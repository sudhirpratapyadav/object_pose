"""Web-server entrypoint: camera + depth worker + WebSocket streamer.

Same pipeline as detect.py, but the viewer is a custom React+Three frontend in
web/. This script streams binary frames (points / mesh / rgb-jpeg) to all
connected browser clients.

Wire format (little-endian, uint8 'kind' tag):
  HEADER       : magic 'P3DF' u32, seq u32, kind u8, _pad u24
  kind=0 pts   : n u32, xyz_f16 [3n], rgb_u8 [3n]
  kind=1 mesh  : nv u32, nf u32, xyz_f16 [3nv], rgb_u8 [3nv], faces_u32 [3nf]
  kind=2 jpeg  : w u16, h u16, jpeg_bytes...
  kind=3 meta  : json_bytes... (intrinsics, fps, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import struct
import time
from typing import Set

import cv2
import numpy as np
import websockets

from camera import NetworkRGB, RealSenseRGB
from depth import create_shm, depth_worker, DEFAULT_MODEL

VIZ_HZ = 30
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
INFER_W, INFER_H = 640, 480

MAGIC = b"P3DF"
KIND_POINTS = 0
KIND_MESH   = 1
KIND_JPEG   = 2
KIND_META   = 3


def _pack_header(seq: int, kind: int) -> bytes:
    return MAGIC + struct.pack("<IB3x", seq & 0xFFFFFFFF, kind)


def _f32_to_f16_bytes(arr_f32: np.ndarray) -> bytes:
    return arr_f32.astype(np.float16).tobytes()


def _frame_points(seq: int, xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    n = xyz.shape[0]
    return (
        _pack_header(seq, KIND_POINTS)
        + struct.pack("<I", n)
        + _f32_to_f16_bytes(xyz)
        + rgb.astype(np.uint8).tobytes()
    )


def _frame_mesh(seq: int, xyz: np.ndarray, rgb: np.ndarray,
                faces: np.ndarray) -> bytes:
    return (
        _pack_header(seq, KIND_MESH)
        + struct.pack("<II", xyz.shape[0], faces.shape[0])
        + _f32_to_f16_bytes(xyz)
        + (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        + faces.astype(np.uint32).tobytes()
    )


def _frame_jpeg(seq: int, bgr: np.ndarray, quality: int = 70) -> bytes:
    h, w = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return _pack_header(seq, KIND_JPEG) + struct.pack("<HH", w, h) + buf.tobytes()


def _frame_meta(seq: int, payload: dict) -> bytes:
    return _pack_header(seq, KIND_META) + json.dumps(payload).encode("utf-8")


class Hub:
    """Tracks connected clients and broadcasts frames."""

    def __init__(self) -> None:
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws) -> None:
        async with self._lock:
            self.clients.add(ws)

    async def remove(self, ws) -> None:
        async with self._lock:
            self.clients.discard(ws)

    async def broadcast(self, data: bytes) -> None:
        if not self.clients:
            return
        # Snapshot to avoid mutation during send
        async with self._lock:
            targets = list(self.clients)
        await asyncio.gather(
            *(self._safe_send(ws, data) for ws in targets),
            return_exceptions=True,
        )

    async def _safe_send(self, ws, data: bytes) -> None:
        try:
            await ws.send(data)
        except Exception:
            await self.remove(ws)


async def handler(ws, hub: Hub, meta_payload: dict) -> None:
    await hub.add(ws)
    try:
        # Send meta (intrinsics, dims) on connect
        await ws.send(_frame_meta(0, meta_payload))
        async for _ in ws:
            pass  # No client messages expected for now
    finally:
        await hub.remove(ws)


async def main_async(args) -> None:
    if args.camera == "realsense":
        cam = RealSenseRGB(width=CAM_W, height=CAM_H, fps=CAM_FPS)
    else:
        cam = NetworkRGB(args.camera)
    intr = cam.start()

    sx = INFER_W / intr.width
    sy = INFER_H / intr.height
    fx_i, fy_i = intr.fx * sx, intr.fy * sy
    cx_i, cy_i = intr.cx * sx, intr.cy * sy

    shm = create_shm(intr.width, intr.height, INFER_W, INFER_H)

    stop_ev = mp.Event()
    status_q = mp.Queue(maxsize=64)
    proc = mp.Process(
        target=depth_worker,
        args=(
            shm.rgb.name, shm.depth.name, shm.pc_xyz.name, shm.pc_rgb.name,
            shm.mesh_xyz.name, shm.mesh_rgb.name, shm.mesh_faces.name,
            shm.rgb_seq, shm.depth_seq, shm.pc_count,
            shm.rgb_w, shm.rgb_h, shm.infer_w, shm.infer_h, shm.n_max,
            shm.mesh_grid_w, shm.mesh_grid_h, shm.mesh_n_faces,
            fx_i, fy_i, cx_i, cy_i,
            stop_ev,
            status_q,
            args.model,
        ),
        daemon=True,
    )
    proc.start()

    rgb_buf    = shm.rgb_arr()
    pc_xyz     = shm.pc_xyz_arr()
    pc_rgb     = shm.pc_rgb_arr()
    mesh_xyz   = shm.mesh_xyz_arr()
    mesh_rgb   = shm.mesh_rgb_arr()
    mesh_faces = shm.mesh_faces_arr()

    meta_payload = {
        "rgb_w": intr.width, "rgb_h": intr.height,
        "infer_w": INFER_W, "infer_h": INFER_H,
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        "fx_infer": fx_i, "fy_infer": fy_i, "cx_infer": cx_i, "cy_infer": cy_i,
        "mesh_grid_w": shm.mesh_grid_w, "mesh_grid_h": shm.mesh_grid_h,
        "viz_hz": VIZ_HZ,
    }

    hub = Hub()

    async def ws_loop():
        async with websockets.serve(
            lambda ws: handler(ws, hub, meta_payload),
            args.host, args.port,
            max_size=None,         # don't reject big binary frames
            compression=None,      # binary already tight
            ping_interval=20,
        ):
            print(f"[ws] listening on ws://{args.host}:{args.port}", flush=True)
            await asyncio.Future()  # run forever

    async def stream_loop():
        period = 1.0 / VIZ_HZ
        seq = 0
        last_depth_seq = 0
        n_rgb = n_depth = 0
        t_log = time.time()

        while True:
            t0 = time.time()

            rgb = cam.get()
            if rgb is not None:
                rgb_buf[...] = rgb
                with shm.rgb_seq.get_lock():
                    shm.rgb_seq.value += 1
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                await hub.broadcast(_frame_jpeg(seq, bgr))
                seq += 1
                n_rgb += 1

            with shm.depth_seq.get_lock():
                cur = shm.depth_seq.value
            if cur != last_depth_seq:
                last_depth_seq = cur
                with shm.pc_count.get_lock():
                    n = shm.pc_count.value
                if n > 0:
                    await hub.broadcast(
                        _frame_points(seq, pc_xyz[:n].copy(), pc_rgb[:n].copy())
                    )
                    seq += 1
                    await hub.broadcast(
                        _frame_mesh(seq, mesh_xyz.copy(), mesh_rgb.copy(),
                                    mesh_faces.copy())
                    )
                    seq += 1
                n_depth += 1

            if time.time() - t_log >= 1.0:
                dt = time.time() - t_log
                print(f"[ws] rgb {n_rgb/dt:.1f}  depth {n_depth/dt:.1f}  "
                      f"clients={len(hub.clients)}", flush=True)
                n_rgb = n_depth = 0
                t_log = time.time()

            elapsed = time.time() - t0
            if elapsed < period:
                await asyncio.sleep(period - elapsed)
            else:
                await asyncio.sleep(0)

    try:
        await asyncio.gather(ws_loop(), stream_loop())
    finally:
        stop_ev.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
        cam.stop()
        shm.close()
        shm.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="realsense",
                    help="'realsense' or HTTP URL like 'http://host:8080'")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
