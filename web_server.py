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

from pathlib import Path

from camera import NetworkRGB, RealSenseRGB, VideoFile
from depth import BACKENDS, create_shm, depth_worker, DEFAULT_MODEL
from segment import (
    BACKENDS as SAM_BACKENDS,
    DEFAULT_MODEL as SAM_DEFAULT_MODEL,
    create_seg_shm,
    segment_worker,
)

VIZ_HZ = 30
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
INFER_W, INFER_H = 640, 480

VIDEO_DIR = Path(__file__).parent / "datasets"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def list_videos() -> list[str]:
    """Recursively list video files under datasets/, returning paths relative
    to VIDEO_DIR so the dropdown shows '<dataset>/<camera>/<episode>.mp4'."""
    if not VIDEO_DIR.exists():
        return []
    out = []
    for p in VIDEO_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        # Skip in-flight download stubs (e.g. ".file-000.mp4.w1kE89")
        if p.name.startswith("."):
            continue
        out.append(str(p.relative_to(VIDEO_DIR)))
    return sorted(out)

MAGIC = b"P3DF"
KIND_POINTS      = 0
KIND_MESH        = 1
KIND_JPEG        = 2  # rgb camera image
KIND_META        = 3  # one-shot connect payload (intrinsics + model keys)
KIND_DEPTH_JPEG  = 4  # turbo-colormapped depth image
KIND_MODEL_STATE = 5  # current depth model + status text + progress
KIND_MASK        = 6  # SAM2 mask (per-point) + AABB
KIND_SAM_STATE   = 7  # current SAM2 model + status
KIND_STATS       = 8  # 1 Hz live stats: rgb fps, depth fps, sam_ms
KIND_NORMAL_JPEG = 9  # colorized surface-normal image


def _pack_header(seq: int, kind: int) -> bytes:
    return MAGIC + struct.pack("<IB3x", seq & 0xFFFFFFFF, kind)


def _f32_to_f16_bytes(arr_f32: np.ndarray) -> bytes:
    return arr_f32.astype(np.float16).tobytes()


def _frame_points(seq: int, xyz: np.ndarray, rgb: np.ndarray,
                  mask: np.ndarray | None = None) -> bytes:
    """Layout: header | n u32 | xyz_f16 [3n] | rgb_u8 [3n] | mask_u8 [n].

    mask is per-point (same length as n). 0 if not segmented.
    """
    n = xyz.shape[0]
    if mask is None:
        mask = np.zeros((n,), dtype=np.uint8)
    return (
        _pack_header(seq, KIND_POINTS)
        + struct.pack("<I", n)
        + _f32_to_f16_bytes(xyz)
        + rgb.astype(np.uint8).tobytes()
        + mask.astype(np.uint8).tobytes()
    )


def _frame_mesh(seq: int, xyz: np.ndarray, rgb: np.ndarray,
                faces: np.ndarray, normal: np.ndarray | None = None) -> bytes:
    """Layout: header | nv u32 | nf u32 | has_normal u8 | xyz_f16 [3nv]
       | rgb_u8 [3nv] | faces_u32 [3nf] | normal_f16 [3nv if has_normal]."""
    has_n = 1 if normal is not None else 0
    body = (
        struct.pack("<IIB", xyz.shape[0], faces.shape[0], has_n)
        + _f32_to_f16_bytes(xyz)
        + (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        + faces.astype(np.uint32).tobytes()
    )
    if has_n:
        body += _f32_to_f16_bytes(normal.astype(np.float32))
    return _pack_header(seq, KIND_MESH) + body


def _frame_jpeg(seq: int, bgr: np.ndarray, kind: int = KIND_JPEG,
                quality: int = 70) -> bytes:
    h, w = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return _pack_header(seq, kind) + struct.pack("<HH", w, h) + buf.tobytes()


def _frame_meta(seq: int, payload: dict) -> bytes:
    return _pack_header(seq, KIND_META) + json.dumps(payload).encode("utf-8")


def _frame_model_state(seq: int, payload: dict) -> bytes:
    return _pack_header(seq, KIND_MODEL_STATE) + json.dumps(payload).encode("utf-8")


def _frame_sam_state(seq: int, payload: dict) -> bytes:
    return _pack_header(seq, KIND_SAM_STATE) + json.dumps(payload).encode("utf-8")


def _frame_stats(seq: int, payload: dict) -> bytes:
    return _pack_header(seq, KIND_STATS) + json.dumps(payload).encode("utf-8")


def _frame_mask(seq: int, mask_seq: int, mask: np.ndarray,
                has_box: bool, box_min: np.ndarray, box_max: np.ndarray) -> bytes:
    """mask: uint8 array, length grid_w*grid_h."""
    n = int(mask.size)
    body = (
        struct.pack("<II", mask_seq & 0xFFFFFFFF, n)
        + mask.astype(np.uint8).tobytes()
        + struct.pack("<B", 1 if has_box else 0)
        + box_min.astype(np.float32).tobytes()
        + box_max.astype(np.float32).tobytes()
    )
    return _pack_header(seq, KIND_MASK) + body


def _depth_to_turbo_bgr(depth_m: np.ndarray, dmax: float = 6.0) -> np.ndarray:
    """Colorize depth (meters) as a TURBO-mapped BGR uint8 image."""
    d = np.clip(depth_m, 0.0, dmax)
    u8 = (d / max(dmax, 1e-3) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _normal_to_bgr(n_img: np.ndarray) -> np.ndarray:
    """Standard normal-map visualization: (n+1)/2 in 0..255, RGB then BGR."""
    rgb = ((np.clip(n_img, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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


async def handler(ws, hub: Hub, get_meta_payload,
                  on_set_model, on_set_sam_model,
                  on_sam_click, on_sam_clear,
                  on_set_source) -> None:
    await hub.add(ws)
    try:
        await ws.send(_frame_meta(0, get_meta_payload()))
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    cmd = json.loads(msg)
                except Exception:
                    continue
                if not isinstance(cmd, dict):
                    continue
                if "set_model" in cmd:
                    await on_set_model(str(cmd["set_model"]))
                elif "set_sam_model" in cmd:
                    await on_set_sam_model(str(cmd["set_sam_model"]))
                elif "sam_click" in cmd and isinstance(cmd["sam_click"], dict):
                    c = cmd["sam_click"]
                    await on_sam_click(int(c.get("x", 0)), int(c.get("y", 0)))
                elif "sam_clear" in cmd:
                    await on_sam_clear()
                elif "set_source" in cmd and isinstance(cmd["set_source"], dict):
                    s = cmd["set_source"]
                    kind = str(s.get("kind", "live"))
                    video = s.get("video")
                    await on_set_source(kind, str(video) if video else None)
    finally:
        await hub.remove(ws)


def _make_camera(kind: str, video_name: str | None, args):
    """Build a camera-like object for the requested source. Caller calls .start()."""
    if kind == "video" and video_name:
        path = VIDEO_DIR / video_name
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        return VideoFile(str(path))
    if args.camera == "realsense":
        return RealSenseRGB(width=CAM_W, height=CAM_H, fps=CAM_FPS)
    return NetworkRGB(args.camera)


async def main_async(args) -> None:
    # ---- Mutable session: cam + shm + seg, rebuildable on source switch ----
    sess: dict = {
        "kind": "live",
        "video": None,
        "cam": None,
        "intr": None,
        "shm": None,
        "seg": None,
        "fx_i": 0.0, "fy_i": 0.0, "cx_i": 0.0, "cy_i": 0.0,
    }

    def build_session(kind: str, video_name: str | None) -> None:
        cam = _make_camera(kind, video_name, args)
        intr = cam.start()
        sx = INFER_W / intr.width
        sy = INFER_H / intr.height
        shm = create_shm(intr.width, intr.height, INFER_W, INFER_H)
        seg = create_seg_shm(shm.mesh_grid_w, shm.mesh_grid_h)
        sess.update(
            kind=kind, video=video_name,
            cam=cam, intr=intr, shm=shm, seg=seg,
            fx_i=intr.fx * sx, fy_i=intr.fy * sy,
            cx_i=intr.cx * sx, cy_i=intr.cy * sy,
        )

    def teardown_session() -> None:
        cam = sess.get("cam")
        shm = sess.get("shm")
        seg = sess.get("seg")
        if cam is not None: cam.stop()
        if seg is not None:
            seg.close(); seg.unlink()
        if shm is not None:
            shm.close(); shm.unlink()

    build_session(args.source, args.video)

    def make_meta_payload() -> dict:
        intr = sess["intr"]
        shm = sess["shm"]
        return {
            "rgb_w": intr.width, "rgb_h": intr.height,
            "infer_w": INFER_W, "infer_h": INFER_H,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "fx_infer": sess["fx_i"], "fy_infer": sess["fy_i"],
            "cx_infer": sess["cx_i"], "cy_infer": sess["cy_i"],
            "mesh_grid_w": shm.mesh_grid_w, "mesh_grid_h": shm.mesh_grid_h,
            "viz_hz": VIZ_HZ,
            "models": list(BACKENDS.keys()),
            "default_model": DEFAULT_MODEL,
            "sam_models": list(SAM_BACKENDS.keys()),
            "sam_default_model": SAM_DEFAULT_MODEL,
            "videos": list_videos(),
            "source": {"kind": sess["kind"], "video": sess["video"]},
        }

    hub = Hub()
    state = {
        "proc": None,
        "stop_ev": None,
        "status_q": None,
        "model": args.model,
        "model_status": "starting",
        "model_progress": "",
        "model_file": "",
    }
    sam_state = {
        "proc": None,
        "stop_ev": None,
        "status_q": None,
        "model": args.sam_model,
        "status": "starting",
        "file": "",
    }

    def spawn_depth(model_key: str) -> None:
        shm = sess["shm"]
        stop_ev  = mp.Event()
        status_q = mp.Queue(maxsize=64)
        proc = mp.Process(
            target=depth_worker,
            args=(
                shm.rgb.name, shm.depth.name, shm.pc_xyz.name, shm.pc_rgb.name,
                shm.pc_grid_idx.name,
                shm.mesh_xyz.name, shm.mesh_rgb.name, shm.mesh_faces.name,
                shm.mesh_normal.name, shm.normal_img.name,
                shm.rgb_seq, shm.depth_seq, shm.pc_count, shm.has_normal,
                shm.rgb_w, shm.rgb_h, shm.infer_w, shm.infer_h, shm.n_max,
                shm.mesh_grid_w, shm.mesh_grid_h, shm.mesh_n_faces,
                sess["fx_i"], sess["fy_i"], sess["cx_i"], sess["cy_i"],
                stop_ev, status_q, model_key,
            ),
            daemon=True,
        )
        proc.start()
        state.update(proc=proc, stop_ev=stop_ev, status_q=status_q,
                     model=model_key, model_status="loading",
                     model_progress="", model_file="")

    def stop_depth() -> None:
        if state["stop_ev"] is not None:
            state["stop_ev"].set()
        if state["proc"] is not None:
            state["proc"].join(timeout=10.0)
            if state["proc"].is_alive():
                state["proc"].terminate()
        if state["status_q"] is not None:
            while True:
                try: state["status_q"].get_nowait()
                except Exception: break

    def _model_has_normals(key: str) -> bool:
        # Probe BackendInfo without instantiating the backend.
        factory = BACKENDS.get(key)
        if factory is None:
            return False
        try:
            return bool(factory(1.0).info.has_normals)
        except Exception:
            return False

    def make_model_state_payload() -> dict:
        return {
            "model": state["model"],
            "status": state["model_status"],
            "progress": state["model_progress"],
            "file": state["model_file"],
            "has_normals": _model_has_normals(state["model"]),
        }

    def make_sam_state_payload() -> dict:
        return {
            "model": sam_state["model"],
            "status": sam_state["status"],
            "file": sam_state["file"],
        }

    def spawn_seg(model_key: str) -> None:
        shm = sess["shm"]
        seg = sess["seg"]
        stop_ev  = mp.Event()
        status_q = mp.Queue(maxsize=64)
        proc = mp.Process(
            target=segment_worker,
            args=(
                shm.rgb.name, shm.mesh_xyz.name,
                seg.mask.name, seg.bbox.name,
                seg.click_seq, seg.click_x, seg.click_y,
                seg.mask_seq, seg.has_mask,
                shm.rgb_w, shm.rgb_h,
                shm.infer_w, shm.infer_h,
                shm.mesh_grid_w, shm.mesh_grid_h,
                4,  # MESH_DOWNSAMPLE — must match depth/runner.MESH_DOWNSAMPLE
                stop_ev, status_q, model_key,
            ),
            daemon=True,
        )
        proc.start()
        sam_state.update(proc=proc, stop_ev=stop_ev, status_q=status_q,
                         model=model_key, status="loading", file="")

    def stop_seg() -> None:
        if sam_state["stop_ev"] is not None:
            sam_state["stop_ev"].set()
        if sam_state["proc"] is not None:
            sam_state["proc"].join(timeout=10.0)
            if sam_state["proc"].is_alive():
                sam_state["proc"].terminate()
        if sam_state["status_q"] is not None:
            while True:
                try: sam_state["status_q"].get_nowait()
                except Exception: break

    async def on_set_model(key: str) -> None:
        if key == state["model"] or key not in BACKENDS:
            return
        state.update(model_status=f"switching to {key} ...",
                     model_progress="", model_file="")
        await hub.broadcast(_frame_model_state(0, make_model_state_payload()))
        # Run blocking spawn/stop in a thread so we don't stall the loop.
        await asyncio.to_thread(stop_depth)
        shm = sess["shm"]
        with shm.pc_count.get_lock():
            shm.pc_count.value = 0
        await asyncio.to_thread(spawn_depth, key)
        await hub.broadcast(_frame_model_state(0, make_model_state_payload()))

    async def on_set_sam_model(key: str) -> None:
        if key == sam_state["model"] or key not in SAM_BACKENDS:
            return
        sam_state.update(status=f"switching to {key} ...", file="")
        await hub.broadcast(_frame_sam_state(0, make_sam_state_payload()))
        await asyncio.to_thread(stop_seg)
        # Wipe any current mask so the UI doesn't keep showing an old object.
        seg = sess["seg"]
        seg.mask_arr()[:] = 0
        with seg.has_mask.get_lock(): seg.has_mask.value = 0
        with seg.mask_seq.get_lock(): seg.mask_seq.value = seg.mask_seq.value + 1
        await asyncio.to_thread(spawn_seg, key)
        await hub.broadcast(_frame_sam_state(0, make_sam_state_payload()))

    sam_pending_t = {"t": 0.0}    # mutable closure cell for click-time
    sam_last_ms = {"v": 0}        # most recent click->mask latency (ms)

    async def on_sam_click(x: int, y: int) -> None:
        # Coordinates arrive in INFERENCE-frame pixels.
        seg = sess["seg"]
        with seg.click_x.get_lock(): seg.click_x.value = int(x)
        with seg.click_y.get_lock(): seg.click_y.value = int(y)
        with seg.click_seq.get_lock(): seg.click_seq.value = seg.click_seq.value + 1
        sam_pending_t["t"] = time.monotonic()

    async def on_sam_clear() -> None:
        # Sentinel: negative coords -> worker clears mask.
        seg = sess["seg"]
        with seg.click_x.get_lock(): seg.click_x.value = -1
        with seg.click_y.get_lock(): seg.click_y.value = -1
        with seg.click_seq.get_lock(): seg.click_seq.value = seg.click_seq.value + 1
        sam_pending_t["t"] = 0.0

    async def on_set_source(kind: str, video_name: str | None) -> None:
        # No-op if already on this source.
        if kind == sess["kind"] and video_name == sess["video"]:
            return
        # Tear down workers and replace the session, then respawn.
        await asyncio.to_thread(stop_seg)
        await asyncio.to_thread(stop_depth)
        await asyncio.to_thread(teardown_session)
        try:
            build_session(kind, video_name)
        except Exception as exc:
            print(f"[ws] set_source failed: {exc}", flush=True)
            # Fall back to live camera.
            build_session("live", None)
        # Push the new meta + reset workers.
        await hub.broadcast(_frame_meta(0, {
            **make_meta_payload(),
            "model_state": make_model_state_payload(),
            "sam_state": make_sam_state_payload(),
        }))
        await asyncio.to_thread(spawn_depth, state["model"])
        await asyncio.to_thread(spawn_seg, sam_state["model"])

    spawn_depth(args.model)
    spawn_seg(args.sam_model)

    async def ws_loop():
        def get_meta():
            return {
                **make_meta_payload(),
                "model_state": make_model_state_payload(),
                "sam_state": make_sam_state_payload(),
            }
        async with websockets.serve(
            lambda ws: handler(
                ws, hub, get_meta,
                on_set_model, on_set_sam_model,
                on_sam_click, on_sam_clear,
                on_set_source,
            ),
            args.host, args.port,
            max_size=None, compression=None, ping_interval=20,
        ):
            print(f"[ws] listening on ws://{args.host}:{args.port}", flush=True)
            await asyncio.Future()

    async def stream_loop():
        period = 1.0 / VIZ_HZ
        seq = 0
        last_depth_seq = 0
        last_mask_seq = 0
        n_rgb = n_depth = 0
        t_log = time.time()

        while True:
            t0 = time.time()
            # Re-read session each iteration so source-switches take effect
            # without restarting the loop.
            cam = sess["cam"]
            shm = sess["shm"]
            seg = sess["seg"]
            rgb_buf = shm.rgb_arr()
            depth_buf = shm.depth_arr()
            pc_xyz = shm.pc_xyz_arr()
            pc_rgb = shm.pc_rgb_arr()
            pc_grid_idx = shm.pc_grid_idx_arr()
            mesh_xyz = shm.mesh_xyz_arr()
            mesh_rgb = shm.mesh_rgb_arr()
            mesh_faces = shm.mesh_faces_arr()
            mesh_normal = shm.mesh_normal_arr()
            normal_img = shm.normal_img_arr()
            seg_mask = seg.mask_arr()
            seg_bbox = seg.bbox_arr()

            # Drain depth-worker status queue and broadcast.
            sq = state["status_q"]
            if sq is not None:
                latest = None
                while True:
                    try: latest = sq.get_nowait()
                    except Exception: break
                if latest is not None:
                    kind = latest[0] if latest else ""
                    if kind == "downloading":
                        fname, progress = latest[1], latest[2]
                        state.update(model_status="downloading",
                                     model_progress=str(progress),
                                     model_file=str(fname))
                    elif kind == "loading":
                        state.update(model_status="loading", model_progress="",
                                     model_file="")
                    elif kind == "warming":
                        state.update(model_status="warming up",
                                     model_progress="", model_file="")
                    elif kind == "ready":
                        state.update(model_status=f"running {state['model']}",
                                     model_progress="", model_file="")
                    elif kind == "error":
                        state.update(model_status="error", model_progress="",
                                     model_file=latest[1] if len(latest) > 1 else "")
                    await hub.broadcast(_frame_model_state(0, make_model_state_payload()))

            # Drain SAM-worker status queue.
            ssq = sam_state["status_q"]
            if ssq is not None:
                latest = None
                while True:
                    try: latest = ssq.get_nowait()
                    except Exception: break
                if latest is not None:
                    kind = latest[0] if latest else ""
                    if kind == "downloading":
                        fname = latest[1] if len(latest) > 1 else ""
                        sam_state.update(status="downloading", file=str(fname))
                    elif kind == "loading":
                        sam_state.update(status="loading", file="")
                    elif kind == "ready":
                        sam_state.update(status=f"running {sam_state['model']}",
                                         file="")
                    elif kind == "ok":
                        sam_state.update(status=f"running {sam_state['model']}",
                                         file="")
                    elif kind == "cleared":
                        sam_state.update(status=f"running {sam_state['model']}",
                                         file="")
                    elif kind == "error":
                        sam_state.update(status="error",
                                         file=latest[1] if len(latest) > 1 else "")
                    await hub.broadcast(_frame_sam_state(0, make_sam_state_payload()))

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
                with shm.has_normal.get_lock():
                    have_n = bool(shm.has_normal.value)
                normal_payload = mesh_normal.copy() if have_n else None
                if n > 0:
                    pc_mask = seg_mask[pc_grid_idx[:n]].copy()
                    await hub.broadcast(
                        _frame_points(seq, pc_xyz[:n].copy(), pc_rgb[:n].copy(),
                                      mask=pc_mask)
                    )
                    seq += 1
                    await hub.broadcast(
                        _frame_mesh(seq, mesh_xyz.copy(), mesh_rgb.copy(),
                                    mesh_faces.copy(), normal=normal_payload)
                    )
                    seq += 1
                # Always send a colorized depth jpeg, even when pc is empty.
                depth_bgr = _depth_to_turbo_bgr(depth_buf)
                await hub.broadcast(
                    _frame_jpeg(seq, depth_bgr, kind=KIND_DEPTH_JPEG, quality=80)
                )
                seq += 1
                # And a normal-map jpeg if the model produces normals.
                if have_n:
                    normal_bgr = _normal_to_bgr(normal_img)
                    await hub.broadcast(
                        _frame_jpeg(seq, normal_bgr, kind=KIND_NORMAL_JPEG, quality=80)
                    )
                    seq += 1
                n_depth += 1

            # Broadcast a mask frame whenever the SAM worker bumps mask_seq.
            with seg.mask_seq.get_lock():
                cur_m = seg.mask_seq.value
            if cur_m != last_mask_seq:
                last_mask_seq = cur_m
                # Click->mask latency
                if sam_pending_t["t"] > 0.0:
                    sam_last_ms["v"] = int((time.monotonic() - sam_pending_t["t"]) * 1000)
                    sam_pending_t["t"] = 0.0
                with seg.has_mask.get_lock():
                    has_box = bool(seg.has_mask.value)
                box_min = seg_bbox[0:3].copy()
                box_max = seg_bbox[3:6].copy()
                await hub.broadcast(
                    _frame_mask(seq, cur_m, seg_mask.copy(), has_box, box_min, box_max)
                )
                seq += 1

            if time.time() - t_log >= 1.0:
                dt = time.time() - t_log
                rgb_fps = n_rgb / dt
                depth_fps = n_depth / dt
                print(f"[ws] rgb {rgb_fps:.1f}  depth {depth_fps:.1f}  "
                      f"clients={len(hub.clients)}", flush=True)
                await hub.broadcast(_frame_stats(seq, {
                    "rgb_fps":   round(rgb_fps, 1),
                    "depth_fps": round(depth_fps, 1),
                    "sam_ms":    sam_last_ms["v"],
                }))
                seq += 1
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
        stop_seg()
        stop_depth()
        teardown_session()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="realsense",
                    help="'realsense' or HTTP URL like 'http://host:8080'")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sam_model", default=SAM_DEFAULT_MODEL)
    ap.add_argument("--source", default="live", choices=["live", "video"],
                    help="Initial input source ('live' uses --camera; 'video' picks --video).")
    ap.add_argument("--video", default=None,
                    help="Video filename inside dataset/videos/ to start with.")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
