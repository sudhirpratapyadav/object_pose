"""Web-server entrypoint: camera + depth worker + (optional) robot + WS streamer.

Streams binary frames (points / mesh / rgb-jpeg / robot transforms / ...) to all
connected browser clients. Wire format is documented in wire/__init__.py and
mirrored on the JS side in web/src/protocol.ts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import time
from typing import Set

import cv2
import numpy as np
import websockets

from pathlib import Path

from camera import NetworkRGB, RealSenseRGB, VideoFile
from camera.realsense import Intrinsics as RsIntrinsics
from config import (
    CamCalib, CamCalibError, Extrinsics, Intrinsics as CfgIntrinsics,
    CAM_CALIB_PATH, SIM_CONFIG_PATH,
    load_cam_calib, load_sim_config,
    matrix_to_quat_wxyz, save_cam_calib,
    write_factory_intrinsics, FACTORY_INTR_PATH,
)
from depth import BACKENDS, create_shm, depth_worker, DEFAULT_MODEL
from segment import (
    BACKENDS as SAM_BACKENDS,
    DEFAULT_MODEL as SAM_DEFAULT_MODEL,
    create_seg_shm,
    segment_worker,
)
from robot import (
    DummySource,
    FKEngine,
    create_robot_shm,
    load_robot_scene,
)
from robot.wire import build_robot_geometry_payload
from wire import (
    KIND_DEPTH_JPEG,
    KIND_NORMAL_JPEG,
    encode_cam_calib,
    encode_jpeg,
    encode_mask,
    encode_mesh,
    encode_meta,
    encode_model_state,
    encode_points,
    encode_robot_geometry,
    encode_robot_status,
    encode_robot_transforms,
    encode_sam_state,
    encode_stats,
)

VIZ_HZ = 30
CAM_FPS = 30
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

def _xform_points(xyz: np.ndarray, T: np.ndarray, mask_zero: bool = False) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points.

    If ``mask_zero`` is set, points that were exactly (0,0,0) on input
    (e.g. mesh vertices flagged invalid by fill_mesh) are forced back to
    (0,0,0) afterwards so the GPU's degenerate-triangle cull still works.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    out = xyz @ R.T + t
    if mask_zero:
        invalid = np.all(xyz == 0.0, axis=-1)
        if invalid.any():
            out[invalid] = 0.0
    return out.astype(np.float32, copy=False)


def _xform_normals(n: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) unit normals (no translation)."""
    R = T[:3, :3]
    return (n @ R.T).astype(np.float32, copy=False)


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


async def handler(ws, hub: Hub, get_connect_frames,
                  on_set_model, on_set_sam_model,
                  on_sam_click, on_sam_clear,
                  on_set_source,
                  on_set_cam_extrinsics, on_save_cam_extrinsics,
                  on_reload_cam_extrinsics,
                  on_set_target_ctrl) -> None:
    await hub.add(ws)
    try:
        for frame in get_connect_frames():
            await ws.send(frame)
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
                elif "set_cam_extrinsics" in cmd and isinstance(cmd["set_cam_extrinsics"], dict):
                    e = cmd["set_cam_extrinsics"]
                    pos = e.get("pos") or [0.0, 0.0, 0.0]
                    eul = e.get("euler_deg") or [0.0, 0.0, 0.0]
                    if len(pos) == 3 and len(eul) == 3:
                        await on_set_cam_extrinsics(
                            [float(x) for x in pos],
                            [float(x) for x in eul],
                        )
                elif "save_cam_extrinsics" in cmd:
                    await on_save_cam_extrinsics()
                elif "reload_cam_extrinsics" in cmd:
                    await on_reload_cam_extrinsics()
                elif "set_target_ctrl" in cmd and isinstance(cmd["set_target_ctrl"], list):
                    vals = [float(x) for x in cmd["set_target_ctrl"]]
                    await on_set_target_ctrl(vals)
    finally:
        await hub.remove(ws)


def _make_camera(kind: str, video_name: str | None, args, cam_calib: CamCalib):
    """Build a camera-like object for the requested source. Caller calls .start()."""
    if args.mode == "sim":
        from sim import MujocoCamera
        return MujocoCamera(cam_calib), False
    if kind == "video" and video_name:
        path = VIDEO_DIR / video_name
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        return VideoFile(str(path)), False
    if args.camera == "realsense":
        # Honour YAML resolution; intrinsics get overridden after start().
        # enable_depth=True; RealSenseRGB will auto-fall-back to color-only
        # if the camera fails to deliver any frames in RGBD mode.
        cam = RealSenseRGB(
            width=cam_calib.intrinsics.width,
            height=cam_calib.intrinsics.height,
            fps=CAM_FPS,
            enable_depth=True,
        )
        return cam, True
    return NetworkRGB(args.camera), False


def _calib_payload(calib: CamCalib) -> dict:
    """Serialise the active CamCalib for the meta / cam-config wire frames."""
    T = calib.T_world_camera()
    quat_wxyz = matrix_to_quat_wxyz(T[:3, :3]).tolist()
    return {
        "extrinsics": {
            "pos":       list(calib.extrinsics.pos),
            "euler_deg": list(calib.extrinsics.euler_deg),
            "pos_world": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
            "quat_wxyz": [float(x) for x in quat_wxyz],
        },
        "intrinsics": {
            "fx": calib.intrinsics.fx, "fy": calib.intrinsics.fy,
            "cx": calib.intrinsics.cx, "cy": calib.intrinsics.cy,
            "width":  calib.intrinsics.width,
            "height": calib.intrinsics.height,
        },
    }


def _intr_drift_warn(factory: RsIntrinsics, cfg: CfgIntrinsics) -> None:
    """Log a warning if YAML intrinsics drift significantly from factory."""
    issues: list[str] = []
    if abs(factory.fx - cfg.fx) / max(factory.fx, 1.0) > 0.05:
        issues.append(f"fx {factory.fx:.1f} -> {cfg.fx:.1f}")
    if abs(factory.fy - cfg.fy) / max(factory.fy, 1.0) > 0.05:
        issues.append(f"fy {factory.fy:.1f} -> {cfg.fy:.1f}")
    if abs(factory.cx - cfg.cx) > 10.0:
        issues.append(f"cx {factory.cx:.1f} -> {cfg.cx:.1f}")
    if abs(factory.cy - cfg.cy) > 10.0:
        issues.append(f"cy {factory.cy:.1f} -> {cfg.cy:.1f}")
    if issues:
        print(f"[cam] WARNING: YAML intrinsics differ from factory: "
              f"{', '.join(issues)}", flush=True)


async def main_async(args) -> None:
    # Calibration config (the depth pipeline's belief): required in both modes.
    try:
        cam_calib = load_cam_calib()
    except CamCalibError as exc:
        print(f"[cfg] {exc}", flush=True)
        return
    print(f"[cfg] cam_calib  {cam_calib.intrinsics.width}x{cam_calib.intrinsics.height}  "
          f"fx={cam_calib.intrinsics.fx:.1f} fy={cam_calib.intrinsics.fy:.1f}  "
          f"pos={cam_calib.extrinsics.pos}  euler={cam_calib.extrinsics.euler_deg}",
          flush=True)

    # Sim ground truth: required in sim mode, ignored in real mode.
    sim_calib: CamCalib | None = None
    if args.mode == "sim":
        try:
            sim_calib = load_sim_config()
        except CamCalibError as exc:
            print(f"[cfg] {exc}", flush=True)
            return
        # Resolution must match between belief and ground truth: the depth
        # pipeline back-projects pixels using cam_calib intrinsics, and those
        # pixels are produced by the sim renderer at sim_calib resolution.
        # Different fx/fy/cx/cy (and pos/euler) is the whole point — different
        # WIDTH/HEIGHT is a configuration bug.
        if (cam_calib.intrinsics.width  != sim_calib.intrinsics.width or
            cam_calib.intrinsics.height != sim_calib.intrinsics.height):
            print(
                f"[cfg] resolution mismatch: cam_calib is "
                f"{cam_calib.intrinsics.width}x{cam_calib.intrinsics.height}, "
                f"sim_config is "
                f"{sim_calib.intrinsics.width}x{sim_calib.intrinsics.height}. "
                f"Both YAMLs must declare the same resolution.",
                flush=True)
            return
        print(f"[cfg] sim_truth {sim_calib.intrinsics.width}x{sim_calib.intrinsics.height}  "
              f"fx={sim_calib.intrinsics.fx:.1f} fy={sim_calib.intrinsics.fy:.1f}  "
              f"pos={sim_calib.extrinsics.pos}  euler={sim_calib.extrinsics.euler_deg}",
              flush=True)

    # ---- Mutable session: cam + shm + seg, rebuildable on source switch ----
    sess: dict = {
        "kind": "live",
        "video": None,
        "cam": None,
        "intr": None,
        "shm": None,
        "seg": None,
        "fx_i": 0.0, "fy_i": 0.0, "cx_i": 0.0, "cy_i": 0.0,
        "calib": cam_calib,    # active CamCalib; mutated by calibration commands
    }

    def build_session(kind: str, video_name: str | None) -> None:
        cam, is_realsense = _make_camera(kind, video_name, args, cam_calib)
        intr = cam.start()
        if is_realsense and kind == "live":
            # Snapshot what the SDK reported, then override with YAML.
            ci = cam_calib.intrinsics
            factory_intr = CfgIntrinsics(
                fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
                width=intr.width, height=intr.height,
            )
            try:
                write_factory_intrinsics(
                    factory_intr,
                    header=f"RealSense factory snapshot from "
                           f"{cam_calib.intrinsics.width}x{cam_calib.intrinsics.height} stream "
                           f"(generated; not read by the server)",
                )
                print(f"[cam] factory intrinsics dumped to {FACTORY_INTR_PATH.name}",
                      flush=True)
            except OSError as exc:
                print(f"[cam] could not write {FACTORY_INTR_PATH}: {exc}", flush=True)
            _intr_drift_warn(intr, ci)
            # Override with YAML for the rest of the pipeline.
            intr = RsIntrinsics(width=ci.width, height=ci.height,
                                fx=ci.fx, fy=ci.fy, cx=ci.cx, cy=ci.cy)
            print(f"[cam] using YAML intrinsics: fx={ci.fx:.1f} fy={ci.fy:.1f} "
                  f"cx={ci.cx:.1f} cy={ci.cy:.1f}", flush=True)

        sx = INFER_W / intr.width
        sy = INFER_H / intr.height
        # Allocate the camera-depth side channel only when the source can
        # actually fill it (RealSense with depth, or sim mode).
        if args.mode == "sim":
            with_depth_stream = True
        elif is_realsense:
            with_depth_stream = bool(getattr(cam, "has_depth", False))
        else:
            with_depth_stream = False
        shm = create_shm(intr.width, intr.height, INFER_W, INFER_H,
                         with_depth_stream=with_depth_stream)
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
            "camera_depth_available": shm.depth_stream is not None,
            "camera_depth_label": (
                "Camera depth (MuJoCo)" if args.mode == "sim"
                else "Camera depth (RealSense)"
            ),
            "sam_models": list(SAM_BACKENDS.keys()),
            "sam_default_model": SAM_DEFAULT_MODEL,
            "videos": list_videos(),
            "source": {"kind": sess["kind"], "video": sess["video"]},
            "robot": {
                "enabled": robot["enabled"],
                "source":  robot["source_kind"],
                "mjcf":    str(args.mjcf) if args.mjcf else None,
                "actuators": (
                    [
                        {"name": a.name, "min": a.ctrl_min,
                         "max": a.ctrl_max, "home": a.home_ctrl}
                        for a in robot["scene"].actuators
                    ] if robot["enabled"] and robot["scene"] is not None else []
                ),
                "ee_body_idx":  (robot["scene"].ee_body_idx
                                 if robot["enabled"] and robot["scene"] is not None
                                 else -1),
                "ee_body_name": (robot["scene"].ee_body_name
                                 if robot["enabled"] and robot["scene"] is not None
                                 else ""),
            },
            "cam_calib": _calib_payload(sess["calib"]),
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

    # ---- Robot (optional) ------------------------------------------------
    robot: dict = {
        "enabled":      bool(args.mjcf),
        "source_kind":  args.robot_source,
        "scene":        None,    # robot.RobotScene
        "fk":           None,    # robot.FKEngine
        "shm":          None,    # robot.RobotShm
        "src":          None,    # robot.DummySource (or future producers)
        "geom_payload": b"",     # cached KIND_ROBOT_GEOMETRY frame
        "last_seq":     0,       # last qpos_seq we broadcast
        "nu":           0,       # number of actuators (for ctrl shm)
        "ctrl_arr":     None,    # mp.Array('d', nu)
        "ctrl_seq":     None,    # mp.Value('I')
    }
    if robot["enabled"]:
        scene = load_robot_scene(args.mjcf)
        fk = FKEngine(scene.model)
        home = scene.home_qpos()
        rshm = create_robot_shm(scene.nq, init_qpos=home)
        bodies, meshes, geoms_json, blob = build_robot_geometry_payload(scene)
        robot.update(scene=scene, fk=fk, shm=rshm)
        robot["geom_payload"] = encode_robot_geometry(0, bodies, meshes,
                                                      geoms_json, blob)
        # Control input slot (sim worker reads each step).
        nu = int(scene.model.nu)
        ctrl_arr = mp.Array("d", max(1, nu), lock=True)
        ctrl_seq = mp.Value("I", 0, lock=False)
        # Seed with home ctrl so initial state matches the home keyframe.
        if nu > 0:
            init_ctrl = np.array([a.home_ctrl for a in scene.actuators],
                                 dtype=np.float64)
            with ctrl_arr.get_lock():
                np.frombuffer(ctrl_arr.get_obj(), dtype=np.float64)[:nu] = init_ctrl
        robot.update(nu=nu, ctrl_arr=ctrl_arr, ctrl_seq=ctrl_seq)
        n_mesh_bytes = len(blob)
        print(f"[robot] loaded {args.mjcf}: {len(scene.bodies)} bodies, "
              f"{len(scene.geoms)} visual geoms, "
              f"{len(meshes)} unique meshes, "
              f"{n_mesh_bytes/1024:.1f} KiB mesh data, nu={nu}", flush=True)
        if args.robot_source == "dummy":
            robot["src"] = DummySource(rshm, home)
            robot["src"].start()
            print("[robot] dummy qpos source started", flush=True)
        # 'sim' source is spawned later, after rgb_shm exists.

    def spawn_depth(model_key: str) -> None:
        shm = sess["shm"]
        stop_ev  = mp.Event()
        status_q = mp.Queue(maxsize=64)
        depth_stream_name = (shm.depth_stream.name
                             if shm.depth_stream is not None else None)
        # Mode-aware label so the dropdown reads naturally.
        depth_label = ("Camera depth (MuJoCo)" if args.mode == "sim"
                       else "Camera depth (RealSense)")
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
                stop_ev, status_q, model_key, "cuda",
                depth_stream_name, depth_label,
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
        await hub.broadcast(encode_model_state(0, make_model_state_payload()))
        # Run blocking spawn/stop in a thread so we don't stall the loop.
        await asyncio.to_thread(stop_depth)
        shm = sess["shm"]
        with shm.pc_count.get_lock():
            shm.pc_count.value = 0
        await asyncio.to_thread(spawn_depth, key)
        await hub.broadcast(encode_model_state(0, make_model_state_payload()))

    async def on_set_sam_model(key: str) -> None:
        if key == sam_state["model"] or key not in SAM_BACKENDS:
            return
        sam_state.update(status=f"switching to {key} ...", file="")
        await hub.broadcast(encode_sam_state(0, make_sam_state_payload()))
        await asyncio.to_thread(stop_seg)
        # Wipe any current mask so the UI doesn't keep showing an old object.
        seg = sess["seg"]
        seg.mask_arr()[:] = 0
        with seg.has_mask.get_lock(): seg.has_mask.value = 0
        with seg.mask_seq.get_lock(): seg.mask_seq.value = seg.mask_seq.value + 1
        await asyncio.to_thread(spawn_seg, key)
        await hub.broadcast(encode_sam_state(0, make_sam_state_payload()))

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

    async def on_set_cam_extrinsics(pos: list[float], euler_deg: list[float]) -> None:
        """Live calibration update from a slider drag. Cheap, in-memory only."""
        sess["calib"] = CamCalib(
            extrinsics=Extrinsics(pos=pos, euler_deg=euler_deg),
            intrinsics=sess["calib"].intrinsics,
        )
        await hub.broadcast(encode_cam_calib(0, _calib_payload(sess["calib"])))

    async def on_save_cam_extrinsics() -> None:
        """Persist the current in-memory calibration to YAML."""
        try:
            await asyncio.to_thread(save_cam_calib, sess["calib"], CAM_CALIB_PATH)
            print(f"[cfg] saved {CAM_CALIB_PATH.name}: "
                  f"pos={sess['calib'].extrinsics.pos} "
                  f"euler={sess['calib'].extrinsics.euler_deg}", flush=True)
        except Exception as exc:
            print(f"[cfg] save failed: {exc}", flush=True)

    async def on_reload_cam_extrinsics() -> None:
        """Re-read cam_calib_config.yaml from disk and broadcast it.

        Discards in-memory edits — useful when the user tweaked the gizmo
        and wants to revert to the saved version. Intrinsics are not
        touched (we never change them at runtime).
        """
        try:
            fresh = await asyncio.to_thread(load_cam_calib, CAM_CALIB_PATH)
        except Exception as exc:
            print(f"[cfg] reload failed: {exc}", flush=True)
            return
        # Replace extrinsics; keep intrinsics from the live session (which
        # for real-camera mode were possibly overridden by factory at boot).
        sess["calib"] = CamCalib(
            extrinsics=fresh.extrinsics,
            intrinsics=sess["calib"].intrinsics,
        )
        print(f"[cfg] reloaded {CAM_CALIB_PATH.name}: "
              f"pos={sess['calib'].extrinsics.pos} "
              f"euler={sess['calib'].extrinsics.euler_deg}", flush=True)
        await hub.broadcast(encode_cam_calib(0, _calib_payload(sess["calib"])))

    async def on_set_target_ctrl(vals: list[float]) -> None:
        """Browser slider -> sim worker. No-op when no robot loaded."""
        nu = robot["nu"]
        if nu == 0 or robot["ctrl_arr"] is None:
            return
        if len(vals) != nu:
            print(f"[ctrl] ignoring set_target_ctrl: got {len(vals)} values, "
                  f"expected {nu}", flush=True)
            return
        with robot["ctrl_arr"].get_lock():
            np.frombuffer(robot["ctrl_arr"].get_obj(), dtype=np.float64)[:nu] = vals
            robot["ctrl_seq"].value = (robot["ctrl_seq"].value + 1) & 0xFFFFFFFF

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
        await hub.broadcast(encode_meta(0, {
            **make_meta_payload(),
            "model_state": make_model_state_payload(),
            "sam_state": make_sam_state_payload(),
        }))
        await asyncio.to_thread(spawn_depth, state["model"])
        await asyncio.to_thread(spawn_seg, sam_state["model"])

    spawn_depth(args.model)
    spawn_seg(args.sam_model)

    # ---- Sim worker (mode == sim) ----------------------------------------
    sim: dict = {"proc": None, "stop_ev": None}

    if args.mode == "sim":
        from sim import sim_worker, mj_camera_params
        # Sim worker uses sim_calib (ground truth) for camera placement +
        # optics. The depth pipeline elsewhere uses sess["calib"] (belief).
        cam_pos, cam_quat, cam_fovy = mj_camera_params(sim_calib)
        sim_stop_ev = mp.Event()
        sim_proc = mp.Process(
            target=sim_worker,
            args=(args.mjcf, args.sim_camera),
            kwargs={
                "rgb_shm_name": sess["shm"].rgb.name,
                "rgb_seq":      sess["shm"].rgb_seq,
                "rgb_w":        sim_calib.intrinsics.width,
                "rgb_h":        sim_calib.intrinsics.height,
                "qpos_arr":     robot["shm"].qpos,
                "qpos_seq":     robot["shm"].qpos_seq,
                "nq":           robot["shm"].nq,
                "ctrl_arr":     robot["ctrl_arr"],
                "ctrl_seq":     robot["ctrl_seq"],
                "nu":           robot["nu"],
                "cam_pos":      tuple(float(x) for x in cam_pos),
                "cam_quat_wxyz": tuple(float(x) for x in cam_quat),
                "cam_fovy_deg": float(cam_fovy),
                # Camera-depth side channel: only allocated when depth_stream
                # shm exists (build_session sets this up in sim mode).
                "depth_shm_name": (sess["shm"].depth_stream.name
                                   if sess["shm"].depth_stream is not None else None),
                "rgbd_seq":     sess["shm"].rgbd_seq,
                "stop_ev":      sim_stop_ev,
                "open_viewer":  bool(args.mujoco_gui),
            },
            daemon=True,
        )
        sim_proc.start()
        sim.update(proc=sim_proc, stop_ev=sim_stop_ev)
        print(f"[sim] worker started (camera={args.sim_camera}, "
              f"viewer={'on' if args.mujoco_gui else 'off'})", flush=True)

    def stop_sim() -> None:
        if sim["stop_ev"] is not None:
            sim["stop_ev"].set()
        if sim["proc"] is not None:
            sim["proc"].join(timeout=5.0)
            if sim["proc"].is_alive():
                sim["proc"].terminate()

    # ---- Hardware worker (--robot-source hardware) -----------------------
    hw_state: dict = {
        "proc": None,
        "stop_ev": None,
        "reset_ev": None,
        "reset_done_ev": None,
        "shm_target": None,
        "shm_gains": None,
        "shm_hz": None,
        "shm_gripper": None,
        "hz": 0.0,
    }

    if args.robot_source == "hardware":
        from hardware import (
            real_robot_process, PinocchioArm, kinova_deg_to_rad,
            HOME_DEG, GAINS_KEYS,
        )

        # Read-only seed: compute home EE pose and write into shm_target so
        # the OSC loop tracks itself at home (no motion until/unless someone
        # writes a new target).
        try:
            arm_init = PinocchioArm(args.robot_arm_mjcf, ee_frame="pinch_site")
        except Exception as exc:
            print(f"[hw] failed to load arm MJCF '{args.robot_arm_mjcf}': "
                  f"{exc}", flush=True)
            return
        home_q = kinova_deg_to_rad(HOME_DEG)
        home_pos, home_rot = arm_init.fk(home_q)
        # Pinocchio uses XYZW quat convention; build it from rotation matrix
        # via SciPy-equivalent path. We just need a valid xyzw: convert via
        # our own helper.
        wxyz = matrix_to_quat_wxyz(home_rot)
        # OSC expects target = [pos(3), quat_xyzw(4)]
        target_init = np.zeros(7, dtype=np.float64)
        target_init[:3] = home_pos
        target_init[3:6] = wxyz[1:4]    # x, y, z
        target_init[6]   = wxyz[0]      # w
        del arm_init  # free pinocchio resources held in this process

        # Conservative defaults — same as cam_calib_old.
        gains_init = np.array([5.0, 0.0, 1.0, 0.0, 10.0, 2.0, 0.0],
                              dtype=np.float64)
        assert len(gains_init) == len(GAINS_KEYS)

        shm_target  = mp.Array("d", target_init, lock=True)
        shm_gains   = mp.Array("d", gains_init, lock=True)
        # lock=True so .get_obj() works for np.frombuffer in stream_loop.
        shm_hz      = mp.Array("d", 1, lock=True)
        shm_gripper = mp.Array("d", 1, lock=True)
        hw_stop_ev      = mp.Event()
        hw_reset_ev     = mp.Event()
        hw_reset_done   = mp.Event()
        hw_state.update(
            stop_ev=hw_stop_ev, reset_ev=hw_reset_ev, reset_done_ev=hw_reset_done,
            shm_target=shm_target, shm_gains=shm_gains,
            shm_hz=shm_hz, shm_gripper=shm_gripper,
        )

        # Reuse the existing robot.shm.qpos slot so the OSC loop's joint-angle
        # output flows through the same FK + transform broadcast as sim/dummy.
        # OSC writes 7-DOF arm (rad); the rest of robot.shm.nq is zero/unused
        # for the FK display (gripper joints stay at 0 until we wire grippers).
        # If nq != 7 we need a tiny adapter: write into the first 7 slots.
        if robot["shm"].nq < 7:
            print(f"[hw] robot shm has nq={robot['shm'].nq}, expected >=7",
                  flush=True)
            return

        # The OSC loop expects shm_q to be 7-long. We have a 15-DOF qpos shm
        # (arm + gripper joints) — slice it. Easiest: pass a separate small
        # 7-DOF shm and copy into the main qpos slot in stream_loop.
        shm_q_arm = mp.Array("d", 7, lock=True)
        hw_state["shm_q_arm"] = shm_q_arm

        hw_proc = mp.Process(
            target=real_robot_process,
            args=(args.robot_ip, args.robot_arm_mjcf,
                  shm_q_arm, shm_target, shm_gains, shm_hz, shm_gripper,
                  hw_stop_ev, hw_reset_ev, hw_reset_done),
            kwargs={"ee_frame": "pinch_site"},
            daemon=True,
        )
        hw_proc.start()
        hw_state["proc"] = hw_proc
        print(f"[hw] OSC process started (ip={args.robot_ip}, "
              f"mjcf={args.robot_arm_mjcf})", flush=True)

    def stop_hardware() -> None:
        if hw_state["stop_ev"] is not None:
            hw_state["stop_ev"].set()
        if hw_state["proc"] is not None:
            # The OSC subprocess's finally: drives back to home in position
            # mode (torque-off → high-level → clear-faults → JointMove home →
            # disconnect). That can take ~15 s; give it 30 before forcing.
            print("[hw] waiting for OSC subprocess to park the arm…",
                  flush=True)
            hw_state["proc"].join(timeout=30.0)
            if hw_state["proc"].is_alive():
                print("[hw] OSC subprocess didn't exit cleanly; terminating",
                      flush=True)
                hw_state["proc"].terminate()
                hw_state["proc"].join(timeout=2.0)
            else:
                print("[hw] OSC subprocess parked.", flush=True)

    async def ws_loop():
        def get_connect_frames() -> list[bytes]:
            meta = encode_meta(0, {
                **make_meta_payload(),
                "model_state": make_model_state_payload(),
                "sam_state": make_sam_state_payload(),
            })
            frames = [meta]
            if robot["enabled"] and robot["geom_payload"]:
                frames.append(robot["geom_payload"])
            return frames
        async with websockets.serve(
            lambda ws: handler(
                ws, hub, get_connect_frames,
                on_set_model, on_set_sam_model,
                on_sam_click, on_sam_clear,
                on_set_source,
                on_set_cam_extrinsics, on_save_cam_extrinsics,
                on_reload_cam_extrinsics,
                on_set_target_ctrl,
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
        last_rgb_seq = 0
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
                    await hub.broadcast(encode_model_state(0, make_model_state_payload()))

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
                    await hub.broadcast(encode_sam_state(0, make_sam_state_payload()))

            # Real-mode: pull RGB from cam and write to shm + bump seq.
            # Sim-mode: cam.get() returns None — the sim worker writes shm
            # and bumps seq directly. Either way, the rgb_seq-advance branch
            # below broadcasts the JPEG.
            rgb = cam.get()
            if rgb is not None:
                rgb_buf[...] = rgb
                with shm.rgb_seq.get_lock():
                    shm.rgb_seq.value += 1

            # If the camera supports depth, copy the latest depth frame into
            # the side-channel shm and bump rgbd_seq. The camera-depth backend
            # (when selected) reads from there instead of running an NN.
            depth_stream_arr = shm.depth_stream_arr()
            if depth_stream_arr is not None:
                if hasattr(cam, "get_depth"):
                    depth_m = cam.get_depth()
                    if depth_m is not None:
                        depth_stream_arr[...] = depth_m
                        with shm.rgbd_seq.get_lock():
                            shm.rgbd_seq.value += 1
                # Sim mode: the sim worker writes depth_stream + rgbd_seq
                # directly (task 17), nothing to do here.

            with shm.rgb_seq.get_lock():
                cur_rgb = shm.rgb_seq.value
            if cur_rgb != last_rgb_seq:
                last_rgb_seq = cur_rgb
                bgr = cv2.cvtColor(rgb_buf, cv2.COLOR_RGB2BGR)
                await hub.broadcast(encode_jpeg(seq, bgr))
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
                # Recompute T_world_camera every frame so live calibration
                # updates ripple through immediately.
                T_wc = sess["calib"].T_world_camera()
                normal_payload = (
                    _xform_normals(mesh_normal, T_wc) if have_n else None
                )
                if n > 0:
                    pc_mask = seg_mask[pc_grid_idx[:n]].copy()
                    pc_world = _xform_points(pc_xyz[:n], T_wc)
                    mesh_world = _xform_points(mesh_xyz, T_wc, mask_zero=True)
                    await hub.broadcast(
                        encode_points(seq, pc_world, pc_rgb[:n].copy(),
                                      mask=pc_mask)
                    )
                    seq += 1
                    await hub.broadcast(
                        encode_mesh(seq, mesh_world, mesh_rgb.copy(),
                                    mesh_faces.copy(), normal=normal_payload)
                    )
                    seq += 1
                # Always send a colorized depth jpeg, even when pc is empty.
                depth_bgr = _depth_to_turbo_bgr(depth_buf)
                await hub.broadcast(
                    encode_jpeg(seq, depth_bgr, kind=KIND_DEPTH_JPEG, quality=80)
                )
                seq += 1
                # And a normal-map jpeg if the model produces normals.
                if have_n:
                    normal_bgr = _normal_to_bgr(normal_img)
                    await hub.broadcast(
                        encode_jpeg(seq, normal_bgr, kind=KIND_NORMAL_JPEG, quality=80)
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
                    encode_mask(seq, cur_m, seg_mask.copy(), has_box, box_min, box_max)
                )
                seq += 1

            # Hardware mode: copy the OSC's 7-DOF arm angles (rad) into the
            # main qpos slot and bump qpos_seq so the FK + transforms branch
            # below picks them up, same as sim/dummy.
            if (args.robot_source == "hardware"
                    and hw_state.get("shm_q_arm") is not None
                    and robot["shm"] is not None):
                with hw_state["shm_q_arm"].get_lock():
                    arm_q = np.frombuffer(hw_state["shm_q_arm"].get_obj(),
                                          dtype=np.float64).copy()
                # Don't write a frame of zeros (OSC hasn't reported yet).
                if np.any(arm_q):
                    cur_qpos = robot["shm"].read_qpos()
                    cur_qpos[:7] = arm_q
                    robot["shm"].write_qpos(cur_qpos)

            # Robot transforms — broadcast whenever qpos_seq advances.
            if robot["enabled"] and robot["shm"] is not None:
                cur_q = robot["shm"].read_seq()
                if cur_q != robot["last_seq"]:
                    robot["last_seq"] = cur_q
                    qpos = robot["shm"].read_qpos()
                    xpos, xquat = robot["fk"].compute(qpos)
                    await hub.broadcast(encode_robot_transforms(seq, xpos, xquat))
                    seq += 1

            if time.time() - t_log >= 1.0:
                dt = time.time() - t_log
                rgb_fps = n_rgb / dt
                depth_fps = n_depth / dt
                print(f"[ws] rgb {rgb_fps:.1f}  depth {depth_fps:.1f}  "
                      f"clients={len(hub.clients)}", flush=True)
                await hub.broadcast(encode_stats(seq, {
                    "rgb_fps":   round(rgb_fps, 1),
                    "depth_fps": round(depth_fps, 1),
                    "sam_ms":    sam_last_ms["v"],
                }))
                seq += 1
                # Hardware mode: broadcast OSC rate.
                if args.robot_source == "hardware" and hw_state["shm_hz"] is not None:
                    try:
                        osc_hz = float(np.frombuffer(hw_state["shm_hz"].get_obj(),
                                                     dtype=np.float64)[0])
                    except (AttributeError, ValueError):
                        osc_hz = 0.0
                    proc_alive = bool(hw_state["proc"] is not None
                                      and hw_state["proc"].is_alive())
                    await hub.broadcast(encode_robot_status(seq, {
                        "source": "hardware",
                        "osc_hz": round(osc_hz, 1),
                        "alive":  proc_alive,
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
        if robot["src"] is not None:
            robot["src"].stop()
        stop_sim()
        stop_hardware()
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
    ap.add_argument("--mjcf", default=None,
                    help="Path to an MJCF scene to display a robot. "
                         "If omitted, no robot is rendered.")
    ap.add_argument("--robot-source", default="none",
                    choices=["none", "dummy", "sim", "hardware"],
                    dest="robot_source",
                    help="Where qpos comes from. 'none' = stays at home. "
                         "'dummy' = sine animation (test only). 'sim' = MuJoCo "
                         "physics process; implied by --mode sim. "
                         "'hardware' = real Kinova OSC loop (--mode real only).")
    ap.add_argument("--robot-ip", default="192.168.1.10", dest="robot_ip",
                    help="Kinova hardware IP (for --robot-source hardware).")
    ap.add_argument("--robot-arm-mjcf", default="robot/mjcf/gen3_gripper.xml",
                    dest="robot_arm_mjcf",
                    help="Bare-arm MJCF for Pinocchio dynamics in hardware "
                         "mode. Pinocchio chokes on full scene MJCFs that "
                         "include world geoms (e.g. floor planes), so this "
                         "must be an arm-only file.")
    ap.add_argument("--mode", default="real", choices=["real", "sim"],
                    help="'real' uses a physical camera (RealSense or HTTP). "
                         "'sim' replaces the camera with a MuJoCo render of "
                         "--mjcf and runs physics in a worker process.")
    ap.add_argument("--sim-camera", default="ext_rgbd",
                    dest="sim_camera",
                    help="Named MJCF camera to render in sim mode.")
    ap.add_argument("--mujoco-gui", action="store_true", dest="mujoco_gui",
                    help="Open the MuJoCo native passive viewer alongside "
                         "the web frontend (sim mode only).")
    args = ap.parse_args()
    if args.robot_source != "none" and not args.mjcf:
        ap.error("--robot-source requires --mjcf")
    if args.mode == "sim":
        if not args.mjcf:
            ap.error("--mode sim requires --mjcf")
        if args.robot_source == "hardware":
            ap.error("--robot-source hardware is not supported with --mode sim")
        # Sim mode is its own qpos producer.
        args.robot_source = "sim"
    if args.robot_source == "hardware" and args.mode != "real":
        ap.error("--robot-source hardware requires --mode real")
    if args.mujoco_gui and args.mode != "sim":
        ap.error("--mujoco-gui requires --mode sim")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
