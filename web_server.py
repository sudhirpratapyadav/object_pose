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
from depth import (
    BACKENDS, create_shm, depth_worker, DEFAULT_MODEL,
    resolve_default_model,
)
from depth.backends import CameraReq, FOUNDATION_STEREO_KEYS
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
from vision import compute_object_pose
from robot.wire import build_robot_geometry_payload
from wire import (
    KIND_DEPTH_JPEG,
    KIND_NORMAL_JPEG,
    encode_cam_calib,
    encode_controller_state,
    encode_jpeg,
    encode_log_lines,
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
                  on_set_target_ctrl,
                  on_set_controller, on_stop_controller, on_home_robot,
                  on_recover_robot, on_restart_transport,
                  on_set_joint_target,
                  on_set_ee_target,
                  on_set_controller_gains, on_save_controller_gains,
                  on_reload_controller_gains,
                  on_set_policy, on_stop_policy,
                  on_save_policy_configs, on_reload_policy_configs) -> None:
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
                elif "set_controller" in cmd:
                    await on_set_controller(str(cmd["set_controller"]))
                elif "stop_controller" in cmd:
                    await on_stop_controller()
                elif "home_robot" in cmd:
                    await on_home_robot()
                elif "recover_robot" in cmd:
                    await on_recover_robot()
                elif "restart_transport" in cmd:
                    await on_restart_transport()
                elif "set_joint_target" in cmd and isinstance(cmd["set_joint_target"], list):
                    vals = [float(x) for x in cmd["set_joint_target"]]
                    await on_set_joint_target(vals)
                elif "set_ee_target" in cmd and isinstance(cmd["set_ee_target"], dict):
                    e = cmd["set_ee_target"]
                    pos = e.get("pos") or []
                    quat = e.get("quat_xyzw") or []
                    if len(pos) == 3 and len(quat) == 4:
                        await on_set_ee_target(
                            [float(x) for x in pos],
                            [float(x) for x in quat],
                        )
                elif "set_controller_gains" in cmd and isinstance(cmd["set_controller_gains"], dict):
                    g = cmd["set_controller_gains"]
                    n = str(g.get("name", ""))
                    kp = [float(x) for x in (g.get("kp") or [])]
                    kd = [float(x) for x in (g.get("kd") or [])]
                    await on_set_controller_gains(n, kp, kd)
                elif "save_controller_gains" in cmd:
                    # Client sends the controller name as the value.
                    n = str(cmd["save_controller_gains"])
                    await on_save_controller_gains(n)
                elif "reload_controller_gains" in cmd:
                    await on_reload_controller_gains()
                elif "set_policy" in cmd:
                    await on_set_policy(str(cmd["set_policy"]))
                elif "stop_policy" in cmd:
                    await on_stop_policy()
                elif "save_policy_configs" in cmd:
                    await on_save_policy_configs()
                elif "reload_policy_configs" in cmd:
                    await on_reload_policy_configs()
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
        # Boot in 'rgbd' mode so the camera-depth backend (the default boot
        # pick) has data ready. on_set_model() will reopen the camera if
        # the user switches to FoundationStereo or a learned monocular.
        cam = RealSenseRGB(
            width=cam_calib.intrinsics.width,
            height=cam_calib.intrinsics.height,
            fps=CAM_FPS,
            mode="rgbd",
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
        # All side channels (depth_stream, IR pair) are always allocated
        # so we can switch camera modes without rebuilding shm.
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

    def _camera_supports_depth() -> bool:
        """Whether the active camera CAN produce depth (regardless of
        current mode). Sim always can; D4xx with reopen() can; video can't.
        """
        if args.mode == "sim":
            return True
        cam = sess["cam"]
        # Real RealSense — has reopen() and at least one of has_depth / a
        # past stereo_calib indicates it's a depth-capable device. Today
        # we just rely on the class type: anything with reopen() is a
        # mode-capable camera.
        return hasattr(cam, "reopen")

    def _camera_supports_stereo() -> bool:
        """Whether the active camera CAN produce a rectified IR pair."""
        if args.mode == "sim":
            return False
        cam = sess["cam"]
        if not hasattr(cam, "reopen"):
            return False
        # Quickest signal: if we're already in stereo mode, the calib
        # payload is populated. Otherwise we don't know without trying.
        # Treat all RealSense as candidates — D4xx ships stereo IR.
        return True

    def _model_camera_reqs() -> dict[str, str]:
        """Map of model_key -> camera_req string ("rgb" | "rgbd" | "rgb_stereo").
        Used by the UI to grey out unsupported entries in the dropdown.
        """
        out: dict[str, str] = {}
        for key, factory in BACKENDS.items():
            try:
                info = factory(1.0).info
            except Exception:
                info = getattr(factory, "info", None)
            req = getattr(info, "camera_req", CameraReq.RGB_ONLY)
            out[key] = req.value if hasattr(req, "value") else str(req)
        return out

    # Resolve the depth-model default now that we know whether the camera
    # supplied a depth stream. Honoured by the rest of the boot path
    # (state["model"], make_meta_payload(), spawn_depth(args.model)).
    cam_depth_avail = _camera_supports_depth()
    args.model = resolve_default_model(cam_depth_avail, args.model)
    print(f"[depth] default model: {args.model} "
          f"(camera_depth_available={cam_depth_avail})", flush=True)

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
            "models":         list(BACKENDS.keys()),
            "model_camera_reqs": _model_camera_reqs(),
            "default_model":  args.model,
            "camera_depth_available": _camera_supports_depth(),
            "camera_stereo_available": _camera_supports_stereo(),
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
        depth_stream_name = shm.depth_stream.name
        # Mode-aware label so the dropdown reads naturally.
        depth_label = ("Camera depth (MuJoCo)" if args.mode == "sim"
                       else "Camera depth (RealSense)")

        # FoundationStereo wiring: pull stereo calib from the active camera.
        ir_left_name = ir_right_name = None
        stereo_seq = None
        stereo_calib_payload = None
        if model_key in FOUNDATION_STEREO_KEYS:
            cam = sess["cam"]
            sc = getattr(cam, "stereo_calib", None)
            if sc is None:
                # Should never reach here — _camera_mode_for() would have
                # rejected the switch earlier. Defensive log + skip.
                print(f"[depth] {model_key} requested but camera has no stereo calib;"
                      f" worker will fail and stay alive for re-pick.",
                      flush=True)
            else:
                ir_left_name  = shm.ir_left.name
                ir_right_name = shm.ir_right.name
                stereo_seq    = shm.stereo_seq
                # Also include color intrinsics so the FS backend can
                # warp depth into the color frame the rest of the
                # pipeline uses.
                ci = sess["intr"]
                stereo_calib_payload = {
                    "fx_ir": sc.fx, "fy_ir": sc.fy,
                    "cx_ir": sc.cx, "cy_ir": sc.cy,
                    "ir_w": sc.width, "ir_h": sc.height,
                    "baseline_m":   sc.baseline_m,
                    "ir_to_color_R": sc.ir_to_color_R.tolist(),
                    "ir_to_color_t": sc.ir_to_color_t.tolist(),
                    "fx_color": ci.fx, "fy_color": ci.fy,
                    "cx_color": ci.cx, "cy_color": ci.cy,
                    "color_w":  ci.width, "color_h":  ci.height,
                }

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
            kwargs=dict(
                rgbd_seq=shm.rgbd_seq,
                ir_left_name=ir_left_name,
                ir_right_name=ir_right_name,
                stereo_seq=stereo_seq,
                stereo_calib=stereo_calib_payload,
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

    def _camera_mode_for(key: str) -> str | None:
        """Pick a camera mode that satisfies the backend's CameraReq.

        Returns the mode string ("rgb" / "rgbd" / "rgb_stereo") or None if
        the active camera can't satisfy this backend's requirement. The
        caller treats None as a hard reject (stay on current model).
        """
        info = None
        factory = BACKENDS.get(key)
        if factory is not None:
            try:
                info = factory(1.0).info  # cheap metadata-only call
            except Exception:
                # Sentinel factories raise; fall back to scanning attrs.
                info = getattr(factory, "info", None)
        req = getattr(info, "camera_req", CameraReq.RGB_ONLY)
        cam = sess["cam"]
        if req == CameraReq.RGB_ONLY:
            return "rgb"
        if req == CameraReq.RGB_DEPTH:
            # Only D4xx + sim can produce factory-aligned depth. Sim mode
            # uses the sim worker's depth, not a camera reopen.
            if args.mode == "sim":
                return "rgb"
            if hasattr(cam, "reopen"):
                return "rgbd"
            return None
        if req == CameraReq.RGB_STEREO:
            if hasattr(cam, "reopen") and getattr(cam, "stereo_calib", None) is not None:
                return "rgb_stereo"
            # Active camera doesn't support stereo and we have no way to
            # try (e.g. video file source).
            if hasattr(cam, "reopen"):
                return "rgb_stereo"   # let reopen() try; failure → revert
            return None
        return None

    async def on_set_model(key: str) -> None:
        if key == state["model"] or key not in BACKENDS:
            return
        # Pick target camera mode for this backend.
        target_cam_mode = _camera_mode_for(key)
        if target_cam_mode is None:
            # Reject up front: the active camera can't satisfy this backend.
            err = (f"'{key}' requires a camera that supports its capture "
                   f"requirement; active camera doesn't.")
            print(f"[depth] {err}", flush=True)
            state.update(model_status="error", model_progress="", model_file=err[:200])
            await hub.broadcast(encode_model_state(0, make_model_state_payload()))
            # Restore previous status after a beat so the UI doesn't stick
            # in 'error'.
            state.update(model_status=f"running {state['model']}",
                         model_progress="", model_file="")
            await hub.broadcast(encode_model_state(0, make_model_state_payload()))
            return

        state.update(model_status=f"switching to {key} ...",
                     model_progress="", model_file="")
        await hub.broadcast(encode_model_state(0, make_model_state_payload()))
        # Tear down the depth worker first so it doesn't read partially
        # filled shm during the camera flip.
        await asyncio.to_thread(stop_depth)
        shm = sess["shm"]
        with shm.pc_count.get_lock():
            shm.pc_count.value = 0

        # Reopen the camera in the new mode if it differs (real-camera path).
        cam = sess["cam"]
        prev_mode = getattr(cam, "mode", None)
        if (prev_mode is not None and prev_mode != target_cam_mode
                and hasattr(cam, "reopen")):
            state.update(model_status=f"reconfiguring camera ({target_cam_mode}) ...")
            await hub.broadcast(encode_model_state(0, make_model_state_payload()))
            try:
                new_intr = await asyncio.to_thread(cam.reopen, target_cam_mode)
                # Refresh inference intrinsics — width/height haven't changed
                # but fx/fy/cx/cy might (D4xx ships per-stream intrinsics).
                sx = INFER_W / new_intr.width
                sy = INFER_H / new_intr.height
                sess["intr"] = new_intr
                sess["fx_i"] = new_intr.fx * sx
                sess["fy_i"] = new_intr.fy * sy
                sess["cx_i"] = new_intr.cx * sx
                sess["cy_i"] = new_intr.cy * sy
                # Push fresh meta to clients (intrinsics may have changed).
                await hub.broadcast(encode_meta(0, {
                    **make_meta_payload(),
                    "model_state": make_model_state_payload(),
                    "sam_state":   make_sam_state_payload(),
                }))
            except Exception as exc:
                err = f"camera reopen('{target_cam_mode}') failed: {exc}"
                print(f"[depth] {err}", flush=True)
                # Try to revert to the previous mode so the rest of the
                # pipeline keeps running.
                try:
                    await asyncio.to_thread(cam.reopen, prev_mode)
                except Exception as exc2:
                    print(f"[cam] revert reopen also failed: {exc2}", flush=True)
                state.update(model_status="error", model_progress="", model_file=err[:200])
                await hub.broadcast(encode_model_state(0, make_model_state_payload()))
                # Re-spawn the *previous* model so depth keeps flowing.
                await asyncio.to_thread(spawn_depth, state["model"])
                return

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

    # Sim worker spawn is deferred until after hw_state shms are allocated
    # (it bridges them when --robot-source sim). See below the hw_state
    # block.

    def stop_sim() -> None:
        if sim["stop_ev"] is not None:
            sim["stop_ev"].set()
        if sim["proc"] is not None:
            sim["proc"].join(timeout=5.0)
            if sim["proc"].is_alive():
                sim["proc"].terminate()

    # ---- Hardware: transport process + controller dispatcher ------------
    from collections import deque
    fault_history: deque = deque(maxlen=32)
    hw_state: dict = {
        # Transport process + its shms.
        "transport_proc": None,
        "transport_stop": None,
        "shm_q":          None,    # joint angles from robot (rad)
        "shm_dq":         None,    # joint velocities (rad/s)
        "shm_state_seq":  None,
        "shm_cmd_mode":   None,
        "shm_tau":        None,
        "shm_qtarget":    None,
        "shm_gripper":    None,
        "shm_cmd_seq":    None,
        "shm_hz":         None,
        "shm_phase":      None,
        "shm_fault_msg":  None,
        "swap_request_ev": None,
        "swap_target_mode": None,
        "swap_done_ev":   None,
        # Controller process state.
        "ctrl_name":   "idle",
        "ctrl_proc":   None,
        "ctrl_stop":   None,
        "ctrl_status": "idle",   # 'idle' | 'loading' | 'running' | 'stopping' | 'fault'
        "ctrl_error":  "",
    }

    def _read_ee_target() -> dict | None:
        """Snapshot shm_qtarget[:7] = [pos(3), quat_xyzw(4)] when meaningful.

        Returns None if hardware isn't loaded. Always present (even when not
        running ee_pose) — the gizmo is hidden client-side based on the
        current controller; the value here is just the last-written setpoint.
        """
        sm = hw_state.get("shm_qtarget")
        if sm is None:
            return None
        with sm.get_lock():
            arr = np.frombuffer(sm.get_obj(), dtype=np.float64)[:7].copy()
        return {
            "pos":       [float(arr[0]), float(arr[1]), float(arr[2])],
            "quat_xyzw": [float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])],
        }

    def _read_policy_block() -> dict:
        """Build the per-broadcast policy snapshot."""
        from policies import POLICIES
        # Live stats (rate + last action + status).
        hz = 0.0
        last_act = [0.0] * 7
        status_code = 0
        if hw_state.get("shm_policy_hz") is not None:
            with hw_state["shm_policy_hz"].get_lock():
                hz = float(np.frombuffer(hw_state["shm_policy_hz"].get_obj(),
                                          dtype=np.float64)[0])
            with hw_state["shm_policy_last_action"].get_lock():
                last_act = [float(x) for x in np.frombuffer(
                    hw_state["shm_policy_last_action"].get_obj(),
                    dtype=np.float64)[:7]]
            with hw_state["shm_policy_status"].get_lock():
                status_code = int(np.frombuffer(
                    hw_state["shm_policy_status"].get_obj(),
                    dtype=np.uint8)[0])
        # Current goal (world frame).
        goal = [0.0, 0.0, 0.0]
        if hw_state.get("shm_policy_goal") is not None:
            with hw_state["shm_policy_goal"].get_lock():
                goal = [float(x) for x in np.frombuffer(
                    hw_state["shm_policy_goal"].get_obj(),
                    dtype=np.float64)[:3]]
        # Latest object pose (world frame).
        obj_pose = None
        if hw_state.get("shm_object_pose") is not None:
            with hw_state["shm_object_pose_seq"].get_lock():
                obj_seq = int(hw_state["shm_object_pose_seq"].value)
            if obj_seq > 0:
                with hw_state["shm_object_pose"].get_lock():
                    arr = np.frombuffer(hw_state["shm_object_pose"].get_obj(),
                                         dtype=np.float64)[:4].copy()
                obj_pose = {"pos": arr[:3].tolist(), "n_points": int(arr[3])}
        return {
            "available": [
                {"name": info.name, "display_name": info.display_name,
                 "description": info.description,
                 "controller": info.controller,
                 "needs_object_pose": info.needs_object_pose}
                for (info, _) in POLICIES.values()
            ],
            "current":     hw_state.get("policy_name", "") or "",
            "status_code": status_code,        # 0 waiting, 1 running, 2 success
            "last_error":  hw_state.get("policy_error", ""),
            "configs":     hw_state.get("pol_configs", {}),
            "hz":          round(hz, 1),
            "last_action": last_act,
            "goal":        goal,
            "object_pose": obj_pose,
        }

    def _broadcast_controller_state_sync():
        """Build the controller-state payload (called from sync code paths)."""
        from controllers import CONTROLLERS
        return {
            "available": [
                {"name": info.name, "display_name": info.display_name,
                 "description": info.description, "command_mode": info.command_mode}
                for (info, _) in CONTROLLERS.values()
            ],
            "current": hw_state["ctrl_name"],
            "status":  hw_state["ctrl_status"],
            "last_error": hw_state["ctrl_error"],
            "configs":  hw_state.get("ctrl_configs", {}),
            "ee_target": _read_ee_target(),
            "policy":   _read_policy_block() if args.robot_source in ("hardware", "sim") else None,
        }

    async def _broadcast_controller_state():
        await hub.broadcast(encode_controller_state(0, _broadcast_controller_state_sync()))

    # Sim mode acts as a drop-in for hardware (same controller dispatcher,
    # same shm protocol, same policy plumbing). The only difference is who
    # runs the 500 Hz state-pub / torque-apply loop: the kortex transport
    # process for real, or the MuJoCo sim worker (with bridge kwargs) for sim.
    if args.robot_source in ("hardware", "sim"):
        from hardware import (
            transport_process,
            CMD_MODE_IDLE, CMD_MODE_TORQUE, CMD_MODE_POSITION,
        )

        # Shared log queue: subprocesses push records, stream_loop drains.
        log_q = mp.Queue(maxsize=2000)

        # Allocate all shms the transport + controllers share.
        shm_q          = mp.Array("d", 7, lock=True)
        shm_dq         = mp.Array("d", 7, lock=True)
        shm_state_seq  = mp.Value("I", 0, lock=True)
        shm_cmd_mode   = mp.Value("B", CMD_MODE_IDLE, lock=True)
        shm_tau        = mp.Array("d", 7, lock=True)
        shm_qtarget    = mp.Array("d", 7, lock=True)
        shm_gripper    = mp.Array("d", 1, lock=True)
        shm_cmd_seq    = mp.Value("I", 0, lock=True)
        # Controller gains slot: 14 doubles. Layout depends on the
        # active controller (e.g. joint_pd uses kp[7] || kd[7]).
        shm_gains      = mp.Array("d", 14, lock=True)
        shm_hz         = mp.Array("d", 1, lock=True)
        shm_phase      = mp.Value("B", 0, lock=True)
        shm_fault_msg  = mp.Array("c", 256, lock=True)
        swap_request_ev = mp.Event()
        swap_target_mode = mp.Value("B", CMD_MODE_IDLE, lock=True)
        swap_done_ev    = mp.Event()
        transport_stop  = mp.Event()

        # Controller defaults loaded from YAML (per-machine tuning).
        from controllers.configs import load_configs as load_ctrl_cfg
        ctrl_configs = load_ctrl_cfg()

        # Policy shared state. Always allocated; the policy subprocess
        # reads/writes these. shm_object_pose is also written by the
        # stream_loop (vision side); shm_policy_goal is set by the
        # server at engage time.
        shm_object_pose       = mp.Array("d", 4, lock=True)   # [x, y, z, n_points]
        shm_object_pose_seq   = mp.Value("Q", 0, lock=True)
        shm_policy_goal       = mp.Array("d", 3, lock=True)
        shm_policy_hz         = mp.Array("d", 1, lock=True)
        shm_policy_last_action = mp.Array("d", 7, lock=True)
        shm_policy_status     = mp.Array("B", 1, lock=True)   # 0=waiting,1=running,2=success

        # Policy defaults loaded from YAML.
        from policies.configs import load_configs as load_pol_cfg
        pol_configs = load_pol_cfg()

        hw_state.update(
            shm_q=shm_q, shm_dq=shm_dq, shm_state_seq=shm_state_seq,
            shm_cmd_mode=shm_cmd_mode, shm_tau=shm_tau,
            shm_qtarget=shm_qtarget, shm_gripper=shm_gripper,
            shm_cmd_seq=shm_cmd_seq, shm_hz=shm_hz,
            shm_phase=shm_phase, shm_fault_msg=shm_fault_msg,
            swap_request_ev=swap_request_ev, swap_target_mode=swap_target_mode,
            swap_done_ev=swap_done_ev, transport_stop=transport_stop,
            shm_gains=shm_gains,
            ctrl_configs=ctrl_configs,
            shm_object_pose=shm_object_pose,
            shm_object_pose_seq=shm_object_pose_seq,
            shm_policy_goal=shm_policy_goal,
            shm_policy_hz=shm_policy_hz,
            shm_policy_last_action=shm_policy_last_action,
            shm_policy_status=shm_policy_status,
            pol_configs=pol_configs,
            # Policy run state (server-owned)
            policy_name="",          # "" = no policy engaged
            policy_proc=None,
            policy_stop=None,
            policy_error="",
            # Snapshot of the user's saved ee_pose gains so we can
            # restore them when a policy disengages.
            saved_ee_pose_gains=None,
            log_q=log_q,
        )

        if args.robot_source == "hardware":
            transport_proc = mp.Process(
                target=transport_process,
                args=(args.robot_ip,),
                kwargs={
                    "shm_q": shm_q, "shm_dq": shm_dq, "shm_state_seq": shm_state_seq,
                    "shm_cmd_mode": shm_cmd_mode, "shm_tau": shm_tau,
                    "shm_qtarget": shm_qtarget, "shm_gripper": shm_gripper,
                    "shm_cmd_seq": shm_cmd_seq,
                    "stop_ev": transport_stop,
                    "swap_request_ev": swap_request_ev,
                    "swap_target_mode": swap_target_mode,
                    "swap_done_ev": swap_done_ev,
                    "shm_hz": shm_hz, "shm_phase": shm_phase,
                    "shm_fault_msg": shm_fault_msg,
                    "log_q": log_q,
                },
                daemon=True,
            )
            transport_proc.start()
            hw_state["transport_proc"] = transport_proc
            print(f"[hw] transport process started (ip={args.robot_ip})",
                  flush=True)
        else:
            # Sim mode: spawn the sim_worker now with bridge kwargs so it
            # owns the 500 Hz state-pub loop and accepts torque commands.
            from sim import sim_worker, mj_camera_params
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
                    "depth_shm_name": sess["shm"].depth_stream.name,
                    "rgbd_seq":       sess["shm"].rgbd_seq,
                    "stop_ev":      sim_stop_ev,
                    "open_viewer":  bool(args.mujoco_gui),
                    # Hardware-bridge kwargs:
                    "shm_q": shm_q, "shm_dq": shm_dq,
                    "shm_state_seq": shm_state_seq,
                    "shm_cmd_mode": shm_cmd_mode, "shm_tau": shm_tau,
                    "shm_qtarget": shm_qtarget, "shm_gripper": shm_gripper,
                    "shm_cmd_seq": shm_cmd_seq,
                    "swap_request_ev": swap_request_ev,
                    "swap_target_mode": swap_target_mode,
                    "swap_done_ev": swap_done_ev,
                    "shm_hz": shm_hz, "shm_phase": shm_phase,
                    "shm_fault_msg": shm_fault_msg,
                },
                daemon=True,
            )
            sim_proc.start()
            sim.update(proc=sim_proc, stop_ev=sim_stop_ev)
            # Track in hw_state so stop_hardware() can stop it too.
            hw_state["transport_proc"] = sim_proc
            print(f"[sim] worker started (camera={args.sim_camera}, "
                  f"viewer={'on' if args.mujoco_gui else 'off'}, "
                  f"bridge=on)", flush=True)

        # The default 'idle' controller process gets spawned lazily by
        # on_set_controller — start it now so the dispatcher state is
        # consistent.
        # (We do this in main_async after make_meta_payload is defined,
        # so push to a deferred init below.)

    async def on_set_controller(name: str) -> None:
        """SST: stop current controller → home robot → start new controller."""
        if args.robot_source not in ("hardware", "sim"):
            return
        from controllers import CONTROLLERS
        from hardware import CMD_MODE_IDLE, CMD_MODE_TORQUE, CMD_MODE_POSITION

        if name not in CONTROLLERS:
            print(f"[ctrl] unknown controller '{name}'", flush=True)
            return
        if name == hw_state["ctrl_name"] and hw_state["ctrl_status"] == "running":
            return  # no-op

        info, factory = CONTROLLERS[name]

        # --- 0. If a policy is running, stop it. It writes to shm_qtarget;
        #         keeping it alive across an SST would race with the swap.
        if hw_state.get("policy_proc") is not None:
            await on_stop_policy()

        # --- 1. Stop current controller ---
        hw_state["ctrl_status"] = "stopping"
        hw_state["ctrl_error"] = ""
        await _broadcast_controller_state()
        if hw_state["ctrl_stop"] is not None:
            hw_state["ctrl_stop"].set()
        if hw_state["ctrl_proc"] is not None:
            await asyncio.to_thread(_join_controller, hw_state["ctrl_proc"])
        hw_state["ctrl_proc"] = None
        hw_state["ctrl_stop"] = None

        # --- 2. Tell transport: SST to target command mode ---
        # Always go through home pose. target_mode = the new controller's mode.
        mode_map = {
            "idle":     CMD_MODE_IDLE,
            "torque":   CMD_MODE_TORQUE,
            "position": CMD_MODE_POSITION,
        }
        target_mode = mode_map[info.command_mode]
        with hw_state["swap_target_mode"].get_lock():
            hw_state["swap_target_mode"].value = target_mode
        hw_state["swap_done_ev"].clear()
        hw_state["swap_request_ev"].set()
        hw_state["ctrl_name"] = name
        hw_state["ctrl_status"] = "loading"
        await _broadcast_controller_state()
        # Wait for transport to finish the swap.
        ok = await asyncio.to_thread(hw_state["swap_done_ev"].wait, 30.0)
        if not ok:
            hw_state["ctrl_status"] = "fault"
            hw_state["ctrl_error"] = "transport SST timed out"
            await _broadcast_controller_state()
            return

        # --- 3. Pre-fill shm_gains from this controller's defaults ---
        cfg = hw_state.get("ctrl_configs", {}).get(name, {})
        _seed_gains(hw_state["shm_gains"], name, cfg)

        # --- 4. Spawn new controller ---
        log_q = hw_state.get("log_q")
        target_callable, ctrl_kwargs = (
            factory(mjcf_path=args.robot_arm_mjcf, log_q=log_q,
                    shm_gains=hw_state["shm_gains"],
                    robot_source=args.robot_source)
            if info.command_mode != "idle"
            else factory(log_q=log_q)
        )
        ctrl_stop = mp.Event()
        ctrl_proc = mp.Process(
            target=target_callable,
            args=(hw_state["shm_q"], hw_state["shm_dq"], hw_state["shm_state_seq"],
                  hw_state["shm_cmd_mode"], hw_state["shm_tau"],
                  hw_state["shm_qtarget"], hw_state["shm_gripper"],
                  hw_state["shm_cmd_seq"],
                  ctrl_stop),
            kwargs=ctrl_kwargs,
            daemon=True,
        )
        ctrl_proc.start()
        hw_state["ctrl_proc"] = ctrl_proc
        hw_state["ctrl_stop"] = ctrl_stop

        # Wait for the controller to finish seeding its setpoint (every
        # torque controller bumps shm_cmd_seq after the first write). This
        # makes the "running" broadcast contain a meaningful ee_target so
        # the browser gizmo snaps to the real seed instead of zeros.
        if info.command_mode != "idle":
            with hw_state["shm_cmd_seq"].get_lock():
                start_seq = int(hw_state["shm_cmd_seq"].value)

            def _wait_seed():
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    with hw_state["shm_cmd_seq"].get_lock():
                        if int(hw_state["shm_cmd_seq"].value) != start_seq:
                            return True
                    time.sleep(0.005)
                return False
            await asyncio.to_thread(_wait_seed)

        hw_state["ctrl_status"] = "running"
        print(f"[ctrl] running '{name}'", flush=True)
        await _broadcast_controller_state()

    async def on_stop_controller() -> None:
        await on_set_controller("idle")

    async def on_home_robot() -> None:
        """Home = Stop + resume.

        Snapshots the active controller, drives to idle (SST passes
        through home), then re-engages the same controller. Use as a
        'panic, return to home' button: works in any mode, doesn't
        leave the user in idle if they were tuning a controller.

        From idle, just stays in idle (the SST already put us at home).
        """
        prev = hw_state["ctrl_name"]
        await on_set_controller("idle")
        if prev != "idle" and hw_state["ctrl_status"] != "fault":
            await on_set_controller(prev)

    async def on_recover_robot() -> None:
        """Recovery from fault: same as set_controller('idle') today since
        the SST already does clear_faults + JointMove home + high-level."""
        await on_set_controller("idle")

    async def on_set_joint_target(vals: list[float]) -> None:
        """UI publishes a 7-DOF q_des (rad). The active controller (e.g.
        joint_pd) reads this each cycle as its joint setpoint. Bumping
        cmd_seq is harmless for torque-mode controllers (they're already
        running at 500 Hz from state_seq) but matters for position mode.
        """
        if args.robot_source not in ("hardware", "sim"):
            return
        if hw_state["shm_qtarget"] is None or hw_state["shm_cmd_seq"] is None:
            return
        if len(vals) != 7:
            print(f"[ctrl] ignoring set_joint_target: got {len(vals)} values, "
                  f"expected 7", flush=True)
            return
        with hw_state["shm_qtarget"].get_lock():
            np.frombuffer(hw_state["shm_qtarget"].get_obj(),
                          dtype=np.float64)[:7] = vals
        with hw_state["shm_cmd_seq"].get_lock():
            hw_state["shm_cmd_seq"].value = (hw_state["shm_cmd_seq"].value + 1) & 0xFFFFFFFF

    async def on_set_ee_target(pos: list[float], quat_xyzw: list[float]) -> None:
        """UI publishes a 6-DoF EE target (world frame). Layout in
        shm_qtarget matches controllers/ee_pose.py: [pos(3), quat_xyzw(4)].

        Quaternion is normalised; degenerate input is rejected.
        """
        if args.robot_source not in ("hardware", "sim"):
            return
        if hw_state["shm_qtarget"] is None or hw_state["shm_cmd_seq"] is None:
            return
        if len(pos) != 3 or len(quat_xyzw) != 4:
            print(f"[ctrl] ignoring set_ee_target: pos={len(pos)} quat={len(quat_xyzw)}",
                  flush=True)
            return
        q = np.asarray(quat_xyzw, dtype=np.float64)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n < 1e-6:
            print(f"[ctrl] ignoring set_ee_target: degenerate quat", flush=True)
            return
        q = q / n
        with hw_state["shm_qtarget"].get_lock():
            arr = np.frombuffer(hw_state["shm_qtarget"].get_obj(), dtype=np.float64)
            arr[0:3] = pos
            arr[3:7] = q
        with hw_state["shm_cmd_seq"].get_lock():
            hw_state["shm_cmd_seq"].value = (hw_state["shm_cmd_seq"].value + 1) & 0xFFFFFFFF

    async def on_set_controller_gains(name: str, kp: list[float],
                                       kd: list[float]) -> None:
        """Live update controller gains. Writes to shm_gains (the active
        controller reads it next cycle) AND updates the in-memory config
        dict so the next save_controller_gains will persist this state.
        Layout follows _seed_gains: joint_pd packs kp[7]||kd[7], ee_pose
        uses scalars in slots 0..6.
        """
        if args.robot_source not in ("hardware", "sim"):
            return
        cfg = hw_state.setdefault("ctrl_configs", {}).setdefault(name, {})
        if name == "joint_pd":
            if len(kp) != 7 or len(kd) != 7:
                print(f"[ctrl] joint_pd needs kp[7]+kd[7]; got {len(kp)}+{len(kd)}",
                      flush=True)
                return
            cfg["kp"] = list(kp)
            cfg["kd"] = list(kd)
        else:
            print(f"[ctrl] set_controller_gains unsupported for '{name}'", flush=True)
            return
        # Push to shm only if this is the *active* controller.
        if hw_state["ctrl_name"] == name:
            _seed_gains(hw_state["shm_gains"], name, cfg)
        await _broadcast_controller_state()

    async def on_save_controller_gains(name: str) -> None:
        """Persist the in-memory config for ``name`` to controllers/configs.yaml."""
        from controllers.configs import save_configs as save_ctrl_cfg
        try:
            await asyncio.to_thread(
                save_ctrl_cfg, hw_state.get("ctrl_configs", {}))
            print(f"[ctrl] saved configs.yaml: {name} = "
                  f"{hw_state.get('ctrl_configs', {}).get(name)}", flush=True)
        except Exception as exc:
            print(f"[ctrl] save_controller_gains failed: {exc}", flush=True)

    async def on_reload_controller_gains() -> None:
        """Re-read controllers/configs.yaml from disk."""
        from controllers.configs import load_configs as load_ctrl_cfg
        try:
            cfg = await asyncio.to_thread(load_ctrl_cfg)
        except Exception as exc:
            print(f"[ctrl] reload_controller_gains failed: {exc}", flush=True)
            return
        hw_state["ctrl_configs"] = cfg
        # Re-seed shm_gains for the active controller from the fresh values.
        active = hw_state["ctrl_name"]
        if active in cfg:
            _seed_gains(hw_state["shm_gains"], active, cfg[active])
        await _broadcast_controller_state()

    # ── Policies ──────────────────────────────────────────────────────────
    async def on_set_policy(name: str) -> None:
        """Engage a policy by name. Refuses if:
        - no policy entry exists in policies/configs.yaml
        - the active camera + SAM aren't producing a fresh object_pose
          (when the policy needs one)
        - the required controller can't be engaged
        """
        if args.robot_source not in ("hardware", "sim"):
            return
        from policies import POLICIES

        if name not in POLICIES:
            err = f"unknown policy '{name}'"
            print(f"[pol] {err}", flush=True)
            hw_state["policy_error"] = err
            await _broadcast_controller_state()
            return
        info, factory = POLICIES[name]
        cfg = hw_state["pol_configs"].get(name, {})
        if not cfg:
            err = f"no config for '{name}' in policies/configs.yaml"
            print(f"[pol] {err}", flush=True)
            hw_state["policy_error"] = err
            await _broadcast_controller_state()
            return

        # Block on fresh object_pose if needed.
        if info.needs_object_pose:
            with hw_state["shm_object_pose_seq"].get_lock():
                seq = int(hw_state["shm_object_pose_seq"].value)
            with hw_state["shm_object_pose"].get_lock():
                obj = np.frombuffer(hw_state["shm_object_pose"].get_obj(),
                                     dtype=np.float64)[:4].copy()
            if seq == 0 or obj[3] < 10:
                err = ("no object_pose available — click the target object "
                       "in the RGB view to produce a SAM mask first")
                print(f"[pol] refusing engage: {err}", flush=True)
                hw_state["policy_error"] = err
                await _broadcast_controller_state()
                return
            handle_world = obj[:3].copy()
        else:
            handle_world = np.zeros(3)

        # Stop any prior policy first.
        if hw_state["policy_proc"] is not None:
            await on_stop_policy()

        # Snapshot the user's saved ee_pose gains so we can restore.
        saved = hw_state["ctrl_configs"].get(info.controller, {}).copy()
        hw_state["saved_ee_pose_gains"] = saved

        # Overlay the policy's controller_gains onto the ee_pose config
        # and re-seed shm_gains. This affects the *active* controller
        # immediately and will persist as long as the policy runs.
        pol_gains = cfg.get("controller_gains", {})
        new_ctrl_cfg = saved.copy()
        new_ctrl_cfg.update(pol_gains)
        hw_state["ctrl_configs"][info.controller] = new_ctrl_cfg

        # If the required controller isn't already running, hot-swap to it.
        # The hot-swap goes through home anyway. After the swap, we'll
        # re-issue a JointMove to the policy's home_deg if it differs.
        if hw_state["ctrl_name"] != info.controller:
            await on_set_controller(info.controller)
        else:
            # Same controller — push the overlay into shm now.
            _seed_gains(hw_state["shm_gains"], info.controller, new_ctrl_cfg)
            await _broadcast_controller_state()

        # TODO: per-policy home_deg JointMove. Today we use the default
        # HOME_DEG inside the SST. If policy's home differs significantly
        # the user can manually drag the EE gizmo near it first, OR we
        # add a TCP path here to JointMove to policy.home_deg before
        # spawning. Left as a follow-up to keep this slice scoped.

        # Compute and store goal pose = handle + offset (world frame).
        goal_offset = np.asarray(cfg.get("goal_offset", [0, 0, 0]), dtype=np.float64)
        goal_world = handle_world + goal_offset
        with hw_state["shm_policy_goal"].get_lock():
            np.frombuffer(hw_state["shm_policy_goal"].get_obj(),
                          dtype=np.float64)[:3] = goal_world
        # Reset stats.
        with hw_state["shm_policy_hz"].get_lock():
            np.frombuffer(hw_state["shm_policy_hz"].get_obj(),
                          dtype=np.float64)[0] = 0.0
        with hw_state["shm_policy_last_action"].get_lock():
            np.frombuffer(hw_state["shm_policy_last_action"].get_obj(),
                          dtype=np.float64)[:7] = 0.0
        with hw_state["shm_policy_status"].get_lock():
            np.frombuffer(hw_state["shm_policy_status"].get_obj(),
                          dtype=np.uint8)[0] = 0
        # Also re-zero object_pose_seq so the policy waits for a fresh
        # frame before producing actions.
        # (Don't zero shm_object_pose itself — keep the latest position
        # so engage gripper handling is smooth.)

        # Spawn the policy subprocess.
        target, kwargs = factory(
            mjcf_path=args.robot_arm_mjcf, cfg=cfg,
            log_q=hw_state.get("log_q"),
        )
        stop_ev = mp.Event()
        proc = mp.Process(
            target=target,
            args=(
                hw_state["shm_q"], hw_state["shm_dq"], hw_state["shm_state_seq"],
                hw_state["shm_qtarget"], hw_state["shm_gripper"], hw_state["shm_cmd_seq"],
                hw_state["shm_object_pose"], hw_state["shm_object_pose_seq"],
                hw_state["shm_policy_goal"], hw_state["shm_policy_hz"],
                hw_state["shm_policy_last_action"], hw_state["shm_policy_status"],
                stop_ev,
            ),
            kwargs=kwargs,
            daemon=True,
        )
        proc.start()
        hw_state["policy_name"]  = name
        hw_state["policy_proc"]  = proc
        hw_state["policy_stop"]  = stop_ev
        hw_state["policy_error"] = ""
        print(f"[pol] running '{name}'  goal_world={goal_world.round(3)}", flush=True)
        await _broadcast_controller_state()

    async def on_stop_policy() -> None:
        """Stop the active policy and restore the user's ee_pose gains."""
        if args.robot_source not in ("hardware", "sim"):
            return
        if hw_state["policy_proc"] is None:
            return
        name = hw_state["policy_name"]
        print(f"[pol] stopping '{name}'", flush=True)
        if hw_state["policy_stop"] is not None:
            hw_state["policy_stop"].set()
        if hw_state["policy_proc"] is not None:
            await asyncio.to_thread(hw_state["policy_proc"].join, 5.0)
            if hw_state["policy_proc"].is_alive():
                hw_state["policy_proc"].terminate()
                await asyncio.to_thread(hw_state["policy_proc"].join, 2.0)
        hw_state["policy_proc"] = None
        hw_state["policy_stop"] = None
        hw_state["policy_name"] = ""

        # Restore the saved ee_pose gains so the controller goes back to
        # the user's interactive tuning. Only restore if we did snapshot.
        saved = hw_state.get("saved_ee_pose_gains")
        if saved is not None:
            hw_state["ctrl_configs"]["ee_pose"] = saved
            if hw_state["ctrl_name"] == "ee_pose":
                _seed_gains(hw_state["shm_gains"], "ee_pose", saved)
            hw_state["saved_ee_pose_gains"] = None
        await _broadcast_controller_state()

    async def on_save_policy_configs() -> None:
        from policies.configs import save_configs as save_pol_cfg
        try:
            await asyncio.to_thread(save_pol_cfg, hw_state.get("pol_configs", {}))
            print(f"[pol] saved policies/configs.yaml", flush=True)
        except Exception as exc:
            print(f"[pol] save failed: {exc}", flush=True)

    async def on_reload_policy_configs() -> None:
        from policies.configs import load_configs as load_pol_cfg
        try:
            cfg = await asyncio.to_thread(load_pol_cfg)
        except Exception as exc:
            print(f"[pol] reload failed: {exc}", flush=True)
            return
        hw_state["pol_configs"] = cfg
        await _broadcast_controller_state()

    async def on_restart_transport() -> None:
        """Hard recovery: kill the transport process, respawn it.

        Use only when the SST itself stalls (transport unresponsive). The
        kortex link gets a fresh connection; controller state is reset to
        idle.
        """
        if args.robot_source not in ("hardware", "sim"):
            return
        print("[hw] restart_transport: stopping current controller…", flush=True)
        if hw_state["ctrl_stop"] is not None:
            hw_state["ctrl_stop"].set()
        if hw_state["ctrl_proc"] is not None:
            await asyncio.to_thread(_join_controller, hw_state["ctrl_proc"])
        hw_state["ctrl_proc"] = None
        hw_state["ctrl_stop"] = None
        hw_state["ctrl_name"] = "idle"
        hw_state["ctrl_status"] = "loading"
        hw_state["ctrl_error"] = ""
        await _broadcast_controller_state()

        print("[hw] restart_transport: stopping transport…", flush=True)
        if hw_state["transport_stop"] is not None:
            hw_state["transport_stop"].set()
        if hw_state["transport_proc"] is not None:
            await asyncio.to_thread(
                hw_state["transport_proc"].join, 15.0)
            if hw_state["transport_proc"].is_alive():
                print("[hw] restart_transport: transport still alive, terminating",
                      flush=True)
                hw_state["transport_proc"].terminate()
                await asyncio.to_thread(
                    hw_state["transport_proc"].join, 2.0)

        # Reset events for fresh life.
        hw_state["transport_stop"] = mp.Event()
        hw_state["swap_request_ev"].clear()
        hw_state["swap_done_ev"].clear()

        from hardware import transport_process
        new_proc = mp.Process(
            target=transport_process,
            args=(args.robot_ip,),
            kwargs={
                "shm_q": hw_state["shm_q"], "shm_dq": hw_state["shm_dq"],
                "shm_state_seq": hw_state["shm_state_seq"],
                "shm_cmd_mode":  hw_state["shm_cmd_mode"],
                "shm_tau":       hw_state["shm_tau"],
                "shm_qtarget":   hw_state["shm_qtarget"],
                "shm_gripper":   hw_state["shm_gripper"],
                "shm_cmd_seq":   hw_state["shm_cmd_seq"],
                "stop_ev":       hw_state["transport_stop"],
                "swap_request_ev":  hw_state["swap_request_ev"],
                "swap_target_mode": hw_state["swap_target_mode"],
                "swap_done_ev":     hw_state["swap_done_ev"],
                "shm_hz":      hw_state["shm_hz"],
                "shm_phase":   hw_state["shm_phase"],
                "shm_fault_msg": hw_state["shm_fault_msg"],
                "log_q":       hw_state.get("log_q"),
            },
            daemon=True,
        )
        new_proc.start()
        hw_state["transport_proc"] = new_proc
        hw_state["ctrl_status"] = "idle"
        await _broadcast_controller_state()
        print("[hw] restart_transport: new transport started", flush=True)

    def _seed_gains(shm, ctrl_name: str, cfg: dict) -> None:
        """Pack the controller's default gains into shm_gains.

        Layout:
          joint_pd: [kp(7), kd(7)]
          ee_pose:  [kp_pos, kd_pos, kp_ori, kd_ori, posture_kp, posture_kd,
                     posture_weight, 0, 0, 0, 0, 0, 0, 0]  (rest unused)
          others:   zeros
        """
        arr = np.zeros(14, dtype=np.float64)
        if ctrl_name == "joint_pd":
            kp = np.asarray(cfg.get("kp", [40, 40, 40, 30, 20, 10, 5]), dtype=np.float64)
            kd = np.asarray(cfg.get("kd", [4, 4, 4, 3, 2, 1, 0.5]), dtype=np.float64)
            arr[:7] = kp[:7]
            arr[7:14] = kd[:7]
        elif ctrl_name == "ee_pose":
            arr[0] = float(cfg.get("kp_pos", 5.0))
            arr[1] = float(cfg.get("kd_pos", 0.0))
            arr[2] = float(cfg.get("kp_ori", 1.0))
            arr[3] = float(cfg.get("kd_ori", 0.0))
            arr[4] = float(cfg.get("posture_kp", 10.0))
            arr[5] = float(cfg.get("posture_kd", 2.0))
            arr[6] = float(cfg.get("posture_weight", 0.0))
        with shm.get_lock():
            np.frombuffer(shm.get_obj(), dtype=np.float64)[:14] = arr

    def _join_controller(proc: mp.Process) -> None:
        proc.join(timeout=5.0)
        if proc.is_alive():
            print(f"[ctrl] controller didn't exit cleanly; terminating", flush=True)
            proc.terminate()
            proc.join(timeout=2.0)

    def stop_hardware() -> None:
        # Stop policy first (it writes to shm_qtarget; stale writes during
        # teardown would confuse the controller).
        if hw_state.get("policy_stop") is not None:
            hw_state["policy_stop"].set()
            if hw_state.get("policy_proc") is not None:
                hw_state["policy_proc"].join(timeout=3.0)
        # Then controller.
        if hw_state["ctrl_stop"] is not None:
            hw_state["ctrl_stop"].set()
            if hw_state["ctrl_proc"] is not None:
                _join_controller(hw_state["ctrl_proc"])
        # Then transport (its finally: parks the arm).
        if hw_state["transport_stop"] is not None:
            print("[hw] stopping transport (will park arm at home)…",
                  flush=True)
            hw_state["transport_stop"].set()
        if hw_state["transport_proc"] is not None:
            hw_state["transport_proc"].join(timeout=30.0)
            if hw_state["transport_proc"].is_alive():
                print("[hw] transport didn't exit cleanly; terminating",
                      flush=True)
                hw_state["transport_proc"].terminate()
                hw_state["transport_proc"].join(timeout=2.0)
            else:
                print("[hw] transport parked.", flush=True)

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
            if args.robot_source in ("hardware", "sim"):
                frames.append(encode_controller_state(
                    0, _broadcast_controller_state_sync()))
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
                on_set_controller, on_stop_controller, on_home_robot,
                on_recover_robot, on_restart_transport,
                on_set_joint_target,
                on_set_ee_target,
                on_set_controller_gains, on_save_controller_gains,
                on_reload_controller_gains,
                on_set_policy, on_stop_policy,
                on_save_policy_configs, on_reload_policy_configs,
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
        last_phase = -1   # for fault history edge detection
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

            # If the camera is in 'rgbd' mode, copy the latest depth frame
            # into the side-channel shm and bump rgbd_seq. The camera-depth
            # backend reads from there instead of running an NN. Sim mode:
            # the sim worker writes depth_stream + rgbd_seq directly.
            depth_stream_arr = shm.depth_stream_arr()
            if hasattr(cam, "get_depth"):
                depth_m = cam.get_depth()
                if depth_m is not None and depth_m.shape == depth_stream_arr.shape:
                    depth_stream_arr[...] = depth_m
                    with shm.rgbd_seq.get_lock():
                        shm.rgbd_seq.value += 1

            # If the camera is in 'rgb_stereo' mode, push the IR pair to
            # their side channels so the FoundationStereo backend can read
            # them. Bump stereo_seq on every successful pair.
            if hasattr(cam, "get_stereo"):
                pair = cam.get_stereo()
                if pair is not None:
                    ir1, ir2 = pair
                    if (ir1.shape == shm.ir_left_arr().shape and
                            ir2.shape == shm.ir_right_arr().shape):
                        shm.ir_left_arr()[...]  = ir1
                        shm.ir_right_arr()[...] = ir2
                        with shm.stereo_seq.get_lock():
                            shm.stereo_seq.value += 1

            with shm.rgb_seq.get_lock():
                cur_rgb = shm.rgb_seq.value
            if cur_rgb != last_rgb_seq:
                last_rgb_seq = cur_rgb
                bgr = cv2.cvtColor(rgb_buf, cv2.COLOR_RGB2BGR)
                await hub.broadcast(encode_jpeg(seq, bgr))
                seq += 1
                n_rgb += 1

            # Drain the log queue and broadcast as a batch (cap at 50/iter
            # so a flood doesn't starve the rest of the loop).
            log_q = hw_state.get("log_q")
            if log_q is not None:
                batch = []
                for _ in range(50):
                    try:
                        batch.append(log_q.get_nowait())
                    except Exception:
                        break
                if batch:
                    await hub.broadcast(encode_log_lines(seq, batch))
                    seq += 1

            # Snapshot the entire depth-worker output under depth_seq's
            # lock so we don't read a partially-written buffer. The depth
            # worker bumps depth_seq LAST after a successful publish, and
            # holds this same lock while writing — so we either see the
            # previous full frame or the new one, never a mix.
            fresh = False
            with shm.depth_seq.get_lock():
                cur = shm.depth_seq.value
                fresh = cur != last_depth_seq
                if fresh:
                    last_depth_seq = cur
                    n = shm.pc_count.value
                    have_n = bool(shm.has_normal.value)
                    # Copy everything we need OUT of shm before releasing.
                    pc_xyz_snap   = pc_xyz[:n].copy()  if n > 0 else None
                    pc_rgb_snap   = pc_rgb[:n].copy()  if n > 0 else None
                    pc_idx_snap   = pc_grid_idx[:n].copy() if n > 0 else None
                    mesh_xyz_snap   = mesh_xyz.copy()
                    mesh_rgb_snap   = mesh_rgb.copy()
                    mesh_faces_snap = mesh_faces.copy()
                    mesh_normal_snap = mesh_normal.copy() if have_n else None
                    depth_buf_snap  = depth_buf.copy()
            if fresh:
                # Recompute T_world_camera every frame so live calibration
                # updates ripple through immediately.
                T_wc = sess["calib"].T_world_camera()
                normal_payload = (
                    _xform_normals(mesh_normal_snap, T_wc) if have_n else None
                )
                if n > 0:
                    pc_mask = seg_mask[pc_idx_snap].copy()
                    pc_world = _xform_points(pc_xyz_snap, T_wc)
                    mesh_world = _xform_points(mesh_xyz_snap, T_wc, mask_zero=True)
                    await hub.broadcast(
                        encode_points(seq, pc_world, pc_rgb_snap, mask=pc_mask)
                    )
                    seq += 1
                    await hub.broadcast(
                        encode_mesh(seq, mesh_world, mesh_rgb_snap,
                                    mesh_faces_snap, normal=normal_payload)
                    )
                    seq += 1
                    # ── Object pose (centroid of masked points, world frame) ──
                    # Updated every depth frame; the policy reads it via shm.
                    if (args.robot_source == "hardware"
                            and hw_state.get("shm_object_pose") is not None):
                        pos_w, n_in = compute_object_pose(
                            seg_mask, pc_xyz_snap, pc_idx_snap, T_wc, n,
                        )
                        if n_in > 0:
                            with hw_state["shm_object_pose"].get_lock():
                                arr = np.frombuffer(
                                    hw_state["shm_object_pose"].get_obj(),
                                    dtype=np.float64)
                                arr[0:3] = pos_w
                                arr[3]   = float(n_in)
                            with hw_state["shm_object_pose_seq"].get_lock():
                                hw_state["shm_object_pose_seq"].value += 1
                # Always send a colorized depth jpeg, even when pc is empty.
                depth_bgr = _depth_to_turbo_bgr(depth_buf_snap)
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

            # Hardware mode: copy the transport's 7-DOF arm angles (rad)
            # from shm_q into the robot scene's qpos slot so the existing
            # FK + transforms broadcast picks them up.
            if (args.robot_source == "hardware"
                    and hw_state.get("shm_q") is not None
                    and robot["shm"] is not None):
                with hw_state["shm_q"].get_lock():
                    arm_q = np.frombuffer(hw_state["shm_q"].get_obj(),
                                          dtype=np.float64)[:7].copy()
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
                # Hardware/sim mode: broadcast transport rate + phase + fault.
                if args.robot_source in ("hardware", "sim") and hw_state["shm_hz"] is not None:
                    try:
                        with hw_state["shm_hz"].get_lock():
                            transport_hz = float(np.frombuffer(
                                hw_state["shm_hz"].get_obj(), dtype=np.float64)[0])
                    except (AttributeError, ValueError):
                        transport_hz = 0.0
                    transport_alive = bool(
                        hw_state["transport_proc"] is not None
                        and hw_state["transport_proc"].is_alive()
                    )
                    with hw_state["shm_phase"].get_lock():
                        phase = int(hw_state["shm_phase"].value)
                    # Read null-terminated string from shm_fault_msg.
                    with hw_state["shm_fault_msg"].get_lock():
                        raw = bytes(hw_state["shm_fault_msg"].get_obj())
                    fault_msg = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
                    phase_names = {
                        0: "boot", 1: "homing", 2: "ready",
                        3: "running", 4: "swapping",
                        5: "fault", 6: "shutdown",
                    }
                    # Edge-detect fault transitions and append to history.
                    if phase == 5 and last_phase != 5:
                        fault_history.append({
                            "ts":     time.time(),
                            "source": "transport",
                            "msg":    fault_msg or "(no message)",
                        })
                    last_phase = phase
                    await hub.broadcast(encode_robot_status(seq, {
                        "source":      args.robot_source,
                        "osc_hz":      round(transport_hz, 1),
                        "alive":       transport_alive,
                        "phase":       phase,
                        "phase_name":  phase_names.get(phase, f"phase_{phase}"),
                        "fault_msg":   fault_msg,
                        "fault_history": list(fault_history),
                    }))
                    seq += 1
                    # Refresh controller_state at the same 1 Hz cadence so
                    # the EE-target gizmo follows the live setpoint when
                    # the user isn't dragging.
                    await hub.broadcast(encode_controller_state(
                        seq, _broadcast_controller_state_sync()))
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
    ap.add_argument("--model", default="auto",
                    help="Depth model key, or 'auto' (default): use the camera's "
                         "own depth stream if available, otherwise the best "
                         "learned metric backend.")
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
