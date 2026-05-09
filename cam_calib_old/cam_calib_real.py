"""
Real-robot camera calibration viewer.

Processes / threads
-------------------
  Process  real_robot  — OSC at 500 Hz against Kinova hardware; writes joint
                         angles to shm_q_real
  Thread   camera      — RealSense capture at 30 Hz; pushes frames / point
                         clouds to thread-safe queues inside RGBDCamera
  Thread   viz         — 30 Hz viser update (robot meshes, camera frustum,
                         point cloud, image panels)
  Main                 — keyboard EE target + viser GUI poll at 100 Hz

Startup order
-------------
  1. Shared memory + sync events
  2. Spawn real_robot process  (before any GPU / CUDA init)
  3. Load Pinocchio + MuJoCo CPU model
  4. Start RealSense camera thread
  5. Start viser + viz thread
  6. Main loop

Controls
--------
  Translation  +X/-X: w/s   +Y/-Y: a/d   +Z/-Z: e/q
  Rotation     +Rx/-Rx: i/k  +Ry/-Ry: j/l  +Rz/-Rz: u/o
  Gripper      g toggle open/close
  r            go home (reset)
  Ctrl+C       quit
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import viser
import yaml
from pynput import keyboard as pynput_kb
from queue import Empty, Full, Queue
from scipy.spatial.transform import Rotation
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from hardware import KinovaHardware
from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML
from viewer import ViserMujocoScene

_CAMERA_BACKENDS = {
    "realsense": ("camera_realsense", "RGBDCamera"),
    "oakd":      ("camera.oak_d",     "RGBDCamera"),
}
RGBDCamera = None  # set in main() after arg parsing

from depth_model import DepthModelThread, MODELS as DEPTH_MODELS, DEFAULT_MODEL as DEPTH_DEFAULT_MODEL

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
_TORQUE_XML = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_gripper_torque.xml"
CAM_CFG     = SCRIPT_DIR / "cam_calib_config.yaml"

# ── Timing ────────────────────────────────────────────────────────────────────
TARGET_HZ = 100
OSC_HZ    = 500
OSC_SUBS  = OSC_HZ // TARGET_HZ
VIZ_HZ    = 30

# ── Default OSC gains ─────────────────────────────────────────────────────────
KP_POS, KD_POS         = 5.0, 0.0
KP_ORI, KD_ORI         = 1.0, 0.0
POSTURE_KP, POSTURE_KD = 10.0, 2.0
POSTURE_WEIGHT         = 0.0

MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS      = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0])
HOME_DEG         = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

GAINS_KEYS = ["kp_pos", "kd_pos", "kp_ori", "kd_ori",
              "posture_kp", "posture_kd", "posture_weight"]

_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# ── Keyboard ──────────────────────────────────────────────────────────────────
DELTA_POS = 0.005
DELTA_ROT = np.deg2rad(1.0)

KEY_DELTAS: dict[str, np.ndarray] = {
    "w": np.array([ DELTA_POS, 0, 0, 0, 0, 0]),
    "s": np.array([-DELTA_POS, 0, 0, 0, 0, 0]),
    "a": np.array([0,  DELTA_POS, 0, 0, 0, 0]),
    "d": np.array([0, -DELTA_POS, 0, 0, 0, 0]),
    "e": np.array([0, 0,  DELTA_POS, 0, 0, 0]),
    "q": np.array([0, 0, -DELTA_POS, 0, 0, 0]),
    "i": np.array([0, 0, 0,  DELTA_ROT, 0, 0]),
    "k": np.array([0, 0, 0, -DELTA_ROT, 0, 0]),
    "j": np.array([0, 0, 0, 0,  DELTA_ROT, 0]),
    "l": np.array([0, 0, 0, 0, -DELTA_ROT, 0]),
    "u": np.array([0, 0, 0, 0, 0,  DELTA_ROT]),
    "o": np.array([0, 0, 0, 0, 0, -DELTA_ROT]),
}

_held_keys:   set[str] = set()
_held_lock    = threading.Lock()
_toggle_keys: set[str] = set()
_toggle_lock  = threading.Lock()


def _on_press(key):
    try:
        c = key.char
        with _held_lock:
            _held_keys.add(c)
        if c in ("g", "r"):
            with _toggle_lock:
                _toggle_keys.add(c)
    except AttributeError:
        pass


def _on_release(key):
    try:
        with _held_lock:
            _held_keys.discard(key.char)
    except AttributeError:
        pass


# ── Shared memory helpers ─────────────────────────────────────────────────────

def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _pack_gains(kp_pos, kd_pos, kp_ori, kd_ori, pkp, pkd, pw) -> np.ndarray:
    return np.array([kp_pos, kd_pos, kp_ori, kd_ori, pkp, pkd, pw])


def _gains_dict(arr: np.ndarray) -> dict:
    return dict(zip(GAINS_KEYS, arr))


# ── Helpers ───────────────────────────────────────────────────────────────────

def kinova_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    s = deg.copy()
    s[s > 180.0] -= 360.0
    return np.deg2rad(s)


def _wxyz(q_xyzw: np.ndarray) -> tuple:
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _rotation_display(R: np.ndarray) -> str:
    rows = [" ".join(f"{v:+.3f}" for v in R[i]) for i in range(3)]
    return " | ".join(rows)


def _K_display(fx, fy, cx, cy) -> str:
    return (f"[{fx:.1f}  0  {cx:.1f}]"
            f" [0  {fy:.1f}  {cy:.1f}]"
            f" [0  0  1]")


# ── Camera YAML ───────────────────────────────────────────────────────────────

_DEFAULT_CAM_CFG: dict = {
    "intrinsics": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0,
                   "width": 640, "height": 480},
    "extrinsics": {"pos": [0.5, 0.0, 0.8],
                   "euler_deg": [180.0, 0.0, 90.0]},
}


def load_cam_cfg() -> dict:
    if CAM_CFG.exists():
        with open(CAM_CFG) as f:
            return yaml.safe_load(f)
    with open(CAM_CFG, "w") as f:
        yaml.dump(_DEFAULT_CAM_CFG, f, default_flow_style=None, sort_keys=False)
    print(f"[cam] Created default config -> {CAM_CFG}")
    return _DEFAULT_CAM_CFG


def cam_extrinsics(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos_world [3], R_world_cam [3,3])  i.e. T_base_cam."""
    pos = np.array(cfg["extrinsics"]["pos"], dtype=np.float64)
    R   = Rotation.from_euler("xyz", cfg["extrinsics"]["euler_deg"], degrees=True).as_matrix()
    return pos, R


# ── Camera frustum ────────────────────────────────────────────────────────────

def _build_frustum_segments(pos: np.ndarray, R: np.ndarray,
                             intr: dict, scale: float = 0.15) -> np.ndarray:
    """
    10 line segments forming a camera frustum pyramid + image-plane cross.
    Returns float32 (10, 2, 3).
    """
    w, h   = intr["width"],  intr["height"]
    fx, fy = intr["fx"],     intr["fy"]
    cx, cy = intr["cx"],     intr["cy"]

    corners_cam = np.array([
        [(0 - cx) / fx * scale, (0 - cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (0 - cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],
        [(0 - cx) / fx * scale, (h - cy) / fy * scale, scale],
    ])
    cw = (R @ corners_cam.T).T + pos   # corners in world frame

    segs = []
    for c in cw:                        # 4 rays
        segs.append([pos, c])
    for i in range(4):                  # 4 rectangle edges
        segs.append([cw[i], cw[(i + 1) % 4]])
    segs.append([cw[0], cw[2]])         # diagonal cross
    segs.append([cw[1], cw[3]])
    return np.array(segs, dtype=np.float32)   # (10, 2, 3)


# ── Pinocchio arm ─────────────────────────────────────────────────────────────

class PinocchioArm:
    def __init__(self, mjcf_path: str, ee_frame: str) -> None:
        self.model = pin.buildModelFromMJCF(mjcf_path)
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data  = self.model.createData()
        self.ee_id = self.model.getFrameId(ee_frame)
        if self.ee_id >= self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' not found in {mjcf_path}")
        self._v_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_v for n in _ARM_JOINT_NAMES],
            dtype=np.intp,
        )
        self._q_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in _ARM_JOINT_NAMES],
            dtype=np.intp,
        )
        self._q_full  = pin.neutral(self.model)
        self._dq_full = np.zeros(self.model.nv)

    def _set_q(self, q, dq=None):
        self._q_full[self._q_idx] = q
        if dq is not None:
            self._dq_full[self._v_idx] = dq

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        self._set_q(q)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        oMf = self.data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def jacobian(self, q) -> np.ndarray:
        self._set_q(q)
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        return J[:, self._v_idx]

    def dynamics(self, q, dq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._set_q(q, dq)
        pin.crba(self.model, self.data, self._q_full)
        M_sub = self.data.M[np.ix_(self._v_idx, self._v_idx)]
        M_arm = 0.5 * (M_sub + M_sub.T)
        pin.nonLinearEffects(self.model, self.data, self._q_full, self._dq_full)
        nle = self.data.nle[self._v_idx].copy()
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, self._q_full, self._dq_full)
        J_dot    = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        J_dot_dq = J_dot[:, self._v_idx] @ dq
        return M_arm, nle, J_dot_dq


def _pose_error_6d(tgt_pos, tgt_quat_xyzw, cur_pos, cur_rot) -> np.ndarray:
    return np.concatenate([
        tgt_pos - cur_pos,
        Rotation.from_matrix(
            Rotation.from_quat(tgt_quat_xyzw).as_matrix() @ cur_rot.T
        ).as_rotvec(),
    ])


def compute_osc_torques(robot: PinocchioArm, tgt_pos, tgt_quat_xyzw,
                        q, dq, *, gains, posture_target) -> np.ndarray:
    ee_pos, ee_rot = robot.fk(q)
    error  = _pose_error_6d(tgt_pos, tgt_quat_xyzw, ee_pos, ee_rot)
    J      = robot.jacobian(q)
    ee_vel = J @ dq
    ddx    = np.empty(6)
    ddx[:3] = gains["kp_pos"] * error[:3] + gains["kd_pos"] * (-ee_vel[:3])
    ddx[3:] = gains["kp_ori"] * error[3:] + gains["kd_ori"] * (-ee_vel[3:])
    M, nle, J_dot_dq = robot.dynamics(q, dq)
    M_inv     = np.linalg.inv(M)
    Lambda    = np.linalg.inv(J @ M_inv @ J.T)
    J_dyn_inv = M_inv @ J.T @ Lambda
    F         = Lambda @ (ddx - J_dot_dq)
    N         = np.eye(7) - J.T @ J_dyn_inv.T
    tau_post  = gains["posture_kp"] * (posture_target - q) + gains["posture_kd"] * (-dq)
    tau       = J.T @ F + nle + gains["posture_weight"] * (N @ tau_post)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── Real robot process ────────────────────────────────────────────────────────

def real_robot_process(ip, shm_q, shm_target, shm_gains, shm_hz, shm_gripper,
                       stop_ev, reset_ev, reset_done_ev):
    inner_dt = 1.0 / OSC_HZ
    robot    = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    posture  = kinova_deg_to_rad(HOME_DEG)

    hw = KinovaHardware(ip)
    try:
        print("[real] Connecting…")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            print("[real] Not ready — aborting")
            return

        hw.set_servoing_mode(low_level=False)
        print("[real] Going to home…")
        hw.go_to_joints(HOME_DEG)
        time.sleep(1.0)
        hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        hw.set_torque_mode(True)

        state   = hw.read_state()
        pos_deg = state.positions_deg.copy()
        vel_deg = state.velocities_deg.copy()
        iters, t_rate = 0, time.time()

        while not stop_ev.is_set():
            if reset_ev.is_set():
                reset_ev.clear()
                print("[real] Reset: going home…")
                hw.set_torque_mode(False)
                hw.set_servoing_mode(low_level=False)
                hw.go_to_joints(HOME_DEG)
                time.sleep(0.5)
                hw.set_servoing_mode(low_level=True)
                time.sleep(0.3)
                hw.set_torque_mode(True)
                state   = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()
                iters, t_rate = 0, time.time()
                reset_done_ev.set()
                continue

            target = _np(shm_target).copy()
            gains  = _gains_dict(_np(shm_gains).copy())

            for _ in range(OSC_SUBS):
                t0 = time.time()
                q  = kinova_deg_to_rad(pos_deg)
                dq = np.deg2rad(vel_deg)
                _np(shm_q)[:] = q

                tau     = compute_osc_torques(robot, target[:3], target[3:], q, dq,
                                              gains=gains, posture_target=posture)
                tau    += TAU_OFFSETS
                state   = hw.send_torques(tau, pos_deg,
                                          gripper_position=float(_np(shm_gripper)[0]))
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()

                iters += 1
                elapsed = time.time() - t0
                if elapsed < inner_dt:
                    time.sleep(inner_dt - elapsed)

            dt = time.time() - t_rate
            if dt >= 1.0:
                _np(shm_hz)[0] = iters / dt
                iters, t_rate = 0, time.time()

    finally:
        try:
            if hw.in_torque_mode:
                hw.set_torque_mode(False)
                time.sleep(0.5)
            hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)
            hw.clear_faults()
            if hw.wait_until_ready(timeout=5.0):
                hw.go_to_joints(HOME_DEG)
        except Exception as exc:
            print(f"[real] Shutdown warning: {exc}")
        hw.disconnect()
        print("[real] Done.")


# ── Viz thread ────────────────────────────────────────────────────────────────

def viz_thread_fn(
    server: viser.ViserServer,
    real_view,
    mj_model_cpu,
    mj_data_real,
    arm_q_idxs: np.ndarray,
    shm_q,
    cam: RGBDCamera,
    cam_state: dict,          # shared mutable: pos, euler_deg, fx, fy, cx, cy
    cam_state_lock: threading.Lock,
    gui_rgb,
    gui_depth_sensor,         # always shows hardware depth
    gui_depth_model,          # always shows model depth (blank if unavailable)
    use_model_for_pc_ref: list,  # [bool] — checkbox value, which depth feeds point cloud
    pc_handle_ref: list,
    frustum_handle_ref: list,
    cam_frame_handle_ref: list,
    ee_frame_handle,
    tgt_frame_handle,
    txt_robot_hz,
    shm_hz,
    shm_target,
    pc_world_frame_ref: list,
    depth_model_q_ref: list,    # [Queue | None] — updated when model switches
    rgb_for_depth_q_ref: list, # [Queue | None] — updated when model switches
    stop_ev: threading.Event,
):
    period = 1.0 / VIZ_HZ
    robot  = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")

    def _read_cam():
        with cam_state_lock:
            pos      = np.array(cam_state["pos"])
            euler    = np.array(cam_state["euler_deg"])
            intr     = {k: cam_state[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
        R = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        return pos, R, intr

    def _rebuild_frustum(pos, R, intr):
        segs = _build_frustum_segments(pos, R, intr)
        h = frustum_handle_ref[0]
        if h is not None:
            h.remove()
        frustum_handle_ref[0] = server.scene.add_line_segments(
            "/camera_frustum",
            points=segs,
            colors=np.tile(np.array([0.2, 0.9, 0.2], dtype=np.float32), (len(segs), 2, 1)),
            line_width=2.0,
        )
        # Move the axis frame to match
        wxyz = _wxyz(Rotation.from_matrix(R).as_quat())
        h2 = cam_frame_handle_ref[0]
        if h2 is not None:
            h2.position = tuple(float(v) for v in pos)
            h2.wxyz     = wxyz

    # Build initial scene objects
    cam_pos0, cam_R0, cam_intr0 = _read_cam()
    _rebuild_frustum(cam_pos0, cam_R0, cam_intr0)
    cam_frame_handle_ref[0] = server.scene.add_frame(
        "/camera_frame",
        position=tuple(float(v) for v in cam_pos0),
        wxyz=_wxyz(Rotation.from_matrix(cam_R0).as_quat()),
        axes_length=0.08,
        axes_radius=0.004,
    )

    prev_cam_state = {k: cam_state[k] for k in cam_state}

    while not stop_ev.is_set():
        t0 = time.time()

        # ── Robot FK → mesh ───────────────────────────────────────────────
        q_real = _np(shm_q).copy()
        mj_data_real.qpos.flat[arm_q_idxs] = q_real
        mujoco.mj_kinematics(mj_model_cpu, mj_data_real)
        real_view.update(mj_data_real)

        ee_pos, ee_rot = robot.fk(q_real)
        ee_frame_handle.position = tuple(float(v) for v in ee_pos)
        ee_frame_handle.wxyz     = _wxyz(Rotation.from_matrix(ee_rot).as_quat())

        tgt = _np(shm_target).copy()
        tgt_frame_handle.position = tuple(float(v) for v in tgt[:3])
        tgt_frame_handle.wxyz     = _wxyz(tgt[3:])

        txt_robot_hz.value = f"{_np(shm_hz)[0]:.0f} Hz"

        # ── Camera extrinsics/intrinsics changed → rebuild frustum ────────
        cam_pos, cam_R, cam_intr = _read_cam()
        with cam_state_lock:
            changed = any(
                np.any(np.array(cam_state[k]) != np.array(prev_cam_state[k]))
                for k in cam_state
            )
            if changed:
                prev_cam_state = {k: cam_state[k] for k in cam_state}
        if changed:
            _rebuild_frustum(cam_pos, cam_R, cam_intr)

        # ── Camera frames → GUI images + depth panels ─────────────────────
        frame = cam.get_latest_frame()
        sensor_depth_m: Optional[np.ndarray] = None  # from camera hardware
        model_depth_m:  Optional[np.ndarray] = None  # from Depth Anything V2

        if frame is not None:
            gui_rgb.image = frame.rgb
            sensor_depth_m = frame.depth.astype(np.float32) / 1000.0

            # Always update sensor depth panel
            dmax = sensor_depth_m.max()
            d_u8 = (sensor_depth_m / dmax * 255).astype(np.uint8) if dmax > 0 \
                   else np.zeros_like(sensor_depth_m, dtype=np.uint8)
            gui_depth_sensor.image = np.stack([d_u8, d_u8, d_u8], axis=-1)

            # Feed RGB into depth model queue (always, if model is running)
            _rgb_q = rgb_for_depth_q_ref[0]
            if _rgb_q is not None:
                try:
                    _rgb_q.put_nowait(frame.rgb.copy())
                except Full:
                    try:
                        _rgb_q.get_nowait()
                    except Empty:
                        pass
                    try:
                        _rgb_q.put_nowait(frame.rgb.copy())
                    except Full:
                        pass

        # Drain latest model depth (independent of whether we got a camera frame)
        _depth_q = depth_model_q_ref[0]
        if _depth_q is not None:
            while True:
                try:
                    model_depth_m = _depth_q.get_nowait()
                except Empty:
                    break
            if model_depth_m is not None:
                dmax = model_depth_m.max()
                d_u8 = (model_depth_m / dmax * 255).astype(np.uint8) if dmax > 0 \
                       else np.zeros_like(model_depth_m, dtype=np.uint8)
                gui_depth_model.image = np.stack([d_u8, d_u8, d_u8], axis=-1)

        # Active depth for point cloud: pick based on checkbox
        use_model = use_model_for_pc_ref[0]
        active_depth_m = (model_depth_m if (use_model and model_depth_m is not None)
                          else sensor_depth_m)

        # Always drain the camera backend's PC queue to prevent stale buildup.
        # Only render it when sensor mode is active.
        cam_pc = cam.get_latest_pc()

        # ── Point cloud from active depth ─────────────────────────────────
        if use_model and model_depth_m is not None and frame is not None:
            # Model mode: backproject model depth with calibration intrinsics
            with cam_state_lock:
                fx = cam_state["fx"]; fy = cam_state["fy"]
                cx = cam_state["cx"]; cy = cam_state["cy"]
            h, w = model_depth_m.shape
            u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))
            z     = model_depth_m
            ds    = np.zeros((h, w), dtype=bool)
            ds[::4, ::4] = True
            valid = (z > 0.05) & (z < 10.0) & ds
            pts_cam = np.stack([
                (u[valid] - cx) * z[valid] / fx,
                (v[valid] - cy) * z[valid] / fy,
                z[valid],
            ], axis=1)
            colors_pc = frame.rgb[valid].astype(np.float32) / 255.0

            if pc_world_frame_ref[0]:
                pts_show = (cam_R @ pts_cam.T).T + cam_pos
            else:
                pts_show = pts_cam

            h_pc = pc_handle_ref[0]
            if h_pc is not None:
                h_pc.remove()
            pc_handle_ref[0] = server.scene.add_point_cloud(
                "/point_cloud",
                points=pts_show.astype(np.float32),
                colors=colors_pc,
                point_size=0.003,
            )
        elif not use_model:
            # Sensor mode: render camera backend's pre-built point cloud
            if cam_pc is not None and len(cam_pc.points) > 0:
                if pc_world_frame_ref[0]:
                    pts = (cam_R @ cam_pc.points.T).T + cam_pos
                else:
                    pts = cam_pc.points

                colors = (cam_pc.colors.astype(np.float32) / 255.0
                          if cam_pc.colors is not None
                          else np.full((len(pts), 3), 0.6, dtype=np.float32))

                h = pc_handle_ref[0]
                if h is not None:
                    h.remove()
                pc_handle_ref[0] = server.scene.add_point_cloud(
                    "/point_cloud",
                    points=pts.astype(np.float32),
                    colors=colors,
                    point_size=0.003,
                )

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-robot camera calibration viewer")
    parser.add_argument("--ip",     default="192.168.1.10", help="Kinova IP")
    parser.add_argument("--no-hw",  action="store_true",
                        help="Skip hardware OSC (camera + viz only)")
    parser.add_argument("--camera", choices=list(_CAMERA_BACKENDS), default="realsense",
                        help="Camera backend (default: realsense)")
    parser.add_argument("--depth",  choices=["sensor", "model"], default="sensor",
                        help="Depth source: sensor=camera hardware, model=Depth Anything V2 (default: sensor)")
    args = parser.parse_args()

    # Deferred import so only the chosen backend is loaded
    global RGBDCamera
    module_name, class_name = _CAMERA_BACKENDS[args.camera]
    import importlib
    RGBDCamera = getattr(importlib.import_module(module_name), class_name)
    print(f"[cam] Using backend: {args.camera}  ({module_name}.{class_name})")

    cam_cfg = load_cam_cfg()

    # ── Shared memory ──────────────────────────────────────────────────────
    shm_q       = mp.Array(ctypes.c_double, 7)
    shm_target  = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_gains   = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_hz      = mp.Array(ctypes.c_double, 1)
    shm_gripper = mp.Array(ctypes.c_double, 1)

    home_rad = kinova_deg_to_rad(HOME_DEG)
    _np(shm_q)[:]      = home_rad
    _np(shm_gains)[:] = _pack_gains(KP_POS, KD_POS, KP_ORI, KD_ORI,
                                    POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT)
    _np(shm_gripper)[0] = 1.0   # start closed

    stop_ev       = mp.Event()
    reset_ev      = mp.Event()
    reset_done_ev = mp.Event()
    th_stop       = threading.Event()

    # ── Spawn real robot process (before any CUDA init) ────────────────────
    real_proc = None
    if not args.no_hw:
        real_proc = mp.Process(
            target=real_robot_process,
            args=(args.ip, shm_q, shm_target, shm_gains, shm_hz, shm_gripper,
                  stop_ev, reset_ev, reset_done_ev),
            daemon=True,
        )
        real_proc.start()

    # ── MuJoCo CPU model for FK visualisation ─────────────────────────────
    spec         = mujoco.MjSpec.from_file(str(_TORQUE_XML))
    mj_model_cpu = spec.compile()
    mj_data_real = mujoco.MjData(mj_model_cpu)
    arm_q_idxs   = np.array([mj_model_cpu.joint(n).qposadr for n in _ARM_JOINT_NAMES])

    mj_data_real.qpos.flat[arm_q_idxs] = home_rad
    mujoco.mj_kinematics(mj_model_cpu, mj_data_real)

    print("Loading Pinocchio…")
    robot_main = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    print("  OK")

    # Initial target = home EE pose
    ee_pos0, ee_rot0  = robot_main.fk(home_rad)
    ee_quat0_xyzw     = Rotation.from_matrix(ee_rot0).as_quat()
    _np(shm_target)[:3] = ee_pos0
    _np(shm_target)[3:] = ee_quat0_xyzw

    # ── Shared mutable camera state (extrinsics + intrinsics) ─────────────
    _ext = cam_cfg["extrinsics"]
    _int = cam_cfg["intrinsics"]
    cam_state: dict = {
        "pos":       list(_ext["pos"]),
        "euler_deg": list(_ext["euler_deg"]),
        "fx": float(_int["fx"]),
        "fy": float(_int["fy"]),
        "cx": float(_int["cx"]),
        "cy": float(_int["cy"]),
        "width":  int(_int["width"]),
        "height": int(_int["height"]),
    }
    cam_state_lock = threading.Lock()

    def _cam_pos_R():
        with cam_state_lock:
            pos   = np.array(cam_state["pos"])
            euler = np.array(cam_state["euler_deg"])
        return pos, Rotation.from_euler("xyz", euler, degrees=True).as_matrix()

    cam_pos, cam_R = _cam_pos_R()
    cam_intr       = {k: cam_state[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}

    # ── RealSense ──────────────────────────────────────────────────────────
    cam = RGBDCamera(
        width=cam_intr["width"],
        height=cam_intr["height"],
        fps=30,
    )
    try:
        live = cam.start()
        print(f"[cam] {live.width}x{live.height}  "
              f"fx={live.fx:.1f}  fy={live.fy:.1f}  "
              f"cx={live.cx:.1f}  cy={live.cy:.1f}")
        # Patch cam_state intrinsics with live values for accurate frustum
        with cam_state_lock:
            cam_state.update({"fx": live.fx, "fy": live.fy,
                              "cx": live.cx, "cy": live.cy,
                              "width": live.width, "height": live.height})
        cam_intr = {k: cam_state[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
    except Exception as exc:
        print(f"[cam] Warning — {exc}. Running without live camera.")

    # ── Depth model (optional) ─────────────────────────────────────────────
    _depth_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    depth_model_thread: Optional[DepthModelThread] = None
    _active_model_key    = [DEPTH_DEFAULT_MODEL]
    depth_model_q_ref    = [None]   # [Queue | None]
    rgb_for_depth_q_ref  = [None]   # [Queue | None]

    def _start_depth_model(model_key: str):
        """Stop any running model thread and start a new one with model_key."""
        nonlocal depth_model_thread
        if depth_model_thread is not None:
            depth_model_thread.stop()
        new_rgb_q          = Queue(maxsize=1)
        depth_model_thread = DepthModelThread(model_key=model_key, device=_depth_device)
        new_depth_q        = depth_model_thread.start(new_rgb_q)
        rgb_for_depth_q_ref[0]  = new_rgb_q
        depth_model_q_ref[0]    = new_depth_q
        _active_model_key[0]    = model_key
        print(f"[depth] Started {model_key} (warming up...)")

    if args.depth == "model":
        _start_depth_model(DEPTH_DEFAULT_MODEL)

    # ── Viser ─────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label="Kinova Cam-Calib Viewer")
    scene     = ViserMujocoScene.create(server, mj_model_cpu)
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.75))
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0,
                                   camera_elevation=30.0)

    # EE actual + target frames
    ee_frame_handle  = server.scene.add_frame(
        "/ee_actual", axes_length=0.08, axes_radius=0.004)
    tgt_frame_handle = server.scene.add_frame(
        "/ee_target", axes_length=0.10, axes_radius=0.005)

    # EE target gizmo (draggable)
    ee_gizmo = server.scene.add_transform_controls(
        "/ee_gizmo",
        scale=0.12,
        position=tuple(float(v) for v in ee_pos0),
        wxyz=_wxyz(ee_quat0_xyzw),
    )
    _gizmo_dirty = [False]
    _gizmo_lock  = threading.Lock()

    @ee_gizmo.on_update
    def _on_gizmo(_):
        with _gizmo_lock:
            _gizmo_dirty[0] = True

    # Mutable refs updated by viz thread
    pc_handle_ref         = [None]
    frustum_handle_ref    = [None]
    cam_frame_handle_ref  = [None]
    pc_world_frame_ref    = [True]

    # ── GUI ───────────────────────────────────────────────────────────────
    w_img = cam_intr["width"]
    h_img = cam_intr["height"]
    blank = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    with server.gui.add_folder("Camera"):
        gui_rgb           = server.gui.add_image(image=blank, label="RGB")
        gui_depth_sensor  = server.gui.add_image(image=blank, label="Depth — sensor")
        gui_depth_model   = server.gui.add_image(image=blank, label="Depth — model")
        cb_use_model_depth = server.gui.add_checkbox(
            "Use model depth for point cloud",
            initial_value=(args.depth == "model"),
        )
        dd_model = server.gui.add_dropdown(
            "Depth model",
            options=list(DEPTH_MODELS.keys()),
            initial_value=DEPTH_DEFAULT_MODEL,
        )

    use_model_for_pc_ref = [args.depth == "model"]

    @cb_use_model_depth.on_update
    def _(_):
        use_model_for_pc_ref[0] = cb_use_model_depth.value

    @dd_model.on_update
    def _(_):
        new_key = dd_model.value
        if new_key == _active_model_key[0] and depth_model_thread is not None:
            return
        # Always enable model depth when a model is (re)selected
        use_model_for_pc_ref[0] = True
        cb_use_model_depth.value = True
        _start_depth_model(new_key)

    with server.gui.add_folder("Point Cloud"):
        cb_world = server.gui.add_checkbox("World frame  (uncheck = camera frame)",
                                           initial_value=True)

        @cb_world.on_update
        def _(_):
            pc_world_frame_ref[0] = cb_world.value

    with server.gui.add_folder("Robot State"):
        txt_robot_hz = server.gui.add_text("OSC rate",  initial_value="— Hz")
        txt_ee_pos   = server.gui.add_text("EE pos",    initial_value="—")
        txt_tgt_pos  = server.gui.add_text("Target pos",initial_value="—")

    with server.gui.add_folder("Control"):
        reset_btn   = server.gui.add_button("Go Home")
        gripper_btn = server.gui.add_button("Toggle Gripper")
        txt_gripper = server.gui.add_text("Gripper", initial_value="Closed")

    @reset_btn.on_click
    def _(_): reset_ev.set()

    @gripper_btn.on_click
    def _(_):
        cur = _np(shm_gripper)[0]
        _np(shm_gripper)[0] = 0.0 if cur > 0.5 else 1.0
        txt_gripper.value = "Open" if _np(shm_gripper)[0] < 0.5 else "Closed"

    with server.gui.add_folder("OSC Gains"):
        sl_kp_pos  = server.gui.add_slider("Kp pos",  0.0, 500.0,  1.0, KP_POS)
        sl_kd_pos  = server.gui.add_slider("Kd pos",  0.0, 100.0,  0.5, KD_POS)
        sl_kp_ori  = server.gui.add_slider("Kp ori",  0.0, 500.0,  1.0, KP_ORI)
        sl_kd_ori  = server.gui.add_slider("Kd ori",  0.0, 100.0,  0.5, KD_ORI)
        sl_post_kp = server.gui.add_slider("Post Kp", 0.0, 100.0,  0.1, POSTURE_KP)
        sl_post_kd = server.gui.add_slider("Post Kd", 0.0,  20.0,  0.1, POSTURE_KD)
        sl_post_w  = server.gui.add_slider("Post w",  0.0,   1.0, 0.01, POSTURE_WEIGHT)

    # ── Camera calibration gizmo (in 3-D scene) ───────────────────────────
    with cam_state_lock:
        _gp = list(cam_state["pos"])
        _ge = list(cam_state["euler_deg"])
    _cam_gizmo_wxyz0 = _wxyz(Rotation.from_euler("xyz", _ge, degrees=True).as_quat())
    cam_gizmo = server.scene.add_transform_controls(
        "/cam_gizmo",
        scale=0.18,
        position=tuple(float(v) for v in _gp),
        wxyz=_cam_gizmo_wxyz0,
    )
    _cam_gizmo_dirty = [False]
    _cam_gizmo_lock  = threading.Lock()

    @cam_gizmo.on_update
    def _on_cam_gizmo(_):
        with _cam_gizmo_lock:
            _cam_gizmo_dirty[0] = True

    # ── Camera calibration GUI folder ─────────────────────────────────────
    with cam_state_lock:
        _ip = [cam_state["pos"][0], cam_state["pos"][1], cam_state["pos"][2]]
        _ie = [cam_state["euler_deg"][0], cam_state["euler_deg"][1], cam_state["euler_deg"][2]]
        _fx0, _fy0, _cx0, _cy0 = cam_state["fx"], cam_state["fy"], cam_state["cx"], cam_state["cy"]
        _w0, _h0 = cam_state["width"], cam_state["height"]

    with server.gui.add_folder("Camera Calibration", expand_by_default=False):

        with server.gui.add_folder("Extrinsics — position (m)"):
            sl_cam_px = server.gui.add_slider("x", -3.0, 3.0, 0.005, _ip[0])
            sl_cam_py = server.gui.add_slider("y", -3.0, 3.0, 0.005, _ip[1])
            sl_cam_pz = server.gui.add_slider("z",  0.0, 3.0, 0.005, _ip[2])

        with server.gui.add_folder("Extrinsics — orientation (euler XYZ °)"):
            sl_cam_rx = server.gui.add_slider("rx", -180.0, 180.0, 0.5, _ie[0])
            sl_cam_ry = server.gui.add_slider("ry", -180.0, 180.0, 0.5, _ie[1])
            sl_cam_rz = server.gui.add_slider("rz", -180.0, 180.0, 0.5, _ie[2])

        txt_ext_mat = server.gui.add_text(
            "R (row-major)",
            initial_value=_rotation_display(Rotation.from_euler("xyz", _ie, degrees=True).as_matrix()),
        )

        with server.gui.add_folder("Intrinsics"):
            sl_fx = server.gui.add_slider("fx (px)", 10.0, 2000.0, 0.5, _fx0)
            sl_fy = server.gui.add_slider("fy (px)", 10.0, 2000.0, 0.5, _fy0)
            sl_cx = server.gui.add_slider("cx (px)",  0.0, float(_w0), 0.5, _cx0)
            sl_cy = server.gui.add_slider("cy (px)",  0.0, float(_h0), 0.5, _cy0)

        txt_intr_mat = server.gui.add_text(
            "K matrix",
            initial_value=_K_display(_fx0, _fy0, _cx0, _cy0),
        )

        save_calib_btn   = server.gui.add_button("Save to YAML")
        txt_calib_status = server.gui.add_text("Status", initial_value="—")

    # ── Calibration helpers ───────────────────────────────────────────────

    def _update_cam_state_from_sliders():
        pos = [sl_cam_px.value, sl_cam_py.value, sl_cam_pz.value]
        eul = [sl_cam_rx.value, sl_cam_ry.value, sl_cam_rz.value]
        with cam_state_lock:
            cam_state["pos"]       = pos
            cam_state["euler_deg"] = eul
            cam_state["fx"] = sl_fx.value
            cam_state["fy"] = sl_fy.value
            cam_state["cx"] = sl_cx.value
            cam_state["cy"] = sl_cy.value
        R = Rotation.from_euler("xyz", eul, degrees=True).as_matrix()
        txt_ext_mat.value  = _rotation_display(R)
        txt_intr_mat.value = _K_display(sl_fx.value, sl_fy.value, sl_cx.value, sl_cy.value)
        # Sync gizmo → sliders wrote, push to gizmo
        wxyz = _wxyz(Rotation.from_euler("xyz", eul, degrees=True).as_quat())
        cam_gizmo.position = tuple(float(v) for v in pos)
        cam_gizmo.wxyz     = wxyz

    def _on_ext_slider(_):
        _update_cam_state_from_sliders()

    def _on_intr_slider(_):
        with cam_state_lock:
            cam_state["fx"] = sl_fx.value
            cam_state["fy"] = sl_fy.value
            cam_state["cx"] = sl_cx.value
            cam_state["cy"] = sl_cy.value
        txt_intr_mat.value = _K_display(sl_fx.value, sl_fy.value, sl_cx.value, sl_cy.value)

    for _sl in (sl_cam_px, sl_cam_py, sl_cam_pz, sl_cam_rx, sl_cam_ry, sl_cam_rz):
        _sl.on_update(_on_ext_slider)
    for _sl in (sl_fx, sl_fy, sl_cx, sl_cy):
        _sl.on_update(_on_intr_slider)

    @save_calib_btn.on_click
    def _save_calib(_):
        with cam_state_lock:
            out = {
                "extrinsics": {
                    "pos":       [round(v, 4) for v in cam_state["pos"]],
                    "euler_deg": [round(v, 4) for v in cam_state["euler_deg"]],
                },
                "intrinsics": {
                    "fx":     round(cam_state["fx"], 3),
                    "fy":     round(cam_state["fy"], 3),
                    "cx":     round(cam_state["cx"], 3),
                    "cy":     round(cam_state["cy"], 3),
                    "width":  cam_state["width"],
                    "height": cam_state["height"],
                },
            }
        with open(CAM_CFG, "w") as f:
            yaml.dump(out, f, default_flow_style=None, sort_keys=False)
        txt_calib_status.value = f"Saved -> {CAM_CFG.name}"
        print(f"[calib] Saved -> {CAM_CFG}")

    # ── Keyboard ──────────────────────────────────────────────────────────
    kb = pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    kb.start()
    print("\nKeyboard:")
    print("  Translate  w/s a/d e/q   Rotate  i/k j/l u/o")
    print("  g  toggle gripper   r  go home   Ctrl+C  quit\n")

    # ── Viz thread ─────────────────────────────────────────────────────────
    threading.Thread(
        target=viz_thread_fn,
        daemon=True,
        args=(
            server, real_view, mj_model_cpu, mj_data_real, arm_q_idxs,
            shm_q, cam, cam_state, cam_state_lock,
            gui_rgb, gui_depth_sensor, gui_depth_model, use_model_for_pc_ref,
            pc_handle_ref, frustum_handle_ref, cam_frame_handle_ref,
            ee_frame_handle, tgt_frame_handle,
            txt_robot_hz, shm_hz,
            shm_target,
            pc_world_frame_ref,
            depth_model_q_ref, rgb_for_depth_q_ref,
            th_stop,
        ),
    ).start()

    # ── Main loop ──────────────────────────────────────────────────────────
    outer_dt = 1.0 / TARGET_HZ
    print("Running. Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()

            # Single-shot key events
            with _toggle_lock:
                toggles = _toggle_keys.copy()
                _toggle_keys.clear()

            if "r" in toggles:
                reset_ev.set()
            if "g" in toggles:
                cur = _np(shm_gripper)[0]
                _np(shm_gripper)[0] = 0.0 if cur > 0.5 else 1.0
                txt_gripper.value = "Open" if _np(shm_gripper)[0] < 0.5 else "Closed"

            # Held-key delta
            with _held_lock:
                held = _held_keys.copy()
            delta = sum((KEY_DELTAS[c] for c in held if c in KEY_DELTAS),
                        np.zeros(6))

            # Gizmo takes priority over keyboard when it was just moved
            with _gizmo_lock:
                gizmo_moved = _gizmo_dirty[0]
                _gizmo_dirty[0] = False

            if gizmo_moved:
                gpos  = np.array(ee_gizmo.position)
                gwxyz = np.array(ee_gizmo.wxyz)
                gxyzw = np.array([gwxyz[1], gwxyz[2], gwxyz[3], gwxyz[0]])
                _np(shm_target)[:3] = gpos
                _np(shm_target)[3:] = gxyzw
            elif np.any(delta != 0):
                q_real   = _np(shm_q).copy()
                ee_p, ee_r = robot_main.fk(q_real)
                new_pos  = ee_p + delta[:3]
                if np.any(delta[3:] != 0):
                    new_quat = (Rotation.from_rotvec(delta[3:]) *
                                Rotation.from_matrix(ee_r)).as_quat()
                else:
                    new_quat = Rotation.from_matrix(ee_r).as_quat()
                _np(shm_target)[:3] = new_pos
                _np(shm_target)[3:] = new_quat
                # Sync gizmo
                ee_gizmo.position = tuple(float(v) for v in new_pos)
                ee_gizmo.wxyz     = _wxyz(new_quat)

            # ── Camera gizmo → update state + sliders ─────────────────────
            with _cam_gizmo_lock:
                cam_gizmo_moved = _cam_gizmo_dirty[0]
                _cam_gizmo_dirty[0] = False

            if cam_gizmo_moved:
                gpos  = list(cam_gizmo.position)
                gwxyz = np.array(cam_gizmo.wxyz)           # w x y z
                gxyzw = np.array([gwxyz[1], gwxyz[2], gwxyz[3], gwxyz[0]])
                eul   = Rotation.from_quat(gxyzw).as_euler("xyz", degrees=True).tolist()
                with cam_state_lock:
                    cam_state["pos"]       = gpos
                    cam_state["euler_deg"] = eul
                # Sync sliders to gizmo
                sl_cam_px.value = gpos[0]
                sl_cam_py.value = gpos[1]
                sl_cam_pz.value = gpos[2]
                sl_cam_rx.value = eul[0]
                sl_cam_ry.value = eul[1]
                sl_cam_rz.value = eul[2]
                R_new = Rotation.from_euler("xyz", eul, degrees=True).as_matrix()
                txt_ext_mat.value = _rotation_display(R_new)

            # Gains
            _np(shm_gains)[:] = _pack_gains(
                sl_kp_pos.value, sl_kd_pos.value,
                sl_kp_ori.value, sl_kd_ori.value,
                sl_post_kp.value, sl_post_kd.value, sl_post_w.value,
            )

            # HUD
            q_r     = _np(shm_q).copy()
            tgt     = _np(shm_target).copy()
            ee_p, _ = robot_main.fk(q_r)
            txt_ee_pos.value  = f"x={ee_p[0]:.3f}  y={ee_p[1]:.3f}  z={ee_p[2]:.3f}"
            txt_tgt_pos.value = f"x={tgt[0]:.3f}  y={tgt[1]:.3f}  z={tgt[2]:.3f}"

            elapsed = time.time() - t0
            if elapsed < outer_dt:
                time.sleep(outer_dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        kb.stop()
        th_stop.set()
        stop_ev.set()
        cam.stop()
        if depth_model_thread is not None:
            depth_model_thread.stop()
        if real_proc is not None:
            real_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
