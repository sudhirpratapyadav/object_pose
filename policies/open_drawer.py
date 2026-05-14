"""Open-drawer NN policy subprocess.

Reads:
  - shm_q, shm_dq          : current arm joints (rad)
  - shm_object_pose        : drawer-handle pose in world frame (vision)
  - shm_policy_goal        : goal pose (world frame), fixed at engage
  - shm_policy_gripper_in  : measured gripper position [0..1]

Writes:
  - shm_qtarget            : ee_pose target = [pos(3), quat_xyzw(4)] (world frame)
  - shm_gripper            : gripper command [0..1]  (transport sends to robot)
  - shm_policy_hz, shm_policy_last_action

Loop:
  At target_hz, build 33-D observation, run policy forward, convert the
  7-D action to a delta EE pose + gripper. Write into shm_qtarget so the
  active ee_pose controller tracks it at 500 Hz.

Observation layout (33-D, matches open_drawer_osc training):
   7  joint_vel
   6  ee_pose = [ee_pos(3), axis_angle(3)] in robot-local frame
   1  gripper_state = driver_joint / GRIPPER_DRIVER_MAX
   3  ee_to_object  = handle_world - ee_world
   3  object_pos    = handle_world
   3  object_to_goal = goal_world - handle_world
   3  goal_pos      = goal_world
   7  last_action
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hardware import (
    CMD_MODE_TORQUE,
    PinocchioArm,
    kinova_deg_to_rad,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _np_arr(shm) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _np_arr_bool(shm) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.uint8)


def _quat_xyzw_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → unit quaternion [x, y, z, w]."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([(m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s,
                         0.25 * s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([0.25 * s,
                         (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s,
                         (m[2, 1] - m[1, 2]) / s])
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 1] + m[1, 0]) / s,
                         0.25 * s,
                         (m[1, 2] + m[2, 1]) / s,
                         (m[0, 2] - m[2, 0]) / s])
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array([(m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s,
                     0.25 * s,
                     (m[1, 0] - m[0, 1]) / s])


def _axis_angle_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle (rotvec, length=angle, direction=axis).

    Matches scipy/Rotation: returns (3,) vector. At the home pose for this
    policy the angle is roughly π; SciPy returns a valid axis. Training
    used the mjlab helpers (quat_from_matrix → axis_angle_from_quat); the
    rotvec convention is equivalent up to a sign on the rotation axis at
    the singular case. We use rotvec for simplicity; if training is sign-
    sensitive at home we can swap in the mjlab helpers.
    """
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def open_drawer_policy_process(
    shm_q, shm_dq, shm_state_seq,
    shm_qtarget, shm_gripper, shm_cmd_seq,
    # Vision side: from stream_loop
    shm_object_pose,
    shm_object_pose_seq,
    # Policy run state (server-owned)
    shm_policy_goal,          # 3 doubles, world frame
    shm_policy_hz,            # 1 double
    shm_policy_last_action,   # 7 doubles
    shm_policy_status,        # 1 uint8: 0=waiting, 1=running, 2=success
    stop_ev,
    *,
    mjcf_path: str,
    cfg: dict,
    log_q=None,
):
    """Open-drawer NN policy. Drives the ee_pose controller's target."""
    from hardware.log_relay import install_log_relay
    install_log_relay(log_q, source="policy-open-drawer")
    log = logging.getLogger("policy-open-drawer")

    from .nn_policy import PolicyAgent
    ckpt = cfg.get("checkpoint", "weights/policies/open_drawer/model_1500.pt")
    ckpt_path = ckpt if Path(ckpt).is_absolute() else str(REPO_ROOT / ckpt)
    log.info(f"loading checkpoint: {ckpt_path}")
    agent = PolicyAgent(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")

    obs_dim    = int(cfg.get("obs_dim", 33))
    action_dim = int(cfg.get("action_dim", 7))
    if agent.obs_dim != obs_dim:
        raise RuntimeError(
            f"checkpoint obs_dim={agent.obs_dim} but cfg expects {obs_dim}")
    if agent.action_dim != action_dim:
        raise RuntimeError(
            f"checkpoint action_dim={agent.action_dim} but cfg expects {action_dim}")

    target_hz       = float(cfg.get("target_hz", 10.0))
    period          = 1.0 / target_hz
    delta_pos_scale = float(cfg.get("delta_pos_scale", 0.005))
    delta_ori_scale = float(cfg.get("delta_ori_scale", 0.01))
    gripper_driver_max = float(cfg.get("gripper_driver_max", 0.8))
    success_thresh  = float(cfg.get("success_thresh", 0.02))
    ws_lo = np.asarray(cfg.get("ws_lo", [-1e3] * 3), dtype=np.float64)
    ws_hi = np.asarray(cfg.get("ws_hi", [+1e3] * 3), dtype=np.float64)

    arm = PinocchioArm(mjcf_path, ee_frame="pinch_site")
    last_action = np.zeros(7, dtype=np.float32)

    # Wait until the transport has published at least one state.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with shm_state_seq.get_lock():
            if int(shm_state_seq.value) > 0:
                break
        time.sleep(0.01)

    # Seed q_target to current EE pose so the active ee_pose controller's
    # first cycle doesn't lurch. The ee_pose controller already does this
    # on its own startup; we redo it here in case we engage *after* it has
    # been running.
    with shm_q.get_lock():
        q0 = _np_arr(shm_q)[:7].copy()
    pos0, R0 = arm.fk(q0)
    qx0 = _quat_xyzw_from_matrix(R0)
    seed = np.zeros(7, dtype=np.float64)
    seed[:3] = pos0
    seed[3:] = qx0
    with shm_qtarget.get_lock():
        _np_arr(shm_qtarget)[:7] = seed
    with shm_cmd_seq.get_lock():
        shm_cmd_seq.value = (shm_cmd_seq.value + 1) & 0xFFFFFFFF

    # Local goal cache; the server writes shm_policy_goal once at engage.
    with shm_policy_goal.get_lock():
        goal_world = _np_arr(shm_policy_goal)[:3].copy()
    log.info(f"goal_world = {goal_world.round(3)}")

    with shm_policy_status.get_lock():
        _np_arr_bool(shm_policy_status)[0] = 1  # running

    n_iter = 0
    t_log = time.time()

    while not stop_ev.is_set():
        t0 = time.time()

        # Read state
        with shm_q.get_lock():
            q = _np_arr(shm_q)[:7].copy()
        with shm_dq.get_lock():
            dq = _np_arr(shm_dq)[:7].copy()
        with shm_gripper.get_lock():
            grip_01 = float(_np_arr(shm_gripper)[0])
        with shm_object_pose.get_lock():
            obj = _np_arr(shm_object_pose)[:4].copy()   # [x, y, z, n_points]
        with shm_object_pose_seq.get_lock():
            obj_seq = int(shm_object_pose_seq.value)
        with shm_policy_goal.get_lock():
            goal_world = _np_arr(shm_policy_goal)[:3].copy()

        # If we lost the object pose (e.g. mask disappeared), freeze
        # action to zero and wait. The transport keeps the last target,
        # so the arm holds in place.
        if obj_seq == 0 or obj[3] < 3:   # need ≥3 masked points
            log.warning("waiting for object_pose (mask gone?); pausing policy")
            with shm_policy_status.get_lock():
                _np_arr_bool(shm_policy_status)[0] = 0
            time.sleep(period)
            continue
        with shm_policy_status.get_lock():
            if _np_arr_bool(shm_policy_status)[0] == 0:
                _np_arr_bool(shm_policy_status)[0] = 1
        handle_world = obj[:3].copy()

        # Success check.
        if float(np.linalg.norm(handle_world - goal_world)) < success_thresh:
            log.info("success: handle within %.3f m of goal", success_thresh)
            with shm_policy_status.get_lock():
                _np_arr_bool(shm_policy_status)[0] = 2
            # Hold position by not writing new targets; loop until stopped.
            time.sleep(period)
            continue

        # FK
        ee_pos, ee_rot = arm.fk(q)
        ee_axis_angle = _axis_angle_from_R(ee_rot)

        # Build obs (33-D)
        gripper_obs    = np.array([grip_01], dtype=np.float32)  # already in [0..1]
        # Training used driver_joint / 0.8 ∈ [0,1]; our shm_gripper is
        # already in [0,1] (0=open, 1=closed), so no further scaling.
        # gripper_driver_max only matters if you wanted to invert; keep
        # the scale here in case a future policy uses a different range.
        if gripper_driver_max != 0.8:
            gripper_obs[0] = grip_01 * (0.8 / gripper_driver_max)
        ee_to_object   = handle_world - ee_pos
        object_pos     = handle_world.copy()
        object_to_goal = goal_world - handle_world
        goal_pos       = goal_world.copy()

        obs = np.concatenate([
            dq.astype(np.float32),                                  # 7
            np.concatenate([ee_pos, ee_axis_angle]).astype(np.float32),  # 6
            gripper_obs,                                            # 1
            ee_to_object.astype(np.float32),                        # 3
            object_pos.astype(np.float32),                          # 3
            object_to_goal.astype(np.float32),                      # 3
            goal_pos.astype(np.float32),                            # 3
            last_action,                                            # 7
        ])                                                          # = 33

        action = agent.get_action(obs)   # (7,)
        last_action[:] = action

        # Convert action → next EE target
        tgt_pos = np.clip(ee_pos + action[:3] * delta_pos_scale, ws_lo, ws_hi)
        delta_rot = Rotation.from_rotvec(action[3:6] * delta_ori_scale)
        tgt_quat_xyzw = (delta_rot * Rotation.from_matrix(ee_rot)).as_quat()

        # Publish target → shm_qtarget for ee_pose
        out = np.zeros(7, dtype=np.float64)
        out[:3] = tgt_pos
        out[3:] = tgt_quat_xyzw
        with shm_qtarget.get_lock():
            _np_arr(shm_qtarget)[:7] = out
        with shm_cmd_seq.get_lock():
            shm_cmd_seq.value = (shm_cmd_seq.value + 1) & 0xFFFFFFFF

        # Gripper command: policy outputs [-1, +1]; map to [0, 1] for transport.
        grip_cmd = 0.5 * (float(np.clip(action[6], -1.0, 1.0)) + 1.0)
        with shm_gripper.get_lock():
            _np_arr(shm_gripper)[0] = grip_cmd

        # Stats
        with shm_policy_last_action.get_lock():
            _np_arr(shm_policy_last_action)[:7] = action.astype(np.float64)

        n_iter += 1
        dt = time.time() - t_log
        if dt >= 1.0:
            with shm_policy_hz.get_lock():
                _np_arr(shm_policy_hz)[0] = n_iter / dt
            n_iter = 0
            t_log = time.time()

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    log.info("stopping")
    with shm_policy_status.get_lock():
        _np_arr_bool(shm_policy_status)[0] = 0
