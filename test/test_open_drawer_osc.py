"""Standalone open-drawer policy test on the real Kinova.

Bypasses the whole web stack (no SAM, no vision, no UI) so we can verify
the policy itself behaves on the real arm. The drawer-handle world pose
is HARDCODED — we want to isolate the policy from any perception bug.

Two processes:
  - real_process  : 500 Hz OSC torque loop on the real robot. Reads
                    target from shm_osc_target, applies torques.
  - policy_process: 10 Hz policy forward pass. Reads q/dq + handle/goal,
                    builds 33-D obs, writes osc_target + gripper_ctrl.

Usage:
    python -m test.test_open_drawer_osc                       # defaults
    python -m test.test_open_drawer_osc --handle 0.85 -0.02 0.5
    python -m test.test_open_drawer_osc --goal-offset -0.15 0 0
    python -m test.test_open_drawer_osc --checkpoint weights/policies/open_drawer/model_1500.pt

Ctrl-C to stop — the robot will home itself + park.

Reference (copy/adapted): sim2real_open_drawer_osc.py from the upstream
nn_policy repo, with sim/viser stripped out.
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hardware import (
    HOME_DEG,
    KinovaHardware,
    MAX_JOINT_TORQUE,
    PinocchioArm,
    TAU_OFFSETS,
    compute_osc_torques,
    kinova_deg_to_rad,
)
from hardware.osc import GAINS_KEYS
from policies.nn_policy import PolicyAgent


REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Hardcoded defaults ────────────────────────────────────────────────────────
DEFAULT_HANDLE = np.array([0.85, -0.02, 0.5])   # world frame, metres
DEFAULT_GOAL_OFFSET = np.array([-0.15, 0.0, 0.0])  # goal = handle + offset

# Must match training. Open-drawer YAML values, mirrored here so this
# script doesn't need the YAML parser.
DEFAULT_CHECKPOINT = REPO_ROOT / "weights/policies/open_drawer/model_1500.pt"
TARGET_HZ = 10
OSC_HZ    = 500
OSC_SUBSTEPS = OSC_HZ // TARGET_HZ
KP_POS, KD_POS = 50.0, 10.0
KP_ORI, KD_ORI = 50.0, 10.0
POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT = 10.0, 2.0, 0.0
DELTA_POS_SCALE = 0.005
DELTA_ORI_SCALE = 0.01
GRIPPER_DRIVER_MAX = 0.8
SUCCESS_THRESH = 0.02
WS_LO = np.array([0.20, -0.50, 0.05])
WS_HI = np.array([0.85,  0.50, 1.00])

# Per-policy home (matches the open_drawer YAML).
POLICY_HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

_ARM_MJCF = str(REPO_ROOT / "robot/mjcf/gen3_gripper.xml")


def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _pack_gains() -> np.ndarray:
    return np.array([KP_POS, KD_POS, KP_ORI, KD_ORI,
                     POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT])


def _gains_dict(arr: np.ndarray) -> dict:
    return dict(zip(GAINS_KEYS, arr))


# ── Observation builder (33-D) — must match training ─────────────────────────

def _axis_angle_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle rotvec. Matches training's mjlab path
    to 4 decimals including the near-π singular case (verified offline)."""
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def build_obs(q, dq, ee_pos, ee_rot, gripper_driver_pos,
              handle_pos_world, goal_handle_pos_world, last_action) -> np.ndarray:
    joint_vel_obs      = dq.astype(np.float32)                                                     # 7
    ee_axis_angle      = _axis_angle_from_R(ee_rot)
    ee_pose_obs        = np.concatenate([ee_pos, ee_axis_angle]).astype(np.float32)               # 6
    gripper_obs        = np.array([gripper_driver_pos / GRIPPER_DRIVER_MAX], dtype=np.float32)    # 1
    ee_to_object_obs   = (handle_pos_world - ee_pos).astype(np.float32)                            # 3
    object_pos_obs     = handle_pos_world.astype(np.float32)                                       # 3
    object_to_goal_obs = (goal_handle_pos_world - handle_pos_world).astype(np.float32)             # 3
    goal_pos_obs       = goal_handle_pos_world.astype(np.float32)                                  # 3
    return np.concatenate([
        joint_vel_obs, ee_pose_obs, gripper_obs,
        ee_to_object_obs, object_pos_obs, object_to_goal_obs, goal_pos_obs,
        last_action,
    ]).astype(np.float32)   # 33


# ── Policy process ────────────────────────────────────────────────────────────

def policy_process_fn(checkpoint_path: str, device_str: str,
                      shm_q, shm_dq,
                      shm_osc_target, shm_action,
                      shm_handle_pos, shm_goal_handle_pos,
                      shm_gripper_driver, shm_gripper_ctrl,
                      shm_policy_hz, shm_task_done,
                      stop_event, ready_event):
    """10 Hz: build obs → forward → write OSC target + gripper ctrl."""
    print(f"[policy] loading checkpoint: {checkpoint_path}", flush=True)
    agent = PolicyAgent(checkpoint_path, device=device_str)
    if agent.obs_dim != 33 or agent.action_dim != 7:
        raise RuntimeError(
            f"checkpoint obs/action mismatch: {agent.obs_dim}/{agent.action_dim}, "
            f"expected 33/7")

    robot = PinocchioArm(_ARM_MJCF, ee_frame="pinch_site")
    last_action = np.zeros(7, dtype=np.float32)
    period = 1.0 / TARGET_HZ
    iters = 0
    t_rate = time.time()

    ready_event.wait()  # wait for real_process to home + populate shm_q
    print("[policy] real ready; starting loop", flush=True)

    while not stop_event.is_set():
        t0 = time.time()

        handle_pos      = _np(shm_handle_pos).copy()
        goal_handle_pos = _np(shm_goal_handle_pos).copy()
        dist = float(np.linalg.norm(handle_pos - goal_handle_pos))
        task_done = dist < SUCCESS_THRESH
        np.frombuffer(shm_task_done.get_obj(), dtype=np.uint8)[0] = task_done
        if task_done:
            _np(shm_action)[:] = 0.0
            time.sleep(0.05)
            continue

        q  = _np(shm_q).copy()
        dq = _np(shm_dq).copy()
        gripper_driver = float(_np(shm_gripper_driver)[0])

        ee_pos, ee_rot = robot.fk(q)

        obs = build_obs(q, dq, ee_pos, ee_rot,
                        gripper_driver, handle_pos, goal_handle_pos, last_action)
        action = agent.get_action(obs)
        last_action[:] = action

        # action[:3] → delta pos (workspace-clipped); action[3:6] → delta ori
        osc_tgt_pos = np.clip(ee_pos + action[:3] * DELTA_POS_SCALE, WS_LO, WS_HI)
        delta_rot   = Rotation.from_rotvec(action[3:6] * DELTA_ORI_SCALE)
        osc_tgt_quat_xyzw = (delta_rot * Rotation.from_matrix(ee_rot)).as_quat()

        _np(shm_osc_target)[:3] = osc_tgt_pos
        _np(shm_osc_target)[3:] = osc_tgt_quat_xyzw

        # Store the un-scaled and scaled action for debugging.
        _np(shm_action)[:3]   = action[:3] * DELTA_POS_SCALE
        _np(shm_action)[3:6]  = osc_tgt_pos - ee_pos
        _np(shm_action)[6]    = action[6]
        _np(shm_action)[7:10] = action[3:6] * DELTA_ORI_SCALE

        # action[6] ∈ [-1, +1] → shm_gripper_ctrl ∈ [0, 1]
        _np(shm_gripper_ctrl)[0] = (float(np.clip(action[6], -1.0, 1.0)) + 1.0) * 0.5

        iters += 1
        dt = time.time() - t_rate
        if dt >= 1.0:
            _np(shm_policy_hz)[0] = iters / dt
            print(f"[policy] hz={iters/dt:.1f}  "
                  f"ee_pos={ee_pos.round(3)}  "
                  f"to_handle={(handle_pos - ee_pos).round(3)}  "
                  f"to_goal={dist*100:.1f}cm  "
                  f"action={action.round(3)}", flush=True)
            iters = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    print("[policy] stopping", flush=True)


# ── Real process ──────────────────────────────────────────────────────────────

def real_process_fn(ip: str, home_deg: np.ndarray,
                    shm_q, shm_dq, shm_osc_target, shm_gains,
                    shm_gripper_driver, shm_gripper_ctrl,
                    shm_real_hz, stop_event, ready_event):
    """500 Hz: read state from kortex, apply OSC torques + gripper ctrl."""
    inner_dt = 1.0 / OSC_HZ
    arm = PinocchioArm(_ARM_MJCF, ee_frame="pinch_site")
    posture = kinova_deg_to_rad(home_deg)

    hw = KinovaHardware(ip)
    try:
        print(f"[real] connecting to {ip}…", flush=True)
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            print("[real] robot not ready", flush=True)
            return

        hw.set_servoing_mode(low_level=False)
        time.sleep(0.5)
        print(f"[real] going home → {home_deg.round(1)}", flush=True)
        if not hw.go_to_joints(home_deg):
            print("[real] initial home FAILED", flush=True)
            return
        time.sleep(1.0)
        hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        hw.set_torque_mode(True)

        state = hw.read_state()
        pos_deg = state.positions_deg.copy()
        vel_deg = state.velocities_deg.copy()
        _np(shm_q)[:]  = kinova_deg_to_rad(pos_deg)
        _np(shm_dq)[:] = np.deg2rad(vel_deg)

        # Seed osc_target = current EE pose so first OSC iteration tracks
        # itself (no lurch before the policy publishes its first target).
        ee_pos, ee_rot = arm.fk(_np(shm_q).copy())
        from scipy.spatial.transform import Rotation as _R
        _np(shm_osc_target)[:3] = ee_pos
        _np(shm_osc_target)[3:] = _R.from_matrix(ee_rot).as_quat()  # xyzw

        ready_event.set()
        print("[real] in torque mode; waiting for policy targets", flush=True)

        iters, t_rate = 0, time.time()
        while not stop_event.is_set():
            target       = _np(shm_osc_target).copy()
            gains        = _gains_dict(_np(shm_gains).copy())
            gripper_ctrl = float(_np(shm_gripper_ctrl)[0])

            gripper_pos_01 = hw.read_gripper_position()
            _np(shm_gripper_driver)[0] = gripper_pos_01 * GRIPPER_DRIVER_MAX

            for _ in range(OSC_SUBSTEPS):
                t_inner = time.time()
                q  = kinova_deg_to_rad(pos_deg)
                dq = np.deg2rad(vel_deg)

                _np(shm_q)[:]  = q
                _np(shm_dq)[:] = dq

                tau = compute_osc_torques(
                    arm, target[:3], target[3:], q, dq,
                    gains=gains, posture_target=posture,
                )
                tau += TAU_OFFSETS

                if not np.all(np.isfinite(tau)):
                    print(f"[real] non-finite tau! {tau}", flush=True)
                    stop_event.set()
                    break

                tau = np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)
                state = hw.send_torques(tau, pos_deg, gripper_position=gripper_ctrl)
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()

                iters += 1
                elapsed = time.time() - t_inner
                if elapsed < inner_dt:
                    time.sleep(inner_dt - elapsed)

            dt = time.time() - t_rate
            if dt >= 1.0:
                _np(shm_real_hz)[0] = iters / dt
                iters, t_rate = 0, time.time()

    finally:
        print("[real] shutting down…", flush=True)
        try:
            if hw.in_torque_mode:
                hw.set_torque_mode(False)
                time.sleep(0.5)
            hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)
            hw.clear_faults()
            if hw.wait_until_ready(timeout=5.0):
                hw.go_to_joints(home_deg)
        except Exception as exc:
            print(f"[real] shutdown warning: {exc}", flush=True)
        hw.disconnect()
        print("[real] done.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # CUDA + fork doesn't work — the policy subprocess inherits a half-
    # initialised CUDA context and torch.load fails. Use spawn so each
    # child reinitializes CUDA from scratch.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.10")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--handle", type=float, nargs=3, default=list(DEFAULT_HANDLE),
                    metavar=("X", "Y", "Z"),
                    help="World-frame handle position (default: hardcoded)")
    ap.add_argument("--goal-offset", type=float, nargs=3,
                    default=list(DEFAULT_GOAL_OFFSET),
                    metavar=("DX", "DY", "DZ"),
                    help="Goal = handle + offset (world frame, metres)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--home", type=float, nargs=7,
                    default=list(POLICY_HOME_DEG),
                    metavar=("j1", "j2", "j3", "j4", "j5", "j6", "j7"),
                    help="Home pose (deg) the arm goes to before the policy starts.")
    args = ap.parse_args()

    handle = np.asarray(args.handle,      dtype=np.float64)
    offset = np.asarray(args.goal_offset, dtype=np.float64)
    goal   = handle + offset
    home   = np.asarray(args.home, dtype=np.float64)

    print(f"  ip:         {args.ip}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  handle:     {handle}")
    print(f"  goal:       {goal}  (handle + {offset})")
    print(f"  home_deg:   {home}")
    print(f"  device:     {args.device}")
    print()

    # ── Shared memory ────────────────────────────────────────────────────────
    shm_q  = mp.Array(ctypes.c_double, 7)
    shm_dq = mp.Array(ctypes.c_double, 7)
    shm_osc_target = mp.Array(ctypes.c_double, 7)    # pos(3) + quat_xyzw(4)
    shm_action     = mp.Array(ctypes.c_double, 10)   # for telemetry
    shm_handle_pos     = mp.Array(ctypes.c_double, 3)
    shm_goal_handle_pos = mp.Array(ctypes.c_double, 3)
    shm_gripper_driver  = mp.Array(ctypes.c_double, 1)
    shm_gripper_ctrl    = mp.Array(ctypes.c_double, 1)
    shm_gains      = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_policy_hz  = mp.Array(ctypes.c_double, 1)
    shm_real_hz    = mp.Array(ctypes.c_double, 1)
    shm_task_done  = mp.Array(ctypes.c_uint8, 1)

    # ── Seed constants ───────────────────────────────────────────────────────
    _np(shm_handle_pos)[:]      = handle
    _np(shm_goal_handle_pos)[:] = goal
    _np(shm_gains)[:]           = _pack_gains()
    # Gripper command starts open.
    _np(shm_gripper_ctrl)[0]    = 0.0

    stop_event  = mp.Event()
    ready_event = mp.Event()

    real = mp.Process(
        target=real_process_fn,
        args=(args.ip, home,
              shm_q, shm_dq, shm_osc_target, shm_gains,
              shm_gripper_driver, shm_gripper_ctrl,
              shm_real_hz, stop_event, ready_event),
        daemon=False,
    )
    policy = mp.Process(
        target=policy_process_fn,
        args=(args.checkpoint, args.device,
              shm_q, shm_dq, shm_osc_target, shm_action,
              shm_handle_pos, shm_goal_handle_pos,
              shm_gripper_driver, shm_gripper_ctrl,
              shm_policy_hz, shm_task_done,
              stop_event, ready_event),
        daemon=False,
    )

    real.start()
    policy.start()

    try:
        while real.is_alive() and policy.is_alive():
            time.sleep(0.5)
            if bool(np.frombuffer(shm_task_done.get_obj(), dtype=np.uint8)[0]):
                print("[main] task_done — handle reached goal. Stopping.", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[main] Ctrl-C → stopping", flush=True)
    finally:
        stop_event.set()
        policy.join(timeout=5.0)
        if policy.is_alive():
            policy.terminate()
        real.join(timeout=30.0)
        if real.is_alive():
            real.terminate()
        print("[main] done.", flush=True)


if __name__ == "__main__":
    main()
