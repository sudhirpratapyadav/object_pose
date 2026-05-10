"""OSC torque loop for the Kinova Gen3.

Copied verbatim from cam_calib_old/cam_calib_real.py — Pinocchio-based 6-DOF
impedance control with null-space posture, running at 500 Hz against
KinovaHardware. Not wired into web_server.py yet; kept intact for future use.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .kinova import KinovaHardware


# ── Timing ────────────────────────────────────────────────────────────────────
TARGET_HZ = 100
OSC_HZ    = 500
OSC_SUBS  = OSC_HZ // TARGET_HZ

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS      = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0])
HOME_DEG         = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

GAINS_KEYS = ["kp_pos", "kd_pos", "kp_ori", "kd_ori",
              "posture_kp", "posture_kd", "posture_weight"]

_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


# ── Shared memory helpers ─────────────────────────────────────────────────────

def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _gains_dict(arr: np.ndarray) -> dict:
    return dict(zip(GAINS_KEYS, arr))


# ── Helpers ───────────────────────────────────────────────────────────────────

def kinova_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    s = deg.copy()
    s[s > 180.0] -= 360.0
    return np.deg2rad(s)


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

    def fk(self, q):
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

    def dynamics(self, q, dq):
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

def real_robot_process(ip, mjcf_path, shm_q, shm_target, shm_gains, shm_hz,
                       shm_gripper, stop_ev, reset_ev, reset_done_ev,
                       ee_frame: str = "pinch_site"):
    """500 Hz OSC loop. Writes 7-DOF joint angles (rad) into shm_q.

    Reads target EE pose (xyz + xyzw quat) from shm_target, gains from
    shm_gains, gripper [0..1] from shm_gripper. Updates shm_hz with measured
    rate. Honors stop_ev / reset_ev / reset_done_ev for lifecycle control.
    """
    # Basic logging so kortex / KinovaHardware messages aren't swallowed.
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="[real:%(levelname)s] %(message)s",
                        force=True)

    inner_dt = 1.0 / OSC_HZ
    robot    = PinocchioArm(str(mjcf_path), ee_frame=ee_frame)
    posture  = kinova_deg_to_rad(HOME_DEG)

    hw = KinovaHardware(ip)
    try:
        print("[real] Connecting…", flush=True)
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            print("[real] Not ready — aborting", flush=True)
            return

        hw.set_servoing_mode(low_level=False)
        # Settle in high-level mode before issuing the motion. Some firmware
        # versions abort the first JointMove if it's issued too soon after a
        # low-level → high-level transition.
        time.sleep(1.0)
        hw.clear_faults()
        if not hw.wait_until_ready(timeout=5.0):
            print("[real] Not ready after clear_faults — aborting", flush=True)
            return
        print("[real] Going to home…", flush=True)
        if not hw.go_to_joints(HOME_DEG):
            print("[real] go_to_joints failed — aborting", flush=True)
            return
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
                print("[real] Reset: going home…", flush=True)
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
            print(f"[real] Shutdown warning: {exc}", flush=True)
        hw.disconnect()
        print("[real] Done.", flush=True)
