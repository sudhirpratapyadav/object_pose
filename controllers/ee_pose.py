"""End-effector pose controller (OSC torque tracking).

Computes torques via the operational-space-control law copied from the
existing hardware/osc.py:

    tau = J^T · Λ · (Kp·e + Kd·(-Jq̇) − J̇q̇)
        + nle(q,q̇)
        + posture_weight · N · (Kp_post·(q_home − q) + Kd_post·(−q̇))

target = [pos(3), quat_xyzw(4)] in shm_qtarget (interpretation reuse —
saves a wire kind for now). Gains are hardcoded conservative defaults;
UI knobs land in slice 4.5+.

On startup, the controller seeds shm_qtarget to the current EE pose
(via FK on the latest joint state) so the first iteration is "track
yourself", i.e. the arm holds still. Operator can then publish a new
target via UI commands (slice 4.5).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from hardware import (
    CMD_MODE_TORQUE,
    HOME_DEG,
    MAX_JOINT_TORQUE,
    PinocchioArm,
    compute_osc_torques,
    kinova_deg_to_rad,
)


CONTROL_HZ = 500
INNER_DT = 1.0 / CONTROL_HZ

# Conservative defaults from cam_calib_old. Slow but safe.
GAINS = {
    "kp_pos": 5.0,
    "kd_pos": 0.0,
    "kp_ori": 1.0,
    "kd_ori": 0.0,
    "posture_kp": 10.0,
    "posture_kd": 2.0,
    "posture_weight": 0.0,
}


def _np_arr(shm) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


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


def ee_pose_controller_process(
    shm_q, shm_dq, shm_state_seq,
    shm_cmd_mode, shm_tau, shm_qtarget, shm_gripper, shm_cmd_seq,
    stop_ev,
    *,
    mjcf_path: str,
    ee_frame: str = "pinch_site",
    log_q=None,
):
    """OSC torque controller. Tracks shm_qtarget = [pos, quat_xyzw]."""
    from hardware.log_relay import install_log_relay
    install_log_relay(log_q, source="ctrl-ee-pose")
    log = logging.getLogger("ctrl-ee-pose")

    log.info(f"loading PinocchioArm({mjcf_path})…")
    arm = PinocchioArm(mjcf_path, ee_frame=ee_frame)
    posture = kinova_deg_to_rad(HOME_DEG)
    log.info("ready (gains hardcoded; UI comes in slice 4.5+)")

    # Seed shm_qtarget = current EE pose so first iteration tracks itself.
    # Wait for at least one fresh state from the transport.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with shm_state_seq.get_lock():
            if int(shm_state_seq.value) > 0:
                break
        time.sleep(0.01)
    with shm_q.get_lock():
        q0 = _np_arr(shm_q)[:7].copy()
    pos0, R0 = arm.fk(q0)
    quat0_xyzw = _quat_xyzw_from_matrix(R0)
    target_init = np.zeros(7, dtype=np.float64)
    target_init[:3] = pos0
    target_init[3:] = quat0_xyzw   # [x, y, z, w]
    with shm_qtarget.get_lock():
        _np_arr(shm_qtarget)[:7] = target_init
    with shm_cmd_seq.get_lock():
        shm_cmd_seq.value = (shm_cmd_seq.value + 1) & 0xFFFFFFFF
    log.info(f"seeded target ee_pos={pos0.round(3)} ee_quat_xyzw={quat0_xyzw.round(3)}")

    with shm_cmd_mode.get_lock():
        shm_cmd_mode.value = CMD_MODE_TORQUE

    last_state_seq = 0
    n_writes = 0
    t_log = time.time()

    while not stop_ev.is_set():
        t0 = time.time()

        with shm_state_seq.get_lock():
            cur_seq = int(shm_state_seq.value)
        if cur_seq == last_state_seq:
            time.sleep(0.0005)
            continue
        last_state_seq = cur_seq

        with shm_q.get_lock():
            q = _np_arr(shm_q)[:7].copy()
        with shm_dq.get_lock():
            dq = _np_arr(shm_dq)[:7].copy()
        with shm_qtarget.get_lock():
            tgt = _np_arr(shm_qtarget)[:7].copy()
        tgt_pos = tgt[:3]
        tgt_quat_xyzw = tgt[3:]

        try:
            tau = compute_osc_torques(arm, tgt_pos, tgt_quat_xyzw, q, dq,
                                      gains=GAINS, posture_target=posture)
        except Exception as exc:
            log.exception(f"OSC failed: {exc}")
            tau = np.zeros(7)

        # MAX_JOINT_TORQUE clipping is already inside compute_osc_torques;
        # belt-and-suspenders here too.
        tau = np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)

        with shm_tau.get_lock():
            _np_arr(shm_tau)[:7] = tau
        with shm_cmd_seq.get_lock():
            shm_cmd_seq.value = (shm_cmd_seq.value + 1) & 0xFFFFFFFF

        n_writes += 1
        if time.time() - t_log >= 2.0:
            log.info(f"{n_writes / (time.time() - t_log):.0f} Hz")
            n_writes = 0
            t_log = time.time()

        elapsed = time.time() - t0
        if elapsed < INNER_DT:
            time.sleep(INNER_DT - elapsed)

    log.info("stopping")
