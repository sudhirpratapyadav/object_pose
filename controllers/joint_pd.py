"""Joint-space PD + gravity-compensation controller.

Runs at 500 Hz in low-level torque mode:

    tau = kp · (q_des − q) − kd · q̇ + nle(q, q̇)

UI publishes ``q_des`` (rad) into ``shm_qtarget`` and gain values
``kp[7], kd[7]`` into ``shm_gains`` (concatenated). Each control cycle
the controller reads them and writes torques.

This is the right shape for "I drag a slider, the robot follows":
slider change reaches the controller within one cycle (≤2 ms), so motion
feels real-time and the response stiffness is operator-tunable.
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
    kinova_deg_to_rad,
)


CONTROL_HZ = 500
INNER_DT = 1.0 / CONTROL_HZ


def _np_arr(shm) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def joint_pd_controller_process(
    shm_q, shm_dq, shm_state_seq,
    shm_cmd_mode, shm_tau, shm_qtarget, shm_gripper, shm_cmd_seq,
    stop_ev,
    *,
    mjcf_path: str,
    ee_frame: str = "pinch_site",
    shm_gains=None,                 # mp.Array('d', 14) — kp[7] || kd[7]
    log_q=None,
):
    from hardware.log_relay import install_log_relay
    install_log_relay(log_q, source="ctrl-joint-pd")
    log = logging.getLogger("ctrl-joint-pd")

    log.info(f"loading PinocchioArm({mjcf_path})…")
    arm = PinocchioArm(mjcf_path, ee_frame=ee_frame)
    log.info("ready")

    # Seed the target to current joint state so the first iteration is
    # "track yourself" — no lurch on engage.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with shm_state_seq.get_lock():
            if int(shm_state_seq.value) > 0:
                break
        time.sleep(0.01)
    with shm_q.get_lock():
        q0 = _np_arr(shm_q)[:7].copy()
    with shm_qtarget.get_lock():
        _np_arr(shm_qtarget)[:7] = q0
    log.info(f"seeded q_des to current pose: {np.rad2deg(q0).round(1)} deg")

    with shm_cmd_mode.get_lock():
        shm_cmd_mode.value = CMD_MODE_TORQUE

    last_state_seq = 0
    n_writes = 0
    t_log = time.time()

    # If gains shm wasn't provided, fall back to a static default (lets the
    # controller still run if the dispatcher hasn't wired the shm slot yet).
    static_kp = np.array([40.0, 40.0, 40.0, 30.0, 20.0, 10.0, 5.0])
    static_kd = np.array([4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5])

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
            q_des = _np_arr(shm_qtarget)[:7].copy()

        if shm_gains is not None:
            with shm_gains.get_lock():
                gains = _np_arr(shm_gains)[:14].copy()
            kp = gains[:7]
            kd = gains[7:]
        else:
            kp = static_kp
            kd = static_kd

        # Gravity + Coriolis + inertia-coupling = nle.
        try:
            _M, nle, _Jdot_dq = arm.dynamics(q, dq)
        except Exception as exc:
            log.exception(f"dynamics() failed: {exc}")
            nle = np.zeros(7)

        tau = kp * (q_des - q) - kd * dq + nle
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
