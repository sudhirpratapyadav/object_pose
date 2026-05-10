"""Gravity-compensation controller.

Runs at 500 Hz against the joint state in shm. Uses Pinocchio's
``nonLinearEffects`` to compute gravity + Coriolis torques and writes them
to shm_tau. Transport (in low-level torque mode) sends those to the robot,
so the arm floats compliantly: you can push it around by hand and it will
follow without resisting.

The transport handles the high→low-level mode switch in the SST. By the
time this controller starts, the robot is already at home + low-level +
torque-mode-on. We only set ``cmd_mode = TORQUE`` and stream torques.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from hardware import (
    CMD_MODE_TORQUE,
    MAX_JOINT_TORQUE,
    PinocchioArm,
)


CONTROL_HZ = 500
INNER_DT = 1.0 / CONTROL_HZ


def _np_arr(shm) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def gravcomp_controller_process(
    shm_q, shm_dq, shm_state_seq,
    shm_cmd_mode, shm_tau, shm_qtarget, shm_gripper, shm_cmd_seq,
    stop_ev,
    *,
    mjcf_path: str,
    ee_frame: str = "pinch_site",
    log_q=None,
):
    """Pinocchio-based gravity + Coriolis compensation."""
    from hardware.log_relay import install_log_relay
    install_log_relay(log_q, source="ctrl-gravcomp")
    log = logging.getLogger("ctrl-gravcomp")

    log.info(f"loading PinocchioArm({mjcf_path})…")
    arm = PinocchioArm(mjcf_path, ee_frame=ee_frame)
    log.info("ready")

    # Tell the transport we want torque mode — the dispatcher should
    # already have done the SST so we're at home, low-level, torque-on.
    with shm_cmd_mode.get_lock():
        shm_cmd_mode.value = CMD_MODE_TORQUE

    last_state_seq = 0
    n_writes = 0
    t_log = time.time()

    while not stop_ev.is_set():
        t0 = time.time()

        # Wait for fresh state.
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

        # Pure nle = gravity + Coriolis. compute_osc_torques unused.
        _M, nle, _Jdot_dq = arm.dynamics(q, dq)
        tau = np.clip(nle, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)

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
