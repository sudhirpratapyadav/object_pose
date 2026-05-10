"""Idle controller: don't send anything.

Sets ``cmd_mode = idle``. The transport sees that and stops dispatching
commands; the robot sits in high-level (position) servoing mode and the
firmware holds the last commanded pose. This is the rest state between
controller swaps and the default after boot.
"""

from __future__ import annotations

import logging
import time

from hardware import CMD_MODE_IDLE


def idle_controller_process(
    shm_q, shm_dq, shm_state_seq,
    shm_cmd_mode, shm_tau, shm_qtarget, shm_gripper, shm_cmd_seq,
    stop_ev,
    *,
    log_q=None,
):
    """Park the cmd_mode at IDLE. No work to do."""
    from hardware.log_relay import install_log_relay
    install_log_relay(log_q, source="ctrl-idle")
    log = logging.getLogger("ctrl-idle")

    with shm_cmd_mode.get_lock():
        shm_cmd_mode.value = CMD_MODE_IDLE
    log.info("idle: cmd_mode=IDLE, no commands sent")
    while not stop_ev.is_set():
        time.sleep(0.1)
    log.info("idle: stopping")
