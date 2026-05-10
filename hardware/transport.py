"""Robot transport process.

Owns the kortex connection. Always running while ``--robot-source hardware``.
Reads joint state from the robot at 500 Hz into shm_state. Reads commands
from shm_cmd at 500 Hz and dispatches them — torques in low-level mode,
position setpoints in high-level mode, or no commands when idle.

Controller swaps go through the Safe State Transition (SST):
    stop sending current commands
      → set_servoing_mode(high)
      → JointMove(home)             (always)
      → set_servoing_mode(target)
      → resume

The transport is the only process touching the kortex API. Controller
subprocesses just read state from shm and write commands to shm; the
transport reads those and forwards. This keeps lifecycle responsibility
in one place.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time

import numpy as np

from .kinova import KinovaHardware
from .osc import HOME_DEG, kinova_deg_to_rad


# ── Timing ────────────────────────────────────────────────────────────────────
TRANSPORT_HZ = 500
INNER_DT = 1.0 / TRANSPORT_HZ

# ── Command modes (must match controllers/base.py) ────────────────────────────
CMD_MODE_IDLE     = 0
CMD_MODE_TORQUE   = 1
CMD_MODE_POSITION = 2

# ── Phase codes for status broadcasting ───────────────────────────────────────
PHASE_BOOT          = 0   # connecting / clearing faults / waiting ready
PHASE_HOMING        = 1   # JointMove home
PHASE_READY         = 2   # high-level servoing, idle
PHASE_RUNNING       = 3   # actively dispatching commands
PHASE_SWAPPING      = 4   # in middle of an SST
PHASE_FAULT         = 5
PHASE_SHUTDOWN      = 6


def _np_arr(shm: mp.Array, dtype=np.float64) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=dtype)


def transport_process(
    ip: str,
    *,
    # Joint state out (read-only for controllers).
    shm_q,                 # mp.Array('d', 7) — joint angles (rad)
    shm_dq,                # mp.Array('d', 7) — joint velocities (rad/s)
    shm_state_seq,         # mp.Value('I')   — bumped at every state read
    # Command in (controllers write here).
    shm_cmd_mode,          # mp.Value('B')  — CMD_MODE_*
    shm_tau,               # mp.Array('d', 7) — torques (Nm)
    shm_qtarget,           # mp.Array('d', 7) — joint setpoints (rad)
    shm_gripper,           # mp.Array('d', 1) — gripper [0..1]
    shm_cmd_seq,           # mp.Value('I')   — bumped by controllers each write
    # Lifecycle / SST events.
    stop_ev,               # mp.Event — exit cleanly
    swap_request_ev,       # mp.Event — set by dispatcher to trigger SST
    swap_target_mode,      # mp.Value('B') — CMD_MODE_* the swap should land in
    swap_done_ev,          # mp.Event — transport sets when swap completes
    # Telemetry.
    shm_hz,                # mp.Array('d', 1) — measured loop rate
    shm_phase,             # mp.Value('B') — PHASE_*
    shm_fault_msg,         # mp.Array('c', 256) — short fault string
    log_q=None,            # mp.Queue — log relay for the browser
):
    """Main transport loop. Run as ``mp.Process(target=transport_process)``."""

    from .log_relay import install_log_relay
    install_log_relay(log_q, source="transport")
    log = logging.getLogger("transport")

    def set_phase(p: int) -> None:
        with shm_phase.get_lock():
            shm_phase.value = p

    def set_fault(msg: str) -> None:
        b = msg.encode("utf-8")[:255]
        with shm_fault_msg.get_lock():
            buf = shm_fault_msg.get_obj()
            for i in range(len(buf)):
                buf[i] = b[i] if i < len(b) else 0

    set_phase(PHASE_BOOT)
    hw = KinovaHardware(ip)
    try:
        log.info(f"connecting to {ip}…")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready(timeout=15.0):
            set_phase(PHASE_FAULT)
            set_fault("not ready after connect")
            log.error("not ready — aborting")
            return

        # Always start by going home in high-level position mode.
        hw.set_servoing_mode(low_level=False)
        time.sleep(0.5)
        set_phase(PHASE_HOMING)
        log.info("going home…")
        if not hw.go_to_joints(HOME_DEG):
            set_phase(PHASE_FAULT)
            set_fault("initial JointMove home failed")
            log.error("initial home failed")
            return

        # We're now at home, high-level, ready.
        set_phase(PHASE_READY)
        log.info("ready (idle, high-level position)")

        # Initial state read. We start in IDLE / high-level / no command stream.
        # Read state at 500 Hz and publish to shm. Don't send any command.
        state = hw.read_state()
        pos_deg = state.positions_deg.copy()
        vel_deg = state.velocities_deg.copy()
        _publish_state(shm_q, shm_dq, shm_state_seq, pos_deg, vel_deg)

        last_cmd_seq = 0
        last_dispatched_mode = CMD_MODE_IDLE
        in_low_level = False
        faulted = False              # latched after fault detected

        n_iters = 0
        t_rate = time.time()
        while not stop_ev.is_set():
            iter_t0 = time.time()

            # ── Honour swap requests ────────────────────────────────────
            if swap_request_ev.is_set():
                with swap_target_mode.get_lock():
                    target_mode = int(swap_target_mode.value)
                _do_swap(hw, log, target_mode, set_phase, set_fault,
                         in_low_level)
                # After swap we always end up at home in the target mode.
                # If target is torque, we're now in low-level.
                in_low_level = (target_mode == CMD_MODE_TORQUE)
                last_dispatched_mode = target_mode
                # Read post-swap state.
                state = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()
                _publish_state(shm_q, shm_dq, shm_state_seq, pos_deg, vel_deg)
                # Clear fault latch — recovery succeeded.
                faulted = False
                set_fault("")
                swap_request_ev.clear()
                swap_done_ev.set()
                set_phase(PHASE_RUNNING if target_mode != CMD_MODE_IDLE
                          else PHASE_READY)
                continue  # restart loop pacing

            # ── Normal iteration: dispatch the current command ──────────
            with shm_cmd_mode.get_lock():
                cmd_mode = int(shm_cmd_mode.value)

            # When latched-faulted, skip command dispatch entirely. Just
            # read state so the UI keeps showing live joint angles.
            # Recovery happens on the next swap_request_ev.
            if faulted:
                state = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()
            elif cmd_mode == CMD_MODE_TORQUE and in_low_level:
                with shm_tau.get_lock():
                    tau = _np_arr(shm_tau)[:7].copy()
                with shm_gripper.get_lock():
                    grip = float(_np_arr(shm_gripper)[0])
                # Defensive NaN/inf guard — refuse to send garbage to robot.
                if not np.all(np.isfinite(tau)):
                    log.error(f"controller produced non-finite tau: {tau}")
                    faulted = True
                    set_phase(PHASE_FAULT)
                    set_fault("controller produced non-finite torques")
                    state = hw.read_state()
                    pos_deg = state.positions_deg.copy()
                    vel_deg = state.velocities_deg.copy()
                else:
                    state = hw.send_torques(tau, pos_deg, gripper_position=grip)
                    pos_deg = state.positions_deg.copy()
                    vel_deg = state.velocities_deg.copy()
            elif cmd_mode == CMD_MODE_POSITION and not in_low_level:
                with shm_cmd_seq.get_lock():
                    cur_seq = int(shm_cmd_seq.value)
                if cur_seq != last_cmd_seq:
                    last_cmd_seq = cur_seq
                    with shm_qtarget.get_lock():
                        qtgt = _np_arr(shm_qtarget)[:7].copy()
                    qtgt_deg = np.rad2deg(qtgt)
                    # Scale duration with the largest joint delta. Min 0.5s
                    # so tiny moves still feel snappy; +0.05 s/deg so big
                    # moves get a smooth ramp.
                    dmax_deg = float(np.max(np.abs(qtgt_deg - pos_deg)))
                    duration = max(0.5, 0.5 + 0.05 * dmax_deg)
                    log.info(f"position step → {qtgt_deg.round(1)}  "
                             f"(Δ={dmax_deg:.1f}°, dur={duration:.1f}s)")
                    hw.go_to_joints(qtgt_deg, duration=duration)
                state = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()
            else:
                state = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()

            # Robot-side fault detection (every cycle, free with the read).
            if not faulted and state.fault_bank != 0:
                actuators = ", ".join(str(i) for i in state.faulted_actuators)
                msg = (f"kortex fault: bank=0x{state.fault_bank:08x} "
                       f"actuator={actuators or 'base'}")
                log.error(msg)
                faulted = True
                set_phase(PHASE_FAULT)
                set_fault(msg[:255])

            _publish_state(shm_q, shm_dq, shm_state_seq, pos_deg, vel_deg)

            # ── Pace ────────────────────────────────────────────────────
            n_iters += 1
            dt = time.time() - t_rate
            if dt >= 1.0:
                with shm_hz.get_lock():
                    _np_arr(shm_hz)[0] = n_iters / dt
                n_iters = 0
                t_rate = time.time()

            elapsed = time.time() - iter_t0
            if elapsed < INNER_DT:
                time.sleep(INNER_DT - elapsed)

    except Exception as exc:
        log.exception(f"transport crashed: {exc}")
        set_phase(PHASE_FAULT)
        set_fault(str(exc)[:255])
    finally:
        set_phase(PHASE_SHUTDOWN)
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
            log.warning(f"shutdown warning: {exc}")
        hw.disconnect()
        log.info("transport done")


def _publish_state(shm_q, shm_dq, shm_state_seq,
                   pos_deg: np.ndarray, vel_deg: np.ndarray) -> None:
    q = kinova_deg_to_rad(pos_deg)
    dq = np.deg2rad(vel_deg)
    with shm_q.get_lock():
        _np_arr(shm_q)[:7] = q
    with shm_dq.get_lock():
        _np_arr(shm_dq)[:7] = dq
    with shm_state_seq.get_lock():
        shm_state_seq.value = (shm_state_seq.value + 1) & 0xFFFFFFFF


def _do_swap(hw: KinovaHardware, log,
             target_mode: int, set_phase, set_fault,
             in_low_level: bool) -> None:
    """The Safe State Transition.

    Always goes through home pose. Steps:
      1. If currently in low-level: turn off torque mode, switch to
         high-level.
      2. Clear faults + wait_until_ready.
      3. JointMove HOME.
      4. If target is torque mode: switch to low_level + torque on.
         Otherwise (idle / position): stay in high-level.
    """
    set_phase(4)  # PHASE_SWAPPING
    log.info(f"SST start → target={target_mode}")
    try:
        if in_low_level:
            hw.set_torque_mode(False)
            time.sleep(0.3)
            hw.set_servoing_mode(low_level=False)
            time.sleep(0.5)
        hw.clear_faults()
        time.sleep(0.5)
        if not hw.wait_until_ready(timeout=10.0):
            log.warning("SST: not ready before homing")
        log.info("SST: homing…")
        if not hw.go_to_joints(HOME_DEG):
            log.error("SST: JointMove home failed")
            set_fault("home failed during SST")
            return
        if target_mode == CMD_MODE_TORQUE:
            time.sleep(0.5)
            hw.set_servoing_mode(low_level=True)
            time.sleep(0.5)
            hw.set_torque_mode(True)
            log.info("SST: now in low_level + torque")
        else:
            log.info("SST: holding in high-level (idle/position)")
    except Exception as exc:
        log.exception(f"SST failed: {exc}")
        set_fault(f"SST: {exc}"[:255])
