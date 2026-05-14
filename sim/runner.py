"""MuJoCo simulator process — acts as a drop-in for hardware transport.

This single process owns the MuJoCo model and provides two services:

  1. **Physics + transport.** Steps physics at sim_hz, bridges joint state
     and torque commands through the same shared-memory protocol the
     hardware transport uses (shm_q / shm_dq / shm_tau / shm_cmd_mode /
     shm_qtarget / shm_phase / swap events …). The controller dispatcher
     and policy plumbing in web_server.py work unchanged.
  2. **Render.** Renders RGB + camera-depth from the same camera the real
     RealSense uses, writes into rgb_shm / depth_stream every Nth step.

When called *without* the hardware-bridge kwargs (legacy path), it falls
back to the original position-mode behaviour driven by ctrl_arr.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import mujoco
import numpy as np

from hardware import HOME_DEG, kinova_deg_to_rad


# Command modes — must match controllers/base.py + hardware/transport.py.
CMD_MODE_IDLE     = 0
CMD_MODE_TORQUE   = 1
CMD_MODE_POSITION = 2

# Phases — must match hardware/transport.py.
PHASE_BOOT     = 0
PHASE_HOMING   = 1
PHASE_READY    = 2
PHASE_RUNNING  = 3
PHASE_SWAPPING = 4
PHASE_FAULT    = 5
PHASE_SHUTDOWN = 6


def _np_arr(shm: mp.Array, dtype=np.float64) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=dtype)


def _set_phase(shm_phase, p: int) -> None:
    if shm_phase is None:
        return
    with shm_phase.get_lock():
        shm_phase.value = p


def _publish_state(shm_q, shm_dq, shm_state_seq, q_rad, dq_rad) -> None:
    if shm_q is None:
        return
    with shm_q.get_lock():
        _np_arr(shm_q)[:7] = q_rad
    with shm_dq.get_lock():
        _np_arr(shm_dq)[:7] = dq_rad
    with shm_state_seq.get_lock():
        shm_state_seq.value = (shm_state_seq.value + 1) & 0xFFFFFFFF


def sim_worker(
    mjcf_path: str,
    cam_name: str,
    *,
    # rgb_shm
    rgb_shm_name: str,
    rgb_seq,                # mp.Value('I')
    rgb_w: int,
    rgb_h: int,
    # qpos_shm (legacy bridge to robot_renderer / FK)
    qpos_arr,               # mp.Array('d', nq)
    qpos_seq,               # mp.Value('I')
    nq: int,
    # control input (legacy position-mode; ignored when hardware-bridge kwargs provided)
    ctrl_arr,               # mp.Array('d', nu)
    ctrl_seq,               # mp.Value('I')
    nu: int,
    # camera placement (intrinsics-aware values from caller's calib)
    cam_pos,
    cam_quat_wxyz,
    cam_fovy_deg: float,
    # optional camera-depth side channel
    depth_shm_name: str | None = None,
    rgbd_seq=None,
    depth_max_m: float = 10.0,
    # lifecycle
    stop_ev=None,
    open_viewer: bool = False,
    sim_hz: int = 500,
    render_hz: int = 30,
    # ── Hardware-bridge kwargs (when --robot-source sim) ──────────────────
    # When ALL of these are provided, sim behaves as a transport: bridges
    # joint state out, accepts torques in, honours SST swap events.
    shm_q=None,             # mp.Array('d', 7) — joint angles (rad)
    shm_dq=None,            # mp.Array('d', 7) — joint velocities (rad/s)
    shm_state_seq=None,
    shm_cmd_mode=None,      # mp.Value('B')
    shm_tau=None,           # mp.Array('d', 7)
    shm_qtarget=None,       # mp.Array('d', 7) — joint setpoints in position mode
    shm_gripper=None,       # mp.Array('d', 1) — [0..1]
    shm_cmd_seq=None,
    swap_request_ev=None,
    swap_target_mode=None,
    swap_done_ev=None,
    shm_hz=None,
    shm_phase=None,
    shm_fault_msg=None,
):
    """Sim process target. Owns the MjModel, MjData, Renderer, and viewer.

    The hardware-bridge kwargs are optional; legacy callers (no bridge)
    get the position-mode behaviour the original sim_worker had.
    """
    # Bridge mode is "on" when the transport-API shms are all wired up.
    hw_bridge = (shm_q is not None and shm_dq is not None
                 and shm_tau is not None and shm_cmd_mode is not None)

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Re-apply calibration on the camera the depth pipeline uses.
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        print(f"[sim] camera '{cam_name}' not found; aborting", flush=True)
        return
    model.cam_pos[cid] = np.asarray(cam_pos, dtype=np.float64)
    model.cam_quat[cid] = np.asarray(cam_quat_wxyz, dtype=np.float64)
    model.cam_fovy[cid] = float(cam_fovy_deg)
    model.cam_resolution[cid] = (int(rgb_w), int(rgb_h))

    # Reset to home keyframe if present.
    home_idx = -1
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
            home_idx = i
            break
    if home_idx >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_idx)
    else:
        mujoco.mj_resetData(model, data)

    # In bridge mode we override the keyframe-derived arm joints with the
    # hardware HOME_DEG so the sim matches what the real transport would
    # publish (the keyframe's joint_1=0 doesn't match our hardware default
    # of joint_1=π/2). After this, mj_forward to settle inertia.
    if hw_bridge:
        home_rad = kinova_deg_to_rad(np.asarray(HOME_DEG, dtype=np.float64))
        data.qpos[:7] = home_rad
        data.qvel[:7] = 0.0
        mujoco.mj_forward(model, data)

    # Seed ctrl from current pose (position-mode legacy path).
    if not hw_bridge and model.nu > 0:
        for ai in range(model.nu):
            j = model.actuator_trnid[ai, 0]
            if model.actuator_trntype[ai] == int(mujoco.mjtTrn.mjTRN_JOINT):
                qadr = model.jnt_qposadr[j]
                data.ctrl[ai] = data.qpos[qadr]
            else:
                data.ctrl[ai] = 0.0
    if ctrl_arr is not None:
        with ctrl_arr.get_lock():
            np.frombuffer(ctrl_arr.get_obj(), dtype=np.float64)[:nu] = data.ctrl[:nu]
        last_ctrl_seq = int(ctrl_seq.value)
    else:
        last_ctrl_seq = 0

    # Renderer (RGB) + optional depth renderer.
    renderer = mujoco.Renderer(model, height=rgb_h, width=rgb_w)
    depth_renderer = None
    depth_shm = None
    depth_buf = None
    if depth_shm_name is not None:
        depth_renderer = mujoco.Renderer(model, height=rgb_h, width=rgb_w)
        depth_renderer.enable_depth_rendering()
        depth_shm = shared_memory.SharedMemory(name=depth_shm_name)
        depth_buf = np.ndarray((rgb_h, rgb_w), dtype=np.float32, buffer=depth_shm.buf)

    rgb_shm = shared_memory.SharedMemory(name=rgb_shm_name)
    rgb_buf = np.ndarray((rgb_h, rgb_w, 3), dtype=np.uint8, buffer=rgb_shm.buf)

    viewer = None
    if open_viewer:
        try:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)
            print("[sim] passive viewer window opened", flush=True)
        except Exception as exc:
            print(f"[sim] viewer not available: {exc}", flush=True)

    inner_dt = 1.0 / sim_hz
    render_every = max(1, int(round(sim_hz / render_hz)))
    step = 0
    n_renders = 0
    n_steps = 0
    t_log = time.time()

    # Bridge-mode state.
    if hw_bridge:
        _set_phase(shm_phase, PHASE_BOOT)
        # Publish initial state.
        _publish_state(shm_q, shm_dq, shm_state_seq,
                       data.qpos[:7].copy(), data.qvel[:7].copy())
        # Seed qtarget = home so position mode doesn't lurch.
        if shm_qtarget is not None:
            with shm_qtarget.get_lock():
                _np_arr(shm_qtarget)[:7] = data.qpos[:7].copy()
        _set_phase(shm_phase, PHASE_READY)
        # Joint indexes in actuator order (arm 1..7 then gripper). The
        # torque XML has motors named joint_1..7 at positions 0..6.
        n_arm = 7
        # Last seen position-mode setpoint sequence.
        last_pos_cmd_seq = 0
        n_iters = 0
        t_rate = time.time()

    print(f"[sim] running ({sim_hz} Hz step, ~{render_hz} Hz render, "
          f"nq={model.nq}, nu={model.nu}, bridge={'on' if hw_bridge else 'off'})",
          flush=True)

    try:
        while not stop_ev.is_set():
            t0 = time.time()

            if hw_bridge:
                # ── Honour SST swap requests ──────────────────────────────
                if swap_request_ev is not None and swap_request_ev.is_set():
                    with swap_target_mode.get_lock():
                        target_mode = int(swap_target_mode.value)
                    _set_phase(shm_phase, PHASE_SWAPPING)
                    # Always pass through home: snap arm joints to HOME, zero vel.
                    home_rad = kinova_deg_to_rad(
                        np.asarray(HOME_DEG, dtype=np.float64))
                    # Smooth ramp would be nicer but a single-step snap keeps the
                    # protocol simple. Real hardware takes ~8 s for this; we just
                    # sleep briefly so the UI status pill is visible.
                    data.qpos[:7] = home_rad
                    data.qvel[:7] = 0.0
                    # Zero gripper joints (8..14 in qpos) too.
                    if model.nq > 7:
                        data.qpos[7:] = 0.0
                        data.qvel[7:] = 0.0
                    # Zero torques + reseed qtarget = home so the next step
                    # doesn't jolt.
                    data.ctrl[:] = 0.0
                    mujoco.mj_forward(model, data)
                    if shm_qtarget is not None:
                        with shm_qtarget.get_lock():
                            _np_arr(shm_qtarget)[:7] = home_rad
                    if shm_cmd_seq is not None:
                        with shm_cmd_seq.get_lock():
                            shm_cmd_seq.value = (shm_cmd_seq.value + 1) & 0xFFFFFFFF
                    _publish_state(shm_q, shm_dq, shm_state_seq,
                                   home_rad, np.zeros(7))
                    # Fake a brief settle so the UI shows the transition.
                    time.sleep(0.3)
                    swap_request_ev.clear()
                    if swap_done_ev is not None:
                        swap_done_ev.set()
                    _set_phase(shm_phase, PHASE_RUNNING
                               if target_mode != CMD_MODE_IDLE else PHASE_READY)
                    continue

                # ── Read command mode and apply ─────────────────────────
                with shm_cmd_mode.get_lock():
                    cmd_mode = int(shm_cmd_mode.value)

                if cmd_mode == CMD_MODE_TORQUE:
                    # Apply joint torques (motors take ctrl == torque directly
                    # in the torque XML).
                    with shm_tau.get_lock():
                        tau = _np_arr(shm_tau)[:n_arm].copy()
                    if np.all(np.isfinite(tau)):
                        data.ctrl[:n_arm] = tau
                    # Gripper actuator (last actuator): map [0,1] -> [0,255].
                    if shm_gripper is not None and model.nu > n_arm:
                        with shm_gripper.get_lock():
                            grip = float(_np_arr(shm_gripper)[0])
                        data.ctrl[n_arm] = float(np.clip(grip, 0.0, 1.0)) * 255.0
                elif cmd_mode == CMD_MODE_POSITION:
                    # Position-mode in sim torque XML: emulate the kortex
                    # JointMove by snapping qpos toward the target. This is
                    # a placeholder — the real transport runs a trajectory.
                    # For sim we just set ctrl=0 and step qpos a fraction
                    # of the way each cycle.
                    if shm_cmd_seq is not None:
                        with shm_cmd_seq.get_lock():
                            cur = int(shm_cmd_seq.value)
                        if cur != last_pos_cmd_seq:
                            last_pos_cmd_seq = cur
                            with shm_qtarget.get_lock():
                                qtgt = _np_arr(shm_qtarget)[:7].copy()
                            data.qpos[:7] = qtgt
                            data.qvel[:7] = 0.0
                            mujoco.mj_forward(model, data)
                    data.ctrl[:n_arm] = 0.0
                else:  # CMD_MODE_IDLE
                    # On real hardware "idle" means high-level position
                    # servoing — the firmware holds the last pose. In sim
                    # we mimic this with gravity compensation so the arm
                    # doesn't drop under its own weight. mj_rne with
                    # flg_acc=0 computes the bias forces (gravity +
                    # Coriolis) needed to hold qvel at its current value;
                    # we apply those as torque ctrl.
                    qfrc = np.zeros(model.nv)
                    mujoco.mj_rne(model, data, 0, qfrc)
                    data.ctrl[:n_arm] = qfrc[:n_arm]

            else:
                # Legacy position-mode bridge (no robot-source=sim).
                cur_ctrl_seq = int(ctrl_seq.value)
                if cur_ctrl_seq != last_ctrl_seq:
                    last_ctrl_seq = cur_ctrl_seq
                    with ctrl_arr.get_lock():
                        incoming = np.frombuffer(ctrl_arr.get_obj(),
                                                 dtype=np.float64)[:nu].copy()
                    data.ctrl[:nu] = incoming

            # ── Step physics ──────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Publish state (bridge + legacy qpos) ──────────────────────
            with qpos_arr.get_lock():
                np.frombuffer(qpos_arr.get_obj(), dtype=np.float64)[:nq] = data.qpos
                qpos_seq.value = (qpos_seq.value + 1) & 0xFFFFFFFF
            if hw_bridge:
                _publish_state(shm_q, shm_dq, shm_state_seq,
                               data.qpos[:7].copy(), data.qvel[:7].copy())
                n_iters += 1
                dt_r = time.time() - t_rate
                if dt_r >= 1.0:
                    if shm_hz is not None:
                        with shm_hz.get_lock():
                            _np_arr(shm_hz)[0] = n_iters / dt_r
                    n_iters = 0
                    t_rate = time.time()

            # ── Render ────────────────────────────────────────────────────
            if step % render_every == 0:
                renderer.update_scene(data, camera=cam_name)
                rgb = renderer.render()
                rgb_buf[...] = rgb
                with rgb_seq.get_lock():
                    rgb_seq.value = (rgb_seq.value + 1) & 0xFFFFFFFF
                if depth_renderer is not None:
                    depth_renderer.update_scene(data, camera=cam_name)
                    depth = depth_renderer.render()
                    np.clip(depth, 0.0, depth_max_m, out=depth)
                    depth_buf[...] = depth
                    if rgbd_seq is not None:
                        with rgbd_seq.get_lock():
                            rgbd_seq.value = (rgbd_seq.value + 1) & 0xFFFFFFFF
                n_renders += 1

            if viewer is not None:
                viewer.sync()

            step += 1
            n_steps += 1

            elapsed = time.time() - t0
            if elapsed < inner_dt:
                time.sleep(inner_dt - elapsed)

            now = time.time()
            if now - t_log >= 2.0:
                step_hz = n_steps / (now - t_log)
                rend_hz = n_renders / (now - t_log)
                print(f"[sim] step {step_hz:.0f} Hz  render {rend_hz:.1f} Hz",
                      flush=True)
                n_steps = n_renders = 0
                t_log = now
    finally:
        if hw_bridge:
            _set_phase(shm_phase, PHASE_SHUTDOWN)
        if viewer is not None:
            try: viewer.close()
            except Exception: pass
        try: renderer.close()
        except Exception: pass
        if depth_renderer is not None:
            try: depth_renderer.close()
            except Exception: pass
        if depth_shm is not None:
            depth_shm.close()
        rgb_shm.close()
        print("[sim] worker exiting", flush=True)
