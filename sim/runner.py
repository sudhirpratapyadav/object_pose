"""MuJoCo simulator process.

Steps physics at 500 Hz, renders RGB into rgb_shm at ~30 Hz (every Nth step),
writes qpos into qpos_shm, optionally opens a passive viewer window.

Reads target ctrl from a small mp.Array each step (placeholder for stage 4
control). For now, ctrl defaults to whatever the home keyframe specifies.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import mujoco
import numpy as np


def sim_worker(
    mjcf_path: str,
    cam_name: str,
    *,
    # rgb_shm
    rgb_shm_name: str,
    rgb_seq,                # mp.Value('I')
    rgb_w: int,
    rgb_h: int,
    # qpos_shm
    qpos_arr,               # mp.Array('d', nq)
    qpos_seq,               # mp.Value('I')
    nq: int,
    # control input (placeholder for stage 4)
    ctrl_arr,               # mp.Array('d', nu) — read each step into data.ctrl
    ctrl_seq,               # mp.Value('I')
    nu: int,
    # camera calibration (already applied to MJCF in parent before pickling
    # would not work; instead the parent serialises pos/quat/fovy/res via the
    # YAML and we re-apply here)
    cam_pos,                # tuple(3) float
    cam_quat_wxyz,          # tuple(4) float
    cam_fovy_deg: float,
    # optional camera-depth side channel
    depth_shm_name: str | None = None,    # rgb_h * rgb_w float32
    rgbd_seq=None,                        # mp.Value bumped on each depth write
    depth_max_m: float = 10.0,            # clip depth to suppress sky/far
    # lifecycle
    stop_ev=None,           # mp.Event
    open_viewer: bool = False,
    sim_hz: int = 500,
    render_hz: int = 30,
):
    """Sim process target. Owns the MjModel, MjData, Renderer, and viewer."""

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Re-apply calibration (parent passed the values; we set them here to
    # avoid pickling the MjModel).
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        print(f"[sim] camera '{cam_name}' not found; aborting", flush=True)
        return
    model.cam_pos[cid] = np.asarray(cam_pos, dtype=np.float64)
    model.cam_quat[cid] = np.asarray(cam_quat_wxyz, dtype=np.float64)
    model.cam_fovy[cid] = float(cam_fovy_deg)
    model.cam_resolution[cid] = (int(rgb_w), int(rgb_h))

    # Initialise from the 'home' keyframe if present, else neutral pose.
    home_idx = -1
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
            home_idx = i
            break
    if home_idx >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_idx)
    else:
        mujoco.mj_resetData(model, data)

    # Seed ctrl from home qpos so position actuators don't snap on startup.
    if model.nu > 0:
        # If actuators are position-mode driven by 'joint' refs, ctrl maps to
        # joint angles. Initialise to current qpos[joint_qposadr].
        for ai in range(model.nu):
            j = model.actuator_trnid[ai, 0]  # joint id (or body id for general)
            if model.actuator_trntype[ai] == int(mujoco.mjtTrn.mjTRN_JOINT):
                qadr = model.jnt_qposadr[j]
                data.ctrl[ai] = data.qpos[qadr]
            else:
                data.ctrl[ai] = 0.0
    # Seed the shared ctrl slot so the parent can later read it back.
    with ctrl_arr.get_lock():
        np.frombuffer(ctrl_arr.get_obj(), dtype=np.float64)[:nu] = data.ctrl[:nu]
    last_ctrl_seq = int(ctrl_seq.value)

    # Renderer (headless EGL/OSMesa). Created after MjData.
    # Need a separate Renderer instance for depth: enable_depth_rendering()
    # switches the renderer permanently into depth mode.
    renderer = mujoco.Renderer(model, height=rgb_h, width=rgb_w)
    depth_renderer = None
    depth_shm = None
    depth_buf = None
    if depth_shm_name is not None:
        depth_renderer = mujoco.Renderer(model, height=rgb_h, width=rgb_w)
        depth_renderer.enable_depth_rendering()
        depth_shm = shared_memory.SharedMemory(name=depth_shm_name)
        depth_buf = np.ndarray((rgb_h, rgb_w), dtype=np.float32, buffer=depth_shm.buf)

    # Map rgb_shm
    rgb_shm = shared_memory.SharedMemory(name=rgb_shm_name)
    rgb_buf = np.ndarray((rgb_h, rgb_w, 3), dtype=np.uint8, buffer=rgb_shm.buf)

    # Optional viewer.
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

    print(f"[sim] running ({sim_hz} Hz step, ~{render_hz} Hz render, "
          f"nq={model.nq}, nu={model.nu})", flush=True)

    try:
        while not stop_ev.is_set():
            t0 = time.time()

            # Pick up new ctrl from parent if seq advanced.
            cur_ctrl_seq = int(ctrl_seq.value)
            if cur_ctrl_seq != last_ctrl_seq:
                last_ctrl_seq = cur_ctrl_seq
                with ctrl_arr.get_lock():
                    incoming = np.frombuffer(ctrl_arr.get_obj(),
                                             dtype=np.float64)[:nu].copy()
                data.ctrl[:nu] = incoming

            mujoco.mj_step(model, data)

            # Publish qpos to the shared slot (latest-wins; bumped each step).
            with qpos_arr.get_lock():
                np.frombuffer(qpos_arr.get_obj(), dtype=np.float64)[:nq] = data.qpos
                qpos_seq.value = (qpos_seq.value + 1) & 0xFFFFFFFF

            if step % render_every == 0:
                renderer.update_scene(data, camera=cam_name)
                rgb = renderer.render()
                rgb_buf[...] = rgb
                with rgb_seq.get_lock():
                    rgb_seq.value = (rgb_seq.value + 1) & 0xFFFFFFFF
                # Camera-depth side channel: render depth (already in metres
                # per MuJoCo's Renderer; clip to suppress sky/zfar values).
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

            # Pace: sleep to maintain sim_hz best-effort.
            elapsed = time.time() - t0
            if elapsed < inner_dt:
                time.sleep(inner_dt - elapsed)

            # 1 Hz heartbeat.
            now = time.time()
            if now - t_log >= 2.0:
                step_hz = n_steps / (now - t_log)
                rend_hz = n_renders / (now - t_log)
                print(f"[sim] step {step_hz:.0f} Hz  render {rend_hz:.1f} Hz",
                      flush=True)
                n_steps = n_renders = 0
                t_log = now
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        try:
            renderer.close()
        except Exception:
            pass
        if depth_renderer is not None:
            try:
                depth_renderer.close()
            except Exception:
                pass
        if depth_shm is not None:
            depth_shm.close()
        rgb_shm.close()
        print("[sim] worker exiting", flush=True)
