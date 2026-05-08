"""
Live object-pose viewer.

Pipeline:
  RealSense (thread)
      └── shm "rgb" ──► depth process (Depth Anything V2)
                            ├── shm "depth"
                            └── shm "pc_xyz" + "pc_rgb"
                                            └── viser viewer (main thread)
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from camera import RealSenseRGB
from depth  import create_shm, depth_worker, BACKENDS, DEFAULT_MODEL
from viewer import Viewer

VIZ_HZ = 30
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
INFER_W, INFER_H = 640, 480


def main():
    cam = RealSenseRGB(width=CAM_W, height=CAM_H, fps=CAM_FPS)
    intr = cam.start()

    # Scale intrinsics from capture to inference resolution
    sx = INFER_W / intr.width
    sy = INFER_H / intr.height
    fx_i, fy_i = intr.fx * sx, intr.fy * sy
    cx_i, cy_i = intr.cx * sx, intr.cy * sy

    shm = create_shm(intr.width, intr.height, INFER_W, INFER_H)

    state = {"proc": None, "stop_ev": None, "status_q": None, "model": DEFAULT_MODEL}

    def spawn_depth(model_key: str) -> None:
        stop_ev  = mp.Event()
        status_q = mp.Queue(maxsize=64)
        proc = mp.Process(
            target=depth_worker,
            args=(
                shm.rgb.name, shm.depth.name, shm.pc_xyz.name, shm.pc_rgb.name,
                shm.rgb_seq, shm.depth_seq, shm.pc_count,
                shm.rgb_w, shm.rgb_h, shm.infer_w, shm.infer_h, shm.n_max,
                fx_i, fy_i, cx_i, cy_i,
                stop_ev,
                status_q,
                model_key,
            ),
            daemon=True,
        )
        proc.start()
        state["proc"], state["stop_ev"], state["status_q"], state["model"] = (
            proc, stop_ev, status_q, model_key,
        )

    def stop_depth() -> None:
        if state["stop_ev"] is not None:
            state["stop_ev"].set()
        if state["proc"] is not None:
            state["proc"].join(timeout=10.0)
            if state["proc"].is_alive():
                state["proc"].terminate()
        # Drain the status queue so it can be GC'd
        if state["status_q"] is not None:
            while True:
                try: state["status_q"].get_nowait()
                except Exception: break

    spawn_depth(DEFAULT_MODEL)

    viewer = Viewer(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy,
                    model_keys=list(BACKENDS.keys()), default_model=DEFAULT_MODEL)

    def on_model_change(key: str) -> None:
        if key == state["model"]:
            return
        viewer.set_model_status(f"switching to {key} ...", "", "")
        stop_depth()
        with shm.pc_count.get_lock():
            shm.pc_count.value = 0
        spawn_depth(key)

    viewer.set_model_change_callback(on_model_change)

    print("Open http://localhost:8080 — Ctrl+C to quit.")

    rgb_buf   = shm.rgb_arr()
    depth_buf = shm.depth_arr()
    pc_xyz    = shm.pc_xyz_arr()
    pc_rgb    = shm.pc_rgb_arr()

    period = 1.0 / VIZ_HZ
    last_depth_seq = 0
    n_rgb = n_depth = n_pc = 0
    t_log = time.time()

    def _format_status(msg: tuple) -> tuple[str, str, str]:
        """Return (status, progress, filename) tuple for the GUI."""
        if not msg: return ("", "", "")
        kind = msg[0]
        if kind == "downloading":
            fname, progress = msg[1], msg[2]
            return ("downloading", str(progress), fname)
        if kind == "loading": return ("loading model ...", "", "")
        if kind == "warming": return ("warming up ...", "", "")
        if kind == "ready":   return (f"running {state['model']}", "", "")
        if kind == "error":   return ("error", "", msg[1] if len(msg)>1 else "")
        return (" ".join(str(m) for m in msg), "", "")

    try:
        while True:
            t0 = time.time()

            rgb = cam.get()
            if rgb is not None:
                rgb_buf[...] = rgb
                with shm.rgb_seq.get_lock():
                    shm.rgb_seq.value += 1
                viewer.update_rgb(rgb)
                n_rgb += 1

            with shm.depth_seq.get_lock():
                cur = shm.depth_seq.value
            if cur != last_depth_seq:
                last_depth_seq = cur
                viewer.update_depth(depth_buf.copy())
                with shm.pc_count.get_lock():
                    n = shm.pc_count.value
                if n > 0:
                    viewer.update_point_cloud(pc_xyz[:n].copy(), pc_rgb[:n].copy())
                    n_pc += 1
                n_depth += 1

            # Drain depth-process status messages
            sq = state["status_q"]
            if sq is not None:
                latest = None
                while True:
                    try: latest = sq.get_nowait()
                    except Exception: break
                if latest is not None:
                    s, p, f = _format_status(latest)
                    viewer.set_model_status(s, p, f)

            if time.time() - t_log >= 1.0:
                dt = time.time() - t_log
                viewer.update_fps(n_rgb/dt, n_depth/dt, n_pc/dt)
                print(f"[viz] rgb {n_rgb/dt:.1f}  depth {n_depth/dt:.1f}  pc {n_pc/dt:.1f}")
                n_rgb = n_depth = n_pc = 0
                t_log = time.time()

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_depth()
        cam.stop()
        viewer.stop()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
