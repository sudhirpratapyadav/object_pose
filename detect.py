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
from depth  import create_shm, depth_worker
from viewer import Viewer

VIZ_HZ = 30


def main():
    cam = RealSenseRGB()
    intr = cam.start()

    shm = create_shm(intr.width, intr.height)
    stop_ev = mp.Event()

    proc = mp.Process(
        target=depth_worker,
        args=(
            shm.rgb.name, shm.depth.name, shm.pc_xyz.name, shm.pc_rgb.name,
            shm.rgb_seq, shm.depth_seq, shm.pc_count,
            shm.width, shm.height, shm.n_max,
            intr.fx, intr.fy, intr.cx, intr.cy,
            stop_ev,
        ),
        daemon=True,
    )
    proc.start()

    viewer = Viewer(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    print("Open http://localhost:8080 — Ctrl+C to quit.")

    rgb_buf   = shm.rgb_arr()
    depth_buf = shm.depth_arr()
    pc_xyz    = shm.pc_xyz_arr()
    pc_rgb    = shm.pc_rgb_arr()

    period = 1.0 / VIZ_HZ
    last_depth_seq = 0
    n_rgb = n_depth = n_pc = 0
    t_log = time.time()

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
        stop_ev.set()
        cam.stop()
        viewer.stop()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
