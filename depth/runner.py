"""
Depth worker process + shared-memory layout.

  RGB in  :  shm "rgb"     (rgb_h, rgb_w, 3)         uint8     written by main, read here
              + mp.Value 'rgb_seq' bumped on each new frame
  Depth   :  shm "depth"   (infer_h, infer_w)        float32   written here
  PC xyz  :  shm "pc_xyz"  (n_max, 3)                float32   written here
  PC rgb  :  shm "pc_rgb"  (n_max, 3)                uint8     written here
              + mp.Value 'pc_count' = number of valid points (<= n_max)
              + mp.Value 'depth_seq' bumped each iteration
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
from PIL import Image

from .backends import DEFAULT_MODEL, make_backend

PC_DOWNSAMPLE = 4
PC_MIN_M, PC_MAX_M = 0.05, 10.0


@dataclass
class DepthShm:
    rgb:    shared_memory.SharedMemory
    depth:  shared_memory.SharedMemory
    pc_xyz: shared_memory.SharedMemory
    pc_rgb: shared_memory.SharedMemory
    rgb_seq:   mp.Value
    depth_seq: mp.Value
    pc_count:  mp.Value
    rgb_w:    int
    rgb_h:    int
    infer_w:  int
    infer_h:  int
    n_max:    int

    def rgb_arr(self) -> np.ndarray:
        return np.ndarray((self.rgb_h, self.rgb_w, 3), dtype=np.uint8, buffer=self.rgb.buf)

    def depth_arr(self) -> np.ndarray:
        return np.ndarray((self.infer_h, self.infer_w), dtype=np.float32, buffer=self.depth.buf)

    def pc_xyz_arr(self) -> np.ndarray:
        return np.ndarray((self.n_max, 3), dtype=np.float32, buffer=self.pc_xyz.buf)

    def pc_rgb_arr(self) -> np.ndarray:
        return np.ndarray((self.n_max, 3), dtype=np.uint8, buffer=self.pc_rgb.buf)

    def close(self):
        for shm in (self.rgb, self.depth, self.pc_xyz, self.pc_rgb):
            shm.close()

    def unlink(self):
        for shm in (self.rgb, self.depth, self.pc_xyz, self.pc_rgb):
            try: shm.unlink()
            except FileNotFoundError: pass


def create_shm(rgb_w: int, rgb_h: int, infer_w: int, infer_h: int) -> DepthShm:
    n_max = (infer_w // PC_DOWNSAMPLE) * (infer_h // PC_DOWNSAMPLE)
    rgb    = shared_memory.SharedMemory(create=True, size=rgb_h * rgb_w * 3)
    depth  = shared_memory.SharedMemory(create=True, size=infer_h * infer_w * 4)
    pc_xyz = shared_memory.SharedMemory(create=True, size=n_max * 3 * 4)
    pc_rgb = shared_memory.SharedMemory(create=True, size=n_max * 3)
    return DepthShm(
        rgb=rgb, depth=depth, pc_xyz=pc_xyz, pc_rgb=pc_rgb,
        rgb_seq=mp.Value("Q", 0),
        depth_seq=mp.Value("Q", 0),
        pc_count=mp.Value("I", 0),
        rgb_w=rgb_w, rgb_h=rgb_h,
        infer_w=infer_w, infer_h=infer_h, n_max=n_max,
    )


def _open_existing(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=name)


def depth_worker(
    rgb_name: str, depth_name: str, pc_xyz_name: str, pc_rgb_name: str,
    rgb_seq, depth_seq, pc_count,
    rgb_w: int, rgb_h: int, infer_w: int, infer_h: int, n_max: int,
    fx: float, fy: float, cx: float, cy: float,
    stop_ev,
    status_q=None,
    model_key: str = DEFAULT_MODEL,
    device: str = "cuda",
):
    """Inference runs at (infer_w, infer_h). Intrinsics fx/fy/cx/cy are in the
    inference frame (already scaled by caller)."""

    def status(*msg):
        if status_q is not None:
            try: status_q.put_nowait(msg)
            except Exception: pass

    rgb_shm    = _open_existing(rgb_name)
    depth_shm  = _open_existing(depth_name)
    pc_xyz_shm = _open_existing(pc_xyz_name)
    pc_rgb_shm = _open_existing(pc_rgb_name)

    rgb_arr    = np.ndarray((rgb_h, rgb_w, 3),  dtype=np.uint8,   buffer=rgb_shm.buf)
    depth_arr  = np.ndarray((infer_h, infer_w), dtype=np.float32, buffer=depth_shm.buf)
    pc_xyz_arr = np.ndarray((n_max, 3),         dtype=np.float32, buffer=pc_xyz_shm.buf)
    pc_rgb_arr = np.ndarray((n_max, 3),         dtype=np.uint8,   buffer=pc_rgb_shm.buf)

    try:
        backend = make_backend(model_key, focal_px=float(fx))
        backend.load(status, device=device)
    except Exception as exc:
        msg = str(exc)
        print(f"[depth] backend '{model_key}' failed to load: {msg}")
        status("error", msg[:120])
        # Stay alive so the GUI can switch to a different backend
        while not stop_ev.is_set():
            time.sleep(0.1)
        for shm in (rgb_shm, depth_shm, pc_xyz_shm, pc_rgb_shm):
            shm.close()
        return

    # Pre-compute pixel grid (inference frame, downsampled)
    ds = PC_DOWNSAMPLE
    u = np.arange(0, infer_w, ds, dtype=np.float32)
    v = np.arange(0, infer_h, ds, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy

    last_seen = 0
    while not stop_ev.is_set():
        with rgb_seq.get_lock():
            cur = rgb_seq.value
        if cur == last_seen:
            time.sleep(0.005)
            continue
        last_seen = cur

        rgb_full = rgb_arr.copy()
        rgb_pil  = Image.fromarray(rgb_full).resize((infer_w, infer_h), Image.BILINEAR)
        rgb_inf  = np.asarray(rgb_pil)

        d = backend.infer(rgb_inf)
        d = np.clip(d.astype(np.float32), 0.0, PC_MAX_M)
        if d.shape != (infer_h, infer_w):
            d = np.asarray(Image.fromarray(d).resize((infer_w, infer_h), Image.BILINEAR))

        # Back-project
        d_ds   = d[::ds, ::ds]
        rgb_ds = rgb_inf[::ds, ::ds]
        valid  = (d_ds > PC_MIN_M) & (d_ds < PC_MAX_M)
        zs = d_ds[valid]
        xs = x_norm[valid] * zs
        ys = y_norm[valid] * zs
        pts  = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        cols = rgb_ds[valid].astype(np.uint8)
        n = min(len(pts), n_max)

        depth_arr[...] = d
        pc_xyz_arr[:n] = pts[:n]
        pc_rgb_arr[:n] = cols[:n]
        with pc_count.get_lock():
            pc_count.value = n
        with depth_seq.get_lock():
            depth_seq.value = depth_seq.value + 1

    for shm in (rgb_shm, depth_shm, pc_xyz_shm, pc_rgb_shm):
        shm.close()
