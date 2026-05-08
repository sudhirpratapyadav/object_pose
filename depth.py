"""
Depth-Anything-V2 metric depth + point cloud, in a separate process.

Communication layout
--------------------
  RGB in  :  shm "rgb"     (H, W, 3) uint8         — written by main, read here
              + mp.Value 'rgb_seq' bumped on each new frame
  Depth   :  shm "depth"   (H, W)    float32       — written here
  PC xyz  :  shm "pc_xyz"  (N_MAX, 3) float32      — written here
  PC rgb  :  shm "pc_rgb"  (N_MAX, 3) uint8        — written here
              + mp.Value 'pc_count' (number of valid points, ≤ N_MAX)
              + mp.Value 'depth_seq' bumped each iteration
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

DEPTH_REPO  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
WEIGHTS_DIR = Path(__file__).parent / "weights" / "v2-indoor-small"

PC_DOWNSAMPLE = 4   # 4× stride per axis  → 1/16 of pixels
PC_MIN_M, PC_MAX_M = 0.05, 10.0


@dataclass
class DepthShm:
    """Holds shared-memory handles + metadata for the depth/pc pipeline."""
    rgb:    shared_memory.SharedMemory
    depth:  shared_memory.SharedMemory
    pc_xyz: shared_memory.SharedMemory
    pc_rgb: shared_memory.SharedMemory
    rgb_seq:   mp.Value
    depth_seq: mp.Value
    pc_count:  mp.Value
    width:  int
    height: int
    n_max:  int

    def rgb_arr(self) -> np.ndarray:
        return np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.rgb.buf)

    def depth_arr(self) -> np.ndarray:
        return np.ndarray((self.height, self.width), dtype=np.float32, buffer=self.depth.buf)

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


def create_shm(width: int, height: int) -> DepthShm:
    n_max = (width // PC_DOWNSAMPLE) * (height // PC_DOWNSAMPLE)
    rgb    = shared_memory.SharedMemory(create=True, size=height * width * 3)
    depth  = shared_memory.SharedMemory(create=True, size=height * width * 4)
    pc_xyz = shared_memory.SharedMemory(create=True, size=n_max * 3 * 4)
    pc_rgb = shared_memory.SharedMemory(create=True, size=n_max * 3)
    return DepthShm(
        rgb=rgb, depth=depth, pc_xyz=pc_xyz, pc_rgb=pc_rgb,
        rgb_seq=mp.Value("Q", 0),    # uint64
        depth_seq=mp.Value("Q", 0),
        pc_count=mp.Value("I", 0),   # uint32
        width=width, height=height, n_max=n_max,
    )


def _open_existing(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=name)


def depth_worker(
    rgb_name: str, depth_name: str, pc_xyz_name: str, pc_rgb_name: str,
    rgb_seq, depth_seq, pc_count,
    width: int, height: int, n_max: int,
    fx: float, fy: float, cx: float, cy: float,
    stop_ev,
    device: str = "cuda",
):
    rgb_shm    = _open_existing(rgb_name)
    depth_shm  = _open_existing(depth_name)
    pc_xyz_shm = _open_existing(pc_xyz_name)
    pc_rgb_shm = _open_existing(pc_rgb_name)

    rgb_arr    = np.ndarray((height, width, 3), dtype=np.uint8, buffer=rgb_shm.buf)
    depth_arr  = np.ndarray((height, width),    dtype=np.float32, buffer=depth_shm.buf)
    pc_xyz_arr = np.ndarray((n_max, 3),         dtype=np.float32, buffer=pc_xyz_shm.buf)
    pc_rgb_arr = np.ndarray((n_max, 3),         dtype=np.uint8,   buffer=pc_rgb_shm.buf)

    from PIL import Image
    from transformers import pipeline as hf_pipeline

    if not (WEIGHTS_DIR / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        print(f"[depth] downloading {DEPTH_REPO} -> {WEIGHTS_DIR}")
        snapshot_download(repo_id=DEPTH_REPO, local_dir=str(WEIGHTS_DIR))

    print(f"[depth] loading model on {device} ...")
    t0 = time.time()
    pipe = hf_pipeline("depth-estimation", model=str(WEIGHTS_DIR), device=device)
    pipe(Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)))  # warmup
    print(f"[depth] ready in {time.time()-t0:.1f}s")

    # Pre-compute pixel grid for back-projection (downsampled)
    ds = PC_DOWNSAMPLE
    u = np.arange(0, width,  ds, dtype=np.float32)
    v = np.arange(0, height, ds, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)        # (h//ds, w//ds)
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

        # Snapshot the RGB buffer (cheap copy keeps inference inputs stable)
        rgb_local = rgb_arr.copy()

        out = pipe(Image.fromarray(rgb_local))
        d = out["predicted_depth"]
        if hasattr(d, "cpu"):
            d = d.cpu()
        d = np.clip(d.numpy().astype(np.float32), 0.0, PC_MAX_M)

        # Back-project (downsampled)
        d_ds = d[::ds, ::ds]
        rgb_ds = rgb_local[::ds, ::ds]
        valid = (d_ds > PC_MIN_M) & (d_ds < PC_MAX_M)
        zs = d_ds[valid]
        xs = x_norm[valid] * zs
        ys = y_norm[valid] * zs
        pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        cols = rgb_ds[valid].astype(np.uint8)
        n = min(len(pts), n_max)

        # Publish
        depth_arr[...] = d
        pc_xyz_arr[:n] = pts[:n]
        pc_rgb_arr[:n] = cols[:n]
        with pc_count.get_lock():
            pc_count.value = n
        with depth_seq.get_lock():
            depth_seq.value = depth_seq.value + 1

    for shm in (rgb_shm, depth_shm, pc_xyz_shm, pc_rgb_shm):
        shm.close()
