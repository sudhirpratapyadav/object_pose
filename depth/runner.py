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
from .mesh import build_faces, fill_mesh, grid_dims, precompute_unproject

PC_DOWNSAMPLE = 4
PC_MIN_M, PC_MAX_M = 0.05, 10.0
MESH_DOWNSAMPLE = 4
MESH_EDGE_THRESHOLD_M = 0.05


@dataclass
class DepthShm:
    rgb:    shared_memory.SharedMemory
    depth:  shared_memory.SharedMemory
    pc_xyz: shared_memory.SharedMemory
    pc_rgb: shared_memory.SharedMemory
    pc_grid_idx: shared_memory.SharedMemory
    mesh_xyz:   shared_memory.SharedMemory
    mesh_rgb:   shared_memory.SharedMemory
    mesh_faces: shared_memory.SharedMemory
    mesh_normal: shared_memory.SharedMemory  # (n_verts, 3) float32, may be all zeros
    normal_img: shared_memory.SharedMemory   # (infer_h, infer_w, 3) float32 normal map
    rgb_seq:   mp.Value
    depth_seq: mp.Value
    pc_count:  mp.Value
    has_normal: mp.Value
    rgb_w:    int
    rgb_h:    int
    infer_w:  int
    infer_h:  int
    n_max:    int
    mesh_grid_w: int
    mesh_grid_h: int
    mesh_n_faces: int

    def rgb_arr(self) -> np.ndarray:
        return np.ndarray((self.rgb_h, self.rgb_w, 3), dtype=np.uint8, buffer=self.rgb.buf)

    def depth_arr(self) -> np.ndarray:
        return np.ndarray((self.infer_h, self.infer_w), dtype=np.float32, buffer=self.depth.buf)

    def pc_xyz_arr(self) -> np.ndarray:
        return np.ndarray((self.n_max, 3), dtype=np.float32, buffer=self.pc_xyz.buf)

    def pc_rgb_arr(self) -> np.ndarray:
        return np.ndarray((self.n_max, 3), dtype=np.uint8, buffer=self.pc_rgb.buf)

    def pc_grid_idx_arr(self) -> np.ndarray:
        return np.ndarray((self.n_max,), dtype=np.uint32, buffer=self.pc_grid_idx.buf)

    def mesh_xyz_arr(self) -> np.ndarray:
        n = self.mesh_grid_w * self.mesh_grid_h
        return np.ndarray((n, 3), dtype=np.float32, buffer=self.mesh_xyz.buf)

    def mesh_rgb_arr(self) -> np.ndarray:
        n = self.mesh_grid_w * self.mesh_grid_h
        return np.ndarray((n, 3), dtype=np.float32, buffer=self.mesh_rgb.buf)

    def mesh_faces_arr(self) -> np.ndarray:
        return np.ndarray((self.mesh_n_faces, 3), dtype=np.int32, buffer=self.mesh_faces.buf)

    def mesh_normal_arr(self) -> np.ndarray:
        n = self.mesh_grid_w * self.mesh_grid_h
        return np.ndarray((n, 3), dtype=np.float32, buffer=self.mesh_normal.buf)

    def normal_img_arr(self) -> np.ndarray:
        return np.ndarray((self.infer_h, self.infer_w, 3), dtype=np.float32,
                          buffer=self.normal_img.buf)

    def close(self):
        for shm in (self.rgb, self.depth, self.pc_xyz, self.pc_rgb,
                    self.pc_grid_idx,
                    self.mesh_xyz, self.mesh_rgb, self.mesh_faces,
                    self.mesh_normal, self.normal_img):
            shm.close()

    def unlink(self):
        for shm in (self.rgb, self.depth, self.pc_xyz, self.pc_rgb,
                    self.pc_grid_idx,
                    self.mesh_xyz, self.mesh_rgb, self.mesh_faces,
                    self.mesh_normal, self.normal_img):
            try: shm.unlink()
            except FileNotFoundError: pass


def create_shm(rgb_w: int, rgb_h: int, infer_w: int, infer_h: int) -> DepthShm:
    n_max = (infer_w // PC_DOWNSAMPLE) * (infer_h // PC_DOWNSAMPLE)
    grid_w, grid_h = grid_dims(infer_w, infer_h, MESH_DOWNSAMPLE)
    n_verts = grid_w * grid_h
    n_faces = 2 * (grid_h - 1) * (grid_w - 1)

    rgb    = shared_memory.SharedMemory(create=True, size=rgb_h * rgb_w * 3)
    depth  = shared_memory.SharedMemory(create=True, size=infer_h * infer_w * 4)
    pc_xyz = shared_memory.SharedMemory(create=True, size=n_max * 3 * 4)
    pc_rgb = shared_memory.SharedMemory(create=True, size=n_max * 3)
    pc_grid_idx = shared_memory.SharedMemory(create=True, size=n_max * 4)
    mesh_xyz   = shared_memory.SharedMemory(create=True, size=n_verts * 3 * 4)
    mesh_rgb   = shared_memory.SharedMemory(create=True, size=n_verts * 3 * 4)
    mesh_faces = shared_memory.SharedMemory(create=True, size=n_faces * 3 * 4)
    mesh_normal = shared_memory.SharedMemory(create=True, size=n_verts * 3 * 4)
    normal_img  = shared_memory.SharedMemory(create=True, size=infer_h * infer_w * 3 * 4)
    return DepthShm(
        rgb=rgb, depth=depth, pc_xyz=pc_xyz, pc_rgb=pc_rgb,
        pc_grid_idx=pc_grid_idx,
        mesh_xyz=mesh_xyz, mesh_rgb=mesh_rgb, mesh_faces=mesh_faces,
        mesh_normal=mesh_normal, normal_img=normal_img,
        rgb_seq=mp.Value("Q", 0),
        depth_seq=mp.Value("Q", 0),
        pc_count=mp.Value("I", 0),
        has_normal=mp.Value("B", 0),
        rgb_w=rgb_w, rgb_h=rgb_h,
        infer_w=infer_w, infer_h=infer_h, n_max=n_max,
        mesh_grid_w=grid_w, mesh_grid_h=grid_h, mesh_n_faces=n_faces,
    )


def _open_existing(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=name)


def depth_worker(
    rgb_name: str, depth_name: str, pc_xyz_name: str, pc_rgb_name: str,
    pc_grid_idx_name: str,
    mesh_xyz_name: str, mesh_rgb_name: str, mesh_faces_name: str,
    mesh_normal_name: str, normal_img_name: str,
    rgb_seq, depth_seq, pc_count, has_normal_flag,
    rgb_w: int, rgb_h: int, infer_w: int, infer_h: int, n_max: int,
    mesh_grid_w: int, mesh_grid_h: int, mesh_n_faces: int,
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
    pc_grid_idx_shm = _open_existing(pc_grid_idx_name)
    mesh_xyz_shm   = _open_existing(mesh_xyz_name)
    mesh_rgb_shm   = _open_existing(mesh_rgb_name)
    mesh_faces_shm = _open_existing(mesh_faces_name)
    mesh_normal_shm = _open_existing(mesh_normal_name)
    normal_img_shm  = _open_existing(normal_img_name)

    rgb_arr    = np.ndarray((rgb_h, rgb_w, 3),  dtype=np.uint8,   buffer=rgb_shm.buf)
    depth_arr  = np.ndarray((infer_h, infer_w), dtype=np.float32, buffer=depth_shm.buf)
    pc_xyz_arr = np.ndarray((n_max, 3),         dtype=np.float32, buffer=pc_xyz_shm.buf)
    pc_rgb_arr = np.ndarray((n_max, 3),         dtype=np.uint8,   buffer=pc_rgb_shm.buf)
    pc_grid_idx_arr = np.ndarray((n_max,),      dtype=np.uint32,  buffer=pc_grid_idx_shm.buf)
    n_verts    = mesh_grid_w * mesh_grid_h
    mesh_xyz_arr   = np.ndarray((n_verts, 3),       dtype=np.float32, buffer=mesh_xyz_shm.buf)
    mesh_rgb_arr   = np.ndarray((n_verts, 3),       dtype=np.float32, buffer=mesh_rgb_shm.buf)
    mesh_faces_arr = np.ndarray((mesh_n_faces, 3),  dtype=np.int32,   buffer=mesh_faces_shm.buf)
    mesh_normal_arr = np.ndarray((n_verts, 3),      dtype=np.float32, buffer=mesh_normal_shm.buf)
    normal_img_arr  = np.ndarray((infer_h, infer_w, 3), dtype=np.float32,
                                 buffer=normal_img_shm.buf)

    all_shms = (rgb_shm, depth_shm, pc_xyz_shm, pc_rgb_shm, pc_grid_idx_shm,
                mesh_xyz_shm, mesh_rgb_shm, mesh_faces_shm,
                mesh_normal_shm, normal_img_shm)

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
        for shm in all_shms:
            shm.close()
        return

    # Pre-compute pixel grid (inference frame, downsampled)
    ds = PC_DOWNSAMPLE
    u = np.arange(0, infer_w, ds, dtype=np.float32)
    v = np.arange(0, infer_h, ds, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy

    # Static mesh topology + back-projection lookups.
    mesh_faces_static = build_faces(mesh_grid_w, mesh_grid_h)
    mesh_x_norm, mesh_y_norm = precompute_unproject(
        mesh_grid_w, mesh_grid_h, MESH_DOWNSAMPLE, fx, fy, cx, cy,
    )

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

        out = backend.infer(rgb_inf)
        if isinstance(out, tuple):
            d, n_img = out
        else:
            d, n_img = out, None
        d = np.clip(d.astype(np.float32), 0.0, PC_MAX_M)
        if d.shape != (infer_h, infer_w):
            d = np.asarray(Image.fromarray(d).resize((infer_w, infer_h), Image.BILINEAR))

        # Back-project
        d_ds   = d[::ds, ::ds]
        rgb_ds = rgb_inf[::ds, ::ds]
        h_ds, w_ds = d_ds.shape
        valid  = (d_ds > PC_MIN_M) & (d_ds < PC_MAX_M)
        zs = d_ds[valid]
        xs = x_norm[valid] * zs
        ys = y_norm[valid] * zs
        pts  = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        cols = rgb_ds[valid].astype(np.uint8)
        # Flat grid index per valid point. Mesh grid is the same as pc grid
        # (PC_DOWNSAMPLE == MESH_DOWNSAMPLE), so this also indexes mesh verts.
        flat_idx = np.flatnonzero(valid.ravel()).astype(np.uint32)
        n = min(len(pts), n_max)

        depth_arr[...] = d
        pc_xyz_arr[:n] = pts[:n]
        pc_rgb_arr[:n] = cols[:n]
        pc_grid_idx_arr[:n] = flat_idx[:n]

        n_grid_for_fill = None
        if n_img is not None and n_img.shape[:2] == (infer_h, infer_w):
            n_grid_for_fill = n_img[::MESH_DOWNSAMPLE, ::MESH_DOWNSAMPLE]
        fill_mesh(
            d, rgb_inf,
            mesh_x_norm, mesh_y_norm,
            mesh_faces_static,
            MESH_DOWNSAMPLE,
            PC_MIN_M, PC_MAX_M,
            MESH_EDGE_THRESHOLD_M,
            mesh_xyz_arr, mesh_rgb_arr, mesh_faces_arr,
            normal_grid=n_grid_for_fill,
        )

        # Normals: write the inference-frame map to normal_img and the
        # downsampled grid to mesh_normal. has_normal toggles per frame.
        if n_img is not None:
            if n_img.shape[:2] != (infer_h, infer_w):
                from PIL import Image as _I
                ch = []
                for c in range(3):
                    ch.append(np.asarray(_I.fromarray(n_img[..., c]).resize(
                        (infer_w, infer_h), _I.BILINEAR)))
                n_img = np.stack(ch, axis=-1)
            normal_img_arr[...] = n_img.astype(np.float32)
            n_grid = n_img[::MESH_DOWNSAMPLE, ::MESH_DOWNSAMPLE]
            mesh_normal_arr[...] = n_grid.reshape(-1, 3).astype(np.float32)
            with has_normal_flag.get_lock(): has_normal_flag.value = 1
        else:
            with has_normal_flag.get_lock(): has_normal_flag.value = 0

        with pc_count.get_lock():
            pc_count.value = n
        with depth_seq.get_lock():
            depth_seq.value = depth_seq.value + 1

    for shm in all_shms:
        shm.close()
