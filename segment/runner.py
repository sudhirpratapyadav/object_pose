"""Segmentation worker process and shared-memory layout.

Inputs:
  - rgb shm (already created by depth.runner.create_shm) + rgb_seq counter
  - mesh_xyz shm (dense grid, written by depth worker) — used to compute the
    3D bbox of the segmented region.
  - click_x, click_y, click_seq shared values: when click_seq increments, the
    worker reruns SAM2 on the latest RGB frame at that pixel.

Outputs (via SegShm):
  - mask_shm: uint8 array of length grid_w*grid_h (mesh-grid order). 1 where
    the pixel belongs to the current segmented object.
  - bbox_shm: float32[6] = [xmin, ymin, zmin, xmax, ymax, zmax].
  - mask_seq counter, bumped on each new mask.
  - has_mask flag (1 if a non-empty mask is present).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
from PIL import Image

from .backends import DEFAULT_MODEL, make_backend


@dataclass
class SegShm:
    mask:  shared_memory.SharedMemory
    bbox:  shared_memory.SharedMemory
    mask_seq:  mp.Value     # bumped on each new mask
    click_seq: mp.Value     # bumped by main when user clicks
    click_x:   mp.Value     # int — pixel x in INFERENCE frame
    click_y:   mp.Value     # int — pixel y in INFERENCE frame
    has_mask:  mp.Value     # 0/1
    grid_w:    int
    grid_h:    int

    @property
    def n(self) -> int:
        return self.grid_w * self.grid_h

    def mask_arr(self) -> np.ndarray:
        return np.ndarray((self.n,), dtype=np.uint8, buffer=self.mask.buf)

    def bbox_arr(self) -> np.ndarray:
        return np.ndarray((6,), dtype=np.float32, buffer=self.bbox.buf)

    def close(self):
        for shm in (self.mask, self.bbox):
            shm.close()

    def unlink(self):
        for shm in (self.mask, self.bbox):
            try: shm.unlink()
            except FileNotFoundError: pass


def create_seg_shm(grid_w: int, grid_h: int) -> SegShm:
    n = grid_w * grid_h
    mask = shared_memory.SharedMemory(create=True, size=n)
    bbox = shared_memory.SharedMemory(create=True, size=6 * 4)
    return SegShm(
        mask=mask, bbox=bbox,
        mask_seq=mp.Value("Q", 0),
        click_seq=mp.Value("Q", 0),
        click_x=mp.Value("i", 0),
        click_y=mp.Value("i", 0),
        has_mask=mp.Value("B", 0),
        grid_w=grid_w, grid_h=grid_h,
    )


def _open_existing(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=name)


def segment_worker(
    rgb_name: str,
    mesh_xyz_name: str,
    mask_name: str, bbox_name: str,
    click_seq, click_x, click_y,
    mask_seq, has_mask,
    rgb_w: int, rgb_h: int,
    infer_w: int, infer_h: int,
    grid_w: int, grid_h: int,
    grid_downsample: int,
    stop_ev,
    status_q=None,
    model_key: str = DEFAULT_MODEL,
    device: str = "cuda",
) -> None:
    """Wait for click_seq to bump, then run SAM2 on the latest RGB and write
    a grid-shaped mask + 3D bbox of the segmented region."""

    def status(*msg):
        if status_q is not None:
            try: status_q.put_nowait(msg)
            except Exception: pass

    rgb_shm    = _open_existing(rgb_name)
    mesh_shm   = _open_existing(mesh_xyz_name)
    mask_shm   = _open_existing(mask_name)
    bbox_shm   = _open_existing(bbox_name)

    n_grid = grid_w * grid_h
    rgb_arr  = np.ndarray((rgb_h, rgb_w, 3), dtype=np.uint8,   buffer=rgb_shm.buf)
    mesh_xyz = np.ndarray((n_grid, 3),       dtype=np.float32, buffer=mesh_shm.buf)
    mask_arr = np.ndarray((n_grid,),         dtype=np.uint8,   buffer=mask_shm.buf)
    bbox_arr = np.ndarray((6,),              dtype=np.float32, buffer=bbox_shm.buf)

    try:
        backend = make_backend(model_key)
        backend.load(status, device=device)
    except Exception as exc:
        msg = str(exc)
        print(f"[seg] backend '{model_key}' failed to load: {msg}", flush=True)
        status("error", msg[:160])
        while not stop_ev.is_set():
            time.sleep(0.1)
        for shm in (rgb_shm, mesh_shm, mask_shm, bbox_shm):
            shm.close()
        return

    last_click_seq = 0
    while not stop_ev.is_set():
        with click_seq.get_lock():
            cur = click_seq.value
        if cur == last_click_seq:
            time.sleep(0.01)
            continue
        last_click_seq = cur

        with click_x.get_lock(): cx_inf = click_x.value
        with click_y.get_lock(): cy_inf = click_y.value

        # Special "clear" sentinel: negative click coords -> wipe mask
        if cx_inf < 0 or cy_inf < 0:
            mask_arr[:] = 0
            with has_mask.get_lock(): has_mask.value = 0
            with mask_seq.get_lock(): mask_seq.value = mask_seq.value + 1
            status("cleared")
            continue

        rgb_full = rgb_arr.copy()
        # Convert click (inference frame) -> capture frame for SAM2.
        sx = rgb_w / infer_w
        sy = rgb_h / infer_h
        sam_x = int(round(cx_inf * sx))
        sam_y = int(round(cy_inf * sy))

        try:
            backend.set_image(rgb_full)
            mask_full = backend.predict_point(sam_x, sam_y, label=1)
        except Exception as exc:
            print(f"[seg] predict failed: {exc}", flush=True)
            with has_mask.get_lock(): has_mask.value = 0
            with mask_seq.get_lock(): mask_seq.value = mask_seq.value + 1
            continue

        # Mask comes back at capture resolution. Downsample to the
        # mesh/point-cloud grid via NEAREST (boolean-friendly).
        if mask_full.shape != (rgb_h, rgb_w):
            mask_full = np.asarray(
                Image.fromarray(mask_full.astype(np.uint8) * 255)
                     .resize((rgb_w, rgb_h), Image.NEAREST)
            ) > 127
        mask_inf = np.asarray(
            Image.fromarray(mask_full.astype(np.uint8) * 255)
                 .resize((infer_w, infer_h), Image.NEAREST)
        ) > 127
        mask_grid = mask_inf[::grid_downsample, ::grid_downsample]
        m = mask_grid.astype(np.uint8).ravel()
        if m.size != n_grid:
            mm = np.zeros(n_grid, dtype=np.uint8)
            k = min(m.size, n_grid)
            mm[:k] = m[:k]
            m = mm
        mask_arr[:] = m

        # 3D bbox from mesh-grid xyz at masked cells (skip zero-z, NaN).
        sel = m.astype(bool)
        xyz = mesh_xyz[sel]
        finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-3)
        xyz = xyz[finite]
        if xyz.shape[0] > 0:
            bbox_arr[0:3] = xyz.min(axis=0)
            bbox_arr[3:6] = xyz.max(axis=0)
            with has_mask.get_lock(): has_mask.value = 1
        else:
            with has_mask.get_lock(): has_mask.value = 0

        with mask_seq.get_lock():
            mask_seq.value = mask_seq.value + 1
        status("ok")

    for shm in (rgb_shm, mesh_shm, mask_shm, bbox_shm):
        shm.close()
