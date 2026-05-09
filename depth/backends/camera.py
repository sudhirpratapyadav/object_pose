"""Camera-depth backend: relays the camera's own depth stream as the depth
output, bypassing any neural-network inference.

Reads from a side-channel shared-memory slot (rgb_h x rgb_w float32 metres)
that the camera or sim worker writes into. The factory needs the shm name
and dimensions, which are only known at runtime — see make_camera_backend()
for the constructor used by depth_worker.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np
from PIL import Image

from .base import BackendInfo, StatusFn


class CameraDepthBackend:
    def __init__(self, info: BackendInfo, *,
                 depth_shm_name: str,
                 src_w: int, src_h: int,
                 infer_w: int, infer_h: int) -> None:
        self.info = info
        self._shm_name = depth_shm_name
        self._src_w = src_w
        self._src_h = src_h
        self._infer_w = infer_w
        self._infer_h = infer_h
        self._shm: shared_memory.SharedMemory | None = None
        self._depth_arr: np.ndarray | None = None

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        try:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            self._depth_arr = np.ndarray(
                (self._src_h, self._src_w), dtype=np.float32, buffer=self._shm.buf,
            )
        except Exception as exc:
            status("error", f"camera depth shm: {exc}")
            raise
        status("ready")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        # rgb is at infer_w/infer_h; we ignore it and read from shm. The
        # depth stream is at native sensor resolution and may need resizing.
        d = self._depth_arr
        if d is None:
            return np.zeros((self._infer_h, self._infer_w), dtype=np.float32)
        # Snapshot to a local copy so we don't tear when the writer updates.
        d_snap = d.copy()
        if d_snap.shape != (self._infer_h, self._infer_w):
            d_snap = np.asarray(
                Image.fromarray(d_snap).resize(
                    (self._infer_w, self._infer_h), Image.NEAREST,
                ),
                dtype=np.float32,
            )
        return d_snap


def make_camera_backend(label: str, *, depth_shm_name: str,
                        src_w: int, src_h: int,
                        infer_w: int, infer_h: int) -> CameraDepthBackend:
    """Build a CameraDepthBackend wired to a specific depth-stream shm."""
    info = BackendInfo(
        key="camera-depth",
        label=label,
        family="camera",
        repo="",
        infer_w=infer_w,
        infer_h=infer_h,
        has_normals=False,
    )
    return CameraDepthBackend(
        info,
        depth_shm_name=depth_shm_name,
        src_w=src_w, src_h=src_h,
        infer_w=infer_w, infer_h=infer_h,
    )
