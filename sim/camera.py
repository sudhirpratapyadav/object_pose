"""Stub camera adapter for sim mode.

The actual RGB producer is the sim worker process — it writes directly into
``rgb_shm`` and bumps ``rgb_seq``. This adapter just satisfies the
camera-source interface (start/get/stop) so web_server.py's session builder
can treat it like RealSenseRGB or VideoFile.
"""

from __future__ import annotations

import numpy as np

from camera.realsense import Intrinsics
from config import CamCalib


class MujocoCamera:
    """Reports YAML intrinsics; ``get()`` always returns None.

    The sim worker is the real producer; the stream loop's "rgb_seq advanced"
    branch (the one that broadcasts the JPEG and bumps the depth pipeline)
    fires whenever the worker writes a new frame.
    """

    def __init__(self, calib: CamCalib) -> None:
        i = calib.intrinsics
        self._intr = Intrinsics(width=i.width, height=i.height,
                                fx=i.fx, fy=i.fy, cx=i.cx, cy=i.cy)

    def start(self) -> Intrinsics:
        print(f"[sim-cam] {self._intr.width}x{self._intr.height}  "
              f"fx={self._intr.fx:.1f}  fy={self._intr.fy:.1f}", flush=True)
        return self._intr

    def get(self) -> np.ndarray | None:
        # Sim worker writes RGB to shm and bumps rgb_seq directly; the stream
        # loop picks up frame events from rgb_seq, not from this method.
        return None

    def stop(self) -> None:
        pass
