"""RealSense color-stream wrapper. Threaded; latest-frame slot."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class RealSenseRGB:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width, self.height, self.fps = width, height, fps
        self.intrinsics: Intrinsics | None = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._pipe: rs.pipeline | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Intrinsics:
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.rgb8, self.fps)
        profile = self._pipe.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(width=intr.width, height=intr.height,
                                     fx=intr.fx, fy=intr.fy,
                                     cx=intr.ppx, cy=intr.ppy)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[cam] {intr.width}x{intr.height}  fx={intr.fx:.1f}  fy={intr.fy:.1f}")
        return self.intrinsics

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipe:
            self._pipe.stop()

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _loop(self):
        while self._running.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            with self._lock:
                self._latest = rgb
