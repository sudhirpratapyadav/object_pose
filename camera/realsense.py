"""RealSense color-stream wrapper. Threaded; latest-frame slot.

When ``enable_depth=True`` is requested, the wrapper also enables a depth
stream aligned to the color frame and exposes per-pixel depth in metres via
``get_depth()``. Depth is best-effort: if the device or USB connection can't
deliver a depth stream, we log a warning and continue with color only
(``has_depth`` becomes False).
"""

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
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_depth: bool = False):
        self.width, self.height, self.fps = width, height, fps
        self._enable_depth = enable_depth
        self.has_depth = False           # set True only after start() succeeds
        self.intrinsics: Intrinsics | None = None
        self._latest: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._depth_scale = 0.001        # metres-per-unit; refreshed in start()
        self._align: rs.align | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._pipe: rs.pipeline | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Intrinsics:
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.rgb8, self.fps)
        if self._enable_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height,
                              rs.format.z16, self.fps)
        try:
            profile = self._pipe.start(cfg)
        except RuntimeError as exc:
            if self._enable_depth:
                # Fall back to color-only if depth stream wasn't accepted.
                print(f"[cam] depth stream unavailable ({exc}); "
                      f"falling back to color only", flush=True)
                self._enable_depth = False
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height,
                                  rs.format.rgb8, self.fps)
                profile = self._pipe.start(cfg)
            else:
                raise

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(width=intr.width, height=intr.height,
                                     fx=intr.fx, fy=intr.fy,
                                     cx=intr.ppx, cy=intr.ppy)

        if self._enable_depth:
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                self._align = rs.align(rs.stream.color)
                self.has_depth = True
                print(f"[cam] depth stream enabled  "
                      f"depth_scale={self._depth_scale:g} m/unit", flush=True)
            except Exception as exc:
                print(f"[cam] depth setup failed ({exc}); color only",
                      flush=True)
                self.has_depth = False
                self._align = None

        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[cam] {intr.width}x{intr.height}  fx={intr.fx:.1f}  "
              f"fy={intr.fy:.1f}  depth={'on' if self.has_depth else 'off'}",
              flush=True)
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

    def get_depth(self) -> np.ndarray | None:
        """Latest depth frame in metres, aligned to the color image. None
        if depth wasn't enabled or no frame has arrived yet."""
        with self._lock:
            return self._latest_depth

    def _loop(self):
        while self._running.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            depth_m: np.ndarray | None = None
            if self.has_depth:
                depth = frames.get_depth_frame()
                if depth:
                    depth_u16 = np.asanyarray(depth.get_data())
                    depth_m = depth_u16.astype(np.float32) * self._depth_scale
            with self._lock:
                self._latest = rgb
                if depth_m is not None:
                    self._latest_depth = depth_m
