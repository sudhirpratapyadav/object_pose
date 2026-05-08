"""Video-file 'camera' source. Plays an MP4 (or any cv2-readable file) on a
loop at its native FPS, exposing the same interface as RealSenseRGB and
NetworkRGB so the rest of the pipeline doesn't care.

Intrinsics: video files don't carry calibration. We fall back to a pinhole
model that's reasonable for most footage:

    fx = fy = 0.85 * width
    cx = width  / 2
    cy = height / 2
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .realsense import Intrinsics


class VideoFile:
    def __init__(self, path: str, fps: float | None = None):
        self.path = path
        self._fps_override = fps
        self._cap = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self.intrinsics: Intrinsics | None = None

    def start(self) -> Intrinsics:
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video file: {self.path}")
        self._cap = cap

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 1.0:
            native_fps = 30.0
        fps = self._fps_override or native_fps

        # Pinhole fallback intrinsics (0.85 * width is a sane default focal).
        fx = fy = 0.85 * w
        cx, cy = w / 2.0, h / 2.0
        self.intrinsics = Intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        self._fps = fps

        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[cam] video {self.path} {w}x{h} @ {fps:.1f} fps", flush=True)
        return self.intrinsics

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        import cv2
        period = 1.0 / max(self._fps, 1.0)
        while self._running.is_set():
            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok:
                # Loop back to start.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # cv2 returns BGR; pipeline expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = rgb
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
