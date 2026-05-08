"""Network camera client. Connects to a `camera_client` MJPEG server and exposes
the same interface as `RealSenseRGB` (start/get/stop, returns Intrinsics).

Usage:
    cam = NetworkRGB("http://other-pc:8080")
    intr = cam.start()
    rgb = cam.get()
"""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np

from .realsense import Intrinsics


class NetworkRGB:
    def __init__(self, url: str, timeout: float = 5.0):
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.intrinsics: Intrinsics | None = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._resp = None

    def start(self) -> Intrinsics:
        with urllib.request.urlopen(f"{self.base_url}/intrinsics", timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        self.intrinsics = Intrinsics(
            width=int(data["width"]), height=int(data["height"]),
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data["cx"]), cy=float(data["cy"]),
        )
        print(f"[netcam] {self.intrinsics.width}x{self.intrinsics.height} "
              f"fx={self.intrinsics.fx:.1f} from {self.base_url}")
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.intrinsics

    def stop(self):
        self._running.clear()
        if self._resp is not None:
            try: self._resp.close()
            except Exception: pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _loop(self):
        import cv2  # only the receiver needs cv2

        while self._running.is_set():
            try:
                self._resp = urllib.request.urlopen(f"{self.base_url}/stream", timeout=self.timeout)
                buf = b""
                while self._running.is_set():
                    chunk = self._resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # MJPEG SOI/EOI markers
                    while True:
                        soi = buf.find(b"\xff\xd8")
                        eoi = buf.find(b"\xff\xd9", soi + 2) if soi != -1 else -1
                        if soi == -1 or eoi == -1:
                            break
                        jpg = buf[soi:eoi + 2]
                        buf = buf[eoi + 2:]
                        bgr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                           cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        with self._lock:
                            self._latest = rgb
            except Exception as e:
                if self._running.is_set():
                    print(f"[netcam] stream error: {e}; reconnecting in 1s")
                    self._running.wait(1.0)
            finally:
                if self._resp is not None:
                    try: self._resp.close()
                    except Exception: pass
                    self._resp = None
