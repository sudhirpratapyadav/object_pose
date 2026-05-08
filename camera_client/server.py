"""Camera-PC side: serve RGB frames as MJPEG + a small JSON intrinsics endpoint.

Run this on the PC that has the camera physically attached. The depth/viewer PC
connects with `camera.NetworkRGB("http://<this-pc>:8080")`.

Endpoints:
  GET /stream      multipart/x-mixed-replace JPEG stream (open in a browser to verify)
  GET /intrinsics  JSON { width, height, fx, fy, cx, cy }
  GET /            tiny landing page

Source can be a RealSense camera (default if pyrealsense2 is importable and a
device is attached) or any OpenCV-compatible camera index / video URL.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import numpy as np

JPEG_BOUNDARY = "frame"


class FrameSource:
    """Latest-frame slot; a worker thread keeps `_latest` up to date."""

    def __init__(self):
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self.intrinsics: dict | None = None

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _set(self, rgb: np.ndarray):
        with self._lock:
            self._latest = rgb

    def _loop(self):
        raise NotImplementedError


class RealSenseSource(FrameSource):
    def __init__(self, width: int, height: int, fps: int):
        super().__init__()
        self.width, self.height, self.fps = width, height, fps
        self._pipe = None

    def start(self):
        import pyrealsense2 as rs

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.rgb8, self.fps)
        profile = self._pipe.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
        }
        print(f"[client] realsense {intr.width}x{intr.height} fx={intr.fx:.1f}")
        super().start()

    def stop(self):
        super().stop()
        if self._pipe:
            self._pipe.stop()

    def _loop(self):
        while self._running.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            self._set(np.asanyarray(color.get_data()).copy())


class OpenCVSource(FrameSource):
    """Fallback: any OpenCV-readable source (webcam index, video file, RTSP url)."""

    def __init__(self, src, width: int, height: int, fps: int,
                 fx: float | None, fy: float | None,
                 cx: float | None, cy: float | None):
        super().__init__()
        self.src = src
        self.width, self.height, self.fps = width, height, fps
        # No real intrinsics from a generic webcam — caller can pass them, or we
        # default to "fx = fy = width" and a centered principal point. This is a
        # rough placeholder; calibrate for real metric depth.
        self.intrinsics = {
            "width": width, "height": height,
            "fx": fx if fx is not None else float(width),
            "fy": fy if fy is not None else float(width),
            "cx": cx if cx is not None else width / 2.0,
            "cy": cy if cy is not None else height / 2.0,
        }
        self._cap = None

    def start(self):
        import cv2
        self._cap = cv2.VideoCapture(self.src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open source: {self.src!r}")
        print(f"[client] opencv source={self.src} {self.width}x{self.height}")
        super().start()

    def stop(self):
        super().stop()
        if self._cap:
            self._cap.release()

    def _loop(self):
        import cv2
        while self._running.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._set(rgb)


def _make_handler(source: FrameSource, jpeg_quality: int) -> Callable:
    import cv2

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._send_landing()
            elif self.path == "/intrinsics":
                self._send_intrinsics()
            elif self.path == "/stream":
                self._send_stream()
            else:
                self.send_error(404)

        def _send_landing(self):
            body = (
                b"<html><body style='font-family:sans-serif'>"
                b"<h2>camera_client</h2>"
                b"<p><a href='/stream'>/stream</a> &mdash; MJPEG</p>"
                b"<p><a href='/intrinsics'>/intrinsics</a> &mdash; JSON</p>"
                b"<img src='/stream' style='max-width:90%'/>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_intrinsics(self):
            payload = json.dumps(source.intrinsics or {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_stream(self):
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY}",
            )
            self.end_headers()
            try:
                while True:
                    rgb = source.get()
                    if rgb is None:
                        time.sleep(0.01)
                        continue
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, jpg = cv2.imencode(
                        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(f"--{JPEG_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 60)  # cap at ~60 Hz
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="realsense",
                    help="'realsense' or an OpenCV source (e.g. '0', '/dev/video0', RTSP url)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1..100")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    args = ap.parse_args()

    if args.source == "realsense":
        source: FrameSource = RealSenseSource(args.width, args.height, args.fps)
    else:
        try:
            src: int | str = int(args.source)
        except ValueError:
            src = args.source
        source = OpenCVSource(src, args.width, args.height, args.fps,
                              args.fx, args.fy, args.cx, args.cy)

    source.start()

    handler = _make_handler(source, args.quality)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[client] serving on http://{args.host}:{args.port}  (open in a browser)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[client] stopping")
    finally:
        httpd.shutdown()
        source.stop()


if __name__ == "__main__":
    main()
