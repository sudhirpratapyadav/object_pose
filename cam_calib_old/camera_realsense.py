"""
Minimal RealSense camera interface for cam_calib.

Import this as `camera_realsense` and use:
    cam = RGBDCamera(width, height, fps)
    cam.start()
    frame = cam.get_latest_frame()   # RGBDFrame | None
    pc    = cam.get_latest_pc()      # PointCloud | None
    cam.stop()

Switch cameras by replacing this file's implementation.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "camera"))

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,      0,      1]], dtype=np.float64)


@dataclass
class RGBDFrame:
    timestamp: float
    rgb: np.ndarray    # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) uint16, millimetres
    intrinsics: CameraIntrinsics


@dataclass
class PointCloud:
    timestamp: float
    points: np.ndarray              # (N, 3) float32, metres, camera frame
    colors: Optional[np.ndarray]    # (N, 3) uint8 RGB


class RGBDCamera:
    """
    Threaded RealSense RGBD camera.

    Frames and point clouds are dropped on overflow — caller always gets the
    latest available data via get_latest_frame() / get_latest_pc().
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        min_depth_m: float = 0.1,
        max_depth_m: float = 3.0,
        pc_downsample: int = 4,
    ):
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed")

        self.width = width
        self.height = height
        self.fps = fps
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.pc_downsample = pc_downsample

        self._frame_q: Queue[RGBDFrame] = Queue(maxsize=2)
        self._pc_q:    Queue[PointCloud] = Queue(maxsize=1)

        self.intrinsics: Optional[CameraIntrinsics] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pc_gen = rs.pointcloud()

    def start(self) -> CameraIntrinsics:
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError("No RealSense devices found")

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8,  self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,   self.fps)

        profile = self._pipeline.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            width=intr.width, height=intr.height,
            fx=intr.fx, fy=intr.fy,
            cx=intr.ppx, cy=intr.ppy,
        )

        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="RealSenseThread")
        self._thread.start()
        return self.intrinsics

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipeline:
            self._pipeline.stop()

    def get_latest_frame(self) -> Optional[RGBDFrame]:
        """Return the most recent frame, or None if none available."""
        frame = None
        while True:
            try:
                frame = self._frame_q.get_nowait()
            except Empty:
                break
        return frame

    def get_latest_pc(self) -> Optional[PointCloud]:
        """Return the most recent point cloud, or None if none available."""
        pc = None
        while True:
            try:
                pc = self._pc_q.get_nowait()
            except Empty:
                break
        return pc

    def _loop(self):
        align = rs.align(rs.stream.color)
        while self._running.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            ts  = color_frame.get_timestamp() / 1000.0
            rgb = np.asanyarray(color_frame.get_data()).copy()   # (H, W, 3) uint8
            dep = np.asanyarray(depth_frame.get_data()).copy()   # (H, W) uint16 mm

            frame = RGBDFrame(timestamp=ts, rgb=rgb, depth=dep, intrinsics=self.intrinsics)
            try:
                self._frame_q.put_nowait(frame)
            except Full:
                try:
                    self._frame_q.get_nowait()
                except Empty:
                    pass
                try:
                    self._frame_q.put_nowait(frame)
                except Full:
                    pass

            # Point cloud
            pc = self._make_pc(depth_frame, color_frame, ts)
            if pc is not None:
                try:
                    self._pc_q.put_nowait(pc)
                except Full:
                    try:
                        self._pc_q.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._pc_q.put_nowait(pc)
                    except Full:
                        pass

    def _make_pc(self, depth_frame, color_frame, ts: float) -> Optional[PointCloud]:
        try:
            points = self._pc_gen.calculate(depth_frame)
            self._pc_gen.map_to(color_frame)

            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex   = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            z = verts[:, 2]
            valid = (z > self.min_depth_m) & (z < self.max_depth_m)
            ds = np.zeros(len(verts), dtype=bool)
            ds[::self.pc_downsample] = True
            mask = valid & ds

            pts = verts[mask]
            rgb_img = np.asanyarray(color_frame.get_data())
            h, w = rgb_img.shape[:2]
            tx = np.clip((tex[mask, 0] * w).astype(np.int32), 0, w - 1)
            ty = np.clip((tex[mask, 1] * h).astype(np.int32), 0, h - 1)
            colors = rgb_img[ty, tx]

            return PointCloud(timestamp=ts, points=pts, colors=colors)
        except Exception:
            return None
