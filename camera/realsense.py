"""RealSense camera wrapper with mode-aware streams.

Three modes are supported, picked at start() / reopen() time:
  - "rgb"        : color only. Works on any RealSense (no depth, no IR).
  - "rgbd"       : color + factory-aligned depth, emitter ON.
                   Used by the camera-depth backend.
  - "rgb_stereo" : color + rectified IR pair (left/right), emitter OFF.
                   Used by FoundationStereo. The IR pair is the same physical
                   stereo D4xx uses internally; the SDK ships them rectified
                   (horizontal epipolar lines) — exactly what learned stereo
                   models want. The emitter is disabled because its dot
                   pattern adds artificial texture that the learned model
                   was not trained on.

Threaded; latest-frame slot. Switching mode tears down the SDK pipeline and
opens a fresh one; ``reopen(mode)`` makes that easy from the rest of the
server. ``has_depth`` / ``has_stereo`` reflect the active mode after start.

If the requested mode can't be opened (e.g. RGBD probe gets no frames after
1.5 s), start() falls back to "rgb" so the rest of the pipeline still works.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyrealsense2 as rs


CameraMode = Literal["rgb", "rgbd", "rgb_stereo"]


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class StereoCalib:
    """Calibration of the rectified IR pair, in the IR-left frame.

    The IR pair is factory-rectified: ``baseline_m`` is purely horizontal
    in IR-left coordinates, so disparity → depth via z = fx*B/d.
    ``ir_to_color_R/t`` map a 3D point from IR-left frame to color frame
    so we can sample colors for an FS point cloud.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline_m: float            # |t_x| between IR-left and IR-right, metres
    ir_to_color_R: np.ndarray    # (3, 3)
    ir_to_color_t: np.ndarray    # (3,) metres


class RealSenseRGB:
    """Backwards-compatible name; the class actually handles all 3 modes.

    Boot mode is taken from ``enable_depth`` for callers that haven't been
    updated yet: True → "rgbd", False → "rgb". New callers should pass
    ``mode=`` explicitly.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_depth: bool = False,
                 mode: CameraMode | None = None):
        self.width, self.height, self.fps = width, height, fps
        if mode is None:
            mode = "rgbd" if enable_depth else "rgb"
        self._mode: CameraMode = mode
        # Public flags reflecting the *active* mode after open.
        self.has_depth = False
        self.has_stereo = False
        self.intrinsics: Intrinsics | None = None
        self.stereo_calib: StereoCalib | None = None
        # Latest-frame slots.
        self._latest: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_ir1: np.ndarray | None = None
        self._latest_ir2: np.ndarray | None = None
        self._depth_scale = 0.001
        self._align: rs.align | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._pipe: rs.pipeline | None = None
        self._thread: threading.Thread | None = None

    # ---- public surface --------------------------------------------------

    @property
    def mode(self) -> CameraMode:
        return self._mode

    def start(self) -> Intrinsics:
        """Open the pipeline in the requested mode.

        Falls back to "rgb" if the requested mode probes empty (e.g. RGBD
        delivers no frames within 1.5 s — a real D455 failure mode).
        """
        if not self._open(self._mode):
            print(f"[cam] WARNING: mode='{self._mode}' probe got no frames; "
                  f"retrying as 'rgb'.", flush=True)
            self._mode = "rgb"
            self._open("rgb")  # raises if this also fails
        intr = self.intrinsics
        if intr is None:
            raise RuntimeError("camera failed to start")
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        flags = []
        if self.has_depth:  flags.append("depth")
        if self.has_stereo: flags.append("stereo")
        flag_s = (" + " + " + ".join(flags)) if flags else ""
        print(f"[cam] {intr.width}x{intr.height}  fx={intr.fx:.1f}  "
              f"fy={intr.fy:.1f}  mode={self._mode}{flag_s}",
              flush=True)
        return intr

    def reopen(self, mode: CameraMode) -> Intrinsics:
        """Tear down the current pipeline and open a fresh one in ``mode``.

        Returns the new intrinsics. Raises if the new mode can't be opened
        (caller should restore the previous mode by calling reopen() again).
        """
        # Stop the read thread but keep the lock + slots.
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Reset latest slots so callers can't read stale data through the
        # gap. They get refilled by the new mode's first frame.
        with self._lock:
            self._latest = None
            self._latest_depth = None
            self._latest_ir1 = None
            self._latest_ir2 = None
        self._mode = mode
        if not self._open(mode):
            raise RuntimeError(
                f"camera reopen('{mode}') failed: probe got no frames"
            )
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.intrinsics  # type: ignore[return-value]

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipe:
            try: self._pipe.stop()
            except Exception: pass

    def get(self) -> np.ndarray | None:
        """Latest color frame (rgb8). None if no frame yet."""
        with self._lock:
            return self._latest

    def get_depth(self) -> np.ndarray | None:
        """Latest aligned depth (m). None if mode != rgbd or no frame yet."""
        with self._lock:
            return self._latest_depth

    def get_stereo(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Latest (IR-left, IR-right) Y8 pair. None if mode != rgb_stereo."""
        with self._lock:
            if self._latest_ir1 is None or self._latest_ir2 is None:
                return None
            return self._latest_ir1, self._latest_ir2

    # ---- internals -------------------------------------------------------

    def _open(self, mode: CameraMode, *,
              probe_timeout_s: float = 1.5) -> bool:
        """Open a fresh pipeline for ``mode`` and probe ≥1 frame.

        On success, populates self.intrinsics, self.has_*, self.stereo_calib,
        and leaves self._pipe running. On failure, stops the pipeline and
        returns False. Raises only if pipe.start itself raises.
        """
        if self._pipe is not None:
            try: self._pipe.stop()
            except Exception: pass
        self._pipe = rs.pipeline()
        cfg = rs.config()
        # Color is always on (used by every mode).
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.rgb8, self.fps)
        if mode == "rgbd":
            cfg.enable_stream(rs.stream.depth, self.width, self.height,
                              rs.format.z16, self.fps)
        elif mode == "rgb_stereo":
            cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                              rs.format.y8, self.fps)
            cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height,
                              rs.format.y8, self.fps)
            # Need depth-sensor handle for emitter control; keep depth off
            # but the sensor still exists.
        # mode == "rgb" needs no extras.

        profile = self._pipe.start(cfg)  # may raise

        # Color intrinsics: every mode publishes these; depth pipeline uses
        # them as the camera frame.
        c_intr = profile.get_stream(rs.stream.color)\
            .as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(
            width=c_intr.width, height=c_intr.height,
            fx=c_intr.fx, fy=c_intr.fy,
            cx=c_intr.ppx, cy=c_intr.ppy,
        )

        # Per-mode setup.
        self.has_depth = False
        self.has_stereo = False
        self._align = None
        self.stereo_calib = None

        if mode == "rgbd":
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                # Emitter ON helps the factory stereo block-matcher.
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                self._align = rs.align(rs.stream.color)
                self.has_depth = True
            except Exception as exc:
                print(f"[cam] depth setup failed ({exc}); rgbd will fall back",
                      flush=True)

        elif mode == "rgb_stereo":
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                # Emitter OFF — the dot pattern hurts the learned stereo model.
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
                ir1 = profile.get_stream(rs.stream.infrared, 1)\
                    .as_video_stream_profile()
                ir2 = profile.get_stream(rs.stream.infrared, 2)\
                    .as_video_stream_profile()
                ir_intr = ir1.get_intrinsics()
                ir_to_ir2 = ir2.get_extrinsics_to(ir1)
                # baseline = horizontal component of IR1->IR2 translation
                baseline_m = abs(float(ir_to_ir2.translation[0]))
                color_profile = profile.get_stream(rs.stream.color)\
                    .as_video_stream_profile()
                ir_to_color = ir1.get_extrinsics_to(color_profile)
                self.stereo_calib = StereoCalib(
                    fx=ir_intr.fx, fy=ir_intr.fy,
                    cx=ir_intr.ppx, cy=ir_intr.ppy,
                    width=ir_intr.width, height=ir_intr.height,
                    baseline_m=baseline_m,
                    ir_to_color_R=np.array(ir_to_color.rotation,
                                           dtype=np.float64).reshape(3, 3),
                    ir_to_color_t=np.array(ir_to_color.translation,
                                           dtype=np.float64).reshape(3),
                )
                self.has_stereo = True
            except Exception as exc:
                print(f"[cam] stereo setup failed ({exc}); rgb_stereo will fall back",
                      flush=True)

        # Probe ≥ 1 frame within the timeout.
        deadline = time.time() + probe_timeout_s
        while time.time() < deadline:
            remaining_ms = max(50, int((deadline - time.time()) * 1000))
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=remaining_ms)
            except RuntimeError:
                continue
            if frames.get_color_frame():
                if mode == "rgbd":
                    print(f"[cam] depth stream enabled  "
                          f"depth_scale={self._depth_scale:g} m/unit",
                          flush=True)
                elif mode == "rgb_stereo" and self.stereo_calib is not None:
                    sc = self.stereo_calib
                    print(f"[cam] stereo enabled  baseline={sc.baseline_m*1000:.1f} mm  "
                          f"ir fx={sc.fx:.1f} fy={sc.fy:.1f}  emitter=OFF",
                          flush=True)
                return True

        # No frames in budget; tear down the pipeline.
        try: self._pipe.stop()
        except Exception: pass
        self._pipe = None
        self.has_depth = False
        self.has_stereo = False
        self.stereo_calib = None
        return False

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
            ir1 = ir2 = None
            if self.has_stereo:
                f1 = frames.get_infrared_frame(1)
                f2 = frames.get_infrared_frame(2)
                if f1 and f2:
                    ir1 = np.asanyarray(f1.get_data()).copy()
                    ir2 = np.asanyarray(f2.get_data()).copy()
            with self._lock:
                self._latest = rgb
                if depth_m is not None:
                    self._latest_depth = depth_m
                if ir1 is not None and ir2 is not None:
                    self._latest_ir1 = ir1
                    self._latest_ir2 = ir2
