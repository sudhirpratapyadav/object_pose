"""Viser viewer: RGB panel, depth panel, 3-D point cloud, origin + frustum."""

from __future__ import annotations

import numpy as np
import viser


def _depth_to_rgb(depth_m: np.ndarray) -> np.ndarray:
    dmax = float(depth_m.max())
    if dmax <= 0:
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    u8 = (depth_m / dmax * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _frustum_segments(width: int, height: int,
                      fx: float, fy: float, cx: float, cy: float,
                      scale: float = 0.3) -> np.ndarray:
    """10 line segments forming a camera-frame frustum: 4 rays + 4 edges + 2 diagonals."""
    corners = np.array([
        [(0     - cx) / fx * scale, (0      - cy) / fy * scale, scale],
        [(width - cx) / fx * scale, (0      - cy) / fy * scale, scale],
        [(width - cx) / fx * scale, (height - cy) / fy * scale, scale],
        [(0     - cx) / fx * scale, (height - cy) / fy * scale, scale],
    ], dtype=np.float32)
    apex = np.zeros(3, dtype=np.float32)
    segs = []
    for c in corners:
        segs.append([apex, c])
    for i in range(4):
        segs.append([corners[i], corners[(i + 1) % 4]])
    segs.append([corners[0], corners[2]])
    segs.append([corners[1], corners[3]])
    return np.array(segs, dtype=np.float32)


class Viewer:
    def __init__(self, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float,
                 model_keys: list[str] | None = None,
                 default_model: str | None = None,
                 label: str = "Object Pose Detector"):
        self.server = viser.ViserServer(label=label)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        with self.server.gui.add_folder("Camera"):
            self.gui_rgb   = self.server.gui.add_image(image=blank, label="RGB")
            self.gui_depth = self.server.gui.add_image(image=blank, label="Depth (model)")

        with self.server.gui.add_folder("Stats"):
            self.txt_rgb_fps   = self.server.gui.add_text("RGB fps",   initial_value="0.0")
            self.txt_depth_fps = self.server.gui.add_text("Depth fps", initial_value="0.0")
            self.txt_pc_fps    = self.server.gui.add_text("PC fps",    initial_value="0.0")

        with self.server.gui.add_folder("Point Cloud"):
            self.sl_pc_size = self.server.gui.add_slider(
                "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.025,
            )

        @self.sl_pc_size.on_update
        def _(_):
            if self._pc_handle is not None:
                self._pc_handle.point_size = self.sl_pc_size.value

        # Depth model dropdown — selection is read by main loop
        self._on_model_change = None
        if model_keys:
            with self.server.gui.add_folder("Depth Model"):
                self.dd_model = self.server.gui.add_dropdown(
                    "Model", options=model_keys,
                    initial_value=default_model or model_keys[0],
                )
                self.txt_model_status = self.server.gui.add_text(
                    "Status", initial_value="ready",
                )

            @self.dd_model.on_update
            def _(_):
                if self._on_model_change is not None:
                    self._on_model_change(self.dd_model.value)

        # Origin frame (world == camera frame for now)
        self.server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.005)

        # Camera frustum at origin (camera is at origin, looking +Z)
        segs = _frustum_segments(width, height, fx, fy, cx, cy)
        self.server.scene.add_line_segments(
            "/camera_frustum",
            points=segs,
            colors=np.tile(np.array([0.2, 0.9, 0.2], dtype=np.float32),
                           (len(segs), 2, 1)),
            line_width=2.0,
        )
        self._pc_handle = None

    def update_rgb(self, rgb: np.ndarray) -> None:
        self.gui_rgb.image = rgb

    def update_depth(self, depth_m: np.ndarray) -> None:
        self.gui_depth.image = _depth_to_rgb(depth_m)

    def update_point_cloud(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if self._pc_handle is not None:
            self._pc_handle.remove()
        colors_f = rgb.astype(np.float32) / 255.0
        self._pc_handle = self.server.scene.add_point_cloud(
            "/point_cloud",
            points=xyz.astype(np.float32),
            colors=colors_f,
            point_size=self.sl_pc_size.value,
        )

    def set_model_change_callback(self, fn) -> None:
        self._on_model_change = fn

    def set_model_status(self, text: str) -> None:
        if hasattr(self, "txt_model_status"):
            self.txt_model_status.value = text

    def update_fps(self, rgb_fps: float, depth_fps: float, pc_fps: float) -> None:
        self.txt_rgb_fps.value   = f"{rgb_fps:.1f}"
        self.txt_depth_fps.value = f"{depth_fps:.1f}"
        self.txt_pc_fps.value    = f"{pc_fps:.1f}"

    def stop(self) -> None:
        self.server.stop()
