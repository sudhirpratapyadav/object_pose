"""Viser viewer: RGB panel, depth panel, 3-D point cloud, origin + frustum."""

from __future__ import annotations

import numpy as np
import trimesh
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
                 label: str = "Object Pose Detector",
                 port: int = 8090):
        self.server = viser.ViserServer(label=label, port=port)

        @self.server.on_client_connect
        def _set_camera(client):
            client.camera.up_direction = (0.0, -1.0, 0.0)
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
                "Point size", min=0.001, max=0.1, step=0.001, initial_value=0.05,
            )
            self.cb_show_camera = self.server.gui.add_checkbox(
                "Show camera", initial_value=True,
            )
            self.dd_display = self.server.gui.add_dropdown(
                "Display", options=["points", "mesh", "both"], initial_value="points",
            )

        @self.sl_pc_size.on_update
        def _(_):
            if self._pc_handle is not None:
                self._pc_handle.point_size = self.sl_pc_size.value

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
                self.txt_model_progress = self.server.gui.add_text(
                    "Progress", initial_value="",
                )
                self.txt_model_file = self.server.gui.add_text(
                    "File", initial_value="",
                )

            @self.dd_model.on_update
            def _(_):
                if self._on_model_change is not None:
                    self._on_model_change(self.dd_model.value)

        self.server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.005)
        # Transform-only parent (no visible axes — axes_length=0).
        self.server.scene.add_frame(
            "/origin/camera",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.0,
            axes_radius=0.0,
        )
        # Toggleable axes glyph (sibling-style child so hiding it doesn't hide pc).
        self._cam_axes = self.server.scene.add_frame(
            "/origin/camera/axes",
            axes_length=0.1,
            axes_radius=0.004,
        )
        segs = _frustum_segments(width, height, fx, fy, cx, cy)
        self._frustum = self.server.scene.add_line_segments(
            "/origin/camera/frustum",
            points=segs,
            colors=np.tile(np.array([0.2, 0.9, 0.2], dtype=np.float32),
                           (len(segs), 2, 1)),
            line_width=2.0,
        )
        self._pc_handle = None
        self._mesh_handle = None

        @self.cb_show_camera.on_update
        def _(_):
            v = self.cb_show_camera.value
            self._cam_axes.visible = v
            self._frustum.visible = v

        @self.dd_display.on_update
        def _(_):
            mode = self.dd_display.value
            if self._pc_handle is not None:
                self._pc_handle.visible = mode in ("points", "both")
            if self._mesh_handle is not None:
                self._mesh_handle.visible = mode in ("mesh", "both")

    def update_rgb(self, rgb: np.ndarray) -> None:
        self.gui_rgb.image = rgb

    def update_depth(self, depth_m: np.ndarray) -> None:
        self.gui_depth.image = _depth_to_rgb(depth_m)

    def update_point_cloud(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if self._pc_handle is not None:
            self._pc_handle.remove()
        colors_f = rgb.astype(np.float32) / 255.0
        self._pc_handle = self.server.scene.add_point_cloud(
            "/origin/camera/point_cloud",
            points=xyz.astype(np.float32),
            colors=colors_f,
            point_size=self.sl_pc_size.value,
            visible=self.dd_display.value in ("points", "both"),
        )

    def update_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                    vertex_colors: np.ndarray) -> None:
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
        vc_u8 = (np.clip(vertex_colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            vertex_colors=vc_u8,
            process=False,
        )
        self._mesh_handle = self.server.scene.add_mesh_trimesh(
            "/origin/camera/mesh",
            mesh=mesh,
            visible=self.dd_display.value in ("mesh", "both"),
        )

    def set_model_change_callback(self, fn) -> None:
        self._on_model_change = fn

    def set_model_status(self, text: str, progress: str = "", filename: str = "") -> None:
        if hasattr(self, "txt_model_status"):
            self.txt_model_status.value = text
        if hasattr(self, "txt_model_progress"):
            self.txt_model_progress.value = progress
        if hasattr(self, "txt_model_file"):
            self.txt_model_file.value = filename

    def update_fps(self, rgb_fps: float, depth_fps: float, pc_fps: float) -> None:
        self.txt_rgb_fps.value   = f"{rgb_fps:.1f}"
        self.txt_depth_fps.value = f"{depth_fps:.1f}"
        self.txt_pc_fps.value    = f"{pc_fps:.1f}"

    def stop(self) -> None:
        self.server.stop()
