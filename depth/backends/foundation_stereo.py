"""FoundationStereo backend: learned stereo on a rectified IR pair.

Reads the IR-left + IR-right Y8 frames from shared memory (written by the
camera thread when the camera is in 'rgb_stereo' mode) and produces a
disparity map → depth (metres) using the upstream FoundationStereo model.

Key facts:
  - The IR pair is factory-rectified (D4xx). z = fx_ir * baseline / disparity.
  - The depth map is in IR-left frame. To match the rest of the pipeline
    (which back-projects with color intrinsics in color frame), we project
    each IR-left point into the color frame using the IR→color extrinsics
    and rasterise depth in the color image. The result lines up with what
    the other backends produce: a (infer_h, infer_w) float32 depth map in
    the color frame at inference resolution.
  - Inference runs at the IR native resolution (typically 848x480 on D455);
    we then warp to color frame and resize to (infer_w, infer_h).

Weights live under ``weights/foundation_stereo/{small,large}/``:
  cfg.yaml + model_best_bp2.pth
"""

from __future__ import annotations

import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import BackendInfo, CameraReq, StatusFn


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = REPO_ROOT / "weights" / "foundation_stereo"
FS_REPO_DIR = REPO_ROOT / "depth" / "foundationstereo"


# Public mapping: dropdown key → variant name on disk.
FOUNDATION_STEREO_VARIANTS = {
    "fs-small": "small",
    "fs-large": "large",
}
FOUNDATION_STEREO_KEYS = tuple(FOUNDATION_STEREO_VARIANTS.keys())


def _ensure_repo_on_path() -> None:
    """Insert the FS repo + its core/ into sys.path. Idempotent."""
    p1, p2 = str(FS_REPO_DIR), str(FS_REPO_DIR / "core")
    if p1 not in sys.path: sys.path.insert(0, p1)
    if p2 not in sys.path: sys.path.insert(0, p2)
    # Utils.py does ``import open3d`` unconditionally; we don't ship it.
    # Stub out the few names it touches at import time.
    import types as _types
    if "open3d" not in sys.modules:
        _o3d = _types.ModuleType("open3d")
        _o3d.geometry = _types.ModuleType("open3d.geometry")
        _o3d.utility  = _types.ModuleType("open3d.utility")
        sys.modules["open3d"] = _o3d
        sys.modules["open3d.geometry"] = _o3d.geometry
        sys.modules["open3d.utility"]  = _o3d.utility


class FoundationStereoBackend:
    def __init__(self, info: BackendInfo, *, variant: str,
                 ir_left_shm_name: str, ir_right_shm_name: str,
                 stereo_seq, src_w: int, src_h: int,
                 infer_w: int, infer_h: int,
                 stereo_calib: dict[str, Any],
                 valid_iters: int = 16,
                 z_max: float = 8.0) -> None:
        self.info = info
        self._variant = variant
        self._ir_left_name = ir_left_shm_name
        self._ir_right_name = ir_right_shm_name
        self._src_w = src_w
        self._src_h = src_h
        self._infer_w = infer_w
        self._infer_h = infer_h
        self._stereo_calib = stereo_calib
        self._valid_iters = valid_iters
        self._z_max = z_max
        self._stereo_seq = stereo_seq
        self._last_seen_seq = 0
        self._cached_depth: np.ndarray | None = None
        self._left_shm: shared_memory.SharedMemory | None = None
        self._right_shm: shared_memory.SharedMemory | None = None
        self._left_arr: np.ndarray | None = None
        self._right_arr: np.ndarray | None = None
        self._model = None
        self._padder = None
        self._device = "cuda"

    # ---- DepthBackend protocol -----------------------------------------

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        self._device = device
        try:
            status("loading")
            _ensure_repo_on_path()

            from omegaconf import OmegaConf
            from core.foundation_stereo import FoundationStereo  # type: ignore
            from core.utils.utils import InputPadder              # type: ignore

            variant_dir = WEIGHTS_ROOT / self._variant
            ckpt_path = variant_dir / "model_best_bp2.pth"
            cfg_path  = variant_dir / "cfg.yaml"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"FoundationStereo weights missing: {ckpt_path}. "
                    f"Copy the cfg.yaml + model_best_bp2.pth from the upstream "
                    f"release into that directory."
                )

            cfg = OmegaConf.load(cfg_path)
            if "vit_size" not in cfg:
                # 'large' cfg.yaml omits this; default upstream is vitl.
                cfg["vit_size"] = "vitl" if self._variant == "large" else "vits"
            cfg_obj = OmegaConf.create(cfg)
            model = FoundationStereo(cfg_obj)
            ckpt = torch.load(str(ckpt_path), map_location="cpu",
                              weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()

            # Open the IR shm slots.
            self._left_shm  = shared_memory.SharedMemory(name=self._ir_left_name)
            self._right_shm = shared_memory.SharedMemory(name=self._ir_right_name)
            self._left_arr  = np.ndarray((self._src_h, self._src_w),
                                         dtype=np.uint8, buffer=self._left_shm.buf)
            self._right_arr = np.ndarray((self._src_h, self._src_w),
                                         dtype=np.uint8, buffer=self._right_shm.buf)

            self._model = model
            self._InputPadder = InputPadder
            status("ready")
        except Exception as exc:
            status("error", str(exc)[:160])
            raise

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Return a (infer_h, infer_w) float32 depth map in the color frame.

        ``rgb`` (color image at infer resolution) is ignored — we read the
        IR pair from shm, do stereo matching, and warp the result into the
        color frame using the IR→color extrinsics.
        """
        # Skip when the camera has no fresh stereo pair (e.g. mode just
        # switched). Return zeros so the rest of the pipeline keeps moving.
        if self._stereo_seq is not None:
            with self._stereo_seq.get_lock():
                cur = int(self._stereo_seq.value)
            if cur == self._last_seen_seq:
                if self._cached_depth is not None:
                    return self._cached_depth
                return np.zeros((self._infer_h, self._infer_w), dtype=np.float32)
            self._last_seen_seq = cur

        if self._left_arr is None or self._right_arr is None or self._model is None:
            return np.zeros((self._infer_h, self._infer_w), dtype=np.float32)

        # Snapshot to local copies so we don't tear during inference.
        left  = self._left_arr.copy()
        right = self._right_arr.copy()

        depth_color = self._infer_once(left, right)

        # Resize to inference resolution.
        if depth_color.shape != (self._infer_h, self._infer_w):
            depth_color = np.asarray(
                Image.fromarray(depth_color).resize(
                    (self._infer_w, self._infer_h), Image.NEAREST,
                ),
                dtype=np.float32,
            )
        self._cached_depth = depth_color
        return depth_color

    # ---- internals ------------------------------------------------------

    def _ir_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Y8 (H,W) -> 1x3xHxW float on device. Tile mono to 3 channels."""
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        t = torch.as_tensor(img).to(self._device).float()[None].permute(0, 3, 1, 2)
        return t

    @torch.inference_mode()
    def _infer_once(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        img0 = self._ir_to_tensor(left)
        img1 = self._ir_to_tensor(right)
        padder = self._InputPadder(img0.shape, divis_by=32, force_square=False)
        img0p, img1p = padder.pad(img0, img1)
        with torch.cuda.amp.autocast(True):
            disp = self._model.forward(img0p, img1p,
                                        iters=self._valid_iters,
                                        test_mode=True)
        disp = padder.unpad(disp.float()).squeeze().detach().cpu().numpy()
        return self._disp_to_color_frame_depth(disp, left)

    def _disp_to_color_frame_depth(self, disp: np.ndarray,
                                    ir_left_y8: np.ndarray) -> np.ndarray:
        """Project disparity → 3D in IR-left frame → color frame depth map.

        Steps:
          1. z = fx_ir * baseline / clip(disp, eps, ∞) in IR-left frame.
          2. Back-project (u, v, z) using IR intrinsics to get xyz_ir.
          3. Apply T_ir->color to map points to color frame.
          4. Project xyz_color through color intrinsics to (u_c, v_c).
          5. Rasterise z_color into a (color_h, color_w) depth map. For
             pixels where multiple IR points project, keep the nearest.
        """
        sc = self._stereo_calib
        fx_ir, fy_ir = sc["fx_ir"], sc["fy_ir"]
        cx_ir, cy_ir = sc["cx_ir"], sc["cy_ir"]
        baseline = sc["baseline_m"]
        H, W = disp.shape

        # Step 1: disparity -> depth in IR-left frame.
        eps = 1e-3
        z_ir = (fx_ir * baseline) / np.clip(disp, eps, None)
        valid_d = (z_ir > 0.05) & (z_ir < self._z_max) & np.isfinite(z_ir)

        if not valid_d.any():
            color_w = sc.get("color_w", W)
            color_h = sc.get("color_h", H)
            return np.zeros((color_h, color_w), dtype=np.float32)

        # Step 2: back-project to IR-left 3D.
        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
        x_ir = (u_grid - cx_ir) * z_ir / fx_ir
        y_ir = (v_grid - cy_ir) * z_ir / fy_ir
        xyz_ir = np.stack([x_ir, y_ir, z_ir], axis=-1)  # (H, W, 3)
        flat_xyz = xyz_ir.reshape(-1, 3)
        flat_valid = valid_d.reshape(-1)
        flat_xyz = flat_xyz[flat_valid]

        # Step 3: IR-left -> color frame.
        R = np.asarray(sc["ir_to_color_R"], dtype=np.float64)  # (3,3)
        t = np.asarray(sc["ir_to_color_t"], dtype=np.float64)  # (3,)
        xyz_color = (R @ flat_xyz.T).T + t  # (N, 3)

        # Step 4: project through color intrinsics.
        fx_c, fy_c = sc["fx_color"], sc["fy_color"]
        cx_c, cy_c = sc["cx_color"], sc["cy_color"]
        color_w   = int(sc.get("color_w", W))
        color_h   = int(sc.get("color_h", H))
        zc = xyz_color[:, 2]
        ok = zc > 0.05
        u_c = np.full(zc.shape, -1.0, dtype=np.float32)
        v_c = np.full(zc.shape, -1.0, dtype=np.float32)
        u_c[ok] = (xyz_color[ok, 0] / zc[ok]) * fx_c + cx_c
        v_c[ok] = (xyz_color[ok, 1] / zc[ok]) * fy_c + cy_c
        u_int = np.round(u_c).astype(np.int32)
        v_int = np.round(v_c).astype(np.int32)
        in_img = ok & (u_int >= 0) & (u_int < color_w) \
                    & (v_int >= 0) & (v_int < color_h)

        # Step 5: rasterise — nearest-z wins per color-frame pixel.
        depth_color = np.zeros((color_h, color_w), dtype=np.float32)
        if in_img.any():
            us = u_int[in_img]
            vs = v_int[in_img]
            zs = zc[in_img].astype(np.float32)
            # Sort by descending z so writes leave the smallest z last.
            order = np.argsort(-zs)
            us, vs, zs = us[order], vs[order], zs[order]
            depth_color[vs, us] = zs
        return depth_color


def make_foundation_stereo_backend(*, key: str,
                                    ir_left_shm_name: str,
                                    ir_right_shm_name: str,
                                    stereo_seq,
                                    src_w: int, src_h: int,
                                    infer_w: int, infer_h: int,
                                    stereo_calib: dict[str, Any],
                                    ) -> FoundationStereoBackend:
    if key not in FOUNDATION_STEREO_VARIANTS:
        raise ValueError(f"unknown FoundationStereo key '{key}'")
    variant = FOUNDATION_STEREO_VARIANTS[key]
    label = "FoundationStereo (ViT-S)" if variant == "small" \
            else "FoundationStereo (ViT-L)"
    info = BackendInfo(
        key=key, label=label, family="foundation-stereo",
        repo="NVlabs/FoundationStereo",
        infer_w=infer_w, infer_h=infer_h, has_normals=False,
        camera_req=CameraReq.RGB_STEREO,
    )
    return FoundationStereoBackend(
        info,
        variant=variant,
        ir_left_shm_name=ir_left_shm_name,
        ir_right_shm_name=ir_right_shm_name,
        stereo_seq=stereo_seq,
        src_w=src_w, src_h=src_h,
        infer_w=infer_w, infer_h=infer_h,
        stereo_calib=stereo_calib,
    )
