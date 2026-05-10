"""Backend registry: maps short keys to a factory that builds a DepthBackend."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import BackendInfo, DepthBackend
from .camera import CameraDepthBackend, make_camera_backend
from .hf_pipeline import HFPipelineBackend
from .metric3d import Metric3DBackend
from .moge import MoGeBackend
from .unidepth import UniDepthBackend

CAMERA_DEPTH_KEY = "camera-depth"

WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "weights"


def _hf(key: str, repo: str, label: str | None = None,
        infer_w: int = 640, infer_h: int = 480) -> Callable[[float], DepthBackend]:
    info = BackendInfo(
        key=key, label=label or key, family="hf-pipeline",
        repo=repo, infer_w=infer_w, infer_h=infer_h,
    )
    return lambda focal_px: HFPipelineBackend(info, WEIGHTS_ROOT)


def _unidepth(key: str, repo: str, label: str,
              infer_w: int = 640, infer_h: int = 480) -> Callable[[float], DepthBackend]:
    info = BackendInfo(
        key=key, label=label, family="unidepth",
        repo=repo, infer_w=infer_w, infer_h=infer_h,
    )
    return lambda focal_px: UniDepthBackend(info)


def _metric3d(key: str, hub_entry: str, label: str,
              infer_w: int = 640, infer_h: int = 480) -> Callable[[float], DepthBackend]:
    info = BackendInfo(
        key=key, label=label, family="metric3d",
        repo=hub_entry, infer_w=infer_w, infer_h=infer_h,
    )
    return lambda focal_px: Metric3DBackend(info, hub_entry, focal_px=focal_px)


def _moge(key: str, repo: str, label: str,
          infer_w: int = 640, infer_h: int = 480,
          has_normals: bool = False) -> Callable[[float], DepthBackend]:
    info = BackendInfo(
        key=key, label=label, family="moge",
        repo=repo, infer_w=infer_w, infer_h=infer_h,
        has_normals=has_normals,
    )
    return lambda focal_px: MoGeBackend(info, WEIGHTS_ROOT, focal_px=focal_px)


# Sentinel factory: the camera-depth backend is built by depth_worker via
# make_camera_backend(...) using runtime shm info, not via BACKENDS[key].
def _camera_sentinel(focal_px: float) -> DepthBackend:
    raise RuntimeError(
        "camera-depth backend must be built via make_camera_backend(); the "
        "depth worker handles this internally when model_key == 'camera-depth'."
    )


# Registry: key -> factory(focal_px) -> DepthBackend
BACKENDS: dict[str, Callable[[float], DepthBackend]] = {
    CAMERA_DEPTH_KEY:      _camera_sentinel,
    "dav2-indoor-small":   _hf("dav2-indoor-small",   "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"),
    "dav2-indoor-base":    _hf("dav2-indoor-base",    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"),
    "dav2-indoor-large":   _hf("dav2-indoor-large",   "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"),
    "dav2-outdoor-small":  _hf("dav2-outdoor-small",  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"),
    "dav2-outdoor-base":   _hf("dav2-outdoor-base",   "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"),
    "dav2-outdoor-large":  _hf("dav2-outdoor-large",  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"),
    "depthpro":            _hf("depthpro",            "apple/DepthPro-hf"),
    "unidepth-v2-vitl14":  _unidepth("unidepth-v2-vitl14",
                                     "lpiccinelli/unidepth-v2-vitl14",
                                     "UniDepth V2 ViT-L/14"),
    "metric3d-v2-small":   _metric3d("metric3d-v2-small",
                                     "metric3d_vit_small",
                                     "Metric3D V2 ViT-S"),
    "metric3d-v2-large":   _metric3d("metric3d-v2-large",
                                     "metric3d_vit_large",
                                     "Metric3D V2 ViT-L"),
    "moge-2-vits-normal":  _moge("moge-2-vits-normal",
                                 "Ruicheng/moge-2-vits-normal",
                                 "MoGe-2 ViT-S + normal",
                                 has_normals=True),
    "moge-2-vitl":         _moge("moge-2-vitl",
                                 "Ruicheng/moge-2-vitl",
                                 "MoGe-2 ViT-L",
                                 has_normals=False),
    "moge-2-vitl-normal":  _moge("moge-2-vitl-normal",
                                 "Ruicheng/moge-2-vitl-normal",
                                 "MoGe-2 ViT-L + normal",
                                 has_normals=True),
}

DEFAULT_MODEL = "dav2-indoor-small"


def make_backend(key: str, focal_px: float) -> DepthBackend:
    if key not in BACKENDS:
        raise ValueError(f"Unknown backend key '{key}'. Available: {list(BACKENDS)}")
    return BACKENDS[key](focal_px)


__all__ = ["BACKENDS", "CAMERA_DEPTH_KEY", "DEFAULT_MODEL",
           "make_backend", "make_camera_backend",
           "BackendInfo", "DepthBackend", "CameraDepthBackend"]
