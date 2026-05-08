"""Metric3D V2 backend (yvanyin/Metric3D, loaded via torch.hub)."""

from __future__ import annotations

import time

import numpy as np

from .base import BackendInfo, StatusFn

# Metric3D's "canonical camera" focal length used for un-canonicalizing depth.
# (See Metric3D V2 paper / repo: predictions are in canonical-camera space, scale
#  by real_focal / canonical_focal to recover true metric depth.)
_CANONICAL_FOCAL = 1000.0


class Metric3DBackend:
    """Wraps Metric3D V2 ViT (`metric3d_vit_small` or `metric3d_vit_large`).

    Unlike DAV2 etc., Metric3D requires intrinsics to un-canonicalize the
    predicted depth into true metres.
    """

    def __init__(self, info: BackendInfo, hub_entry: str, focal_px: float = _CANONICAL_FOCAL):
        self.info = info
        self.hub_entry = hub_entry  # "metric3d_vit_small" | "metric3d_vit_large"
        self.focal_px = focal_px    # real focal in pixels at infer resolution
        self._model = None
        self._device = "cuda"

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        status("loading")
        print(f"[depth] loading {self.info.key} via torch.hub ({self.hub_entry}) ...")
        t0 = time.time()
        import torch
        # Disable strict (torch.hub re-clones every time without it)
        self._device = device
        self._model = torch.hub.load(
            "yvanyin/metric3d", self.hub_entry, pretrain=True, trust_repo=True,
        ).to(device).eval()

        status("warming")
        dummy = torch.zeros(1, 3, self.info.infer_h, self.info.infer_w, device=device)
        with torch.no_grad():
            self._model.inference({"input": dummy})
        print(f"[depth] {self.info.key} ready in {time.time()-t0:.1f}s")
        status("ready")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        # Metric3D expects (B, 3, H, W) float32 normalized by ImageNet mean/std
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            pred_depth, _conf, _out = self._model.inference({"input": x})

        d = pred_depth
        if d.ndim == 4:
            d = d[0, 0]
        elif d.ndim == 3:
            d = d[0]
        d = d.detach().cpu().numpy().astype(np.float32)

        # Un-canonicalize: d_true = d_canonical * (real_f / canonical_f)
        d = d * (self.focal_px / _CANONICAL_FOCAL)
        return d
