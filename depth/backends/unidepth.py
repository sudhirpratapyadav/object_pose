"""UniDepth V2 backend (lpiccinelli-eth/UniDepth)."""

from __future__ import annotations

import time

import numpy as np

from .base import BackendInfo, StatusFn


class UniDepthBackend:
    def __init__(self, info: BackendInfo):
        self.info = info
        self._model = None
        self._device = "cuda"

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        status("loading")
        print(f"[depth] loading {self.info.key} ({self.info.repo}) on {device} ...")
        t0 = time.time()
        try:
            from unidepth.models import UniDepthV2  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "unidepth package not installed. Install with: "
                "pip install git+https://github.com/lpiccinelli-eth/UniDepth.git"
            ) from e
        import torch

        self._device = device
        self._model = UniDepthV2.from_pretrained(self.info.repo).to(device).eval()

        status("warming")
        dummy = torch.zeros(3, self.info.infer_h, self.info.infer_w, dtype=torch.uint8, device=device)
        with torch.no_grad():
            self._model.infer(dummy)
        print(f"[depth] {self.info.key} ready in {time.time()-t0:.1f}s")
        status("ready")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self._device)
        with torch.no_grad():
            out = self._model.infer(t)
        d = out["depth"]
        if d.ndim == 4:
            d = d[0, 0]
        elif d.ndim == 3:
            d = d[0]
        return d.detach().cpu().numpy().astype(np.float32)
