"""Common interface for depth-model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class BackendInfo:
    key: str               # short id used in dropdown
    label: str             # human-readable label
    family: str            # "hf-pipeline" | "unidepth" | "metric3d" | "moge"
    repo: str              # HuggingFace repo (or torch.hub source)
    infer_w: int = 640     # preferred inference width
    infer_h: int = 480     # preferred inference height
    has_normals: bool = False  # whether infer() also returns surface normals


# status callback takes a tuple of strings/numbers
StatusFn = Callable[..., None]


class DepthBackend(Protocol):
    info: BackendInfo

    def load(self, status: StatusFn, device: str = "cuda") -> None: ...

    def infer(self, rgb: np.ndarray):
        """Run inference. rgb is (H, W, 3) uint8 already at infer_w/infer_h.

        Returns either:
          - depth: (H, W) float32 in metres (no normals), or
          - (depth, normal): with normal a (H, W, 3) float32 unit-vector map
            in camera frame. Backends that don't produce normals return only
            depth.
        """
        ...
