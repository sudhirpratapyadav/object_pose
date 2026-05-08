"""Common interface for depth-model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class BackendInfo:
    key: str               # short id used in dropdown
    label: str             # human-readable label
    family: str            # "hf-pipeline" | "unidepth" | "metric3d"
    repo: str              # HuggingFace repo (or torch.hub source)
    infer_w: int = 640     # preferred inference width
    infer_h: int = 480     # preferred inference height


# status callback takes a tuple of strings/numbers
StatusFn = Callable[..., None]


class DepthBackend(Protocol):
    info: BackendInfo

    def load(self, status: StatusFn, device: str = "cuda") -> None: ...

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Run inference. rgb is (H, W, 3) uint8 already at infer_w/infer_h.
        Returns metric depth (H, W) float32 in metres."""
        ...
