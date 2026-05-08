"""SAM2 backend registry. Mirrors depth/backends in style.

Each backend is identified by a key like 'sam2-tiny'. Loading happens lazily
inside the worker process so the parent doesn't pull torch+sam2 into memory
just to advertise the dropdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

StatusFn = Callable[..., None]


@dataclass
class BackendInfo:
    key: str
    hf_repo: str
    config_name: str   # config file name in the sam2 package
    description: str


BACKENDS: dict[str, BackendInfo] = {
    "sam2-tiny": BackendInfo(
        key="sam2-tiny",
        hf_repo="facebook/sam2-hiera-tiny",
        config_name="sam2_hiera_t.yaml",
        description="Fastest. ~50ms/click on a 30-series GPU.",
    ),
    "sam2-small": BackendInfo(
        key="sam2-small",
        hf_repo="facebook/sam2-hiera-small",
        config_name="sam2_hiera_s.yaml",
        description="Slightly better masks, ~2x slower than tiny.",
    ),
    "sam2-base": BackendInfo(
        key="sam2-base",
        hf_repo="facebook/sam2-hiera-base-plus",
        config_name="sam2_hiera_b+.yaml",
        description="Strong quality, moderate speed.",
    ),
    "sam2-large": BackendInfo(
        key="sam2-large",
        hf_repo="facebook/sam2-hiera-large",
        config_name="sam2_hiera_l.yaml",
        description="Best masks, slowest.",
    ),
}

DEFAULT_MODEL = "sam2-tiny"


class SamBackend:
    """Thin wrapper around a SAM2 image predictor.

    set_image(rgb)              — prepare embeddings for a frame
    predict(points, labels)     — return best mask (H, W) bool
    """

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        self._predictor = None

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        import glob
        from huggingface_hub import snapshot_download
        # SAM2 official repo path. Imports are lazy so the registry can be
        # listed without sam2 installed.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        status("downloading", self.info.hf_repo, 0)
        local_dir = snapshot_download(self.info.hf_repo)
        status("loading")

        # HF snapshots ship the checkpoint as 'sam2_hiera_<size>.pt'. Glob
        # rather than guess the exact filename.
        pts = glob.glob(f"{local_dir}/*.pt")
        if not pts:
            raise FileNotFoundError(f"no .pt checkpoint in {local_dir}")
        ckpt_path = pts[0]
        model = build_sam2(self.info.config_name, ckpt_path, device=device)
        self._predictor = SAM2ImagePredictor(model)
        status("ready")

    def set_image(self, rgb: np.ndarray) -> None:
        self._predictor.set_image(rgb)

    def predict_point(self, x: int, y: int, label: int = 1) -> np.ndarray:
        """Return (H, W) bool mask for a single positive/negative click."""
        import numpy as np
        pts = np.array([[x, y]], dtype=np.float32)
        lbls = np.array([label], dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=True,
        )
        # Pick highest-scoring mask
        best = int(np.argmax(scores))
        return masks[best].astype(bool)


def make_backend(key: str) -> SamBackend:
    if key not in BACKENDS:
        raise KeyError(f"unknown sam2 backend: {key}")
    return SamBackend(BACKENDS[key])
