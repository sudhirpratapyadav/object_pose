"""MoGe / MoGe-2 backend (Microsoft, NeurIPS 2025).

MoGe-2 predicts a metric 3D point map directly. We only need the Z-channel
to slot into the existing depth/pc pipeline.

Repo:   https://github.com/microsoft/MoGe
Models: facebook/sam2-style, hosted on HF under 'Ruicheng/<key>'.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import BackendInfo, DepthBackend, StatusFn


class MoGeBackend:
    def __init__(self, info: BackendInfo, weights_root: Path):
        self.info = info
        self.weights_dir = weights_root / info.key
        self._model = None
        self._device = "cuda"

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        self._device = device
        import torch
        # Lazy imports — keep the registry listable without moge installed.
        from moge.model.v2 import MoGeModel

        if not (self.weights_dir / "model.pt").exists() and \
           not any(self.weights_dir.glob("*.safetensors")):
            _download_snapshot(self.info.repo, self.weights_dir, status)

        status("loading")
        print(f"[depth] loading {self.info.key} on {device} ...", flush=True)
        t0 = time.time()
        # MoGeModel.from_pretrained takes either a checkpoint *file* or an HF
        # repo id, not a directory. Find the checkpoint file in our snapshot.
        ckpt_files = list(self.weights_dir.glob("model.pt")) \
                     + list(self.weights_dir.glob("*.safetensors"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"no model.pt / .safetensors in {self.weights_dir}"
            )
        self._model = MoGeModel.from_pretrained(str(ckpt_files[0]))
        self._model = self._model.to(device).eval()

        status("warming")
        with torch.no_grad():
            dummy = torch.zeros(3, self.info.infer_h, self.info.infer_w,
                                dtype=torch.float32, device=device)
            self._model.infer(dummy)
        print(f"[depth] {self.info.key} ready in {time.time()-t0:.1f}s", flush=True)
        status("ready")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        # MoGe expects (3, H, W) float in [0, 1].
        rgb_t = torch.from_numpy(rgb).to(self._device).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self._model.infer(rgb_t)
        # out is a dict; use 'depth' if present, else derive from 'points' z.
        if "depth" in out and out["depth"] is not None:
            d = out["depth"]
        else:
            d = out["points"][..., 2]
        d = d.detach().cpu().numpy().astype(np.float32)
        # Replace NaNs/inf (invalid pixels) with 0 — pipeline filters z<=0.
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if d.shape != (self.info.infer_h, self.info.infer_w):
            from PIL import Image as _I
            d = np.asarray(_I.fromarray(d).resize(
                (self.info.infer_w, self.info.infer_h), _I.BILINEAR))
        return d


def _download_snapshot(repo: str, wdir: Path, status: StatusFn) -> None:
    """Same byte-polling progress reporting used by the HF backend."""
    import os
    import threading
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    status("downloading", "...", "starting")
    print(f"[depth] downloading {repo} -> {wdir}", flush=True)

    try:
        info = HfApi().model_info(repo, files_metadata=True)
        total_bytes = sum((s.size or 0) for s in (info.siblings or []))
    except Exception:
        total_bytes = 0

    wdir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    hub_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cache_dirs = [
        hub_root / f"models--{repo.replace('/', '--')}",
        wdir,
    ]

    def _dir_size(p: Path) -> int:
        n = 0
        if not p.exists():
            return 0
        for f in p.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    n += f.stat().st_size
            except OSError:
                pass
        return n

    def _poll():
        while not stop.is_set():
            cur = max(_dir_size(d) for d in cache_dirs)
            mb = cur / 1_000_000
            if total_bytes:
                pct = 100.0 * cur / total_bytes
                tot_mb = total_bytes / 1_000_000
                status("downloading", "all files",
                       f"{pct:.0f}% ({mb:.0f}/{tot_mb:.0f} MB)")
            else:
                status("downloading", "all files", f"{mb:.0f} MB")
            stop.wait(0.5)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        snapshot_download(repo_id=repo, local_dir=str(wdir))
    finally:
        stop.set()
        t.join(timeout=1.0)
    status("downloading", "all files", "100%")
