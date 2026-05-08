"""Backend wrapping `transformers.pipeline("depth-estimation", ...)` for any
HF depth model that exposes the `predicted_depth` output (DAV2-Metric, DepthPro,
ZoeDepth)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import BackendInfo, DepthBackend, StatusFn


class HFPipelineBackend:
    def __init__(self, info: BackendInfo, weights_root: Path):
        self.info = info
        self.weights_dir = weights_root / info.key
        self._pipe = None

    def load(self, status: StatusFn, device: str = "cuda") -> None:
        from PIL import Image
        from transformers import pipeline as hf_pipeline

        if not (self.weights_dir / "model.safetensors").exists():
            _download_snapshot(self.info.repo, self.weights_dir, status)

        status("loading")
        print(f"[depth] loading {self.info.key} on {device} ...")
        t0 = time.time()
        self._pipe = hf_pipeline(
            "depth-estimation",
            model=str(self.weights_dir),
            device=device,
        )

        status("warming")
        dummy = Image.fromarray(np.zeros((self.info.infer_h, self.info.infer_w, 3), dtype=np.uint8))
        self._pipe(dummy)
        print(f"[depth] {self.info.key} ready in {time.time()-t0:.1f}s")
        status("ready")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        out = self._pipe(Image.fromarray(rgb))
        d = out["predicted_depth"]
        if hasattr(d, "cpu"):
            d = d.cpu()
        d = d.numpy().astype(np.float32)
        # Ensure (infer_h, infer_w)
        if d.shape != (self.info.infer_h, self.info.infer_w):
            from PIL import Image as _I
            d = np.asarray(_I.fromarray(d).resize(
                (self.info.infer_w, self.info.infer_h), _I.BILINEAR))
        return d


def _download_snapshot(repo: str, wdir: Path, status: StatusFn) -> None:
    """Download an HF snapshot. Progress is polled from the on-disk byte count
    in a background thread — robust whether hf_hub uses pure-Python tqdm or
    hf_transfer (Rust)."""
    import os
    import threading
    # Xet sometimes stalls indefinitely on non-Xet repos; force the classic
    # HTTP downloader. Must be set before importing huggingface_hub here.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    status("downloading", "...", "starting")
    print(f"[depth] downloading {repo} -> {wdir}")

    # Find the total expected bytes via the HF API.
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        total_bytes = sum((s.size or 0) for s in (info.siblings or []))
    except Exception:
        total_bytes = 0

    wdir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    # hf_hub stores blobs in ~/.cache/huggingface/hub/models--<org>--<name>/blobs.
    import os
    hub_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cache_dirs = [
        hub_root / f"models--{repo.replace('/', '--')}",
        wdir,  # local_dir copies once cache blob is complete
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
