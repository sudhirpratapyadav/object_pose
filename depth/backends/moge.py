"""MoGe / MoGe-2 backend (Microsoft, NeurIPS 2025).

MoGe-2 predicts a metric 3D point map directly. We only need the Z-channel
to slot into the existing depth/pc pipeline.

We pass ``fov_x`` (derived from the YAML cam_calib intrinsics) to MoGe so
its predicted depth is consistent with the camera the rest of the pipeline
believes in. Without this hint MoGe estimates its own FOV, which can
disagree with the YAML and bias the back-projection.

Repo:   https://github.com/microsoft/MoGe
Models: facebook/sam2-style, hosted on HF under 'Ruicheng/<key>'.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .base import BackendInfo, DepthBackend, StatusFn


class MoGeBackend:
    def __init__(self, info: BackendInfo, weights_root: Path,
                 focal_px: float | None = None):
        self.info = info
        self.weights_dir = weights_root / info.key
        # Horizontal FOV in degrees, derived from the calibrated focal length
        # and the inference width. None if we don't know — MoGe will estimate.
        if focal_px and focal_px > 0:
            self._fov_x_deg = math.degrees(
                2.0 * math.atan(0.5 * info.infer_w / float(focal_px))
            )
        else:
            self._fov_x_deg = None
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
        if self._fov_x_deg is not None:
            print(f"[depth] {self.info.key} fov_x={self._fov_x_deg:.2f}° "
                  f"(from cam_calib)", flush=True)
        else:
            print(f"[depth] {self.info.key} fov_x=auto (MoGe estimates it)",
                  flush=True)
        print(f"[depth] {self.info.key} ready in {time.time()-t0:.1f}s", flush=True)
        status("ready")

    def infer(self, rgb: np.ndarray):
        import torch
        # MoGe expects (3, H, W) float in [0, 1].
        rgb_t = torch.from_numpy(rgb).to(self._device).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self._model.infer(rgb_t, fov_x=self._fov_x_deg)

        # One-shot diagnostic: log MoGe's reported intrinsics + depth stats so
        # we can verify the model honoured the fov_x we passed (vs estimating
        # its own). MoGe's 'intrinsics' is in normalized image coords
        # (fx = focal / width). Suppress after the first frame.
        if not getattr(self, "_logged_intr", False):
            self._logged_intr = True
            try:
                if "intrinsics" in out and out["intrinsics"] is not None:
                    K = out["intrinsics"].detach().cpu().numpy()
                    fx_n = float(K[..., 0, 0])
                    fy_n = float(K[..., 1, 1])
                    fx_px = fx_n * self.info.infer_w
                    fy_px = fy_n * self.info.infer_h
                    print(f"[depth] {self.info.key} reported fx={fx_px:.1f}px "
                          f"fy={fy_px:.1f}px (normalized {fx_n:.4f},{fy_n:.4f})",
                          flush=True)
                d_dbg = (out["depth"] if "depth" in out and out["depth"] is not None
                         else out["points"][..., 2])
                d_np = d_dbg.detach().cpu().numpy()
                d_finite = d_np[np.isfinite(d_np)]
                if d_finite.size:
                    print(f"[depth] {self.info.key} depth stats "
                          f"min={float(d_finite.min()):.3f} "
                          f"max={float(d_finite.max()):.3f} "
                          f"med={float(np.median(d_finite)):.3f} m",
                          flush=True)
            except Exception as exc:
                print(f"[depth] {self.info.key} diag log failed: {exc}",
                      flush=True)

        # Depth
        if "depth" in out and out["depth"] is not None:
            d = out["depth"]
        else:
            d = out["points"][..., 2]
        d = d.detach().cpu().numpy().astype(np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if d.shape != (self.info.infer_h, self.info.infer_w):
            from PIL import Image as _I
            d = np.asarray(_I.fromarray(d).resize(
                (self.info.infer_w, self.info.infer_h), _I.BILINEAR))

        # Normals (only the *-normal MoGe variants emit these)
        n_arr = None
        if self.info.has_normals and "normal" in out and out["normal"] is not None:
            n = out["normal"].detach().cpu().numpy().astype(np.float32)
            # Expect (H, W, 3); some versions ship (3, H, W).
            if n.ndim == 3 and n.shape[0] == 3:
                n = np.transpose(n, (1, 2, 0))
            n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
            if n.shape[:2] != (self.info.infer_h, self.info.infer_w):
                # Resize per-channel (rare; output usually matches infer dims)
                from PIL import Image as _I
                ch = []
                for c in range(3):
                    ch.append(np.asarray(_I.fromarray(n[..., c]).resize(
                        (self.info.infer_w, self.info.infer_h), _I.BILINEAR)))
                n = np.stack(ch, axis=-1)
            # Re-normalize to unit vectors.
            norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
            n_arr = (n / norm).astype(np.float32)

        return (d, n_arr) if n_arr is not None else d


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
