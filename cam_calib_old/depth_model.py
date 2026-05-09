"""
Depth model inference thread for cam_calib_real.py.

Runs Depth Anything V2 Metric asynchronously in a daemon thread. The camera
thread pushes RGB frames into an input queue; this thread consumes the latest
frame, runs inference, and pushes metric depth maps (float32, metres) into an
output queue. Both queues are size-1 drop-on-overflow so the caller always gets
the freshest result and never blocks.

Usage
-----
    from depth_model import DepthModelThread, MODELS

    dm = DepthModelThread(model_key="v2-indoor-large", device="cuda")
    depth_q = dm.start(rgb_q)       # rgb_q: Queue[np.ndarray] (H,W,3 uint8)
    # ...
    depth = depth_q.get_nowait()    # np.ndarray float32 (H,W), metres
    dm.stop()

Model keys
----------
    v2-indoor-small   v2-indoor-base   v2-indoor-large
    v2-outdoor-small  v2-outdoor-base  v2-outdoor-large

Weights are downloaded on first use to weights/<model_key>/ next to this script.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np

WEIGHTS_BASE = Path(__file__).parent / "weights"

# All available Depth Anything V2 metric models
MODELS: dict[str, str] = {
    "v2-indoor-small":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "v2-indoor-base":    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "v2-indoor-large":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "v2-outdoor-small":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "v2-outdoor-base":   "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "v2-outdoor-large":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}

DEFAULT_MODEL = "v2-indoor-small"

# Legacy alias kept for backwards compatibility
WEIGHTS_DIR = WEIGHTS_BASE / DEFAULT_MODEL


# ── Download helper ───────────────────────────────────────────────────────────

def download_weights(model_key: str = DEFAULT_MODEL) -> Path:
    """Download model weights from HuggingFace Hub if not already present."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS)}")
    repo_id   = MODELS[model_key]
    local_dir = WEIGHTS_BASE / model_key
    marker    = local_dir / "model.safetensors"
    if marker.exists():
        print(f"[depth_model] Weights already at {local_dir}")
        return local_dir

    from huggingface_hub import snapshot_download
    print(f"[depth_model] Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[depth_model] Download complete.")
    return local_dir


# ── Depth model thread ────────────────────────────────────────────────────────

class DepthModelThread:
    """
    Async wrapper around the Depth Anything V2 transformers pipeline.

    The inference loop runs at whatever rate the GPU allows (~12 Hz on T600).
    It always pulls the *latest* RGB frame so it never falls behind.

    Output depth is float32, metres, same spatial size as the input RGB.
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS)}")
        self.model_key   = model_key
        self.weights_dir = WEIGHTS_BASE / model_key
        self.device      = device

        self._rgb_q:   Optional[Queue] = None   # set by start()
        self._depth_q: Queue           = Queue(maxsize=1)

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipe    = None   # transformers pipeline, loaded in thread

        self._infer_ms: float = 0.0   # exponential moving average of inference time

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, rgb_q: Queue) -> Queue:
        """
        Start the inference thread. Downloads weights if needed.

        Args:
            rgb_q: queue that the camera thread pushes np.ndarray (H,W,3 uint8) into.

        Returns:
            depth_q: queue from which callers can pop np.ndarray (H,W) float32 metres.
        """
        if not (self.weights_dir / "model.safetensors").exists():
            print(f"[depth_model] Weights not found — downloading {self.model_key}...")
            download_weights(self.model_key)

        self._rgb_q = rgb_q
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DepthModelThread")
        self._thread.start()
        return self._depth_q

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def infer_ms(self) -> float:
        """Smoothed inference latency in milliseconds."""
        return self._infer_ms

    # ── Inference loop ────────────────────────────────────────────────────────

    def _load_pipeline(self):
        import torch
        from transformers import pipeline as hf_pipeline

        print(f"[depth_model] Loading {self.model_key} from {self.weights_dir} on {self.device} ...")
        t0 = time.time()
        self._pipe = hf_pipeline(
            "depth-estimation",
            model=str(self.weights_dir),
            device=self.device,
        )
        print(f"[depth_model] Ready in {time.time()-t0:.1f}s")

        # Warmup — first inference is slow due to CUDA JIT
        from PIL import Image
        dummy = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        self._pipe(dummy)
        print("[depth_model] Warmup done")

    def _loop(self):
        self._load_pipeline()

        from PIL import Image
        import torch

        while self._running.is_set():
            # Drain to the latest frame
            rgb = None
            while True:
                try:
                    rgb = self._rgb_q.get_nowait()
                except Empty:
                    break

            if rgb is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            pil_img = Image.fromarray(rgb)
            out     = self._pipe(pil_img)

            # predicted_depth: torch.Tensor (H, W) float32, metric metres
            depth_m = out["predicted_depth"]
            if hasattr(depth_m, "cpu"):
                depth_m = depth_m.cpu()
            depth_np = depth_m.numpy().astype(np.float32)

            # Clamp to sane range (sensor usually valid 0.1–10 m)
            depth_np = np.clip(depth_np, 0.0, 10.0)

            ms = (time.time() - t0) * 1000.0
            self._infer_ms = 0.9 * self._infer_ms + 0.1 * ms

            self._put_latest(self._depth_q, depth_np)

    @staticmethod
    def _put_latest(q: Queue, item) -> None:
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Full:
                pass


# ── CLI: download weights ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download model weights to weights/ directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                        help=f"Model variant (default: {DEFAULT_MODEL})")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick inference test after loading")
    args = parser.parse_args()

    if args.download:
        download_weights(args.model)

    if args.test:
        import time
        from queue import Queue

        rgb_q = Queue(maxsize=1)
        dm    = DepthModelThread(model_key=args.model)
        depth_q = dm.start(rgb_q)

        # Feed dummy frames
        print("Feeding frames...")
        for i in range(3):
            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            try:
                rgb_q.put_nowait(rgb)
            except Full:
                rgb_q.get_nowait()
                rgb_q.put_nowait(rgb)
            time.sleep(0.5)
            try:
                depth = depth_q.get_nowait()
                print(f"  frame {i}: depth {depth.shape} range [{depth.min():.2f}, {depth.max():.2f}] m  ({dm.infer_ms:.0f}ms)")
            except Empty:
                print(f"  frame {i}: no depth yet")

        dm.stop()
        print("Test done.")
