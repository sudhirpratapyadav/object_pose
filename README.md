# object_pose

Live RGB + monocular metric depth + 3D point cloud viewer.

Pipeline:

```
RealSense (thread)
    └── shm "rgb" ──► depth process (any backend)
                          ├── shm "depth"
                          └── shm "pc_xyz" + "pc_rgb"
                                          └── viser viewer (main thread)
```

## Layout

| Path                          | What it does                                              |
| ----------------------------- | --------------------------------------------------------- |
| `camera/realsense.py`         | Threaded RealSense color stream, returns intrinsics       |
| `depth/runner.py`             | Worker process + shared-memory layout                     |
| `depth/backends/base.py`      | `DepthBackend` interface (`load`, `infer`)                |
| `depth/backends/hf_pipeline.py` | Wraps `transformers.pipeline("depth-estimation")`       |
| `depth/backends/unidepth.py`  | UniDepth V2 (predicts intrinsics + metric depth)          |
| `depth/backends/metric3d.py`  | Metric3D V2 (torch.hub)                                   |
| `viewer/server.py`            | Viser GUI: RGB / depth panels, 3D point cloud, frustum    |
| `detect.py`                   | Main entry point — wires camera → depth → viewer          |

## Models

The dropdown lists drop-in backends; weights download on first selection.

- **Depth Anything V2 — Metric** (Indoor / Outdoor × Small / Base / Large)
- **Apple DepthPro** — sharpest edges, ~2 GB weights
- **UniDepth V2 ViT-L/14** — predicts intrinsics + metric depth (extra install)
- **Metric3D V2 ViT-S / ViT-L** — top benchmarks; loaded via `torch.hub`

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

### HuggingFace auth (recommended)

Anonymous downloads from HuggingFace are aggressively rate-limited and can
**stall on large repos** (DepthPro, etc.). Set up a free token once:

```bash
huggingface-cli login
# paste a token from https://huggingface.co/settings/tokens (read access is fine)
```

The token is cached in `~/.cache/huggingface/token` and used automatically.

### Optional backends (UniDepth / Metric3D)

These have problematic build deps so they aren't installed by default. Add
them only if you want those dropdown entries to work:

```bash
# UniDepth V2
uv pip install --no-build-isolation \
  "unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git"

# Metric3D V2 — pulls code via torch.hub at runtime; mmcv may also be needed
uv pip install timm
```

## Run

```bash
uv run detect.py
```

Open <http://localhost:8080>. The GUI shows live RGB, model depth, a 3D point
cloud, the camera frustum + origin frame, FPS counters, and a depth-model
dropdown with download progress.

## Hardware

- Intel RealSense camera (color stream only).
- CUDA-capable GPU recommended.

## License

MIT — see `LICENSE`.
