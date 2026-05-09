# object_pose

Live RGB + monocular metric depth + 3D point cloud viewer.

Pipeline:

```
RealSense (thread)
    └── shm "rgb" ──► depth process (any backend)
                          ├── shm "depth"
                          ├── shm "pc_xyz" + "pc_rgb"
                          └── shm "mesh_*"
                                  └── web_server.py ──► WebSocket ──► browser (React + Three.js)
```

## Layout

| Path                            | What it does                                              |
| ------------------------------- | --------------------------------------------------------- |
| `camera/realsense.py`           | Threaded RealSense color stream, returns intrinsics       |
| `depth/runner.py`               | Worker process + shared-memory layout                     |
| `depth/backends/base.py`        | `DepthBackend` interface (`load`, `infer`)                |
| `depth/backends/hf_pipeline.py` | Wraps `transformers.pipeline("depth-estimation")`         |
| `depth/backends/unidepth.py`    | UniDepth V2 (predicts intrinsics + metric depth)          |
| `depth/backends/metric3d.py`    | Metric3D V2 (torch.hub)                                   |
| `segment/`                      | SAM2 worker — click-driven masks + 3D bbox                |
| `robot/`                        | MJCF loader, robot-state shm, server-side FK              |
| `hardware/`                     | Kinova Gen3 OSC torque loop (Pinocchio dynamics)          |
| `web_server.py`                 | WebSocket server bridging shm → browser                   |
| `web/`                          | React + Three.js frontend                                 |

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

## Calibration config

`cam_calib_config.yaml` at the repo root is required (see
`cam_calib_config.yaml.example` for a template). It is the **single source of
truth** for both real and sim modes — extrinsics + intrinsics + resolution.
The server hard-errors at startup if the file is missing.

In real mode the server requests the camera at the YAML resolution, dumps the
RealSense factory intrinsics to `cam_factory_intrinsics.yaml` (read-only
snapshot for reference), and uses the YAML intrinsics for the depth pipeline.
In sim mode the YAML values patch the named MJCF camera at runtime.

The browser has live extrinsics sliders that publish updates back to the
server; clicking *Save to YAML* persists them.

## Run

Real mode (RealSense + camera):

```bash
uv run web_server.py
```

Real mode + robot display (animated dummy joints):

```bash
uv run web_server.py --mjcf robot/mjcf/scene.xml --robot-source dummy
```

Sim mode (MuJoCo provides the camera + physics; robot is interactive):

```bash
uv run web_server.py --mode sim --mjcf robot/mjcf/scene.xml
# add --mujoco-gui to also open the native passive viewer window
```

Then open <http://localhost:5173> (Vite dev) — or run `npm run build` inside
`web/` to ship the static bundle.

Flags:

- `--mode {real,sim}` — `sim` swaps RealSense for a MuJoCo render of `--mjcf`.
- `--mjcf PATH` — load an MJCF for robot display (required in `sim` mode).
- `--robot-source {none,dummy,sim}` — `dummy` animates joints; `sim` is
  implied by `--mode sim`.
- `--sim-camera NAME` — which MJCF camera to render in sim mode
  (default `ext_rgbd`).
- `--mujoco-gui` — open MuJoCo's passive native viewer (sim mode only).

## Hardware

- Intel RealSense camera (color stream only).
- CUDA-capable GPU recommended.

## License

MIT — see `LICENSE`.
