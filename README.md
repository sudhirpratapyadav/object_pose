# object_pose

Live RGB + monocular metric depth + 3D point cloud viewer.

Pipeline:

```
RealSense (thread)
    └── shm "rgb" ──► depth process (Depth-Anything-V2-Metric-Indoor-Small)
                          ├── shm "depth"
                          └── shm "pc_xyz" + "pc_rgb"
                                          └── viser viewer (main thread)
```

## Layout

| File         | What it does                                              |
| ------------ | --------------------------------------------------------- |
| `camera.py`  | Threaded RealSense color stream, returns intrinsics       |
| `depth.py`   | Multiprocessing depth worker + shared-memory layout       |
| `viewer.py`  | Viser GUI: RGB / depth panels, 3D point cloud, frustum    |
| `detect.py`  | Main entry point — wires camera → depth → viewer          |

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

The Depth Anything V2 weights live in `weights/v2-indoor-small/`. If absent
they are downloaded from HuggingFace on first run.

## Run

```bash
uv run detect.py
```

Open <http://localhost:8080>. The GUI shows live RGB, model depth, a 3D point
cloud, the camera frustum + origin frame, and FPS counters.

## Hardware

- Intel RealSense camera (color stream only).
- CUDA-capable GPU recommended (depth model runs ~5 fps on a small GPU,
  faster on bigger ones).

## License

MIT — see `LICENSE`.
