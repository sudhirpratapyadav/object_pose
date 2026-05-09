"""
Collect camera calibration data from MuJoCo simulation.

Moves the Kinova Gen3 to a set of poses, captures RGB and depth images from
the external camera and wrist camera, and records the EE pose.

Two modes:
  • Pose-file mode  (--poses_file poses.yaml): iterate over the poses defined
    by configure_cam.py.  The robot is moved using the saved qpos directly.
  • Random mode     (default): sample random joint configurations.

Each run creates a timestamped folder under `data/`:
    data/20260411_153042/
        rgb_ext/    00000.png ...
        depth_ext/  00000.npy ...
        rgb_wrist/  00000.png ...
        depth_wrist/00000.npy ...
        metadata.json
        preview.mp4          ← 2x2 grid (ext-rgb | ext-depth / wrist-rgb | wrist-depth), if --video

Usage:
    uv run collect_data.py --poses_file poses.yaml [--settle_steps 500] [--video] [--fps 5]
    uv run collect_data.py [--n_samples 200] [--settle_steps 500] [--video] [--fps 5]
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco
import imageio.v3 as iio
import imageio
import yaml
from scipy.spatial.transform import Rotation


# Joint limits for the 7 arm joints (rad)
JOINT_LIMITS = np.array([
    [-6.28318,  6.28318],  # joint_1
    [-2.24,     2.24    ],  # joint_2
    [-6.28318,  6.28318],  # joint_3
    [-2.57,     2.57    ],  # joint_4
    [-6.28318,  6.28318],  # joint_5
    [-2.09,     2.09    ],  # joint_6
    [-6.28318,  6.28318],  # joint_7
])

# Home configuration (safe starting pose)
HOME_QPOS = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])


CAMERA_CONFIG_PATH = Path(__file__).parent / "camera_config.yaml"
EXT_CAM_NAME = "ext_rgbd"


def load_camera_config() -> dict:
    with open(CAMERA_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def apply_camera_config(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict) -> None:
    """Override ext_rgbd camera pose and fovy from config dict."""
    cam_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EXT_CAM_NAME)
    pos       = np.array(cfg["extrinsics"]["pos"], dtype=np.float64)
    euler_deg = np.array(cfg["extrinsics"]["euler_deg"], dtype=np.float64)
    rot       = Rotation.from_euler("xyz", euler_deg, degrees=True)
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]   # scipy xyzw -> mujoco wxyz
    # cam_quat + cam_pos drive cam_xmat/cam_xpos via mj_forward
    model.cam_pos[cam_id]  = pos
    model.cam_quat[cam_id] = quat_wxyz
    model.cam_fovy[cam_id] = cfg["intrinsics"]["fovy"]


def sample_random_joints(rng: np.random.Generator, margin: float = 0.1) -> np.ndarray:
    """Sample random joint angles within limits, with a safety margin."""
    lo = JOINT_LIMITS[:, 0] * (1 - margin)
    hi = JOINT_LIMITS[:, 1] * (1 - margin)
    return rng.uniform(lo, hi)


def render_camera(model: mujoco.MjModel, data: mujoco.MjData,
                  renderer: mujoco.Renderer, cam_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and depth images from named camera. Returns (rgb, depth)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()  # HxWx3, uint8

    # Depth pass
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth_raw = renderer.render().copy()  # HxW, float32, values in [0,1] (linearized)
    renderer.disable_depth_rendering()

    return rgb, depth_raw


def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Return EE position and quaternion (xyzw) from the pinch_site."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    pos = data.site_xpos[site_id].copy()          # (3,)
    # site_xmat is a 3x3 rotation matrix (row-major); convert to quaternion
    rot_mat = data.site_xmat[site_id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_mat.flatten())  # mujoco convention: w,x,y,z
    return {"pos": pos.tolist(), "quat_wxyz": quat.tolist()}


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert float32 depth map [0,1] to uint8 RGB using a colormap (viridis-like)."""
    # Normalize to [0, 255], clip outliers
    d = np.clip(depth, 0.0, 1.0)
    d_u8 = (d * 255).astype(np.uint8)
    # Apply a simple blue→yellow colormap manually (no matplotlib needed)
    r = d_u8
    g = (d_u8 * 0.8).astype(np.uint8)
    b = (255 - d_u8)
    return np.stack([r, g, b], axis=-1)


def make_grid_frame(rgb_ext: np.ndarray, depth_ext: np.ndarray,
                    rgb_wrist: np.ndarray, depth_wrist: np.ndarray) -> np.ndarray:
    """Build a 2x2 grid: [ext-rgb | ext-depth] / [wrist-rgb | wrist-depth]. Returns HxWx3 uint8."""
    d_ext   = depth_to_rgb(depth_ext)
    d_wrist = depth_to_rgb(depth_wrist)
    top    = np.concatenate([rgb_ext,   d_ext],   axis=1)
    bottom = np.concatenate([rgb_wrist, d_wrist], axis=1)
    return np.concatenate([top, bottom], axis=0)


def settle(model: mujoco.MjModel, data: mujoco.MjData, target_qpos: np.ndarray,
           n_steps: int) -> None:
    """Hold position setpoints and step the simulation to let it settle."""
    # Actuators 0-6 are position controllers for arm joints
    data.ctrl[:7] = target_qpos
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def load_poses_file(path: Path) -> list[dict]:
    """Load poses from a poses.yaml produced by configure_cam.py."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    poses = raw.get("poses", [])
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def collect(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).parent
    scene_xml  = script_dir / "scene.xml"

    # Determine targets: poses from file or random
    use_poses = args.poses_file is not None
    if use_poses:
        poses = load_poses_file(Path(args.poses_file))
        n_samples = len(poses)
        print(f"Pose-file mode: {n_samples} pose(s) from {args.poses_file}")
    else:
        n_samples = args.n_samples
        print(f"Random mode: {n_samples} samples (seed={args.seed})")

    # Timestamped run folder
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("data") / run_name

    dirs = {
        "rgb_ext":    run_dir / "rgb_ext",
        "depth_ext":  run_dir / "depth_ext",
        "rgb_wrist":  run_dir / "rgb_wrist",
        "depth_wrist": run_dir / "depth_wrist",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data  = mujoco.MjData(model)

    # Apply camera config (overrides scene.xml)
    cam_cfg = load_camera_config()
    apply_camera_config(model, data, cam_cfg)
    width  = cam_cfg["intrinsics"]["width"]
    height = cam_cfg["intrinsics"]["height"]
    print(f"Camera config: pos={cam_cfg['extrinsics']['pos']}  "
          f"euler={cam_cfg['extrinsics']['euler_deg']}  fovy={cam_cfg['intrinsics']['fovy']}")

    renderer = mujoco.Renderer(model, height=height, width=width)

    rng = np.random.default_rng(args.seed)
    metadata     = []
    video_frames = [] if args.video else None

    # Move to home first
    mujoco.mj_resetData(model, data)
    settle(model, data, HOME_QPOS, n_steps=args.settle_steps)

    print(f"Run: {run_name}")
    print(f"Collecting {n_samples} samples -> {run_dir}")
    if args.video:
        print(f"Video: preview.mp4 @ {args.fps} fps")
    t0 = time.time()

    for i in range(n_samples):
        if use_poses:
            # Move directly to saved qpos
            target = np.array(poses[i]["qpos"], dtype=np.float64)
        else:
            target = sample_random_joints(rng)

        settle(model, data, target, n_steps=args.settle_steps)

        actual_qpos = data.qpos[:7].copy().tolist()
        ee_pose     = get_ee_pose(model, data)

        fname = f"{i:05d}"

        # External camera
        rgb_ext, depth_ext = render_camera(model, data, renderer, "ext_rgbd")
        iio.imwrite(str(dirs["rgb_ext"]   / f"{fname}.png"), rgb_ext)
        np.save(str(dirs["depth_ext"]     / f"{fname}.npy"), depth_ext.astype(np.float32))

        # Wrist camera
        rgb_wrist, depth_wrist = render_camera(model, data, renderer, "wrist_cam")
        iio.imwrite(str(dirs["rgb_wrist"] / f"{fname}.png"), rgb_wrist)
        np.save(str(dirs["depth_wrist"]   / f"{fname}.npy"), depth_wrist.astype(np.float32))

        if args.video:
            video_frames.append(make_grid_frame(rgb_ext, depth_ext, rgb_wrist, depth_wrist))

        entry = {
            "id":            i,
            "target_qpos":   target.tolist(),
            "actual_qpos":   actual_qpos,
            "ee_pos":        ee_pose["pos"],
            "ee_quat_wxyz":  ee_pose["quat_wxyz"],
        }
        if use_poses:
            entry["pose_id"] = poses[i].get("id", i)
        metadata.append(entry)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/{n_samples}] {elapsed:.1f}s  ee={[f'{v:.3f}' for v in ee_pose['pos']]}")

    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"run": run_name, "n_samples": len(metadata), "samples": metadata}, f, indent=2)

    print(f"\nDone. {run_dir}")
    print(f"  rgb_ext/    {n_samples} PNGs")
    print(f"  depth_ext/  {n_samples} NPYs")
    print(f"  rgb_wrist/  {n_samples} PNGs")
    print(f"  depth_wrist/{n_samples} NPYs")
    print(f"  metadata.json")

    if args.video:
        video_path = run_dir / "preview.mp4"
        print(f"Writing video ({len(video_frames)} frames @ {args.fps} fps)...", end=" ", flush=True)
        with imageio.get_writer(str(video_path), fps=args.fps, codec="libx264",
                                quality=7, pixelformat="yuv420p") as writer:
            for frame in video_frames:
                writer.append_data(frame)
        print(f"done -> {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect camera calibration data from MuJoCo sim.")
    parser.add_argument("--poses_file",   type=str, default=None,
                        help="Path to poses.yaml from configure_cam.py. "
                             "If provided, uses predefined poses instead of random sampling.")
    parser.add_argument("--n_samples",    type=int, default=200,
                        help="Number of random configurations to sample (default: 200, ignored if --poses_file)")
    parser.add_argument("--settle_steps", type=int, default=500,
                        help="Simulation steps to settle at each pose (default: 500)")
    parser.add_argument("--seed",  type=int,  default=42,  help="Random seed (default: 42)")
    parser.add_argument("--video", action="store_true",    help="Write preview.mp4 (2x2 grid)")
    parser.add_argument("--fps",   type=int,  default=5,   help="Video FPS (default: 5)")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
