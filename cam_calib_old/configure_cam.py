"""
Interactive external camera configurator + EE pose collector + camera calibrator.

Loads the MuJoCo scene, lets you:
  1. Adjust the external RGBD camera's extrinsics/intrinsics via sliders/gizmo.
  2. Move the robot EE to a desired pose via sliders or a draggable gizmo.
  3. Save the current EE pose + RGB image + 3D checkerboard keypoints to
     data/ground_truth/pose_<N>/ with "Save Pose" button.
  4. Run OpenCV camera calibration with "Calibrate Camera" button; result is
     written to calibration_result.yaml.

Usage:
    uv run configure_cam.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import viser
import yaml
from scipy.spatial.transform import Rotation

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
SCENE_XML   = SCRIPT_DIR / "scene.xml"
CAM_CFG     = SCRIPT_DIR / "camera_config.yaml"
POSES_FILE  = SCRIPT_DIR / "poses.yaml"
EXT_CAM     = "ext_rgbd"
WRIST_CAM   = "wrist_cam"
EE_SITE     = "pinch_site"
CB_SITE     = "checkerboard_origin"   # site name in scene.xml

# ── Checkerboard parameters ────────────────────────────────────────────────────
# Inner corners (cols, rows) — same convention as cv2.findChessboardCorners
CB_COLS        = 4       # number of inner corners along the horizontal axis (5-square board)
CB_ROWS        = 3       # number of inner corners along the vertical axis  (4-square board)
CB_SQUARE_MM   = 45.0    # physical size of one square in millimetres

# ── Data / output paths ────────────────────────────────────────────────────────
DATA_DIR       = SCRIPT_DIR / "data"
GT_DIR         = DATA_DIR / "ground_truth"
CALIB_OUT      = SCRIPT_DIR / "calibration_result.yaml"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from viewer import ViserMujocoScene

# ── Defaults ───────────────────────────────────────────────────────────────────
HOME_QPOS = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])
VIZ_HZ    = 10   # preview refresh rate

# IK parameters
IK_STEPS      = 200    # gradient steps per IK solve
IK_ALPHA      = 0.5    # step size
IK_DAMPING    = 1e-4   # damped least-squares regularization
SETTLE_STEPS  = 500    # sim steps to settle after IK


# ── Camera helpers ─────────────────────────────────────────────────────────────

def load_camera_config() -> dict:
    if CAM_CFG.exists():
        with open(CAM_CFG) as f:
            return yaml.safe_load(f)
    return {
        "extrinsics": {"pos": [1.2, 0.0, 1.0], "euler_deg": [120.0, 0.0, 90.0]},
        "intrinsics": {"fovy": 60.0, "width": 640, "height": 480},
    }


def save_camera_config(pos, euler_deg, fovy, width, height) -> None:
    cfg = {
        "extrinsics": {
            "pos":       [round(float(v), 4) for v in pos],
            "euler_deg": [round(float(v), 4) for v in euler_deg],
        },
        "intrinsics": {
            "fovy":   round(float(fovy), 2),
            "width":  int(width),
            "height": int(height),
        },
    }
    with open(CAM_CFG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    print(f"Saved camera config -> {CAM_CFG}")


def apply_camera(model: mujoco.MjModel, data: mujoco.MjData, pos, euler_deg, fovy) -> None:
    cam_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EXT_CAM)
    rot       = Rotation.from_euler("xyz", euler_deg, degrees=True)
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]
    model.cam_pos[cam_id]  = np.array(pos)
    model.cam_quat[cam_id] = quat_wxyz
    model.cam_fovy[cam_id] = fovy
    mujoco.mj_forward(model, data)


def fovy_to_intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Build a 3×3 camera matrix from MuJoCo fovy (vertical FOV in degrees)."""
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy   # MuJoCo assumes square pixels
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ── Render helpers ─────────────────────────────────────────────────────────────

def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    d = np.clip(depth, 0.0, 1.0)
    d_u8 = (d * 255).astype(np.uint8)
    return np.stack([d_u8, (d_u8 * 0.8).astype(np.uint8), 255 - d_u8], axis=-1)


def make_grid(rgb_ext, depth_ext, rgb_wrist, depth_wrist) -> np.ndarray:
    top    = np.concatenate([rgb_ext,   depth_to_rgb(depth_ext)],   axis=1)
    bottom = np.concatenate([rgb_wrist, depth_to_rgb(depth_wrist)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def render_cam(renderer: mujoco.Renderer, model: mujoco.MjModel,
               data: mujoco.MjData, cam_name: str):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


# ── IK helpers ─────────────────────────────────────────────────────────────────

def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return current EE (pos [3], quat_wxyz [4]) from ee_site."""
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    pos      = data.site_xpos[site_id].copy()
    rot_mat  = data.site_xmat[site_id].reshape(3, 3)
    quat     = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_mat.flatten())
    return pos, quat   # quat is wxyz


def ik_solve(model: mujoco.MjModel, data: mujoco.MjData,
             target_pos: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Damped-least-squares 6-DOF IK for the 7-DOF arm.
    Tracks both position and orientation of EE_SITE.
    Modifies data.qpos in-place; returns the resulting qpos[:7].
    """
    site_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    nv         = model.nv
    target_rot = Rotation.from_quat(target_quat_wxyz[[1, 2, 3, 0]]).as_matrix()

    for _ in range(IK_STEPS):
        mujoco.mj_forward(model, data)

        pos_err = target_pos - data.site_xpos[site_id]
        cur_rot = data.site_xmat[site_id].reshape(3, 3)
        ori_err = Rotation.from_matrix(target_rot @ cur_rot.T).as_rotvec()

        err = np.concatenate([pos_err, ori_err])   # (6,)
        if np.linalg.norm(err) < 1e-4:
            break

        jac_pos = np.zeros((3, nv))
        jac_rot = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        J = np.vstack([jac_pos, jac_rot])   # (6, nv)

        dq_full = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), err)
        data.qpos[:7] += IK_ALPHA * dq_full[:7]

        for j in range(7):
            lo, hi = model.jnt_range[j]
            if lo < hi:
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)

    mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()


def settle(model: mujoco.MjModel, data: mujoco.MjData, qpos7: np.ndarray) -> None:
    data.ctrl[:7] = qpos7
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


# ── Poses file helpers ─────────────────────────────────────────────────────────

def load_poses() -> list[dict]:
    if POSES_FILE.exists():
        with open(POSES_FILE) as f:
            raw = yaml.safe_load(f) or {}
        return raw.get("poses", [])
    return []


def save_poses(poses: list[dict]) -> None:
    with open(POSES_FILE, "w") as f:
        yaml.dump({"poses": poses}, f, default_flow_style=None, sort_keys=False)
    print(f"Saved {len(poses)} pose(s) -> {POSES_FILE}")


# ── Checkerboard 3-D keypoint helpers ─────────────────────────────────────────

def has_cb_site(model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CB_SITE) != -1


def get_cb_origin_pose(model: mujoco.MjModel,
                       data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return checkerboard origin site world pose: (pos [3], rot_mat [3,3])."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CB_SITE)
    pos     = data.site_xpos[site_id].copy()
    rot_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot_mat


def checkerboard_3d_points(origin_pos: np.ndarray,
                           origin_rot: np.ndarray) -> np.ndarray:
    """
    Compute world-frame 3-D positions of all CB_ROWS×CB_COLS inner corners.

    Convention (OpenCV):
      - Corner (0,0) = checkerboard_origin site position.
      - Corners advance along the site's local +X axis (columns) and
        local +Y axis (rows).
      - The board lies in the site's local XY plane (Z = 0 in local frame).

    Returns float32 array of shape (CB_ROWS*CB_COLS, 3).
    """
    sq = CB_SQUARE_MM * 1e-3   # convert mm -> metres
    pts_world = []
    for r in range(CB_ROWS):
        for c in range(CB_COLS):
            local = np.array([c * sq, r * sq, 0.0])
            world = origin_pos + origin_rot @ local
            pts_world.append(world)
    return np.array(pts_world, dtype=np.float32)


# ── Per-pose data saving ───────────────────────────────────────────────────────

def save_pose_data(pose_id: int,
                   rgb_ext: np.ndarray,
                   depth_ext: np.ndarray,
                   pts3d: np.ndarray,
                   pose_entry: dict) -> Path:
    """
    Save everything for one calibration pose into GT_DIR/pose_<id>/:
      rgb.png        — ext camera RGB image
      depth.npy      — ext camera depth map (float32, metres)
      keypoints_3d.npy  — (N,3) world-frame checkerboard corners
      pose.yaml      — EE pose + qpos + checkerboard origin
    Returns the pose directory path.
    """
    pose_dir = GT_DIR / f"pose_{pose_id:04d}"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # RGB: MuJoCo renders RGB; OpenCV expects BGR for imwrite
    cv2.imwrite(str(pose_dir / "rgb.png"), cv2.cvtColor(rgb_ext, cv2.COLOR_RGB2BGR))

    np.save(str(pose_dir / "depth.npy"), depth_ext.astype(np.float32))
    np.save(str(pose_dir / "keypoints_3d.npy"), pts3d)

    with open(pose_dir / "pose.yaml", "w") as f:
        yaml.dump(pose_entry, f, default_flow_style=None, sort_keys=False)

    print(f"  Saved data for pose {pose_id} -> {pose_dir}")
    return pose_dir


# ── Calibration ────────────────────────────────────────────────────────────────

def run_calibration(width: int, height: int) -> str:
    """
    Load all saved pose data from GT_DIR, detect 2-D checkerboard corners in
    each RGB image, then run cv2.calibrateCamera to recover intrinsics and
    extrinsics.  Results are written to CALIB_OUT.

    Returns a short status string for the GUI.
    """
    pose_dirs = sorted(GT_DIR.glob("pose_*"))
    if not pose_dirs:
        return "No pose data found in data/ground_truth/"

    obj_points_all: list[np.ndarray] = []   # 3-D world points per image
    img_points_all: list[np.ndarray] = []   # 2-D detected corners per image
    used_poses: list[str]            = []

    cb_pattern = (CB_COLS, CB_ROWS)   # (cols, rows) for findChessboardCorners

    for pose_dir in pose_dirs:
        rgb_path = pose_dir / "rgb.png"
        kp_path  = pose_dir / "keypoints_3d.npy"
        if not rgb_path.exists() or not kp_path.exists():
            print(f"  Skipping {pose_dir.name}: missing rgb.png or keypoints_3d.npy")
            continue

        img_bgr = cv2.imread(str(rgb_path))
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, cb_pattern, None)
        if not ret:
            print(f"  Skipping {pose_dir.name}: checkerboard not detected")
            continue

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        pts3d = np.load(str(kp_path))   # (N, 3) float32
        obj_points_all.append(pts3d)
        img_points_all.append(corners)
        used_poses.append(pose_dir.name)

    n_good = len(obj_points_all)
    if n_good < 4:
        return f"Only {n_good} usable pose(s); need at least 4 for calibration"

    print(f"  Running calibrateCamera on {n_good} poses ...")

    # Initial guess from current fovy
    cfg    = load_camera_config()
    K_init = fovy_to_intrinsics(cfg["intrinsics"]["fovy"], width, height)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all,
        img_points_all,
        (width, height),
        K_init.copy(),
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    # ── Compute mean extrinsics (camera pose in world frame) ──────────────
    # Each rvec/tvec is the transform that maps world points into camera frame.
    # We average the camera-in-world translations for a summary.
    cam_positions = []
    extrinsics_per_pose = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rv)
        t    = tv.flatten()
        # Camera position in world: -R^T @ t
        cam_pos_world = (-R.T @ t).tolist()
        cam_positions.append(cam_pos_world)
        extrinsics_per_pose.append({
            "pose":    used_poses[i],
            "rvec":    rv.flatten().tolist(),
            "tvec":    t.tolist(),
            "cam_pos_world": cam_pos_world,
        })

    result = {
        "calibration": {
            "rms_reprojection_error_px": float(rms),
            "n_poses_used":              n_good,
            "poses_used":                used_poses,
        },
        "intrinsics": {
            "fx":   float(K[0, 0]),
            "fy":   float(K[1, 1]),
            "cx":   float(K[0, 2]),
            "cy":   float(K[1, 2]),
            "dist": dist.flatten().tolist(),
            "K":    K.tolist(),
        },
        "extrinsics_per_pose": extrinsics_per_pose,
        "checkerboard": {
            "cols":       CB_COLS,
            "rows":       CB_ROWS,
            "square_mm":  CB_SQUARE_MM,
        },
    }

    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_OUT, "w") as f:
        yaml.dump(result, f, default_flow_style=None, sort_keys=False)
    print(f"  Calibration result -> {CALIB_OUT}")

    return f"RMS={rms:.3f}px  n={n_good}  -> {CALIB_OUT.name}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = load_camera_config()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    # Start at home pose
    data.qpos[:7] = HOME_QPOS.copy()
    data.ctrl[:7] = HOME_QPOS.copy()
    mujoco.mj_forward(model, data)

    width  = cfg["intrinsics"]["width"]
    height = cfg["intrinsics"]["height"]
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Apply loaded camera config
    init_pos   = list(cfg["extrinsics"]["pos"])
    init_euler = list(cfg["extrinsics"]["euler_deg"])
    init_fovy  = cfg["intrinsics"]["fovy"]
    apply_camera(model, data, init_pos, init_euler, init_fovy)

    # Initial EE pose
    init_ee_pos, init_ee_quat = get_ee_pose(model, data)
    init_ee_euler = Rotation.from_quat(init_ee_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)

    # Saved poses list (mutable, shared)
    saved_poses: list[dict] = load_poses()

    # Keep last rendered ext-cam RGB so Save Pose can capture it
    _last_rgb_ext: list[np.ndarray]   = [np.zeros((height, width, 3), dtype=np.uint8)]
    _last_depth_ext: list[np.ndarray] = [np.zeros((height, width), dtype=np.float32)]

    # ── Viser server ──────────────────────────────────────────────────────
    server = viser.ViserServer(label="Camera & Pose Configurator")
    scene  = ViserMujocoScene.create(server, model)
    robot  = scene.add_robot("robot", color=(0.75, 0.75, 0.75, 1.0))
    scene.create_visualization_gui(camera_distance=1.5, camera_azimuth=135.0, camera_elevation=30.0)
    robot.update(data)

    # ── Checkerboard corner markers (one icosphere per corner) ────────────
    cb_origin0, cb_rot0 = get_cb_origin_pose(model, data)
    pts0 = checkerboard_3d_points(cb_origin0, cb_rot0)
    cb_markers = []
    for idx in range(CB_ROWS * CB_COLS):
        # colour: corner (0,0) is red (origin), rest are yellow
        color = (1.0, 0.0, 0.0) if idx == 0 else (1.0, 1.0, 0.0)
        handle = server.scene.add_icosphere(
            f"/cb_corner/{idx}",
            radius=0.004,
            position=tuple(float(v) for v in pts0[idx]),
            color=color,
        )
        cb_markers.append(handle)

    # Axis frame at checkerboard origin showing its pose
    cb_wxyz0 = Rotation.from_matrix(cb_rot0).as_quat()[[3, 0, 1, 2]]
    cb_frame = server.scene.add_frame(
        "/cb_origin_frame",
        position=tuple(float(v) for v in cb_origin0),
        wxyz=tuple(float(v) for v in cb_wxyz0),
        axes_length=0.05,
        axes_radius=0.003,
    )

    _lock     = threading.Lock()
    _dirty    = [True]    # camera params changed
    _ee_dirty = [False]   # EE target changed, need IK re-solve

    cam_state = {
        "pos":       list(init_pos),
        "euler_deg": list(init_euler),
        "fovy":      init_fovy,
    }

    ee_state = {
        "pos":       init_ee_pos.tolist(),
        "euler_deg": init_ee_euler.tolist(),
    }

    # ── Camera GUI ────────────────────────────────────────────────────────
    with server.gui.add_folder("Camera Extrinsics (position)"):
        sl_px = server.gui.add_slider("x (m)",  min=-3.0, max=3.0, step=0.01, initial_value=init_pos[0])
        sl_py = server.gui.add_slider("y (m)",  min=-3.0, max=3.0, step=0.01, initial_value=init_pos[1])
        sl_pz = server.gui.add_slider("z (m)",  min= 0.0, max=3.0, step=0.01, initial_value=init_pos[2])

    with server.gui.add_folder("Camera Extrinsics (orientation euler XYZ deg)"):
        sl_rx = server.gui.add_slider("rx (°)", min=-180.0, max=180.0, step=0.5, initial_value=init_euler[0])
        sl_ry = server.gui.add_slider("ry (°)", min=-180.0, max=180.0, step=0.5, initial_value=init_euler[1])
        sl_rz = server.gui.add_slider("rz (°)", min=-180.0, max=180.0, step=0.5, initial_value=init_euler[2])

    with server.gui.add_folder("Camera Intrinsics"):
        sl_fovy = server.gui.add_slider("fovy (°)", min=10.0, max=150.0, step=0.5, initial_value=init_fovy)

    with server.gui.add_folder("Camera preview"):
        img_handle = server.gui.add_image(
            image=np.zeros((height * 2, width * 2, 3), dtype=np.uint8),
            label="ext-rgb | ext-depth / wrist-rgb | wrist-depth",
        )

    save_cam_btn   = server.gui.add_button("💾  Save Camera Config")
    txt_cam_status = server.gui.add_text("Camera Status", initial_value="Loaded from camera_config.yaml")

    @save_cam_btn.on_click
    def _save_cam(_):
        with _lock:
            pos       = list(cam_state["pos"])
            euler_deg = list(cam_state["euler_deg"])
            fovy      = cam_state["fovy"]
        save_camera_config(pos, euler_deg, fovy, width, height)
        txt_cam_status.value = f"Saved  pos={[round(v,3) for v in pos]}  fovy={fovy:.1f}°"

    # Camera gizmo
    init_cam_rot  = Rotation.from_euler("xyz", init_euler, degrees=True)
    init_cam_wxyz = init_cam_rot.as_quat()[[3, 0, 1, 2]]

    cam_gizmo = server.scene.add_transform_controls(
        "/ext_camera",
        scale=0.25,
        position=tuple(float(v) for v in init_pos),
        wxyz=tuple(float(v) for v in init_cam_wxyz),
    )

    def _sliders_to_cam_gizmo():
        rot  = Rotation.from_euler("xyz", cam_state["euler_deg"], degrees=True)
        wxyz = rot.as_quat()[[3, 0, 1, 2]]
        cam_gizmo.position = tuple(float(v) for v in cam_state["pos"])
        cam_gizmo.wxyz     = tuple(float(v) for v in wxyz)

    def _cam_gizmo_to_sliders():
        pos   = list(cam_gizmo.position)
        wxyz  = np.array(cam_gizmo.wxyz)
        euler = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_euler("xyz", degrees=True).tolist()
        with _lock:
            cam_state["pos"]       = pos
            cam_state["euler_deg"] = euler
            _dirty[0]              = True
        sl_px.value = pos[0]; sl_py.value = pos[1]; sl_pz.value = pos[2]
        sl_rx.value = euler[0]; sl_ry.value = euler[1]; sl_rz.value = euler[2]

    @cam_gizmo.on_update
    def _on_cam_gizmo(_):
        _cam_gizmo_to_sliders()

    def _on_cam_slider(_):
        with _lock:
            cam_state["pos"]       = [sl_px.value, sl_py.value, sl_pz.value]
            cam_state["euler_deg"] = [sl_rx.value, sl_ry.value, sl_rz.value]
            cam_state["fovy"]      = sl_fovy.value
            _dirty[0]              = True
        _sliders_to_cam_gizmo()

    for sl in (sl_px, sl_py, sl_pz, sl_rx, sl_ry, sl_rz, sl_fovy):
        sl.on_update(_on_cam_slider)

    # ── EE Target GUI ─────────────────────────────────────────────────────
    with server.gui.add_folder("EE Target Position (m)"):
        sl_ee_x = server.gui.add_slider("x", min=-1.0, max=1.0, step=0.005, initial_value=init_ee_pos[0])
        sl_ee_y = server.gui.add_slider("y", min=-1.0, max=1.0, step=0.005, initial_value=init_ee_pos[1])
        sl_ee_z = server.gui.add_slider("z", min= 0.0, max=1.5, step=0.005, initial_value=init_ee_pos[2])

    with server.gui.add_folder("EE Target Orientation (euler XYZ deg)"):
        sl_ee_rx = server.gui.add_slider("rx (°)", min=-180.0, max=180.0, step=1.0, initial_value=init_ee_euler[0])
        sl_ee_ry = server.gui.add_slider("ry (°)", min=-180.0, max=180.0, step=1.0, initial_value=init_ee_euler[1])
        sl_ee_rz = server.gui.add_slider("rz (°)", min=-180.0, max=180.0, step=1.0, initial_value=init_ee_euler[2])

    # EE gizmo
    init_ee_wxyz = Rotation.from_euler("xyz", init_ee_euler, degrees=True).as_quat()[[3, 0, 1, 2]]
    ee_gizmo = server.scene.add_transform_controls(
        "/ee_target",
        scale=0.15,
        position=tuple(float(v) for v in init_ee_pos),
        wxyz=tuple(float(v) for v in init_ee_wxyz),
    )

    def _sliders_to_ee_gizmo():
        wxyz = Rotation.from_euler("xyz", ee_state["euler_deg"], degrees=True).as_quat()[[3, 0, 1, 2]]
        ee_gizmo.position = tuple(float(v) for v in ee_state["pos"])
        ee_gizmo.wxyz     = tuple(float(v) for v in wxyz)

    def _ee_gizmo_to_sliders():
        pos   = list(ee_gizmo.position)
        wxyz  = np.array(ee_gizmo.wxyz)
        euler = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_euler("xyz", degrees=True).tolist()
        with _lock:
            ee_state["pos"]       = pos
            ee_state["euler_deg"] = euler
            _ee_dirty[0]          = True
        sl_ee_x.value  = pos[0]; sl_ee_y.value  = pos[1]; sl_ee_z.value  = pos[2]
        sl_ee_rx.value = euler[0]; sl_ee_ry.value = euler[1]; sl_ee_rz.value = euler[2]

    @ee_gizmo.on_update
    def _on_ee_gizmo(_):
        _ee_gizmo_to_sliders()

    def _on_ee_slider(_):
        with _lock:
            ee_state["pos"]       = [sl_ee_x.value, sl_ee_y.value, sl_ee_z.value]
            ee_state["euler_deg"] = [sl_ee_rx.value, sl_ee_ry.value, sl_ee_rz.value]
            _ee_dirty[0]          = True
        _sliders_to_ee_gizmo()

    for sl in (sl_ee_x, sl_ee_y, sl_ee_z, sl_ee_rx, sl_ee_ry, sl_ee_rz):
        sl.on_update(_on_ee_slider)

    # ── Pose management GUI ───────────────────────────────────────────────
    n_existing = len(list(GT_DIR.glob("pose_*"))) if GT_DIR.exists() else 0
    txt_pose_count  = server.gui.add_text("Saved poses", initial_value=f"{n_existing} pose(s) in data/ground_truth/")
    txt_ee_status   = server.gui.add_text("EE Status",   initial_value="Home pose")

    save_pose_btn   = server.gui.add_button("📍  Save Current EE Pose")
    clear_poses_btn = server.gui.add_button("🗑️  Clear All Poses")
    go_home_btn     = server.gui.add_button("🏠  Go Home")

    @save_pose_btn.on_click
    def _save_pose(_):
        with _lock:
            pos_now, quat_now  = get_ee_pose(model, data)
            euler_now          = Rotation.from_quat(quat_now[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
            qpos_now           = data.qpos[:7].copy().tolist()
            cb_origin, cb_rot  = get_cb_origin_pose(model, data)
            pts3d              = checkerboard_3d_points(cb_origin, cb_rot)
            rgb_snap           = _last_rgb_ext[0].copy()
            depth_snap         = _last_depth_ext[0].copy()

        pose_id = len(list(GT_DIR.glob("pose_*"))) if GT_DIR.exists() else 0

        cb_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cb_quat, cb_rot.flatten())

        pose_entry = {
            "id":              pose_id,
            "ee_pos":          [round(float(v), 5) for v in pos_now],
            "ee_euler_deg":    [round(float(v), 3) for v in euler_now],
            "ee_quat_wxyz":    [round(float(v), 6) for v in quat_now],
            "qpos":            [round(float(v), 6) for v in qpos_now],
            "cb_origin_pos":   [round(float(v), 6) for v in cb_origin],
            "cb_origin_quat_wxyz": [round(float(v), 6) for v in cb_quat],
            "checkerboard": {
                "cols":      CB_COLS,
                "rows":      CB_ROWS,
                "square_mm": CB_SQUARE_MM,
            },
        }

        save_pose_data(pose_id, rgb_snap, depth_snap, pts3d, pose_entry)

        # Also keep the lightweight poses.yaml index
        saved_poses.append(pose_entry)
        save_poses(saved_poses)

        n_saved = len(list(GT_DIR.glob("pose_*")))
        txt_pose_count.value = f"{n_saved} pose(s) in data/ground_truth/"
        txt_ee_status.value  = f"Pose #{pose_id} saved  pos={[round(v,3) for v in pos_now]}"
        print(f"  Saved pose #{pose_id}: pos={pose_entry['ee_pos']}")

    @clear_poses_btn.on_click
    def _clear_poses(_):
        import shutil
        if GT_DIR.exists():
            shutil.rmtree(GT_DIR)
        GT_DIR.mkdir(parents=True, exist_ok=True)
        saved_poses.clear()
        save_poses(saved_poses)
        txt_pose_count.value = "0 pose(s) in data/ground_truth/"
        txt_ee_status.value  = "All poses cleared"
        print("Cleared all poses.")

    @go_home_btn.on_click
    def _go_home(_):
        with _lock:
            data.qpos[:7] = HOME_QPOS.copy()
            data.ctrl[:7] = HOME_QPOS.copy()
            mujoco.mj_forward(model, data)
            pos_h, quat_h = get_ee_pose(model, data)
            euler_h = Rotation.from_quat(quat_h[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
            ee_state["pos"]       = pos_h.tolist()
            ee_state["euler_deg"] = euler_h.tolist()
            _dirty[0]             = True

        sl_ee_x.value  = pos_h[0]; sl_ee_y.value  = pos_h[1]; sl_ee_z.value  = pos_h[2]
        sl_ee_rx.value = euler_h[0]; sl_ee_ry.value = euler_h[1]; sl_ee_rz.value = euler_h[2]
        _sliders_to_ee_gizmo()
        robot.update(data)
        txt_ee_status.value = "Home pose"

    # ── Calibration GUI ───────────────────────────────────────────────────
    calibrate_btn    = server.gui.add_button("📷  Calibrate Camera")
    txt_calib_status = server.gui.add_text("Calibration Status", initial_value="Not run yet")

    @calibrate_btn.on_click
    def _calibrate(_):
        txt_calib_status.value = "Running calibration ..."
        with _lock:
            w = width
            h = height
        status = run_calibration(w, h)
        txt_calib_status.value = status

    # ── Viz loop ──────────────────────────────────────────────────────────
    print("Viser running — open http://localhost:8080 in your browser.")
    print("  • Drag EE gizmo or use sliders to position the arm.")
    print("  • Click 'Save Current EE Pose' to save pose + image + 3D keypoints.")
    print("  • Click 'Calibrate Camera' to run OpenCV calibration.")
    print("  • Click 'Save Camera Config' to update camera_config.yaml.")
    print("Ctrl+C to quit.\n")

    period = 1.0 / VIZ_HZ
    try:
        while True:
            t0 = time.time()

            with _lock:
                cam_dirty  = _dirty[0]
                ee_dirty   = _ee_dirty[0]
                pos        = list(cam_state["pos"])
                euler_deg  = list(cam_state["euler_deg"])
                fovy       = cam_state["fovy"]
                ee_pos_tgt = list(ee_state["pos"])
                ee_eul_tgt = list(ee_state["euler_deg"])
                if cam_dirty:
                    _dirty[0]    = False
                if ee_dirty:
                    _ee_dirty[0] = False

            if ee_dirty:
                target_quat_xyzw = Rotation.from_euler("xyz", ee_eul_tgt, degrees=True).as_quat()
                target_quat_wxyz = target_quat_xyzw[[3, 0, 1, 2]]
                target_pos       = np.array(ee_pos_tgt)

                qpos7 = ik_solve(model, data, target_pos, target_quat_wxyz)
                settle(model, data, qpos7)

                pos_actual, _ = get_ee_pose(model, data)
                pos_err = np.linalg.norm(pos_actual - target_pos)
                txt_ee_status.value = (
                    f"pos={[round(v,3) for v in pos_actual]}  "
                    f"err={pos_err*1000:.1f}mm"
                )
                robot.update(data)
                cam_dirty = True   # re-render camera view with new robot pose

                # Update checkerboard corner markers and origin frame
                cb_o, cb_r = get_cb_origin_pose(model, data)
                pts = checkerboard_3d_points(cb_o, cb_r)
                for idx, handle in enumerate(cb_markers):
                    handle.position = tuple(float(v) for v in pts[idx])
                cb_wxyz = Rotation.from_matrix(cb_r).as_quat()[[3, 0, 1, 2]]
                cb_frame.position = tuple(float(v) for v in cb_o)
                cb_frame.wxyz     = tuple(float(v) for v in cb_wxyz)

            if cam_dirty:
                apply_camera(model, data, pos, euler_deg, fovy)

                rgb_ext,   depth_ext   = render_cam(renderer, model, data, EXT_CAM)
                rgb_wrist, depth_wrist = render_cam(renderer, model, data, WRIST_CAM)
                img_handle.image = make_grid(rgb_ext, depth_ext, rgb_wrist, depth_wrist)

                # Cache latest ext-cam frames for pose saving
                with _lock:
                    _last_rgb_ext[0]   = rgb_ext
                    _last_depth_ext[0] = depth_ext

                txt_cam_status.value = (
                    f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                    f"euler=[{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}]°  "
                    f"fovy={fovy:.1f}°"
                )

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
