"""Apply CamCalib (OpenCV-frame) to a named camera in a loaded MjModel.

MuJoCo cameras follow OpenGL convention: the camera looks down its **-Z**
axis, with +X right and +Y up. Real cameras (and RealSense, OpenCV) use the
opposite Y/Z: +Z forward, +Y down. The patch flips Y and Z to convert.

MuJoCo also assumes the principal point is at the image centre. If the
calibration's cx/cy are off-centre, we log a warning and use centred values
in sim — meaning sim and real won't match exactly when calibration places
the principal point off-centre. A future patch could render oversized + crop
for true off-centre support.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from config import CamCalib, matrix_to_quat_wxyz


# OpenCV (Y down, Z fwd) -> MuJoCo (Y up, Z back): flip Y and Z columns of R.
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def mj_camera_params(calib: CamCalib) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a YAML calibration into MuJoCo camera params.

    Returns (pos[3], quat_wxyz[4], fovy_deg). Use this when you need the
    values to pass into another process; ``patch_mjcf_camera`` uses the same
    conversion when patching a model in place.
    """
    R_cv = calib.T_world_camera()[:3, :3]
    R_mj = R_cv @ _CV_TO_MJ
    quat = matrix_to_quat_wxyz(R_mj)
    pos  = np.asarray(calib.extrinsics.pos, dtype=np.float64)
    fovy = math.degrees(2.0 * math.atan(0.5 * calib.intrinsics.height
                                        / calib.intrinsics.fy))
    return pos, quat, fovy


def patch_mjcf_camera(model: mujoco.MjModel, calib: CamCalib,
                      camera_name: str = "ext_rgbd") -> None:
    """Patch a named camera in the loaded model with extrinsics, intrinsics, resolution.

    Mutates ``model`` in place. Must be called *before* MjData is created
    (well, actually MuJoCo recomputes xpos on mj_kinematics so it's fine to
    call after too; but for clarity do it as part of init).

    Raises ValueError if the camera name isn't found.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"camera '{camera_name}' not found in MJCF "
                         f"(available: {_list_camera_names(model)})")

    pos, quat, fovy = mj_camera_params(calib)
    model.cam_pos[cam_id]  = pos
    model.cam_quat[cam_id] = quat
    model.cam_fovy[cam_id] = fovy

    i = calib.intrinsics
    cx_center = 0.5 * i.width
    cy_center = 0.5 * i.height
    if abs(i.cx - cx_center) > 1.0 or abs(i.cy - cy_center) > 1.0:
        print(f"[sim] WARNING: principal point ({i.cx:.1f}, {i.cy:.1f}) is "
              f"off-centre vs ({cx_center:.1f}, {cy_center:.1f}); MuJoCo "
              f"cannot model this exactly, sim camera uses centred principal "
              f"point. Real-mode pipeline still uses your YAML cx/cy.",
              flush=True)
    model.cam_resolution[cam_id] = (int(i.width), int(i.height))


def _list_camera_names(model: mujoco.MjModel) -> list[str]:
    out = []
    for i in range(model.ncam):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        out.append(n or f"<cam_{i}>")
    return out
