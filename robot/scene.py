"""Load an MJCF and extract a static description suitable for streaming.

Produces:

  - bodies: list of dicts (id, name, parent) — kinematic tree
  - geoms:  list of dicts (body_id, type, local pos/quat, size, color, mesh data)

Visual geoms (group 0–2) are kept; collision-only geoms (group 3) are dropped.
Mesh geometry is emitted as a flat (verts, faces) pair in MuJoCo's body-local
frame. Primitive geoms (box / sphere / capsule / cylinder / ellipsoid) carry
their size + local transform; the browser renders them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


# Visual group threshold: group <= this is rendered, group > this is skipped.
# Default mujoco visual is group 2; collision is group 3.
VISUAL_GROUP_MAX = 2

# Geom type codes mirrored from mjtGeom.
GEOM_PLANE     = int(mujoco.mjtGeom.mjGEOM_PLANE)
GEOM_SPHERE    = int(mujoco.mjtGeom.mjGEOM_SPHERE)
GEOM_CAPSULE   = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
GEOM_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
GEOM_CYLINDER  = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
GEOM_BOX       = int(mujoco.mjtGeom.mjGEOM_BOX)
GEOM_MESH      = int(mujoco.mjtGeom.mjGEOM_MESH)


@dataclass
class BodyInfo:
    id: int
    name: str
    parent: int  # body id; world is its own parent (id 0)


@dataclass
class GeomInfo:
    body_id: int
    type: int
    local_pos: np.ndarray   # (3,) f32
    local_quat: np.ndarray  # (4,) f32, wxyz
    size: np.ndarray        # (3,) f32
    color: np.ndarray       # (4,) f32 rgba [0..1]
    # mesh fields (only populated for type == GEOM_MESH)
    mesh_verts: np.ndarray | None = None  # (V, 3) f32
    mesh_faces: np.ndarray | None = None  # (F, 3) i32


@dataclass
class ActuatorInfo:
    name: str
    ctrl_min: float
    ctrl_max: float
    home_ctrl: float        # initial ctrl from the 'home' keyframe (or 0)


@dataclass
class RobotScene:
    mjcf_path: Path
    model: mujoco.MjModel
    bodies: list[BodyInfo]
    geoms: list[GeomInfo]
    nq: int
    actuators: list[ActuatorInfo]
    ee_body_idx: int       # body whose world transform represents the EE
    ee_body_name: str

    def home_qpos(self) -> np.ndarray:
        # If a 'home' keyframe exists, use it; else neutral (zeros for hinge joints).
        for i in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name == "home":
                return self.model.key_qpos[i].copy()
        return np.zeros(self.model.nq, dtype=np.float64)


def _actuator_info(model: mujoco.MjModel) -> list[ActuatorInfo]:
    """Pull per-actuator name + effective ctrl range + home-keyframe ctrl.

    If the 'home' keyframe declares ``ctrl=...`` we honour it; otherwise we
    fall back to deriving each joint-position actuator's home ctrl from the
    home qpos (so a slider Reset matches the initial pose).
    """
    home_ctrl = None
    home_qpos = None
    for i in range(model.nkey):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        if n == "home":
            home_ctrl = model.key_ctrl[i].copy()
            home_qpos = model.key_qpos[i].copy()
            break

    use_qpos_for_home = (
        home_ctrl is not None and home_qpos is not None
        and float(np.max(np.abs(home_ctrl))) == 0.0
    )

    out: list[ActuatorInfo] = []
    for ai in range(model.nu):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
                or f"act_{ai}")
        is_joint = int(model.actuator_trntype[ai]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        if bool(model.actuator_ctrllimited[ai]):
            lo, hi = float(model.actuator_ctrlrange[ai, 0]), float(model.actuator_ctrlrange[ai, 1])
        elif is_joint:
            j = int(model.actuator_trnid[ai, 0])
            if bool(model.jnt_limited[j]):
                lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
            else:
                lo, hi = -np.pi * 2.0, np.pi * 2.0
        else:
            lo, hi = -1.0, 1.0

        if use_qpos_for_home and is_joint:
            j = int(model.actuator_trnid[ai, 0])
            qadr = int(model.jnt_qposadr[j])
            h = float(home_qpos[qadr])
        elif home_ctrl is not None:
            h = float(home_ctrl[ai])
        else:
            h = 0.0
        h = max(lo, min(hi, h))
        out.append(ActuatorInfo(name=name, ctrl_min=lo, ctrl_max=hi, home_ctrl=h))
    return out


def load_robot_scene(mjcf_path: str | Path) -> RobotScene:
    """Parse an MJCF and pull out the static visual description."""
    mjcf_path = Path(mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    bodies: list[BodyInfo] = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        bodies.append(BodyInfo(id=i, name=name, parent=int(model.body_parentid[i])))

    geoms: list[GeomInfo] = []
    for i in range(model.ngeom):
        if int(model.geom_group[i]) > VISUAL_GROUP_MAX:
            continue
        gtype = int(model.geom_type[i])
        if gtype == GEOM_PLANE:
            # Skip the floor — the browser already draws its own ground grid.
            continue

        color = _geom_color(model, i)
        info = GeomInfo(
            body_id=int(model.geom_bodyid[i]),
            type=gtype,
            local_pos=model.geom_pos[i].astype(np.float32).copy(),
            local_quat=model.geom_quat[i].astype(np.float32).copy(),  # wxyz
            size=model.geom_size[i].astype(np.float32).copy(),
            color=color,
        )
        if gtype == GEOM_MESH:
            mid = int(model.geom_dataid[i])
            if mid >= 0:
                v_start = int(model.mesh_vertadr[mid])
                v_count = int(model.mesh_vertnum[mid])
                f_start = int(model.mesh_faceadr[mid])
                f_count = int(model.mesh_facenum[mid])
                info.mesh_verts = (
                    model.mesh_vert[v_start:v_start + v_count].astype(np.float32).copy()
                )
                info.mesh_faces = (
                    model.mesh_face[f_start:f_start + f_count].astype(np.int32).copy()
                )
        geoms.append(info)

    actuators = _actuator_info(model)
    ee_idx, ee_name = _pick_ee_body(model, bodies)
    return RobotScene(mjcf_path=mjcf_path, model=model, bodies=bodies,
                      geoms=geoms, nq=int(model.nq), actuators=actuators,
                      ee_body_idx=ee_idx, ee_body_name=ee_name)


# Common end-effector body names, in priority order. The first match wins.
_EE_BODY_PREFERENCE = (
    "end_effector_link",   # Kinova Gen3 convention (this repo's MJCF)
    "ee_link",
    "tcp",
    "tool0",
    "wrist_link",
    "bracelet_link",
)


def _pick_ee_body(model: mujoco.MjModel,
                  bodies: list[BodyInfo]) -> tuple[int, str]:
    """Pick a body to display the EE axes at.

    Strategy: prefer well-known names; otherwise fall back to the deepest
    body in the kinematic chain (likely the leaf of the arm).
    """
    by_name = {b.name: b.id for b in bodies}
    for name in _EE_BODY_PREFERENCE:
        if name in by_name:
            return by_name[name], name
    # Fallback: pick the body whose ancestry chain to world is longest.
    depth = [0] * model.nbody
    for i in range(1, model.nbody):
        depth[i] = depth[int(model.body_parentid[i])] + 1
    leaf = int(np.argmax(depth))
    return leaf, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, leaf) or f"body_{leaf}"


def _geom_color(model: mujoco.MjModel, geom_idx: int) -> np.ndarray:
    """Resolve geom color: use material rgba if present, else the geom's own rgba."""
    mat_id = int(model.geom_matid[geom_idx])
    if mat_id >= 0:
        return model.mat_rgba[mat_id].astype(np.float32).copy()
    return model.geom_rgba[geom_idx].astype(np.float32).copy()
