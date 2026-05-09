"""Camera calibration config: shared by real-mode and sim-mode.

YAML is the single source of truth. Both modes load it at startup. Real mode
ignores the RealSense factory intrinsics for the depth pipeline (a separate
factory snapshot is dumped to ``cam_factory_intrinsics.yaml`` for reference);
sim mode patches the MJCF camera with these values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


CAM_CALIB_PATH    = Path(__file__).parent / "cam_calib_config.yaml"
SIM_CONFIG_PATH   = Path(__file__).parent / "sim_config.yaml"
FACTORY_INTR_PATH = Path(__file__).parent / "cam_factory_intrinsics.yaml"


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([0.25 * s,
                         (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([(m[2, 1] - m[1, 2]) / s,
                         0.25 * s,
                         (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 2] - m[2, 0]) / s,
                         (m[0, 1] + m[1, 0]) / s,
                         0.25 * s,
                         (m[1, 2] + m[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array([(m[1, 0] - m[0, 1]) / s,
                     (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s,
                     0.25 * s])


def _euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic XYZ Euler (degrees) -> 3x3 rotation matrix.

    Matches scipy ``Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)``:
    R = Rx(rx) @ Ry(ry) @ Rz(rz).
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


@dataclass
class Extrinsics:
    pos: list[float]        # length 3, [x, y, z] in world
    euler_deg: list[float]  # length 3, XYZ Euler in degrees


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class CamCalib:
    extrinsics: Extrinsics
    intrinsics: Intrinsics

    def T_world_camera(self) -> np.ndarray:
        """4x4 homogeneous transform from camera frame to world frame."""
        R = _euler_xyz_deg_to_matrix(*self.extrinsics.euler_deg)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = np.asarray(self.extrinsics.pos, dtype=np.float64)
        return T

    def fovy_deg(self) -> float:
        """Vertical field-of-view in degrees, derived from fy + height.

        MuJoCo cameras assume principal point at the image center; sim mode
        cannot exactly reproduce a non-centered (cx, cy). See ``is_centered``.
        """
        return 2.0 * math.degrees(math.atan(0.5 * self.intrinsics.height / self.intrinsics.fy))

    def is_centered(self, tol_px: float = 1.0) -> bool:
        i = self.intrinsics
        return abs(i.cx - 0.5 * i.width) < tol_px and abs(i.cy - 0.5 * i.height) < tol_px


class CamCalibError(RuntimeError):
    """Raised when the calibration YAML is missing or malformed."""


def load_cam_calib(path: Path | str = CAM_CALIB_PATH) -> CamCalib:
    """Load and validate the calibration YAML (the user's *belief* about the
    camera pose + intrinsics). Hard error if missing."""
    return _load_calib_yaml(
        Path(path),
        missing_hint=(
            "Create one (see cam_calib_config.yaml.example) before starting "
            "the server. This file is the depth pipeline's belief about "
            "where the camera is; in real mode it is also where you persist "
            "calibration results."
        ),
    )


def load_sim_config(path: Path | str = SIM_CONFIG_PATH) -> CamCalib:
    """Load the sim ground-truth YAML (where MuJoCo actually places the
    rendered camera). Same shape as cam_calib_config.yaml."""
    return _load_calib_yaml(
        Path(path),
        missing_hint=(
            "Create one (see sim_config.yaml.example) before starting in "
            "--mode sim. This file is the sim's GROUND TRUTH; it is "
            "intentionally separate from cam_calib_config.yaml so you can "
            "test calibration by letting them differ."
        ),
    )


def _load_calib_yaml(p: Path, *, missing_hint: str) -> CamCalib:
    if not p.exists():
        raise CamCalibError(f"Config not found: {p}\n{missing_hint}")
    with p.open() as f:
        data = yaml.safe_load(f) or {}

    try:
        ex = data["extrinsics"]
        intr = data["intrinsics"]
        cfg = CamCalib(
            extrinsics=Extrinsics(
                pos=[float(x) for x in ex["pos"]],
                euler_deg=[float(x) for x in ex["euler_deg"]],
            ),
            intrinsics=Intrinsics(
                fx=float(intr["fx"]), fy=float(intr["fy"]),
                cx=float(intr["cx"]), cy=float(intr["cy"]),
                width=int(intr["width"]), height=int(intr["height"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CamCalibError(f"Malformed {p}: {exc}") from exc

    _validate(cfg, p)
    return cfg


def save_cam_calib(cfg: CamCalib, path: Path | str = CAM_CALIB_PATH) -> None:
    """Serialise back to YAML, preserving structure."""
    payload = {
        "extrinsics": asdict(cfg.extrinsics),
        "intrinsics": asdict(cfg.intrinsics),
    }
    with Path(path).open("w") as f:
        yaml.safe_dump(payload, f, default_flow_style=None, sort_keys=False)


def write_factory_intrinsics(intr: Intrinsics, path: Path | str = FACTORY_INTR_PATH,
                             *, header: str | None = None) -> None:
    """Dump the camera's reported intrinsics to a separate YAML.

    Read-only snapshot for the user — the server never reads this back.
    """
    payload = {"intrinsics": asdict(intr)}
    p = Path(path)
    text = yaml.safe_dump(payload, default_flow_style=None, sort_keys=False)
    with p.open("w") as f:
        if header:
            for line in header.strip().splitlines():
                f.write(f"# {line}\n")
        f.write(text)


def _validate(cfg: CamCalib, path: Path) -> None:
    i = cfg.intrinsics
    if i.width <= 0 or i.height <= 0:
        raise CamCalibError(f"{path}: width/height must be positive")
    if i.fx <= 0 or i.fy <= 0:
        raise CamCalibError(f"{path}: fx/fy must be positive")
    if not (0.0 <= i.cx <= i.width) or not (0.0 <= i.cy <= i.height):
        raise CamCalibError(
            f"{path}: principal point ({i.cx}, {i.cy}) outside image "
            f"({i.width}x{i.height})"
        )
    if len(cfg.extrinsics.pos) != 3 or len(cfg.extrinsics.euler_deg) != 3:
        raise CamCalibError(f"{path}: extrinsics.pos/euler_deg must have length 3")
