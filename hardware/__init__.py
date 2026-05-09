"""Kinova Gen3 hardware interface + OSC torque loop.

Copied verbatim from cam_calib_old/cam_calib_real.py — kept intact so it can
be wired into web_server.py later without rewriting.
"""

from .kinova import KinovaHardware, RobotState
from .osc import (
    PinocchioArm,
    compute_osc_torques,
    real_robot_process,
    kinova_deg_to_rad,
    HOME_DEG,
    MAX_JOINT_TORQUE,
    TAU_OFFSETS,
    GAINS_KEYS,
    OSC_HZ,
    TARGET_HZ,
    OSC_SUBS,
)

__all__ = [
    "KinovaHardware",
    "RobotState",
    "PinocchioArm",
    "compute_osc_torques",
    "real_robot_process",
    "kinova_deg_to_rad",
    "HOME_DEG",
    "MAX_JOINT_TORQUE",
    "TAU_OFFSETS",
    "GAINS_KEYS",
    "OSC_HZ",
    "TARGET_HZ",
    "OSC_SUBS",
]
