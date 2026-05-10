"""Kinova Gen3 hardware: low-level kortex wrapper, transport process,
and the OSC torque math (kept here for the future EE-pose controller)."""

from .kinova import KinovaHardware, RobotState
from .osc import (
    PinocchioArm,
    compute_osc_torques,
    real_robot_process,   # legacy single-process OSC loop, no longer spawned
    kinova_deg_to_rad,
    HOME_DEG,
    MAX_JOINT_TORQUE,
    TAU_OFFSETS,
    GAINS_KEYS,
    OSC_HZ,
    TARGET_HZ,
    OSC_SUBS,
)
from .transport import (
    transport_process,
    CMD_MODE_IDLE,
    CMD_MODE_TORQUE,
    CMD_MODE_POSITION,
    PHASE_BOOT,
    PHASE_HOMING,
    PHASE_READY,
    PHASE_RUNNING,
    PHASE_SWAPPING,
    PHASE_FAULT,
    PHASE_SHUTDOWN,
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
    "transport_process",
    "CMD_MODE_IDLE",
    "CMD_MODE_TORQUE",
    "CMD_MODE_POSITION",
    "PHASE_BOOT",
    "PHASE_HOMING",
    "PHASE_READY",
    "PHASE_RUNNING",
    "PHASE_SWAPPING",
    "PHASE_FAULT",
    "PHASE_SHUTDOWN",
]
