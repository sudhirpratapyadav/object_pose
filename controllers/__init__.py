"""Controller plugin registry.

Each entry is (ControllerInfo, factory).  The factory takes per-spawn
kwargs (mjcf path, etc.) and returns a target callable + kwargs suitable
for ``mp.Process(target=…, kwargs=…)``.
"""

from __future__ import annotations

from typing import Callable

from .base import ControllerInfo
from .ee_pose import ee_pose_controller_process
from .gravcomp import gravcomp_controller_process
from .idle import idle_controller_process
from .joint_pd import joint_pd_controller_process


# A factory returns (target_callable, fixed_kwargs_dict).
ControllerFactory = Callable[..., tuple[Callable, dict]]


# Factories accept **kwargs so the dispatcher can pass shm_gains uniformly;
# controllers that don't use it ignore the parameter.

def _idle_factory(*, log_q=None, **_unused) -> tuple[Callable, dict]:
    return idle_controller_process, {"log_q": log_q}


def _gravcomp_factory(*, mjcf_path: str, log_q=None,
                      **_unused) -> tuple[Callable, dict]:
    return gravcomp_controller_process, {
        "mjcf_path": mjcf_path,
        "ee_frame": "pinch_site",
        "log_q": log_q,
    }


def _joint_pd_factory(*, mjcf_path: str, log_q=None,
                      shm_gains=None, **_unused) -> tuple[Callable, dict]:
    return joint_pd_controller_process, {
        "mjcf_path": mjcf_path,
        "ee_frame":  "pinch_site",
        "shm_gains": shm_gains,
        "log_q":     log_q,
    }


def _ee_pose_factory(*, mjcf_path: str, log_q=None,
                     shm_gains=None, robot_source: str = "hardware",
                     **_unused) -> tuple[Callable, dict]:
    # ee_pose doesn't yet read shm_gains (gains hardcoded); kwarg is
    # accepted for uniformity with the dispatcher.
    del shm_gains
    return ee_pose_controller_process, {
        "mjcf_path": mjcf_path,
        "ee_frame":  "pinch_site",
        "log_q":     log_q,
        # Real-robot joint bias correction stays on for hardware, off for
        # sim (sim dynamics are symmetric, no bias needed).
        "apply_tau_offsets": robot_source == "hardware",
    }


# name -> (info, factory)
CONTROLLERS: dict[str, tuple[ControllerInfo, ControllerFactory]] = {
    "idle": (
        ControllerInfo(
            name="idle",
            display_name="Idle",
            description="No commands. Robot held by firmware in high-level "
                        "position mode. Default rest state between swaps.",
            command_mode="idle",
        ),
        _idle_factory,
    ),
    "gravcomp": (
        ControllerInfo(
            name="gravcomp",
            display_name="Gravity comp",
            description="Compliant: gravity + Coriolis compensation only. "
                        "Push the arm by hand and it follows without resisting.",
            command_mode="torque",
        ),
        _gravcomp_factory,
    ),
    "joint_pd": (
        ControllerInfo(
            name="joint_pd",
            display_name="Joint PD",
            description="500 Hz joint-space PD + gravity. "
                        "tau = kp·(q_des−q) − kd·q̇ + nle. "
                        "UI sliders publish q_des; gains are tunable live "
                        "from the Robot tab. Stiffness ranges from "
                        "compliant (low kp) to position-like (high kp).",
            command_mode="torque",
        ),
        _joint_pd_factory,
    ),
    "ee_pose": (
        ControllerInfo(
            name="ee_pose",
            display_name="EE pose (OSC)",
            description="Operational-space-control torque tracking. Holds "
                        "the EE at a target pose with conservative gains. "
                        "Compliant — push it and it pushes back, but not "
                        "rigidly. Gains UI lands in slice 4.5.",
            command_mode="torque",
        ),
        _ee_pose_factory,
    ),
}


DEFAULT_CONTROLLER = "idle"


__all__ = [
    "ControllerInfo",
    "ControllerFactory",
    "CONTROLLERS",
    "DEFAULT_CONTROLLER",
    "idle_controller_process",
    "gravcomp_controller_process",
    "joint_pd_controller_process",
    "ee_pose_controller_process",
]
