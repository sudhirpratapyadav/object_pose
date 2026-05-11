"""Policy plugin registry.

A *policy* writes setpoints into shared memory at its own outer-loop rate
(typically 10-50 Hz). The active *controller* (ee_pose) reads those
setpoints and tracks them at 500 Hz. Policy and controller are decoupled
on purpose: the same NN policy can drive any torque controller, and the
same controller can be driven by a UI gizmo, a scripted recording, or an
NN policy.

Engaging a policy:
  1. Server checks that a per-policy YAML entry exists in
     ``policies/configs.yaml`` (or the .example fallback).
  2. Server checks that the policy can read its observations
     (e.g. open_drawer needs object_pose from SAM).
  3. Server hot-swaps the controller to ``ee_pose`` using the policy's
     gains + home pose (overlaid for the duration of the run).
  4. Policy subprocess starts; writes shm_qtarget at TARGET_HZ.
  5. On stop, the saved user gains + default home are restored.
"""

from __future__ import annotations

from typing import Callable

from .base import PolicyInfo
from .open_drawer import open_drawer_policy_process


PolicyFactory = Callable[..., tuple[Callable, dict]]


def _open_drawer_factory(*, mjcf_path: str, cfg: dict,
                          log_q=None, **_unused) -> tuple[Callable, dict]:
    return open_drawer_policy_process, {
        "mjcf_path": mjcf_path,
        "cfg":       cfg,
        "log_q":     log_q,
    }


# name -> (info, factory)
POLICIES: dict[str, tuple[PolicyInfo, PolicyFactory]] = {
    "open_drawer": (
        PolicyInfo(
            name="open_drawer",
            display_name="Open drawer (NN)",
            description="RSL-RL PPO policy. 33-D obs, 7-D action "
                        "(delta_pos[3] + delta_ori[3] + gripper). "
                        "Needs object_pose (SAM mask on the drawer handle). "
                        "Drives the ee_pose controller at the policy's "
                        "trained gains.",
            controller="ee_pose",
            needs_object_pose=True,
        ),
        _open_drawer_factory,
    ),
}


DEFAULT_POLICY: str | None = None   # boots with no policy engaged


__all__ = [
    "PolicyInfo", "PolicyFactory", "POLICIES", "DEFAULT_POLICY",
    "open_drawer_policy_process",
]
