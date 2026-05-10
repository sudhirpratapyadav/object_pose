"""Controller plugin interface.

Each controller is a callable that runs as its own ``mp.Process``. Shape:

    controller_process(
        # Joint state IN (read-only).
        shm_q, shm_dq, shm_state_seq,
        # Commands OUT (the controller fills these; the transport reads).
        shm_cmd_mode, shm_tau, shm_qtarget, shm_gripper, shm_cmd_seq,
        # Lifecycle.
        stop_ev,
        # Plugin-specific extras (mjcf path, etc.) via kwargs.
    )

The controller writes its output into the same shm slots the transport
reads. Transport never imports any controller; the dispatcher orchestrates
who's the active writer.

A registry of controller metadata + factories lives in ``controllers/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerInfo:
    name: str               # short id used in dropdown / commands
    display_name: str       # human-readable
    description: str        # one-line tooltip shown in the UI
    command_mode: str       # 'idle' | 'torque' | 'position'
