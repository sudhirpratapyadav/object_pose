"""Producers that write qpos into the RobotShm slot.

Stage-1 sources:

  - ``none``:   no-op; whoever else writes the shm wins.
  - ``dummy``:  background thread that sweeps the arm joints in a slow sine,
                useful for verifying the rendering loop end-to-end.

Later stages will add ``sim`` (MuJoCo step loop) and ``hardware`` (Kinova
OSC loop, see ``hardware.real_robot_process``).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from .state import RobotShm


class DummySource:
    """Animates the first 7 hinge joints (Kinova arm) with a slow sine."""

    def __init__(self, shm: RobotShm, init_qpos: np.ndarray, hz: float = 60.0,
                 amplitude: float = 0.4, period_s: float = 6.0) -> None:
        self._shm = shm
        self._base = init_qpos.astype(np.float64).copy()
        self._dt = 1.0 / hz
        self._amp = amplitude
        self._omega = 2.0 * np.pi / period_s
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thr is not None:
            return
        self._thr = threading.Thread(target=self._loop, daemon=True, name="robot-dummy")
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)
            self._thr = None

    def _loop(self) -> None:
        t0 = time.time()
        q = self._base.copy()
        # Per-joint phase offsets so the motion is visible across the chain.
        phase = np.linspace(0.0, np.pi, 7)
        n_arm = min(7, self._shm.nq)
        while not self._stop.is_set():
            t = time.time() - t0
            q[:n_arm] = self._base[:n_arm] + self._amp * np.sin(self._omega * t + phase[:n_arm])
            self._shm.write_qpos(q)
            time.sleep(self._dt)
