"""Robot joint state in shared memory + server-side FK.

The qpos slot is a small, latest-wins shared-memory buffer that any worker
can write into (sim, hardware OSC, manual setter). Web server reads it and
runs MuJoCo forward kinematics inline to get per-body world transforms.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class RobotShm:
    """Shared-memory slot for a single qpos vector."""
    qpos: mp.Array            # float64, length nq
    qpos_seq: mp.Value        # uint32 — bumped each time qpos is written
    nq: int

    def read_qpos(self) -> np.ndarray:
        return np.frombuffer(self.qpos.get_obj(), dtype=np.float64).copy()

    def read_seq(self) -> int:
        return int(self.qpos_seq.value)

    def write_qpos(self, q: np.ndarray) -> None:
        if q.shape != (self.nq,):
            raise ValueError(f"qpos shape {q.shape}, expected ({self.nq},)")
        with self.qpos.get_lock():
            np.frombuffer(self.qpos.get_obj(), dtype=np.float64)[:] = q
            self.qpos_seq.value = (self.qpos_seq.value + 1) & 0xFFFFFFFF


def create_robot_shm(nq: int, init_qpos: np.ndarray | None = None) -> RobotShm:
    arr = mp.Array("d", nq, lock=True)
    if init_qpos is not None:
        if init_qpos.shape != (nq,):
            raise ValueError(f"init_qpos shape {init_qpos.shape}, expected ({nq},)")
        np.frombuffer(arr.get_obj(), dtype=np.float64)[:] = init_qpos
    seq = mp.Value("I", 1 if init_qpos is not None else 0, lock=False)
    return RobotShm(qpos=arr, qpos_seq=seq, nq=nq)


class FKEngine:
    """Wraps a per-process MjData and runs forward kinematics on demand.

    Not multiprocess-safe — keep one instance per process.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)

    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xpos[nbody, 3] f32, xquat[nbody, 4] f32 wxyz) at the given qpos."""
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos shape {qpos.shape}, expected ({self.model.nq},)"
            )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        xpos = self.data.xpos.astype(np.float32, copy=True)
        xquat = self.data.xquat.astype(np.float32, copy=True)
        return xpos, xquat
