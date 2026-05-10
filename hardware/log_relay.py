"""Subprocess-side logging that relays records into a shared mp.Queue.

Each subprocess (transport, controller) calls ``install_log_relay(log_q,
source)`` early in its target. From then on, every ``logging`` call is
duplicated to the queue so the server can broadcast log lines to the
browser. Records also go to stderr as usual so terminal users still see
them.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time


class _QueueHandler(logging.Handler):
    """Push each record into a shared mp.Queue. Drop on full to avoid
    blocking the hot path."""

    def __init__(self, q: mp.Queue, source: str) -> None:
        super().__init__()
        self._q = q
        self._source = source

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        try:
            self._q.put_nowait({
                "ts": time.time(),
                "level": record.levelname,
                "source": self._source,
                "msg": msg,
            })
        except Exception:
            # Queue full or pickle error — drop the line silently. Terminal
            # stderr still has it.
            pass


def install_log_relay(log_q: mp.Queue | None, source: str,
                      level: int = logging.INFO) -> None:
    """Install both a stderr handler (so terminal output keeps working)
    and the queue relay. Idempotent — safe to call from process target."""
    fmt = logging.Formatter("[%(name)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    # Wipe any existing handlers (subprocesses inherit them via fork on
    # some platforms; we want a clean slate).
    for h in list(root.handlers):
        root.removeHandler(h)

    stderr_h = logging.StreamHandler()
    stderr_h.setFormatter(fmt)
    root.addHandler(stderr_h)

    if log_q is not None:
        q_h = _QueueHandler(log_q, source)
        q_h.setFormatter(fmt)
        root.addHandler(q_h)
