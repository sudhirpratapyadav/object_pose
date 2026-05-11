"""Per-policy defaults YAML loader.

Each policy entry carries the constants its trained checkpoint expects:
checkpoint path, obs/action dims, OSC gains, action scales, home pose,
target rate, and any task-specific knobs (success threshold, goal offset).

The runtime file ``configs.yaml`` is per-machine and gitignored. A
``configs.yaml.example`` ships in the repo as a template; if the runtime
file is missing the loader falls back to the example so users still get
something sensible at first boot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIGS_PATH = Path(__file__).parent / "configs.yaml"
EXAMPLE_PATH = Path(__file__).parent / "configs.yaml.example"


def load_configs(path: Path | str = CONFIGS_PATH) -> dict[str, dict[str, Any]]:
    """Returns a name → config-dict mapping.

    Falls back to ``configs.yaml.example`` if the runtime file is missing
    so a fresh checkout still has sensible defaults.
    """
    p = Path(path)
    if not p.exists():
        p = EXAMPLE_PATH
        if not p.exists():
            return {}
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected top-level mapping")
    return data


def save_configs(configs: dict[str, dict[str, Any]],
                 path: Path | str = CONFIGS_PATH) -> None:
    """Write the full configs map to YAML (runtime path by default)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(configs, f, default_flow_style=None, sort_keys=False)
