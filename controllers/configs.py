"""Per-controller defaults YAML loader.

The configs.yaml file holds default gains (and any other tunables) per
controller. The dispatcher loads it at startup; the browser can publish
``set_controller_gains`` to change in-memory values, and
``save_controller_gains`` to persist back to YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIGS_PATH = Path(__file__).parent / "configs.yaml"


def load_configs(path: Path | str = CONFIGS_PATH) -> dict[str, dict[str, Any]]:
    """Returns a name → config-dict mapping. Empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected top-level mapping")
    return data


def save_configs(configs: dict[str, dict[str, Any]],
                 path: Path | str = CONFIGS_PATH) -> None:
    """Write the full configs map to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(configs, f, default_flow_style=None, sort_keys=False)
