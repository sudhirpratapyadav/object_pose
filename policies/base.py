"""Common interface for policy plugins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyInfo:
    name: str                   # short id used in dropdown
    display_name: str
    description: str
    controller: str             # required controller name (always "ee_pose" today)
    needs_object_pose: bool     # whether engage blocks on a fresh object_pose
