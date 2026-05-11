"""Estimate an object's world-frame position from a SAM mask + point cloud.

The SAM worker produces a binary mask keyed by mesh-grid cell. The depth
worker produces a point cloud with a parallel ``pc_grid_idx`` array — the
mesh-grid cell each point comes from. Cross-reference the two and average
the masked points to get a centroid in *camera* frame, then transform by
``T_world_camera`` to get the world-frame pose.

For the open-drawer policy this is "good enough" — the drawer handle is a
rigid, mostly-symmetric object, so the visible centroid is close to the
training-time ``object_site``. If we need finer pose later (orientation,
or a precise handle-tip offset), we can replace this with PCA-based
oriented-bounding-box fitting.
"""

from __future__ import annotations

import numpy as np


MIN_POINTS = 10   # below this, the mask is too sparse to trust


def compute_object_pose(
    seg_mask: np.ndarray,        # (mesh_grid_h * mesh_grid_w,) uint8, 1 inside object
    pc_xyz: np.ndarray,          # (n, 3) float32, points in CAMERA frame
    pc_grid_idx: np.ndarray,     # (n,) uint32, mesh-grid index per point
    T_world_camera: np.ndarray,  # (4, 4) float64
    n_valid: int,                # how many of the first rows of pc_xyz are valid
) -> tuple[np.ndarray, int]:
    """Return ``(pos_world, n_inside)``.

    ``pos_world`` is the centroid of all points that fall inside the
    SAM mask, transformed into the world frame. ``n_inside`` is the count
    of masked points used.  If too few points (< MIN_POINTS), returns
    (zeros, 0) — callers should treat that as "no fresh estimate".
    """
    if n_valid <= 0 or seg_mask is None or pc_xyz is None or pc_grid_idx is None:
        return np.zeros(3, dtype=np.float64), 0

    # seg_mask is keyed by mesh-grid cell; pc_grid_idx maps each point to
    # its grid cell. Boolean mask of "inside object" per point.
    idx = pc_grid_idx[:n_valid]
    if idx.size == 0:
        return np.zeros(3, dtype=np.float64), 0
    inside = seg_mask[idx] != 0
    n_in = int(inside.sum())
    if n_in < MIN_POINTS:
        return np.zeros(3, dtype=np.float64), n_in

    pts_cam = pc_xyz[:n_valid][inside]              # (n_in, 3)
    centroid_cam = pts_cam.mean(axis=0)             # (3,)

    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    centroid_world = (R @ centroid_cam) + t
    return centroid_world.astype(np.float64), n_in
