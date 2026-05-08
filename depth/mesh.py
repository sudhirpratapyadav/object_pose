"""Per-frame depth-to-mesh on a fixed-topology grid.

Topology (face indices) is computed once at startup. Each frame the worker
fills a fixed-size vertex buffer and applies an edge mask: triangles whose
vertices straddle a depth jump > `edge_threshold` are collapsed to a degenerate
triangle (3 copies of vertex 0, GPU-culled). Invalid pixels are also collapsed.

This avoids reallocating the index buffer or shipping a variable-length face
array each frame.
"""

from __future__ import annotations

import numpy as np


def grid_dims(infer_w: int, infer_h: int, ds: int) -> tuple[int, int]:
    return infer_w // ds, infer_h // ds


def build_faces(grid_w: int, grid_h: int) -> np.ndarray:
    """Static index buffer for an (h x w) grid.

    Two triangles per quad, ordered (a, c, b) and (b, c, d) where a=top-left,
    b=top-right, c=bottom-left, d=bottom-right.

    Returns int32 array of shape (2*(h-1)*(w-1), 3). The order of triangles is
    deterministic so the worker can index into per-quad info row-major.
    """
    h, w = grid_h, grid_w
    i, j = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    a = (i * w + j).ravel()
    b = (i * w + j + 1).ravel()
    c = ((i + 1) * w + j).ravel()
    d = ((i + 1) * w + j + 1).ravel()
    tri1 = np.stack([a, c, b], axis=1)
    tri2 = np.stack([b, c, d], axis=1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int32)


def precompute_unproject(
    grid_w: int, grid_h: int, ds: int,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_norm, y_norm) so xyz = (x_norm*z, y_norm*z, z) per grid pixel."""
    u = np.arange(grid_w, dtype=np.float32) * ds
    v = np.arange(grid_h, dtype=np.float32) * ds
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy
    return x_norm, y_norm


def fill_mesh(
    depth: np.ndarray,            # (infer_h, infer_w) float32
    rgb_inf: np.ndarray,          # (infer_h, infer_w, 3) uint8
    x_norm: np.ndarray,           # (grid_h, grid_w)
    y_norm: np.ndarray,           # (grid_h, grid_w)
    faces_static: np.ndarray,     # (M, 3) int32, output of build_faces
    ds: int,
    z_min: float, z_max: float,
    edge_threshold: float,
    out_xyz: np.ndarray,          # (grid_h*grid_w, 3) float32 — written
    out_rgb: np.ndarray,          # (grid_h*grid_w, 3) float32 — written, [0..1]
    out_faces: np.ndarray,        # (M, 3) int32 — written (collapsed copy)
    normal_grid: "np.ndarray | None" = None,    # (grid_h, grid_w, 3) float32
    normal_cos_threshold: float = 0.5,          # cos(60°) = 0.5; lower angle → tighter
) -> None:
    z = depth[::ds, ::ds]
    h, w = z.shape
    valid = (z > z_min) & (z < z_max)

    x = x_norm * z
    y = y_norm * z
    out_xyz[:, 0] = x.ravel()
    out_xyz[:, 1] = y.ravel()
    out_xyz[:, 2] = z.ravel()

    out_rgb[...] = rgb_inf[::ds, ::ds].reshape(-1, 3).astype(np.float32) * (1.0 / 255.0)

    # Per-quad depth-edge test. Face order matches build_faces:
    # quad indexed by (i, j) for i in [0..h-2], j in [0..w-2], row-major.
    # Triangles are tri1 (a,c,b), tri2 (b,c,d).
    n_quads = (h - 1) * (w - 1)
    za = z[:-1, :-1].ravel()
    zb = z[:-1, 1: ].ravel()
    zc = z[1: , :-1].ravel()
    zd = z[1: , 1: ].ravel()
    va = valid[:-1, :-1].ravel()
    vb = valid[:-1, 1: ].ravel()
    vc = valid[1: , :-1].ravel()
    vd = valid[1: , 1: ].ravel()

    et = edge_threshold
    keep1 = (
        va & vb & vc
        & (np.abs(za - zb) < et)
        & (np.abs(za - zc) < et)
        & (np.abs(zb - zc) < et)
    )
    keep2 = (
        vb & vc & vd
        & (np.abs(zb - zc) < et)
        & (np.abs(zb - zd) < et)
        & (np.abs(zc - zd) < et)
    )

    # Optional normal-angle test: drop triangles whose vertex normals diverge
    # too much (i.e. cross a surface boundary). Cheap dot products on already
    # unit-length vectors.
    if normal_grid is not None and normal_grid.shape[:2] == (h, w):
        na = normal_grid[:-1, :-1].reshape(-1, 3)
        nb = normal_grid[:-1, 1: ].reshape(-1, 3)
        nc = normal_grid[1: , :-1].reshape(-1, 3)
        nd = normal_grid[1: , 1: ].reshape(-1, 3)
        ct = normal_cos_threshold
        keep1 &= (
            (np.einsum("ij,ij->i", na, nb) > ct)
            & (np.einsum("ij,ij->i", na, nc) > ct)
            & (np.einsum("ij,ij->i", nb, nc) > ct)
        )
        keep2 &= (
            (np.einsum("ij,ij->i", nb, nc) > ct)
            & (np.einsum("ij,ij->i", nb, nd) > ct)
            & (np.einsum("ij,ij->i", nc, nd) > ct)
        )

    out_faces[...] = faces_static
    # Collapse rejected triangles to (0, 0, 0) — degenerate, GPU culls.
    bad1 = ~keep1
    bad2 = ~keep2
    if bad1.any():
        out_faces[:n_quads][bad1] = 0
    if bad2.any():
        out_faces[n_quads:][bad2] = 0
