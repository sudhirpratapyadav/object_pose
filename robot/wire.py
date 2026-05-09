"""Translate a RobotScene into the wire-format robot-geometry payload."""

from __future__ import annotations

import struct

import numpy as np

from .scene import GEOM_MESH, RobotScene


def build_robot_geometry_payload(scene: RobotScene
                                 ) -> tuple[list[dict], list[dict], list[dict], bytes]:
    """Returns (bodies_json, meshes_json, geoms_json, mesh_blob_bytes).

    Mesh data is deduplicated by (verts ptr, faces ptr) — meshes shared between
    geoms (e.g. left/right gripper drivers) are only sent once.
    """
    bodies = [{"name": b.name, "parent": int(b.parent)} for b in scene.bodies]

    # Dedup meshes: key on (verts_id, faces_id) pointer identity.
    mesh_idx_for_key: dict[tuple[int, int], int] = {}
    meshes: list[dict] = []
    blob = bytearray()

    geoms_json: list[dict] = []
    for g in scene.geoms:
        mesh_idx: int | None = None
        if g.type == GEOM_MESH and g.mesh_verts is not None and g.mesh_faces is not None:
            key = (id(g.mesh_verts), id(g.mesh_faces))
            mesh_idx = mesh_idx_for_key.get(key)
            if mesh_idx is None:
                v = np.ascontiguousarray(g.mesh_verts, dtype=np.float32)
                f = np.ascontiguousarray(g.mesh_faces, dtype=np.uint32)
                v_off = len(blob)
                blob.extend(v.tobytes())
                f_off = len(blob)
                blob.extend(f.tobytes())
                mesh_idx = len(meshes)
                meshes.append({
                    "vert_offset": v_off,
                    "vert_count":  int(v.shape[0]),
                    "face_offset": f_off,
                    "face_count":  int(f.shape[0]),
                })
                mesh_idx_for_key[key] = mesh_idx

        geoms_json.append({
            "body":  int(g.body_id),
            "type":  int(g.type),
            "pos":   [float(x) for x in g.local_pos],
            "quat":  [float(x) for x in g.local_quat],   # wxyz
            "size":  [float(x) for x in g.size],
            "color": [float(x) for x in g.color],
            "mesh":  mesh_idx,
        })

    return bodies, meshes, geoms_json, bytes(blob)
