"""Binary frame encoders for the WS protocol."""

from __future__ import annotations

import json
import struct

import cv2
import numpy as np


MAGIC = b"P3DF"

KIND_POINTS           = 0
KIND_MESH             = 1
KIND_JPEG             = 2
KIND_META             = 3
KIND_DEPTH_JPEG       = 4
KIND_MODEL_STATE      = 5
KIND_MASK             = 6
KIND_SAM_STATE        = 7
KIND_STATS            = 8
KIND_NORMAL_JPEG      = 9
KIND_ROBOT_GEOMETRY   = 10
KIND_ROBOT_TRANSFORMS = 11
KIND_CAM_CALIB        = 12
KIND_ROBOT_STATUS     = 13
KIND_CONTROLLER_STATE = 14
KIND_LOG_LINES        = 15


def pack_header(seq: int, kind: int) -> bytes:
    return MAGIC + struct.pack("<IB3x", seq & 0xFFFFFFFF, kind)


def _f32_to_f16_bytes(arr_f32: np.ndarray) -> bytes:
    return arr_f32.astype(np.float16).tobytes()


def encode_points(seq: int, xyz: np.ndarray, rgb: np.ndarray,
                  mask: np.ndarray | None = None) -> bytes:
    n = xyz.shape[0]
    if mask is None:
        mask = np.zeros((n,), dtype=np.uint8)
    return (
        pack_header(seq, KIND_POINTS)
        + struct.pack("<I", n)
        + _f32_to_f16_bytes(xyz)
        + rgb.astype(np.uint8).tobytes()
        + mask.astype(np.uint8).tobytes()
    )


def encode_mesh(seq: int, xyz: np.ndarray, rgb: np.ndarray,
                faces: np.ndarray, normal: np.ndarray | None = None) -> bytes:
    has_n = 1 if normal is not None else 0
    body = (
        struct.pack("<IIB", xyz.shape[0], faces.shape[0], has_n)
        + _f32_to_f16_bytes(xyz)
        + (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        + faces.astype(np.uint32).tobytes()
    )
    if has_n:
        body += _f32_to_f16_bytes(normal.astype(np.float32))
    return pack_header(seq, KIND_MESH) + body


def encode_jpeg(seq: int, bgr: np.ndarray, kind: int = KIND_JPEG,
                quality: int = 70) -> bytes:
    h, w = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return pack_header(seq, kind) + struct.pack("<HH", w, h) + buf.tobytes()


def encode_meta(seq: int, payload: dict) -> bytes:
    return pack_header(seq, KIND_META) + json.dumps(payload).encode("utf-8")


def encode_model_state(seq: int, payload: dict) -> bytes:
    return pack_header(seq, KIND_MODEL_STATE) + json.dumps(payload).encode("utf-8")


def encode_sam_state(seq: int, payload: dict) -> bytes:
    return pack_header(seq, KIND_SAM_STATE) + json.dumps(payload).encode("utf-8")


def encode_stats(seq: int, payload: dict) -> bytes:
    return pack_header(seq, KIND_STATS) + json.dumps(payload).encode("utf-8")


def encode_mask(seq: int, mask_seq: int, mask: np.ndarray,
                has_box: bool, box_min: np.ndarray, box_max: np.ndarray) -> bytes:
    n = int(mask.size)
    body = (
        struct.pack("<II", mask_seq & 0xFFFFFFFF, n)
        + mask.astype(np.uint8).tobytes()
        + struct.pack("<B", 1 if has_box else 0)
        + box_min.astype(np.float32).tobytes()
        + box_max.astype(np.float32).tobytes()
    )
    return pack_header(seq, KIND_MASK) + body


def encode_robot_geometry(seq: int, bodies: list[dict], meshes: list[dict],
                          geoms: list[dict], mesh_blob: bytes) -> bytes:
    """One-shot description of a robot.

    bodies: [{"name": str, "parent": int}]
    meshes: [{"vert_offset": int, "vert_count": int,
              "face_offset": int, "face_count": int}]
              — all offsets are byte offsets into mesh_blob; vert_count and
              face_count are vertex / triangle counts (each vertex = 3 f32,
              each face = 3 u32).
    geoms:  [{"body": int, "type": int,
              "pos": [x,y,z], "quat": [w,x,y,z],
              "size": [a,b,c], "color": [r,g,b,a],
              "mesh": int | null}]
    mesh_blob: concatenated bytes for all meshes (verts_f32 + faces_u32 per mesh,
               in the order listed in `meshes`).
    """
    header = json.dumps({
        "bodies": bodies,
        "meshes": meshes,
        "geoms":  geoms,
    }, separators=(",", ":")).encode("utf-8")
    body = struct.pack("<I", len(header)) + header + mesh_blob
    return pack_header(seq, KIND_ROBOT_GEOMETRY) + body


def encode_cam_calib(seq: int, payload: dict) -> bytes:
    """Live update of the active camera calibration (pos + euler + intrinsics)."""
    return pack_header(seq, KIND_CAM_CALIB) + json.dumps(payload).encode("utf-8")


def encode_robot_status(seq: int, payload: dict) -> bytes:
    """1 Hz robot status: OSC rate, mode, error string, etc."""
    return pack_header(seq, KIND_ROBOT_STATUS) + json.dumps(payload).encode("utf-8")


def encode_controller_state(seq: int, payload: dict) -> bytes:
    """Controller dispatcher state: available list, current name, status."""
    return pack_header(seq, KIND_CONTROLLER_STATE) + json.dumps(payload).encode("utf-8")


def encode_log_lines(seq: int, lines: list[dict]) -> bytes:
    """Batch of log records:  [{ts, level, source, msg}, ...]."""
    return pack_header(seq, KIND_LOG_LINES) + json.dumps(lines).encode("utf-8")


def encode_robot_transforms(seq: int, xpos: np.ndarray, xquat: np.ndarray) -> bytes:
    """Per-frame body transforms.

    xpos: (nbody, 3) f32 world positions.
    xquat: (nbody, 4) f32 world orientations (wxyz, MuJoCo convention).
    """
    nbody = int(xpos.shape[0])
    body = (
        struct.pack("<I", nbody)
        + xpos.astype(np.float32).tobytes()
        + xquat.astype(np.float32).tobytes()
    )
    return pack_header(seq, KIND_ROBOT_TRANSFORMS) + body
