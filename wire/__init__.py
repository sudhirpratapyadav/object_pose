"""Server-side wire-format encoders.

Mirrors web/src/protocol.ts; bytes here are the bytes parsed there.

Header (all little-endian):
    magic 'P3DF' (4 bytes) | seq u32 | kind u8 | _pad u24

Kinds:
   0  POINTS               n u32 | xyz_f16[3n] | rgb_u8[3n] | mask_u8[n]
   1  MESH                 nv u32 | nf u32 | has_normal u8 | xyz_f16[3nv] |
                           rgb_u8[3nv] | faces_u32[3nf] | normal_f16[3nv?]
   2  JPEG (rgb)           w u16 | h u16 | jpeg_bytes...
   3  META                 json bytes
   4  DEPTH_JPEG           w u16 | h u16 | jpeg_bytes...
   5  MODEL_STATE          json bytes
   6  MASK                 mask_seq u32 | n u32 | mask_u8[n] |
                           has_box u8 | box_min f32[3] | box_max f32[3]
   7  SAM_STATE            json bytes
   8  STATS                json bytes
   9  NORMAL_JPEG          w u16 | h u16 | jpeg_bytes...
  10  ROBOT_GEOMETRY       json header bytes... (see encode_robot_geometry)
  11  ROBOT_TRANSFORMS     nbody u32 | xpos_f32[3*nbody] | xquat_f32[4*nbody]
"""

from .frames import (
    MAGIC,
    KIND_POINTS,
    KIND_MESH,
    KIND_JPEG,
    KIND_META,
    KIND_DEPTH_JPEG,
    KIND_MODEL_STATE,
    KIND_MASK,
    KIND_SAM_STATE,
    KIND_STATS,
    KIND_NORMAL_JPEG,
    KIND_ROBOT_GEOMETRY,
    KIND_ROBOT_TRANSFORMS,
    KIND_CAM_CALIB,
    KIND_ROBOT_STATUS,
    KIND_CONTROLLER_STATE,
    KIND_LOG_LINES,
    pack_header,
    encode_points,
    encode_mesh,
    encode_jpeg,
    encode_meta,
    encode_model_state,
    encode_sam_state,
    encode_stats,
    encode_mask,
    encode_robot_geometry,
    encode_robot_transforms,
    encode_cam_calib,
    encode_robot_status,
    encode_controller_state,
    encode_log_lines,
)

__all__ = [
    "MAGIC",
    "KIND_POINTS",
    "KIND_MESH",
    "KIND_JPEG",
    "KIND_META",
    "KIND_DEPTH_JPEG",
    "KIND_MODEL_STATE",
    "KIND_MASK",
    "KIND_SAM_STATE",
    "KIND_STATS",
    "KIND_NORMAL_JPEG",
    "KIND_ROBOT_GEOMETRY",
    "KIND_ROBOT_TRANSFORMS",
    "KIND_CAM_CALIB",
    "KIND_ROBOT_STATUS",
    "KIND_CONTROLLER_STATE",
    "KIND_LOG_LINES",
    "pack_header",
    "encode_points",
    "encode_mesh",
    "encode_jpeg",
    "encode_meta",
    "encode_model_state",
    "encode_sam_state",
    "encode_stats",
    "encode_mask",
    "encode_robot_geometry",
    "encode_robot_transforms",
    "encode_cam_calib",
    "encode_robot_status",
    "encode_controller_state",
    "encode_log_lines",
]
