/**
 * Binary wire protocol parser.
 * Header: 'P3DF' (4) + seq u32 + kind u8 + 3 pad = 12 bytes.
 *   kind=0 points: n u32, xyz_f16 [3n], rgb_u8 [3n]
 *   kind=1 mesh:   nv u32, nf u32, xyz_f16 [3nv], rgb_u8 [3nv], faces_u32 [3nf]
 *   kind=2 jpeg:   w u16, h u16, jpeg bytes
 *   kind=3 meta:   utf-8 json
 */

export const KIND_POINTS = 0;
export const KIND_MESH = 1;
export const KIND_JPEG = 2;
export const KIND_META = 3;
export const KIND_DEPTH_JPEG = 4;
export const KIND_MODEL_STATE = 5;
export const KIND_MASK = 6;
export const KIND_SAM_STATE = 7;
export const KIND_STATS = 8;
export const KIND_NORMAL_JPEG = 9;
export const KIND_ROBOT_GEOMETRY = 10;
export const KIND_ROBOT_TRANSFORMS = 11;
export const KIND_CAM_CALIB = 12;
export const KIND_ROBOT_STATUS = 13;
export const KIND_CONTROLLER_STATE = 14;
export const KIND_LOG_LINES = 15;

// MuJoCo geom type codes (mjtGeom).
export const GEOM_PLANE     = 0;
export const GEOM_HFIELD    = 1;
export const GEOM_SPHERE    = 2;
export const GEOM_CAPSULE   = 3;
export const GEOM_ELLIPSOID = 4;
export const GEOM_CYLINDER  = 5;
export const GEOM_BOX       = 6;
export const GEOM_MESH      = 7;

export type ModelState = {
  model: string;
  status: string;
  progress: string;
  file: string;
  has_normals?: boolean;
};

export type SamState = {
  model: string;
  status: string;
  file: string;
};

export type Stats = {
  rgb_fps: number;
  depth_fps: number;
  sam_ms: number;
};

export type CamCalibPayload = {
  extrinsics: {
    pos: [number, number, number];
    euler_deg: [number, number, number];
    pos_world: [number, number, number];
    quat_wxyz: [number, number, number, number];
  };
  intrinsics: {
    fx: number; fy: number; cx: number; cy: number;
    width: number; height: number;
  };
};

export type Meta = {
  rgb_w: number; rgb_h: number;
  infer_w: number; infer_h: number;
  fx: number; fy: number; cx: number; cy: number;
  fx_infer: number; fy_infer: number; cx_infer: number; cy_infer: number;
  mesh_grid_w: number; mesh_grid_h: number;
  viz_hz: number;
  models: string[];
  /** model key -> camera requirement ("rgb" | "rgbd" | "rgb_stereo"). */
  model_camera_reqs?: Record<string, string>;
  default_model: string;
  camera_depth_available?: boolean;
  camera_stereo_available?: boolean;
  camera_depth_label?: string;
  model_state?: ModelState;
  sam_models: string[];
  sam_default_model: string;
  sam_state?: SamState;
  videos: string[];
  source: { kind: "live" | "video"; video: string | null };
  robot?: {
    enabled: boolean; source: string; mjcf: string | null;
    actuators: { name: string; min: number; max: number; home: number }[];
    ee_body_idx: number;
    ee_body_name: string;
  };
  cam_calib?: CamCalibPayload;
};

export type RobotBody = { name: string; parent: number };
export type RobotMeshIndex = {
  vert_offset: number; vert_count: number;
  face_offset: number; face_count: number;
};
export type RobotGeom = {
  body: number;
  type: number;             // GEOM_*
  pos: [number, number, number];
  quat: [number, number, number, number];   // wxyz
  size: [number, number, number];
  color: [number, number, number, number];
  mesh: number | null;       // index into RobotGeometryFrame.meshes
};

export type RobotGeometryFrame = {
  kind: typeof KIND_ROBOT_GEOMETRY; seq: number;
  bodies: RobotBody[];
  meshes: RobotMeshIndex[];
  geoms: RobotGeom[];
  blob: ArrayBuffer;          // concatenated verts_f32 + faces_u32 per mesh
};

export type RobotTransformsFrame = {
  kind: typeof KIND_ROBOT_TRANSFORMS; seq: number;
  xpos: Float32Array;         // length 3*nbody
  xquat: Float32Array;        // length 4*nbody, wxyz
  nbody: number;
};

export type CamCalibFrame = {
  kind: typeof KIND_CAM_CALIB; seq: number;
  calib: CamCalibPayload;
};

export type FaultEvent = {
  ts: number;       // unix seconds
  source: string;
  msg: string;
};

export type RobotStatus = {
  source: string;
  osc_hz?: number;
  alive?: boolean;
  phase?: number;
  phase_name?: string;     // "boot" | "homing" | "ready" | "running" | "swapping" | "fault" | "shutdown"
  fault_msg?: string;
  fault_history?: FaultEvent[];
};

export type RobotStatusFrame = {
  kind: typeof KIND_ROBOT_STATUS; seq: number;
  status: RobotStatus;
};

export type ControllerInfo = {
  name: string;
  display_name: string;
  description: string;
  command_mode: "idle" | "torque" | "position";
};

export type ControllerStatus = "idle" | "loading" | "running" | "stopping" | "fault";

export type EeTarget = {
  pos: [number, number, number];           // world frame, metres
  quat_xyzw: [number, number, number, number]; // unit, [x,y,z,w]
};

export type PolicyInfo = {
  name: string;
  display_name: string;
  description: string;
  controller: string;
  needs_object_pose: boolean;
};

export type ObjectPose = {
  pos: [number, number, number];   // world frame, metres
  n_points: number;                // masked-point count behind this estimate
};

export type PolicyState = {
  available: PolicyInfo[];
  selected: string;                // dropdown pick; "" if nothing selected
  current: string;                 // running subprocess; "" if not running
  lifecycle: "idle" | "paused" | "running";
  status_code: number;             // 0 waiting, 1 running, 2 success
  last_error: string;
  configs: Record<string, Record<string, unknown>>;
  hz: number;
  last_action: number[];           // length action_dim (7 for open_drawer)
  goal: [number, number, number];  // world frame
  object_pose: ObjectPose | null;
};

export type ControllerState = {
  available: ControllerInfo[];
  current: string;
  status: ControllerStatus;
  last_error: string;
  // Per-controller config dicts (keyed by name). Free-form: each
  // controller defines its own fields. Used for live gain UI + save/load.
  configs: Record<string, Record<string, number | number[] | string>>;
  // Latest EE target (shm_qtarget snapshot). Present in hardware mode;
  // null otherwise. The browser uses this to seed/refresh the EE-target
  // gizmo when it's not actively being dragged.
  ee_target?: EeTarget | null;
  // Policy block — present in hardware mode, null otherwise.
  policy?: PolicyState | null;
};

export type ControllerStateFrame = {
  kind: typeof KIND_CONTROLLER_STATE; seq: number;
  state: ControllerState;
};

export type LogLine = {
  ts: number;       // unix seconds
  level: string;    // 'INFO' | 'WARNING' | 'ERROR' | ...
  source: string;   // 'transport' | 'ctrl-gravcomp' | etc.
  msg: string;
};

export type LogLinesFrame = {
  kind: typeof KIND_LOG_LINES; seq: number;
  lines: LogLine[];
};

export type PointsFrame = {
  kind: typeof KIND_POINTS; seq: number;
  xyz: Float16Array;       // length 3n  (Float16 stored, exposed as f16)
  rgb: Uint8Array;         // length 3n, [0..255]
  mask: Uint8Array;        // length n, 0 or 1
  n: number;
};

export type MeshFrame = {
  kind: typeof KIND_MESH; seq: number;
  xyz: Float16Array;       // length 3nv
  rgb: Uint8Array;         // length 3nv
  faces: Uint32Array;      // length 3nf
  normal: Float16Array | null;  // length 3nv, or null if model has no normals
  nv: number; nf: number;
};

export type JpegFrame = {
  kind: typeof KIND_JPEG | typeof KIND_DEPTH_JPEG | typeof KIND_NORMAL_JPEG;
  seq: number;
  w: number; h: number; bytes: Uint8Array;
};

export type MetaFrame = {
  kind: typeof KIND_META; seq: number; meta: Meta;
};

export type ModelStateFrame = {
  kind: typeof KIND_MODEL_STATE; seq: number; state: ModelState;
};

export type SamStateFrame = {
  kind: typeof KIND_SAM_STATE; seq: number; state: SamState;
};

export type MaskFrame = {
  kind: typeof KIND_MASK; seq: number; maskSeq: number;
  mask: Uint8Array;
  hasBox: boolean;
  boxMin: Float32Array;
  boxMax: Float32Array;
};

export type StatsFrame = {
  kind: typeof KIND_STATS; seq: number; stats: Stats;
};

export type Frame =
  | PointsFrame | MeshFrame | JpegFrame | MetaFrame
  | ModelStateFrame | SamStateFrame | MaskFrame | StatsFrame
  | RobotGeometryFrame | RobotTransformsFrame | CamCalibFrame
  | RobotStatusFrame | ControllerStateFrame | LogLinesFrame;

const MAGIC = 0x46443350; // 'P3DF' little-endian

// Float16 polyfill for browsers without Float16Array (most don't yet).
// We expose Float16Array as a thin wrapper that returns a Float32Array.
export class Float16Array extends Float32Array {}

function f16BytesToF32(buf: ArrayBuffer, byteOffset: number, n: number): Float32Array {
  // Copy first so we don't depend on byteOffset being 2-byte aligned.
  const slice = buf.slice(byteOffset, byteOffset + n * 2);
  const u16 = new Uint16Array(slice);
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = halfToFloat(u16[i]);
  return f32;
}

// IEEE-754 half -> single. Adapted from typical reference impls.
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7C00) >> 10;
  const f = h & 0x03FF;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  } else if (e === 0x1F) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function parseFrame(buf: ArrayBuffer): Frame | null {
  if (buf.byteLength < 12) return null;
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) return null;
  const seq = dv.getUint32(4, true);
  const kind = dv.getUint8(8);
  let off = 12;
  switch (kind) {
    case KIND_POINTS: {
      const n = dv.getUint32(off, true); off += 4;
      const xyz = f16BytesToF32(buf, off, 3 * n) as unknown as Float16Array;
      off += 6 * n;
      const rgb = new Uint8Array(buf.slice(off, off + 3 * n)); off += 3 * n;
      const mask = new Uint8Array(buf.slice(off, off + n)); off += n;
      return { kind: KIND_POINTS, seq, xyz, rgb, mask, n };
    }
    case KIND_MESH: {
      const nv = dv.getUint32(off, true); off += 4;
      const nf = dv.getUint32(off, true); off += 4;
      const hasN = dv.getUint8(off) !== 0; off += 1;
      const xyz = f16BytesToF32(buf, off, 3 * nv) as unknown as Float16Array;
      off += 6 * nv;
      const rgb = new Uint8Array(buf.slice(off, off + 3 * nv)); off += 3 * nv;
      const faces = new Uint32Array(buf.slice(off, off + 12 * nf));
      off += 12 * nf;
      let normal: Float16Array | null = null;
      if (hasN) {
        normal = f16BytesToF32(buf, off, 3 * nv) as unknown as Float16Array;
        off += 6 * nv;
      }
      return { kind: KIND_MESH, seq, xyz, rgb, faces, normal, nv, nf };
    }
    case KIND_JPEG:
    case KIND_DEPTH_JPEG:
    case KIND_NORMAL_JPEG: {
      const w = dv.getUint16(off, true); off += 2;
      const h = dv.getUint16(off, true); off += 2;
      const bytes = new Uint8Array(buf, off);
      return {
        kind: kind as typeof KIND_JPEG | typeof KIND_DEPTH_JPEG | typeof KIND_NORMAL_JPEG,
        seq, w, h, bytes,
      };
    }
    case KIND_META: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const meta = JSON.parse(text) as Meta;
      return { kind: KIND_META, seq, meta };
    }
    case KIND_MODEL_STATE: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const ms = JSON.parse(text) as ModelState;
      return { kind: KIND_MODEL_STATE, seq, state: ms };
    }
    case KIND_SAM_STATE: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const ms = JSON.parse(text) as SamState;
      return { kind: KIND_SAM_STATE, seq, state: ms };
    }
    case KIND_MASK: {
      const maskSeq = dv.getUint32(off, true); off += 4;
      const n = dv.getUint32(off, true); off += 4;
      const mask = new Uint8Array(buf.slice(off, off + n)); off += n;
      const hasBox = dv.getUint8(off) !== 0; off += 1;
      const boxMin = new Float32Array(buf.slice(off, off + 12)); off += 12;
      const boxMax = new Float32Array(buf.slice(off, off + 12)); off += 12;
      return { kind: KIND_MASK, seq, maskSeq, mask, hasBox, boxMin, boxMax };
    }
    case KIND_STATS: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const stats = JSON.parse(text) as Stats;
      return { kind: KIND_STATS, seq, stats };
    }
    case KIND_ROBOT_GEOMETRY: {
      const jsonLen = dv.getUint32(off, true); off += 4;
      const jsonBytes = new Uint8Array(buf, off, jsonLen); off += jsonLen;
      const text = new TextDecoder().decode(jsonBytes);
      const j = JSON.parse(text) as {
        bodies: RobotBody[]; meshes: RobotMeshIndex[]; geoms: RobotGeom[];
      };
      // The mesh blob is the rest of the buffer. Slice into a standalone
      // ArrayBuffer so callers don't have to track byteOffset.
      const blob = buf.slice(off);
      return {
        kind: KIND_ROBOT_GEOMETRY, seq,
        bodies: j.bodies, meshes: j.meshes, geoms: j.geoms,
        blob,
      };
    }
    case KIND_ROBOT_TRANSFORMS: {
      const nbody = dv.getUint32(off, true); off += 4;
      const xpos = new Float32Array(buf.slice(off, off + 12 * nbody));
      off += 12 * nbody;
      const xquat = new Float32Array(buf.slice(off, off + 16 * nbody));
      off += 16 * nbody;
      return { kind: KIND_ROBOT_TRANSFORMS, seq, xpos, xquat, nbody };
    }
    case KIND_CAM_CALIB: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const calib = JSON.parse(text) as CamCalibPayload;
      return { kind: KIND_CAM_CALIB, seq, calib };
    }
    case KIND_ROBOT_STATUS: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const status = JSON.parse(text) as RobotStatus;
      return { kind: KIND_ROBOT_STATUS, seq, status };
    }
    case KIND_CONTROLLER_STATE: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const state = JSON.parse(text) as ControllerState;
      return { kind: KIND_CONTROLLER_STATE, seq, state };
    }
    case KIND_LOG_LINES: {
      const text = new TextDecoder().decode(new Uint8Array(buf, off));
      const lines = JSON.parse(text) as LogLine[];
      return { kind: KIND_LOG_LINES, seq, lines };
    }
  }
  return null;
}
