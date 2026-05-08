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

export type ModelState = {
  model: string;
  status: string;
  progress: string;
  file: string;
};

export type Meta = {
  rgb_w: number; rgb_h: number;
  infer_w: number; infer_h: number;
  fx: number; fy: number; cx: number; cy: number;
  fx_infer: number; fy_infer: number; cx_infer: number; cy_infer: number;
  mesh_grid_w: number; mesh_grid_h: number;
  viz_hz: number;
  models: string[];
  default_model: string;
  model_state?: ModelState;
};

export type PointsFrame = {
  kind: typeof KIND_POINTS; seq: number;
  xyz: Float16Array;       // length 3n  (Float16 stored, exposed as f16)
  rgb: Uint8Array;         // length 3n, [0..255]
  n: number;
};

export type MeshFrame = {
  kind: typeof KIND_MESH; seq: number;
  xyz: Float16Array;       // length 3nv
  rgb: Uint8Array;         // length 3nv
  faces: Uint32Array;      // length 3nf
  nv: number; nf: number;
};

export type JpegFrame = {
  kind: typeof KIND_JPEG | typeof KIND_DEPTH_JPEG; seq: number;
  w: number; h: number; bytes: Uint8Array;
};

export type MetaFrame = {
  kind: typeof KIND_META; seq: number; meta: Meta;
};

export type ModelStateFrame = {
  kind: typeof KIND_MODEL_STATE; seq: number; state: ModelState;
};

export type Frame = PointsFrame | MeshFrame | JpegFrame | MetaFrame | ModelStateFrame;

const MAGIC = 0x46443350; // 'P3DF' little-endian

// Float16 polyfill for browsers without Float16Array (most don't yet).
// We expose Float16Array as a thin wrapper that returns a Float32Array.
export class Float16Array extends Float32Array {}

function f16BytesToF32(buf: ArrayBuffer, byteOffset: number, n: number): Float32Array {
  const u16 = new Uint16Array(buf, byteOffset, n);
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
      const rgb = new Uint8Array(buf, off, 3 * n);
      return { kind: KIND_POINTS, seq, xyz, rgb, n };
    }
    case KIND_MESH: {
      const nv = dv.getUint32(off, true); off += 4;
      const nf = dv.getUint32(off + 0, true); off += 4;
      const xyz = f16BytesToF32(buf, off, 3 * nv) as unknown as Float16Array;
      off += 6 * nv;
      const rgb = new Uint8Array(buf.slice(off, off + 3 * nv)); off += 3 * nv;
      // Faces follow at u32 alignment. Bytes-position must be multiple of 4.
      // 6*nv is always even but not always /4; rgb is 3*nv bytes; total may be odd.
      const faces = new Uint32Array(buf.slice(off, off + 12 * nf));
      return { kind: KIND_MESH, seq, xyz, rgb, faces, nv, nf };
    }
    case KIND_JPEG:
    case KIND_DEPTH_JPEG: {
      const w = dv.getUint16(off, true); off += 2;
      const h = dv.getUint16(off, true); off += 2;
      const bytes = new Uint8Array(buf, off);
      return { kind: kind as typeof KIND_JPEG | typeof KIND_DEPTH_JPEG,
               seq, w, h, bytes };
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
  }
  return null;
}
