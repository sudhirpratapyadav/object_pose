import { useCallback, useEffect, useRef, useState } from "react";
import {
  Frame, KIND_DEPTH_JPEG, KIND_JPEG, KIND_MASK, KIND_MESH, KIND_META,
  KIND_MODEL_STATE, KIND_NORMAL_JPEG, KIND_POINTS, KIND_SAM_STATE, KIND_STATS,
  Meta, ModelState, SamState, Stats, parseFrame,
} from "./protocol";

type JpegRef = React.MutableRefObject<{ blobUrl: string | null; seq: number }>;

export type MaskData = {
  mask: Uint8Array;
  hasBox: boolean;
  boxMin: Float32Array;
  boxMax: Float32Array;
  seq: number;
};

export type StreamState = {
  meta: Meta | null;
  modelState: ModelState | null;
  samState: SamState | null;
  stats: Stats | null;
  pointsRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; mask: Uint8Array; n: number; seq: number } | null>;
  meshRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; faces: Uint32Array; normal: Float32Array | null; nv: number; nf: number; seq: number } | null>;
  jpegRef: JpegRef;
  depthJpegRef: JpegRef;
  normalJpegRef: JpegRef;
  maskRef: React.MutableRefObject<MaskData | null>;
  connected: boolean;
  setModel: (key: string) => void;
  setSamModel: (key: string) => void;
  samClick: (x: number, y: number) => void;
  samClear: () => void;
};

export function useStream(url: string): StreamState {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [samState, setSamState] = useState<SamState | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const pointsRef = useRef<StreamState["pointsRef"]["current"]>(null);
  const meshRef = useRef<StreamState["meshRef"]["current"]>(null);
  const jpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });
  const depthJpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });
  const normalJpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });
  const maskRef = useRef<MaskData | null>(null);

  useEffect(() => {
    let cancelled = false;

    // Defer URL.revokeObjectURL so any <img> still pointing at the previous
    // blob has time to swap in the new one before the old URL is invalidated.
    // Without this, slow producers cause a flash to "broken image" between
    // the new URL being chosen and the new bytes being decoded.
    const setBlob = (ref: JpegRef, bytes: Uint8Array, seq: number) => {
      const blob = new Blob([bytes], { type: "image/jpeg" });
      const u = URL.createObjectURL(blob);
      const old = ref.current.blobUrl;
      ref.current = { blobUrl: u, seq };
      if (old) setTimeout(() => URL.revokeObjectURL(old), 500);
    };

    const connect = () => {
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) setTimeout(connect, 500);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") return;
        const frame: Frame | null = parseFrame(ev.data as ArrayBuffer);
        if (!frame) return;
        switch (frame.kind) {
          case KIND_META:
            setMeta(frame.meta);
            if (frame.meta.model_state) setModelState(frame.meta.model_state);
            if (frame.meta.sam_state) setSamState(frame.meta.sam_state);
            break;
          case KIND_MODEL_STATE:
            setModelState(frame.state);
            break;
          case KIND_SAM_STATE:
            setSamState(frame.state);
            break;
          case KIND_STATS:
            setStats(frame.stats);
            break;
          case KIND_MASK:
            maskRef.current = {
              mask: frame.mask,
              hasBox: frame.hasBox,
              boxMin: frame.boxMin,
              boxMax: frame.boxMax,
              seq: frame.maskSeq,
            };
            break;
          case KIND_POINTS:
            pointsRef.current = {
              xyz: frame.xyz as unknown as Float32Array,
              rgb: frame.rgb,
              mask: frame.mask,
              n: frame.n,
              seq: frame.seq,
            };
            break;
          case KIND_MESH:
            meshRef.current = {
              xyz: frame.xyz as unknown as Float32Array,
              rgb: frame.rgb,
              faces: frame.faces,
              normal: (frame.normal as unknown as Float32Array | null),
              nv: frame.nv,
              nf: frame.nf,
              seq: frame.seq,
            };
            break;
          case KIND_JPEG:
            setBlob(jpegRef, frame.bytes, frame.seq);
            break;
          case KIND_DEPTH_JPEG:
            setBlob(depthJpegRef, frame.bytes, frame.seq);
            break;
          case KIND_NORMAL_JPEG:
            setBlob(normalJpegRef, frame.bytes, frame.seq);
            break;
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [url]);

  const send = (obj: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  };
  const setModel = useCallback((key: string) => send({ set_model: key }), []);
  const setSamModel = useCallback((key: string) => send({ set_sam_model: key }), []);
  const samClick = useCallback((x: number, y: number) => send({ sam_click: { x, y } }), []);
  const samClear = useCallback(() => send({ sam_clear: true }), []);

  return {
    meta, modelState, samState, stats,
    pointsRef, meshRef, jpegRef, depthJpegRef, normalJpegRef, maskRef,
    connected, setModel, setSamModel, samClick, samClear,
  };
}
