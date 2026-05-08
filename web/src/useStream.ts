import { useCallback, useEffect, useRef, useState } from "react";
import {
  Frame, KIND_DEPTH_JPEG, KIND_JPEG, KIND_MESH, KIND_META, KIND_MODEL_STATE,
  KIND_POINTS, Meta, ModelState, parseFrame,
} from "./protocol";

type JpegRef = React.MutableRefObject<{ blobUrl: string | null; seq: number }>;

export type StreamState = {
  meta: Meta | null;
  modelState: ModelState | null;
  pointsRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; n: number; seq: number } | null>;
  meshRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; faces: Uint32Array; nv: number; nf: number; seq: number } | null>;
  jpegRef: JpegRef;
  depthJpegRef: JpegRef;
  connected: boolean;
  setModel: (key: string) => void;
};

export function useStream(url: string): StreamState {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const pointsRef = useRef<StreamState["pointsRef"]["current"]>(null);
  const meshRef = useRef<StreamState["meshRef"]["current"]>(null);
  const jpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });
  const depthJpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });

  useEffect(() => {
    let cancelled = false;

    const setBlob = (ref: JpegRef, bytes: Uint8Array, seq: number) => {
      const blob = new Blob([bytes], { type: "image/jpeg" });
      const u = URL.createObjectURL(blob);
      const old = ref.current.blobUrl;
      ref.current = { blobUrl: u, seq };
      if (old) URL.revokeObjectURL(old);
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
            break;
          case KIND_MODEL_STATE:
            setModelState(frame.state);
            break;
          case KIND_POINTS:
            pointsRef.current = {
              xyz: frame.xyz as unknown as Float32Array,
              rgb: frame.rgb,
              n: frame.n,
              seq: frame.seq,
            };
            break;
          case KIND_MESH:
            meshRef.current = {
              xyz: frame.xyz as unknown as Float32Array,
              rgb: frame.rgb,
              faces: frame.faces,
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
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [url]);

  const setModel = useCallback((key: string) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ set_model: key }));
    }
  }, []);

  return { meta, modelState, pointsRef, meshRef, jpegRef, depthJpegRef, connected, setModel };
}
