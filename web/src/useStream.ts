import { useEffect, useRef, useState } from "react";
import { Frame, KIND_JPEG, KIND_MESH, KIND_META, KIND_POINTS, Meta, parseFrame } from "./protocol";

export type StreamState = {
  meta: Meta | null;
  pointsRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; n: number; seq: number } | null>;
  meshRef: React.MutableRefObject<{ xyz: Float32Array; rgb: Uint8Array; faces: Uint32Array; nv: number; nf: number; seq: number } | null>;
  jpegRef: React.MutableRefObject<{ blobUrl: string | null; seq: number }>;
  connected: boolean;
};

export function useStream(url: string): StreamState {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [connected, setConnected] = useState(false);
  const pointsRef = useRef<StreamState["pointsRef"]["current"]>(null);
  const meshRef = useRef<StreamState["meshRef"]["current"]>(null);
  const jpegRef = useRef<{ blobUrl: string | null; seq: number }>({ blobUrl: null, seq: -1 });

  useEffect(() => {
    let ws: WebSocket | null = null;
    let cancelled = false;

    const connect = () => {
      ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) setTimeout(connect, 500);
      };
      ws.onerror = () => ws?.close();
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") return;
        const frame: Frame | null = parseFrame(ev.data as ArrayBuffer);
        if (!frame) return;
        switch (frame.kind) {
          case KIND_META:
            setMeta(frame.meta);
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
          case KIND_JPEG: {
            const blob = new Blob([frame.bytes], { type: "image/jpeg" });
            const url = URL.createObjectURL(blob);
            const old = jpegRef.current.blobUrl;
            jpegRef.current = { blobUrl: url, seq: frame.seq };
            if (old) URL.revokeObjectURL(old);
            break;
          }
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      ws?.close();
    };
  }, [url]);

  return { meta, pointsRef, meshRef, jpegRef, connected };
}
