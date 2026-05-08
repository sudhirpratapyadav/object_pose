import { useEffect, useRef, useState } from "react";
import { Viewer } from "./Viewer";
import { StreamState, useStream } from "./useStream";

export default function App() {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const stream = useStream(wsUrl);

  const [pointSize, setPointSize] = useState(0.01);
  const [inversePerspective, setInversePerspective] = useState(true);
  const [display, setDisplay] = useState<"points" | "mesh" | "both">("points");
  const [showCamera, setShowCamera] = useState(true);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [depthUrl, setDepthUrl] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let lastRgb = -1, lastDepth = -1;
    const tick = () => {
      const j = stream.jpegRef.current;
      if (j.seq !== lastRgb && j.blobUrl) {
        lastRgb = j.seq;
        setImgUrl(j.blobUrl);
      }
      const d = stream.depthJpegRef.current;
      if (d.seq !== lastDepth && d.blobUrl) {
        lastDepth = d.seq;
        setDepthUrl(d.blobUrl);
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    let raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [stream]);

  const models = stream.meta?.models ?? [];
  const currentModel = stream.modelState?.model ?? stream.meta?.default_model ?? "";
  const samModels = stream.meta?.sam_models ?? [];
  const currentSamModel = stream.samState?.model ?? stream.meta?.sam_default_model ?? "";

  const onSegClickFromElement = (rect: DOMRect, clientX: number, clientY: number) => {
    if (!stream.meta) return;
    const px = (clientX - rect.left) / rect.width;
    const py = (clientY - rect.top) / rect.height;
    // Click is sent in INFERENCE-frame pixels (matches backend convention).
    const x = Math.round(px * stream.meta.infer_w);
    const y = Math.round(py * stream.meta.infer_h);
    stream.samClick(x, y);
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", height: "100%" }}>
      <div style={{ padding: 12, borderRight: "1px solid #1f242b", overflow: "auto" }}>
        <h2 style={{ margin: "4px 0 12px" }}>Object Pose</h2>
        <div style={{ marginBottom: 8, fontSize: 12, color: stream.connected ? "#7ee787" : "#ff7b72" }}>
          {stream.connected ? "● connected" : "● disconnected"} ({wsUrl})
        </div>

        <Section title="Depth model">
          <select
            value={currentModel}
            disabled={!models.length}
            onChange={(e) => stream.setModel(e.target.value)}
            style={{ width: "100%", padding: 4, background: "#11151a", color: "#e5e7eb", border: "1px solid #1f242b" }}
          >
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <div style={{ fontSize: 11, opacity: 0.7, marginTop: 4 }}>
            {stream.modelState ? renderStatus(stream.modelState) : "—"}
          </div>
        </Section>

        <Section title="Display">
          {(["points", "mesh", "both"] as const).map((m) => (
            <label key={m} style={{ marginRight: 8 }}>
              <input
                type="radio"
                name="display"
                checked={display === m}
                onChange={() => setDisplay(m)}
              />
              {m}
            </label>
          ))}
        </Section>

        <Section title="Point size">
          <Slider min={0.001} max={0.1} step={0.001} value={pointSize} onChange={setPointSize} />
          <label style={{ display: "block", marginTop: 4 }}>
            <input
              type="checkbox"
              checked={inversePerspective}
              onChange={(e) => setInversePerspective(e.target.checked)}
            />
            Inverse-perspective
          </label>
          <div style={{ fontSize: 11, opacity: 0.6 }}>
            On: every point renders at the same screen size regardless of depth.
            Off: world-space size, far points shrink with perspective.
          </div>
        </Section>

        <Section title="Scene">
          <label>
            <input
              type="checkbox"
              checked={showCamera}
              onChange={(e) => setShowCamera(e.target.checked)}
            />
            Camera axes
          </label>
        </Section>

        <Section title="Segmentation (SAM2)">
          <select
            value={currentSamModel}
            disabled={!samModels.length}
            onChange={(e) => stream.setSamModel(e.target.value)}
            style={{ width: "100%", padding: 4, background: "#11151a",
                     color: "#e5e7eb", border: "1px solid #1f242b" }}
          >
            {samModels.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <div style={{ fontSize: 11, opacity: 0.7, marginTop: 4 }}>
            {stream.samState ? renderSamStatus(stream.samState) : "—"}
          </div>
          <div style={{ fontSize: 11, opacity: 0.6, marginTop: 4 }}>
            Click anywhere on the RGB image to segment.
          </div>
          <button
            onClick={() => stream.samClear()}
            style={{ marginTop: 6, width: "100%", padding: 4,
                     background: "#11151a", color: "#e5e7eb",
                     border: "1px solid #1f242b", cursor: "pointer" }}
          >
            Clear selection
          </button>
        </Section>

        <Section title="RGB">
          {imgUrl
            ? <img src={imgUrl}
                   style={{ width: "100%", display: "block" }} />
            : <div style={{ color: "#666" }}>—</div>}
        </Section>

        <Section title="Segmentation (mask overlay)">
          <MaskedRgb
            imgUrl={imgUrl}
            stream={stream}
            onClick={(rect, x, y) => onSegClickFromElement(rect, x, y)}
          />
        </Section>

        <Section title="Depth">
          {depthUrl
            ? <img src={depthUrl} style={{ width: "100%", display: "block" }} />
            : <div style={{ color: "#666" }}>—</div>}
        </Section>

        <Section title="Stream">
          <div style={{ fontSize: 11, opacity: 0.7 }}>
            {stream.meta
              ? `${stream.meta.rgb_w}x${stream.meta.rgb_h} @ ${stream.meta.viz_hz} Hz`
              : "waiting for meta..."}
          </div>
        </Section>
      </div>
      <div style={{ position: "relative" }}>
        <Viewer
          stream={stream}
          pointSize={pointSize}
          inversePerspective={inversePerspective}
          display={display}
          showCamera={showCamera}
        />
      </div>
    </div>
  );
}

function renderStatus(s: { status: string; progress: string; file: string }): string {
  if (s.status === "downloading") {
    const pct = s.progress ? `${s.progress}%` : "";
    return `downloading ${s.file} ${pct}`.trim();
  }
  if (s.status === "error") return `error: ${s.file}`;
  return s.status;
}

function renderSamStatus(s: { status: string; file: string }): string {
  if (s.status === "downloading") return `downloading ${s.file}`.trim();
  if (s.status === "error") return `error: ${s.file}`;
  return s.status;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function MaskedRgb({ imgUrl, stream, onClick }: {
  imgUrl: string | null;
  stream: StreamState;
  onClick: (rect: DOMRect, clientX: number, clientY: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const lastBlobRef = useRef<string | null>(null);

  useEffect(() => {
    let alive = true;
    const draw = () => {
      const canvas = canvasRef.current;
      const meta = stream.meta;
      if (!canvas || !meta) {
        if (alive) raf = requestAnimationFrame(draw);
        return;
      }

      // Lazy-create the <img>; reuse the same element when blob URL changes.
      if (imgUrl && lastBlobRef.current !== imgUrl) {
        const img = new Image();
        img.src = imgUrl;
        imgRef.current = img;
        lastBlobRef.current = imgUrl;
      }
      const img = imgRef.current;

      const W = canvas.clientWidth || meta.rgb_w;
      const targetH = Math.round(W * meta.rgb_h / meta.rgb_w);
      if (canvas.width !== W || canvas.height !== targetH) {
        canvas.width = W;
        canvas.height = targetH;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        if (alive) raf = requestAnimationFrame(draw);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (img && img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      }

      // Overlay the mask. Mask is at mesh_grid_w x mesh_grid_h; build a small
      // ImageData and stretch-blit it on top with multiply/source-over alpha.
      const m = stream.maskRef.current;
      if (m && m.mask.length === meta.mesh_grid_w * meta.mesh_grid_h) {
        const gw = meta.mesh_grid_w, gh = meta.mesh_grid_h;
        const overlay = ctx.createImageData(gw, gh);
        const buf = overlay.data;
        // Highlight = orange (255, 170, 51), alpha ~140.
        for (let i = 0; i < gw * gh; i++) {
          const on = m.mask[i] !== 0;
          buf[i * 4 + 0] = 255;
          buf[i * 4 + 1] = 170;
          buf[i * 4 + 2] = 51;
          buf[i * 4 + 3] = on ? 140 : 0;
        }
        // Use an offscreen canvas to upscale with smoothing off (so mask edges
        // align to grid cells, no halos).
        const off = document.createElement("canvas");
        off.width = gw; off.height = gh;
        const offCtx = off.getContext("2d");
        if (offCtx) {
          offCtx.putImageData(overlay, 0, 0);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
        }
      }

      if (alive) raf = requestAnimationFrame(draw);
    };
    let raf = requestAnimationFrame(draw);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [stream, imgUrl]);

  return (
    <canvas
      ref={canvasRef}
      onClick={(e) =>
        onClick(e.currentTarget.getBoundingClientRect(), e.clientX, e.clientY)
      }
      style={{ width: "100%", display: "block", cursor: "crosshair",
               background: "#11151a" }}
    />
  );
}

function Slider({ min, max, step, value, onChange }: {
  min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%" }}
      />
      <div style={{ fontSize: 11, opacity: 0.7 }}>{value.toFixed(3)}</div>
    </div>
  );
}
