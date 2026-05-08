import { useEffect, useState } from "react";
import { Viewer } from "./Viewer";
import { useStream } from "./useStream";

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

        <Section title="RGB">
          {imgUrl
            ? <img src={imgUrl} style={{ width: "100%", display: "block" }} />
            : <div style={{ color: "#666" }}>—</div>}
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
