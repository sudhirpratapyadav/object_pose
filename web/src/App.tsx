import { useEffect, useState } from "react";
import { Viewer } from "./Viewer";
import { useStream } from "./useStream";

export default function App() {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const stream = useStream(wsUrl);

  const [pointSize, setPointSize] = useState(0.005);          // base meters
  const [depthSizeFactor, setDepthSizeFactor] = useState(0.0); // far-bigger
  const [display, setDisplay] = useState<"points" | "mesh" | "both">("points");
  const [showCamera, setShowCamera] = useState(true);
  const [imgUrl, setImgUrl] = useState<string | null>(null);

  // Pull most recent jpeg URL into state (~30fps)
  useEffect(() => {
    let alive = true;
    let last = -1;
    const tick = () => {
      const j = stream.jpegRef.current;
      if (j.seq !== last && j.blobUrl) {
        last = j.seq;
        setImgUrl(j.blobUrl);
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    let raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [stream]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", height: "100%" }}>
      <div style={{ padding: 12, borderRight: "1px solid #1f242b", overflow: "auto" }}>
        <h2 style={{ margin: "4px 0 12px" }}>Object Pose</h2>
        <div style={{ marginBottom: 8, fontSize: 12, color: stream.connected ? "#7ee787" : "#ff7b72" }}>
          {stream.connected ? "● connected" : "● disconnected"} ({wsUrl})
        </div>

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

        <Section title="Point base size (m)">
          <Slider
            min={0.001} max={0.05} step={0.001}
            value={pointSize}
            onChange={setPointSize}
          />
        </Section>

        <Section title="Far-bigger factor (per m)">
          <Slider
            min={0} max={2} step={0.05}
            value={depthSizeFactor}
            onChange={setDepthSizeFactor}
          />
          <div style={{ fontSize: 11, opacity: 0.6 }}>
            0 = uniform world size; &gt;0 makes far points larger.
          </div>
        </Section>

        <Section title="Scene">
          <label>
            <input
              type="checkbox"
              checked={showCamera}
              onChange={(e) => setShowCamera(e.target.checked)}
            />
            Show camera
          </label>
        </Section>

        <Section title="RGB">
          {imgUrl
            ? <img src={imgUrl} style={{ width: "100%", display: "block" }} />
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
          depthSizeFactor={depthSizeFactor}
          display={display}
          showCamera={showCamera}
        />
      </div>
    </div>
  );
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
