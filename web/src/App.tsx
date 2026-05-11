import { useEffect, useState } from "react";
import { Viewer } from "./Viewer";
import { useStream } from "./useStream";
import { Card } from "./ui/Card";
import { StatusBar } from "./ui/StatusBar";
import { TabKey, TabStrip } from "./ui/TabStrip";
import { VisionTab } from "./tabs/VisionTab";
import { RobotTab } from "./tabs/RobotTab";
import { PolicyTab } from "./tabs/PolicyTab";
import { DiagnosticsTab } from "./tabs/DiagnosticsTab";

export default function App() {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const stream = useStream(wsUrl);

  // Viewer-state lifted here so it persists across tab changes (the
  // <Viewer> itself is rendered once as a fixed-position background and
  // the Vision tab's controls write to this same state).
  const [pointSize, setPointSize] = useState(0.01);
  const [inversePerspective, setInversePerspective] = useState(true);
  const [display, setDisplay] = useState<"points" | "mesh" | "both">("points");
  const [showCamera, setShowCamera] = useState(false);
  const [showBBox, setShowBBox] = useState(false);
  const [showRobot, setShowRobot] = useState(true);
  const [showEeAxes, setShowEeAxes] = useState(true);
  const [showWorldAxes, setShowWorldAxes] = useState(false);
  const [showWorldHandle, setShowWorldHandle] = useState(false);
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate">("translate");

  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [depthUrl, setDepthUrl] = useState<string | null>(null);
  const [normalUrl, setNormalUrl] = useState<string | null>(null);
  const [hideHud, setHideHud] = useState(false);
  const [tab, setTab] = useState<TabKey>("vision");

  // Keyboard shortcuts: H toggles HUD, T/R switch the camera-pose gizmo.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "h" || e.key === "H") setHideHud((v) => !v);
      else if (e.key === "t" || e.key === "T") setGizmoMode("translate");
      else if (e.key === "r" || e.key === "R") setGizmoMode("rotate");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Pull blob-url state from refs at ~30fps without re-rendering everything.
  useEffect(() => {
    let alive = true;
    let lastRgb = -1, lastDepth = -1, lastNormal = -1;
    const tick = () => {
      const j = stream.jpegRef.current;
      if (j.seq !== lastRgb && j.blobUrl) { lastRgb = j.seq; setImgUrl(j.blobUrl); }
      const d = stream.depthJpegRef.current;
      if (d.seq !== lastDepth && d.blobUrl) { lastDepth = d.seq; setDepthUrl(d.blobUrl); }
      const n = stream.normalJpegRef.current;
      if (n.seq !== lastNormal && n.blobUrl) { lastNormal = n.seq; setNormalUrl(n.blobUrl); }
      if (alive) raf = requestAnimationFrame(tick);
    };
    let raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [stream]);

  const onSegClick = (rect: DOMRect, clientX: number, clientY: number) => {
    if (!stream.meta) return;
    const px = (clientX - rect.left) / rect.width;
    const py = (clientY - rect.top) / rect.height;
    const x = Math.round(px * stream.meta.infer_w);
    const y = Math.round(py * stream.meta.infer_h);
    stream.samClick(x, y);
  };

  const cardTitle =
    tab === "vision" ? "Vision"
      : tab === "robot" ? "Robot"
      : tab === "policy" ? "Policy"
      : "Diagnostics";

  return (
    <div style={{ position: "fixed", inset: 0 }}>
      {/* 3D viewer fills the whole window as the background. */}
      <div style={{ position: "fixed", inset: 0, zIndex: 0 }}>
        <Viewer
          stream={stream}
          pointSize={pointSize}
          inversePerspective={inversePerspective}
          display={display}
          showCamera={showCamera}
          showBBox={showBBox}
          showRobot={showRobot}
          showEeAxes={showEeAxes}
          showWorldAxes={showWorldAxes}
          showWorldHandle={showWorldHandle}
          gizmoMode={gizmoMode}
        />
      </div>

      {/* Persistent top status bar. */}
      <StatusBar stream={stream} />

      {/* Tab strip below the status bar. */}
      <TabStrip active={tab} onChange={setTab} />

      {/* Tab content lives in a single floating Card so the look stays
          consistent across tabs. The card sits in the existing top-left
          slot, just shifted down to make room for the bar + strip. */}
      <Card
        id={`tab-${tab}`}
        title={cardTitle}
        slot="slot-tl-tabbed"
        hidden={hideHud}
        headerRight={
          <span
            className={`conn-dot ${stream.connected ? "ok" : "bad"}`}
            title={`${stream.connected ? "Connected" : "Disconnected"} — ${wsUrl}`}
          />
        }
      >
        {tab === "vision" && (
          <VisionTab
            stream={stream}
            pointSize={pointSize} setPointSize={setPointSize}
            inversePerspective={inversePerspective} setInversePerspective={setInversePerspective}
            display={display} setDisplay={setDisplay}
            showCamera={showCamera} setShowCamera={setShowCamera}
            showBBox={showBBox} setShowBBox={setShowBBox}
            showRobot={showRobot} setShowRobot={setShowRobot}
            showEeAxes={showEeAxes} setShowEeAxes={setShowEeAxes}
            showWorldAxes={showWorldAxes} setShowWorldAxes={setShowWorldAxes}
            showWorldHandle={showWorldHandle} setShowWorldHandle={setShowWorldHandle}
            gizmoMode={gizmoMode} setGizmoMode={setGizmoMode}
            imgUrl={imgUrl} depthUrl={depthUrl} normalUrl={normalUrl}
            onSegClick={onSegClick}
          />
        )}
        {tab === "robot" && <RobotTab stream={stream} />}
        {tab === "policy" && <PolicyTab stream={stream} />}
        {tab === "diagnostics" && <DiagnosticsTab stream={stream} />}

        <div className="help" style={{ marginTop: 8 }}>press H to toggle HUD</div>
      </Card>
    </div>
  );
}
