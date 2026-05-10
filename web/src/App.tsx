import { useEffect, useRef, useState } from "react";
import { CamCalibPanel } from "./CamCalibPanel";
import { HardwareStatusPanel } from "./HardwareStatusPanel";
import { JointTargetsPanel } from "./JointTargetsPanel";
import { Viewer } from "./Viewer";
import { StreamState, useStream } from "./useStream";
import { Card } from "./ui/Card";

export default function App() {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const stream = useStream(wsUrl);

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
      if (j.seq !== lastRgb && j.blobUrl) {
        lastRgb = j.seq;
        setImgUrl(j.blobUrl);
      }
      const d = stream.depthJpegRef.current;
      if (d.seq !== lastDepth && d.blobUrl) {
        lastDepth = d.seq;
        setDepthUrl(d.blobUrl);
      }
      const n = stream.normalJpegRef.current;
      if (n.seq !== lastNormal && n.blobUrl) {
        lastNormal = n.seq;
        setNormalUrl(n.blobUrl);
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
    const x = Math.round(px * stream.meta.infer_w);
    const y = Math.round(py * stream.meta.infer_h);
    stream.samClick(x, y);
  };

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

      <Card
        id="controls"
        title="Controls"
        slot="slot-tl"
        hidden={hideHud}
        headerRight={
          <span
            className={`conn-dot ${stream.connected ? "ok" : "bad"}`}
            title={`${stream.connected ? "Connected" : "Disconnected"} — ${wsUrl}`}
          />
        }
      >
        <div className="row">
          <div className="label">Source</div>
          <div className="segmented">
            <button
              className={stream.meta?.source.kind === "live" ? "active" : ""}
              onClick={() => stream.setSource("live")}
            >
              live
            </button>
            <button
              className={stream.meta?.source.kind === "video" ? "active" : ""}
              onClick={() => {
                const v = stream.meta?.source.video
                  ?? stream.meta?.videos?.[0]
                  ?? null;
                if (v) stream.setSource("video", v);
              }}
              disabled={!stream.meta?.videos?.length}
            >
              video
            </button>
          </div>
          {stream.meta?.source.kind === "video" && (
            <select
              className="select"
              value={stream.meta?.source.video ?? ""}
              disabled={!stream.meta?.videos?.length}
              onChange={(e) => stream.setSource("video", e.target.value)}
            >
              {(stream.meta?.videos ?? []).map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          )}
        </div>

        <div className="row">
          <div className="label">Depth model</div>
          <select
            className="select"
            value={currentModel}
            disabled={!models.length}
            onChange={(e) => stream.setModel(e.target.value)}
          >
            {models.map((m) => {
              const isCamera = m === "camera-depth";
              const label = isCamera
                ? (stream.meta?.camera_depth_label ?? "Camera depth")
                : m;
              const disabled = isCamera && !stream.meta?.camera_depth_available;
              return (
                <option key={m} value={m} disabled={disabled}>
                  {label}{disabled ? " (unavailable)" : ""}
                </option>
              );
            })}
          </select>
          <div className="help">{stream.modelState ? renderStatus(stream.modelState) : "—"}</div>
        </div>

        <div className="row">
          <div className="label">Display</div>
          <div className="segmented">
            {(["points", "mesh", "both"] as const).map((m) => (
              <button
                key={m}
                className={display === m ? "active" : ""}
                onClick={() => setDisplay(m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="row">
          <div className="label">Point size · {pointSize.toFixed(3)}</div>
          <input
            className="range"
            type="range" min={0.001} max={0.1} step={0.001}
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={inversePerspective}
              onChange={(e) => setInversePerspective(e.target.checked)}
            />
            Inverse-perspective
          </label>
        </div>

        <div className="row">
          <label className="toggle">
            <input
              type="checkbox"
              checked={showCamera}
              onChange={(e) => setShowCamera(e.target.checked)}
            />
            Camera axes
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showWorldAxes}
              onChange={(e) => setShowWorldAxes(e.target.checked)}
            />
            World axes
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showWorldHandle}
              onChange={(e) => setShowWorldHandle(e.target.checked)}
            />
            World handle
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showBBox}
              onChange={(e) => setShowBBox(e.target.checked)}
            />
            Show 3D bounding box
          </label>
        </div>

        {showCamera && (
          <div className="row">
            <div className="label">Gizmo</div>
            <div className="segmented">
              <button
                className={gizmoMode === "translate" ? "active" : ""}
                onClick={() => setGizmoMode("translate")}
                title="Drag colored arrows to translate (T)"
              >
                Translate
              </button>
              <button
                className={gizmoMode === "rotate" ? "active" : ""}
                onClick={() => setGizmoMode("rotate")}
                title="Drag colored rings to rotate (R)"
              >
                Rotate
              </button>
            </div>
          </div>
        )}


        {stream.meta?.robot?.enabled && (
          <div className="row">
            <label className="toggle">
              <input
                type="checkbox"
                checked={showRobot}
                onChange={(e) => setShowRobot(e.target.checked)}
              />
              Show robot
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={showEeAxes}
                onChange={(e) => setShowEeAxes(e.target.checked)}
              />
              Show EE axes
            </label>
            <span style={{ opacity: 0.6, fontSize: "0.85em", marginLeft: 8 }}>
              {stream.meta.robot.source}
            </span>
          </div>
        )}

        <CamCalibPanel stream={stream} />

        <JointTargetsPanel stream={stream} />

        <HardwareStatusPanel stream={stream} />

        <div className="row">
          <div className="label">SAM2 model</div>
          <select
            className="select"
            value={currentSamModel}
            disabled={!samModels.length}
            onChange={(e) => stream.setSamModel(e.target.value)}
          >
            {samModels.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <div className="kv">
            <span>{stream.samState ? renderSamStatus(stream.samState) : "—"}</span>
            <span className="kv-val mono">
              {stream.stats && stream.stats.sam_ms > 0
                ? `${stream.stats.sam_ms} ms`
                : "—"}
            </span>
          </div>
        </div>

        <div className="row">
          <div className="label">RGB · click to segment</div>
          <div className="img-frame">
            <MaskedRgb
              imgUrl={imgUrl}
              stream={stream}
              onClick={(rect, x, y) => onSegClickFromElement(rect, x, y)}
            />
            <span className="badge">
              <b>{stream.stats ? stream.stats.rgb_fps.toFixed(1) : "—"}</b>
              {" fps "}
              <span className="dim">
                {stream.meta ? `· ${stream.meta.rgb_w}×${stream.meta.rgb_h}` : ""}
              </span>
            </span>
          </div>
        </div>

        <div className="row">
          <div className="label">Depth</div>
          {depthUrl ? (
            <div className="img-frame">
              <img src={depthUrl} className="thumb" />
              <span className="badge">
                <b>{stream.stats ? stream.stats.depth_fps.toFixed(1) : "—"}</b>
                {" fps "}
                <span className="dim">
                  {stream.meta ? `· ${stream.meta.infer_w}×${stream.meta.infer_h}` : ""}
                </span>
              </span>
            </div>
          ) : <div className="help">—</div>}
        </div>

        {stream.modelState?.has_normals && (
          <div className="row">
            <div className="label">Normals</div>
            {normalUrl ? (
              <div className="img-frame">
                <img src={normalUrl} className="thumb" />
                <span className="badge">
                  <span className="dim">
                    {stream.meta ? `${stream.meta.infer_w}×${stream.meta.infer_h}` : ""}
                  </span>
                </span>
              </div>
            ) : <div className="help">—</div>}
          </div>
        )}

        <button className="button accent" onClick={() => stream.samClear()}>
          Clear selection
        </button>

        <div className="help">press H to toggle HUD</div>
      </Card>
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

function MaskedRgb({ imgUrl, stream, onClick }: {
  imgUrl: string | null;
  stream: StreamState;
  onClick: (rect: DOMRect, clientX: number, clientY: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const lastBlobRef = useRef<string | null>(null);
  const [pulse, setPulse] = useState<{ x: number; y: number; key: number } | null>(null);
  const [pending, setPending] = useState(false);
  const pendingMaskSeqRef = useRef<number>(-1);

  // Drop "pending" once the mask seq advances after our click.
  useEffect(() => {
    let alive = true;
    const tick = () => {
      const m = stream.maskRef.current;
      if (m && m.seq !== pendingMaskSeqRef.current) {
        pendingMaskSeqRef.current = m.seq;
        setPending(false);
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    let raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [stream]);

  useEffect(() => {
    let alive = true;
    const draw = () => {
      const canvas = canvasRef.current;
      const meta = stream.meta;
      if (!canvas || !meta) {
        if (alive) raf = requestAnimationFrame(draw);
        return;
      }

      // Preload the next image off-screen and only swap in once it's decoded.
      // This avoids a flash to "no image yet" when blob URLs change while the
      // depth pipeline is slow and the canvas redraws between url updates.
      if (imgUrl && lastBlobRef.current !== imgUrl) {
        const next = new Image();
        next.onload = () => { imgRef.current = next; };
        next.src = imgUrl;
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

      const m = stream.maskRef.current;
      if (m && m.mask.length === meta.mesh_grid_w * meta.mesh_grid_h) {
        const gw = meta.mesh_grid_w, gh = meta.mesh_grid_h;
        const overlay = ctx.createImageData(gw, gh);
        const buf = overlay.data;
        for (let i = 0; i < gw * gh; i++) {
          const on = m.mask[i] !== 0;
          buf[i * 4 + 0] = 255;
          buf[i * 4 + 1] = 170;
          buf[i * 4 + 2] = 51;
          buf[i * 4 + 3] = on ? 140 : 0;
        }
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

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setPulse({ x: e.clientX - rect.left, y: e.clientY - rect.top, key: Date.now() });
    setPending(true);
    pendingMaskSeqRef.current = stream.maskRef.current?.seq ?? -1;
    onClick(rect, e.clientX, e.clientY);
  };

  return (
    <div className="mask-wrap">
      <canvas ref={canvasRef} className="canvas-mask" onClick={handleClick} />
      {pulse && (
        <span
          key={pulse.key}
          className="pulse"
          style={{ left: pulse.x, top: pulse.y }}
          onAnimationEnd={() => setPulse(null)}
        />
      )}
      {pending && <div className="spinner" />}
    </div>
  );
}
