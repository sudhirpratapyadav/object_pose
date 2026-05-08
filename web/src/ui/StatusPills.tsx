import { ModelState, SamState } from "../protocol";

type Props = {
  connected: boolean;
  modelState: ModelState | null;
  samState: SamState | null;
};

export function StatusPills({ connected, modelState, samState }: Props) {
  return (
    <div className="status-row">
      <span className={`pill ${connected ? "ok" : "bad"}`}>
        <span className="dot" />
        {connected ? "connected" : "disconnected"}
      </span>
      <span className={`pill ${modelStateClass(modelState)}`}>
        <span className="dot" />
        {renderDepth(modelState)}
      </span>
      <span className={`pill ${samStateClass(samState)}`}>
        <span className="dot" />
        {renderSam(samState)}
      </span>
    </div>
  );
}

function modelStateClass(s: ModelState | null) {
  if (!s) return "";
  if (s.status === "error") return "bad";
  if (s.status === "loading" || s.status === "downloading" || s.status === "warming up") return "warn";
  if (s.status.startsWith("running")) return "ok";
  return "warn";
}

function samStateClass(s: SamState | null) {
  if (!s) return "";
  if (s.status === "error") return "bad";
  if (s.status === "loading" || s.status === "downloading") return "warn";
  if (s.status.startsWith("running")) return "ok";
  return "warn";
}

function renderDepth(s: ModelState | null) {
  if (!s) return "depth: —";
  if (s.status === "downloading") {
    const pct = s.progress ? `${s.progress}%` : "";
    return `depth: dl ${pct}`.trim();
  }
  if (s.status === "loading") return "depth: loading";
  if (s.status.startsWith("running")) return s.status;
  return `depth: ${s.status}`;
}

function renderSam(s: SamState | null) {
  if (!s) return "sam: —";
  if (s.status === "downloading") return "sam: downloading";
  if (s.status === "loading") return "sam: loading";
  if (s.status.startsWith("running")) return s.status;
  return `sam: ${s.status}`;
}
