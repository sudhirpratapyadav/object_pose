/**
 * Persistent top status bar.
 *
 * Drives the controller dispatcher: pick a controller, hit STOP, hit HOME.
 * Shows live transport rate + controller status. Visible from every tab.
 */

import { StreamState } from "../useStream";

type Props = {
  stream: StreamState;
};

export function StatusBar({ stream }: Props) {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const robotSrc = stream.meta?.robot?.source ?? "none";
  const oscHz = stream.robotStatus?.osc_hz;
  const transportAlive = stream.robotStatus?.alive;
  const phaseName = stream.robotStatus?.phase_name ?? "";
  const faultMsg = stream.robotStatus?.fault_msg ?? "";
  const ctrl = stream.controllerState;

  // Map transport phase to a visual class.
  const phasePillClass = (() => {
    if (phaseName === "fault") return "status-pill-bad";
    if (phaseName === "ready" || phaseName === "running") return "status-pill-ok";
    if (phaseName === "" || phaseName === "shutdown") return "";
    return "status-pill-warn";   // boot / homing / swapping
  })();

  // Sim and hardware both have the controller dispatcher + transport.
  const isHardware = robotSrc === "hardware" || robotSrc === "sim";
  const ctrlAvailable = ctrl?.available ?? [];
  const ctrlCurrent = ctrl?.current ?? "idle";
  const ctrlStatus = ctrl?.status ?? "idle";
  const ctrlError = ctrl?.last_error ?? "";

  const onCtrlChange = (name: string) => {
    if (!isHardware) return;
    if (name === ctrlCurrent && ctrlStatus === "running") return;
    stream.setController(name);
  };

  const onStop = () => {
    if (!isHardware) return;
    stream.stopController();
  };
  const onHome = () => {
    if (!isHardware) return;
    stream.homeRobot();
  };

  // Status pill text + class.
  let statusText: string;
  let statusClass = "";
  switch (ctrlStatus) {
    case "loading":
      statusText = `starting ${ctrlCurrent}…`;
      statusClass = "status-pill-warn";
      break;
    case "running":
      statusText = `running ${ctrlCurrent}`;
      statusClass = "status-pill-ok";
      break;
    case "stopping":
      statusText = `stopping ${ctrlCurrent}…`;
      statusClass = "status-pill-warn";
      break;
    case "fault":
      statusText = `fault: ${ctrlError || "unknown"}`;
      statusClass = "status-pill-bad";
      break;
    default:
      statusText = "idle";
  }

  return (
    <div className="status-bar">
      <div className="status-group">
        <span
          className={`conn-pill ${stream.connected ? "ok" : "bad"}`}
          title={`${stream.connected ? "Connected" : "Disconnected"} — ${wsUrl}`}
        >
          <span className={`conn-dot ${stream.connected ? "ok" : "bad"}`} />
          <span>{stream.connected ? "online" : "offline"}</span>
        </span>

        {robotSrc !== "none" && (
          <span className="status-pill" title="Robot source">
            <span className="status-pill-key">robot</span>
            <span className="status-pill-val mono">{robotSrc}</span>
          </span>
        )}
      </div>

      <div className="status-group">
        <label className="status-pill" title="Active controller">
          <span className="status-pill-key">controller</span>
          <select
            className="status-select"
            value={ctrlCurrent}
            onChange={(e) => onCtrlChange(e.target.value)}
            disabled={!isHardware || !ctrlAvailable.length
              || ctrlStatus === "loading" || ctrlStatus === "stopping"}
          >
            {ctrlAvailable.length === 0 && (
              <option value="idle">idle</option>
            )}
            {ctrlAvailable.map((c) => (
              <option key={c.name} value={c.name} title={c.description}>
                {c.display_name}
              </option>
            ))}
          </select>
        </label>

        <span className={`status-pill ${statusClass}`} title={ctrlError || statusText}>
          <span className="status-pill-key">status</span>
          <span className="status-pill-val">{statusText}</span>
        </span>

        {isHardware && (
          <>
            <span
              className={`status-pill ${transportAlive === false ? "status-pill-bad" : "status-pill-ok"}`}
              title="Kortex link to the robot"
            >
              <span className="status-pill-key">kortex</span>
              <span className="status-pill-val mono">
                {transportAlive === false ? "offline" : "ok"}
              </span>
            </span>
            {phaseName && (
              <span
                className={`status-pill ${phasePillClass}`}
                title={faultMsg || `Robot transport phase: ${phaseName}`}
              >
                <span className="status-pill-key">phase</span>
                <span className="status-pill-val mono">{phaseName}</span>
              </span>
            )}
            {ctrlStatus === "running" && (
              <span className="status-pill" title="Active controller loop rate">
                <span className="status-pill-key">control</span>
                <span className="status-pill-val mono">
                  {oscHz != null ? `${oscHz.toFixed(0)} Hz` : "—"}
                </span>
              </span>
            )}
          </>
        )}
      </div>

      <div className="status-group status-group-right">
        <button className="status-btn" onClick={onHome} title="Send arm to home pose">
          ⌂ Home
        </button>
        <button className="status-btn status-btn-bad" onClick={onStop} title="Stop active controller (back to idle + home)">
          ■ Stop
        </button>
      </div>
    </div>
  );
}
