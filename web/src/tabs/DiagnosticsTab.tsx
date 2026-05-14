/**
 * Diagnostics tab: connection state, raw meta dump, fault history (later).
 */

import { LogViewer } from "../ui/LogViewer";
import { StreamState } from "../useStream";

type Props = { stream: StreamState };

export function DiagnosticsTab({ stream }: Props) {
  const wsUrl = `ws://${window.location.hostname || "localhost"}:8765`;
  const meta = stream.meta;
  const robot = meta?.robot;
  const oscHz = stream.robotStatus?.osc_hz;
  const alive = stream.robotStatus?.alive;
  const phaseName = stream.robotStatus?.phase_name ?? "";
  const faultMsg = stream.robotStatus?.fault_msg ?? "";

  return (
    <>
      {phaseName === "fault" && (
        <div className="row" style={{
          padding: 10,
          borderRadius: 6,
          background: "rgba(255, 107, 107, 0.12)",
          border: "1px solid rgba(255, 107, 107, 0.4)",
          gap: 6,
        }}>
          <div className="label" style={{ color: "var(--bad)" }}>
            ⚠ Robot fault
          </div>
          <div className="mono" style={{ fontSize: "0.85em" }}>
            {faultMsg || "(no message)"}
          </div>
          <button
            className="button accent"
            onClick={() => stream.recoverRobot()}
            title="Clear faults, switch to high-level, JointMove home"
          >
            Recover
          </button>
        </div>
      )}

      {robot?.source === "hardware" && (
        <div className="row">
          <div className="label">Recovery</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              className="button"
              onClick={() => stream.recoverRobot()}
              title="Same as Stop: stop controller, clear faults, JointMove home"
            >
              Recover
            </button>
            <button
              className="button"
              onClick={() => {
                if (window.confirm(
                  "Restart the kortex transport process?\n\n" +
                  "This kills the link to the robot and reopens it. " +
                  "The robot will home itself again. Use only when the " +
                  "normal recovery is stuck."
                )) {
                  stream.restartTransport();
                }
              }}
              title="Kill + respawn the kortex transport process"
            >
              Restart transport
            </button>
          </div>
          <div className="help">
            Recover does the standard sequence (clear faults → home →
            high-level position mode → idle). Restart transport is the
            heavy hammer when even that hangs.
          </div>
        </div>
      )}

      {(robot?.source === "hardware" || robot?.source === "sim") && (
        <details className="row" style={{ paddingLeft: 0 }}>
          <summary className="label" style={{ cursor: "pointer" }}>
            Fault history ({stream.robotStatus?.fault_history?.length ?? 0})
          </summary>
          <div className="mono" style={{
            fontSize: "0.74em",
            background: "rgba(0,0,0,0.32)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: 6,
            maxHeight: 200,
            overflow: "auto",
          }}>
            {!stream.robotStatus?.fault_history?.length ? (
              <span style={{ opacity: 0.5 }}>(no faults this session)</span>
            ) : (
              stream.robotStatus.fault_history.slice().reverse().map((f, i) => (
                <div key={`${f.ts}-${i}`} style={{ marginBottom: 4 }}>
                  <span style={{ color: "var(--text-faint)" }}>
                    {new Date(f.ts * 1000).toLocaleTimeString()}
                  </span>{" "}
                  <span style={{ color: "var(--bad)" }}>
                    [{f.source}]
                  </span>{" "}
                  {f.msg}
                </div>
              ))
            )}
          </div>
        </details>
      )}
      <div className="row">
        <div className="label">Connection</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>WebSocket</span>
          <span className="kv-val">
            {stream.connected ? "connected" : "disconnected"}
            {" · "}{wsUrl}
          </span>
        </div>
      </div>

      <div className="row">
        <div className="label">Robot</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>source</span>
          <span className="kv-val">{robot?.source ?? "—"}</span>
        </div>
        {(robot?.source === "hardware" || robot?.source === "sim") && (
          <>
            <div className="kv mono" style={{ fontSize: "0.85em" }}>
              <span>phase</span>
              <span className="kv-val">{phaseName || "—"}</span>
            </div>
            <div className="kv mono" style={{ fontSize: "0.85em" }}>
              <span>transport</span>
              <span className="kv-val">
                {oscHz != null ? `${oscHz.toFixed(0)} Hz` : "—"}
                {alive === false && " (dead)"}
              </span>
            </div>
            <div className="kv mono" style={{ fontSize: "0.85em" }}>
              <span>EE body</span>
              <span className="kv-val">
                {robot?.ee_body_name ?? "—"} (id {robot?.ee_body_idx ?? "—"})
              </span>
            </div>
          </>
        )}
      </div>

      <div className="row">
        <div className="label">Camera calibration</div>
        {meta?.cam_calib ? (
          <div className="kv mono" style={{ fontSize: "0.78em", flexDirection: "column", alignItems: "stretch" }}>
            <span>pos: [{meta.cam_calib.extrinsics.pos.map(v => v.toFixed(4)).join(", ")}]</span>
            <span>euler: [{meta.cam_calib.extrinsics.euler_deg.map(v => v.toFixed(2)).join(", ")}]</span>
            <span>fx={meta.cam_calib.intrinsics.fx.toFixed(1)} fy={meta.cam_calib.intrinsics.fy.toFixed(1)} cx={meta.cam_calib.intrinsics.cx.toFixed(1)} cy={meta.cam_calib.intrinsics.cy.toFixed(1)}</span>
            <span>{meta.cam_calib.intrinsics.width}×{meta.cam_calib.intrinsics.height}</span>
          </div>
        ) : (
          <div className="help">— waiting —</div>
        )}
      </div>

      <div className="row">
        <div className="label">Stats</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>RGB</span>
          <span className="kv-val">{stream.stats?.rgb_fps?.toFixed(1) ?? "—"} fps</span>
        </div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>depth</span>
          <span className="kv-val">{stream.stats?.depth_fps?.toFixed(1) ?? "—"} fps</span>
        </div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>SAM</span>
          <span className="kv-val">{stream.stats?.sam_ms ?? "—"} ms</span>
        </div>
      </div>

      <LogViewer lines={stream.logLines} />

      <details className="row" style={{ paddingLeft: 0 }}>
        <summary className="label" style={{ cursor: "pointer" }}>
          raw meta (JSON)
        </summary>
        <pre className="mono" style={{
          fontSize: "0.72em",
          background: "rgba(255,255,255,0.04)",
          padding: 8,
          borderRadius: 4,
          maxHeight: 240,
          overflow: "auto",
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}>{JSON.stringify(meta, null, 2)}</pre>
      </details>
    </>
  );
}
