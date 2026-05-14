/**
 * Policy tab — pick a learned policy and control its lifecycle.
 *
 * State machine (server-driven):
 *   idle    — nothing selected, or selected but never engaged.
 *   running — subprocess writing shm_qtarget at target_hz; ee_pose tracks.
 *   paused  — selection kept, subprocess dead, arm holds at current pose.
 *
 * Buttons (one row, in order):
 *   ▶ Play / ■ Stop   — toggle. Play: idle → running (needs object_pose).
 *                       Stop: any → idle (drives home via SST).
 *   ⏸ Pause / ▶ Resume — toggle. running ↔ paused.
 *   ⟲ Reset           — drive home, leave selection in paused state.
 */

import { StreamState } from "../useStream";

type Props = { stream: StreamState };

const STATUS_LABELS: Record<number, string> = {
  0: "waiting",
  1: "running",
  2: "success",
};

export function PolicyTab({ stream }: Props) {
  const robot = stream.meta?.robot;
  const isHardware = robot?.source === "hardware" || robot?.source === "sim";
  const pol = stream.controllerState?.policy;

  if (!isHardware) {
    return (
      <div className="row">
        <div className="label">Policy</div>
        <div className="help">
          Only available with <code>--robot-source hardware</code> or
          <code>--robot-source sim</code>.
        </div>
      </div>
    );
  }
  if (!pol) {
    return (
      <div className="row">
        <div className="label">Policy</div>
        <div className="help">— waiting for policy state —</div>
      </div>
    );
  }

  const available = pol.available;
  const selected = pol.selected || "";
  const running = pol.lifecycle === "running";
  const paused  = pol.lifecycle === "paused";
  const idle    = pol.lifecycle === "idle";
  const hasSelection = !!selected;
  const statusLabel = STATUS_LABELS[pol.status_code] ?? `code=${pol.status_code}`;

  const activeInfo = available.find((p) => p.name === selected);

  // Error is only meaningful once a policy is selected; otherwise the
  // server's "select a policy first" / "no object_pose" messages are noise.
  const errorVisible = hasSelection && !!pol.last_error;

  const playOrStop = () => {
    if (running || paused) stream.stopPolicy();
    else stream.playPolicy();
  };
  const pauseOrResume = () => stream.pausePolicy();
  const reset = () => stream.resetPolicy();

  return (
    <>
      <div className="row">
        <div className="label">Select policy</div>
        <select
          className="select"
          value={selected}
          onChange={(e) => stream.selectPolicy(e.target.value)}
        >
          <option value="">(none)</option>
          {available.map((p) => (
            <option key={p.name} value={p.name}>
              {p.display_name}
            </option>
          ))}
        </select>
        {activeInfo?.description && (
          <div className="help">{activeInfo.description}</div>
        )}
      </div>

      {/* Controls row — only shown once a policy is selected */}
      {hasSelection && (
        <div className="row">
          <div className="label">Controls</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              className={"button " + (idle ? "accent" : "")}
              onClick={playOrStop}
              title={
                running || paused ? "Disengage and drive home"
                  : "Engage the selected policy"
              }
            >
              {running || paused ? "■ Stop" : "▶ Play"}
            </button>
            {(running || paused) && (
              <button
                className="button"
                onClick={pauseOrResume}
                title={
                  running ? "Pause: hold arm at current pose"
                    : "Resume: continue from current pose"
                }
              >
                {running ? "⏸ Pause" : "▶ Resume"}
              </button>
            )}
            {(running || paused) && (
              <button
                className="button"
                onClick={reset}
                title="Drive home; selection remains, paused at home"
              >
                ⟲ Reset
              </button>
            )}
          </div>
          <div className="help">
            State: <b>{pol.lifecycle}</b>
            {running && <> · {statusLabel}</>}
          </div>
        </div>
      )}

      {/* Per-policy error — only meaningful once selected */}
      {errorVisible && (
        <div className="row">
          <div className="help" style={{ color: "salmon" }}>
            {pol.last_error}
          </div>
        </div>
      )}

      {/* Object-pose status — what the policy reads from vision */}
      {hasSelection && activeInfo?.needs_object_pose && (
        <div className="row">
          <div className="label">Object pose</div>
          {pol.object_pose ? (
            <>
              <div className="kv mono" style={{ fontSize: "0.85em" }}>
                <span>handle (world)</span>
                <span className="kv-val">
                  [{pol.object_pose.pos.map((v) => v.toFixed(3)).join(", ")}]
                </span>
              </div>
              <div className="help">
                {pol.object_pose.n_points} masked points
              </div>
            </>
          ) : (
            <div className="help" style={{ color: "salmon" }}>
              no object pose — click the target in the RGB view to make a SAM mask
            </div>
          )}
        </div>
      )}

      {/* Goal (set at Play time = handle + goal_offset) */}
      {(running || paused) && (
        <div className="row">
          <div className="label">Goal</div>
          <div className="kv mono" style={{ fontSize: "0.85em" }}>
            <span>handle goal (world)</span>
            <span className="kv-val">
              [{pol.goal.map((v) => v.toFixed(3)).join(", ")}]
            </span>
          </div>
        </div>
      )}

      {/* Live HZ + last action */}
      {running && (
        <div className="row">
          <div className="label">Live</div>
          <div className="kv mono" style={{ fontSize: "0.85em" }}>
            <span>policy hz</span>
            <span className="kv-val">{pol.hz.toFixed(1)}</span>
          </div>
          <div className="kv mono" style={{ fontSize: "0.78em" }}>
            <span>last action</span>
            <span className="kv-val">
              [{pol.last_action.map((v) => v.toFixed(2)).join(", ")}]
            </span>
          </div>
        </div>
      )}
    </>
  );
}
