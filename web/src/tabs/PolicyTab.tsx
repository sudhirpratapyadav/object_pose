/**
 * Policy tab — engage / disengage a learned NN policy.
 *
 * A policy writes setpoints into shm_qtarget at its own outer-loop rate
 * (10 Hz for open_drawer). The active ee_pose controller tracks those
 * setpoints at 500 Hz.
 *
 * Engage flow (server-side):
 *   1. checks fresh object_pose if the policy needs one;
 *   2. overlays the policy's controller_gains on ee_pose for the run;
 *   3. hot-swaps to ee_pose if not already running;
 *   4. spawns the policy subprocess.
 *
 * UI shows: dropdown, status pill (waiting/running/success), object-pose
 * indicator, goal block, live HZ, and the last 7-D action.
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
  const current = pol.current || "";
  const engaged = !!current;
  const statusLabel = STATUS_LABELS[pol.status_code] ?? `code=${pol.status_code}`;

  const activeInfo = available.find((p) => p.name === current);

  const onSelect = (name: string) => {
    if (!name) return;
    if (engaged && name === current) return;
    if (engaged) stream.stopPolicy();
    stream.setPolicy(name);
  };

  return (
    <>
      <div className="row">
        <div className="label">Active policy</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>{engaged ? activeInfo?.display_name ?? current : "(none)"}</span>
          <span className={
            "kv-val " + (pol.status_code === 1 ? "ok"
              : pol.status_code === 2 ? "ok"
              : pol.status_code === 0 && engaged ? "warn"
              : "")
          }>
            {engaged ? statusLabel : "—"}
          </span>
        </div>
        {pol.last_error && (
          <div className="help" style={{ color: "salmon" }}>
            {pol.last_error}
          </div>
        )}
        {activeInfo?.description && (
          <div className="help">{activeInfo.description}</div>
        )}
      </div>

      <div className="row">
        <div className="label">Pick policy</div>
        <select
          className="select"
          value={current}
          onChange={(e) => onSelect(e.target.value)}
        >
          <option value="">(none)</option>
          {available.map((p) => (
            <option key={p.name} value={p.name}>
              {p.display_name}
            </option>
          ))}
        </select>
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <button
            className="button"
            onClick={() => stream.stopPolicy()}
            disabled={!engaged}
            title="Stop the active policy and restore the user's ee_pose gains"
          >
            Stop
          </button>
          <button
            className="button"
            onClick={() => stream.reloadPolicyConfigs()}
            title="Re-read policies/configs.yaml from disk"
          >
            Reload configs
          </button>
        </div>
      </div>

      {/* Object-pose status — what the policy reads from vision */}
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

      {/* Goal (set at engage time = handle + goal_offset) */}
      {engaged && (
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
      {engaged && (
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
