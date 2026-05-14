/**
 * Read-only hardware status: joint angles, EE pose, OSC rate, Stop button.
 *
 * Visible only when meta.robot.source == "hardware". Joint angles and EE
 * pose are derived from the existing KIND_ROBOT_TRANSFORMS stream
 * (xpos/xquat per body) — we don't need a new wire frame for read-only.
 *
 * The Stop button is intentionally NOT wired to a server command yet; the
 * stage-1 cut is read-only. When control commands arrive in stage 2 we'll
 * hook a `robot_stop` JSON command into stream.send().
 */

import { useEffect, useState } from "react";
import { StreamState } from "./useStream";

type Props = { stream: StreamState };

export function HardwareStatusPanel({ stream }: Props) {
  const robot = stream.meta?.robot;
  // Live joint angles + EE pose are useful in sim too.
  const visible = robot?.source === "hardware" || robot?.source === "sim";

  // Throttled snapshot of EE pose + joint angles (5 Hz) so we don't re-render
  // on every transform tick.
  const [snap, setSnap] = useState<{
    eePos: [number, number, number] | null;
    eeQuat: [number, number, number, number] | null;
  }>({ eePos: null, eeQuat: null });

  useEffect(() => {
    if (!visible) return;
    const eeIdx = robot.ee_body_idx;
    let alive = true;
    const tick = () => {
      const x = stream.robotXformRef.current;
      if (x && eeIdx >= 0 && eeIdx < x.nbody) {
        const i3 = 3 * eeIdx, i4 = 4 * eeIdx;
        setSnap({
          eePos: [x.xpos[i3], x.xpos[i3 + 1], x.xpos[i3 + 2]],
          // wxyz on wire — keep as-is for display.
          eeQuat: [x.xquat[i4], x.xquat[i4 + 1], x.xquat[i4 + 2], x.xquat[i4 + 3]],
        });
      }
      if (alive) handle = window.setTimeout(tick, 200);
    };
    let handle = window.setTimeout(tick, 200);
    return () => { alive = false; window.clearTimeout(handle); };
  }, [visible, robot?.ee_body_idx, stream.robotXformRef]);

  if (!visible) return null;

  const oscHz = stream.robotStatus?.osc_hz;
  const alive = stream.robotStatus?.alive;

  return (
    <>
      <div className="row">
        <div className="label">Robot status</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>OSC</span>
          <span className="kv-val">
            {oscHz != null ? `${oscHz.toFixed(0)} Hz` : "—"}
            {alive === false && (
              <span style={{ color: "#ff6666", marginLeft: 6 }}>(dead)</span>
            )}
          </span>
        </div>
      </div>
      <div className="row">
        <div className="label">EE position (m)</div>
        <div className="mono" style={{ fontSize: "0.85em", flex: 1 }}>
          {snap.eePos
            ? snap.eePos.map((v) => v.toFixed(3)).join("  ")
            : "—"}
        </div>
      </div>
      <div className="row">
        <div className="label">EE quat (wxyz)</div>
        <div className="mono" style={{ fontSize: "0.85em", flex: 1 }}>
          {snap.eeQuat
            ? snap.eeQuat.map((v) => v.toFixed(3)).join("  ")
            : "—"}
        </div>
      </div>
      <div className="row">
        <button
          className="button accent"
          onClick={() => alert("Stop not wired yet — stage-2 work")}
          title="Stop the OSC loop (not wired in this read-only build)"
        >
          Stop (TODO)
        </button>
        <span style={{ opacity: 0.6, fontSize: "0.85em", marginLeft: 8 }}>
          read-only
        </span>
      </div>
    </>
  );
}
