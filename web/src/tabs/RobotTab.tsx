/**
 * Robot tab: per-controller config + source-specific panels.
 *
 * Slice 2: shows the active controller's display name + description.
 * Per-controller config panels (joint sliders for joint-position, gain
 * sliders for ee-pose, etc.) come in later slices.
 */

import { HardwareStatusPanel } from "../HardwareStatusPanel";
import { JointPdPanel } from "../JointPdPanel";
import { JointTargetsPanel } from "../JointTargetsPanel";
import { StreamState } from "../useStream";

type Props = { stream: StreamState };

export function RobotTab({ stream }: Props) {
  const robot = stream.meta?.robot;
  const robotEnabled = !!robot?.enabled;
  const robotSrc = robot?.source ?? "none";
  const ctrl = stream.controllerState;
  const isHardware = robotSrc === "hardware" || robotSrc === "sim";

  if (!robotEnabled) {
    return (
      <div className="row">
        <div className="label">Robot</div>
        <div className="help">
          No robot loaded. Pass <code>--mjcf robot/mjcf/scene.xml</code>
          (and optionally <code>--robot-source dummy / sim / hardware</code>)
          when launching the server.
        </div>
      </div>
    );
  }

  // Find the currently active controller's metadata.
  const activeInfo = ctrl?.available.find(c => c.name === ctrl.current);

  return (
    <>
      <div className="row">
        <div className="label">Source</div>
        <div className="kv mono" style={{ fontSize: "0.85em" }}>
          <span>{robotSrc}</span>
          <span className="kv-val">
            {robotSrc === "hardware" ? "(real Kinova)"
              : robotSrc === "sim" ? "(MuJoCo)"
              : robotSrc === "dummy" ? "(test sine)"
              : "(home pose)"}
          </span>
        </div>
      </div>

      {isHardware && (
        <div className="row">
          <div className="label">Active controller</div>
          {activeInfo ? (
            <>
              <div className="kv mono" style={{ fontSize: "0.85em" }}>
                <span>{activeInfo.display_name}</span>
                <span className="kv-val">{activeInfo.command_mode}</span>
              </div>
              <div className="help">{activeInfo.description}</div>
            </>
          ) : (
            <div className="help">— no controller info —</div>
          )}
          <div className="help">
            Switch controllers from the dropdown in the top status bar.
            Per-controller config panels (joint sliders / EE-pose targets /
            gains / etc.) will appear here as later slices land.
          </div>
        </div>
      )}

      {/* Per-controller config panel for joint_pd (hardware mode). */}
      <JointPdPanel stream={stream} />

      {/* Existing per-source panels.
          JointTargetsPanel is sim-only (publishes set_target_ctrl).
          HardwareStatusPanel shows live joint angles + EE pose in hardware. */}
      <JointTargetsPanel stream={stream} />
      <HardwareStatusPanel stream={stream} />
    </>
  );
}
