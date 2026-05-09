/**
 * Per-actuator sliders that publish set_target_ctrl.
 *
 * Visible only when meta.robot.source == "sim". The sim worker tracks these
 * setpoints with the MJCF's position actuators (or whatever generic actuator
 * the model declares).
 */

import { useEffect, useState } from "react";
import { StreamState } from "./useStream";

type Props = { stream: StreamState };

export function JointTargetsPanel({ stream }: Props) {
  const robot = stream.meta?.robot;
  const actuators = robot?.actuators ?? [];
  const visible = robot?.source === "sim" && actuators.length > 0;

  // Local slider state. Seeded from `home` on first arrival.
  const [vals, setVals] = useState<number[] | null>(null);
  useEffect(() => {
    if (vals === null && actuators.length > 0) {
      setVals(actuators.map((a) => a.home));
    }
  }, [actuators.length]);

  if (!visible || vals === null) return null;

  const update = (idx: number, v: number) => {
    const next = vals.slice();
    next[idx] = v;
    setVals(next);
    stream.setTargetCtrl(next);
  };

  const resetHome = () => {
    const homes = actuators.map((a) => a.home);
    setVals(homes);
    stream.setTargetCtrl(homes);
  };

  return (
    <>
      <div className="row">
        <div className="label">Joint targets</div>
        <button className="button" onClick={resetHome} title="Reset to home keyframe">
          Home
        </button>
      </div>
      <div className="row" style={{ flexDirection: "column", gap: 4 }}>
        {actuators.map((a, i) => (
          <ActuatorSlider key={a.name} name={a.name}
                          min={a.min} max={a.max}
                          value={vals[i]}
                          onChange={(v) => update(i, v)} />
        ))}
      </div>
    </>
  );
}


type SliderProps = {
  name: string; min: number; max: number; value: number;
  onChange: (v: number) => void;
};

function ActuatorSlider({ name, min, max, value, onChange }: SliderProps) {
  const span = max - min;
  // Step: 1 / 1000 of range, with min step 0.001.
  const step = Math.max(0.001, span / 1000);
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <span className="mono" style={{ width: 80, fontSize: "0.85em", opacity: 0.75 }}>
        {name}
      </span>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1 }}
      />
      <input
        type="number"
        value={value.toFixed(3)}
        step={step} min={min} max={max}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className="mono"
        style={{
          width: 70, fontSize: "0.8em", padding: "2px 4px",
          background: "rgba(255,255,255,0.05)", color: "inherit",
          border: "1px solid rgba(255,255,255,0.15)", borderRadius: 3,
        }}
      />
    </div>
  );
}
