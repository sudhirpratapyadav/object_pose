/**
 * Joint-PD controller panel.
 *
 * Visible only when stream.controllerState.current === "joint_pd" AND
 * status === "running". Provides:
 *   - 7 joint-target sliders (q_des in deg)
 *   - 7 kp + 7 kd sliders for tuning
 *   - Save / Reload buttons that persist or revert via configs.yaml
 *
 * Live updates: every slider drag publishes immediately. The controller
 * subprocess sees the new values within one 500 Hz cycle (≤ 2 ms).
 */

import { useEffect, useRef, useState } from "react";
import { StreamState } from "./useStream";


type Props = { stream: StreamState };

const HOME_DEG = [90, 30, 0, 90, 0, 60, -90];
const KP_MIN = 0, KP_MAX = 100;
const KD_MIN = 0, KD_MAX = 20;


export function JointPdPanel({ stream }: Props) {
  const robot = stream.meta?.robot;
  const isHardware = robot?.source === "hardware";
  const ctrl = stream.controllerState;
  const isActive = ctrl?.current === "joint_pd" && ctrl?.status === "running";

  const actuators = (robot?.actuators ?? []).slice(0, 7);

  // Slider state in radians.
  const [qDes, setQDes] = useState<number[] | null>(null);
  const [kp, setKp] = useState<number[] | null>(null);
  const [kd, setKd] = useState<number[] | null>(null);
  const seededRef = useRef(false);

  // Re-seed when the controller becomes active OR when the server's
  // configs change (e.g. after Reload).
  useEffect(() => {
    if (!isActive) {
      seededRef.current = false;
      return;
    }
    const cfg = ctrl?.configs?.joint_pd as
      | { kp?: number[]; kd?: number[] }
      | undefined;
    const cfgKp = cfg?.kp;
    const cfgKd = cfg?.kd;

    if (!seededRef.current) {
      seededRef.current = true;
      setQDes(HOME_DEG.map((d) => (d * Math.PI) / 180));
    }
    if (cfgKp && cfgKp.length >= 7) setKp(cfgKp.slice(0, 7));
    if (cfgKd && cfgKd.length >= 7) setKd(cfgKd.slice(0, 7));
  }, [isActive, ctrl?.configs?.joint_pd]);

  if (!isHardware) return null;
  if (!isActive) {
    return (
      <div className="row">
        <div className="label">Joint PD</div>
        <div className="help">
          Pick <b>Joint PD</b> from the controller dropdown to tune.
        </div>
      </div>
    );
  }
  if (!qDes || !kp || !kd || actuators.length < 7) {
    return (
      <div className="row">
        <div className="label">Joint PD</div>
        <div className="help">— waiting for joint metadata + gains —</div>
      </div>
    );
  }

  const updateQ = (i: number, v: number) => {
    const next = qDes.slice(); next[i] = v;
    setQDes(next);
    stream.setJointTarget(next);
  };
  const updateKp = (i: number, v: number) => {
    const next = kp.slice(); next[i] = v;
    setKp(next);
    stream.setControllerGains("joint_pd", next, kd);
  };
  const updateKd = (i: number, v: number) => {
    const next = kd.slice(); next[i] = v;
    setKd(next);
    stream.setControllerGains("joint_pd", kp, next);
  };

  const goHome = () => {
    const home = HOME_DEG.map((d) => (d * Math.PI) / 180);
    setQDes(home);
    stream.setJointTarget(home);
  };
  const onSave = () => stream.saveControllerGains("joint_pd");
  const onReload = () => stream.reloadControllerGains();

  return (
    <>
      <div className="row" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
        <div className="label" style={{ flex: 1 }}>Joint targets (deg)</div>
        <button className="button" onClick={goHome} title="Send all joints to HOME_DEG">
          Home pose
        </button>
      </div>
      <div className="row" style={{ flexDirection: "column", gap: 4 }}>
        {actuators.map((a, i) => (
          <NumSlider
            key={`q-${a.name}`} name={a.name}
            min={a.min} max={a.max} value={qDes[i]}
            display="deg"
            onChange={(v) => updateQ(i, v)}
          />
        ))}
      </div>

      <div className="row" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
        <div className="label" style={{ flex: 1 }}>Gains</div>
        <button className="button" onClick={onReload} title="Re-read configs.yaml">
          Reload
        </button>
        <button className="button accent" onClick={onSave} title="Save to controllers/configs.yaml">
          Save
        </button>
      </div>
      <div className="row" style={{ flexDirection: "column", gap: 4 }}>
        <div className="help" style={{ marginBottom: 2 }}>kp (stiffness)</div>
        {actuators.map((a, i) => (
          <NumSlider
            key={`kp-${a.name}`} name={a.name}
            min={KP_MIN} max={KP_MAX} value={kp[i]}
            display="raw"
            step={1}
            onChange={(v) => updateKp(i, v)}
          />
        ))}
        <div className="help" style={{ marginTop: 6, marginBottom: 2 }}>kd (damping)</div>
        {actuators.map((a, i) => (
          <NumSlider
            key={`kd-${a.name}`} name={a.name}
            min={KD_MIN} max={KD_MAX} value={kd[i]}
            display="raw"
            step={0.1}
            onChange={(v) => updateKd(i, v)}
          />
        ))}
      </div>
    </>
  );
}


type NumSliderProps = {
  name: string; min: number; max: number; value: number;
  display: "deg" | "raw";
  step?: number;
  onChange: (v: number) => void;
};

function NumSlider({ name, min, max, value, display, step, onChange }: NumSliderProps) {
  const span = max - min;
  const sliderStep = step ?? Math.max(0.001, span / 1000);
  const shown = display === "deg" ? (value * 180) / Math.PI : value;
  const numStep = display === "deg" ? 1 : (step ?? 0.1);
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <span className="mono" style={{ width: 70, fontSize: "0.82em", opacity: 0.75 }}>
        {name}
      </span>
      <input
        type="range"
        min={min} max={max} step={sliderStep} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1 }}
      />
      <input
        type="number"
        value={shown.toFixed(display === "deg" ? 1 : 2)}
        step={numStep}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!Number.isFinite(n)) return;
          const raw = display === "deg" ? (n * Math.PI) / 180 : n;
          onChange(Math.max(min, Math.min(max, raw)));
        }}
        className="mono"
        style={{
          width: 64, fontSize: "0.78em", padding: "2px 4px",
          background: "rgba(255,255,255,0.05)", color: "inherit",
          border: "1px solid rgba(255,255,255,0.15)", borderRadius: 3,
        }}
      />
    </div>
  );
}
