/**
 * Camera-calibration sliders + Save button.
 *
 * Holds local state for pos + euler so dragging is smooth and immediate;
 * seeds from server's camCalib on first arrival and on explicit Reload.
 * Each change is sent live to the server, which transforms the point cloud
 * and broadcasts an updated calib that all clients can see.
 */

import { useEffect, useRef, useState } from "react";
import { StreamState } from "./useStream";

type Props = {
  stream: StreamState;
};

const POS_STEP = 0.005;   // 5 mm
const ROT_STEP = 0.5;     // 0.5 deg
const POS_MIN = -2.0, POS_MAX = 2.0;
const ROT_MIN = -180.0, ROT_MAX = 180.0;

export function CamCalibPanel({ stream }: Props) {
  const calib = stream.camCalib;
  const [pos, setPos] = useState<[number, number, number] | null>(null);
  const [eul, setEul] = useState<[number, number, number] | null>(null);
  const seededRef = useRef(false);
  const [savedAt, setSavedAt] = useState<string>("");

  // Seed from server on first calib arrival.
  useEffect(() => {
    if (calib && !seededRef.current) {
      seededRef.current = true;
      setPos(calib.extrinsics.pos);
      setEul(calib.extrinsics.euler_deg);
    }
  }, [calib]);

  if (!calib || !pos || !eul) {
    return (
      <div className="row">
        <div className="label">Camera calibration</div>
        <div className="help">— waiting —</div>
      </div>
    );
  }

  const sendPos = (next: [number, number, number]) => {
    setPos(next);
    stream.setCamExtrinsics(next, eul);
  };
  const sendEul = (next: [number, number, number]) => {
    setEul(next);
    stream.setCamExtrinsics(pos, next);
  };

  const reload = () => {
    setPos(calib.extrinsics.pos);
    setEul(calib.extrinsics.euler_deg);
    stream.setCamExtrinsics(calib.extrinsics.pos, calib.extrinsics.euler_deg);
  };

  const save = () => {
    stream.saveCamExtrinsics();
    setSavedAt(new Date().toLocaleTimeString());
  };

  const i = calib.intrinsics;

  return (
    <>
      <div className="row">
        <div className="label">Camera position (m)</div>
        <Triplet labels={["x", "y", "z"]} values={pos} step={POS_STEP}
                 min={POS_MIN} max={POS_MAX} digits={4}
                 onChange={sendPos} />
      </div>
      <div className="row">
        <div className="label">Camera rotation (deg)</div>
        <Triplet labels={["rx", "ry", "rz"]} values={eul} step={ROT_STEP}
                 min={ROT_MIN} max={ROT_MAX} digits={2}
                 onChange={sendEul} />
      </div>
      <div className="row">
        <div className="label">Intrinsics</div>
        <div className="kv mono" style={{ fontSize: "0.85em", opacity: 0.85 }}>
          <span>{i.width}×{i.height}</span>
          <span className="kv-val">
            fx={i.fx.toFixed(1)} fy={i.fy.toFixed(1)}<br />
            cx={i.cx.toFixed(1)} cy={i.cy.toFixed(1)}
          </span>
        </div>
      </div>
      <div className="row">
        <button className="button" onClick={reload} title="Reload from server">
          Reload
        </button>
        <button className="button" onClick={save} title="Save to cam_calib_config.yaml">
          Save to YAML
        </button>
        {savedAt && (
          <span style={{ marginLeft: 8, fontSize: "0.85em", opacity: 0.65 }}>
            saved {savedAt}
          </span>
        )}
      </div>
    </>
  );
}


type TripletProps = {
  labels: [string, string, string];
  values: [number, number, number];
  step: number;
  min: number;
  max: number;
  digits: number;
  onChange: (v: [number, number, number]) => void;
};

function Triplet({ labels, values, step, min, max, digits, onChange }: TripletProps) {
  const update = (idx: 0 | 1 | 2, raw: string) => {
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    const clamped = Math.max(min, Math.min(max, n));
    const next: [number, number, number] = [...values];
    next[idx] = clamped;
    onChange(next);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
      {[0, 1, 2].map((i) => (
        <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span className="mono" style={{ width: 18, opacity: 0.6 }}>{labels[i]}</span>
          <input
            type="range"
            min={min} max={max} step={step}
            value={values[i]}
            onChange={(e) => update(i as 0|1|2, e.target.value)}
            style={{ flex: 1 }}
          />
          <input
            type="number"
            value={values[i].toFixed(digits)}
            step={step} min={min} max={max}
            onChange={(e) => update(i as 0|1|2, e.target.value)}
            className="mono"
            style={{
              width: 80, fontSize: "0.85em", padding: "2px 4px",
              background: "rgba(255,255,255,0.05)", color: "inherit",
              border: "1px solid rgba(255,255,255,0.15)", borderRadius: 3,
            }}
          />
        </div>
      ))}
    </div>
  );
}
