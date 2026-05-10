/**
 * Top horizontal tab strip below the StatusBar.
 *
 * Tabs are local UI state — no server interaction. The Vision tab keeps
 * the calibration/depth/RGB controls; Robot has the controller-specific
 * panels; Policy is a placeholder for slice 6; Diagnostics has connection
 * + raw state debugging.
 */

export type TabKey = "vision" | "robot" | "policy" | "diagnostics";

type Props = {
  active: TabKey;
  onChange: (k: TabKey) => void;
};

const TABS: { key: TabKey; label: string }[] = [
  { key: "vision",      label: "Vision" },
  { key: "robot",       label: "Robot" },
  { key: "policy",      label: "Policy" },
  { key: "diagnostics", label: "Diagnostics" },
];

export function TabStrip({ active, onChange }: Props) {
  return (
    <div className="tab-strip">
      {TABS.map((t) => (
        <button
          key={t.key}
          className={`tab-btn ${active === t.key ? "active" : ""}`}
          onClick={() => onChange(t.key)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
