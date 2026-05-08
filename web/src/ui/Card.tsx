import { useEffect, useState } from "react";

type Props = {
  id: string;
  title: string;
  /** CSS class controlling fixed position (e.g. "slot-tl"). */
  slot: string;
  zIndex?: number;
  hidden?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  /** Optional element rendered on the right side of the header (status, etc). */
  headerRight?: React.ReactNode;
  children: React.ReactNode;
};

const KEY = (id: string) => `obj-pose:collapsed:${id}`;

/** Glass HUD panel pinned to a slot. Click header to toggle collapse. */
export function Card({
  id, title, slot, zIndex = 20, hidden = false,
  collapsible = true, defaultCollapsed = false,
  headerRight,
  children,
}: Props) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (!collapsible) return false;
    if (typeof window === "undefined") return defaultCollapsed;
    const v = localStorage.getItem(KEY(id));
    return v == null ? defaultCollapsed : v === "1";
  });

  useEffect(() => {
    if (!collapsible) return;
    try { localStorage.setItem(KEY(id), collapsed ? "1" : "0"); } catch {}
  }, [id, collapsed, collapsible]);

  const onHeaderClick = () => {
    if (collapsible) setCollapsed((v) => !v);
  };

  const style: React.CSSProperties = { zIndex };
  if (hidden) style.display = "none";

  return (
    <div
      className={`card ${slot} ${collapsed ? "collapsed" : ""}`}
      style={style}
    >
      <div
        className="card-header"
        onClick={onHeaderClick}
        style={{ cursor: collapsible ? "pointer" : "default" }}
      >
        <span>{title}</span>
        <span className="card-header-actions">
          {headerRight}
          {collapsible && <span className="chevron" aria-hidden>▾</span>}
        </span>
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}
