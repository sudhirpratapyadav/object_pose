/**
 * Live log viewer.
 *
 * Reads stream.logLines (a ring buffer) and renders a scrollable list,
 * color-coded by level. Sticky auto-scroll: jumps to newest unless the
 * user has scrolled up. Filter dropdown: all / WARN+ / ERROR only.
 */

import { useEffect, useRef, useState } from "react";
import { LogLine } from "../protocol";


type Filter = "all" | "warn" | "error";

const LEVEL_COLOR: Record<string, string> = {
  DEBUG:    "rgba(255,255,255,0.5)",
  INFO:     "var(--text)",
  WARNING:  "var(--warn)",
  ERROR:    "var(--bad)",
  CRITICAL: "var(--bad)",
};

const LEVEL_RANK: Record<string, number> = {
  DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50,
};


export function LogViewer({ lines }: { lines: LogLine[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Track whether the user is at the bottom; stop auto-scroll if they scroll up.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 12;
    setAutoScroll(atBottom);
  };

  // Auto-scroll to bottom on new lines, only if user hasn't scrolled up.
  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines, autoScroll]);

  const minRank = filter === "error" ? 40 : filter === "warn" ? 30 : 0;
  const visible = lines.filter(l => (LEVEL_RANK[l.level] ?? 20) >= minRank);

  return (
    <>
      <div className="row" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
        <div className="label" style={{ flex: 1 }}>Log ({visible.length}/{lines.length})</div>
        <select
          className="select"
          value={filter}
          onChange={(e) => setFilter(e.target.value as Filter)}
          style={{ width: 110 }}
        >
          <option value="all">all levels</option>
          <option value="warn">WARN+</option>
          <option value="error">ERROR only</option>
        </select>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="mono"
        style={{
          fontSize: "0.74em",
          background: "rgba(0,0,0,0.32)",
          border: "1px solid var(--border)",
          borderRadius: 4,
          padding: 6,
          maxHeight: 260,
          minHeight: 160,
          overflow: "auto",
          lineHeight: 1.4,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {visible.length === 0 ? (
          <span style={{ opacity: 0.5 }}>(no log lines)</span>
        ) : (
          visible.map((l, i) => (
            <div
              key={`${l.ts}-${i}`}
              style={{ color: LEVEL_COLOR[l.level] ?? "var(--text)" }}
            >
              [{formatTs(l.ts)} {l.level} {l.source}] {l.msg}
            </div>
          ))
        )}
      </div>
    </>
  );
}


function formatTs(ts: number): string {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}
