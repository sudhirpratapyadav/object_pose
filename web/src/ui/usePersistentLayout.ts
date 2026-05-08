import { useEffect, useRef, useState } from "react";

export type Layout = {
  x: number;
  y: number;
  w?: number;
  h?: number;
  collapsed?: boolean;
};

const KEY = (id: string) => `obj-pose:layout:${id}`;

/** Persist a panel's position/size/collapsed state to localStorage by id. */
export function usePersistentLayout(id: string, initial: Layout) {
  const [layout, setLayout] = useState<Layout>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = localStorage.getItem(KEY(id));
      if (raw) return { ...initial, ...JSON.parse(raw) };
    } catch {}
    return initial;
  });
  const ref = useRef(layout);
  ref.current = layout;

  useEffect(() => {
    try { localStorage.setItem(KEY(id), JSON.stringify(layout)); } catch {}
  }, [id, layout]);

  return [layout, setLayout, ref] as const;
}
