"use client";
// Walkthrough mode: phase list + commentary + phase widget in the side panel, timeline at the bottom.
import { useEffect } from "react";
import { animate, store, useStore } from "@/lib/store";
import { Phase, phasesFor } from "@/walkthrough/phases";

export function goPhase(id: string) {
  const ph = phasesFor(store.get().regime).find((p) => p.id === id);
  if (!ph) return;
  store.set((s) => ({ phase: id, flyTo: { id: ph.focus.join(","), n: (s.flyTo?.n ?? 0) + 1 }, trace: null }));
}

function step(d: number) {
  const list = phasesFor(store.get().regime);
  const i = list.findIndex((p) => p.id === store.get().phase);
  const j = Math.min(list.length - 1, Math.max(0, (i < 0 ? 0 : i) + d));
  goPhase(list[j].id);
}

export function useWalkKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (store.get().mode !== "walk" || (e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export function WalkPanel({ widget }: { widget: (p: Phase) => React.ReactNode }) {
  const regime = useStore((s) => s.regime);
  const phaseId = useStore((s) => s.phase);
  const open = useStore((s) => s.sidebar);
  const list = phasesFor(regime);
  const i = Math.max(0, list.findIndex((p) => p.id === phaseId));
  const ph = list[i];
  return (
    <aside className={`absolute bottom-14 left-0 top-11 z-20 flex w-full flex-col border-r border-white/10 bg-[#0d1117]/95 backdrop-blur transition-transform sm:w-[380px] ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <div className="flex-1 overflow-y-auto px-4 py-3 text-[12.5px] leading-relaxed text-slate-300">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          {ph.group}
          {ph.atom ? ` · atom ${ph.atom}` : ""} · {i + 1}/{list.length}
        </div>
        <h2 className="mb-2 text-base font-semibold text-slate-100">{ph.title}</h2>
        {ph.body}
        {ph.grad && (
          <div className="mb-2 rounded bg-orange-500/10 px-2 py-1.5 text-[11px] text-orange-200">
            gradient view: blocks show ∂L/∂(activation); orange = positive, blue = negative, dark = no gradient
            <button
              className="mt-1.5 block rounded bg-orange-500/20 px-2 py-0.5 text-orange-100 hover:bg-orange-500/30"
              onClick={() => animate(3200, (u) => store.set({ reveal: u }), () => store.set({ reveal: 1 }))}
            >
              ▶ play the backward pass (loss → input)
            </button>
          </div>
        )}
        {widget(ph)}
      </div>
      <div className="flex items-center gap-2 border-t border-white/10 px-3 py-2">
        <button className="rounded bg-white/5 px-3 py-1 text-xs hover:bg-white/10 disabled:opacity-30" disabled={i === 0} onClick={() => step(-1)}>
          ← back
        </button>
        <button className="ml-auto rounded bg-sky-500/20 px-3 py-1 text-xs text-sky-100 hover:bg-sky-500/30 disabled:opacity-30" disabled={i === list.length - 1} onClick={() => step(1)}>
          next →
        </button>
      </div>
    </aside>
  );
}

const GROUP_COLOR: Record<Phase["group"], string> = { Overview: "#94a3b8", "I-JEPA": "#f0a04b", LeJEPA: "#62d196", Shared: "#a3b6c9", Context: "#c792ea" };

export function Timeline() {
  const regime = useStore((s) => s.regime);
  const phaseId = useStore((s) => s.phase);
  const list = phasesFor(regime);
  return (
    <div className="absolute inset-x-0 bottom-0 z-30 flex h-14 items-center gap-1 overflow-x-auto border-t border-white/10 bg-[#0b0e13]/95 px-2">
      {list.map((p) => (
        <button
          key={p.id}
          onClick={() => goPhase(p.id)}
          title={p.title}
          className={`flex h-10 min-w-[74px] shrink-0 flex-col items-start justify-center rounded px-1.5 text-left ${p.id === phaseId ? "bg-white/15" : "hover:bg-white/5"}`}
          style={{ borderTop: `2px solid ${p.id === phaseId ? GROUP_COLOR[p.group] : GROUP_COLOR[p.group] + "44"}` }}
        >
          <span className="text-[9px] uppercase tracking-wider" style={{ color: GROUP_COLOR[p.group] }}>
            {p.atom ? `atom ${p.atom}` : p.group}
          </span>
          <span className="max-w-[120px] truncate text-[10.5px] text-slate-300">{p.title}</span>
        </button>
      ))}
    </div>
  );
}
