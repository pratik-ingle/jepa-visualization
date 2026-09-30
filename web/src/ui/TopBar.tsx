"use client";
import { Regime, store, useStore } from "@/lib/store";
import { phasesFor } from "@/walkthrough/phases";

const REGIMES: [Regime, string, string][] = [
  ["train", "Training", "Train"],
  ["val", "Validation", "Val"],
  ["eval", "Evaluation", "Eval"],
];

const btn = "rounded-md border border-white/10 px-2 py-1 text-xs hover:bg-white/10 sm:px-2.5";

export default function TopBar() {
  const regime = useStore((s) => s.regime);
  const mode = useStore((s) => s.mode);
  const sidebar = useStore((s) => s.sidebar);
  const explorer = useStore((s) => s.explorer);
  return (
    <div className="absolute inset-x-0 top-0 z-30 flex h-11 items-center gap-1.5 border-b border-white/10 bg-[#0b0e13]/90 px-2 backdrop-blur sm:px-3">
      <button className={`${btn} ${sidebar ? "bg-white/10" : ""}`} onClick={() => store.set({ sidebar: !sidebar })} title="toggle side panel">
        ☰
      </button>
      <span className="hidden text-sm font-semibold text-slate-100 md:inline">JEPA Visualization</span>
      <div className="flex rounded-md border border-white/10 p-0.5 sm:ml-1">
        {REGIMES.map(([r, label, short]) => (
          <button
            key={r}
            onClick={() => {
              const ph = phasesFor(r)[0];
              store.set((s) => ({ regime: r, phase: ph.id, gradView: false, flyTo: { id: ph.focus.join(","), n: (s.flyTo?.n ?? 0) + 1 } }));
            }}
            className={`rounded px-1.5 py-0.5 text-xs sm:px-2 ${r === regime ? "bg-slate-200 text-slate-900" : "text-slate-300 hover:bg-white/10"}`}
          >
            <span className="hidden sm:inline">{label}</span>
            <span className="sm:hidden">{short}</span>
          </button>
        ))}
      </div>
      <div className="flex rounded-md border border-white/10 p-0.5">
        {(["walk", "free"] as const).map((m) => (
          <button key={m} onClick={() => store.set({ mode: m, sidebar: true })} className={`rounded px-1.5 py-0.5 text-xs sm:px-2 ${m === mode ? "bg-white/15 text-white" : "text-slate-400 hover:text-slate-200"}`}>
            <span className="hidden sm:inline">{m === "walk" ? "Walkthrough" : "Free explore"}</span>
            <span className="sm:hidden">{m === "walk" ? "Walk" : "Free"}</span>
          </button>
        ))}
      </div>
      <div className="ml-auto flex shrink-0 gap-1">
        <button className={`${btn} hidden sm:block`} onClick={() => store.set((s) => ({ flyTo: { id: "ijepa:cloud,lejepa:cloud", n: (s.flyTo?.n ?? 0) + 1 } }))}>
          clouds
        </button>
        <button className={btn} onClick={() => store.set((s) => ({ resetView: s.resetView + 1 }))} title="reset view">
          <span className="hidden sm:inline">reset view</span>
          <span className="sm:hidden">⤢</span>
        </button>
        <button className={`${btn} ${explorer ? "bg-white/15 text-white" : ""}`} onClick={() => store.set({ explorer: !explorer })} title="hyperparameter explorer">
          <span className="hidden sm:inline">explorer</span>
          <span className="sm:hidden">runs</span>
        </button>
      </div>
    </div>
  );
}
