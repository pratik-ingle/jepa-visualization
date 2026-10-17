"use client";
import { REPO_URL } from "@/lib/data";
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
        <a href={REPO_URL} target="_blank" rel="noreferrer" className={`${btn} flex items-center gap-1.5 text-slate-200`} title="Source code, data and training runs on GitHub">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          <span className="hidden sm:inline">GitHub</span>
          <span className="sr-only sm:hidden">GitHub repository</span>
        </a>
      </div>
    </div>
  );
}
