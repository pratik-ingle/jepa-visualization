"use client";
import { useStore } from "@/lib/store";

function fmt(v: number) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e4) ? v.toExponential(3) : v.toFixed(4);
}

export default function Tooltip({ right = 12, bottom = 12 }: { right?: number; bottom?: number }) {
  const h = useStore((s) => s.hover);
  const trace = useStore((s) => s.trace);
  return (
    <div className="pointer-events-none absolute z-40 flex max-w-[min(92vw,360px)] flex-col items-end gap-2" style={{ right, bottom }}>
      {trace && (
        <div className="rounded-md border border-white/10 bg-black/70 px-3 py-1.5 text-xs text-slate-300 backdrop-blur">
          tracing {trace.space === "view" ? `view ${trace.token + 1}` : `token ${trace.token} (row ${Math.floor(trace.token / 8)}, col ${trace.token % 8})`} · click empty space to clear
        </div>
      )}
      {h && (
        <div className="w-full rounded-lg border border-white/10 bg-black/75 p-3 text-xs shadow-xl backdrop-blur">
          <div className="flex items-baseline justify-between gap-3">
            <span className="font-semibold text-slate-100">{h.label}</span>
            <span className="font-mono text-slate-400">{h.shape}</span>
          </div>
          <div className="mt-0.5 text-[11px] text-slate-400">ATOMS {h.atom}</div>
          <div className="mt-2 font-mono text-lg text-amber-300">{fmt(h.value)}</div>
          <div className="mt-1 text-slate-300">
            {h.rowName} · {h.colName}
          </div>
          <div className="text-[11px] text-slate-500">
            row {h.row}, col {h.col}
          </div>
          {h.note && <div className="mt-1.5 text-[11px] text-slate-400">{h.note}</div>}
        </div>
      )}
    </div>
  );
}
