"use client";
// Hyperparameter / optimiser / regulariser explorer (milestone 4): pick a setting -> the matching
// offline run is swapped in; curves update instantly; the checkpoint scrubber drives the 3D cloud.
import { useEffect, useState } from "react";
import { AXIS_LABEL, Cloud, fmtValue, loadCloud, Metrics, RunEntry, RunIndex } from "@/lib/runs";
import { store, useStore } from "@/lib/store";
import { LineChart, Series } from "./charts";
import { ALGO_COLOR, ProbeWidget, useMetrics } from "./widgets";

type Algo = "ijepa" | "lejepa";
const col = (m: Metrics | null, part: "train" | "eval", k: string) => ((m?.[part] as Record<string, (number | null)[]>)?.[k] ?? []) as (number | null)[];
const BASE: Record<Algo, string> = { ijepa: "ijepa-base", lejepa: "lejepa-base" };
const PRESET_LABEL: Record<string, string> = {
  "representation-collapse": "representation collapse",
  "masking-too-easy": "masking too easy",
  "tau-too-low": "τ too low",
  "no-warmup": "no warmup",
  "lambda-too-high": "λ too high",
};

// The two 2-way grids of the sweep: which settings span them, and their baseline values.
const GRID_AXES: Record<string, { a: string; b: string; base: Record<string, unknown>; title: string }> = {
  "lam-x-slices": { a: "reg.lam", b: "reg.slices", base: { "reg.lam": 0.05, "reg.slices": 256 }, title: "λ × SIGReg slices M" },
  "npred-x-scale": { a: "mask.npred", b: "mask.pred_scale", base: { "mask.npred": 4, "mask.pred_scale": [0.15, 0.2] }, title: "target blocks × target scale" },
};

function GridTable({ grid, runs, index, algo, current, onPick }: { grid: string; runs: string[]; index: RunIndex; algo: Algo; current: string; onPick: (r: string) => void }) {
  const g = GRID_AXES[grid];
  if (!g) return null;
  const entries = runs.map((n) => index.runs.find((r) => r.name === n)).filter(Boolean) as RunEntry[];
  const v = (r: RunEntry, k: string) => JSON.stringify(r.set[k] ?? g.base[k]);
  const rows = [...new Set(entries.map((r) => v(r, g.a)))].sort((x, y) => (JSON.parse(x) > JSON.parse(y) ? 1 : -1));
  const cols = [...new Set(entries.map((r) => v(r, g.b)))].sort((x, y) => (JSON.parse(x) > JSON.parse(y) ? 1 : -1));
  const key = algo === "ijepa" ? "target.linear_acc" : "backbone.linear_acc";
  const show = (s: string) => fmtValue(JSON.parse(s));
  return (
    <div>
      <div className="text-[11px] text-slate-400">{g.title} · linear-probe accuracy</div>
      <table className="mt-0.5 border-separate border-spacing-0.5 text-[10px]">
        <thead>
          <tr>
            <th className="px-1 text-left font-normal text-slate-500">{AXIS_LABEL[g.a] ?? g.a} ↓ / {AXIS_LABEL[g.b] ?? g.b} →</th>
            {cols.map((c) => (
              <th key={c} className="px-1 font-normal text-slate-400">{show(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((rv) => (
            <tr key={rv}>
              <td className="px-1 text-slate-400">{show(rv)}</td>
              {cols.map((cv) => {
                const e = entries.find((r) => v(r, g.a) === rv && v(r, g.b) === cv);
                const acc = e ? Number(e.final[key]) : NaN;
                return (
                  <td key={cv}>
                    {e ? (
                      <button onClick={() => onPick(e.name)} title={e.name} className={`w-14 rounded px-1 py-0.5 font-mono ${e.name === current ? "ring-1 ring-white" : ""}`} style={{ background: `rgba(98, 209, 150, ${Math.max(0.08, (acc - 0.2) / 0.55)})`, color: acc > 0.5 ? "#0b0e13" : "#e2e8f0" }}>
                        {(acc * 100).toFixed(1)}%
                      </button>
                    ) : (
                      <span className="block w-14 text-center text-slate-600">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function useCloud(name: string): Cloud | null {
  const [c, setC] = useState<Cloud | null>(null);
  useEffect(() => {
    let live = true;
    loadCloud(name).then((x) => live && setC(x)).catch(() => live && setC(null));
    return () => {
      live = false;
    };
  }, [name]);
  return c;
}

function pair(m: Metrics | null, b: Metrics | null, part: "train" | "eval", key: string, color: string, label: string, transform?: (v: number) => number): Series[] {
  const tf = (a: (number | null)[]) => (transform ? a.map((v) => (v === null ? null : transform(v))) : a);
  const out: Series[] = [];
  if (b && b !== m) out.push({ x: col(b, part, "epoch"), y: tf(col(b, part, key)), color: "#5c6672", dash: true, label: "baseline" });
  out.push({ x: col(m, part, "epoch"), y: tf(col(m, part, key)), color, label, width: 1.8 });
  return out;
}

export default function Explorer({ index }: { index: RunIndex | null }) {
  const regime = useStore((s) => s.regime);
  const runSel = useStore((s) => s.runSel);
  const ckpt = useStore((s) => s.ckpt);
  const [algo, setAlgo] = useState<Algo>("lejepa");
  const run = runSel[algo];
  const m = useMetrics(run), b = useMetrics(BASE[algo]);
  const cloud = useCloud(run);
  if (!index) return <div className="p-4 text-xs text-slate-400">Loading run index…</div>;
  const entry = index.runs.find((r) => r.name === run);
  const color = ALGO_COLOR[algo];
  const feat = algo === "ijepa" ? "target" : "backbone";
  const d = 64;
  const spe = m?.steps_per_epoch ?? 195;
  const marker = cloud ? cloud.steps[ckpt < 0 ? cloud.steps.length - 1 : Math.min(ckpt, cloud.steps.length - 1)] / spe : null;
  const pick = (name: string) => store.set((s) => ({ runSel: { ...s.runSel, [algo]: name } }));
  const lossKeys = algo === "ijepa" ? [["loss_pred", "prediction loss"]] : [["loss_total", "total"], ["loss_inv", "invariance"], ["loss_reg", "SIGReg"]];
  const f = entry?.final ?? {};

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 border-b border-white/10 px-3 py-2">
        {(["ijepa", "lejepa"] as const).map((a) => (
          <button key={a} onClick={() => setAlgo(a)} className={`rounded px-2 py-0.5 text-xs ${a === algo ? "bg-white/15 text-white" : "text-slate-400 hover:text-slate-200"}`} style={{ color: a === algo ? ALGO_COLOR[a] : undefined }}>
            {a === "ijepa" ? "I-JEPA" : "LeJEPA"}
          </button>
        ))}
        <span className="ml-auto text-[10px] text-slate-500">{index.runs.length} runs · 200 epochs each</span>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto px-3 py-2">
        <section>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Failure-mode presets</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(index.presets).map(([p, runs]) => (
              <button
                key={p}
                onClick={() => {
                  const sel = { ...store.get().runSel };
                  for (const r of runs) {
                    const e = index.runs.find((x) => x.name === r);
                    if (e) sel[e.algo] = r;
                  }
                  const first = index.runs.find((x) => x.name === runs[0]);
                  if (first) setAlgo(first.algo);
                  store.set({ runSel: sel, ckpt: -1 });
                }}
                className="rounded border border-red-400/30 bg-red-500/10 px-1.5 py-0.5 text-[11px] text-red-200 hover:bg-red-500/20"
              >
                {PRESET_LABEL[p] ?? p}
              </button>
            ))}
            <button onClick={() => pick(BASE[algo])} className="rounded border border-white/15 px-1.5 py-0.5 text-[11px] text-slate-300 hover:bg-white/10">
              baseline
            </button>
          </div>
        </section>
        <section className="space-y-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">One setting at a time (★ = baseline)</div>
          {Object.entries(index.axes[algo]).map(([axis, pts]) => (
            <div key={axis}>
              <div className="text-[11px] text-slate-400">{AXIS_LABEL[axis] ?? axis}</div>
              <div className="flex flex-wrap gap-1">
                {pts.map((p) => (
                  <button key={p.run} onClick={() => pick(p.run)} title={p.run} className={`rounded px-1.5 py-0.5 text-[11px] ${p.run === run ? "text-slate-900" : "bg-white/5 text-slate-300 hover:bg-white/10"}`} style={{ background: p.run === run ? color : undefined }}>
                    {fmtValue(p.value)}
                    {p.baseline ? " ★" : ""}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </section>
        {Object.entries(index.grids)
          .filter(([, runs]) => index.runs.find((r) => r.name === runs[0])?.algo === algo)
          .map(([grid, runs]) => (
            <section key={grid}>
              <GridTable grid={grid} runs={runs} index={index} algo={algo} current={run} onPick={pick} />
            </section>
          ))}
        {entry && (
          <section className="rounded-md border border-white/10 bg-black/25 p-2 text-[11px]">
            <div className="flex items-baseline justify-between">
              <span className="font-semibold" style={{ color }}>{run}</span>
              <span className={f[`${feat}.collapse`] === "complete" ? "text-red-300" : "text-slate-400"}>{f[`${feat}.collapse`] === "complete" ? "COLLAPSED" : entry.diverged ? "DIVERGED" : "healthy"}</span>
            </div>
            <div className="mt-0.5 font-mono text-[10px] text-slate-500">{Object.keys(entry.set).filter((k) => !k.startsWith("log.")).map((k) => `${k}=${JSON.stringify(entry.set[k])}`).join("  ") || "baseline settings"}</div>
            <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-0.5 text-slate-300">
              <span>k-NN <b>{(Number(f[`${feat}.knn_acc`]) * 100).toFixed(1)}%</b></span>
              <span>linear <b>{(Number(f[`${feat}.linear_acc`]) * 100).toFixed(1)}%</b></span>
              <span>RankMe <b>{Number(f[`${feat}.rankme`]).toFixed(1)}</b></span>
              <span>eff. dims <b>{Number(f[`${feat}.eff_dims`]).toFixed(1)}</b></span>
              <span>spread <b>{Number(f[`${feat}.spread`]).toPrecision(2)}</b></span>
              <span>{(entry.wall_s / 60).toFixed(0)} min</span>
            </div>
          </section>
        )}
        {cloud && (
          <section>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              checkpoint
              <input type="range" min={0} max={cloud.steps.length - 1} value={ckpt < 0 ? cloud.steps.length - 1 : ckpt} onChange={(e) => store.set({ ckpt: +e.target.value >= cloud.steps.length - 1 ? -1 : +e.target.value })} className="flex-1" />
              <span className="w-24 text-right font-mono text-slate-200">epoch {marker?.toFixed(marker < 1 ? 2 : 0)}</span>
            </label>
            <div className="text-[10px] text-slate-500">drives the PCA cloud in the scene (fly there with the “clouds” button)</div>
          </section>
        )}
        {regime === "train" && (
          <>
            {lossKeys.map(([k, lbl]) => (
              <LineChart key={k} title={`train ${lbl} (log)`} log marker={marker} series={pair(m, b, "train", k, color, run)} height={110} />
            ))}
            <LineChart title="k-NN accuracy (frozen features)" marker={marker} series={pair(m, b, "eval", `${feat}.knn_acc`, color, run)} height={110} />
            <LineChart title="effective dimensions" marker={marker} series={pair(m, b, "eval", `${feat}.iso_pr`, color, run, (v) => v * d)} height={100} />
            <LineChart title="encoder gradient norm (log)" log marker={marker} series={pair(m, b, "train", "grad_encoder", color, run)} height={100} />
            <LineChart title={algo === "ijepa" ? "learning rate and τ" : "learning rate"} marker={marker} series={[{ x: col(m, "train", "epoch"), y: col(m, "train", "lr").map((v) => (v === null ? null : v * 1000)), color, label: "lr ×1e3" }, ...(algo === "ijepa" ? [{ x: col(m, "train", "epoch"), y: col(m, "train", "tau"), color: "#7fb8e6", label: "τ" }] : [])]} height={90} />
          </>
        )}
        {regime === "val" && (
          <>
            <LineChart title="val vs train loss (log)" log marker={marker} series={[{ x: col(m, "train", "epoch"), y: col(m, "train", algo === "ijepa" ? "loss_pred" : "loss_total"), color: `${color}77`, label: "train" }, { x: col(m, "eval", "epoch"), y: col(m, "eval", algo === "ijepa" ? "val_loss_pred" : "val_loss_total"), color, width: 2, label: "val" }]} height={120} />
            <LineChart title="val loss vs baseline (log)" log marker={marker} series={pair(m, b, "eval", algo === "ijepa" ? "val_loss_pred" : "val_loss_total", color, run)} height={110} />
            <LineChart title="k-NN accuracy: the signal the val loss should track" marker={marker} series={pair(m, b, "eval", `${feat}.knn_acc`, color, run)} height={110} />
            <LineChart title="RankMe" marker={marker} series={pair(m, b, "eval", `${feat}.rankme`, color, run)} height={100} />
          </>
        )}
        {regime === "eval" && (
          <>
            <LineChart title="k-NN accuracy" marker={marker} series={pair(m, b, "eval", `${feat}.knn_acc`, color, run)} height={110} />
            <LineChart title="linear-probe accuracy" marker={marker} series={pair(m, b, "eval", `${feat}.linear_acc`, color, run)} height={110} />
            <LineChart title="isotropy (participation ratio)" marker={marker} series={pair(m, b, "eval", `${feat}.iso_pr`, color, run)} height={100} />
            <ProbeWidget index={index} />
          </>
        )}
      </div>
    </div>
  );
}
