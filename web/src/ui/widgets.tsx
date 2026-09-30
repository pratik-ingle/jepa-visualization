"use client";
// Phase widgets: every number comes from a real forward pass in the browser or a real logged run.
import { useEffect, useMemo, useState } from "react";
import type { BatchFeats } from "@/lib/batch";
import type { IJepaRun } from "@/lib/compute";
import { AXIS_LABEL, fmtValue, loadMetrics, Metrics, RunIndex } from "@/lib/runs";
import { animate, store, useStore } from "@/lib/store";
import type { SigregOut } from "@/model/lejepa";
import { CLASS_COLORS } from "@/scene/layout";
import { spearman } from "@/lib/stats";
import { HistDensity, LineChart, Scatter, Series } from "./charts";
import { MaskEditor } from "./Panel";

export const ALGO_COLOR = { ijepa: "#f0a04b", lejepa: "#62d196" };

export function useMetrics(name: string | null): Metrics | null {
  const [m, setM] = useState<Metrics | null>(null);
  useEffect(() => {
    if (!name) return;
    let live = true;
    loadMetrics(name).then((x) => live && setM(x)).catch(() => live && setM(null));
    return () => {
      live = false;
    };
  }, [name]);
  return m;
}

const col = (m: Metrics | null, part: "train" | "eval", k: string) => ((m?.[part] as Record<string, (number | null)[]>)?.[k] ?? []) as (number | null)[];

function Box({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="mb-2 rounded-md border border-white/10 bg-black/20 p-2">
      {title && <div className="mb-1 text-[11px] font-semibold text-slate-300">{title}</div>}
      {children}
    </div>
  );
}

function Slider({ label, min, max, step, value, onChange, fmt }: { label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="flex items-center justify-between gap-2 text-[11px] text-slate-400">
      <span className="shrink-0">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} className="w-32" />
      <span className="w-14 text-right font-mono text-slate-200">{fmt ? fmt(value) : value}</span>
    </label>
  );
}

// ------------------------------------------------------------------ atom 2
export function MaskWidget({ ij }: { ij: IJepaRun }) {
  return (
    <Box title="Mask editor (drag the coloured blocks)">
      <div className="flex gap-3">
        <MaskEditor run={ij} />
        <div className="text-[11px] text-slate-400">
          context <span className="text-slate-200">{ij.masks.ctxIdx.length}</span> tokens
          <br />
          targets <span className="text-slate-200">{ij.masks.tgtIdx.length}×{ij.masks.tgtIdx[0].length}</span>
          <br />
          loss <span className="font-mono text-slate-200">{ij.loss.toFixed(4)}</span>
          <br />
          <button className="mt-1 rounded bg-white/5 px-1.5 py-0.5 hover:bg-white/10" onClick={() => store.set({ maskSeed: Math.floor(Math.random() * 1e6), targetsOverride: null })}>
            re-sample
          </button>
        </div>
      </div>
    </Box>
  );
}

// ------------------------------------------------------------------ atom 8
export function OptimWidget() {
  const run = useStore((s) => s.runSel.ijepa);
  const m = useMetrics(run);
  const ep = col(m, "train", "epoch");
  return (
    <>
      <Box title={`Schedules logged by ${run}`}>
        <LineChart series={[{ x: ep, y: col(m, "train", "lr").map((v) => (v === null ? null : v * 1000)), color: "#f0a04b", label: "learning rate ×1e3" }, { x: ep, y: col(m, "train", "wd"), color: "#7fb8e6", label: "weight decay" }]} height={120} />
      </Box>
      <Box title="Gradient norm (log)">
        <LineChart log series={[{ x: ep, y: col(m, "train", "grad_encoder"), color: "#f0a04b", label: "context encoder θ" }, { x: ep, y: col(m, "train", "grad_head"), color: "#c792ea", label: "predictor φ" }]} height={120} />
      </Box>
    </>
  );
}

// ------------------------------------------------------------------ atom 9
export function EmaWidget() {
  const tau = useStore((s) => s.tau);
  const steps = useStore((s) => s.emaSteps);
  const run = useStore((s) => s.runSel.ijepa);
  const m = useMetrics(run);
  return (
    <>
      <Box title="Apply the EMA update to the real weights">
        <Slider label="τ" min={0.9} max={0.9999} step={0.0001} value={tau} onChange={(v) => store.set({ tau: v })} fmt={(v) => v.toFixed(4)} />
        <Slider label="steps k" min={0} max={3} step={0.01} value={Math.log10(steps)} onChange={(v) => store.set({ emaSteps: Math.max(1, Math.round(10 ** v)) })} fmt={() => String(steps)} />
        <button
          className="mt-1 rounded bg-sky-500/20 px-2 py-0.5 text-[11px] text-sky-100 hover:bg-sky-500/30"
          onClick={() => animate(4000, (u) => store.set({ emaSteps: Math.max(1, Math.round(10 ** (u * 3.3))) }))}
        >
          ▶ animate k = 1 → 2000 updates
        </button>
        <p className="mt-1 text-[11px] text-slate-400">
          θ̄ − θ shrinks by τ^k = <span className="font-mono text-slate-200">{Math.pow(tau, steps).toPrecision(3)}</span>. With τ = 0.996 the teacher remembers ~{Math.round(1 / (1 - 0.996))} steps.
        </p>
      </Box>
      <Box title={`‖θ̄ − θ‖/‖θ‖ and τ during training (${run})`}>
        <LineChart series={[{ x: col(m, "eval", "epoch"), y: col(m, "eval", "ema_gap"), color: "#7fb8e6", label: "EMA gap" }]} height={110} />
        <LineChart series={[{ x: col(m, "train", "epoch"), y: col(m, "train", "tau"), color: "#e2e8f0", label: "τ schedule" }]} height={90} />
      </Box>
    </>
  );
}

// ------------------------------------------------------------------ atom 13
export interface SigState {
  z: Float32Array;
  n: number;
  K: number;
  A: Float32Array;
  M: number;
  out: SigregOut;
  t: Float32Array;
}

function CycleButton() {
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!on) return;
    const id = setInterval(() => store.set((s) => ({ dirSeed: s.dirSeed + 1 })), 900);
    return () => clearInterval(id);
  }, [on]);
  return (
    <button className="rounded bg-pink-500/20 px-2 py-0.5 text-pink-200 hover:bg-pink-500/30" onClick={() => setOn(!on)}>
      {on ? "■ stop" : "▶ cycle directions"}
    </button>
  );
}

export function SigregWidget({ sig, progress }: { sig: SigState | null; progress: number }) {
  const slice = useStore((s) => s.sigSlice);
  const M = useStore((s) => s.sigM);
  if (!sig)
    return (
      <Box title="Computing z for 256 held-out images in a Web Worker…">
        <div className="h-1.5 w-full overflow-hidden rounded bg-white/10">
          <div className="h-full bg-pink-400" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      </Box>
    );
  const m = Math.min(slice, sig.M - 1);
  const y = new Float32Array(sig.n);
  for (let i = 0; i < sig.n; i++) y[i] = sig.out.y.data[i * sig.M + m];
  const T = sig.t.length;
  const t = Array.from(sig.t);
  const re = Array.from(sig.out.ecfRe.data.subarray(m * T, (m + 1) * T));
  const im = Array.from(sig.out.ecfIm.data.subarray(m * T, (m + 1) * T));
  const phi = t.map((v) => Math.exp(-0.5 * v * v));
  const ep = sig.out.ep.data[m];
  return (
    <>
      <Box title="Directions">
        <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-400">
          M =
          {[2, 8, 16, 32, 64].map((v) => (
            <button key={v} onClick={() => store.set({ sigM: v, sigSlice: 0 })} className={`rounded px-1.5 py-0.5 ${v === M ? "bg-slate-200 text-slate-900" : "bg-white/5 hover:bg-white/10"}`}>
              {v}
            </button>
          ))}
          <span className="ml-auto flex gap-1">
            <button className="rounded bg-pink-500/20 px-2 py-0.5 text-pink-200 hover:bg-pink-500/30" onClick={() => store.set((s) => ({ dirSeed: s.dirSeed + 1 }))}>
              re-draw
            </button>
            <CycleButton />
          </span>
        </div>
        <Slider label="inspect slice" min={0} max={sig.M - 1} step={1} value={m} onChange={(v) => store.set({ sigSlice: v })} />
        <p className="text-[11px] text-slate-400">
          SIGReg (mean over slices) = <span className="font-mono text-pink-200">{sig.out.value.toFixed(3)}</span> · this slice EP = <span className="font-mono text-pink-200">{ep.toFixed(3)}</span>
        </p>
      </Box>
      <Box title={`Slice ${m}: histogram of y = z·a over 256 images vs N(0, 1)`}>
        <HistDensity samples={y} density={(x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)} />
      </Box>
      <Box title="Characteristic function: empirical (colour) vs Gaussian e^(−t²/2) (white)">
        <LineChart
          xLabel="t"
          series={[
            { x: t, y: phi, color: "#e2e8f0", label: "target Re" },
            { x: t, y: re, color: "#ef8fb7", label: "Re φ̂(t)" },
            { x: t, y: t.map(() => 0), color: "#94a3b8", dash: true, label: "target Im" },
            { x: t, y: im, color: "#7fb8e6", label: "Im φ̂(t)" },
          ]}
          height={130}
        />
      </Box>
    </>
  );
}

// ------------------------------------------------------------------ atom 15
export function CloudWidget({ index }: { index: RunIndex | null }) {
  const runSel = useStore((s) => s.runSel);
  const ckpt = useStore((s) => s.ckpt);
  const cloudKey = useStore((s) => s.cloudKey);
  const mi = useMetrics(runSel.ijepa), ml = useMetrics(runSel.lejepa);
  const names = (algo: "ijepa" | "lejepa") => index?.runs.filter((r) => r.algo === algo).map((r) => r.name) ?? [];
  const quick: [string, "ijepa" | "lejepa", string][] = [
    ["I-JEPA baseline", "ijepa", "ijepa-base"],
    ["no stop-grad", "ijepa", "ijepa-no-stopgrad"],
    ["τ = 0.9", "ijepa", "ijepa-tau-low"],
    ["LeJEPA baseline", "lejepa", "lejepa-base"],
    ["λ = 0", "lejepa", "lejepa-lambda0"],
    ["2 fixed slices", "lejepa", "lejepa-M2-fixed"],
  ];
  return (
    <>
      <Box title="Runs shown as clouds">
        <div className="mb-1.5 flex flex-wrap gap-1">
          {quick.map(([label, algo, name]) => (
            <button key={name} onClick={() => store.set((s) => ({ runSel: { ...s.runSel, [algo]: name } }))} className={`rounded px-1.5 py-0.5 text-[11px] ${runSel[algo] === name ? "bg-slate-200 text-slate-900" : "bg-white/5 text-slate-300 hover:bg-white/10"}`}>
              {label}
            </button>
          ))}
        </div>
        {(["ijepa", "lejepa"] as const).map((a) => (
          <div key={a} className="mb-1 flex items-center gap-2 text-[11px]">
            <span style={{ color: ALGO_COLOR[a] }} className="w-12">{a === "ijepa" ? "I-JEPA" : "LeJEPA"}</span>
            <select value={runSel[a]} onChange={(e) => store.set((s) => ({ runSel: { ...s.runSel, [a]: e.target.value } }))} className="flex-1 rounded bg-white/5 px-1 py-0.5 text-slate-200">
              {names(a).map((n) => (
                <option key={n} value={n} className="bg-slate-900">{n}</option>
              ))}
            </select>
            <select
              value={cloudKey[a]}
              onChange={(e) => store.set((s) => ({ cloudKey: { ...s.cloudKey, [a]: e.target.value } as typeof s.cloudKey }))}
              className="rounded bg-white/5 px-1 py-0.5 text-slate-200"
            >
              {(a === "ijepa" ? ["target", "context"] : ["backbone", "z"]).map((k) => (
                <option key={k} value={k} className="bg-slate-900">{k}</option>
              ))}
            </select>
          </div>
        ))}
        <Slider label="checkpoint" min={0} max={23} step={1} value={ckpt < 0 ? 23 : ckpt} onChange={(v) => store.set({ ckpt: v >= 23 ? -1 : v })} fmt={(v) => (v >= 23 ? "final" : `#${v}`)} />
      </Box>
      <GeometryCharts m={mi} name={runSel.ijepa} feat={cloudKey.ijepa} d={64} color={ALGO_COLOR.ijepa} />
      <GeometryCharts m={ml} name={runSel.lejepa} feat={cloudKey.lejepa} d={cloudKey.lejepa === "z" ? 32 : 64} color={ALGO_COLOR.lejepa} />
    </>
  );
}

export function GeometryCharts({ m, name, feat, d, color }: { m: Metrics | null; name: string; feat: string; d: number; color: string }) {
  const ep = col(m, "eval", "epoch");
  const v = col(m, "eval", `${feat}.var_mean`), nm = col(m, "eval", `${feat}.norm_mean`);
  const spread = v.map((x, i) => (x === null || nm[i] === null ? null : (x * d) / Math.max(1e-12, nm[i]! ** 2)));
  const eff = col(m, "eval", `${feat}.iso_pr`).map((x) => (x === null ? null : x * d));
  return (
    <Box title={`${name} · ${feat}`}>
      <LineChart series={[{ x: ep, y: col(m, "eval", `${feat}.rankme`), color, label: "RankMe" }, { x: ep, y: eff, color: "#e2e8f0", label: "effective dims" }]} height={100} />
      <LineChart log series={[{ x: ep, y: spread, color, label: "spread (log)" }]} height={80} />
      <LineChart series={[{ x: ep, y: col(m, "eval", `${feat}.knn_acc`), color, label: "k-NN acc" }, { x: ep, y: col(m, "eval", `${feat}.linear_acc`), color: "#e2e8f0", dash: true, label: "linear acc" }]} height={90} />
    </Box>
  );
}

// ------------------------------------------------------------------ validation
/** Across logged runs: does the (validation) loss predict probe accuracy? */
export function LossVsProbe({ index }: { index: RunIndex | null }) {
  if (!index) return null;
  const rows = (algo: "ijepa" | "lejepa", norm: boolean) =>
    index.runs
      .filter((r) => r.algo === algo && Number.isFinite(Number(r.final.val_loss)) && (!norm || Number(r.final.lam) > 0))
      .map((r) => {
        const key = algo === "ijepa" ? "target" : "backbone";
        const loss = Number(r.final.val_loss) / (norm ? Number(r.final.lam) ** 0.4 : 1);
        return { name: r.name, loss, acc: Number(r.final[`${key}.linear_acc`]) };
      })
      .filter((r) => Number.isFinite(r.acc) && Number.isFinite(r.loss));
  const block = (algo: "ijepa" | "lejepa", norm: boolean, title: string) => {
    const d = rows(algo, norm);
    const rho = d.length > 2 ? spearman(d.map((r) => r.loss), d.map((r) => r.acc)) : NaN;
    return (
      <Box title={`${title} · Spearman ρ(loss, linear acc) = ${Number.isFinite(rho) ? rho.toFixed(2) : "—"} over ${d.length} runs`}>
        <Scatter points={d.map((r) => ({ x: Math.log10(Math.max(r.loss, 1e-12)), y: r.acc, color: ALGO_COLOR[algo], label: r.name }))} xLabel={`log10 final val loss${norm ? " / λ^0.4" : ""}`} yLabel="linear acc" height={150} onPick={(name) => store.set((s) => ({ runSel: { ...s.runSel, [algo]: name } }))} />
      </Box>
    );
  };
  return (
    <>
      {block("ijepa", false, "I-JEPA runs")}
      {block("lejepa", false, "LeJEPA runs")}
      {block("lejepa", true, "LeJEPA runs, loss / λ^0.4 (paper Eq. 10)")}
      <p className="text-[11px] text-slate-500">A negative ρ means lower loss ↔ better probes (a useful training signal). Each point is one 200-epoch run; click to load it.</p>
    </>
  );
}

/** Within one run: Spearman ρ between val loss and linear-probe accuracy over eval checkpoints (epoch >= 10). */
function withinRho(m: Metrics | null, lossKey: string, accKey: string): { rho: number; n: number } {
  const ep = col(m, "eval", "epoch"), v = col(m, "eval", lossKey), a = col(m, "eval", accKey);
  const pts = ep.map((e, i) => [e, v[i], a[i]] as const).filter(([e, x, y]) => e !== null && e >= 10 && x !== null && y !== null);
  return { rho: pts.length > 5 ? spearman(pts.map((p) => p[1]!), pts.map((p) => p[2]!)) : NaN, n: pts.length };
}

export function ValWidget({ index }: { index?: RunIndex | null }) {
  const runSel = useStore((s) => s.runSel);
  const mi = useMetrics(runSel.ijepa), ml = useMetrics(runSel.lejepa);
  const ri = withinRho(mi, "val_loss_pred", "target.linear_acc"), rl = withinRho(ml, "val_loss_total", "backbone.linear_acc");
  const fmtR = (r: { rho: number; n: number }) => (Number.isFinite(r.rho) ? `${r.rho >= 0 ? "+" : ""}${r.rho.toFixed(2)}` : "—");
  return (
    <>
      <Box title="Within one run: does val loss track the probe? (checkpoints after epoch 10)">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded bg-white/5 p-1.5">
            <div style={{ color: ALGO_COLOR.ijepa }}>{runSel.ijepa}</div>
            <div className="font-mono text-lg text-slate-100">ρ = {fmtR(ri)}</div>
            <div className="text-slate-500">{ri.n} checkpoints</div>
          </div>
          <div className="rounded bg-white/5 p-1.5">
            <div style={{ color: ALGO_COLOR.lejepa }}>{runSel.lejepa}</div>
            <div className="font-mono text-lg text-slate-100">ρ = {fmtR(rl)}</div>
            <div className="text-slate-500">{rl.n} checkpoints</div>
          </div>
        </div>
        <p className="mt-1 text-[11px] text-slate-500">ρ near −1: as the loss falls the probe improves, so the loss is a usable signal. Near 0 or positive: it is not.</p>
      </Box>
      <Box title={`I-JEPA (${runSel.ijepa}): val vs train loss, and k-NN`}>
        <LineChart log series={[{ x: col(mi, "train", "epoch"), y: col(mi, "train", "loss_pred"), color: "#f0a04b88", label: "train loss" }, { x: col(mi, "eval", "epoch"), y: col(mi, "eval", "val_loss_pred"), color: "#f0a04b", width: 2, label: "val loss" }]} height={110} />
        <LineChart series={[{ x: col(mi, "eval", "epoch"), y: col(mi, "eval", "target.knn_acc"), color: "#e2e8f0", label: "k-NN acc (target enc.)" }]} height={90} />
      </Box>
      <Box title={`LeJEPA (${runSel.lejepa}): val loss terms and linear probe`}>
        <LineChart log series={[{ x: col(ml, "eval", "epoch"), y: col(ml, "eval", "val_loss_total"), color: "#62d196", width: 2, label: "val total" }, { x: col(ml, "eval", "epoch"), y: col(ml, "eval", "val_loss_inv"), color: "#e6c35a", label: "val invariance" }, { x: col(ml, "eval", "epoch"), y: col(ml, "eval", "val_loss_reg"), color: "#ef8fb7", label: "val SIGReg" }]} height={110} />
        <LineChart series={[{ x: col(ml, "eval", "epoch"), y: col(ml, "eval", "backbone.linear_acc"), color: "#e2e8f0", label: "linear acc (backbone)" }]} height={90} />
      </Box>
      <p className="mb-2 text-[11px] text-slate-500">Try the collapse runs in the explorer: their val loss is the lowest of all.</p>
      <LossVsProbe index={index ?? null} />
    </>
  );
}

// ------------------------------------------------------------------ evaluation: live k-NN
export function looKnn(F: Float32Array, n: number, d: number, labels: number[], k = 10): number {
  const X = new Float32Array(F);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += X[i * d + j] ** 2;
    s = Math.sqrt(s) || 1;
    for (let j = 0; j < d; j++) X[i * d + j] /= s;
  }
  let correct = 0;
  const sims = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += X[i * d + c] * X[j * d + c];
      sims[j] = j === i ? -Infinity : s;
    }
    const idx = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, k);
    const votes = new Float64Array(10);
    for (const j of idx) votes[labels[j]] += Math.exp(sims[j] / 0.07);
    let best = 0;
    for (let c = 1; c < 10; c++) if (votes[c] > votes[best]) best = c;
    if (best === labels[i]) correct++;
  }
  return correct / n;
}

export function KnnWidget({ fi, fl, pi, pl, labels, classes }: { fi: BatchFeats | null; fl: BatchFeats | null; pi: number; pl: number; labels: number[]; classes: string[] }) {
  const accI = useMemo(() => (fi ? looKnn(fi.feats.target, fi.n, 64, labels) : null), [fi, labels]);
  const accL = useMemo(() => (fl ? looKnn(fl.feats.backbone, fl.n, 64, labels) : null), [fl, labels]);
  const ckI = useStore((s) => s.ijepaCkpt), ckL = useStore((s) => s.lejepaCkpt);
  const row = (name: string, color: string, acc: number | null, p: number, ck: string) => (
    <div className="mb-1.5">
      <div className="flex justify-between text-[11px]">
        <span style={{ color }}>{name}</span>
        <span className="text-slate-500">{ck.replace(".bin", "")}</span>
      </div>
      {acc === null ? (
        <div className="h-1.5 w-full overflow-hidden rounded bg-white/10">
          <div className="h-full" style={{ width: `${Math.round(p * 100)}%`, background: color }} />
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="h-3 flex-1 overflow-hidden rounded bg-white/10">
            <div className="h-full" style={{ width: `${acc * 100}%`, background: color }} />
          </div>
          <span className="w-12 text-right font-mono text-sm text-slate-100">{(acc * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
  return (
    <Box title="Leave-one-out k-NN on 256 held-out images (computed live)">
      {row("I-JEPA target encoder", ALGO_COLOR.ijepa, accI, pi, ckI)}
      {row("LeJEPA backbone", ALGO_COLOR.lejepa, accL, pl, ckL)}
      <p className="text-[11px] text-slate-500">
        Chance is 10%. Switch the weights in the free-explore controls (e.g. step 0 vs final) to watch it change. Classes:{" "}
        {classes.map((c, i) => (
          <span key={c} style={{ color: CLASS_COLORS[i] }}>
            {c}{i < 9 ? ", " : ""}
          </span>
        ))}
      </p>
    </Box>
  );
}

// ------------------------------------------------------------------ evaluation: probe vs isotropy, axes
export function ProbeWidget({ index }: { index: RunIndex | null }) {
  const runSel = useStore((s) => s.runSel);
  if (!index) return null;
  const pts = index.runs.map((r) => {
    const key = r.algo === "ijepa" ? "target" : "backbone";
    return {
      x: Number(r.final[`${key}.iso_pr`]), y: Number(r.final[`${key}.linear_acc`]), color: ALGO_COLOR[r.algo], label: r.name, hl: runSel[r.algo] === r.name,
    };
  });
  return (
    <Box title="Linear-probe accuracy vs isotropy of the frozen features (every run)">
      <Scatter points={pts} xLabel="isotropy (participation ratio)" yLabel="linear acc" onPick={(name) => {
        const r = index.runs.find((x) => x.name === name);
        if (r) store.set((s) => ({ runSel: { ...s.runSel, [r.algo]: name } }));
      }} />
      <p className="text-[11px] text-slate-500">
        <span style={{ color: ALGO_COLOR.ijepa }}>● I-JEPA</span> <span style={{ color: ALGO_COLOR.lejepa }}>● LeJEPA</span> · click a point to load its cloud
      </p>
    </Box>
  );
}

export function AxesWidget({ index }: { index: RunIndex | null }) {
  if (!index) return null;
  const final = (name: string, algo: string) => index.runs.find((r) => r.name === name)?.final[`${algo === "ijepa" ? "target" : "backbone"}.linear_acc`];
  return (
    <>
      {(["ijepa", "lejepa"] as const).map((algo) => (
        <Box key={algo} title={algo === "ijepa" ? "I-JEPA: linear-probe accuracy per setting" : "LeJEPA: linear-probe accuracy per setting"}>
          {Object.entries(index.axes[algo]).map(([axis, pts]) => (
            <div key={axis} className="mb-1.5">
              <div className="text-[10px] text-slate-400">{AXIS_LABEL[axis] ?? axis}</div>
              {pts.map((p) => {
                const acc = Number(final(p.run, algo) ?? 0);
                return (
                  <button key={p.run} onClick={() => store.set((s) => ({ runSel: { ...s.runSel, [algo]: p.run } }))} className="flex w-full items-center gap-1.5 text-left">
                    <span className="w-24 truncate text-[10px] text-slate-300">{fmtValue(p.value)}{p.baseline ? " ★" : ""}</span>
                    <span className="h-2 flex-1 overflow-hidden rounded bg-white/5">
                      <span className="block h-full" style={{ width: `${acc * 100}%`, background: ALGO_COLOR[algo], opacity: p.baseline ? 1 : 0.7 }} />
                    </span>
                    <span className="w-9 text-right font-mono text-[10px] text-slate-300">{(acc * 100).toFixed(0)}%</span>
                  </button>
                );
              })}
            </div>
          ))}
        </Box>
      ))}
    </>
  );
}

export type { Series };
