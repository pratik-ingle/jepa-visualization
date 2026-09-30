"use client";
import { useMemo, useRef, useState } from "react";
import type { IJepaRun, LeJepaRun } from "@/lib/compute";
import type { ModelJson } from "@/lib/data";
import { store, useStore } from "@/lib/store";
import type { Rect } from "@/model/masks";
import type { Assets } from "./App";

const BLOCK_COLORS = ["#f56b5c", "#5ccc80", "#6b99ff", "#f5cc4d"];

function thumbUrl(img: Uint8Array): string {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const ctx = c.getContext("2d")!;
  const d = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    d.data[4 * i] = img[i];
    d.data[4 * i + 1] = img[1024 + i];
    d.data[4 * i + 2] = img[2048 + i];
    d.data[4 * i + 3] = 255;
  }
  ctx.putImageData(d, 0, 0);
  return c.toDataURL();
}

function Section({ title, color, children }: { title: string; color: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-white/10 px-4 py-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider" style={{ color }}>
        {title}
      </h2>
      <div className="space-y-2.5">{children}</div>
    </section>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="text-slate-400">{label}</span>
      <div className="flex items-center gap-1.5">{children}</div>
    </div>
  );
}

function Pills({ n, value, onChange, labels }: { n: number; value: number; onChange: (i: number) => void; labels?: string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {Array.from({ length: n }, (_, i) => (
        <button
          key={i}
          onClick={() => onChange(i)}
          className={`min-w-7 rounded px-1.5 py-0.5 text-xs ${i === value ? "bg-slate-200 text-slate-900" : "bg-white/5 text-slate-300 hover:bg-white/10"}`}
        >
          {labels?.[i] ?? i + 1}
        </button>
      ))}
    </div>
  );
}

function Seed({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <>
      <span className="font-mono text-slate-300">{value}</span>
      <button className="rounded bg-white/5 px-2 py-0.5 hover:bg-white/10" onClick={() => onChange(Math.floor(Math.random() * 1e6))} title="re-sample">
        re-sample
      </button>
    </>
  );
}

function CkptSelect({ model, value, onChange }: { model: ModelJson; value: string; onChange: (f: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="rounded bg-white/5 px-1.5 py-0.5 text-xs text-slate-200">
      {model.checkpoints.map((c) => (
        <option key={c.file} value={c.file} className="bg-slate-900">
          {c.file === "final.bin" ? `final (epoch ${Math.round(c.step / model.steps_per_epoch)})` : `step ${c.step} (epoch ${(c.step / model.steps_per_epoch).toFixed(c.step < 1000 ? 2 : 0)})`}
        </option>
      ))}
    </select>
  );
}

/** 8x8 token grid: grey = context, colours = target blocks (drag to move), dark = unused. */
export function MaskEditor({ run }: { run: IJepaRun }) {
  const G = 8, cell = 22;
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ i: number; dy: number; dx: number } | null>(null);
  const ctx = useMemo(() => new Set(run.masks.ctxIdx), [run]);
  const targets = run.masks.targets;
  const at = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [Math.floor((e.clientY - r.top) / cell), Math.floor((e.clientX - r.left) / cell)];
  };
  const move = (e: React.PointerEvent) => {
    if (!drag) return;
    const [gy, gx] = at(e);
    const t = targets[drag.i];
    const top = Math.min(Math.max(gy - drag.dy, 0), G - t.h), left = Math.min(Math.max(gx - drag.dx, 0), G - t.w);
    if (top === t.top && left === t.left) return;
    const next: Rect[] = targets.map((r, i) => (i === drag.i ? { ...r, top, left } : r));
    store.set({ targetsOverride: next });
  };
  return (
    <div
      ref={ref}
      className="relative shrink-0 touch-none select-none"
      style={{ width: G * cell, height: G * cell }}
      onPointerMove={move}
      onPointerUp={() => setDrag(null)}
      onPointerLeave={() => setDrag(null)}
    >
      {Array.from({ length: G * G }, (_, k) => (
        <div
          key={k}
          className="absolute border border-black/40"
          style={{ left: (k % G) * cell, top: Math.floor(k / G) * cell, width: cell, height: cell, background: ctx.has(k) ? "#9ea4ad" : "#171a21" }}
        />
      ))}
      {targets.map((t, i) => (
        <div
          key={i}
          className="absolute cursor-grab rounded-sm border-2 active:cursor-grabbing"
          style={{ left: t.left * cell, top: t.top * cell, width: t.w * cell, height: t.h * cell, borderColor: BLOCK_COLORS[i % 4], background: `${BLOCK_COLORS[i % 4]}55` }}
          onPointerDown={(e) => {
            (e.currentTarget.parentElement as HTMLElement).setPointerCapture(e.pointerId);
            const [gy, gx] = at(e);
            setDrag({ i, dy: gy - t.top, dx: gx - t.left });
            store.set({ tgtBlock: i });
          }}
        />
      ))}
    </div>
  );
}

export default function Panel({ assets, ij, le }: { assets: Assets; ij: IJepaRun; le: LeJepaRun }) {
  const s = {
    open: useStore((x) => x.sidebar),
    imageIdx: useStore((x) => x.imageIdx),
    maskSeed: useStore((x) => x.maskSeed),
    npred: useStore((x) => x.npred),
    predScaleMax: useStore((x) => x.predScaleMax),
    tgtBlock: useStore((x) => x.tgtBlock),
    head: useStore((x) => x.head),
    showTarget: useStore((x) => x.showTarget),
    ijepaCkpt: useStore((x) => x.ijepaCkpt),
    viewSeed: useStore((x) => x.viewSeed),
    viewIdx: useStore((x) => x.viewIdx),
    color: useStore((x) => x.color),
    lejepaCkpt: useStore((x) => x.lejepaCkpt),
    override: useStore((x) => x.targetsOverride),
    grad: useStore((x) => x.gradView),
  };
  const thumbs = useMemo(() => assets.images.map(thumbUrl), [assets.images]);
  const labels = assets.dataset.labels.showcase;
  const heads = (assets.ijepa.config.model as { heads: number }).heads;
  const Vg = (assets.lejepa.config.views as { n_global: number }).n_global;

  return (
    <>
      <aside
        className={`absolute bottom-0 left-0 top-11 z-20 w-full overflow-y-auto border-r border-white/10 bg-[#0d1117]/95 pt-2 backdrop-blur transition-transform sm:w-[330px] ${s.open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <header className="px-4 pb-3">
          <h1 className="text-base font-semibold text-slate-100">JEPA Visualization</h1>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">
            Two tiny joint-embedding predictive architectures trained on CIFAR-10, running live in your browser. Every value is computed by a real forward pass.
            Left: <span className="text-amber-300">I-JEPA</span> (EMA teacher + predictor). Right: <span className="text-emerald-300">LeJEPA</span> (one encoder + SIGReg).
          </p>
        </header>

        <Section title="Input image" color="#cbd5e1">
          <div className="grid grid-cols-8 gap-1">
            {thumbs.map((u, i) => (
              <button key={i} onClick={() => store.set({ imageIdx: i })} title={assets.dataset.classes[labels[i]]} className={`overflow-hidden rounded ring-2 ${i === s.imageIdx ? "ring-slate-100" : "ring-transparent hover:ring-white/30"}`}>
                <img src={u} alt={assets.dataset.classes[labels[i]]} className="block w-full [image-rendering:pixelated]" />
              </button>
            ))}
          </div>
          <div className="text-xs text-slate-400">class: {assets.dataset.classes[labels[s.imageIdx]]} (the models never saw labels)</div>
        </Section>

        <Section title="I-JEPA" color="#f0a04b">
          <Row label="weights">
            <CkptSelect model={assets.ijepa} value={s.ijepaCkpt} onChange={(f) => store.set({ ijepaCkpt: f })} />
          </Row>
          <Row label="mask seed">
            <Seed value={s.maskSeed} onChange={(v) => store.set({ maskSeed: v, targetsOverride: null })} />
          </Row>
          <Row label={`target blocks: ${s.npred}`}>
            <input type="range" min={1} max={4} value={s.npred} onChange={(e) => store.set({ npred: +e.target.value, targetsOverride: null, tgtBlock: 0 })} />
          </Row>
          <Row label={`max block scale: ${s.predScaleMax.toFixed(2)}`}>
            <input type="range" min={0.15} max={0.4} step={0.01} value={s.predScaleMax} onChange={(e) => store.set({ predScaleMax: +e.target.value, targetsOverride: null })} />
          </Row>
          <div className="flex items-start gap-3">
            <MaskEditor run={ij} />
            <div className="space-y-1 text-[11px] leading-snug text-slate-400">
              <p>Drag a coloured target block; the grey context is re-carved and both encoders re-run.</p>
              <p>
                context: <span className="text-slate-200">{ij.masks.ctxIdx.length}</span> tokens
                <br />
                targets: <span className="text-slate-200">{ij.masks.tgtIdx.length}×{ij.masks.tgtIdx[0].length}</span>
              </p>
              {s.override && (
                <button className="rounded bg-white/5 px-1.5 py-0.5 hover:bg-white/10" onClick={() => store.set({ targetsOverride: null })}>
                  reset
                </button>
              )}
            </div>
          </div>
          <Row label="predictor follows block">
            <Pills n={ij.masks.tgtIdx.length} value={s.tgtBlock} onChange={(i) => store.set({ tgtBlock: i })} />
          </Row>
          <Row label="attention head">
            <Pills n={heads} value={s.head} onChange={(i) => store.set({ head: i })} />
          </Row>
          <Row label="show">
            <Pills n={2} value={s.grad ? 1 : 0} labels={["activations", "gradients"]} onChange={(i) => store.set({ gradView: i === 1 })} />
          </Row>
          <Row label="target encoder">
            <Pills n={2} value={s.showTarget ? 1 : 0} labels={["hide", "show"]} onChange={(i) => store.set({ showTarget: i === 1 })} />
          </Row>
          <div className="rounded bg-white/5 px-2 py-1.5 font-mono text-xs text-slate-300">
            loss {ij.loss.toFixed(4)} · forward {ij.ms.toFixed(0)} ms
          </div>
        </Section>

        <Section title="LeJEPA" color="#62d196">
          <Row label="weights">
            <CkptSelect model={assets.lejepa} value={s.lejepaCkpt} onChange={(f) => store.set({ lejepaCkpt: f })} />
          </Row>
          <Row label="crop seed">
            <Seed value={s.viewSeed} onChange={(v) => store.set({ viewSeed: v })} />
          </Row>
          <Row label="colour augmentations">
            <Pills n={2} value={s.color ? 1 : 0} labels={["off", "on"]} onChange={(i) => store.set({ color: i === 1 })} />
          </Row>
          <Row label="view in the encoder">
            <Pills n={le.views.length} value={s.viewIdx} labels={le.views.map((_, i) => (i < Vg ? `G${i + 1}` : `L${i - Vg + 1}`))} onChange={(i) => store.set({ viewIdx: i })} />
          </Row>
          <div className="rounded bg-white/5 px-2 py-1.5 font-mono text-xs text-slate-300">
            invariance {le.inv.toFixed(4)} · forward {le.ms.toFixed(0)} ms
          </div>
        </Section>

        <Section title="How to look" color="#94a3b8">
          <ul className="list-disc space-y-1 pl-4 text-[11px] leading-relaxed text-slate-400">
            <li>Drag to pan, scroll or pinch to zoom, right-drag (two fingers) to orbit.</li>
            <li>Hover any cell for its value, shape and ATOMS reference.</li>
            <li>Click a token row to trace that token through every tensor it touches.</li>
            <li>
              Colours: <span className="text-orange-300">positive</span> / <span className="text-sky-400">negative</span> (per-tensor scale); attention and squared errors use a dark→yellow scale.
            </li>
          </ul>
          <p className="text-[11px] leading-relaxed text-slate-500">
            I-JEPA: Assran et al., CVPR 2023 (arXiv:2301.08243). LeJEPA: Balestriero &amp; LeCun, 2025 (arXiv:2511.08544).
          </p>
        </Section>
      </aside>
    </>
  );
}
