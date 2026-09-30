// Scene layout: a pure function from recorded activations (or their gradients) + UI state to
// positioned blocks. World units: 1 cell = 1 unit; x right, y up; data flows downward.
// I-JEPA on the left, LeJEPA on the right; batch-level views (EMA, SIGReg, embedding clouds) below.
import type { IJepaRun, LeJepaRun, Rec } from "@/lib/compute";
import type { Cloud } from "@/lib/runs";
import type { SigregOut } from "@/model/lejepa";
import { Tensor } from "@/model/tensor";
import type { Params } from "@/model/vit";
import type { BlockSpec, Mode } from "./TensorBox";

export interface ImageSpec {
  id: string;
  rgb: Float32Array;
  res: number;
  x: number;
  y: number;
  size: number;
  outline?: string;
  view?: number;
}

export interface Frame {
  id: string;
  title: string;
  subtitle?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}

export interface Label {
  text: string;
  x: number;
  y: number;
  size: number;
  color?: string;
}

export interface CloudSpec {
  id: string;
  cx: number; // centre of the cube
  cy: number;
  size: number;
  pos: Float32Array; // n * 3, already in world units relative to the centre
  colors: Float32Array; // n * 3 (linear 0-1)
  n: number;
}

export interface Layout {
  blocks: BlockSpec[];
  images: ImageSpec[];
  frames: Frame[];
  labels: Label[];
  clouds: CloudSpec[];
  bounds: { x0: number; x1: number; y0: number; y1: number };
}

const GAP = 8;
const ROWGAP = 16;
const LAYERGAP = 34;
const PAD = 16;
export const TINT = { ctx: "#f0a04b", tgt: "#5aa9e6", pred: "#c792ea", io: "#9aa5b1", le: "#62d196", proj: "#e6c35a", ema: "#7fb8e6", sig: "#ef8fb7", cloud: "#a3b6c9" };
export const CLASS_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"];

export const tokName = (k: number) => (k < 0 ? "mask token" : `token ${k} (row ${Math.floor(k / 8)}, col ${k % 8})`);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const l2 = (a: Float32Array) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

function mat(t: Tensor, b: number): { data: Float32Array; rows: number; cols: number } {
  const [, S, W] = t.shape;
  return { data: t.data.subarray(b * S * W, (b + 1) * S * W), rows: S, cols: W };
}
function headsConcat(t: Tensor, b: number) {
  const [, h, S, dh] = t.shape;
  const out = new Float32Array(S * h * dh);
  for (let hh = 0; hh < h; hh++)
    for (let s = 0; s < S; s++)
      for (let d = 0; d < dh; d++) out[s * h * dh + hh * dh + d] = t.data[((b * h + hh) * S + s) * dh + d];
  return { data: out, rows: S, cols: h * dh };
}
function headMap(t: Tensor, b: number, head: number) {
  const [, h, S] = t.shape;
  const o = (b * h + head) * S * S;
  return { data: t.data.subarray(o, o + S * S), rows: S, cols: S };
}

type AddOpts = Partial<BlockSpec> & { label: string; atom: string; tint: string; mode?: Mode };

class Builder {
  blocks: BlockSpec[] = [];
  labels: Label[] = [];
  frames: Frame[] = [];
  images: ImageSpec[] = [];
  clouds: CloudSpec[] = [];
  constructor(
    public group: "ijepa" | "lejepa",
    public rec: Rec,
    public grads: Rec | null, // non-null = gradient view
  ) {}

  /** The tensor shown for a recorded key: the activation, or dL/d(activation) in gradient view. */
  t(key: string): Tensor {
    if (!this.grads) return this.rec[key];
    const g = this.grads[key];
    return g ?? new Tensor(new Float32Array(this.rec[key].size), this.rec[key].shape);
  }
  noGrad(key: string) {
    return !!this.grads && !this.grads[key];
  }

  add(id: string, d: { data: Float32Array; rows: number; cols: number }, x: number, y: number, o: AddOpts, recKey: string | null = id) {
    const gm = !!this.grads && recKey !== null;
    const noGrad = gm && this.noGrad(recKey!);
    const label = gm ? `∂L/∂ ${o.label}` : o.label;
    const mode: Mode = gm && o.mode === 1 ? 0 : o.mode ?? 0;
    const spec: BlockSpec = { id: `${this.group}:${id}`, rows: d.rows, cols: d.cols, data: d.data, x, y, group: this.group, ...o, mode, label, noGrad, order: this.blocks.length, grad: gm && !noGrad };
    if (noGrad) spec.note = "no gradient reaches this tensor (stop-gradient)";
    this.blocks.push(spec);
    const cell = spec.cell ?? 1;
    const norm = gm && !noGrad ? `  ‖·‖=${l2(d.data).toExponential(1)}` : "";
    const txt = d.cols * cell < 48 ? label : `${label}  [${d.rows}×${d.cols}]${norm}`;
    this.labels.push({ text: noGrad ? `${txt}  · no gradient` : txt, x, y: y + 2.2, size: 3.6, color: noGrad ? "#5b6470" : undefined });
    return x + d.cols * cell + GAP;
  }

  /** One transformer layer as two rows: attention row, then MLP row. Returns the y below it. */
  layer(rp: string, b: number, head: number, x0: number, y: number, rowH: number, rowTokens: number[], tint: string, name: string): number {
    const q = this.rec[rp + "q"];
    const dh = q.shape[3];
    const tok = { rowTokens, rowName: (r: number) => tokName(rowTokens[r]) };
    const ch = { colName: (c: number) => `channel ${c}` };
    const hd = { colName: (c: number) => `head ${Math.floor(c / dh)}, dim ${c % dh}` };
    if (name) this.labels.push({ text: name, x: x0 - 58, y: y - 6, size: 10, color: "#8b95a3" });
    let x = x0;
    x = this.add(rp + "ln1", mat(this.t(rp + "ln1"), b), x, y, { label: "LN₁", atom: "T1 LayerNorm", tint, ...tok, ...ch });
    x = this.add(rp + "q", headsConcat(this.t(rp + "q"), b), x, y, { label: "Q (all heads)", atom: "T2 QKV projection", tint, ...tok, ...hd });
    x = this.add(rp + "k", headsConcat(this.t(rp + "k"), b), x, y, { label: "K", atom: "T2 QKV projection", tint, ...tok, ...hd });
    x = this.add(rp + "v", headsConcat(this.t(rp + "v"), b), x, y, { label: "V", atom: "T2 QKV projection", tint, ...tok, ...hd });
    x = this.add(rp + "attn", headMap(this.t(rp + "attn"), b, head), x, y, {
      label: `attn h${head}`, atom: `T3-T4 softmax(QKᵀ/√${dh}), head ${head}`, tint, mode: 1, ...tok,
      colTokens: rowTokens, colName: (c: number) => `key ${tokName(rowTokens[c])}`, note: "row = query token, column = key token; each row sums to 1",
    });
    this.add(rp + "o", mat(this.t(rp + "o"), b), x, y, { label: "A·V (heads concat)", atom: "T5 weighted sum", tint, ...tok, ...hd });
    const y2 = y - rowH - ROWGAP;
    x = x0;
    x = this.add(rp + "x_attn", mat(this.t(rp + "x_attn"), b), x, y2, { label: "x + attn·W_o", atom: "T6 out-proj + residual", tint, ...tok, ...ch });
    x = this.add(rp + "ln2", mat(this.t(rp + "ln2"), b), x, y2, { label: "LN₂", atom: "T7 LayerNorm", tint, ...tok, ...ch });
    x = this.add(rp + "fc1", mat(this.t(rp + "fc1"), b), x, y2, { label: "MLP up (W₁)", atom: "T8 MLP up", tint, ...tok, colName: (c) => `hidden ${c}` });
    x = this.add(rp + "gelu", mat(this.t(rp + "gelu"), b), x, y2, { label: "GELU", atom: "T9 GELU", tint, ...tok, colName: (c) => `hidden ${c}` });
    this.add(rp + "x_out", mat(this.t(rp + "x_out"), b), x, y2, { label: "x + MLP (W₂)", atom: "T10 MLP down + residual", tint, ...tok, ...ch });
    return y2 - rowH - LAYERGAP;
  }

  frame(id: string, title: string, subtitle: string, x: number, y: number, w: number, h: number, color: string) {
    this.frames.push({ id: `${this.group}:${id}`, title, subtitle, x: x - PAD, y: y + PAD + 10, w: w + 2 * PAD, h: h + 2 * PAD + 10, color });
  }

  bottom(): number {
    return Math.min(...this.frames.map((f) => f.y - f.h));
  }
}

export const TOWER_W = 3 * 64 + 2 * 256 + 4 * GAP; // MLP row of a D=64 layer: the widest row
export const LEJEPA_X0 = 2 * TOWER_W + 360;

function maskGrid(run: IJepaRun): Float32Array {
  const g = new Float32Array(64);
  run.masks.ctxIdx.forEach((k) => (g[k] = 1));
  run.masks.tgtIdx.forEach((blk, i) => blk.forEach((k) => (g[k] = 2 + (i % 4))));
  return g;
}

export interface IJepaOpts {
  head: number;
  tgtBlock: number;
  showTarget: boolean;
  grads: Rec | null;
  frozen: boolean;
  ema: { P: Params; tau: number; steps: number } | null;
}

export function layoutIJepa(run: IJepaRun, img01: Float32Array, st: IJepaOpts): Builder {
  const rec = run.rec;
  const B = new Builder("ijepa", rec, st.grads);
  const gm = !!st.grads;
  const Nc = run.masks.ctxIdx.length;
  const blk = Math.min(st.tgtBlock, run.masks.tgtIdx.length - 1);
  const tgtTok = run.masks.tgtIdx[blk];
  const Nt = tgtTok.length;
  const all = range(64);
  const ctxTok = run.masks.ctxIdx;
  const depth = Object.keys(rec).filter((k) => /^ctx\.blk\d+\.ln1$/.test(k)).length;
  const pdepth = Object.keys(rec).filter((k) => /^pred\.blk\d+\.ln1$/.test(k)).length;
  const frozen = st.frozen ? " (frozen)" : "";

  // Header: image, mask, patches, embeddings (atoms 1-2)
  const y = 0;
  B.images.push({ id: "ijepa:image", rgb: img01, res: 32, x: 0, y, size: 32 });
  B.add("mask", { data: maskGrid(run), rows: 8, cols: 8 }, 42, y, {
    label: "mask (8×8 tokens)", atom: "2 multi-block masking", tint: TINT.io, mode: 2, cell: 4,
    rowName: (r) => `grid row ${r}`, colName: (c) => `grid col ${c}`, note: "grey = context, colours = the target blocks, dark = unused",
  }, null);
  let x = 90;
  const tok = { rowTokens: all, rowName: (r: number) => tokName(r) };
  x = B.add("ctx.patches", mat(rec["ctx.patches"], 0), x, y, { label: "patches", atom: "1b patchify", tint: TINT.io, ...tok, colName: (c) => `ch ${Math.floor(c / 16)}, y ${Math.floor((c % 16) / 4)}, x ${c % 4}` }, null);
  x = B.add("ctx.patch_embed", mat(B.t("ctx.patch_embed"), 0), x, y, { label: "patch · W_e", atom: "1c linear embed", tint: TINT.io, ...tok, colName: (c) => `channel ${c}` });
  x = B.add("ctx.pos", { data: rec["ctx.pos"].data, rows: 64, cols: 64 }, x, y, { label: "+ sin-cos position", atom: "1d positional embedding (fixed)", tint: TINT.io, ...tok, colName: (c) => (c < 32 ? `column code ${c}` : `row code ${c - 32}`) }, null);
  x = B.add("ctx.tokens", mat(B.t("ctx.tokens"), 0), x, y, { label: "tokens", atom: "1 embeddings", tint: TINT.io, ...tok, colName: (c) => `channel ${c}` });
  B.frame("embed", "Input → patches → embeddings", gm ? "gradient reaches W_e only through the visible context tokens (masked rows stay 0)" : "atoms 1–2 · 32×32 image, 4×4 patches, 8×8 = 64 tokens", 0, y, x - GAP, 64, TINT.io);

  // Towers (atoms 3-4)
  const yTop = y - 64 - 90;
  const xT = TOWER_W + 110;
  B.add("ctx.kept", mat(B.t("ctx.kept"), 0), 0, yTop, { label: "visible context tokens", atom: "3 gather context", tint: TINT.ctx, rowTokens: ctxTok, rowName: (r) => tokName(ctxTok[r]), colName: (c) => `channel ${c}` });
  if (st.showTarget)
    B.add("tgt.tokens", mat(B.t("tgt.tokens"), 0), xT, yTop, { label: "all 64 tokens", atom: "4 target encoder input", tint: TINT.tgt, ...tok, colName: (c) => `channel ${c}` });
  let yc = yTop - 64 - LAYERGAP;
  for (let i = 0; i < depth; i++) {
    const yNext = B.layer(`ctx.blk${i}.`, 0, st.head, 0, yc, 64, ctxTok, TINT.ctx, `layer ${i + 1}`);
    if (st.showTarget) B.layer(`tgt.blk${i}.`, 0, st.head, xT, yc, 64, all, TINT.tgt, "");
    yc = yNext;
  }
  B.add("ctx.out", mat(B.t("ctx.out"), 0), 0, yc, { label: "s_x = LN(context)", atom: "3 context encoder output", tint: TINT.ctx, rowTokens: ctxTok, rowName: (r) => tokName(ctxTok[r]), colName: (c) => `channel ${c}` });
  const yOut = yc;
  B.frame("ctx", `Context encoder f_θ${frozen}`, gm ? "∂L/∂ activations · the gradient flows back from the predictor through every layer" : `atom 3 · sees only the ${Nc} visible tokens · trained by backprop`, 0, yTop, TOWER_W, yTop - yOut + 64, TINT.ctx);
  if (st.showTarget) {
    let xx = xT;
    xx = B.add("tgt.out", mat(B.t("tgt.out"), 0), xx, yOut, { label: "encoder output", atom: "4 target encoder", tint: TINT.tgt, ...tok, colName: (c) => `channel ${c}` });
    xx = B.add("tgt.ln", mat(B.t("tgt.ln"), 0), xx, yOut, { label: "LayerNorm (no affine)", atom: "4 target normalisation", tint: TINT.tgt, ...tok, colName: (c) => `channel ${c}` });
    B.add("tgt.s_y", mat(B.t("tgt.s_y"), blk), xx, yOut, { label: `targets s_y · block ${blk + 1}`, atom: "4 gather targets (stop-grad)", tint: TINT.tgt, rowTokens: tgtTok, rowName: (r) => tokName(tgtTok[r]), colName: (c) => `channel ${c}` });
    B.frame("tgt", gm ? "Target encoder f_θ̄ — receives NO gradient" : `Target encoder f_θ̄ — EMA copy, no gradient${frozen}`, gm ? "stop-gradient: θ̄ changes only through the EMA update (atom 9)" : "atom 4 · the FULL image · θ̄ ← τθ̄ + (1−τ)θ", xT, yTop, TOWER_W, yTop - yOut + 64, TINT.tgt);
  }

  // Predictor (atom 5) + loss (atom 6)
  const seqTok = [...ctxTok, ...tgtTok];
  const S = Nc + Nt;
  const yP = yOut - 64 - 110;
  let yLoss = yP;
  if (rec["pred.embed"]) {
    x = 0;
    x = B.add("pred.embed", mat(B.t("pred.embed"), 0), x, yP, { label: "s_x · W (to width 32)", atom: "5a predictor embed", tint: TINT.pred, rowTokens: ctxTok, rowName: (r) => tokName(ctxTok[r]), colName: (c) => `channel ${c}` });
    x = B.add("pred.ctx", mat(B.t("pred.ctx"), 0), x, yP, { label: "+ context positions", atom: "5b", tint: TINT.pred, rowTokens: ctxTok, rowName: (r) => tokName(ctxTok[r]), colName: (c) => `channel ${c}` });
    x = B.add("pred.mask_q", mat(B.t("pred.mask_q"), blk), x, yP, {
      label: `mask token + target positions · block ${blk + 1}`, atom: "5c mask queries", tint: TINT.pred, rowTokens: tgtTok, rowName: (r) => `query for ${tokName(tgtTok[r])}`, colName: (c) => `channel ${c}`,
    });
    B.add("pred.in", mat(B.t("pred.in"), blk), x, yP, { label: "[context ; queries]", atom: "5d concat (one sequence per block)", tint: TINT.pred, rowTokens: seqTok, rowName: (r) => (r < Nc ? tokName(seqTok[r]) : `query for ${tokName(seqTok[r])}`), colName: (c) => `channel ${c}` });
    let yy = yP - S - LAYERGAP;
    for (let i = 0; i < pdepth; i++) yy = B.layer(`pred.blk${i}.`, blk, Math.min(st.head, 1), 0, yy, S, seqTok, TINT.pred, `pred ${i + 1}`);
    B.add("pred.norm", mat(B.t("pred.norm"), blk), 0, yy, { label: "LN, keep queries", atom: "5f", tint: TINT.pred, rowTokens: tgtTok, rowName: (r) => tokName(tgtTok[r]), colName: (c) => `channel ${c}` });
    B.frame("pred", `Predictor g_φ — target block ${blk + 1}`, "atom 5 · context + mask tokens carrying the target positions → predicted target embeddings", 0, yP, TOWER_W, yP - yy + Nt, TINT.pred);
    x = xT;
    const out = mat(B.t("pred.out"), blk), tgt = mat(rec["tgt.s_y"], blk), pout = mat(rec["pred.out"], blk);
    const err = new Float32Array(Nt * 64);
    for (let i = 0; i < err.length; i++) err[i] = pout.data[i] - tgt.data[i];
    const t2 = { rowTokens: tgtTok, rowName: (r: number) => tokName(tgtTok[r]), colName: (c: number) => `channel ${c}` };
    x = B.add("pred.out", out, x, yy, { label: "prediction ŝ_y", atom: "5g predictor proj", tint: TINT.pred, ...t2 });
    x = B.add("loss.target", gm ? { data: new Float32Array(Nt * 64), rows: Nt, cols: 64 } : tgt, x, yy, { label: "target s_y (no grad)", atom: "4", tint: TINT.tgt, ...t2 }, gm ? "tgt.s_y" : null);
    x = B.add("loss.err", { data: err, rows: Nt, cols: 64 }, x, yy, { label: "ŝ_y − s_y", atom: "6 loss (smooth-L1 of this)", tint: "#ff6b6b", ...t2 }, null);
    B.frame("loss", `Loss = ${run.loss.toFixed(4)}`, "atom 6 · smooth-L1 between prediction and target, averaged over all blocks", xT, yy, x - xT - GAP, Nt, "#ff6b6b");
    yLoss = yy - Nt;
  }

  // EMA update of the target encoder (atom 9), on real weights from the loaded checkpoint
  if (st.ema && st.ema.P["target_encoder.norm.weight"]) emaGroup(B, st.ema.P, st.ema.tau, st.ema.steps, xT, yLoss - 110);
  return B;
}

const EMA_TENSOR = "blocks.0.attn.proj.weight";

function emaGroup(B: Builder, P: Params, tau: number, steps: number, x0: number, y0: number) {
  const th = P["encoder." + EMA_TENSOR], tb = P["target_encoder." + EMA_TENSOR];
  const [R, C] = th.shape;
  const f = Math.pow(tau, steps);
  const after = new Float32Array(th.size), gap = new Float32Array(th.size), gapAfter = new Float32Array(th.size);
  for (let i = 0; i < th.size; i++) {
    gap[i] = tb.data[i] - th.data[i];
    gapAfter[i] = f * gap[i];
    after[i] = th.data[i] + gapAfter[i];
  }
  const w = { rowName: (r: number) => `output ${r}`, colName: (c: number) => `input ${c}` };
  const sc = Math.max(...Array.from(th.data, Math.abs), ...Array.from(tb.data, Math.abs));
  const gsc = Math.max(...Array.from(gap, Math.abs)) || 1;
  let x = x0;
  x = B.add("ema.theta", { data: th.data, rows: R, cols: C }, x, y0, { label: "θ (context encoder)", atom: "9 · layer-1 attn out-proj W_o", tint: TINT.ctx, scale: sc, ...w }, null);
  x = B.add("ema.target", { data: tb.data, rows: R, cols: C }, x, y0, { label: "θ̄ (target encoder)", atom: "9", tint: TINT.tgt, scale: sc, ...w }, null);
  x = B.add("ema.after", { data: after, rows: R, cols: C }, x, y0, { label: `θ̄ after k=${steps}`, atom: "9 θ̄ ← τθ̄ + (1−τ)θ", tint: TINT.ema, scale: sc, ...w }, null);
  x = B.add("ema.gap", { data: gap, rows: R, cols: C }, x, y0, { label: "θ̄ − θ (now)", atom: "9 EMA gap", tint: TINT.ema, scale: gsc, ...w }, null);
  x = B.add("ema.gapAfter", { data: gapAfter, rows: R, cols: C }, x, y0, { label: "θ̄ − θ after (τ^k·gap)", atom: "9 EMA gap shrinks geometrically", tint: TINT.ema, scale: gsc, ...w }, null);
  // Per-tensor relative gap across the whole encoder
  const names = Object.keys(P).filter((k) => k.startsWith("encoder.") && P["target_" + k]);
  const rel = new Float32Array(names.length);
  names.forEach((k, i) => {
    const a = P[k].data, b = P["target_" + k].data;
    let d = 0, n = 0;
    for (let j = 0; j < a.length; j++) {
      d += (b[j] - a[j]) ** 2;
      n += a[j] ** 2;
    }
    rel[i] = Math.sqrt(d / (n || 1));
  });
  const yb = y0 - R - 40;
  B.add("ema.rel", { data: rel, rows: 1, cols: names.length }, x0, yb, {
    label: "‖θ̄ − θ‖ / ‖θ‖ per encoder tensor", atom: "9 EMA gap per tensor", tint: TINT.ema, mode: 1, cell: 6, rowName: () => "relative gap", colName: (c) => names[c].replace("encoder.", ""),
  }, null);
  const total = Math.sqrt(names.reduce((s, k) => s + Array.from(P[k].data).reduce((a, v, j) => a + (P["target_" + k].data[j] - v) ** 2, 0), 0));
  B.frame("ema", "EMA update  θ̄ ← τ·θ̄ + (1 − τ)·θ", `atom 9 · real weights from the loaded checkpoint · total ‖θ̄ − θ‖ = ${total.toFixed(3)} → ${(total * f).toFixed(3)} after ${steps} step(s)`, x0, y0, Math.max(x - x0 - GAP, names.length * 6), R + 46, TINT.ema);
}

export interface LeJepaOpts {
  head: number;
  viewIdx: number;
  grads: Rec | null;
  frozen: boolean;
  sig: { z: Float32Array; n: number; K: number; A: Float32Array; M: number; out: SigregOut; slice: number } | null;
}

export function layoutLeJepa(run: LeJepaRun, src01: Float32Array, st: LeJepaOpts, X0: number): Builder {
  const rec = run.rec;
  const B = new Builder("lejepa", rec, st.grads);
  const gm = !!st.grads;
  const Vg = rec["g.tokens"].shape[0];
  const V = run.views.length;
  const v = Math.min(st.viewIdx, V - 1);
  const isG = v < Vg;
  const rp = isG ? "g." : "l.";
  const b = isG ? v : v - Vg;
  const S = rec[rp + "tokens"].shape[1];
  const g = Math.sqrt(S);
  const vt = range(S);
  const vName = (r: number) => `view ${v + 1} token ${r} (row ${Math.floor(r / g)}, col ${r % g})`;
  const frozen = st.frozen ? " (frozen)" : "";

  // Views (atom 10)
  const y = 0;
  B.images.push({ id: "lejepa:source", rgb: src01, res: 32, x: X0, y, size: 32 });
  let x = X0 + 50;
  run.views01.forEach((rgb, i) => {
    const res = i < Vg ? 32 : 16;
    B.images.push({ id: `lejepa:view${i}`, rgb, res, x, y, size: res * 1.5, outline: i === v ? "#ffffff" : undefined, view: i });
    B.labels.push({ text: i < Vg ? `global ${i + 1}` : `local ${i - Vg + 1}`, x, y: y + 2.2, size: 3.2 });
    x += res * 1.5 + GAP;
  });
  B.frame("views", "Views: 2 global + 6 local crops of ONE image", "atom 10 · random resized crops + flip / colour jitter / grayscale / solarize · click a view", X0, y, x - X0 - GAP, 48, TINT.le);

  // Shared encoder (atom 11)
  const yTop = y - 64 - 90;
  const tk = { rowTokens: vt, rowName: vName, colName: (c: number) => `channel ${c}` };
  B.add(rp + "tokens", mat(B.t(rp + "tokens"), b), X0, yTop, { label: `view ${v + 1} tokens (${g}×${g})`, atom: "11a patchify + embed + position", tint: TINT.le, ...tk });
  let yy = yTop - 64 - LAYERGAP;
  const depth = Object.keys(rec).filter((k) => /^g\.blk\d+\.ln1$/.test(k)).length;
  for (let i = 0; i < depth; i++) yy = B.layer(`${rp}blk${i}.`, b, st.head, X0, yy, 64, vt, TINT.le, `layer ${i + 1}`);
  B.add(rp + "out", mat(B.t(rp + "out"), b), X0, yy, { label: "encoder output", atom: "11b", tint: TINT.le, ...tk });
  B.frame(
    "enc",
    `Shared encoder f_θ${frozen} — view ${v + 1} shown`,
    gm ? "the invariance gradient flows into this ONE encoder through every view (pick another view: it is non-zero there too)" : "atom 11 · ONE set of weights for all 8 views · no teacher, no EMA, no stop-gradient",
    X0, yTop, TOWER_W, yTop - yy + 64, TINT.le,
  );

  // Projector + z + invariance (atoms 11c-12)
  const yP = yy - 64 - 110;
  const views = { rowTokens: range(V), space: "view" as const, rowName: (r: number) => `view ${r + 1} (${r < Vg ? "global" : "local"})` };
  x = X0;
  x = B.add("pool", { data: B.t("pool").data, rows: V, cols: rec["pool"].shape[1] }, x, yP, { label: "mean-pool (backbone e)", atom: "11c", tint: TINT.le, ...views, colName: (c) => `channel ${c}` });
  const projKeys = Object.keys(rec).filter((k) => k.startsWith("proj.")).sort((a, c) => parseInt(a.split(".")[1]) - parseInt(c.split(".")[1]));
  const names: Record<string, string> = { linear: "Linear", batchnorm1d: "BatchNorm", relu: "ReLU" };
  let zKey = "";
  let yRow = yP;
  let xMax = x;
  for (const k of projKeys) {
    const t = B.t(k);
    const kind = k.split(".")[2];
    zKey = k;
    const last = k === projKeys[projKeys.length - 1];
    if (x + t.shape[1] > X0 + TOWER_W) {
      xMax = Math.max(xMax, x);
      x = X0;
      yRow -= V + 26;
    }
    x = B.add(k, { data: t.data, rows: t.shape[0], cols: t.shape[1] }, x, yRow, { label: last ? "z (projector output)" : names[kind], atom: last ? "11d → z ∈ ℝ^K" : "11d projector", tint: TINT.proj, ...views, colName: (c) => `${last ? "z dim" : "unit"} ${c}` });
  }
  const z = rec[zKey];
  const K = z.shape[1];
  const cen = rec["centers"];
  const err = new Float32Array(V * K);
  for (let i = 0; i < V; i++) for (let k = 0; k < K; k++) err[i * K + k] = (z.data[i * K + k] - cen.data[k]) ** 2;
  xMax = Math.max(xMax, x);
  const y2 = yRow - V - 40;
  let x2 = X0;
  x2 = B.add("centers", { data: B.t("centers").data, rows: 1, cols: K }, x2, y2, { label: "μ = mean of the 2 global z", atom: "12 centroid", tint: TINT.proj, rowName: () => "centroid", colName: (c) => `z dim ${c}` });
  x2 = B.add("inv.err", { data: err, rows: V, cols: K }, x2, y2, { label: "(μ − z_v)²", atom: "12 invariance (mean of this)", tint: "#ff6b6b", mode: 1, ...views, colName: (c) => `z dim ${c}` }, null);
  B.frame(
    "proj",
    `Projector g_φ → z · invariance = ${run.inv.toFixed(4)}`,
    "atoms 11–12 · every view is pulled toward the centroid of the global views",
    X0, yP, Math.max(xMax, x2) - X0 - GAP, yP - y2 + V, TINT.proj,
  );

  // SIGReg on a real batch (atom 13)
  if (st.sig) sigregGroup(B, st.sig, X0, y2 - V - 120);
  return B;
}

function sigregGroup(B: Builder, s: NonNullable<LeJepaOpts["sig"]>, x0: number, y0: number) {
  const { z, n, K, A, M, out, slice } = s;
  const img = { rowName: (r: number) => `held-out image ${r}` };
  const G = 44; // extra room so the column labels do not collide
  let x = x0;
  x = B.add("sig.z", { data: z, rows: n, cols: K }, x, y0, { label: `z [${n}×${K}]`, atom: "13 · embeddings of held-out images (uncropped view)", tint: TINT.sig, ...img, colName: (c) => `z dim ${c}` }, null) + G;
  x = B.add("sig.A", { data: A, rows: K, cols: M }, x, y0, { label: `A [${K}×${M}]`, atom: "13a random unit directions (re-drawn every step)", tint: TINT.sig, cell: Math.max(1, Math.min(3, 48 / M)), rowName: (r) => `z dim ${r}`, colName: (c) => `direction ${c}` }, null) + G;
  x = B.add("sig.y", { data: out.y.data, rows: n, cols: M }, x, y0, { label: "y = z·A", atom: "13b projections: one 1-D slice per column", tint: TINT.sig, ...img, colName: (c) => `slice ${c}` }, null) + G;
  const ySpec = B.blocks[B.blocks.length - 1];
  ySpec.colHL = slice; // the slice inspected in the side panel
  x = B.add("sig.ep", { data: out.ep.data, rows: 1, cols: M }, x, y0, {
    label: "EP per slice", atom: "13e EP_m = N·∫|φ̂(t) − e^{−t²/2}|² e^{−t²/2} dt", tint: TINT.sig, mode: 1, cell: Math.max(2, Math.min(8, 128 / M)), rowName: () => "EP statistic", colName: (c) => `slice ${c}${c === slice ? " (selected)" : ""}`,
  }, null);
  B.frame("sig", `SIGReg on ${n} held-out images = ${out.value.toFixed(3)}`, `atom 13 · mean of ${M} slice statistics · a perfectly Gaussian slice scores ≈ 1.06 (finite-sample floor)`, x0, y0, Math.max(x - x0 - 8, 300), n, TINT.sig);
}

/** A PCA-3D embedding cloud of a logged run at one checkpoint (atom 15/16). */
export function cloudGroup(B: Builder, cloud: Cloud, key: string, ckpt: number, labels: number[] | null, cx: number, cy: number, size: number, title: string, subtitle: string): void {
  const c = cloud.keys[key];
  if (!c) return;
  const C = cloud.steps.length, n = c.n;
  const k = ckpt < 0 || ckpt >= C ? C - 1 : ckpt;
  // One scale for the whole run (so collapse visibly shrinks the cloud), set by the largest RMS
  // radius over checkpoints rather than the single most extreme point.
  let rms = (c as { rms?: number }).rms;
  if (rms === undefined) {
    rms = 0;
    for (let q = 0; q < C; q++) {
      let acc = 0;
      for (let i = 0; i < n * 3; i++) acc += c.data[q * n * 3 + i] ** 2;
      rms = Math.max(rms, Math.sqrt(acc / n));
    }
    (c as { rms?: number }).rms = rms;
  }
  const s = size / 2 / (2.4 * (rms || 1));
  const pos = new Float32Array(n * 3), colors = new Float32Array(n * 3);
  const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) pos[i * 3 + d] = c.data[(k * n + i) * 3 + d] * s;
    const col = hex(labels ? CLASS_COLORS[labels[i] % 10] : TINT.cloud);
    colors.set(col, i * 3);
  }
  B.clouds.push({ id: `${B.group}:cloud`, cx, cy, size, pos, colors, n });
  const evr = c.evr[k] ?? [];
  B.frame("cloud", title, `${subtitle} · step ${cloud.steps[k]} · PCA explains ${(evr.reduce((a, b) => a + (b ?? 0), 0) * 100).toFixed(0)}% · fixed scale over the run`, cx - size / 2, cy + size / 2, size, size, TINT.cloud);
}

export { Builder };

export function combine(parts: Builder[]): Layout {
  const blocks = parts.flatMap((p) => p.blocks);
  const frames = parts.flatMap((p) => p.frames);
  const xs = frames.flatMap((f) => [f.x, f.x + f.w]);
  const ys = frames.flatMap((f) => [f.y, f.y - f.h]);
  return {
    blocks,
    images: parts.flatMap((p) => p.images),
    frames,
    labels: parts.flatMap((p) => p.labels),
    clouds: parts.flatMap((p) => p.clouds),
    bounds: { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) },
  };
}

/** Bounding box of frames whose id matches any of the given ids (for camera fly-to). */
export function focusBox(layout: Layout, ids: string[]): { x0: number; x1: number; y0: number; y1: number } | null {
  const fs = layout.frames.filter((f) => ids.includes(f.id));
  if (!fs.length) return null;
  // Clouds are cubes extending toward the camera; perspective enlarges their near faces.
  const pad = (f: Frame) => (f.id.endsWith(":cloud") ? f.w * 0.22 : 0);
  return {
    x0: Math.min(...fs.map((f) => f.x - pad(f))),
    x1: Math.max(...fs.map((f) => f.x + f.w + pad(f))),
    y0: Math.min(...fs.map((f) => f.y - f.h - pad(f))),
    y1: Math.max(...fs.map((f) => f.y + 30 + pad(f))),
  };
}
