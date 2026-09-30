// Run the real forward passes for the current UI state (single image, main thread: ~10-100 ms).
import { ijepaForward, IJepaCfg } from "@/model/ijepa";
import { LeJepaCfg, lejepaForward, normalize, renderView, ViewParams } from "@/model/lejepa";
import { DEFAULT_MASK, MaskSet, sampleMasks, withTargets } from "@/model/masks";
import { mulberry32 } from "@/model/random";
import { IntTensor, Tensor } from "@/model/tensor";
import { Params } from "@/model/vit";
import { sampleView } from "@/model/views";
import { State } from "./store";

export type Rec = Record<string, Tensor>;

export interface IJepaRun {
  rec: Rec;
  masks: MaskSet;
  loss: number;
  ms: number;
}

export interface LeJepaRun {
  rec: Rec;
  views: ViewParams[];
  views01: Float32Array[]; // [3, r, r] each, in [0, 1]
  inv: number;
  ms: number;
}

export function makeMasks(s: State): MaskSet {
  const m = sampleMasks({ ...DEFAULT_MASK, npred: s.npred, predScale: [0.15, s.predScaleMax] }, mulberry32(s.maskSeed));
  return s.targetsOverride && s.targetsOverride.length === m.targets.length ? withTargets(m, s.targetsOverride, DEFAULT_MASK.G) : m;
}

const toNetInput = (img: Uint8Array) => {
  const x01 = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) x01[i] = img[i] / 255;
  return normalize(x01, 1, 32);
};

export function runIJepa(P: Params, cfg: IJepaCfg, img: Uint8Array, masks: MaskSet): IJepaRun {
  const t0 = performance.now();
  const Nt = masks.tgtIdx[0].length;
  const ctx: IntTensor = { data: Int32Array.from(masks.ctxIdx), shape: [1, masks.ctxIdx.length] };
  const tgt: IntTensor = { data: Int32Array.from(masks.tgtIdx.flat()), shape: [masks.tgtIdx.length, 1, Nt] };
  const rec: Rec = {};
  const out = ijepaForward(P, cfg, toNetInput(img), ctx, tgt, rec);
  return { rec, masks, loss: out.loss, ms: performance.now() - t0 };
}

export function runLeJepa(P: Params, cfg: LeJepaCfg, img: Uint8Array, seed: number, color: boolean): LeJepaRun {
  const t0 = performance.now();
  const v = cfg.views;
  const rng = mulberry32(seed);
  const views = [
    ...Array.from({ length: v.n_global }, () => sampleView(rng, v.global_scale as [number, number], color)),
    ...Array.from({ length: v.n_local }, () => sampleView(rng, v.local_scale as [number, number], color)),
  ];
  const views01 = views.map((p, i) => renderView(img, p, i < v.n_global ? v.global_res : v.local_res));
  const cat = (a: Float32Array[]) => {
    const out = new Float32Array(a.reduce((s, x) => s + x.length, 0));
    let o = 0;
    for (const x of a) (out.set(x, o), (o += x.length));
    return out;
  };
  const g = normalize(cat(views01.slice(0, v.n_global)), v.n_global, v.global_res);
  const l = v.n_local ? normalize(cat(views01.slice(v.n_global)), v.n_local, v.local_res) : null;
  const rec: Rec = {};
  const out = lejepaForward(P, cfg, g, l, 1, null, rec);
  return { rec, views, views01, inv: out.inv, ms: performance.now() - t0 };
}
