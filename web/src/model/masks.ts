// I-JEPA multi-block masking for ONE image (ATOMS atom 2). Mirrors model/jepa_viz/masks.py,
// except that with a single image there is no batch truncation. Block positions are explicit so
// the UI can drag target blocks and re-carve the context.
import { randint, Rng } from "./random";

export interface MaskOpts {
  npred: number;
  predScale: [number, number];
  predAspect: [number, number];
  encScale: [number, number];
  minKeep: number;
  edgeFix: boolean;
  G: number;
}

export const DEFAULT_MASK: MaskOpts = {
  npred: 4, predScale: [0.15, 0.2], predAspect: [0.75, 1.5], encScale: [0.85, 1.0], minKeep: 4, edgeFix: true, G: 8,
};

export interface Rect {
  top: number;
  left: number;
  h: number;
  w: number;
}

export interface MaskSet {
  targets: Rect[];
  context: Rect; // the square before carving
  ctxIdx: number[]; // sorted token indices kept as context
  tgtIdx: number[][]; // per block, row-major token indices
}

export function blockSize(o: MaskOpts, u: number, scale: [number, number], aspect: [number, number]): [number, number] {
  const s = scale[0] + u * (scale[1] - scale[0]);
  const k = Math.floor(o.G * o.G * s);
  const ar = aspect[0] + u * (aspect[1] - aspect[0]);
  const lim = o.edgeFix ? o.G : o.G - 1;
  // Python round() is banker's rounding; sizes here never land exactly on .5 for the defaults.
  const h = Math.max(1, Math.min(Math.round(Math.sqrt(k * ar)), lim));
  const w = Math.max(1, Math.min(Math.round(Math.sqrt(k / ar)), lim));
  return [h, w];
}

const place = (o: MaskOpts, rng: Rng, h: number, w: number): Rect => ({
  top: randint(rng, 0, Math.max(o.edgeFix ? o.G - h + 1 : o.G - h, 1)),
  left: randint(rng, 0, Math.max(o.edgeFix ? o.G - w + 1 : o.G - w, 1)),
  h, w,
});

export const rectTokens = (r: Rect, G: number): number[] => {
  const out: number[] = [];
  for (let i = r.top; i < r.top + r.h; i++) for (let j = r.left; j < r.left + r.w; j++) out.push(i * G + j);
  return out;
};

/** Context = square minus every target block (dropping the last constraints if needed, like the official sampler). */
export function carve(context: Rect, targets: Rect[], G: number, dropLast = 0): number[] {
  const blocked = new Set<number>();
  targets.slice(0, targets.length - dropLast).forEach((t) => rectTokens(t, G).forEach((k) => blocked.add(k)));
  return rectTokens(context, G).filter((k) => !blocked.has(k)).sort((a, b) => a - b);
}

export function sampleMasks(o: MaskOpts, rng: Rng): MaskSet {
  const [h, w] = blockSize(o, rng(), o.predScale, o.predAspect);
  const [hc, wc] = blockSize(o, rng(), o.encScale, [1, 1]);
  const targets = Array.from({ length: o.npred }, () => place(o, rng, h, w));
  let tries = 0, timeout = 20;
  for (;;) {
    const context = place(o, rng, hc, wc);
    const ctxIdx = carve(context, targets, o.G, Math.min(tries, targets.length));
    if (ctxIdx.length > o.minKeep) return { targets, context, ctxIdx, tgtIdx: targets.map((t) => rectTokens(t, o.G)) };
    if (--timeout === 0) {
      tries++;
      timeout = 20;
    }
  }
}

/** Rebuild index sets after the user moved a block. */
export function withTargets(m: MaskSet, targets: Rect[], G: number): MaskSet {
  return { targets, context: m.context, ctxIdx: carve(m.context, targets, G), tgtIdx: targets.map((t) => rectTokens(t, G)) };
}
