// LeJEPA forward (ATOMS atoms 10-14): render views -> ONE encoder -> projector -> invariance + SIGReg.
import { concat0, linear, meanTokens, relu, Tensor } from "./tensor";
import { ModelCfg, Params, put, Rec, vit } from "./vit";

export interface LeJepaCfg {
  model: ModelCfg;
  views: { n_global: number; n_local: number; global_res: number; local_res: number; global_scale: number[]; local_scale: number[]; color: boolean };
  proj: { hidden: number; depth: number; out: number };
  reg: { kind: string; lam: number; slices: number; t_max: number; t_points: number };
}

/** One view's replayable parameters (model/jepa_viz/augment.py). */
export interface ViewParams {
  box: [number, number, number, number]; // x0, y0, w, h in source pixels
  flip: boolean;
  jitter: boolean;
  bright: number;
  contrast: number;
  sat: number;
  hue: number;
  gray: boolean;
  sol: boolean;
}

export const MEAN = [0.4914, 0.4822, 0.4465];
export const STD = [0.247, 0.2435, 0.2616];
const GW = [0.299, 0.587, 0.114];
const YIQ = [
  [0.299, 0.587, 0.114],
  [0.596, -0.274, -0.322],
  [0.211, -0.523, 0.312],
];

function inv3(m: number[][]): number[][] {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
const mul3 = (x: number[][], y: number[][]) => x.map((r) => [0, 1, 2].map((j) => r[0] * y[0][j] + r[1] * y[1][j] + r[2] * y[2][j]));
const YIQ_INV = inv3(YIQ);
const clip01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** img: uint8 [3, S, S] (CHW) -> view in [0, 1], Float32 [3, res, res]. Mirrors augment.render_views. */
export function renderView(img: ArrayLike<number>, p: ViewParams, res: number, S = 32): Float32Array {
  const [x0, y0, w, h] = p.box;
  const out = new Float32Array(3 * res * res);
  const N = res * res;
  for (let i = 0; i < res; i++) {
    const ky = (i + 0.5) / res;
    const fy = Math.min(Math.max(y0 + ky * h - 0.5, 0), S - 1);
    const r0 = Math.floor(fy), r1 = Math.min(r0 + 1, S - 1), wy = fy - r0;
    for (let j = 0; j < res; j++) {
      const kx = (j + 0.5) / res;
      const xs = p.flip ? x0 + w - kx * w : x0 + kx * w;
      const fx = Math.min(Math.max(xs - 0.5, 0), S - 1);
      const c0 = Math.floor(fx), c1 = Math.min(c0 + 1, S - 1), wx = fx - c0;
      for (let c = 0; c < 3; c++) {
        const base = c * S * S;
        const v =
          (img[base + r0 * S + c0] * (1 - wx) * (1 - wy) + img[base + r0 * S + c1] * wx * (1 - wy) +
            img[base + r1 * S + c0] * (1 - wx) * wy + img[base + r1 * S + c1] * wx * wy) / 255;
        out[c * N + i * res + j] = v;
      }
    }
  }
  if (p.jitter) {
    for (let k = 0; k < 3 * N; k++) out[k] = clip01(out[k] * p.bright);
    let m = 0;
    for (let k = 0; k < N; k++) m += GW[0] * out[k] + GW[1] * out[N + k] + GW[2] * out[2 * N + k];
    m /= N;
    for (let k = 0; k < 3 * N; k++) out[k] = clip01(p.contrast * out[k] + (1 - p.contrast) * m);
    for (let k = 0; k < N; k++) {
      const g = GW[0] * out[k] + GW[1] * out[N + k] + GW[2] * out[2 * N + k];
      for (let c = 0; c < 3; c++) out[c * N + k] = clip01(p.sat * out[c * N + k] + (1 - p.sat) * g);
    }
    const th = 2 * Math.PI * p.hue;
    const R = [
      [1, 0, 0],
      [0, Math.cos(th), -Math.sin(th)],
      [0, Math.sin(th), Math.cos(th)],
    ];
    const M = mul3(mul3(YIQ_INV, R), YIQ);
    for (let k = 0; k < N; k++) {
      const r = out[k], g = out[N + k], b = out[2 * N + k];
      for (let c = 0; c < 3; c++) out[c * N + k] = clip01(M[c][0] * r + M[c][1] * g + M[c][2] * b);
    }
  }
  if (p.gray)
    for (let k = 0; k < N; k++) {
      const g = GW[0] * out[k] + GW[1] * out[N + k] + GW[2] * out[2 * N + k];
      out[k] = out[N + k] = out[2 * N + k] = g;
    }
  if (p.sol) for (let k = 0; k < 3 * N; k++) if (out[k] >= 0.5) out[k] = 1 - out[k];
  return out;
}

/** views in [0,1] [n, 3, r, r] -> network input (channel normalised). */
export function normalize(x01: Float32Array, n: number, res: number): Tensor {
  const y = new Float32Array(x01.length);
  const N = res * res;
  for (let v = 0; v < n; v++)
    for (let c = 0; c < 3; c++)
      for (let k = 0; k < N; k++) {
        const i = (v * 3 + c) * N + k;
        y[i] = (x01[i] - MEAN[c]) / STD[c];
      }
  return new Tensor(y, [n, 3, res, res]);
}

export function projector(e: Tensor, P: Params, depth: number, rec: Rec): Tensor {
  let x = e;
  const lin = (i: number, t: Tensor) => linear(t, P[`projector.net.${i}.weight`], P[`projector.net.${i}.bias`]);
  for (let j = 0; j < depth - 1; j++) {
    const i = 3 * j;
    x = lin(i, x);
    put(rec, `proj.${i}.linear`, x);
    const bn = `projector.net.${i + 1}.`;
    const [rm, rv, w, b] = [P[bn + "running_mean"], P[bn + "running_var"], P[bn + "weight"], P[bn + "bias"]];
    const H = x.shape[1];
    const y = new Float32Array(x.size);
    for (let r = 0; r < x.shape[0]; r++)
      for (let k = 0; k < H; k++) y[r * H + k] = ((x.data[r * H + k] - rm.data[k]) / Math.sqrt(rv.data[k] + 1e-5)) * w.data[k] + b.data[k];
    x = new Tensor(y, x.shape);
    put(rec, `proj.${i + 1}.batchnorm1d`, x);
    x = relu(x);
    put(rec, `proj.${i + 2}.relu`, x);
  }
  const i = 3 * (depth - 1);
  x = lin(i, x);
  put(rec, `proj.${i}.linear`, x);
  return x;
}

export function tGrid(tMax: number, n: number): Float32Array {
  const t = new Float32Array(n);
  for (let j = 0; j < n; j++) t[j] = -tMax + (2 * tMax * j) / (n - 1);
  return t;
}

export interface SigregOut {
  y: Tensor; // [V, B, M] projections
  ecfRe: Tensor; // [V, M, T]
  ecfIm: Tensor;
  ep: Tensor; // [V, M] Epps-Pulley statistic per view and slice
  value: number; // mean EP
}

/** z [V, B, K], A [K, M] unit directions. Mirrors sigreg.epps_pulley (trapezoid, times B). */
export function sigreg(z: Tensor, A: Tensor, t: Float32Array): SigregOut {
  const [V, B, K] = z.shape;
  const M = A.shape[1], T = t.length;
  const y = new Float32Array(V * B * M);
  for (let r = 0; r < V * B; r++)
    for (let m = 0; m < M; m++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += z.data[r * K + k] * A.data[k * M + m];
      y[r * M + m] = s;
    }
  const re = new Float32Array(V * M * T), im = new Float32Array(V * M * T), ep = new Float32Array(V * M);
  let total = 0;
  for (let v = 0; v < V; v++)
    for (let m = 0; m < M; m++) {
      const o = (v * M + m) * T;
      for (let j = 0; j < T; j++) {
        let c = 0, s = 0;
        for (let b = 0; b < B; b++) {
          const a = t[j] * y[(v * B + b) * M + m];
          c += Math.cos(a);
          s += Math.sin(a);
        }
        re[o + j] = c / B;
        im[o + j] = s / B;
      }
      let integ = 0;
      const err = (j: number) => {
        const phi = Math.exp(-0.5 * t[j] * t[j]);
        return ((re[o + j] - phi) ** 2 + im[o + j] ** 2) * phi;
      };
      for (let j = 0; j + 1 < T; j++) integ += ((t[j + 1] - t[j]) * (err(j) + err(j + 1))) / 2;
      ep[v * M + m] = integ * B;
      total += integ * B;
    }
  return {
    y: new Tensor(y, [V, B, M]),
    ecfRe: new Tensor(re, [V, M, T]),
    ecfIm: new Tensor(im, [V, M, T]),
    ep: new Tensor(ep, [V, M]),
    value: total / (V * M),
  };
}

export interface LeJepaOut {
  loss: number; // NaN when SIGReg is skipped
  inv: number;
  reg: number;
  e: Tensor; // pooled backbone features [V*B, D]
  z: Tensor; // [V, B, K]
  sig: SigregOut | null;
}

/** g/l: normalised global [Vg*B,3,32,32] and local [Vl*B,3,16,16] views (view-major).
 *  A = null skips SIGReg (a batch statistic: meaningless for a single image). */
export function lejepaForward(P: Params, cfg: LeJepaCfg, g: Tensor, l: Tensor | null, B: number, A: Tensor | null, rec: Rec): LeJepaOut {
  const parts = [meanTokens(vit(g, P, "encoder.", cfg.model, rec, "g."))];
  if (l && l.shape[0] > 0) parts.push(meanTokens(vit(l, P, "encoder.", cfg.model, rec, "l.")));
  const e = concat0(parts);
  put(rec, "pool", e);
  const zf = projector(e, P, cfg.proj.depth, rec);
  const V = e.shape[0] / B, K = zf.shape[1];
  const z = zf.reshape([V, B, K]);
  const Vg = cfg.views.n_global;
  const centers = new Float32Array(B * K);
  for (let v = 0; v < Vg; v++) for (let i = 0; i < B * K; i++) centers[i] += z.data[v * B * K + i] / Vg;
  put(rec, "centers", new Tensor(centers, [B, K]));
  let inv = 0;
  for (let v = 0; v < V; v++) for (let i = 0; i < B * K; i++) inv += (centers[i] - z.data[v * B * K + i]) ** 2;
  inv /= V * B * K;
  if (!A) return { loss: NaN, inv, reg: NaN, e, z, sig: null };
  const sig = sigreg(z, A, tGrid(cfg.reg.t_max, cfg.reg.t_points));
  put(rec, "sigreg.y", sig.y);
  put(rec, "sigreg.ecf_re", sig.ecfRe);
  put(rec, "sigreg.ecf_im", sig.ecfIm);
  put(rec, "sigreg.ep", sig.ep);
  const lam = cfg.reg.lam;
  const loss = (1 - lam) * inv + lam * sig.value;
  put(rec, "loss", new Tensor(new Float32Array([loss]), []));
  return { loss, inv, reg: sig.value, e, z, sig };
}
