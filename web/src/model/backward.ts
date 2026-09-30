// Reverse-mode gradients for both JEPA losses (ATOMS atoms 7 and 14), hand-derived per op and
// checked against the PyTorch goldens (per-row gradient norms of every recorded activation and
// per-parameter gradient norms). Uses the forward's recorded activations as the tape.
import { erf, Tensor } from "./tensor";
import type { IJepaCfg } from "./ijepa";
import type { LeJepaCfg } from "./lejepa";
import { Params } from "./vit";

export type Grads = Record<string, Tensor>; // dL/d(activation), keyed like the forward `rec`
export type ParamGrads = Record<string, Float32Array>; // dL/d(param), keyed like the weights

function acc(pg: ParamGrads, name: string, n: number): Float32Array {
  if (!pg[name]) pg[name] = new Float32Array(n);
  return pg[name];
}

/** y = x W^T + b. Returns dx; accumulates dW, db into pg. */
function linearBack(x: Tensor, W: Tensor, dy: Tensor, pg: ParamGrads | null, name?: string, hasBias = true): Tensor {
  const [out, inn] = W.shape;
  const rows = x.size / inn;
  const dx = new Float32Array(x.size);
  for (let r = 0; r < rows; r++)
    for (let o = 0; o < out; o++) {
      const g = dy.data[r * out + o];
      if (g === 0) continue;
      const wo = o * inn, xo = r * inn;
      for (let k = 0; k < inn; k++) dx[xo + k] += g * W.data[wo + k];
    }
  if (pg && name) {
    const dW = acc(pg, name + ".weight", out * inn);
    for (let r = 0; r < rows; r++)
      for (let o = 0; o < out; o++) {
        const g = dy.data[r * out + o];
        if (g === 0) continue;
        const wo = o * inn, xo = r * inn;
        for (let k = 0; k < inn; k++) dW[wo + k] += g * x.data[xo + k];
      }
    if (hasBias) {
      const db = acc(pg, name + ".bias", out);
      for (let r = 0; r < rows; r++) for (let o = 0; o < out; o++) db[o] += dy.data[r * out + o];
    }
  }
  return new Tensor(dx, x.shape);
}

/** LayerNorm over the last axis (optionally affine). Returns dx; accumulates dγ, dβ. */
function layerNormBack(x: Tensor, dy: Tensor, eps: number, w?: Tensor, pg?: ParamGrads | null, name?: string): Tensor {
  const D = x.shape[x.shape.length - 1];
  const rows = x.size / D;
  const dx = new Float32Array(x.size);
  const dw = w && pg && name ? acc(pg, name + ".weight", D) : null;
  const db = w && pg && name ? acc(pg, name + ".bias", D) : null;
  const xhat = new Float64Array(D), dxh = new Float64Array(D);
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let mu = 0;
    for (let k = 0; k < D; k++) mu += x.data[o + k];
    mu /= D;
    let v = 0;
    for (let k = 0; k < D; k++) v += (x.data[o + k] - mu) ** 2;
    const rstd = 1 / Math.sqrt(v / D + eps);
    let s1 = 0, s2 = 0;
    for (let k = 0; k < D; k++) {
      xhat[k] = (x.data[o + k] - mu) * rstd;
      const g = dy.data[o + k];
      dxh[k] = w ? g * w.data[k] : g;
      if (dw) dw[k] += g * xhat[k];
      if (db) db[k] += g;
      s1 += dxh[k];
      s2 += dxh[k] * xhat[k];
    }
    for (let k = 0; k < D; k++) dx[o + k] = (rstd / D) * (D * dxh[k] - s1 - xhat[k] * s2);
  }
  return new Tensor(dx, x.shape);
}

function geluBack(u: Tensor, dg: Tensor): Tensor {
  const du = new Float32Array(u.size);
  const c = 1 / Math.sqrt(2 * Math.PI);
  for (let i = 0; i < u.size; i++) {
    const x = u.data[i];
    du[i] = dg.data[i] * (0.5 * (1 + erf(x / Math.SQRT2)) + x * c * Math.exp(-0.5 * x * x));
  }
  return new Tensor(du, u.shape);
}

const addInto = (a: Float32Array, b: Float32Array) => {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
};

/** Backward through one pre-LN block. xIn is the block input; rec holds its activations. */
function blockBack(rec: Grads, rp: string, P: Params, pre: string, xIn: Tensor, dOut: Tensor, pg: ParamGrads, g: Grads): Tensor {
  const [B, S, W] = xIn.shape;
  const heads = rec[rp + "q"].shape[1], dh = W / heads;
  g[rp + "x_out"] = dOut;
  // MLP branch: x_out = x_attn + fc2(gelu(fc1(ln2(x_attn))))
  const dG = linearBack(rec[rp + "gelu"], P[pre + "fc2.weight"], dOut, pg, pre + "fc2");
  g[rp + "gelu"] = dG;
  const dU = geluBack(rec[rp + "fc1"], dG);
  g[rp + "fc1"] = dU;
  const dM = linearBack(rec[rp + "ln2"], P[pre + "fc1.weight"], dU, pg, pre + "fc1");
  g[rp + "ln2"] = dM;
  const dXa = new Float32Array(dOut.data);
  addInto(dXa, layerNormBack(rec[rp + "x_attn"], dM, 1e-6, P[pre + "norm2.weight"], pg, pre + "norm2").data);
  const dXattn = new Tensor(dXa, [B, S, W]);
  g[rp + "x_attn"] = dXattn;
  // Attention branch: x_attn = x_in + proj(o)
  const dO = linearBack(rec[rp + "o"], P[pre + "attn.proj.weight"], dXattn, pg, pre + "attn.proj");
  g[rp + "o"] = dO;
  const q = rec[rp + "q"].data, k = rec[rp + "k"].data, v = rec[rp + "v"].data, A = rec[rp + "attn"].data;
  const n = B * heads * S * dh;
  const dq = new Float32Array(n), dk = new Float32Array(n), dv = new Float32Array(n);
  const dA = new Float32Array(B * heads * S * S), dSc = new Float32Array(B * heads * S * S);
  const scale = 1 / Math.sqrt(dh);
  for (let b = 0; b < B; b++)
    for (let h = 0; h < heads; h++) {
      const bh = (b * heads + h) * S;
      for (let i = 0; i < S; i++) {
        const row = (bh + i) * S;
        let dot = 0;
        for (let j = 0; j < S; j++) {
          let s = 0;
          for (let d = 0; d < dh; d++) s += dO.data[(b * S + i) * W + h * dh + d] * v[(bh + j) * dh + d];
          dA[row + j] = s;
          dot += s * A[row + j];
          for (let d = 0; d < dh; d++) dv[(bh + j) * dh + d] += A[row + j] * dO.data[(b * S + i) * W + h * dh + d];
        }
        for (let j = 0; j < S; j++) dSc[row + j] = A[row + j] * (dA[row + j] - dot);
        for (let j = 0; j < S; j++) {
          const gs = dSc[row + j] * scale;
          for (let d = 0; d < dh; d++) {
            dq[(bh + i) * dh + d] += gs * k[(bh + j) * dh + d];
            dk[(bh + j) * dh + d] += gs * q[(bh + i) * dh + d];
          }
        }
      }
    }
  g[rp + "attn"] = new Tensor(dA, [B, heads, S, S]);
  g[rp + "scores"] = new Tensor(dSc, [B, heads, S, S]);
  g[rp + "q"] = new Tensor(dq, [B, heads, S, dh]);
  g[rp + "k"] = new Tensor(dk, [B, heads, S, dh]);
  g[rp + "v"] = new Tensor(dv, [B, heads, S, dh]);
  const dqkv = new Float32Array(B * S * 3 * W);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++)
      for (let h = 0; h < heads; h++)
        for (let d = 0; d < dh; d++) {
          const src = ((b * heads + h) * S + s) * dh + d, dst = (b * S + s) * 3 * W + h * dh + d;
          dqkv[dst] = dq[src];
          dqkv[dst + W] = dk[src];
          dqkv[dst + 2 * W] = dv[src];
        }
  const dLn1 = linearBack(rec[rp + "ln1"], P[pre + "attn.qkv.weight"], new Tensor(dqkv, [B, S, 3 * W]), pg, pre + "attn.qkv");
  g[rp + "ln1"] = dLn1;
  const dx = new Float32Array(dXattn.data);
  addInto(dx, layerNormBack(xIn, dLn1, 1e-6, P[pre + "norm1.weight"], pg, pre + "norm1").data);
  return new Tensor(dx, [B, S, W]);
}

/** Backward through a ViT given dL/d(out). Records grads under rp; returns dL/d(tokens) [B, N, D]. */
function vitBack(rec: Grads, P: Params, pre: string, rp: string, depth: number, dOut: Tensor, pg: ParamGrads, g: Grads, keepIdx?: Int32Array): Tensor {
  g[rp + "out"] = dOut;
  const last = rec[`${rp}blk${depth - 1}.x_out`];
  let d = layerNormBack(last, dOut, 1e-6, P[pre + "norm.weight"], pg, pre + "norm");
  for (let i = depth - 1; i >= 0; i--) {
    const xIn = i > 0 ? rec[`${rp}blk${i - 1}.x_out`] : keepIdx ? rec[rp + "kept"] : rec[rp + "tokens"];
    d = blockBack(rec, `${rp}blk${i}.`, P, `${pre}blocks.${i}.`, xIn, d, pg, g);
  }
  let dTok = d;
  if (keepIdx) {
    g[rp + "kept"] = d;
    const [B, K, D] = d.shape;
    const N = rec[rp + "tokens"].shape[1];
    const full = new Float32Array(B * N * D);
    for (let b = 0; b < B; b++)
      for (let kk = 0; kk < K; kk++) {
        const t = keepIdx[b * K + kk];
        for (let c = 0; c < D; c++) full[(b * N + t) * D + c] += d.data[(b * K + kk) * D + c];
      }
    dTok = new Tensor(full, [B, N, D]);
  }
  g[rp + "tokens"] = dTok;
  g[rp + "patch_embed"] = dTok; // the positional embedding is fixed: same gradient
  linearBack(rec[rp + "patches"], P[pre + "patch_embed.weight"], dTok, pg, pre + "patch_embed");
  return dTok;
}

export interface BackOut {
  grads: Grads;
  params: ParamGrads;
}

/** I-JEPA: gradients of the smooth-L1 (or L2) prediction loss. The target branch is stop-grad. */
export function ijepaBackward(P: Params, cfg: IJepaCfg, rec: Grads, ctxIdx: Int32Array, tgtIdx: Int32Array, Mt: number): BackOut {
  const g: Grads = {}, pg: ParamGrads = {};
  const shat = rec["pred.out"], sy = rec["tgt.s_y"];
  const n = shat.size;
  const dShat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = shat.data[i] - sy.data[i];
    dShat[i] = (cfg.loss === "l2" ? 2 * d : Math.max(-1, Math.min(1, d))) / n;
  }
  g["pred.out"] = new Tensor(dShat, shat.shape);
  const [MB, Nt, D] = shat.shape;
  const B = MB / Mt;
  const sx = rec["ctx.out"];
  const Nc = sx.shape[1];
  let dSx: Tensor;
  const pc = cfg.pred;
  if (!pc.enabled) {
    const d = new Float32Array(B * Nc * D);
    for (let i = 0; i < Mt; i++)
      for (let b = 0; b < B; b++)
        for (let t = 0; t < Nt; t++)
          for (let s = 0; s < Nc; s++)
            for (let c = 0; c < D; c++) d[(b * Nc + s) * D + c] += dShat[((i * B + b) * Nt + t) * D + c] / Nc;
    dSx = new Tensor(d, [B, Nc, D]);
  } else {
    const Dp = pc.dim;
    const S = Nc + Nt;
    const dNorm = linearBack(rec["pred.norm"], P["predictor.proj.weight"], g["pred.out"], pg, "predictor.proj");
    g["pred.norm"] = dNorm;
    // pred.norm = LN(z)[:, Nc:] -> rows < Nc receive no gradient from the loss
    const zLast = rec[`pred.blk${pc.depth - 1}.x_out`];
    const dzFull = new Float32Array(MB * S * Dp);
    for (let r = 0; r < MB; r++) dzFull.set(dNorm.data.subarray(r * Nt * Dp, (r + 1) * Nt * Dp), (r * S + Nc) * Dp);
    let d = layerNormBack(zLast, new Tensor(dzFull, [MB, S, Dp]), 1e-6, P["predictor.norm.weight"], pg, "predictor.norm");
    for (let l = pc.depth - 1; l >= 0; l--) {
      const xIn = l > 0 ? rec[`pred.blk${l - 1}.x_out`] : rec["pred.in"];
      d = blockBack(rec, `pred.blk${l}.`, P, `predictor.blocks.${l}.`, xIn, d, pg, g);
    }
    g["pred.in"] = d;
    const dCtx = new Float32Array(B * Nc * Dp), dMq = new Float32Array(MB * Nt * Dp);
    const dTok = acc(pg, "predictor.mask_token", Dp);
    for (let r = 0; r < MB; r++) {
      const b = r % B;
      for (let s = 0; s < Nc; s++) for (let c = 0; c < Dp; c++) dCtx[(b * Nc + s) * Dp + c] += d.data[(r * S + s) * Dp + c];
      for (let t = 0; t < Nt; t++)
        for (let c = 0; c < Dp; c++) {
          const v = d.data[(r * S + Nc + t) * Dp + c];
          dMq[(r * Nt + t) * Dp + c] = v;
          dTok[c] += v;
        }
    }
    g["pred.mask_q"] = new Tensor(dMq, [MB, Nt, Dp]);
    g["pred.ctx"] = new Tensor(dCtx, [B, Nc, Dp]);
    g["pred.embed"] = g["pred.ctx"];
    dSx = linearBack(sx, P["predictor.embed.weight"], g["pred.ctx"], pg, "predictor.embed");
  }
  vitBack(rec, P, "encoder.", "ctx.", cfg.model.depth, dSx, pg, g, ctxIdx);
  void tgtIdx;
  return { grads: g, params: pg };
}

/** LeJEPA: gradients of (1-λ)·invariance + λ·SIGReg through the projector and the ONE encoder, for every view.
 *  BatchNorm is in eval mode (running statistics), exactly as recorded in the goldens. */
export function lejepaBackward(P: Params, cfg: LeJepaCfg, rec: Grads, B: number, A: Tensor | null): BackOut {
  const g: Grads = {}, pg: ParamGrads = {};
  const lam = cfg.reg.lam;
  const projKeys = Object.keys(rec).filter((k) => k.startsWith("proj.")).sort((a, b) => parseInt(a.split(".")[1]) - parseInt(b.split(".")[1]));
  const zKey = projKeys[projKeys.length - 1];
  const zf = rec[zKey];
  const [VB, K] = zf.shape;
  const V = VB / B, Vg = cfg.views.n_global;
  const cen = rec["centers"].data;
  const dz = new Float32Array(VB * K);
  const dCen = new Float32Array(B * K);
  const ci = (2 * (1 - lam)) / (V * B * K);
  for (let v = 0; v < V; v++)
    for (let i = 0; i < B * K; i++) {
      const diff = zf.data[v * B * K + i] - cen[i];
      dz[v * B * K + i] += ci * diff;
      dCen[i] -= ci * diff;
    }
  g["centers"] = new Tensor(dCen, [B, K]);
  for (let v = 0; v < Vg; v++) for (let i = 0; i < B * K; i++) dz[v * B * K + i] += dCen[i] / Vg;
  if (A && lam > 0) {
    // SIGReg = mean_{v,m} EP_{v,m};  EP = B * trapz_t [((re - φ)^2 + im^2) φ]
    const y = rec["sigreg.y"], re = rec["sigreg.ecf_re"].data, im = rec["sigreg.ecf_im"].data;
    const M = A.shape[1];
    const T = rec["sigreg.ecf_re"].shape[2];
    const tMax = cfg.reg.t_max;
    const t = Array.from({ length: T }, (_, j) => -tMax + (2 * tMax * j) / (T - 1));
    const wq = t.map((_, j) => ((j > 0 ? t[j] - t[j - 1] : 0) + (j < T - 1 ? t[j + 1] - t[j] : 0)) / 2); // trapezoid weights
    const phi = t.map((tj) => Math.exp(-0.5 * tj * tj));
    const scale = lam / (V * M);
    g["sigreg.ep"] = new Tensor(new Float32Array(V * M).fill(scale), [V, M]); // loss = ... + λ·mean(EP)
    const dRe = new Float32Array(re.length), dIm = new Float32Array(im.length);
    for (let v = 0; v < V; v++)
      for (let m = 0; m < M; m++)
        for (let j = 0; j < T; j++) {
          const o = (v * M + m) * T + j;
          dRe[o] = scale * B * wq[j] * phi[j] * 2 * (re[o] - phi[j]);
          dIm[o] = scale * B * wq[j] * phi[j] * 2 * im[o];
        }
    g["sigreg.ecf_re"] = new Tensor(dRe, [V, M, T]);
    g["sigreg.ecf_im"] = new Tensor(dIm, [V, M, T]);
    const dy = new Float32Array(y.size);
    for (let v = 0; v < V; v++)
      for (let b = 0; b < B; b++)
        for (let m = 0; m < M; m++) {
          const yy = y.data[(v * B + b) * M + m];
          let s = 0;
          for (let j = 0; j < T; j++) {
            const o = (v * M + m) * T + j;
            s += (-dRe[o] * Math.sin(t[j] * yy) + dIm[o] * Math.cos(t[j] * yy)) * t[j];
          }
          dy[(v * B + b) * M + m] = s / B;
        }
    g["sigreg.y"] = new Tensor(dy, y.shape);
    for (let r = 0; r < VB; r++)
      for (let k = 0; k < K; k++) {
        let s = 0;
        for (let m = 0; m < M; m++) s += dy[r * M + m] * A.data[k * M + m];
        dz[r * K + k] += s;
      }
  }
  // Projector: [Linear, BN(eval), ReLU] x (depth - 1), Linear
  let d = new Tensor(dz, [VB, K]);
  const depth = cfg.proj.depth;
  for (let j = depth - 1; j >= 0; j--) {
    const i = 3 * j;
    g[`proj.${i}.linear`] = d;
    const x = i === 0 ? rec["pool"] : rec[`proj.${i - 1}.relu`];
    d = linearBack(x, P[`projector.net.${i}.weight`], d, pg, `projector.net.${i}`);
    if (i === 0) break;
    // ReLU (i-1), BatchNorm (i-2)
    g[`proj.${i - 1}.relu`] = d;
    const pre = rec[`proj.${i - 2}.batchnorm1d`];
    const dr = new Float32Array(d.size);
    for (let q = 0; q < d.size; q++) dr[q] = pre.data[q] > 0 ? d.data[q] : 0;
    g[`proj.${i - 2}.batchnorm1d`] = new Tensor(dr, d.shape);
    const bn = `projector.net.${i - 2}.`;
    const H = d.shape[1];
    const lin = rec[`proj.${i - 3}.linear`];
    const dx = new Float32Array(dr.length);
    const dw = acc(pg, bn + "weight", H), db = acc(pg, bn + "bias", H);
    for (let r = 0; r < VB; r++)
      for (let k = 0; k < H; k++) {
        const inv = 1 / Math.sqrt(P[bn + "running_var"].data[k] + 1e-5);
        const gv = dr[r * H + k];
        dx[r * H + k] = gv * P[bn + "weight"].data[k] * inv;
        dw[k] += gv * (lin.data[r * H + k] - P[bn + "running_mean"].data[k]) * inv;
        db[k] += gv;
      }
    d = new Tensor(dx, d.shape);
  }
  g["pool"] = d;
  // Mean-pool back into each view's tokens, then through the shared encoder.
  const D = d.shape[1];
  const nG = rec["g.out"].shape[0];
  for (const [rp, off, n] of [["g.", 0, nG], ["l.", nG, rec["l.out"]?.shape[0] ?? 0]] as const) {
    if (!n) continue;
    const S = rec[rp + "out"].shape[1];
    const dOut = new Float32Array(n * S * D);
    for (let r = 0; r < n; r++) for (let s = 0; s < S; s++) for (let c = 0; c < D; c++) dOut[(r * S + s) * D + c] = d.data[(off + r) * D + c] / S;
    vitBack(rec, P, "encoder.", rp, cfg.model.depth, new Tensor(dOut, [n, S, D]), pg, g);
  }
  return { grads: g, params: pg };
}

/** Per-row L2 norm over the last axis (what the goldens store as grad.<key>). */
export function rowNorms(t: Tensor): Float32Array {
  const D = t.shape[t.shape.length - 1];
  const out = new Float32Array(t.size / D);
  for (let r = 0; r < out.length; r++) {
    let s = 0;
    for (let k = 0; k < D; k++) s += t.data[r * D + k] ** 2;
    out[r] = Math.sqrt(s);
  }
  return out;
}

export function l2(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
