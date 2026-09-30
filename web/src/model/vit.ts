// Tiny ViT forward (ATOMS atom 1 + transformer sub-atoms T1-T10). Recorded keys match the
// PyTorch `rec` dict exactly (e.g. "ctx.blk0.attn"), so goldens compare key by key.
import { add, gatherTokens, gelu, IntTensor, layerNorm, linear, patchify, sincos2d, Tensor } from "./tensor";

export type Params = Record<string, Tensor>;
export type Rec = Record<string, Tensor> | undefined;

export interface ModelCfg {
  patch: number;
  dim: number;
  depth: number;
  heads: number;
  mlp_ratio: number;
}

export const put = (rec: Rec, key: string, t: Tensor) => {
  if (rec) rec[key] = t;
};

export function block(x: Tensor, P: Params, pre: string, heads: number, rec: Rec, rp: string): Tensor {
  const [B, S, W] = x.shape;
  const dh = W / heads;
  const a = layerNorm(x, P[pre + "norm1.weight"], P[pre + "norm1.bias"]);
  put(rec, rp + "ln1", a);
  const qkv = linear(a, P[pre + "attn.qkv.weight"], P[pre + "attn.qkv.bias"]); // [B, S, 3W]
  const n = B * heads * S * dh;
  const q = new Float32Array(n), k = new Float32Array(n), v = new Float32Array(n);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++) {
      const src = (b * S + s) * 3 * W;
      for (let h = 0; h < heads; h++)
        for (let d = 0; d < dh; d++) {
          const dst = ((b * heads + h) * S + s) * dh + d;
          q[dst] = qkv.data[src + h * dh + d];
          k[dst] = qkv.data[src + W + h * dh + d];
          v[dst] = qkv.data[src + 2 * W + h * dh + d];
        }
    }
  const scale = 1 / Math.sqrt(dh);
  const scores = new Float32Array(B * heads * S * S);
  const attn = new Float32Array(B * heads * S * S);
  const o = new Float32Array(B * S * W);
  for (let b = 0; b < B; b++)
    for (let h = 0; h < heads; h++) {
      const bh = (b * heads + h) * S;
      for (let i = 0; i < S; i++) {
        const row = (bh + i) * S;
        let mx = -Infinity;
        for (let j = 0; j < S; j++) {
          let s = 0;
          for (let d = 0; d < dh; d++) s += q[(bh + i) * dh + d] * k[(bh + j) * dh + d];
          s *= scale;
          scores[row + j] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let j = 0; j < S; j++) {
          const e = Math.exp(scores[row + j] - mx);
          attn[row + j] = e;
          z += e;
        }
        for (let j = 0; j < S; j++) attn[row + j] /= z;
        for (let d = 0; d < dh; d++) {
          let s = 0;
          for (let j = 0; j < S; j++) s += attn[row + j] * v[(bh + j) * dh + d];
          o[(b * S + i) * W + h * dh + d] = s;
        }
      }
    }
  put(rec, rp + "q", new Tensor(q, [B, heads, S, dh]));
  put(rec, rp + "k", new Tensor(k, [B, heads, S, dh]));
  put(rec, rp + "v", new Tensor(v, [B, heads, S, dh]));
  put(rec, rp + "scores", new Tensor(scores, [B, heads, S, S]));
  put(rec, rp + "attn", new Tensor(attn, [B, heads, S, S]));
  const oT = new Tensor(o, [B, S, W]);
  put(rec, rp + "o", oT);
  const x1 = add(x, linear(oT, P[pre + "attn.proj.weight"], P[pre + "attn.proj.bias"]));
  put(rec, rp + "x_attn", x1);
  const m = layerNorm(x1, P[pre + "norm2.weight"], P[pre + "norm2.bias"]);
  put(rec, rp + "ln2", m);
  const u = linear(m, P[pre + "fc1.weight"], P[pre + "fc1.bias"]);
  put(rec, rp + "fc1", u);
  const g = gelu(u);
  put(rec, rp + "gelu", g);
  const x2 = add(x1, linear(g, P[pre + "fc2.weight"], P[pre + "fc2.bias"]));
  put(rec, rp + "x_out", x2);
  return x2;
}

const posCache = new Map<string, Tensor>();
export function posEmbed(dim: number, g: number, G = 8): Tensor {
  const key = `${dim}/${g}/${G}`;
  if (!posCache.has(key)) posCache.set(key, sincos2d(dim, g, G));
  return posCache.get(key)!;
}

/** x: normalised images [B, 3, r, r]; keepIdx: [B, K] tokens kept (context encoder). */
export function vit(x: Tensor, P: Params, pre: string, m: ModelCfg, rec: Rec, rp: string, keepIdx?: IntTensor): Tensor {
  const p = patchify(x, m.patch);
  put(rec, rp + "patches", p);
  const e = linear(p, P[pre + "patch_embed.weight"], P[pre + "patch_embed.bias"]);
  put(rec, rp + "patch_embed", e);
  const pos = posEmbed(m.dim, x.shape[3] / m.patch);
  put(rec, rp + "pos", pos);
  let t = add(e, pos);
  put(rec, rp + "tokens", t);
  if (keepIdx) {
    t = gatherTokens(t, keepIdx);
    put(rec, rp + "kept", t);
  }
  for (let i = 0; i < m.depth; i++) t = block(t, P, `${pre}blocks.${i}.`, m.heads, rec, `${rp}blk${i}.`);
  t = layerNorm(t, P[pre + "norm.weight"], P[pre + "norm.bias"]);
  put(rec, rp + "out", t);
  return t;
}
