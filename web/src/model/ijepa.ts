// I-JEPA forward (ATOMS atoms 3-6) for a batch of images with explicit mask index sets.
import { concat0, gatherTokens, IntTensor, layerNorm, linear, meanTokens, takeRows, Tensor } from "./tensor";
import { block, ModelCfg, Params, posEmbed, put, Rec, vit } from "./vit";

export interface IJepaCfg {
  model: ModelCfg;
  pred: { enabled: boolean; linear?: boolean; dim: number; depth: number; heads: number };
  ema: { shared: boolean };
  loss: string;
}

export interface IJepaOut {
  loss: number;
  sx: Tensor; // context encoder output [B, Nc, D]
  sy: Tensor; // targets [Mt*B, Nt, D]
  shat: Tensor; // predictions [Mt*B, Nt, D]
}

const TARGET_LN_EPS = 1e-5;

/** x [B,3,32,32] normalised; ctxIdx [B,Nc]; tgtIdx [Mt,B,Nt] (block-major). */
export function ijepaForward(P: Params, cfg: IJepaCfg, x: Tensor, ctxIdx: IntTensor, tgtIdx: IntTensor, rec: Rec): IJepaOut {
  const [Mt, B, Nt] = tgtIdx.shape;
  const D = cfg.model.dim;
  const sx = vit(x, P, "encoder.", cfg.model, rec, "ctx.", ctxIdx);
  const Nc = sx.shape[1];

  // Target path: full image through the EMA encoder, LayerNorm without affine, gather per block.
  const h = layerNorm(vit(x, P, cfg.ema.shared ? "encoder." : "target_encoder.", cfg.model, rec, "tgt."), undefined, undefined, TARGET_LN_EPS);
  put(rec, "tgt.ln", h);
  const blockIdx = (i: number): IntTensor => ({ data: tgtIdx.data.subarray(i * B * Nt, (i + 1) * B * Nt), shape: [B, Nt] });
  const sy = concat0(Array.from({ length: Mt }, (_, i) => gatherTokens(h, blockIdx(i))));
  put(rec, "tgt.s_y", sy);

  let shat: Tensor;
  const pc = cfg.pred;
  if (!pc.enabled) {
    // Ablation: pooled context, identity map, no positional conditioning.
    const pooled = meanTokens(sx);
    const y = new Float32Array(Mt * B * Nt * D);
    for (let i = 0; i < Mt; i++)
      for (let b = 0; b < B; b++)
        for (let t = 0; t < Nt; t++) y.set(pooled.data.subarray(b * D, b * D + D), ((i * B + b) * Nt + t) * D);
    shat = new Tensor(y, [Mt * B, Nt, D]);
  } else if (pc.linear) {
    const pos = posEmbed(pc.dim, 8);
    const c = linear(meanTokens(sx), P["predictor.ctx.weight"], P["predictor.ctx.bias"]); // [B, D]
    const pp = linear(takeRows(pos, tgtIdx.data), P["predictor.pos_proj.weight"]); // [Mt*B*Nt, D]
    for (let r = 0; r < Mt * B * Nt; r++) {
      const b = Math.floor(r / Nt) % B;
      for (let d = 0; d < D; d++) pp.data[r * D + d] += c.data[b * D + d];
    }
    shat = pp.reshape([Mt * B, Nt, D]);
  } else {
    const Dp = pc.dim;
    const pos = posEmbed(Dp, 8);
    const c = linear(sx, P["predictor.embed.weight"], P["predictor.embed.bias"]);
    put(rec, "pred.embed", c);
    const cp = new Float32Array(c.data);
    for (let b = 0; b < B; b++)
      for (let k = 0; k < Nc; k++) {
        const pr = ctxIdx.data[b * Nc + k] * Dp;
        for (let d = 0; d < Dp; d++) cp[(b * Nc + k) * Dp + d] += pos.data[pr + d];
      }
    const ctx = new Tensor(cp, [B, Nc, Dp]);
    put(rec, "pred.ctx", ctx);
    const mt = P["predictor.mask_token"].data;
    const mq = takeRows(pos, tgtIdx.data);
    for (let r = 0; r < Mt * B * Nt; r++) for (let d = 0; d < Dp; d++) mq.data[r * Dp + d] += mt[d];
    put(rec, "pred.mask_q", mq.reshape([Mt * B, Nt, Dp]));
    const S = Nc + Nt;
    const inp = new Float32Array(Mt * B * S * Dp);
    for (let i = 0; i < Mt; i++)
      for (let b = 0; b < B; b++) {
        const seq = (i * B + b) * S * Dp;
        inp.set(cp.subarray(b * Nc * Dp, (b + 1) * Nc * Dp), seq);
        inp.set(mq.data.subarray((i * B + b) * Nt * Dp, (i * B + b + 1) * Nt * Dp), seq + Nc * Dp);
      }
    let z = new Tensor(inp, [Mt * B, S, Dp]);
    put(rec, "pred.in", z);
    for (let l = 0; l < pc.depth; l++) z = block(z, P, `predictor.blocks.${l}.`, pc.heads, rec, `pred.blk${l}.`);
    const zn = layerNorm(z, P["predictor.norm.weight"], P["predictor.norm.bias"]);
    const tail = new Float32Array(Mt * B * Nt * Dp);
    for (let r = 0; r < Mt * B; r++) tail.set(zn.data.subarray((r * S + Nc) * Dp, (r + 1) * S * Dp), r * Nt * Dp);
    const tailT = new Tensor(tail, [Mt * B, Nt, Dp]);
    put(rec, "pred.norm", tailT);
    shat = linear(tailT, P["predictor.proj.weight"], P["predictor.proj.bias"]);
  }
  put(rec, "pred.out", shat);

  let loss = 0;
  for (let i = 0; i < shat.size; i++) {
    const d = shat.data[i] - sy.data[i];
    const a = Math.abs(d);
    loss += cfg.loss === "l2" ? d * d : a < 1 ? 0.5 * d * d : a - 0.5;
  }
  loss /= shat.size;
  put(rec, "loss", new Tensor(new Float32Array([loss]), []));
  return { loss, sx, sy, shat };
}
