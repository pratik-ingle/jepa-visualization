/// <reference lib="webworker" />
// Batch forward passes off the main thread: pooled features (and LeJEPA z) for the 256-image batch.
import { decodeBundle, Manifest } from "@/model/bundle";
import { normalize, projector } from "@/model/lejepa";
import { meanTokens } from "@/model/tensor";
import { ModelCfg, Params, vit } from "@/model/vit";

export interface BatchRequest {
  id: number;
  algo: "ijepa" | "lejepa";
  weightsUrl: string;
  tensors: Manifest;
  config: { model: ModelCfg; proj?: { depth: number }; ema?: { shared: boolean } };
  images: Uint8Array; // n * 3 * 32 * 32
  n: number;
}

const paramCache = new Map<string, Params>();

self.onmessage = async (e: MessageEvent<BatchRequest>) => {
  const r = e.data;
  try {
    let P = paramCache.get(r.weightsUrl);
    if (!P) {
      const buf = await (await fetch(r.weightsUrl)).arrayBuffer();
      P = decodeBundle(r.tensors, buf).f;
      paramCache.set(r.weightsUrl, P);
    }
    const D = r.config.model.dim, px = 3 * 32 * 32, chunk = 16;
    const out: Record<string, Float32Array> = {};
    const keys = r.algo === "ijepa" ? ["target", "context"] : ["backbone", "z"];
    for (let i = 0; i < r.n; i += chunk) {
      const m = Math.min(chunk, r.n - i);
      const x01 = new Float32Array(m * px);
      for (let k = 0; k < m * px; k++) x01[k] = r.images[i * px + k] / 255;
      const x = normalize(x01, m, 32);
      if (r.algo === "ijepa") {
        const tgt = meanTokens(vit(x, P, r.config.ema?.shared ? "encoder." : "target_encoder.", r.config.model, undefined, ""));
        const ctx = meanTokens(vit(x, P, "encoder.", r.config.model, undefined, ""));
        (out.target ??= new Float32Array(r.n * D)).set(tgt.data, i * D);
        (out.context ??= new Float32Array(r.n * D)).set(ctx.data, i * D);
      } else {
        const eb = meanTokens(vit(x, P, "encoder.", r.config.model, undefined, ""));
        const z = projector(eb, P, r.config.proj!.depth, undefined);
        (out.backbone ??= new Float32Array(r.n * D)).set(eb.data, i * D);
        (out.z ??= new Float32Array(r.n * z.shape[1])).set(z.data, i * z.shape[1]);
      }
      (self as unknown as Worker).postMessage({ id: r.id, progress: (i + m) / r.n });
    }
    (self as unknown as Worker).postMessage({ id: r.id, done: true, feats: out, keys }, Object.values(out).map((a) => a.buffer));
  } catch (err) {
    (self as unknown as Worker).postMessage({ id: r.id, error: String(err) });
  }
};
