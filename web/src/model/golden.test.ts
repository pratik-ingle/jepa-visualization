// The browser forward passes must reproduce the PyTorch goldens tensor-for-tensor.
// Goldens: data/golden/<name>/ (regenerate: python -m jepa_viz.export --only golden).
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { decodeBundle, Manifest } from "./bundle";
import { ijepaForward, IJepaCfg } from "./ijepa";
import { LeJepaCfg, lejepaForward, normalize, renderView, ViewParams } from "./lejepa";
import { Tensor } from "./tensor";

const DATA = path.resolve(__dirname, "../../../data");

function load(dir: string, json: string, bin: string, manifestKey: string) {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, dir, json), "utf8"));
  const buf = fs.readFileSync(path.join(DATA, dir, bin));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  return { meta, bundle: decodeBundle(meta[manifestKey] as Manifest, ab) };
}

interface Row {
  key: string;
  err: number;
  ref: number;
  ok: boolean;
}

function compare(rec: Record<string, Tensor>, golden: Record<string, Tensor>, loose: string[] = []): Row[] {
  const rows: Row[] = [];
  for (const [key, t] of Object.entries(rec)) {
    const g = golden[key];
    if (!g) continue;
    expect(t.size, `size of ${key}`).toBe(g.size);
    const tol = loose.includes(key) ? 1e-3 : 2e-4;
    let err = 0, ref = 0, ok = true;
    for (let i = 0; i < g.size; i++) {
      const e = Math.abs(t.data[i] - g.data[i]);
      err = Math.max(err, e);
      ref = Math.max(ref, Math.abs(g.data[i]));
      if (e > tol + tol * Math.abs(g.data[i])) ok = false;
    }
    rows.push({ key, err, ref, ok });
  }
  return rows;
}

function report(name: string, rows: Row[]) {
  const worst = rows.reduce((a, r) => (r.err / (r.ref + 1e-6) > a.err / (a.ref + 1e-6) ? r : a));
  console.log(`${name}: ${rows.length} tensors, worst ${worst.key} max|err|=${worst.err.toExponential(2)} (max|ref|=${worst.ref.toExponential(2)})`);
  const bad = rows.filter((r) => !r.ok);
  for (const r of bad.slice(0, 10)) console.log(`  FAIL ${r.key}: ${r.err.toExponential(3)}`);
  return bad;
}

const haveGoldens = fs.existsSync(path.join(DATA, "golden/ijepa-base/golden.json"));

describe.skipIf(!haveGoldens)("forward passes match PyTorch goldens", () => {
  it("I-JEPA: context encoder, EMA target encoder, predictor, loss", () => {
    const { meta: mj, bundle: model } = load("models/ijepa-base", "model.json", "final.bin", "tensors");
    const { meta: gj, bundle: G } = load("golden/ijepa-base", "golden.json", "golden.bin", "tensors");
    const rec: Record<string, Tensor> = {};
    const out = ijepaForward(model.f, mj.config as IJepaCfg, G.f["input.x"], G.i["input.ctx_idx"], G.i["input.tgt_idx"], rec);
    const rows = compare(rec, G.f);
    const bad = report("ijepa-base", rows);
    expect(rows.length).toBeGreaterThan(130);
    expect(bad.map((r) => r.key)).toEqual([]);
    expect(out.loss).toBeCloseTo(gj.loss, 5);
  });

  it("LeJEPA: view rendering, shared encoder, projector, invariance, SIGReg", () => {
    const { meta: mj, bundle: model } = load("models/lejepa-base", "model.json", "final.bin", "tensors");
    const { meta: gj, bundle: G } = load("golden/lejepa-base", "golden.json", "golden.bin", "tensors");
    const cfg = mj.config as LeJepaCfg;
    const imgs = G.u8["input.images"];
    const B = imgs.shape[0];
    const px = 3 * 32 * 32;
    const views = (tag: string, n: number, res: number) => {
      const f = (k: string) => G.f[`input.${tag}.${k}`].data;
      const out = new Float32Array(n * B * 3 * res * res);
      for (let i = 0; i < n * B; i++) {
        const p: ViewParams = {
          box: [f("box")[4 * i], f("box")[4 * i + 1], f("box")[4 * i + 2], f("box")[4 * i + 3]],
          flip: f("flip")[i] > 0.5, jitter: f("jitter")[i] > 0.5, bright: f("bright")[i], contrast: f("contrast")[i],
          sat: f("sat")[i], hue: f("hue")[i], gray: f("gray")[i] > 0.5, sol: f("sol")[i] > 0.5,
        };
        const src = imgs.data.subarray((i % B) * px, (i % B + 1) * px); // view-major: view j of image b = row j*B + b
        out.set(renderView(src, p, res), i * 3 * res * res);
      }
      return out;
    };
    const g01 = views("global", cfg.views.n_global, cfg.views.global_res);
    const l01 = views("local", cfg.views.n_local, cfg.views.local_res);
    const rec: Record<string, Tensor> = {
      "input.global01": new Tensor(g01, [cfg.views.n_global * B, 3, 32, 32]),
      "input.local01": new Tensor(l01, [cfg.views.n_local * B, 3, 16, 16]),
    };
    const g = normalize(g01, cfg.views.n_global * B, cfg.views.global_res);
    const l = normalize(l01, cfg.views.n_local * B, cfg.views.local_res);
    const out = lejepaForward(model.f, cfg, g, l, B, G.f["sigreg.A"], rec);
    const rows = compare(rec, G.f, ["sigreg.ep"]);
    const bad = report("lejepa-base", rows);
    expect(rows.length).toBeGreaterThan(110);
    expect(bad.map((r) => r.key)).toEqual([]);
    expect(out.loss).toBeCloseTo(gj.loss, 4);
  });
});

// ---- gradients (atoms 7 and 14) -------------------------------------------------------------
import { ijepaBackward, l2, lejepaBackward, rowNorms } from "./backward";

function compareGrads(name: string, grads: Record<string, Tensor>, G: Record<string, Tensor>, params: Record<string, Float32Array>, gj: any) {
  let n = 0, worst = 0, worstKey = "";
  const bad: string[] = [];
  for (const [key, ref] of Object.entries(G)) {
    if (!key.startsWith("grad.")) continue;
    const t = grads[key.slice(5)];
    if (!t) continue;
    const got = rowNorms(t);
    expect(got.length, key).toBe(ref.size);
    let refMax = 0, err = 0;
    for (let i = 0; i < got.length; i++) {
      refMax = Math.max(refMax, ref.data[i]);
      err = Math.max(err, Math.abs(got[i] - ref.data[i]));
    }
    const rel = err / (refMax + 1e-12);
    if (rel > worst) (worst = rel), (worstKey = key);
    if (rel > 2e-3) bad.push(`${key} rel ${rel.toExponential(2)}`);
    n++;
  }
  let pn = 0, pworst = 0;
  for (const [p, ref] of Object.entries(gj.param_grad_norm as Record<string, number | null>)) {
    if (ref === null) continue;
    const got = params[p] ? l2(params[p]) : 0;
    const rel = Math.abs(got - ref) / (ref + 1e-12);
    pworst = Math.max(pworst, rel);
    if (rel > 2e-3) bad.push(`param ${p}: ${got} vs ${ref}`);
    pn++;
  }
  console.log(`${name} grads: ${n} activation grads (worst rel ${worst.toExponential(2)} at ${worstKey}), ${pn} param grads (worst rel ${pworst.toExponential(2)})`);
  return { n, pn, bad };
}

describe.skipIf(!haveGoldens)("backward passes match PyTorch gradients", () => {
  it("I-JEPA: predictor + context encoder receive gradient; target encoder none", () => {
    const { meta: mj, bundle: model } = load("models/ijepa-base", "model.json", "final.bin", "tensors");
    const { meta: gj, bundle: G } = load("golden/ijepa-base", "golden.json", "golden.bin", "tensors");
    const rec: Record<string, Tensor> = {};
    const tgt = G.i["input.tgt_idx"];
    ijepaForward(model.f, mj.config as IJepaCfg, G.f["input.x"], G.i["input.ctx_idx"], tgt, rec);
    const { grads, params } = ijepaBackward(model.f, mj.config as IJepaCfg, rec, G.i["input.ctx_idx"].data, tgt.data, tgt.shape[0]);
    const r = compareGrads("ijepa-base", grads, G.f, params, gj);
    expect(r.n).toBe(Object.keys(G.f).filter((k) => k.startsWith("grad.")).length); // every golden gradient compared
    expect(r.bad).toEqual([]);
    expect(Object.values(gj.target_param_grad_norm).every((v) => v === null)).toBe(true);
    expect(Object.keys(params).some((p) => p.startsWith("target_encoder"))).toBe(false);
  });

  it("LeJEPA: invariance + SIGReg gradients reach the one encoder through every view", () => {
    const { meta: mj, bundle: model } = load("models/lejepa-base", "model.json", "final.bin", "tensors");
    const { meta: gj, bundle: G } = load("golden/lejepa-base", "golden.json", "golden.bin", "tensors");
    const cfg = mj.config as LeJepaCfg;
    const B = G.u8["input.images"].shape[0];
    const rec: Record<string, Tensor> = {};
    const g = normalize(G.f["input.global01"].data, cfg.views.n_global * B, 32);
    const l = normalize(G.f["input.local01"].data, cfg.views.n_local * B, 16);
    lejepaForward(model.f, cfg, g, l, B, G.f["sigreg.A"], rec);
    const { grads, params } = lejepaBackward(model.f, cfg, rec, B, G.f["sigreg.A"]);
    const r = compareGrads("lejepa-base", grads, G.f, params, gj);
    expect(r.n).toBe(Object.keys(G.f).filter((k) => k.startsWith("grad.")).length);
    expect(r.bad).toEqual([]);
  });
});
