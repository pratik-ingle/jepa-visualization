// Logged training runs (model/runs -> data/runs): index, columnar metrics, PCA-3D checkpoint clouds.
import { decodeBundle, Manifest } from "@/model/bundle";
import { fetchJson } from "./data";
import { BASE } from "./data";

export interface RunEntry {
  name: string;
  algo: "ijepa" | "lejepa";
  tags: string[];
  set: Record<string, unknown>;
  steps: number;
  diverged: boolean;
  wall_s: number;
  final: Record<string, number | string | null>;
}
export interface AxisPoint {
  value: number | string | number[];
  run: string;
  baseline?: boolean;
}
export interface RunIndex {
  runs: RunEntry[];
  axes: Record<"ijepa" | "lejepa", Record<string, AxisPoint[]>>;
  presets: Record<string, string[]>;
  grids: Record<string, string[]>;
}
export type Columns = Record<string, (number | null)[]>;
export interface Metrics {
  steps_per_epoch: number;
  train: Columns;
  eval: Columns & Record<string, unknown>;
}
export interface Cloud {
  steps: number[];
  keys: Record<string, { data: Float32Array; evr: number[][]; scale: number; n: number }>;
}

let indexP: Promise<RunIndex> | null = null;
export const loadIndex = () => (indexP ??= fetchJson<RunIndex>("runs/index.json"));

const metricsCache = new Map<string, Promise<Metrics>>();
export function loadMetrics(name: string): Promise<Metrics> {
  if (!metricsCache.has(name)) metricsCache.set(name, fetchJson<Metrics>(`runs/${name}/metrics.json`));
  return metricsCache.get(name)!;
}

const cloudCache = new Map<string, Promise<Cloud>>();
export function loadCloud(name: string): Promise<Cloud> {
  if (!cloudCache.has(name))
    cloudCache.set(
      name,
      (async () => {
        const meta = await fetchJson<{ steps: number[]; tensors: Manifest; keys: Record<string, { evr: number[][]; scale: number }> }>(`runs/${name}/pca.json`);
        const buf = await (await fetch(`${BASE}/data/runs/${name}/pca.bin`)).arrayBuffer();
        const b = decodeBundle(meta.tensors, buf);
        const keys: Cloud["keys"] = {};
        for (const [k, t] of Object.entries(b.f)) {
          const base = k.replace(/\.fixed$/, "");
          const m = meta.keys[base];
          keys[k] = { data: t.data, evr: m?.evr ?? [], scale: m?.scale ?? 1, n: t.shape[1] };
        }
        return { steps: meta.steps, keys };
      })(),
    );
  return cloudCache.get(name)!;
}

/** "reg.lam" -> "λ", etc. for slider labels. */
export const AXIS_LABEL: Record<string, string> = {
  teacher: "teacher (EMA τ / stop-grad)",
  predictor: "predictor",
  "mask.pred_scale": "target block scale",
  "mask.npred": "target blocks",
  "mask.enc_scale": "context block scale",
  "pred.depth": "predictor depth",
  "optim.lr": "peak learning rate",
  "optim.warmup_epochs": "warmup epochs",
  "optim.schedule": "LR schedule",
  "optim.wd_schedule": "weight-decay schedule",
  "optim.beta2": "Adam β₂",
  "optim.name": "optimiser",
  "model.depth": "encoder depth",
  loss: "loss",
  "reg.lam": "λ (SIGReg weight)",
  slices: "SIGReg slices M",
  "views.n_local": "local views V_l",
  "views.local_scale": "local crop scale",
  "proj.out": "embedding dim K",
  "proj.depth": "projector depth",
  "reg.kind": "regulariser",
  "reg.t_max": "Epps–Pulley t range",
};

export const fmtValue = (v: AxisPoint["value"]) => (Array.isArray(v) ? `${v[0]}–${v[1]}` : String(v));
