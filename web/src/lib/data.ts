// Fetch and decode the /data exports in the browser.
import { decodeBundle, Manifest } from "@/model/bundle";
import { Params } from "@/model/vit";

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const url = (p: string) => `${BASE}/data/${p}`;

export interface Checkpoint {
  file: string;
  step: number;
  bytes: number;
}

export interface ModelJson {
  name: string;
  algo: "ijepa" | "lejepa";
  config: Record<string, any>;
  tensors: Manifest;
  checkpoints: Checkpoint[];
  steps_per_epoch: number;
}

export interface DatasetJson {
  classes: string[];
  mean: number[];
  std: number[];
  labels: Record<string, number[]>;
  atlases: Record<string, { file: string; cols: number; n: number }>;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const r = await fetch(url(path));
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const paramCache = new Map<string, Promise<Params>>();

export function loadParams(model: ModelJson, file: string): Promise<Params> {
  const key = `${model.name}/${file}`;
  if (!paramCache.has(key))
    paramCache.set(
      key,
      fetch(url(`models/${model.name}/${file}`))
        .then((r) => r.arrayBuffer())
        .then((buf) => decodeBundle(model.tensors, buf).f),
    );
  return paramCache.get(key)!;
}

/** Decode an image atlas (32x32 tiles) into uint8 CHW images. */
export async function loadAtlas(file: string, n: number, cols: number): Promise<Uint8Array[]> {
  const img = new Image();
  img.src = url(file);
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  const out: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    const ox = (i % cols) * 32, oy = Math.floor(i / cols) * 32;
    const a = new Uint8Array(3 * 32 * 32);
    for (let y = 0; y < 32; y++)
      for (let x = 0; x < 32; x++) {
        const s = ((oy + y) * c.width + ox + x) * 4;
        for (let ch = 0; ch < 3; ch++) a[ch * 1024 + y * 32 + x] = px[s + ch];
      }
    out.push(a);
  }
  return out;
}
