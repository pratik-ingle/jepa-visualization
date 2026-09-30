"use client";
// Hook: real batch forward passes in a Web Worker, cached per (model, checkpoint).
import { useEffect, useState } from "react";
import type { BatchRequest } from "./batch.worker";
import { BASE, ModelJson } from "./data";

export interface BatchFeats {
  feats: Record<string, Float32Array>; // key -> [n, d]
  n: number;
}

let worker: Worker | null = null;
let nextId = 1;
const cache = new Map<string, Promise<BatchFeats>>();
const listeners = new Map<number, (m: { progress?: number; done?: boolean; feats?: Record<string, Float32Array>; error?: string }) => void>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./batch.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e) => listeners.get(e.data.id)?.(e.data);
  }
  return worker;
}

export function runBatch(model: ModelJson, ckpt: string, images: Uint8Array, n: number, onProgress: (p: number) => void): Promise<BatchFeats> {
  const key = `${model.name}/${ckpt}`;
  if (!cache.has(key))
    cache.set(
      key,
      new Promise((resolve, reject) => {
        const id = nextId++;
        listeners.set(id, (m) => {
          if (m.error) reject(new Error(m.error));
          else if (m.done) {
            listeners.delete(id);
            resolve({ feats: m.feats!, n });
          } else if (m.progress !== undefined) onProgress(m.progress);
        });
        const req: BatchRequest = {
          id, algo: model.algo, n, images: images.slice(), tensors: model.tensors, config: model.config as BatchRequest["config"],
          weightsUrl: new URL(`${BASE}/data/models/${model.name}/${ckpt}`, location.href).href,
        };
        getWorker().postMessage(req, [req.images.buffer]);
      }),
    );
  return cache.get(key)!;
}

export function useBatch(model: ModelJson | null, ckpt: string, images: Uint8Array | null, n: number, enabled: boolean) {
  const [state, setState] = useState<{ data: BatchFeats | null; progress: number; error?: string }>({ data: null, progress: 0 });
  useEffect(() => {
    if (!model || !images || !enabled) return;
    let live = true;
    setState((s) => ({ ...s, data: null, progress: 0 }));
    runBatch(model, ckpt, images, n, (p) => live && setState((s) => ({ ...s, progress: p })))
      .then((d) => live && setState({ data: d, progress: 1 }))
      .catch((e) => live && setState({ data: null, progress: 0, error: String(e) }));
    return () => {
      live = false;
    };
  }, [model, ckpt, images, n, enabled]);
  return state;
}
