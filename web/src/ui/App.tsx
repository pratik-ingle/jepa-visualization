"use client";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { useBatch } from "@/lib/batch";
import { IJepaRun, LeJepaRun, makeMasks, runIJepa, runLeJepa } from "@/lib/compute";
import { DatasetJson, fetchJson, loadAtlas, loadParams, ModelJson } from "@/lib/data";
import { Cloud, loadCloud, loadIndex, RunIndex } from "@/lib/runs";
import { store, useStore } from "@/lib/store";
import { ijepaBackward, lejepaBackward } from "@/model/backward";
import { IJepaCfg } from "@/model/ijepa";
import { LeJepaCfg, sigreg, tGrid } from "@/model/lejepa";
import { mulberry32 } from "@/model/random";
import { Tensor } from "@/model/tensor";
import { Params } from "@/model/vit";
import { sampleDirections } from "@/model/views";
import { cloudGroup, combine, LEJEPA_X0, layoutIJepa, layoutLeJepa, TOWER_W } from "@/scene/layout";
import { phasesFor, type Phase } from "@/walkthrough/phases";
import Explorer from "./Explorer";
import Panel from "./Panel";
import Tooltip from "./Tooltip";
import TopBar from "./TopBar";
import { goPhase, Timeline, useWalkKeys, WalkPanel } from "./WalkPanel";
import { AxesWidget, CloudWidget, EmaWidget, KnnWidget, MaskWidget, OptimWidget, ProbeWidget, SigregWidget, SigState, ValWidget } from "./widgets";

const Scene = dynamic(() => import("@/scene/Scene"), { ssr: false });

export interface Assets {
  dataset: DatasetJson;
  images: Uint8Array[];
  ijepa: ModelJson;
  lejepa: ModelJson;
}

const to01 = (img: Uint8Array) => Float32Array.from(img, (v) => v / 255);
const CLOUD = 420;

function useCloud(name: string) {
  const [c, setC] = useState<Cloud | null>(null);
  useEffect(() => {
    let live = true;
    loadCloud(name).then((x) => live && setC(x)).catch(() => live && setC(null));
    return () => {
      live = false;
    };
  }, [name]);
  return c;
}

export default function App() {
  const [assets, setAssets] = useState<Assets | null>(null);
  const [batchImgs, setBatchImgs] = useState<Uint8Array | null>(null);
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<{ ijepa: Params; lejepa: Params } | null>(null);
  const [runs, setRuns] = useState<{ ij: IJepaRun; le: LeJepaRun } | null>(null);
  const [wide, setWide] = useState(true);

  const s = {
    imageIdx: useStore((x) => x.imageIdx), maskSeed: useStore((x) => x.maskSeed), npred: useStore((x) => x.npred),
    predScaleMax: useStore((x) => x.predScaleMax), targetsOverride: useStore((x) => x.targetsOverride), ijepaCkpt: useStore((x) => x.ijepaCkpt),
    lejepaCkpt: useStore((x) => x.lejepaCkpt), viewSeed: useStore((x) => x.viewSeed), color: useStore((x) => x.color), head: useStore((x) => x.head),
    tgtBlock: useStore((x) => x.tgtBlock), showTarget: useStore((x) => x.showTarget), viewIdx: useStore((x) => x.viewIdx), regime: useStore((x) => x.regime),
    mode: useStore((x) => x.mode), phase: useStore((x) => x.phase), gradView: useStore((x) => x.gradView), tau: useStore((x) => x.tau),
    emaSteps: useStore((x) => x.emaSteps), sigSlice: useStore((x) => x.sigSlice), dirSeed: useStore((x) => x.dirSeed), sigM: useStore((x) => x.sigM),
    runSelI: useStore((x) => x.runSel.ijepa), runSelL: useStore((x) => x.runSel.lejepa), ckpt: useStore((x) => x.ckpt),
    keyI: useStore((x) => x.cloudKey.ijepa), keyL: useStore((x) => x.cloudKey.lejepa), sidebar: useStore((x) => x.sidebar), explorer: useStore((x) => x.explorer),
  };
  useWalkKeys();

  const phases = phasesFor(s.regime);
  const phase: Phase = phases.find((p) => p.id === s.phase) ?? phases[0];
  const gradOn = s.regime === "train" && (s.mode === "walk" ? !!phase.grad : s.gradView);
  const frozen = s.regime === "eval";

  // ---- assets ------------------------------------------------------------------------------
  useEffect(() => {
    const narrow = window.innerWidth < 640;
    setWide(!narrow);
    if (narrow) store.set({ sidebar: false });
    (async () => {
      const [dataset, ijepa, lejepa] = await Promise.all([
        fetchJson<DatasetJson>("dataset.json"),
        fetchJson<ModelJson>("models/ijepa-base/model.json"),
        fetchJson<ModelJson>("models/lejepa-base/model.json"),
      ]);
      const a = dataset.atlases.showcase;
      setAssets({ dataset, images: await loadAtlas(a.file, a.n, a.cols), ijepa, lejepa });
      loadIndex().then(setIndex).catch(() => undefined);
      const bt = dataset.atlases.batch;
      const imgs = await loadAtlas(bt.file, bt.n, bt.cols);
      const flat = new Uint8Array(imgs.length * 3072);
      imgs.forEach((im, i) => flat.set(im, i * 3072));
      setBatchImgs(flat);
    })().catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!assets) return;
    Promise.all([loadParams(assets.ijepa, s.ijepaCkpt), loadParams(assets.lejepa, s.lejepaCkpt)])
      .then(([ij, le]) => setParams({ ijepa: ij, lejepa: le }))
      .catch((e) => setError(String(e)));
  }, [assets, s.ijepaCkpt, s.lejepaCkpt]);

  // ---- forward passes (per input change) -------------------------------------------------------
  useEffect(() => {
    if (!assets || !params) return;
    const img = assets.images[s.imageIdx];
    const ij = runIJepa(params.ijepa, assets.ijepa.config as IJepaCfg, img, makeMasks(store.get()));
    const le = runLeJepa(params.lejepa, assets.lejepa.config as LeJepaCfg, img, s.viewSeed, s.color);
    setRuns({ ij, le });
  }, [assets, params, s.imageIdx, s.maskSeed, s.npred, s.predScaleMax, s.targetsOverride, s.viewSeed, s.color]);

  // ---- backward passes (only when a gradient view is on) ---------------------------------------
  const grads = useMemo(() => {
    if (!gradOn || !runs || !params || !assets) return null;
    const m = runs.ij.masks;
    const gi = ijepaBackward(params.ijepa, assets.ijepa.config as IJepaCfg, runs.ij.rec, Int32Array.from(m.ctxIdx), Int32Array.from(m.tgtIdx.flat()), m.tgtIdx.length);
    const gl = lejepaBackward(params.lejepa, assets.lejepa.config as LeJepaCfg, runs.le.rec, 1, null);
    return { ij: gi.grads, le: gl.grads };
  }, [gradOn, runs, params, assets]);

  // ---- batch features (Web Worker) + SIGReg on the batch -----------------------------------------
  const batchLE = useBatch(assets?.lejepa ?? null, s.lejepaCkpt, batchImgs, 256, true);
  const batchIJ = useBatch(assets?.ijepa ?? null, s.ijepaCkpt, batchImgs, 256, s.regime === "eval");
  const sig: SigState | null = useMemo(() => {
    const f = batchLE.data;
    if (!f || !assets) return null;
    const cfg = assets.lejepa.config as LeJepaCfg;
    const K = cfg.proj.out, n = f.n, M = s.sigM;
    const A = sampleDirections(mulberry32(s.dirSeed * 7919 + M), K, M);
    const t = tGrid(cfg.reg.t_max, cfg.reg.t_points);
    const out = sigreg(new Tensor(f.feats.z, [1, n, K]), new Tensor(A, [K, M]), t);
    return { z: f.feats.z, n, K, A, M, out, t };
  }, [batchLE.data, assets, s.sigM, s.dirSeed]);

  // ---- logged runs --------------------------------------------------------------------------------
  const cloudI = useCloud(s.runSelI), cloudL = useCloud(s.runSelL);

  const layout = useMemo(() => {
    if (!assets || !runs || !params) return null;
    const img01 = to01(assets.images[s.imageIdx]);
    const ij = layoutIJepa(runs.ij, img01, {
      head: s.head, tgtBlock: s.tgtBlock, showTarget: s.showTarget, grads: grads?.ij ?? null, frozen,
      ema: s.regime === "train" ? { P: params.ijepa, tau: s.tau, steps: s.emaSteps } : null,
    });
    const le = layoutLeJepa(runs.le, img01, {
      head: s.head, viewIdx: s.viewIdx, grads: grads?.le ?? null, frozen,
      sig: s.regime !== "eval" && sig ? { ...sig, slice: Math.min(s.sigSlice, sig.M - 1) } : null,
    }, LEJEPA_X0);
    const labels = s.regime === "eval" ? assets.dataset.labels.pca : null;
    const colourNote = labels ? "coloured by class" : "held-out images";
    if (cloudI) cloudGroup(ij, cloudI, s.keyI, s.ckpt, labels, TOWER_W / 2, ij.bottom() - 110 - CLOUD / 2, CLOUD, `Embedding cloud · ${s.runSelI} (${s.keyI})`, `atom 15 · 512 ${colourNote}`);
    if (cloudL) cloudGroup(le, cloudL, s.keyL, s.ckpt, labels, LEJEPA_X0 + TOWER_W / 2, le.bottom() - 110 - CLOUD / 2, CLOUD, `Embedding cloud · ${s.runSelL} (${s.keyL})`, `atom 15 · 512 ${colourNote}`);
    return combine([ij, le]);
  }, [assets, runs, params, grads, sig, cloudI, cloudL, frozen, s.imageIdx, s.head, s.tgtBlock, s.showTarget, s.viewIdx, s.regime, s.tau, s.emaSteps, s.sigSlice, s.keyI, s.keyL, s.ckpt, s.runSelI, s.runSelL]);

  // Fly to the current phase once the scene exists.
  useEffect(() => {
    if (layout && s.mode === "walk" && store.get().flyTo === null) goPhase(phase.id);
  }, [layout, s.mode, phase.id]);

  const insets = {
    left: s.sidebar && wide ? (s.mode === "walk" ? 380 : 330) : 0,
    right: s.explorer && wide ? 430 : 0,
    bottom: s.mode === "walk" ? 56 : 0,
  };

  const widget = (p: Phase) => {
    if (!runs || !assets) return null;
    switch (p.widget) {
      case "mask": return <MaskWidget ij={runs.ij} />;
      case "optim": return <OptimWidget />;
      case "ema": return <EmaWidget />;
      case "sigreg": return <SigregWidget sig={sig} progress={batchLE.progress} />;
      case "cloud": return <CloudWidget index={index} />;
      case "valcurve": return <ValWidget index={index} />;
      case "knn": return <KnnWidget fi={batchIJ.data} fl={batchLE.data} pi={batchIJ.progress} pl={batchLE.progress} labels={assets.dataset.labels.batch} classes={assets.dataset.classes} />;
      case "probe": return <ProbeWidget index={index} />;
      case "axes": return <AxesWidget index={index} />;
      default: return null;
    }
  };

  return (
    <div className="relative h-dvh w-screen overflow-hidden bg-[#0b0e13] text-slate-200">
      <TopBar />
      {layout ? (
        <div className="absolute inset-0 top-11">
          <Scene layout={layout} insets={insets} />
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-sm text-slate-400">{error ?? "Loading models and running the forward passes…"}</div>
      )}
      {assets && runs && (s.mode === "walk" ? (
        <>
          <WalkPanel widget={widget} />
          <Timeline />
        </>
      ) : (
        <Panel assets={assets} ij={runs.ij} le={runs.le} />
      ))}
      {s.explorer && (
        <aside className="absolute right-0 top-11 z-20 w-full border-l border-white/10 bg-[#0d1117]/95 backdrop-blur sm:w-[420px]" style={{ bottom: insets.bottom }}>
          <Explorer index={index} />
        </aside>
      )}
      <Tooltip right={insets.right + 12} bottom={insets.bottom + 12} />
    </div>
  );
}
