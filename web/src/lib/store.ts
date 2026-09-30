// Tiny external store for UI state (useSyncExternalStore), no dependency.
import { useSyncExternalStore } from "react";
import { MaskSet, Rect } from "@/model/masks";

export interface Hover {
  label: string;
  shape: string;
  atom: string;
  row: number;
  col: number;
  value: number;
  rowName: string;
  colName: string;
  note?: string;
}

export interface State {
  imageIdx: number;
  // I-JEPA
  maskSeed: number;
  npred: number;
  predScaleMax: number;
  targetsOverride: Rect[] | null; // set when the user drags blocks
  tgtBlock: number; // which target block the predictor view follows
  head: number;
  showTarget: boolean;
  ijepaCkpt: string;
  // LeJEPA
  viewSeed: number;
  viewIdx: number;
  color: boolean;
  lejepaCkpt: string;
  // interaction
  trace: { group: "ijepa" | "lejepa"; space: "token" | "view"; token: number } | null;
  hover: Hover | null;
  sidebar: boolean;
  resetView: number; // bump to re-frame the camera
  // regimes, walkthrough, explorer (milestones 3-5)
  regime: Regime;
  mode: "walk" | "free";
  phase: string; // phase id
  gradView: boolean; // colour blocks by dL/d(activation) instead of the activation
  tau: number; // EMA momentum slider
  emaSteps: number; // how many EMA updates to apply with that tau
  sigSlice: number; // SIGReg: which 1-D slice is inspected
  dirSeed: number; // SIGReg: direction sample
  sigM: number; // SIGReg: number of slices drawn in the browser
  explorer: boolean;
  runSel: { ijepa: string; lejepa: string };
  ckpt: number; // PCA checkpoint index; -1 = last
  cloudKey: { ijepa: "target" | "context"; lejepa: "backbone" | "z" };
  flyTo: { id: string; n: number } | null; // camera request
  reveal: number; // backward animation: fraction of blocks (from the loss backwards) showing gradient
}

export type Regime = "train" | "val" | "eval";

export const initialState: State = {
  imageIdx: 0,
  maskSeed: 1,
  npred: 4,
  predScaleMax: 0.2,
  targetsOverride: null,
  tgtBlock: 0,
  head: 0,
  showTarget: true,
  ijepaCkpt: "final.bin",
  viewSeed: 1,
  viewIdx: 0,
  color: true,
  lejepaCkpt: "final.bin",
  trace: null,
  hover: null,
  sidebar: true,
  resetView: 0,
  regime: "train",
  mode: "walk",
  phase: "intro",
  gradView: false,
  tau: 0.996,
  emaSteps: 1,
  sigSlice: 0,
  dirSeed: 7,
  sigM: 16,
  explorer: false,
  runSel: { ijepa: "ijepa-base", lejepa: "lejepa-base" },
  ckpt: -1,
  cloudKey: { ijepa: "target", lejepa: "z" },
  flyTo: null,
  reveal: 1,
};

type Listener = () => void;
let state = initialState;
const listeners = new Set<Listener>();

export const store = {
  get: () => state,
  set(patch: Partial<State> | ((s: State) => Partial<State>)) {
    state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
    listeners.forEach((l) => l());
  },
  subscribe(l: Listener) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};

export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(store.subscribe, () => sel(state), () => sel(initialState));
}

export type { MaskSet };

/** Animate a numeric store field with requestAnimationFrame (used for the backward / EMA animations). */
export function animate(duration: number, onFrame: (u: number) => void, done?: () => void) {
  const t0 = performance.now();
  const tick = () => {
    const u = Math.min(1, (performance.now() - t0) / duration);
    onFrame(u);
    if (u < 1) requestAnimationFrame(tick);
    else done?.();
  };
  requestAnimationFrame(tick);
}
