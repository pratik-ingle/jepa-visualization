import { describe, expect, it } from "vitest";
import { blockSize, carve, DEFAULT_MASK, rectTokens, sampleMasks, withTargets } from "./masks";
import { mulberry32 } from "./random";
import { sampleDirections, sampleView } from "./views";

describe("mask sampler (single image)", () => {
  it("targets are rectangles, context excludes every target, sizes in range", () => {
    for (let seed = 0; seed < 200; seed++) {
      const m = sampleMasks(DEFAULT_MASK, mulberry32(seed));
      expect(m.targets).toHaveLength(4);
      const ctx = new Set(m.ctxIdx);
      expect(ctx.size).toBe(m.ctxIdx.length);
      expect(m.ctxIdx.length).toBeGreaterThan(DEFAULT_MASK.minKeep);
      for (const t of m.tgtIdx) {
        expect(t.length).toBeGreaterThanOrEqual(9);
        expect(t.length).toBeLessThanOrEqual(16);
        for (const k of t) expect(ctx.has(k)).toBe(false);
      }
    }
  });

  it("matches the Python block-size rule", () => {
    // python: MultiBlockMaskSampler().block_size(u, (0.15, 0.2), (0.75, 1.5)) for u = 0, 0.5, 1
    expect(blockSize(DEFAULT_MASK, 0, [0.15, 0.2], [0.75, 1.5])).toEqual([3, 3]);
    expect(blockSize(DEFAULT_MASK, 1, [0.15, 0.2], [0.75, 1.5])).toEqual([4, 3]);
    expect(blockSize(DEFAULT_MASK, 1, [0.85, 1.0], [1, 1])).toEqual([8, 8]);
  });

  it("moving a target re-carves the context", () => {
    const m = sampleMasks(DEFAULT_MASK, mulberry32(3));
    const moved = withTargets(m, [{ ...m.targets[0], top: 0, left: 0 }, ...m.targets.slice(1)], 8);
    for (const k of rectTokens(moved.targets[0], 8)) expect(moved.ctxIdx).not.toContain(k);
    expect(carve({ top: 0, left: 0, h: 8, w: 8 }, [], 8)).toHaveLength(64);
  });
});

describe("view + direction samplers", () => {
  it("crop boxes stay inside the image with the requested area", () => {
    const rng = mulberry32(1);
    for (let i = 0; i < 500; i++) {
      const [x0, y0, w, h] = sampleView(rng, [0.05, 0.3]).box;
      expect(x0).toBeGreaterThanOrEqual(0);
      expect(y0).toBeGreaterThanOrEqual(0);
      expect(x0 + w).toBeLessThanOrEqual(32 + 1e-9);
      expect(y0 + h).toBeLessThanOrEqual(32 + 1e-9);
      expect((w * h) / 1024).toBeLessThanOrEqual(0.3 + 1e-9);
    }
  });

  it("directions are unit columns", () => {
    const A = sampleDirections(mulberry32(2), 32, 16);
    for (let m = 0; m < 16; m++) {
      let n = 0;
      for (let k = 0; k < 32; k++) n += A[k * 16 + m] ** 2;
      expect(n).toBeCloseTo(1, 5);
    }
  });
});
