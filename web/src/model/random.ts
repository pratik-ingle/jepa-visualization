// Seeded PRNG for the browser's own mask / crop / direction sampling. It is not PyTorch's RNG:
// goldens pin explicit indices and boxes, so the forward pass is tested independently of sampling.

export type Rng = () => number; // uniform [0, 1)

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const randint = (rng: Rng, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo)); // [lo, hi)

export function normal(rng: Rng): number {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
