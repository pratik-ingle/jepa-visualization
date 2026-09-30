// LeJEPA multi-crop view sampling for ONE image (ATOMS atom 10), mirroring augment.sample_view_params.
import { ViewParams } from "./lejepa";
import { normal, Rng } from "./random";

function sampleBox(rng: Rng, scale: [number, number], size = 32, ratio: [number, number] = [3 / 4, 4 / 3]): [number, number, number, number] {
  for (let k = 0; k < 10; k++) {
    const s = scale[0] + (scale[1] - scale[0]) * rng();
    const r = Math.exp(Math.log(ratio[0]) + (Math.log(ratio[1]) - Math.log(ratio[0])) * rng());
    const w = Math.sqrt(s * size * size * r), h = Math.sqrt((s * size * size) / r);
    if (w <= size && h <= size) return [rng() * (size - w), rng() * (size - h), w, h];
  }
  return [0, 0, size, size];
}

export function sampleView(rng: Rng, scale: [number, number], color = true): ViewParams {
  const box = sampleBox(rng, scale);
  const u = () => rng();
  return {
    box,
    flip: u() < 0.5,
    jitter: color && u() < 0.8,
    bright: 0.6 + 0.8 * u(),
    contrast: 0.6 + 0.8 * u(),
    sat: 0.8 + 0.4 * u(),
    hue: -0.1 + 0.2 * u(),
    gray: color && u() < 0.2,
    sol: color && u() < 0.2,
  };
}

export const IDENTITY_VIEW: ViewParams = {
  box: [0, 0, 32, 32], flip: false, jitter: false, bright: 1, contrast: 1, sat: 1, hue: 0, gray: false, sol: false,
};

/** Unit-norm random directions A [K, M] (columns), as in sigreg.sample_directions. */
export function sampleDirections(rng: Rng, K: number, M: number): Float32Array {
  const A = new Float32Array(K * M);
  for (let i = 0; i < K * M; i++) A[i] = normal(rng);
  for (let m = 0; m < M; m++) {
    let n = 0;
    for (let k = 0; k < K; k++) n += A[k * M + m] ** 2;
    n = Math.sqrt(n);
    for (let k = 0; k < K; k++) A[k * M + m] /= n;
  }
  return A;
}
