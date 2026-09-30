// Minimal dense tensors (row-major) and the ops the two JEPA forward passes need.
// Every op mirrors model/jepa_viz/np_forward.py, which is itself checked against PyTorch.

export class Tensor {
  constructor(
    public data: Float32Array,
    public shape: number[],
  ) {
    if (data.length !== numel(shape)) throw new Error(`size ${data.length} != shape ${shape}`);
  }
  static zeros(shape: number[]): Tensor {
    return new Tensor(new Float32Array(numel(shape)), shape);
  }
  get size(): number {
    return this.data.length;
  }
  reshape(shape: number[]): Tensor {
    return new Tensor(this.data, shape);
  }
}

export interface IntTensor {
  data: Int32Array;
  shape: number[];
}

export const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

// y[r, o] = sum_k x[r, k] * W[o, k] + b[o]   (PyTorch Linear layout: W is [out, in])
export function linear(x: Tensor, W: Tensor, b?: Tensor): Tensor {
  const [out, inn] = W.shape;
  const rows = x.size / inn;
  const y = new Float32Array(rows * out);
  const xd = x.data, wd = W.data, bd = b?.data;
  for (let r = 0; r < rows; r++) {
    const xo = r * inn;
    for (let o = 0; o < out; o++) {
      const wo = o * inn;
      let s = bd ? bd[o] : 0;
      for (let k = 0; k < inn; k++) s += xd[xo + k] * wd[wo + k];
      y[r * out + o] = s;
    }
  }
  return new Tensor(y, [...x.shape.slice(0, -1), out]);
}

export function layerNorm(x: Tensor, w?: Tensor, b?: Tensor, eps = 1e-6): Tensor {
  const D = x.shape[x.shape.length - 1];
  const rows = x.size / D;
  const y = new Float32Array(x.size);
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let mu = 0;
    for (let k = 0; k < D; k++) mu += x.data[o + k];
    mu /= D;
    let v = 0;
    for (let k = 0; k < D; k++) {
      const d = x.data[o + k] - mu;
      v += d * d;
    }
    const inv = 1 / Math.sqrt(v / D + eps);
    for (let k = 0; k < D; k++) {
      const n = (x.data[o + k] - mu) * inv;
      y[o + k] = w && b ? n * w.data[k] + b.data[k] : n;
    }
  }
  return new Tensor(y, x.shape);
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 (JS has no Math.erf).
export function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

export function gelu(x: Tensor): Tensor {
  const y = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i];
    y[i] = 0.5 * v * (1 + erf(v / Math.SQRT2));
  }
  return new Tensor(y, x.shape);
}

export function add(a: Tensor, b: Tensor): Tensor {
  const y = new Float32Array(a.size);
  if (b.size === a.size) for (let i = 0; i < a.size; i++) y[i] = a.data[i] + b.data[i];
  else for (let i = 0; i < a.size; i++) y[i] = a.data[i] + b.data[i % b.size]; // broadcast trailing
  return new Tensor(y, a.shape);
}

export function relu(x: Tensor): Tensor {
  const y = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) y[i] = x.data[i] > 0 ? x.data[i] : 0;
  return new Tensor(y, x.shape);
}

// [B, C, H, W] -> [B, (H/p)*(W/p), C*p*p]; token = row*g + col; value = c*p*p + py*p + px
export function patchify(x: Tensor, p: number): Tensor {
  const [B, C, H, W] = x.shape;
  const gh = H / p, gw = W / p, pd = C * p * p;
  const y = new Float32Array(B * gh * gw * pd);
  let i = 0;
  for (let b = 0; b < B; b++)
    for (let r = 0; r < gh; r++)
      for (let c = 0; c < gw; c++)
        for (let ch = 0; ch < C; ch++)
          for (let py = 0; py < p; py++)
            for (let px = 0; px < p; px++) y[i++] = x.data[((b * C + ch) * H + r * p + py) * W + c * p + px];
  return new Tensor(y, [B, gh * gw, pd]);
}

// Fixed 2D sin-cos for a g x g grid inside a G x G frame: [g*g, dim].
// First half of channels encodes the column, second half the row; coord(i) = (i + 0.5) * G / g - 0.5.
export function sincos2d(dim: number, g: number, G = 8): Tensor {
  const out = new Float32Array(g * g * dim);
  const half = dim / 2, q = half / 2;
  const coord = (i: number) => ((i + 0.5) * G) / g - 0.5;
  for (let r = 0; r < g; r++)
    for (let c = 0; c < g; c++) {
      const o = (r * g + c) * dim;
      for (let i = 0; i < q; i++) {
        const om = 1 / Math.pow(10000, i / q);
        out[o + i] = Math.sin(coord(c) * om);
        out[o + q + i] = Math.cos(coord(c) * om);
        out[o + half + i] = Math.sin(coord(r) * om);
        out[o + half + q + i] = Math.cos(coord(r) * om);
      }
    }
  return new Tensor(out, [g * g, dim]);
}

// x [B, N, W], idx [B, K] -> [B, K, W]
export function gatherTokens(x: Tensor, idx: IntTensor): Tensor {
  const [B, N, W] = x.shape;
  const K = idx.shape[1];
  const y = new Float32Array(B * K * W);
  for (let b = 0; b < B; b++)
    for (let k = 0; k < K; k++) {
      const src = (b * N + idx.data[b * K + k]) * W;
      y.set(x.data.subarray(src, src + W), (b * K + k) * W);
    }
  return new Tensor(y, [B, K, W]);
}

// table [N, W] rows picked by flat indices -> [n, W]
export function takeRows(table: Tensor, idx: ArrayLike<number>): Tensor {
  const W = table.shape[1];
  const y = new Float32Array(idx.length * W);
  for (let i = 0; i < idx.length; i++) y.set(table.data.subarray(idx[i] * W, idx[i] * W + W), i * W);
  return new Tensor(y, [idx.length, W]);
}

export function meanTokens(x: Tensor): Tensor {
  const [B, S, W] = x.shape;
  const y = new Float32Array(B * W);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++) for (let k = 0; k < W; k++) y[b * W + k] += x.data[(b * S + s) * W + k] / S;
  return new Tensor(y, [B, W]);
}

export function concat0(ts: Tensor[]): Tensor {
  const size = ts.reduce((a, t) => a + t.size, 0);
  const y = new Float32Array(size);
  let o = 0;
  for (const t of ts) {
    y.set(t.data, o);
    o += t.size;
  }
  return new Tensor(y, [ts.reduce((a, t) => a + t.shape[0], 0), ...ts[0].shape.slice(1)]);
}
