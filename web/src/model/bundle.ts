// Decoding of /data tensor bundles: a .bin of little-endian arrays + a JSON manifest
// {name: {dtype, shape, offset}} (see model/jepa_viz/export.py: Bundle).
import { IntTensor, Tensor } from "./tensor";

export type DType = "f16" | "f32" | "i32" | "u8";
export interface TensorMeta {
  dtype: DType;
  shape: number[];
  offset: number;
}
export type Manifest = Record<string, TensorMeta>;

export interface Bundle {
  f: Record<string, Tensor>; // f16 / f32 arrays, decoded to Float32
  i: Record<string, IntTensor>; // i32 arrays
  u8: Record<string, { data: Uint8Array; shape: number[] }>;
}

let F16: Float32Array | null = null;
function f16Table(): Float32Array {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    F16[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  return F16;
}

const count = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

export function decodeBundle(manifest: Manifest, buf: ArrayBuffer): Bundle {
  const out: Bundle = { f: {}, i: {}, u8: {} };
  const dv = new DataView(buf);
  for (const [name, m] of Object.entries(manifest)) {
    const n = count(m.shape);
    if (m.dtype === "f16") {
      const t = f16Table();
      const a = new Float32Array(n);
      for (let k = 0; k < n; k++) a[k] = t[dv.getUint16(m.offset + 2 * k, true)];
      out.f[name] = new Tensor(a, m.shape);
    } else if (m.dtype === "f32") {
      const a = new Float32Array(n);
      for (let k = 0; k < n; k++) a[k] = dv.getFloat32(m.offset + 4 * k, true);
      out.f[name] = new Tensor(a, m.shape);
    } else if (m.dtype === "i32") {
      const a = new Int32Array(n);
      for (let k = 0; k < n; k++) a[k] = dv.getInt32(m.offset + 4 * k, true);
      out.i[name] = { data: a, shape: m.shape };
    } else {
      out.u8[name] = { data: new Uint8Array(buf, m.offset, n).slice(), shape: m.shape };
    }
  }
  return out;
}
