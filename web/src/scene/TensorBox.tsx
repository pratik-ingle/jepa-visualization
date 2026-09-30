"use client";
// One box per tensor; cell values are read from a float texture in the fragment shader
// (the llm-viz approach: a 64x256 activation is one draw call, not 16k instances).
import { ThreeEvent, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export type Mode = 0 | 1 | 2; // 0 diverging (signed), 1 sequential (>= 0, e.g. attention), 2 categorical

export interface BlockSpec {
  id: string;
  label: string;
  atom: string; // ATOMS.md reference shown in the tooltip
  rows: number;
  cols: number;
  data: Float32Array; // row-major rows x cols
  x: number; // top-left corner (world units)
  y: number;
  z?: number;
  cell?: number; // world units per cell (default 1)
  mode: Mode;
  scale?: number; // colour normalisation; default max |v|
  tint: string; // frame / side colour
  rowTokens?: number[]; // token id per row (for tracing), -1 = none
  colTokens?: number[];
  rowName?: (r: number) => string;
  colName?: (c: number) => string;
  note?: string;
  group: "ijepa" | "lejepa";
  space?: "token" | "view"; // what rows index, for tracing (default token)
  noGrad?: boolean; // gradient view: this tensor receives no gradient (drawn dimmed)
  colHL?: number; // a column highlighted regardless of tracing (e.g. the inspected SIGReg slice)
  order?: number; // position in the forward computation (for the backward reveal)
  grad?: boolean; // shows a gradient
}

const vert = /* glsl */ `
out vec3 vPos;
out vec3 vNormal;
void main() {
  vPos = position;
  vNormal = normal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const frag = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D uData;
uniform vec2 uSize;     // (cols, rows)
uniform float uCell;
uniform float uScale;
uniform int uMode;
uniform vec3 uTint;
uniform float uRowHL;
uniform float uColHL;
uniform float uDim;
in vec3 vPos;
in vec3 vNormal;
out vec4 fragColor;

vec3 diverging(float v) {
  v = clamp(v, -1.0, 1.0);
  float a = pow(abs(v), 0.75);
  vec3 zero = vec3(0.105, 0.12, 0.155);
  return v >= 0.0 ? mix(zero, vec3(1.0, 0.63, 0.2), a) : mix(zero, vec3(0.26, 0.62, 1.0), a);
}
vec3 sequential(float v) {
  v = pow(clamp(v, 0.0, 1.0), 0.55);
  vec3 c0 = vec3(0.06, 0.07, 0.12), c1 = vec3(0.1, 0.45, 0.55), c2 = vec3(0.99, 0.92, 0.4);
  return v < 0.5 ? mix(c0, c1, v * 2.0) : mix(c1, c2, (v - 0.5) * 2.0);
}
vec3 categorical(float v) {
  int k = int(v + 0.5);
  if (k == 0) return vec3(0.09, 0.1, 0.13);   // unused
  if (k == 1) return vec3(0.62, 0.65, 0.7);   // context
  if (k == 2) return vec3(0.96, 0.42, 0.36);  // target blocks 1-4
  if (k == 3) return vec3(0.36, 0.8, 0.5);
  if (k == 4) return vec3(0.42, 0.6, 1.0);
  return vec3(0.96, 0.8, 0.3);
}

void main() {
  float cx = vPos.x / uCell + uSize.x * 0.5;
  float cy = uSize.y * 0.5 - vPos.y / uCell;
  ivec2 ij = ivec2(clamp(floor(cx), 0.0, uSize.x - 1.0), clamp(floor(cy), 0.0, uSize.y - 1.0));
  float raw = texelFetch(uData, ij, 0).r;
  vec3 col = uMode == 0 ? diverging(raw / uScale) : uMode == 1 ? sequential(raw / uScale) : categorical(raw);

  if (abs(vNormal.z) < 0.5) {            // side faces: tinted, darker
    col = mix(col, uTint, 0.55) * 0.55;
  } else {
    vec2 c = vec2(cx, cy);
    vec2 w = fwidth(c);
    float px = max(w.x, w.y);             // cells per pixel
    vec2 f = abs(fract(c) - 0.5);
    float edge = (0.5 - max(f.x, f.y)) / max(px, 1e-4); // distance to cell border, in pixels
    float grid = (1.0 - smoothstep(0.12, 0.35, px)) * (1.0 - smoothstep(0.0, 1.2, edge));
    col *= 1.0 - 0.5 * grid;
    // slab outline in the block tint
    float bx = min(cx, uSize.x - cx) / max(w.x, 1e-4), by = min(cy, uSize.y - cy) / max(w.y, 1e-4);
    col = mix(col, uTint, 1.0 - smoothstep(0.5, 1.8, min(bx, by)));
    bool hr = uRowHL >= 0.0 && float(ij.y) == uRowHL;
    bool hc = uColHL >= 0.0 && float(ij.x) == uColHL;
    if (hr || hc) col = mix(col, vec3(1.0), hr && hc ? 0.6 : 0.3);
  }
  fragColor = vec4(col * uDim, 1.0);
}`;

export function autoScale(d: Float32Array, mode: Mode): number {
  let m = 0;
  for (let i = 0; i < d.length; i++) m = Math.max(m, mode === 1 ? d[i] : Math.abs(d[i]));
  return m > 0 ? m : 1;
}

interface Props {
  spec: BlockSpec;
  rowHL: number;
  colHL: number;
  dim: boolean;
  onHover: (spec: BlockSpec | null, row: number, col: number) => void;
  onPick: (spec: BlockSpec, row: number, col: number) => void;
}

export function TensorBox({ spec, rowHL, colHL, dim, onHover, onPick }: Props) {
  const invalidate = useThree((s) => s.invalidate);
  const cell = spec.cell ?? 1;
  const tex = useMemo(() => {
    const t = new THREE.DataTexture(new Float32Array(spec.rows * spec.cols), spec.cols, spec.rows, THREE.RedFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    return t;
  }, [spec.rows, spec.cols]);
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: vert,
        fragmentShader: frag,
        uniforms: {
          uData: { value: tex },
          uSize: { value: new THREE.Vector2(spec.cols, spec.rows) },
          uCell: { value: cell },
          uScale: { value: 1 },
          uMode: { value: spec.mode },
          uTint: { value: new THREE.Color(spec.tint).convertLinearToSRGB() }, // shader writes sRGB directly
          uRowHL: { value: -1 },
          uColHL: { value: -1 },
          uDim: { value: 1 },
        },
      }),
    [tex, spec.cols, spec.rows, cell, spec.mode, spec.tint],
  );
  useEffect(() => {
    (tex.image.data as Float32Array).set(spec.data);
    tex.needsUpdate = true;
    mat.uniforms.uScale.value = spec.scale ?? autoScale(spec.data, spec.mode);
    invalidate();
  }, [spec.data, spec.scale, spec.mode, tex, mat, invalidate]);
  useEffect(() => {
    mat.uniforms.uRowHL.value = rowHL;
    mat.uniforms.uColHL.value = colHL;
    mat.uniforms.uDim.value = dim ? 0.35 : 1;
    invalidate();
  }, [rowHL, colHL, dim, mat, invalidate]);
  useEffect(() => () => (tex.dispose(), mat.dispose()), [tex, mat]);

  const mesh = useRef<THREE.Mesh>(null);
  const cellAt = (e: ThreeEvent<PointerEvent | MouseEvent>): [number, number] | null => {
    if (!mesh.current || !e.face || e.face.normal.z < 0.5) return null;
    const p = mesh.current.worldToLocal(e.point.clone());
    const c = Math.floor(p.x / cell + spec.cols / 2), r = Math.floor(spec.rows / 2 - p.y / cell);
    return r >= 0 && r < spec.rows && c >= 0 && c < spec.cols ? [r, c] : null;
  };
  const w = spec.cols * cell, h = spec.rows * cell;
  return (
    <mesh
      ref={mesh}
      position={[spec.x + w / 2, spec.y - h / 2, spec.z ?? 0]}
      material={mat}
      onPointerMove={(e) => {
        e.stopPropagation();
        const rc = cellAt(e);
        if (rc) onHover(spec, rc[0], rc[1]);
      }}
      onPointerOut={() => onHover(null, -1, -1)}
      onClick={(e) => {
        e.stopPropagation();
        const rc = cellAt(e);
        if (rc) onPick(spec, rc[0], rc[1]);
      }}
    >
      <boxGeometry args={[w, h, Math.min(cell * 0.6, 2)]} />
    </mesh>
  );
}

/** RGB image [3, r, r] in [0, 1] drawn as a textured slab. */
export function ImageBox({ rgb, res, x, y, size, outline, onPick }: { rgb: Float32Array; res: number; x: number; y: number; size: number; outline?: string; onPick?: () => void }) {
  const invalidate = useThree((s) => s.invalidate);
  const tex = useMemo(() => {
    const t = new THREE.DataTexture(new Uint8Array(res * res * 4), res, res, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.colorSpace = THREE.SRGBColorSpace;
    t.flipY = false;
    return t;
  }, [res]);
  useEffect(() => {
    const d = tex.image.data as Uint8Array;
    const N = res * res;
    // DataTexture row 0 is the bottom of the plane; write rows flipped so the image is upright.
    for (let r = 0; r < res; r++)
      for (let c = 0; c < res; c++) {
        const src = r * res + c, dst = ((res - 1 - r) * res + c) * 4;
        d[dst] = rgb[src] * 255;
        d[dst + 1] = rgb[N + src] * 255;
        d[dst + 2] = rgb[2 * N + src] * 255;
        d[dst + 3] = 255;
      }
    tex.needsUpdate = true;
    invalidate();
  }, [rgb, res, tex, invalidate]);
  return (
    <group position={[x + size / 2, y - size / 2, 0]}>
      {outline && (
        <mesh position={[0, 0, -0.6]}>
          <planeGeometry args={[size + 2.4, size + 2.4]} />
          <meshBasicMaterial color={outline} toneMapped={false} />
        </mesh>
      )}
      <mesh onClick={(e) => (e.stopPropagation(), onPick?.())}>
        <planeGeometry args={[size, size]} />
        <meshBasicMaterial map={tex} toneMapped={false} />
      </mesh>
    </group>
  );
}
