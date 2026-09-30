"use client";
import { OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { memo, useCallback, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { BASE } from "@/lib/data";
import { store, useStore } from "@/lib/store";
import { focusBox, type Layout } from "./layout";
import { BlockSpec, ImageBox, TensorBox } from "./TensorBox";

// DejaVu Sans subset renamed "JEPA Viz Sans" (Latin + Greek + math; licence in /fonts/LICENSE-dejavu.txt)
const FONT = `${BASE}/fonts/viz-400.woff`;
const FONT_B = `${BASE}/fonts/viz-600.woff`;
const BG = "#0b0e13";
const FOV = 40;

/** Mix two hex colours in sRGB (THREE.Color.lerp works in linear space, which reads far brighter here). */
function mixHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

export interface Insets {
  left: number;
  right: number;
  bottom: number;
}

/** Camera pose that fits a box into the part of the canvas not covered by panels. */
function fit(box: { x0: number; x1: number; y0: number; y1: number }, ins: Insets, pad = 1.08) {
  const W = window.innerWidth, H = window.innerHeight - 44; // canvas sits below the 44 px top bar
  const tanH = Math.tan(((FOV / 2) * Math.PI) / 180);
  const freeW = Math.max(200, W - ins.left - ins.right), freeH = Math.max(160, H - ins.bottom);
  const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
  // world half-height of the whole canvas such that the box fits the free rectangle
  const halfH = Math.max((bh / 2) * (H / freeH), ((bw / 2) * (W / freeW)) / (W / H)) * pad;
  const dist = halfH / tanH;
  const worldPerPx = (2 * halfH) / H;
  const cx = (box.x0 + box.x1) / 2 - ((ins.left - ins.right) / 2) * worldPerPx;
  const cy = (box.y0 + box.y1) / 2 - (ins.bottom / 2) * worldPerPx;
  return { pos: new THREE.Vector3(cx, cy, dist), target: new THREE.Vector3(cx, cy, 0) };
}

/** Exposes per-frame renderer stats (draw calls, triangles) for performance checks. */
function RenderStats() {
  useFrame(({ gl }) => {
    const w = window as unknown as { __jepaRender?: { calls: number; triangles: number; frames: number } };
    w.__jepaRender = { calls: gl.info.render.calls, triangles: gl.info.render.triangles, frames: (w.__jepaRender?.frames ?? 0) + 1 };
  });
  return null;
}

/** Flies the camera to the frames named by store.flyTo, or re-frames everything on reset. */
function CameraRig({ layout, insets }: { layout: Layout; insets: Insets }) {
  const fly = useStore((s) => s.flyTo);
  const reset = useStore((s) => s.resetView);
  const { camera, controls, invalidate } = useThree();
  const anim = useRef<{ t0: number; from: THREE.Vector3; to: THREE.Vector3; fromT: THREE.Vector3; toT: THREE.Vector3 } | null>(null);
  const start = useCallback(
    (box: { x0: number; x1: number; y0: number; y1: number } | null) => {
      if (!box || !controls) return;
      const c = controls as unknown as { target: THREE.Vector3 };
      const { pos, target } = fit(box, insets);
      anim.current = { t0: performance.now(), from: camera.position.clone(), to: pos, fromT: c.target.clone(), toT: target };
      invalidate();
    },
    [camera, controls, insets, invalidate],
  );
  useEffect(() => {
    if (fly) start(fly.id === "*" ? layout.bounds : focusBox(layout, fly.id.split(",")));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fly?.n, controls]);
  useEffect(() => {
    if (reset) start(layout.bounds);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reset]);
  useFrame(() => {
    const a = anim.current;
    if (!a || !controls) return;
    const c = controls as unknown as { target: THREE.Vector3; update: () => void };
    const u = Math.min(1, (performance.now() - a.t0) / 950);
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; // easeInOutCubic
    camera.position.lerpVectors(a.from, a.to, e);
    c.target.lerpVectors(a.fromT, a.toT, e);
    c.update();
    if (u >= 1) anim.current = null;
    else invalidate();
  });
  return null;
}

const titleSize = (w: number) => Math.min(18, Math.max(8, w * 0.028));

const Frames = memo(function Frames({ frames }: { frames: Layout["frames"] }) {
  return (
    <>
      {frames.map((f) => (
        <group key={f.id}>
          {/* Frames are pure backdrop: drawn first, no depth writes, so they can never z-fight the data. */}
          <mesh position={[f.x + f.w / 2, f.y - f.h / 2, -2]} renderOrder={-2}>
            <planeGeometry args={[f.w + 1.6, f.h + 1.6]} />
            <meshBasicMaterial color={mixHex(f.color, BG, 0.6)} toneMapped={false} depthWrite={false} />
          </mesh>
          <mesh position={[f.x + f.w / 2, f.y - f.h / 2, -2]} renderOrder={-1}>
            <planeGeometry args={[f.w, f.h]} />
            <meshBasicMaterial color={mixHex(f.color, BG, 0.93)} toneMapped={false} depthWrite={false} />
          </mesh>
          {/* Title size follows the frame so close-ups of small groups are not dominated by their title. */}
          <Text font={FONT_B} fontSize={titleSize(f.w)} color={f.color} anchorX="left" anchorY="bottom" position={[f.x, f.y + titleSize(f.w) * 0.35 + 9, 0]} maxWidth={f.w * 1.6}>
            {f.title}
          </Text>
          {f.subtitle && (
            <Text font={FONT} fontSize={Math.max(4, titleSize(f.w) * 0.38)} color="#8b95a3" anchorX="left" anchorY="bottom" position={[f.x, f.y + 4, 0]} maxWidth={f.w * 1.8}>
              {f.subtitle}
            </Text>
          )}
        </group>
      ))}
    </>
  );
});

const Labels = memo(function Labels({ labels }: { labels: Layout["labels"] }) {
  return (
    <>
      {labels.map((l, i) => (
        <Text key={i} font={FONT} fontSize={l.size} color={l.color ?? "#c3cad3"} anchorX="left" anchorY="bottom" position={[l.x, l.y, 0.5]}>
          {l.text}
        </Text>
      ))}
    </>
  );
});

/** PCA-3D embedding cloud: real points in a wireframe cube (orbit with right-drag to see depth). */
function CloudPoints({ c }: { c: Layout["clouds"][number] }) {
  const invalidate = useThree((s) => s.invalidate);
  const geom = useMemo(() => new THREE.BufferGeometry(), []);
  useEffect(() => {
    geom.setAttribute("position", new THREE.BufferAttribute(c.pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(c.colors, 3));
    geom.computeBoundingSphere();
    invalidate();
  }, [c.pos, c.colors, geom, invalidate]);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(c.size, c.size, c.size)), [c.size]);
  return (
    <group position={[c.cx, c.cy, 0]}>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color="#3a4452" />
      </lineSegments>
      <points geometry={geom}>
        <pointsMaterial size={Math.max(2.5, c.size / 90)} vertexColors sizeAttenuation toneMapped={false} />
      </points>
    </group>
  );
}

function Blocks({ blocks }: { blocks: BlockSpec[] }) {
  const trace = useStore((s) => s.trace);
  const reveal = useStore((s) => s.reveal);
  // Backward animation: gradient blocks appear from the loss back toward the input.
  const maxOrder = useMemo(() => {
    const m: Record<string, number> = {};
    for (const b of blocks) if (b.grad) m[b.group] = Math.max(m[b.group] ?? 0, b.order ?? 0);
    return m;
  }, [blocks]);
  const hidden = (b: BlockSpec) => !!b.grad && reveal < 1 && (b.order ?? 0) < (1 - reveal) * (maxOrder[b.group] ?? 0);
  const onHover = useCallback((spec: BlockSpec | null, r: number, c: number) => {
    if (!spec) return store.set({ hover: null });
    const v = spec.data[r * spec.cols + c];
    store.set({
      hover: {
        label: spec.label, atom: spec.atom, shape: `[${spec.rows}×${spec.cols}]`, row: r, col: c, value: v,
        rowName: spec.rowName?.(r) ?? `row ${r}`, colName: spec.colName?.(c) ?? `col ${c}`, note: spec.note,
      },
    });
  }, []);
  const onPick = useCallback((spec: BlockSpec, r: number) => {
    const tok = spec.rowTokens?.[r];
    if (tok === undefined || tok < 0) return store.set({ trace: null });
    const cur = store.get().trace;
    const space = spec.space ?? "token";
    const same = cur && cur.group === spec.group && cur.space === space && cur.token === tok;
    store.set({ trace: same ? null : { group: spec.group, space, token: tok } });
  }, []);
  return (
    <>
      {blocks.map((b) => {
        const on = trace && trace.group === b.group && trace.space === (b.space ?? "token");
        const rowHL = on && b.rowTokens ? b.rowTokens.indexOf(trace!.token) : -1;
        const colHL = b.colHL ?? (on && b.colTokens ? b.colTokens.indexOf(trace!.token) : -1);
        return <TensorBox key={b.id} spec={b} rowHL={rowHL} colHL={colHL} dim={!!b.noGrad || hidden(b)} onHover={onHover} onPick={onPick} />;
      })}
    </>
  );
}

export default function Scene({ layout, insets }: { layout: Layout; insets: Insets }) {
  const init = useMemo(() => fit(layout.bounds, insets), []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 2]}
      camera={{ fov: FOV, position: init.pos.toArray() as [number, number, number], near: 10, far: 60000 }}
      gl={{ antialias: true }}
      onPointerMissed={() => store.set({ trace: null, hover: null })}
      style={{ background: BG, touchAction: "none" }}
    >
      <OrbitControls
        makeDefault
        target={init.target.toArray() as [number, number, number]}
        enableDamping={false}
        zoomToCursor
        screenSpacePanning
        minDistance={40}
        maxDistance={20000}
        mouseButtons={{ LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }}
        touches={{ ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE }}
      />
      <CameraRig layout={layout} insets={insets} />
      <RenderStats />
      <Frames frames={layout.frames} />
      <Blocks blocks={layout.blocks} />
      {layout.clouds.map((c) => (
        <CloudPoints key={c.id} c={c} />
      ))}
      {layout.images.map((im) => (
        <ImageBox
          key={im.id}
          rgb={im.rgb}
          res={im.res}
          x={im.x}
          y={im.y}
          size={im.size}
          outline={im.outline}
          onPick={im.view !== undefined ? () => store.set({ viewIdx: im.view! }) : undefined}
        />
      ))}
      <Labels labels={layout.labels} />
    </Canvas>
  );
}
