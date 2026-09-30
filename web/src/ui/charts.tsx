"use client";
// Minimal SVG charts (no dependencies): line charts, scatter, histogram-with-density.

export interface Series {
  x: (number | null)[];
  y: (number | null)[];
  color: string;
  label?: string;
  dash?: boolean;
  width?: number;
}

const W = 380, H = 150, M = { l: 44, r: 10, t: 12, b: 26 };

function ticks(lo: number, hi: number, n = 4): number[] {
  if (!(hi > lo)) return [lo];
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n)));
  const err = (hi - lo) / n / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-12; v += s) out.push(+v.toPrecision(6));
  return out;
}
const fmt = (v: number) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(0) : +v.toPrecision(3) + "");

export function LineChart({ series, title, xLabel = "epoch", log = false, height = H, marker, yRange }: {
  series: Series[];
  title?: string;
  xLabel?: string;
  log?: boolean;
  height?: number;
  marker?: number | null;
  yRange?: [number, number];
}) {
  const tf = (v: number) => (log ? Math.log10(Math.max(v, 1e-12)) : v);
  const xs: number[] = [], ys: number[] = [];
  for (const s of series)
    s.x.forEach((xv, i) => {
      const yv = s.y[i];
      if (xv !== null && yv !== null && Number.isFinite(yv) && (!log || yv > 0)) {
        xs.push(xv);
        ys.push(tf(yv));
      }
    });
  if (!xs.length) return <div className="text-[11px] text-slate-500">no data</div>;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = yRange ? tf(yRange[0]) : Math.min(...ys), y1 = yRange ? tf(yRange[1]) : Math.max(...ys);
  if (y1 - y0 < 1e-9) (y0 -= 0.5), (y1 += 0.5);
  const pad = (y1 - y0) * 0.06;
  y0 -= pad;
  y1 += pad;
  const h = height;
  const px = (v: number) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const py = (v: number) => h - M.b - ((v - y0) / (y1 - y0)) * (h - M.t - M.b);
  const yt = log ? ticks(Math.floor(y0), Math.ceil(y1), 3).filter((v) => Number.isInteger(v)) : ticks(y0, y1);
  return (
    <div>
      {title && <div className="mb-0.5 text-[11px] font-medium text-slate-300">{title}</div>}
      <svg viewBox={`0 0 ${W} ${h}`} className="w-full">
        {yt.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={py(v)} y2={py(v)} stroke="#1f2630" />
            <text x={M.l - 4} y={py(v) + 3} fontSize={9} textAnchor="end" fill="#7c8794">
              {log ? `1e${v}` : fmt(v)}
            </text>
          </g>
        ))}
        {ticks(x0, x1, 5).map((v) => (
          <text key={v} x={px(v)} y={h - M.b + 12} fontSize={9} textAnchor="middle" fill="#7c8794">
            {fmt(v)}
          </text>
        ))}
        <text x={W - M.r} y={h - 3} fontSize={9} textAnchor="end" fill="#5c6672">
          {xLabel}
        </text>
        {marker !== undefined && marker !== null && <line x1={px(marker)} x2={px(marker)} y1={M.t} y2={h - M.b} stroke="#ffffff55" strokeDasharray="3 3" />}
        {series.map((s, si) => {
          let d = "";
          s.x.forEach((xv, i) => {
            const yv = s.y[i];
            if (xv === null || yv === null || !Number.isFinite(yv) || (log && yv <= 0)) return;
            d += `${d ? "L" : "M"}${px(xv).toFixed(1)},${py(tf(yv)).toFixed(1)}`;
          });
          return <path key={si} d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.4} strokeDasharray={s.dash ? "4 3" : undefined} />;
        })}
      </svg>
      {series.some((s) => s.label) && (
        <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-400">
          {series.filter((s) => s.label).map((s, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3" style={{ background: s.color, opacity: s.dash ? 0.6 : 1 }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export interface Pt {
  x: number;
  y: number;
  color: string;
  label: string;
  hl?: boolean;
}

export function Scatter({ points, xLabel, yLabel, onPick, height = 190 }: { points: Pt[]; xLabel: string; yLabel: string; onPick?: (label: string) => void; height?: number }) {
  const ok = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!ok.length) return <div className="text-[11px] text-slate-500">no data</div>;
  let x0 = Math.min(...ok.map((p) => p.x)), x1 = Math.max(...ok.map((p) => p.x));
  let y0 = Math.min(...ok.map((p) => p.y)), y1 = Math.max(...ok.map((p) => p.y));
  const dx = (x1 - x0) * 0.08 || 0.1, dy = (y1 - y0) * 0.08 || 0.1;
  (x0 -= dx), (x1 += dx), (y0 -= dy), (y1 += dy);
  const h = height;
  const px = (v: number) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
  const py = (v: number) => h - M.b - ((v - y0) / (y1 - y0)) * (h - M.t - M.b);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} className="w-full">
      {ticks(y0, y1).map((v) => (
        <g key={v}>
          <line x1={M.l} x2={W - M.r} y1={py(v)} y2={py(v)} stroke="#1f2630" />
          <text x={M.l - 4} y={py(v) + 3} fontSize={9} textAnchor="end" fill="#7c8794">
            {fmt(v)}
          </text>
        </g>
      ))}
      {ticks(x0, x1, 5).map((v) => (
        <text key={v} x={px(v)} y={h - M.b + 12} fontSize={9} textAnchor="middle" fill="#7c8794">
          {fmt(v)}
        </text>
      ))}
      <text x={W - M.r} y={h - 3} fontSize={9} textAnchor="end" fill="#5c6672">
        {xLabel}
      </text>
      <text x={4} y={M.t - 2} fontSize={9} fill="#5c6672">
        {yLabel}
      </text>
      {ok.map((p, i) => (
        <g key={i} onClick={() => onPick?.(p.label)} className={onPick ? "cursor-pointer" : ""}>
          <circle cx={px(p.x)} cy={py(p.y)} r={p.hl ? 5.5 : 3.5} fill={p.color} fillOpacity={p.hl ? 1 : 0.75} stroke={p.hl ? "#fff" : "none"} strokeWidth={1.2}>
            <title>{`${p.label}: (${p.x.toPrecision(3)}, ${p.y.toPrecision(3)})`}</title>
          </circle>
        </g>
      ))}
    </svg>
  );
}

/** Histogram of samples with a reference density curve (e.g. N(0,1)) on the same axis. */
export function HistDensity({ samples, density, range = [-4, 4], bins = 32, height = 130 }: { samples: Float32Array | number[]; density: (x: number) => number; range?: [number, number]; bins?: number; height?: number }) {
  const [a, b] = range;
  const w = (b - a) / bins;
  const counts = new Array(bins).fill(0);
  let n = 0;
  for (const v of samples) {
    const k = Math.floor((v - a) / w);
    if (k >= 0 && k < bins) counts[k]++;
    n++;
  }
  const dens = counts.map((c) => c / (n * w));
  const ref = Array.from({ length: 121 }, (_, i) => a + ((b - a) * i) / 120);
  const ymax = Math.max(...dens, ...ref.map(density)) * 1.1;
  const h = height;
  const px = (v: number) => M.l + ((v - a) / (b - a)) * (W - M.l - M.r);
  const py = (v: number) => h - M.b - (v / ymax) * (h - M.t - M.b);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} className="w-full">
      {dens.map((d, i) => (
        <rect key={i} x={px(a + i * w) + 0.5} width={Math.max(0, px(a + (i + 1) * w) - px(a + i * w) - 1)} y={py(d)} height={h - M.b - py(d)} fill="#ef8fb7" fillOpacity={0.55} />
      ))}
      <path d={ref.map((x, i) => `${i ? "L" : "M"}${px(x).toFixed(1)},${py(density(x)).toFixed(1)}`).join("")} fill="none" stroke="#e2e8f0" strokeWidth={1.4} />
      {ticks(a, b, 8).map((v) => (
        <text key={v} x={px(v)} y={h - M.b + 12} fontSize={9} textAnchor="middle" fill="#7c8794">
          {fmt(v)}
        </text>
      ))}
      <line x1={M.l} x2={W - M.r} y1={h - M.b} y2={h - M.b} stroke="#2a323d" />
    </svg>
  );
}
