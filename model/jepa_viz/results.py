"""Markdown tables for RESULTS.md, generated from data/runs/index.json (after `export --only runs`).

  python -m jepa_viz.results > ../RESULTS_tables.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
LABEL = {
    "teacher": "teacher (EMA τ / stop-grad)", "predictor": "predictor", "mask.pred_scale": "target block scale", "mask.npred": "target blocks",
    "mask.enc_scale": "context block scale", "pred.depth": "predictor depth", "optim.lr": "peak LR", "optim.warmup_epochs": "warmup epochs",
    "optim.schedule": "LR schedule", "optim.wd_schedule": "WD schedule", "optim.beta2": "Adam β₂", "optim.name": "optimiser", "model.depth": "encoder depth",
    "loss": "loss", "reg.lam": "λ", "slices": "SIGReg slices M", "views.n_local": "local views V_l", "views.local_scale": "local crop scale",
    "proj.out": "K", "proj.depth": "projector depth", "reg.kind": "regulariser", "reg.t_max": "EP t range",
}


def fmt_v(v) -> str:
    return f"{v[0]}–{v[1]}" if isinstance(v, list) else str(v)


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * float(x):.1f}"


def spearman(x: list[float], y: list[float]) -> float:
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) or 1
    return num / den


def main() -> None:
    idx = json.loads((DATA / "runs" / "index.json").read_text())
    runs = {r["name"]: r for r in idx["runs"]}
    out = []
    for algo, key, d in (("ijepa", "target", 64), ("lejepa", "backbone", 64)):
        out.append(f"\n### {'I-JEPA' if algo == 'ijepa' else 'LeJEPA'} (frozen {key} features; ★ = baseline)\n")
        out.append("| setting | value | run | k-NN % | linear % | RankMe | eff. dims | spread | collapse |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for axis, pts in idx["axes"][algo].items():
            for p in pts:
                f = runs[p["run"]]["final"]
                out.append(
                    f"| {LABEL.get(axis, axis)} | {fmt_v(p['value'])}{' ★' if p.get('baseline') else ''} | {p['run']} | {pct(f.get(f'{key}.knn_acc'))} | {pct(f.get(f'{key}.linear_acc'))} "
                    f"| {f.get(f'{key}.rankme')} | {f.get(f'{key}.eff_dims')} | {f.get(f'{key}.spread')} | {f.get(f'{key}.collapse')} |")
    out.append("\n### Failure-mode presets\n")
    out.append("| preset | run | k-NN % | linear % | spread | collapse |")
    out.append("|---|---|---|---|---|---|")
    for preset, names in idx["presets"].items():
        for n in names:
            r = runs[n]
            key = "target" if r["algo"] == "ijepa" else "backbone"
            f = r["final"]
            out.append(f"| {preset} | {n} | {pct(f.get(f'{key}.knn_acc'))} | {pct(f.get(f'{key}.linear_acc'))} | {f.get(f'{key}.spread')} | {f.get(f'{key}.collapse')} |")
    out.append("\n### Does the loss track probe accuracy? (Spearman ρ across runs, final val loss vs linear acc)\n")
    for algo, key in (("ijepa", "target"), ("lejepa", "backbone")):
        rs = [r for r in idx["runs"] if r["algo"] == algo and r["final"].get("val_loss") is not None]
        x = [math.log10(max(float(r["final"]["val_loss"]), 1e-12)) for r in rs]
        y = [float(r["final"][f"{key}.linear_acc"]) for r in rs]
        out.append(f"- {algo}: ρ = {spearman(x, y):.2f} over {len(rs)} runs")
        if algo == "lejepa":
            rs2 = [r for r in rs if (r["final"].get("lam") or 0) > 0]
            x2 = [float(r["final"]["val_loss"]) / float(r["final"]["lam"]) ** 0.4 for r in rs2]
            y2 = [float(r["final"][f"{key}.linear_acc"]) for r in rs2]
            out.append(f"- lejepa, loss / λ^0.4 (paper Eq. 10): ρ = {spearman(x2, y2):.2f} over {len(rs2)} runs with λ > 0")
    out.append("\n### Within a run: does val loss track linear-probe accuracy over training? (eval checkpoints after epoch 10)\n")
    out.append("| run | ρ(val loss, linear acc) |")
    out.append("|---|---|")
    per_algo: dict[str, list[float]] = {"ijepa": [], "lejepa": []}
    for r in idx["runs"]:
        m = json.loads((DATA / "runs" / r["name"] / "metrics.json").read_text())["eval"]
        key = "target" if r["algo"] == "ijepa" else "backbone"
        vk = "val_loss_pred" if r["algo"] == "ijepa" else "val_loss_total"
        pts = [(v, a) for e, v, a in zip(m["epoch"], m.get(vk, []), m.get(f"{key}.linear_acc", [])) if e is not None and e >= 10 and v is not None and a is not None]
        if len(pts) > 5:
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            per_algo[r["algo"]].append(rho)
            if r["name"] in ("ijepa-base", "lejepa-base"):
                out.append(f"| {r['name']} | {rho:.2f} |")
    for a, v in per_algo.items():
        if v:
            v = sorted(v)
            out.append(f"| median over {len(v)} {a} runs | {v[len(v) // 2]:.2f} |")
    out.append("\n### Isotropy vs probe accuracy (Spearman ρ across runs, participation ratio vs linear acc)\n")
    for algo, key in (("ijepa", "target"), ("lejepa", "backbone")):
        rs = [r for r in idx["runs"] if r["algo"] == algo]
        x = [float(r["final"][f"{key}.iso_pr"]) for r in rs]
        y = [float(r["final"][f"{key}.linear_acc"]) for r in rs]
        out.append(f"- {algo}: ρ = {spearman(x, y):.2f} over {len(rs)} runs")
    print("\n".join(out))


if __name__ == "__main__":
    main()
