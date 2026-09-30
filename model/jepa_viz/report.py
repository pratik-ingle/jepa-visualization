"""Markdown summary of finished runs (for milestone reviews).

  python -m jepa_viz.report
"""
from __future__ import annotations

import json

from .diagnostics import spread_and_verdict
from .train import RUNS


def final_eval(run_dir) -> dict | None:
    if not (run_dir / "DONE").exists():
        return None
    ev = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if '"eval"' in l]
    return ev[-1] if ev else None


def compare(other: str) -> None:
    """Side-by-side final kNN / linear / RankMe of runs/ vs another runs directory (e.g. fp32 archive)."""
    base = RUNS.parent / other
    print(f"| run | kNN ({other} -> runs) | linear | RankMe |")
    print("|---|---|---|---|")
    for run in sorted(RUNS.iterdir()):
        a, b = final_eval(base / run.name), final_eval(run)
        if not (run.is_dir() and a and b):
            continue
        k = "target" if "target.knn_acc" in b else "backbone"
        f = lambda m: f"{a[f'{k}.{m}']:.3f} -> {b[f'{k}.{m}']:.3f}"
        print(f"| {run.name} | {f('knn_acc')} | {f('linear_acc')} | {a[f'{k}.rankme']:.1f} -> {b[f'{k}.rankme']:.1f} |")


def main() -> None:
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        return compare(sys.argv[2])
    rows = []
    for run in sorted(RUNS.iterdir()):
        if run.name.startswith("_") or not (run / "DONE").exists():
            continue
        s = json.loads((run / "summary.json").read_text())
        cfg = json.loads((run / "config.json").read_text())
        ev = [json.loads(l) for l in open(run / "metrics.jsonl")]
        ev = [r for r in ev if r["kind"] == "eval"]
        key = "target" if s["algo"] == "ijepa" else "backbone"
        d = cfg["model"]["dim"]
        f = ev[-1]
        best_knn = max(r[f"{key}.knn_acc"] for r in ev)
        spread, eff, verdict = spread_and_verdict(f, key, d)
        min_spread = min(spread_and_verdict(r, key, d)[0] for r in ev)
        z = ""
        if s["algo"] == "lejepa":
            zs, ze, zv = spread_and_verdict(f, "z", cfg["proj"]["out"])
            z = f" | z: RankMe {f['z.rankme']:.1f}/{cfg['proj']['out']}, eff {ze:.1f}, spread {zs:.2f}, {zv}"
        val = f.get("val_loss_pred", f.get("val_loss_total"))
        rows.append(
            f"| {s['name']} | {f[f'{key}.knn_acc']:.3f} ({best_knn:.3f}) | {f[f'{key}.linear_acc']:.3f} "
            f"| {f[f'{key}.rankme']:.1f} | {eff:.1f} | {spread:.3g} (min {min_spread:.2g}) | {verdict} | {val:.4g} "
            f"| {'DIVERGED ' if s['diverged'] else ''}{s['wall_s'] / 60:.0f} min{z} |")
    print("| run | kNN (best) | linear | RankMe/64 | eff. dims | spread (min) | collapse | val loss | wall / z |")
    print("|---|---|---|---|---|---|---|---|---|")
    base = RUNS / "ijepa-base" / "metrics.jsonl"
    if base.exists():  # step-0 eval = the randomly initialised encoder (same seed for every run)
        r0 = next(json.loads(l) for l in open(base) if '"eval"' in l)
        sp, eff, _ = spread_and_verdict(r0, "target", 64)
        print(f"| *random init (step 0)* | {r0['target.knn_acc']:.3f} | {r0['target.linear_acc']:.3f} "
              f"| {r0['target.rankme']:.1f} | {eff:.1f} | {sp:.3g} | — | — | — |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
