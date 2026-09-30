"""Launch a YAML list of runs as concurrent processes on one GPU (resumable: skips runs with DONE).

  python -m jepa_viz.sweep sweeps/m1.yaml --jobs 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .train import RUNS


def load_runs(path: Path) -> list[dict]:
    spec = yaml.safe_load(path.read_text())
    return [{"set": {}, "tags": [], **r} for r in spec["runs"]]


def command(r: dict) -> list[str]:
    cmd = [sys.executable, "-m", "jepa_viz.train", "--algo", r["algo"], "--name", r["name"]]
    for k, v in r["set"].items():
        cmd += ["--set", f"{k}={json.dumps(v) if not isinstance(v, str) else v}"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--meta-only", action="store_true", help="rewrite meta.json (tags) without launching")
    a = ap.parse_args()
    runs = [r for r in load_runs(a.spec) if a.only is None or r["name"] in a.only]
    todo = []
    for r in runs:
        d = RUNS / r["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({k: r[k] for k in ("name", "algo", "set", "tags")}, indent=1))
        if (d / "DONE").exists():
            print(f"skip {r['name']} (done)")
        else:
            todo.append(r)
    if a.meta_only:
        print(f"meta.json rewritten for {len(runs)} runs")
        return

    running: list[tuple[dict, subprocess.Popen, float]] = []
    failed = []
    while todo or running:
        while todo and len(running) < a.jobs:
            r = todo.pop(0)
            log = open(RUNS / r["name"] / "stdout.log", "w")
            p = subprocess.Popen(command(r), stdout=log, stderr=subprocess.STDOUT,
                                 cwd=Path(__file__).resolve().parents[1])
            running.append((r, p, time.time()))
            print(f"start {r['name']}", flush=True)
        time.sleep(2)
        for item in list(running):
            r, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
                if p.returncode != 0:
                    failed.append(r["name"])
                print(f"end {r['name']}: {status} in {time.time() - t0:.0f}s", flush=True)
    print("failed:", failed if failed else "none")


if __name__ == "__main__":
    main()
