"""Train one run and log everything the explorer needs.

  python -m jepa_viz.train --algo ijepa --name ijepa-base
  python -m jepa_viz.train --algo lejepa --name lejepa-lambda0 --set reg.lam=0

Writes model/runs/<name>/: config.json, metrics.jsonl (train + eval records), pca_feats.pt,
ckpt/*.pt (weights), summary.json, and a DONE marker.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, build, to_dict
from .data import load_data
from .diagnostics import collapse_verdict, geometry
from .evaluate import extract, knn_accuracy, linear_probe
from .ijepa import IJEPA
from .lejepa import LeJEPA
from .schedules import lr_at, tau_at, wd_at

RUNS = Path(__file__).resolve().parents[1] / "runs"
N_TRAIN = 50_000


def make_model(cfg: Config) -> torch.nn.Module:
    return IJEPA(cfg) if cfg.algo == "ijepa" else LeJEPA(cfg)


def make_optimizer(model: torch.nn.Module, o) -> torch.optim.Optimizer:
    groups = []
    for gname, params in model.param_groups().items():
        # No decay on biases / LayerNorm / BatchNorm (all 1-D); the 3-D mask token IS decayed (official).
        decay = [p for p in params if p.requires_grad and p.ndim > 1]
        decay_ids = {id(p) for p in decay}
        no_decay = [p for p in params if p.requires_grad and id(p) not in decay_ids]
        groups.append({"params": decay, "decay": True, "gname": gname})
        groups.append({"params": no_decay, "decay": False, "gname": gname, "weight_decay": 0.0})
    if o.name == "adamw":
        return torch.optim.AdamW(groups, lr=o.lr, betas=(o.beta1, o.beta2), weight_decay=o.wd)
    if o.name == "adam":  # L2-coupled weight decay
        return torch.optim.Adam(groups, lr=o.lr, betas=(o.beta1, o.beta2), weight_decay=o.wd)
    if o.name == "sgd":
        return torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, weight_decay=o.wd, nesterov=True)
    raise ValueError(o.name)


def logspace_steps(total: int, n: int, first: int = 25) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [total]
    pts = np.geomspace(first, max(total, first + 1), n - 1)
    return sorted({0, total} | {int(round(p)) for p in pts if p <= total})


def eval_steps(total: int, spe: int, every_epochs: float) -> list[int]:
    early = {0, 25, 50, 100, 200, 400}
    interval = max(1, int(round(every_epochs * spe)))
    regular = set(range(0, total + 1, interval))
    return sorted({s for s in early | regular if s <= total} | {total})


def grad_norm(params) -> float:
    gs = [p.grad.detach().norm() for p in params if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v.tolist()
        out[k] = v
    return out


def run(cfg: Config, device: str = "cuda", max_steps: int | None = None, do_eval: bool = True) -> dict:
    run_dir = RUNS / cfg.name
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=1))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = load_data(device)
    model = make_model(cfg).to(device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.system.precision == "bf16")
    o = cfg.optim
    opt = make_optimizer(model, o)
    B = o.batch_size
    spe = N_TRAIN // B
    total = int(round(o.epochs * spe))
    if max_steps is not None:
        total = min(total, max_steps)
    warmup = int(round(o.warmup_epochs * spe))

    ev_steps = set(eval_steps(total, spe, cfg.log.eval_every_epochs)) if do_eval else set()
    pca_steps = set(logspace_steps(total, cfg.log.n_pca_ckpts)) if do_eval else set()
    w_steps = set(logspace_steps(total, cfg.log.n_weight_ckpts))

    val_x, _ = data.subset("val")
    ptr_x, ptr_y = data.subset("probe_train")
    pte_x, pte_y = data.subset("eval_test")
    pca_x, _ = data.subset("pca")
    groups = model.param_groups()

    log_f = open(run_dir / "metrics.jsonl", "w")

    def write(rec: dict) -> None:
        log_f.write(json.dumps(_jsonable(rec)) + "\n")
        log_f.flush()

    def evaluate_now(step: int) -> dict:
        model.eval()
        rec: dict = {"kind": "eval", "step": step, "epoch": step / spe}
        vt: dict[str, list[float]] = {}
        n_chunks = val_x.shape[0] // B
        for c in range(n_chunks):
            for k, v in model.val_loss(val_x[c * B : (c + 1) * B], c, data.normalize).items():
                vt.setdefault(k, []).append(float(v))
        rec.update({f"val_{k}": float(np.mean(v)) for k, v in vt.items()})
        fv = extract(model, val_x, data.prep)
        ftr = extract(model, ptr_x, data.prep)
        fte = extract(model, pte_x, data.prep)
        for key in fv:
            g = geometry(fv[key], prefix=f"{key}.")
            rec.update(g)
            rec[f"{key}.collapse"] = collapse_verdict(g, fv[key].shape[1], prefix=f"{key}.")
            rec[f"{key}.knn_acc"] = knn_accuracy(ftr[key], ptr_y, fte[key], pte_y)
            rec.update({f"{key}.{k}": v for k, v in linear_probe(ftr[key], ptr_y, fte[key], pte_y).items()})
        if cfg.algo == "ijepa":
            rec["ema_gap"] = model.ema_gap()
        model.train()
        write(rec)
        return rec

    pca_store: dict[str, list[torch.Tensor]] = {}
    pca_done: list[int] = []
    last_eval: dict = {}
    diverged = False
    t_start = time.time()
    perm = torch.randperm(N_TRAIN, device=device)
    model.train()
    step = 0
    for step in range(total + 1):
        if step in ev_steps:
            last_eval = evaluate_now(step)
        if step in pca_steps:
            f = extract(model, pca_x, data.prep)
            for k, v in f.items():
                pca_store.setdefault(k, []).append(v.half().cpu())
            pca_done.append(step)
        if step in w_steps:
            torch.save(model.state_dict(), run_dir / "ckpt" / f"step{step:07d}.pt")
        if step == total:
            break

        i = step % spe
        if i == 0:
            perm = torch.randperm(N_TRAIN, device=device)
        xb = data.train_x[perm[i * B : (i + 1) * B]]
        lr = lr_at(step, total, warmup, o.lr, o.start_lr_ratio, o.min_lr_ratio, o.schedule)
        wd = wd_at(step, total, o.wd, o.wd_end, o.wd_schedule)
        for g in opt.param_groups:
            g["lr"] = lr
            g["weight_decay"] = wd if g["decay"] else 0.0

        with amp:
            loss, terms = model.train_loss(xb, step, data.normalize)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        log_now = step % cfg.log.every == 0
        if log_now:
            gn = {f"grad_{k}": grad_norm(v) for k, v in groups.items()}
        opt.step()
        tau = tau_at(step, total, cfg.ema.tau0, cfg.ema.tau1)
        model.after_step(tau)
        if log_now:
            lv = float(loss)
            rec = {"kind": "train", "step": step, "epoch": step / spe, "lr": lr, "wd": wd, "tau": tau,
                   "time": time.time() - t_start, **gn, **{k: float(v) for k, v in terms.items()}}
            write(rec)
            if not math.isfinite(lv):
                diverged = True
                break

    wall = time.time() - t_start
    if pca_store:
        torch.save({"steps": pca_done, "feats": {k: torch.stack(v) for k, v in pca_store.items()}},
                   run_dir / "pca_feats.pt")
    torch.save(model.state_dict(), run_dir / "ckpt" / "final.pt")
    summary = {"name": cfg.name, "algo": cfg.algo, "steps": step, "total_steps": total,
               "steps_per_epoch": spe, "wall_s": wall, "diverged": diverged,
               "final_eval": {k: v for k, v in last_eval.items() if not isinstance(v, list)}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "DONE").write_text("ok\n")
    log_f.close()
    return summary


def parse_sets(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ijepa", "lejepa"])
    ap.add_argument("--name", default="")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. reg.lam=0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    cfg = build(a.algo, parse_sets(a.set), a.name)
    s = run(cfg, a.device, a.max_steps, not a.no_eval)
    print(json.dumps({k: v for k, v in s.items() if k != "final_eval"}))


if __name__ == "__main__":
    main()
