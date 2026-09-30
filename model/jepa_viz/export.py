"""Export everything the web app needs into /data (ATOMS contract).

  python -m jepa_viz.export                       # dataset, models, goldens, runs
  python -m jepa_viz.export --only runs

Layout:
  data/dataset.json                  classes, normalisation, labels of the browser image sets
  data/splits.json                   all split indices (not needed by the browser)
  data/images/showcase.png           16 images, 4x4 atlas of 32x32 (class round-robin)
  data/images/batch.png              256 images, 16x16 atlas (browser batch for SIGReg / PCA)
  data/models/<name>/model.json      architecture + tensor manifest + checkpoint list
  data/models/<name>/final.bin       fp16 weights (PyTorch layout: Linear weight is [out, in])
  data/models/<name>/step<k>.bin     fp16 weights at logged checkpoints (same manifest)
  data/golden/<name>/golden.json     manifest of recorded tensors (+ metadata, grad norms)
  data/golden/<name>/golden.bin      fp32 / int32 recorded tensors (TEST ONLY, not shipped)
  data/runs/index.json               one entry per run: tags, overrides, summary numbers
  data/runs/<name>/metrics.json      columnar train + eval logs (4 significant figures)
  data/runs/<name>/pca.bin           fp16 [ckpt, 512, 3] per feature key: "<key>" per-checkpoint PCA,
                                     Procrustes-aligned; "<key>.fixed" final-checkpoint basis
  data/runs/<name>/pca.json          manifest + steps + explained-variance ratios
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .augment import render_views, sample_view_params
from .config import from_dict
from .diagnostics import spread_and_verdict
from .data import CLASSES, MEAN, STD, get_splits, load_cifar_numpy
from .ijepa import IJEPA
from .lejepa import LeJEPA
from .masks import MultiBlockMaskSampler
from .sigreg import t_grid
from .train import RUNS, make_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BASELINES = ["ijepa-base", "lejepa-base"]
GOLDEN_IMAGES = 3  # first 3 showcase images


# ---------------------------------------------------------------------------------------------
# binary bundles
# ---------------------------------------------------------------------------------------------
class Bundle:
    """Concatenated little-endian arrays + a manifest {name: {dtype, shape, offset}} (offset in bytes)."""

    DT = {"f16": np.float16, "f32": np.float32, "i32": np.int32, "u8": np.uint8}

    def __init__(self):
        self.parts: list[bytes] = []
        self.manifest: dict[str, dict] = {}
        self.offset = 0

    def add(self, name: str, arr, dtype: str) -> None:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        a = np.ascontiguousarray(np.asarray(arr), dtype=self.DT[dtype])
        pad = (-self.offset) % 8  # keep every array 8-byte aligned for typed-array views
        if pad:
            self.parts.append(b"\0" * pad)
            self.offset += pad
        self.manifest[name] = {"dtype": dtype, "shape": list(a.shape), "offset": self.offset}
        b = a.astype(a.dtype.newbyteorder("<")).tobytes()
        self.parts.append(b)
        self.offset += len(b)

    def write(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(self.parts))
        return self.offset


def sig(x: float, n: int = 4):
    if x is None or not isinstance(x, (int, float)) or isinstance(x, bool):
        return x
    if not math.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def sig_list(v):
    return [sig_list(x) if isinstance(x, list) else sig(x) for x in v]


# ---------------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------------
def atlas(imgs: np.ndarray, cols: int) -> Image.Image:
    n = imgs.shape[0]
    rows = -(-n // cols)
    out = np.zeros((rows * 32, cols * 32, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = im.transpose(1, 2, 0)
    return Image.fromarray(out)


def export_dataset() -> None:
    x_tr, y_tr, x_te, y_te = load_cifar_numpy()
    s = get_splits(y_tr, y_te)
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    atlas(x_te[s["showcase"]], 4).save(DATA / "images" / "showcase.png", optimize=True)
    atlas(x_te[s["batch"]], 16).save(DATA / "images" / "batch.png", optimize=True)
    ds = {
        "name": "CIFAR-10",
        "classes": CLASSES,
        "mean": MEAN,
        "std": STD,
        "image_size": 32,
        "sizes": {k: int(len(v)) for k, v in s.items()},
        "labels": {k: y_te[s[k]].tolist() for k in ("pca", "batch", "showcase")},
        "atlases": {"showcase": {"file": "images/showcase.png", "cols": 4, "n": 16},
                    "batch": {"file": "images/batch.png", "cols": 16, "n": 256}},
    }
    (DATA / "dataset.json").write_text(json.dumps(ds))
    (DATA / "splits.json").write_text(json.dumps({
        "note": "indices: probe_train into CIFAR-10 train; all others into CIFAR-10 test (canonical order)",
        **{k: v.tolist() for k, v in s.items()}}))
    print("dataset: ok")


# Files the web app fetches before first render (everything else is lazy).
INITIAL_LOAD = ["dataset.json", "images/showcase.png", "runs/index.json",
                *[f"models/{n}/{f}" for n in BASELINES for f in ("model.json", "final.bin")]]


def payload_report() -> int:
    total = 0
    for f in INITIAL_LOAD:
        n = (DATA / f).stat().st_size
        total += n
        print(f"  {n / 1e3:8.1f} KB  {f}")
    print(f"  {total / 1e6:8.2f} MB  initial data payload (excludes the JS bundle; budget 5 MB total)")
    lazy = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file() and "golden" not in p.parts) - total
    print(f"  {lazy / 1e6:8.2f} MB  lazily loaded (checkpoints, run logs, PCA clouds, batch atlas)")
    return total


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
def load_run_model(name: str, ckpt: str = "final.pt") -> torch.nn.Module:
    run = RUNS / name
    cfg = from_dict(json.loads((run / "config.json").read_text()))
    m = make_model(cfg)
    m.load_state_dict(torch.load(run / "ckpt" / ckpt, map_location="cpu", weights_only=True))
    return m.eval()


def round_fp16_(m: torch.nn.Module) -> None:
    """Round exactly the exported tensors (state_dict) to fp16. Non-persistent buffers such as the
    predictor's fixed sin-cos table are not shipped (the browser recomputes them) and stay exact."""
    with torch.no_grad():
        for t in weight_tensors(m).values():
            t.copy_(t.half().float())


def weight_tensors(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    sd = m.state_dict()
    return {k: v for k, v in sd.items() if v.dtype == torch.float32}


def export_model(name: str) -> None:
    run = RUNS / name
    cfg = json.loads((run / "config.json").read_text())
    out = DATA / "models" / name
    ckpts = sorted(p.name for p in (run / "ckpt").glob("step*.pt"))
    files = []
    manifest = None
    for ck in ckpts + ["final.pt"]:
        m = load_run_model(name, ck)
        b = Bundle()
        for k, v in weight_tensors(m).items():
            b.add(k, v, "f16")
        fname = ck.replace(".pt", ".bin")
        size = b.write(out / fname)
        manifest = manifest or b.manifest
        assert b.manifest == manifest, "checkpoint layouts differ"
        step = int(ck[4:11]) if ck.startswith("step") else None
        files.append({"file": fname, "step": step, "bytes": size})
    summary = json.loads((run / "summary.json").read_text())
    for f in files:
        if f["step"] is None:
            f["step"] = summary["steps"]
    model_json = {
        "name": name,
        "algo": cfg["algo"],
        "config": {k: cfg[k] for k in ("model", "mask", "pred", "ema", "views", "proj", "reg", "loss")},
        "layout": {
            "linear": "weight [out, in]; y = x @ weight.T + bias",
            "qkv": "qkv.weight [3W, W]: rows [q | k | v], each split into heads of d_h",
            "patchify": "token = row*G + col; value = c*P*P + py*P + px",
            "pos_embed": "fixed 2D sin-cos, first half = column, second half = row; coords (i+0.5)*G/g-0.5",
            "layernorm_eps": 1e-6,
            "target_layernorm_eps": 1e-5,
            "gelu": "exact (erf)",
        },
        "tensors": manifest,
        "checkpoints": files,
        "steps_per_epoch": summary["steps_per_epoch"],
    }
    (out / "model.json").write_text(json.dumps(model_json, indent=1))
    print(f"model {name}: {len(files)} checkpoints, {files[-1]['bytes'] / 1e6:.2f} MB each")


# ---------------------------------------------------------------------------------------------
# goldens
# ---------------------------------------------------------------------------------------------
def _golden_images() -> tuple[torch.Tensor, list[int]]:
    x_tr, y_tr, x_te, y_te = load_cifar_numpy()
    s = get_splits(y_tr, y_te)
    idx = s["showcase"][:GOLDEN_IMAGES]
    return torch.as_tensor(x_te[idx]), idx.tolist()


def _prep(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def export_golden(name: str) -> None:
    m = load_run_model(name)
    round_fp16_(m)  # goldens are computed with exactly the weights the browser receives
    x_u8, img_idx = _golden_images()
    B = x_u8.shape[0]
    b = Bundle()
    b.add("input.images", x_u8, "u8")
    meta: dict = {"name": name, "algo": m.cfg.algo, "images": img_idx, "bn_mode": "eval"}
    rec: dict = {}

    if isinstance(m, IJEPA):
        x = _prep(x_u8.float() / 255.0).requires_grad_(False)
        masks = MultiBlockMaskSampler.from_cfg(m.cfg.mask, seed=0)(B, 0)
        b.add("input.ctx_idx", masks.ctx_idx, "i32")
        b.add("input.tgt_idx", masks.tgt_idx, "i32")
        meta["mask"] = {"tgt_hw": masks.tgt_hw, "ctx_hw": masks.ctx_hw, "ctx_full": masks.ctx_full.tolist()}
        b.add("input.x", x, "f32")
        loss, _ = m.loss_on(x, masks, rec=rec)
    else:
        gen = torch.Generator().manual_seed(0)
        vc = m.cfg.views
        pg = sample_view_params(vc.n_global * B, vc.global_scale, gen, "cpu", color=vc.color)
        pl = sample_view_params(vc.n_local * B, vc.local_scale, gen, "cpu", color=vc.color)
        for tag, p in (("global", pg), ("local", pl)):
            for k, v in p.items():
                b.add(f"input.{tag}.{k}", v.float() if v.dtype == torch.bool else v, "f32")
        g01 = render_views(x_u8.repeat(vc.n_global, 1, 1, 1), pg, vc.global_res)
        l01 = render_views(x_u8.repeat(vc.n_local, 1, 1, 1), pl, vc.local_res)
        b.add("input.global01", g01, "f32")
        b.add("input.local01", l01, "f32")
        loss, _ = m.loss_on(_prep(g01), _prep(l01), B, step=0, rec=rec)
        meta["sigreg"] = {"seed": 0, "t": t_grid(m.cfg.reg.t_max, m.cfg.reg.t_points).tolist(),
                          "lam": m.cfg.reg.lam, "note": "directions = sample_directions(K, M, seed=0); "
                          "browser can't reproduce torch RNG, so A is stored as sigreg.A"}

    # gradients: per-parameter norms and per-activation gradient norms over the last axis
    for t in rec.values():
        if t.requires_grad:
            t.retain_grad()
    m.zero_grad(set_to_none=True)
    loss.backward()
    meta["param_grad_norm"] = {n: (float(p.grad.norm()) if p.grad is not None else None)
                               for n, p in m.named_parameters()}
    if isinstance(m, IJEPA) and m.target_encoder is not None:
        meta["target_param_grad_norm"] = {n: (float(p.grad.norm()) if p.grad is not None else None)
                                          for n, p in m.target_encoder.named_parameters()}
    for k, t in rec.items():
        b.add(k, t.detach().float(), "f32")
        if t.grad is not None and t.dim() >= 2:
            b.add("grad." + k, t.grad.float().norm(dim=-1), "f32")
    meta["loss"] = float(loss)
    out = DATA / "golden" / name
    size = b.write(out / "golden.bin")
    meta["tensors"] = b.manifest
    (out / "golden.json").write_text(json.dumps(meta))
    print(f"golden {name}: {len(b.manifest)} tensors, {size / 1e6:.2f} MB, loss={float(loss):.5f}")


# ---------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------
def columnar(records: list[dict]) -> dict:
    keys: list[str] = []
    for r in records:
        for k in r:
            if k not in keys and k != "kind":
                keys.append(k)
    cols: dict = {}
    for k in keys:
        vals = [r.get(k) for r in records]
        if any(isinstance(v, list) for v in vals):
            cols[k] = [sig_list(v) if isinstance(v, list) else None for v in vals]
        elif any(isinstance(v, str) for v in vals):
            cols[k] = vals
        else:
            cols[k] = [sig(v) for v in vals]
    return cols


def pca3_aligned(feats: torch.Tensor) -> tuple[np.ndarray, list[list[float]]]:
    """feats [ckpt, n, d] -> [ckpt, n, 3] PCA projections, Procrustes-aligned backwards from the
    final checkpoint (rotation/reflection only: absolute scale kept), and explained-variance ratios."""
    F = feats.double()
    P, evr = [], []
    for f in F:
        fc = f - f.mean(0)
        U, S, Vt = torch.linalg.svd(fc, full_matrices=False)
        k = min(3, Vt.shape[0])
        p = fc @ Vt[:k].T
        if k < 3:
            p = torch.cat([p, torch.zeros(p.shape[0], 3 - k, dtype=p.dtype)], 1)
        tot = (S**2).sum().clamp_min(1e-30)
        evr.append([float(x) for x in (S[:3] ** 2 / tot)])
        P.append(p)
    for i in range(len(P) - 2, -1, -1):
        M = P[i].T @ P[i + 1]
        U, _, Vt = torch.linalg.svd(M)
        P[i] = P[i] @ (U @ Vt)
    return torch.stack(P).numpy(), evr


def pca3_fixed(feats: torch.Tensor) -> np.ndarray:
    """Every checkpoint projected onto the FINAL checkpoint's top-3 PCA basis (each checkpoint
    centred on its own mean). Smoother to scrub than per-checkpoint PCA, but less faithful to the
    shape of early clouds, whose variance may lie outside that basis."""
    F = feats.double()
    Fc = F - F.mean(1, keepdim=True)
    _, _, Vt = torch.linalg.svd(Fc[-1], full_matrices=False)
    k = min(3, Vt.shape[0])
    return (Fc @ Vt[:k].T).numpy()


def summarize(eval_rec: dict, cfg: dict) -> dict:
    """Final-eval numbers for the run index. `collapse` is the post-hoc verdict (spread / RankMe),
    not the heuristic logged during training."""
    algo = cfg["algo"]
    keys = [("target", cfg["model"]["dim"])] if algo == "ijepa" else [("backbone", cfg["model"]["dim"]), ("z", cfg["proj"]["out"])]
    pick = ["knn_acc", "linear_acc", "rankme", "iso_pr", "iso_minmax", "cos_mean", "var_mean", "ep_std"]
    out: dict = {}
    for key, d in keys:
        out.update({f"{key}.{p}": sig(eval_rec.get(f"{key}.{p}")) for p in pick})
        if eval_rec:
            spread, eff, verdict = spread_and_verdict(eval_rec, key, d)
            out.update({f"{key}.spread": sig(spread), f"{key}.eff_dims": sig(eff), f"{key}.collapse": verdict})
    return out


def get_dotted(cfg: dict, key: str):
    v = cfg
    for part in key.split("."):
        v = v[part]
    return list(v) if isinstance(v, tuple) else v


BASE_RUN = {"ijepa": "ijepa-base", "lejepa": "lejepa-base"}


def explorer_maps(index: list[dict], configs: dict[str, dict]) -> dict:
    """axes: algo -> axis -> [{value, run}] (baseline slotted into every axis of its algo);
    presets: name -> [runs]; grids: name -> [runs]."""
    axes: dict = {"ijepa": {}, "lejepa": {}}
    presets: dict = {}
    grids: dict = {}
    for e in index:
        for tag in e["tags"]:
            kind, _, rest = tag.partition(":")
            if kind == "axis":
                key, _, label = rest.partition("=")
                value = label if label else get_dotted(configs[e["name"]], key)
                axes[e["algo"]].setdefault(key, []).append({"value": value, "run": e["name"]})
            elif kind == "preset":
                presets.setdefault(rest, []).append(e["name"])
            elif kind == "grid":
                grids.setdefault(rest, []).append(e["name"])
    for algo, ax in axes.items():
        base = BASE_RUN[algo]
        if base not in configs:
            continue
        for key, lst in ax.items():
            if not any(x["run"] == base for x in lst):
                v = get_dotted(configs[base], key) if "." in key or key in configs[base] else None
                lst.append({"value": v, "run": base, "baseline": True})
            else:
                for x in lst:
                    if x["run"] == base:
                        x["baseline"] = True
            num = all(isinstance(x["value"], (int, float)) and not isinstance(x["value"], bool) for x in lst)
            if num:
                lst.sort(key=lambda x: x["value"])
    for grid, runs in grids.items():
        algo = next(e["algo"] for e in index if e["name"] == runs[0])
        if BASE_RUN[algo] in configs and BASE_RUN[algo] not in runs:
            runs.append(BASE_RUN[algo])
    return {"axes": axes, "presets": presets, "grids": grids}


def export_runs() -> None:
    index = []
    configs: dict[str, dict] = {}
    for run in sorted(RUNS.iterdir()):
        if run.name.startswith("_") or not (run / "DONE").exists():
            continue
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        summary = json.loads((run / "summary.json").read_text())
        recs = [json.loads(l) for l in open(run / "metrics.jsonl")]
        train = [r for r in recs if r["kind"] == "train"]
        ev = [r for r in recs if r["kind"] == "eval"]
        out = DATA / "runs" / run.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(
            {"steps_per_epoch": summary["steps_per_epoch"], "train": columnar(train), "eval": columnar(ev)},
            separators=(",", ":")))
        pca_meta = {}
        if (run / "pca_feats.pt").exists():
            pf = torch.load(run / "pca_feats.pt", weights_only=True)
            b = Bundle()
            for key, feats in pf["feats"].items():
                proj, evr = pca3_aligned(feats.float())
                b.add(key, proj, "f16")
                b.add(key + ".fixed", pca3_fixed(feats.float()), "f16")
                pca_meta[key] = {"evr": [[sig(x) for x in e] for e in evr],
                                 "scale": sig(float(np.abs(proj).max()))}
            b.write(out / "pca.bin")
            (out / "pca.json").write_text(json.dumps(
                {"steps": pf["steps"], "tensors": b.manifest, "keys": pca_meta, "labels": "dataset.json:labels.pca"}))
        final = ev[-1] if ev else {}
        configs[run.name] = json.loads((run / "config.json").read_text())
        algo = summary["algo"]
        lk = "loss_pred" if algo == "ijepa" else "loss_total"
        tail = [r[lk] for r in train[-20:] if r.get(lk) is not None and math.isfinite(r[lk])]
        losses = {
            "train_loss": sig(sum(tail) / len(tail)) if tail else None,  # mean of the last 20 logged steps
            "val_loss": sig(final.get("val_" + lk)),
            "lam": configs[run.name]["reg"]["lam"] if algo == "lejepa" else None,
        }
        index.append({
            "name": run.name,
            "algo": summary["algo"],
            "tags": meta.get("tags", []),
            "set": meta.get("set", {}),
            "steps": summary["steps"],
            "diverged": summary["diverged"],
            "wall_s": sig(summary["wall_s"]),
            "final": {**summarize(final, json.loads((run / "config.json").read_text())), **losses},
        })
        print(f"run {run.name}: {len(train)} train / {len(ev)} eval records")
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    (DATA / "runs" / "index.json").write_text(json.dumps({"runs": index, **explorer_maps(index, configs)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dataset", "models", "golden", "runs", "payload"], nargs="*")
    a = ap.parse_args()
    parts = a.only or ["dataset", "models", "golden", "runs", "payload"]
    if "dataset" in parts:
        export_dataset()
    if "models" in parts:
        for n in BASELINES:
            export_model(n)
    if "golden" in parts:
        for n in BASELINES:
            export_golden(n)
    if "runs" in parts:
        export_runs()
    if "payload" in parts:
        payload_report()


if __name__ == "__main__":
    main()
