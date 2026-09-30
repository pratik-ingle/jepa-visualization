"""Run configuration: nested dataclasses, per-algorithm baselines, dotted overrides.

A run is fully described by (algo, overrides). Its name defaults to a short hash of the
overrides, so sweep runs are addressable by the settings that differ from baseline.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelCfg:
    patch: int = 4
    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4


@dataclass
class MaskCfg:  # I-JEPA multi-block masking (ATOMS atom 2)
    npred: int = 4
    pred_scale: tuple = (0.15, 0.2)
    pred_aspect: tuple = (0.75, 1.5)
    enc_scale: tuple = (0.85, 1.0)
    min_keep: int = 4
    edge_fix: bool = True


@dataclass
class PredCfg:  # I-JEPA predictor (atom 5)
    enabled: bool = True
    linear: bool = False  # ablation: s_hat_j = A mean(s_x) + B pi(pos_j) + b (position-aware, no attention)
    dim: int = 32
    depth: int = 2
    heads: int = 2


@dataclass
class EmaCfg:  # I-JEPA target encoder (atoms 4, 9)
    tau0: float = 0.996
    tau1: float = 1.0
    shared: bool = False  # target encoder IS the context encoder (tau = 0)
    stopgrad: bool = True


@dataclass
class ViewCfg:  # LeJEPA multi-crop (atom 10)
    n_global: int = 2
    n_local: int = 6
    global_res: int = 32
    local_res: int = 16
    global_scale: tuple = (0.3, 1.0)
    local_scale: tuple = (0.05, 0.3)
    color: bool = True


@dataclass
class ProjCfg:  # LeJEPA projector (atom 11d)
    hidden: int = 256
    depth: int = 3
    out: int = 32  # K


@dataclass
class RegCfg:  # LeJEPA regulariser (atoms 13, 14)
    kind: str = "sigreg"  # sigreg | vicreg
    lam: float = 0.05
    slices: int = 256
    t_max: float = 5.0
    t_points: int = 17
    resample: bool = True
    vic_var_w: float = 1.0
    vic_cov_w: float = 0.04


@dataclass
class OptimCfg:  # atom 8
    name: str = "adamw"  # adamw | adam | sgd
    lr: float = 1e-3
    start_lr_ratio: float = 0.2
    min_lr_ratio: float = 1e-3
    warmup_epochs: float = 10.0
    schedule: str = "cosine"  # cosine | constant
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9  # sgd only
    wd: float = 0.04
    wd_end: float = 0.4
    wd_schedule: str = "cosine"  # cosine | fixed
    batch_size: int = 256
    epochs: float = 200.0


@dataclass
class LogCfg:
    every: int = 25
    eval_every_epochs: float = 2.0
    n_pca_ckpts: int = 24
    n_weight_ckpts: int = 0  # >0 keeps full weights at log-spaced steps (baselines)


@dataclass
class SystemCfg:  # numerics (changes how the maths is computed, not what is computed)
    # bf16 autocast for the networks; losses, SIGReg, EMA, optimiser state and all evaluation stay fp32.
    # Measured: LeJEPA 26 -> 19 ms/step with plain attention (vit.ATTN_IMPL). torch.compile was tried
    # and dropped (NaN at step 0 combined with bf16; per-block recompiles; <= 10% gain at this size).
    precision: str = "bf16"  # bf16 | fp32


@dataclass
class Config:
    algo: str = "ijepa"  # ijepa | lejepa
    name: str = ""
    seed: int = 0
    loss: str = "smooth_l1"  # I-JEPA: smooth_l1 | l2
    ijepa_crop_scale: tuple = (0.3, 1.0)
    model: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    pred: PredCfg = field(default_factory=PredCfg)
    ema: EmaCfg = field(default_factory=EmaCfg)
    views: ViewCfg = field(default_factory=ViewCfg)
    proj: ProjCfg = field(default_factory=ProjCfg)
    reg: RegCfg = field(default_factory=RegCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    log: LogCfg = field(default_factory=LogCfg)
    system: SystemCfg = field(default_factory=SystemCfg)


# Per-algorithm baseline settings layered on top of the dataclass defaults.
ALGO_BASE: dict[str, dict[str, Any]] = {
    "ijepa": {},
    "lejepa": {  # LeJEPA: fixed weight decay, no schedule (paper section 6.1)
        "optim.wd": 0.05,
        "optim.wd_schedule": "fixed",
    },
}


def set_dotted(cfg: Any, key: str, value: Any) -> None:
    obj = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown config key: {key}")
    old = getattr(obj, parts[-1])
    if isinstance(value, str) and not isinstance(old, str):
        value = yaml.safe_load(value)
    if isinstance(old, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, parts[-1], value)


def build(algo: str, overrides: dict[str, Any] | None = None, name: str = "") -> Config:
    cfg = Config(algo=algo)
    for k, v in ALGO_BASE[algo].items():
        set_dotted(cfg, k, v)
    overrides = dict(overrides or {})
    for k, v in overrides.items():
        set_dotted(cfg, k, v)
    cfg.name = name or f"{algo}-{overrides_hash(overrides)}"
    return cfg


def overrides_hash(overrides: dict[str, Any]) -> str:
    blob = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8] if overrides else "base"


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def from_dict(d: dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in d.items():
        if isinstance(v, dict):
            sub = getattr(cfg, k)
            for kk, vv in v.items():
                setattr(sub, kk, tuple(vv) if isinstance(vv, list) else vv)
        else:
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg
