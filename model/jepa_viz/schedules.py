"""LR, weight-decay and EMA-momentum schedules (ATOMS atoms 8, 9). All are pure functions of step."""
from __future__ import annotations

import math


def lr_at(step: int, total: int, warmup: int, peak: float, start_ratio: float, min_ratio: float,
          kind: str = "cosine") -> float:
    if warmup > 0 and step < warmup:
        return peak * (start_ratio + (1 - start_ratio) * step / warmup)
    if kind == "constant":
        return peak
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    lo = peak * min_ratio
    return lo + (peak - lo) * 0.5 * (1 + math.cos(math.pi * p))


def wd_at(step: int, total: int, wd0: float, wd1: float, kind: str = "cosine") -> float:
    if kind == "fixed":
        return wd0
    p = min(1.0, step / max(1, total))
    return wd1 + (wd0 - wd1) * 0.5 * (1 + math.cos(math.pi * p))


def tau_at(step: int, total: int, tau0: float, tau1: float) -> float:
    return tau0 + (tau1 - tau0) * min(1.0, step / max(1, total))
