"""Collapse / geometry diagnostics (ATOMS atom 15). Logged, never trained on."""
from __future__ import annotations

import torch

from .sigreg import epps_pulley, sample_directions, t_grid

DIAG_DIRECTION_SEED = 4242
N_DIAG_DIRECTIONS = 64
N_COS_PAIRS = 4096


@torch.no_grad()
def geometry(F: torch.Tensor, prefix: str = "") -> dict[str, object]:
    """F [n, d] features -> scalar metrics and the covariance spectrum (float lists)."""
    F = F.float()
    n, d = F.shape
    Fc = F - F.mean(0)
    cov = Fc.T @ Fc / (n - 1)
    eig = torch.linalg.eigvalsh(cov.double()).flip(0).clamp_min(0).float()  # descending
    var = torch.diagonal(cov)
    s = torch.linalg.svdvals(F.double())
    p = s / s.sum() + 1e-7
    rankme = torch.exp(-(p * torch.log(p)).sum())

    A = sample_directions(d, N_DIAG_DIRECTIONS, DIAG_DIRECTION_SEED, F.device)
    t = t_grid(5.0, 17, F.device)
    y = F @ A
    ep_raw = epps_pulley(y, t).mean()
    y_std = (y - y.mean(0)) / (y.std(0) + 1e-8)
    ep_std = epps_pulley(y_std, t).mean()

    g = torch.Generator(device=F.device).manual_seed(DIAG_DIRECTION_SEED)
    i = torch.randint(0, n, (N_COS_PAIRS,), generator=g, device=F.device)
    j = torch.randint(0, n, (N_COS_PAIRS,), generator=g, device=F.device)
    keep = i != j
    cos = torch.nn.functional.cosine_similarity(F[i[keep]], F[j[keep]], dim=1)

    lam_max = eig[0].clamp_min(1e-12)
    out = {
        "var_mean": float(var.mean()),
        "rankme": float(rankme),
        "iso_minmax": float(eig[-1] / lam_max),
        "iso_minmean": float(eig[-1] / eig.mean().clamp_min(1e-12)),
        # participation ratio / d, in (0, 1]; 1 = perfectly isotropic. Robust to a single null
        # direction (LayerNorm'd features with beta ~ 0 always have one, making min/max ~ 0).
        "iso_pr": float(eig.sum() ** 2 / (d * (eig**2).sum()).clamp_min(1e-24)),
        "ep_raw": float(ep_raw),
        "ep_std": float(ep_std),
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
        "norm_mean": float(F.norm(dim=1).mean()),
        "spectrum": eig.tolist(),
        "var": var.tolist(),
    }
    return {prefix + k: v for k, v in out.items()}


def collapse_verdict(g: dict[str, object], d: int, prefix: str = "") -> str:
    """Heuristic label for a feature set: complete / dimensional / none."""
    rank = float(g[prefix + "rankme"])
    if float(g[prefix + "cos_mean"]) > 0.98 or rank < 1.5:
        return "complete"
    if rank < 0.25 * d:
        return "dimensional"
    return "none"


def spread_and_verdict(rec: dict, key: str, d: int) -> tuple[float, float, str]:
    """Post-hoc collapse call from logged scalars (used by report/export).

    spread   = d * mean per-dim variance / mean(||f||)^2: the fraction of feature energy that varies
               across images. Unlike raw pair-cosine it ignores a shared offset (mean-pooled
               LayerNorm features all share one, which pushes raw cosine toward 1 even when healthy).
    eff_dims = participation ratio * d (centred covariance).
    verdict  = "complete" if spread < 1e-3 or RankMe < 1.5, else "none".
    """
    spread = float(rec[f"{key}.var_mean"]) * d / max(float(rec[f"{key}.norm_mean"]) ** 2, 1e-12)
    eff = float(rec[f"{key}.iso_pr"]) * d
    verdict = "complete" if spread < 1e-3 or float(rec[f"{key}.rankme"]) < 1.5 else "none"
    return spread, eff, verdict
