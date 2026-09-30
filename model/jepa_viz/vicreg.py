"""VICReg variance + covariance terms (Bardes, Ponce & LeCun, 2022) as a SIGReg swap-in.

Applied per view to z [V, N, K] and averaged over views, mirroring how SIGReg is applied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vicreg_reg(z: torch.Tensor, var_w: float = 1.0, cov_w: float = 0.04) -> tuple[torch.Tensor, dict]:
    z = z.float()
    if z.dim() == 2:
        z = z[None]
    V, N, K = z.shape
    zc = z - z.mean(1, keepdim=True)
    std = torch.sqrt(zc.var(1) + 1e-4)  # [V, K]
    var = F.relu(1 - std).mean()
    cov = zc.transpose(1, 2) @ zc / (N - 1)  # [V, K, K]
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    cov_loss = (off**2).sum((1, 2)).mean() / K
    return var_w * var + cov_w * cov_loss, {"vic_var": var.detach(), "vic_cov": cov_loss.detach()}
