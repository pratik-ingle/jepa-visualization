"""SIGReg: sketched isotropic Gaussian regularisation (ATOMS atom 13).

Follows the LeJEPA paper listing (arXiv 2511.08544):
    A = randn(K, M), columns normalised, seeded by the global step (resampled every step)
    t = linspace(-5, 5, 17);  phi(t) = exp(-t^2 / 2)
    err = |ecf(t) - phi(t)|^2 * phi(t);  EP = N * trapz(err, t)
    SIGReg = mean over slices (and over views)
Always computed in float32.
"""
from __future__ import annotations

import math

import torch


def t_grid(t_max: float = 5.0, n: int = 17, device=None) -> torch.Tensor:
    return torch.linspace(-t_max, t_max, n, device=device)


def sample_directions(K: int, M: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    A = torch.randn(K, M, generator=g, device=device)
    return A / A.norm(dim=0, keepdim=True)


def ecf(y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical characteristic function of samples y [..., N, M] at t [T] -> (re, im) [..., M, T]."""
    yt = y.unsqueeze(-1) * t
    return torch.cos(yt).mean(-3), torch.sin(yt).mean(-3)


def epps_pulley(y: torch.Tensor, t: torch.Tensor, return_parts: bool = False):
    """Epps-Pulley statistic of each column of y [..., N, M] against N(0, 1) -> [..., M]."""
    N = y.shape[-2]
    re, im = ecf(y.float(), t)
    phi = torch.exp(-0.5 * t**2)
    err = ((re - phi) ** 2 + im**2) * phi
    ep = torch.trapezoid(err, t, dim=-1) * N
    if return_parts:
        return ep, {"ecf_re": re, "ecf_im": im, "err": err}
    return ep


def sigreg(z: torch.Tensor, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z [V, N, K] (or [N, K]) embeddings, A [K, M] unit directions -> scalar mean EP."""
    y = z.float() @ A
    return epps_pulley(y, t).mean()


def ep_closed_form(y: torch.Tensor) -> torch.Tensor:
    """Exact N * integral over R of |ecf - phi|^2 phi dt for y [N, M] (weight exp(-t^2/2)).

    Expanding the square and using int cos(a t) exp(-t^2/2) dt = sqrt(2 pi) exp(-a^2/2):
      sqrt(2pi)/N * sum_jk exp(-(y_j - y_k)^2 / 2) - 2 sqrt(pi) * sum_j exp(-y_j^2 / 4) + N sqrt(2pi/3)
    (the BHEP / Epps-Pulley closed form up to the weight's normalising constant). Test oracle.
    """
    y = y.double()
    N = y.shape[0]
    d = y[:, None, :] - y[None, :, :]
    a = math.sqrt(2 * math.pi) / N * torch.exp(-0.5 * d**2).sum((0, 1))
    b = 2 * math.sqrt(math.pi) * torch.exp(-(y**2) / 4).sum(0)
    return a - b + N * math.sqrt(2 * math.pi / 3)
