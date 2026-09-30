"""Batched GPU view generation with explicit, replayable parameters (ATOMS atom 10).

A view is fully determined by
  box        (x0, y0, w, h) in source pixels (continuous; the source is S x S, S = 32)
  flip       horizontal flip bit
  jitter     (on, brightness b, contrast c, saturation s, hue shift) -- ColorJitter(0.4, 0.4, 0.2, 0.1)
  gray, sol  grayscale / solarize bits
`render_views` is a pure function of (images, params), so the browser can reproduce any view.

Crop resampling: output pixel (i, j) of an r x r view samples the source at
    x = x0 + (j + 0.5) * w / r        (flipped: x = x0 + w - (j + 0.5) * w / r)
    y = y0 + (i + 0.5) * h / r
in continuous pixel coordinates (pixel k spans [k, k+1)), bilinear, edge-clamped
(grid_sample with align_corners=False, padding_mode='border').

Photometric ops run in a fixed order (brightness, contrast, saturation, hue), clamping to
[0, 1] after each. torchvision randomises the order; hue here is a YIQ-space rotation by
2*pi*hue, a close, cheap stand-in for torchvision's HSV hue shift.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

GRAY_W = (0.299, 0.587, 0.114)
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def _u(gen: torch.Generator, shape, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def sample_boxes(n: int, scale: tuple, gen: torch.Generator, device, size: int = 32,
                 ratio: tuple = (3 / 4, 4 / 3), attempts: int = 10) -> torch.Tensor:
    """Vectorised RandomResizedCrop box sampler -> [n, 4] (x0, y0, w, h). Fallback: whole image."""
    s = _u(gen, (n, attempts), scale[0], scale[1], device)
    r = torch.exp(_u(gen, (n, attempts), math.log(ratio[0]), math.log(ratio[1]), device))
    w = torch.sqrt(s * size * size * r)
    h = torch.sqrt(s * size * size / r)
    ok = (w <= size) & (h <= size)
    first = ok.float().argmax(1)
    rows = torch.arange(n, device=device)
    w, h = w[rows, first], h[rows, first]
    none_ok = ~ok.any(1)
    w = torch.where(none_ok, torch.full_like(w, size), w)
    h = torch.where(none_ok, torch.full_like(h, size), h)
    x0 = torch.rand(n, generator=gen, device=device) * (size - w)
    y0 = torch.rand(n, generator=gen, device=device) * (size - h)
    return torch.stack([x0, y0, w, h], -1)


def sample_view_params(n: int, scale: tuple, gen: torch.Generator, device, color: bool = True,
                       flip: bool = True) -> dict[str, torch.Tensor]:
    p = {"box": sample_boxes(n, scale, gen, device)}
    rnd = lambda: torch.rand(n, generator=gen, device=device)
    p["flip"] = (rnd() < 0.5) if flip else torch.zeros(n, dtype=torch.bool, device=device)
    on = color
    p["jitter"] = (rnd() < 0.8) & on
    p["bright"] = _u(gen, n, 0.6, 1.4, device)
    p["contrast"] = _u(gen, n, 0.6, 1.4, device)
    p["sat"] = _u(gen, n, 0.8, 1.2, device)
    p["hue"] = _u(gen, n, -0.1, 0.1, device)
    p["gray"] = (rnd() < 0.2) & on
    p["sol"] = (rnd() < 0.2) & on
    return p


def identity_params(n: int, device, size: int = 32) -> dict[str, torch.Tensor]:
    z = torch.zeros(n, device=device)
    f = torch.zeros(n, dtype=torch.bool, device=device)
    box = torch.tensor([0.0, 0.0, size, size], device=device).expand(n, 4).clone()
    return {"box": box, "flip": f, "jitter": f, "bright": z + 1, "contrast": z + 1, "sat": z + 1,
            "hue": z, "gray": f, "sol": f}


def _gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(GRAY_W, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def hue_matrix(hue: torch.Tensor) -> torch.Tensor:
    """[n] hue shifts (fraction of a turn) -> [n, 3, 3] RGB->RGB matrices via YIQ rotation."""
    yiq = _YIQ.to(hue.device)
    inv = torch.linalg.inv(yiq)
    th = 2 * math.pi * hue
    c, s = torch.cos(th), torch.sin(th)
    R = torch.zeros(hue.shape[0], 3, 3, device=hue.device)
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
    return inv @ R @ yiq


def render_views(img_u8: torch.Tensor, p: dict[str, torch.Tensor], res: int) -> torch.Tensor:
    """img_u8 [n, 3, S, S] (the source image of each view) -> views in [0, 1], [n, 3, res, res]."""
    n, _, S, _ = img_u8.shape
    dev = img_u8.device
    x = img_u8.float() / 255.0
    x0, y0, w, h = p["box"].unbind(-1)
    k = (torch.arange(res, device=dev, dtype=torch.float32) + 0.5) / res
    xs = x0[:, None] + k[None] * w[:, None]
    xs = torch.where(p["flip"][:, None], x0[:, None] + w[:, None] - k[None] * w[:, None], xs)
    ys = y0[:, None] + k[None] * h[:, None]
    gx = (2 * xs / S - 1)[:, None, :].expand(n, res, res)
    gy = (2 * ys / S - 1)[:, :, None].expand(n, res, res)
    out = F.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border",
                        align_corners=False)

    j = p["jitter"].view(n, 1, 1, 1)
    y = (out * p["bright"].view(n, 1, 1, 1)).clamp(0, 1)
    m = _gray(y).mean((2, 3), keepdim=True)
    y = (p["contrast"].view(n, 1, 1, 1) * y + (1 - p["contrast"].view(n, 1, 1, 1)) * m).clamp(0, 1)
    g = _gray(y)
    y = (p["sat"].view(n, 1, 1, 1) * y + (1 - p["sat"].view(n, 1, 1, 1)) * g).clamp(0, 1)
    y = torch.einsum("nij,njhw->nihw", hue_matrix(p["hue"]), y).clamp(0, 1)
    out = torch.where(j, y, out)
    out = torch.where(p["gray"].view(n, 1, 1, 1), _gray(out).expand(-1, 3, -1, -1), out)
    out = torch.where(p["sol"].view(n, 1, 1, 1) & (out >= 0.5), 1 - out, out)
    return out
