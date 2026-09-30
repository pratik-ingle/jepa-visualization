"""Tiny ViT (ATOMS atom 1 and the transformer-block sub-atoms T1-T10).

Conventions shared with the browser port and np_forward.py:
  patchify   token index = row * g + col; value index = c * P^2 + py * P + px
  qkv        Linear(W, 3W) output columns: [q | k | v], each split into heads of d_h columns
  pos-embed  fixed 2D sin-cos; first half of channels encodes the column, second half the row;
             a view with g x g tokens covering the G x G reference frame puts token i at
             coordinate (i + 0.5) * G / g - 0.5 (so a 4x4 local view spans the full frame)
  LayerNorm  eps = 1e-6;  GELU exact (erf)

Pass `rec={}` (and optionally a key prefix) to any forward to record every intermediate.
Recorded tensors stay attached to the graph so callers can retain_grad() them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-6
# "math": explicit softmax(q k^T / sqrt(d_h)) v with batched matmuls -- faster than the fused SDPA
# kernels at these sizes (d_h = 16, <= 64 tokens) and identical to the recorded/golden path.
ATTN_IMPL = "math"


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/p)*(W/p), C*p*p]"""
    B, C, H, W = x.shape
    g_h, g_w = H // p, W // p
    return x.reshape(B, C, g_h, p, g_w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g_h * g_w, C * p * p)


def sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0)
    omega = 1.0 / 10000**omega
    out = pos.double()[:, None] * omega[None]
    return torch.cat([torch.sin(out), torch.cos(out)], 1)


def grid_coords(g: int, G: int) -> torch.Tensor:
    return (torch.arange(g, dtype=torch.float64) + 0.5) * G / g - 0.5


def sincos_pos_embed_2d(dim: int, g: int, G: int = 8) -> torch.Tensor:
    """[g*g, dim] float32 positional embedding for a g x g token grid in a G x G frame."""
    c = grid_coords(g, G)
    rows = c.repeat_interleave(g)
    cols = c.repeat(g)
    return torch.cat([sincos_1d(dim // 2, cols), sincos_1d(dim // 2, rows)], 1).float()


def _rec(rec: dict | None, key: str, t: torch.Tensor) -> None:
    if rec is not None:
        rec[key] = t


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rec: dict | None = None, pre: str = "") -> torch.Tensor:
        B, S, W = x.shape
        h, dh = self.heads, W // self.heads
        q, k, v = self.qkv(x).reshape(B, S, 3, h, dh).permute(2, 0, 3, 1, 4)  # each [B, h, S, dh]
        if rec is None and ATTN_IMPL == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
            attn = scores.softmax(-1)
            o = attn @ v
            for name, t in (("q", q), ("k", k), ("v", v), ("scores", scores), ("attn", attn)):
                _rec(rec, pre + name, t)
        o = o.transpose(1, 2).reshape(B, S, W)
        _rec(rec, pre + "o", o)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x: torch.Tensor, rec: dict | None = None, pre: str = "") -> torch.Tensor:
        a = self.norm1(x)
        _rec(rec, pre + "ln1", a)
        x = x + self.attn(a, rec, pre)
        _rec(rec, pre + "x_attn", x)
        m = self.norm2(x)
        _rec(rec, pre + "ln2", m)
        u = self.fc1(m)
        _rec(rec, pre + "fc1", u)
        g = F.gelu(u)
        _rec(rec, pre + "gelu", g)
        x = x + self.fc2(g)
        _rec(rec, pre + "x_out", x)
        return x


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, W], idx [B, K] -> [B, K, W]"""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class ViT(nn.Module):
    def __init__(self, patch: int = 4, dim: int = 64, depth: int = 4, heads: int = 4,
                 mlp_ratio: int = 4, in_ch: int = 3, ref_grid: int = 8):
        super().__init__()
        self.patch, self.dim, self.ref_grid = patch, dim, ref_grid
        self.patch_embed = nn.Linear(in_ch * patch * patch, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=LN_EPS)
        self._pos: dict[int, torch.Tensor] = {}
        self.apply(init_weights)
        # Official I-JEPA keeps PyTorch's default init for the (conv) patch embedding. With
        # std-0.02 weights, patch content (std ~0.14) is swamped by the sin-cos positions (RMS ~0.7)
        # and the predictor can hit targets from position alone.
        self.patch_embed.reset_parameters()

    def pos_embed(self, g: int, device) -> torch.Tensor:
        if g not in self._pos or self._pos[g].device != device:
            self._pos[g] = sincos_pos_embed_2d(self.dim, g, self.ref_grid).to(device)
        return self._pos[g]

    def embed(self, x: torch.Tensor, rec: dict | None = None, pre: str = "") -> torch.Tensor:
        p = patchify(x, self.patch)
        g = x.shape[-1] // self.patch
        e = self.patch_embed(p)
        pos = self.pos_embed(g, x.device)
        t = e + pos
        _rec(rec, pre + "patches", p)
        _rec(rec, pre + "patch_embed", e)
        _rec(rec, pre + "pos", pos)
        _rec(rec, pre + "tokens", t)
        return t

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None,
                rec: dict | None = None, pre: str = "") -> torch.Tensor:
        """x: normalised images [B, C, r, r]; keep_idx: [B, K] token indices to keep (context)."""
        t = self.embed(x, rec, pre)
        if keep_idx is not None:
            t = gather_tokens(t, keep_idx)
            _rec(rec, pre + "kept", t)
        for i, blk in enumerate(self.blocks):
            t = blk(t, rec, f"{pre}blk{i}.")
        t = self.norm(t)
        _rec(rec, pre + "out", t)
        return t
