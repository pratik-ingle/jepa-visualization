"""I-JEPA multi-block masking (ATOMS atom 2), vectorised over the batch.

Follows facebookresearch/ijepa src/masks/multiblock.py:
  * block sizes drawn once per batch; scale and aspect share one uniform draw (official quirk)
  * target blocks placed independently per image; context = one square block per image with
    every target block carved out; rejected while it keeps <= min_keep tokens; after 20 failed
    draws for an image, its last remaining target constraint is dropped
  * index sets sorted and truncated to the batch minimum (only ever shortens the context)
Deviation D1 (edge_fix=True): official placement `randint(0, G - h)` (exclusive) plus h <= G-1
never touches the last row/column; we allow top in [0, G-h] and h <= G.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MaskBatch:
    ctx_idx: torch.Tensor  # [B, N_ctx] sorted token indices
    tgt_idx: torch.Tensor  # [M_t, B, N_tgt] token indices, row-major within each block
    tgt_hw: tuple[int, int]
    ctx_hw: tuple[int, int]
    ctx_full: torch.Tensor  # [B] context size before batch truncation

    def to(self, device) -> "MaskBatch":
        return MaskBatch(self.ctx_idx.to(device, non_blocking=True), self.tgt_idx.to(device, non_blocking=True),
                         self.tgt_hw, self.ctx_hw, self.ctx_full)


class MultiBlockMaskSampler:
    def __init__(self, npred=4, pred_scale=(0.15, 0.2), pred_aspect=(0.75, 1.5), enc_scale=(0.85, 1.0),
                 min_keep=4, edge_fix=True, grid=8, seed=0, timeout=20):
        self.npred, self.pred_scale, self.pred_aspect = npred, pred_scale, pred_aspect
        self.enc_scale, self.min_keep, self.edge_fix = enc_scale, min_keep, edge_fix
        self.G, self.seed, self.timeout = grid, seed, timeout

    @classmethod
    def from_cfg(cls, m, grid=8, seed=0) -> "MultiBlockMaskSampler":
        return cls(m.npred, m.pred_scale, m.pred_aspect, m.enc_scale, m.min_keep, m.edge_fix, grid, seed)

    def block_size(self, u: float, scale, aspect) -> tuple[int, int]:
        s = scale[0] + u * (scale[1] - scale[0])
        k = int(self.G * self.G * s)
        ar = aspect[0] + u * (aspect[1] - aspect[0])
        h = int(round(math.sqrt(k * ar)))
        w = int(round(math.sqrt(k / ar)))
        limit = self.G if self.edge_fix else self.G - 1
        return max(1, min(h, limit)), max(1, min(w, limit))

    def _place(self, n: int, h: int, w: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        hi_t = self.G - h + 1 if self.edge_fix else self.G - h
        hi_l = self.G - w + 1 if self.edge_fix else self.G - w
        top = torch.randint(0, max(hi_t, 1), (n,), generator=gen)
        left = torch.randint(0, max(hi_l, 1), (n,), generator=gen)
        return top, left

    def _mask(self, top: torch.Tensor, left: torch.Tensor, h: int, w: int) -> torch.Tensor:
        r = torch.arange(self.G)
        rows = (r[None] >= top[:, None]) & (r[None] < top[:, None] + h)
        cols = (r[None] >= left[:, None]) & (r[None] < left[:, None] + w)
        return rows[:, :, None] & cols[:, None, :]  # [n, G, G]

    def __call__(self, B: int, step: int) -> MaskBatch:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        u_p, u_e = torch.rand(2, generator=gen).tolist()
        h, w = self.block_size(u_p, self.pred_scale, self.pred_aspect)
        hc, wc = self.block_size(u_e, self.enc_scale, (1.0, 1.0))
        G = self.G

        tgt_masks, tgt_idx = [], []
        for _ in range(self.npred):
            top, left = self._place(B, h, w, gen)
            tgt_masks.append(self._mask(top, left, h, w))
            rr = top[:, None, None] + torch.arange(h)[None, :, None]
            cc = left[:, None, None] + torch.arange(w)[None, None, :]
            tgt_idx.append((rr * G + cc).reshape(B, h * w))
        keep_ok = torch.stack([~m for m in tgt_masks])  # [npred, B, G, G] acceptable regions

        ctx = torch.zeros(B, G, G, dtype=torch.bool)
        valid = torch.zeros(B, dtype=torch.bool)
        tries = torch.zeros(B, dtype=torch.long)
        timeout = torch.full((B,), self.timeout, dtype=torch.long)
        while not bool(valid.all()):
            top, left = self._place(B, hc, wc, gen)
            m = self._mask(top, left, hc, wc)
            n_constr = (self.npred - tries).clamp(min=0)
            for k in range(self.npred):
                use = (k < n_constr).view(B, 1, 1)
                m = m & (keep_ok[k] | ~use)
            ok = m.flatten(1).sum(1) > self.min_keep
            pending = ~valid
            new = pending & ok
            ctx[new] = m[new]
            valid |= new
            fail = pending & ~ok
            timeout[fail] -= 1
            expired = fail & (timeout == 0)
            tries[expired] += 1
            timeout[expired] = self.timeout

        flat = ctx.flatten(1)
        counts = flat.sum(1)
        n_ctx = int(counts.min())
        ctx_idx = torch.argsort((~flat).to(torch.int8), dim=1, stable=True)[:, :n_ctx]
        return MaskBatch(ctx_idx, torch.stack(tgt_idx), (h, w), (hc, wc), counts)
