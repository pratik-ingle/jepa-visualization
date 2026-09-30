"""I-JEPA (ATOMS atoms 1-9): context encoder, EMA target encoder, predictor, loss, EMA step."""
from __future__ import annotations

import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import render_views, sample_view_params
from .config import Config
from .masks import MaskBatch, MultiBlockMaskSampler
from .vit import LN_EPS, Block, ViT, _rec, gather_tokens, init_weights, sincos_pos_embed_2d

TARGET_LN_EPS = 1e-5  # F.layer_norm default, as in the official train.py


class Predictor(nn.Module):
    """Atom 5. Each target block is its own sequence [context ; mask queries], batched block-major."""

    def __init__(self, enc_dim: int, dim: int, depth: int, heads: int, mlp_ratio: int = 4, grid: int = 8):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.proj = nn.Linear(dim, enc_dim)
        self.register_buffer("pos", sincos_pos_embed_2d(dim, grid, grid), persistent=False)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, s_x: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor,
                rec: dict | None = None) -> torch.Tensor:
        Mt, B, Nt = tgt_idx.shape
        Nc = s_x.shape[1]
        c = self.embed(s_x)
        _rec(rec, "pred.embed", c)
        c = c + self.pos[ctx_idx]
        _rec(rec, "pred.ctx", c)
        m = self.mask_token + self.pos[tgt_idx.reshape(Mt * B, Nt)]
        _rec(rec, "pred.mask_q", m)
        x = torch.cat([c.repeat(Mt, 1, 1), m], 1)  # [Mt*B, Nc+Nt, Dp]
        _rec(rec, "pred.in", x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rec, f"pred.blk{i}.")
        x = self.norm(x)[:, Nc:]
        _rec(rec, "pred.norm", x)
        out = self.proj(x)
        _rec(rec, "pred.out", out)
        return out


class LinearPredictor(nn.Module):
    """Ablation: position-aware but linear, no attention.
    s_hat_j = A * mean(s_x) + B * pi(pos_j) + b  (pi: fixed sin-cos at the predictor width)."""

    def __init__(self, enc_dim: int, dim: int, grid: int = 8):
        super().__init__()
        self.ctx = nn.Linear(enc_dim, enc_dim)
        self.pos_proj = nn.Linear(dim, enc_dim, bias=False)
        self.register_buffer("pos", sincos_pos_embed_2d(dim, grid, grid), persistent=False)
        self.apply(init_weights)

    def forward(self, s_x: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor,
                rec: dict | None = None) -> torch.Tensor:
        Mt, B, Nt = tgt_idx.shape
        c = self.ctx(s_x.mean(1, keepdim=True)).repeat(Mt, 1, 1)  # [Mt*B, 1, D]
        out = c + self.pos_proj(self.pos[tgt_idx.reshape(Mt * B, Nt)])
        _rec(rec, "pred.out", out)
        return out


class IJEPA(nn.Module):
    def __init__(self, cfg: Config, grid: int = 8):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.encoder = ViT(m.patch, m.dim, m.depth, m.heads, m.mlp_ratio, ref_grid=grid)
        self.shared = cfg.ema.shared
        self.target_encoder = None
        if not self.shared:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        pc = cfg.pred
        self.predictor = None
        if pc.enabled:
            self.predictor = (LinearPredictor(m.dim, pc.dim, grid) if pc.linear
                              else Predictor(m.dim, pc.dim, pc.depth, pc.heads, m.mlp_ratio, grid))
        self.sampler = MultiBlockMaskSampler.from_cfg(cfg.mask, grid, seed=cfg.seed)
        self.val_sampler = MultiBlockMaskSampler.from_cfg(cfg.mask, grid, seed=cfg.seed + 7919)
        self.aug_gen: torch.Generator | None = None

    # -- parameter groups ------------------------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        g = {"encoder": list(self.encoder.parameters())}
        if self.predictor is not None:
            g["head"] = list(self.predictor.parameters())
        return g

    @property
    def target(self) -> ViT:
        return self.encoder if self.shared else self.target_encoder

    # -- atoms 3-6 ---------------------------------------------------------------------------
    def forward_target(self, x: torch.Tensor, tgt_idx: torch.Tensor, rec: dict | None = None) -> torch.Tensor:
        ctx = torch.no_grad() if self.cfg.ema.stopgrad else contextlib.nullcontext()
        with ctx:
            h = self.target(x, rec=rec, pre="tgt.")
            h = F.layer_norm(h, (h.shape[-1],), eps=TARGET_LN_EPS)
            _rec(rec, "tgt.ln", h)
            s_y = torch.cat([gather_tokens(h, idx) for idx in tgt_idx], 0)  # [Mt*B, Nt, D]
            _rec(rec, "tgt.s_y", s_y)
        return s_y

    def predict(self, s_x: torch.Tensor, masks: MaskBatch, rec: dict | None = None) -> torch.Tensor:
        Mt, B, Nt = masks.tgt_idx.shape
        if self.predictor is not None:
            return self.predictor(s_x, masks.ctx_idx, masks.tgt_idx, rec)
        # Ablation "no predictor": pooled context, identity map, no positional conditioning.
        pooled = s_x.mean(1, keepdim=True).repeat(Mt, 1, 1).expand(-1, Nt, -1)
        _rec(rec, "pred.out", pooled)
        return pooled

    def loss_on(self, x: torch.Tensor, masks: MaskBatch, rec: dict | None = None):
        s_x = self.encoder(x, keep_idx=masks.ctx_idx, rec=rec, pre="ctx.")
        s_y = self.forward_target(x, masks.tgt_idx, rec)
        s_hat = self.predict(s_x, masks, rec)
        with torch.autocast(device_type=x.device.type, enabled=False):  # loss in fp32
            if self.cfg.loss == "l2":
                loss = F.mse_loss(s_hat.float(), s_y.float())
            else:
                loss = F.smooth_l1_loss(s_hat.float(), s_y.float())
        _rec(rec, "loss", loss)
        return loss, {"loss_pred": loss.detach()}

    # -- training interface --------------------------------------------------------------------
    def train_loss(self, x_u8: torch.Tensor, step: int, normalize):
        B = x_u8.shape[0]
        dev = x_u8.device
        if self.aug_gen is None:
            self.aug_gen = torch.Generator(device=dev).manual_seed(self.cfg.seed + 101)
        p = sample_view_params(B, self.cfg.ijepa_crop_scale, self.aug_gen, dev, color=False, flip=False)
        x = normalize(render_views(x_u8, p, x_u8.shape[-1]))
        masks = self.sampler(B, step).to(dev)
        loss, terms = self.loss_on(x, masks)
        terms["n_ctx"] = torch.tensor(float(masks.ctx_idx.shape[1]))
        terms["n_tgt"] = torch.tensor(float(masks.tgt_idx.shape[2]))
        return loss, terms

    @torch.no_grad()
    def val_loss(self, x_u8: torch.Tensor, chunk_idx: int, normalize) -> dict[str, torch.Tensor]:
        """Forward only on held-out images: uncropped, masks from a fixed seed per chunk."""
        B = x_u8.shape[0]
        x = normalize(x_u8.float() / 255.0)
        masks = self.val_sampler(B, chunk_idx).to(x_u8.device)
        _, terms = self.loss_on(x, masks)
        return terms

    @torch.no_grad()
    def after_step(self, tau: float) -> None:
        """Atom 9: theta_target <- tau * theta_target + (1 - tau) * theta_context."""
        if self.shared:
            return
        tp = [p for p in self.target_encoder.parameters()]
        cp = [p.detach() for p in self.encoder.parameters()]
        torch._foreach_mul_(tp, tau)
        torch._foreach_add_(tp, cp, alpha=1.0 - tau)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Atom 16: mean-pooled final-LN tokens of the full image."""
        return {"target": self.target(x).mean(1), "context": self.encoder(x).mean(1)}

    @torch.no_grad()
    def ema_gap(self) -> float:
        if self.shared:
            return 0.0
        num = sum(((a - b) ** 2).sum() for a, b in zip(self.target_encoder.parameters(), self.encoder.parameters()))
        den = sum((b**2).sum() for b in self.encoder.parameters())
        return float(torch.sqrt(num / den))
