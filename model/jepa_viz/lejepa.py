"""LeJEPA (ATOMS atoms 10-14): multi-crop views -> ONE encoder -> projector -> invariance + SIGReg.

No EMA, no stop-gradient, no teacher, no predictor.
Loss (paper Eq. 9): (1 - lam) * mean_{n,v,k} (mu_n - z_{n,v})^2 + lam * mean_v SIGReg(z_v),
with mu_n the centroid of the GLOBAL views and the error taken over ALL views.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .augment import render_views, sample_view_params
from .config import Config
from .sigreg import epps_pulley, sample_directions, t_grid
from .vicreg import vicreg_reg
from .vit import ViT, _rec

FIXED_DIRECTION_SEED = 12345


class Projector(nn.Module):
    """Atom 11d: [Linear -> BatchNorm -> ReLU] x (depth - 1) -> Linear."""

    def __init__(self, in_dim: int, hidden: int, out: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, rec: dict | None = None) -> torch.Tensor:
        for i, layer in enumerate(self.net):
            x = layer(x)
            _rec(rec, f"proj.{i}.{type(layer).__name__.lower()}", x)
        return x


class LeJEPA(nn.Module):
    def __init__(self, cfg: Config, grid: int = 8):
        super().__init__()
        m, pc = cfg.model, cfg.proj
        self.cfg = cfg
        self.encoder = ViT(m.patch, m.dim, m.depth, m.heads, m.mlp_ratio, ref_grid=grid)
        self.projector = Projector(m.dim, pc.hidden, pc.out, pc.depth)
        self.aug_gen: torch.Generator | None = None

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"encoder": list(self.encoder.parameters()), "head": list(self.projector.parameters())}

    # -- atoms 11-14 ---------------------------------------------------------------------------
    def embed_views(self, g: torch.Tensor, l: torch.Tensor | None, B: int, rec: dict | None = None):
        """g [Vg*B, 3, 32, 32], l [Vl*B, 3, 16, 16] view-major -> e [V*B, D], z [V, B, K]."""
        e = [self.encoder(g, rec=rec, pre="g.").mean(1)]
        if l is not None and l.shape[0] > 0:
            e.append(self.encoder(l, rec=rec, pre="l.").mean(1))
        e = torch.cat(e)
        _rec(rec, "pool", e)
        z = self.projector(e, rec)
        return e, z.view(-1, B, z.shape[-1])

    def directions(self, step: int, device) -> torch.Tensor:
        r = self.cfg.reg
        seed = step if r.resample else FIXED_DIRECTION_SEED
        return sample_directions(self.cfg.proj.out, r.slices, seed, device)

    def loss_on(self, g, l, B: int, step: int, rec: dict | None = None):
        _, z = self.embed_views(g, l, B, rec)
        # Losses always in fp32, even under bf16 autocast: cos(t * y) with |t| up to 5 needs it.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._loss_fp32(z.float(), B, step, rec)

    def _loss_fp32(self, zf: torch.Tensor, B: int, step: int, rec: dict | None):
        r = self.cfg.reg
        z = zf
        centers = zf[: self.cfg.views.n_global].mean(0)
        inv = (centers - zf).square().mean()
        _rec(rec, "centers", centers)
        terms = {"loss_inv": inv.detach()}
        if r.kind == "vicreg":
            reg, extra = vicreg_reg(zf, r.vic_var_w, r.vic_cov_w)
            terms.update(extra)
        else:
            A = self.directions(step, z.device)
            t = t_grid(r.t_max, r.t_points, z.device)
            y = zf @ A  # [V, B, M]
            ep, parts = epps_pulley(y, t, return_parts=True)  # [V, M]
            reg = ep.mean()
            _rec(rec, "sigreg.A", A)
            _rec(rec, "sigreg.y", y)
            _rec(rec, "sigreg.ecf_re", parts["ecf_re"])
            _rec(rec, "sigreg.ecf_im", parts["ecf_im"])
            _rec(rec, "sigreg.ep", ep)
        loss = (1 - r.lam) * inv + r.lam * reg
        _rec(rec, "loss", loss)
        terms["loss_reg"] = reg.detach()
        terms["loss_total"] = loss.detach()
        return loss, terms

    # -- training interface --------------------------------------------------------------------
    def make_views(self, x_u8: torch.Tensor, gen: torch.Generator, normalize):
        vc = self.cfg.views
        B = x_u8.shape[0]
        dev = x_u8.device
        pg = sample_view_params(vc.n_global * B, vc.global_scale, gen, dev, color=vc.color)
        g = normalize(render_views(x_u8.repeat(vc.n_global, 1, 1, 1), pg, vc.global_res))
        l = None
        if vc.n_local > 0:
            pl = sample_view_params(vc.n_local * B, vc.local_scale, gen, dev, color=vc.color)
            l = normalize(render_views(x_u8.repeat(vc.n_local, 1, 1, 1), pl, vc.local_res))
        return g, l

    def train_loss(self, x_u8: torch.Tensor, step: int, normalize):
        if self.aug_gen is None:
            self.aug_gen = torch.Generator(device=x_u8.device).manual_seed(self.cfg.seed + 101)
        g, l = self.make_views(x_u8, self.aug_gen, normalize)
        return self.loss_on(g, l, x_u8.shape[0], step)

    @torch.no_grad()
    def val_loss(self, x_u8: torch.Tensor, chunk_idx: int, normalize) -> dict[str, torch.Tensor]:
        """Forward only (BatchNorm in eval mode): views and directions from fixed seeds per chunk."""
        gen = torch.Generator(device=x_u8.device).manual_seed(10_000 + chunk_idx)
        g, l = self.make_views(x_u8, gen, normalize)
        _, terms = self.loss_on(g, l, x_u8.shape[0], step=1_000_000_000 + chunk_idx)
        return terms

    def after_step(self, tau: float) -> None:  # no EMA in LeJEPA
        return None

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e = self.encoder(x).mean(1)
        return {"backbone": e, "z": self.projector(e)}
