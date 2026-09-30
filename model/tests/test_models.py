import copy

import pytest
import torch

from jepa_viz.augment import identity_params, render_views, sample_view_params
from jepa_viz.config import build
from jepa_viz.ijepa import IJEPA
from jepa_viz.lejepa import LeJEPA
from jepa_viz.masks import MultiBlockMaskSampler
from jepa_viz.train import make_optimizer
from jepa_viz.vit import ViT, patchify, sincos_pos_embed_2d

norm = lambda x: (x - 0.5) / 0.25


def _imgs(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, 3, 32, 32), generator=g, dtype=torch.uint8)


def test_patchify_order():
    x = torch.arange(3 * 32 * 32, dtype=torch.float32).view(1, 3, 32, 32)
    p = patchify(x, 4)
    assert p.shape == (1, 64, 48)
    # token 9 = row 1, col 1; value 0 = channel 0, py 0, px 0 -> pixel (4, 4)
    assert p[0, 9, 0] == x[0, 0, 4, 4]
    # value index c*16 + py*4 + px
    assert p[0, 9, 16 * 2 + 4 * 3 + 1] == x[0, 2, 7, 5]


def test_pos_embed_local_view_spans_frame():
    full = sincos_pos_embed_2d(64, 8, 8)
    assert full.shape == (64, 64)
    # first half encodes the column: token (0, 1) differs from (0, 0) only in the first half
    assert torch.allclose(full[0, 32:], full[1, 32:]) and not torch.allclose(full[0, :32], full[1, :32])
    local = sincos_pos_embed_2d(64, 4, 8)
    # 4x4 local token 0 sits at coordinate 0.5: between full tokens 0 and 1
    assert local.shape == (16, 64)
    assert not torch.allclose(local[0], full[0])


def test_identity_render_is_exact():
    x = _imgs()
    out = render_views(x, identity_params(4, "cpu"), 32)
    assert torch.allclose(out * 255, x.float(), atol=1e-3)


def test_rec_forward_matches_plain_forward():
    vit = ViT().eval()
    x = torch.randn(2, 3, 32, 32)
    rec = {}
    a = vit(x, rec=rec)
    b = vit(x)
    assert torch.allclose(a, b, atol=1e-5)
    assert "blk3.attn" in rec and rec["blk3.attn"].shape == (2, 4, 64, 64)


def _ijepa(**over):
    return IJEPA(build("ijepa", over))


def test_ijepa_target_encoder_gets_no_grad():
    m = _ijepa()
    loss, _ = m.train_loss(_imgs(8), 0, norm)
    loss.backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert all(p.grad is not None for p in m.encoder.parameters())
    assert all(p.grad is not None for p in m.predictor.parameters())


def test_ijepa_no_stopgrad_shared_grads_both_branches():
    m = _ijepa(**{"ema.shared": True, "ema.stopgrad": False})
    assert m.target_encoder is None
    x = norm(_imgs(8).float() / 255)
    masks = MultiBlockMaskSampler(seed=0)(8, 0)
    # gradient through the target branch alone reaches the encoder only when stopgrad is off
    s_y = m.forward_target(x, masks.tgt_idx)
    s_y.sum().backward()
    assert m.encoder.blocks[0].fc1.weight.grad is not None


def test_ijepa_context_only_patches_get_embed_grad():
    m = _ijepa()
    x = norm(_imgs(2).float() / 255).requires_grad_(True)
    masks = MultiBlockMaskSampler(seed=0)(2, 0)
    loss, _ = m.loss_on(x, masks)
    loss.backward()
    g = patchify(x.grad, 4).abs().sum(-1)  # [B, 64] pixel-gradient per patch
    for b in range(2):
        ctx = set(masks.ctx_idx[b].tolist())
        for tok in range(64):
            assert (g[b, tok] > 0) == (tok in ctx)


def test_ema_update_closed_form_and_tau_one_noop():
    m = _ijepa()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(0.1)
    before = copy.deepcopy(m.target_encoder.state_dict())
    ctx = m.encoder.state_dict()
    m.after_step(0.9)
    after = m.target_encoder.state_dict()
    for k in before:
        assert torch.allclose(after[k], 0.9 * before[k] + 0.1 * ctx[k], atol=1e-6)
    snap = copy.deepcopy(after)
    m.after_step(1.0)
    for k in snap:
        assert torch.equal(m.target_encoder.state_dict()[k], snap[k])


def test_ijepa_no_predictor_runs():
    m = _ijepa(**{"pred.enabled": False})
    loss, _ = m.train_loss(_imgs(8), 0, norm)
    loss.backward()
    assert m.predictor is None


def test_lejepa_single_encoder_and_grad_through_every_view():
    m = LeJEPA(build("lejepa"))
    names = [n for n, _ in m.named_modules() if n.endswith("encoder")]
    assert names == ["encoder"]
    B = 8
    x = _imgs(B)
    gen = torch.Generator().manual_seed(0)
    pg = sample_view_params(2 * B, (0.3, 1.0), gen, "cpu")
    pl = sample_view_params(6 * B, (0.05, 0.3), gen, "cpu")
    g = norm(render_views(x.repeat(2, 1, 1, 1), pg, 32)).requires_grad_(True)
    l = norm(render_views(x.repeat(6, 1, 1, 1), pl, 16)).requires_grad_(True)
    loss, terms = m.loss_on(g, l, B, step=0)
    loss.backward()
    assert g.grad.view(2, B, -1).abs().sum(-1).min() > 0
    assert l.grad.view(6, B, -1).abs().sum(-1).min() > 0
    assert set(terms) >= {"loss_inv", "loss_reg", "loss_total"}


@pytest.mark.parametrize("kind", ["sigreg", "vicreg"])
def test_lejepa_reg_kinds(kind):
    m = LeJEPA(build("lejepa", {"reg.kind": kind}))
    loss, _ = m.train_loss(_imgs(8), 0, norm)
    assert torch.isfinite(loss)


def test_optimizer_groups_exclude_1d_from_decay():
    m = _ijepa()
    opt = make_optimizer(m, m.cfg.optim)
    for g in opt.param_groups:
        if g["decay"]:
            assert all(p.ndim > 1 for p in g["params"])
        else:
            assert all(p.ndim == 1 for p in g["params"])


def test_ijepa_linear_predictor_is_position_aware():
    m = _ijepa(**{"pred.linear": True})
    x = norm(_imgs(2).float() / 255)
    masks = MultiBlockMaskSampler(seed=0)(2, 0)
    s_x = m.encoder(x, keep_idx=masks.ctx_idx)
    out = m.predict(s_x, masks)
    Mt, B, Nt = masks.tgt_idx.shape
    assert out.shape == (Mt * B, Nt, 64)
    assert not torch.allclose(out[0, 0], out[0, 1])  # different target positions -> different predictions
    loss, _ = m.loss_on(x, masks)
    loss.backward()
    assert m.predictor.pos_proj.weight.grad is not None


def test_bf16_autocast_keeps_losses_fp32():
    m = LeJEPA(build("lejepa"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = m.train_loss(_imgs(8), 0, norm)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
