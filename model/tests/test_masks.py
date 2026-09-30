import torch

from jepa_viz.masks import MultiBlockMaskSampler


def _sets(mb, b):
    ctx = set(mb.ctx_idx[b].tolist())
    tgts = [set(mb.tgt_idx[i, b].tolist()) for i in range(mb.tgt_idx.shape[0])]
    return ctx, tgts


def test_context_disjoint_from_targets_and_shapes():
    s = MultiBlockMaskSampler(seed=0)
    for step in range(50):
        mb = s(64, step)
        Mt, B, Nt = mb.tgt_idx.shape
        assert (Mt, B) == (4, 64)
        h, w = mb.tgt_hw
        assert Nt == h * w
        assert mb.ctx_idx.shape[1] > s.min_keep
        for b in range(B):
            ctx, tgts = _sets(mb, b)
            assert len(ctx) == mb.ctx_idx.shape[1]  # no duplicates
            for t in tgts:
                assert not (ctx & t)


def test_block_size_bounds():
    s = MultiBlockMaskSampler(seed=0)
    for u in torch.linspace(0, 1, 21).tolist():
        h, w = s.block_size(u, s.pred_scale, s.pred_aspect)
        assert 9 <= h * w <= 16 and 1 <= h <= 8 and 1 <= w <= 8
        hc, wc = s.block_size(u, s.enc_scale, (1.0, 1.0))
        assert hc == wc and hc in (7, 8)


def test_target_blocks_are_rectangles_and_sorted_context():
    s = MultiBlockMaskSampler(seed=1)
    mb = s(16, 3)
    h, w = mb.tgt_hw
    for i in range(mb.tgt_idx.shape[0]):
        for b in range(16):
            idx = mb.tgt_idx[i, b]
            r, c = idx // 8, idx % 8
            assert r.max() - r.min() + 1 == h and c.max() - c.min() + 1 == w
    assert torch.all(mb.ctx_idx[:, 1:] > mb.ctx_idx[:, :-1])


def test_official_edge_behaviour_never_touches_last_row_col():
    s = MultiBlockMaskSampler(seed=0, edge_fix=False)
    for step in range(30):
        mb = s(32, step)
        assert (mb.tgt_idx // 8).max() <= 6 and (mb.tgt_idx % 8).max() <= 6
        assert (mb.ctx_idx // 8).max() <= 6 and (mb.ctx_idx % 8).max() <= 6


def test_edge_fix_reaches_last_row_col():
    s = MultiBlockMaskSampler(seed=0, edge_fix=True)
    rows = set()
    for step in range(30):
        mb = s(32, step)
        rows |= set((mb.tgt_idx // 8).flatten().tolist())
    assert 7 in rows


def test_deterministic_given_seed_and_step():
    a = MultiBlockMaskSampler(seed=5)(8, 11)
    b = MultiBlockMaskSampler(seed=5)(8, 11)
    assert torch.equal(a.ctx_idx, b.ctx_idx) and torch.equal(a.tgt_idx, b.tgt_idx)
