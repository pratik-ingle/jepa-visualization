import math

import torch

from jepa_viz.sigreg import ecf, ep_closed_form, epps_pulley, sample_directions, sigreg, t_grid


def test_directions_unit_norm_and_seeded():
    A = sample_directions(32, 64, seed=3, device="cpu")
    assert torch.allclose(A.norm(dim=0), torch.ones(64), atol=1e-6)
    assert torch.equal(A, sample_directions(32, 64, seed=3, device="cpu"))
    assert not torch.equal(A, sample_directions(32, 64, seed=4, device="cpu"))


def test_ecf_matches_gaussian_cf():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(200_000, 1, generator=g)
    t = t_grid()
    re, im = ecf(y, t)
    assert torch.allclose(re[0], torch.exp(-0.5 * t**2), atol=0.01)
    assert torch.allclose(im[0], torch.zeros_like(t), atol=0.01)


def test_quadrature_matches_closed_form():
    g = torch.Generator().manual_seed(1)
    y = torch.cat([torch.randn(300, 2, generator=g), 0.3 * torch.rand(300, 1, generator=g)], 1)
    t_fine = torch.linspace(-12, 12, 4001, dtype=torch.float64)
    quad = epps_pulley(y.double(), t_fine)
    exact = ep_closed_form(y)
    assert torch.allclose(quad.double(), exact, rtol=1e-4, atol=1e-6)


def test_ep_small_for_gaussian_large_for_collapse_scale_uniform():
    g = torch.Generator().manual_seed(2)
    t = t_grid()
    N = 256
    gauss = epps_pulley(torch.randn(N, 64, generator=g), t).mean()
    collapsed = epps_pulley(torch.zeros(N, 8), t).mean()
    scaled = epps_pulley(3 * torch.randn(N, 8, generator=g), t).mean()
    uniform = epps_pulley((torch.rand(N, 8, generator=g) - 0.5) * math.sqrt(12), t).mean()
    # For N(0,1) samples E[N |ecf - phi|^2] = 1 - exp(-t^2), so
    # E[EP] = int (1 - e^{-t^2}) e^{-t^2/2} dt = sqrt(2 pi) - sqrt(2 pi / 3) ~= 1.059, for any N.
    expected = math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)
    assert abs(gauss - expected) < 0.15
    assert collapsed > 50 * gauss and scaled > 10 * gauss
    assert uniform > gauss


def test_sigreg_gradients_flow_to_every_view():
    z = torch.randn(8, 64, 16, requires_grad=True)
    A = sample_directions(16, 32, 0, "cpu")
    sigreg(z, A, t_grid()).backward()
    assert all(z.grad[v].abs().sum() > 0 for v in range(8))
