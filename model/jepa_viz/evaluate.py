"""Frozen-feature evaluation (ATOMS atom 16): k-NN and linear probe."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract(model, x_u8: torch.Tensor, prep, bs: int = 1000) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    outs: dict[str, list[torch.Tensor]] = {}
    for i in range(0, x_u8.shape[0], bs):
        for k, v in model.features(prep(x_u8[i : i + bs])).items():
            outs.setdefault(k, []).append(v.float())
    model.train(was_training)
    return {k: torch.cat(v) for k, v in outs.items()}


@torch.no_grad()
def knn_accuracy(f_tr, y_tr, f_te, y_te, k: int = 20, temp: float = 0.07, n_cls: int = 10,
                 chunk: int = 1000) -> float:
    f_tr = F.normalize(f_tr.float(), dim=1)
    f_te = F.normalize(f_te.float(), dim=1)
    correct = 0
    for i in range(0, f_te.shape[0], chunk):
        sims = f_te[i : i + chunk] @ f_tr.T
        s, idx = sims.topk(k, dim=1)
        votes = torch.zeros(s.shape[0], n_cls, device=s.device)
        votes.scatter_add_(1, y_tr[idx], torch.exp(s / temp))
        correct += (votes.argmax(1) == y_te[i : i + chunk]).sum().item()
    return correct / f_te.shape[0]


def linear_probe(f_tr, y_tr, f_te, y_te, n_cls: int = 10, l2: float = 1e-4, iters: int = 100) -> dict[str, float]:
    """Multinomial logistic regression on standardised features, full-batch L-BFGS."""
    f_tr, f_te = f_tr.float(), f_te.float()
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    a, b = (f_tr - mu) / sd, (f_te - mu) / sd
    if not torch.isfinite(a).all():
        return {"linear_acc": float("nan"), "linear_train_acc": float("nan")}
    W = torch.zeros(a.shape[1], n_cls, device=a.device, requires_grad=True)
    bias = torch.zeros(n_cls, device=a.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, bias], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(a @ W + bias, y_tr) + l2 * (W**2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        tr = ((a @ W + bias).argmax(1) == y_tr).float().mean().item()
        te = ((b @ W + bias).argmax(1) == y_te).float().mean().item()
    return {"linear_acc": te, "linear_train_acc": tr}
