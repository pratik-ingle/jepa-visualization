"""Independent NumPy forward pass over the /data exports -- the reference the TS port must match.

Reads only data/models/<name>/{model.json, final.bin} and data/golden/<name>/{golden.json,
golden.bin}; shares no code with the PyTorch model. Re-implements every op the browser will
(including view rendering and an erf approximation, since JS has no Math.erf) and compares each
intermediate against the recorded goldens.

  python -m jepa_viz.np_forward            # exit code 1 if any tensor is out of tolerance
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"
DT = {"f16": np.float16, "f32": np.float32, "i32": np.int32, "u8": np.uint8}
MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)


def load_bundle(manifest: dict, blob: bytes) -> dict[str, np.ndarray]:
    out = {}
    for k, m in manifest.items():
        dt = np.dtype(DT[m["dtype"]]).newbyteorder("<")
        n = int(np.prod(m["shape"])) if m["shape"] else 1
        out[k] = np.frombuffer(blob, dtype=dt, count=n, offset=m["offset"]).reshape(m["shape"]).astype(
            np.float32 if m["dtype"] in ("f16", "f32") else DT[m["dtype"]])
    return out


# ---- primitive ops (mirrored 1:1 in web/src/model) -------------------------------------------
def erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26, |error| < 1.5e-7."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(np.float32)


def gelu(x):
    return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)


def layernorm(x, w=None, b=None, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if w is not None:
        y = y * w + b
    return y.astype(np.float32)


def linear(x, W, b):
    return (x @ W.T + b).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def patchify(x, p):
    B, C, H, W = x.shape
    return x.reshape(B, C, H // p, p, W // p, p).transpose(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


def sincos_2d(dim: int, g: int, G: int = 8) -> np.ndarray:
    c = (np.arange(g, dtype=np.float64) + 0.5) * G / g - 0.5
    rows, cols = np.repeat(c, g), np.tile(c, g)

    def one(d, pos):
        om = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        o = pos[:, None] * om[None]
        return np.concatenate([np.sin(o), np.cos(o)], 1)

    return np.concatenate([one(dim // 2, cols), one(dim // 2, rows)], 1).astype(np.float32)


class Checker:
    def __init__(self, golden: dict[str, np.ndarray]):
        self.g = golden
        self.rows: list[tuple[str, float, float, bool]] = []

    def __call__(self, key: str, val: np.ndarray, atol=2e-4, rtol=2e-4) -> None:
        if key not in self.g:
            return
        ref = self.g[key]
        val = np.asarray(val, np.float32).reshape(ref.shape)
        err = np.abs(val - ref)
        ok = bool(np.all(err <= atol + rtol * np.abs(ref)))
        self.rows.append((key, float(err.max()), float(np.abs(ref).max()), ok))

    def report(self, name: str) -> bool:
        bad = [r for r in self.rows if not r[3]]
        worst = max(self.rows, key=lambda r: r[1] / (r[2] + 1e-6))
        print(f"{name}: {len(self.rows)} tensors compared, {len(bad)} out of tolerance; "
              f"worst {worst[0]} max|err|={worst[1]:.2e} (max|ref|={worst[2]:.2e})")
        for r in bad[:20]:
            print(f"   FAIL {r[0]}: max|err|={r[1]:.3e} max|ref|={r[2]:.3e}")
        return not bad


# ---- network pieces -----------------------------------------------------------------------------
def block(x, P, pre, heads, chk, rec_pre):
    B, S, W = x.shape
    dh = W // heads
    a = layernorm(x, P[pre + "norm1.weight"], P[pre + "norm1.bias"])
    chk(rec_pre + "ln1", a)
    qkv = linear(a, P[pre + "attn.qkv.weight"], P[pre + "attn.qkv.bias"])
    qkv = qkv.reshape(B, S, 3, heads, dh).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    chk(rec_pre + "q", q), chk(rec_pre + "k", k), chk(rec_pre + "v", v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)
    chk(rec_pre + "scores", scores)
    attn = softmax(scores).astype(np.float32)
    chk(rec_pre + "attn", attn)
    o = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, W)
    chk(rec_pre + "o", o)
    x = x + linear(o, P[pre + "attn.proj.weight"], P[pre + "attn.proj.bias"])
    chk(rec_pre + "x_attn", x)
    m = layernorm(x, P[pre + "norm2.weight"], P[pre + "norm2.bias"])
    chk(rec_pre + "ln2", m)
    u = linear(m, P[pre + "fc1.weight"], P[pre + "fc1.bias"])
    chk(rec_pre + "fc1", u)
    g = gelu(u)
    chk(rec_pre + "gelu", g)
    x = x + linear(g, P[pre + "fc2.weight"], P[pre + "fc2.bias"])
    chk(rec_pre + "x_out", x)
    return x


def vit(x, P, pre, cfg, chk, rec_pre, keep_idx=None):
    m = cfg["model"]
    p = patchify(x, m["patch"])
    chk(rec_pre + "patches", p)
    e = linear(p, P[pre + "patch_embed.weight"], P[pre + "patch_embed.bias"])
    chk(rec_pre + "patch_embed", e)
    t = e + sincos_2d(m["dim"], x.shape[-1] // m["patch"])
    chk(rec_pre + "tokens", t)
    if keep_idx is not None:
        t = np.take_along_axis(t, keep_idx[..., None].astype(np.int64), 1)
        chk(rec_pre + "kept", t)
    for i in range(m["depth"]):
        t = block(t, P, f"{pre}blocks.{i}.", m["heads"], chk, f"{rec_pre}blk{i}.")
    t = layernorm(t, P[pre + "norm.weight"], P[pre + "norm.bias"])
    chk(rec_pre + "out", t)
    return t


def ijepa(P, cfg, G, chk):
    x = G["input.x"]
    ctx_idx, tgt_idx = G["input.ctx_idx"].astype(np.int64), G["input.tgt_idx"].astype(np.int64)
    Mt, B, Nt = tgt_idx.shape
    s_x = vit(x, P, "encoder.", cfg, chk, "ctx.", keep_idx=ctx_idx)
    tpre = "encoder." if cfg["ema"]["shared"] else "target_encoder."
    h = vit(x, P, tpre, cfg, chk, "tgt.")
    h = layernorm(h, eps=1e-5)
    chk("tgt.ln", h)
    s_y = np.concatenate([np.take_along_axis(h, tgt_idx[i][..., None], 1) for i in range(Mt)], 0)
    chk("tgt.s_y", s_y)
    pc = cfg["pred"]
    Nc = s_x.shape[1]
    if pc["enabled"]:
        pos = sincos_2d(pc["dim"], 8)
        c = linear(s_x, P["predictor.embed.weight"], P["predictor.embed.bias"])
        chk("pred.embed", c)
        c = c + pos[ctx_idx]
        chk("pred.ctx", c)
        mq = P["predictor.mask_token"].reshape(1, 1, -1) + pos[tgt_idx.reshape(Mt * B, Nt)]
        chk("pred.mask_q", mq)
        z = np.concatenate([np.tile(c, (Mt, 1, 1)), mq], 1)
        chk("pred.in", z)
        for i in range(pc["depth"]):
            z = block(z, P, f"predictor.blocks.{i}.", pc["heads"], chk, f"pred.blk{i}.")
        z = layernorm(z, P["predictor.norm.weight"], P["predictor.norm.bias"])[:, Nc:]
        chk("pred.norm", z)
        s_hat = linear(z, P["predictor.proj.weight"], P["predictor.proj.bias"])
    else:
        s_hat = np.broadcast_to(np.tile(s_x.mean(1, keepdims=True), (Mt, 1, 1)), (Mt * B, Nt, s_x.shape[-1]))
    chk("pred.out", s_hat)
    d = s_hat - s_y
    ad = np.abs(d)
    loss = np.where(ad < 1.0, 0.5 * d * d, ad - 0.5).mean() if cfg["loss"] != "l2" else (d * d).mean()
    chk("loss", np.float32(loss), atol=1e-5)
    return float(loss)


# ---- LeJEPA ------------------------------------------------------------------------------------
YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
GW = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)


def bilinear_border(img, xs, ys):
    """img [3, S, S]; xs [r], ys [r] continuous pixel coords (centres at k+0.5) -> [3, r, r]."""
    S = img.shape[-1]
    fx = np.clip(xs - 0.5, 0, S - 1)
    fy = np.clip(ys - 0.5, 0, S - 1)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, S - 1), np.minimum(y0 + 1, S - 1)
    wx, wy = (fx - x0)[None, None, :], (fy - y0)[None, :, None]
    a = img[:, y0][:, :, x0]
    b = img[:, y0][:, :, x1]
    c = img[:, y1][:, :, x0]
    d = img[:, y1][:, :, x1]
    return (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy).astype(np.float32)


def render(img_u8, p, i, res):
    x = img_u8.astype(np.float32) / 255.0
    x0, y0, w, h = p["box"][i]
    k = (np.arange(res, dtype=np.float32) + 0.5) / res
    xs = x0 + w - k * w if p["flip"][i] > 0.5 else x0 + k * w
    ys = y0 + k * h
    out = bilinear_border(x, xs, ys)[None]
    if p["jitter"][i] > 0.5:
        y = np.clip(out * p["bright"][i], 0, 1)
        m = (y * GW).sum(1, keepdims=True).mean((2, 3), keepdims=True)
        y = np.clip(p["contrast"][i] * y + (1 - p["contrast"][i]) * m, 0, 1)
        g = (y * GW).sum(1, keepdims=True)
        y = np.clip(p["sat"][i] * y + (1 - p["sat"][i]) * g, 0, 1)
        th = 2 * math.pi * p["hue"][i]
        R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
        M = (np.linalg.inv(YIQ) @ R @ YIQ).astype(np.float32)
        out = np.clip(np.einsum("ij,njhw->nihw", M, y), 0, 1)
    if p["gray"][i] > 0.5:
        out = np.repeat((out * GW).sum(1, keepdims=True), 3, 1)
    if p["sol"][i] > 0.5:
        out = np.where(out >= 0.5, 1 - out, out)
    return out[0]


def lejepa(P, cfg, G, chk):
    imgs = G["input.images"]
    B = imgs.shape[0]
    vc = cfg["views"]
    views = {}
    for tag, n, res in (("global", vc["n_global"], vc["global_res"]), ("local", vc["n_local"], vc["local_res"])):
        p = {k.split(".")[-1]: v for k, v in G.items() if k.startswith(f"input.{tag}.")}
        # views are view-major: view j of image b is row j*B + b
        v = np.stack([render(imgs[i % B], p, i, res) for i in range(n * B)])
        chk(f"input.{tag}01", v, atol=1e-5)
        views[tag] = ((v - MEAN) / STD).astype(np.float32)
    eg = vit(views["global"], P, "encoder.", cfg, chk, "g.").mean(1)
    el = vit(views["local"], P, "encoder.", cfg, chk, "l.").mean(1)
    e = np.concatenate([eg, el])
    chk("pool", e)
    # projector.net = [Linear, BatchNorm1d (eval), ReLU] x (depth-1), then Linear
    x = e
    lin = lambda i, x: linear(x, P[f"projector.net.{i}.weight"], P[f"projector.net.{i}.bias"])
    for j in range(cfg["proj"]["depth"] - 1):
        i = 3 * j
        x = lin(i, x)
        chk(f"proj.{i}.linear", x)
        bn = f"projector.net.{i + 1}."
        x = ((x - P[bn + "running_mean"]) / np.sqrt(P[bn + "running_var"] + 1e-5) * P[bn + "weight"]
             + P[bn + "bias"]).astype(np.float32)
        chk(f"proj.{i + 1}.batchnorm1d", x)
        x = np.maximum(x, 0)
        chk(f"proj.{i + 2}.relu", x)
    i = 3 * (cfg["proj"]["depth"] - 1)
    x = lin(i, x)
    chk(f"proj.{i}.linear", x)
    z = x.reshape(-1, B, x.shape[-1])
    centers = z[: vc["n_global"]].mean(0)
    chk("centers", centers)
    inv = ((centers - z) ** 2).mean()
    A = G["sigreg.A"]
    t = np.linspace(-cfg["reg"]["t_max"], cfg["reg"]["t_max"], cfg["reg"]["t_points"]).astype(np.float32)
    y = z @ A
    chk("sigreg.y", y)
    yt = y[..., None] * t
    re, im = np.cos(yt).mean(1), np.sin(yt).mean(1)
    chk("sigreg.ecf_re", re), chk("sigreg.ecf_im", im)
    phi = np.exp(-0.5 * t**2)
    err = ((re - phi) ** 2 + im**2) * phi
    ep = np.trapezoid(err, t, axis=-1) * B
    chk("sigreg.ep", ep, atol=1e-3, rtol=1e-3)
    lam = cfg["reg"]["lam"]
    loss = (1 - lam) * inv + lam * ep.mean()
    chk("loss", np.float32(loss), atol=1e-4)
    return float(loss)


def check(name: str) -> bool:
    mj = json.loads((DATA / "models" / name / "model.json").read_text())
    P = load_bundle(mj["tensors"], (DATA / "models" / name / "final.bin").read_bytes())
    gj = json.loads((DATA / "golden" / name / "golden.json").read_text())
    G = load_bundle(gj["tensors"], (DATA / "golden" / name / "golden.bin").read_bytes())
    chk = Checker(G)
    loss = ijepa(P, mj["config"], G, chk) if mj["algo"] == "ijepa" else lejepa(P, mj["config"], G, chk)
    print(f"{name}: numpy loss {loss:.6f} vs golden {gj['loss']:.6f}")
    return chk.report(name)


def main() -> None:
    names = sys.argv[1:] or ["ijepa-base", "lejepa-base"]
    ok = all([check(n) for n in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
