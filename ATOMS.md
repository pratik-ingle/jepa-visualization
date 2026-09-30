# ATOMS — every discrete operation in I-JEPA and LeJEPA

This document breaks both algorithms into **atoms**: the smallest operations the visualisation
shows, from raw pixels to the optimiser step. Each atom lists:
- its math;
- its input and output shapes at the baseline tiny sizes;
- its learnable parameters;
- **whether gradients flow through it**;
- the code that implements it (`model/jepa_viz/…`);
- what the 3D scene draws for it.

It is the contract between the PyTorch reference (`/model`), the exported data (`/data`) and the
browser re-implementation (`/web`). Where this project deliberately departs from a paper or its
official code, the departure is marked **[deviation]** and collected in §D.

---

## 0. Symbols and baseline sizes

| symbol | meaning | baseline |
|---|---|---|
| B | batch size | 256 |
| C, H, W | channels, height, width of an input image | 3, 32, 32 |
| P | patch side (pixels) | 4 |
| G | token-grid side = H/P | 8 |
| N | tokens per full image = G² | 64 |
| d_patch | values per patch = C·P² | 48 |
| D | encoder width | 64 |
| L | encoder depth (transformer blocks) | 4 |
| h, d_h | encoder heads, head width = D/h | 4, 16 |
| D_mlp | encoder MLP hidden width = 4·D | 256 |
| D_p | I-JEPA predictor width | 32 |
| L_p, h_p | predictor depth, heads | 2, 2 |
| M_t | I-JEPA target blocks per image | 4 |
| N_tgt | tokens per target block (same for every block in a batch) | 9 or 12 (3×3 / 4×3; measured) |
| N_ctx | context tokens per image, after target removal and truncation to the batch minimum | 6–30, median 18 (measured, B = 256) |
| V_g, V_l, V | LeJEPA global views, local views, total | 2, 6, 8 |
| r_g, r_l | global / local view resolution (px) | 32, 16 |
| N_l | tokens per local view = (r_l/P)² | 16 |
| D_proj | LeJEPA projector hidden width | 256 |
| K | LeJEPA embedding dimension (projector output) | 32 |
| M | SIGReg slices (random 1-D directions) | 256 |
| T | Epps–Pulley integration points | 17 |
| λ | LeJEPA trade-off, `(1−λ)·inv + λ·SIGReg` | 0.05 |
| τ | I-JEPA EMA momentum | 0.996 → 1.0 |

**Parameter counts** (baseline):

| component | parameters |
|---|---|
| ViT encoder | 203,200 |
| — patch embed | 3,136 |
| — each block | 49,984 (×4) |
| — final LayerNorm | 128 |
| I-JEPA predictor | 29,696 |
| LeJEPA projector | 91,680 (+1,024 BatchNorm running-stat buffers) |

**Totals:**
- **I-JEPA:** 203,200 context encoder + 203,200 EMA target encoder (not optimised) + 29,696 predictor.
- **LeJEPA:** 203,200 + 91,680 = 294,880. That is the whole model: there is no second copy.

Tensor layout convention: batch first, tokens row-major over the grid (token index = row·G + col).

---

## T. The transformer block (shared sub-atoms)

The encoders (atoms 3, 4, 11) and the predictor (atom 5) are stacks of the same pre-LN block,
applied to a sequence **x ∈ ℝ^{S×W}**:
- **Encoder:** width W = D, heads h, MLP 4W.
- **Predictor:** width W = D_p, heads h_p, MLP 4W.

The sequence length S depends on the caller:
- 64 for a full image;
- N_ctx for the context encoder;
- N_ctx + N_tgt for the predictor;
- 16 for a LeJEPA local view.

The same sub-atoms are drawn inside every block, so the walkthrough explains them once (in
atom 3) and then refers back.

| # | op | math | in → out | params | grad |
|---|---|---|---|---|---|
| T1 | LayerNorm 1 | x̂ = (x − μ_row)/σ_row · γ₁ + β₁ | S×W → S×W | γ₁, β₁ ∈ ℝ^W | yes |
| T2 | QKV projection | [q k v] = x̂ W_qkv + b_qkv, split into h heads | S×W → 3 × h×S×d_h | W_qkv ∈ ℝ^{W×3W}, b_qkv | yes |
| T3 | attention scores | a = q kᵀ / √d_h | h×S×d_h → h×S×S | — | yes |
| T4 | softmax | A = softmax over keys (last axis) | h×S×S → h×S×S | — | yes |
| T5 | weighted sum | o = A v, heads concatenated | h×S×S, h×S×d_h → S×W | — | yes |
| T6 | out-proj + residual | x′ = x + o W_o + b_o | S×W → S×W | W_o ∈ ℝ^{W×W}, b_o | yes |
| T7 | LayerNorm 2 | x̃ = LN(x′; γ₂, β₂) | S×W → S×W | γ₂, β₂ | yes |
| T8 | MLP up | u = x̃ W₁ + b₁ | S×W → S×4W | W₁ ∈ ℝ^{W×4W}, b₁ | yes |
| T9 | GELU | g = u·Φ(u) (exact, erf form) | S×4W → S×4W | — | yes |
| T10 | MLP down + residual | x″ = x′ + g W₂ + b₂ | S×4W → S×W | W₂ ∈ ℝ^{4W×W}, b₂ | yes |

LayerNorm uses ε = 1e-6. After the last block a **final LayerNorm** (γ_f, β_f) is applied.

Code: `vit.py: Block`, `vit.py: ViT.forward(..., rec=None)`. Passing `rec` records every T-output.

**Scene:**
- One stacked "layer slab" per block:
  - LN outputs as S×W sheets;
  - Q/K/V as three h×S×d_h stacks;
  - each head's S×S attention map as a square sheet (the head selector picks which is shown opaque);
  - the MLP hidden as a wide S×4W sheet.
- Residual adds are drawn as the stream passing *through* the slab.
- Weights are drawn as thin blocks beside the op that uses them.

---

## I-JEPA path (atoms 1–9)

### Atom 1 — Input → patchify → patch embedding + positional embedding

| sub | op | math | in → out | params | grad |
|---|---|---|---|---|---|
| 1a | normalise | x = (img/255 − mean_c)/std_c (CIFAR-10 channel stats) | B×3×32×32 uint8 → float | — | no (input) |
| 1b | patchify | cut into G×G patches of P×P; each flattened in (c, py, px) order, i.e. index = c·P² + py·P + px | B×3×32×32 → B×N×d_patch = B×64×48 | — | — |
| 1c | linear embed | e = patch · W_e + b_e | B×64×48 → B×64×D | W_e ∈ ℝ^{48×64}, b_e ∈ ℝ^{64} (3,136) | yes (context tokens only, see atom 7) |
| 1d | + positional | e ← e + π(r, c) | B×64×D → B×64×D | none (fixed) | — |

**Positional embedding π.** It is fixed 2D sin-cos, as in I-JEPA / MAE:
- The first D/2 channels encode the column coordinate and the last D/2 the row coordinate.
- Each half is `[sin(p·ω_i), cos(p·ω_i)]` for i = 0…D/4−1, with ω_i = 10000^(−i/(D/4)).

**Continuous coordinates [deviation, see D3].** Coordinates are evaluated at continuous positions.
A view with g×g tokens that spans the reference G×G frame gives token i the coordinate
`(i + 0.5)·G/g − 0.5`:
- A global view (g = 8) gets coordinates 0…7.
- A 16-px local view (g = 4) gets 0.5, 2.5, 4.5, 6.5.

This is the sin-cos equivalent of the positional-embedding interpolation that DINO/LeJEPA use
for small crops. It is exact and easy to reproduce in the browser.

**Ordering.** As in the official code, positional embeddings are added **before** masking. Every
token therefore carries its own position into whichever encoder sees it.

**Initialisation matters here.**
- W_e keeps PyTorch's default init (kaiming-uniform). This matches the official I-JEPA, whose conv patch embedding is never re-initialised. All other Linear layers use trunc-normal(0.02).
- With std-0.02 weights on W_e as well, patch content (std ≈ 0.14) is swamped by π (RMS ≈ 0.7). Every token then mostly encodes *where* it is, and the predictor can hit the targets from position alone.
- The result is a tiny loss with near-collapsed features (random-pair cosine 0.95, RankMe ≈ 2). This **positional shortcut** was observed in a smoke test and is worth showing in the walkthrough.

Code: `vit.py: patchify`, `vit.py: sincos_pos_embed_2d`, `vit.py: ViT.embed`.

**Scene:**
- The 32×32 RGB image, with an 8×8 patch grid overlay.
- The patches fly out into a 64×48 sheet, meet the 48×64 W_e block, and become the 64×64
  embedding sheet.
- The 64×64 π sheet (a visible sinusoidal banding) is added on top.

### Atom 2 — Multi-block masking: context and target index sets

No learnable params. Gradients: none (discrete sampling). Follows the official
`src/masks/multiblock.py`.

1. **Block sizes, once per batch** (seeded by the step counter):
   - Draw one number `u ~ U(0,1)`.
   - Target scale s = 0.15 + 0.05u and aspect ar = 0.75 + 0.75u. Scale and aspect share **the
     same u**, so they are correlated. This is an official-code quirk and is reproduced.
   - `k = int(N·s)`, `h = round(√(k·ar))`, `w = round(√(k/ar))`.
   - The context block draws u again: scale 0.85–1.0, aspect 1.
2. **Target blocks, M_t = 4 per image, placed independently:** a random top-left corner, then a
   rectangle of h×w tokens.
   - Target blocks may overlap each other.
   - Every target block in the batch has exactly h·w tokens, so N_tgt = h·w (9–12 at baseline).
3. **Context block, 1 per image:**
   - A random 7×7 or 8×8 square.
   - Multiplied by the complement of every target block, which removes all target tokens.
   - Rejected and resampled if it keeps ≤ `min_keep` = 4 tokens **[deviation D2: official 10 at 14×14]**.
   - After 20 failures, the last target constraint is dropped.
4. **Batch truncation:**
   - Index sets are sorted (`nonzero`) and truncated to the **minimum** length in the batch, so
     tensors are rectangular.
   - This only ever shortens *context* sets (target sets are equal-sized).
   - Because the indices are sorted, truncation always drops the bottom-right-most context tokens.
     This is an official quirk and is reproduced.
5. **Edge fix [deviation D1]:**
   - The official sampler draws `top = randint(0, G − h)`, whose upper bound is exclusive, and caps
     blocks at G−1. Together, **no block ever touches the last row or column**.
   - At 14×14 the context block (13×13) is then always pinned to the top-left, and 27 of 196
     tokens are never context or target. At 8×8 it would lose 15 of 64 tokens (23%).
   - We draw `top ∈ [0, G − h]` and allow blocks up to G. `mask.edge_fix=false` restores the
     official behaviour.

| output | shape |
|---|---|
| `ctx_idx` | B×N_ctx (int) |
| `tgt_idx` | M_t × B×N_tgt (int) |
| sampled sizes | (h, w) target, (h_c, w_c) context |

Code: `masks.py: MultiBlockMaskSampler.__call__`.

**Scene:**
- The 8×8 grid, with the four target blocks outlined in four colours and the context square in grey.
- An animation removes overlap: target cells are carved out of the context square.
- Truncated tokens are shown hatched.
- Live controls: seed, target scale and count, and dragging blocks.

### Atom 3 — Context encoder f_θ on visible context tokens only

- **Gather:** `z₀ = gather(e + π, ctx_idx)`, taking B×64×D → B×N_ctx×D.
- **Blocks:** L blocks (T1–T10, width D) plus the final LN, giving **s_x ∈ ℝ^{B×N_ctx×D}**.
- The masked-out tokens are *absent*, not zeroed. Attention is N_ctx×N_ctx.

| params | grad |
|---|---|
| θ = all encoder weights (203,200) | **yes**: the context encoder is trained by backprop |

Code: `ijepa.py: IJEPA.forward_context`.

**Scene:**
- A short, narrow column of tokens (only the visible ones) runs up the context-encoder tower.
- Its attention maps are N_ctx×N_ctx.

### Atom 4 — Target encoder f_θ̄ (EMA copy) on the full image → target embeddings

1. **Encode the full image:** `h = f_θ̄(e + π)` on **all 64 tokens**, giving B×64×D.
2. **Normalise:** `h ← LayerNorm(h)` over the feature dimension, **without affine parameters**. This
   is on top of the encoder's own final LN, as in the official `train.py`.
3. **Gather targets:** for each block i, `s_y^{(i)} = gather(h, tgt_idx_i)`, giving B×N_tgt×D for
   i = 1…M_t.
4. **Stop-gradient:** the whole atom runs under `torch.no_grad()`.

| params | grad |
|---|---|
| θ̄ = a separate copy of the encoder weights, **not in the optimiser**, updated only by EMA (atom 9) | **none**: `stop-grad`, and θ̄ never receives gradient |

Code: `ijepa.py: IJEPA.forward_target`.

**Scene:**
- A second full-width tower beside the context tower, drawn in a cooler colour.
- Its full 64×64 attention maps contrast with the context tower's narrower ones.
- A "no-gradient" barrier sits at its output.

### Atom 5 — Predictor g_φ: context embeddings + mask tokens → predicted target embeddings

**Steps**, for each target block i (all blocks are batched along the batch axis, giving M_t·B
sequences):

| sub | op | in → out | params |
|---|---|---|---|
| 5a | predictor embed: `c = s_x W_pe + b_pe` | B×N_ctx×D → B×N_ctx×D_p | W_pe ∈ ℝ^{D×D_p}, b_pe (2,080) |
| 5b | `c ← c + π_p(ctx positions)` (sin-cos at width D_p) | — | none |
| 5c | mask queries: `m_i = mask_token + π_p(tgt_idx_i)` | → B×N_tgt×D_p | mask_token ∈ ℝ^{D_p} (32) |
| 5d | concat `[c ; m_i]`, context repeated once per block | → (M_t·B)×(N_ctx+N_tgt)×D_p | — |
| 5e | L_p blocks (T1–T10 at width D_p, h_p heads) | same | 25,408 |
| 5f | final LN, keep the last N_tgt tokens | → (M_t·B)×N_tgt×D_p | γ, β (64) |
| 5g | predictor proj: `ŝ_y = · W_pp + b_pp` | → (M_t·B)×N_tgt×D | W_pp ∈ ℝ^{D_p×D}, b_pp (2,112) |

**Notes:**
- Each target block is its **own sequence**, so target blocks never attend to each other. That is
  "one pass per target block", implemented as one batched pass.
- The predictor is deliberately **narrow**: D_p = D/2, like the official 384-wide predictor behind a
  768–1280-wide encoder.
- Grad: **yes**. Parameters φ are trained, and gradient continues back into s_x.

Code: `ijepa.py: Predictor.forward`.

**Scene:**
- Context tokens (from atom 3) and M_t sets of mask-token queries, each coloured by its target
  block, feed a small tower.
- Its (N_ctx+N_tgt)² attention maps show the mask queries attending into the context.
- Clicking a target token traces which context tokens it attends to.

### Atom 6 — Loss: predicted vs. target embeddings

- **Official code:** `L = smooth_L1(ŝ_y, s_y)` with β = 1, averaged over every element (blocks ×
  batch × tokens × channels).
- **Paper:** `(1/M_t) Σ_i Σ_{j∈B_i} ‖ŝ_{y_j} − s_{y_j}‖²`.
- Because the targets are layer-normed, most errors are < 1, where smooth-L1 = ½·MSE.
- **Baseline:** smooth-L1. `loss.type=l2` switches to MSE.

| in | out | params | grad |
|---|---|---|---|
| (M_t·B)×N_tgt×D twice | scalar | — | yes (into ŝ_y only; s_y is a constant) |

Code: `ijepa.py: ijepa_loss`.

**Scene:**
- The predicted and target sheets side by side, with an error sheet between them (colour = error).
- The errors reduce to the scalar loss, whose value is plotted.

### Atom 7 — Backward pass

- Autograd computes ∂L/∂φ (predictor) and ∂L/∂θ (context encoder, including the patch embed W_e).
- **θ̄ receives no gradient.**
- **Where the gradient reaches:**
  - W_e receives gradient **only through context tokens**. Patches that were masked out, or that
    are target tokens, contribute nothing to the context branch.
  - The positional embedding is fixed, so it receives no gradient.
  - The mask token receives gradient.

| params touched | not touched |
|---|---|
| θ, φ | θ̄ |

Code: `ijepa.py: IJEPA.train_step` (`loss.backward()`).

**Scene:**
- Per-token gradient magnitudes ‖∂L/∂activation‖ flow backwards down the predictor tower and then
  the context tower.
- Per-weight gradient norms light up the weight blocks.
- The target tower stays dark. A "∅" badge shows `grad is None` for θ̄.

### Atom 8 — Optimiser step (AdamW + schedules)

**Update**, for each trained param p with gradient g at step k:
- m ← β₁m + (1−β₁)g
- v ← β₂v + (1−β₂)g²
- p ← p − lr_k · ( m̂/(√v̂ + ε) + wd_k·p )
- Bias correction: m̂ = m/(1−β₁^k), v̂ = v/(1−β₂^k).
- Hyperparameters: β = (0.9, 0.999), ε = 1e-8.

**Schedules and settings:**
- **LR:** linear warmup from 0.2·lr_peak to lr_peak over `warmup` steps, then cosine to 1e-3·lr_peak.
- **WD:** cosine from 0.04 to 0.4 (official I-JEPA).
- **No weight decay** on biases and on all 1-D params (LayerNorm γ/β). The mask token is 3-D, so it
  is decayed, as in the official code.
- No gradient clipping.

| params | grad |
|---|---|
| updates θ, φ; optimiser state (m, v) is the same size as θ ∪ φ | n/a (not differentiated) |

Code: `schedules.py`, `train.py: make_optimizer`.

**Scene:**
- Each weight block shimmers by its update magnitude.
- A small timeline shows lr_k and wd_k.

### Atom 9 — EMA update of the target encoder

- **Update:** θ̄ ← τ_k·θ̄ + (1−τ_k)·θ, applied element-wise to every encoder tensor after the
  optimiser step.
- **Schedule:** τ_k is linear from τ₀ = 0.996 to τ₁ = 1.0 over the run, so the target freezes by the end.

| params | grad |
|---|---|
| updates θ̄ (no learnable params of its own) | none: a no-grad, in-place update |

Code: `ijepa.py: IJEPA.ema_update`.

**Scene:**
- Target-tower weight blocks morph toward the context-tower weights.
- A τ slider shows how far one step moves them.
- The ‖θ̄ − θ‖ gap is plotted over training.

---

## LeJEPA path (atoms 10–14)

### Atom 10 — Multi-crop augmentation

For each image, draw V_g = 2 **global** and V_l = 6 **local** views:
- **Crop box:** RandomResizedCrop, area scale ∈ (0.3, 1) for global and (0.05, 0.3) for local, log-uniform
  aspect ∈ (3/4, 4/3). Resize bilinearly to 32×32 (global) or 16×16 (local).
- **Photometric:** horizontal flip (p = 0.5); colour jitter (brightness 0.4, contrast 0.4, saturation
  0.2, hue 0.1) with p = 0.8; grayscale (p = 0.2); solarize (p = 0.2, threshold 0.5).
- These follow the official LeJEPA README recipe. **No Gaussian blur** **[deviation D5]**: at 32 px, a
  blur is a large fraction of the image.

**Output:**
- globals: B×V_g×3×32×32;
- locals: B×V_l×3×16×16.

Every view's crop box (x₀, y₀, w, h), flip bit and jitter parameters are **explicit tensors**, so the
browser can reproduce any view exactly.

No params. Gradients: no (the inputs are data).

Code: `augment.py: sample_views`, `augment.py: render_views`.

**Scene:**
- The source image with 8 coloured crop rectangles.
- Each crop flies out, is resized, and becomes a small image tile (32 px or 16 px).

### Atom 11 — Shared encoder f_θ → projector g_φ → embeddings z

| sub | op | in → out | params | grad |
|---|---|---|---|---|
| 11a | patchify + embed + π (atom 1). Local views get 4×4 tokens at continuous coordinates (D3) | global: B·V_g×64×D; local: B·V_l×16×D | W_e, b_e | yes |
| 11b | **the same** encoder f_θ (T1–T10 ×L + final LN) on every view | same shapes | θ (203,200), **one copy** | **yes, through every view** |
| 11c | mean-pool over tokens | → e ∈ ℝ^{V·B×D} | — | yes |
| 11d | projector: Linear(D,256) → BN → ReLU → Linear(256,256) → BN → ReLU → Linear(256,K) | V·B×D → **z ∈ ℝ^{V·B×K}** | φ (91,680); BN running stats (buffers) | yes |

**Notes:**
- Global and local views have different token counts, so they go through the encoder as two calls.
  The **projector runs once on all V·B pooled vectors**, so BatchNorm normalises over all views
  jointly.
- LeJEPA's paper finds 3-layer projectors best (Table 4). `proj.depth=2` (Linear-BN-ReLU-Linear) is
  the spec's 2-layer option.
- The **backbone features e** are what downstream probes use. **z** is where the loss acts.

Code: `lejepa.py: LeJEPA.forward_views`, `lejepa.py: Projector`.

**Scene:**
- Eight view tiles enter **one** encoder tower, drawn once, with the views stacked through it.
- The tower pools, then projects to eight K-vectors per image.
- One set of weights is highlighted. There is no teacher tower, unlike the I-JEPA branch.

### Atom 12 — Invariance (prediction) loss

- **Centroid:** μ_n = (1/V_g) Σ_{v ≤ V_g} z_{n,v}, the centre of the **global** views of image n.
- **Loss:** L_inv = (1/(B·V·K)) Σ_n Σ_{v=1}^{V} ‖μ_n − z_{n,v}‖².
- The loss covers **all** V views, globals included **[deviation from the task spec, which says "each
  local view"; this follows the paper's Eq. 8 and official listing]**.
- The official `.mean()` also divides by K.

| in | out | params | grad |
|---|---|---|---|
| z: V×B×K | scalar | — | **yes, into every view and into μ** (no stop-gradient anywhere) |

Code: `lejepa.py: invariance_loss`.

**Scene:**
- The 8 z-vectors per image as points.
- The global centroid as a star.
- Error springs from each point to the star.

### Atom 13 — SIGReg: sketched isotropic Gaussian regularisation (Epps–Pulley on random 1-D slices)

Computed **separately for each view v** on Z_v ∈ ℝ^{B×K}, then averaged over views.

| sub | op | in → out |
|---|---|---|
| 13a | directions: A = randn(K, M) with each column scaled to unit norm. Seeded by the global step, so it is **resampled every step** | → K×M |
| 13b | project: Y = Z_v A, B samples on each of M 1-D slices | B×K → B×M |
| 13c | empirical characteristic function on the grid t_j ∈ linspace(−5, 5, T=17): φ̂_m(t) = (1/B) Σ_n e^{i t Y_nm} = mean cos(tY) + i·mean sin(tY) | → M×T (complex) |
| 13d | weighted squared error vs. N(0,1): err_m(t) = \|φ̂_m(t) − e^{−t²/2}\|² · e^{−t²/2} | → M×T |
| 13e | Epps–Pulley statistic: EP_m = B · trapz_t(err_m) | → M |
| 13f | average: SIGReg_v = (1/M) Σ_m EP_m; SIGReg = (1/V) Σ_v SIGReg_v | → scalar |

**Facts the visualisation relies on:**
- **Why slices work.** By the Cramér–Wold theorem, a distribution is determined by all of its 1-D
  projections. Matching every slice to N(0,1) therefore forces z toward the isotropic Gaussian
  N(0, I_K). Random slices resampled every step cover all directions over training. LeJEPA Fig. 7
  shows that as few as 16 resampled directions beat thousands of fixed ones.
- **Cost.** It is **linear in B, in K and in M**: O(B·K·M + B·M·T). It needs no K×K covariance, unlike
  VICReg, and no B×B similarity matrix, unlike contrastive losses.
- **Why the target is an isotropic Gaussian.**
  - LeJEPA **Theorem 1**: among embedding distributions with a fixed scalar covariance budget, the
    isotropic Gaussian uniquely minimises the integrated squared bias of k-NN and kernel probes.
  - **Lemmas 1–2**: anisotropy amplifies the bias and variance of linear probes.
- **The statistic.** It is the Epps & Pulley (1983) normality test statistic. It has bounded
  gradients: |∂EP/∂z| ≤ 4σ²/N (LeJEPA Thm 4).

| params | grad |
|---|---|
| none (A is sampled, not learned) | **yes**, into every z_v |

Code: `sigreg.py: sample_directions`, `sigreg.py: epps_pulley`, `sigreg.py: sigreg`.

**Scene:**
- The K-dimensional cloud of z, with M direction arrows sweeping through it.
- For a chosen slice:
  - the 1-D histogram of Y over the N(0,1) density;
  - the real and imaginary parts of φ̂(t) against e^{−t²/2};
  - the shaded weighted error area;
  - the EP value.
- Bars for all slices average into SIGReg.
- "Resample directions" redraws A live.

### Atom 14 — Total loss, single backward pass, single optimiser, no EMA

- **Total loss:** L = (1−λ)·L_inv + λ·SIGReg. This is the paper's Eq. 9 and official code, with λ =
  0.05 as the paper default.
- **Relation to the spec's form:** the task spec writes `inv + λ′·SIGReg`. That is the same objective
  with λ′ = λ/(1−λ) and the learning rate rescaled by (1−λ).
- **Update:** one `backward()` and one AdamW step (atom 8's update rule).
- **Schedules:** LR warmup + cosine, **fixed** weight decay 0.05 (LeJEPA uses no WD schedule).
- **No EMA, no stop-gradient, no teacher, no predictor.**

| params | grad |
|---|---|
| θ (encoder), φ (projector) | yes, from both terms, through every view |

Code: `lejepa.py: LeJEPA.train_step`.

**Scene:**
- The two loss terms meet at a weighted sum (the λ dial).
- Gradients flow back through **all eight** view paths into the **one** encoder.
- Visually, the only weights anywhere are the one encoder and the projector.

---

## Shared (atoms 15–16)

### Atom 15 — Collapse and geometry diagnostics (no gradients; logged, never trained on)

Computed on a fixed set of 5,000 held-out images (the val split). Each metric is computed on
**pooled backbone features** (I-JEPA: target encoder **and** context encoder; LeJEPA: encoder) and,
for LeJEPA, also on **z**.

Let F ∈ ℝ^{n×d} be the features, F̄ the column-centred features, Σ = F̄ᵀF̄/(n−1), and λ₁ ≥ … ≥ λ_d
the eigenvalues of Σ.

| metric | definition | collapse signature |
|---|---|---|
| per-dim variance | diag(Σ) (vector) and its mean | → 0 |
| covariance spectrum | λ₁…λ_d (sorted, vector) | a few large, rest ≈ 0 |
| effective rank (RankMe) | exp(−Σ_k p_k log p_k), where p_k = σ_k/‖σ‖₁ + ε, σ = singular values of F (uncentred, as in RankMe), ε = 1e-7 | → 1 |
| isotropy | λ_d/λ₁ (min/max, the spec's score); λ_d/mean(λ); **participation ratio** (Σλ)²/(d·Σλ²) ∈ (0, 1] | → 0 (anisotropic); 1 = isotropic |
| slice Gaussianity | mean Epps–Pulley statistic over 64 **fixed** random unit directions: (a) raw, (b) each slice standardised | large |
| random-pair cosine | mean and std of cos(f_i, f_j) over 4,096 random pairs (raw, uncentred features) | mean → 1, std → 0 |
| spread | d·mean(diag Σ) / mean(‖f‖)²: the fraction of feature energy that varies across images | → 0 |

**Collapse call.** A run counts as completely collapsed when spread < 10⁻³ or RankMe < 1.5.
- Raw random-pair cosine is *not* used for this call. Mean-pooled LayerNorm features share a large common offset, which pushes raw cosine toward 1 even for a healthy encoder. For example, the LeJEPA baseline backbone scores cosine 0.98 and spread 0.02 while reaching 65% kNN accuracy.
- Anisotropy is reported as a number: effective dimensions = participation ratio × d. It is not a thresholded label.

**Why the participation ratio.** For LayerNorm'd features with β ≈ 0, every token's features sum to ≈ 0. Mean-pooled features therefore have one near-null direction, and min/max ≈ 0 regardless of how well spread the rest is. The participation ratio is insensitive to a single null direction, so it is the isotropy axis used in the probe-accuracy-vs-isotropy plots. min/max is still logged.

Reference: RankMe, Garrido et al., ICML 2023, arXiv:2210.02885.

Code: `diagnostics.py`.

**Scene:**
- The PCA-3D point cloud of features, scrubbable over checkpoints.
- The spectrum as a bar chart.
- A collapsed run visibly shrinks to a point (complete collapse) or flattens to a line or plane
  (dimensional collapse).

### Atom 16 — Evaluation: frozen encoder → pooled features → k-NN and linear probe

1. **Freeze:** gradients off. Features are the **mean-pooled final-LN tokens** of the full 32×32 image:
   - I-JEPA: the **target** encoder, as in the paper; the context encoder is also logged.
   - LeJEPA: the encoder backbone **e**, not z.
2. **Labelled data:**
   - Probe-train: a fixed class-balanced 10,000-image subset of CIFAR-10 train (the "small labelled set").
   - Probe-test: 5,000 CIFAR-10 test images, disjoint from the 5,000 val images used in atom 15.
3. **k-NN:** cosine similarity, k = 20, temperature-weighted votes, exp(s/0.07), as in DINO.
4. **Linear probe:** multinomial logistic regression on standardised features. Full-batch L-BFGS
   with a small L2 penalty.

Gradients: none into the encoder. The probe's own parameters are fitted separately.

"Testing" a self-supervised model means **testing the representation** (what these probes read
off frozen features), **not the pretext prediction**. A low I-JEPA prediction loss can coexist with
a collapsed, useless representation (see the ablations).

Code: `evaluate.py: extract_features`, `evaluate.py: knn_accuracy`, `evaluate.py: linear_probe`.

**Scene:**
- The frozen encoder is greyed.
- The PCA-3D feature cloud is coloured by class.
- The probe accuracy is shown against the isotropy of the same features.

---

## A. Ablation definitions (what exactly is switched off)

| name | definition | why it is defined this way |
|---|---|---|
| **I-JEPA without stop-gradient** | θ̄ ≡ θ (τ = 0: shared weights) and the target branch is **not** detached, so gradient flows into the encoder through both branches | With a separate EMA encoder, removing `no_grad` changes nothing: θ̄ is not in the optimiser, so the gradient has nowhere to go. The meaningful ablation is shared weights with gradient through both paths. The encoder can then make targets trivially predictable, and it collapses. |
| **I-JEPA τ = 0 *with* stop-gradient** | θ̄ ≡ θ each step, but targets are detached (SimSiam-style) | Isolates what stop-grad alone buys without the EMA. |
| **I-JEPA without predictor** | ŝ_y = the mean of the context tokens s_x, projected by an identity map and broadcast to every target position (no learned predictor and **no positional conditioning**) | The spec's "predictor removed". Without a position-aware predictor, the only way to predict every target token from the same vector is to make all tokens alike. |
| **τ too low** | τ fixed at 0.9 (vs. 0.996 → 1) | The target chases the online network too fast. |
| **LeJEPA λ = 0** | invariance only | Nothing prevents z from being constant: complete collapse. |
| **LeJEPA M too small** | M = 2 directions, **fixed** for the whole run (`sigreg.resample=false`) | Only 2 of K = 32 directions are ever constrained, so the remaining directions may collapse (dimensional collapse, visible in the spectrum). The companion run with M = 2 **resampled** every step tests the paper's Fig. 7 claim. |
| **λ too high** | λ = 0.95 | Embeddings become nearly perfectly Gaussian, but the views need not agree, so the features carry little image information. |

## C. Contrasts

**JEPA vs. masked autoencoders (MAE; He et al., 2022).**
- MAE masks patches and reconstructs their **pixels** with a decoder. The loss therefore spends
  capacity on everything that is hard to predict: exact textures, noise, lighting. The resulting
  features usually need fine-tuning to be linearly useful.
- A JEPA predicts the **representation** of the masked region, computed by a target encoder. That
  encoder is free to discard unpredictable detail, so the objective is closer to "predict what is
  there" than "predict how it looks". I-JEPA reaches strong linear-probe results with less compute
  than MAE.
- The price is that a representation-space target can move. A pixel target is fixed, but an encoder
  that maps everything to a constant predicts itself perfectly. **Collapse is a JEPA problem that MAE
  does not have.** Both variants shown here are answers to it.

**JEPA vs. contrastive methods (SimCLR, MoCo).**
- Contrastive methods prevent collapse **explicitly with negatives**: every image's embedding is
  pushed away from the other images' embeddings. This needs large batches or memory queues, costs
  O(B²) similarities, and its invariances come from hand-designed augmentations.
- I-JEPA uses **no negatives and no photometric augmentations** (crop only). Masking in the token
  grid is its only "view".
- LeJEPA also uses no negatives, but it does use augmented views. Its anti-collapse term is a
  **distribution-level** constraint: match N(0, I) along random slices. It costs O(B), not a pairwise
  repulsion.

**JEPA vs. BYOL / DINO.**
- BYOL and DINO share I-JEPA's machinery: an EMA teacher and stop-gradient, plus a predictor in BYOL
  and centring/sharpening in DINO.
- But they compare **global embeddings of two augmented views** (an invariance objective). Their
  "prediction" is not conditioned on *where*.
- I-JEPA's predictor receives **target positions** (mask tokens + positional embeddings) and predicts
  spatially localised content. It learns from the image's own structure rather than from augmentation
  invariance.
- LeJEPA is structurally closer to DINO (global/local multi-crop, invariance to a global centre), but
  it removes the teacher, EMA, centring and stop-gradient entirely.

**SIGReg vs. VICReg variance/covariance terms vs. EMA + stop-gradient.**
- **EMA + stop-gradient** is an **implicit, dynamical** defence. No term in the loss forbids collapse.
  Training avoids it only while the predictor adapts faster than the slowly moving target. That makes
  it sensitive to τ, predictor capacity and learning rate, and it makes the loss value a poor guide to
  representation quality.
- **VICReg** (Bardes, Ponce & LeCun, 2022) is **explicit**:
  - a hinge keeps each dimension's standard deviation ≥ 1 (variance);
  - off-diagonal covariances are penalised (covariance).
  - It matches the first two moments. It costs O(B·K²) and says nothing about higher moments, so
    non-Gaussian "shortcut" distributions satisfy it.
- **SIGReg** is also explicit, but it matches the **whole distribution** to N(0, I) through 1-D slices.
  - It costs O(B·K·M), linear in K.
  - It has one trade-off knob, λ.
  - LeJEPA Thm 9 shows VICReg-like moment matching is SIGReg's limit with a moment-based test, and
    Thm 3 explains why finitely many moments are not enough.
- `reg=vicreg` swaps SIGReg for VICReg's variance + covariance terms in the explorer.

## V. Where V-JEPA differs (notes for the commentary)

- **Tokens:** video clips of 16 frames are cut into 2×16×16 **tubelets** (space-time patches).
- **3D multi-block masks:** blocks are extended through time as tubes. There are two mask families,
  each a separate prediction task:
  - short-range: 8 small blocks, spatial scale 0.15;
  - long-range: 2 large blocks, spatial scale 0.7.
  - Together they mask roughly 90% of the clip.
- **Loss and teacher:** L1 regression in feature space to layer-normed EMA-teacher targets. EMA goes
  0.998 → 1.
- **Evaluation:** a frozen backbone read by an **attentive probe** (cross-attention pooling), not
  mean-pooling.
- **V-JEPA 2:**
  - adds 3D-RoPE and a ViT-g encoder (≈1B params);
  - uses a constant EMA;
  - adds **V-JEPA 2-AC**: an action-conditioned predictor trained on robot video, used for
    model-predictive planning by minimising the distance to a goal embedding.
- **The anti-collapse answer is still EMA + stop-gradient.** LeJEPA's SIGReg is a drop-in alternative.

## D. Deviations from papers and official code (tiny scale)

| # | what | official | here | reason |
|---|---|---|---|---|
| D1 | mask block placement | `top = randint(0, G−h)` (exclusive), h ≤ G−1: last row/column never used | top ∈ [0, G−h], h ≤ G; `mask.edge_fix=false` restores | at 8×8 the official rule discards 15/64 tokens every step |
| D2 | `min_keep` | 10 (of 196) | 4 (of 64) | scaled to grid size (also the collator's class default) |
| D3 | positional embedding for small views | bicubic interpolation of the learned or fixed grid (DINO/LeJEPA) | fixed sin-cos evaluated at continuous coordinates | exact, parameter-free, easy to reproduce in the browser |
| D4 | image and view sizes | 224 px, 16-px patches; locals 96–98 px | 32 px, 4-px patches (8×8 grid); locals 16 px (4×4 grid) | CIFAR-10 |
| D5 | LeJEPA augmentations | + Gaussian blur p=0.5 | no blur | a blur kernel is a large fraction of a 32-px image |
| D6 | LeJEPA loss form | (1−λ)·inv + λ·SIGReg | same (the spec's `inv + λ·SIGReg` differs only by rescaling) | follow the paper |
| D7 | LeJEPA invariance | all V views vs. the global centroid | same (the spec says local views only) | follow the paper |
| D8 | model scale | ViT-B…H, predictor 384×6–12; projector 2048-2048-K | ViT D=64 ×4, predictor 32×2, projector 256-256-32 | in-browser inference, readable tensors |
| D9 | LeJEPA SIGReg defaults | M = 1024 (listing: 256), T = 17 on [−5, 5] | M = 256, T = 17 on [−5, 5] | K = 32 is small; Fig. 7 shows ≥16 resampled slices suffice |
| D10 | training numerics | bf16 autocast (I-JEPA official); SDPA attention | bf16 autocast for the networks; losses, SIGReg, EMA, optimiser state and all evaluation in fp32; attention as explicit softmax(QKᵀ/√d_h)V | fused SDPA kernels are slower at d_h = 16 and ≤ 64 tokens; the explicit form is exactly what the goldens record. Exports and goldens are fp32 compute on fp16-rounded weights |

## R. References

- **I-JEPA:** M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, N. Ballas.
  *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023.
  arXiv:2301.08243. Code: github.com/facebookresearch/ijepa.
- **V-JEPA:** A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, N. Ballas.
  *Revisiting Feature Prediction for Learning Visual Representations from Video.* 2024.
  arXiv:2404.08471.
- **V-JEPA 2:** M. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning.* 2025. arXiv:2506.09985.
- **LeJEPA:** R. Balestriero, Y. LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without
  the Heuristics.* 2025. arXiv:2511.08544. Code: github.com/galilai-group/lejepa.
- **RankMe:** Q. Garrido, R. Balestriero, L. Najman, Y. LeCun. *RankMe: Assessing the Downstream
  Performance of Pretrained Self-Supervised Representations by Their Rank.* ICML 2023.
  arXiv:2210.02885.
- **Epps & Pulley:** T. W. Epps, L. B. Pulley. *A test for normality based on the empirical
  characteristic function.* Biometrika 70(3):723–726, 1983.
- **MAE:** K. He et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
  arXiv:2111.06377.
- **SimCLR:** T. Chen et al. 2020, arXiv:2002.05709.
- **MoCo:** K. He et al. 2020, arXiv:1911.05722.
- **BYOL:** J.-B. Grill et al. 2020, arXiv:2006.07733.
- **SimSiam:** X. Chen, K. He. 2021, arXiv:2011.10566.
- **DINO:** M. Caron et al. 2021, arXiv:2104.14294.
- **VICReg:** A. Bardes, J. Ponce, Y. LeCun. ICLR 2022, arXiv:2105.04906.
