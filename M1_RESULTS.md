# Milestone 1: results for review

Covers `ATOMS.md`, two trained tiny models, nine ablations, and the exports in `/data`.

- **Setup:** CIFAR-10, 200 epochs, batch 256, one RTX 4070 SUPER.
- **Numerics:** bf16 autocast for the networks. Losses, SIGReg, EMA, optimiser state and all evaluation are fp32.
- **Sweep:** 11 runs in 2 h 01 min wall-clock, 3 at a time.
- **Reproduce:** `python -m jepa_viz.sweep sweeps/m1.yaml`, then `python -m jepa_viz.export`.
- **fp32 archive:** the first fp32 sweep is archived in `model/runs_m1_fp32/`. See the comparison below.

## Final numbers

All numbers are measured on frozen, mean-pooled features: the target encoder for I-JEPA, the backbone for LeJEPA.
- **kNN:** k = 20, cosine, 10k labelled images → 5k test images.
- **Linear probe:** logistic regression on the same split.
- **Effective dims:** participation ratio × 64.
- **Spread:** the fraction of feature energy that varies across images. A value below 1e-3 or RankMe below 1.5 counts as complete collapse.

| run | kNN (best during run) | linear | RankMe /64 | eff. dims | spread (min over run) | z (LeJEPA): RankMe /32, eff. dims |
|---|---|---|---|---|---|---|
| *random init (step 0)* | 0.258 | 0.316 | 6.6 | 1.6 | 0.27 | |
| **ijepa-base** | **0.488** (0.511) | **0.534** | 31.1 | 12.2 | 0.50 (0.043) | |
| ijepa-no-stopgrad | 0.100 (0.273) | 0.287 | 1.0 | — | 8e-11 (5e-11) | |
| ijepa-no-predictor | 0.229 (0.264) | 0.361 | 4.5 | 1.6 | 0.28 (0.038) | |
| ijepa-linear-predictor | 0.420 (0.430) | 0.475 | 55.8 | 34.8 | 0.93 (0.16) | |
| ijepa-tau-low (τ = 0.9) | 0.354 (0.460) | 0.464 | 19.9 | 5.4 | 0.18 (9e-5) | |
| ijepa-tau0-stopgrad | 0.354 (0.360) | 0.425 | 18.9 | 7.9 | 0.43 (9e-4) | |
| **lejepa-base** | **0.647** (0.651) | **0.710** | 13.1 | 15.9 | 0.022 (0.022) | 31.3, 26.9 |
| lejepa-lambda0 | 0.206 (0.267) | 0.285 | 1.3 | 1.0 | 0.010 (0.007) | 1.0, 0.0 (z collapsed to a point) |
| lejepa-M2-fixed | 0.374 (0.380) | 0.450 | 12.6 | 2.2 | 0.23 (0.20) | 1.3, 1.0 (z collapsed to a line) |
| lejepa-M2-resampled | 0.426 (0.432) | 0.476 | 14.9 | 7.3 | 0.12 (0.039) | 21.9, 14.0 |
| lejepa-lambda-high (λ = 0.95) | 0.358 (0.359) | 0.377 | 18.0 | 13.1 | 0.19 (0.095) | 31.7, 30.0 (near-perfect Gaussian) |

## Does each ablation show what it should?

- **I-JEPA without stop-gradient.** Yes: complete collapse by about step 200.
  - Spread falls to 8e-11 and kNN accuracy to chance.
  - The prediction loss falls to about 1e-6. **The loss looks best exactly when the representation is useless.**
- **I-JEPA without predictor** (pooled identity, no positions). Collapse onto about 1.6 effective dimensions.
  - Probe accuracy ends *below* random init.
- **I-JEPA with a linear predictor** (added after review). A position-aware linear map from the pooled context, with no attention.
  - **It does not collapse.** Its features are the highest-rank and most isotropic of any I-JEPA run (RankMe 56 of 64, 35 effective dimensions).
  - **It probes worse than the baseline** (42% vs 49%).
  - Knowing *where* to predict is what prevents collapse. The attention predictor's capacity is what buys quality.
  - Higher rank does not by itself mean better probes, which is a useful counterpoint for milestone 5's plot of probe accuracy against isotropy.
- **τ too low (0.9).** Transient complete collapse at epochs 2–6.
  - Spread reaches 9e-5.
  - It then escapes, reaching 46% at best, and drifts down to 35%.
- **τ = 0 with stop-gradient (SimSiam-style).** Near-collapse for the first ~30 epochs, then partial recovery to 35%.
- **LeJEPA λ = 0.** z collapses to a single point, and the invariance loss reaches 0. The backbone also collapses (RankMe 1.3).
- **LeJEPA M = 2 fixed directions.** Anisotropic collapse: z keeps large variance along the constrained directions but has about 1 effective dimension. kNN is 37%.
- **LeJEPA M = 2 resampled.** This only partly tests the paper's claim that resampling rescues small M.
  - Resampling helps (37% → 43%, z RankMe 1.3 → 21.9), but two slices are far from the M = 256 baseline (65%).
  - The paper's own claim is about |A| = 16, which belongs on the M-axis in milestone 4.
- **λ too high (0.95).** Yes: z is the most isotropic of any run (30 of 32 effective dimensions), but kNN is 36% because view agreement is ignored.

## fp32 vs bf16 (same seeds; `python -m jepa_viz.report --compare runs_m1_fp32`)

Both baselines are within 0.6 points of their fp32 runs.
- **I-JEPA:** kNN 48.2% → 48.8%; linear probe 53.8% → 53.4%.
- **LeJEPA:** kNN 64.9% → 64.7%; linear probe 71.6% → 71.0%.

Most ablations move by about 1 point or less. The exceptions are the chaotic runs:
- **`ijepa-tau-low`:** final kNN 40% → 35%, but its best accuracy is the same (46%) and so is the collapse-and-recover shape.
- **`lejepa-lambda-high`:** linear probe 41% → 38%.

No conclusion changed.

**Speed.** LeJEPA runs at 19 ms/step, down from 26: bf16 plus explicit-matmul attention, which beats the fused SDPA kernels at d_h = 16 and ≤ 64 tokens. I-JEPA is CPU-bound at about 8 ms/step either way. What didn't help:
- `torch.compile`: NaN at step 0 in combination with bf16, and at most about 10% gain.
- Several concurrent runs: the GPU time-slices between processes, so one LeJEPA run already saturates it.
- CUDA MPS, which would let processes share the GPU: it won't start in this environment.

## Verification

| check | result |
|---|---|
| pytest (26 tests) | all pass: masks, SIGReg including a closed-form Epps–Pulley oracle, gradient flow, EMA, recording, linear predictor, fp32 losses under bf16 |
| gradient reach, recorded in the I-JEPA golden | target encoder 0 of 52 params, context encoder 52 of 52, predictor 31 of 31 |
| independent NumPy forward pass vs goldens | 258 of 258 tensors within tolerance (worst: 1.4e-5 for I-JEPA, 1.9e-5 for LeJEPA), including the view rendering (crop, flip, jitter, grayscale, solarize) |
| initial data payload | 1.53 MB, within the 5 MB budget before the JS bundle |

## Things found along the way

Most of these are already written into ATOMS.md.

1. **Initialisation.** Trunc-normal(0.02) on the patch embedding lets the fixed sin-cos positions swamp image content. I-JEPA then predicts from position alone, giving a tiny loss and near-collapsed features. The official code keeps the default conv init, and so do we now.
2. **Official sampler edge bug.** The official multi-block sampler never places a block on the last row or column, which at 8×8 would waste 23% of tokens. We fix it by default (`mask.edge_fix`).
3. **Small context.** Truncating index sets to the batch minimum, an official quirk we reproduce, leaves the context at a median of 18 tokens (range 6–30) at B = 256.
4. **Diagnostics.** Raw random-pair cosine and min/max isotropy are misleading for mean-pooled LayerNorm features. The LeJEPA backbone scores cosine 0.98 while reaching 65% kNN. Collapse is therefore called from spread and RankMe, and isotropy uses the participation ratio.
5. **Val loss vs quality.** I-JEPA's val loss is lowest during early near-collapse, rises as the representation improves, and keeps falling late in training while kNN accuracy declines.
6. **Features keep rotating** between late checkpoints (both LeJEPA loss terms are rotation-invariant). The PCA export therefore carries per-checkpoint aligned clouds plus a fixed-basis version for smooth scrubbing.

## Settled at review

1. The I-JEPA baseline stays at the official defaults; the milestone-4 sweeps will show what helps.
2. Faster numerics (bf16 + explicit attention) are adopted, and all milestone-1 runs were re-run to match.
3. The no-predictor ablation is kept, and a linear-predictor variant was added.
4. Goldens stay git-ignored. Regenerate them with `python -m jepa_viz.export --only golden`.

## Carried into the milestone-4 plan

**Grid size.** The mixed sweep runs at about **5.5 runs/hour** (LeJEPA about 27–35 min per run and I-JEPA about 20–35 min, 3 at a time). A full one-axis-at-a-time grid across every axis in the spec would take about 100 runs, or roughly 18 h. The milestone-4 plan should pick a smaller grid:
- about 30–40 runs, roughly 6–7 h;
- prioritised on the axes the walkthrough and presets actually use: mask scale and count, λ, M, τ, LR/warmup, predictor depth, and K.
