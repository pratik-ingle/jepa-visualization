# Results: 49 tiny JEPA runs on CIFAR-10

Every run used the same recipe:
- CIFAR-10 at 32×32, a ViT with D=64 and 4 layers (~200k params), 200 epochs, batch 256;
- bf16 networks with fp32 losses, on one RTX 4070 SUPER;
- probes on frozen, mean-pooled features: the I-JEPA target encoder and the LeJEPA backbone;
- k-NN (k=20, cosine) and a linear probe, trained on 10k labelled images and tested on 5k held-out images.

The runs break down as:
- 11 milestone-1 runs: the baselines and ablations (details in `M1_RESULTS.md`);
- 38 milestone-4 explorer runs: one setting at a time from each baseline, 5 failure presets, and two 2-way grids.

Every number here is also browsable in the guide's explorer. Regenerate the tables with `python -m jepa_viz.results`.

**Collapse call.** A run counts as completely collapsed when spread < 1e-3 or RankMe < 1.5.
- **Spread** is the fraction of feature energy that varies across images.
- **Effective dimensions** is the covariance participation ratio × 64.

## Headline findings

1. **Both anti-collapse mechanisms work, and removing them collapses the model.**
   - I-JEPA without stop-gradient collapses completely (spread 8e-11, k-NN at chance).
   - LeJEPA with λ = 0 collapses completely (z becomes a single point).
   - I-JEPA without a predictor collapses onto ~1.6 dimensions.
   - With only 2 *fixed* SIGReg directions, LeJEPA's z collapses onto a line.
2. **LeJEPA beats I-JEPA at this scale.** Its baseline reaches 64.7% k-NN / 71.0% linear, against 48.8% / 53.4% for I-JEPA. The best LeJEPA setting (larger local crops) reaches 68.6% / 73.9%.
3. **The optimiser choice can collapse I-JEPA.** Adam and SGD, run with I-JEPA's 0.04 → 0.4 weight-decay schedule applied as coupled L2, collapse completely (10% k-NN, spread 1e-15). The encoder's gradients are ~1e-4, so the decay term dominates each update and pulls the weights to zero. AdamW's decoupled decay does not.
4. **Positional conditioning prevents I-JEPA collapse; the predictor's attention adds quality.**
   - Removing the predictor collapses the model.
   - A linear, position-aware predictor does not collapse. It gives the highest-rank features of any I-JEPA run, yet probes worse than the baseline (42.0% vs 48.8% k-NN).
5. **Validation loss.**
   - **Within a run**, LeJEPA's val loss tracks linear-probe accuracy almost perfectly (median Spearman ρ = −0.98 over 24 runs). I-JEPA's does not (median +0.33).
   - **Across configurations**, the paper's loss–accuracy correlation does not reproduce at this scale: ρ = −0.21 raw and +0.03 after dividing by λ^0.4.
   - For I-JEPA the across-run correlation is **inverted** (ρ = +0.75): the collapsed runs have the lowest loss.
6. **Isotropy vs probes.** Across runs, the isotropy of frozen features correlates with linear accuracy at ρ = 0.68 for LeJEPA and 0.42 for I-JEPA. That is a real but partial trend; the linear-predictor run shows isotropy is not sufficient.
7. **SIGReg settings.**
   - More slices help: M = 2 → 42.6%, 16 → 56.4%, 64 → 63.0%, 256 → 64.7% k-NN.
   - Re-drawing directions beats fixing them (56.4% vs 50.7% at M = 16). Small M is also fragile to λ: at M = 16, λ = 0.2 drops to 52.4% linear.
   - VICReg as a drop-in comes close (68.2% linear vs 71.0%).
   - λ is forgiving between 0.01 and 0.05, costs 5 points at 0.2, and fails at 0.95.
8. **Presets that did not fail.**
   - No warmup does not hurt at this scale (I-JEPA 50.5%, LeJEPA 65.0%). The preset stays so it can be seen not to fail.
   - "Masking too easy" (1 small block) cuts val loss 3× and loses 7 points of k-NN. The default masking is the best of every masking setting tried.
   - "τ too low" (0.9) collapses at epochs 2–6, then escapes to 35% k-NN.

## Tables

### I-JEPA (frozen target features; ★ = baseline)

| setting | value | run | k-NN % | linear % | RankMe | eff. dims | spread | collapse |
|---|---|---|---|---|---|---|---|---|
| optimiser | adam | ijepa-adam | 10.0 | 10.0 | 1.0 | 1.0 | 5.857e-15 | complete |
| optimiser | sgd | ijepa-sgd | 10.0 | 10.0 | 1.0 | 1.0 | 5.003e-15 | complete |
| optimiser | adamw ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| teacher (EMA τ / stop-grad) | EMA 0.996→1 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| teacher (EMA τ / stop-grad) | shared, no stop-grad | ijepa-no-stopgrad | 10.0 | 28.7 | 1.001 | 2.931 | 7.897e-11 | complete |
| teacher (EMA τ / stop-grad) | EMA fixed 0.9 | ijepa-tau-low | 35.4 | 46.4 | 19.85 | 5.426 | 0.1818 | none |
| teacher (EMA τ / stop-grad) | shared, stop-grad | ijepa-tau0-stopgrad | 35.4 | 42.5 | 18.9 | 7.877 | 0.4328 | none |
| teacher (EMA τ / stop-grad) | EMA 0.99→1 | ijepa-tau0.99 | 48.9 | 53.1 | 35.81 | 12.85 | 0.6144 | none |
| predictor | transformer ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| predictor | linear | ijepa-linear-predictor | 42.0 | 47.5 | 55.83 | 34.78 | 0.9332 | none |
| predictor | none | ijepa-no-predictor | 22.9 | 36.1 | 4.51 | 1.601 | 0.2831 | none |
| Adam β₂ | 0.95 | ijepa-beta2-0.95 | 49.5 | 54.9 | 32.07 | 12.62 | 0.4358 | none |
| Adam β₂ | 0.999 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| LR schedule | constant | ijepa-const-lr | 48.4 | 56.1 | 32.81 | 12.21 | 0.8148 | none |
| LR schedule | cosine ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| context block scale | 0.4–0.6 | ijepa-ctx-small | 46.1 | 49.9 | 30.12 | 11.25 | 0.4645 | none |
| context block scale | 0.85–1.0 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| encoder depth | 2 | ijepa-depth2 | 46.5 | 51.5 | 29.71 | 10.83 | 0.4413 | none |
| encoder depth | 4 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| loss | l2 | ijepa-l2 | 50.7 | 55.6 | 32.5 | 12.11 | 0.4833 | none |
| loss | smooth_l1 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| peak LR | 0.0003 | ijepa-lr3e-4 | 48.8 | 50.9 | 33.52 | 12.34 | 0.5556 | none |
| peak LR | 0.001 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| peak LR | 0.003 | ijepa-lr3e-3 | 50.8 | 57.0 | 30.78 | 10.47 | 0.6746 | none |
| warmup epochs | 0 | ijepa-no-warmup | 50.5 | 54.0 | 31.58 | 13.36 | 0.3727 | none |
| warmup epochs | 10.0 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| target blocks | 1 | ijepa-npred1 | 44.4 | 48.8 | 34.43 | 13.96 | 0.6456 | none |
| target blocks | 2 | ijepa-npred2 | 43.3 | 45.3 | 28.83 | 11.08 | 0.3194 | none |
| target blocks | 4 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| predictor depth | 1 | ijepa-pred-depth1 | 50.4 | 54.6 | 30.0 | 12.94 | 0.3079 | none |
| predictor depth | 2 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| predictor depth | 4 | ijepa-pred-depth4 | 50.7 | 54.0 | 35.91 | 14.87 | 0.5787 | none |
| target block scale | 0.3–0.4 | ijepa-tgt-large | 41.5 | 48.6 | 28.58 | 9.434 | 0.395 | none |
| target block scale | 0.05–0.1 | ijepa-tgt-small | 41.4 | 45.6 | 29.91 | 13.45 | 0.2127 | none |
| target block scale | 0.15–0.2 ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |
| WD schedule | fixed | ijepa-wd-fixed | 49.7 | 55.0 | 31.57 | 12.92 | 0.4964 | none |
| WD schedule | cosine ★ | ijepa-base | 48.8 | 53.4 | 31.1 | 12.2 | 0.4987 | none |

### LeJEPA (frozen backbone features; ★ = baseline)

| setting | value | run | k-NN % | linear % | RankMe | eff. dims | spread | collapse |
|---|---|---|---|---|---|---|---|---|
| K | 16 | lejepa-K16 | 65.4 | 72.2 | 12.61 | 17.38 | 0.02091 | none |
| K | 32 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| K | 64 | lejepa-K64 | 64.4 | 70.2 | 13.79 | 17.99 | 0.02403 | none |
| SIGReg slices M | 16 | lejepa-M16 | 56.4 | 61.3 | 24.0 | 13.84 | 0.1042 | none |
| SIGReg slices M | 16 fixed | lejepa-M16-fixed | 50.7 | 53.6 | 19.58 | 2.419 | 0.2112 | none |
| SIGReg slices M | 2 fixed | lejepa-M2-fixed | 37.4 | 45.0 | 12.6 | 2.245 | 0.2247 | none |
| SIGReg slices M | 2 | lejepa-M2-resampled | 42.6 | 47.6 | 14.88 | 7.339 | 0.1178 | none |
| SIGReg slices M | 64 | lejepa-M64 | 63.0 | 68.8 | 16.53 | 15.94 | 0.03786 | none |
| SIGReg slices M | 256 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| encoder depth | 2 | lejepa-depth2 | 60.0 | 65.9 | 16.21 | 15.39 | 0.03425 | none |
| encoder depth | 4 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| λ | 0.0 | lejepa-lambda0 | 20.6 | 28.5 | 1.332 | 1.042 | 0.01013 | complete |
| λ | 0.01 | lejepa-lam0.01 | 65.6 | 71.7 | 12.75 | 12.18 | 0.02364 | none |
| λ | 0.05 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| λ | 0.2 | lejepa-lam0.2 | 61.1 | 66.1 | 16.79 | 18.51 | 0.03642 | none |
| λ | 0.95 | lejepa-lambda-high | 35.8 | 37.7 | 18.02 | 13.14 | 0.1906 | none |
| local crop scale | 0.3–0.6 | lejepa-local-large | 68.6 | 73.9 | 16.78 | 16.38 | 0.03766 | none |
| local crop scale | 0.05–0.3 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| local views V_l | 0 | lejepa-local0 | 66.4 | 70.3 | 15.6 | 16.76 | 0.03342 | none |
| local views V_l | 2 | lejepa-local2 | 64.3 | 70.1 | 16.12 | 16.63 | 0.03446 | none |
| local views V_l | 6 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| peak LR | 0.0003 | lejepa-lr3e-4 | 58.9 | 64.1 | 18.23 | 14.16 | 0.04882 | none |
| peak LR | 0.001 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| peak LR | 0.003 | lejepa-lr3e-3 | 67.2 | 73.2 | 15.11 | 16.94 | 0.02936 | none |
| warmup epochs | 0 | lejepa-no-warmup | 65.0 | 70.9 | 13.73 | 17.27 | 0.02419 | none |
| warmup epochs | 10.0 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| projector depth | 2 | lejepa-proj2 | 63.6 | 69.8 | 24.16 | 18.13 | 0.08753 | none |
| projector depth | 3 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| EP t range | 3.0 | lejepa-tmax3 | 65.2 | 72.0 | 13.11 | 17.1 | 0.02217 | none |
| EP t range | 5.0 ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |
| regulariser | vicreg | lejepa-vicreg | 64.3 | 68.2 | 13.04 | 13.03 | 0.02353 | none |
| regulariser | sigreg ★ | lejepa-base | 64.7 | 71.0 | 13.15 | 15.94 | 0.02245 | none |

### Failure-mode presets

| preset | run | k-NN % | linear % | spread | collapse |
|---|---|---|---|---|---|
| masking-too-easy | ijepa-mask-easy | 41.7 | 44.9 | 0.5428 | none |
| representation-collapse | ijepa-no-stopgrad | 10.0 | 28.7 | 7.897e-11 | complete |
| representation-collapse | lejepa-lambda0 | 20.6 | 28.5 | 0.01013 | complete |
| no-warmup | ijepa-no-warmup | 50.5 | 54.0 | 0.3727 | none |
| no-warmup | lejepa-no-warmup | 65.0 | 70.9 | 0.02419 | none |
| tau-too-low | ijepa-tau-low | 35.4 | 46.4 | 0.1818 | none |
| lambda-too-high | lejepa-lambda-high | 35.8 | 37.7 | 0.1906 | none |

### Does the loss track probe accuracy? (Spearman ρ across runs, final val loss vs linear acc)

- ijepa: ρ = 0.75 over 25 runs
- lejepa: ρ = -0.21 over 24 runs
- lejepa, loss / λ^0.4 (paper Eq. 10): ρ = 0.03 over 23 runs with λ > 0

### Within a run: does val loss track linear-probe accuracy over training? (eval checkpoints after epoch 10)

| run | ρ(val loss, linear acc) |
|---|---|
| ijepa-base | 0.10 |
| lejepa-base | -0.99 |
| median over 25 ijepa runs | 0.33 |
| median over 24 lejepa runs | -0.98 |

### Isotropy vs probe accuracy (Spearman ρ across runs, participation ratio vs linear acc)

- ijepa: ρ = 0.42 over 25 runs
- lejepa: ρ = 0.68 over 24 runs


## Caveats

- One seed per setting. Differences of 1–2 points are within run-to-run noise and should not be read as effects.
- The 200-epoch budget, the tiny ViT and 32-px CIFAR-10 are far from the papers' ImageNet scale. In particular, the across-configuration loss–accuracy correlation and the paper's M = 16 claim may behave differently at scale.
- Adam and SGD used the I-JEPA weight-decay schedule unchanged (coupled L2). Tuning their decay would likely avoid the collapse; the run isolates the effect of coupled vs decoupled decay.
