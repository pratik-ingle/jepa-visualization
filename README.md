# JEPA Visualization

**Live demo: <https://pratik-ingle.github.io/jepa-visualization-/>**

An interactive, illustrated 3D guide to Joint-Embedding Predictive Architectures, in the style of
[bbycroft.net/llm](https://bbycroft.net/llm). Two real tiny models run live in your browser:

- **I-JEPA** (Assran et al., 2023) avoids representation collapse with block masking, an EMA target
  encoder, stop-gradient and a predictor.
- **LeJEPA** (Balestriero & LeCun, 2025) uses multi-view invariance plus SIGReg, with no EMA, no
  stop-gradient and no predictor.

They make the same predictive bet and give two different answers to "how do you stop representation
collapse".

[`ATOMS.md`](ATOMS.md) breaks both algorithms into every operation, with shapes, gradient flow and
every deviation from the papers.
[`M1_RESULTS.md`](M1_RESULTS.md) holds the trained models and ablations.
[`RESULTS.md`](RESULTS.md) holds the full hyperparameter sweep.

## What the guide shows

- **One 3D scene with two branches.**
  - I-JEPA: image → patches → mask → context and EMA target encoders → predictor → loss.
  - LeJEPA: 8 crops → one shared encoder → projector → z → invariance + SIGReg.
  - Every tensor is a block coloured by value. Hover a cell for its value, shape and ATOMS reference;
    click a token to trace it through every layer.
- **A walkthrough**, one phase per atom, in which the camera flies to the tensors the commentary
  discusses. Use ←/→ or the timeline to move through it. The phases include:
  - a live **backward pass** that plays gradients from the loss back to the input and shows that the
    target encoder receives none;
  - the **EMA update** applied to real weights;
  - **SIGReg** on a real batch of 256 held-out embeddings computed in a Web Worker: random directions,
    per-slice histograms against N(0, 1), empirical vs. Gaussian characteristic functions, and
    per-slice Epps–Pulley statistics.
- **Live controls.** Each change re-runs the real forward (and backward) passes:
  - the input image;
  - re-sampling the mask or crops, and the target count and scale;
  - dragging mask blocks;
  - any attention head;
  - which encoder or view is shown;
  - the checkpoint.
- **A hyperparameter explorer** covering every run: one setting at a time, failure-mode presets and
  2-way grids.
  - Choosing a setting swaps in the matching offline run.
  - Curves update instantly against a baseline ghost.
  - A checkpoint scrubber drives each run's PCA-3D embedding cloud in the scene.
- **Three regimes** (tabs):
  - **Training:** forward, backward, optimiser and EMA.
  - **Validation:** forward only on held-out images, and why JEPA val loss is a weak signal.
  - **Evaluation:** frozen encoders, live k-NN, class-coloured features, probe accuracy vs isotropy,
    and per-axis probe shifts.

Every number on screen comes from a real forward or backward pass in the browser or from a real
logged training run.

## Layout

| path | contents |
|---|---|
| `model/` | PyTorch reference: tiny ViT, both algorithms, training, sweeps, export, NumPy verifier |
| `data/` | exports consumed by the web app: fp16 weights, images, run logs, PCA clouds, goldens |
| `web/` | Next.js (static export) + react-three-fiber app |

## Reproduce

```bash
conda env create -f model/environment.yml && conda activate jepa
cd model
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C .cache
python -m pytest                                   # 26 tests
python -m jepa_viz.sweep sweeps/m1.yaml --jobs 3   # baselines + ablations (~2 h on one RTX 4070 SUPER)
python -m jepa_viz.sweep sweeps/m4.yaml --jobs 4   # explorer sweep, 38 runs (~7 h)
python -m jepa_viz.export                          # -> ../data (weights, logs, clouds, goldens)
python -m jepa_viz.np_forward                      # independent NumPy check of the exports
python -m jepa_viz.report                          # results table

cd ../web
npm install
npm test          # TypeScript forward AND backward passes vs the PyTorch goldens
npm run dev       # http://localhost:3000
npm run build     # static site in web/out (any static file server)
```

Every push to `main` rebuilds the site and deploys it to GitHub Pages
([`.github/workflows/pages.yml`](.github/workflows/pages.yml)). To serve the site from a subpath
yourself, set `NEXT_PUBLIC_BASE_PATH=/<subpath>` when you run `npm run build`.

`model/splits.json` fixes the data splits once. The browser port in `web/src/model/` is
dependency-free TypeScript. Its forward passes match PyTorch on every recorded tensor: 140 for I-JEPA
and 122 for LeJEPA, including crop rendering. Its hand-derived backward passes match autograd on
every recorded gradient: 82 + 115 activation gradients and 83 + 62 parameter gradients.

## Performance and size

- **Rendering.** One box per tensor, with cell values read from a float texture in a shader
  (following llm-viz). That comes to about 460 draw calls and about 10k triangles per frame, rendered
  on demand only.
- **Frame rate.** Chromium's CPU software renderer draws camera flights at 46–50 fps at 1600×900.
  A 60 fps check on a real laptop integrated GPU has not been run here.
- **First load.** At most 4.6 MB raw (about 2.9 MB gzipped), counting every JS chunk, both models'
  fp16 weights, the image atlases, the fonts and the baseline clouds. Other run logs and clouds load
  on demand.

Labels use "JEPA Viz Sans", a renamed subset of DejaVu Sans (licence:
`web/public/fonts/LICENSE-dejavu.txt`).
