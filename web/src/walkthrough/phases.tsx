// Walkthrough: one phase per ATOMS.md atom (plus overview / contrast phases), per regime.
import type { ReactNode } from "react";
import type { Regime } from "@/lib/store";

export type Widget = "mask" | "optim" | "ema" | "sigreg" | "cloud" | "valcurve" | "probe" | "knn" | "axes";

export interface Phase {
  id: string;
  group: "Overview" | "I-JEPA" | "LeJEPA" | "Shared" | "Context";
  title: string;
  atom?: string;
  regimes: Regime[];
  focus: string[]; // frame ids to fly to ("*" = everything)
  grad?: boolean; // show dL/d(activation) instead of activations
  widget?: Widget;
  body: ReactNode;
}

const REFS: Record<string, [string, string]> = {
  ijepa: ["Assran et al., CVPR 2023", "https://arxiv.org/abs/2301.08243"],
  vjepa: ["Bardes et al., 2024", "https://arxiv.org/abs/2404.08471"],
  vjepa2: ["Assran et al., 2025", "https://arxiv.org/abs/2506.09985"],
  lejepa: ["Balestriero & LeCun, 2025", "https://arxiv.org/abs/2511.08544"],
  rankme: ["Garrido et al., ICML 2023", "https://arxiv.org/abs/2210.02885"],
  ep: ["Epps & Pulley, Biometrika 1983", "https://doi.org/10.1093/biomet/70.3.723"],
  mae: ["He et al., CVPR 2022", "https://arxiv.org/abs/2111.06377"],
  vicreg: ["Bardes, Ponce & LeCun, ICLR 2022", "https://arxiv.org/abs/2105.04906"],
  byol: ["Grill et al., 2020", "https://arxiv.org/abs/2006.07733"],
  dino: ["Caron et al., 2021", "https://arxiv.org/abs/2104.14294"],
  simclr: ["Chen et al., 2020", "https://arxiv.org/abs/2002.05709"],
};

export function Cite({ id }: { id: keyof typeof REFS }) {
  const [t, u] = REFS[id];
  return (
    <a href={u} target="_blank" rel="noreferrer" className="whitespace-nowrap text-sky-400 hover:underline">
      [{t}]
    </a>
  );
}

const P = ({ children }: { children: ReactNode }) => <p className="mb-2">{children}</p>;
const K = ({ children }: { children: ReactNode }) => <span className="font-mono text-[11px] text-amber-200">{children}</span>;
const Note = ({ children }: { children: ReactNode }) => <p className="mb-2 rounded border-l-2 border-sky-500/60 bg-sky-500/5 px-2 py-1 text-slate-300">{children}</p>;

const ALL: Regime[] = ["train", "val", "eval"];
const TRAIN: Regime[] = ["train"];

export const PHASES: Phase[] = [
  // ---------------------------------------------------------------- Overview
  {
    id: "intro", group: "Overview", title: "Two answers to one question", regimes: TRAIN, focus: ["*"],
    body: (
      <>
        <P>
          A <b>joint-embedding predictive architecture</b> learns by predicting the <i>representation</i> of a hidden part of an image from the representation of a visible part,
          never touching pixels. That makes it cheap and semantic, and it creates one danger: an encoder that maps every image to the same vector predicts itself perfectly.
          This is <b>representation collapse</b>.
        </P>
        <P>
          <span className="text-amber-300">I-JEPA</span> (left) avoids it with asymmetry: an EMA teacher, stop-gradient and a predictor <Cite id="ijepa" />.{" "}
          <span className="text-emerald-300">LeJEPA</span> (right) makes it impossible: SIGReg forces the embeddings toward an isotropic Gaussian <Cite id="lejepa" />.
        </P>
        <P>
          Both are real 200k-parameter ViTs trained on CIFAR-10 (8×8 tokens of 4×4 pixels). Every value you hover is computed by a forward (and, later, backward) pass running in
          your browser. The PyTorch reference matches it to about 1e-5.
        </P>
        <Note>Drag to pan, scroll to zoom, right-drag to orbit. Use ←/→ or the timeline to move through the atoms.</Note>
      </>
    ),
  },
  {
    id: "val-intro", group: "Overview", title: "Validation: forward only, on held-out images", regimes: ["val"], focus: ["*"],
    body: (
      <>
        <P>
          The scene now runs the same two objectives on <b>held-out images</b> (CIFAR-10 test images the models never trained on): no backward pass, no optimiser step, no EMA update.
        </P>
        <P>
          The validation loss is simply the training objective measured on new images. For a supervised model that is the whole story. For a self-supervised JEPA it is a weak signal,
          and this regime shows why.
        </P>
      </>
    ),
  },
  {
    id: "eval-intro", group: "Overview", title: "Evaluation: testing the representation", regimes: ["eval"], focus: ["*"],
    body: (
      <>
        <P>
          A JEPA is never deployed to predict masked embeddings. Its product is the <b>encoder</b>. So &ldquo;testing&rdquo; a self-supervised model means testing the{" "}
          <b>representation</b>, not the pretext prediction.
        </P>
        <P>
          Protocol: freeze the encoder (the I-JEPA <i>target</i> encoder, as in the paper; the LeJEPA backbone before the projector), mean-pool its tokens, then fit two cheap probes on a
          small labelled set (10k CIFAR-10 images): a <b>k-NN</b> classifier and a <b>linear</b> classifier. They are evaluated on 5k held-out test images.
        </P>
      </>
    ),
  },
  // ---------------------------------------------------------------- I-JEPA
  {
    id: "embed", group: "I-JEPA", atom: "1", title: "Patches → embeddings (+ fixed positions)", regimes: ALL, focus: ["ijepa:embed"],
    body: (
      <>
        <P>
          The 32×32 image is cut into an 8×8 grid of 4×4 patches: 64 tokens of 3·4·4 = 48 numbers. A linear map <K>W_e</K> (48→64) embeds each patch, and a <b>fixed 2D sin-cos</b>{" "}
          code of the token&rsquo;s row and column is added. The first 32 channels encode the column, the last 32 the row.
        </P>
        <Note>
          Found while building this: if <K>W_e</K> is initialised too small, the positions (RMS ≈ 0.7) swamp the patch content (std ≈ 0.14). The predictor can then hit every target from
          position alone: the loss is tiny and the features are nearly collapsed. The official I-JEPA keeps PyTorch&rsquo;s default patch-embedding init, which avoids this.
        </Note>
      </>
    ),
  },
  {
    id: "mask", group: "I-JEPA", atom: "2", title: "Multi-block masking", regimes: ALL, focus: ["ijepa:embed"], widget: "mask",
    body: (
      <>
        <P>
          Four <b>target blocks</b> (each 15–20% of the grid, aspect 0.75–1.5; here 3×3 or 4×3 tokens) are sampled. One <b>context block</b> (85–100% of the grid) is sampled too, and
          every target token is carved out of it. The encoder must understand the image well enough that the predictor can say what is <i>inside</i> the holes.
        </P>
        <P>
          Two quirks of the official sampler are reproduced (see ATOMS.md §D). Block sizes are shared across a batch, and context sets are truncated to the batch minimum, which leaves
          median 18 context tokens at batch 256. One is fixed: the official placement never touches the last row or column, which would waste 23% of an 8×8 grid.
        </P>
        <P>
          In our sweep the default is the best of every masking setting tried. Fewer targets (1 or 2), smaller targets (5–10%) or larger ones (30–40%) all probe 4–8 points worse. A
          single small target (&ldquo;masking too easy&rdquo;) lowers the val loss 3× and loses 7 points of k-NN.
        </P>
        <Note>Drag a coloured block below: the context is re-carved and every tensor on the left is recomputed.</Note>
      </>
    ),
  },
  {
    id: "ctx", group: "I-JEPA", atom: "3", title: "Context encoder f_θ — only what is visible", regimes: ALL, focus: ["ijepa:ctx"],
    body: (
      <>
        <P>
          Only the visible tokens enter the context encoder: they are <i>gathered</i>, not zeroed, so attention is N_ctx×N_ctx and the masked content cannot leak in. Four pre-LayerNorm
          transformer blocks (width 64, 4 heads of 16) and a final LayerNorm produce <K>s_x</K>.
        </P>
        <P>Click any token row to trace it through every layer. Its row lights up in each tensor, and its query row and key column light up in each attention map.</P>
      </>
    ),
  },
  {
    id: "block", group: "I-JEPA", atom: "T1–T10", title: "Inside one transformer block", regimes: ALL, focus: ["ijepa:ctx"],
    body: (
      <>
        <P>
          <b>Top row, attention:</b> LN₁ normalises each token. Q, K and V are three linear maps of it, shown with all 4 heads side by side. The attention map (pick the head in the controls) is
          softmax(QKᵀ/√16), so each row sums to 1. A·V mixes value vectors, and the output projection is added back to the residual stream (<K>x + attn·W_o</K>).
        </P>
        <P>
          <b>Bottom row, MLP:</b> LN₂, an up-projection to 256, GELU, and a down-projection added to the residual stream again.
        </P>
        <P>Every block you see in both encoders, the predictor and the LeJEPA encoder is this same pattern.</P>
      </>
    ),
  },
  {
    id: "tgt", group: "I-JEPA", atom: "4", title: "Target encoder f_θ̄ — the EMA teacher", regimes: ALL, focus: ["ijepa:tgt"],
    body: (
      <>
        <P>
          The target encoder is a <b>copy</b> of the context encoder whose weights θ̄ are never trained by gradient. It sees the <b>full</b> image (64 tokens). Its output is layer-normed
          again, with no affine parameters, and the target-block tokens are gathered as the prediction targets <K>s_y</K>. Everything here runs under stop-gradient.
        </P>
        <P>Compare its attention maps with the context encoder&rsquo;s: same weights (almost), different inputs.</P>
      </>
    ),
  },
  {
    id: "pred", group: "I-JEPA", atom: "5", title: "Predictor g_φ — predicting at a position", regimes: ALL, focus: ["ijepa:pred"],
    body: (
      <>
        <P>
          The predictor is deliberately <b>narrow</b> (width 32, 2 blocks). It receives the context tokens plus one <b>mask token</b> per target position, each carrying that
          position&rsquo;s sin-cos code. Every target block is its own sequence, so blocks never see each other.
        </P>
        <Note>
          Our ablations: with <b>no predictor</b> (pooled context, no position) the representation collapses onto ~1.6 effective dimensions and probes below random init. A{" "}
          <b>linear, position-aware</b> predictor does not collapse (the highest rank of any I-JEPA run) but probes worse than the transformer (42% vs 49% k-NN). Knowing <i>where</i> to
          predict is what prevents collapse; attention is what buys quality.
        </Note>
      </>
    ),
  },
  {
    id: "loss", group: "I-JEPA", atom: "6", title: "Loss: prediction vs target", regimes: ALL, focus: ["ijepa:loss", "ijepa:pred"],
    body: (
      <>
        <P>
          The loss is smooth-L1 between the predicted and target embeddings, averaged over all blocks. That is the official code; the paper writes L2. On layer-normed targets most errors
          are below 1, where smooth-L1 is ½·MSE. Only this image and this mask contribute here.
        </P>
      </>
    ),
  },
  {
    id: "backward", group: "I-JEPA", atom: "7", title: "Backward: where the gradient goes", regimes: TRAIN, focus: ["ijepa:ctx", "ijepa:tgt", "ijepa:pred", "ijepa:loss"], grad: true,
    body: (
      <>
        <P>
          Every block now shows <b>∂L/∂(activation)</b>, computed by a backward pass in your browser. It is hand-derived per op and matches PyTorch&rsquo;s autograd on every recorded tensor
          (worst relative error 4e-6).
        </P>
        <P>
          The gradient enters at the prediction, flows back through the predictor into <K>s_x</K>, and then through every layer of the context encoder. It reaches the patch embedding only
          through the <b>visible</b> tokens: masked rows of <K>∂L/∂ tokens</K> are exactly 0.
        </P>
        <Note>The whole target encoder is dark: stop-gradient. Its weights change only through the EMA update (two steps ahead).</Note>
      </>
    ),
  },
  {
    id: "optim", group: "I-JEPA", atom: "8", title: "Optimiser step: AdamW + schedules", regimes: TRAIN, focus: ["ijepa:ctx", "ijepa:pred"], widget: "optim",
    body: (
      <>
        <P>
          AdamW updates θ (context encoder) and φ (predictor) with per-parameter adaptive steps. The learning rate warms up linearly over 10 epochs, then follows a cosine to 1e-3× the
          peak. Weight decay rises on a cosine from 0.04 to 0.4; biases and LayerNorm parameters are not decayed. The curves below are the logged schedules and gradient norms of the
          training run whose weights you are looking at.
        </P>
        <Note>
          Decoupled weight decay matters here. In our sweep, swapping AdamW for Adam or SGD with the same decay schedule applied as ordinary L2 collapsed I-JEPA completely (k-NN
          10%, spread 1e-15). The encoder&rsquo;s gradients are tiny (~1e-4), so the coupled decay term dominates the update and drives every weight to zero. Load{" "}
          <K>ijepa-adam</K> in the explorer.
        </Note>
      </>
    ),
  },
  {
    id: "ema", group: "I-JEPA", atom: "9", title: "EMA update of the target encoder", regimes: TRAIN, focus: ["ijepa:ema"], widget: "ema",
    body: (
      <>
        <P>
          After each optimiser step: <K>θ̄ ← τ·θ̄ + (1 − τ)·θ</K>. τ rises linearly from 0.996 to 1.0 over training, so the teacher trails the student by an average of ~250 steps early on
          and freezes at the end. The blocks show one real weight matrix (layer-1 W_o) from both encoders of the loaded checkpoint, and what k updates with your τ would do.
        </P>
        <Note>
          &ldquo;τ too low&rdquo; (fixed 0.9) collapsed completely at epochs 2–6 in our runs, then slowly escaped. A teacher that chases the student too closely lets the pair agree on a
          trivial solution.
        </Note>
      </>
    ),
  },
  // ---------------------------------------------------------------- LeJEPA
  {
    id: "views", group: "LeJEPA", atom: "10", title: "Multi-crop views", regimes: ALL, focus: ["lejepa:views"],
    body: (
      <>
        <P>
          LeJEPA sees one image through <b>8 views</b>: 2 global crops (30–100% of the area, resized to 32 px) and 6 local crops (5–30%, resized to 16 px). Each gets random flips,
          colour jitter, grayscale and solarize. Local views become 4×4 token grids whose sin-cos positions span the same frame as a global view.
        </P>
        <Note>Click any view tile (or use the controls) to route it through the encoder below. Re-sample the crops to see all activations change.</Note>
      </>
    ),
  },
  {
    id: "enc", group: "LeJEPA", atom: "11", title: "One encoder, then a projector", regimes: ALL, focus: ["lejepa:enc", "lejepa:proj"],
    body: (
      <>
        <P>
          Every view passes through <b>the same encoder</b>. There is no second copy: no teacher, no EMA, no stop-gradient. Tokens are mean-pooled into the backbone feature <K>e</K>
          (what probes use), then a projector (Linear-BatchNorm-ReLU ×2 → Linear) maps it to <K>z ∈ ℝ³²</K>, where the loss acts.
        </P>
      </>
    ),
  },
  {
    id: "inv", group: "LeJEPA", atom: "12", title: "Invariance: agree with the global centre", regimes: ALL, focus: ["lejepa:proj"],
    body: (
      <>
        <P>
          The centre μ is the mean of the two global views&rsquo; z. The invariance loss is the mean squared distance from <b>every</b> view, globals included, to μ (paper Eq. 8). Local
          views must predict the global content they are part of: this is the &ldquo;predictive&rdquo; half of LeJEPA.
        </P>
        <Note>Alone, invariance collapses. With λ = 0 our run drove z to a single point (invariance loss exactly 0) and k-NN to 21%.</Note>
      </>
    ),
  },
  {
    id: "sigreg", group: "LeJEPA", atom: "13", title: "SIGReg: sketched isotropic Gaussian regularisation", regimes: ["train", "val"], focus: ["lejepa:sig"], widget: "sigreg",
    body: (
      <>
        <P>
          SIGReg asks the <b>batch</b> of embeddings to look like an isotropic Gaussian N(0, I). Checking that in 32-D directly is hard. By Cramér–Wold, a distribution is determined by all
          its 1-D projections, so SIGReg draws M random unit directions, projects the batch onto each, and tests every 1-D slice against N(0, 1).
        </P>
        <P>
          The test is Epps–Pulley <Cite id="ep" />: <K>EP = N·∫|φ̂(t) − e^{"{−t²/2}"}|² e^{"{−t²/2}"} dt</K>, the weighted distance between the slice&rsquo;s empirical characteristic
          function and the Gaussian&rsquo;s, integrated with the trapezoid rule over 17 points in [−5, 5]. It costs O(N·K·M): no covariance matrix, no pairwise similarities.
        </P>
        <P>
          Below: the real z of 256 held-out images from the loaded LeJEPA checkpoint, computed in a Web Worker. Pick a slice to see its histogram against N(0, 1) and its characteristic
          function against the target. A perfectly Gaussian slice still scores ≈ 1.06: the statistic&rsquo;s finite-sample floor.
        </P>
        <Note>
          The optimality claim: among embeddings with a fixed covariance budget, the isotropic Gaussian uniquely minimises the bias of k-NN and kernel probes (LeJEPA Thm 1), and anisotropy
          hurts linear probes (Lemmas 1–2). Directions are re-drawn every step. With 2 <i>fixed</i> directions our z collapsed onto a line. In the sweep, k-NN rises with M: 43% at M = 2,
          56% at 16, 63% at 64 and 65% at 256. Re-drawing beats fixed directions (56% vs 51% at M = 16).
        </Note>
      </>
    ),
  },
  {
    id: "lejepa-back", group: "LeJEPA", atom: "14", title: "One loss, one backward, no teacher", regimes: TRAIN, focus: ["lejepa:enc", "lejepa:proj"], grad: true,
    body: (
      <>
        <P>
          <K>L = (1 − λ)·invariance + λ·SIGReg</K>, with λ = 0.05 (the paper&rsquo;s default and its only trade-off knob). One backward pass, one AdamW optimiser, fixed weight decay.
        </P>
        <P>
          The blocks show ∂L/∂(activation) for this image&rsquo;s invariance term, backpropagated live through the projector and the encoder. Switch views: the gradient reaches the{" "}
          <b>same weights</b> through every one of the 8 views. The SIGReg term&rsquo;s gradient needs a batch; our browser implementation matches PyTorch for both terms (tested on the
          goldens).
        </P>
      </>
    ),
  },
  // ---------------------------------------------------------------- Shared
  {
    id: "valweak", group: "Shared", title: "Why JEPA validation loss is a weak signal", regimes: ["val"], focus: ["ijepa:loss", "lejepa:proj"], widget: "valcurve",
    body: (
      <>
        <P>
          In a JEPA the target is itself produced by the network. An encoder can lower the loss by making its representations <i>easier to predict</i> rather than more useful, and a
          collapsed encoder reaches almost zero loss.
        </P>
        <P>
          In our I-JEPA baseline the val loss is lowest (0.057) in the early near-collapse phase and <i>rises</i> as features become useful. Late in training it keeps falling while k-NN
          accuracy slides from 51% to 48%. Without stop-gradient it falls to ~1e-6 while k-NN drops to chance. The curves: val loss vs k-NN for the selected run.
        </P>
        <Note>
          LeJEPA claims the opposite for its objective: its training loss correlates with linear-probe accuracy (Spearman ≈ 85%, ≈ 99% after dividing by λ^0.4) because SIGReg pins the
          scale. In our 49 runs it holds <b>within</b> a run: the median ρ(val loss, linear accuracy) over training is −0.98 for LeJEPA and +0.33 for I-JEPA. It does{" "}
          <b>not</b> hold <b>across</b> configurations at this scale: over 24 LeJEPA runs ρ = −0.21 (+0.03 after dividing by λ^0.4). Over 25 I-JEPA runs ρ = +0.75, inverted,
          because collapsed runs have the lowest loss. The scatter below plots every run.
        </Note>
      </>
    ),
  },
  {
    id: "diag", group: "Shared", atom: "15", title: "Collapse & geometry diagnostics", regimes: ALL, focus: ["ijepa:cloud", "lejepa:cloud"], widget: "cloud",
    body: (
      <>
        <P>
          The clouds are the pooled features of 512 held-out images from a logged training run, projected to 3D by PCA at the selected checkpoint (right-drag to orbit). The scale is fixed
          over the run, so collapse <b>shrinks</b> the cloud. Pick a run and scrub through training.
        </P>
        <P>
          Numbers: <b>RankMe</b> <Cite id="rankme" /> (entropy of the singular values), <b>effective dims</b> (participation ratio of the covariance) and <b>spread</b> (the fraction of
          feature energy that varies across images). A run is called collapsed when spread &lt; 1e-3 or RankMe &lt; 1.5. Raw pair-cosine is logged but misleads: pooled LayerNorm
          features share a large offset (0.98 for a healthy encoder).
        </P>
      </>
    ),
  },
  {
    id: "knn", group: "Shared", atom: "16", title: "Frozen features → k-NN, live", regimes: ["eval"], focus: ["ijepa:cloud", "lejepa:cloud"], widget: "knn",
    body: (
      <>
        <P>
          Both loaded encoders are frozen and run on 256 held-out images in a Web Worker. Their pooled features are classified by leave-one-out k-NN (cosine similarity, k = 10, labels
          never used in training). It is computed live, right now. The logged runs used the full protocol (10k labelled → 5k test).
        </P>
      </>
    ),
  },
  {
    id: "iso", group: "Shared", atom: "16", title: "Probe accuracy vs isotropy", regimes: ["eval"], focus: ["ijepa:cloud", "lejepa:cloud"], widget: "probe",
    body: (
      <>
        <P>
          LeJEPA&rsquo;s central claim is that isotropic embeddings make better probes. Every point is a trained run: x is the isotropy of its frozen features (participation ratio of
          the covariance), y its linear-probe accuracy. Click a point to load that run&rsquo;s cloud.
        </P>
        <Note>
          Over our 49 runs the rank correlation between isotropy and linear accuracy is ρ = 0.68 for LeJEPA and 0.42 for I-JEPA: a real but partial trend. Collapsed runs are anisotropic
          and probe badly. The linear-predictor I-JEPA, though, has the most isotropic features of its family and still probes worse than the baseline. Isotropy helps; it is not
          sufficient.
        </Note>
      </>
    ),
  },
  {
    id: "axes", group: "Shared", title: "How each hyperparameter moves the probes", regimes: ["eval"], focus: ["ijepa:cloud", "lejepa:cloud"], widget: "axes",
    body: (
      <>
        <P>
          Each bar is the final probe accuracy of one training run that changes a single setting from its baseline. Open the <b>explorer</b> (top right) for the full curves.
        </P>
      </>
    ),
  },
  // ---------------------------------------------------------------- Context
  {
    id: "contrast", group: "Context", title: "JEPA vs MAE, contrastive, BYOL/DINO, VICReg", regimes: ALL, focus: ["*"],
    body: (
      <>
        <P>
          <b>vs masked autoencoders</b> <Cite id="mae" />: MAE reconstructs <i>pixels</i>, spending capacity on unpredictable texture and noise. A JEPA predicts the <i>representation</i>,
          which can drop what cannot be predicted. The price is that a representation target can move, so collapse is a JEPA problem MAE does not have.
        </P>
        <P>
          <b>vs contrastive</b> <Cite id="simclr" />: negatives push different images apart (O(B²) similarities, large batches, hand-picked augmentations). I-JEPA uses no negatives
          and only crops. LeJEPA uses no negatives either; its anti-collapse term is a distribution constraint of cost O(B).
        </P>
        <P>
          <b>vs BYOL / DINO</b> <Cite id="byol" /> <Cite id="dino" />: the same EMA + stop-gradient machinery, but comparing global embeddings of two augmentations. I-JEPA&rsquo;s
          predictor is told <i>where</i> to predict. LeJEPA keeps DINO&rsquo;s multi-crop but deletes the teacher, centring and stop-gradient.
        </P>
        <P>
          <b>SIGReg vs VICReg vs EMA + stop-gradient</b> <Cite id="vicreg" />: EMA + stop-grad is an implicit, dynamical defence. Nothing in the loss forbids collapse, so it is
          sensitive to τ and to the predictor. VICReg is explicit but matches only two moments at cost O(B·K²). SIGReg matches the whole distribution slice by slice at cost O(B·K·M), and
          VICReg is its moment-matching limit (LeJEPA Thm 9).
        </P>
      </>
    ),
  },
  {
    id: "vjepa", group: "Context", title: "Where V-JEPA differs", regimes: ALL, focus: ["*"],
    body: (
      <>
        <P>
          <b>V-JEPA</b> <Cite id="vjepa" /> applies the I-JEPA recipe to video:
        </P>
        <ul className="mb-2 list-disc pl-4">
          <li>16-frame clips are cut into 2×16×16 <b>tubelets</b>;</li>
          <li>multi-block masks become <b>tubes through time</b>: 8 small short-range blocks or 2 large long-range blocks, masking ~90% of the clip;</li>
          <li>the loss is L1 against layer-normed EMA targets (τ 0.998 → 1);</li>
          <li>evaluation uses an <b>attentive probe</b> (cross-attention pooling), not mean-pooling.</li>
        </ul>
        <P>
          <b>V-JEPA 2</b> <Cite id="vjepa2" /> scales the encoder to ~1B parameters, uses 3D-RoPE and a constant EMA. It adds an action-conditioned predictor (V-JEPA 2-AC) trained on robot
          video and used for model-predictive planning. The anti-collapse answer is still EMA + stop-gradient; SIGReg would be a drop-in alternative.
        </P>
      </>
    ),
  },
];

export const phasesFor = (r: Regime) => PHASES.filter((p) => p.regimes.includes(r));
