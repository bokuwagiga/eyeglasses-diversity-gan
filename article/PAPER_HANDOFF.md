# Paper handoff - everything needed to write the article

Single source of truth for writing the paper. Consolidates
`article/experiment_log.md` (2,326 lines of chronological notes),
`article/findings_checkpoint_sweep.md`, the figure/table assets, and the
supervisor correspondence. Written 2026-09-06.

Every number here traces to a file on disk. Where a number was later
withdrawn or corrected, the retraction is recorded next to it rather than
the number being deleted - several of the retractions are themselves
article material.

---

## 1. The article

**Title (working, from the supervisor's template):** Generating Diverse
Synthetic Eyeglass Datasets Using GANs. Target: MDPI. Template already
supplied by the supervisor at
`article/2026_Generating_Diverse_Synthetic_Eyeglass_Datasets_Using_GANs__MDPI_/`
(`article.tex`, `refs_LLM_embed.bib`, `images/`, `Definitions`).

**Supervisor:** Dalius Matuzevicius, Vilnius Tech. Author: Giga Shubitidze.

**Goal.** Maximise the *diversity* of a GAN-generated eyeglass-frame
dataset (256x512 RGB, frontal catalogue-style), while keeping quality -
i.e. produce and characterise a quality-diversity Pareto front rather than
a single "best FID" model.

**Core contribution.** A suite of eyeglass-tailored diversity metrics that
standard GAN evaluation (FID/KID/LPIPS) cannot express, plus the
experimental demonstration that optimising the standard metrics actively
destroys diversity.

**Explicitly out of scope.** No mask generation (semantic masks for frame
front + temples are a later article). This project removed the thesis's
mask branch entirely.

**Predecessor.** MSc thesis "Research on Generative AI Algorithms for
Building the Dataset of Eyeglass Frame Images" (Vilnius Tech, defended
2026). This is the follow-up article.

---

## 2. Data

- **Real set:** 6,511 catalogue frames, `data/source/images`, 256x512 (H x W,
  2:1), frontal, near-uniform background.
- Sources are ~1461 px square; the pipeline centre-crops to 2:1 then
  resizes. **The resize filter is a load-bearing detail - see section 9.1.**
- Frame silhouettes are extracted by background-distance thresholding, so
  no GAN masks are needed for geometry metrics.
- Effective data doubled to ~13k by dataset-level x-flips (added run 15).
- 6,511 images is small for a GAN. Almost every failure in this project
  traces back to the discriminator memorising this set.

---

## 3. Starting point (what the thesis left)

Base run `gan_20260515`: StyleGAN2, 8-layer mapping, ADA, PPL, feature
matching, KID model selection. Sharp samples, best domain metrics of the
thesis, but:

- colour diversity 0.057 vs 0.106 for an earlier thesis run
- FID 38.5
- ADA never activated (`ada_p ~= 0` for 400 epochs): D too weak to overfit
  (lr_d 5e-5 + R1 gamma 1.0 + narrow channels), so the heuristic never fired

Removing the mask supervision destabilised training badly, which produced
the first three failures below. That is worth one paragraph in the paper:
mask supervision had been silently grounding G early in training.

### 3.1 Three failures before a stable baseline existed

1. **Broken TTUR.** `lr_d` had been doubled to 1e-4; D outran G in 3
   epochs. Fixed to 5e-5 (G 2e-4). Also added integer-pixel translation to
   ADA (geometric augs are the strongest ADA category; the inherited
   pipeline had only flip/brightness/noise/cutout).
2. **Hinge loss has zero gradient outside its margins.** Inherited from
   mask-supervised runs where D never built margins > 1. Unconditionally,
   D separates real/fake early, hinge `d_loss` hits exactly 0, D gets no
   gradient, collapse trip-wire fires. Fixed: non-saturating logistic
   (softplus) loss, the standard StyleGAN2/ADA choice.
3. **Two generator defects** (the big one). Samples were saturated colour
   noise after 5 epochs.
   - **tanh on the RGB output.** Catalogue images are ~90% white
     background, so every loss pushes most pixels to +1 where tanh has zero
     gradient. Measured: 99.5% of pixels saturated, G gradient norm ~0
     within 10 optimizer steps. Official StyleGAN2 uses a **linear** RGB
     head. Fixed.
   - **Collapsed mapping network.** Eight stacked default `nn.Linear` +
     LeakyReLU shrink the w-code std to **0.018** at init - styles carried
     almost no information, and image variation came mostly from per-pixel
     noise injection. Fixed with equalized-lr linear layers, lr multiplier
     0.01 (official SG2 mapping). w std after fix: **1.04**.

   Defect 2 is a plausible cause of the thesis run's low colour diversity:
   with near-constant w, per-image colour is driven by noise injection
   rather than styles. **Worth a paragraph in the article.**

---

## 4. Final architecture and training recipe

StyleGAN2-ADA derivative, stock PyTorch only (no custom CUDA ops):

- 8-layer equalized-lr mapping network (lr mult 0.01), w_dim 512
- weight-demodulated synthesis, skip-RGB, **linear** RGB output
- self-attention at 32x64
- 6 synthesis stages: 8x16 -> 256x512
- G channels [512,256,256,128,64,32,16] (~8.5M params); D
  [16,32,64,128,256,256] (~2.5M). **Both width multipliers were tested and
  falsified - see 8.4.**
- logistic (non-saturating) loss + lazy R1 (every 16 steps), gamma 5
- PPL regularisation, weight 4.0, lazy every 16 steps, reduced PPL batch
  (16) to fit VRAM
- feature matching (random pairing - nearest pairing here was load-bearing
  *against* collapse, see 8.5), weight 2.0
- VGG perceptual loss 0.75 with **nearest-real** pairing
- ADA: flip / integer translation / brightness / noise / cutout, `ada_max_p`
  0.95. **Colour ADA (hue rotation + saturation jitter) OFF
  (`--ada-color-max-p 0`)** - see 8.2
- group-wise MinibatchSTD (groups of 4)
- blur-before-avgpool in D only (blurpool)
- dataset-level x-flips
- inverse-frequency sample weighting over the CIELAB a*b* histogram
  (`data/sample_weights_invfreq.json`, alpha 0.5, cap 4)
- EMA of G for all generation and evaluation
- batch 16 at 256x512 on an RTX 4090 (24 GB); ~180 s/epoch

**Checkpointing:** `checkpoint_latest.pth` and `checkpoint_best.pth`
(min training KID) only, **both overwritten**. This destroyed two results -
see 11.1. `--snapshot-every N` was added afterwards to keep permanent
generator-only snapshots (34 MB vs 166 MB).

---

## 5. The metric suite (the core contribution)

All computed without GAN masks. `metrics/`:

| module | what it does |
|---|---|
| `diversity_metrics.py` | geometry, colour, feature-space, perceptual, symmetry, pose |
| `evaluate_diversity.py` | the main report driver -> `diversity_report.json` |
| `diagnose_artifacts.py` | 12-check structural defect suite, real-percentile calibrated |
| `diagnose_ab_gap.py` | which CIELAB colour bins are missing, and which real images live there |
| `compute_sample_weights.py` | inverse-frequency oversampling weights from the ab histogram |
| `filter_by_precision.py` | manifold-based rejection sampling (PRDC precision mask) |
| `filter_by_artifacts.py` | clean-set builder from the artifact suite |

### 5.1 Metric families

- **Geometry (9 descriptors):** area ratio, filled ratio, aspect ratio,
  height fraction, mean/max rim thickness, hollowness, lens hole count,
  solidity. Reported as dispersion (std, p05-p95) vs real plus
  per-descriptor Jensen-Shannon distance.
- **Colour:** pairwise CIELAB distance of per-image frame colour;
  chroma-weighted circular hue entropy; a*b* histogram coverage
  (`ab_coverage`, 16x16 bins).
- **Feature space (Inception-2048):** Vendi score, improved
  precision/recall, density/coverage, nearest-real distance ratio
  (memorisation check).
- **Perceptual:** mean pairwise LPIPS vs the same statistic on reals.
- **Quality anchors:** FID, KID.
- **Structural artifacts (12 checks):** sharpness, rim_breaks,
  rim_contrast, n_fragments, speckle_frac, contour_wobble, symmetry_iou,
  edge_sym, **edge_sym_aligned**, appearance_sym, and others. Thresholds
  calibrated at real **p2-p98**.

### 5.2 The two genuinely new metrics (article-critical)

Both came out of a wrong reading of an existing metric, which is itself
part of the story.

**a) Pose-aligned mirror symmetry (`edge_sym_aligned`).**
`edge_sym` mirrors the rim boundary band about the bbox vertical centre.
That axis is only the true symmetry axis if the photograph is perfectly
fronto-parallel. Validated on synthetic silhouettes:

| silhouette | fixed | aligned |
|---|---|---|
| perfect | 1.0000 | 1.0000 |
| tilted 2 deg | 0.4867 | 0.9800 |
| true asymmetry | 0.4505 | 0.5221 |
| asymmetry + tilt | 0.2223 | 0.5167 |

A 2 deg tilt costs 0.51 of `edge_sym` while genuine temple asymmetry costs
0.07 - **the check is ~7x more sensitive to pose than to the defect it is
named for.** The aligned version searches horizontal shift (+-8 px) and
rotation (+-3 deg) and keeps the best IoU. 17.8 ms/image.

**b) Mirror-axis offset dispersion (yaw diversity).**
`pose_dispersion()` reports std / IQR / fraction off-centre of the
per-image mirror-axis offset, plus the gen/real std ratio. It exposes a
failure mode invisible to FID, KID, LPIPS and ab_coverage: a symmetry
prior collapsing pose variation.

| set | offset std | frac >= 3 px | ratio vs real |
|---|---|---|---|
| real | 2.89 | 41.9% | 1.00 |
| sharp1 ep1600 | 3.08 | 35.5% | **1.07** |
| mirror1 | 1.25 | 4.9% | **0.42** |
| mirror2 | 1.30 | 5.2% | **0.45** |

**Important correction (2026-08-20):** the offset is **translation-invariant
by construction** - the axis search re-centres on the bbox, so lateral
placement cannot register. It measures **differential temple
foreshortening, i.e. yaw**, not image placement. Docstrings and CLI help
that claimed otherwise were corrected in commit 7c69102. Do not repeat the
placement reading in the paper.

### 5.3 Measured noise floor (use this everywhere)

Same checkpoint, same protocol, generation seed 42 vs seed 0:

| metric | max abs delta |
|---|---|
| FID | **0.052** |
| recall | **0.019** |
| ab_coverage | **0.016** |
| precision | 0.008 |
| coverage | 0.003 |

**No difference smaller than this should be discussed anywhere in the
paper.** All existing tables should be re-checked against it.

---

## 6. The generation protocol (12 strategies from 5 levers)

Generation is a separate axis from training and was swept independently.
Levers: truncation psi; layered truncation (coarse/fine split); pure-z vs
w-blending; multi-modal truncation centres; post-hoc filtering.

Full sweep on the nocolorada checkpoint (FID / precision / recall /
ab_coverage / Vendi / yield):

| protocol | FID | prec | recall | ab_cov | Vendi | yield |
|---|---|---|---|---|---|---|
| psi 0.85 mixed | 13.08 | 0.963 | 0.103 | 0.219 | 2.29 | 100% |
| psi 1.0 mixed | 11.25 | 0.924 | 0.154 | 0.270 | 2.52 | 100% |
| psi 1.2 mixed | 10.46 | 0.844 | 0.235 | 0.352 | 2.83 | 100% |
| psi 1.0 pure-z | 10.41 | 0.890 | 0.189 | 0.313 | 2.70 | 100% |
| psi 1.2 pure-z | 10.76 | 0.765 | 0.324 | 0.395 | 3.05 | 100% |
| psi 1.2 pz + precision filter | 10.35 | 0.923 | 0.250 | 0.355 | 2.85 | 70% |
| **layered 1.0/1.2 @3, pure-z** | **10.20** | 0.879 | 0.212 | 0.367 | 2.74 | **100%** |
| layered + filter | 10.51 | 0.957 | 0.169 | 0.344 | 2.65 | 83% |
| layered + centres k=8 | 10.22 | 0.879 | 0.203 | 0.352 | 2.73 | 100% |

**Findings:**

1. **Truncation psi < 1 is a pure loss, not a trade-off.** psi 0.7 -> 1.0
   improved FID *and every diversity metric simultaneously*. psi 0.7 is not
   on the Pareto front at all. Quality and diversity improve together up to
   at least psi 1.2.
2. **w-blending suppresses diversity.** Pure-z (100% single z) beats the
   30/40/20/10 blended default at equal psi on nearly every axis. Blending
   multiple w vectors averages toward the mean.
3. **Layered truncation is the win.** Capping only the 3 coarse stages
   (silhouette/proportion) while leaving the 3 fine stages
   (colour/texture) at psi 1.2 gives the best FID at 100% yield, and beats
   the filtered psi-1.2 set on colour, because the stages carrying colour
   are never truncated.
4. **Multi-modal truncation centres (k=8) are a dead end.** FID 10.22 vs
   10.20 - a wash within noise. At psi >= 1.0 the truncation contraction is
   too weak for the choice of centre to matter. Documented negative result.
5. **Filtering became unnecessary.** Once layered truncation removed the
   manifold-outlier tail, stacking the precision filter moves along a
   trade-off curve instead of lifting the front.

**The frozen headline protocol, used for every result from July onward:**

```
--truncation-psi 1.0 --truncation-psi-fine 1.2 --truncation-cutoff 3 \
--pure-frac 1.0 --num-images 10000 --seed 42
```

**Caveat that must be stated:** this protocol was tuned on the
`nocolorada` checkpoint at FID 10.2 and then frozen. The final models are
at FID 3.6-4.1. Nobody has verified it is still optimal. Worse, the
best-vs-last sweep later showed nocolorada's *last* checkpoint beats the
one the protocol was tuned on (see 10.2).

---

## 7. Model progression (chronological, with reasoning)

19 runs. 18 trained models; 2 (`baseline`, `wide2`) diverged and were never
fully evaluated. Ordered as they happened.

| # | run | change | outcome |
|---|---|---|---|
| 1 | baseline | R1 gamma 2 | first stable run; D overfits, turnover ep115 |
| 2 | r1gamma5 | gamma 5 | KID 0.0041, 3x better; turnover delayed to ~220 |
| 3 | r1gamma10 | gamma 10 | over-regularised, strictly dominated |
| 4 | divcfg | perceptual weight 0 | quality collapse; ablation point |
| 5 | divcfg2 | nearest-real perceptual pairing + colour ADA | FID 12.5, but ab_cov halved |
| 6 | **nocolorada** | colour ADA OFF | FID 11.25, best of era; protocol tuned here |
| 7 | ppl4 | PPL 2.0 -> 4.0 | FID 8.90 at ep310; late collapse ep325+ |
| 8 | rebalance | x8/x4 colour oversampling | colour gap closed but overshot; FID 18.6 |
| 9 | **rebalance2** | invfreq oversampling (alpha 0.5, cap 4) | FID 7.80; headline for a while |
| 11 | edgeD | Sobel edge channel in D | **null result**; D's input was not the bottleneck |
| 12 | wide2 | G x2, D x2 | **diverged ~ep100**; wider D overfits *faster* |
| 13 | wide_g2d15 | G x2, D x1.5 | trained, but colour collapse (confounded, see 12) |
| 14 | sg2f | 5 SG2-faithful changes at once | **mode collapse**, recall 0.000 |
| 15 | rebalance3 | 3 defensible pieces kept | best recall so far, but colour collapse from a dropped flag |
| 16 | rebalance4 | flag restored, 800 epochs | FID 5.03, wins on everything |
| 17 | **sharp1** | resize-filter fix | FID 4.50 -> 3.90 -> 3.58 across ep800/1200/1600 |
| 18 | mirror1 | flip-concat mirror coupling | fixes symmetry, destroys colour + pose diversity |
| 19 | **mirror2** | learned per-sample mirror axis | **best diversity of any run** |

### 7.1 The key mechanism findings

**R1 gamma sweep brackets the optimum from both sides:**

| gamma | best train KID | at epoch | post-peak |
|---|---|---|---|
| 2 | 0.0114 | 115 | fast D-overfit turnover |
| 5 | **0.0041** | 190 | turnover delayed to ~220 |
| 10 | 0.0102 | ~370 | flat plateau, capped quality |

gamma 10 is *strictly dominated* by gamma 5 - worse quality AND worse on
nearly every diversity metric (recall 0.013 vs 0.102). **The "weaker D ->
more diversity" hypothesis is refuted.** Diversity needs a D strong enough
to *demand* coverage.

**Colour signature is diagnostic.** pairwise LAB up while ab_coverage and
hue entropy are down = a few well-separated colour clusters rather than
broad coverage. This pattern identified over-regularisation in gamma 10.

**Nearest-real perceptual pairing.** The mode-averaging problem was the
random *pairing*, not the perceptual gradient: pulling each fake toward a
random real converges on the dataset mean in expectation. Matching each
fake to its nearest real in the batch (pooled deepest VGG features,
selection under `no_grad`) fixed it. Verified: shuffled near-copies match
their sources (loss 0.27 matched vs 7.0 index-aligned).

**But nearest pairing on the FM loss too was catastrophic** (run 14): with
*both* reconstruction losses using nearest matching, the last
mode-*covering* force is gone, because a collapsed G is always near *some*
real. Random-pair FM was load-bearing in every good run.

**Colour rebalancing dose-response** (a figure in its own right):

| tilt (boosted share) | ab_coverage | FID |
|---|---|---|
| 0% | 0.4219 | 8.90 |
| 3.7% (invfreq) | 0.5391 | 7.80 |
| 20.2% (x8/x4) | 0.6836 (overshoot) | 18.60 |

The colour gap is fixable by *sampling alone* - no capacity change needed.
The `diagnose_ab_gap.py` analysis showed only 153 real images (2.35%) lived
in the missing bins, i.e. a **rare-mode problem**, dominated by
purple/violet/magenta acetate. That diagnosis is what made oversampling
the right lever rather than a guess.

**D memorisation runaway and the blurpool fix.** Mean D margin
(d_real - d_fake) over matched windows:

| run | ep400-450 | ep450-500 | ep500-550 | ep550-600 | ep700-744 |
|---|---|---|---|---|---|
| rebalance2 | 4.52 | 8.82 | 21.04 | 37.50 | - |
| rebalance4 | 1.90 | 2.53 | 3.41 | 4.43 | 7.18 |

Blur-before-avgpool denies D the high-frequency shortcuts it was
memorising the 6,511 reals with. reb3 and reb4 are the only runs with it,
which makes this the project's cleanest single-factor ablation.

**Attribution for rebalance2 -> rebalance4 (4 changes):**

| change | effect |
|---|---|
| colour ADA off | restores colour diversity (ab 0.18 -> 0.61) |
| D blurpool | prevents D memorisation runaway; recall up |
| dataset x-flips | fixes frame asymmetry (**but see retraction 12.2**) |
| group minibatch-std | standard SG2 practice, no isolated evidence |
| longer budget (800 ep) | KID still improving past reb2's 600 |

---

## 8. Falsified hypotheses (all article material)

1. **"Weaker D gives more diversity."** R1 gamma 10 refuted it - worse on
   nearly every diversity metric than gamma 5.
2. **"Colour ADA increases colour diversity."** It *cost* ab_coverage in
   every clean comparison and never bought anything at convergence.
   Mechanism: the ADA paper's augmentation-leaking regime - with `ada_p`
   pinned at its ceiling, hue/saturation jitter is applied nearly every
   step, D becomes colour-invariant, and G's colour distribution is left
   unconstrained. Signature in the ab_gap report: 111 missing bins, *all*
   chromatic (|a*| >= 8), neutrals fully covered.
3. **"D's input is the bottleneck" (edge channel).** Run 11 was a null
   result: defect rates digit-identical to the parent model
   (sharpness 8.06% vs 8.05%, rim_breaks 3.20% vs 3.20%).
   Also: no blend or scale schedule can hide a per-pixel signal from a
   pooling D - aggregate separability goes like SNR*sqrt(N) over ~131k
   pixels. Three separate attempts (full strength, linear down-scaling,
   noise blending) all failed identically at ep15-21.
4. **"G capacity is the bottleneck."** Tested from both ends and falsified.
   wide2 (G x2, D x2) diverged ~ep100 - **wider D overfits faster, not
   slower**, hitting the ADA ceiling by ep17. wide_g2d15 (G x2, D x1.5)
   trained but the 19M-parameter G converged onto dominant colour modes
   under sustained D pressure. **Data volume, not capacity, is the binding
   constraint.** (The colour part of this is confounded - see 12.3.)
5. **"Five SG2-faithful changes at once will help."** sg2f mode-collapsed:
   FID 41.5, recall 0.000, one pale wire-thin frame repeated. The
   "unprecedented stability" (margins +-0.2 for 800 epochs) was a toothless
   D, not health. Two culprits identified: G-side blur *on top of* bilinear
   upsample (5x cumulative low-pass on the skip-RGB path, so pale thin mush
   is G's optimum), and nearest-pairing FM removing the last mode-covering
   force. **Lesson: 5 entangled changes at once = no attribution without a
   failed run to pay for it.**
6. **"Asymmetry lives in the temples" (first test).** Refuted at ep800 -
   the frame front was worse (5.93% vs 4.20%). **Then un-refuted at
   ep1200:** at ep800 *both* regions were undertrained, masking the split.
   Given 400 more epochs the frame front converges to near-real symmetry
   (-3.6% residual) while temples barely move (-12.9%). **Method lesson: do
   not diagnose a defect as structural from an unconverged checkpoint.**
7. **"The model mirrors tortoiseshell patterns."** Refuted: over-mirroring
   on patterned frames was 0.36% against an expected 10%.
8. **"mirror1's yaw collapse came from reflecting about a fixed column."**
   This motivated mirror2 (learned per-sample axis) - about 60 GPU-hours.
   **Refuted:** mirror2's yaw std ratio is 0.45 vs mirror1's 0.42,
   essentially no recovery. Enforcing mirror symmetry forbids differential
   temple foreshortening regardless of which axis is reflected about.
   **Yaw collapse is intrinsic to a mirror prior.**

---

## 9. Methodological findings (each deserves its own subsection)

### 9.1 The train/eval resize-filter mismatch (strongest finding)

The artifact suite's one persistent gap was sharpness (gen 6.22% flagged
vs real 2.01%). Traced properly:

**Step 1 - the gap is global, not a bad tail.** Bbox Laplacian variance
medians: real 1125, rebalance4 753 (-33%). Gen p90 (1141) only just
reaches real p50. Meanwhile `rim_contrast` was nearly matched (344 vs 377):
outlines crisp, interiors soft.

**Step 2 - the deficit is identical in every run ever trained:**

| run | sharp p10 | p50 | p90 | rimC p50 | notes |
|---|---|---|---|---|---|
| REAL (bicubic) | 495 | 1125 | 1599 | 377 | |
| rebalance2 | 306 | 789 | 1203 | 370 | |
| edgeD | 306 | 789 | 1203 | 370 | |
| wide_g2d15 | 327 | 715 | 1100 | 343 | 2x G capacity |
| sg2f | 263 | 729 | 925 | 341 | equal-lr + G blur |
| rebalance3 | 337 | 761 | 1158 | 348 | blurpool |
| rebalance4 | 330 | 753 | 1141 | 344 | |

A +-5% band across 2x D, 2x G, equalized-lr, blurpool and no blurpool.
**An invariant that survives every architectural change is not
architectural.** (2x G was *worse* at 715, independently killing the
capacity explanation.)

**Step 3 - root cause.** The training dataloader and the metric pipeline
resized with different filters:

```
training: transforms.Resize((256,512))         -> PIL BILINEAR
metric:   load_rgb -> img.resize(..., BICUBIC) -> PIL BICUBIC
```

Both centre-crop identically, so geometry was never distorted - but the
sources are ~1461 px, a ~2.85x reduction, where the triangle filter erases
interior detail. Scoring 800 reals through each path:

| path | p10 | p50 | p90 |
|---|---|---|---|
| reals TRAIN (crop+bilinear) | 337 | 780 | 1113 |
| generated rebalance4 | 330 | 753 | 1141 |
| reals METRIC (crop+bicubic) | 495 | 1125 | 1599 |

**rebalance4 matched its own training distribution within 3.5% at every
percentile.** The generator was never under-rendering - it faithfully
reproduced targets ~30% softer than the reals it was scored against.

**Confirmation.** sharp1 = rebalance4 re-run with BICUBIC training
transform, `run_config.json` byte-identical except `output`. At **ep300**
already: sharpness p50 1113 = 98.9% of the real median, from a checkpoint
3x worse in KID than rebalance4's final. `rim_contrast` p50 344 -> 382
(real 377) moved with it, so this is rendered detail, not high-frequency
noise inflating a variance statistic.

**Hypotheses refuted along the way, all withdrawn:** G capacity; the VGG
perceptual loss acting as a blur prior (`perc_loss` flat at 2.65-2.87 was
a *converged* term, not a smoothing force); an aspect-ratio squash (gen
aspect 3.047 vs real 3.067).

**Article value:** a train/eval preprocessing mismatch silently capped
output quality across **six** architecture experiments and produced
**three** plausible-but-wrong explanations before it was found.

### 9.2 KID checkpoint selection is a collapse guard, not an optimiser

Driven by the supervisor's question: *"if the best KID does not lead to the
best set of metrics, then it is not good."* Answered three ways.

**(a) Three checkpoints of one run (sharp1):**

| checkpoint | FID | prec | recall | coverage | ab_cov | hue ent |
|---|---|---|---|---|---|---|
| ep~800 | 4.50 | 0.898 | 0.617 | 0.918 | **0.664** | **0.876** |
| ep~1200 | 3.90 | 0.887 | **0.655** | 0.930 | 0.629 | 0.865 |
| ep~1600 | **3.58** | **0.918** | 0.626 | **0.950** | 0.590 | 0.859 |

Every fidelity metric improves monotonically; every diversity metric peaks
earlier and declines. **Not a single exception in either direction.**

**(b) Best vs last for all 16 usable models** (`results/checkpoint_comparison.csv`):

- **5 runs: best is clearly right** - all late collapses. rebalance2
  FID 7.87 -> 71.13 (recall 0.382 -> 0.000); ppl4 8.90 -> 27.16; wide2
  14.18 -> 45.09; wide_g2d15 9.00 -> 11.95; divcfg 33.67 -> 48.69.
- **3 runs: last is better.** nocolorada is the clearest -
  FID 10.20 -> 8.93, recall 0.211 -> 0.353, coverage 0.746 -> 0.850, all
  at once.
- **3 runs: the axes disagree.** r1gamma5 (best FID, but colour 0.305 vs
  0.434 for last); rebalance4 (FID tied, recall 0.623 vs 0.656).
- **5 runs: no difference** above the noise floor.

**(c) The direct test - mirror2 continued 300 epochs with snapshots
every 50** (`results/mirror2_epoch_curve.csv`):

| epoch | FID | prec | recall | ab_cov |
|---|---|---|---|---|
| 1194 (old best) | **4.064** | **0.869** | 0.705 | **0.613** |
| 1250 | 4.363 | 0.844 | 0.688 | 0.574 |
| 1300 | 4.313 | 0.851 | 0.716 | 0.594 |
| 1350 | 4.153 | 0.855 | 0.703 | 0.574 |
| 1400 | 4.236 | 0.854 | 0.710 | 0.594 |
| 1450 | 4.173 | 0.857 | **0.717** | 0.605 |
| 1500 | 4.115 | 0.852 | 0.693 | 0.566 |

Training KID **improved** (0.000975 at ep1420 vs 0.001088 at ep1195) but
**no suite metric improved beyond the noise floor**, and ab_coverage fell
0.047 (3x noise). **Better KID did not give a better dataset.** This is the
demonstration the supervisor asked for, and the answer is negative.

**(d) The KID argmin is largely noise.** Training KID uses only
`kid_n_real = kid_n_fake = 1000`, subset size 100
(`gan/gan_train.py:100`). Over the 27 evals after ep1200 of sharp1:
std 0.000403, max/min spread 6x. Best (ep1555, 0.000443) beats last
(ep1600, 0.000638) by **0.48 sigma**.

**Recommendation for the paper:** describe the rule honestly as an early
stopping guard against collapse, not a claim of optimality; report the
epoch curve as a figure; apply the noise floor everywhere; raise KID
sampling to 5000 in future work.

### 9.3 Two silent failure modes found in the trainer

Both fixed in commit `d085510`; both are worth a methods note because
neither raises an error.

**AMP silently skips generator steps.** `GradScaler` skips the optimiser
step when gradients contain inf/NaN and says nothing - losses keep being
computed and logged while G never moves. **edgeD spent its final 20 epochs
that way:** its EMA weights are bit-identical between ep579 and ep599 (0 of
102 tensors differ), G differs in only 2 of 108 (spectral-norm power
iteration buffers), while D updated normally (49 of 51). It surfaced only
because both checkpoints scored the same FID to five decimals. Now counted
and reported per epoch, with `G DID NOT TRAIN` when an entire epoch is
skipped. On mirror2's continuation the counter read 64 skips over 300
epochs (max 10, at ep1201 right after resume) - benign scale
recalibration, materially different from edgeD's whole-epoch freeze.

**Partial EMA load.** The generate path copied EMA weights only for
parameters present in `ckpt['ema']`, silently keeping raw-G values for the
rest - a mixed generator with no warning. Now refuses.

### 9.4 A checkpoint can be invalidated by a code change with no key or shape mismatch

`sg2f` published FID 41.54 cannot be reproduced. The checkpoint loads with
**0 missing and 0 unexpected keys** under a strict load, and generates
recognisable frame geometry with a blown-out yellow/red/black palette
(FID 206, PRDC all exactly 0).

Cause: commit `c23dc0e` (2026-08-01) added G-side blur and equalized-lr;
sg2f trained and was evaluated under it; commit `894564f`
(2026-08-04 14:09) reverted both - **70 minutes after the published
evaluation**. Equalized-lr is a *runtime per-layer weight scale*, not a
stored parameter, so removing it changes what the same tensors compute
while every key and shape stays valid.

**Diagnostic worth publishing:** all-zero PRDC with a plausible FID is the
signature of a checkpoint/code mismatch, not of a bad model - a genuinely
weak model keeps sane precision (baseline 0.803, divcfg 0.378).

---

## 10. Final results

### 10.1 Main results table (`article/figures/results_table.{txt,tex}`)

| model | FID | KID | Prec | Recall | Dens | Cov | ab_cov | LPIPS | yaw ratio | mem |
|---|---|---|---|---|---|---|---|---|---|---|
| rebalance4 | 5.031 | 0.00197 | 0.835 | 0.623 | 0.948 | 0.886 | 0.6133 | 0.2302 | - | 1.03 |
| sharp1 ep800 | 4.499 | 0.00143 | 0.898 | 0.617 | 1.093 | 0.918 | 0.6641 | 0.2309 | - | 1.02 |
| sharp1 ep1200 | 3.901 | 0.00097 | 0.887 | 0.655 | 1.110 | 0.930 | 0.6289 | 0.2322 | - | 1.03 |
| sharp1 ep1600 | **3.583** | 0.00082 | **0.918** | 0.626 | 1.190 | **0.950** | 0.5898 | 0.2286 | 1.07 | 1.00 |
| mirror1 | 4.479 | 0.00138 | 0.848 | 0.667 | 0.920 | 0.920 | 0.5234 | 0.2358 | 0.42 | 1.03 |
| **mirror2** | 4.064 | 0.00116 | 0.869 | **0.705** | 1.020 | 0.924 | **0.6133** | 0.2313 | 0.45 | 1.02 |

Real reference: ab_coverage **0.5664**, LPIPS 0.2333, pairwise_lab 24.54,
hue_entropy 0.8607, Vendi 3.4, aligned sym p50 0.6055.

### 10.2 The two headline models

**The article puts forward two models, not one, and that is the point.**

- **mirror2 - the main result.** Best recall of any run (0.705), colour
  coverage above real, aligned symmetry gap -0.017 vs sharp1's -0.059,
  no memorisation (ratio 1.02). It fixes the one defect plain training
  could not, without over-mirroring texture (appearance_sym 1.06% vs
  mirror1's 0.30% against real 2.01%).
- **sharp1 ep1600 - the fidelity end.** Best FID (3.583), precision
  (0.918), coverage (0.950). Also the model that shows the cost:
  it has the *worst* diversity of the modern runs.

**Never order the results by FID alone without saying so.** FID is
dominated by fidelity and barely responds to diversity, so an FID-sorted
table silently re-ranks everything by the axis the paper is not studying.
Neither model dominates the other; that non-domination *is* the finding.

**Careful with colour:** both models sit *above* real ab_coverage (0.5664),
and sharp1 ep1600 (0.590) is the closer of the two. Exceeding the real
spread is not a win - it means placing frame colours where the catalogue
has no examples. Report distance-to-real, not raw coverage.

### 10.3 Model x generation-strategy matrix

`article/figures/ablation_matrix.{txt,csv,html}` - 19 evaluated rows x 12
strategies = 228 cells, **30 measured (13%)**.

The filled cells form an **L**, not a scatter: one dense row (nocolorada x
9 protocols) and one dense column (the layered protocol x 13 models). The
search was greedy/coordinate-descent - sweep protocols on one model, freeze
the winner, then vary architecture. **This means model x strategy
independence cannot be claimed, and the paper should say so** rather than
imply a factorial design. A full factorial would be ~2,400 evaluations, and
6 of the rows are known failures.

### 10.4 Filtering as a Pareto knob

The `unfiltered / max-flags 1 / max-flags 0` triple traces a front between
structural cleanliness and colour coverage (defects concentrate in the
rare-colour tail):

| set | FID | ab_cov (5k) | precision | recall |
|---|---|---|---|---|
| unfiltered | 7.80 | 0.5664 | 0.880 | 0.368 |
| max-flags 1 | 7.96 | 0.5469 | 0.892 | 0.342 |
| max-flags 0 | 8.57 | 0.5195 | 0.890 | 0.342 |

Note: this was measured on rebalance2. From rebalance4 onward the models
are at real-catalogue artifact levels **without any filtering**
(86.37% vs real 86.67%), which removes the filter as a confound from the
main Pareto figures. Keep filtering as its own analysis section.

---

## 11. What cannot be reproduced (state this in the paper)

### 11.1 Two results have no weights

`gan_train.py` kept only `checkpoint_latest.pth` and `checkpoint_best.pth`
and overwrote both. sharp1 was one run continued through **six segments**
to ep1600, so:

- **sharp1 ep800** - metrics recorded, **weights gone**
- **sharp1 ep1200** - metrics recorded, **weights gone** (and it is on the
  Pareto front in both panels)
- sharp1 ep1600, rebalance4, mirror1, mirror2 - weights intact

Their numbers stand as recorded measurements of a training trajectory, and
the trajectory is itself the finding, but they cannot carry a "headline
model" claim because nobody can ever look at their output.
`--snapshot-every` was added afterwards to prevent recurrence.

### 11.2 sg2f cannot be reproduced at all

See 9.4. Third unreproducible result, and the only one whose cause is
semantic rather than a deleted file.

### 11.3 mirror2 ep1194 lives in exactly one place

The 300-epoch continuation improved training KID, so the trainer
**overwrote** `results/mirror2/checkpoints/checkpoint_best.pth` with
ep1420. The ep1194 weights behind every published mirror2 number now exist
only at `results/_checkpoint_archive/mirror2_checkpoint_best.pth` on the
training PC. **This must be backed up elsewhere.**

### 11.4 Bitwise reproduction is not achievable and should not be claimed

cuDNN autotuning and non-deterministic reduction kernels are not pinned,
and each resume re-seeds the RNG from `cfg.seed` rather than restoring
generator state, so batch order after a resume is not what an uninterrupted
run would have had. **The honest claim is method-level:** fixed seed,
pinned commit, published config and data, with run-to-run variation not
quantified. Quantifying it would need a repeat run at a different seed
(~63 GPU-hours).

Since 2026-08-15 (commit `32059ba`), `run_config.json`,
`diversity_report.json` and `artifact_report.json` all carry the git
commit, a dirty-tree flag, and torch/CUDA/GPU strings; every launch appends
to `run_config_history.json`. **Runs before that date have reconstructed
provenance only.** sharp1's segment structure was reconstructed from
`total_hrs` resets:

| segment | epochs | hours | s/epoch |
|---|---|---|---|
| 1 | 1-300 | 14.9 | 179 |
| 2 | 301-500 | 10.2 | 183 |
| 3 | 501-800 | 15.3 | 184 |
| 4 | 801-1157 | 19.0 | 192 |
| 5 | 1158-1200 | 3.9 | 323 (machine was busy) |
| 6 | 1201-1600 | 20.4 | 184 |

Total 83.6 h over 6 segments.

---

## 12. Retractions - do NOT quote these

Every one of these was written down as a result and later withdrawn. They
are listed so they cannot leak into the draft.

1. **All "better than real" symmetry results from runs 1-16**
   (`edge_sym`, `appearance_sym`, `symmetry_iou`). Soft images have
   artificially symmetric edges, because the left/right detail differences
   that break symmetry are exactly what the bilinear filter erased. Do not
   quote any cross-`cd724ff` `edge_sym`.
2. **The x-flip credit for rebalance4's symmetry** (edge_sym 0.60%,
   appearance_sym 0.67%). Withdrawn - it was the blur confound above.
3. **"wide_g2d15 shows 2x-G capacity causes colour collapse."** Withdrawn:
   wide_g2d15 and sg2f both ran with colour ADA *on*, so their ab_coverage
   was never a clean measurement. sg2f's recall 0.000 mode collapse stands
   independently.
4. **"appearance_sym 1.44% vs real 2.01% is a win"** (run 17). The check is
   one-sided and the bulk distribution goes the other way.
5. **"mirror1 is much more symmetric than real."** 73% of its apparent lead
   was **pose, not shape**. Genuine shape advantage is +0.019, marginal
   parity - which matched the user's visual read against the metric.
6. **mirror1's clean rate 89.58% > real 86.67%** is an artifact of
   one-sided checks that cannot flag over-symmetry.
7. **"Kill the run above D margin ~10."** Wrong heuristic. rebalance2's
   runaway had margin climbing *while KID degraded 25x*; sharp1's margin
   climbed while KID *improved* 2x. Margin alone does not diagnose
   memorisation - check recall/density/nearest-real.
8. **"Asymmetry does not live in the temples"** (the ep800 refutation).
   Withdrawn at ep1200 - both regions were undertrained.
9. **The mirror-axis offset measures lateral placement.** It is
   translation-invariant by construction; it measures yaw.
10. **`results/ppl4/diversity` is VOID** - it measured the post-collapse
    ep400 checkpoint instead of ep310 because the eval used
    `find_latest_checkpoint`. Superseded by `ppl4_best`.
11. **Clean-rate comparability.** The check count changed from 11 to 12
    mid-project, and thresholds changed from p0.5-p99.5 to p2-p98. **Any
    table mixing them is wrong.** ep1600's headline 83.29% is not
    comparable to ep1200's 86.19%.
12. **KID is not comparable across the `cd724ff` boundary** (the bicubic
    fix changed the target distribution).

---

## 13. Method lessons (a short discussion subsection writes itself)

1. **Twice, an architecture change was built on an untested reading of a
   metric** - about 60 GPU-hours each time. First reading `edge_sym` as
   shape asymmetry when it was mostly pose; second assuming the axis offset
   measured lateral placement when it is translation-invariant. Both would
   have been refuted by a seconds-long synthetic-input test. **Rule adopted:
   any new metric needs a controlled-input test of what it responds to
   before conclusions - let alone experiments - are built on it.**
2. **Five entangled changes at once = no attribution** without a failed run
   to pay for it (sg2f).
3. **Every launch command must be diffed against the previous run's saved
   `run_config.json`** before it is issued. Three runs (wide_g2d15, sg2f,
   rebalance3) carried an unintended colour-ADA change because the command
   was retyped from plan text. This rule later caught a `--no-resume` in
   mirror2's config that would have destroyed 85 GPU-hours.
4. **Do not diagnose a defect as structural from an unconverged
   checkpoint.**
5. **A metric that "beats real" is a warning sign**, usually meaning the
   check is one-sided or confounded.
6. **Visual inspection outranks the metric suite.** The user's eyeball
   checks caught two blind spots the numbers missed.

---

## 14. Assets on disk

**Figures** (`article/figures/`):

| file | contents |
|---|---|
| `pareto.{png,pdf}` | two panels: FID vs recall, FID vs ab_coverage; Pareto staircase; real colour line |
| `results_table.{txt,tex}` | the 6-row main table, same script as the figure so they cannot disagree |
| `grid_models.png` | real / sharp1 ep1600 / mirror2, 18 random samples each, seed 42 |
| `grid_sharp1_ep1600.png`, `grid_mirror2.png` | per-model grids |
| `ablation_matrix.{txt,csv,html}` | 19 x 12 matrix; HTML pastes into email/Word |
| `evaluation_inventory.{txt,csv}` | every evaluation ever run, with provenance |

**Data files** (`results/`, not git-tracked):
`checkpoint_comparison.csv` (best vs last, 18 runs + 2 seed-noise rows),
`mirror2_epoch_curve.csv` (the snapshot curve).

**Honest caption for `grid_models.png`:** at n=18 the recall gap
(0.626 vs 0.705) is **not** visible. Both blocks are catalogue-quality.
The grid shows quality; it must not be captioned as showing the diversity
gap - the metrics carry that claim.

**Scripts:** `scripts/make_pareto_figure.py`,
`make_report_grid.py`, `make_ablation_matrix.py`,
`diagnose_symmetry_axis.py`, `diagnose_symmetry_region.py`,
`inspect_mirror_axis.py`.

---

## 15. Suggested paper structure

1. **Introduction** - synthetic catalogue data for eyeglass frames; why
   diversity, not just fidelity, is the useful property.
2. **Related work** - StyleGAN2/ADA, GAN evaluation (FID/KID/PRDC/Vendi),
   limited-data GANs, rejection sampling.
3. **Data and base model** - 6,511 frames; the thesis starting point; the
   three stabilisation fixes (tanh head, collapsed mapping, hinge loss).
4. **Metric suite** (the contribution) - families, and specifically the two
   new metrics with their synthetic validation tables (5.2).
5. **Methods: training** - final recipe; colour rebalancing from ab-gap
   diagnosis; blurpool; mirror coupling variants.
6. **Methods: generation protocol** - the 5 levers, the psi/pure-z finding,
   layered truncation.
7. **Results** - main table; Pareto front; the ablation matrix with its
   L-shape caveat.
8. **Checkpoint selection study** (9.2) - the supervisor asked for this and
   it is a self-contained result.
9. **Methodological findings** - resize-filter mismatch (9.1); silent
   trainer failures (9.3); checkpoint invalidation (9.4); the noise floor.
10. **Negative results and limitations** - section 8; yaw collapse as a
    measured property of symmetry priors; reproducibility limits (11).
11. **Conclusion and future work** - masks article; wider KID sampling;
    the 12-cell protocol re-sweep.

---

## 16. Open items

1. **The 12-cell protocol re-sweep.** The headline protocol was tuned on
   nocolorada at FID 10.2; current models are at 3.6-4.1. Re-running the 6
   main protocols on sharp1 ep1600 and mirror2 is 12 cells, generation +
   evaluation only, no training. This is the single most defensible gap.
2. **Re-check every existing table against the noise floor** (5.3). Some
   differences currently discussed may not survive FID 0.052 / recall
   0.019.
3. **`rebalance4`'s best checkpoint is recorded as both ep730
   (KID 0.00152) and ep745 (KID 0.00138)** in the same log entry, attached
   to the same results table. Needs `results/rebalance4/metrics/` to
   settle.
4. **The regenerated `checkpoint_comparison.csv`** with sg2f marked FAILED
   and edgeD annotated exists on the training PC but had not synced as of
   2026-09-06. The local copy still shows sg2f FID 206.28 as if it were a
   measurement.
5. **The yaw bound reading is unresolved and the paper does not need it.**
   Two readings fit the axis probe equally well: (a) the coupling is a yaw
   bottleneck, (b) the axis head merely follows what G already produced. A
   static probe cannot separate them. Cheap test if ever wanted: an
   inference-time gamma sweep (0, 0.5, 1.0) on the existing checkpoint,
   ~1 hour, no training.
6. **contour_wobble is untouched by everything tried.** p50 0.3633 (sharp1)
   vs 0.3631 (rebalance4) - identical despite the resize fix tripling
   sharpness. Its worst cases are thin-wire / rimless / transparent frames
   where threshold-based silhouette extraction is itself unreliable, so
   part of the gap is metric noise.

---

## 17. Supervisor correspondence - what has been promised

Latest exchange (2026-08-28). Dalius asked to focus on 1-2 day tasks and
approved the best-vs-last checkpoint evaluation for the article. He was
sent the four-group result and the mirror2 curve, and replied:

> Let's present both, like you described here. It was an experiment.
> Let's start writing. I will prepare LaTeX template today.

**Outstanding commitments:**

- a LaTeX table of the 16 models in the four groups (9.2b)
- a PDF figure of the mirror2 metric-vs-epoch curve (9.2c)
- both promised "by Monday"; Giga to start with the methods and results
  sections once the template lands

The template has since arrived at
`article/2026_Generating_Diverse_Synthetic_Eyeglass_Datasets_Using_GANs__MDPI_/`.

---

## 18. Conventions

- ASCII only in source files; no absolute paths (the project must copy
  as-is to the training PC at `C:\Users\AI\Desktop\eyeglasses-diversity-gan`)
- all data/output locations come from CLI arguments with relative defaults
- pip-installable from `requirements.txt` alone; stock PyTorch, no compiled
  custom CUDA ops
- every run writes `results/<run_name>/` with a config snapshot, `metrics/`,
  `samples/`, `checkpoints/`
- **commits must not mention AI assistance or carry Co-Authored-By trailers**
- `data/` and `results/` are not git-tracked; the dev machine has neither
  the dataset nor the generated images, only the JSON reports
