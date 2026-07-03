# Eyeglasses Diversity GAN

Research project: maximizing the diversity of a StyleGAN2-generated eyeglass
frame dataset (256x512 frontal catalogue images), with new eyeglass-tailored
diversity metrics and a systematic architecture/hyperparameter study.

Follow-up to the MSc thesis project `synthetic-eyeglasses-generator`
(conditional StyleGAN2 + downstream segmentation, IoU 0.867 vs 0.831 real baseline).

## Structure

- `gan/` - StyleGAN2 training and generation (no mask branch in this project)
- `metrics/` - diversity + quality metric suite
- `scripts/` - data preparation and utilities
- `notebooks/` - analysis notebooks
- `reference/` - read-only script snapshots from the thesis repo
- `article/` - article draft
- `data/`, `results/` - not git-tracked

## Research phases

1. **Metric suite** - eyeglass-tailored diversity metrics (geometric descriptors,
   frame-style distribution, perceptual colour space, feature-space
   recall/coverage/Vendi score), validated on three existing datasets
   (real, thesis-run generated, gan_20260515 generated).
2. **Base model fixes** - remove mask branch, fix ADA, re-balance discriminator.
3. **Diversity study** - architecture modifications and hyperparameter sweeps,
   evaluated on the metric suite; quality-diversity Pareto analysis.
4. **Comparison + article** - generated vs original dataset diversity.
