# data/ - expected layout (not git-tracked)

Copy these from the thesis repo / previous runs onto the training PC:

```
data/
  source/
    images/          ~6,500 real catalogue frame images (from thesis repo
                     data/source/images) - GAN TRAINING SET and the real
                     reference for all diversity comparisons
  generated_thesis/
    images/          9,695 accepted images from the thesis GAN run
                     (thesis repo results/gan/generated or the accepted set)
  generated_20260515/
    images/          10,000 images from results/gan_20260515/generated/images
  smoke/             tiny subsets for local smoke tests (created by dev scripts)
```

Only `source/images` is required for training. The two `generated_*` sets are
used to validate the diversity metric suite (it should rank
thesis-run > 20260515-run on diversity).

Typical commands (run from the project root):

```
# Train
python gan/gan_train.py --images data/source/images --output results/<run_name> --end-epoch 400

# Generate 10k images from the best checkpoint
python gan/gan_train.py --generate-only results/<run_name>/checkpoints/checkpoint_best.pth \
    --output results/<run_name> --truncation-psi 0.7

# Evaluate diversity vs real
python metrics/evaluate_diversity.py --generated results/<run_name>/generated/images \
    --real data/source/images --out results/<run_name>/diversity
```
