"""
Compute per-image training sample weights to counter rare-colour mode dropping.

Two modes:

1. invfreq (default, recommended): smooth inverse-frequency weighting from the
   real set's own a*b* histogram (16x16 bins of per-image mean frame colour,
   same binning as the ab_coverage metric):

       w = clip((c_ref / c_bin) ** alpha, 1, cap)

   where c_bin is the image's bin count and c_ref the median occupied bin
   count. Rare bins are boosted proportionally to their rarity, dense bins
   stay at 1, no cliff between adjacent bins. alpha=0.5 (sqrt tempering) and
   cap=4 give a mild tilt (~8-10% boosted share of drawn samples).
   No ab-gap report needed.

2. bins: binary boost of the missing/underrepresented bins listed in an
   ab_gap_report.json (diagnose_ab_gap.py). Used by the first rebalance run
   (x8/x4, 20% boosted share) - too aggressive: colour coverage overshot the
   real set while FID doubled and recall collapsed. Kept for reference.

Writes sample_weights.json mapping image FILENAME -> weight, consumed by
gan/gan_train.py --sample-weights. Keys are basenames so the file stays valid
when the data folder lives at a different path on the training PC.

Usage (invfreq):
  python metrics/compute_sample_weights.py \
      --real data/source/images --out data/sample_weights_invfreq.json

Usage (bins):
  python metrics/compute_sample_weights.py --mode bins \
      --real data/source/images \
      --ab-gap results/ppl4_best/ab_gap/ab_gap_report.json \
      --out data/sample_weights.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_ab_gap import AB_LO, AB_HI, N_BINS, mean_frame_colors
from evaluate_diversity import list_images


def bin_index(a, b):
    step = (AB_HI - AB_LO) / N_BINS
    i = int(np.clip((a - AB_LO) / step, 0, N_BINS - 1))
    j = int(np.clip((b - AB_LO) / step, 0, N_BINS - 1))
    return i, j


def weights_invfreq(kept, means, alpha, cap):
    """Smooth inverse-frequency weights from the real ab histogram."""
    bin_counts = Counter(bin_index(m[1], m[2]) for m in means)
    c_ref = float(np.median(list(bin_counts.values())))
    weights = {}
    for p, m in zip(kept, means):
        c = bin_counts[bin_index(m[1], m[2])]
        w = (c_ref / c) ** alpha
        weights[Path(p).name] = float(np.clip(w, 1.0, cap))
    meta = {'mode': 'invfreq', 'alpha': alpha, 'cap': cap,
            'median_occupied_bin_count': c_ref,
            'n_occupied_bins': len(bin_counts)}
    return weights, meta


def weights_bins(kept, means, gap_path, missing_weight, under_weight):
    """Binary boost of missing/underrepresented bins from an ab-gap report."""
    with open(gap_path) as f:
        gap = json.load(f)
    missing_bins = {bin_index(d['a_center'], d['b_center'])
                    for d in gap['missing_bins']}
    under_bins = {bin_index(d['a_center'], d['b_center'])
                  for d in gap['underrepresented_bins']}
    weights = {}
    counts = Counter()
    for p, m in zip(kept, means):
        ij = bin_index(m[1], m[2])
        if ij in missing_bins:
            w, tag = missing_weight, 'missing'
        elif ij in under_bins:
            w, tag = under_weight, 'under'
        else:
            w, tag = 1.0, 'normal'
        weights[Path(p).name] = w
        counts[tag] += 1
    meta = {'mode': 'bins', 'ab_gap_report': str(gap_path),
            'missing_weight': missing_weight, 'under_weight': under_weight,
            'counts': dict(counts)}
    return weights, meta


def main():
    ap = argparse.ArgumentParser(
        description='Per-image sampling weights to counter rare-colour dropping.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--real', required=True, help='Real images dir')
    ap.add_argument('--out', required=True, help='Output sample_weights.json')
    ap.add_argument('--mode', choices=['invfreq', 'bins'], default='invfreq')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='invfreq: tempering exponent (0 = uniform, 1 = full '
                         'inverse frequency)')
    ap.add_argument('--cap', type=float, default=4.0,
                    help='invfreq: maximum weight')
    ap.add_argument('--ab-gap', default=None,
                    help='bins mode: ab_gap_report.json from diagnose_ab_gap.py')
    ap.add_argument('--missing-weight', type=float, default=8.0,
                    help='bins mode: weight for missing-bin images')
    ap.add_argument('--under-weight', type=float, default=4.0,
                    help='bins mode: weight for underrepresented-bin images')
    args = ap.parse_args()

    if args.mode == 'bins' and not args.ab_gap:
        ap.error('--mode bins requires --ab-gap')

    paths = list_images(args.real)
    kept, means = mean_frame_colors(paths, 'Real frame colours')
    kept_set = set(kept)

    if args.mode == 'invfreq':
        weights, meta = weights_invfreq(kept, means, args.alpha, args.cap)
    else:
        weights, meta = weights_bins(kept, means, args.ab_gap,
                                     args.missing_weight, args.under_weight)

    # Images whose silhouette extraction failed: neutral weight
    n_failed = 0
    for p in paths:
        if p not in kept_set:
            weights[Path(p).name] = 1.0
            n_failed += 1

    total_w = sum(weights.values())
    boosted = {k: w for k, w in weights.items() if w > 1.0}
    boosted_w = sum(boosted.values())
    meta.update({
        'real_dir': args.real, 'n_images': len(weights),
        'n_failed_silhouette': n_failed,
        'n_boosted': len(boosted),
        'boosted_share_of_samples': boosted_w / total_w,
        'max_weight': max(weights.values()),
        'mean_weight': total_w / len(weights),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'meta': meta, 'weights': weights}, f, indent=2)

    print(f'Mode: {args.mode}  images: {len(weights)}  '
          f'boosted (w>1): {len(boosted)}  failed_sil: {n_failed}')
    print(f'Max weight: {meta["max_weight"]:.2f}  '
          f'mean weight: {meta["mean_weight"]:.3f}')
    print(f'Boosted images will make up {100 * boosted_w / total_w:.1f}% '
          f'of drawn training samples '
          f'(uniform would be {100 * len(boosted) / len(weights):.1f}%)')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
