"""
Compute per-image training sample weights from the ab-gap diagnosis.

Reads the missing / underrepresented a*b* bins found by diagnose_ab_gap.py,
assigns every real image a sampling weight:

    missing bin           -> --missing-weight  (default 8)
    underrepresented bin  -> --under-weight    (default 4)
    everything else       -> 1

and writes sample_weights.json mapping image FILENAME -> weight, consumed by
gan/gan_train.py --sample-weights. Keys are basenames so the file stays valid
when the data folder lives at a different path on the training PC.

Usage:
  python metrics/compute_sample_weights.py \
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


def main():
    ap = argparse.ArgumentParser(
        description='Per-image sampling weights from the ab-gap diagnosis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--real', required=True, help='Real images dir')
    ap.add_argument('--ab-gap', required=True,
                    help='ab_gap_report.json from diagnose_ab_gap.py')
    ap.add_argument('--out', required=True, help='Output sample_weights.json')
    ap.add_argument('--missing-weight', type=float, default=8.0)
    ap.add_argument('--under-weight', type=float, default=4.0)
    args = ap.parse_args()

    with open(args.ab_gap) as f:
        gap = json.load(f)
    missing_bins = {bin_index(d['a_center'], d['b_center'])
                    for d in gap['missing_bins']}
    under_bins = {bin_index(d['a_center'], d['b_center'])
                  for d in gap['underrepresented_bins']}

    paths = list_images(args.real)
    kept, means = mean_frame_colors(paths, 'Real frame colours')
    kept_set = set(kept)

    weights = {}
    counts = Counter()
    for p, m in zip(kept, means):
        ij = bin_index(m[1], m[2])
        if ij in missing_bins:
            w, tag = args.missing_weight, 'missing'
        elif ij in under_bins:
            w, tag = args.under_weight, 'under'
        else:
            w, tag = 1.0, 'normal'
        weights[Path(p).name] = w
        counts[tag] += 1
    # Images whose silhouette extraction failed: neutral weight
    for p in paths:
        if p not in kept_set:
            weights[Path(p).name] = 1.0
            counts['failed_sil'] += 1

    total_w = sum(weights.values())
    boosted_w = sum(w for w in weights.values() if w > 1.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'meta': {
                'real_dir': args.real, 'ab_gap_report': args.ab_gap,
                'missing_weight': args.missing_weight,
                'under_weight': args.under_weight,
                'n_images': len(weights), 'counts': dict(counts),
                'boosted_share_of_samples': boosted_w / total_w,
            },
            'weights': weights,
        }, f, indent=2)

    print(f'Images: {len(weights)}  '
          f'missing x{args.missing_weight:g}: {counts["missing"]}  '
          f'under x{args.under_weight:g}: {counts["under"]}  '
          f'normal: {counts["normal"]}  failed_sil: {counts["failed_sil"]}')
    print(f'Boosted images will make up {100 * boosted_w / total_w:.1f}% '
          f'of drawn training samples (was '
          f'{100 * (counts["missing"] + counts["under"]) / len(weights):.1f}%)')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
