"""Does the ab_coverage ranking depend on the histogram bin count?

ab_coverage is the fraction of occupied bins in a 16x16 histogram of
per-image mean frame colour over CIELAB a*b* in [-40, 40]. Occupancy-based
coverage depends on binning, so this probe recomputes it at 12x12, 16x16
and 20x20 for the real set and any number of generated sets, on matched
subsamples. The claim it supports is not that the absolute value is stable
(it cannot be) but that the ORDERING of the compared sets is.

  python metrics/ab_bin_stability.py --real data/source/images \
      --sets name1=path1 name2=path2 --out results/_probes/ab_bin_stability.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.diversity_metrics import (  # noqa: E402
    code_version, extract_silhouette, frame_lab_stats)
from metrics.evaluate_diversity import list_images, load_rgb  # noqa: E402


def mean_colours(directory, n, seed):
    paths = list_images(directory)
    if n and n < len(paths):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    means, failed = [], 0
    for p in paths:
        img = load_rgb(p)
        sil = extract_silhouette(img)
        if sil is None:
            failed += 1
            continue
        mean_lab, _ = frame_lab_stats(img, sil)
        means.append(mean_lab)
    return np.asarray(means), failed


def coverage(means, bins):
    hist, _, _ = np.histogram2d(means[:, 1], means[:, 2], bins=bins,
                                range=[[-40, 40], [-40, 40]])
    return float((hist > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sets', nargs='+', required=True,
                    help='name=path pairs of generated image directories')
    ap.add_argument('--bins', type=int, nargs='+', default=[12, 16, 20])
    ap.add_argument('--max-images', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/_probes/ab_bin_stability.json')
    args = ap.parse_args()

    sets = [('real', args.real)]
    for spec in args.sets:
        name, path = spec.split('=', 1)
        sets.append((name, path))

    rows = {}
    for name, path in sets:
        means, failed = mean_colours(path, args.max_images, args.seed)
        rows[name] = {'n': len(means), 'failed_silhouettes': failed,
                      'coverage': {str(b): round(coverage(means, b), 4)
                                   for b in args.bins}}
        print('%-14s n=%-5d  %s' % (name, len(means), rows[name]['coverage']))

    rankings = {}
    for b in args.bins:
        order = sorted((n for n, _ in sets),
                       key=lambda n: -rows[n]['coverage'][str(b)])
        rankings[str(b)] = order
    stable = len({tuple(v) for v in rankings.values()}) == 1
    print('ranking per bin count: %s' % rankings)
    print('ordering stable across bin counts: %s' % stable)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'code': code_version(),
                   'results': rows, 'rankings': rankings,
                   'ordering_stable': stable}, f, indent=2)
    print('written: %s' % args.out)


if __name__ == '__main__':
    main()
