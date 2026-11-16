"""Recompute the pose section (mirror symmetry and axis offset) of a diversity report.

Uses the same image subsamples as metrics/evaluate_diversity.py (seed 0, at
most 5,000 images per set) and writes the fixed-axis and pose-aligned symmetry
medians, the axis-offset dispersion and the per-image values. Only CPU work.

  python metrics/recompute_pose.py --real data/source/images \
      --generated results/mirror2/eval/generated/images --out results/mirror2/eval/pose_v2
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_metrics as dm  # noqa: E402
from evaluate_diversity import list_images, load_rgb  # noqa: E402


def axes(paths, desc):
    rows = []
    for p in tqdm(paths, desc=desc):
        sil = dm.extract_silhouette(load_rgb(p))
        r = dm.mirror_axis_symmetry(sil) if sil is not None else None
        if r is not None:
            rows.append([os.path.basename(p), r[0], r[1], r[2], r[3]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--generated', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-images', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gen_paths = list_images(args.generated)
    real_paths = list_images(args.real)

    def subsample(paths, n):
        if len(paths) <= n:
            return paths
        idx = rng.choice(len(paths), size=n, replace=False)
        return [paths[i] for i in sorted(idx)]

    gen_s = subsample(gen_paths, args.max_images)
    real_s = subsample(real_paths, args.max_images)

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.out)), '_pose_v2_real_cache.json')
    key = {'real': os.path.abspath(args.real), 'n': len(real_s), 'first': os.path.basename(real_s[0]),
           'last': os.path.basename(real_s[-1]), 'code': dm.code_version()}
    real_rows = None
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get('key') == key:
            real_rows = c['rows']
    if real_rows is None:
        real_rows = axes(real_s, 'real')
        json.dump({'key': key, 'rows': real_rows}, open(cache, 'w'))
    gen_rows = axes(gen_s, 'generated')

    g = np.array([r[1:] for r in gen_rows], dtype=float)
    r = np.array([x[1:] for x in real_rows], dtype=float)
    report = {
        'meta': {'real': args.real, 'generated': args.generated, 'n_real': len(real_rows),
                 'n_gen': len(gen_rows), 'seed': args.seed, 'code': dm.code_version()},
        'fixed_p50_gen': float(np.median(g[:, 0])), 'fixed_p50_real': float(np.median(r[:, 0])),
        'aligned_p50_gen': float(np.median(g[:, 1])), 'aligned_p50_real': float(np.median(r[:, 1])),
        'dispersion': dm.pose_dispersion(g[:, 2], r[:, 2]),
        'offcentre_frac_real': float(np.mean(np.abs(r[:, 2]) >= 3)),
        'offset_mean_gen': float(g[:, 2].mean()), 'offset_mean_real': float(r[:, 2].mean()),
    }
    json.dump(report, open(os.path.join(args.out, 'pose_report.json'), 'w'), indent=1)
    json.dump({'gen': gen_rows, 'real': real_rows},
              open(os.path.join(args.out, 'pose_scores.json'), 'w'))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
