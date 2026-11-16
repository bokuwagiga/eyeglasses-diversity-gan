"""Check how often the mirror-axis search is clipped by the image border.

mirror_axis_symmetry crops the silhouette to its bounding box plus a 16 px
margin and clips the crop at the canvas edge. When the margin on the two sides
is clipped by different amounts, the crop is no longer centred on the bounding
box and the recovered offset contains a lateral-placement term. This probe
measures, for a folder of images, the horizontal margins of the silhouette and
the axis offset with and without the clipping (the second by zero-padding the
silhouette before the search).

  python metrics/probe_axis_border.py --images data/source/images --out results/_probes/axis_border_real.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_metrics as dm  # noqa: E402
from evaluate_diversity import list_images, load_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-images', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pad', type=int, default=32)
    args = ap.parse_args()

    paths = list_images(args.images)
    if len(paths) > args.max_images:
        rng = np.random.default_rng(args.seed)
        paths = [paths[i] for i in sorted(rng.choice(len(paths), size=args.max_images, replace=False))]
    rows = []
    for k, p in enumerate(paths):
        sil = dm.extract_silhouette(load_rgb(p))
        if sil is None:
            continue
        xs = np.nonzero(sil)[1]
        a = dm.mirror_axis_symmetry(sil)
        b = dm.mirror_axis_symmetry(np.pad(sil, args.pad))
        if a is None or b is None:
            continue
        rows.append({'file': os.path.basename(p), 'left': int(xs.min()),
                     'right': int(sil.shape[1] - 1 - xs.max()),
                     'offset_clipped': a[2], 'aligned_clipped': a[1],
                     'offset_padded': b[2], 'aligned_padded': b[1], 'fixed_padded': b[0]})
        if k % 500 == 0:
            print('  %d / %d' % (k, len(paths)), flush=True)

    left = np.array([r['left'] for r in rows]); right = np.array([r['right'] for r in rows])
    oc = np.array([r['offset_clipped'] for r in rows]); op = np.array([r['offset_padded'] for r in rows])
    summary = {
        'n': len(rows),
        'margin_left_median': float(np.median(left)), 'margin_right_median': float(np.median(right)),
        'frac_any_margin_below_16': float(np.mean((left < 16) | (right < 16))),
        'frac_unequal_clipping': float(np.mean(np.minimum(left, 16) != np.minimum(right, 16))),
        'offset_clipped_mean': float(oc.mean()), 'offset_clipped_std': float(oc.std()),
        'offset_padded_mean': float(op.mean()), 'offset_padded_std': float(op.std()),
        'frac_offset_changed': float(np.mean(oc != op)),
        'frac_abs_ge3_clipped': float(np.mean(np.abs(oc) >= 3)),
        'frac_abs_ge3_padded': float(np.mean(np.abs(op) >= 3)),
        'corr_offset_clipped_vs_margin_diff': float(np.corrcoef(oc, right - left)[0, 1]),
        'corr_offset_padded_vs_margin_diff': float(np.corrcoef(op, right - left)[0, 1]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'summary': summary, 'rows': rows}, open(args.out, 'w'))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
