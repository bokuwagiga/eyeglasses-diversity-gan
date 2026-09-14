"""Realistic novelty rate: generated frames that exist nowhere in the catalogue.

The diversity suite measures how well the generated set covers the real
distribution (recall, coverage, ab_coverage) and whether it copies it
(nearest-real memorisation ratio). This metric asks the remaining question:
does the generator produce frames that are genuinely NEW - far from every
catalogue image - while still being realistic frames rather than artifacts?

Construction:
  1. Reference set R = real images (plus their horizontal flips by default,
     since training saw x-flips; a mirrored copy of a catalogue frame is not
     a novel frame).
  2. Calibration: leave-one-out nearest-neighbour distance of each real
     image to the rest of R (its own flip excluded). The novelty threshold
     tau is the P-th percentile (default 95) of this distribution, i.e. the
     distance scale at which a real catalogue frame counts as "far" from
     the rest of the catalogue.
  3. A generated image is NOVEL if its nearest-reference distance exceeds
     tau, and REALISTIC if it lies inside the k-NN feature manifold of R
     (the improved-precision criterion). The headline number is the
     realistic novelty rate: the fraction of generated images that are both.

Calibration checks (both cheap, both reported):
  - gen == real with the same subsample -> every distance is 0, novelty 0.
  - gen = a disjoint half of the real set -> real images are exchangeable
    with the reference, so the novelty rate should sit near (100 - P)%.

Features are Inception-v3 pool3 (2048-d), the same space as FID/PRDC in
this project. All distances are Euclidean in that space.

  python metrics/novelty_rate.py --real data/source/images \
      --generated results/<run>/generated/images --out results/<run>/novelty
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.diversity_metrics import (  # noqa: E402
    code_version, precision_recall_density_coverage)
from metrics.evaluate_diversity import (  # noqa: E402
    InceptionFeatures, list_images, load_rgb)


def subsample(paths, n, seed):
    if n is None or n >= len(paths):
        return list(paths)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def extract(inc, paths, desc, flip=False):
    imgs = []
    for p in paths:
        a = load_rgb(p)
        imgs.append(np.fliplr(a).copy() if flip else a)
    return inc.extract(imgs, desc=desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--generated', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-images', type=int, default=5000,
                    help='subsample size per set (matches the eval protocol)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gen-seed', type=int, default=None,
                    help='separate subsample seed for the generated set '
                         '(used by the disjoint-real calibration check)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--pct', type=float, default=95.0)
    ap.add_argument('--no-flips', action='store_true',
                    help='exclude horizontal flips from the reference set')
    ap.add_argument('--copy-top', type=int, default=24,
                    help='copy the N most novel realistic images (with their '
                         'nearest real) into <out>/novel_examples')
    ap.add_argument('--calibration-check', action='store_true',
                    help='ignore --generated: split the real subsample into '
                         'two disjoint halves and score one against the '
                         'other. The novelty rate should land near '
                         '(100 - pct) percent.')
    args = ap.parse_args()

    if args.calibration_check:
        n = args.max_images if args.max_images else 5000
        pool = subsample(list_images(args.real), 2 * (n // 2), args.seed)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(pool))
        real_paths = [pool[i] for i in sorted(perm[:len(pool) // 2])]
        gen_paths = [pool[i] for i in sorted(perm[len(pool) // 2:])]
        gen_seed = args.seed
    else:
        real_paths = subsample(list_images(args.real), args.max_images,
                               args.seed)
        gen_seed = args.seed if args.gen_seed is None else args.gen_seed
        gen_paths = subsample(list_images(args.generated), args.max_images,
                              gen_seed)

    inc = InceptionFeatures()
    f_real = extract(inc, real_paths, 'Features (real)')
    f_gen = extract(inc, gen_paths, 'Features (generated)')
    ref_feats = [f_real]
    if not args.no_flips:
        ref_feats.append(extract(inc, real_paths, 'Features (real, flipped)',
                                 flip=True))
    ref = np.concatenate(ref_feats).astype(np.float32)
    n_real = len(f_real)

    from scipy.spatial.distance import cdist

    # Leave-one-out calibration over the real set. Each real row excludes
    # itself and, when flips are in the reference, its own flip.
    d_loo = cdist(f_real.astype(np.float32), ref)
    idx = np.arange(n_real)
    d_loo[idx, idx] = np.inf
    if not args.no_flips:
        d_loo[idx, idx + n_real] = np.inf
    d_cal = d_loo.min(axis=1)
    tau = float(np.percentile(d_cal, args.pct))

    d_gen_all = cdist(f_gen.astype(np.float32), ref)
    nn_idx = d_gen_all.argmin(axis=1)
    d_gen = d_gen_all[np.arange(len(f_gen)), nn_idx]
    novel = d_gen > tau

    prdc = precision_recall_density_coverage(ref, f_gen.astype(np.float32),
                                             k=args.k, return_mask=True)
    inside = prdc.pop('inside_real_mask')
    realistic_novel = novel & inside

    def pcts(a):
        return {p: round(float(np.percentile(a, p)), 3)
                for p in (5, 25, 50, 75, 95)}

    report = {
        'config': {
            'real': args.real, 'generated': args.generated,
            'n_real': n_real, 'n_generated': len(f_gen),
            'reference_includes_flips': not args.no_flips,
            'reference_size': int(ref.shape[0]),
            'k': args.k, 'threshold_percentile': args.pct,
            'seed': args.seed, 'gen_seed': gen_seed,
        },
        'code': code_version(),
        'tau': round(tau, 3),
        'calibration_nn_dist': pcts(d_cal),
        'generated_nn_dist': pcts(d_gen),
        'novelty_rate': round(float(novel.mean()), 4),
        'realism_rate_inside_manifold': round(float(inside.mean()), 4),
        'realistic_novelty_rate': round(float(realistic_novel.mean()), 4),
        'novel_but_unrealistic_rate':
            round(float((novel & ~inside).mean()), 4),
        'prdc_vs_reference': {k2: round(v, 4) if isinstance(v, float) else v
                              for k2, v in prdc.items()},
    }

    os.makedirs(args.out, exist_ok=True)
    order = np.argsort(-d_gen)
    with open(os.path.join(args.out, 'novelty_scores.csv'), 'w',
              newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['gen_file', 'nn_real_file', 'nn_dist', 'novel',
                     'inside_real_manifold'])
        for i in order:
            wr.writerow([os.path.basename(gen_paths[i]),
                         os.path.basename(real_paths[nn_idx[i] % n_real]),
                         round(float(d_gen[i]), 3), int(novel[i]),
                         int(inside[i])])

    copied = 0
    if args.copy_top:
        ex_dir = os.path.join(args.out, 'novel_examples')
        os.makedirs(ex_dir, exist_ok=True)
        for rank, i in enumerate([j for j in order if realistic_novel[j]]):
            if copied >= args.copy_top:
                break
            shutil.copy2(gen_paths[i], os.path.join(
                ex_dir, '%02d_gen_%s' % (rank, os.path.basename(gen_paths[i]))))
            shutil.copy2(real_paths[nn_idx[i] % n_real], os.path.join(
                ex_dir, '%02d_nnreal_%s'
                % (rank, os.path.basename(real_paths[nn_idx[i] % n_real]))))
            copied += 1
    report['n_example_pairs_copied'] = copied

    with open(os.path.join(args.out, 'novelty_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('reference size          %d (flips: %s)'
          % (ref.shape[0], not args.no_flips))
    print('tau (real p%.0f LOO)     %.3f' % (args.pct, tau))
    print('novelty rate            %.4f' % report['novelty_rate'])
    print('realism rate            %.4f' % report['realism_rate_inside_manifold'])
    print('REALISTIC NOVELTY RATE  %.4f' % report['realistic_novelty_rate'])
    print('novel but unrealistic   %.4f' % report['novel_but_unrealistic_rate'])
    print('written: %s' % os.path.join(args.out, 'novelty_report.json'))


if __name__ == '__main__':
    main()
