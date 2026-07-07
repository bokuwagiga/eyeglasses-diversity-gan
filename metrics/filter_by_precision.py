"""
Precision-based rejection filter for generated eyeglass datasets.

Idea: instead of lowering truncation psi (which discards good and bad
samples alike), keep a high-diversity generation setting and reject only
the specific samples that fall outside the real manifold in Inception
feature space (the same per-sample check used by the PRDC 'precision'
metric in diversity_metrics.py). This targets the visually-bad outliers
directly rather than trading away diversity uniformly.

Usage:
  python metrics/filter_by_precision.py --generated <dir> --real <dir> \
      --out <dir> [--k 5] [--max-real 5000] [--max-generated 10000]

Writes the kept images into --out (copied, original filenames preserved)
and a filter_report.json with kept/rejected counts and the fraction kept.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_metrics as dm
from evaluate_diversity import InceptionFeatures, list_images, load_images


def main():
    parser = argparse.ArgumentParser(
        description='Reject generated images that fall outside the real manifold.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generated', required=True, help='Generated images dir')
    parser.add_argument('--real', required=True, help='Real images dir')
    parser.add_argument('--out', required=True, help='Output dir for kept images')
    parser.add_argument('--k', type=int, default=5,
                        help='k for the k-NN real-manifold radius (same as PRDC)')
    parser.add_argument('--max-real', type=int, default=5000,
                        help='Max real images used to build the manifold')
    parser.add_argument('--max-generated', type=int, default=0,
                        help='Max generated images to filter (0 = all)')
    args = parser.parse_args()

    gen_paths = list_images(args.generated)
    real_paths = list_images(args.real)
    if args.max_generated > 0:
        gen_paths = gen_paths[:args.max_generated]
    real_paths = real_paths[:args.max_real]

    gen_images = load_images(gen_paths, 'Load generated')
    real_images = load_images(real_paths, 'Load real')

    extractor = InceptionFeatures()
    feats_gen = extractor.extract(gen_images, desc='Features gen')
    feats_real = extractor.extract(real_images, desc='Features real')

    prdc = dm.precision_recall_density_coverage(
        feats_real, feats_gen, k=args.k, return_mask=True)
    mask = prdc.pop('inside_real_mask')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for path, keep in zip(gen_paths, mask):
        if keep:
            shutil.copy2(path, out_dir / Path(path).name)
            kept += 1
    rejected = len(gen_paths) - kept

    report = {
        'generated_dir': args.generated, 'real_dir': args.real,
        'n_generated': len(gen_paths), 'n_real': len(real_paths),
        'k': args.k, 'kept': kept, 'rejected': rejected,
        'fraction_kept': kept / len(gen_paths),
        'precision_at_filter_time': prdc['precision'],
    }
    with open(out_dir.parent / 'filter_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Kept {kept}/{len(gen_paths)} ({report['fraction_kept']:.1%}) -> {out_dir}")


if __name__ == '__main__':
    main()
