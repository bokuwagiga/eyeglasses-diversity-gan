"""Metric-vs-epoch figure for the continued mirror2 run.

Reads the snapshot evaluations in results/mirror2_epoch_curve.csv and the
training KID series, and plots each metric against epoch with the measured
noise floor drawn as a band around the value of the selected checkpoint.

  python scripts/make_epoch_curve_figure.py --out article/figures
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NOISE = {'fid': 0.052, 'precision': 0.008, 'recall': 0.019,
         'ab_coverage': 0.016}


def read_curve(path):
    rows = []
    for r in csv.DictReader(l for l in open(path) if not l.startswith('#')):
        ep = int(r['epoch'])
        # the selected checkpoint is listed by its zero-based checkpoint index;
        # snapshots are listed by epochs completed
        if r['cell'].startswith('ckpt_best'):
            ep += 1
        rows.append({'epoch': ep,
                     'fid': float(r['fid']),
                     'precision': float(r['precision']),
                     'recall': float(r['recall']),
                     'ab_coverage': float(r['ab_coverage'])})
    return sorted(rows, key=lambda r: r['epoch'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curve', default='results/mirror2_epoch_curve.csv')
    ap.add_argument('--metrics',
                    default='results/mirror2/metrics/training_metrics.json')
    ap.add_argument('--start', type=int, default=1195)
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    rows = read_curve(args.curve)
    ref = rows[0]
    m = json.load(open(args.metrics))
    kid = [(e, k) for e, k in zip(m['epoch'], m['kid'])
           if k is not None and e >= args.start]

    fig, grid = plt.subplots(2, 3, figsize=(11, 6.4))
    axes = list(grid.flat)[:5]
    grid.flat[5].axis('off')
    ax = axes[0]
    ax.plot([e for e, _ in kid], [k * 1000 for _, k in kid], '-o', ms=3,
            color='0.3')
    ax.axvline(ref['epoch'], ls=':', color='tab:red', lw=1)
    ax.set_title('(a) Training KID (x1000)', fontsize=12)

    panels = [('fid', '(b) FID (lower is better)'),
              ('precision', '(c) Precision'),
              ('recall', '(d) Recall'),
              ('ab_coverage', '(e) Colour coverage')]
    eps = [r['epoch'] for r in rows]
    for ax, (key, title) in zip(axes[1:], panels):
        v0 = ref[key]
        ax.axhspan(v0 - NOISE[key], v0 + NOISE[key], color='tab:blue',
                   alpha=0.15, lw=0)
        ax.axhline(v0, color='tab:blue', lw=0.8)
        ax.plot(eps, [r[key] for r in rows], '-o', ms=4, color='tab:blue')
        ax.plot([ref['epoch']], [v0], 'o', ms=6, color='tab:red')
        ax.set_title(title, fontsize=12)

    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out / f'epoch_curve.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'  wrote {p}')


if __name__ == '__main__':
    main()
