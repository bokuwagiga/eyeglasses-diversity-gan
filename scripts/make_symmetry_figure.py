"""Figure for the pose-aligned symmetry axis and the axis-offset dispersion.

Top row: two real catalogue frames with the bounding-box centre line and the
recovered mirror axis. Bottom row: distribution of the axis offset in the
real set against each generated set.

The real offsets are computed on the same 5,000-image subsample that
metrics/evaluate_diversity.py draws (seed 0, 10,000 generated images) and are
cached in a JSON file. The generated offsets are read from the scores_gen.json
files written by metrics/diagnose_artifacts.py.

  python scripts/make_symmetry_figure.py --real data/source/images \
      --sets "Plain=<scores_gen.json>=<diversity_report.json>" ... \
      --out article/figures
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'metrics'))
import diversity_metrics as dm  # noqa: E402
from evaluate_diversity import list_images, load_rgb  # noqa: E402

BLUE = '#1f5fa8'
GREY = '0.72'


def real_offsets(real_dir, cache, n_gen=10000, n=5000, seed=0):
    if os.path.exists(cache):
        return json.load(open(cache))
    paths = list_images(real_dir)
    rng = np.random.default_rng(seed)
    rng.choice(n_gen, size=n, replace=False)
    idx = sorted(rng.choice(len(paths), size=n, replace=False))
    rows = []
    for k, i in enumerate(idx):
        sil = dm.extract_silhouette(load_rgb(paths[i]))
        r = dm.mirror_axis_symmetry(sil) if sil is not None else None
        if r is not None:
            rows.append({'file': os.path.basename(paths[i]), 'fixed': r[0],
                         'aligned': r[1], 'offset': r[2], 'rot': r[3]})
        if k % 500 == 0:
            print('  real %d / %d' % (k, len(idx)), flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(rows, open(cache, 'w'))
    return rows


def axis_panel(ax, path, title):
    img = load_rgb(path)
    sil = dm.extract_silhouette(img)
    fixed, aligned, shift, rot = dm.mirror_axis_symmetry(sil)
    xs = np.nonzero(sil)[1]
    centre = (xs.min() + xs.max()) / 2.0
    ax.imshow(img)
    ax.axvline(centre, color='0.35', lw=1.0, ls='--')
    ax.axvline(centre + shift / 2.0, color=BLUE, lw=1.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('0.8')
    ax.set_title('%s offset %d px, aligned IoU %.2f' % (title, int(shift), aligned),
                 fontsize=8.5, loc='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sets', nargs='+', required=True, help='label=scores_gen.json=diversity_report.json')
    ap.add_argument('--cache', default='results/_probes/real_axis_offsets.json')
    ap.add_argument('--examples', nargs=2, default=None,
                    help='two real file names (frontal, rotated); chosen automatically if omitted')
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    rows = real_offsets(args.real, args.cache)
    off_real = np.array([r['offset'] for r in rows])
    print('real: n %d, offset std %.3f, |offset|>=3 %.3f'
          % (len(off_real), off_real.std(), np.mean(np.abs(off_real) >= 3)))

    if args.examples:
        ex = args.examples
    else:
        front = max((r for r in rows if r['offset'] == 0), key=lambda r: r['aligned'])
        yawed = max((r for r in rows if abs(r['offset']) >= 6 and r['rot'] == 0),
                    key=lambda r: r['aligned'] - r['fixed'])
        ex = [front['file'], yawed['file']]
    print('examples:', ex)

    sets = []
    for spec in args.sets:
        label, path, report = spec.split('=')
        d = json.load(open(path))
        ratio = json.load(open(report))['pose']['dispersion']['offset_std_ratio']
        o = np.array([v['mirror_offset'] for v in d.values()
                      if v.get('mirror_offset') is not None])
        sets.append((label, o, ratio))
        print('%s: n %d, std %.3f, ratio %.2f' % (label, len(o), o.std(), o.std() / off_real.std()))

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.9)
    axis_panel(fig.add_subplot(gs[0, 0:3]), os.path.join(args.real, ex[0]), '(a)')
    axis_panel(fig.add_subplot(gs[0, 3:6]), os.path.join(args.real, ex[1]), '(b)')

    bins = np.arange(-8.5, 9.5, 1.0)
    centres = np.arange(-8, 9)
    h_real = np.histogram(off_real, bins=bins)[0] / len(off_real)
    for k, (label, o, ratio) in enumerate(sets):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        h = np.histogram(o, bins=bins)[0] / len(o)
        ax.bar(centres, h_real, width=0.9, color=GREY, linewidth=0)
        ax.step(np.r_[centres - 0.5, centres[-1] + 0.5], np.r_[h, h[-1]], where='post',
                color=BLUE, lw=1.4)
        ax.set_title('(%s) %s: ratio %.2f' % ('cde'[k], label, ratio),
                     fontsize=8.5, loc='left')
        ax.set_xlabel('axis offset (px)', fontsize=8)
        if k == 0:
            ax.set_ylabel('fraction of images', fontsize=8)
            ax.text(-8.4, 0.58, 'real', color='0.4', fontsize=8, va='top')
            ax.text(-8.4, 0.50, 'generated', color=BLUE, fontsize=8, va='top')
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(0, max(0.62, h.max() * 1.05))
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    os.makedirs(args.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(args.out, 'symmetry_axis.' + ext)
        fig.savefig(p, dpi=250 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
