"""Figure of novel realistic images next to their nearest catalogue image.

For each model the examples are not hand-picked: the novel_examples/ folder
of metrics/novelty_rate.py is ordered by novelty, and the first --count
images with zero artifact flags are taken.

  python scripts/make_novelty_figure.py \
      --sets "Plain=results/sharp1/eval1600/novelty=results/sharp1/eval1600/artifacts_p2p98/scores_gen.json" \
             "Mirror axis=results/mirror2/eval/novelty=results/mirror2/eval/artifacts_p2p98/scores_gen.json" \
      --out article/figures
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def load(path, w=512, h=256):
    im = Image.open(path).convert('RGB')
    iw, ih = im.size
    if abs(iw / ih - w / h) > 0.01:
        ch = int(iw * h / w)
        top = (ih - ch) // 2
        im = im.crop((0, top, iw, top + ch))
    return im.resize((w, h), Image.BICUBIC)


def clean_examples(novelty_dir, scores_path, count):
    scores = json.load(open(scores_path))
    out = []
    for g in sorted(glob.glob(os.path.join(novelty_dir, 'novel_examples', '*_gen_*.png'))):
        base = os.path.basename(g)
        name = re.sub(r'^\d+_gen_', '', base)
        if scores.get(name, {}).get('n_flags') != 0:
            continue
        real = glob.glob(os.path.join(novelty_dir, 'novel_examples', base.split('_')[0] + '_nnreal_*'))[0]
        out.append((g, real))
        if len(out) == count:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', required=True,
                    help='label=novelty_dir=scores_gen.json')
    ap.add_argument('--count', type=int, default=3)
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    sets = []
    for spec in args.sets:
        label, nd, sc = spec.split('=')
        sets.append((label, clean_examples(nd, sc, args.count)))

    ncol = 2 * len(sets)
    fig, axes = plt.subplots(args.count, ncol, figsize=(7.2, 0.95 * args.count + 0.3))
    for s, (label, pairs) in enumerate(sets):
        for r, (g, real) in enumerate(pairs):
            for c, (path, head) in enumerate(((g, label + ': generated'),
                                              (real, 'nearest catalogue image'))):
                ax = axes[r, 2 * s + c]
                ax.imshow(load(path))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color('0.8')
                if r == 0:
                    ax.set_title(head, fontsize=8)
            print(label, os.path.basename(g), os.path.basename(real))
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    os.makedirs(args.out, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(args.out, 'novel_examples.' + ext)
        fig.savefig(p, dpi=250, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
