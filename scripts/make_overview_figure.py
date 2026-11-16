"""Overview diagram of the study: data, training, generation, evaluation.

  python scripts/make_overview_figure.py --out article/figures
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

BLUE = '#1f5fa8'
ORANGE = '#d9730d'

BOXES = [
    # x, y, w, h, title, lines, colour
    (0.00, 0.50, 0.175, 0.46, 'Real catalogue',
     ['6,511 frontal frames', '256 x 512 pixels', 'light background'], '0.35'),
    (0.225, 0.50, 0.225, 0.46, 'Training',
     ['StyleGAN2-based generator', 'colour-aware sampling', 'ADA without colour ops',
      'optional mirror axis'], BLUE),
    (0.50, 0.50, 0.205, 0.46, 'Generation',
     ['layered truncation', r'($\psi$ 1.0 coarse, 1.2 fine)', 'one latent per image',
      '10,000 images'], BLUE),
    (0.755, 0.04, 0.245, 0.92, 'Evaluation vs. real',
     ['FID, KID', 'precision and recall', 'density and coverage', 'colour coverage',
      'silhouette geometry', 'pose-aligned symmetry', 'axis-offset dispersion',
      'realistic novelty', 'artifact checks'], '0.35'),
    (0.225, 0.04, 0.48, 0.30, 'Baseline',
     ['official StyleGAN2-ADA', 'same images, same evaluation'], ORANGE),
]
ARROWS = [
    ((0.175, 0.73), (0.225, 0.73), '0.35'),
    ((0.45, 0.73), (0.50, 0.73), BLUE),
    ((0.705, 0.73), (0.755, 0.73), BLUE),
    ((0.0875, 0.50), (0.0875, 0.19), '0.35'),
    ((0.0875, 0.19), (0.225, 0.19), '0.35'),
    ((0.705, 0.19), (0.755, 0.19), ORANGE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis('off')
    for x, y, w, h, title, lines, colour in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.015',
                                    fc='white', ec=colour, lw=1.2))
        ax.text(x + 0.012, y + h - 0.05, title, fontsize=7.6, fontweight='bold',
                color=colour if colour != '0.35' else 'black', va='top')
        for i, line in enumerate(lines):
            ax.text(x + 0.012, y + h - 0.15 - i * 0.082, line, fontsize=6.4, va='top', color='0.15')
    for (x0, y0), (x1, y1), colour in ARROWS:
        head = '-|>' if (x1, y1) != (0.0875, 0.19) else '-'
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=head, mutation_scale=9,
                                     lw=1.1, color=colour, shrinkA=0, shrinkB=0))
    os.makedirs(args.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(args.out, 'overview.' + ext)
        fig.savefig(p, dpi=250 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
