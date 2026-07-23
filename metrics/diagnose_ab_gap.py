"""
Diagnose the ab_coverage gap: which frame colours does the model miss?

Recomputes the exact same statistic behind the ab_coverage metric (2D
histogram of per-image MEAN frame colour on the a*b* plane, 16x16 bins over
[-40, 40]) for the real and generated sets, then:

  1. Plots real vs generated occupancy heatmaps and a "missing bins" map.
  2. Lists every bin that real images occupy but generated ones do not
     (plus bins where the generated density is far below real).
  3. Copies example real images from each missing bin into
     <out>/missing_examples/ and builds a contact sheet, so the missing
     colour modes can be inspected by eye.

No GPU needed. Run wherever the images live (training PC):

  python metrics/diagnose_ab_gap.py \
      --real data/source/images \
      --generated results/ppl4_best/generated/images \
      --out results/ppl4_best/ab_gap
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_metrics as dm
from evaluate_diversity import list_images, load_rgb

AB_LO, AB_HI, N_BINS = -40.0, 40.0, 16


def mean_frame_colors(paths, desc):
    """Per-image mean frame LAB colour. Returns (paths_kept, means (N,3))."""
    kept, means = [], []
    for p in tqdm(paths, desc=desc):
        img = load_rgb(p)
        sil = dm.extract_silhouette(img)
        if sil is None:
            continue
        mean_lab, _ = dm.frame_lab_stats(img, sil)
        kept.append(p)
        means.append(mean_lab)
    return kept, np.asarray(means)


def ab_hist(means):
    hist, _, _ = np.histogram2d(means[:, 1], means[:, 2], bins=N_BINS,
                                range=[[AB_LO, AB_HI], [AB_LO, AB_HI]])
    return hist  # (a_bins, b_bins), counts


def bin_center(i, j):
    step = (AB_HI - AB_LO) / N_BINS
    return AB_LO + (i + 0.5) * step, AB_LO + (j + 0.5) * step


def lab_to_rgb01(l, a, b):
    """Approximate sRGB swatch colour for a LAB value (via OpenCV)."""
    import cv2
    lab = np.array([[[l * 255.0 / 100.0, a + 128.0, b + 128.0]]], dtype=np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]
    return rgb.astype(np.float64) / 255.0


def make_heatmaps(hist_real, hist_gen, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    missing = (hist_real > 0) & (hist_gen == 0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [AB_LO, AB_HI, AB_LO, AB_HI]
    for ax, h, title in ((axes[0], hist_real, 'Real (log count)'),
                         (axes[1], hist_gen, 'Generated (log count)')):
        im = ax.imshow(np.log1p(h).T, origin='lower', extent=extent,
                       cmap='viridis')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    im = axes[2].imshow(missing.T.astype(float), origin='lower', extent=extent,
                        cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title(f'Missing bins (real yes, gen no): {int(missing.sum())}')
    for ax in axes:
        ax.set_xlabel('a* (green - red)')
        ax.set_ylabel('b* (blue - yellow)')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_contact_sheet(bins_info, out_path, thumb_w=192, per_bin=6):
    """One row per missing bin: colour swatch + example real images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = len(bins_info)
    if n_rows == 0:
        return
    n_cols = per_bin + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 1.3 * n_rows), squeeze=False)
    for r, info in enumerate(bins_info):
        a, b = info['a_center'], info['b_center']
        swatch = np.ones((32, 64, 3)) * lab_to_rgb01(info['mean_L'], a, b)
        axes[r][0].imshow(swatch)
        axes[r][0].set_title(f"a={a:.0f} b={b:.0f} n={info['n_real']}",
                             fontsize=8)
        for c in range(1, n_cols):
            ax = axes[r][c]
            k = c - 1
            if k < len(info['examples']):
                ax.imshow(load_rgb(info['examples'][k]))
        for ax in axes[r]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Diagnose which a*b* colour bins the generated set misses.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--real', required=True, help='Real images dir')
    ap.add_argument('--generated', required=True, help='Generated images dir')
    ap.add_argument('--out', required=True, help='Output dir')
    ap.add_argument('--max-images', type=int, default=0,
                    help='Cap per set (0 = use all images)')
    ap.add_argument('--examples-per-bin', type=int, default=6,
                    help='Real example images to copy per missing bin')
    ap.add_argument('--under-factor', type=float, default=0.2,
                    help='Report bins where gen fraction < factor * real fraction')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    real_paths = list_images(args.real)
    gen_paths = list_images(args.generated)
    if args.max_images > 0:
        if len(real_paths) > args.max_images:
            idx = rng.choice(len(real_paths), args.max_images, replace=False)
            real_paths = [real_paths[i] for i in sorted(idx)]
        if len(gen_paths) > args.max_images:
            idx = rng.choice(len(gen_paths), args.max_images, replace=False)
            gen_paths = [gen_paths[i] for i in sorted(idx)]

    real_kept, real_means = mean_frame_colors(real_paths, 'Real frame colours')
    gen_kept, gen_means = mean_frame_colors(gen_paths, 'Gen frame colours')

    hist_real = ab_hist(real_means)
    hist_gen = ab_hist(gen_means)
    frac_real = hist_real / hist_real.sum()
    frac_gen = hist_gen / hist_gen.sum()

    occ_real = hist_real > 0
    occ_gen = hist_gen > 0
    missing = occ_real & ~occ_gen
    under = occ_real & occ_gen & (frac_gen < args.under_factor * frac_real)

    # Assign each real image to its bin for example lookup
    step = (AB_HI - AB_LO) / N_BINS
    bi = np.clip(((real_means[:, 1] - AB_LO) / step).astype(int), 0, N_BINS - 1)
    bj = np.clip(((real_means[:, 2] - AB_LO) / step).astype(int), 0, N_BINS - 1)

    def bin_report(mask):
        infos = []
        for i, j in zip(*np.nonzero(mask)):
            sel = (bi == i) & (bj == j)
            idxs = np.nonzero(sel)[0]
            pick = rng.permutation(idxs)[:args.examples_per_bin]
            a_c, b_c = bin_center(i, j)
            infos.append({
                'a_center': a_c, 'b_center': b_c,
                'chroma': float(np.hypot(a_c, b_c)),
                'hue_deg': float(np.degrees(np.arctan2(b_c, a_c)) % 360),
                'n_real': int(hist_real[i, j]),
                'n_gen': int(hist_gen[i, j]),
                'mean_L': float(real_means[idxs, 0].mean()),
                'examples': [real_kept[k] for k in pick],
            })
        infos.sort(key=lambda d: -d['n_real'])
        return infos

    missing_info = bin_report(missing)
    under_info = bin_report(under)

    # Copy example real images of missing bins
    ex_dir = out / 'missing_examples'
    if ex_dir.exists():
        shutil.rmtree(ex_dir)
    for info in missing_info:
        d = ex_dir / f"a{info['a_center']:+.0f}_b{info['b_center']:+.0f}"
        d.mkdir(parents=True, exist_ok=True)
        for p in info['examples']:
            shutil.copy(p, d / Path(p).name)

    make_heatmaps(hist_real, hist_gen, out / 'ab_heatmaps.png')
    make_contact_sheet(missing_info, out / 'missing_contact_sheet.png',
                       per_bin=args.examples_per_bin)

    n_missing_imgs = sum(d['n_real'] for d in missing_info)
    summary = {
        'n_real': len(real_kept), 'n_gen': len(gen_kept),
        'ab_coverage_real': float(occ_real.mean()),
        'ab_coverage_gen': float(occ_gen.mean()),
        'bins_real': int(occ_real.sum()), 'bins_gen': int(occ_gen.sum()),
        'bins_missing': int(missing.sum()),
        'bins_underrepresented': int(under.sum()),
        'real_images_in_missing_bins': n_missing_imgs,
        'real_frac_in_missing_bins': n_missing_imgs / max(len(real_kept), 1),
        'missing_bins': missing_info,
        'underrepresented_bins': under_info,
    }
    with open(out / 'ab_gap_report.json', 'w') as f:
        json.dump(summary, f, indent=2)

    lines = []
    add = lines.append
    add('=' * 70)
    add('ab_coverage gap diagnosis')
    add(f'  real {len(real_kept)} imgs, generated {len(gen_kept)} imgs')
    add(f'  coverage: real {occ_real.mean():.4f} ({int(occ_real.sum())} bins), '
        f'gen {occ_gen.mean():.4f} ({int(occ_gen.sum())} bins)')
    add(f'  missing bins: {int(missing.sum())}  '
        f'underrepresented (<{args.under_factor}x real share): {int(under.sum())}')
    add(f'  real images living in missing bins: {n_missing_imgs} '
        f'({100.0 * n_missing_imgs / max(len(real_kept), 1):.2f}% of real set)')
    add('')
    add('--- Missing bins (real occupied, generated empty) ---')
    add(f'  {"a*":>6}{"b*":>6}{"chroma":>8}{"hue":>6}{"L":>6}{"n_real":>8}  example')
    for d in missing_info:
        ex = Path(d['examples'][0]).name if d['examples'] else '-'
        add(f"  {d['a_center']:>6.0f}{d['b_center']:>6.0f}{d['chroma']:>8.1f}"
            f"{d['hue_deg']:>6.0f}{d['mean_L']:>6.1f}{d['n_real']:>8}  {ex}")
    add('')
    add('--- Underrepresented bins (generated share far below real) ---')
    add(f'  {"a*":>6}{"b*":>6}{"chroma":>8}{"n_real":>8}{"n_gen":>8}')
    for d in under_info:
        add(f"  {d['a_center']:>6.0f}{d['b_center']:>6.0f}{d['chroma']:>8.1f}"
            f"{d['n_real']:>8}{d['n_gen']:>8}")
    add('=' * 70)
    with open(out / 'ab_gap_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nOutputs written to {out}')


if __name__ == '__main__':
    main()
