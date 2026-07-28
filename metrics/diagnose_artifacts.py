"""
Structural defect diagnosis for generated eyeglass frames.

FID/precision live in Inception feature space and can score visibly broken
frames as "close to real". This script checks eyeglass-specific structural
priors directly on the frame silhouette, calibrated against the real set:

  n_fragments   connected components of the silhouette after speckle removal
                (a frame should be one piece; extras = floating artifacts)
  n_holes       enclosed lens openings (should be 2; rimless/semi-rimless
                reals may score 0-1, hence real calibration)
  symmetry_iou  IoU of the silhouette with its horizontal mirror about the
                bbox centre (frontal frames are near-symmetric)
  lens_mismatch relative area difference of the two largest lens openings
  speckle_frac  foreground area OUTSIDE the largest component (smudges)
  sharpness     variance of Laplacian inside the frame bbox (blur check)
  rim_breaks    lens openings that only appear after morphologically closing
                small gaps = broken rims (the opening leaks into background)
  rim_contrast  median image-gradient strength along the silhouette boundary
                (low = washed-out / ghost frames whose rims fade out)
  contour_wobble mean deviation of the silhouette contours from their
                low-pass-smoothed version (high = wonky circles / wavy
                lines instead of smooth manufactured curves)
  edge_sym      mirror IoU of the rim BOUNDARY band (not the filled shape,
                which is too forgiving): sensitive to left/right lens shape
                and rim thickness differences
  appearance_sym mean CIELAB difference between frame pixels and their
                mirrored counterparts (colour/shading asymmetry)

Thresholds are the real set's [lo_pct, hi_pct] percentiles per check
(default 0.5 / 99.5): a generated image is flagged when it falls outside
the range that 99% of real images occupy.

Outputs into --out:
  artifact_report.txt / .json   flag rates real vs generated, per check
  scores_gen.json               per-image scores + flags (for filtering)
  worst_<check>.png             contact sheet of the worst offenders
  worst_overall.png             images failing the most checks

Run wherever the images live (CPU only):
  python metrics/diagnose_artifacts.py \
      --real data/source/images \
      --generated results/rebalance2/generated/images \
      --out results/rebalance2/artifacts
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_metrics as dm
from evaluate_diversity import list_images, load_rgb

CHECKS = ['n_fragments', 'n_holes', 'symmetry_iou', 'lens_mismatch',
          'speckle_frac', 'sharpness', 'rim_breaks', 'rim_contrast',
          'contour_wobble', 'edge_sym', 'appearance_sym']

# Direction of badness per check: 'high' = larger is worse, 'low' = smaller
# is worse, 'both' = outside the real range either way is bad.
DIRECTION = {
    'n_fragments': 'high',
    'n_holes': 'both',
    'symmetry_iou': 'low',
    'lens_mismatch': 'high',
    'speckle_frac': 'high',
    'sharpness': 'low',
    'rim_breaks': 'high',    # lens holes that only exist after gap-closing
    'rim_contrast': 'low',   # weak boundary gradient = ghost/fading frame
    'contour_wobble': 'high',  # wavy/wonky rim outlines
    'edge_sym': 'low',         # boundary-band mirror IoU
    'appearance_sym': 'high',  # Lab distance to mirrored frame
}

MIN_BLOB_FRAC = 0.0002  # blobs smaller than this fraction of the image are noise


def artifact_scores(img_rgb):
    """Per-image structural scores. Returns dict or None if no silhouette."""
    sil = dm.extract_silhouette(img_rgb)
    if sil is None:
        return None
    sil_u8 = sil.astype(np.uint8)
    h, w = sil_u8.shape
    img_area = h * w

    # Connected components of the silhouette
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(sil_u8, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    significant = areas > MIN_BLOB_FRAC * img_area
    n_fragments = int(significant.sum())
    main_label = 1 + int(np.argmax(areas)) if len(areas) else 0
    main_area = float(areas.max()) if len(areas) else 0.0
    speckle_frac = float((sil_u8.sum() - main_area) / max(sil_u8.sum(), 1))

    # Lens openings
    from scipy.ndimage import binary_fill_holes

    def count_holes(mask_u8):
        filled = binary_fill_holes(mask_u8).astype(np.uint8)
        holes = (filled - mask_u8).astype(np.uint8)
        _, _, hstats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        hareas = hstats[1:, cv2.CC_STAT_AREA]
        return np.sort(hareas[hareas > 0.0005 * img_area])[::-1], filled

    hsig, _ = count_holes(sil_u8)
    n_holes = int(len(hsig))

    # Rim breaks: a broken rim leaks its lens opening into the background, so
    # the hole is not counted. Morphologically closing small gaps (9x9,
    # bridges breaks up to ~8 px) makes the hole (re)appear. Extra holes
    # after closing = number of broken rims. Rare on real frames, and any
    # false bridging there is absorbed by the real-percentile calibration.
    closed = cv2.morphologyEx(sil_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    hsig_closed, _ = count_holes(closed)
    rim_breaks = max(0, int(len(hsig_closed)) - n_holes)
    if n_holes >= 2:
        lens_mismatch = float((hsig[0] - hsig[1]) / hsig[0])
    else:
        lens_mismatch = 1.0 if n_holes == 1 else 0.0

    # Symmetry: mirror the silhouette about the bbox vertical centre line
    ys, xs = np.nonzero(sil_u8)
    x0, x1 = xs.min(), xs.max()
    crop = sil_u8[:, x0:x1 + 1]
    mirrored = crop[:, ::-1]
    inter = np.logical_and(crop, mirrored).sum()
    union = np.logical_or(crop, mirrored).sum()
    symmetry_iou = float(inter / max(union, 1))

    # Sharpness inside the frame bbox
    y0, y1 = ys.min(), ys.max()
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_full[y0:y1 + 1, x0:x1 + 1]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Rim contrast: median image-gradient strength along the silhouette
    # boundary. Washed-out / ghost frames (pale rims fading into the
    # background) score low; crisp catalogue frames score high.
    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    boundary = cv2.morphologyEx(sil_u8, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)) > 0
    bvals = grad[boundary]
    rim_contrast = float(np.median(bvals)) if bvals.size else 0.0

    # Contour wobble: real rims are smooth manufactured curves. Low-pass each
    # significant contour (circular moving average over the point sequence)
    # and measure the mean point-wise deviation of the raw contour from its
    # smoothed version. Wonky circles / wavy lines deviate by pixels; smooth
    # rims only by quantization noise. Genuine sharp corners contribute a
    # small localized baseline that the real-percentile calibration absorbs.
    from scipy.ndimage import uniform_filter1d
    contours, _ = cv2.findContours(sil_u8, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_NONE)
    k = 15
    wobbles, weights = [], []
    for cnt in contours:
        if len(cnt) < 4 * k:
            continue  # too short to distinguish wobble from corners
        pts = cnt[:, 0, :].astype(np.float64)
        sx = uniform_filter1d(pts[:, 0], size=k, mode='wrap')
        sy = uniform_filter1d(pts[:, 1], size=k, mode='wrap')
        dev = np.sqrt((pts[:, 0] - sx) ** 2 + (pts[:, 1] - sy) ** 2)
        wobbles.append(float(dev.mean()))
        weights.append(float(len(cnt)))
    contour_wobble = (float(np.average(wobbles, weights=weights))
                      if wobbles else 0.0)

    # Edge symmetry: mirror IoU on the rim boundary band. The filled-shape
    # IoU (symmetry_iou) is too forgiving - large areas overlap even when
    # lens shapes / rim thickness differ left vs right. A ~7 px boundary
    # band makes those differences count.
    bound_u8 = cv2.morphologyEx(crop, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
    band = cv2.dilate(bound_u8, np.ones((5, 5), np.uint8))
    band_mir = band[:, ::-1]
    edge_sym = float(np.logical_and(band, band_mir).sum()
                     / max(np.logical_or(band, band_mir).sum(), 1))

    # Appearance symmetry: mean CIELAB distance between frame pixels and
    # their mirrored counterparts, over pixels where both the silhouette
    # and its mirror are frame. Catches colour/shading asymmetry.
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_crop = lab[:, x0:x1 + 1]
    common = np.logical_and(crop, crop[:, ::-1])
    if common.sum() > 0:
        d = lab_crop - lab_crop[:, ::-1]
        appearance_sym = float(np.sqrt((d * d).sum(axis=2))[common].mean())
    else:
        appearance_sym = 255.0

    return {'n_fragments': n_fragments, 'n_holes': n_holes,
            'symmetry_iou': symmetry_iou, 'lens_mismatch': lens_mismatch,
            'speckle_frac': speckle_frac, 'sharpness': sharpness,
            'rim_breaks': rim_breaks, 'rim_contrast': rim_contrast,
            'contour_wobble': contour_wobble, 'edge_sym': edge_sym,
            'appearance_sym': appearance_sym}


def score_set(paths, desc):
    scores, kept, failed = [], [], 0
    for p in tqdm(paths, desc=desc):
        s = artifact_scores(load_rgb(p))
        if s is None:
            failed += 1
            continue
        scores.append(s)
        kept.append(p)
    return kept, scores, failed


def calibrate(real_scores, lo_pct, hi_pct):
    """Per-check acceptance range from the real set's percentiles."""
    thresholds = {}
    for c in CHECKS:
        vals = np.array([s[c] for s in real_scores], dtype=np.float64)
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        d = DIRECTION[c]
        thresholds[c] = {
            'lo': float(lo) if d in ('low', 'both') else None,
            'hi': float(hi) if d in ('high', 'both') else None,
        }
    return thresholds


def flag(scores, thresholds):
    """Boolean flag matrix (N, len(CHECKS))."""
    out = np.zeros((len(scores), len(CHECKS)), dtype=bool)
    for i, s in enumerate(scores):
        for j, c in enumerate(CHECKS):
            t = thresholds[c]
            v = s[c]
            if t['lo'] is not None and v < t['lo']:
                out[i, j] = True
            if t['hi'] is not None and v > t['hi']:
                out[i, j] = True
    return out


def contact_sheet(paths, labels, out_path, n_cols=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(paths)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 1.5 * n_rows), squeeze=False)
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        ax.axis('off')
        if k < n:
            ax.imshow(load_rgb(paths[k]))
            ax.set_title(labels[k], fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Structural defect diagnosis, calibrated on the real set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--real', required=True)
    ap.add_argument('--generated', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-images', type=int, default=0,
                    help='Cap per set (0 = all)')
    ap.add_argument('--lo-pct', type=float, default=0.5)
    ap.add_argument('--hi-pct', type=float, default=99.5)
    ap.add_argument('--worst-n', type=int, default=24,
                    help='Images per worst-offender contact sheet')
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

    real_kept, real_scores, real_failed = score_set(real_paths, 'Real')
    gen_kept, gen_scores, gen_failed = score_set(gen_paths, 'Generated')

    thresholds = calibrate(real_scores, args.lo_pct, args.hi_pct)
    flags_real = flag(real_scores, thresholds)
    flags_gen = flag(gen_scores, thresholds)

    n_flags_gen = flags_gen.sum(axis=1)
    clean_gen = float((n_flags_gen == 0).mean())
    clean_real = float((flags_real.sum(axis=1) == 0).mean())

    # Contact sheets: worst offenders per check
    for j, c in enumerate(CHECKS):
        d = DIRECTION[c]
        vals = np.array([s[c] for s in gen_scores], dtype=np.float64)
        flagged = np.nonzero(flags_gen[:, j])[0]
        if len(flagged) == 0:
            continue
        if d == 'high':
            order = flagged[np.argsort(-vals[flagged])]
        elif d == 'low':
            order = flagged[np.argsort(vals[flagged])]
        else:  # both: furthest outside the range
            t = thresholds[c]
            dist = np.maximum(t['lo'] - vals[flagged], vals[flagged] - t['hi'])
            order = flagged[np.argsort(-dist)]
        pick = order[:args.worst_n]
        contact_sheet([gen_kept[i] for i in pick],
                      [f'{c}={vals[i]:.3g}' for i in pick],
                      out / f'worst_{c}.png')

    # Overall worst: most simultaneous flags
    order = np.argsort(-n_flags_gen)
    pick = [i for i in order[:args.worst_n] if n_flags_gen[i] > 0]
    contact_sheet([gen_kept[i] for i in pick],
                  [f'{int(n_flags_gen[i])} flags' for i in pick],
                  out / 'worst_overall.png')

    # Per-image scores for downstream filtering
    with open(out / 'scores_gen.json', 'w') as f:
        json.dump({Path(p).name: {**s, 'n_flags': int(nf)}
                   for p, s, nf in zip(gen_kept, gen_scores, n_flags_gen)},
                  f, indent=2)

    report = {
        'meta': {'real_dir': args.real, 'generated_dir': args.generated,
                 'n_real': len(real_kept), 'n_gen': len(gen_kept),
                 'real_failed_sil': real_failed, 'gen_failed_sil': gen_failed,
                 'lo_pct': args.lo_pct, 'hi_pct': args.hi_pct},
        'thresholds': thresholds,
        'flag_rate_real': {c: float(flags_real[:, j].mean())
                           for j, c in enumerate(CHECKS)},
        'flag_rate_gen': {c: float(flags_gen[:, j].mean())
                          for j, c in enumerate(CHECKS)},
        'clean_fraction_real': clean_real,
        'clean_fraction_gen': clean_gen,
        'gen_flag_count_hist': {str(k): int((n_flags_gen == k).sum())
                                for k in range(int(n_flags_gen.max()) + 1)},
    }
    with open(out / 'artifact_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    lines = []
    add = lines.append
    add('=' * 70)
    add('Structural artifact report (thresholds = real '
        f'p{args.lo_pct}-p{args.hi_pct})')
    add(f'  real {len(real_kept)} imgs ({real_failed} sil-failed), '
        f'gen {len(gen_kept)} imgs ({gen_failed} sil-failed)')
    add('')
    add(f'  {"check":<16}{"real flag%":>12}{"gen flag%":>12}{"range":>28}')
    for j, c in enumerate(CHECKS):
        t = thresholds[c]
        lo = f"{t['lo']:.3g}" if t['lo'] is not None else '-'
        hi = f"{t['hi']:.3g}" if t['hi'] is not None else '-'
        add(f'  {c:<16}{100 * flags_real[:, j].mean():>11.2f}%'
            f'{100 * flags_gen[:, j].mean():>11.2f}%'
            f'{"[" + lo + ", " + hi + "]":>28}')
    add('')
    add(f'  clean (0 flags): real {100 * clean_real:.2f}%   '
        f'gen {100 * clean_gen:.2f}%')
    add(f'  gen flag-count histogram: '
        + '  '.join(f'{k}:{v}' for k, v in report['gen_flag_count_hist'].items()))
    add('=' * 70)
    with open(out / 'artifact_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nOutputs written to {out}')


if __name__ == '__main__':
    main()
