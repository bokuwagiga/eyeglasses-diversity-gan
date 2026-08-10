"""Split edge_sym into frame-front vs temple regions.

edge_sym (metrics/diagnose_artifacts.py) is a mirror IoU over the whole rim
boundary band. sharp1 flags 3.82% vs real 2.01%, while appearance_sym is
BETTER than real - i.e. the asymmetry is geometric, not tonal. The working
hypothesis is that it lives in the temples: they sit at opposite extremes of
a 512 px image, so the high-resolution conv layers that decide their fine
boundary shape have no causal path between them, and symmetry can only be
inherited from the 4x8 / 8x16 layers.

This script recomputes the same boundary-band mirror IoU separately over
the central band of the bbox (frame front) and the two outer bands
(temples). Both region masks are mirror-symmetric about the bbox centre, so
the restriction does not itself bias the IoU.

If the generated-vs-real gap is concentrated in the outer band, the
long-range-coordination diagnosis holds and the fix belongs in the
architecture. If the gap is flat across regions, it does not.

Second check: TEXTURE symmetry, which is a different question. Frame SHAPE
should be mirror-symmetric, but frame TEXTURE should not - tortoiseshell /
patterned acetate is cut from a sheet, so the left and right rims carry
different pattern. diagnose_artifacts.appearance_sym is one-sided (it only
flags texture that is too ASYMMETRIC), so a generator that mirrors patterns
it should not mirror scores BETTER than real and the suite cannot see it.
Here appearance_sym is reported two-sided, and split into patterned vs
plain frames by intra-frame Lab variance, since only patterned frames can
exhibit the defect.
"""
import argparse
import json
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics.diversity_metrics import extract_silhouette  # noqa: E402
from metrics.evaluate_diversity import load_rgb  # noqa: E402

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def region_edge_sym(sil_u8, center_frac):
    """Mirror IoU of the rim boundary band, split into centre / outer bands.

    center_frac is the fraction of the bbox width treated as frame front.
    Returns (edge_sym_all, edge_sym_center, edge_sym_outer) or None.
    """
    ys, xs = np.nonzero(sil_u8)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    crop = sil_u8[:, x0:x1 + 1]
    w = crop.shape[1]
    if w < 8:
        return None

    bound = cv2.morphologyEx(crop, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    band = cv2.dilate(bound, np.ones((5, 5), np.uint8)).astype(bool)
    band_mir = band[:, ::-1]

    inter = np.logical_and(band, band_mir)
    union = np.logical_or(band, band_mir)

    # Mirror-symmetric column masks about the bbox centre.
    half = (1.0 - center_frac) / 2.0
    lo = int(round(half * w))
    hi = w - lo
    cmask = np.zeros(w, dtype=bool)
    cmask[lo:hi] = True
    omask = ~cmask

    def iou(colmask):
        i = inter[:, colmask].sum()
        u = union[:, colmask].sum()
        return float(i / u) if u > 0 else np.nan

    return (float(inter.sum() / max(union.sum(), 1)), iou(cmask), iou(omask))


def texture_symmetry(img_rgb, sil_u8):
    """Mean CIELAB distance between frame pixels and their mirror, plus a
    'patternedness' score.

    Same construction as diagnose_artifacts.appearance_sym, but returned raw
    so it can be read two-sided. patternedness is the mean per-channel std of
    Lab inside the frame: plain black/white acetate is near 0, tortoiseshell
    and other multi-tone patterns are high.
    """
    ys, xs = np.nonzero(sil_u8)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    crop = sil_u8[:, x0:x1 + 1].astype(bool)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_crop = lab[:, x0:x1 + 1]

    common = np.logical_and(crop, crop[:, ::-1])
    if common.sum() < 50:
        return None
    d = lab_crop - lab_crop[:, ::-1]
    app_sym = float(np.sqrt((d * d).sum(axis=2))[common].mean())

    vals = lab_crop[crop]
    patternedness = float(vals.std(axis=0).mean())
    return app_sym, patternedness


def scan(folder, limit, center_frac):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    if limit:
        names = names[:limit]
    rows = []
    for i, n in enumerate(names):
        try:
            img = load_rgb(os.path.join(folder, n))
            sil = extract_silhouette(img).astype(np.uint8)
            r = region_edge_sym(sil, center_frac)
            t = texture_symmetry(img, sil)
        except Exception:
            r, t = None, None
        if r is not None and t is not None:
            rows.append((r[0], r[1], r[2], t[0], t[1]))
        if (i + 1) % 500 == 0:
            print('  %d/%d' % (i + 1, len(names)), flush=True)
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--generated', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-images', type=int, default=3000)
    ap.add_argument('--center-frac', type=float, default=0.5,
                    help='fraction of bbox width treated as frame front')
    args = ap.parse_args()

    print('scanning real ...', flush=True)
    R = scan(args.real, args.max_images, args.center_frac)
    print('scanning generated ...', flush=True)
    G = scan(args.generated, args.max_images, args.center_frac)

    labels = ['all', 'center(front)', 'outer(temples)']
    lines = []
    lines.append('=' * 66)
    lines.append('edge_sym by region  (higher = more mirror-symmetric)')
    lines.append('  real %d imgs, gen %d imgs, center_frac %.2f'
                 % (len(R), len(G), args.center_frac))
    lines.append('')
    lines.append('  region            real p2   real p50 |  gen p2    gen p50 |'
                 '  flag%% vs real p2')
    for j, lab in enumerate(labels):
        r = R[:, j][np.isfinite(R[:, j])]
        g = G[:, j][np.isfinite(G[:, j])]
        thr = np.percentile(r, 2)
        flag = float((g < thr).mean() * 100)
        lines.append('  %-16s %7.4f  %7.4f | %7.4f  %7.4f |  %5.2f%%'
                     % (lab, thr, np.median(r), np.percentile(g, 2),
                        np.median(g), flag))
    lines.append('')
    lines.append('  Real flag%% is 2.00%% by construction. Excess in the outer')
    lines.append('  band but not the centre supports the long-range')
    lines.append('  coordination hypothesis.')
    lines.append('')

    # --- texture symmetry, two-sided ---
    ra, rp = R[:, 3], R[:, 4]
    ga, gp = G[:, 3], G[:, 4]
    pthr = np.percentile(rp, 70)  # 'patterned' = top 30% of real by Lab std
    lines.append('-' * 66)
    lines.append('TEXTURE symmetry (appearance_sym, LOW = more mirror-alike)')
    lines.append('  Frame SHAPE should mirror; frame TEXTURE should NOT.')
    lines.append('  patterned = Lab std above real p70 = %.2f' % pthr)
    lines.append('')
    lines.append('  subset            n_real  n_gen | real p10/p50/p90 |'
                 '  gen p10/p50/p90')
    for lab_, rm, gm in (('all', np.ones_like(rp, bool), np.ones_like(gp, bool)),
                         ('patterned', rp >= pthr, gp >= pthr),
                         ('plain', rp < pthr, gp < pthr)):
        r_, g_ = ra[rm], ga[gm]
        if len(r_) < 10 or len(g_) < 10:
            continue
        lines.append('  %-16s %6d %6d | %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f'
                     % (lab_, len(r_), len(g_),
                        *np.percentile(r_, [10, 50, 90]),
                        *np.percentile(g_, [10, 50, 90])))
    pm_r, pm_g = ra[rp >= pthr], ga[gp >= pthr]
    if len(pm_r) >= 10 and len(pm_g) >= 10:
        over = float((pm_g < np.percentile(pm_r, 10)).mean() * 100)
        lines.append('')
        lines.append('  OVER-MIRRORING on patterned frames: %.2f%% of generated'
                     % over)
        lines.append('  fall below the real p10 (expected 10%% if matched).')
        lines.append('  Much above 10%% = the model mirrors patterns that real')
        lines.append('  cut-from-sheet acetate does not. The p2-p98 suite is')
        lines.append('  one-sided here and cannot detect this.')
    lines.append('=' * 66)
    txt = '\n'.join(lines)
    print(txt)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(txt + '\n')
        with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
            json.dump({'labels': labels, 'real': R.tolist(),
                       'generated': G.tolist()}, f)


if __name__ == '__main__':
    main()
