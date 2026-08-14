"""Separate SHAPE asymmetry from POSE misalignment in edge_sym.

metrics/diagnose_artifacts.edge_sym mirrors the rim boundary band about the
bbox vertical centre line. That axis is only the frame's true symmetry axis
if the photograph is perfectly fronto-parallel and untilted. Real catalogue
shots are not: a degree of tilt, or slight yaw that makes one temple extend
further, moves the bbox centre off the true axis and depresses edge_sym for
photographic reasons rather than shape reasons.

This matters because mirror1 (flip-concat in G) scores edge_sym p50 0.5946
against real 0.5535 - i.e. "more symmetric than real" - while direct visual
inspection says its frames are more symmetric than sharp1's but NOT more
symmetric than real. A GAN trained with x-flips can produce more canonically
posed frames than the photographs it learned from, which would raise the
fixed-axis score without the shape being more symmetric.

The test: recompute the same boundary-band mirror IoU while searching over
the mirror axis (horizontal shift and small rotation) and keeping the best.
That removes pose from the measurement and leaves shape.

  - If real gains much more from alignment than generated, the fixed-axis
    metric was pose-confounded and the "more symmetric than real" reading is
    an artifact.
  - If both gain about the same, the fixed-axis gap is real shape asymmetry
    and the visual impression is what needs revisiting.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics.diversity_metrics import extract_silhouette  # noqa: E402
from metrics.evaluate_diversity import load_rgb  # noqa: E402

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def shift_h(a, s):
    """Shift a bool array horizontally by s pixels, zero fill."""
    if s == 0:
        return a
    out = np.zeros_like(a)
    if s > 0:
        out[:, s:] = a[:, :-s]
    else:
        out[:, :s] = a[:, -s:]
    return out


def band_of(sil_u8):
    """Rim boundary band, same construction as diagnose_artifacts.edge_sym."""
    bound = cv2.morphologyEx(sil_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.dilate(bound, np.ones((5, 5), np.uint8))


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else np.nan


def axis_search(sil_u8, max_shift, max_deg, deg_step):
    """Mirror IoU at the bbox centre axis, and maximized over axis pose.

    Returns (baseline, aligned, best_shift_px, best_deg).
    """
    ys, xs = np.nonzero(sil_u8)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if x1 - x0 < 8:
        return None

    # Pad so rotation and shift do not clip the frame.
    pad = max_shift + 8
    sub = sil_u8[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
    if sub.size == 0:
        return None

    h, w = sub.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    best = (-1.0, 0, 0.0)
    baseline = None
    degs = [0.0] if max_deg <= 0 else list(
        np.arange(-max_deg, max_deg + 1e-9, deg_step))

    for deg in degs:
        if deg == 0.0:
            rot = sub
        else:
            M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
            rot = cv2.warpAffine(sub, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderValue=0)
        band = band_of(rot).astype(bool)
        flipped = band[:, ::-1]
        for s in range(-max_shift, max_shift + 1):
            # Mirroring about an axis offset by s/2 from centre == flip then
            # shift by s.
            v = iou(band, shift_h(flipped, s))
            if deg == 0.0 and s == 0:
                baseline = v
            if v > best[0]:
                best = (v, s, float(deg))

    if baseline is None:
        return None
    return baseline, best[0], best[1], best[2]


def scan(folder, limit, max_shift, max_deg, deg_step, tag):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    if limit:
        names = names[:limit]
    rows = []
    for i, n in enumerate(names):
        try:
            sil = extract_silhouette(load_rgb(os.path.join(folder, n)))
            r = axis_search(sil.astype(np.uint8), max_shift, max_deg, deg_step)
        except Exception:
            r = None
        if r is not None:
            rows.append(r)
        if (i + 1) % 250 == 0:
            print('  %s %d/%d' % (tag, i + 1, len(names)), flush=True)
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--generated', required=True, nargs='+',
                    help='one or more generated image folders')
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-images', type=int, default=1500)
    ap.add_argument('--max-shift', type=int, default=8,
                    help='half-width of the mirror-axis shift search, px')
    ap.add_argument('--max-deg', type=float, default=3.0)
    ap.add_argument('--deg-step', type=float, default=1.0)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(p.rstrip('/\\'))
                             for p in args.generated]
    sets = [('real', args.real)] + list(zip(labels, args.generated))

    res = {}
    for tag, folder in sets:
        print('scanning %s ...' % tag, flush=True)
        res[tag] = scan(folder, args.max_images, args.max_shift,
                        args.max_deg, args.deg_step, tag)

    lines = []
    lines.append('=' * 72)
    lines.append('edge_sym: fixed bbox axis vs pose-aligned axis')
    lines.append('  shift search +-%d px, rotation +-%.1f deg step %.1f'
                 % (args.max_shift, args.max_deg, args.deg_step))
    lines.append('')
    lines.append('  set          n   fixed p50   aligned p50    gain |'
                 '  med |shift|  med |deg|')
    for tag, _ in sets:
        A = res[tag]
        if len(A) == 0:
            continue
        fx, al = np.median(A[:, 0]), np.median(A[:, 1])
        lines.append('  %-10s %4d      %.4f        %.4f  %+.4f |'
                     '      %4.1f      %4.1f'
                     % (tag, len(A), fx, al, al - fx,
                        np.median(np.abs(A[:, 2])), np.median(np.abs(A[:, 3]))))

    rA = res['real']
    if len(rA):
        rfx, ral = np.median(rA[:, 0]), np.median(rA[:, 1])
        lines.append('')
        lines.append('  gap vs real (positive = generated scores higher):')
        lines.append('  %-10s %12s %14s' % ('set', 'fixed axis', 'aligned axis'))
        for tag, _ in sets[1:]:
            A = res[tag]
            if not len(A):
                continue
            lines.append('  %-10s %+12.4f %+14.4f'
                         % (tag, np.median(A[:, 0]) - rfx,
                            np.median(A[:, 1]) - ral))
        lines.append('')
        lines.append('  If a generated set leads on the fixed axis but not on')
        lines.append('  the aligned axis, its edge_sym advantage was POSE')
        lines.append('  (canonical framing), not shape symmetry.')
    lines.append('=' * 72)

    txt = '\n'.join(lines)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(txt + '\n')
        with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in res.items()}, f)


if __name__ == '__main__':
    main()
