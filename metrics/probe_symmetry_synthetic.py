"""Controlled-input validation of the mirror-symmetry metrics.

Twice in this project an architecture change was built on an untested
reading of a symmetry metric. The rule adopted afterwards: any metric gets
a controlled-input test of what it responds to before conclusions are
built on it. This probe is that test for mirror_axis_symmetry, run on
synthetic eyeglass silhouettes whose ground truth is known exactly.

Cases and what each one establishes:

  perfect          both scores at ceiling; recovered offset 0
  tilt 2 deg       pose alone: the fixed-axis score must drop hard while
                   the aligned score recovers (the search covers +-3 deg)
  temple -25%      genuine shape asymmetry: BOTH scores must drop, and the
                   aligned search must NOT be able to hide it
  tilt + temple    the two effects together
  shift +12 px     pure translation: scores unchanged and offset exactly 0,
  shift -20 px     because the search re-centres on the bounding box -
                   the offset is translation-invariant by construction
  temple -25%      the offset must move away from 0: differential temple
    (offset check)   length is what the offset responds to, i.e. yaw

The probe also times the aligned search per image. Numbers in the paper
come from this probe's JSON, not from any earlier ad-hoc run.

  python metrics/probe_symmetry_synthetic.py --out results/_probes/symmetry_synthetic.json
  python metrics/probe_symmetry_synthetic.py --real <real image dir> --time-n 50
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.diversity_metrics import mirror_axis_symmetry  # noqa: E402

H, W = 256, 512


def make_silhouette(temple_right_scale=1.0, tilt_deg=0.0, shift_px=0):
    """Synthetic frontal eyeglass silhouette: two elliptical rims, a bridge,
    and two temple bars. Perfectly mirror-symmetric unless one temple is
    scaled (yaw / genuine shape asymmetry) or the canvas is rotated (pose)."""
    m = np.zeros((H, W), np.uint8)
    cx, cy = W // 2, H // 2
    # Margins: mirror_axis_symmetry pads the bbox by max_shift + 8 = 16 px
    # and clamps at the canvas edge, so a silhouette closer than 16 px to
    # the edge gets an asymmetric crop and a biased fixed axis (discovered
    # by this probe's first run). Real catalogue frames are centred with
    # far larger margins; the synthetic shapes keep >= 30 px after every
    # transform below so the probe tests the metric, not the clamp.
    lens_w, lens_h, rim = 65, 55, 10
    half_bridge, temple_len, temple_th = 25, 50, 8

    for lens_cx in (cx - half_bridge - lens_w, cx + half_bridge + lens_w):
        cv2.ellipse(m, (lens_cx, cy), (lens_w, lens_h), 0, 0, 360, 1,
                    thickness=rim)
    cv2.rectangle(m, (cx - half_bridge, cy - lens_h // 2 - 3),
                  (cx + half_bridge, cy - lens_h // 2 + 5), 1, -1)

    y_t = cy - lens_h + 14
    left_edge = cx - half_bridge - 2 * lens_w
    right_edge = cx + half_bridge + 2 * lens_w
    cv2.rectangle(m, (left_edge - temple_len, y_t), (left_edge, y_t + temple_th),
                  1, -1)
    right_len = int(round(temple_len * temple_right_scale))
    cv2.rectangle(m, (right_edge, y_t), (right_edge + right_len, y_t + temple_th),
                  1, -1)

    if tilt_deg:
        rot = cv2.getRotationMatrix2D((cx, cy), tilt_deg, 1.0)
        m = cv2.warpAffine(m, rot, (W, H), flags=cv2.INTER_NEAREST)
    if shift_px:
        m = np.roll(m, shift_px, axis=1)
        if shift_px > 0:
            m[:, :shift_px] = 0
        else:
            m[:, shift_px:] = 0
    return m


CASES = [
    ('perfect',            dict()),
    ('tilt_2deg',          dict(tilt_deg=2.0)),
    ('temple_10pct_short', dict(temple_right_scale=0.90)),
    ('temple_25pct_short', dict(temple_right_scale=0.75)),
    ('tilt_plus_temple',   dict(tilt_deg=2.0, temple_right_scale=0.75)),
    ('shift_plus12px',     dict(shift_px=12)),
    ('shift_minus20px',    dict(shift_px=-20)),
    ('temple_plus_shift',  dict(temple_right_scale=0.75, shift_px=12)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/_probes/symmetry_synthetic.json')
    ap.add_argument('--real', default=None,
                    help='optional real image dir for per-image timing')
    ap.add_argument('--time-n', type=int, default=50)
    args = ap.parse_args()

    rows = []
    for name, kw in CASES:
        sil = make_silhouette(**kw)
        fixed, aligned, shift, rot = mirror_axis_symmetry(sil)
        rows.append({'case': name, 'params': kw, 'fixed_iou': round(fixed, 4),
                     'aligned_iou': round(aligned, 4),
                     'recovered_shift_px': shift, 'recovered_rot_deg': rot})
        print('%-20s fixed %.4f  aligned %.4f  shift %+5.1f  rot %+4.1f'
              % (name, fixed, aligned, shift, rot))

    by = {r['case']: r for r in rows}
    base_f = by['perfect']['fixed_iou']
    base_a = by['perfect']['aligned_iou']
    pose_cost_fixed = base_f - by['tilt_2deg']['fixed_iou']
    pose_cost_aligned = base_a - by['tilt_2deg']['aligned_iou']
    asym_cost_fixed = base_f - by['temple_25pct_short']['fixed_iou']
    asym_cost_aligned = base_a - by['temple_25pct_short']['aligned_iou']
    summary = {
        'pose_cost_fixed_axis': round(pose_cost_fixed, 4),
        'pose_cost_aligned_axis': round(pose_cost_aligned, 4),
        'asymmetry_cost_fixed_axis': round(asym_cost_fixed, 4),
        'asymmetry_cost_aligned_axis': round(asym_cost_aligned, 4),
        'pose_penalty_removed_by_alignment':
            round(1.0 - pose_cost_aligned / max(pose_cost_fixed, 1e-9), 4),
        'asymmetry_penalty_kept_by_alignment':
            round(asym_cost_aligned / max(asym_cost_fixed, 1e-9), 4),
        'offset_translation_invariant': bool(
            by['shift_plus12px']['recovered_shift_px'] == 0
            and by['shift_minus20px']['recovered_shift_px'] == 0),
        'offset_responds_to_yaw': bool(
            abs(by['temple_25pct_short']['recovered_shift_px']) >= 2),
    }
    print('\npose (2 deg tilt) cost:  fixed %.4f   aligned %.4f'
          % (pose_cost_fixed, pose_cost_aligned))
    print('shape asymmetry cost:    fixed %.4f   aligned %.4f'
          % (asym_cost_fixed, asym_cost_aligned))
    print('alignment removes %.1f%% of the pose penalty, keeps %.1f%% of the'
          ' asymmetry penalty'
          % (100 * summary['pose_penalty_removed_by_alignment'],
             100 * summary['asymmetry_penalty_kept_by_alignment']))
    print('translation-invariant       %s' % summary['offset_translation_invariant'])
    print('offset responds to yaw      %s' % summary['offset_responds_to_yaw'])

    timing = None
    sils = []
    if args.real:
        from metrics.evaluate_diversity import list_images, load_rgb
        from metrics.diversity_metrics import extract_silhouette
        for p in list_images(args.real)[:args.time_n]:
            s = extract_silhouette(load_rgb(p))
            if s is not None:
                sils.append(s)
    if not sils:
        sils = [make_silhouette() for _ in range(args.time_n)]
    for s in sils[:5]:
        mirror_axis_symmetry(s)  # warm-up
    t0 = time.perf_counter()
    for s in sils:
        mirror_axis_symmetry(s)
    dt = (time.perf_counter() - t0) / len(sils)
    timing = {'n_images': len(sils),
              'source': 'real' if args.real else 'synthetic',
              'ms_per_image': round(dt * 1000, 1),
              'minutes_per_10k': round(dt * 10000 / 60, 1)}
    print('\ntiming: %.1f ms/image (%s, n=%d) = %.1f min per 10k'
          % (timing['ms_per_image'], timing['source'], timing['n_images'],
             timing['minutes_per_10k']))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'cases': rows, 'summary': summary, 'timing': timing}, f,
                  indent=2)
    print('written: %s' % args.out)


if __name__ == '__main__':
    main()
