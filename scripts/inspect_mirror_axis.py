"""Report what a mirror-coupling generator actually learned.

Two ways a mirror2 run can be uninformative without any metric noticing:

  1. gamma stays at its zero init, so the coupling is switched off and the
     run is just a slower baseline.
  2. gamma opens but the axis head stays at its zero init, so every sample
     reflects about the image centre - i.e. it silently degenerates into
     mirror1 and the ablation compares nothing.

Case 2 is the reason this script exists. It reports the spread of the
predicted axis across random latents, converted to pixels at full 512
width so it can be read against the measured pose statistics from
evaluate_diversity section F (real offset std 2.89 px, mirror1 1.25).

    python scripts/inspect_mirror_axis.py --checkpoint results/mirror2/checkpoints/checkpoint_best.pth
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_train_module():
    path = Path(__file__).resolve().parent.parent / 'gan' / 'gan_train.py'
    spec = importlib.util.spec_from_file_location('gan_train', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--ema', action='store_true', default=True,
                    help='use EMA weights (what generation uses)')
    ap.add_argument('--raw', dest='ema', action='store_false')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    gt = load_train_module()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    keys = ckpt['G'].keys()
    has_axis = any('.axis.' in k for k in keys)
    has_mirror = any(k.startswith('mirror.') for k in keys)
    if not has_mirror:
        print('No mirror coupling in this checkpoint.')
        return 1
    print(f'variant: {"mirror2 (learned axis)" if has_axis else "mirror1 (fixed centre axis)"}')
    print(f'epoch  : {ckpt.get("epoch", "?")}')

    G = gt.Generator(mapping_depth=8, mirror_coupling=not has_axis,
                     mirror_axis=has_axis)
    G.load_state_dict(ckpt['G'])
    if args.ema and 'ema' in ckpt:
        for name, p in G.named_parameters():
            if name in ckpt['ema']:
                p.data.copy_(ckpt['ema'][name])
        print('weights : EMA')
    else:
        print('weights : raw G')
    G.eval()

    torch.manual_seed(args.seed)
    with torch.no_grad():
        w = G.get_w(torch.randn(args.samples, G.w_dim))

    print()
    for idx, m in G.mirror.items():
        stage_w = 16 * 2 ** int(idx)          # 8x16 seed, stage i -> width 16*2^i
        print(f'stage {idx} ({stage_w // 2}x{stage_w} feature map)')
        print(f'  gamma            : {m.gamma.item():+.4f}')
        if m.axis is None:
            print('  axis             : fixed at feature-map centre')
            continue
        with torch.no_grad():
            a = torch.tanh(m.axis(w))[:, 0] * m.MAX_AXIS
        px = a * 512.0                        # comparable to metric offsets
        sat = (a.abs() > 0.95 * m.MAX_AXIS).float().mean()
        print(f'  axis W norm      : {m.axis.weight.norm().item():.4f}  '
              f'bias {m.axis.bias.item():+.4f}')
        print(f'  predicted offset : mean {px.mean():+.2f} px  std {px.std():.2f} px '
              f'(at 512 wide)')
        print(f'                     p05 {px.quantile(0.05):+.2f}  '
              f'p50 {px.median():+.2f}  p95 {px.quantile(0.95):+.2f}')
        print(f'  clipping at tanh : {100 * sat:.1f}% of samples '
              f'(high means MAX_AXIS is too small)')
        if px.std() < 0.25:
            print('  WARNING: axis is effectively constant - this run has '
                  'degenerated into mirror1')
    print()
    print('Reference: real axis-offset std 2.89 px, sharp1 3.08, mirror1 1.25.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
