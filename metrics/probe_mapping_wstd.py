"""Measure the style-code dispersion of the mapping network at initialisation.

The inherited (thesis) mapping network stacked `depth` default nn.Linear +
LeakyReLU layers. Default PyTorch initialisation scales each weight by
1/sqrt(fan_in) without the equalized-learning-rate correction, so the signal
contracts at every layer and the w-code collapses toward a constant. A style
code with near-zero variance carries almost no per-image information, which
leaves per-pixel noise injection as the dominant source of image variation -
a plausible mechanism for the low colour diversity of the thesis run.

This probe builds both mapping networks at initialisation, pushes the same
standard-normal z batch through each, and reports the per-dimension standard
deviation of the resulting w codes. No training and no checkpoint required;
the effect is a property of initialisation alone.

  python metrics/probe_mapping_wstd.py --out results/_probes/mapping_wstd.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        'gt', os.path.join(ROOT, 'gan', 'gan_train.py'))
    gt = importlib.util.module_from_spec(spec)
    sys.modules['gt'] = gt
    spec.loader.exec_module(gt)
    return gt


class LegacyPixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class LegacyMapping(nn.Module):
    """The pre-fix mapping network, reproduced from
    reference/gan_train_20260515_snapshot.py: PixelNorm followed by plain
    nn.Linear + LeakyReLU, no equalized learning rate."""

    def __init__(self, z_dim, w_dim, depth):
        super().__init__()
        layers = [LegacyPixelNorm()]
        for _ in range(depth):
            layers.append(nn.Linear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def w_stats(net, z):
    with torch.no_grad():
        w = net(z)
    return {
        'w_std': float(w.std().item()),
        'w_std_per_dim_mean': float(w.std(dim=0).mean().item()),
        'w_abs_mean': float(w.abs().mean().item()),
        'w_shape': list(w.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z-dim', type=int, default=512)
    ap.add_argument('--w-dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/_probes/mapping_wstd.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    z = torch.randn(args.batch, args.z_dim)

    gt = load_trainer()
    torch.manual_seed(args.seed)
    fixed = gt.MappingNetwork(args.z_dim, args.w_dim, depth=args.depth)
    torch.manual_seed(args.seed)
    legacy = LegacyMapping(args.z_dim, args.w_dim, args.depth)

    rep = {
        'config': vars(args),
        'legacy_plain_linear': w_stats(legacy, z),
        'equalized_lr_lr_mul_0.01': w_stats(fixed, z),
        'z_reference': {'z_std': float(z.std().item())},
    }
    rep['ratio_fixed_over_legacy'] = (
        rep['equalized_lr_lr_mul_0.01']['w_std']
        / max(rep['legacy_plain_linear']['w_std'], 1e-12))

    # Per-layer contraction of the legacy stack, to show it is compounding.
    prof, h = [], z
    with torch.no_grad():
        for i, layer in enumerate(legacy.net):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                prof.append({'linear_index': len(prof) + 1,
                             'std_after': float(h.std().item())})
    rep['legacy_per_layer_std'] = prof

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=2)

    print('z std                        %.4f' % rep['z_reference']['z_std'])
    print('legacy plain-Linear  w std   %.4f' % rep['legacy_plain_linear']['w_std'])
    print('equalized-lr (0.01)  w std   %.4f'
          % rep['equalized_lr_lr_mul_0.01']['w_std'])
    print('ratio                        %.1fx' % rep['ratio_fixed_over_legacy'])
    print('\nlegacy per-Linear std profile:')
    for p in prof:
        print('  after linear %d : %.4f' % (p['linear_index'], p['std_after']))
    print('\nwritten: %s' % args.out)


if __name__ == '__main__':
    main()
