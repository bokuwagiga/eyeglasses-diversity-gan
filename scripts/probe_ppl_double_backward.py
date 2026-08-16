"""Check that the mirror-axis coupling survives PPL on THIS machine's CUDA.

PPL differentiates the generator output with create_graph=True and then
backprops through that gradient, so every op in the synthesis path needs a
double-backward formula. The first mirror2 launch died on the first
optimizer step because grid_sample has none on CUDA (the CPU kernel does,
which is why it could not be reproduced on the dev box).

Run this before committing GPU-days to a mirror2 run:

    python scripts/probe_ppl_double_backward.py

It reports PASS/FAIL for each variant under autocast + GradScaler, which is
how training actually runs, and separately confirms whether raw
grid_sample double backward is the thing that is unsupported here.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_train_module():
    path = Path(__file__).resolve().parent.parent / 'gan' / 'gan_train.py'
    spec = importlib.util.spec_from_file_location('gan_train', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ppl_step(G, device, batch, latent_dim, amp):
    """One PPL-regularised G step, matching gan_train.train()."""
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    opt = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=amp):
        z = torch.randn(batch, latent_dim, device=device)
        w = G.get_w(z)
        imgs = G.synthesis(w)
    imgs = imgs.float()
    noise = torch.randn_like(imgs) / math.sqrt(imgs.shape[2] * imgs.shape[3])
    grad = torch.autograd.grad(outputs=(imgs * noise).sum(),
                               inputs=w, create_graph=True)[0]
    path_lengths = grad.float().pow(2).sum(-1).mean(-1).sqrt() \
        if grad.dim() == 3 else grad.float().pow(2).sum(-1).sqrt()
    penalty = (path_lengths - path_lengths.mean().detach()).pow(2).mean()
    scaler.scale(penalty).backward()
    scaler.step(opt)
    scaler.update()
    return penalty.item()


def raw_grid_sample_check(device):
    x = torch.randn(2, 3, 8, 16, device=device, requires_grad=True)
    a = torch.zeros(2, 1, device=device, requires_grad=True)
    theta = x.new_zeros(2, 2, 3)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * torch.tanh(a)[:, 0]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    y = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
    g, = torch.autograd.grad((y * torch.randn_like(y)).sum(), a,
                             create_graph=True)
    g.sum().backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=16,
                    help='matches --ppl-batch-size')
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('CUDA not available - this probe is meaningless on CPU, '
              'because the CPU kernel HAS the double-backward formula.')
        return 1

    gt = load_train_module()
    device = torch.device('cuda')
    amp = not args.no_amp
    print(f'torch {torch.__version__}  cuda {torch.version.cuda}  '
          f'{torch.cuda.get_device_name(0)}')
    print(f'batch {args.batch}  amp {amp}\n')

    print('--- raw grid_sample double backward (the suspected culprit) ---')
    try:
        raw_grid_sample_check(device)
        print('  OK - grid_sample is NOT the problem here; report the full '
              'error text from the failing run instead.\n')
    except RuntimeError as e:
        print(f'  FAILS as expected: {e}\n')

    print('--- generator variants under PPL ---')
    variants = [('baseline (no coupling)', {}),
                ('mirror1 --g-mirror-coupling', dict(mirror_coupling=True)),
                ('mirror2 --g-mirror-axis', dict(mirror_axis=True))]
    ok = True
    for name, kw in variants:
        G = gt.Generator(mapping_depth=8, **kw).to(device)
        for m in getattr(G, 'mirror', {}).values():
            torch.nn.init.constant_(m.gamma, 0.3)   # force the path open
            if m.axis is not None:
                torch.nn.init.normal_(m.axis.weight, std=0.05)
        try:
            for _ in range(3):
                val = ppl_step(G, device, args.batch, 512, amp)
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f'  PASS  {name:<30} ppl {val:.4f}  peak {mem:.2f} GB')
        except RuntimeError as e:
            ok = False
            print(f'  FAIL  {name:<30} {e}')
        del G
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print('\n' + ('All variants OK - safe to launch mirror2.' if ok
                  else 'A variant still fails - do NOT launch; send the text above.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
