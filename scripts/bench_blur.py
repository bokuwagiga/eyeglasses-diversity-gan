"""Benchmark blur implementations inside a realistic training step (GPU).

Times G+D forward+backward under autocast (batch 16, 256x512) with three
Blur variants patched into gan/gan_train.py's models:
  identity  - blur disabled (upper bound on any blur speedup)
  conv      - original depthwise conv2d with runtime-expanded kernel
  shiftadd  - separable shift-and-add (current gan_train.py implementation)

Run on the training PC (stop training first - needs the VRAM):
  python scripts/bench_blur.py
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('gt', ROOT / 'gan' / 'gan_train.py')
gt = importlib.util.module_from_spec(spec)
sys.modules['gt'] = gt
spec.loader.exec_module(gt)

KER = None


def conv_forward(self, x):
    global KER
    if KER is None or KER.device != x.device:
        k = torch.tensor([1.0, 2.0, 1.0], device=x.device)
        k = torch.outer(k, k)
        KER = (k / k.sum()).view(1, 1, 3, 3)
    c = x.size(1)
    return F.conv2d(x, KER.expand(c, 1, 3, 3).to(x.dtype), padding=1, groups=c)


def shiftadd_forward(self, x):
    x = F.pad(x, (1, 1, 1, 1))
    x = (x[..., :-2] + 2.0 * x[..., 1:-1] + x[..., 2:]) * 0.25
    x = (x[..., :-2, :] + 2.0 * x[..., 1:-1, :] + x[..., 2:, :]) * 0.25
    return x


def identity_forward(self, x):
    return x


def bench(name, fwd, batch_size, iters, warmup, device):
    gt.Blur.forward = fwd
    torch.manual_seed(0)
    G = gt.Generator(512, 512, 8).to(device)
    D = gt.Discriminator().to(device)
    z = torch.randn(batch_size, 512, device=device)

    def step():
        with torch.amp.autocast(device, enabled=(device == 'cuda')):
            fake = G(z)
            pred = D(fake)
            loss = pred.mean()
        loss.backward()
        G.zero_grad(set_to_none=True)
        D.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f'{name:<10} {dt * 1000:8.1f} ms / G+D fwd+bwd step')
    del G, D
    if device == 'cuda':
        torch.cuda.empty_cache()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}, batch {args.batch_size}, {args.iters} iters')

    t_id = bench('identity', identity_forward, args.batch_size, args.iters,
                 args.warmup, device)
    t_conv = bench('conv', conv_forward, args.batch_size, args.iters,
                   args.warmup, device)
    t_sa = bench('shiftadd', shiftadd_forward, args.batch_size, args.iters,
                 args.warmup, device)

    print()
    print(f'blur cost:  conv +{(t_conv - t_id) * 1000:.1f} ms, '
          f'shiftadd +{(t_sa - t_id) * 1000:.1f} ms per step')
    if t_sa < t_conv:
        print(f'shiftadd is {t_conv / t_sa:.2f}x faster overall -> keep it')
    else:
        print(f'conv is {t_sa / t_conv:.2f}x faster overall -> report back, '
              f'we will revert the blur implementation')


if __name__ == '__main__':
    main()
