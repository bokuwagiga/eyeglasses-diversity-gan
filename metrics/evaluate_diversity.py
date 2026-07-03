"""
Evaluate the diversity (and quality) of a generated eyeglass dataset against
the real reference set.

Usage:
  python metrics/evaluate_diversity.py --generated <dir> --real <dir> \
      --out results/eval_run1 [--max-images 5000] [--feature-subset 2000]

Outputs into --out:
  diversity_report.json   all metric values
  diversity_report.txt    human-readable summary

Metric groups (see diversity_metrics.py for definitions):
  A. Quality reference: FID, KID (torchmetrics, Inception-v3)
  B. Geometric descriptor diversity: dispersion + JSD vs real per descriptor
  C. Colour diversity: frame-pixel CIELAB pairwise distance, hue entropy,
     a*b* coverage - reported for both sets
  D. Feature-space: Vendi Score, Precision/Recall, Density/Coverage,
     nearest-real-neighbour distances (memorisation check)
  E. Perceptual: mean pairwise LPIPS within each set
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_metrics as dm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_H, IMG_W = 256, 512


def list_images(directory):
    paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f'No images found under {directory}')
    return paths


def load_rgb(path):
    """Load, center-crop to 2:1 and resize to 256x512. Returns uint8 (H, W, 3)."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    target = IMG_W / IMG_H
    if w / h > target:
        new_w = int(h * target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((IMG_W, IMG_H), Image.BICUBIC)
    return np.asarray(img)


def load_images(paths, desc):
    return [load_rgb(p) for p in tqdm(paths, desc=desc)]


# Inception features


class InceptionFeatures:
    """Inception-v3 pool3 (2048-d) feature extractor."""

    def __init__(self):
        from torchvision import models, transforms
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval().to(DEVICE)
        self.net = net
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, images, batch_size=32, desc='Features'):
        feats = []
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            batch = torch.stack([self.tf(Image.fromarray(im))
                                 for im in images[i:i + batch_size]]).to(DEVICE)
            feats.append(self.net(batch).cpu().numpy())
        return np.concatenate(feats)


# FID / KID


@torch.no_grad()
def fid_kid(real_images, gen_images, batch_size=32):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
    kid = KernelInceptionDistance(
        subset_size=min(1000, len(real_images) // 2, len(gen_images) // 2)).to(DEVICE)

    def feed(images, real):
        for i in tqdm(range(0, len(images), batch_size),
                      desc=f'FID/KID {"real" if real else "gen"}'):
            batch = np.stack(images[i:i + batch_size])
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(DEVICE)
            fid.update(t, real=real)
            kid.update(t, real=real)

    feed(real_images, True)
    feed(gen_images, False)
    kid_mean, kid_std = kid.compute()
    return {'fid': float(fid.compute().item()),
            'kid_mean': float(kid_mean.item()),
            'kid_std': float(kid_std.item())}


# LPIPS pairwise


@torch.no_grad()
def lpips_pairwise(images, n_pairs=500, seed=0, desc='LPIPS'):
    import lpips
    net = lpips.LPIPS(net='vgg', verbose=False).to(DEVICE)
    rng = np.random.default_rng(seed)
    n = len(images)
    dists = []
    for _ in tqdm(range(n_pairs), desc=desc):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a = torch.from_numpy(images[i]).permute(2, 0, 1).float().div(127.5).sub(1)
        b = torch.from_numpy(images[j]).permute(2, 0, 1).float().div(127.5).sub(1)
        d = net(a.unsqueeze(0).to(DEVICE), b.unsqueeze(0).to(DEVICE))
        dists.append(d.item())
    return {'mean': float(np.mean(dists)), 'std': float(np.std(dists))}


# Report


def write_txt_report(report, path):
    lines = []
    add = lines.append
    add('=' * 70)
    add('Eyeglass Dataset Diversity Report')
    add(f'  generated: {report["meta"]["generated_dir"]} '
        f'({report["meta"]["n_generated"]} images)')
    add(f'  real:      {report["meta"]["real_dir"]} '
        f'({report["meta"]["n_real"]} images)')
    add('=' * 70)

    if 'quality' in report:
        add('')
        add('--- A. Quality reference ---')
        q = report['quality']
        add(f'  FID  : {q["fid"]:.3f}   (lower = closer to real distribution)')
        add(f'  KID  : {q["kid_mean"]:.5f} +- {q["kid_std"]:.5f}')

    add('')
    add('--- B. Geometric descriptor diversity ---')
    add(f'  Silhouette extraction failures: gen '
        f'{report["geometry"]["failed_gen"]}, real {report["geometry"]["failed_real"]}')
    add(f'  {"descriptor":<20}{"gen std":>10}{"real std":>10}'
        f'{"std ratio":>11}{"JSD":>8}')
    disp_g = report['geometry']['dispersion_gen']
    disp_r = report['geometry']['dispersion_real']
    jsd = report['geometry']['jsd']
    for name in dm.DESCRIPTOR_NAMES:
        sg, sr = disp_g[name]['std'], disp_r[name]['std']
        ratio = sg / sr if sr > 0 else float('nan')
        add(f'  {name:<20}{sg:>10.4f}{sr:>10.4f}{ratio:>11.2f}{jsd[name]:>8.3f}')
    add('  (std ratio > 1: generated more dispersed; JSD near 0: covers real distribution)')

    add('')
    add('--- C. Colour diversity (frame pixels, CIELAB) ---')
    add(f'  {"metric":<20}{"generated":>12}{"real":>12}')
    for key in ['pairwise_lab', 'hue_entropy', 'ab_coverage']:
        add(f'  {key:<20}{report["color"]["gen"][key]:>12.4f}'
            f'{report["color"]["real"][key]:>12.4f}')

    add('')
    add('--- D. Feature-space diversity (Inception-v3) ---')
    v = report['feature_space']
    add(f'  Vendi Score      : gen {v["vendi_gen"]["vendi"]:.1f} / {v["vendi_gen"]["n"]}'
        f'   real {v["vendi_real"]["vendi"]:.1f} / {v["vendi_real"]["n"]}')
    add(f'  Vendi per sample : gen {v["vendi_gen"]["vendi_per_sample"]:.4f}'
        f'   real {v["vendi_real"]["vendi_per_sample"]:.4f}')
    p = v['prdc']
    add(f'  Precision {p["precision"]:.3f}  Recall {p["recall"]:.3f}  '
        f'Density {p["density"]:.3f}  Coverage {p["coverage"]:.3f}  (k={p["k"]})')
    add('  (recall & coverage measure diversity relative to the real set)')
    m = v['memorisation']
    add(f'  Nearest-real dist: gen->real {m["gen_to_real_mean"]:.3f}  '
        f'real->real {m["real_to_real_mean"]:.3f}  '
        f'ratio {m["ratio_gen_over_real"]:.2f}')
    add('  (ratio << 1 with low p05 would indicate memorisation)')

    if 'lpips' in report:
        add('')
        add('--- E. Perceptual diversity (LPIPS, pairwise within set) ---')
        add(f'  generated: {report["lpips"]["gen"]["mean"]:.4f} '
            f'+- {report["lpips"]["gen"]["std"]:.4f}')
        add(f'  real:      {report["lpips"]["real"]["mean"]:.4f} '
            f'+- {report["lpips"]["real"]["std"]:.4f}')

    add('=' * 70)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate diversity of generated eyeglass dataset vs real.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generated', required=True, help='Generated images dir')
    parser.add_argument('--real', required=True, help='Real images dir')
    parser.add_argument('--out', required=True, help='Output directory for reports')
    parser.add_argument('--max-images', type=int, default=5000,
                        help='Max images per set for geometry/colour metrics')
    parser.add_argument('--feature-subset', type=int, default=2000,
                        help='Subset size for Vendi/PRDC (eigendecomposition cost)')
    parser.add_argument('--lpips-pairs', type=int, default=500,
                        help='Random pairs for LPIPS (0 to skip)')
    parser.add_argument('--skip-fid', action='store_true',
                        help='Skip FID/KID computation')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_paths = list_images(args.generated)
    real_paths = list_images(args.real)

    def subsample(paths, n):
        if len(paths) <= n:
            return paths
        idx = rng.choice(len(paths), size=n, replace=False)
        return [paths[i] for i in sorted(idx)]

    gen_paths_s = subsample(gen_paths, args.max_images)
    real_paths_s = subsample(real_paths, args.max_images)

    gen_images = load_images(gen_paths_s, 'Load generated')
    real_images = load_images(real_paths_s, 'Load real')

    report = {'meta': {
        'generated_dir': args.generated, 'real_dir': args.real,
        'n_generated': len(gen_images), 'n_real': len(real_images),
        'n_generated_total': len(gen_paths), 'n_real_total': len(real_paths),
        'seed': args.seed,
    }}

    # A. Quality
    if not args.skip_fid:
        print('\n[A] FID / KID ...')
        report['quality'] = fid_kid(real_images, gen_images)

    # B. Geometry
    print('\n[B] Geometric descriptors ...')
    sil_gen = [dm.extract_silhouette(im) for im in tqdm(gen_images, desc='Silhouettes gen')]
    sil_real = [dm.extract_silhouette(im) for im in tqdm(real_images, desc='Silhouettes real')]
    desc_gen, failed_gen = dm.descriptor_matrix(gen_images)
    desc_real, failed_real = dm.descriptor_matrix(real_images)
    report['geometry'] = {
        'failed_gen': failed_gen, 'failed_real': failed_real,
        'dispersion_gen': dm.descriptor_dispersion(desc_gen),
        'dispersion_real': dm.descriptor_dispersion(desc_real),
        'jsd': dm.descriptor_jsd(desc_gen, desc_real),
    }

    # C. Colour
    print('\n[C] Colour diversity ...')
    report['color'] = {
        'gen': dm.color_diversity(gen_images, sil_gen, seed=args.seed),
        'real': dm.color_diversity(real_images, sil_real, seed=args.seed),
    }

    # D. Feature space
    print('\n[D] Feature-space diversity ...')
    extractor = InceptionFeatures()
    n_sub = args.feature_subset
    gen_sub_idx = rng.choice(len(gen_images), size=min(n_sub, len(gen_images)), replace=False)
    real_sub_idx = rng.choice(len(real_images), size=min(n_sub, len(real_images)), replace=False)
    feats_gen = extractor.extract([gen_images[i] for i in gen_sub_idx], desc='Features gen')
    feats_real = extractor.extract([real_images[i] for i in real_sub_idx], desc='Features real')
    report['feature_space'] = {
        'vendi_gen': dm.vendi_score(feats_gen),
        'vendi_real': dm.vendi_score(feats_real),
        'prdc': dm.precision_recall_density_coverage(feats_real, feats_gen),
        'memorisation': dm.nearest_real_distances(feats_real, feats_gen),
    }

    # E. LPIPS
    if args.lpips_pairs > 0:
        print('\n[E] LPIPS pairwise ...')
        report['lpips'] = {
            'gen': lpips_pairwise(gen_images, args.lpips_pairs, args.seed, 'LPIPS gen'),
            'real': lpips_pairwise(real_images, args.lpips_pairs, args.seed, 'LPIPS real'),
        }

    with open(out_dir / 'diversity_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    write_txt_report(report, out_dir / 'diversity_report.txt')
    print(f'\nReports written to {out_dir}')


if __name__ == '__main__':
    main()
