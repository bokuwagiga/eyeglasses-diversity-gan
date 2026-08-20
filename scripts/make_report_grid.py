"""Build a labeled real-vs-generated comparison grid for reports.

Samples images at random (fixed seed, no cherry-picking), puts real rows
on top and one labeled block per model below.

--generated may be given more than once, as label=path, so several models
can be compared against a single real block instead of one figure each:

    python scripts/make_report_grid.py \
        --real data/source/images \
        --generated "sharp1 ep1600=results/sharp1/eval1600/generated/images" \
        --generated "mirror2=results/mirror2/eval/generated/images" \
        --out article/figures/grid_models.png

Each block draws from its own seeded stream keyed by its label, so adding
or reordering a model does not change which images the others show.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def list_images(folder):
    files = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f'No images found in {folder}')
    return files


def load_thumb(path, w, h):
    """Center-crop the image to the w:h aspect ratio, then resize to w x h.

    Matches how real images are prepared for training/evaluation, so real
    and generated thumbnails look the same in the grid.
    """
    img = Image.open(path).convert('RGB')
    target = w / h
    if img.width / img.height > target:  # too wide: crop sides
        cw = round(img.height * target)
        x0 = (img.width - cw) // 2
        img = img.crop((x0, 0, x0 + cw, img.height))
    else:  # too tall: crop top/bottom
        ch = round(img.width / target)
        y0 = (img.height - ch) // 2
        img = img.crop((0, y0, img.width, y0 + ch))
    return img.resize((w, h), Image.LANCZOS)


def sample_block(label, folder, n, seed):
    """n random images from folder, on a stream that depends only on label."""
    return random.Random(f'{seed}:{label}').sample(list_images(folder), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True, help='folder with real images')
    ap.add_argument('--generated', required=True, action='append',
                    help='label=folder, repeatable (one block per model)')
    ap.add_argument('--out', required=True, help='output PNG path')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--rows', type=int, default=3, help='rows PER block')
    ap.add_argument('--thumb-width', type=int, default=256, help='thumbnail width px (2:1 aspect)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    n = args.cols * args.rows
    blocks = [('Real catalogue images', sample_block('real', args.real, n, args.seed))]
    for spec in args.generated:
        label, sep, folder = spec.partition('=')
        if not sep:
            label, folder = 'Generated', spec
        blocks.append((label, sample_block(label, folder, n, args.seed)))

    tw = args.thumb_width
    th = tw // 2  # images are 2:1 (512x256 W x H)
    pad = 4
    label_h = 30
    font = ImageFont.load_default(size=17)

    grid_w = args.cols * tw + (args.cols + 1) * pad
    block_h = args.rows * th + args.rows * pad
    total_h = len(blocks) * (label_h + block_h) + pad

    canvas = Image.new('RGB', (grid_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    y = 0
    for label, files in blocks:
        draw.text((pad + 2, y + 7), f'{label}  (random sample, seed {args.seed})',
                  fill='black', font=font)
        y += label_h
        for i, f in enumerate(files):
            r, c = divmod(i, args.cols)
            x = pad + c * (tw + pad)
            canvas.paste(load_thumb(f, tw, th), (x, y + r * (th + pad)))
        y += block_h

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f'Wrote {out} ({canvas.width}x{canvas.height}, '
          f'{len(blocks)} blocks of {n}, seed {args.seed})')


if __name__ == '__main__':
    main()
