"""Contact sheet of novel realistic images next to their nearest real image.

Reads the novel_examples/ folder written by metrics/novelty_rate.py and,
optionally, the artifact scores of the same generated set to print the
number of artifact flags under each generated image.

  python scripts/make_novelty_sheet.py --examples results/mirror2/eval/novelty/novel_examples \
      --scores results/mirror2/eval/artifacts_p2p98/scores_gen.json --out sheet.png
"""
import argparse
import glob
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont


def load(path, w, h):
    im = Image.open(path).convert('RGB')
    iw, ih = im.size
    if abs(iw / ih - w / h) > 0.01:
        ch = int(iw * h / w)
        top = (ih - ch) // 2
        im = im.crop((0, top, iw, top + ch))
    return im.resize((w, h), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--examples', required=True)
    ap.add_argument('--scores', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=12)
    ap.add_argument('--thumb', type=int, default=360)
    args = ap.parse_args()

    scores = json.load(open(args.scores)) if args.scores else {}
    gens = sorted(glob.glob(os.path.join(args.examples, '*_gen_*.png')))
    gens = gens[args.first:args.first + args.count]
    w, h = args.thumb, args.thumb // 2
    pad, lab = 8, 20
    font = ImageFont.load_default(size=14)
    sheet = Image.new('RGB', (2 * w + 3 * pad, len(gens) * (h + lab + pad) + pad),
                      'white')
    d = ImageDraw.Draw(sheet)
    for i, g in enumerate(gens):
        rank = os.path.basename(g).split('_')[0]
        real = glob.glob(os.path.join(args.examples, rank + '_nnreal_*'))[0]
        name = re.sub(r'^\d+_gen_', '', os.path.basename(g))
        flags = scores.get(name, {}).get('n_flags')
        y = pad + i * (h + lab + pad)
        sheet.paste(load(g, w, h), (pad, y + lab))
        sheet.paste(load(real, w, h), (2 * pad + w, y + lab))
        txt = 'rank %s  %s' % (rank, name)
        if flags is not None:
            txt += '  flags %d' % flags
        d.text((pad, y + 2), txt, fill='black', font=font)
        d.text((2 * pad + w, y + 2), 'nearest real', fill='black', font=font)
    sheet.save(args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
