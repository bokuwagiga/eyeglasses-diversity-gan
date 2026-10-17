"""Contact sheets for the StyleGAN2-ADA baseline checks.

--mode visual   : folder of gen_images.py outputs (seedNNNN.png, 512x512, seeds 0-23).
                  Writes sheet_<name>.png, 6 x 4 tiles at half size, each with the file name
                  and a blue box around the content band (rows 128-383).
--mode samples  : folder of 10,000 cropped images (512x256). Draws 24 at random
                  (random.Random(0).sample over the sorted file list, as for the first run),
                  copies them into --out, writes sample_list.txt and
                  sample_sheet_random24_seed0.png (2 columns x 12 rows, full size).

Usage:
  python scripts/sg2ada_contact_sheet.py --mode visual  --src DIR --out PNG
  python scripts/sg2ada_contact_sheet.py --mode samples --src DIR --out DIR
"""
import argparse, os, random, shutil
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["visual", "samples"], required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.src) if f.lower().endswith(".png"))
if len(files) < a.n:
    raise SystemExit(f"only {len(files)} PNGs in {a.src}")

if a.mode == "visual":
    files = files[: a.n]
    COLS, ROWS, S = 6, 4, 256  # tiles scaled 512 -> 256
    sheet = Image.new("RGB", (COLS * (S + 8) + 8, ROWS * (S + 24) + 8), "white")
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(os.path.join(a.src, f)).convert("RGB")
        if im.size != (512, 512):
            raise SystemExit(f"{f}: unexpected size {im.size}")
        im = im.resize((S, S), Image.LANCZOS)
        x = 8 + (i % COLS) * (S + 8)
        y = 8 + (i // COLS) * (S + 24)
        dr.text((x, y + 2), f, fill="black")
        sheet.paste(im, (x, y + 16))
        dr.rectangle((x, y + 16 + 64, x + S - 1, y + 16 + 191), outline=(0, 90, 255), width=2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out} with {len(files)} tiles")
else:
    rng = random.Random(0)
    sample = sorted(rng.sample(files, a.n))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sample_list.txt"), "w") as fh:
        fh.write("\n".join(sample) + "\n")
    COLS, ROWS, W, H = 2, (a.n + 1) // 2, 512, 256
    sheet = Image.new("RGB", (COLS * (W + 8) + 8, ROWS * (H + 24) + 8), "white")
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(sample):
        shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
        im = Image.open(os.path.join(a.src, f)).convert("RGB")
        if im.size != (W, H):
            raise SystemExit(f"{f}: unexpected size {im.size}")
        x = 8 + (i % COLS) * (W + 8)
        y = 8 + (i // COLS) * (H + 24)
        dr.text((x, y + 2), f, fill="black")
        sheet.paste(im, (x, y + 16))
    sheet.save(os.path.join(a.out, "sample_sheet_random24_seed0.png"))
    print(f"copied {len(sample)} samples and wrote the sheet into {a.out}: {sample[:3]} ...")
