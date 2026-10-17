"""Pad the catalogue images to 512x512 for the StyleGAN2-ADA baseline.

Content = metrics.evaluate_diversity.load_rgb (centre-crop 2:1, BICUBIC to 256x512).
Padding = 128 rows top + 128 bottom, filled with the median of the border pixels
(the same background estimate as metrics.diversity_metrics.extract_silhouette).
"""
import os, sys, json
import numpy as np
from PIL import Image
from tqdm import tqdm
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from metrics.evaluate_diversity import load_rgb   # noqa: E402

src = os.path.join(repo, "data", "source", "images")
files = sorted(f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".jpeg")))
os.makedirs(out, exist_ok=True)
bgs = []
for f in tqdm(files, desc="pad"):
    img = load_rgb(os.path.join(src, f))                       # uint8 (256, 512, 3)
    assert img.shape == (256, 512, 3), (f, img.shape)
    border = np.concatenate([img[0, :].reshape(-1, 3), img[-1, :].reshape(-1, 3),
                             img[:, 0].reshape(-1, 3), img[:, -1].reshape(-1, 3)]).astype(np.float32)
    bg = np.median(border, axis=0)                              # same estimate as extract_silhouette
    bg8 = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    canvas = np.empty((512, 512, 3), np.uint8); canvas[:] = bg8
    canvas[128:384] = img
    Image.fromarray(canvas).save(os.path.join(out, os.path.splitext(f)[0] + ".png"), compress_level=1)
    bgs.append(bg8.tolist())
json.dump({"n": len(files), "src": src, "bg_examples": dict(zip(files[:5], bgs[:5]))},
          open(os.path.join(os.path.dirname(out.rstrip("/\\")), "prep_summary.json"), "w"), indent=1)
print("written", len(files), "images to", out)
