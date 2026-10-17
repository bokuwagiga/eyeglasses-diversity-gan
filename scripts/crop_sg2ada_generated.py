"""Crop StyleGAN2-ADA baseline outputs (512x512) back to the 256x512 content band.

The training images were 256x512 frames padded with 128 rows above and below
(see the Phase 1 dataset preparation), so rows 128-383 of every generated image
hold the frame content. Writes the crops with the same file names and a
generation_config.json with the provenance.

Usage: python scripts/crop_sg2ada_generated.py --src DIR --dst DIR --snapshot PKL
"""
import argparse, hashlib, json, os, subprocess, sys, time
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder with gen_images.py output (seedNNNN.png, 512x512)")
ap.add_argument("--dst", required=True, help="output folder for the 256x512 crops")
ap.add_argument("--snapshot", required=True, help="network pickle the images were generated from")
ap.add_argument("--stylegan3", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "stylegan3"),
                help="stylegan3 clone (default: next to this repository)")
ap.add_argument("--gen-command", default="", help="exact gen_images.py command line, for the record")
ap.add_argument("--expect", type=int, default=10000, help="required number of input PNGs")
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.src) if f.lower().endswith(".png"))
if len(files) != a.expect:
    sys.exit(f"expected {a.expect} PNGs in {a.src}, found {len(files)}")
os.makedirs(a.dst, exist_ok=True)
for f in files:
    im = Image.open(os.path.join(a.src, f)).convert("RGB")
    if im.size != (512, 512):
        sys.exit(f"{f}: unexpected size {im.size}")
    im.crop((0, 128, 512, 384)).save(os.path.join(a.dst, f), compress_level=6)

h = hashlib.sha256()
with open(a.snapshot, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
try:
    commit = subprocess.check_output(["git", "-C", a.stylegan3, "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", a.stylegan3, "status", "--porcelain"], text=True).strip())
except Exception as e:  # noqa: BLE001
    commit, dirty = f"unknown ({e})", None
cfg = {
    "model": "StyleGAN2-ADA (NVlabs stylegan3 repo, --cfg=stylegan2)",
    "stylegan3_commit": commit, "stylegan3_working_tree_dirty": dirty,
    "snapshot": os.path.abspath(a.snapshot), "snapshot_sha256": h.hexdigest(),
    "seeds": "0-9999", "num_images": len(files), "truncation_psi": 1.0, "noise_mode": "const",
    "generated_resolution": "512x512", "crop_rows": "128-383 (256x512 content band)",
    "gen_command": a.gen_command, "cropped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(os.path.join(os.path.dirname(os.path.dirname(a.dst.rstrip(os.sep + "/"))), "generation_config.json"), "w") as fh:
    json.dump(cfg, fh, indent=2)
print(f"cropped {len(files)} images into {a.dst}")
