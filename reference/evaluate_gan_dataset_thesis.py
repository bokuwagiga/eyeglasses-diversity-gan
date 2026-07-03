"""
evaluate_dataset_v2.py - Domain-specific evaluation for generated eyeglass frame datasets.

Standard metrics (FID, IS, LPIPS) are computed alongside three domain-specific metrics
designed for product images with small foreground regions on uniform backgrounds:

  1. Edge Coherence     - how well the mask boundary aligns with image edges (Canny).
                          A well-trained joint generator should produce masks whose
                          boundaries coincide with real frame edges in the RGB output.
                          Score: mean F1 between dilated mask boundary and Canny edges,
                          computed only within a band around the mask boundary.

  2. Mask Regularity   - measures contour smoothness and closure.
                          Fragmented or jagged masks indicate the decoder has not learned
                          a stable frame shape. Score: ratio of largest contour area to
                          total white pixel area (compactness), penalised by contour count.

  3. Frame Symmetry    - eyeglass frames are left-right symmetric by design.
                          Score: 1 - normalised pixel difference between the masked region
                          and its horizontal mirror, computed only within the mask.

Usage:
    python evaluate_dataset_v2.py \\
        --real_imgs   ./data/frames_orig \\
        --gen_imgs    ./output/generated/images \\
        --gen_masks   ./output/generated/masks \\
        --real_masks  ./data/masks \\
        --output      evaluation_results_v2.txt \\
        --n_sample    2000   # use subset for speed; -1 for all

Dependencies:
    pip install torch torchvision scipy lpips scikit-image opencv-python-headless numpy Pillow
"""

import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torchvision.models import inception_v3
from tqdm import tqdm

warnings.filterwarnings("ignore")

#  Constants

IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


#  File helpers

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_rgb(path, size=(299, 299)):
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.LANCZOS)


def load_mask(path, orig_size=None):
    """Load mask as a numpy uint8 array (0 or 255)."""
    mask = Image.open(path).convert("L")
    if orig_size:
        mask = mask.resize(orig_size, Image.NEAREST)
    arr = np.array(mask)
    return (arr > 127).astype(np.uint8) * 255


def sample_paths(paths, n):
    if n == -1 or n >= len(paths):
        return paths
    return random.sample(paths, n)


#  Inception features for FID

def build_inception():
    model = inception_v3(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval().to(DEVICE)
    return model


def preprocess_to_canvas(img: Image.Image, target_w=512, target_h=256) -> Image.Image:
    """
    Bring any image into the same 256x512 space used by the generator:
      1. Center-crop to the target aspect ratio (2:1 width:height).
      2. Resize to exactly 256x512.
    This ensures real and generated images are in the same pixel distribution
    before FID features are extracted.
    """
    w, h = img.size
    target_ratio = target_w / target_h  # 2.0

    if w / h > target_ratio:
        # Image is wider than 2:1 - crop width
        new_w = int(h * target_ratio)
        left  = (w - new_w) // 2
        img   = img.crop((left, 0, left + new_w, h))
    else:
        # Image is taller than 2:1 - crop height
        new_h = int(w / target_ratio)
        top   = (h - new_h) // 2
        img   = img.crop((0, top, w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


class _ImageListDataset(torch.utils.data.Dataset):
    """Thin dataset that loads and transforms images from a list of paths."""
    def __init__(self, paths, transform, normalize_to_canvas=False):
        self.paths = paths
        self.transform = transform
        self.normalize_to_canvas = normalize_to_canvas

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.normalize_to_canvas:
            img = preprocess_to_canvas(img)
        return self.transform(img)


def get_features(img_paths, model, batch_size=32, normalize_to_canvas=False,
                 num_workers=6):
    """
    Extract Inception-v3 features.
    Set normalize_to_canvas=True for real training images so they are
    center-cropped and resized to 256x512 before feature extraction,
    matching the resolution of the generated images.
    """
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = _ImageListDataset(img_paths, transform, normalize_to_canvas)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    feats = []
    for batch in tqdm(loader, desc="  Extracting features", unit="batch"):
        with torch.no_grad():
            f = model(batch.to(DEVICE)).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def compute_fid(real_feats, gen_feats):
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


#  Inception Score

def compute_is(img_paths, model, batch_size=32, splits=10):
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    softmax = torch.nn.Softmax(dim=1)
    # Re-attach classifier head temporarily
    inc = inception_v3(weights="IMAGENET1K_V1").eval().to(DEVICE)
    preds = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="  Inception Score", unit="batch"):
        batch = [transform(Image.open(p).convert("RGB")) for p in img_paths[i:i+batch_size]]
        batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            p = softmax(inc(batch)).cpu().numpy()
        preds.append(p)
    preds = np.concatenate(preds, axis=0)
    scores = []
    n = len(preds)
    split_size = n // splits
    for s in range(splits):
        part = preds[s * split_size:(s + 1) * split_size]
        py = part.mean(axis=0)
        scores.append(np.exp(np.mean(np.sum(part * (np.log(part + 1e-10) - np.log(py + 1e-10)), axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


#  LPIPS

def compute_lpips(img_paths_a, img_paths_b, batch_size=32):
    loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                            T.Normalize([0.5]*3, [0.5]*3)])
    vals = []
    pairs = list(zip(img_paths_a, img_paths_b))
    for i in tqdm(range(0, len(pairs), batch_size), desc="  LPIPS", unit="batch"):
        chunk = pairs[i:i+batch_size]
        a = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in chunk]).to(DEVICE)
        b = torch.stack([transform(Image.open(p).convert("RGB")) for _, p in chunk]).to(DEVICE)
        with torch.no_grad():
            d = loss_fn(a, b).cpu().numpy().flatten()
        vals.extend(d.tolist())
    return float(np.mean(vals)), float(np.std(vals))


#  Domain-specific metric 1: Edge Coherence

def edge_coherence_score(img_path, mask_path, dilation_px=3):
    """
    Measures alignment between the mask boundary and real image edges.

    Method:
      1. Extract mask boundary pixels (morphological gradient).
      2. Detect edges in the RGB image using Canny within a band around the boundary.
      3. Compute F1 (precision * recall) between dilated boundary and Canny edges.

    A score near 1.0 means the generator placed mask boundaries exactly where
    the image has colour discontinuities (i.e. at the actual frame edges).
    A low score means the mask does not align with what is visible in the image.
    """
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None

    mask_bin = (mask > 127).astype(np.uint8)

    # Mask boundary via morphological gradient
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, kernel)

    # Dilate boundary to create evaluation band (tolerance window)
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2*dilation_px+1, 2*dilation_px+1))
    band = cv2.dilate(boundary, dil_kernel)

    # Canny edges in grayscale image, restricted to the band
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=80)
    edges_in_band = (edges > 0) & (band > 0)
    boundary_bool = boundary > 0

    tp = np.sum(edges_in_band & boundary_bool).astype(float)
    fp = np.sum(edges_in_band & ~boundary_bool).astype(float)
    fn = np.sum(~edges_in_band & boundary_bool).astype(float)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def batch_edge_coherence(img_paths, mask_paths, dilation_px=3):
    scores = []
    for ip, mp in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="  Edge coherence", unit="img"):
        s = edge_coherence_score(ip, mp, dilation_px)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)), float(np.std(scores))


#  Domain-specific metric 2: Mask Regularity

def mask_regularity_score(mask_path):
    """
    Measures how well-formed the generated mask is.

    A good mask should have:
      - One dominant connected component (the frame outline)
      - Smooth, closed contours
      - Minimal fragmentation (small isolated blobs)

    Score = (area of largest contour / total white area) * (1 / num_contours_penalty)
    Score range: 0 (fragmented/noisy) to 1 (single clean contour).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    total_white = int(np.sum(binary > 0))
    if total_white == 0:
        return 0.0

    # For hollow frame shapes (ring outlines), cv2.contourArea() returns the
    # area enclosed by the outer boundary - larger than actual white pixels.
    # Instead, draw each contour filled and count pixel overlap with the real mask,
    # keeping the score in [0, 1] regardless of shape topology.
    filled_areas = []
    for c in contours:
        tmp = np.zeros_like(binary)
        cv2.drawContours(tmp, [c], -1, 255, thickness=cv2.FILLED)
        filled_areas.append(cv2.countNonZero(tmp))

    best_idx = int(np.argmax(filled_areas))
    tmp = np.zeros_like(binary)
    cv2.drawContours(tmp, [contours[best_idx]], -1, 255, thickness=cv2.FILLED)
    overlap = cv2.countNonZero(cv2.bitwise_and(tmp, binary))

    # Compactness: fraction of actual white pixels captured by the largest contour
    compactness = overlap / (total_white + 1e-8)

    # Penalise fragmentation: more contours = lower score
    # 1 contour -> 1.0, 2 -> 0.85, 5 -> 0.60, 10+ -> ~0.40
    n = len(contours)
    frag_penalty = 1.0 / (1.0 + 0.2 * (n - 1))

    return float(min(compactness * frag_penalty, 1.0))


def batch_mask_regularity(mask_paths):
    scores = [mask_regularity_score(p) for p in tqdm(mask_paths, desc="  Mask regularity", unit="img")]
    scores = [s for s in scores if s is not None]
    return float(np.mean(scores)), float(np.std(scores))


#  Domain-specific metric 3: Frame Symmetry

def frame_symmetry_score(img_path, mask_path):
    """
    Eyeglass frames are left-right symmetric by design.
    Measures how symmetric the generated frame is by comparing the masked
    region with its horizontal mirror image.

    Score = 1 - (mean absolute pixel difference between left and right halves
                 of the masked region, normalised to [0, 1]).

    Score near 1.0 = highly symmetric (as expected for real frames).
    Score near 0.0 = asymmetric (unlikely for a well-trained model).

    Note: some asymmetry is expected due to perspective and lighting;
    scores above 0.75 are considered good for this domain.
    """
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None

    mask_bin = (mask > 127).astype(np.float32)
    img_f    = img.astype(np.float32) / 255.0

    # Flip image and mask horizontally
    img_flip  = cv2.flip(img_f,    1)
    mask_flip = cv2.flip(mask_bin, 1)

    # Intersection mask: pixels that are frame in BOTH original and flipped
    intersection = (mask_bin > 0) & (mask_flip > 0)
    if intersection.sum() < 100:
        return None  # not enough overlap to measure

    diff = np.abs(img_f - img_flip)           # per-pixel per-channel difference
    diff_masked = diff[intersection]           # only within frame region
    mean_diff = diff_masked.mean()             # 0 = identical, 1 = fully different

    return float(1.0 - mean_diff)


def batch_frame_symmetry(img_paths, mask_paths):
    scores = []
    for ip, mp in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="  Frame symmetry", unit="img"):
        s = frame_symmetry_score(ip, mp)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)), float(np.std(scores))


#  Color and mask diversity

def color_diversity(img_paths, n_sample=500):
    paths = random.sample(img_paths, min(n_sample, len(img_paths)))
    means = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB")).reshape(-1, 3)
        means.append(img.mean(axis=0))
    means = np.array(means)
    dists = []
    idx   = random.sample(range(len(means)), min(200, len(means)))
    for i in idx:
        for j in idx:
            if i != j:
                dists.append(np.linalg.norm(means[i] - means[j]) / 255.0)
    return float(np.mean(dists)) if dists else 0.0


def mask_shape_diversity(mask_paths, n_sample=500):
    paths = random.sample(mask_paths, min(n_sample, len(mask_paths)))
    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(cv2.resize(m, (64, 32)) > 127)
    masks = np.array(masks).reshape(len(masks), -1).astype(np.float32)
    dists = []
    idx   = random.sample(range(len(masks)), min(200, len(masks)))
    for i in idx:
        for j in idx:
            if i != j:
                dists.append(np.mean(masks[i] != masks[j]))
    return float(np.mean(dists)) if dists else 0.0


#  Main

def main(
    real_imgs,
    gen_imgs,
    gen_masks,
    real_masks,
    output="evaluation_results_v2.txt",
    n_sample=10000,
    no_fid=False,
    no_is=False,
    no_lpips=False,
    no_edge_coherence=False,
    no_mask_regularity=False,
    no_frame_symmetry=False,
    no_diversity=False,
    previous_results=None
):
    real_img_paths  = list_images(Path(real_imgs))
    gen_img_paths   = list_images(Path(gen_imgs))
    gen_mask_paths  = list_images(Path(gen_masks))
    real_mask_paths = list_images(Path(real_masks))

    print(f"Real images : {len(real_img_paths)}")
    print(f"Gen images  : {len(gen_img_paths)}")
    print(f"Gen masks   : {len(gen_mask_paths)}")
    print(f"Device      : {DEVICE}")

    # Sample for speed
    gen_img_s   = sample_paths(gen_img_paths,  n_sample)
    gen_mask_s  = sample_paths(gen_mask_paths, n_sample)

    # Align gen_img and gen_mask by stem name
    gen_mask_map = {p.stem.replace("mask_", "glass_"): p for p in gen_mask_paths}
    paired_imgs, paired_masks = [], []
    for ip in gen_img_s:
        mp = gen_mask_map.get(ip.stem)
        if mp:
            paired_imgs.append(ip)
            paired_masks.append(mp)

    print(f"\nPaired samples for domain metrics: {len(paired_imgs)}")

    #  Load previous results if provided 
    def load_previous(path):
        """Parse a previous results .txt file into a dict of {key: float}."""
        prev = {}
        key_map = {
            "FID"                    : "FID",
            "IS_mean"                : "IS_mean",
            "IS_std"                 : "IS_std",
            "LPIPS_mean"             : "LPIPS_mean",
            "LPIPS_std"              : "LPIPS_std",
            "edge_coherence_mean"    : "edge_coherence_mean",
            "edge_coherence_std"     : "edge_coherence_std",
            "mask_regularity_mean"   : "mask_regularity_mean",
            "mask_regularity_std"    : "mask_regularity_std",
            "frame_symmetry_mean"    : "frame_symmetry_mean",
            "frame_symmetry_std"     : "frame_symmetry_std",
            "color_diversity"        : "color_diversity",
            "mask_shape_diversity"   : "mask_shape_diversity",
        }
        with open(path) as f:
            for line in f:
                line = line.strip()
                for txt_key, dict_key in key_map.items():
                    if line.startswith(txt_key):
                        parts = line.split(":")
                        if len(parts) == 2:
                            try:
                                prev[dict_key] = float(parts[1].strip())
                            except ValueError:
                                pass
        return prev

    results = {}

    if previous_results:
        prev = load_previous(previous_results)
        results.update(prev)
        print(f"\nLoaded {len(prev)} previous results from: {previous_results}")
        print("  These will appear in the final report even for skipped metrics.")

    step = 1
    total_steps = sum([
        not no_fid, not no_is, not no_lpips,
        not no_edge_coherence, not no_mask_regularity,
        not no_frame_symmetry, not no_diversity,
    ])

    #  Standard metrics 
    need_inception = not no_fid or not no_is
    inception = build_inception() if need_inception else None

    if not no_fid:
        print(f"\n[{step}/{total_steps}] Computing FID...")
        print("  (Real images: center-cropped and resized to 256x512 to match generated)")
        real_feats = get_features(real_img_paths, inception, normalize_to_canvas=True)
        gen_feats  = get_features(gen_img_s,      inception, normalize_to_canvas=False)
        results["FID"] = compute_fid(real_feats, gen_feats)
        print(f"  FID = {results['FID']:.3f}")
        step += 1
    else:
        print("\n[SKIPPED] FID  (--no-fid)")
        results["FID"] = None

    if not no_is:
        print(f"\n[{step}/{total_steps}] Computing Inception Score...")
        results["IS_mean"], results["IS_std"] = compute_is(gen_img_s, inception)
        print(f"  IS  = {results['IS_mean']:.3f} +- {results['IS_std']:.3f}")
        step += 1
    else:
        print("\n[SKIPPED] Inception Score  (--no-is)")
        results["IS_mean"] = results["IS_std"] = None

    if not no_lpips:
        print(f"\n[{step}/{total_steps}] Computing LPIPS...")
        real_sample = sample_paths(real_img_paths, len(gen_img_s))
        results["LPIPS_mean"], results["LPIPS_std"] = compute_lpips(gen_img_s, real_sample)
        print(f"  LPIPS = {results['LPIPS_mean']:.4f} +- {results['LPIPS_std']:.4f}")
        step += 1
    else:
        print("\n[SKIPPED] LPIPS  (--no-lpips)")
        results["LPIPS_mean"] = results["LPIPS_std"] = None

    #  Domain-specific metrics
    if not no_edge_coherence:
        print(f"\n[{step}/{total_steps}] Edge Coherence (mask boundary vs image edges)...")
        ec_mean, ec_std = batch_edge_coherence(paired_imgs, paired_masks)
        results["edge_coherence_mean"] = ec_mean
        results["edge_coherence_std"]  = ec_std
        print(f"  Edge Coherence = {ec_mean:.4f} +- {ec_std:.4f}")
        step += 1
    else:
        print("\n[SKIPPED] Edge Coherence  (--no-edge-coherence)")
        results["edge_coherence_mean"] = results["edge_coherence_std"] = None

    if not no_mask_regularity:
        print(f"\n[{step}/{total_steps}] Mask Regularity (contour smoothness and closure)...")
        mr_mean, mr_std = batch_mask_regularity(paired_masks)
        results["mask_regularity_mean"] = mr_mean
        results["mask_regularity_std"]  = mr_std
        print(f"  Mask Regularity = {mr_mean:.4f} +- {mr_std:.4f}")
        step += 1
    else:
        print("\n[SKIPPED] Mask Regularity  (--no-mask-regularity)")
        results["mask_regularity_mean"] = results["mask_regularity_std"] = None

    if not no_frame_symmetry:
        print(f"\n[{step}/{total_steps}] Frame Symmetry Score...")
        fs_mean, fs_std = batch_frame_symmetry(paired_imgs, paired_masks)
        results["frame_symmetry_mean"] = fs_mean
        results["frame_symmetry_std"]  = fs_std
        print(f"  Frame Symmetry = {fs_mean:.4f} +- {fs_std:.4f}")
        step += 1
    else:
        print("\n[SKIPPED] Frame Symmetry  (--no-frame-symmetry)")
        results["frame_symmetry_mean"] = results["frame_symmetry_std"] = None

    #  Diversity
    if not no_diversity:
        print(f"\n[{step}/{total_steps}] Diversity metrics...")
        results["color_diversity"]      = color_diversity(gen_img_s)
        results["mask_shape_diversity"] = mask_shape_diversity(list(paired_masks))
        print(f"  Color diversity      = {results['color_diversity']:.4f}")
        print(f"  Mask shape diversity = {results['mask_shape_diversity']:.4f}")
    else:
        print("\n[SKIPPED] Diversity  (--no-diversity)")
        results["color_diversity"] = results["mask_shape_diversity"] = None

    #  Write report

    def rating(val, good, great, higher_is_better=True):
        """Return a short verdict string, or SKIPPED if val is None."""
        if val is None:
            return "SKIPPED"
        if higher_is_better:
            if val >= great: return "EXCELLENT"
            if val >= good:  return "GOOD"
            return "NEEDS IMPROVEMENT"
        else:
            if val <= great: return "EXCELLENT"
            if val <= good:  return "GOOD"
            return "NEEDS IMPROVEMENT"

    def fmt(val, decimals=4):
        """Format a metric value, showing N/A if skipped."""
        return f"{val:.{decimals}f}" if val is not None else "N/A (skipped)"

    fid_r  = rating(results["FID"],  50, 20, higher_is_better=False)
    ec_r   = rating(results["edge_coherence_mean"],  0.35, 0.55)
    mr_r   = rating(results["mask_regularity_mean"], 0.60, 0.80)
    fs_r   = rating(results["frame_symmetry_mean"],  0.75, 0.85)
    cd_r   = rating(results["color_diversity"],      0.05, 0.09)
    msd_r  = rating(results["mask_shape_diversity"], 0.05, 0.08)

    lines = [
        "=" * 70,
        "GAN Eyeglass Dataset Evaluation Results (v2 - 256x512)",
        f"Generated images : {len(gen_img_paths)}",
        f"Evaluated sample : {len(paired_imgs)}",
        "=" * 70,
        "",
        "--- Standard Metrics ---",
        "",
        f"  FID        : {fmt(results['FID'])}  [{fid_r}]",
        "  What it measures : Distance between real and generated image distributions",
        "  in Inception-v3 feature space. Lower is better. FID < 10 is considered",
        "  excellent for a narrow product domain. Real images were center-cropped",
        "  and resized to 256x512 to match generated images before comparison.",
        "",
        f"  IS (mean)  : {fmt(results['IS_mean'])} +- {fmt(results['IS_std'])}",
        "  What it measures : Sharpness and variety of generated images using an",
        "  ImageNet classifier. NOTE: IS is unreliable for eyeglass frames because",
        "  the Inception classifier was not trained on this domain. Low IS values",
        "  are expected and should not be interpreted as poor generation quality.",
        "",
        f"  LPIPS      : {fmt(results['LPIPS_mean'])} +- {fmt(results['LPIPS_std'])}",
        "  What it measures : Perceptual distance between generated images and",
        "  randomly paired real images using deep VGG features. Higher values",
        "  indicate the generated images look perceptually different from the real",
        "  ones - which is desirable as it means the model is not memorising",
        "  training data. Values in the range 0.25-0.45 are typical for generative",
        "  models on product imagery.",
        "",
        "--- Domain-Specific Metrics ---",
        "",
        f"  Edge Coherence   : {fmt(results['edge_coherence_mean'])} +- {fmt(results['edge_coherence_std'])}  [{ec_r}]",
        "  What it measures : How well the generated mask boundary aligns with",
        "  actual frame edges in the RGB image. Computed as F1 between dilated",
        "  mask boundary pixels and Canny edge pixels within a 3-pixel tolerance",
        "  band. A high score means the model consistently places mask edges where",
        "  real colour discontinuities (frame boundaries) appear in the image.",
        "  Score > 0.35 = acceptable; > 0.55 = excellent joint generation quality.",
        "",
        f"  Mask Regularity  : {fmt(results['mask_regularity_mean'])} +- {fmt(results['mask_regularity_std'])}  [{mr_r}]",
        "  What it measures : How clean and well-formed the generated masks are.",
        "  Measures the fraction of mask pixels captured by the single largest",
        "  contour, penalised by the number of disconnected fragments. Score near",
        "  1.0 = one clean closed shape; score near 0 = fragmented or noisy mask.",
        "  For hollow frame outlines the filled-contour overlap method is used",
        "  to keep the metric in [0, 1]. Score > 0.60 = good; > 0.80 = excellent.",
        "",
        f"  Frame Symmetry   : {fmt(results['frame_symmetry_mean'])} +- {fmt(results['frame_symmetry_std'])}  [{fs_r}]",
        "  What it measures : Left-right symmetry of the generated frame within",
        "  the masked region. Eyeglass frames are symmetric by design, so a high",
        "  score indicates the model has learned physically realistic structure.",
        "  Computed as 1 - normalised mean pixel difference between the frame",
        "  region and its horizontal mirror. Score > 0.75 = realistic;",
        "  > 0.85 = highly symmetric (excellent).",
        "",
        "--- Diversity Metrics ---",
        "",
        f"  Color Diversity      : {fmt(results['color_diversity'])}  [{cd_r}]",
        "  What it measures : Mean pairwise colour distance between generated",
        "  images in RGB space. Higher values indicate a wider range of frame",
        "  colours and materials. Low scores suggest mode collapse toward a single",
        "  colour. Score > 0.05 = acceptable variety; > 0.09 = strong diversity.",
        "",
        f"  Mask Shape Diversity : {fmt(results['mask_shape_diversity'])}  [{msd_r}]",
        "  What it measures : Mean pairwise pixel difference between generated",
        "  masks (downsampled to 64x32). Higher values mean the model generates",
        "  a variety of frame geometries (round, rectangular, cat-eye, etc.)",
        "  rather than repeating the same shape. Score > 0.05 = good variety.",
        "",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(output, "w") as f:
        f.write(report)
    print(f"\nResults saved to: {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a generated eyeglass dataset.")
    parser.add_argument("--real_imgs",   required=True,  help="Directory of real training images")
    parser.add_argument("--real_masks",  required=True,  help="Directory of real training masks")
    parser.add_argument("--gen_imgs",    required=True,  help="Directory of generated images")
    parser.add_argument("--gen_masks",   required=True,  help="Directory of generated masks")
    parser.add_argument("--output",      default="evaluation_results.txt")
    parser.add_argument("--n_sample",    type=int, default=2000, help="-1 for all")
    parser.add_argument("--previous_results", default=None, help="Path to a previous results .txt for comparison")
    parser.add_argument("--no_fid",             action="store_true")
    parser.add_argument("--no_is",              action="store_true")
    parser.add_argument("--no_lpips",           action="store_true")
    parser.add_argument("--no_edge_coherence",  action="store_true")
    parser.add_argument("--no_mask_regularity", action="store_true")
    parser.add_argument("--no_frame_symmetry",  action="store_true")
    parser.add_argument("--no_diversity",       action="store_true")
    args = parser.parse_args()

    main(
        real_imgs=args.real_imgs,
        real_masks=args.real_masks,
        gen_imgs=args.gen_imgs,
        gen_masks=args.gen_masks,
        output=args.output,
        n_sample=args.n_sample,
        previous_results=args.previous_results,
        no_fid=args.no_fid,
        no_is=args.no_is,
        no_lpips=args.no_lpips,
        no_edge_coherence=args.no_edge_coherence,
        no_mask_regularity=args.no_mask_regularity,
        no_frame_symmetry=args.no_frame_symmetry,
        no_diversity=args.no_diversity,
    )