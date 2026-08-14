"""
Eyeglass-tailored diversity metrics.

All images are frontal eyeglass frames on a near-uniform background, so the
frame silhouette can be extracted by background-distance thresholding without
any segmentation masks.

Metric groups:

1. Geometric descriptor diversity (eyeglass-specific)
   Per-image descriptors from the extracted silhouette: frame area ratio,
   bounding-box aspect ratio, rim thickness (distance-transform based),
   lens-opening count and area, hollowness. Diversity = per-descriptor
   dispersion, plus Jensen-Shannon distance between generated and real
   descriptor histograms (distribution coverage).

2. Colour diversity (eyeglass-specific)
   Computed over FRAME PIXELS ONLY in CIELAB space (perceptually uniform),
   unlike the thesis metric which used raw RGB over the whole image:
   mean pairwise LAB distance of per-image frame colour, hue circular
   entropy, and 2D a*b* chromaticity histogram coverage.

3. Feature-space diversity (domain-agnostic references)
   Inception-v3 pool features: Vendi Score (effective number of distinct
   samples), Precision/Recall (Kynkaanniemi et al. 2019) and
   Density/Coverage (Naeem et al. 2020) against the real set, and
   nearest-real-neighbour distance distribution (memorisation check).

Every public function takes numpy image arrays or feature matrices and is
independent of file layout; see evaluate_diversity.py for the CLI driver.
"""

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import jensenshannon


# 1. Silhouette extraction


def extract_silhouette(img_rgb, bg_tolerance=30, min_area_frac=0.005):
    """Binary frame silhouette from a frame photo on near-uniform background.

    img_rgb: uint8 (H, W, 3).
    Background colour is estimated from the border pixels (median); foreground
    is everything further than bg_tolerance from it (Euclidean RGB). Small
    speckles are removed with a morphological open.

    Returns a bool (H, W) array, or None if extraction fails (no plausible
    foreground found).
    """
    h, w = img_rgb.shape[:2]
    border = np.concatenate([
        img_rgb[0, :].reshape(-1, 3), img_rgb[-1, :].reshape(-1, 3),
        img_rgb[:, 0].reshape(-1, 3), img_rgb[:, -1].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(border, axis=0)

    dist = np.linalg.norm(img_rgb.astype(np.float32) - bg, axis=2)
    fg = (dist > bg_tolerance).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    if fg.sum() < min_area_frac * h * w:
        return None
    return fg.astype(bool)


# 2. Geometric descriptors


DESCRIPTOR_NAMES = [
    'area_ratio',       # silhouette area / image area
    'filled_ratio',     # filled-silhouette area / image area (overall frame size)
    'aspect_ratio',     # bounding box width / height
    'height_frac',      # bounding box height / image height (frame vertical size)
    'rim_thickness',    # 2 * mean distance-transform value (px, thin wire vs acetate)
    'rim_thickness_max',# 2 * max distance-transform value (thickest part)
    'hollowness',       # 1 - silhouette / filled silhouette (lens openness)
    'n_holes',          # number of enclosed openings (lens count proxy)
    'solidity',         # filled area / convex hull area (shape compactness)
]


def geometric_descriptors(silhouette):
    """Compute the eyeglass geometric descriptor vector from a bool silhouette.

    Returns dict name -> float, or None for degenerate silhouettes.
    """
    sil = silhouette.astype(np.uint8)
    h, w = sil.shape
    area = float(sil.sum())
    if area <= 0:
        return None

    ys, xs = np.nonzero(sil)
    bh = float(ys.max() - ys.min() + 1)
    bw = float(xs.max() - xs.min() + 1)

    filled = binary_fill_holes(sil).astype(np.uint8)
    filled_area = float(filled.sum())

    # Lens openings: connected components of (filled - silhouette)
    holes = (filled - sil).astype(np.uint8)
    n_holes, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    # Ignore tiny hole fragments (< 0.05% of image)
    significant = int(((stats[1:, cv2.CC_STAT_AREA]) > 0.0005 * h * w).sum())

    # Rim thickness via distance transform of the silhouette
    dt = cv2.distanceTransform(sil, cv2.DIST_L2, 5)
    dt_vals = dt[sil > 0]
    rim_mean = 2.0 * float(dt_vals.mean())
    rim_max = 2.0 * float(dt_vals.max())

    # Solidity of the filled shape
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = 0.0
    for c in contours:
        hull_area += cv2.contourArea(cv2.convexHull(c))
    solidity = filled_area / hull_area if hull_area > 0 else 0.0

    return {
        'area_ratio': area / (h * w),
        'filled_ratio': filled_area / (h * w),
        'aspect_ratio': bw / bh,
        'height_frac': bh / h,
        'rim_thickness': rim_mean,
        'rim_thickness_max': rim_max,
        'hollowness': 1.0 - area / filled_area if filled_area > 0 else 0.0,
        'n_holes': float(significant),
        'solidity': solidity,
    }


def mirror_axis_symmetry(silhouette, max_shift=8, max_deg=3.0, deg_step=1.0):
    """Boundary-band mirror IoU at the bbox axis and at the best-fit axis.

    The plain mirror IoU (diagnose_artifacts.edge_sym) reflects about the
    bbox vertical centre line, which is the frame's true symmetry axis only
    when the shot is fronto-parallel and centred. It is not: on synthetic
    silhouettes a 2 degree tilt costs 0.51 of IoU on a PERFECTLY symmetric
    frame, while genuine temple asymmetry costs only 0.07 - roughly 7x more
    sensitive to pose than to the defect it is named for. Measured on the
    real set, 40% of catalogue photos need a >=3 px axis shift.

    Searching the axis over horizontal shift and small rotation and keeping
    the best IoU removes pose, leaving shape. The offset itself is a useful
    quantity in its own right: its dispersion across a set measures POSE
    diversity, which mirror coupling in G was found to collapse (real shift
    std 2.97, mirror1 1.25) without any other metric noticing.

    Returns (fixed, aligned, shift_px, rot_deg) or None.
    """
    sil = silhouette.astype(np.uint8)
    ys, xs = np.nonzero(sil)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if x1 - x0 < 8:
        return None

    pad = max_shift + 8
    sub = sil[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
    if sub.size == 0:
        return None
    h, w = sub.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    def shift_h(a, s):
        if s == 0:
            return a
        out = np.zeros_like(a)
        if s > 0:
            out[:, s:] = a[:, :-s]
        else:
            out[:, :s] = a[:, -s:]
        return out

    best = (-1.0, 0, 0.0)
    fixed = None
    degs = [0.0] if max_deg <= 0 else list(
        np.arange(-max_deg, max_deg + 1e-9, deg_step))

    for deg in degs:
        if deg == 0.0:
            rot = sub
        else:
            M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
            rot = cv2.warpAffine(sub, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderValue=0)
        band = cv2.dilate(cv2.morphologyEx(rot, cv2.MORPH_GRADIENT, k3),
                          k5).astype(bool)
        flipped = band[:, ::-1]
        for s in range(-max_shift, max_shift + 1):
            m = shift_h(flipped, s)
            u = np.logical_or(band, m).sum()
            v = float(np.logical_and(band, m).sum() / u) if u else 0.0
            if deg == 0.0 and s == 0:
                fixed = v
            if v > best[0]:
                best = (v, s, float(deg))

    if fixed is None:
        return None
    return fixed, best[0], float(best[1]), best[2]


def pose_dispersion(offsets_gen, offsets_real):
    """Dispersion of the mirror-axis offset: a POSE diversity measure.

    Real catalogue frames sit off-centre by a few px (slight yaw extends one
    temple); a generator that pins every frame's symmetry axis to the image
    centre has lost a real axis of variation that FID, KID, LPIPS and
    ab_coverage are all blind to.

    Returns dict of std/IQR for both sets plus the generated/real std ratio
    (1.0 = matched, << 1.0 = pose collapse).
    """
    g = np.asarray(offsets_gen, dtype=float)
    r = np.asarray(offsets_real, dtype=float)
    g = g[np.isfinite(g)]
    r = r[np.isfinite(r)]
    if g.size == 0 or r.size == 0:
        return None

    def iqr(a):
        return float(np.percentile(a, 75) - np.percentile(a, 25))

    return {
        'offset_std_gen': float(g.std()),
        'offset_std_real': float(r.std()),
        'offset_std_ratio': float(g.std() / r.std()) if r.std() > 0 else float('nan'),
        'offset_iqr_gen': iqr(g),
        'offset_iqr_real': iqr(r),
        'offcentre_frac_gen': float((np.abs(g) >= 3).mean()),
        'offcentre_frac_real': float((np.abs(r) >= 3).mean()),
    }


def descriptor_matrix(images_rgb):
    """Descriptor matrix (N, len(DESCRIPTOR_NAMES)) for a list of RGB images.

    Rows where silhouette extraction failed are dropped.
    Returns (matrix, n_failed).
    """
    rows = []
    failed = 0
    for img in images_rgb:
        sil = extract_silhouette(img)
        d = geometric_descriptors(sil) if sil is not None else None
        if d is None:
            failed += 1
            continue
        rows.append([d[name] for name in DESCRIPTOR_NAMES])
    return np.asarray(rows, dtype=np.float64), failed


def descriptor_dispersion(desc):
    """Per-descriptor dispersion: std and 5-95 percentile range."""
    out = {}
    for i, name in enumerate(DESCRIPTOR_NAMES):
        col = desc[:, i]
        out[name] = {
            'mean': float(col.mean()),
            'std': float(col.std()),
            'p05_p95_range': float(np.percentile(col, 95) - np.percentile(col, 5)),
        }
    return out


def descriptor_jsd(desc_gen, desc_real, n_bins=32):
    """Jensen-Shannon distance per descriptor between generated and real.

    Bins are defined on the pooled range so the comparison is symmetric.
    0 = identical distributions; 1 = disjoint. Low JSD together with equal or
    higher dispersion means the generated set covers the real distribution.
    """
    out = {}
    for i, name in enumerate(DESCRIPTOR_NAMES):
        pooled = np.concatenate([desc_gen[:, i], desc_real[:, i]])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if hi <= lo:
            out[name] = 0.0
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        h_gen, _ = np.histogram(desc_gen[:, i], bins=bins, density=False)
        h_real, _ = np.histogram(desc_real[:, i], bins=bins, density=False)
        p = h_gen / max(h_gen.sum(), 1)
        q = h_real / max(h_real.sum(), 1)
        out[name] = float(jensenshannon(p, q, base=2))
    return out


# 3. Colour diversity (frame pixels only, CIELAB)


def frame_lab_stats(img_rgb, silhouette):
    """Mean LAB colour and chroma histogram data of the frame pixels."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV LAB: L in [0,255] (=L*255/100), a/b offset by 128
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    px = lab[silhouette]
    return px.mean(axis=0), px  # (3,), (n_px, 3)


def color_diversity(images_rgb, silhouettes, max_pairs=20000, seed=0):
    """Colour diversity of frame pixels in CIELAB.

    Returns:
      pairwise_lab: mean pairwise Euclidean distance between per-image mean
        frame colours (higher = more diverse palette)
      hue_entropy: normalised entropy of the hue-angle histogram over chromatic
        frame pixels (0 = single hue family, 1 = uniform hue coverage)
      ab_coverage: fraction of occupied bins in a 2D a*b* histogram of
        per-image mean colours (colour gamut coverage)
    """
    rng = np.random.default_rng(seed)
    means = []
    hue_hist = np.zeros(36)
    for img, sil in zip(images_rgb, silhouettes):
        if sil is None:
            continue
        mean_lab, px = frame_lab_stats(img, sil)
        means.append(mean_lab)
        a, b = px[:, 1], px[:, 2]
        chroma = np.sqrt(a ** 2 + b ** 2)
        chromatic = chroma > 10.0  # ignore near-neutral pixels
        if chromatic.any():
            hue = np.degrees(np.arctan2(b[chromatic], a[chromatic])) % 360
            h, _ = np.histogram(hue, bins=36, range=(0, 360),
                                weights=chroma[chromatic])
            hue_hist += h
    means = np.asarray(means)
    n = len(means)
    if n < 2:
        return {'pairwise_lab': 0.0, 'hue_entropy': 0.0, 'ab_coverage': 0.0}

    # Mean pairwise distance on a random subset of pairs
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    valid = i != j
    d = np.linalg.norm(means[i[valid]] - means[j[valid]], axis=1)
    pairwise_lab = float(d.mean())

    # Hue entropy, normalised by log(n_bins)
    p = hue_hist / hue_hist.sum() if hue_hist.sum() > 0 else np.ones(36) / 36
    p = p[p > 0]
    hue_entropy = float(-(p * np.log(p)).sum() / np.log(36))

    # a*b* occupancy of per-image mean colours ([-40, 40] covers frame colours)
    hist2d, _, _ = np.histogram2d(means[:, 1], means[:, 2], bins=16,
                                  range=[[-40, 40], [-40, 40]])
    ab_coverage = float((hist2d > 0).mean())

    return {'pairwise_lab': pairwise_lab, 'hue_entropy': hue_entropy,
            'ab_coverage': ab_coverage}


# 4. Feature-space diversity


def vendi_score(features, q=1.0):
    """Vendi Score (Friedman & Dieng 2023): effective number of distinct samples.

    features: (N, D) matrix. Rows are L2-normalised; VS = exp(entropy of the
    eigenvalues of the normalised cosine-similarity kernel K/N).
    VS is in [1, N]: 1 = all samples identical, N = all orthogonal.
    Also returned normalised by N for cross-set comparability.
    """
    x = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    n = x.shape[0]
    k = (x @ x.T) / n
    eigvals = np.linalg.eigvalsh(k)
    eigvals = np.clip(eigvals, 0, None)
    eigvals = eigvals / eigvals.sum()
    nz = eigvals[eigvals > 1e-12]
    if q == 1.0:
        entropy = -(nz * np.log(nz)).sum()
        vs = float(np.exp(entropy))
    else:
        vs = float((nz ** q).sum() ** (1.0 / (1.0 - q)))
    return {'vendi': vs, 'vendi_per_sample': vs / n, 'n': n}


def precision_recall_density_coverage(real_feats, gen_feats, k=5, return_mask=False):
    """Improved Precision/Recall (Kynkaanniemi 2019) + Density/Coverage (Naeem 2020).

    k-NN manifold estimation in feature space:
      precision - fraction of generated samples inside the real manifold (fidelity)
      recall    - fraction of real samples inside the generated manifold (DIVERSITY)
      density   - how densely generated samples pack the real manifold
      coverage  - fraction of real k-NN balls containing a generated sample (DIVERSITY)

    return_mask=True additionally returns the per-generated-sample boolean
    inside_real array, usable to reject samples that fall outside the real
    manifold (precision-based rejection sampling) without retraining.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist

    def knn_radii(x, k):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
        d, _ = nn.kneighbors(x)
        return d[:, -1]  # distance to k-th neighbour (excluding self)

    r_real = knn_radii(real_feats, k)
    r_gen = knn_radii(gen_feats, k)
    dmat = cdist(gen_feats, real_feats)  # (G, R)
    inside_real = (dmat <= r_real[None, :]).any(axis=1)
    precision = float(inside_real.mean())

    dmat_t = dmat.T  # (R, G)
    inside_gen = (dmat_t <= r_gen[None, :]).any(axis=1)
    recall = float(inside_gen.mean())

    density = float((dmat <= r_real[None, :]).sum(axis=1).mean() / k)
    coverage = float((dmat_t.min(axis=1) <= r_real).mean())

    result = {'precision': precision, 'recall': recall,
              'density': density, 'coverage': coverage, 'k': k}
    if return_mask:
        result['inside_real_mask'] = inside_real
    return result


def nearest_real_distances(real_feats, gen_feats):
    """Distance from each generated sample to its nearest real sample.

    A memorisation check: a spike of near-zero distances means copied
    training samples; a distribution similar to real-to-real nearest
    neighbour distances is healthy.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(real_feats)
    d_gen, _ = nn.kneighbors(gen_feats, n_neighbors=1)
    d_real, _ = nn.kneighbors(real_feats, n_neighbors=2)  # 2nd = nearest other real
    return {
        'gen_to_real_mean': float(d_gen.mean()),
        'gen_to_real_p05': float(np.percentile(d_gen, 5)),
        'real_to_real_mean': float(d_real[:, 1].mean()),
        'ratio_gen_over_real': float(d_gen.mean() / (d_real[:, 1].mean() + 1e-12)),
    }
