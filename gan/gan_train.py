"""
StyleGAN2 for eyeglass frame generation (256x512), image-only (no mask branch).

Derived from the thesis gan_20260515 snapshot with these changes:
  - Mask decoder, mask losses and discriminator mask channel removed
    (this article studies image diversity only; masks come in a later article)
  - Discriminator re-balanced: lr_d and r1_gamma raised from the over-weakened
    20260515 values (lr_d 5e-5 -> 1e-4, r1_gamma 1.0 -> 2.0) so D provides a
    meaningful adversarial signal (20260515 hinge margins collapsed to ~0.5)
  - Style mixing regularization added (per-stage w, StyleGAN2 default 0.9)
  - All diversity-relevant hyperparameters exposed as CLI arguments so the
    architecture/hyperparameter study can be driven from the command line
  - Every run writes a config snapshot (run_config.json) into its output dir

Kept from the snapshot: 8-layer mapping, ADA, lazy R1 + PPL, feature matching,
VGG perceptual loss, EMA, KID tracking with best-checkpoint saving.
"""

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import models, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# Configuration


class Config:
    """Central hyperparameter store. CLI arguments override these defaults."""

    img_height = 256
    img_width = 512  # 2:1 aspect ratio for eyeglass frames

    latent_dim = 512     # dimensionality of the random noise vector z
    w_dim = 512          # dimensionality of the w-space style code
    mapping_depth = 8    # MLP depth in mapping network

    batch_size = 32
    lr_g = 0.0002    # generator learning rate
    # TTUR: D must learn slower than G. Without the mask supervision that
    # grounded G early in the thesis runs, lr_d=1e-4 let D saturate within
    # 3 epochs (hinge d_loss -> 0, G diverging). 5e-5 was stable for the
    # full 400 epochs of the 20260515 run.
    lr_d = 0.00005
    beta1 = 0.0
    beta2 = 0.99

    # R1 gradient penalty on real images (lazy, every r1_interval steps).
    # 20260515 used gamma=1.0 which, combined with the low lr_d and narrow D,
    # collapsed the hinge margins. 2.0 is a middle ground; sweep 1-10.
    r1_gamma = 2.0
    r1_interval = 16

    # Feature matching: G matches D's intermediate feature statistics on reals.
    fm_weight = 2.0

    # Path length regularization (StyleGAN2), lazy like R1.
    ppl_weight = 2.0
    ppl_interval = 16
    ppl_decay = 0.99
    ppl_batch_size = 16  # reduced batch: create_graph=True is memory hungry

    # VGG perceptual loss between fakes and (randomly paired) reals.
    perceptual_weight = 0.75

    # Style mixing regularization: probability of using two independent w codes
    # for a random split of synthesis stages (StyleGAN2 default 0.9).
    style_mixing_prob = 0.9

    ema_decay = 0.999
    save_every = 10

    kid_start = 100      # start evaluating KID after this epoch
    kid_interval = 15    # evaluate KID every N epochs after kid_start
    kid_n_real = 1000
    kid_n_fake = 1000

    d_width_mult = 1.0   # discriminator channel width multiplier

    ada_target = 0.6     # target fraction of real logits scoring positive
    ada_interval = 4
    ada_step = 0.002     # 20260515 used 0.001; slightly faster adaptation
    ada_max_p = 0.85
    ada_color_max_p = 0.85  # separate ceiling for saturation/hue augmentation

    # Optional path to sample_weights.json (metrics/compute_sample_weights.py):
    # oversamples rare-colour images via WeightedRandomSampler.
    sample_weights = ''

    num_workers = 2
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()


cfg = Config()

# Paths, set from CLI in main()
DATASET_PATH = Path('data/source/images')
OUTPUT_ROOT = Path('results/run')
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
SAMPLES_DIR = OUTPUT_ROOT / 'samples'
METRICS_DIR = OUTPUT_ROOT / 'metrics'
GENERATED_DIR = OUTPUT_ROOT / 'generated'


def setup_directories():
    for d in [CHECKPOINT_DIR, SAMPLES_DIR, METRICS_DIR, GENERATED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_run_config(args):
    """Snapshot the effective configuration of this run for reproducibility.

    Training runs write run_config.json. Generation-only runs write
    generation_config.json instead, so re-generating into a training run's
    output directory can never clobber the original training config
    snapshot (this happened to results/nocolorada: its training config -
    r1_gamma 5, batch 16, ada_color_max_p 0 - was overwritten with CLI
    defaults by a later --generate-only call; the true values survive only
    in the experiment log).
    """
    merged = {**vars(Config), **vars(cfg)}  # class defaults + CLI overrides
    payload = {k: (str(v) if isinstance(v, torch.device) else v)
               for k, v in merged.items()
               if not k.startswith('_')
               and isinstance(v, (int, float, str, bool, torch.device))}
    payload['cli_args'] = vars(args)
    name = 'generation_config.json' if getattr(args, 'generate_only', None) \
        else 'run_config.json'
    with open(OUTPUT_ROOT / name, 'w') as f:
        json.dump(payload, f, indent=2)


# Dataset


class EyeglassesDataset(Dataset):
    """Image-only dataset of frontal eyeglass frame photos.

    Images are center-cropped to 2:1 before resizing to avoid distortion.
    """

    def __init__(self, images_dir, transform):
        self.transform = transform
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(
                glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
        self.image_paths.sort()
        if not self.image_paths:
            raise FileNotFoundError(f'No images found under {images_dir}')
        print(f'Found {len(self.image_paths)} images')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Center-crop to the target 2:1 ratio before resizing
        w, h = image.size
        target_ratio = cfg.img_width / cfg.img_height  # 2.0
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            image = image.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            image = image.crop((0, top, w, top + new_h))

        return self.transform(image)


# Model building blocks


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate (StyleGAN).

    Weights are stored as N(0, 1)/lr_mul and scaled by lr_mul/sqrt(fan_in) at
    runtime. This keeps activation variance constant across depth and gives
    the layer an effective learning rate of lr_mul * optimizer lr. With plain
    nn.Linear default init, an 8-layer LeakyReLU MLP shrinks its output std
    to ~0.02, collapsing w-space so that styles carry almost no information
    (measured on this network; also the likely cause of the 20260515 run's
    low colour diversity). The official mapping network uses lr_mul=0.01.
    """

    def __init__(self, in_dim, out_dim, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class ScaledLeakyReLU(nn.Module):
    """LeakyReLU(0.2) with the sqrt(2) gain that keeps variance constant."""

    def forward(self, x):
        return F.leaky_relu(x, 0.2) * math.sqrt(2.0)


class MappingNetwork(nn.Module):
    """Noise z -> style code w via a pixel-normalised, equalized-lr MLP."""

    def __init__(self, z_dim, w_dim, depth=8):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(depth):
            layers.append(EqualLinear(z_dim, w_dim, lr_mul=0.01))
            layers.append(ScaledLeakyReLU())
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyleMod(nn.Module):
    """Per-channel affine (scale, shift) parameters from a style vector w."""

    def __init__(self, in_channels, w_dim):
        super().__init__()
        self.fc = nn.Linear(w_dim, in_channels * 2)
        self.fc.bias.data[:in_channels] = 1.0
        self.fc.bias.data[in_channels:] = 0.0

    def forward(self, w):
        style = self.fc(w).unsqueeze(2).unsqueeze(3)
        return style.chunk(2, dim=1)


class ModulatedConv2d(nn.Module):
    """Weight-modulated convolution, the core StyleGAN2 operation."""

    def __init__(self, in_channels, out_channels, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.demodulate = demodulate
        self.eps = 1e-8
        self.pad = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in',
                                nonlinearity='leaky_relu')
        self.style = StyleMod(in_channels, w_dim)

    def forward(self, x, w):
        B = x.size(0)
        scale, _ = self.style(w)
        weight = self.weight * scale.unsqueeze(1)
        if self.demodulate:
            sigma = torch.sqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + self.eps)
            weight = weight / sigma
        weight = weight.reshape(B * weight.size(1), weight.size(2),
                                weight.size(3), weight.size(4))
        x = x.reshape(1, B * x.size(1), x.size(2), x.size(3))
        out = F.conv2d(x, weight, padding=self.pad, groups=B)
        return out.reshape(B, -1, out.size(2), out.size(3))


class StyledConvBlock(nn.Module):
    """Modulated conv -> noise injection -> bias + activation."""

    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x, w)
        x = self.noise(x)
        return self.act(x + self.bias)


class SelfAttention(nn.Module):
    """Non-local self-attention; applied at low resolution only (O(H^2 W^2))."""

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 1)
        self.query = spectral_norm(nn.Conv2d(channels, mid, 1))
        self.key = spectral_norm(nn.Conv2d(channels, mid, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = torch.bmm(q, k) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


# Generator


class Generator(nn.Module):
    """StyleGAN2 synthesis network for 256x512 eyeglass frames (image only).

    - Learned 4x8 constant seed tensor
    - 6 upsampling stages of StyledConvBlock pairs with skip-RGB accumulation
    - Self-attention at the 32x64 stage
    - Per-stage style input to support style mixing regularization:
      synthesis() accepts w of shape (B, w_dim) or (B, n_stages, w_dim)
    """

    CHANNELS = [512, 256, 256, 128, 64, 32, 16]

    def __init__(self, z_dim=512, w_dim=512, mapping_depth=8):
        super().__init__()
        self.w_dim = w_dim
        self.num_stages = len(self.CHANNELS) - 1
        self.mapping = MappingNetwork(z_dim, w_dim, depth=mapping_depth)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 8))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)

        ch = self.CHANNELS
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.attn_idx = 2
        self.attention = SelfAttention(ch[self.attn_idx + 1])

        for i in range(self.num_stages):
            self.blocks.append(nn.ModuleList([
                StyledConvBlock(ch[i], ch[i + 1], w_dim),
                StyledConvBlock(ch[i + 1], ch[i + 1], w_dim),
            ]))
            self.to_rgb.append(ModulatedConv2d(ch[i + 1], 3, 1, w_dim,
                                               demodulate=False))

    def get_w(self, z):
        return self.mapping(z)

    def synthesis(self, w):
        """w: (B, w_dim) shared across stages, or (B, num_stages, w_dim)."""
        if w.dim() == 2:
            w = w.unsqueeze(1).expand(-1, self.num_stages, -1)

        batch_size = w.size(0)
        x = self.const.repeat(batch_size, 1, 1, 1)
        rgb = None

        for i, (block, rgb_layer) in enumerate(zip(self.blocks, self.to_rgb)):
            w_i = w[:, i]
            x = self.upsample(x)
            x = block[0](x, w_i)
            x = block[1](x, w_i)
            if i == self.attn_idx:
                x = self.attention(x)
            rgb_out = rgb_layer(x, w_i)
            rgb = rgb_out if rgb is None else self.upsample(rgb) + rgb_out

        # Linear RGB output (official StyleGAN2). A tanh here saturates within
        # a few optimizer steps on this data: the near-white catalogue
        # backgrounds push most pixels to +1, where tanh has zero gradient,
        # freezing the generator (measured: 99.5% of pixels saturated and
        # G grad norm ~0 after 10 steps of regression toward a white image).
        return rgb

    def forward(self, z, style_mixing_prob=0.0):
        w = self.get_w(z)  # (B, w_dim)
        w = w.unsqueeze(1).expand(-1, self.num_stages, -1).contiguous()

        if style_mixing_prob > 0 and random.random() < style_mixing_prob:
            # Second w for stages >= random crossover point (per-batch cut).
            z2 = torch.randn_like(z)
            w2 = self.get_w(z2)
            cut = random.randint(1, self.num_stages - 1)
            w[:, cut:] = w2.unsqueeze(1).expand(-1, self.num_stages - cut, -1)

        return self.synthesis(w)


# Discriminator


class MinibatchSTD(nn.Module):
    """Appends a channel holding the mean std across the batch (anti mode collapse)."""

    def forward(self, x):
        std = x.std(dim=0).mean()
        std_map = std.view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_map], dim=1)


class DiscBlock(nn.Module):
    """Residual downsampling block: two convs + 1x1 skip, then avg-pool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.down = nn.AvgPool2d(2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        return self.down(y) + self.down(self.skip(x))


class Discriminator(nn.Module):
    """Discriminator on 3-channel RGB input (mask channel removed).

    Channel widths scale with d_width_mult so discriminator capacity can be
    part of the hyperparameter study (20260515 narrowed D so far it never
    provided a strong adversarial signal).
    """

    def __init__(self, d_width_mult=1.0):
        super().__init__()
        base = [16, 32, 64, 128, 256, 256]
        ch = [max(8, int(round(c * d_width_mult))) for c in base]

        self.from_rgb = nn.Conv2d(3, ch[0], 1)
        self.act = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(*[
            DiscBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1)
        ])
        self.attn = SelfAttention(ch[3])
        self.mbstd = MinibatchSTD()
        self.final = nn.Conv2d(ch[-1] + 1, ch[-1], 3, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(ch[-1] * 16, 1),
        )

    def forward(self, img, return_features=False):
        x = self.act(self.from_rgb(img))
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:
                x = self.attn(x)
            if return_features:
                features.append(x)
        x = self.mbstd(x)
        x = self.act(self.final(x))
        logit = self.head(x).squeeze(1)
        if return_features:
            return logit, features
        return logit


# Losses


# Non-saturating logistic loss (StyleGAN2 / StyleGAN2-ADA). The hinge loss
# used in the mask-supervised runs has zero gradient outside its margins:
# once D separates real/fake by more than 1 (normal in the first epochs of
# unconditional training, where G has no mask supervision to ground it), D
# receives no corrective data gradient at all and d_loss sits at exactly 0.
# Softplus is smooth everywhere: D always gets a small restoring gradient,
# and G's gradient is strongest precisely when it is losing.
def d_logistic(real_pred, fake_pred):
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def g_nonsat(fake_pred):
    return F.softplus(-fake_pred).mean()


def r1_penalty(real_pred, real_img):
    grad = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img,
                               create_graph=True)[0]
    return (grad.view(grad.size(0), -1).norm(2, dim=1) ** 2).mean()


class VGGPerceptualLoss(nn.Module):
    """Frozen VGG16 features at relu1_2, relu2_2, relu3_3.

    Each fake is compared against its NEAREST real in the batch (nearest in
    deep VGG feature space, selection under no_grad) instead of a randomly
    paired one. Minimizing distance to a random real pulls every fake toward
    the dataset mean in expectation, a mode-averaging force that suppresses
    diversity; pulling toward the nearest real keeps the "look like a
    catalogue photo" gradient while letting each fake stay in its own mode.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])
        self.slice2 = nn.Sequential(*list(vgg)[4:9])
        self.slice3 = nn.Sequential(*list(vgg)[9:18])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return ((x + 1) / 2.0 - self.mean) / self.std

    def forward(self, pred, target):
        p = self._normalize(pred)
        t = self._normalize(target)
        p_feats, t_feats = [], []
        for layer in [self.slice1, self.slice2, self.slice3]:
            p = layer(p)
            with torch.no_grad():
                t = layer(t)
            p_feats.append(p)
            t_feats.append(t)

        # Match each fake to the closest real in the batch using pooled
        # deepest-slice features (cheap: B x B distances on B x C vectors).
        with torch.no_grad():
            pv = p_feats[-1].mean([2, 3]).float()
            tv = t_feats[-1].mean([2, 3]).float()
            match = torch.cdist(pv, tv).argmin(dim=1)

        loss = 0.0
        for pf, tf, w in zip(p_feats, t_feats, [1.0, 1.0, 0.5]):
            loss += w * F.l1_loss(pf, tf[match])
        return loss


# Adaptive Discriminator Augmentation


class ADAugment:
    """Adaptive Discriminator Augmentation (Karras et al. 2020), image-only.

    Tracks rt = fraction of real logits > 0 and adjusts augmentation
    probability p toward ada_target. Augmentations are applied to both real
    and fake images before D sees them.
    """

    def __init__(self, target=0.6, step=0.002, max_p=0.85, interval=4,
                 color_max_p=None):
        self.target = target
        self.step = step
        self.max_p = max_p
        self.interval = interval
        # Colour augmentations (saturation/hue) can leak the frame-colour
        # distribution into D less than pixel/geometric ones do, and can
        # also suppress G's own colour diversity if driven too hard; keep
        # a separate, independently tunable ceiling for them.
        self.color_max_p = max_p if color_max_p is None else color_max_p
        self.p = 0.0
        self._signs = []

    def update(self, real_pred):
        self._signs.append((real_pred > 0).float().mean().item())
        if len(self._signs) >= self.interval:
            rt = sum(self._signs) / len(self._signs)
            self._signs.clear()
            if rt > self.target:
                self.p = min(self.p + self.step, self.max_p)
            else:
                self.p = max(self.p - self.step, 0.0)

    def __call__(self, imgs):
        if self.p <= 0.0:
            return imgs
        B = imgs.size(0)
        device = imgs.device
        color_p = min(self.p, self.color_max_p)

        # Horizontal flip
        flip = torch.rand(B, device=device) < self.p
        if flip.any():
            imgs = torch.where(flip.view(B, 1, 1, 1), imgs.flip(-1), imgs)

        # Integer-pixel translation, per-sample (geometric augmentations are
        # the most effective category in the ADA paper; without them ADA
        # cannot restrain D once rt exceeds the target)
        H, W = imgs.shape[2], imgs.shape[3]
        max_ty, max_tx = int(H * 0.125), int(W * 0.125)
        for b in range(B):
            if random.random() < self.p:
                ty = random.randint(-max_ty, max_ty)
                tx = random.randint(-max_tx, max_tx)
                if ty != 0 or tx != 0:
                    imgs[b] = torch.roll(imgs[b], shifts=(ty, tx), dims=(1, 2))

        # Brightness jitter, per-sample
        bright = (torch.rand(B, device=device) < self.p).view(B, 1, 1, 1)
        scale = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
        # No clamping (official ADA): clamps zero the gradient for
        # out-of-range pixels and assume bounded G output.
        imgs = torch.where(bright, imgs * scale, imgs)

        # Saturation jitter, per-sample: scale distance from per-pixel luma.
        # Colour augmentations stop D from keying on the exact frame colours
        # of the small training set, which otherwise pushes G toward the
        # dominant colour modes (low a*b* coverage in the diversity reports).
        luma_w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
        sat = (torch.rand(B, device=device) < color_p).view(B, 1, 1, 1)
        sat_scale = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 1.0
        luma = (imgs * luma_w).sum(1, keepdim=True)
        imgs = torch.where(sat, luma + sat_scale * (imgs - luma), imgs)

        # Hue rotation, per-sample: rotate RGB about the (1,1,1) luma axis
        # (Rodrigues formula), the official ADA colour transform. Applied
        # with probability < 1, so the true colour distribution stays
        # identifiable (ADA non-leakage argument).
        hue = torch.rand(B, device=device) < color_p
        if hue.any():
            theta = (torch.rand(B, device=device) * 2.0 - 1.0) * math.pi
            theta = torch.where(hue, theta, torch.zeros_like(theta))
            c = theta.cos().view(B, 1, 1)
            s = theta.sin().view(B, 1, 1)
            v = 1.0 / math.sqrt(3.0)
            eye = torch.eye(3, device=device).unsqueeze(0)
            vvt = torch.full((1, 3, 3), v * v, device=device)
            K = torch.tensor([[0.0, -v, v], [v, 0.0, -v], [-v, v, 0.0]],
                             device=device).unsqueeze(0)
            R = c * eye + s * K + (1.0 - c) * vvt  # (B, 3, 3)
            imgs = torch.einsum('bij,bjhw->bihw', R, imgs)

        # Gaussian noise, per-sample
        noisy = (torch.rand(B, device=device) < self.p * 0.5).view(B, 1, 1, 1)
        imgs = torch.where(noisy, imgs + 0.05 * torch.randn_like(imgs), imgs)

        # Cutout, per-sample
        H, W = imgs.shape[2], imgs.shape[3]
        ph, pw = H // 4, W // 4
        cut = torch.ones_like(imgs)
        for b in range(B):
            if random.random() < self.p * 0.5:
                t = random.randint(0, H - ph)
                l = random.randint(0, W - pw)
                cut[b, :, t:t + ph, l:l + pw] = 0.0
        return imgs * cut


# EMA


class EMA:
    """Exponential moving average of G's parameters, used for all inference."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone()
                       for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# Utilities


def save_metrics(history):
    path = METRICS_DIR / 'training_metrics.json'
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def find_latest_checkpoint(checkpoint_dir):
    p = Path(checkpoint_dir) / 'checkpoint_latest.pth'
    return p if p.exists() else None


def autocast():
    return torch.amp.autocast('cuda', enabled=cfg.use_cuda)


# Training loop


def train(end_epoch, resume=False, checkpoint_path=None):
    img_transform = transforms.Compose([
        transforms.Resize((cfg.img_height, cfg.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [0,1] -> [-1,1]
    ])

    dataset = EyeglassesDataset(str(DATASET_PATH), img_transform)
    sampler = None
    if cfg.sample_weights:
        with open(cfg.sample_weights) as f:
            wmap = json.load(f)['weights']
        weights = [wmap.get(os.path.basename(p), 1.0) for p in dataset.image_paths]
        n_boosted = sum(1 for w in weights if w > 1.0)
        total = sum(weights)
        print(f'Weighted sampling: {n_boosted}/{len(weights)} boosted images, '
              f'{100 * sum(w for w in weights if w > 1.0) / total:.1f}% of drawn samples '
              f'({cfg.sample_weights})')
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True,
                        pin_memory=cfg.use_cuda)

    G = Generator(cfg.latent_dim, cfg.w_dim, cfg.mapping_depth).to(cfg.device)
    D = Discriminator(cfg.d_width_mult).to(cfg.device)

    vgg = VGGPerceptualLoss().to(cfg.device) if cfg.perceptual_weight > 0 else None
    ada = ADAugment(cfg.ada_target, cfg.ada_step, cfg.ada_max_p, cfg.ada_interval,
                    color_max_p=cfg.ada_color_max_p)

    g_optim = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    d_optim = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    scaler_g = torch.amp.GradScaler('cuda', enabled=cfg.use_cuda)
    scaler_d = torch.amp.GradScaler('cuda', enabled=cfg.use_cuda)

    ema = EMA(G, cfg.ema_decay)
    fixed_z = torch.randn(16, cfg.latent_dim, device=cfg.device)

    history = {
        'epoch': [], 'g_loss': [], 'd_loss': [], 'perc_loss': [], 'fm_loss': [],
        'g_total': [], 'd_real': [], 'd_fake': [], 'ada_p': [],
        'kid': [], 'epoch_sec': [], 'total_hrs': [],
    }

    best_kid = float('inf')
    ppl_mean_ema = 0.0
    start_epoch = 0

    if resume and checkpoint_path and os.path.exists(str(checkpoint_path)):
        print(f'Loading checkpoint: {checkpoint_path}')
        ckpt = torch.load(str(checkpoint_path), map_location=cfg.device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        ema.shadow = ckpt.get('ema', ema.shadow)
        ada.p = ckpt.get('ada_p', 0.0)
        history = ckpt.get('history', history)
        best_kid = ckpt.get('best_kid', best_kid)
        ppl_mean_ema = ckpt.get('ppl_mean_ema', 0.0)
        start_epoch = ckpt['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f'Generator params:     {g_params:,}')
    print(f'Discriminator params: {d_params:,}')
    print(f'Training epochs {start_epoch} -> {end_epoch}\n')

    kid_real_pool = None
    kid_metric = None

    t0 = time.time()
    global_step = 0
    d_collapse_counter = 0

    for epoch in range(start_epoch, end_epoch):
        t_epoch = time.time()
        G.train()
        D.train()
        sums = {'g': 0.0, 'd': 0.0, 'perc': 0.0, 'fm': 0.0,
                'g_total': 0.0, 'dr': 0.0, 'df': 0.0}

        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{end_epoch}', leave=False)
        for real_imgs in pbar:
            real_imgs = real_imgs.to(cfg.device)
            batch_size = real_imgs.size(0)

            # --- Discriminator step ---
            d_optim.zero_grad()

            aug_imgs = ada(real_imgs.detach().clone())
            with autocast():
                real_pred, real_feats = D(aug_imgs, return_features=True)
            ada.update(real_pred.detach().float())

            with torch.no_grad():
                z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
                with autocast():
                    fake_imgs = G(z, cfg.style_mixing_prob)
                fake_aug = ada(fake_imgs.float().clone())

            with autocast():
                fake_pred = D(fake_aug)
                d_loss = d_logistic(real_pred.float(), fake_pred.float())

            # R1 on non-augmented reals, float32, lazy with interval scaling
            if global_step % cfg.r1_interval == 0:
                real_imgs_r1 = real_imgs.detach().requires_grad_(True)
                real_pred_r1 = D(real_imgs_r1)
                r1 = r1_penalty(real_pred_r1, real_imgs_r1)
                d_loss = d_loss + cfg.r1_gamma * 0.5 * cfg.r1_interval * r1

            # D-collapse trip-wire (failsafe only). With the logistic loss,
            # d_loss < 1e-4 requires |logits| > ~9 on every sample for 100
            # consecutive steps, i.e. genuine divergence, not the normal
            # early-training phase where D leads.
            if d_loss.item() < 1e-4:
                d_collapse_counter += 1
                if d_collapse_counter > 100:
                    print(f'\nD saturated (d_loss={d_loss.item():.2e} for 100 steps). Stopping.')
                    return
            else:
                d_collapse_counter = 0

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(d_optim)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler_d.step(d_optim)
            scaler_d.update()

            # --- Generator step ---
            g_optim.zero_grad()

            z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            with autocast():
                fake_imgs = G(z, cfg.style_mixing_prob)
            fake_imgs = fake_imgs.float()
            fake_aug = ada(fake_imgs.clone())

            with autocast():
                fake_pred, fake_feats = D(fake_aug, return_features=True)

            g_adv = g_nonsat(fake_pred.float())

            perc_loss = torch.tensor(0.0, device=cfg.device)
            if vgg is not None:
                perc_loss = vgg(fake_imgs, real_imgs)

            fm_loss = torch.stack([
                F.l1_loss(f.float().mean([2, 3]), r.detach().float().mean([2, 3]))
                for f, r in zip(fake_feats, real_feats)
            ]).mean()

            g_total = (g_adv
                       + cfg.perceptual_weight * perc_loss
                       + cfg.fm_weight * fm_loss)

            # Path length regularization: lazy, float32, gradients flow through mapping
            if cfg.ppl_weight > 0 and global_step % cfg.ppl_interval == 0:
                z_ppl = torch.randn(cfg.ppl_batch_size, cfg.latent_dim, device=cfg.device)
                w_ppl = G.get_w(z_ppl)
                ppl_imgs = G.synthesis(w_ppl)
                noise = torch.randn_like(ppl_imgs) / math.sqrt(
                    ppl_imgs.shape[2] * ppl_imgs.shape[3])
                grad = torch.autograd.grad(outputs=(ppl_imgs * noise).sum(),
                                           inputs=w_ppl, create_graph=True)[0]
                path_lengths = grad.pow(2).sum(1).sqrt()
                path_mean = path_lengths.mean()
                if ppl_mean_ema == 0.0:
                    ppl_mean_ema = path_mean.item()
                else:
                    ppl_penalty = (path_lengths - ppl_mean_ema).pow(2).mean()
                    with torch.no_grad():
                        ppl_mean_ema = (ppl_mean_ema * cfg.ppl_decay
                                        + path_mean.item() * (1 - cfg.ppl_decay))
                    g_total = g_total + cfg.ppl_weight * ppl_penalty

            scaler_g.scale(g_total).backward()
            scaler_g.unscale_(g_optim)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_g.step(g_optim)
            scaler_g.update()
            ema.update(G)

            sums['g'] += g_adv.item()
            sums['d'] += d_loss.item()
            sums['perc'] += perc_loss.item()
            sums['fm'] += fm_loss.item()
            sums['g_total'] += g_total.item()
            sums['dr'] += real_pred.mean().item()
            sums['df'] += fake_pred.mean().item()
            global_step += 1

            pbar.set_postfix(G=f'{g_adv.item():.3f}', D=f'{d_loss.item():.3f}',
                             ADA_p=f'{ada.p:.3f}')

        # --- End of epoch ---
        n_batches = len(loader)
        epoch_time = time.time() - t_epoch
        total_hours = (time.time() - t0) / 3600

        kid_mean = None
        with torch.no_grad():
            ema.apply_shadow(G)
            G.eval()

            eval_imgs = G(fixed_z)
            save_image(eval_imgs.clamp(-1, 1),
                       str(SAMPLES_DIR / f'epoch_{epoch + 1:04d}_imgs.png'),
                       nrow=4, normalize=True, value_range=(-1, 1))

            if (epoch + 1) >= cfg.kid_start and \
                    (epoch + 1 - cfg.kid_start) % cfg.kid_interval == 0:
                if kid_real_pool is None:
                    print(f'Building real image pool for KID ({cfg.kid_n_real} images) ...')
                    _pool = []
                    for real_batch in loader:
                        imgs_uint8 = ((real_batch.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                        _pool.append(imgs_uint8)
                        if sum(x.size(0) for x in _pool) >= cfg.kid_n_real:
                            break
                    kid_real_pool = torch.cat(_pool)[:cfg.kid_n_real]
                    kid_metric = KernelInceptionDistance(
                        subset_size=min(100, cfg.kid_n_real // 2)).to(cfg.device)
                kid_metric.reset()
                for i in range(0, len(kid_real_pool), 64):
                    kid_metric.update(kid_real_pool[i:i + 64].to(cfg.device), real=True)
                n_fake_done = 0
                while n_fake_done < cfg.kid_n_fake:
                    bs_kid = min(32, cfg.kid_n_fake - n_fake_done)
                    z_kid = torch.randn(bs_kid, cfg.latent_dim, device=cfg.device)
                    with autocast():
                        fake_kid = G(z_kid)
                    fake_uint8 = ((fake_kid.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                    kid_metric.update(fake_uint8, real=False)
                    n_fake_done += bs_kid
                kid_mean, kid_std = kid_metric.compute()
                kid_mean = kid_mean.item()
                print(f'  KID: {kid_mean:.5f} +- {kid_std.item():.5f}')

            G.train()
            ema.restore(G)

        history['epoch'].append(epoch + 1)
        history['g_loss'].append(sums['g'] / n_batches)
        history['d_loss'].append(sums['d'] / n_batches)
        history['perc_loss'].append(sums['perc'] / n_batches)
        history['fm_loss'].append(sums['fm'] / n_batches)
        history['g_total'].append(sums['g_total'] / n_batches)
        history['d_real'].append(sums['dr'] / n_batches)
        history['d_fake'].append(sums['df'] / n_batches)
        history['ada_p'].append(ada.p)
        history['kid'].append(kid_mean)
        history['epoch_sec'].append(epoch_time)
        history['total_hrs'].append(total_hours)

        kid_str = f'  KID={kid_mean:.5f}' if kid_mean is not None else ''
        print(f'Ep {epoch + 1:4d}/{end_epoch}  '
              f'G={sums["g"] / n_batches:.4f}  D={sums["d"] / n_batches:.4f}  '
              f'FM={sums["fm"] / n_batches:.4f}  '
              f'Dreal={sums["dr"] / n_batches:+.3f}  Dfake={sums["df"] / n_batches:+.3f}  '
              f'ADA_p={ada.p:.3f}{kid_str}  '
              f'{epoch_time:.0f}s  ({total_hours:.1f}h total)')

        save_metrics(history)

        ema.apply_shadow(G)
        ckpt_data = {
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'g_optim': g_optim.state_dict(),
            'd_optim': d_optim.state_dict(),
            'ema': ema.shadow,
            'ada_p': ada.p,
            'best_kid': best_kid,
            'ppl_mean_ema': ppl_mean_ema,
            'history': history,
        }
        torch.save(ckpt_data, str(CHECKPOINT_DIR / 'checkpoint_latest.pth'))
        if kid_mean is not None and kid_mean < best_kid:
            best_kid = kid_mean
            ckpt_data['best_kid'] = best_kid
            torch.save(ckpt_data, str(CHECKPOINT_DIR / 'checkpoint_best.pth'))
            print(f'  Best checkpoint saved (KID {kid_mean:.5f})')
        ema.restore(G)

    print(f'\nfinished, {(time.time() - t0) / 3600:.2f}h total')
    save_metrics(history)


# Generation


def generate(checkpoint_path, num_images=10000, truncation_psi=0.7, batch_size=32,
             pure_frac=0.30, truncation_psi_fine=None, truncation_cutoff=3,
             truncation_centers=1):
    """Generate a dataset from a trained checkpoint (images only).

    Generation modes (metadata records the mode of each sample), split as
    pure_frac / interp2 / interp3 / interp4 in a fixed 4:2:1 remaining
    ratio (default pure_frac=0.30 reproduces the original 30/40/20/10
    split; pure_frac=1.0 gives 100% single-z samples, useful to isolate
    whether w-blending itself is suppressing diversity vs. the model):
      pure_frac  pure     - single z sample
      remainder  interp2  - blend two w vectors (bimodal Beta weights), 4/7 share
      remainder  interp3  - blend three w vectors (Dirichlet weights), 2/7 share
      remainder  interp4  - blend four w vectors (Dirichlet weights), 1/7 share

    truncation_psi pulls w toward its truncation center: lower = quality,
    higher = diversity.

    Per-layer truncation: stages [0, truncation_cutoff) (coarse, silhouette
    and proportions - the region most prone to going off-manifold at high
    psi) use truncation_psi; stages [truncation_cutoff, num_stages) (fine,
    colour/texture/rim detail - where diversity is wanted) use
    truncation_psi_fine. If truncation_psi_fine is None, it defaults to
    truncation_psi (original single-psi behaviour, bit-exact). This is the
    standard StyleGAN "truncation on a layer subset" trick: it lets quality
    be pulled in on the coarse stages without sacrificing fine-detail
    diversity. The 6 stages here span 8x16 up to 256x512.

    Multi-modal truncation centers: truncation_centers=1 (default) pulls
    every sample toward the single global mean w (original behaviour,
    bit-exact). truncation_centers=k>1 instead k-means-clusters a large
    w-space sample into k centroids and pulls each generated sample toward
    its OWN nearest centroid rather than one global "average frame". This
    should curb precision loss from truncation without collapsing distinct
    frame styles toward a single mode, since e.g. a wide/angular frame and
    a small/round frame get pulled toward their own respective center
    instead of both being dragged toward the dataset-wide average.
    """
    if truncation_psi_fine is None:
        truncation_psi_fine = truncation_psi
    img_dir = GENERATED_DIR / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(checkpoint_path), map_location=cfg.device)
    G = Generator(cfg.latent_dim, cfg.w_dim, cfg.mapping_depth).to(cfg.device)
    G.load_state_dict(ckpt['G'])
    if 'ema' in ckpt:
        for name, param in G.named_parameters():
            if name in ckpt['ema']:
                param.data.copy_(ckpt['ema'][name])
    G.eval()

    print('Computing w_mean ...')
    w_samples = []
    with torch.no_grad():
        for _ in range(200):
            z = torch.randn(50, cfg.latent_dim, device=cfg.device)
            w_samples.append(G.get_w(z))
    w_all = torch.cat(w_samples)
    w_mean = w_all.mean(0, keepdim=True)

    if truncation_centers > 1:
        from sklearn.cluster import KMeans
        print(f'Clustering w-space into {truncation_centers} truncation centers ...')
        km = KMeans(n_clusters=truncation_centers, random_state=cfg.seed, n_init=10)
        km.fit(w_all.cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=w_mean.dtype,
                               device=cfg.device)  # (truncation_centers, w_dim)
    else:
        centers = w_mean  # (1, w_dim); every sample uses this one center

    num_stages = G.num_stages
    psi_per_stage = torch.tensor(
        [truncation_psi] * truncation_cutoff +
        [truncation_psi_fine] * (num_stages - truncation_cutoff),
        device=cfg.device).view(1, num_stages, 1)

    n_pure = int(num_images * pure_frac)
    n_remaining = num_images - n_pure
    n_interp2 = int(n_remaining * 4 / 7)
    n_interp3 = int(n_remaining * 2 / 7)
    n_interp4 = num_images - n_pure - n_interp2 - n_interp3
    plan = ([('pure', 1)] * n_pure + [('interp2', 2)] * n_interp2 +
            [('interp3', 3)] * n_interp3 + [('interp4', 4)] * n_interp4)
    random.shuffle(plan)

    generated = 0
    with open(GENERATED_DIR / 'metadata.csv', 'w') as meta_file:
        meta_file.write('id,type,n_z,psi,psi_fine,cutoff,centers,center_id\n')
        pbar = tqdm(total=num_images, desc='Generating')
        with torch.no_grad():
            while generated < num_images:
                bs = min(batch_size, num_images - generated)
                batch_plan = plan[generated:generated + bs]
                w_batch = torch.zeros(bs, G.w_dim, device=cfg.device)
                rows = []

                for j, (gen_type, n_z) in enumerate(batch_plan):
                    idx = generated + j
                    z_j = torch.randn(n_z, cfg.latent_dim, device=cfg.device)
                    w_j = G.get_w(z_j)
                    if n_z == 1:
                        w_blend = w_j[0]
                    elif n_z == 2:
                        alpha = float(np.random.beta(0.4, 0.4))
                        w_blend = alpha * w_j[0] + (1 - alpha) * w_j[1]
                    else:
                        conc = 0.7 if n_z == 3 else 0.5
                        weights = np.random.dirichlet([conc] * n_z)
                        w_blend = sum(float(weights[k]) * w_j[k] for k in range(n_z))
                    w_batch[j] = w_blend

                if truncation_centers > 1:
                    dists = torch.cdist(w_batch, centers)  # (bs, truncation_centers)
                    assign = dists.argmin(dim=1)            # (bs,)
                    center_batch = centers[assign]           # (bs, w_dim)
                else:
                    assign = torch.zeros(bs, dtype=torch.long, device=cfg.device)
                    center_batch = centers.expand(bs, -1)

                for j, (gen_type, n_z) in enumerate(batch_plan):
                    idx = generated + j
                    rows.append(f'{idx:05d},{gen_type},{n_z},{truncation_psi:.2f},'
                                f'{truncation_psi_fine:.2f},{truncation_cutoff},'
                                f'{truncation_centers},{int(assign[j])}\n')

                w_expanded = w_batch.unsqueeze(1).expand(-1, num_stages, -1)
                center_expanded = center_batch.unsqueeze(1).expand(-1, num_stages, -1)
                w_expanded = center_expanded + psi_per_stage * (w_expanded - center_expanded)
                imgs = G.synthesis(w_expanded)

                for j in range(bs):
                    save_image(imgs[j].clamp(-1, 1),
                               str(img_dir / f'glass_{generated + j:05d}.png'),
                               normalize=True, value_range=(-1, 1))
                    meta_file.write(rows[j])

                generated += bs
                pbar.update(bs)
        pbar.close()

    print(f'Generated {generated} images -> {GENERATED_DIR}')


# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='StyleGAN2 for eyeglass frame generation (image-only).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', default='data/source/images',
                        help='Directory of training frame images')
    parser.add_argument('--output', default='results/run',
                        help='Root output directory (one folder per experiment run)')
    parser.add_argument('--end-epoch', type=int, default=400)
    parser.add_argument('--no-resume', action='store_true',
                        help='Start from scratch, ignoring existing checkpoints')
    parser.add_argument('--checkpoint', default=None,
                        help='Explicit checkpoint path to resume from')
    parser.add_argument('--generate', action='store_true',
                        help='Generate images after training completes')
    parser.add_argument('--generate-only', default=None, metavar='CHECKPOINT',
                        help='Skip training; generate from the given checkpoint')
    parser.add_argument('--num-images', type=int, default=10000,
                        help='Number of images to generate')
    parser.add_argument('--pure-frac', type=float, default=0.30,
                        help='Fraction of generated images that are single-z '
                             '(no w-blending); remainder splits 4:2:1 across '
                             'interp2/3/4 as before. Set 1.0 for a 100%% '
                             'pure-z variant to isolate blending effects')
    parser.add_argument('--truncation-psi', type=float, default=0.7,
                        help='Truncation psi (lower=quality, higher=diversity). '
                             'Applied to the coarse stages (see --truncation-cutoff); '
                             'applied to ALL stages if --truncation-psi-fine is unset')
    parser.add_argument('--truncation-psi-fine', type=float, default=None,
                        help='Separate truncation psi for the fine synthesis stages '
                             '(colour/texture/rim detail). If unset, defaults to '
                             '--truncation-psi (original single-psi behaviour)')
    parser.add_argument('--truncation-cutoff', type=int, default=3,
                        help='First stage index using --truncation-psi-fine instead '
                             'of --truncation-psi; stages 0..cutoff-1 are coarse '
                             '(silhouette/proportions), cutoff..5 are fine '
                             '(colour/texture/detail); 6 stages total (8x16..256x512)')
    parser.add_argument('--truncation-centers', type=int, default=1,
                        help='Number of w-space k-means cluster centers to truncate '
                             'toward. 1 = original single global-mean truncation '
                             '(bit-exact). k>1 truncates each sample toward its own '
                             'nearest cluster center instead of one global mean, to '
                             'avoid collapsing distinct frame styles together')
    # Sweep hyperparameters
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    parser.add_argument('--lr-g', type=float, default=Config.lr_g)
    parser.add_argument('--lr-d', type=float, default=Config.lr_d)
    parser.add_argument('--r1-gamma', type=float, default=Config.r1_gamma)
    parser.add_argument('--fm-weight', type=float, default=Config.fm_weight)
    parser.add_argument('--ppl-weight', type=float, default=Config.ppl_weight)
    parser.add_argument('--perceptual-weight', type=float, default=Config.perceptual_weight)
    parser.add_argument('--style-mixing-prob', type=float, default=Config.style_mixing_prob)
    parser.add_argument('--mapping-depth', type=int, default=Config.mapping_depth)
    parser.add_argument('--d-width-mult', type=float, default=1.0,
                        help='Discriminator channel width multiplier')
    parser.add_argument('--ada-target', type=float, default=Config.ada_target)
    parser.add_argument('--ada-max-p', type=float, default=Config.ada_max_p,
                        help='Upper limit for the ADA augmentation probability')
    parser.add_argument('--ada-color-max-p', type=float, default=Config.ada_color_max_p,
                        help='Upper limit for the saturation/hue augmentation '
                             'probability specifically; set 0 to disable colour '
                             'augmentation while keeping geometric/photometric ADA')
    parser.add_argument('--kid-start', type=int, default=Config.kid_start)
    parser.add_argument('--sample-weights', default=Config.sample_weights,
                        help='sample_weights.json from metrics/compute_sample_weights.py '
                             '(oversamples rare-colour images; empty = uniform)')
    parser.add_argument('--num-workers', type=int, default=Config.num_workers)
    parser.add_argument('--seed', type=int, default=Config.seed)
    args = parser.parse_args()

    # Apply CLI overrides onto cfg
    cfg.batch_size = args.batch_size
    cfg.lr_g = args.lr_g
    cfg.lr_d = args.lr_d
    cfg.r1_gamma = args.r1_gamma
    cfg.fm_weight = args.fm_weight
    cfg.ppl_weight = args.ppl_weight
    cfg.perceptual_weight = args.perceptual_weight
    cfg.style_mixing_prob = args.style_mixing_prob
    cfg.mapping_depth = args.mapping_depth
    cfg.d_width_mult = args.d_width_mult
    cfg.ada_target = args.ada_target
    cfg.ada_max_p = args.ada_max_p
    cfg.ada_color_max_p = args.ada_color_max_p
    cfg.sample_weights = args.sample_weights
    cfg.kid_start = args.kid_start
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed

    DATASET_PATH = Path(args.images)
    OUTPUT_ROOT = Path(args.output)
    CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
    SAMPLES_DIR = OUTPUT_ROOT / 'samples'
    METRICS_DIR = OUTPUT_ROOT / 'metrics'
    GENERATED_DIR = OUTPUT_ROOT / 'generated'

    setup_directories()
    save_run_config(args)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    print(f'PyTorch {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB)')
    print(f'Data: {DATASET_PATH}')
    print(f'Output: {OUTPUT_ROOT}')

    if args.generate_only:
        generate(args.generate_only, num_images=args.num_images,
                 truncation_psi=args.truncation_psi, pure_frac=args.pure_frac,
                 truncation_psi_fine=args.truncation_psi_fine,
                 truncation_cutoff=args.truncation_cutoff,
                 truncation_centers=args.truncation_centers)
    else:
        resume = not args.no_resume
        checkpoint_path = args.checkpoint
        if checkpoint_path is None and resume:
            checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        print(f'Checkpoint: {checkpoint_path}' if checkpoint_path else 'Starting from scratch.')

        train(end_epoch=args.end_epoch, resume=resume,
              checkpoint_path=checkpoint_path if resume else None)

        if args.generate:
            # Prefer the best-KID checkpoint: the latest one may be past a
            # late-training collapse (this burned the ppl4 run, where ep400
            # was post-collapse while ep310 held the project-best train KID).
            best_ckpt = CHECKPOINT_DIR / 'checkpoint_best.pth'
            final_ckpt = str(best_ckpt) if best_ckpt.exists() \
                else find_latest_checkpoint(CHECKPOINT_DIR)
            if final_ckpt:
                generate(final_ckpt, num_images=args.num_images,
                         truncation_psi=args.truncation_psi, pure_frac=args.pure_frac,
                         truncation_psi_fine=args.truncation_psi_fine,
                         truncation_cutoff=args.truncation_cutoff,
                         truncation_centers=args.truncation_centers)
