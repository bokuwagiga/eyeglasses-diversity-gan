"""
Conditional StyleGAN2 for eyeglass frame generation (256x512).

Outputs RGB image + binary mask. Compared to full StyleGAN2:
single discriminator with feature matching loss for stability
8-layer mapping network (matches original StyleGAN2)
reduced discriminator channel widths to match 6.5k training images
path length regularization (PPL) for smooth latent-to-image mapping
VGG perceptual loss sampled at three feature levels
adaptive discriminator augmentation (ADA) for small datasets
"""

import os
import glob
import json
import math
import random
import time
from pathlib import Path
import shutil

import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm


# Configuration

class Config:
    """Central hyperparameter store. Modify here rather than in individual functions."""

    img_height = 256
    img_width = 512  # 2:1 aspect ratio for eyeglass frames

    latent_dim = 512       # dimensionality of the random noise vector z
    style_embed_dim = 512  # dimensionality of the w-space style code
    mapping_depth = 8      # MLP depth in mapping network (matches original StyleGAN2)

    batch_size = 32
    lr_g = 0.0002   # generator learning rate
    lr_d = 0.00005  # discriminator learning rate
    beta1 = 0.0     # Adam beta1, 0 is standard for GAN training
    beta2 = 0.99    # Adam beta2

    # R1 gradient penalty, penalises D for sharp gradients on real images
    # Official StyleGAN2 paper256 config uses gamma=1.0; paper512 uses 0.5.
    # Previous value of 10 was 10x too strong, over-regularizing D.
    r1_gamma = 1.0
    r1_interval = 16  # apply R1 every N optimiser steps (lazy regularization, per paper)

    # Feature matching: G matches D's intermediate feature statistics on real images.
    # Provides a stable gradient signal even when D is fully saturated (d_loss = 0),
    # preventing G from collapsing to blank output.
    fm_weight = 2.0   # weight of feature matching loss (low to avoid over-averaging)

    # Path length regularization (StyleGAN2): penalises the mapping network when
    # equal steps in w-space do not produce equal-magnitude changes in the image.
    # This forces the mapping network to use w-space uniformly, preventing many
    # different z's from collapsing to the same w -> same output (mode collapse).
    ppl_weight      = 2.0   # weight of PPL penalty
    ppl_interval    = 16    # lazy: apply every N steps (same as R1)
    ppl_decay       = 0.99  # EMA decay for running mean path length
    # PPL uses create_graph=True in float32, which stores the full synthesis
    # computation graph. At batch_size=32 and 256x512 this exceeds 10 GB.
    # Official StyleGAN2 uses pl_minibatch_shrink=2, i.e. half the main batch.
    # With batch_size=32, that is 16. Reduce to 8 or 4 if OOM on PPL step.
    ppl_batch_size  = 16

    # Auxiliary loss weights.
    # Reduced from mask_weight=15, perceptual_weight=1.5: at those values mask loss
    # was ~70% of g_total and perceptual was second-largest, leaving the adversarial
    # signal (g_adv) drowned out. New weights give g_adv ~20% of g_total.
    perceptual_weight = 0.75  # was 1.5
    mask_weight = 7.0         # was 15.0; ramps from 3 over first 50 epochs

    ema_decay = 0.999   # EMA decay for G weights used at inference
    save_every = 10     # checkpoint frequency in epochs

    kid_start    = 250    # start evaluating KID after this epoch
    kid_interval = 15     # evaluate KID every N epochs after kid_start
    kid_n_real   = 1000   # real images to use for KID (subset of training set)
    kid_n_fake   = 1000   # generated images to use for KID

    ada_target   = 0.6    # ADA: target fraction of real images scoring positive in D
    ada_interval = 4      # steps between ADA probability updates
    ada_step     = 0.001  # probability adjustment per update
    ada_max_p    = 0.85   # maximum augmentation probability

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths (defaults; all overridable via CLI args below)
# --images       path to training frame images   (default: data/gan/raw/full_masks/accepted/images)
# --masks        path to training frame masks    (default: data/gan/raw/full_masks/accepted/masks)
# --output       root output directory           (default: results/gan)
# --end-epoch    total epochs to train           (default: 350)
# --no-resume    start from scratch instead of auto-resuming latest checkpoint
# --checkpoint   explicit checkpoint path to resume from

DATA_ROOT = Path('./data')
DATASET_PATH = DATA_ROOT / 'gan' / 'raw' / 'full_masks' / 'accepted' / 'images'
MASKS_PATH   = DATA_ROOT / 'gan' / 'raw' / 'full_masks' / 'accepted' / 'masks'

OUTPUT_ROOT = Path('results/gan')
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
OUTPUT_DIR = OUTPUT_ROOT / 'samples'
METRICS_DIR = OUTPUT_ROOT / 'metrics'
GENERATED_DIR = OUTPUT_ROOT / 'generated'

RESUME_TRAINING = True
START_EPOCH = 0
END_EPOCH = 350
CHECKPOINT_PATH = None  # None = auto-find latest

cfg = Config()


def setup_directories():
    """Create output folders if they don't exist."""
    for d in [CHECKPOINT_DIR, OUTPUT_DIR, METRICS_DIR, GENERATED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f'Device: {cfg.device}')
    print(f'Data: {DATASET_PATH}')
    print(f'Output: {OUTPUT_ROOT}')


# Dataset

class EyeglassesDataset(Dataset):
    """Paired image + mask dataset.

    Images are center-cropped to 2:1 before resizing to avoid distortion.
    If a mask file is missing, a blank (all-zero) mask is used as fallback.
    The dataset index doubles as the style ID passed to the generator.
    """

    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Collect images recursively; sort for reproducible ordering
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
        self.image_paths.sort()
        self.num_styles = len(self.image_paths)
        print(f'Found {self.num_styles} images')

    def _mask_path(self, img_path):
        """Resolve mask path by mapping the image filename into masks_dir."""
        return os.path.join(self.masks_dir, os.path.basename(img_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self._mask_path(img_path)

        image = Image.open(img_path).convert('RGB')

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)  # blank mask fallback

        # Center-crop to the target 2:1 ratio before resizing to avoid squashing
        w, h = image.size
        target_ratio = cfg.img_width / cfg.img_height  # 2.0

        if w / h > target_ratio:
            # Image is wider than needed, trim left and right equally
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            box = (left, 0, left + new_w, h)
        else:
            # Image is taller than needed, trim top and bottom equally
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            box = (0, top, w, top + new_h)

        image = image.crop(box)
        mask = mask.crop(box)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.Compose([
                transforms.Resize((cfg.img_height, cfg.img_width)),
                transforms.ToTensor(),
            ])(mask)

        return image, mask, idx



# Model Building Blocks


class PixelNorm(nn.Module):
    """Per-sample feature-vector L2 normalisation; keeps activations from growing unbounded."""
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    """Transforms noise z -> style code w via a pixel-normalised MLP.

    w-space is more disentangled than z-space because the MLP can
    'unfold' the Gaussian distribution to match the data manifold.
    """
    def __init__(self, z_dim, w_dim, depth=4):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(depth):
            layers.append(nn.Linear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoiseInjection(nn.Module):
    """Adds spatially independent noise scaled by a per-channel learned weight.

    This introduces stochastic fine detail (hair, texture) without
    affecting the global structure controlled by the style code.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Single-channel noise broadcast across all spatial positions
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyleMod(nn.Module):
    """Produces per-channel affine parameters (scale gamma, shift beta) from a style vector w."""
    def __init__(self, in_channels, w_dim):
        super().__init__()
        self.fc = nn.Linear(w_dim, in_channels * 2)
        # Initialise to identity: gamma=1, beta=0, so the block is a no-op at the start
        self.fc.bias.data[:in_channels] = 1.0
        self.fc.bias.data[in_channels:] = 0.0

    def forward(self, w):
        # Returns (scale, shift) each of shape (B, C, 1, 1)
        style = self.fc(w).unsqueeze(2).unsqueeze(3)
        return style.chunk(2, dim=1)


class ModulatedConv2d(nn.Module):
    """Style-modulated convolution, the core operation of StyleGAN2.

    Each sample's conv weights are scaled by its style vector (modulation),
    then normalised so output features have unit variance (demodulation).
    This is equivalent to instance-normalisation but fused into the conv,
    which avoids the need for explicit normalisation layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.demodulate = demodulate
        self.eps = 1e-8
        self.pad = kernel_size // 2

        # Leading batch dim is 1 here; it becomes B at forward time via broadcasting
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        self.style = StyleMod(in_channels, w_dim)

    def forward(self, x, w):
        B = x.size(0)
        scale, _ = self.style(w)

        # Modulate: multiply each input-channel slice of the kernel by its style scale
        weight = self.weight * scale.unsqueeze(1)

        if self.demodulate:
            # Demodulate: normalise each output filter to unit RMS so variance is preserved
            sigma = torch.sqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + self.eps)
            weight = weight / sigma

        # Fold the batch into the channel axis so a single grouped conv processes
        # all samples simultaneously, each with its own modulated kernel
        weight = weight.reshape(B * weight.size(1), weight.size(2), weight.size(3), weight.size(4))
        x = x.reshape(1, B * x.size(1), x.size(2), x.size(3))
        out = F.conv2d(x, weight, padding=self.pad, groups=B)
        return out.reshape(B, -1, out.size(2), out.size(3))


class StyledConvBlock(nn.Module):
    """One synthesis step: modulated conv -> noise injection -> bias + activation."""
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x, w)
        x = self.noise(x)
        x = x + self.bias
        return self.act(x)


class SelfAttention(nn.Module):
    """Non-local self-attention block for capturing long-range spatial dependencies.

    Helps the model learn global symmetry constraints (e.g. left-right balance
    of a frame). Applied only at the 32x64 resolution because attention cost
    is O(H^2W^2) and is prohibitive at higher resolutions.

    gamma starts at zero so the block is bypassed initially and learns to
    contribute gradually as training progresses.
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 1)
        self.query = spectral_norm(nn.Conv2d(channels, mid, 1))
        self.key   = spectral_norm(nn.Conv2d(channels, mid, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, mid)
        k = self.key(x).view(B, -1, H * W)                      # (B, mid, HW)
        v = self.value(x).view(B, -1, H * W)                    # (B, C,   HW)

        # Scaled dot-product attention
        attn = torch.bmm(q, k) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out



# Generator


class MaskDecoder(nn.Module):
    """U-Net-style decoder that reconstructs a binary mask from G's skip features.

    Takes the deepest skip tensor as the starting feature map, then at each
    step upsamples and concatenates the next shallower skip before applying
    two conv layers, similar to U-Net's decoder path.
    skip_channels lists channel widths from deepest (index 0) to shallowest.
    """
    def __init__(self, skip_channels):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_channels = skip_channels[0]
        for i in range(1, len(skip_channels)):
            out_channels = skip_channels[i]
            self.blocks.append(nn.Sequential(
                # After concatenation the input has current + skip channels
                nn.Conv2d(current_channels + out_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            ))
            current_channels = out_channels

        self.to_mask = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 3, 1, 1),
            nn.InstanceNorm2d(current_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(current_channels, 1, 1),  # project to single-channel mask
            nn.Sigmoid()
        )

    def forward(self, skips):
        """skips: list of feature maps ordered deepest -> shallowest."""
        x = skips[0]
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, size=skips[i + 1].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[i + 1]], dim=1)
            x = block(x)
        return self.to_mask(x)


class Generator(nn.Module):
    """StyleGAN2 synthesis network for 256x512 eyeglass frames.

    Architecture:
      - Learned 4x8 constant tensor as the seed feature map
      - 6 upsampling stages (x2 each) via StyledConvBlock pairs
      - Progressive RGB accumulation: each stage adds residual detail
      - Self-attention inserted at the 32x64 resolution stage
      - MaskDecoder branch fed with skip features from the last 4 stages

    Channel progression: 512 -> 256 -> 256 -> 128 -> 64 -> 32 -> 16
    """
    CHANNELS = [512, 256, 256, 128, 64, 32, 16]

    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        self.w_dim = w_dim
        # 8-layer mapping network z -> w, matching original StyleGAN2
        self.mapping = MappingNetwork(z_dim, w_dim, depth=cfg.mapping_depth)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 8))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        ch = self.CHANNELS
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        # Attention applied after the block that outputs ch[attn_idx + 1] channels
        self.attn_idx = 2
        self.attention = SelfAttention(ch[self.attn_idx + 1])

        for i in range(len(ch) - 1):
            self.blocks.append(nn.ModuleList([
                StyledConvBlock(ch[i], ch[i + 1], w_dim),
                StyledConvBlock(ch[i + 1], ch[i + 1], w_dim),
            ]))
            # 1x1 modulated conv projects features to RGB (no demodulation needed for output)
            self.to_rgb.append(ModulatedConv2d(ch[i + 1], 3, 1, w_dim, demodulate=False))

        # Feed skip features from the last 4 synthesis stages to the mask decoder
        self.mask_decoder = MaskDecoder(skip_channels=ch[-4:])

    def get_w(self, z):
        """Map noise z -> disentangled w-space code via the 8-layer mapping network."""
        # PixelNorm is the first layer of self.mapping (matches original StyleGAN2).
        # F.normalize here was redundant: L2-normalising to the unit sphere before
        # PixelNorm would rescale the result by sqrt(latent_dim) ~= 22.6, giving a
        # combined normalisation that is neither pure PixelNorm nor pure L2-norm.
        return self.mapping(z)

    def synthesis(self, w):
        """Run the synthesis network given a w-space code.

        RGB is built progressively: each stage upsamples the current RGB
        and adds the new stage's contribution, similar to skip connections.
        """
        batch_size = w.size(0)
        x = self.const.repeat(batch_size, 1, 1, 1)

        rgb = None   # accumulated RGB; starts as None so the first stage sets it directly
        skips = []   # feature maps from the last 4 stages for the mask decoder

        for i, (block, rgb_layer) in enumerate(zip(self.blocks, self.to_rgb)):
            x = self.upsample(x)
            x = block[0](x, w)
            x = block[1](x, w)

            if i == self.attn_idx:
                x = self.attention(x)

            rgb_out = rgb_layer(x, w)
            if rgb is None:
                rgb = rgb_out
            else:
                # Upsample previous RGB to current resolution and add residual detail
                rgb = self.upsample(rgb) + rgb_out

            if i >= len(self.blocks) - 4:
                skips.append(x)

        return torch.tanh(rgb), self.mask_decoder(skips)

    def forward(self, z):
        w = self.get_w(z)
        return self.synthesis(w)



# Discriminator


class MinibatchSTD(nn.Module):
    """Appends a constant-value channel whose value is the mean std across the batch.

    When G produces a mode collapse, all samples look alike and the std is near
    zero, a signal D can exploit to distinguish fake batches from real ones.
    """
    def forward(self, x):
        std = x.std(dim=0).mean()
        std_map = std.view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_map], dim=1)


class DiscBlock(nn.Module):
    """Residual downsampling block: two convs + identity skip, then avg-pool.

    Spectral norm removed: StyleGAN2 paper (Karras et al. 2019) states that
    enabling spectral normalization in addition to their contributions (or
    instead of them) invariably compromises FID. R1 regularization alone is
    sufficient for discriminator stability at this scale.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.down  = nn.AvgPool2d(2)
        # 1x1 skip projection to match output channels before residual addition
        self.skip  = nn.Conv2d(in_channels, out_channels, 1)
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        return self.down(y) + self.down(self.skip(x))


class Discriminator(nn.Module):
    """Discriminator operating on 4-channel input (RGB + mask).

    Uses R1 regularization only (no spectral norm). StyleGAN2 paper explicitly
    states spectral norm invariably compromises FID when combined with their
    demodulation and R1 contributions. Minibatch std and self-attention
    discourage mode collapse and capture long-range structure.
    """
    def __init__(self):
        super().__init__()

        # Reduced channel widths to match the 6.5k training set.
        # Original StyleGAN2 uses ~400 params/image; [32,64,128,256,512,512] gives
        # ~3000 params/image (7.5x too much), causing D to memorise rather than learn.
        ch = [16, 32, 64, 128, 256, 256]

        self.from_rgb = nn.Conv2d(4, ch[0], 1)  # 3 RGB + 1 mask
        self.act = nn.LeakyReLU(0.2)

        self.blocks = nn.Sequential(*[
            DiscBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1)
        ])

        # Attention at the 128-channel resolution (after 3 downsampling blocks)
        self.attn = SelfAttention(ch[3])

        self.mbstd = MinibatchSTD()
        self.final = nn.Conv2d(ch[-1] + 1, ch[-1], 3, padding=1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(ch[-1] * 16, 1)
        )

    def forward(self, img, mask, return_features=False):
        x = torch.cat([img, mask], dim=1)
        x = self.act(self.from_rgb(x))

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:
                x = self.attn(x)
            if return_features:
                features.append(x)

        x = self.mbstd(x)
        x = self.act(self.final(x))
        logit = self.head(x).squeeze()

        if return_features:
            return logit, features
        return logit



# Losses


def d_hinge(real_pred, fake_pred):
    """Hinge adversarial loss for D.

    Penalises real scores below +1 and fake scores above -1, leaving a
    margin of 2 between the two sides.
    """
    real_loss = torch.relu(1.0 - real_pred).mean()
    fake_loss = torch.relu(1.0 + fake_pred).mean()
    return real_loss + fake_loss


def g_hinge(fake_pred):
    """Hinge adversarial loss for G, maximises D's score on generated images."""
    return -fake_pred.mean()


def r1_penalty(real_pred, real_img):
    """R1 gradient penalty (Mescheder et al. 2018).

    Penalises the magnitude of D's gradients with respect to real images,
    encouraging D to be locally Lipschitz around the real data manifold
    rather than memorising sharp decision boundaries.
    """
    grad = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True
    )[0]
    return (grad.view(grad.size(0), -1).norm(2, dim=1) ** 2).mean()


class FocalBCE(nn.Module):
    """Focal binary cross-entropy for mask supervision.

    Multiplies the per-pixel BCE by (1 - p_t)^gamma so easily-classified
    pixels contribute less to the gradient, forcing the model to focus
    on hard boundary and thin-structure regions.
    alpha weights the positive (foreground) class to handle class imbalance.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)  # guard against exact 0/1 which crash CUDA BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)                    # p_t: probability assigned to the correct class
        alpha_t = torch.where(target > 0.5, self.alpha, 1 - self.alpha)   # class-balancing weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma                    # down-weight confident predictions
        return (focal_weight * bce).mean()


def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss, complements BCE for small/sparse mask regions.

    Dice is volume-based so it is not dominated by the large background
    class the way per-pixel BCE can be.
    """
    pred_flat   = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    # smooth avoids division by zero and provides a small gradient even for empty masks
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG16 features at three levels.

    relu1_2 captures low-level edges, relu2_2 textures, relu3_3 mid-level
    structure (shape of lenses and frame outline). Using all three levels
    gives a richer gradient signal than edges alone.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # relu3_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return ((x + 1) / 2.0 - self.mean) / self.std

    def forward(self, pred, target):
        p = self._normalize(pred)
        t = self._normalize(target)
        loss = 0.0
        for layer, w in [(self.slice1, 1.0), (self.slice2, 1.0), (self.slice3, 0.5)]:
            p = layer(p)
            with torch.no_grad():
                t = layer(t)
            loss += w * F.l1_loss(p, t)
        return loss



# Adaptive Discriminator Augmentation


class ADAugment:
    """Adaptive Discriminator Augmentation (Karras et al. 2020).

    Tracks the fraction of real images that D scores above zero (r_t) and
    adjusts augmentation probability p so that r_t stays near ada_target.
    When D overfits to real images r_t rises above target and p increases,
    making D's job harder; when D underfits p decreases.

    Augmentations are applied to both real and fake images before D sees
    them, so G's task difficulty scales with D's. Only the image is colour-
    jittered; the mask is only flipped/cut so it stays a valid binary map.
    """
    def __init__(self, target=0.6, step=0.001, max_p=0.85, interval=4):
        self.target   = target
        self.step     = step
        self.max_p    = max_p
        self.interval = interval
        self.p        = 0.0
        self._signs   = []

    def update(self, real_pred):
        """Call once per D step with the raw real predictions tensor."""
        # Fraction of real logits that are positive, in [0, 1].
        # Previously used real_pred.sign().mean() which gives a value in [-1, +1],
        # making the 0.6 target equivalent to requiring ~80% positive logits and
        # causing ADA to shut off prematurely. The paper defines rt in [0, 1].
        self._signs.append((real_pred > 0).float().mean().item())
        if len(self._signs) >= self.interval:
            rt = sum(self._signs) / len(self._signs)
            self._signs.clear()
            if rt > self.target:
                self.p = min(self.p + self.step, self.max_p)
            else:
                self.p = max(self.p - self.step, 0.0)

    def __call__(self, imgs, masks):
        """Apply stochastic augmentations; returns augmented (imgs, masks)."""
        if self.p <= 0.0:
            return imgs, masks

        B      = imgs.size(0)
        device = imgs.device

        # Horizontal flip - applied to both image and mask
        flip = torch.rand(B, device=device) < self.p
        if flip.any():
            imgs  = torch.where(flip.view(B, 1, 1, 1), imgs.flip(-1),  imgs)
            masks = torch.where(flip.view(B, 1, 1, 1), masks.flip(-1), masks)

        # Brightness jitter - per-sample, image only.
        # Previously gated by one random.random() call, meaning the whole batch
        # got jitter or none did. ADA's rt controller is per-sample, so all-or-
        # nothing augmentation miscalibrates p. Per-sample masks fix this.
        brightness_mask = (torch.rand(B, device=device) < self.p).view(B, 1, 1, 1)
        scale = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
        imgs = torch.where(brightness_mask, (imgs * scale).clamp(-1.0, 1.0), imgs)

        # Gaussian noise - per-sample, image only
        noise_mask = (torch.rand(B, device=device) < self.p * 0.5).view(B, 1, 1, 1)
        imgs = torch.where(noise_mask, (imgs + 0.05 * torch.randn_like(imgs)).clamp(-1.0, 1.0), imgs)

        # Cutout: zero a random rectangular patch - per-sample
        cutout_mask = torch.ones_like(imgs)
        H, W = imgs.shape[2], imgs.shape[3]
        ph, pw = H // 4, W // 4
        for b in range(B):
            if random.random() < self.p * 0.5:
                t = random.randint(0, H - ph)
                l = random.randint(0, W - pw)
                cutout_mask[b, :, t:t + ph, l:l + pw] = 0.0
        imgs = imgs * cutout_mask

        return imgs, masks


# EMA (Exponential Moving Average)


class EMA:
    """Exponential moving average of G's trainable parameters.

    The EMA weights typically produce cleaner outputs than the raw
    training weights because they average out the per-step noise.
    apply_shadow / restore swap the EMA weights in and out so the
    raw weights are still available for continued training.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model):
        """Blend current model weights into the shadow: shadow = decay*shadow + (1-decay)*param."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Swap live weights for EMA weights, saving the originals in backup."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore the live training weights from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}



# Utilities


def save_metrics(history, cfg):
    """Dump training metrics as JSON."""
    path = os.path.join(cfg.metrics_dir, 'training_metrics.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Metrics saved: {path}')


def find_latest_checkpoint(checkpoint_dir):
    """Return checkpoint_latest.pth if it exists, otherwise None."""
    p = Path(checkpoint_dir) / 'checkpoint_latest.pth'
    return p if p.exists() else None



# Training Loop


def train(start_epoch, end_epoch, resume=False, checkpoint_path=None):
    # ColorJitter removed: ADA is the sole source of calibrated augmentation.
    # Static jitter at data-load time means D sees both sources but ADA's rt
    # controller only tracks ADA augmentations, causing systematic miscalibration
    # of the augmentation probability p.
    img_transform = transforms.Compose([
        transforms.Resize((cfg.img_height, cfg.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)   # map [0,1] -> [-1, 1]
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((cfg.img_height, cfg.img_width)),
        transforms.ToTensor(),
    ])

    dataset = EyeglassesDataset(str(DATASET_PATH), str(MASKS_PATH), img_transform, mask_transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    G = Generator(cfg.latent_dim, cfg.style_embed_dim).to(cfg.device)
    D = Discriminator().to(cfg.device)

    vgg   = VGGPerceptualLoss().to(cfg.device)
    focal = FocalBCE().to(cfg.device)
    ada   = ADAugment(cfg.ada_target, cfg.ada_step, cfg.ada_max_p, cfg.ada_interval)

    g_optim = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    d_optim = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')


    ema = EMA(G, cfg.ema_decay)

    # Fixed noise for consistent visual comparisons across epochs
    fixed_z = torch.randn(16, cfg.latent_dim, device=cfg.device)

    history = {
        'epoch': [], 'g_loss': [], 'd_loss': [], 'mask_loss': [], 'perc_loss': [],
        'fm_loss': [], 'g_total': [], 'd_real': [], 'd_fake': [], 'ada_p': [],
        'mask_iou': [], 'mask_dice': [], 'kid': [], 'epoch_sec': [], 'total_hrs': [],
    }

    best_kid     = float('inf')
    ppl_mean_ema = 0.0   # running mean path length for PPL; restored from checkpoint

    if resume and checkpoint_path and os.path.exists(str(checkpoint_path)):
        print(f'Loading checkpoint: {checkpoint_path}')
        ckpt = torch.load(str(checkpoint_path), map_location=cfg.device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        if 'ema' in ckpt:
            ema.shadow = ckpt['ema']
        if 'ada_p' in ckpt:
            ada.p = ckpt['ada_p']
        if 'history' in ckpt:
            history = ckpt['history']
        if 'best_kid' in ckpt:
            best_kid = ckpt['best_kid']
        if 'ppl_mean_ema' in ckpt:
            ppl_mean_ema = ckpt['ppl_mean_ema']
        start_epoch = ckpt['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f'Generator params:     {g_params:,}')
    print(f'Discriminator params: {d_params:,}')
    print(f'Training epochs {start_epoch} -> {end_epoch}')
    print()

    # KID pool and metric are built lazily on first use (epoch kid_start).
    # Building them upfront holds ~400 MB CPU RAM + Inception model on GPU
    # for hundreds of epochs before they are ever needed.
    kid_real_pool = None
    kid_metric    = None

    t0 = time.time()
    global_step = 0
    d_collapse_counter = 0  # counts consecutive steps with d_loss < 1e-4

    for epoch in range(start_epoch, end_epoch):
        t_epoch = time.time()
        G.train()
        D.train()

        sums = {'g': 0.0, 'd': 0.0, 'mask': 0.0, 'perc': 0.0, 'fm': 0.0,
                'g_total': 0.0, 'dr': 0.0, 'df': 0.0}

        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{end_epoch}', leave=False)
        for real_imgs, real_masks, _idx in pbar:
            real_imgs  = real_imgs.to(cfg.device)
            real_masks = real_masks.to(cfg.device)
            batch_size = real_imgs.size(0)

            # --- Discriminator step ---
            d_optim.zero_grad()

            aug_imgs, aug_masks = ada(real_imgs.detach().clone(), real_masks.detach().clone())

            with torch.amp.autocast('cuda'):
                real_pred, real_feats = D(aug_imgs, aug_masks, return_features=True)

            ada.update(real_pred.detach().float())

            with torch.no_grad():
                z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
                with torch.amp.autocast('cuda'):
                    fake_imgs, fake_masks = G(z)
                fake_imgs  = fake_imgs.float()
                fake_masks = fake_masks.float().clamp(0.0, 1.0)
                fake_aug, fake_mask_aug = ada(fake_imgs.clone(), fake_masks.clone())

            with torch.amp.autocast('cuda'):
                fake_pred = D(fake_aug, fake_mask_aug)
                d_loss = d_hinge(real_pred.float(), fake_pred.float())

            # R1 on non-augmented real images (StyleGAN2-ADA Appendix C).
            # R1 must regularise D on the true real data manifold, not on
            # augmented samples. Applying it to aug_imgs would penalise D for
            # sharp gradients on flipped/jittered images rather than on real ones,
            # defeating the purpose of the regulariser. Separate D forward pass.
            # The forward runs in float32 (no autocast): create_graph=True inside
            # autocast produces NaN because fp16 intermediates appear in the
            # higher-order gradient graph.
            if global_step % cfg.r1_interval == 0:
                real_imgs_r1 = real_imgs.detach().requires_grad_(True)
                real_pred_r1 = D(real_imgs_r1, real_masks, return_features=False)
                r1 = r1_penalty(real_pred_r1, real_imgs_r1)
                # Lazy R1 scaling (StyleGAN2 Section B): when R1 fires every r1_interval
                # steps instead of every step, multiply the weight by r1_interval
                # so the average gradient contribution per step stays constant.
                # Without this, R1 is r1_interval times weaker than intended
                # (10 * 0.5 / 16 ~= 0.3 effective weight vs 5 intended).
                d_loss = d_loss + cfg.r1_gamma * 0.5 * cfg.r1_interval * r1

            # D-collapse trip-wire: if d_loss stays near zero for 100 consecutive
            # steps D has fully saturated and G will receive no useful gradient.
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
            with torch.amp.autocast('cuda'):
                fake_imgs, fake_masks = G(z)
            fake_imgs  = fake_imgs.float()
            fake_masks = fake_masks.float().clamp(0.0, 1.0)
            # Clone before ADA so fake_imgs stays unmodified for perceptual loss backward
            fake_aug, fake_mask_aug = ada(fake_imgs.clone(), fake_masks.clone())

            with torch.amp.autocast('cuda'):
                fake_pred, fake_feats = D(fake_aug, fake_mask_aug, return_features=True)

            g_adv     = g_hinge(fake_pred.float())
            mask_loss = focal(fake_masks, real_masks) + dice_loss(fake_masks, real_masks)
            perc_loss = vgg(fake_imgs, real_imgs)

            # Feature matching: G matches D's intermediate feature statistics on real images.
            # Gives G a meaningful gradient even when D is saturated (d_loss = 0).
            fm_loss = torch.stack([
                F.l1_loss(f.float().mean([2, 3]), r.detach().float().mean([2, 3]))
                for f, r in zip(fake_feats, real_feats)
            ]).mean()

            # Ramp mask_weight from 3 to cfg.mask_weight over the first 50 epochs.
            mask_w = min(cfg.mask_weight, 3.0 + (cfg.mask_weight - 3.0) * min(epoch / 50.0, 1.0))

            g_total = (g_adv
                       + mask_w               * mask_loss
                       + cfg.perceptual_weight * perc_loss
                       + cfg.fm_weight        * fm_loss)

            # Path length regularization (StyleGAN2): lazy, every ppl_interval steps.
            # Penalises the mapping network when equal steps in w-space produce
            # unequal changes in the image, preventing many z -> same w collapse.
            if global_step % cfg.ppl_interval == 0:
                z_ppl = torch.randn(cfg.ppl_batch_size, cfg.latent_dim, device=cfg.device)
                # Run entirely in float32 outside autocast: create_graph=True with
                # autocast produces NaN gradients because fp16 intermediate tensors
                # appear inside the higher-order gradient graph.
                # Do NOT detach w_ppl from the mapping network. The original StyleGAN2
                # lets PPL gradients flow all the way back through the mapping so the
                # mapping is penalised when it produces w vectors that land in poorly-
                # conditioned regions of the synthesis Jacobian. Detaching severs this
                # and means only the synthesis network receives PPL signal, leaving the
                # mapping free to collapse many z -> same w undetected.
                w_ppl = G.get_w(z_ppl)
                ppl_imgs, _ = G.synthesis(w_ppl)
                noise = torch.randn_like(ppl_imgs) / math.sqrt(ppl_imgs.shape[2] * ppl_imgs.shape[3])
                grad = torch.autograd.grad(
                    outputs=(ppl_imgs * noise).sum(),
                    inputs=w_ppl,
                    create_graph=True,
                )[0]
                path_lengths = grad.pow(2).sum(1).sqrt()
                path_mean = path_lengths.mean()
                if ppl_mean_ema == 0.0:
                    # First call: seed the EMA from the actual path length so the first
                    # penalty is not (path_lengths - 0)^2, which is a large destabilising
                    # spike on the mapping network at step ppl_interval.
                    ppl_mean_ema = path_mean.item()
                else:
                    ppl_penalty = (path_lengths - ppl_mean_ema).pow(2).mean()
                    with torch.no_grad():
                        ppl_mean_ema = ppl_mean_ema * cfg.ppl_decay + path_mean.item() * (1 - cfg.ppl_decay)
                    g_total = g_total + cfg.ppl_weight * ppl_penalty

            scaler_g.scale(g_total).backward()
            scaler_g.unscale_(g_optim)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_g.step(g_optim)
            scaler_g.update()
            ema.update(G)

            sums['g']       += g_adv.item()
            sums['d']       += d_loss.item()
            sums['mask']    += mask_loss.item()
            sums['perc']    += perc_loss.item()
            sums['fm']      += fm_loss.item()
            sums['g_total'] += g_total.item()
            sums['dr']      += real_pred.mean().item()
            sums['df']      += fake_pred.mean().item()
            global_step += 1

            pbar.set_postfix(G=f'{g_adv.item():.3f}', D=f'{d_loss.item():.3f}',
                             FM=f'{fm_loss.item():.3f}', ADA_p=f'{ada.p:.3f}')

        # end of epoch
        n_batches = len(loader)
        epoch_time = time.time() - t_epoch
        total_hours = (time.time() - t0) / 3600

        # --- End-of-epoch evaluation (EMA weights, no grad) ---
        with torch.no_grad():
            ema.apply_shadow(G)
            G.eval()

            # Fixed-z samples give a stable visual reference across epochs
            eval_imgs, eval_masks = G(fixed_z)

            # Random samples used for mask quality metrics
            eval_z = torch.randn(cfg.batch_size, cfg.latent_dim, device=cfg.device)
            gen_imgs, gen_masks = G(eval_z)

            # Draw a fresh batch of real masks to compare against generated masks
            real_masks_batch = next(iter(loader))[1].to(cfg.device)
            n_eval = min(gen_masks.size(0), real_masks_batch.size(0))

            gen_binary  = (gen_masks[:n_eval] > 0.5).float()
            real_binary = (real_masks_batch[:n_eval] > 0.5).float()

            intersection = (gen_binary * real_binary).sum(dim=[1, 2, 3])
            union        = (gen_binary + real_binary).clamp(0, 1).sum(dim=[1, 2, 3])

            iou  = (intersection / (union + 1e-8)).mean().item()
            dice = (2 * intersection / (gen_binary.sum(dim=[1, 2, 3]) + real_binary.sum(dim=[1, 2, 3]) + 1e-8)).mean().item()

            save_image(eval_imgs,  f'{cfg.output_dir}/epoch_{epoch + 1:04d}_imgs.png',  nrow=4, normalize=True)
            save_image(eval_masks, f'{cfg.output_dir}/epoch_{epoch + 1:04d}_masks.png', nrow=4)

            # KID evaluation: after kid_start epochs, every kid_interval epochs
            kid_mean = None
            # >= so KID fires at exactly kid_start, not one epoch after it.
            if (epoch + 1) >= cfg.kid_start and (epoch + 1 - cfg.kid_start) % cfg.kid_interval == 0:
                # Lazy init: build pool and metric only when first needed.
                if kid_real_pool is None:
                    print(f'Building real image pool for KID ({cfg.kid_n_real} images) ...')
                    _pool = []
                    for real_batch, _, _ in loader:
                        imgs_uint8 = ((real_batch.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                        _pool.append(imgs_uint8)
                        if sum(x.size(0) for x in _pool) >= cfg.kid_n_real:
                            break
                    kid_real_pool = torch.cat(_pool)[:cfg.kid_n_real]
                    kid_metric = KernelInceptionDistance(
                        subset_size=min(100, cfg.kid_n_real // 2)
                    ).to(cfg.device)
                kid_metric.reset()
                # Feed real images in batches
                for i in range(0, len(kid_real_pool), 64):
                    kid_metric.update(kid_real_pool[i:i + 64].to(cfg.device), real=True)
                # Generate fake images and feed them
                n_fake_done = 0
                while n_fake_done < cfg.kid_n_fake:
                    bs_kid = min(32, cfg.kid_n_fake - n_fake_done)
                    z_kid = torch.randn(bs_kid, cfg.latent_dim, device=cfg.device)
                    with torch.amp.autocast('cuda'):
                        fake_kid, _ = G(z_kid)
                    fake_uint8 = ((fake_kid.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                    kid_metric.update(fake_uint8, real=False)
                    n_fake_done += bs_kid
                kid_mean, kid_std = kid_metric.compute()
                kid_mean = kid_mean.item()
                kid_std  = kid_std.item()
                print(f'  KID: {kid_mean:.5f} +- {kid_std:.5f}')

            G.train()
            ema.restore(G)

        # Update history
        history['epoch'].append(epoch + 1)
        history['g_loss'].append(sums['g'] / n_batches)
        history['d_loss'].append(sums['d'] / n_batches)
        history['mask_loss'].append(sums['mask'] / n_batches)
        history['perc_loss'].append(sums['perc'] / n_batches)
        history['fm_loss'].append(sums['fm'] / n_batches)
        history['g_total'].append(sums['g_total'] / n_batches)
        history['d_real'].append(sums['dr'] / n_batches)
        history['d_fake'].append(sums['df'] / n_batches)
        history['ada_p'].append(ada.p)
        history['mask_iou'].append(iou)
        history['mask_dice'].append(dice)
        history['kid'].append(kid_mean)
        history['epoch_sec'].append(epoch_time)
        history['total_hrs'].append(total_hours)

        kid_str = f'  KID={kid_mean:.5f}' if kid_mean is not None else ''
        print(f'Ep {epoch + 1:4d}/{end_epoch}  '
              f'G={sums["g"] / n_batches:.4f}  D={sums["d"] / n_batches:.4f}  '
              f'FM={sums["fm"] / n_batches:.4f}  Mask={sums["mask"] / n_batches:.4f}  '
              f'IoU={iou:.4f}  ADA_p={ada.p:.3f}{kid_str}  {epoch_time:.0f}s  ({total_hours:.1f}h total)')
        # GradScaler diagnostic: if either scale drops below ~64 and stays there,
        # there is persistent fp16 overflow that is silently skipping optimizer steps.
        print(f'  scaler_g={scaler_g.get_scale():.0f}  scaler_d={scaler_d.get_scale():.0f}')

        save_metrics(history, cfg)

        # Always save latest checkpoint (overwrites previous latest)
        ema.apply_shadow(G)
        ckpt_data = {
            'epoch':        epoch,
            'G':            G.state_dict(),
            'D':            D.state_dict(),
            'g_optim':      g_optim.state_dict(),
            'd_optim':      d_optim.state_dict(),
            'ema':          ema.shadow,
            'ada_p':        ada.p,
            'best_kid':     best_kid,
            'ppl_mean_ema': ppl_mean_ema,
            'history':      history,
        }
        torch.save(ckpt_data, os.path.join(cfg.checkpoint_dir, 'checkpoint_latest.pth'))

        # Save best checkpoint when KID improves (lower is better)
        if kid_mean is not None and kid_mean < best_kid:
            best_kid = kid_mean
            torch.save(ckpt_data, os.path.join(cfg.checkpoint_dir, 'checkpoint_best.pth'))
            print(f'  Best checkpoint saved (KID {kid_mean:.5f})')

        ema.restore(G)

    print(f'\nfinished, {(time.time() - t0) / 3600:.2f}h total')
    save_metrics(history, cfg)



# Generation


def generate(checkpoint_path, num_images=10000, truncation_psi=0.7, batch_size=16):
    """Generate a diverse dataset from a trained checkpoint.

    Without style embeddings, diversity comes entirely from sampling different
    z vectors and interpolating in w-space. Generation modes:
      30 % pure     - single z sample
      40 % interp2  - blend two w vectors (Beta bimodal weights)
      20 % interp3  - blend three w vectors (Dirichlet weights)
      10 % interp4  - blend four w vectors (Dirichlet weights)

    The truncation trick pulls each w toward the population mean w_mean,
    trading diversity for visual quality. psi=0.7 is a good default.
    """
    out_dir  = str(GENERATED_DIR)
    img_dir  = os.path.join(out_dir, 'images')
    mask_dir = os.path.join(out_dir, 'masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    ckpt = torch.load(str(checkpoint_path), map_location=cfg.device)
    G = Generator(cfg.latent_dim, cfg.style_embed_dim).to(cfg.device)
    G.load_state_dict(ckpt['G'])
    if 'ema' in ckpt:
        for name, param in G.named_parameters():
            if name in ckpt['ema']:
                param.data.copy_(ckpt['ema'][name])
    G.eval()

    # Estimate w_mean over 10 000 random z samples for the truncation trick
    print('Computing w_mean ...')
    w_samples = []
    with torch.no_grad():
        for _ in range(200):
            z = torch.randn(50, cfg.latent_dim, device=cfg.device)
            w_samples.append(G.get_w(z))
    w_mean = torch.cat(w_samples).mean(0, keepdim=True)

    # Build generation plan
    n_pure    = int(num_images * 0.30)
    n_interp2 = int(num_images * 0.40)
    n_interp3 = int(num_images * 0.20)
    n_interp4 = num_images - n_pure - n_interp2 - n_interp3

    plan = (
        [('pure',    1)] * n_pure    +
        [('interp2', 2)] * n_interp2 +
        [('interp3', 3)] * n_interp3 +
        [('interp4', 4)] * n_interp4
    )
    random.shuffle(plan)

    generated = 0
    meta_path = os.path.join(out_dir, 'metadata.csv')
    meta_file = open(meta_path, 'w')
    meta_file.write('id,type,n_z,psi\n')

    try:
        pbar = tqdm(total=num_images, desc='Generating')

        with torch.no_grad():
            while generated < num_images:
                bs         = min(batch_size, num_images - generated)
                batch_plan = plan[generated:generated + bs]

                # For each item: sample n_z vectors, compute w codes, blend in w-space
                w_batch = torch.zeros(bs, G.w_dim, device=cfg.device)
                rows    = []

                for j, (gen_type, n_z) in enumerate(batch_plan):
                    idx = generated + j
                    z_j = torch.randn(n_z, cfg.latent_dim, device=cfg.device)
                    w_j = G.get_w(z_j)  # (n_z, w_dim)

                    if n_z == 1:
                        w_blend = w_j[0]
                    elif n_z == 2:
                        alpha = float(np.random.beta(0.4, 0.4))
                        w_blend = alpha * w_j[0] + (1 - alpha) * w_j[1]
                    else:
                        conc    = 0.7 if n_z == 3 else 0.5
                        weights = np.random.dirichlet([conc] * n_z)
                        w_blend = sum(float(weights[k]) * w_j[k] for k in range(n_z))

                    w_batch[j] = w_blend
                    rows.append(f'{idx:05d},{gen_type},{n_z},{truncation_psi:.2f}\n')

                # Truncation trick
                w_batch = w_mean + truncation_psi * (w_batch - w_mean)
                imgs, masks = G.synthesis(w_batch)

                for j in range(bs):
                    img_idx = generated + j
                    save_image(imgs[j],  f'{img_dir}/glass_{img_idx:05d}.png', normalize=True)
                    save_image(masks[j], f'{mask_dir}/mask_{img_idx:05d}.png')
                    meta_file.write(rows[j])

                generated += bs
                pbar.update(bs)

        pbar.close()
        print(f'Generated {generated} images -> {out_dir}')

        type_counts: dict = {}
        for gen_type, _ in plan:
            type_counts[gen_type] = type_counts.get(gen_type, 0) + 1
        for t, c in sorted(type_counts.items()):
            print(f'  {t}: {c} ({c / num_images * 100:.1f}%)')

    finally:
        meta_file.close()



# Main

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train conditional StyleGAN2 for eyeglass frame generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--images',     default=str(DATASET_PATH),
                        help='Directory of training frame images')
    parser.add_argument('--masks',      default=str(MASKS_PATH),
                        help='Directory of training frame masks')
    parser.add_argument('--output',     default=str(OUTPUT_ROOT),
                        help='Root output directory for checkpoints, samples, metrics')
    parser.add_argument('--end-epoch',  type=int, default=END_EPOCH,
                        help='Final training epoch')
    parser.add_argument('--no-resume',  action='store_true',
                        help='Start from scratch, ignoring any existing checkpoints')
    parser.add_argument('--checkpoint', default=None,
                        help='Explicit checkpoint path to resume from')
    parser.add_argument('--generate',   action='store_true',
                        help='Generate 10k images after training completes')
    parser.add_argument('--generate-only', default=None, metavar='CHECKPOINT',
                        help='Skip training; generate from the given checkpoint path')
    parser.add_argument('--truncation-psi', type=float, default=0.7,
                        help='Truncation psi for generation (0.5=quality, 1.0=diversity). Default 0.7')
    args = parser.parse_args()

    DATASET_PATH   = Path(args.images)
    MASKS_PATH     = Path(args.masks)
    OUTPUT_ROOT    = Path(args.output)
    CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
    OUTPUT_DIR     = OUTPUT_ROOT / 'samples'
    METRICS_DIR    = OUTPUT_ROOT / 'metrics'
    GENERATED_DIR  = OUTPUT_ROOT / 'generated'
    END_EPOCH      = args.end_epoch
    RESUME_TRAINING = not args.no_resume

    setup_directories()

    # Fix all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg.output_dir = str(OUTPUT_DIR)
    cfg.checkpoint_dir = str(CHECKPOINT_DIR)
    cfg.metrics_dir = str(METRICS_DIR)

    print(f'PyTorch {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB)')

    checkpoint_path = args.checkpoint
    if checkpoint_path is None and RESUME_TRAINING:
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)

    if checkpoint_path:
        print(f'Checkpoint: {checkpoint_path}')
    else:
        print('Starting from scratch.')

    if args.generate_only:
        generate(args.generate_only, num_images=10000, truncation_psi=args.truncation_psi, batch_size=32)
    else:
        train(
            start_epoch=START_EPOCH,
            end_epoch=END_EPOCH,
            resume=RESUME_TRAINING,
            checkpoint_path=checkpoint_path if RESUME_TRAINING else None
        )

        if args.generate:
            final_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
            if final_ckpt:
                generate(final_ckpt, num_images=10000, truncation_psi=args.truncation_psi, batch_size=32)