"""
dataset.py — Dataset class with augmentation pipeline optimized for rare classes.

ROUND 4 UPGRADES (targeting 0.60+ mIoU):
  - CopyPaste augmentation: copies rare class pixels from random donors
    (Logs 0.1%, Ground Clutter 4.4%, Rocks 1.2%, Dry Bushes 1.1%)
  - Multi-scale training: resize to 1.5× then RandomResizedCrop
    (objects appear at varying scales → better generalization)
  - Stronger geometric augmentations
  - ElasticTransform for learning deformable object boundaries
  - CLAHE for better contrast in desert environments
"""

import os
import ssl
import random

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import VALUE_MAP, IMG_SIZE, MEAN, STD


# ============================================================================
# Mask Conversion — LUT-based (fastest possible)
# ============================================================================

_MAX_RAW_VAL = max(VALUE_MAP.keys()) + 1
_MASK_LUT = np.zeros(_MAX_RAW_VAL + 1, dtype=np.uint8)
for raw_val, class_id in VALUE_MAP.items():
    _MASK_LUT[raw_val] = class_id


def convert_mask(mask_pil):
    """Convert raw 16-bit mask pixel values to class indices (0–10) via LUT."""
    arr = np.array(mask_pil, dtype=np.int32)
    arr_clamped = np.clip(arr, 0, _MAX_RAW_VAL)
    return _MASK_LUT[arr_clamped]


# ============================================================================
# Augmentation Pipelines
# ============================================================================

def get_train_transforms(img_size=IMG_SIZE):
    """
    Training augmentation — Round 4 (multi-scale + aggressive for IoU push).
    
    Key changes from R3:
    - Multi-scale: resize to 1.5× base, then RandomResizedCrop
      → objects appear at 0.3× to 1.0× of enlarged size
      → gives both zoom-in AND zoom-out variation
    - Slightly stronger photometric augmentation
    - Rest unchanged (already strong from R3)
    """
    # Multi-scale base: resize to 1.5× then crop to target
    # This creates BOTH zoom-in and zoom-out effects:
    #   scale=1.0: crop full 768→512 (zoom out, objects smaller)
    #   scale=0.3: crop 230→512 (zoom in, objects larger)
    enlarged_h = int(img_size[0] * 1.5)
    enlarged_w = int(img_size[1] * 1.5)

    return A.Compose([
        # --- Multi-scale resize + crop (replaces fixed Resize) ---
        A.Resize(height=enlarged_h, width=enlarged_w),
        A.RandomResizedCrop(
            size=(img_size[0], img_size[1]),
            scale=(0.3, 1.0),    # ⚡ R4: Full multi-scale range
            ratio=(0.75, 1.33),
            p=1.0
        ),

        # --- Geometric Augmentations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.25),

        # Additional geometric variation
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.3
        ),

        # Elastic/Grid distortion — helps with deformable objects
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=8, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.2),

        # --- Photometric Augmentations (stronger for desert) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.5
        ),

        # CLAHE: Adaptive histogram equalization (great for desert scenes!)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.2),

        # --- Color space augmentations ---
        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.1),

        # --- Regularization ---
        A.CoarseDropout(
            num_holes_range=(3, 10),
            hole_height_range=(20, 48),
            hole_width_range=(20, 48),
            p=0.4
        ),

        # --- Normalize + Tensor ---
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    """Validation — just resize, normalize, tensor."""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=IMG_SIZE):
    """
    Test-Time Augmentation transforms — Round 4.
    
    Multi-scale + flip TTA for better predictions at inference.
    Average softmax predictions across all variants → take argmax.
    Expected boost: +0.01–0.015 mIoU (free, no retraining needed).
    """
    transforms = []

    # 1. Original
    transforms.append(A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]))

    # 2. Horizontal flip
    transforms.append(A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]))

    # 3-5. Multi-scale (0.75×, 1.25×, 1.5×)
    for scale in [0.75, 1.25, 1.5]:
        h = int(img_size[0] * scale)
        w = int(img_size[1] * scale)
        transforms.append(A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]))

    # 6. Flip + larger scale
    transforms.append(A.Compose([
        A.Resize(height=int(img_size[0] * 1.25), width=int(img_size[1] * 1.25)),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]))

    return transforms


# ============================================================================
# Dataset Class with CopyPaste Support
# ============================================================================

class OffroadSegDataset(Dataset):
    """
    Off-Road Desert Segmentation Dataset.

    Expects:
        data_dir/
            Color_Images/    ← RGB images (.png)
            Segmentation/    ← Mask images (.png, 16-bit)
    
    ROUND 4: Optional CopyPaste augmentation for rare classes.
    Copies rare class pixels from random donor images and pastes them
    onto the current image. This dramatically increases the effective
    representation of classes like Logs (0.1%) and Ground Clutter (4.4%).
    """

    def __init__(self, data_dir, transform=None, return_filename=False,
                 copy_paste=False, copy_paste_prob=0.3, rare_classes=None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.return_filename = return_filename
        self.copy_paste = copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.rare_classes = rare_classes or [4, 5, 7, 8]  # Dry Bushes, Ground Clutter, Logs, Rocks

        # Get sorted file list
        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Verify mask files exist
        missing = [f for f in self.filenames
                   if not os.path.exists(os.path.join(self.mask_dir, f))]
        if missing:
            print(f"[WARNING] {len(missing)} images have no matching mask!")
            self.filenames = [f for f in self.filenames if f not in set(missing)]

        print(f"[INFO] Loaded {len(self.filenames)} samples from {data_dir}")

        # Build rare class index for CopyPaste
        self.rare_index = {}
        if self.copy_paste:
            self._build_rare_index()

    def _build_rare_index(self):
        """Pre-index which images contain each rare class for CopyPaste."""
        from tqdm import tqdm

        self.rare_index = {cls: [] for cls in self.rare_classes}

        print(f"[INFO] Building CopyPaste index for rare classes...")
        for idx in tqdm(range(len(self.filenames)), desc="Indexing rare classes", leave=False):
            mask_path = os.path.join(self.mask_dir, self.filenames[idx])
            mask_pil = Image.open(mask_path)
            mask = convert_mask(mask_pil)
            unique_classes = set(np.unique(mask))
            for cls in self.rare_classes:
                if cls in unique_classes:
                    self.rare_index[cls].append(idx)

        class_names = [
            "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
            "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
        ]
        print(f"[INFO] CopyPaste rare class index:")
        for cls in self.rare_classes:
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            print(f"  {name} (class {cls}): found in {len(self.rare_index[cls])} images")

    def _load_raw(self, idx):
        """Load raw image and mask as numpy arrays (no transforms)."""
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_path = os.path.join(self.mask_dir, fname)
        mask_pil = Image.open(mask_path)
        mask = convert_mask(mask_pil)

        return image, mask

    def _apply_copy_paste(self, image, mask):
        """
        Copy rare class pixels from a random donor image and paste onto current image.
        
        This is the standard CopyPaste augmentation adapted for semantic segmentation.
        It dramatically helps rare classes by artificially increasing their pixel count
        in each training batch.
        """
        # Pick a random rare class
        cls = random.choice(self.rare_classes)
        if not self.rare_index.get(cls):
            return image, mask

        # Pick a random donor image containing that class
        donor_idx = random.choice(self.rare_index[cls])
        donor_image, donor_mask = self._load_raw(donor_idx)

        # Resize donor to match current image dimensions
        h, w = image.shape[:2]
        dh, dw = donor_image.shape[:2]
        if (dh, dw) != (h, w):
            donor_image = np.array(
                Image.fromarray(donor_image).resize((w, h), Image.BILINEAR)
            )
            donor_mask = np.array(
                Image.fromarray(donor_mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
            )

        # Extract pixels of the rare class and paste them
        paste_pixels = (donor_mask == cls)

        if paste_pixels.sum() > 10:  # Only paste if meaningful number of pixels
            # Random horizontal flip of pasted content for variety
            if random.random() > 0.5:
                donor_image = np.fliplr(donor_image).copy()
                donor_mask = np.fliplr(donor_mask).copy()
                paste_pixels = np.fliplr(paste_pixels).copy()

            image[paste_pixels] = donor_image[paste_pixels]
            mask[paste_pixels] = cls

        return image, mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load raw image + mask
        image, mask = self._load_raw(idx)

        # CopyPaste augmentation (BEFORE spatial transforms)
        if self.copy_paste and random.random() < self.copy_paste_prob:
            image, mask = self._apply_copy_paste(image, mask)

        # Apply augmentations (resize, crop, normalize, etc.)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        if self.return_filename:
            return image, mask, fname
        return image, mask


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    from config import TRAIN_DIR, VAL_DIR

    train_ds = OffroadSegDataset(TRAIN_DIR, transform=get_train_transforms(), copy_paste=True)
    val_ds = OffroadSegDataset(VAL_DIR, transform=get_val_transforms())

    img, mask = train_ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape:  {mask.shape}")
    print(f"Mask classes: {mask.unique()}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
