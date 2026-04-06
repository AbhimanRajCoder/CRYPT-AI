"""
losses.py — Loss functions optimized for class imbalance.

ROUND 4 UPGRADES (targeting 0.60+ mIoU):
  - Lovász-Softmax loss: DIRECTLY optimizes the IoU metric
  - IoU-adaptive class weights: auto-boost classes based on their current IoU
  - OHEM ratio 0.7 (less aggressive, keeps more useful gradients)
  - Loss mix: Lovász(0.3) + Dice(0.2) + Focal(0.3) + Boundary(0.2)
  - Background (class 0) fully excluded via ignore_index=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DICE_WEIGHT, CE_WEIGHT, FOCAL_GAMMA, IGNORE_INDEX


# ============================================================================
# Lovász-Softmax Loss — Direct IoU Optimization
# Reference: "The Lovász-Softmax loss" (Berman et al., CVPR 2018)
# ============================================================================

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors.
    
    This is the key mathematical insight: the Lovász extension provides
    a tight convex surrogate for the IoU loss.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for multi-class semantic segmentation.
    
    Unlike Dice/CE which are pixel-level surrogates, Lovász directly
    optimizes the mean IoU metric. This is critical for pushing past
    the 0.52 plateau.
    
    Key advantage: naturally handles class imbalance because IoU is
    computed per-class then averaged.
    """

    def __init__(self, ignore_index=IGNORE_INDEX, per_image=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, pred, target):
        probas = F.softmax(pred, dim=1)

        if self.per_image:
            losses = []
            for i in range(pred.shape[0]):
                loss = self._lovasz_softmax_flat(
                    probas[i:i+1], target[i:i+1]
                )
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            return self._lovasz_softmax_flat(probas, target)

    def _lovasz_softmax_flat(self, probas, target):
        B, C, H, W = probas.shape
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.reshape(-1)

        # Remove ignored pixels
        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = (target_flat != self.ignore_index)
            if valid.sum() == 0:
                return torch.tensor(0.0, device=probas.device, requires_grad=True)
            probas_flat = probas_flat[valid]
            target_flat = target_flat[valid]

        losses = []
        for c in range(C):
            if c == self.ignore_index:
                continue
            fg = (target_flat == c).float()
            if fg.sum() == 0:
                continue  # Skip classes not present in this batch
            errors = (fg - probas_flat[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if len(losses) == 0:
            return torch.tensor(0.0, device=probas.device, requires_grad=True)
        return torch.stack(losses).mean()


# ============================================================================
# Class-Weighted Dice Loss
# ============================================================================

class ClassWeightedDiceLoss(nn.Module):
    """
    Class-Weighted Dice Loss — gives rare classes proportionally more gradient.
    
    CRITICAL: Standard Dice loss averages over all classes equally, so Logs (0.1% pixels)
    gets the same weight as Sky (37.6% pixels). With class weights, we can boost rare classes.
    """

    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-6, ignore_index=IGNORE_INDEX, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Create mask to ignore background
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index).unsqueeze(1).float()
            pred_soft = pred_soft * valid_mask
            target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (pred_soft * target_one_hot).sum(dim=dims)
        cardinality = pred_soft.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Exclude background from computation
        valid_classes = [i for i in range(self.num_classes) if i != self.ignore_index]
        dice_score = dice_score[valid_classes]

        # Apply class weights to Dice loss (KEY IMPROVEMENT)
        if self.class_weights is not None:
            w = self.class_weights[valid_classes]
            w = w.to(dice_score.device)
            w = w / w.sum()
            dice_loss = 1.0 - (w * dice_score).sum()
        else:
            dice_loss = 1.0 - dice_score.mean()

        return dice_loss


# ============================================================================
# OHEM Focal Loss
# ============================================================================

class OHEMFocalLoss(nn.Module):
    """
    Online Hard Example Mining + Focal Loss.
    
    Only backpropagates through the hardest pixels (highest loss).
    R4: ratio=0.7 keeps 70% of pixels (was 50% — too aggressive).
    """

    def __init__(self, gamma=FOCAL_GAMMA, weight=None, ignore_index=IGNORE_INDEX, ohem_ratio=0.7):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ohem_ratio = ohem_ratio

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # OHEM: only keep the hardest pixels
        if self.ohem_ratio < 1.0:
            valid_mask = (target != self.ignore_index).view(-1)
            focal_flat = focal_loss.view(-1)
            valid_losses = focal_flat[valid_mask]

            if valid_losses.numel() > 0:
                num_hard = max(int(valid_losses.numel() * self.ohem_ratio), 1)
                hard_losses, _ = torch.topk(valid_losses, num_hard)
                return hard_losses.mean()

        return focal_loss.mean()


class FocalLoss(nn.Module):
    """Standard Focal Loss (fallback without OHEM)."""

    def __init__(self, gamma=FOCAL_GAMMA, weight=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# Boundary Loss
# ============================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss — penalizes errors at class boundaries.
    
    Detects boundary pixels using Laplacian edge detection on the mask,
    then applies extra loss weight to those pixels.
    """

    def __init__(self, weight=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        target_float = target.float().unsqueeze(1)  # (B, 1, H, W)

        # Sobel-like boundary detection
        padded = F.pad(target_float, (1, 1, 1, 1), mode='replicate')

        shifts = [
            padded[:, :, 1:-1, 2:],   # right
            padded[:, :, 1:-1, :-2],   # left
            padded[:, :, 2:, 1:-1],    # down
            padded[:, :, :-2, 1:-1],   # up
        ]

        boundary = torch.zeros_like(target_float)
        for s in shifts:
            boundary = boundary + (s != target_float).float()

        boundary = (boundary > 0).float().squeeze(1)  # (B, H, W)

        # CE loss with boundary weighting
        ce_loss = F.cross_entropy(
            pred, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none"
        )

        # Boost loss on boundary pixels (2× weight on boundaries)
        boundary_weight = 1.0 + boundary * 2.0
        weighted_loss = ce_loss * boundary_weight

        return weighted_loss.mean()


# ============================================================================
# Combined Loss — Round 4
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss for Round 4 — targeting 0.60+ mIoU:
      - Lovász-Softmax (direct IoU optimization — the KEY addition)
      - Class-weighted Dice (weighted by class importance)
      - OHEM Focal (hard pixel mining, class-weighted)
      - Boundary loss (focus on class borders)
    
    Default mix: Lovász(0.3) + Dice(0.2) + Focal(0.3) + Boundary(0.2)
    All components ignore Background (class 0).
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dice_weight=DICE_WEIGHT,
        ce_weight=CE_WEIGHT,
        lovasz_weight=0.0,
        boundary_weight=0.0,
        class_weights=None,
        focal_gamma=FOCAL_GAMMA,
        use_ohem=False,
        ohem_ratio=0.7,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight

        # Lovász-Softmax (NEW in R4)
        self.lovasz_loss = None
        if lovasz_weight > 0:
            self.lovasz_loss = LovaszSoftmaxLoss(
                ignore_index=IGNORE_INDEX, per_image=False
            )
            print(f"[INFO] Lovász-Softmax Loss enabled (weight={lovasz_weight})")

        # Class-weighted Dice
        self.dice_loss = ClassWeightedDiceLoss(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            class_weights=class_weights,
        )

        # Focal with optional OHEM
        if use_ohem:
            self.ce_loss = OHEMFocalLoss(
                gamma=focal_gamma, weight=class_weights,
                ignore_index=IGNORE_INDEX, ohem_ratio=ohem_ratio
            )
            print(f"[INFO] Using OHEM Focal Loss (ratio={ohem_ratio}, γ={focal_gamma})")
        elif focal_gamma > 0:
            self.ce_loss = FocalLoss(gamma=focal_gamma, weight=class_weights, ignore_index=IGNORE_INDEX)
            print(f"[INFO] Using Focal Loss (γ={focal_gamma})")
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
            print(f"[INFO] Using CrossEntropy Loss")

        # Boundary loss
        self.boundary_loss_fn = None
        if boundary_weight > 0:
            self.boundary_loss_fn = BoundaryLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
            print(f"[INFO] Boundary Loss enabled (weight={boundary_weight})")

        # Print summary
        parts = []
        if lovasz_weight > 0: parts.append(f"Lovász({lovasz_weight})")
        if dice_weight > 0: parts.append(f"Dice({dice_weight})")
        if ce_weight > 0: parts.append(f"Focal/CE({ce_weight})")
        if boundary_weight > 0: parts.append(f"Boundary({boundary_weight})")
        print(f"[INFO] Combined Loss: {' + '.join(parts)}")
        print(f"[INFO] Background (class 0) IGNORED in all loss components")

    def forward(self, pred, target):
        total = 0.0

        if self.dice_weight > 0:
            d_loss = self.dice_loss(pred, target)
            total = total + self.dice_weight * d_loss

        if self.ce_weight > 0:
            c_loss = self.ce_loss(pred, target)
            total = total + self.ce_weight * c_loss

        if self.lovasz_loss is not None and self.lovasz_weight > 0:
            l_loss = self.lovasz_loss(pred, target)
            total = total + self.lovasz_weight * l_loss

        if self.boundary_loss_fn is not None and self.boundary_weight > 0:
            b_loss = self.boundary_loss_fn(pred, target)
            total = total + self.boundary_weight * b_loss

        return total


# ============================================================================
# IoU-Adaptive Class Weight Computation — Round 4
# ============================================================================

def compute_class_weights(dataset, num_classes=NUM_CLASSES, max_samples=None):
    """
    Compute class weights using sqrt-inverse-frequency + IoU-adaptive boosting.
    
    ROUND 4 IMPROVEMENTS:
      - IoU-adaptive boost: classes with lower IoU get exponentially higher weight
      - This automatically focuses training on Ground Clutter, Rocks, Logs
      - Landscape (24.4% pixels, IoU=0.475) finally gets proper weighting
      - Background weight = 0 (excluded from loss)
    """
    from config import CLASS_NAMES
    import numpy as np
    import os
    from PIL import Image
    from tqdm import tqdm
    from dataset import convert_mask

    if max_samples is None:
        n = len(dataset)
    else:
        n = min(len(dataset), max_samples)

    print(f"[INFO] Computing class weights from {n}/{len(dataset)} samples...")

    class_counts = np.zeros(num_classes, dtype=np.int64)

    mask_dir = dataset.mask_dir
    filenames = dataset.filenames[:n]

    for fname in tqdm(filenames, desc="Scanning ALL classes", leave=False):
        mask_path = os.path.join(mask_dir, fname)
        mask_pil = Image.open(mask_path)
        mask = convert_mask(mask_pil)
        counts = np.bincount(mask.flatten(), minlength=num_classes)
        class_counts += counts[:num_classes]

    total = class_counts.sum()

    # Print raw distribution
    zero_classes = []
    print("[INFO] Raw class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        pct = 100 * count / total if total > 0 else 0
        print(f"  [{i}] {name:<20}: {pct:5.1f}% ({count:,} px)")
        if count == 0:
            zero_classes.append(name)

    if zero_classes:
        print(f"[WARNING] Classes with ZERO pixels: {zero_classes}")

    # ----------------------------------------------------------------
    # Step 1: sqrt-inverse frequency (EXCLUDE Background)
    # ----------------------------------------------------------------
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[0] = 0.0  # Background → ignored

    non_bg_total = class_counts[1:].sum()
    for i in range(1, num_classes):
        freq = class_counts[i] / (non_bg_total + 1e-6)
        weights[i] = 1.0 / np.sqrt(freq + 1e-6)

    # Normalize so mean = 1
    non_bg_weights = weights[1:]
    non_bg_weights = non_bg_weights / non_bg_weights.mean()
    weights[1:] = non_bg_weights

    # ----------------------------------------------------------------
    # Step 2: IoU-ADAPTIVE BOOST (Round 4 — key improvement!)
    # 
    # Classes below target IoU get boosted proportionally to how far
    # they are from the target. This automatically focuses on:
    #   Ground Clutter (0.270 → big boost)
    #   Rocks (0.361 → big boost)
    #   Logs (0.384 → medium boost)
    #   Dry Bushes (0.440 → medium boost)
    #   Landscape (0.475 → moderate boost)  ← was UNDER-WEIGHTED!
    # ----------------------------------------------------------------
    latest_iou = {
        1: 0.626,   # Trees
        2: 0.513,   # Lush Bushes
        3: 0.592,   # Dry Grass
        4: 0.440,   # Dry Bushes
        5: 0.270,   # Ground Clutter  ← worst!
        6: 0.580,   # Flowers
        7: 0.384,   # Logs
        8: 0.361,   # Rocks
        9: 0.475,   # Landscape       ← under-weighted before!
        10: 0.979,  # Sky
    }

    target_iou = 0.60
    boost = np.ones(num_classes)
    boost[0] = 0.0  # Background — IGNORE

    for cls_id, iou in latest_iou.items():
        if iou < target_iou:
            # Quadratic-ish penalty for being below target
            # Ground Clutter: (0.60/0.27)^1.5 ≈ 3.30
            # Rocks:          (0.60/0.36)^1.5 ≈ 2.15
            # Logs:           (0.60/0.38)^1.5 ≈ 1.97
            # Dry Bushes:     (0.60/0.44)^1.5 ≈ 1.59
            # Landscape:      (0.60/0.475)^1.5 ≈ 1.41
            # Lush Bushes:    (0.60/0.51)^1.5 ≈ 1.27
            boost[cls_id] = (target_iou / max(iou, 0.1)) ** 1.5
        else:
            # Slight down-weight for saturated classes (Sky, Trees)
            boost[cls_id] = 0.8

    weights = weights * boost

    # Re-normalize non-background
    non_bg_weights = weights[1:]
    non_bg_weights = non_bg_weights / non_bg_weights.mean()
    weights[1:] = non_bg_weights

    # Wider clip range for more differentiation
    weights[1:] = np.clip(weights[1:], 0.15, 8.0)
    weights[0] = 0.0

    print("\n[INFO] Final class weights (IoU-adaptive, Background excluded):")
    for name, w, count in zip(CLASS_NAMES, weights, class_counts):
        pct = 100 * count / total if total > 0 else 0
        iou_str = ""
        cls_idx = CLASS_NAMES.index(name)
        if cls_idx in latest_iou:
            iou_str = f" [IoU={latest_iou[cls_idx]:.3f}, boost={boost[cls_idx]:.2f}×]"
        status = "IGNORED" if w == 0 else f"weight={w:.3f}"
        print(f"  {name:<20}: {status} ({pct:.1f}% of pixels){iou_str}")

    return torch.FloatTensor(weights)
