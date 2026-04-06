"""
metrics.py — Evaluation metrics for semantic segmentation.

ROUND 3 FIX:
  - compute_iou_per_class and mean now EXCLUDE Background (class 0) from mIoU
  - This gives a truer metric since Background has 0 pixels and always returns NaN
  - Per-class IoU still computed for all classes (NaN for absent classes)
"""

import numpy as np
import torch
from config import NUM_CLASSES, CLASS_NAMES, IGNORE_INDEX


def compute_iou_per_class(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """
    Compute IoU for each class.
    
    Args:
        pred: (B, C, H, W) logits OR (B, H, W) predictions
        target: (B, H, W) ground truth
        ignore_index: class index to exclude from mean computation
    Returns:
        per_class_iou (list), mean_iou (float, excluding ignored classes)
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    per_class_iou = []
    for cls in range(num_classes):
        pred_mask = (pred_flat == cls)
        target_mask = (target_flat == cls)

        intersection = (pred_mask & target_mask).sum().float().item()
        union = (pred_mask | target_mask).sum().float().item()

        if union == 0:
            per_class_iou.append(float('nan'))
        else:
            per_class_iou.append(intersection / union)

    # ⚡ FIXED: Exclude ignored class from mean IoU computation
    valid_ious = [per_class_iou[i] for i in range(num_classes) if i != ignore_index]
    mean_iou = np.nanmean(valid_ious)
    return per_class_iou, mean_iou


def compute_dice_per_class(pred, target, num_classes=NUM_CLASSES, smooth=1e-6, ignore_index=IGNORE_INDEX):
    """Compute Dice coefficient for each class (excluding ignored class from mean)."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    per_class_dice = []
    for cls in range(num_classes):
        pred_mask = (pred_flat == cls)
        target_mask = (target_flat == cls)

        intersection = (pred_mask & target_mask).sum().float().item()
        total = pred_mask.sum().float().item() + target_mask.sum().float().item()

        dice = (2.0 * intersection + smooth) / (total + smooth)
        per_class_dice.append(dice)

    # Exclude ignored class from mean
    valid_dices = [per_class_dice[i] for i in range(num_classes) if i != ignore_index]
    mean_dice = np.nanmean(valid_dices)
    return per_class_dice, mean_dice


def compute_pixel_accuracy(pred, target, ignore_index=IGNORE_INDEX):
    """Compute pixel accuracy, excluding ignored class pixels."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    # Only count pixels that are not the ignored class
    valid_mask = (target != ignore_index)
    correct = ((pred == target) & valid_mask).sum().float().item()
    total = valid_mask.sum().float().item()
    return correct / (total + 1e-6)


def compute_confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """Compute confusion matrix using vectorized ops (fast)."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    pred_flat = pred.view(-1).cpu().long()
    target_flat = target.view(-1).cpu().long()

    mask = (target_flat >= 0) & (target_flat < num_classes) & \
           (pred_flat >= 0) & (pred_flat < num_classes)
    indices = target_flat[mask] * num_classes + pred_flat[mask]
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes).numpy()

    return cm


def print_metrics_table(per_class_iou, per_class_dice, pixel_acc, ignore_index=IGNORE_INDEX):
    """Pretty-print metrics as a table, clearly marking ignored classes."""
    print("\n" + "=" * 65)
    print(f"{'Class':<20} {'IoU':>10} {'Dice':>10}")
    print("-" * 65)

    for i, name in enumerate(CLASS_NAMES):
        if i == ignore_index:
            print(f"{name:<20} {'IGNORED':>10} {'IGNORED':>10}")
            continue
        iou_str = f"{per_class_iou[i]:.4f}" if not np.isnan(per_class_iou[i]) else "  N/A"
        dice_str = f"{per_class_dice[i]:.4f}" if not np.isnan(per_class_dice[i]) else "  N/A"
        print(f"{name:<20} {iou_str:>10} {dice_str:>10}")

    print("-" * 65)
    # Mean excluding ignored class
    valid_ious = [per_class_iou[i] for i in range(len(per_class_iou)) if i != ignore_index]
    valid_dices = [per_class_dice[i] for i in range(len(per_class_dice)) if i != ignore_index]
    mean_iou = np.nanmean(valid_ious)
    mean_dice = np.nanmean(valid_dices)
    print(f"{'Mean (excl. BG)':<20} {mean_iou:>10.4f} {mean_dice:>10.4f}")
    print(f"{'Pixel Accuracy':<20} {pixel_acc:>10.4f}")
    print("=" * 65)
