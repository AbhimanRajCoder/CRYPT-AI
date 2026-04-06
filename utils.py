"""
utils.py — Helper utilities for the segmentation project.

SPEED OPTIMIZATIONS:
  - Removed deterministic mode (was slowing down training)
  - set_seed now allows cudnn.benchmark = True
"""

import os
import random
import numpy as np
import torch
from config import COLOR_PALETTE, NUM_CLASSES


def set_seed(seed=42):
    """Set random seeds for reproducibility WITHOUT sacrificing speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ⚡ Do NOT set deterministic=True or benchmark=False
    # Those cause ~15-30% slowdown for minimal reproducibility gain
    print(f"[INFO] Random seed set to {seed}")


def save_checkpoint(state, filepath):
    """Save model checkpoint with automatic Google Drive sync for Colab."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[INFO] Checkpoint saved to {filepath}")
    
    # --- Google Drive Sync ---
    # Detect if we are in Colab and if Drive is mounted
    drive_mount = "/content/drive/MyDrive"
    if os.path.exists(drive_mount):
        # Create unique project folder in Drive
        drive_path = os.path.join(drive_mount, "Offroad_Segmentation", "checkpoints")
        os.makedirs(drive_path, exist_ok=True)
        
        fname = os.path.basename(filepath)
        drive_dest = os.path.join(drive_path, fname)
        
        try:
            import shutil
            shutil.copy2(filepath, drive_dest)
            print(f"[INFO] ☁️ Synced to Google Drive: {drive_dest}")
        except Exception as e:
            print(f"[WARNING] Google Drive sync failed: {e}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """Load model checkpoint with corruption check."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Checkpoint file NOT found: {filepath}")
    
    # Check if file is empty or too small (corrupted)
    file_size = os.path.getsize(filepath)
    if file_size < 1000:  # A real checkpoint should be several MBs
        raise RuntimeError(f"[ERROR] Checkpoint file is too small ({file_size} bytes). It is likely corrupted or incomplete.")

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load checkpoint. The file might be corrupted: {e}")
        
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_iou = checkpoint.get("best_iou", 0.0)
    print(f"[INFO] Loaded checkpoint from {filepath} (epoch {epoch}, best IoU: {best_iou:.4f})")
    return epoch, best_iou


def mask_to_color(mask):
    """Convert a class-index mask (H, W) to an RGB color image (H, W, 3)."""
    palette = np.array(COLOR_PALETTE, dtype=np.uint8)
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        color_img[mask == cls_id] = palette[cls_id]
    return color_img


def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize an image tensor back to [0, 1] range for display."""
    mean = np.array(mean)
    std = np.array(std)
    img = img_tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1)  # CHW → HWC
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


class AverageMeter:
    """Tracks running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
