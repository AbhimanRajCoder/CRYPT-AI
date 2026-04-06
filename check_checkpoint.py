
import torch
import os
from models import SegmentationModel
from utils import load_checkpoint
from config import DEVICE, CHECKPOINT_DIR

def check():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_40.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Found checkpoint at {checkpoint_path}")
    model = SegmentationModel(use_aux=False)
    
    try:
        start_epoch, best_iou = load_checkpoint(checkpoint_path, model, device=DEVICE)
        print(f"Successfully loaded checkpoint!")
        print(f"Start Epoch: {start_epoch}")
        print(f"Best IoU: {best_iou:.4f}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

if __name__ == "__main__":
    check()
