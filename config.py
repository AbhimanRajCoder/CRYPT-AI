"""
config.py — Central configuration for Off-Road Desert Segmentation

ROUND 4 — PUSH TO 0.60+ mIoU (from epoch 47 checkpoint, mIoU=0.5201):
  - KEPT: ResNet50 backbone (resume-compatible with existing checkpoint)
  - NEW: Lovász-Softmax loss (direct IoU optimization)
  - NEW: IoU-adaptive class weights (auto-boost underperforming classes)
  - NEW: CopyPaste augmentation for rare classes
  - TUNED: OHEM ratio 0.7 (less aggressive filtering)
  - TUNED: Loss mix = Lovász(0.3) + Dice(0.2) + Focal(0.3) + Boundary(0.2)
  - Target: mIoU 0.60+
"""

import os
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Offroad_Segmentation_Training_Dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PREDICTION_DIR = os.path.join(BASE_DIR, "predictions")

for d in [CHECKPOINT_DIR, OUTPUT_DIR, PREDICTION_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# Device Configuration
# ============================================================================
def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.ones(1).to("mps")
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

DEVICE = get_device()
USE_CUDA = DEVICE.type == "cuda"
USE_MPS = DEVICE.type == "mps"

# ============================================================================
# Class Definitions (11 classes including Background)
# NOTE: Background (class 0) has 0 pixels in this dataset and is IGNORED
# ============================================================================
VALUE_MAP = {
    0:     0,   # Background (ignored in loss — 0 pixels)
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

NUM_CLASSES = len(CLASS_NAMES)  # 11

# IGNORE_INDEX: Class 0 (Background) has 0 pixels, exclude from loss & metrics
IGNORE_INDEX = 0

# Color palette for visualization (RGB)
COLOR_PALETTE = [
    [0,   0,   0],      # Background — black
    [34,  139, 34],     # Trees — forest green
    [0,   255, 0],      # Lush Bushes — lime
    [210, 180, 140],    # Dry Grass — tan
    [139, 90,  43],     # Dry Bushes — brown
    [128, 128, 0],      # Ground Clutter — olive
    [255, 20,  147],    # Flowers — deep pink
    [139, 69,  19],     # Logs — saddle brown
    [128, 128, 128],    # Rocks — gray
    [160, 82,  45],     # Landscape — sienna
    [135, 206, 235],    # Sky — sky blue
]

# ============================================================================
# Training Hyperparameters — ROUND 4 PUSH TO 0.60+
# ============================================================================
IMG_SIZE = (512, 512)           # ⚡ Upgraded from 448: better for small objects

if USE_CUDA:
    BATCH_SIZE = 12             # Adjusted for 512×512 + ResNet50
    NUM_WORKERS = 2
    BACKBONE = "resnet50"       # ⚡ KEPT: resume-compatible with existing checkpoint
elif USE_MPS:
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    BACKBONE = "resnet50"
else:
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    BACKBONE = "resnet50"

GRAD_ACCUM_STEPS = 2            # Effective batch = 24 (12×2)

NUM_EPOCHS = 80                 # ⚡ R4: More epochs for convergence
LEARNING_RATE = 2e-5            # ⚡ Proper fine-tuning LR (was 2.3e-4 = too high!)
WEIGHT_DECAY = 1e-4
PRETRAINED = True

# Loss Configuration — ROUND 4 OPTIMIZED MIX
DICE_WEIGHT = 0.2               # ⚡ R4: Reduced (Lovász also optimizes IoU)
CE_WEIGHT = 0.3                 # Focal for hard pixel mining
LOVASZ_WEIGHT = 0.3             # ⚡ R4 NEW: Direct IoU optimization
FOCAL_GAMMA = 2.0               # ⚡ Reduced from 3.0 (3.0 can suppress gradients too much)
AUX_LOSS_WEIGHT = 0.0           # Disabled: checkpoint has no aux classifier
USE_OHEM = True                 # ⚡ NEW: Online Hard Example Mining
OHEM_RATIO = 0.7                # ⚡ R4: Keep 70% (was 50% — too aggressive)
BOUNDARY_WEIGHT = 0.2           # ⚡ NEW: Boundary-aware loss component

# LR Scheduler
LR_SCHEDULER = "cosine_warm_restarts"  # ⚡ Warm restarts for escaping local minima
LR_MIN = 1e-7
LR_WARMUP_EPOCHS = 2
LR_RESTART_PERIOD = 10          # Restart every 10 epochs

# ============================================================================
# Data Augmentation
# ============================================================================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ============================================================================
# CopyPaste Augmentation (for rare classes)
# ============================================================================
USE_COPY_PASTE = True           # ⚡ R4 NEW: CopyPaste for rare classes
COPY_PASTE_PROB = 0.3           # Probability of pasting per sample
RARE_CLASSES = [4, 5, 7, 8]     # Dry Bushes, Ground Clutter, Logs, Rocks

# ============================================================================
# Logging
# ============================================================================
PRINT_FREQ = 50
SAVE_BEST_ONLY = True
EARLY_STOPPING_PATIENCE = 25   # ⚡ R4: More patience for longer training
