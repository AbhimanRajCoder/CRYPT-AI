# 🏜️ Off-Road Desert Semantic Segmentation
### Duality AI Offroad Autonomy Segmentation Challenge
**Presentation — 4:30 PM**

---

---

## SLIDE 1 — SYNOPSIS (Problem Statement)

### 🎯 "Teaching a Robot to See the Desert"

**The Challenge**
> Given a synthetic RGB image from a desert digital twin (Falcon simulator), predict the class of every single pixel — in real time.

**Why Does This Matter?**
- Autonomous vehicles in off-road/military terrain cannot rely on GPS or maps
- They need to *see and understand* the environment pixel-by-pixel to navigate safely
- Bad segmentation = vehicle drives into a rock or off a cliff

**The Setting**
- Dataset: **2,857 training images + 317 validation images** from Duality AI's **Falcon digital twin** simulator
- No external data allowed — purely synthetic desert environments
- Evaluation: **Mean IoU (mIoU)** across all 10 terrain classes

**Our Goal**
> Build a model that generalizes to *unseen* desert environments with mIoU > 0.55 → 0.60+

---

---

## SLIDE 2 — ARCHITECTURE

### 🏗️ The Model: DeepLabV3+ with ResNet50

```
Input Image (3 × 512 × 512)
        │
   ┌────▼───────────┐
   │   ResNet50      │ ← Pretrained on COCO/ImageNet
   │   Backbone      │   Hierarchical feature extraction
   │   (39.6M params)│   (edges → textures → objects)
   └────┬───────────┘
        │
   ┌────▼────────────────┐
   │    ASPP Module       │ ← Atrous Spatial Pyramid Pooling
   │ Dilation: 1, 6, 12, │   Multi-scale context capture
   │ 18, GlobalAvgPool   │   Tiny rocks  Large sky  Wide landscape
   └────┬────────────────┘
        │
   ┌────▼───────────┐
   │  Classifier     │ ← Conv2d(256 → 10 classes)
   │  Head           │   Per-pixel class assignment
   └────┬───────────┘
        │
  Segmentation Mask (10 × 512 × 512)
        │
  Argmax → Final prediction per pixel
```

**Why DeepLabV3+?**
| Feature | Benefit |
|---------|---------|
| ASPP (multi-scale) | Handles vast sky AND tiny logs in the same scene |
| Pretrained backbone | Strong feature extraction from Day 1 |
| Dilated convolutions | Large receptive field without downsampling |
| Proven SOTA | Benchmark winner on COCO, Pascal VOC |

**Key Config**
- Input: **512 × 512** (upgraded from 448 for better small-object coverage)
- Optimizer: **AdamW** (weight decay = 1e-4)
- Scheduler: **CosineAnnealingWarmRestarts** (restarts every 10 epochs)
- Mixed Precision: **FP16** on CUDA (2x speedup)

---

---

## SLIDE 3 — STATISTICS & RESULTS

### 📊 Dataset & Class Distribution

| Class | Pixel % | Weight | Phase 2 IoU | Phase 3 IoU |
|-------|---------|--------|-------------|-------------|
| Sky | 37.6% | 0.20 | 0.974 ✅ | 0.978 ✅ |
| Landscape | 24.4% | 0.20 | 0.560 ✅ | ~0.475 ⚠️ |
| Dry Grass | 18.9% | 0.20 | 0.602 ✅ | ~0.590 ✅ |
| Lush Bushes | 5.9% | 0.33 | 0.478 ⚠️ | ~0.474 ⚠️ |
| Ground Clutter | 4.4% | 0.74 | 0.277 ❌ | ~0.270 ❌ |
| Flowers | 2.8% | 0.45 | 0.511 ⚠️ | ~0.562 ✅ |
| Trees | 3.5% | 0.33 | 0.599 ✅ | ~0.611 ✅ |
| Rocks | 1.2% | 1.14 | 0.361 ❌ | ~0.350 ⚠️ |
| Dry Bushes | 1.1% | 1.07 | 0.444 ⚠️ | ~0.423 ⚠️ |
| **Logs** | **0.1%** | **5.57** | **0.317 ❌** | **~0.352 ⚠️** |

### Training Progress
| Phase | Epochs | Best mIoU | Strategy |
|-------|--------|-----------|----------|
| Phase 1 (Scratch) | 1–25 | **0.5123** | ResNet50, Dice + Focal |
| Phase 2 (Fine-Tune) | 26–60 | **In Progress** | OHEM + Boundary + Weighted Dice |

### Runtime Stats (Google Colab T4 GPU)
- Epoch time: ~**285 seconds** (~4.7 min/epoch)
- Batch size: **12** (effective 24 with gradient accumulation)
- Image size: **512 × 512**
- Model size: **~320 MB**

---

---

## SLIDE 4 — PROBLEMS & SOLUTIONS

### 🔥 Real Challenges We Hit (and Fixed)

---

**Problem 1: Extreme Class Imbalance**
> Sky = 37.6% of all pixels. Logs = 0.1%. The model just predicted "Sky" and "Landscape" everywhere and got decent loss — but IoU was catastrophically low for rare classes.

**What we tried first:** Raw inverse-frequency class weights
**What went wrong:** Weights of 50x+ for Logs caused gradient explosion → training diverged

**Solution:**
- Switched to **√inverse-frequency weights** (gentle, stable)
- Added **manual per-class multipliers** (Logs: 5.57x, Rocks: 1.14x)
- Used **OHEM** (Online Hard Example Mining) — only backpropagate through the hardest 50% of pixels per batch. Sky pixels never make the "hard" list.

---

**Problem 2: Torchvision `aux_loss` Incompatibility**
> When loading pretrained ResNet50 weights, Torchvision throws:
> `ValueError: 'aux_loss' expected True but got False`
> The pretrained weights bake in the aux classifier; you can't opt-out after the fact.

**Solution:**
- Always load the model *without* specifying `aux_loss` parameter
- Then immediately `model.aux_classifier = None` to remove it programmatically
- Our checkpoint was trained without aux → perfectly compatible ✅

---

**Problem 3: Checkpoint Architecture Mismatch**
> After upgrading to ResNet101, the model failed to load the old ResNet50 checkpoint with:
> `Missing key(s): model.aux_classifier.X.weight`

**Solution:**
- Reverted to **ResNet50** to keep checkpoint compatibility
- Tied `use_aux` flag to `AUX_LOSS_WEIGHT > 0.0` — so setting weight=0 in config automatically prevents the aux head from being built

---

**Problem 4: `NameError: BACKBONE is not defined` during training**
> We added the `BACKBONE` variable to `config.py` but forgot to import it in `train.py`.

**Solution:** Simple fix — added `BACKBONE` to the import block in `train.py`. Lesson: always audit imports when adding new config variables.

---

**Problem 5: mIoU Inflated by Background Class**
> Background class (class 0) has exactly 0 pixels. Its IoU was being calculated as ~1.0 (numerically: 0/0+ε ≈ 1.0), inflating the reported mIoU.

**Solution:**
- Modified `metrics.py`, `losses.py`, and `test.py` to set `IGNORE_INDEX=0`
- Now mIoU is computed only over the 10 real classes — a much truer metric

---

**Problem 6: Learning Rate Too High on Resume**
> Previous runs used LR = 2.3×10⁻⁴ which was a good LR for training from scratch. Resuming with the same LR after 25 epochs caused the model to "unlearn" — IoU dropped.

**Solution:**
- Fine-tuning LR = **2×10⁻⁵** (10x lower)
- Added `--forced LR` logic in `train.py` that overrides the optimizer LR from checkpoint on resume
- Added `CosineAnnealingWarmRestarts` instead of `OneCycleLR` (which is not resume-friendly)

---

---

## SLIDE 5 — KEY LEARNINGS & NEXT STEPS

### 🚀 What We Learned

**On Model Design**
- ASPP + DeepLabV3 is highly effective for multi-scale desert scenes
- Pretrained weights are non-negotiable — training from scratch takes 10× longer
- Resolution matters: 448→512 gives meaningfully more signal for tiny objects (Logs, Rocks)

**On Class Imbalance**
- There is no single silver bullet — you need a **combination**: weighted loss + OHEM + augmentation + resolution
- √inverse frequency is the stable choice; raw inverse explodes
- Boundary-aware loss is a free +0.5-1% by penalizing class border confusion

**On Training Stability**
- Never use `OneCycleLR` if you plan to resume — it hardcodes epoch count
- `CosineAnnealingWarmRestarts` is resume-friendly and helps escape local minima
- Gradient clipping (`max_norm=1.0`) prevents occasional spikes during warm restarts

**On Evaluation**
- Always exclude dummy/empty classes (Background) from mIoU — otherwise your metric lies
- TTA (Test-Time Augmentation) is a **free +2-4% mIoU** at inference — always use it for final submission

### ⏭️ Immediate Next Steps
1. Let training complete to epoch 60 (~2 hours remaining on Colab T4)
2. Run `python test.py --tta` → expect +2-3% mIoU bonus on final score
3. Analyze confusion matrix → identify if Landscape↔DryGrass is still the biggest confusion pair
4. If mIoU < 0.58 after TTA → boost Ground Clutter and Logs weights further in `losses.py`

### 🎯 Target
> **0.51 (baseline) → 0.55+ (fine-tuning) → 0.58-0.60+ (with TTA)**
