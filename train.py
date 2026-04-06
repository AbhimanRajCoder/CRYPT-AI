"""
train.py — Full training pipeline for Off-Road Desert Segmentation.

Usage:
    python train.py                          # Train from scratch
    python train.py --epochs 80 --batch 12   # Custom settings
    python train.py --resume checkpoints/best_model.pth  # Resume/fine-tune

ROUND 4 UPGRADES (targeting 0.60+ mIoU):
    ✓ ResNet101 backbone with auxiliary classifier
    ✓ 512×512 input for better small object detection
    ✓ Auxiliary loss (0.4× weight) for better gradient flow
    ✓ Class-weighted Dice loss (rare classes get proportional gradient)
    ✓ OHEM Focal Loss (only hardest 50% pixels)
    ✓ Boundary-aware loss component
    ✓ CosineAnnealingWarmRestarts scheduler (escape local minima)
    ✓ Gradient accumulation (effective batch=16 with batch=8)
    ✓ Proper fine-tuning LR (2e-5, not 2.3e-4)
    ✓ Background excluded from mIoU metric
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
try:
    from torch import amp
    has_torch_amp = True
except ImportError:
    has_torch_amp = False
from tqdm import tqdm

from config import (
    DEVICE, USE_CUDA, USE_MPS, TRAIN_DIR, VAL_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, GRAD_ACCUM_STEPS, BACKBONE,
    LEARNING_RATE, WEIGHT_DECAY, LR_MIN, LR_WARMUP_EPOCHS, LR_RESTART_PERIOD,
    NUM_CLASSES, CLASS_NAMES, IGNORE_INDEX,
    EARLY_STOPPING_PATIENCE, SAVE_BEST_ONLY, LR_SCHEDULER,
    AUX_LOSS_WEIGHT, USE_OHEM, OHEM_RATIO, BOUNDARY_WEIGHT,
    DICE_WEIGHT, CE_WEIGHT, LOVASZ_WEIGHT, FOCAL_GAMMA,
    USE_COPY_PASTE, COPY_PASTE_PROB, RARE_CLASSES,
)
from dataset import OffroadSegDataset, get_train_transforms, get_val_transforms
from models import SegmentationModel, count_parameters
from losses import CombinedLoss, compute_class_weights
from metrics import compute_iou_per_class, compute_dice_per_class, compute_pixel_accuracy, print_metrics_table
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter
from visualize import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Off-Road Segmentation Model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=GRAD_ACCUM_STEPS,
                        help="Gradient accumulation steps (effective batch = batch × accum)")
    parser.add_argument("--no-aux", action="store_true", help="Disable auxiliary loss")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch, total_epochs, scaler=None, use_amp=False,
                    accum_steps=1, use_step_scheduler=False,
                    aux_criterion=None, aux_weight=AUX_LOSS_WEIGHT):
    """Train one epoch with AMP, gradient accumulation, and auxiliary loss."""
    model.train()
    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [TRAIN]", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            if has_torch_amp:
                with amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    
                    # Handle auxiliary output
                    if isinstance(outputs, tuple):
                        main_out, aux_out = outputs
                        main_loss = criterion(main_out, masks)
                        aux_loss = aux_criterion(aux_out, masks) if aux_criterion else criterion(aux_out, masks)
                        loss = (main_loss + aux_weight * aux_loss) / accum_steps
                    else:
                        main_loss = criterion(outputs, masks)
                        aux_loss = torch.tensor(0.0)
                        loss = main_loss / accum_steps
            else:
                with autocast():
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        main_out, aux_out = outputs
                        main_loss = criterion(main_out, masks)
                        aux_loss = aux_criterion(aux_out, masks) if aux_criterion else criterion(aux_out, masks)
                        loss = (main_loss + aux_weight * aux_loss) / accum_steps
                    else:
                        main_loss = criterion(outputs, masks)
                        aux_loss = torch.tensor(0.0)
                        loss = main_loss / accum_steps
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if use_step_scheduler and scheduler is not None:
                    scheduler.step()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                main_loss = criterion(main_out, masks)
                aux_loss = aux_criterion(aux_out, masks) if aux_criterion else criterion(aux_out, masks)
                loss = (main_loss + aux_weight * aux_loss) / accum_steps
            else:
                main_loss = criterion(outputs, masks)
                aux_loss = torch.tensor(0.0)
                loss = main_loss / accum_steps
            
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if use_step_scheduler and scheduler is not None:
                    scheduler.step()

        loss_meter.update(loss.item() * accum_steps, images.size(0))
        main_loss_meter.update(main_loss.item(), images.size(0))
        if isinstance(outputs, tuple):
            aux_loss_meter.update(aux_loss.item(), images.size(0))
        
        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            main=f"{main_loss_meter.avg:.4f}",
        )

    # Handle leftover gradients
    if len(loader) % accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    """Validate and compute metrics (Background excluded from mIoU)."""
    model.eval()
    loss_meter = AverageMeter()
    all_iou, all_dice, all_acc = [], [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [VAL]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss_meter.update(loss.item(), images.size(0))

        per_iou, mean_iou = compute_iou_per_class(outputs, masks)
        _, mean_dice = compute_dice_per_class(outputs, masks)
        pixel_acc = compute_pixel_accuracy(outputs, masks)

        all_iou.append(per_iou)
        all_dice.append(mean_dice)
        all_acc.append(pixel_acc)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", iou=f"{mean_iou:.4f}")

    avg_class_iou = np.nanmean(all_iou, axis=0)
    
    # Mean IoU excluding Background (class 0)
    valid_ious = [avg_class_iou[i] for i in range(NUM_CLASSES) if i != IGNORE_INDEX]
    mean_iou = np.nanmean(valid_ious)
    
    mean_dice = np.mean(all_dice)
    mean_acc = np.mean(all_acc)

    return loss_meter.avg, mean_iou, mean_dice, mean_acc, avg_class_iou


def main():
    args = parse_args()
    set_seed(args.seed)

    device = DEVICE
    use_amp = USE_CUDA and not args.no_amp
    use_aux = (not args.no_aux) and (AUX_LOSS_WEIGHT > 0.0)

    print(f"\n{'='*60}")
    print(f"  OFF-ROAD DESERT SEGMENTATION — ROUND 4 TRAINING")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch} (effective: {args.batch * args.accum_steps})")
    print(f"  Learning rate:{args.lr}")
    print(f"  Image size:   {IMG_SIZE}")
    print(f"  Backbone:     {BACKBONE}")
    print(f"  Classes:      {NUM_CLASSES} (Background ignored)")
    print(f"  Scheduler:    {LR_SCHEDULER}")
    print(f"  Aux Loss:     {'Enabled (weight=' + str(AUX_LOSS_WEIGHT) + ')' if use_aux else 'Disabled'}")
    print(f"  OHEM:         {'Enabled (ratio=' + str(OHEM_RATIO) + ')' if USE_OHEM else 'Disabled'}")
    print(f"  Lovász:       {'Enabled (weight=' + str(LOVASZ_WEIGHT) + ')' if LOVASZ_WEIGHT > 0 else 'Disabled'}")
    print(f"  Boundary:     {'Enabled (weight=' + str(BOUNDARY_WEIGHT) + ')' if BOUNDARY_WEIGHT > 0 else 'Disabled'}")
    print(f"  CopyPaste:    {'Enabled (prob=' + str(COPY_PASTE_PROB) + ')' if USE_COPY_PASTE else 'Disabled'}")
    print(f"  Mixed Prec:   {'Enabled ⚡' if use_amp else 'Disabled'}")
    print(f"  Grad Accum:   {args.accum_steps} steps")
    print(f"{'='*60}\n")

    # ---- Datasets ----
    train_transform = get_val_transforms() if args.no_augment else get_train_transforms()
    val_transform = get_val_transforms()

    train_ds = OffroadSegDataset(
        TRAIN_DIR, transform=train_transform,
        copy_paste=USE_COPY_PASTE,
        copy_paste_prob=COPY_PASTE_PROB,
        rare_classes=RARE_CLASSES,
    )
    val_ds = OffroadSegDataset(VAL_DIR, transform=val_transform)

    use_persistent = args.workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(USE_CUDA or USE_MPS),
        drop_last=True, persistent_workers=use_persistent,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(USE_CUDA or USE_MPS),
        persistent_workers=use_persistent,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # ---- Model ----
    print("\n[STEP 1] Building model...")
    model = SegmentationModel(use_aux=use_aux)
    count_parameters(model)
    model = model.to(device)

    if USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # ---- Class weights ----
    print("\n[STEP 2] Computing class weights for loss...")
    class_weights = compute_class_weights(train_ds)
    class_weights = class_weights.to(device)

    # ---- Loss ----
    criterion = CombinedLoss(
        class_weights=class_weights,
        dice_weight=DICE_WEIGHT,
        ce_weight=CE_WEIGHT,
        lovasz_weight=LOVASZ_WEIGHT,
        boundary_weight=BOUNDARY_WEIGHT,
        focal_gamma=FOCAL_GAMMA,
        use_ohem=USE_OHEM,
        ohem_ratio=OHEM_RATIO,
    )
    
    # Auxiliary loss (simpler — just CE + Dice, no OHEM/boundary)
    aux_criterion = None
    if use_aux:
        aux_criterion = CombinedLoss(
            class_weights=class_weights,
            dice_weight=0.5,
            ce_weight=0.5,
            boundary_weight=0.0,
            focal_gamma=2.0,
            use_ohem=False,
        )
        print(f"[INFO] Auxiliary loss: Dice(0.5) + Focal(0.5, γ=2.0)")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    # ---- Resume Detection ----
    start_epoch = 1
    best_iou = 0.0

    # ---- LR Scheduler ----
    steps_per_epoch = len(train_loader) // args.accum_steps
    use_step_scheduler = False

    if LR_SCHEDULER == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=LR_RESTART_PERIOD, T_mult=2, eta_min=LR_MIN
        )
        print(f"[INFO] Using CosineAnnealingWarmRestarts: T_0={LR_RESTART_PERIOD}, T_mult=2, min_lr={LR_MIN}")
    elif LR_SCHEDULER == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=LR_WARMUP_EPOCHS / args.epochs,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100,
        )
        use_step_scheduler = True
        print(f"[INFO] Using OneCycleLR: max_lr={args.lr}, warmup={LR_WARMUP_EPOCHS} epochs")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=LR_MIN
        )
        print(f"[INFO] Using CosineAnnealingLR: T_max={args.epochs}, min_lr={LR_MIN}")

    # ---- Mixed Precision Scaler ----
    if use_amp:
        if has_torch_amp:
            scaler = amp.GradScaler('cuda')
        else:
            scaler = GradScaler()
    else:
        scaler = None

    # ---- Resume ----
    if args.resume:
        start_epoch, best_iou = load_checkpoint(args.resume, model, optimizer, device=device)
        start_epoch += 1

        # Force LR on resume
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr
        print(f"[INFO] ⚡ Forced LR to {args.lr} for fine-tuning")

        # Fresh scheduler from resume point
        remaining_epochs = args.epochs - start_epoch + 1
        if LR_SCHEDULER == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=min(LR_RESTART_PERIOD, remaining_epochs), T_mult=2, eta_min=LR_MIN
            )
            print(f"[INFO] Fresh CosineWarmRestarts: T_0={min(LR_RESTART_PERIOD, remaining_epochs)}")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=LR_MIN
            )
            print(f"[INFO] Fresh CosineAnnealingLR: T_max={remaining_epochs}")

    # ---- History ----
    history = {
        "train_loss": [], "val_loss": [],
        "val_iou": [], "val_dice": [], "val_acc": [],
        "lr": [],
    }

    # ---- Training loop ----
    print(f"\n[STEP 3] Starting training for {args.epochs} epochs...\n")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, args.epochs, scaler=scaler, use_amp=use_amp,
            accum_steps=args.accum_steps, use_step_scheduler=use_step_scheduler,
            aux_criterion=aux_criterion, aux_weight=AUX_LOSS_WEIGHT,
        )

        # Validate
        val_loss, val_iou, val_dice, val_acc, per_class_iou = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        # Step epoch-level scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if not use_step_scheduler and scheduler is not None:
            if LR_SCHEDULER == "cosine_warm_restarts":
                scheduler.step(epoch)
            else:
                scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t_start

        # Check improvement
        improved = ""
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            improved = " ★ NEW BEST"

            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou,
                "history": history,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            patience_counter += 1

        # Print per-class IoU (skip Background)
        class_strs = []
        for i in range(NUM_CLASSES):
            if i == IGNORE_INDEX:
                continue  # Skip Background in display
            if not np.isnan(per_class_iou[i]):
                marker = "❌" if per_class_iou[i] < 0.35 else "⚠️" if per_class_iou[i] < 0.50 else "✅"
                class_strs.append(f"{CLASS_NAMES[i]}={per_class_iou[i]:.3f}{marker}")
        print(f"  ↳ Per-class: {' | '.join(class_strs)}")

        print(
            f"Epoch {epoch:3d}/{args.epochs} │ "
            f"Train: {train_loss:.4f} │ "
            f"Val: {val_loss:.4f} │ "
            f"IoU: {val_iou:.4f} │ "
            f"Dice: {val_dice:.4f} │ "
            f"Acc: {val_acc:.4f} │ "
            f"LR: {current_lr:.2e} │ "
            f"{elapsed:.0f}s{improved}"
        )

        # Save every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou,
                "history": history,
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n[INFO] Early stopping after {patience_counter} epochs without improvement.")
            break

    # ---- Save history ----
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[INFO] Training history saved to {history_path}")

    # ---- Plot ----
    print("[INFO] Plotting training curves...")
    plot_training_curves(history, OUTPUT_DIR)

    # ---- Final evaluation ----
    print("\n[STEP 4] Final evaluation with best model...")
    load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_model.pth"), model, device=device)
    model = model.to(device)

    _, final_iou, final_dice, final_acc, _ = validate(
        model, val_loader, criterion, device, 0, 0
    )

    model.eval()
    all_per_iou, all_per_dice = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            per_iou, _ = compute_iou_per_class(outputs, masks)
            per_dice, _ = compute_dice_per_class(outputs, masks)
            all_per_iou.append(per_iou)
            all_per_dice.append(per_dice)

    avg_per_iou = np.nanmean(all_per_iou, axis=0).tolist()
    avg_per_dice = np.nanmean(all_per_dice, axis=0).tolist()

    print_metrics_table(avg_per_iou, avg_per_dice, final_acc)

    # ---- Save results ----
    results = {
        "best_val_iou": best_iou,
        "final_val_iou": final_iou,
        "final_val_dice": final_dice,
        "final_val_acc": final_acc,
        "per_class_iou": {name: iou for name, iou in zip(CLASS_NAMES, avg_per_iou)},
    }
    results_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Best Val IoU:  {best_iou:.4f} (excl. Background)")
    print(f"  Model saved:   {CHECKPOINT_DIR}/best_model.pth")
    print(f"  Plots saved:   {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
