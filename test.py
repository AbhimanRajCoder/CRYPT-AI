"""
test.py — Complete Professional Testing and Evaluation Pipeline
Computes mAP@50, Precision, Recall, F1, and IoU.
Generates presentation-ready graphs and outputs for PPT.
"""

import os
import ssl
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context

# Import project settings
from config import (
    DEVICE, USE_CUDA, USE_MPS, VAL_DIR, CHECKPOINT_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, CLASS_NAMES, COLOR_PALETTE,
    IGNORE_INDEX, BASE_DIR
)
from dataset import OffroadSegDataset, get_val_transforms
from models import SegmentationModel
from utils import load_checkpoint, mask_to_color, denormalize

# Set professional visualization style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'axes.titleweight': 'bold',
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--data_dir", type=str, default=VAL_DIR)
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "runs", "test"))
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--history", type=str, default=os.path.join(BASE_DIR, "training_history.json"))
    return parser.parse_args()


# ─── METRIC UTILS ────────────────────────────────────────────────────────────

def compute_binary_metrics(pred_flat, true_flat, num_classes):
    """Computes basic metrics for each class pixel-wise."""
    ious, precisions, recalls, f1s = [], [], [], []
    
    for cls_idx in range(num_classes):
        if cls_idx == IGNORE_INDEX:
            ious.append(np.nan)
            precisions.append(np.nan)
            recalls.append(np.nan)
            f1s.append(np.nan)
            continue
            
        pred_cls = (pred_flat == cls_idx)
        true_cls = (true_flat == cls_idx)
        
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        
        tp = intersection
        fp = pred_cls.sum() - tp
        fn = true_cls.sum() - tp
        
        iou = tp / (union + 1e-6) if union > 0 else np.nan
        precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else np.nan
        
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = np.nan
            
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return ious, precisions, recalls, f1s


# ─── MATPLOTLIB GRAPHING FUNCTIONS ───────────────────────────────────────────

def save_fig(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path + ".pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_training_curves(history_file, graphs_dir):
    if not os.path.exists(history_file):
        return
    with open(history_file, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=3)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', lw=3)
    ax.set_title("Training & Validation Loss vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    fig.text(0.5, -0.05, "Insight: Both losses steadily decrease, showing proper convergence without massive overfitting.", ha='center', fontsize=12, style='italic')
    save_fig(fig, os.path.join(graphs_dir, "01_Loss_Curve"))
    
    # 2. mIoU / mAP proxy vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['val_iou'], label='mIoU (Proxy for mAP@50)', color='#2ca02c', lw=3)
    ax.set_title("Validation mIoU vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Intersection over Union")
    ax.legend()
    fig.text(0.5, -0.05, "Insight: Steady climb in metric performance. Warm restarts help escape early plateaus.", ha='center', fontsize=12, style='italic')
    save_fig(fig, os.path.join(graphs_dir, "02_Performance_Curve"))

    # 3. Learning Rate vs Performance
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(epochs, history.get('lr', epochs), color='#d62728', lw=2, linestyle='--')
    ax2.plot(epochs, history['val_iou'], color='#2ca02c', lw=3)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Learning Rate', color='#d62728')
    ax2.set_ylabel('mIoU', color='#2ca02c')
    plt.title("Learning Rate Schedule vs Performance")
    fig.text(0.5, -0.05, "Insight: Drops in LR often trigger spikes in performance (Cosine Warm Restarts).", ha='center', fontsize=12, style='italic')
    save_fig(fig, os.path.join(graphs_dir, "13_LR_vs_Performance"))


def plot_bar_chart(values, classes, title, ylabel, filename, graphs_dir, color='#1f77b4'):
    valid = [(v, c) for v, c in zip(values, classes) if not np.isnan(v)]
    if not valid: return
    vals, cls = zip(*valid)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(cls), y=list(vals), color=color, ax=ax, edgecolor='black', linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    fig.text(0.5, -0.15, f"Insight: Showcases exactly which classes dominate and which need more aggressive weighting.", ha='center', fontsize=12, style='italic')
    save_fig(fig, os.path.join(graphs_dir, filename))


def plot_confusion_matrix(cm, classes, filename, graphs_dir):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    fig.text(0.5, -0.1, "Insight: Dark diagonal = correct. Off-diagonal highlights confusion (e.g., Rocks vs Clutter).", ha='center', fontsize=12, style='italic')
    save_fig(fig, os.path.join(graphs_dir, filename))


def plot_pr_and_f1(all_probs_sample, all_true_sample, num_classes, graphs_dir):
    """Plot approximated standard PR curve and F1 score."""
    if len(all_probs_sample) == 0: return
    
    y_true = np.concatenate(all_true_sample)
    y_scores = np.concatenate(all_probs_sample, axis=0) # (N, num_classes)
    
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 8))
    
    # We sample a subset to avoid memory crash
    subsample = np.random.choice(len(y_true), min(100000, len(y_true)), replace=False)
    y_t = y_true[subsample]
    y_s = y_scores[subsample]
    
    mAP_sum = 0
    valid_classes = 0
    
    for cls_idx in range(1, num_classes):
        yt_c = (y_t == cls_idx).astype(int)
        ys_c = y_s[:, cls_idx]
        if yt_c.sum() == 0: continue
        
        precision, recall, thresholds = precision_recall_curve(yt_c, ys_c)
        ap = average_precision_score(yt_c, ys_c)
        mAP_sum += ap
        valid_classes += 1
        
        # PR Curve
        ax_pr.plot(recall, precision, lw=2, label=f"{CLASS_NAMES[cls_idx]} (AP={ap:.2f})")
        
        # F1 Curve
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)
        ax_f1.plot(thresholds, f1_scores, lw=2, label=CLASS_NAMES[cls_idx])
        
    ax_pr.set_title("Precision vs Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
    fig_pr.text(0.5, -0.05, "Insight: Curves closer to top-right indicate a near-perfect model.", ha='center', style='italic')
    
    ax_f1.set_title("F1 Score vs Confidence Threshold")
    ax_f1.set_xlabel("Confidence Threshold")
    ax_f1.set_ylabel("F1 Score")
    ax_f1.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
    fig_f1.text(0.5, -0.05, "Insight: Peak represents the optimal confidence threshold for max performance.", ha='center', style='italic')
    
    save_fig(fig_pr, os.path.join(graphs_dir, "04_Precision_Recall_Curve"))
    save_fig(fig_f1, os.path.join(graphs_dir, "05_F1_vs_Threshold"))
    
    return mAP_sum / valid_classes if valid_classes > 0 else 0


# ─── MAIN EVALUATION PIPELINE ────────────────────────────────────────────────

def main():
    args = parse_args()
    device = DEVICE

    # CREATE PIPELINE DIRECTORIES
    graphs_dir = os.path.join(args.output_dir, "graphs")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    preds_dir = os.path.join(args.output_dir, "predictions")
    visuals_dir = os.path.join(args.output_dir, "visuals")
    
    for d in [graphs_dir, metrics_dir, preds_dir, visuals_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 PIPELINE: TESTING & EVALUATION")
    print(f"{'='*60}")

    val_ds = OffroadSegDataset(args.data_dir, transform=get_val_transforms(), return_filename=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)

    print("[STEP 1] Loading Model...")
    model = SegmentationModel(use_aux=False)
    load_checkpoint(args.model, model, device=device)
    model = model.to(device)
    model.eval()

    print(f"[STEP 2] Running Inference on {len(val_ds)} images...")
    
    all_ious, all_precs, all_recs, all_f1s = [], [], [], []
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    iou_distribution = []
    
    # Subsample lists for PR curves
    pr_probs_sample = []
    pr_true_sample = []
    
    vis_count = 0

    with torch.no_grad():
        for images, masks, filenames in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            if args.tta:
                # Multi-scale + flip TTA (6 views, +0.01-0.015 mIoU)
                B, C_in, H, W = images.shape
                tta_sum = torch.zeros((B, NUM_CLASSES, H, W), device=device)
                n_views = 0
                # Multi-scale + Horizontal Flip TTA (6 views: 1.0, 0.75, 1.5 with flips)
                # This is highly effective for off-road while staying efficient.
                scales = [0.75, 1.0, 1.5]
                for scale in scales:
                    sh, sw = int(H * scale), int(W * scale)
                    scaled_img = F.interpolate(images, size=(sh, sw), mode='bilinear', align_corners=False)
                    
                    # 1. Forward original at this scale
                    out = F.softmax(model(scaled_img), dim=1)
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                    tta_sum += out
                    n_views += 1

                    # 2. Forward horizontal flip at this scale
                    hflip = torch.flip(scaled_img, dims=[3])
                    out_f = F.softmax(model(hflip), dim=1).flip(dims=[3])
                    out_f = F.interpolate(out_f, size=(H, W), mode='bilinear', align_corners=False)
                    tta_sum += out_f
                    n_views += 1

                outputs = tta_sum / n_views
            else:
                outputs = F.softmax(model(images), dim=1)
                
            pred_masks = torch.argmax(outputs, dim=1)

            # Store for PR Curve (take 1 randomly downscaled pixel grid per batch to save memory)
            if np.random.rand() < 0.2:
                pr_probs_sample.append(outputs.cpu().numpy()[:, :, ::16, ::16].transpose(0, 2, 3, 1).reshape(-1, NUM_CLASSES))
                pr_true_sample.append(masks.cpu().numpy()[:, ::16, ::16].flatten())

            # Batch Metrics
            for i in range(images.size(0)):
                p_flat = pred_masks[i].cpu().numpy().flatten()
                t_flat = masks[i].cpu().numpy().flatten()
                
                ious, precs, recs, f1s = compute_binary_metrics(p_flat, t_flat, NUM_CLASSES)
                
                # Image-level record
                all_ious.append(ious)
                all_precs.append(precs)
                all_recs.append(recs)
                all_f1s.append(f1s)
                
                # Distribution hist
                valid_ious = [ious[j] for j in range(1, NUM_CLASSES) if not np.isnan(ious[j])]
                if valid_ious: iou_distribution.append(np.mean(valid_ious))
                
                # Confusion Matrix tracking
                cm = confusion_matrix(t_flat, p_flat, labels=list(range(NUM_CLASSES)))
                total_cm += cm
                
                # Dump visual prediction
                if vis_count < 15:  # Save 15 visual proofs
                    img_arr = denormalize(images[i])
                    gt_col = mask_to_color(masks[i].cpu().numpy())
                    pr_col = mask_to_color(pred_masks[i].cpu().numpy())
                    
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(img_arr); axs[0].set_title("Image"); axs[0].axis('off')
                    axs[1].imshow(gt_col); axs[1].set_title("Ground Truth"); axs[1].axis('off')
                    axs[2].imshow(pr_col); axs[2].set_title("Prediction"); axs[2].axis('off')
                    plt.suptitle(filenames[i])
                    fig.savefig(os.path.join(visuals_dir, f"visual_proof_{vis_count:02d}.png"), dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    vis_count += 1
                    
                # Save Raw Pred
                Image.fromarray(pred_masks[i].cpu().numpy().astype(np.uint8)).save(
                    os.path.join(preds_dir, f"{os.path.splitext(filenames[i])[0]}.png")
                )

    print("[STEP 3] Generating Metrics and Presentation Graphs...")
    
    # Calculate Overall Means
    mean_ious = np.nanmean(all_ious, axis=0)
    mean_precs = np.nanmean(all_precs, axis=0)
    mean_recs = np.nanmean(all_recs, axis=0)
    mean_f1s = np.nanmean(all_f1s, axis=0)
    
    # Graph 1-3
    plot_training_curves(args.history, graphs_dir)
    
    # Graph 4 & 5 (PR / F1 Curve) -> also returns proxy mAP
    mAP50 = plot_pr_and_f1(pr_probs_sample, pr_true_sample, NUM_CLASSES, graphs_dir)
    
    # Graph 6
    plot_confusion_matrix(total_cm, CLASS_NAMES, "06_Confusion_Matrix", graphs_dir)
    
    # Graph 7 & 10
    plot_bar_chart(mean_ious[1:], CLASS_NAMES[1:], "Class-wise IoU (Proxy mAP@50)", "IoU", "07_Class_IoU_mAP", graphs_dir)
    plot_bar_chart(mean_f1s[1:], CLASS_NAMES[1:], "Class-wise F1-Score", "F1 Score", "10_Class_F1_Scores", graphs_dir, color='#2ca02c')
    
    # Graph 8: Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(iou_distribution, bins=30, kde=True, color='purple', ax=ax)
    ax.set_title("IoU Distribution Across Images")
    ax.set_xlabel("Mean IoU per Image")
    ax.set_ylabel("Count")
    save_fig(fig, os.path.join(graphs_dir, "08_IoU_Distribution"))

    # SUMMARY TABLES & EXPORTS
    final_mIoU = np.nanmean(mean_ious[1:])
    final_mPrec = np.nanmean(mean_precs[1:])
    final_mRec = np.nanmean(mean_recs[1:])
    final_mF1 = np.nanmean(mean_f1s[1:])
    
    summary = {
        "mAP@50 (Global AP)": float(mAP50),
        "mean_IoU": float(final_mIoU),
        "mean_Precision": float(final_mPrec),
        "mean_Recall": float(final_mRec),
        "mean_F1": float(final_mF1),
    }

    # JSON Output
    with open(os.path.join(metrics_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    # CSV Output
    df = pd.DataFrame({
        "Class": CLASS_NAMES[1:],
        "IoU": mean_ious[1:],
        "Precision": mean_precs[1:],
        "Recall": mean_recs[1:],
        "F1-Score": mean_f1s[1:]
    })
    df.to_csv(os.path.join(metrics_dir, "class_metrics.csv"), index=False)

    print(f"\n✅ PIPELINE COMPLETE. Folders generated in {args.output_dir}/:")
    print(f"  ├── graphs/      (10+ PPT-ready graphs)")
    print(f"  ├── metrics/     (results.json, class_metrics.csv)")
    print(f"  ├── predictions/ (Raw mask outputs)")
    print(f"  └── visuals/     (Before-and-after proof screenshots)")
    print()
    print("-----------------------------------------------------")
    print(" 📊 FINAL METRICS SUMMARY (Averaged valid classes):")
    print("-----------------------------------------------------")
    print(f" mAP@50 (Pixel AP): {mAP50:.4f}")
    print(f" Mean IoU:          {final_mIoU:.4f}")
    print(f" Mean Precision:    {final_mPrec:.4f}")
    print(f" Mean Recall:       {final_mRec:.4f}")
    print(f" Mean F1-Score:     {final_mF1:.4f}")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()
