"""
visualize.py — Visualization utilities for Off-Road Segmentation.

Usage:
    python visualize.py
    python visualize.py --history outputs/training_history.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, CLASS_NAMES, COLOR_PALETTE, NUM_CLASSES


def plot_training_curves(history, output_dir=OUTPUT_DIR):
    """Plot comprehensive training curves."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], 'b-', label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label="Val", linewidth=2)
    ax.set_title("Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU
    ax = axes[0, 1]
    ax.plot(epochs, history["val_iou"], 'g-', linewidth=2, marker='o', markersize=3)
    best_epoch = np.argmax(history["val_iou"]) + 1
    best_iou = max(history["val_iou"])
    ax.axhline(y=best_iou, color='green', linestyle='--', alpha=0.5)
    ax.annotate(f'Best: {best_iou:.4f} (epoch {best_epoch})',
                xy=(best_epoch, best_iou), fontsize=10,
                xytext=(5, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold')
    ax.set_title("Validation IoU", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean IoU")
    ax.grid(True, alpha=0.3)

    # Dice
    ax = axes[1, 0]
    ax.plot(epochs, history["val_dice"], 'm-', linewidth=2, marker='s', markersize=3)
    ax.set_title("Validation Dice Score", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice")
    ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], 'orange', linewidth=2)
    ax.set_title("Learning Rate Schedule", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved training curves to {path}")

    # Loss zoomed
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    ax.fill_between(epochs, history["train_loss"], alpha=0.1, color='blue')
    ax.fill_between(epochs, history["val_loss"], alpha=0.1, color='red')
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved loss curve to {path}")


def plot_per_class_iou(per_class_iou, output_dir=OUTPUT_DIR, title="Per-Class IoU"):
    """Plot horizontal bar chart of per-class IoU."""
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(per_class_iou, dict):
        names = list(per_class_iou.keys())
        values = [v if v is not None else 0 for v in per_class_iou.values()]
    else:
        names = CLASS_NAMES
        values = [v if not np.isnan(v) else 0 for v in per_class_iou]

    sorted_indices = np.argsort(values)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_colors = [np.array(COLOR_PALETTE[i]) / 255.0 for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(sorted_names)), sorted_values, color=sorted_colors,
                   edgecolor='black', linewidth=0.5, height=0.7)

    for bar, val in zip(bars, sorted_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

    mean_iou = np.mean([v for v in values if v > 0])
    ax.axvline(x=mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.4f}')

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=12)
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_iou.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved per-class IoU chart to {path}")


def create_legend_image(output_dir=OUTPUT_DIR):
    """Create a color legend showing class-to-color mapping."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, COLOR_PALETTE)):
        color_norm = np.array(color) / 255.0
        ax.barh(i, 1, color=color_norm, edgecolor='black', height=0.8)
        ax.text(0.5, i, name, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white' if sum(color) < 300 else 'black')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Class Color Legend", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "class_legend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved class legend to {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--history", type=str,
                        default=os.path.join(OUTPUT_DIR, "training_history.json"),
                        help="Path to training history JSON")
    parser.add_argument("--results", type=str,
                        default=os.path.join(OUTPUT_DIR, "final_results.json"),
                        help="Path to final results JSON")
    args = parser.parse_args()

    if os.path.exists(args.history):
        with open(args.history, "r") as f:
            history = json.load(f)
        plot_training_curves(history)
        print("[INFO] Training curves plotted successfully!")
    else:
        print(f"[WARNING] Training history not found at {args.history}")

    if os.path.exists(args.results):
        with open(args.results, "r") as f:
            results = json.load(f)
        if "per_class_iou" in results:
            plot_per_class_iou(results["per_class_iou"])

    create_legend_image()
    print("\n[INFO] All visualizations saved to outputs/")


if __name__ == "__main__":
    main()
