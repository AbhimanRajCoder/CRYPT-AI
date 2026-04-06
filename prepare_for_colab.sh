#!/bin/bash
# prepare_for_colab.sh — Packages the project for Google Colab upload
#
# This creates TWO zip files:
#   1. offroad_scripts.zip   — All Python scripts (small, ~50KB)
#   2. offroad_dataset.zip   — The training dataset (large)
#
# Upload BOTH to Google Drive, then run the Colab notebook.

cd "$(dirname "$0")"

echo "============================================"
echo "  Packaging project for Google Colab"
echo "============================================"

# 1. Package the scripts (small file - easy to upload)
echo ""
echo "[1/2] Creating offroad_scripts.zip (Python scripts only)..."
zip -j offroad_scripts.zip \
    config.py \
    dataset.py \
    models.py \
    losses.py \
    metrics.py \
    train.py \
    test.py \
    visualize.py \
    utils.py \
    requirements.txt

echo "  ✓ offroad_scripts.zip created ($(du -h offroad_scripts.zip | cut -f1))"

# 2. Package the dataset
echo ""
echo "[2/2] Creating offroad_dataset.zip (training dataset)..."
echo "  This may take a few minutes for ~3000 images..."
cd Offroad_Segmentation_Training_Dataset
zip -r ../offroad_dataset.zip train/ val/
cd ..

echo "  ✓ offroad_dataset.zip created ($(du -h offroad_dataset.zip | cut -f1))"

echo ""
echo "============================================"
echo "  DONE! Two files created:"
echo "============================================"
echo "  1. offroad_scripts.zip  — Upload to Google Drive"
echo "  2. offroad_dataset.zip  — Upload to Google Drive"
echo ""
echo "  Then open Offroad_Segmentation_Colab.ipynb in Google Colab"
echo "  and follow the step-by-step instructions."
echo "============================================"
