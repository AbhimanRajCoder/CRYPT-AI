#!/bin/bash

# run_train.sh — Helper script to run training on Mac
# Automatically activates the environment and optimizes for Mac/MPS

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Check if EDU_env exists
if [ ! -d "./EDU_env" ]; then
    echo "Error: Virtual environment 'EDU_env' not found."
    echo "Please run: bash setup_env.sh first."
    exit 1
fi

# Activate environment
echo "Activating virtual environment..."
source ./EDU_env/bin/activate

# Check for Apple Silicon / MPS support
IS_MAC=$(uname -s)
if [ "$IS_MAC" == "Darwin" ]; then
    echo "MacOS detected. Training will be optimized for Apple Silicon (MPS)."
else
    echo "Non-Mac system detected. Training will proceed with CUDA/CPU."
fi

# Run training
# Note: config.py has been updated to handle BATCH_SIZE and WORKERS settings for MPS
echo "Starting training pipeline..."
python3 train.py "$@"

# Post-training: Visualizing results (if training finished successfully)
if [ $? -eq 0 ]; then
    echo "Training complete! Generating visualizations..."
    python3 visualize.py
    echo "Check the 'outputs/' and 'checkpoints/' directories for results."
fi
