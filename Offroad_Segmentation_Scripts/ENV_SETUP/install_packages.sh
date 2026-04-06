#!/bin/bash

# Try activating the Conda environment
echo "Activating the Conda environment 'EDU'..."

# Initialize conda for shell interaction
# This is often needed in scripts to make 'conda activate' work
eval "$(conda shell.bash hook)"

conda activate EDU

if [ $? -ne 0 ]; then
    echo "Failed to activate environment 'EDU'. Please ensure it was created correctly."
    exit 1
fi

# Install the required packages
echo "Installing PyTorch, Torchvision, and Ultralytics (Mac optimized)..."
# Note: Removed CUDA specific packages as they are not applicable to macOS
conda install -c pytorch -c conda-forge pytorch torchvision ultralytics -y

# Install pip packages
echo "Installing OpenCV and tqdm..."
pip install opencv-contrib-python tqdm

echo "Environment setup complete. You can now run your code in the 'EDU' environment."
