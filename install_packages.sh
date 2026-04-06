#!/bin/bash

# Activating the virtual environment
echo "Activating the virtual environment 'EDU_env'..."

# Ensure we are in the correct directory (optional if script is called from the same dir)
# source ./EDU_env/bin/activate

if [ -f "./EDU_env/bin/activate" ]; then
    source ./EDU_env/bin/activate
else
    echo "Error: Virtual environment 'EDU_env' not found in $(pwd). Please run create_env.sh first."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the required packages
echo "Installing PyTorch, Torchvision, and Ultralytics (Mac optimized)..."
# On macOS, pip handles the CPU/MPS backend automatically for torch/torchvision
pip install torch torchvision torchaudio ultralytics opencv-contrib-python tqdm

echo "--------------------------------------------------------"
echo "Environment setup complete!"
echo "To use this environment later, run: source ./EDU_env/bin/activate"
echo "--------------------------------------------------------"
