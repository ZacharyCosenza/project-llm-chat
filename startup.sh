#!/bin/bash

apt update -y
apt install -y vim tmux python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# Install PyTorch first (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other packages
pip install numpy pandas matplotlib scikit-learn \
    transformers datasets accelerate \
    lightning wandb pyarrow hf_transfer

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"