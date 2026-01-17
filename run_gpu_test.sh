#!/bin/bash

# Simple launcher script for gpu_test.py with multiple GPUs
# Usage: ./run_gpu_test.sh [num_gpus]
# Example: ./run_gpu_test.sh 2  (runs on 2 GPUs)
#          ./run_gpu_test.sh    (auto-detects and uses all available GPUs)

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

echo "Running GPU test on $NUM_GPUS GPU(s)..."
echo "========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single GPU mode - running directly"
    python gpu_test.py
else
    echo "Multi-GPU mode - using torchrun"
    torchrun --standalone --nproc_per_node=$NUM_GPUS gpu_test.py
fi
