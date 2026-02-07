#!/bin/bash
# Run multi-GPU training experiments to verify GPU setup
# Usage: ./run_gpu_test.sh <N>
# This will run experiments with 1 GPU and N GPUs sequentially

set -e

N_GPUS=${1:-2}  # Default to 2 if not specified

EPOCHS=${EPOCHS:-50}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
LR=${LR:-0.001}
N_SAMPLES=${N_SAMPLES:-10000}
WANDB_PROJECT=${WANDB_PROJECT:-"multi-gpu-test"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/test_loss_gpu.py"

echo "============================================"
echo "Multi-GPU Training Test"
echo "============================================"
echo "Testing with 1 GPU and ${N_GPUS} GPUs"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Training samples: ${N_SAMPLES}"
echo "W&B project: ${WANDB_PROJECT}"
echo "============================================"
echo ""

# Run with 1 GPU
echo ">>> Running with 1 GPU..."
echo "    Effective batch size: ${BATCH_SIZE_PER_GPU}"
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 "$SCRIPT_PATH" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --n-samples "$N_SAMPLES" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "1gpu_bs${BATCH_SIZE_PER_GPU}"

echo ""
echo ">>> Running with ${N_GPUS} GPUs..."
EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * N_GPUS))
echo "    Effective batch size: ${EFFECTIVE_BS}"

# Build CUDA_VISIBLE_DEVICES string (e.g., "0,1,2" for 3 GPUs)
CUDA_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --standalone --nproc_per_node="$N_GPUS" "$SCRIPT_PATH" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --n-samples "$N_SAMPLES" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "${N_GPUS}gpu_bs${EFFECTIVE_BS}"

echo ""
echo "============================================"
echo "Experiments complete!"
echo "View results at: https://wandb.ai/${WANDB_PROJECT}"
echo "============================================"
