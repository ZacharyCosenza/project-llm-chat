#!/bin/bash
# Run ResNet-18 CIFAR-10 training on N GPUs
# Usage: ./run_gpu_test.sh <N>

set -e

N_GPUS=${1:-1}  # Default to 1 if not specified

EPOCHS=${EPOCHS:-50}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-128}
LR=${LR:-0.1}
DATA_DIR=${DATA_DIR:-"./data"}
WANDB_PROJECT=${WANDB_PROJECT:-"multi-gpu-test"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/test_loss_gpu.py"

EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * N_GPUS))

echo "============================================"
echo "ResNet-18 CIFAR-10 Training"
echo "============================================"
echo "GPUs: ${N_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Effective batch size: ${EFFECTIVE_BS}"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "W&B project: ${WANDB_PROJECT}"
echo "============================================"
echo ""

# Build CUDA_VISIBLE_DEVICES string (e.g., "0,1,2" for 3 GPUs)
CUDA_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --standalone --nproc_per_node="$N_GPUS" "$SCRIPT_PATH" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --data-dir "$DATA_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "${N_GPUS}gpu_bs${EFFECTIVE_BS}"

echo ""
echo "============================================"
echo "Experiment complete!"
echo "View results at: https://wandb.ai/${WANDB_PROJECT}"
echo "============================================"
