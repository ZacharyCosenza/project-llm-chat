#!/bin/bash
# Run multi-GPU training experiments to verify GPU setup
# Usage: ./run_gpu_test.sh <N>
# Runs 1-GPU and N-GPU experiments IN PARALLEL (requires N+1 GPUs total)
# GPU 0 is used for the 1-GPU experiment, GPUs 1..N for the N-GPU experiment

set -e

N_GPUS=${1:-2}  # Default to 2 if not specified

EPOCHS=${EPOCHS:-50}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
LR=${LR:-0.001}
N_SAMPLES=${N_SAMPLES:-10000}
WANDB_PROJECT=${WANDB_PROJECT:-"multi-gpu-test"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/test_loss_gpu.py"

EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * N_GPUS))

echo "============================================"
echo "Multi-GPU Training Test (PARALLEL)"
echo "============================================"
echo "Running in parallel: 1 GPU (GPU 0) and ${N_GPUS} GPUs (GPUs 1-${N_GPUS})"
echo "Total GPUs required: $((N_GPUS + 1))"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Training samples: ${N_SAMPLES}"
echo "W&B project: ${WANDB_PROJECT}"
echo "============================================"
echo ""

# Build CUDA_VISIBLE_DEVICES string for N-GPU experiment (GPUs 1 through N)
CUDA_DEVICES_N=$(seq -s, 1 "$N_GPUS")

echo ">>> Starting 1-GPU experiment (GPU 0, effective batch size: ${BATCH_SIZE_PER_GPU})..."
echo ">>> Starting ${N_GPUS}-GPU experiment (GPUs ${CUDA_DEVICES_N}, effective batch size: ${EFFECTIVE_BS})..."
echo ""

# Run 1-GPU experiment in background (use different master port)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT_PATH" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --n-samples "$N_SAMPLES" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "1gpu_bs${BATCH_SIZE_PER_GPU}" &
PID_1GPU=$!

# Run N-GPU experiment in background (use different master port)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_N torchrun --standalone --nproc_per_node="$N_GPUS" --master_port=29501 "$SCRIPT_PATH" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --n-samples "$N_SAMPLES" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "${N_GPUS}gpu_bs${EFFECTIVE_BS}" &
PID_NGPU=$!

# Wait for both to complete
echo "Waiting for experiments to complete (PIDs: $PID_1GPU, $PID_NGPU)..."
wait $PID_1GPU
EXIT_1GPU=$?
wait $PID_NGPU
EXIT_NGPU=$?

echo ""
echo "============================================"
if [ $EXIT_1GPU -eq 0 ] && [ $EXIT_NGPU -eq 0 ]; then
    echo "Both experiments completed successfully!"
else
    echo "Some experiments failed (1-GPU exit: $EXIT_1GPU, ${N_GPUS}-GPU exit: $EXIT_NGPU)"
fi
echo "View results at: https://wandb.ai/${WANDB_PROJECT}"
echo "============================================"
