#!/bin/bash
# Run run_6_pytorch.py on N GPUs
# Usage: ./run_pytorch.sh <N>

set -e

N_GPUS=${1:-1}  # Default to 1 if not specified

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/run_6_pytorch.py"

echo "============================================"
echo "TinyGPT Training"
echo "============================================"
echo "GPUs: ${N_GPUS}"
echo "Batch size: 32 (per GPU)"
echo "Max steps: 10000"
echo "============================================"
echo ""

torchrun --standalone --nproc_per_node="$N_GPUS" "$SCRIPT_PATH"
