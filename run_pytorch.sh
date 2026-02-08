#!/bin/bash
# Run TinyGPT training on N GPUs
# Usage: ./run_pytorch.sh <N>
#   N - Number of GPUs (default: 1)

set -e

N_GPUS=${1:-1}

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "TinyGPT Training - ${N_GPUS} GPU(s)"
echo "============================================"

torchrun --standalone --nproc_per_node="$N_GPUS" "${SCRIPT_DIR}/run_7_opt.py"
