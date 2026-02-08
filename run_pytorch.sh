#!/bin/bash
# Run TinyGPT training on N GPUs
# Usage: ./run_pytorch.sh <N> [--gpt2-opt]
#   N         - Number of GPUs (default: 1)
#   --gpt2-opt - Use GPT-2 standard optimizer (run_7_opt.py)

set -e

N_GPUS=1
USE_GPT2_OPT=false

for arg in "$@"; do
    case $arg in
        --gpt2-opt)
            USE_GPT2_OPT=true
            ;;
        *)
            if [[ $arg =~ ^[0-9]+$ ]]; then
                N_GPUS=$arg
            fi
            ;;
    esac
done

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$USE_GPT2_OPT" = true ]; then
    SCRIPT="${SCRIPT_DIR}/run_7_opt.py"
    OPT_NAME="GPT-2 Standard"
else
    SCRIPT="${SCRIPT_DIR}/run_6_pytorch.py"
    OPT_NAME="Custom"
fi

echo "============================================"
echo "TinyGPT Training - ${N_GPUS} GPU(s)"
echo "Optimizer: ${OPT_NAME}"
echo "============================================"

torchrun --standalone --nproc_per_node="$N_GPUS" "$SCRIPT"
