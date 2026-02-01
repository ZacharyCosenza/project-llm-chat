export NCCL_DEBUG=WARN  # Change to WARN for less spam
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket

torchrun --nproc_per_node=8 run_6_pytorch.py --batch_size 22