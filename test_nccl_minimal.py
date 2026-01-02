"""
Minimal NCCL/Distributed Test Script

This script tests ONLY the distributed backend initialization to isolate
whether the hang is caused by NCCL/torch.distributed or PyTorch Lightning.

Usage:
    torchrun --nproc_per_node=2 test_nccl_minimal.py
"""

import torch
import torch.distributed as dist
import os
import sys
import time

# Global script start time
SCRIPT_START_TIME = time.time()

def debug_print(msg, force_all_ranks=True):
    """Print debug message with rank, timestamp, and process info"""
    rank = int(os.environ.get('LOCAL_RANK', -1))
    global_rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    elapsed = time.time() - SCRIPT_START_TIME

    if force_all_ranks or rank <= 0:
        prefix = f"[{timestamp}][+{elapsed:.3f}s][R{global_rank}/{world_size}][LR{rank}][PID:{os.getpid()}]"
        print(f"{prefix} {msg}", flush=True)
        sys.stdout.flush()

def print_env_vars():
    """Print all relevant environment variables"""
    debug_print("=" * 80)
    debug_print("ENVIRONMENT VARIABLES:")
    env_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
        'MASTER_ADDR', 'MASTER_PORT',
        'CUDA_VISIBLE_DEVICES',
        'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE',
        'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'
    ]
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        debug_print(f"  {var} = {value}")
    debug_print("=" * 80)

def test_cuda():
    """Test CUDA availability and setup"""
    debug_print("=" * 80)
    debug_print("STEP 1: Testing CUDA")
    debug_print("=" * 80)

    debug_print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        debug_print("ERROR: CUDA not available!")
        return False

    debug_print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    debug_print(f"Setting CUDA device to LOCAL_RANK={local_rank}")
    torch.cuda.set_device(local_rank)

    debug_print(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
    debug_print(f"CUDA device name: {torch.cuda.get_device_name()}")

    debug_print("STEP 1: CUDA test PASSED")
    debug_print("=" * 80)
    return True

def test_dist_available():
    """Test if distributed is available"""
    debug_print("=" * 80)
    debug_print("STEP 2: Testing torch.distributed availability")
    debug_print("=" * 80)

    debug_print(f"torch.distributed.is_available() = {torch.distributed.is_available()}")
    debug_print(f"torch.distributed.is_nccl_available() = {torch.distributed.is_nccl_available()}")

    if not torch.distributed.is_available():
        debug_print("ERROR: torch.distributed not available!")
        return False

    if not torch.distributed.is_nccl_available():
        debug_print("WARNING: NCCL not available, trying gloo instead")
        return "gloo"

    debug_print("STEP 2: torch.distributed availability test PASSED")
    debug_print("=" * 80)
    return "nccl"

def test_dist_init(backend="nccl"):
    """Test distributed initialization"""
    debug_print("=" * 80)
    debug_print(f"STEP 3: Initializing distributed backend ({backend})")
    debug_print("=" * 80)

    debug_print(f"About to call dist.init_process_group(backend='{backend}')...")
    sys.stdout.flush()

    try:
        # This is where the hang might occur
        dist.init_process_group(backend=backend)

        debug_print("dist.init_process_group() RETURNED successfully!")
        debug_print(f"torch.distributed.is_initialized() = {dist.is_initialized()}")
        debug_print(f"dist.get_backend() = {dist.get_backend()}")
        debug_print(f"dist.get_rank() = {dist.get_rank()}")
        debug_print(f"dist.get_world_size() = {dist.get_world_size()}")

        debug_print("STEP 3: Distributed initialization PASSED")
        debug_print("=" * 80)
        return True

    except Exception as e:
        debug_print(f"ERROR during dist.init_process_group(): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collective_ops():
    """Test basic collective operations"""
    debug_print("=" * 80)
    debug_print("STEP 4: Testing collective operations")
    debug_print("=" * 80)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Test barrier
    debug_print("Testing barrier...")
    dist.barrier()
    debug_print("Barrier PASSED")

    # Test all_reduce
    debug_print("Testing all_reduce...")
    tensor = torch.ones(1).cuda() * rank
    debug_print(f"Before all_reduce: tensor = {tensor.item()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    debug_print(f"After all_reduce: tensor = {tensor.item()} (expected: {expected})")

    if tensor.item() == expected:
        debug_print("all_reduce PASSED")
    else:
        debug_print(f"all_reduce FAILED: got {tensor.item()}, expected {expected}")
        return False

    # Test broadcast
    debug_print("Testing broadcast...")
    if rank == 0:
        broadcast_tensor = torch.tensor([42.0]).cuda()
    else:
        broadcast_tensor = torch.zeros(1).cuda()

    debug_print(f"Before broadcast: tensor = {broadcast_tensor.item()}")
    dist.broadcast(broadcast_tensor, src=0)
    debug_print(f"After broadcast: tensor = {broadcast_tensor.item()}")

    if broadcast_tensor.item() == 42.0:
        debug_print("broadcast PASSED")
    else:
        debug_print(f"broadcast FAILED: got {broadcast_tensor.item()}, expected 42.0")
        return False

    debug_print("STEP 4: Collective operations test PASSED")
    debug_print("=" * 80)
    return True

def cleanup():
    """Cleanup distributed"""
    debug_print("=" * 80)
    debug_print("STEP 5: Cleanup")
    debug_print("=" * 80)

    if dist.is_initialized():
        debug_print("Calling dist.destroy_process_group()...")
        dist.destroy_process_group()
        debug_print("Cleanup complete")

    debug_print("=" * 80)

def main():
    debug_print("=" * 80)
    debug_print("MINIMAL NCCL/DISTRIBUTED TEST SCRIPT")
    debug_print("=" * 80)

    # Print environment
    print_env_vars()

    # Step 1: Test CUDA
    if not test_cuda():
        debug_print("CUDA test failed, exiting")
        return 1

    # Step 2: Check distributed availability
    backend = test_dist_available()
    if not backend:
        debug_print("Distributed not available, exiting")
        return 1

    # Step 3: Initialize distributed
    if not test_dist_init(backend):
        debug_print("Distributed initialization failed, exiting")
        return 1

    # Step 4: Test collective operations
    if not test_collective_ops():
        debug_print("Collective operations failed")
        cleanup()
        return 1

    # Step 5: Cleanup
    cleanup()

    debug_print("=" * 80)
    debug_print("ALL TESTS PASSED!")
    debug_print("=" * 80)

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        debug_print(f"Script exiting with code {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        debug_print("Interrupted by user")
        cleanup()
        sys.exit(130)
    except Exception as e:
        debug_print(f"UNHANDLED EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
