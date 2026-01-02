# Minimal NCCL/Distributed Test Script

## Purpose

This script tests **ONLY** the PyTorch distributed backend initialization (NCCL) without any PyTorch Lightning code. This helps isolate whether the hang is caused by:
- **NCCL/torch.distributed** (network, CUDA, or distributed backend issue)
- **PyTorch Lightning** (DDP strategy, trainer initialization, etc.)

## What It Tests

The script runs 5 steps in sequence:

1. **CUDA availability** - Verifies CUDA is available and sets device
2. **Distributed availability** - Checks if NCCL is available
3. **Distributed initialization** - Calls `dist.init_process_group()` ⚠️ **Most likely hang point**
4. **Collective operations** - Tests barrier, all_reduce, broadcast
5. **Cleanup** - Destroys process group

## Usage

### Basic Test (2 GPUs)
```bash
cd /home/zaccosenza/code/project-llm-chat
source .venv/bin/activate
torchrun --nproc_per_node=2 test_nccl_minimal.py
```

### With NCCL Debug Output
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 test_nccl_minimal.py
```

### With Specific Network Interface
If you have multiple network interfaces and NCCL picks the wrong one:
```bash
NCCL_SOCKET_IFNAME=eth0 torchrun --nproc_per_node=2 test_nccl_minimal.py
```

### Disable InfiniBand (if causing issues)
```bash
NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 test_nccl_minimal.py
```

### Fallback to Gloo (CPU-based backend)
If NCCL fails, the script will automatically try `gloo` backend.

## Expected Output

### Success Case
Both ranks should print all 5 steps:
```
[time][+0.123s][R0/2][LR0][PID] STEP 1: CUDA test PASSED
[time][+0.124s][R1/2][LR1][PID] STEP 1: CUDA test PASSED
[time][+0.234s][R0/2][LR0][PID] STEP 2: torch.distributed availability test PASSED
[time][+0.235s][R1/2][LR1][PID] STEP 2: torch.distributed availability test PASSED
[time][+0.345s][R0/2][LR0][PID] About to call dist.init_process_group(backend='nccl')...
[time][+0.346s][R1/2][LR1][PID] About to call dist.init_process_group(backend='nccl')...
[time][+1.500s][R0/2][LR0][PID] dist.init_process_group() RETURNED successfully!
[time][+1.501s][R1/2][LR1][PID] dist.init_process_group() RETURNED successfully!
[time][+1.600s][R0/2][LR0][PID] STEP 3: Distributed initialization PASSED
[time][+1.601s][R1/2][LR1][PID] STEP 3: Distributed initialization PASSED
...
[time][+2.000s][R0/2][LR0][PID] ALL TESTS PASSED!
[time][+2.001s][R1/2][LR1][PID] ALL TESTS PASSED!
```

### Hang Case
If it hangs during distributed init:
```
[time][+0.345s][R0/2][LR0][PID] About to call dist.init_process_group(backend='nccl')...
[time][+0.346s][R1/2][LR1][PID] About to call dist.init_process_group(backend='nccl')...
<HANG - no more output>
```

This confirms the issue is in NCCL/torch.distributed, NOT PyTorch Lightning.

## Troubleshooting

### If the script hangs at STEP 3

This means the distributed backend itself has an issue. Common causes:

1. **Network Configuration**
   - Check if `MASTER_ADDR` and `MASTER_PORT` are set correctly by torchrun
   - Verify network connectivity between nodes (if multi-node)
   - Try setting `NCCL_SOCKET_IFNAME` to the correct network interface

2. **NCCL Version Issues**
   ```bash
   # Check NCCL version
   python -c "import torch; print(torch.cuda.nccl.version())"

   # Check CUDA version
   nvcc --version
   ```

3. **Firewall/Port Blocking**
   - Make sure the MASTER_PORT is not blocked
   - On single node, this is usually not an issue

4. **CUDA/GPU Issues**
   - Check `nvidia-smi` to ensure GPUs are visible
   - Verify `CUDA_VISIBLE_DEVICES` is not restricting GPUs incorrectly

### If the script succeeds but run_6_test.py still hangs

This means:
- ✅ NCCL/torch.distributed works fine
- ❌ The issue is in PyTorch Lightning's DDP initialization or trainer.fit()

In this case, the problem is likely:
- PyTorch Lightning's strategy selection
- WandbLogger interfering with distributed setup
- DataLoader worker processes blocking distributed init
- Model synchronization during DDP setup

### Environment Variables to Try

```bash
# Maximum NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Force specific network interface (check with `ip addr`)
export NCCL_SOCKET_IFNAME=eth0  # or ens, ib0, etc.

# Disable InfiniBand if causing issues
export NCCL_IB_DISABLE=1

# Disable peer-to-peer transfers
export NCCL_P2P_DISABLE=1

# Increase timeout for slow networks
export NCCL_TIMEOUT=1800  # 30 minutes in seconds
```

## Next Steps Based on Results

### If test_nccl_minimal.py HANGS:
→ The issue is NCCL/network configuration
→ Focus on fixing NCCL setup, not PyTorch Lightning

### If test_nccl_minimal.py PASSES:
→ The issue is in PyTorch Lightning or your training script
→ Next step: Test PyTorch Lightning DDP without WandbLogger
→ Then: Test with explicit strategy='ddp' instead of 'auto'

## Quick Comparison Test

Run both scripts and compare:
```bash
# Test 1: Pure NCCL
torchrun --nproc_per_node=2 test_nccl_minimal.py

# Test 2: Full training script
torchrun --nproc_per_node=2 run_6_test.py --fast_dev_run 1 --num_workers 0
```

If Test 1 passes but Test 2 hangs, the issue is in the training script configuration.
