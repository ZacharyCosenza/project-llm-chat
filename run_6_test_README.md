# run_6_test.py - Multi-GPU Hang Debug Script

## Overview
This is a hyper-verbose debug version of the training script designed to pinpoint EXACTLY where the multi-GPU hang occurs.

## Features

### 1. Enhanced Debug Output
Every debug print includes:
- **Timestamp**: Current time with milliseconds
- **Elapsed time**: Time since script start
- **Rank info**: Global rank, local rank, world size
- **Process ID**: To identify worker processes

Example output:
```
[10:23:45.123][+2.456s][R0/2][LR0][PID:12345] CHECKPOINT 5: Model created
```

### 2. Numbered Checkpoints
12 checkpoints track progress through initialization:
1. Script started
2. Args parsed
3. Logs directory ready
4. Tokenizer loaded
5. Model created
6. LLMModule ready
7. DataModule ready
8. Accelerator configured
9. WandbLogger ready ⚠️ PRIME SUSPECT
10. MemoryLoggingCallback ready
11. Trainer ready ⚠️ PRIME SUSPECT
12. Training finished

### 3. Critical Sections Marked
Two main suspects for multi-GPU hangs:
- **WandbLogger initialization** (Checkpoint 9)
- **PyTorch Lightning Trainer initialization** (Checkpoint 11)

These sections have extra verbose output before/after the constructor calls.

### 4. Environment & CUDA Status
At script start and before/after Trainer init:
- All relevant environment variables (RANK, CUDA_VISIBLE_DEVICES, WANDB_*, NCCL_*, etc.)
- CUDA availability and device info
- Distributed backend initialization status

### 5. All Ranks Print
Unlike the original script, this version forces ALL GPU ranks to print debug messages, making it clear if one rank hangs while others continue.

## Usage

### Basic Run (with multiple GPUs)
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 run_6_test.py --fast_dev_run 5 --num_workers 0

# Using python -m torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 run_6_test.py --fast_dev_run 5 --num_workers 0
```

### Arguments
- `--batch_size`: Batch size (default: 32)
- `--max_steps`: Max training steps (default: 10000)
- `--fast_dev_run`: Quick test with N batches (default: 0, disabled)
- `--num_workers`: Number of dataloader workers (default: 0, recommended for debugging)

### Recommended Test Sequence

**Test 1: Minimal run**
```bash
torchrun --nproc_per_node=2 run_6_test.py --fast_dev_run 1 --num_workers 0
```

**Test 2: If Test 1 works, try with workers**
```bash
torchrun --nproc_per_node=2 run_6_test.py --fast_dev_run 1 --num_workers 1
```

**Test 3: Single GPU baseline**
```bash
python run_6_test.py --fast_dev_run 1 --num_workers 0
```

## Reading the Output

### Successful Run
All ranks should reach CHECKPOINT 12:
```
[time][+Xs][R0/2][LR0][PID] CHECKPOINT 11: Trainer ready
[time][+Xs][R1/2][LR1][PID] CHECKPOINT 11: Trainer ready
[time][+Xs][R0/2][LR0][PID] CHECKPOINT 12: Training finished
[time][+Xs][R1/2][LR1][PID] CHECKPOINT 12: Training finished
```

### Identifying the Hang

**Example 1: WandbLogger hang**
```
[time][+2.1s][R0/2][LR0][PID] CHECKPOINT 8: Accelerator configured
[time][+2.1s][R1/2][LR1][PID] CHECKPOINT 8: Accelerator configured
[time][+2.2s][R0/2][LR0][PID] About to call WandbLogger constructor...
[time][+2.2s][R1/2][LR1][PID] About to call WandbLogger constructor...
<HANG - no more output>
```
→ **Problem**: WandbLogger initialization blocks on all ranks

**Example 2: Trainer hang**
```
[time][+3.5s][R0/2][LR0][PID] CHECKPOINT 9: WandbLogger ready
[time][+3.5s][R1/2][LR1][PID] CHECKPOINT 9: WandbLogger ready
[time][+3.6s][R0/2][LR0][PID] About to call pl.Trainer with strategy=auto...
[time][+3.6s][R1/2][LR1][PID] About to call pl.Trainer with strategy=auto...
<HANG - no more output>
```
→ **Problem**: Trainer initialization (likely DDP setup) blocks

**Example 3: Partial hang (one rank)**
```
[time][+2.5s][R0/2][LR0][PID] CHECKPOINT 9: WandbLogger ready
[time][+5.5s][R1/2][LR1][PID] CHECKPOINT 9: WandbLogger ready
<HANG after rank 1 completes>
```
→ **Problem**: Rank synchronization issue, one rank waiting for the other

## Common Issues & Solutions

### Issue 1: Hang at WandbLogger (Checkpoint 9)
**Cause**: All ranks trying to initialize WandB simultaneously
**Solution**:
- Set `WANDB_MODE=disabled` for rank != 0
- Or only create WandbLogger on rank 0

### Issue 2: Hang at Trainer init (Checkpoint 11)
**Cause**: Distributed backend initialization or strategy selection
**Solution**:
- Explicitly set strategy='ddp' instead of 'auto'
- Check NCCL environment variables
- Verify network connectivity between ranks

### Issue 3: Hang during directory creation (Checkpoint 3)
**Cause**: Race condition on mkdir
**Solution**: Guard with rank 0 check + barrier

### Issue 4: Hang at tokenizer load (Checkpoint 4)
**Cause**: Multiple ranks hitting HuggingFace Hub simultaneously
**Solution**: Use offline mode or cache tokenizer first

## Next Steps

Once you identify the exact hang location from the checkpoint output:

1. **If WandbLogger hangs**: Modify to only initialize on rank 0
2. **If Trainer hangs**: Check distributed environment setup
3. **If during fit()**: The issue is in training loop, not initialization

You can then create a targeted fix for that specific component.
