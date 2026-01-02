# Progressive PyTorch Lightning Test Suite

## Purpose

Since `test_nccl_minimal.py` **PASSED** (NCCL works fine), the hang must be caused by **PyTorch Lightning** or the interaction between Lightning components.

This script tests Lightning components progressively to isolate which one causes the hang.

## Test Progression

### Test 1: Baseline ✓ (Should pass)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 1
```
- **Tests**: Minimal Trainer + DummyModel
- **No**: Logger, callbacks, workers
- **Expected**: PASS (baseline sanity check)

---

### Test 2: WandbLogger - All Ranks ⚠️ (Might hang)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 2
```
- **Tests**: Trainer + WandbLogger initialized on **ALL ranks**
- **Expected**: Might HANG or have issues
- **Why**: Multiple ranks trying to initialize WandB simultaneously

---

### Test 3: WandbLogger - Rank 0 Only ✓ (Should pass)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 3
```
- **Tests**: WandbLogger only on rank 0, `logger=False` on others
- **Expected**: PASS
- **Why**: Proper rank-aware logger initialization

---

### Test 4: Explicit DDPStrategy ✓ (Should pass)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 4
```
- **Tests**: `strategy='ddp'` instead of `'auto'`
- **Expected**: PASS
- **Why**: Explicit strategy selection avoids auto-detection issues

---

### Test 5: DataLoader Workers ⚠️ (Might hang)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 5
```
- **Tests**: DataLoader with `num_workers=2`
- **Expected**: Might have issues
- **Why**: Worker processes + distributed can cause deadlocks

---

### Test 6: Full Combination ⚠️ (Tests the fix)
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 6
```
- **Tests**: WandB (rank 0 only) + workers + explicit DDP
- **Expected**: Should PASS if we've identified the issues
- **Why**: Combines all best practices

---

## How to Use

### Run all tests in sequence:
```bash
cd /home/zaccosenza/code/project-llm-chat
source .venv/bin/activate

for i in {1..6}; do
    echo "========== Running Test $i =========="
    torchrun --nproc_per_node=2 test_lightning_progressive.py --test $i
    if [ $? -ne 0 ]; then
        echo "TEST $i FAILED - stopping"
        break
    fi
    echo ""
done
```

### Run specific test:
```bash
torchrun --nproc_per_node=2 test_lightning_progressive.py --test 2
```

---

## Interpreting Results

### Scenario 1: Test 1 passes, Test 2 hangs
→ **Problem**: WandbLogger initialization on all ranks
→ **Solution**: Only initialize WandB on rank 0

### Scenario 2: Test 2 passes, Test 5 hangs
→ **Problem**: DataLoader workers with distributed
→ **Solution**: Use `num_workers=0` or properly configure worker initialization

### Scenario 3: All tests pass
→ **Problem**: Specific to your actual model/data
→ **Next step**: Test with your real TinyGPT model but dummy data

### Scenario 4: Test 1 fails
→ **Problem**: Basic Lightning + DDP issue
→ **Action**: Check PyTorch Lightning installation or version

---

## Expected Findings

Based on your earlier output where:
- NCCL test PASSED ✓
- Training script hangs during `trainer.fit()` distributed init

Most likely outcomes:
1. **Test 2 will hang or fail** → WandB on all ranks is the issue
2. **Test 3 will pass** → Confirms WandB needs rank guard
3. **Tests 4-6 should pass** → Shows the proper configuration

---

## Next Steps Based on Results

### If Test 2 hangs:
Apply this fix to `run_6_test.py`:
```python
rank = int(os.environ.get('RANK', 0))

if rank == 0:
    wandb_logger = WandbLogger(...)
else:
    wandb_logger = False  # Or set WANDB_MODE=disabled
```

### If Test 5 hangs:
Use `num_workers=0` in your training script, or configure properly:
```python
# In your dataloader
num_workers=0  # Safest for multi-GPU
# OR
persistent_workers=False  # If you need workers
```

### If all tests pass:
The issue is specific to your model/data. Create Test 7:
- Use your real TinyGPT model
- Use DummyDataModule (not your real data)
- Isolate if it's model size or complexity

---

## Quick Reference

| Test | Component | Expected | Purpose |
|------|-----------|----------|---------|
| 1 | Baseline | PASS | Sanity check |
| 2 | WandB (all ranks) | FAIL | Identify WandB issue |
| 3 | WandB (rank 0) | PASS | Confirm fix |
| 4 | Explicit DDP | PASS | Strategy check |
| 5 | Workers | ? | Worker compatibility |
| 6 | Full combo | PASS | Final validation |

---

## Troubleshooting

### Test hangs with no output
→ Likely distributed init deadlock, same as your training script
→ Try with `NCCL_DEBUG=INFO` for more details

### Test fails with exception
→ Good! We can debug exceptions. Check the traceback

### All tests pass but training still hangs
→ Issue is in your specific model/data
→ Create incremental tests with real components
