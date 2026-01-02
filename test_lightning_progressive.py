"""
Progressive PyTorch Lightning Test

Since NCCL works fine (test_nccl_minimal.py passed), this script tests
PyTorch Lightning components one by one to find which causes the hang.

Usage:
    torchrun --nproc_per_node=2 test_lightning_progressive.py --test 1
    torchrun --nproc_per_node=2 test_lightning_progressive.py --test 2
    ...etc
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import sys
import time
import argparse
from pathlib import Path

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


class DummyModel(pl.LightningModule):
    """Minimal LightningModule for testing"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x).mean()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DummyDataModule(pl.LightningDataModule):
    """Minimal DataModule for testing"""
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 10)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)


def test_1_basic_trainer():
    """Test 1: Minimal Trainer with auto strategy (should work)"""
    debug_print("=" * 80)
    debug_print("TEST 1: Basic Trainer + Dummy Model (no logger, no callbacks)")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()
    debug_print("DummyModel created")

    debug_print("Creating DummyDataModule...")
    datamodule = DummyDataModule()
    debug_print("DummyDataModule created")

    debug_print("Creating Trainer with strategy='auto', devices=2...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='auto',
        max_steps=5,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    debug_print("Trainer created successfully!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 1: PASSED ✓")
    return True


def test_2_with_wandb():
    """Test 2: Trainer + WandbLogger (might hang)"""
    debug_print("=" * 80)
    debug_print("TEST 2: Trainer + WandbLogger (all ranks initialize)")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()

    debug_print("Creating DummyDataModule...")
    datamodule = DummyDataModule()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    debug_print(f"Logs directory: {logs_dir}")

    debug_print("Creating WandbLogger (ALL RANKS)...")
    sys.stdout.flush()

    wandb_logger = WandbLogger(
        project="lightning-test",
        name="test-2-wandb",
        save_dir=str(logs_dir),
    )
    debug_print("WandbLogger created!")

    debug_print("Creating Trainer with WandbLogger...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='auto',
        max_steps=5,
        enable_checkpointing=False,
        logger=wandb_logger,
        enable_progress_bar=False,
    )
    debug_print("Trainer created!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 2: PASSED ✓")
    return True


def test_3_wandb_rank0_only():
    """Test 3: WandbLogger only on rank 0"""
    debug_print("=" * 80)
    debug_print("TEST 3: WandbLogger only on rank 0")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()

    debug_print("Creating DummyDataModule...")
    datamodule = DummyDataModule()

    rank = int(os.environ.get('RANK', 0))

    if rank == 0:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        debug_print("Creating WandbLogger (RANK 0 ONLY)...")
        wandb_logger = WandbLogger(
            project="lightning-test",
            name="test-3-wandb-rank0",
            save_dir=str(logs_dir),
        )
        debug_print("WandbLogger created on rank 0")
    else:
        wandb_logger = False
        debug_print("Rank > 0: Using logger=False")

    debug_print("Creating Trainer...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='auto',
        max_steps=5,
        enable_checkpointing=False,
        logger=wandb_logger,
        enable_progress_bar=False,
    )
    debug_print("Trainer created!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 3: PASSED ✓")
    return True


def test_4_explicit_ddp():
    """Test 4: Explicit DDPStrategy instead of 'auto'"""
    debug_print("=" * 80)
    debug_print("TEST 4: Explicit DDPStrategy (strategy='ddp')")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()

    debug_print("Creating DummyDataModule...")
    datamodule = DummyDataModule()

    debug_print("Creating Trainer with strategy='ddp'...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp',
        max_steps=5,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    debug_print("Trainer created!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 4: PASSED ✓")
    return True


def test_5_with_dataloader_workers():
    """Test 5: DataLoader with num_workers > 0"""
    debug_print("=" * 80)
    debug_print("TEST 5: DataLoader with num_workers=2")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()

    class WorkerDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            dataset = torch.utils.data.TensorDataset(
                torch.randn(100, 10),
                torch.randn(100, 10)
            )
            debug_print("Creating DataLoader with num_workers=2...")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                num_workers=2,
                persistent_workers=False
            )

    debug_print("Creating WorkerDataModule...")
    datamodule = WorkerDataModule()

    debug_print("Creating Trainer...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='auto',
        max_steps=5,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    debug_print("Trainer created!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 5: PASSED ✓")
    return True


def test_6_full_combo():
    """Test 6: Everything together (WandB + workers)"""
    debug_print("=" * 80)
    debug_print("TEST 6: WandbLogger + num_workers=2 + DDPStrategy")
    debug_print("=" * 80)

    debug_print("Creating DummyModel...")
    model = DummyModel()

    class WorkerDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            dataset = torch.utils.data.TensorDataset(
                torch.randn(100, 10),
                torch.randn(100, 10)
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                num_workers=2
            )

    debug_print("Creating WorkerDataModule...")
    datamodule = WorkerDataModule()

    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        wandb_logger = WandbLogger(
            project="lightning-test",
            name="test-6-full",
            save_dir=str(logs_dir),
        )
        debug_print("WandbLogger created on rank 0")
    else:
        wandb_logger = False

    debug_print("Creating Trainer...")
    sys.stdout.flush()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp',
        max_steps=5,
        enable_checkpointing=False,
        logger=wandb_logger,
        enable_progress_bar=False,
    )
    debug_print("Trainer created!")

    debug_print("Calling trainer.fit()...")
    sys.stdout.flush()

    trainer.fit(model, datamodule)

    debug_print("trainer.fit() completed!")
    debug_print("TEST 6: PASSED ✓")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=1, choices=[1,2,3,4,5,6],
                      help='Which test to run (1-6)')
    args = parser.parse_args()

    debug_print("=" * 80)
    debug_print(f"PROGRESSIVE LIGHTNING TEST - TEST #{args.test}")
    debug_print("=" * 80)

    tests = {
        1: ("Basic Trainer (no logger)", test_1_basic_trainer),
        2: ("WandbLogger all ranks", test_2_with_wandb),
        3: ("WandbLogger rank 0 only", test_3_wandb_rank0_only),
        4: ("Explicit DDPStrategy", test_4_explicit_ddp),
        5: ("DataLoader workers", test_5_with_dataloader_workers),
        6: ("Full combo", test_6_full_combo),
    }

    test_name, test_func = tests[args.test]
    debug_print(f"Running: {test_name}")

    try:
        success = test_func()
        if success:
            debug_print("=" * 80)
            debug_print(f"TEST {args.test} COMPLETED SUCCESSFULLY ✓")
            debug_print("=" * 80)
            return 0
        else:
            debug_print(f"TEST {args.test} FAILED")
            return 1
    except Exception as e:
        debug_print("=" * 80)
        debug_print(f"TEST {args.test} FAILED WITH EXCEPTION: {type(e).__name__}: {e}")
        debug_print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
