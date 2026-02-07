#!/usr/bin/env python3
"""
Multi-GPU training test script.
Tests that multi-GPU setup is working by comparing convergence speed
with different GPU counts. With more GPUs, effective batch size is larger,
so loss should decrease faster per epoch.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import wandb


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single GPU fallback
        torch.cuda.set_device(0)
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_data(n_samples: int = 10000, n_features: int = 100, n_classes: int = 10):
    """Generate synthetic classification data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    # Create linearly separable data with some noise
    true_weights = torch.randn(n_features, n_classes)
    logits = X @ true_weights + torch.randn(n_samples, n_classes) * 0.5
    y = logits.argmax(dim=1)
    return X, y


class SimpleNN(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, n_features: int = 100, n_hidden: int = 256, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU training test")
    parser.add_argument("--batch-size-per-gpu", type=int, default=64,
                        help="Batch size per GPU (effective batch = this * n_gpus)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--wandb-project", type=str, default="multi-gpu-test",
                        help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # Effective batch size scales with GPU count
    effective_batch_size = args.batch_size_per_gpu * world_size

    if is_main:
        print(f"Running with {world_size} GPU(s)")
        print(f"Batch size per GPU: {args.batch_size_per_gpu}")
        print(f"Effective batch size: {effective_batch_size}")

        # Initialize wandb only on main process
        run_name = args.run_name or f"{world_size}gpu_bs{effective_batch_size}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "num_gpus": world_size,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "effective_batch_size": effective_batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "n_samples": args.n_samples,
            }
        )

    # Generate data
    X, y = generate_data(n_samples=args.n_samples)

    # Split into train/val
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    model = SimpleNN().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Aggregate losses across processes
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
            val_loss = val_loss_tensor.item()

        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

    if is_main:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
