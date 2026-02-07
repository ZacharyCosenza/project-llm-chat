#!/usr/bin/env python3
"""
Multi-GPU training test script.
Trains ResNet-18 on CIFAR-10 to verify multi-GPU setup.
With more GPUs, effective batch size is larger, so loss should decrease faster per epoch.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18
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


def get_cifar10_loaders(batch_size: int, data_dir: str, world_size: int, num_workers: int = 4):
    """Get CIFAR-10 train and val dataloaders with proper transforms."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def create_resnet18_cifar(num_classes: int = 10):
    """Create ResNet-18 adapted for CIFAR-10 (32x32 images)."""
    model = resnet18(weights=None, num_classes=num_classes)
    # Adapt first conv layer for 32x32 images (smaller kernel, no stride)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool (not needed for small images)
    model.maxpool = nn.Identity()
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU ResNet-18 CIFAR-10 training test")
    parser.add_argument("--batch-size-per-gpu", type=int, default=128,
                        help="Batch size per GPU (effective batch = this * n_gpus)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to store CIFAR-10 data")
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
        print(f"Training ResNet-18 on CIFAR-10")
        print(f"Running with {world_size} GPU(s)")
        print(f"Batch size per GPU: {args.batch_size_per_gpu}")
        print(f"Effective batch size: {effective_batch_size}")

        # Initialize wandb only on main process
        run_name = args.run_name or f"{world_size}gpu_bs{effective_batch_size}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "resnet18",
                "dataset": "cifar10",
                "num_gpus": world_size,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "effective_batch_size": effective_batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
            }
        )

    # Load CIFAR-10 data
    train_loader, val_loader, train_sampler = get_cifar10_loaders(
        batch_size=args.batch_size_per_gpu,
        data_dir=args.data_dir,
        world_size=world_size,
    )

    # Create model
    model = create_resnet18_cifar(num_classes=10).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Aggregate metrics across processes
        if world_size > 1:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()

        if is_main:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            })

    if is_main:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
