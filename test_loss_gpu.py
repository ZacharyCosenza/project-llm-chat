#!/usr/bin/env python3
"""
Multi-GPU training test script using PyTorch Lightning.
Trains ResNet-18 on CIFAR-10 to verify multi-GPU setup.
With more GPUs, effective batch size is larger, so loss should decrease faster per epoch.
"""

import argparse

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torchvision.models import resnet18


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.train_transform
        )
        self.val_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ResNet18CIFAR(L.LightningModule):
    def __init__(self, lr: float = 0.1, epochs: int = 50):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs

        # Create ResNet-18 adapted for CIFAR-10 (32x32 images)
        self.model = resnet18(weights=None, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        self.val_acc(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--wandb-project", type=str, default="multi-gpu-test",
                        help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    args = parser.parse_args()

    # Determine number of GPUs
    num_gpus = args.num_gpus or torch.cuda.device_count()
    effective_batch_size = args.batch_size_per_gpu * num_gpus

    # Auto-generate run name if not specified
    run_name = args.run_name or f"{num_gpus}gpu_bs{effective_batch_size}"

    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": "resnet18",
            "dataset": "cifar10",
            "num_gpus": num_gpus,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "effective_batch_size": effective_batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        }
    )

    # Create data module and model
    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size_per_gpu,
    )
    model = ResNet18CIFAR(lr=args.lr, epochs=args.epochs)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp" if num_gpus > 1 else "auto",
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        enable_progress_bar=True,
    )

    print(f"Training ResNet-18 on CIFAR-10")
    print(f"Running with {num_gpus} GPU(s)")
    print(f"Batch size per GPU: {args.batch_size_per_gpu}")
    print(f"Effective batch size: {effective_batch_size}")

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
