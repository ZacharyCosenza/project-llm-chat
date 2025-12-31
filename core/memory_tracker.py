"""
VRAM Memory Tracking Utilities for PyTorch
"""
import torch
import pytorch_lightning as pl
from typing import Dict, Optional


class MemoryTracker:
    """Utility class to track GPU memory usage broken down by component"""

    def __init__(self, device: torch.device):
        self.device = device
        self.baseline_allocated = 0
        self.baseline_reserved = 0

    def reset_peak_stats(self):
        """Reset peak memory stats"""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics in MB"""
        if self.device.type != "cuda":
            return {}

        stats = {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2,
            'peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }

        # Calculate active memory (reserved but not allocated = fragmentation)
        stats['fragmentation_mb'] = stats['reserved_mb'] - stats['allocated_mb']

        return stats

    def set_baseline(self):
        """Set baseline memory (model parameters + optimizer state)"""
        if self.device.type == "cuda":
            self.baseline_allocated = torch.cuda.memory_allocated(self.device)
            self.baseline_reserved = torch.cuda.memory_reserved(self.device)

    def get_detailed_breakdown(self) -> Dict[str, float]:
        """Get detailed memory breakdown using PyTorch memory snapshot"""
        if self.device.type != "cuda":
            return {}

        stats = self.get_memory_stats()

        # Get memory summary for detailed breakdown
        try:
            memory_summary = torch.cuda.memory_summary(self.device, abbreviated=False)
            # Parse the summary (it's a string)
            # This gives us insights into different tensor categories
        except:
            pass

        return stats


class MemoryLoggingCallback(pl.Callback):
    """PyTorch Lightning callback to log memory usage at different training stages"""

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.tracker = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize memory tracker when training starts"""
        if pl_module.device.type == "cuda":
            self.tracker = MemoryTracker(pl_module.device)

            # Get initial memory (model parameters)
            initial_stats = self.tracker.get_memory_stats()
            if trainer.global_rank == 0:
                print(f"\n=== Initial Memory (Model Parameters) ===")
                print(f"Allocated: {initial_stats['allocated_mb']:.2f} MB")
                print(f"Reserved: {initial_stats['reserved_mb']:.2f} MB")

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        """Track memory before forward pass"""
        if self.tracker and trainer.global_step % self.log_every_n_steps == 0:
            self.tracker.reset_peak_stats()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Track memory after backward pass"""
        if not self.tracker:
            return

        if trainer.global_step % self.log_every_n_steps == 0:
            stats = self.tracker.get_memory_stats()

            # Log to tensorboard/wandb
            if trainer.logger:
                pl_module.log('memory/allocated_mb', stats['allocated_mb'], prog_bar=False, sync_dist=False)
                pl_module.log('memory/reserved_mb', stats['reserved_mb'], prog_bar=False, sync_dist=False)
                pl_module.log('memory/peak_allocated_mb', stats['peak_allocated_mb'], prog_bar=False, sync_dist=False)
                pl_module.log('memory/fragmentation_mb', stats['fragmentation_mb'], prog_bar=False, sync_dist=False)


def estimate_tensor_memory(model: torch.nn.Module, batch_size: int, seq_len: int,
                           vocab_size: int, dtype=torch.float32) -> Dict[str, float]:
    """
    Estimate memory usage for different components analytically.
    This is useful for planning before training.
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2  # float32 or float16/bfloat16

    # Model parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_memory_mb = (param_count * bytes_per_param) / 1024**2

    # Gradients (same size as parameters)
    grad_memory_mb = param_memory_mb

    # Optimizer state (AdamW has 2 states per parameter: momentum and variance)
    optimizer_memory_mb = param_memory_mb * 2

    # Activations (rough estimate for transformer)
    # This is highly architecture-dependent
    # For transformers: embeddings + attention + MLP activations
    n_layers = len([m for m in model.modules() if hasattr(m, 'attn')])
    activation_memory_mb = (batch_size * seq_len * param_count / n_layers * 4 * bytes_per_param) / 1024**2 if n_layers > 0 else 0

    # Input data
    data_memory_mb = (batch_size * seq_len * 4) / 1024**2  # int64 tokens

    # Logits (output)
    logits_memory_mb = (batch_size * seq_len * vocab_size * bytes_per_param) / 1024**2

    total_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb + data_memory_mb + logits_memory_mb

    return {
        'parameters_mb': param_memory_mb,
        'gradients_mb': grad_memory_mb,
        'optimizer_state_mb': optimizer_memory_mb,
        'activations_mb': activation_memory_mb,
        'data_mb': data_memory_mb,
        'logits_mb': logits_memory_mb,
        'total_estimated_mb': total_mb,
    }
