import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset
import pyarrow.parquet as pq
from collections import deque
from core.dataset import list_parquet_files
from core.utils import print0, get_dist_info
from core.models import TinyGPT


class ParquetTokenDataset(IterableDataset):
    """
    Iterable dataset that loads parquet files and tokenizes on-the-fly.
    Properly handles DDP sharding using PyTorch's worker_info.
    """

    def __init__(self, tokenizer, batch_size, max_seq_len, split="train", device=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.split = split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Get parquet files for this split
        all_paths = list_parquet_files()
        self.paths = all_paths[:-1] if split == "train" else all_paths[-1:]

    def __iter__(self):
        # Get DDP worker info from PyTorch's distributed context
        worker_info = torch.utils.data.get_worker_info()

        # Determine rank and world size for DDP sharding
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # If using multiple dataloader workers, further shard the data
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Combine DDP rank with worker ID for sharding
            shard_id = rank * num_workers + worker_id
            total_shards = world_size * num_workers
        else:
            shard_id = rank
            total_shards = world_size

        # Token buffer for accumulating tokens
        needed_tokens = self.batch_size * self.max_seq_len + 1
        token_buffer = deque()

        # Iterate through parquet files indefinitely
        while True:
            for filepath in self.paths:
                pf = pq.ParquetFile(filepath)

                # Shard row groups across DDP processes and workers
                for rg_idx in range(shard_id, pf.num_row_groups, total_shards):
                    # Read row group and extract texts
                    texts = pf.read_row_group(rg_idx).column('text').to_pylist()

                    # Tokenize texts and add to buffer
                    for text in texts:
                        tokens = self.tokenizer.encode(
                            text,
                            add_special_tokens=True,
                            max_length=2048,
                            truncation=True
                        )
                        token_buffer.extend(tokens)

                        # Yield batches whenever we have enough tokens
                        while len(token_buffer) >= needed_tokens:
                            # Extract exactly the needed tokens
                            batch_tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

                            # Create tensors
                            use_pin = self.device == "cuda"
                            scratch = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=use_pin)

                            x = scratch[:-1].view(self.batch_size, self.max_seq_len).to(self.device, non_blocking=use_pin)
                            y = scratch[1:].view(self.batch_size, self.max_seq_len).to(self.device, non_blocking=use_pin)

                            if self.split == "train":
                                # Training data includes state (unused but kept for compatibility)
                                yield x, y, {}
                            else:
                                # Validation data is just (x, y)
                                yield x, y


class LightningGPT(pl.LightningModule):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=2,
                 base_lr=1e-3, weight_decay=0.01,
                 warmup_ratio=0.0, warmdown_ratio=0.2, final_lr_frac=0.0,
                 tokenizer=None, max_seq_len=128, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])

        self.model = TinyGPT(vocab_size, dim, n_layers, n_heads, max_seq_len)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.flops_per_token = self.model.estimate_flops(max_seq_len)
        self._train_loader = None
        self._val_loader = None
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        B, T = x.shape  # batch size and sequence length
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        _, _, _, world_size = get_dist_info()
        flops_so_far = self.flops_per_token * B * T * world_size * (self.global_step + 1)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('flops_so_far', flops_so_far, prog_bar=False, sync_dist=True)

        # Log optimizer parameters
        optimizer = self.optimizers()
        if optimizer is not None:
            # Log learning rates for each parameter group
            param_groups = optimizer.param_groups
            self.log('lr/head', param_groups[0]['lr'], prog_bar=False, sync_dist=True)
            self.log('lr/embeddings', param_groups[1]['lr'], prog_bar=False, sync_dist=True)
            self.log('lr/other', param_groups[2]['lr'], prog_bar=False, sync_dist=True)

            # Log scheduler progress
            current_step = self.global_step
            max_steps = self.trainer.max_steps
            warmup_steps = int(self.hparams.warmup_ratio * max_steps)
            warmdown_start = max_steps - int(self.hparams.warmdown_ratio * max_steps)

            if current_step < warmup_steps:
                phase = 0  # warmup
            elif current_step < warmdown_start:
                phase = 1  # constant
            else:
                phase = 2  # warmdown

            self.log('opt/phase', phase, prog_bar=False, sync_dist=True)
            self.log('opt/step_progress', current_step / max_steps, prog_bar=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_perplexity', perplexity, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        embedding_params = list(self.model.tok_emb.parameters()) + list(self.model.pos_emb.parameters())
        head_params = list(self.model.head.parameters())
        other_params = list(self.model.blocks.parameters()) + list(self.model.ln_f.parameters())
        lr_scale = (self.hparams.dim / 768) ** -0.5
        
        optimizer = torch.optim.AdamW([
            {'params': head_params, 'lr': 0.004 * lr_scale * self.hparams.base_lr},
            {'params': embedding_params, 'lr': 0.2 * lr_scale * self.hparams.base_lr},
            {'params': other_params, 'lr': 0.02 * self.hparams.base_lr}
        ], betas=(0.8, 0.95), eps=1e-10, weight_decay=self.hparams.weight_decay)
        
        def lr_lambda(current_step):
            max_steps = self.trainer.max_steps
            warmup_steps = int(self.hparams.warmup_ratio * max_steps)
            warmdown_steps = int(self.hparams.warmdown_ratio * max_steps)
            
            if current_step < warmup_steps:
                # Linear warmup
                return (current_step + 1) / warmup_steps
            elif current_step <= max_steps - warmdown_steps:
                # Constant
                return 1.0
            else:
                # Linear warmdown
                progress = (max_steps - current_step) / warmdown_steps
                return progress + (1 - progress) * self.hparams.final_lr_frac
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        print(f'loading train dataloader')
        if self._train_loader is None:
            device = "cuda" if self.device.type == "cuda" else "cpu"
            dataset = ParquetTokenDataset(
                self.tokenizer,
                self.batch_size,
                self.max_seq_len,
                split="train",
                device=device
            )
            # Use PyTorch DataLoader with DDP-compatible settings
            # batch_size=None because the dataset already yields batched data
            # num_workers=0 to avoid multiprocessing issues with tokenizers
            # The dataset handles DDP sharding internally using torch.distributed
            self._train_loader = DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,  # Already handled in the dataset
                persistent_workers=False
            )
        return self._train_loader

    def val_dataloader(self):
        print(f'loading val dataloader')
        # Lazy initialization: create dataloader after DDP has spawned processes
        # This is called AFTER setup(), so self.device will be properly set by Lightning
        if self._val_loader is None:
            device = "cuda" if self.device.type == "cuda" else "cpu"
            dataset = ParquetTokenDataset(
                self.tokenizer,
                self.batch_size,
                self.max_seq_len,
                split="val",
                device=device
            )
            # Use PyTorch DataLoader with DDP-compatible settings
            self._val_loader = DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False
            )
        return self._val_loader

def train_gpt(
    batch_size=32,
    max_seq_len=128,
    max_steps=1000,
    val_max_steps=2,
    smoke_test=False,
    wandb_name=None
):
    # Model architecture parameters\
    n_layers = 20
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)

    # Optimizer parameters
    base_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0

    # Training parameters
    eval_every = 250
    grad_clip = 1.0
    grad_accum_steps = 1
    devices = None

    # Logging parameters
    wandb_project = "llm-chat"
    wandb_enabled = True
    # Get distributed info (will be correct after DDP spawns processes)
    is_ddp, rank, local_rank, world_size = get_dist_info()

    # Print distributed info
    print0("=== Training Configuration ===")
    print0(f"DDP Mode: {is_ddp}")
    if is_ddp:
        print0(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    print0(f"Accelerator: {accelerator}")
    print0(f"Devices: {devices}")

    torch.set_float32_matmul_precision('medium')

    # Logger
    logger = None
    if wandb_enabled and not smoke_test:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            log_model=True,
            save_dir="logs",
            config={
                "dim": dim,
                "n_layers": n_layers,
                "base_lr": base_lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_steps": max_steps,
            }
        )

    # Tokenizer setup (no CUDA calls here)
    print0("\n=== Loading Data ===")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print0(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print0(f"Batch size: {batch_size}, Max sequence length: {max_seq_len}")
    print0("Note: Dataloaders will be created lazily after DDP initialization")
    
    # Model
    print0("\n=== Model Configuration ===")
    print0(f"Dimension: {dim}")
    print0(f"Layers: {n_layers}")
    print0(f"Heads: {n_heads}")
    print0(f"Vocab size: {tokenizer.vocab_size}")

    model = LightningGPT(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmdown_ratio=warmdown_ratio,
        final_lr_frac=final_lr_frac,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(f"Total parameters: {total_params:,}")
    print0(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer
    print0("\n=== Training Settings ===")
    print0(f"Max steps: {max_steps}")
    print0(f"Eval every: {eval_every} steps")
    print0(f"Validation max steps: {val_max_steps}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Gradient clipping: {grad_clip if grad_clip > 0 else 'None'}")
    print0(f"Strategy: {'ddp' if devices > 1 else 'auto'}")
    print0(f"Precision: {'bf16-mixed' if accelerator == 'gpu' else '32-true'}")
    print0(f"Smoke test: {smoke_test}")

    print0("\n=== Optimizer Settings ===")
    lr_scale = (dim / 768) ** -0.5
    print0(f"Base LR: {base_lr}")
    print0(f"LR scale factor: {lr_scale:.4f}")
    print0(f"Head LR: {0.004 * lr_scale * base_lr:.6f}")
    print0(f"Embedding LR: {0.2 * lr_scale * base_lr:.6f}")
    print0(f"Other LR: {0.02 * base_lr:.6f}")
    print0(f"Weight decay: {weight_decay}")
    print0(f"Warmup ratio: {warmup_ratio}")
    print0(f"Warmdown ratio: {warmdown_ratio}")

    warmup_steps = int(warmup_ratio * max_steps)
    warmdown_steps = int(warmdown_ratio * max_steps)
    constant_steps = max_steps - warmup_steps - warmdown_steps
    print0(f"Warmup steps: {warmup_steps}")
    print0(f"Constant steps: {constant_steps}")
    print0(f"Warmdown steps: {warmdown_steps}")

    if wandb_enabled and not smoke_test:
        print0(f"\nWandB logging enabled: {wandb_project}/{wandb_name}")
    else:
        print0("\nWandB logging disabled")

    print0("\n=== Starting Training ===\n")

    trainer = pl.Trainer(
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        strategy="ddp" if devices > 1 else "auto",
        val_check_interval=eval_every,
        limit_val_batches=val_max_steps,
        fast_dev_run=smoke_test,
        precision="bf16-mixed" if accelerator == "gpu" else '32-true',
        gradient_clip_val=grad_clip if grad_clip > 0 else None,
        accumulate_grad_batches=grad_accum_steps,
        logger=logger,
        default_root_dir="logs"
    )

    trainer.fit(model)

    print0("\n=== Training Complete ===")
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print0(f"Peak GPU memory: {max_memory_mb:.2f} MB")

    return model, trainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per rank")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--val_max_steps", type=int, default=10, help="Maximum validation steps")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test (fast dev run)")
    parser.add_argument("--wandb_name", type=str, default="test", help="Weights & Biases run name")

    args = parser.parse_args()

    train_gpt(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        val_max_steps=args.val_max_steps,
        smoke_test=args.smoke_test,
        wandb_name=args.wandb_name
    )