"""
TinyGPT Multi-GPU Training Script with PyTorch Lightning
Memory-efficient streaming dataloader for large parquet datasets
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from typing import Iterator
import argparse
import wandb


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=n_heads, 
                dim_feedforward=dim * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
    def estimate_flops(self, sequence_len):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        l = len(self.blocks)
        h = self.blocks[0].self_attn.num_heads
        q = self.blocks[0].self_attn.embed_dim // h
        t = sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token 
       
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def predict(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.pos_emb.num_embeddings else idx[:, -self.pos_emb.num_embeddings:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class StreamingParquetDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that:
    - Reads parquet files one at a time
    - Processes row groups within files to limit memory
    - Shards data across DDP workers
    - Uses a token buffer to pack sequences efficiently
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        seq_len: int = 128,
        text_column: str = "text",
        shuffle_files: bool = True,
        row_group_batch_size: int = 1000,  # rows to load at once from parquet
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.shuffle_files = shuffle_files
        self.row_group_batch_size = row_group_batch_size
        
        self.parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")
        
    def _get_worker_files(self) -> list:
        """Shard files across DDP ranks and dataloader workers."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Get DDP rank info
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Shard by DDP rank first
        files = self.parquet_files[rank::world_size]
        
        # Then shard by dataloader worker
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
        
        return files
    
    def _stream_tokens(self, files: list) -> Iterator[int]:
        """Stream tokens from parquet files one row group at a time."""
        for filepath in files:
            pf = pq.ParquetFile(filepath)
            
            for batch in pf.iter_batches(
                batch_size=self.row_group_batch_size,
                columns=[self.text_column]
            ):
                texts = batch[self.text_column].to_pylist()
                
                for text in texts:
                    if text:
                        tokens = self.tokenizer.encode(text)
                        yield from tokens
                        
                # Free memory after each batch
                del texts, batch
    
    def __iter__(self) -> Iterator[dict]:
        """Yield packed sequences of seq_len + 1 tokens (for input/target shift)."""
        files = self._get_worker_files()
        
        if self.shuffle_files:
            import random
            # Use a seed based on epoch for reproducibility
            # Worker info provides different shuffles per worker
            worker_info = torch.utils.data.get_worker_info()
            seed = worker_info.id if worker_info else 0
            rng = random.Random(seed)
            files = files.copy()
            rng.shuffle(files)
        
        token_buffer = []
        required_len = self.seq_len + 1  # +1 for shifted target
        
        for token in self._stream_tokens(files):
            token_buffer.append(token)
            
            if len(token_buffer) >= required_len:
                # Yield a complete sequence
                seq = torch.tensor(token_buffer[:required_len], dtype=torch.long)
                yield {"input_ids": seq[:-1], "labels": seq[1:]}
                
                # Keep remainder for next sequence (packing)
                token_buffer = token_buffer[required_len:]
        
        # Don't yield partial final buffer - it would cause issues with batching


class TinyGPTLightning(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_seq_len: int = 128,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TinyGPT(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
    def forward(self, x):
        return self.model(x)
    
    def _get_memory_stats(self) -> dict:
        """Get CUDA memory statistics for the current device."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'memory/allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'memory/reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'memory/peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2,
            'memory/peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        ppl = torch.exp(loss)
        
        # Log training metrics
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/ppl", ppl, prog_bar=True, sync_dist=True)
        
        # Log memory stats (only on rank 0 to avoid noise)
        if self.global_rank == 0 and batch_idx % 10 == 0:
            memory_stats = self._get_memory_stats()
            for key, value in memory_stats.items():
                self.log(key, value, rank_zero_only=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        ppl = torch.exp(loss)
        
        # Log validation metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ppl", ppl, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        # Separate weight decay for different param groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "ln" in name or "bias" in name or "emb" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.learning_rate, betas=(0.9, 0.95))
        
        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


class ParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        tokenizer_name: str = "gpt2",
        seq_len: int = 128,
        batch_size: int = 64,
        num_workers: int = 4,
        text_column: str = "text",
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir or train_dir  # Use train for val if not specified
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_column = text_column
        
        self.tokenizer = None
        
    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def train_dataloader(self):
        dataset = StreamingParquetDataset(
            data_dir=self.train_dir,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            text_column=self.text_column,
            shuffle_files=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        dataset = StreamingParquetDataset(
            data_dir=self.val_dir,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            text_column=self.text_column,
            shuffle_files=False,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def main():
    parser = argparse.ArgumentParser(description="Train TinyGPT on parquet data")
    
    # Data args
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training parquet files")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory with validation parquet files")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HuggingFace tokenizer name")
    
    # Model args
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--val_check_interval", type=int, default=500, help="Validation every N steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers per GPU")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    
    # Hardware args
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs (-1 for all)")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--run_name", type=str, default="tinygpt", help="Run name for logging")
    
    # Wandb args
    parser.add_argument("--wandb_project", type=str, default="tinygpt", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Create data module
    data_module = ParquetDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_column=args.text_column,
    )
    
    # Create model
    model = TinyGPTLightning(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.run_name),
            filename="step_{step:06d}-loss_{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            every_n_train_steps=args.val_check_interval,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Determine number of GPUs
    num_gpus = args.gpus if args.gpus > 0 else torch.cuda.device_count()
    
    # Wandb logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            save_dir=args.output_dir,
            log_model=False,  # Don't auto-upload checkpoints
            config={
                "vocab_size": args.vocab_size,
                "dim": args.dim,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "precision": args.precision,
                "num_gpus": num_gpus,
            }
        )

    
    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else "auto",
        precision=args.precision,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Finish wandb run
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()