import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
import pyarrow.parquet as pq
from pathlib import Path
from itertools import cycle
from collections import deque
import random

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
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
    def __init__(self, parquet_dir: str, tokenizer, seq_length: int = 2048,
                 rank: int = 0, world_size: int = 1, shuffle: bool = False,
                 max_sequences: int = None):
        self.files = sorted(Path(parquet_dir).glob("*.parquet"))
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.max_sequences = max_sequences
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
        
        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id
        print(f"Rank {self.rank}, shard_id {shard_id}, total_shards {total_shards}, num_files {len(self.files)}")
        
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)
        
        buffer = deque()
        sequences_yielded = 0
        
        for file_idx in cycle(range(len(files))) if self.max_sequences else range(len(files)):
            if file_idx % total_shards != shard_id:
                continue
            filepath = files[file_idx]
            
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=['text'])
                for text in table['text'].to_pylist():
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    buffer.extend(tokens)
                    buffer.append(self.tokenizer.eos_token_id)
                    
                    while len(buffer) >= self.seq_length + 1:
                        chunk = [buffer.popleft() for _ in range(self.seq_length + 1)]
                        yield {
                            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                            'labels': torch.tensor(chunk[1:], dtype=torch.long)
                        }
                        sequences_yielded += 1
                        if self.max_sequences and sequences_yielded >= self.max_sequences:
                            return


class LLMDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, tokenizer, batch_size: int = 8,
                 seq_length: int = 2048, num_workers: int = 4, val_sequences: int = 1000):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_workers = num_workers
        self.val_sequences = val_sequences
    
    def _collate(self, batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }
    
    def train_dataloader(self):
        dataset = StreamingParquetDataset(
            self.train_dir, self.tokenizer, self.seq_length,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            shuffle=True
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self._collate,
            num_workers=self.num_workers, pin_memory=True, prefetch_factor=2
        )
    
    def val_dataloader(self):
        per_gpu_sequences = self.val_sequences // self.trainer.world_size
        dataset = StreamingParquetDataset(
            self.val_dir, self.tokenizer, self.seq_length,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            shuffle=False,
            max_sequences=per_gpu_sequences
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self._collate,
            num_workers=self.num_workers, pin_memory=True
        )


class LLMModule(pl.LightningModule):
    def __init__(self, model: TinyGPT, tokenizer, base_lr: float = 1e-3, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.0, warmdown_ratio: float = 0.2, final_lr_frac: float = 0.0,
                 max_steps: int = 100000, dim: int = 64):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.warmdown_ratio = warmdown_ratio
        self.final_lr_frac = final_lr_frac
        self.max_steps = max_steps
        self.dim = dim
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def _compute_loss(self, batch):
        input_ids, labels = batch['input_ids'], batch['labels']
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        # Log learning rates for each parameter group
        optimizer = self.trainer.optimizers[0]
        if optimizer is not None:
            param_groups = optimizer.param_groups
            self.log('lr/head', param_groups[0]['lr'], prog_bar=False, sync_dist=False)
            self.log('lr/embeddings', param_groups[1]['lr'], prog_bar=False, sync_dist=False)
            self.log('lr/other', param_groups[2]['lr'], prog_bar=False, sync_dist=False)

            # Log optimizer phase
            current_step = self.global_step
            max_steps = self.trainer.max_steps
            warmup_steps = int(self.warmup_ratio * max_steps)
            warmdown_start = max_steps - int(self.warmdown_ratio * max_steps)

            if current_step < warmup_steps:
                phase = 0  # warmup
            elif current_step < warmdown_start:
                phase = 1  # constant
            else:
                phase = 2  # warmdown

            self.log('opt/phase', phase, prog_bar=False, sync_dist=False)
            self.log('opt/step_progress', current_step / max_steps, prog_bar=False, sync_dist=False)

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
            memory_cached = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB

            self.log('memory/allocated_gb', memory_allocated, prog_bar=False, sync_dist=False)
            self.log('memory/reserved_gb', memory_reserved, prog_bar=False, sync_dist=False)
            self.log('memory/max_allocated_gb', memory_cached, prog_bar=False, sync_dist=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._compute_loss(batch)
        ppl = torch.exp(loss)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/perplexity', ppl, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Run validation only on rank 0
        if self.global_rank == 0:
            from core.validation import run_world_knowledge_validation, run_sentence_completion

            # Run world knowledge validation with metrics
            world_knowledge_results = run_world_knowledge_validation(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=20,
                temperature=0.3,
                top_k=40
            )

            # Log world knowledge metrics to wandb
            if self.logger and 'metrics' in world_knowledge_results:
                metrics_to_log = {f"val/{k}": v for k, v in world_knowledge_results['metrics'].items()}
                self.log_dict(metrics_to_log, sync_dist=False)

            # Run sentence completion
            sentence_completions = run_sentence_completion(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )

            # Log sentence completions to wandb
            if self.logger and sentence_completions:
                sentence_completion_data = [
                    [self.global_step, item['prompt'], item['completion']]
                    for item in sentence_completions
                ]
                self.logger.log_text(
                    key="sentence_completions",
                    columns=["step", "prompt", "completion"],
                    data=sentence_completion_data
                )
    
    def configure_optimizers(self):
        # Group parameters by type (head, embeddings, other)
        embedding_params = list(self.model.tok_emb.parameters()) + list(self.model.pos_emb.parameters())
        head_params = list(self.model.head.parameters())
        other_params = list(self.model.blocks.parameters()) + list(self.model.ln_f.parameters())

        # LR scaling based on model dimension
        lr_scale = (self.dim / 768) ** -0.5

        optimizer = torch.optim.AdamW([
            {'params': head_params, 'lr': 0.004 * lr_scale * self.base_lr},
            {'params': embedding_params, 'lr': 0.2 * lr_scale * self.base_lr},
            {'params': other_params, 'lr': 0.02 * self.base_lr}
        ], betas=(0.8, 0.95), eps=1e-10, weight_decay=self.weight_decay)

        def lr_lambda(current_step):
            max_steps = self.trainer.max_steps
            warmup_steps = int(self.warmup_ratio * max_steps)
            warmdown_steps = int(self.warmdown_ratio * max_steps)

            if current_step < warmup_steps:
                # Linear warmup
                return (current_step + 1) / warmup_steps if warmup_steps > 0 else 1.0
            elif current_step <= max_steps - warmdown_steps:
                # Constant
                return 1.0
            else:
                # Linear warmdown
                progress = (max_steps - current_step) / warmdown_steps
                return progress + (1 - progress) * self.final_lr_frac

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from pytorch_lightning.loggers import WandbLogger
    from core.memory_tracker import MemoryLoggingCallback, estimate_tensor_memory

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TinyGPT model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--smoke_test', action='store_true', help='Run a quick smoke test with reduced steps')
    args = parser.parse_args()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Override max_steps if smoke_test is enabled
    if args.smoke_test:
        max_steps = 100
        val_check_interval = 50
        print("Running in smoke test mode: max_steps=100, val_check_interval=50")
    else:
        max_steps = args.max_steps
        val_check_interval = 1000

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Model architecture parameters
    n_layers = 20
    max_seq_len = 2048
    batch_size = args.batch_size
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )

    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = model.estimate_flops(2048)
    print(f"Parameters: {num_params:,}")
    print(f"FLOPs/token: {flops_per_token:,}")

    # Memory estimation
    if torch.cuda.is_available():
        print("\n=== Memory Estimation ===")
        grad_accum_steps = 4
        memory_estimate = estimate_tensor_memory(
            model,
            batch_size * grad_accum_steps,  # Effective batch size
            max_seq_len,
            tokenizer.vocab_size,
            dtype=torch.bfloat16  # Using bf16-mixed
        )
        print(f"Parameters: {memory_estimate['parameters_mb']:.2f} MB")
        print(f"Gradients: {memory_estimate['gradients_mb']:.2f} MB")
        print(f"Optimizer State: {memory_estimate['optimizer_state_mb']:.2f} MB")
        print(f"Activations (est): {memory_estimate['activations_mb']:.2f} MB")
        print(f"Data (per batch): {memory_estimate['data_mb']:.2f} MB")
        print(f"Logits (per batch): {memory_estimate['logits_mb']:.2f} MB")
        print(f"Total Estimated: {memory_estimate['total_estimated_mb']:.2f} MB")

    # Optimizer parameters
    base_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0

    # Print optimizer configuration
    lr_scale = (dim / 768) ** -0.5
    print(f"\n=== Optimizer Configuration ===")
    print(f"Base LR: {base_lr}")
    print(f"LR scale factor: {lr_scale:.4f}")
    print(f"Head LR: {0.004 * lr_scale * base_lr:.6f}")
    print(f"Embedding LR: {0.2 * lr_scale * base_lr:.6f}")
    print(f"Other LR: {0.02 * base_lr:.6f}")
    print(f"Weight decay: {weight_decay}")
    print(f"Warmup ratio: {warmup_ratio}")
    print(f"Warmdown ratio: {warmdown_ratio}")

    warmup_steps = int(warmup_ratio * max_steps)
    warmdown_steps = int(warmdown_ratio * max_steps)
    constant_steps = max_steps - warmup_steps - warmdown_steps
    print(f"Warmup steps: {warmup_steps}")
    print(f"Constant steps: {constant_steps}")
    print(f"Warmdown steps: {warmdown_steps}")

    llm_module = LLMModule(
        model,
        tokenizer,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmdown_ratio=warmdown_ratio,
        final_lr_frac=final_lr_frac,
        max_steps=max_steps,
        dim=dim
    )
    data_module = LLMDataModule(
        train_dir='data/base_data',
        val_dir='data/base_data',
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_length=max_seq_len,
        num_workers=4,
        val_sequences=10
    )

    # Auto-detect available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        accelerator = 'gpu'
        devices = num_gpus
        strategy = 'ddp' if num_gpus > 1 else 'auto'
        print(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
        strategy = 'auto'
        print("Detected Apple MPS device")
    else:
        accelerator = 'cpu'
        devices = 1
        strategy = 'auto'
        print("No GPU detected, using CPU")

    # Initialize wandb logger with logs directory
    wandb_logger = WandbLogger(
        project="llm-training",
        name=f"tinygpt-{num_params//1000}k-params",
        save_dir=str(logs_dir),  # Save all wandb logs to logs directory
        log_model=True,
        config={
            "model_params": num_params,
            "flops_per_token": flops_per_token,
            "vocab_size": tokenizer.vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "seq_length": max_seq_len,
            "base_lr": base_lr,
            "lr_scale": lr_scale,
            "head_lr": 0.004 * lr_scale * base_lr,
            "embedding_lr": 0.2 * lr_scale * base_lr,
            "other_lr": 0.02 * base_lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
            "max_steps": max_steps,
            "accelerator": accelerator,
            "devices": devices,
            "strategy": strategy,
            "smoke_test": args.smoke_test,
        }
    )

    # Create memory logging callback
    memory_callback = MemoryLoggingCallback(log_every_n_steps=10)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision='bf16-mixed' if accelerator == 'gpu' else '32',
        max_steps=max_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        logger=wandb_logger,  # Add wandb logger
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[memory_callback],
    )

    trainer.fit(llm_module, data_module)