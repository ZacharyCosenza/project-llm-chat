import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from core.utils import print0
from core.models import TinyGPT
from core.dataloader import LLMDataModule
import argparse
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from core.memory_tracker import MemoryLoggingCallback
import time
import os

# Debug helper to print with rank and timestamp
def debug_print(msg, force_all_ranks=False):
    """Print debug message with rank, timestamp, and process info"""
    rank = int(os.environ.get('LOCAL_RANK', -1))
    global_rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    timestamp = time.strftime("%H:%M:%S")

    if force_all_ranks or rank <= 0:
        prefix = f"[{timestamp}][R{global_rank}/{world_size}][LR{rank}]"
        print(f"{prefix} {msg}", flush=True)

class LLMModule(pl.LightningModule):
    def __init__(self, model: TinyGPT, tokenizer, base_lr: float = 1e-3, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.0, warmdown_ratio: float = 0.2, final_lr_frac: float = 0.0,
                 max_steps: int = 100000, dim: int = 64):
        debug_print("LLMModule.__init__() START")
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
        debug_print("LLMModule.__init__() END")

    def forward(self, input_ids):
        return self.model(input_ids)

    def _compute_loss(self, batch):
        input_ids, labels = batch['input_ids'], batch['labels']
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        debug_print(f"training_step() START - batch_idx={batch_idx}, global_step={self.global_step}")

        loss, _, _ = self._compute_loss(batch)
        debug_print(f"training_step() - loss computed: {loss.item():.4f}")

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        debug_print(f"training_step() - loss logged")

        optimizer = self.trainer.optimizers[0]
        if optimizer is not None:
            param_groups = optimizer.param_groups
            self.log('lr/head', param_groups[0]['lr'], prog_bar=False, sync_dist=False)
            self.log('lr/embeddings', param_groups[1]['lr'], prog_bar=False, sync_dist=False)
            self.log('lr/other', param_groups[2]['lr'], prog_bar=False, sync_dist=False)

            current_step = self.global_step
            max_steps = self.trainer.max_steps
            warmup_steps = int(self.warmup_ratio * max_steps)
            warmdown_start = max_steps - int(self.warmdown_ratio * max_steps)

            if current_step < warmup_steps:
                phase = 0
            elif current_step < warmdown_start:
                phase = 1
            else:
                phase = 2

            self.log('opt/phase', phase, prog_bar=False, sync_dist=False)
            self.log('opt/step_progress', current_step / max_steps, prog_bar=False, sync_dist=False)

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_cached = torch.cuda.max_memory_allocated() / 1024**3

            self.log('memory/allocated_gb', memory_allocated, prog_bar=False, sync_dist=False)
            self.log('memory/reserved_gb', memory_reserved, prog_bar=False, sync_dist=False)
            self.log('memory/max_allocated_gb', memory_cached, prog_bar=False, sync_dist=False)

        debug_print(f"training_step() END - batch_idx={batch_idx}")
        return loss

    def validation_step(self, batch, batch_idx):
        debug_print(f"validation_step() START - batch_idx={batch_idx}, rank={self.global_rank}")

        loss, logits, labels = self._compute_loss(batch)
        ppl = torch.exp(loss)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        debug_print(f"validation_step() - metrics computed, about to log")
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/perplexity', ppl, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True)

        debug_print(f"validation_step() END - batch_idx={batch_idx}")
        return loss

    def on_validation_epoch_start(self):
        debug_print(f"on_validation_epoch_start() - rank={self.global_rank}", force_all_ranks=True)

    def on_validation_epoch_end(self):
        debug_print(f"on_validation_epoch_end() START - rank={self.global_rank}", force_all_ranks=True)

        if self.global_rank == 0:
            debug_print("on_validation_epoch_end() - rank 0 running world knowledge validation")
            from core.validation import run_world_knowledge_validation, run_sentence_completion

            world_knowledge_results = run_world_knowledge_validation(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=20,
                temperature=0.3,
                top_k=40
            )
            debug_print("on_validation_epoch_end() - world knowledge validation complete")

            if self.logger and 'metrics' in world_knowledge_results:
                metrics_to_log = {f"val/{k}": v for k, v in world_knowledge_results['metrics'].items()}
                self.log_dict(metrics_to_log, sync_dist=False)
                debug_print("on_validation_epoch_end() - world knowledge metrics logged")

            debug_print("on_validation_epoch_end() - running sentence completion")
            sentence_completions = run_sentence_completion(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )
            debug_print("on_validation_epoch_end() - sentence completion complete")

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
                debug_print("on_validation_epoch_end() - sentence completions logged")
        else:
            debug_print(f"on_validation_epoch_end() - rank {self.global_rank} waiting for rank 0")

        debug_print(f"on_validation_epoch_end() END - rank={self.global_rank}", force_all_ranks=True)

    def configure_optimizers(self):
        debug_print("configure_optimizers() START")

        embedding_params = list(self.model.tok_emb.parameters()) + list(self.model.pos_emb.parameters())
        head_params = list(self.model.head.parameters())
        other_params = list(self.model.blocks.parameters()) + list(self.model.ln_f.parameters())

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
                return (current_step + 1) / warmup_steps if warmup_steps > 0 else 1.0
            elif current_step <= max_steps - warmdown_steps:
                return 1.0
            else:
                progress = (max_steps - current_step) / warmdown_steps
                return progress + (1 - progress) * self.final_lr_frac

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        debug_print("configure_optimizers() END")
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % 50 == 0:
            debug_print(f"on_train_batch_start() - batch_idx={batch_idx}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:
            debug_print(f"on_train_batch_end() - batch_idx={batch_idx}")


if __name__ == "__main__":
    debug_print("=" * 80)
    debug_print("SCRIPT START")
    debug_print("=" * 80)

    parser = argparse.ArgumentParser(description='Train TinyGPT model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--fast_dev_run', type=int, default=0, help='Run a quick test with N batches')
    args = parser.parse_args()

    debug_print(f"Args parsed: batch_size={args.batch_size}, max_steps={args.max_steps}, fast_dev_run={args.fast_dev_run}")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    debug_print(f"Logs directory created/verified: {logs_dir}")

    max_steps = args.max_steps
    val_check_interval = 250

    debug_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    debug_print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    n_layers = 20
    max_seq_len = 2048
    batch_size = args.batch_size
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)

    debug_print("Creating model...")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )
    debug_print(f"Model created: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")

    base_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0

    lr_scale = (dim / 768) ** -0.5
    print0(f"\n=== Optimizer Configuration ===")
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

    debug_print("Creating LLMModule...")
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
    debug_print("LLMModule created")

    debug_print("Creating DataModule...")
    data_module = LLMDataModule(
        train_dir='data/base_data',
        val_dir='data/base_data',
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_length=max_seq_len,
        num_workers=4,
        val_sequences=10
    )
    debug_print("DataModule created")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        accelerator = 'gpu'
        devices = num_gpus
        strategy = 'auto'
        print0(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        debug_print(f"GPU configuration: {num_gpus} devices, strategy={strategy}")
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
        strategy = 'auto'
        print0("Detected Apple MPS device")
    else:
        accelerator = 'cpu'
        devices = 1
        strategy = 'auto'
        print0("No GPU detected, using CPU")

    debug_print("Creating WandbLogger...")
    wandb_logger = WandbLogger(
        project="llm-training",
        name=f"test-debug",
        save_dir=str(logs_dir),
        log_model=True,
        config={
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
        }
    )
    debug_print("WandbLogger created")

    debug_print("Creating MemoryLoggingCallback...")
    memory_callback = MemoryLoggingCallback(log_every_n_steps=10)
    debug_print("MemoryLoggingCallback created")

    debug_print("Creating Trainer...")
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
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[memory_callback],
        fast_dev_run=args.fast_dev_run if args.fast_dev_run > 0 else False,
    )
    debug_print("Trainer created")

    debug_print("=" * 80)
    debug_print("CALLING trainer.fit()...")
    debug_print("=" * 80)

    trainer.fit(llm_module, data_module)

    debug_print("=" * 80)
    debug_print("trainer.fit() COMPLETED")
    debug_print("SCRIPT END")
    debug_print("=" * 80)
