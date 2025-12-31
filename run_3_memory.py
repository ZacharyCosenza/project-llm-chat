import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from core.dataloader import simple_dataloader, tokenizing_dataloader
from core.utils import print0, get_dist_info
from core.validation import run_sentence_completion, run_world_knowledge_validation
from core.models import TinyGPT
from core.memory_tracker import MemoryLoggingCallback, estimate_tensor_memory

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

        optimizer = self.optimizers()
        if optimizer is not None:
            param_groups = optimizer.param_groups
            self.log('lr/head', param_groups[0]['lr'], prog_bar=False, sync_dist=True)
            self.log('lr/embeddings', param_groups[1]['lr'], prog_bar=False, sync_dist=True)
            self.log('lr/other', param_groups[2]['lr'], prog_bar=False, sync_dist=True)

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

    def on_validation_epoch_end(self):

        # Some validations are just sanity checks/prints
        if self.trainer.global_rank == 0:
            run_sentence_completion(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=50,
                temperature=1.0,
                top_k=40
            )

            # Other validations are recorded in log
            validation_results = run_world_knowledge_validation(
                self.model,
                self.tokenizer,
                device=self.device,
                max_new_tokens=20,
                temperature=0.3,
                top_k=40
            )

            if self.logger is not None:
                self.logger.experiment.log({
                    "world_knowledge/accuracy": validation_results["accuracy"],
                    "world_knowledge/token_f1": validation_results["token_f1"],
                    "global_step": self.global_step
                })

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
        if self._train_loader is None:
            device = "cuda" if self.device.type == "cuda" else "cpu"
            self._train_loader = tokenizing_dataloader(
                self.tokenizer,
                self.batch_size,
                self.max_seq_len,
                split="train",
                device=device
            )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            device = "cuda" if self.device.type == "cuda" else "cpu"
            self._val_loader = simple_dataloader(
                self.tokenizer,
                self.batch_size,
                self.max_seq_len,
                split="val",
                device=device
            )
        return self._val_loader

def train_gpt(
    batch_size=32,
    max_seq_len=128,
    max_steps=1000,
    val_max_steps=2,
    smoke_test=False,
    wandb_name=None,
):
    # Model architecture parameters
    n_layers = 20
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)

    # Memory logging
    memory_log_every = 10
    memory_detailed_every = 100

    # Optimizer parameters
    base_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0

    # Training parameters
    eval_every = 250
    grad_clip = 1.0
    grad_accum_steps = 4
    devices = None

    # Logging parameters
    wandb_project = "llm-chat"
    wandb_enabled = True
    is_ddp, rank, local_rank, world_size = get_dist_info()

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

    print0("\n=== Loading Data ===")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print0(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print0(f"Batch size: {batch_size}, Max sequence length: {max_seq_len}")
    print0("Note: Dataloaders will be created lazily after DDP initialization")
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

    # Memory estimation
    if torch.cuda.is_available():
        print0("\n=== Memory Estimation ===")
        memory_estimate = estimate_tensor_memory(
            model.model,
            batch_size * grad_accum_steps,  # Effective batch size
            max_seq_len,
            tokenizer.vocab_size,
            dtype=torch.bfloat16  # Using bf16-mixed
        )
        print0(f"Parameters: {memory_estimate['parameters_mb']:.2f} MB")
        print0(f"Gradients: {memory_estimate['gradients_mb']:.2f} MB")
        print0(f"Optimizer State: {memory_estimate['optimizer_state_mb']:.2f} MB")
        print0(f"Activations (est): {memory_estimate['activations_mb']:.2f} MB")
        print0(f"Data (per batch): {memory_estimate['data_mb']:.2f} MB")
        print0(f"Logits (per batch): {memory_estimate['logits_mb']:.2f} MB")
        print0(f"Total Estimated: {memory_estimate['total_estimated_mb']:.2f} MB")

    print0("\n=== Training Settings ===")
    print0(f"Max steps: {max_steps}")
    print0(f"Eval every: {eval_every} steps")
    print0(f"Validation max steps: {val_max_steps}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Gradient clipping: {grad_clip if grad_clip > 0 else 'None'}")
    print0(f"Strategy: {'ddp' if devices > 1 else 'auto'}")
    print0(f"Precision: {'bf16-mixed' if accelerator == 'gpu' else '32-true'}")
    print0(f"Smoke test: {smoke_test}")
    print0(f"Memory logging: every {memory_log_every} steps")
    print0(f"Detailed memory logging: every {memory_detailed_every} steps")
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

    # Create memory logging callback
    memory_callback = MemoryLoggingCallback(
        log_every_n_steps=memory_log_every,
        detailed_every_n_steps=memory_detailed_every
    )

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
        default_root_dir="logs",
        callbacks=[memory_callback]
    )

    trainer.fit(model)

    print0("\n=== Training Complete ===")

    return model, trainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT model with memory tracking")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per rank")
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
        wandb_name=args.wandb_name,
    )
