import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer
import wandb

# custom imports
from core.dataloader import simple_dataloader, tokenizing_dataloader

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=n_heads, 
                dim_feedforward=dim*4,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
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

class LightningGPT(pl.LightningModule):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=2, lr=1e-3,
                 train_dataloader=None, build_val_loader=None):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataloader', 'val_dataloader'])
        self.model = TinyGPT(vocab_size, dim, n_layers, n_heads)
        self.lr = lr
        self._train_dataloader = train_dataloader
        self.build_val_loader = build_val_loader
        self.flops_per_token = self._calculate_flops_per_token(vocab_size, dim, n_layers, n_heads)
    
    def forward(self, x):
        return self.model(x)

    def _calculate_flops_per_token(self, vocab_size, dim, n_layers, n_heads):
        """
        Calculate approximate FLOPs per token for the transformer model.
        Based on standard transformer FLOP counting methods.
        """
        # Embedding: vocab_size * dim (negligible, but included for completeness)
        embedding_flops = 2 * dim  # lookup is ~2 ops per dim

        # Per-layer FLOPs (for each transformer block)
        # Attention: Q, K, V projections + attention scores + output projection
        # Q, K, V projections: 3 * (2 * dim * dim) per token
        qkv_flops = 3 * 2 * dim * dim
        # Attention scores: 2 * seq_len * dim (simplified, depends on sequence length)
        # We'll use dim as approximation since seq_len varies
        attn_flops = 2 * dim * dim
        # Output projection: 2 * dim * dim
        out_proj_flops = 2 * dim * dim

        # FFN: two linear layers with dim_feedforward = 4*dim
        # First linear: 2 * dim * (4*dim)
        # Second linear: 2 * (4*dim) * dim
        ffn_flops = 2 * dim * (4 * dim) + 2 * (4 * dim) * dim

        # LayerNorm (approximate): 2 * dim per norm, 2 norms per layer
        ln_flops = 2 * 2 * dim

        # Total per layer
        flops_per_layer = qkv_flops + attn_flops + out_proj_flops + ffn_flops + ln_flops

        # Final layer norm
        final_ln_flops = 2 * dim

        # Output head: 2 * dim * vocab_size
        head_flops = 2 * dim * vocab_size

        # Total FLOPs per token (forward pass)
        total_flops = embedding_flops + (n_layers * flops_per_layer) + final_ln_flops + head_flops

        return total_flops

    def training_step(self, batch, batch_idx):
        x, y, dataloader_state_dict = batch
        logits = self(x)  # [B, T-1, vocab]
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        # Calculate FLOPs for this batch
        # Forward pass FLOPs: batch_size * seq_len * flops_per_token
        # Backward pass is approximately 2x forward pass
        B, T = x.shape
        forward_flops = B * T * self.flops_per_token
        total_flops = 3 * forward_flops  # forward + backward (2x forward)

        # Convert to GFLOPs for easier reading
        gflops = total_flops / 1e9

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_gflops_per_batch', gflops, prog_bar=False, sync_dist=True)
        self.log('train_flops_per_token', self.flops_per_token, prog_bar=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_perplexity', perplexity, prog_bar=False, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        # val_loader only constructed when needed
        val_loader = self.build_val_loader()
        return val_loader

def train_gpt(
    dim=64,
    max_seq_len=128,
    n_layers=2,
    n_heads=2,
    lr=1e-3,
    batch_size=32,
    max_steps=1000,
    eval_every=250,
    grad_clip=1.0,
    grad_accum_steps=1,
    devices=None,
    smoke_test=False,
    wandb_project="llm-chat",
    wandb_name=None,
    wandb_enabled=True
):
    """
    Args:
        train_df: DataFrame with 'text' column
        val_df: Optional validation DataFrame
        test_df: Optional test DataFrame
        max_steps: Maximum number of training steps
        eval_every: Run validation every N steps
        devices: Number of GPUs (None = auto-detect)
        smoke_test: If True, runs 1 batch only
    """

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    torch.set_float32_matmul_precision('medium')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize WandB logger with unified logging directory
    # Disable logging when smoke_test is True
    logger = None
    if wandb_enabled and not smoke_test:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            log_model=True,
            save_dir="logs",  # Unified logging directory
            config={
                "dim": dim,
                "max_seq_len": max_seq_len,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "lr": lr,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "eval_every": eval_every,
                "grad_clip": grad_clip,
                "grad_accum_steps": grad_accum_steps,
                "devices": devices,
                "accelerator": accelerator,
            }
        )

    # Create custom dataloaders
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    resuming = False
    meta_data = {}
    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    train_loader = tokenizing_dataloader(tokenizer, batch_size, max_seq_len, split="train", device=device,
                                        resume_state=dataloader_resume_state_dict)
    build_val_loader = lambda: simple_dataloader(tokenizer, batch_size, max_seq_len, split="val", device=device)
    model = LightningGPT(vocab_size=tokenizer.vocab_size, dim=dim,
                        n_layers=n_layers, n_heads=n_heads, lr=lr,
                        train_dataloader=train_loader, build_val_loader=build_val_loader)

    trainer = pl.Trainer(
        max_steps=max_steps if not smoke_test else 100,
        devices=devices,
        accelerator=accelerator,
        strategy="ddp" if devices > 1 else "auto",
        enable_progress_bar=True,
        val_check_interval=eval_every if not smoke_test else 2,
        limit_val_batches=2 if smoke_test else None,
        fast_dev_run=smoke_test,  # Use PyTorch Lightning's default smoke test when enabled
        precision="bf16-mixed" if device == "cuda" else '32-true', # drop-in for the autocast
        gradient_clip_val=grad_clip if grad_clip > 0.0 else None,
        accumulate_grad_batches=grad_accum_steps,
        logger=logger,
        default_root_dir="logs"  # Unified directory for all Lightning outputs
    )

    trainer.fit(model)

if __name__ == "__main__":

    SMOKE = True
    WANDB_NAME = 'test'
    
    train_gpt(
        dim=64,
        max_seq_len=128,
        n_layers=2,
        batch_size=16,
        max_steps=1000,
        eval_every=250,
        grad_clip = 1.0, # gradient clipping value (0.0 = disabled)
        devices=None,
        smoke_test=SMOKE,
        wandb_name=WANDB_NAME
    )