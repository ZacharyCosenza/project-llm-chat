import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from core.optimizer import setup_optimizers, get_lr_multiplier
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
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, dataloader_state_dict = batch
        logits = self(x)  # [B, T-1, vocab]
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        # Calculate validation
        perplexity = torch.exp(loss)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_perplexity', perplexity, prog_bar=False, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

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
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        strategy="ddp" if devices > 1 else "auto",
        enable_progress_bar=True,
        val_check_interval=eval_every,
        limit_val_batches=2,
        fast_dev_run=smoke_test,
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