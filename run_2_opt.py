import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
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
    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=2,
                 base_lr=1e-3, weight_decay=0.01,
                 warmup_ratio=0.0, warmdown_ratio=0.2, final_lr_frac=0.0,
                 train_dataloader=None, build_val_loader=None):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataloader', 'build_val_loader'])
        
        self.model = TinyGPT(vocab_size, dim, n_layers, n_heads)
        self._train_dataloader = train_dataloader
        self.build_val_loader = build_val_loader
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

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
        return self._train_dataloader
    
    def val_dataloader(self):
        return self.build_val_loader()

def train_gpt(
    dim=64,
    max_seq_len=128,
    n_layers=2,
    n_heads=2,
    base_lr=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.0,
    warmdown_ratio=0.2,
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
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    torch.set_float32_matmul_precision('medium')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    
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
    
    # Dataloaders
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader = tokenizing_dataloader(tokenizer, batch_size, max_seq_len, split="train", device=device)
    build_val_loader = lambda: simple_dataloader(tokenizer, batch_size, max_seq_len, split="val", device=device)
    
    # Model
    model = LightningGPT(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmdown_ratio=warmdown_ratio,
        train_dataloader=train_loader,
        build_val_loader=build_val_loader
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        strategy="ddp" if devices > 1 else "auto",
        val_check_interval=eval_every,
        limit_val_batches=2,
        fast_dev_run=smoke_test,
        precision="bf16-mixed" if device == "cuda" else '32-true',
        gradient_clip_val=grad_clip if grad_clip > 0 else None,
        accumulate_grad_batches=grad_accum_steps,
        logger=logger,
        default_root_dir="logs"
    )
    
    trainer.fit(model)
    return model, trainer

if __name__ == "__main__":
    train_gpt(
        dim=768,
        max_seq_len=2048,
        n_layers=2,
        batch_size=32,
        max_steps=1000,
        smoke_test=False,
        wandb_name='test'
    )