import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer

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
    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyGPT(vocab_size, dim, n_layers, n_heads)
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # [B, T-1, vocab]
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.hparams.vocab_size), y.reshape(-1))
        
        self.log('test_loss', loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])[:self.max_len * 5]
        
        enc = self.tokenizer(text, max_length=self.max_len, 
                           padding='max_length', truncation=True,
                           return_tensors='pt')
        
        tokens = enc['input_ids'].squeeze(0)
        x = tokens[:-1]  # input
        y = tokens[1:]   # target
        return x, y

def train_gpt(
    train_df,
    val_df=None,
    test_df=None,
    dim=64,
    max_seq_len=128,
    n_layers=2,
    n_heads=2,
    lr=1e-3,
    batch_size=32,
    epochs=3,
    devices=None,
    smoke_test=False
):
    """
    Train TinyGPT with automatic multi-GPU or CPU support.
    
    Args:
        train_df: DataFrame with 'text' column
        val_df: Optional validation DataFrame
        test_df: Optional test DataFrame
        devices: Number of GPUs (None = auto-detect)
        smoke_test: If True, runs 1 batch only
    """
    torch.set_float32_matmul_precision('medium')
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_ds = TextDataset(train_df, tokenizer, max_seq_len)
    val_ds = TextDataset(val_df, tokenizer, max_seq_len) if val_df is not None else None
    test_ds = TextDataset(test_df, tokenizer, max_seq_len) if test_df is not None else None
    
    # Create dataloaders
    num_workers = 0 if not torch.cuda.is_available() else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available()) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available()) if test_ds else None
    
    # Initialize model
    model = LightningGPT(vocab_size=tokenizer.vocab_size, dim=dim, 
                        n_layers=n_layers, n_heads=n_heads, lr=lr)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=devices,
        accelerator=accelerator,
        strategy="ddp" if devices > 1 else "auto",
        enable_progress_bar=True,
        fast_dev_run=smoke_test
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    if test_loader:
        trainer.test(model, test_loader)
    
    return model, trainer

if __name__ == "__main__":

    SMOKE = True
    
    if SMOKE:
        train_df = pd.DataFrame({'text': ['Hello world'] * 100})
        val_df = pd.DataFrame({'text': ['Test text'] * 20})
    
    # Train on all available GPUs (or CPU)
    model, trainer = train_gpt(
        train_df,
        val_df,
        dim=64,
        max_seq_len=128,
        n_layers=2,
        batch_size=16,
        epochs=3,
        devices=None,  # auto-detect
        smoke_test=SMOKE
    )
    
    print("Training complete!")