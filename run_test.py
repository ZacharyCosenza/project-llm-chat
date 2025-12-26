import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

class SimpleTokenDataset(IterableDataset):
    """Simple iterable dataset that yields batches directly."""
    def __init__(self, tokenizer, batch_size, max_seq_len, split="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Dummy data - replace with your actual data loading
        self.texts = ["This is example text."] * 1000
    
    def __iter__(self):
        # Get DDP info
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Each rank processes different data
        for i in range(rank, len(self.texts), world_size):
            text = self.texts[i]
            tokens = self.tokenizer.encode(text, max_length=self.max_seq_len, truncation=True)
            
            # Pad to max_seq_len
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
            
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            
            yield x.unsqueeze(0), y.unsqueeze(0)  # Add batch dim


class SimpleGPT(pl.LightningModule):
    def __init__(self, vocab_size, dim=128, tokenizer=None, batch_size=4, max_seq_len=128):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.head = torch.nn.Linear(dim, vocab_size)
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
    
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.head(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.squeeze(1), y.squeeze(1)  # Remove extra batch dim
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.squeeze(1), y.squeeze(1)
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        dataset = SimpleTokenDataset(self.tokenizer, self.batch_size, self.max_seq_len, split="train")
        return DataLoader(dataset, batch_size=None, num_workers=0)
    
    def val_dataloader(self):
        dataset = SimpleTokenDataset(self.tokenizer, self.batch_size, self.max_seq_len, split="val")
        return DataLoader(dataset, batch_size=None, num_workers=0)


def train():
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = SimpleGPT(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        tokenizer=tokenizer,
        batch_size=4,
        max_seq_len=128
    )
    
    # Trainer with DDP
    trainer = pl.Trainer(
        max_steps=100,
        accelerator="cpu",
        strategy="ddp",
        precision="bf16-mixed",
        val_check_interval=50,
        limit_val_batches=5
    )
    
    trainer.fit(model)


if __name__ == "__main__":
    train()