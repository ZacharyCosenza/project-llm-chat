"""Minimal multi-GPU Lightning test with fake data"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(32, 10)
    
    def training_step(self, batch, idx):
        x, y = batch
        loss = self.net(x).sum()
        print(f'[Rank {self.global_rank}] Step {idx}, loss={loss.item():.4f}')
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == '__main__':
    # Fake data
    x = torch.randn(1000, 32)
    y = torch.randint(0, 10, (1000,))
    dl = DataLoader(TensorDataset(x, y), batch_size=32)
    
    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy='ddp' if num_gpus > 1 else 'auto',
        max_steps=5,
        enable_progress_bar=False,
        logger=False,
    )
    
    trainer.fit(SimpleModel(), dl)
    print('Done!')