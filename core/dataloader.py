from collections import deque
import torch
from core.dataset import list_parquet_files
from core.utils import get_dist_info
import random
from itertools import cycle
from collections import deque
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from pathlib import Path
import pytorch_lightning as pl
from core.utils import print0

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
        print0(f"Rank {self.rank}, shard_id {shard_id}, total_shards {total_shards}, num_files {len(self.files)}")
        
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
    
### Depreciated dataloaders and stuff

def tokenizing_dataloader(tokenizer, B, T, split="train", device=None, resume_state=None, stream=True):
    assert split in ["train", "val"]

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    token_buffer = deque()

    # Pre-allocate pinned memory buffer for CUDA to avoid repeated allocations
    use_pin = device == "cuda"
    if use_pin:
        pinned_buffer = torch.zeros(needed_tokens, dtype=torch.long, pin_memory=True)
    else:
        pinned_buffer = None

    # Document batch generator
    def doc_batches():
        paths = list_parquet_files()
        paths = paths[:-1] if split == "train" else paths[-1:]

        pq_idx = resume_state.get("pq_idx", 0) if resume_state else 0
        resume_rg = resume_state.get("rg_idx") if resume_state else None

        epoch = 0
        while True:
            for pq_idx_iter in range(pq_idx, len(paths)):
                pf = pq.ParquetFile(paths[pq_idx_iter])

                # Resume logic: skip ahead to avoid repeats
                if resume_rg is not None and pq_idx_iter == pq_idx:
                    rg_idx = ((resume_rg // ddp_world_size) + 1) * ddp_world_size + ddp_rank
                    resume_rg = None
                else:
                    rg_idx = ddp_rank

                while rg_idx < pf.num_row_groups:
                    texts = pf.read_row_group(rg_idx).column('text').to_pylist()
                    yield texts, (pq_idx_iter, rg_idx)
                    rg_idx += ddp_world_size

                # Explicitly close the parquet file to avoid file handle leaks
                del pf

            # After first pass
            if not stream:
                return

            epoch += 1
            pq_idx = 0  # restart from beginning for multi-epoch

    batches = doc_batches()

    # Main loop: accumulate tokens and yield batches
    try:
        while True:
            # Fill buffer (limit buffer size to prevent unbounded growth)
            while len(token_buffer) < needed_tokens:
                texts, (pq_idx, rg_idx) = next(batches)
                for text in texts:
                    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=2048, truncation=True)
                    token_buffer.extend(tokens)
                    
                    # Limit buffer size to avoid excessive memory usage
                    if len(token_buffer) >= needed_tokens * 2:
                        break
                if len(token_buffer) >= needed_tokens * 2:
                    break

            # Pop exact tokens needed
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

            # Reuse pinned memory buffer instead of allocating new one each time
            if use_pin:
                pinned_buffer[:] = torch.tensor(tokens, dtype=torch.long)
                scratch = pinned_buffer
            else:
                scratch = torch.tensor(tokens, dtype=torch.long)

            x = scratch[:-1].view(B, T).to(device, non_blocking=use_pin)
            y = scratch[1:].view(B, T).to(device, non_blocking=use_pin)

            state = {"pq_idx": pq_idx, "rg_idx": rg_idx}
            yield x, y, state

    except StopIteration:
        # Only happens when stream=False and data exhausted
        return

def simple_dataloader(tokenizer, B, T, split="train", device=None, stream=True):
    """Wrapper that only yields (x, y) without state."""
    for x, y, _ in tokenizing_dataloader(tokenizer, B, T, split, device, stream=stream):
        yield x, y