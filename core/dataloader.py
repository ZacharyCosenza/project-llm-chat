from collections import deque
import torch
import os
import pyarrow.parquet as pq

from core.dataset import list_parquet_files
from core.utils import get_dist_info

def tokenizing_dataloader(tokenizer, B, T, split="train", device=None, resume_state=None, stream=True):
    assert split in ["train", "val"]
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    token_buffer = deque()
    
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
            
            # After first pass
            if not stream:
                return
            
            epoch += 1
            pq_idx = 0  # restart from beginning for multi-epoch
    
    batches = doc_batches()
    
    # Main loop: accumulate tokens and yield batches
    try:
        while True:
            # Fill buffer
            while len(token_buffer) < needed_tokens:
                texts, (pq_idx, rg_idx) = next(batches)
                for text in texts:
                    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=2048, truncation=True)
                    token_buffer.extend(tokens)
            
            # Pop exact tokens needed
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            
            # Create batch (only use pinned memory on CUDA)
            use_pin = device == "cuda" and torch.cuda.is_available()
            scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_pin)
            
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