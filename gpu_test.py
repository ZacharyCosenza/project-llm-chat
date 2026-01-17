import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from core.models import TinyGPT
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from pathlib import Path
from collections import deque
import time
import os

class SimpleParquetDataset(IterableDataset):
    """Simple dataset for GPU testing with multi-GPU support"""
    def __init__(self, parquet_dir: str, tokenizer, seq_length: int = 512, max_sequences: int = 100,
                 rank: int = 0, world_size: int = 1):
        self.files = sorted(Path(parquet_dir).glob("*.parquet"))[:1]  # Use only first file for testing
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_sequences = max_sequences
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # Get worker info for DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        # Calculate shard for this rank and worker
        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        buffer = deque()
        sequences_yielded = 0

        for file_idx, filepath in enumerate(self.files):
            # Skip files that don't belong to this shard
            if file_idx % total_shards != shard_id:
                continue

            pf = pq.ParquetFile(filepath)
            for rg_idx in range(min(2, pf.metadata.num_row_groups)):  # Only read 2 row groups
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
                        if sequences_yielded >= self.max_sequences:
                            return


def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, rank):
    """Print only from rank 0"""
    if rank == 0:
        print(msg)


def test_gpu():
    """Test GPU detection and run TinyGPT model for a few iterations"""

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    print_rank0("="*60, rank)
    print_rank0("GPU Test Function", rank)
    print_rank0("="*60, rank)

    # Detect CUDA GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if rank == 0:
            print(f"\nDetected {num_gpus} CUDA GPU(s):")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Multi-Processors: {props.multi_processor_count}")

            print(f"\nCUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

        device = torch.device(f'cuda:{local_rank}')
        print(f"[Rank {rank}] Using device: {device}")
    else:
        print_rank0("\nNo CUDA GPUs detected. Using CPU.", rank)
        device = torch.device('cpu')
        print_rank0(f"Current Device: {device}", rank)

    print_rank0(f"\nPyTorch Version: {torch.__version__}", rank)
    print_rank0(f"World Size: {world_size}", rank)
    print_rank0(f"Training on {world_size} GPU(s)\n", rank)

    # Initialize tokenizer
    print_rank0("\nInitializing tokenizer...", rank)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create small TinyGPT model for testing
    print_rank0("\nCreating TinyGPT model...", rank)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=1,
        n_heads=4,
        max_seq_len=128
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Model parameters: {num_params:,}", rank)

    # Wrap model in DDP if using multiple GPUs
    if world_size > 1:
        print(f"[Rank {rank}] Wrapping model in DistributedDataParallel...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] DDP wrapper complete")

    # Create dataset and dataloader
    print_rank0("\nLoading dataset...", rank)
    seq_length = 128  # Match model's max_seq_len
    dataset = SimpleParquetDataset(
        parquet_dir='data/base_data',
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_sequences=10,
        rank=rank,
        world_size=world_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        },
        num_workers=0  # Keep simple for testing
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop for a few iterations
    print_rank0("\nRunning training for 5 iterations...", rank)
    print_rank0("-"*60, rank)

    model.train()
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Only run 5 iterations
            break

        start_time = time.time()

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        elapsed = time.time() - start_time
        tokens_per_sec = (input_ids.numel() / elapsed)

        # Synchronize loss across all GPUs for accurate reporting
        if world_size > 1:
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        else:
            avg_loss = loss.item()

        # Each rank prints its own metrics
        print(f"[Rank {rank}] Iteration {i+1}/5:")
        print(f"  Loss (local): {loss.item():.4f} | Loss (avg): {avg_loss:.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")

        # Detailed GPU monitoring - each rank monitors its own GPU
        if torch.cuda.is_available():
            print(f"\n  [Rank {rank}] GPU {local_rank} Information ({torch.cuda.get_device_name(local_rank)}):")

            # Memory statistics for this GPU
            mem_allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
            mem_free = (torch.cuda.get_device_properties(local_rank).total_memory / 1024**3) - mem_allocated

            print(f"    Memory Allocated: {mem_allocated:.2f} GB")
            print(f"    Memory Reserved:  {mem_reserved:.2f} GB")
            print(f"    Memory Free:      {mem_free:.2f} GB")

            # Memory stats
            try:
                mem_stats = torch.cuda.memory_stats(local_rank)
                active_bytes = mem_stats.get('active_bytes.all.current', 0) / 1024**3
                inactive_bytes = mem_stats.get('inactive_split_bytes.all.current', 0) / 1024**3

                print(f"    Active Memory:    {active_bytes:.2f} GB")
                print(f"    Inactive Memory:  {inactive_bytes:.2f} GB")

                # Allocation stats
                alloc_count = mem_stats.get('allocation.all.current', 0)
                free_count = mem_stats.get('segment.all.current', 0)
                print(f"    Allocations:      {alloc_count}")
                print(f"    Segments:         {free_count}")
            except:
                pass

            # Temperature and utilization (if nvidia-smi is available)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                     '--format=csv,noheader,nounits', f'--id={local_rank}'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    gpu_util, mem_util, temp, power_draw, power_limit = result.stdout.strip().split(',')
                    print(f"    GPU Utilization:  {gpu_util.strip()}%")
                    print(f"    Mem Utilization:  {mem_util.strip()}%")
                    print(f"    Temperature:      {temp.strip()}°C")
                    print(f"    Power Draw:       {power_draw.strip()}W / {power_limit.strip()}W")
            except:
                pass
        print()

        # Synchronize all ranks before next iteration
        if world_size > 1:
            dist.barrier()

    print_rank0("="*60, rank)
    print_rank0("GPU test completed successfully!", rank)
    print_rank0("="*60, rank)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    test_gpu()
