import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models import TinyGPT
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from pathlib import Path
from collections import deque
import time
import os

class SimpleParquetDataset(IterableDataset):
    """Simple dataset for GPU testing"""
    def __init__(self, parquet_dir: str, tokenizer, seq_length: int = 512, max_sequences: int = 100):
        self.files = sorted(Path(parquet_dir).glob("*.parquet"))[:1]  # Use only first file for testing
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_sequences = max_sequences

    def __iter__(self):
        buffer = deque()
        sequences_yielded = 0

        for filepath in self.files:
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


def test_gpu():
    """Test GPU detection and run TinyGPT model for a few iterations"""

    print("="*60)
    print("GPU Test Function")
    print("="*60)

    os.chdir('/home/zaccosenza/code/project-llm-chat')

    # Detect CUDA GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nDetected {num_gpus} CUDA GPU(s):")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Multi-Processors: {props.multi_processor_count}")

        device = torch.device('cuda:0')
        current_device = torch.cuda.current_device()
        print(f"\nCurrent Device: cuda:{current_device}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        print("\nNo CUDA GPUs detected. Using CPU.")
        device = torch.device('cpu')
        print(f"Current Device: {device}")

    print(f"\nPyTorch Version: {torch.__version__}")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create small TinyGPT model for testing
    print("\nCreating TinyGPT model...")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=1,
        n_heads=4,
        max_seq_len=128
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataset and dataloader
    print("\nLoading dataset...")
    seq_length = 128  # Match model's max_seq_len
    dataset = SimpleParquetDataset(
        parquet_dir='data/base_data',
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_sequences=10
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop for a few iterations
    print("\nRunning training for 5 iterations...")
    print("-"*60)

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

        print(f"Iteration {i+1}/5:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        print()

    print("="*60)
    print("GPU test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_gpu()
