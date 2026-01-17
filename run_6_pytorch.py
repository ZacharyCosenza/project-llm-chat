import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
from core.utils import print0
from core.models import TinyGPT
import argparse
from transformers import AutoTokenizer
from collections import deque
import random
from itertools import cycle
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import os
import wandb
from tqdm import tqdm

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

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

def create_dataloaders(train_dir, val_dir, tokenizer, batch_size, seq_length,
                       num_workers, val_sequences, rank, world_size):
    train_dataset = StreamingParquetDataset(
        train_dir, tokenizer, seq_length,
        rank=rank,
        world_size=world_size,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    per_gpu_sequences = val_sequences // world_size
    val_dataset = StreamingParquetDataset(
        val_dir, tokenizer, seq_length,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        max_sequences=per_gpu_sequences
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)  # Set device BEFORE init_process_group
        dist.init_process_group(
            backend='gloo',
            device_id=torch.device(f'cuda:{local_rank}')  # Explicit device
        )

    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_lr_schedule(optimizer, current_step, max_steps, warmup_ratio, warmdown_ratio, final_lr_frac):
    warmup_steps = int(warmup_ratio * max_steps)
    warmdown_steps = int(warmdown_ratio * max_steps)

    if current_step < warmup_steps:
        lr_mult = (current_step + 1) / warmup_steps if warmup_steps > 0 else 1.0
    elif current_step <= max_steps - warmdown_steps:
        lr_mult = 1.0
    else:
        progress = (max_steps - current_step) / warmdown_steps
        lr_mult = progress + (1 - progress) * final_lr_frac

    return lr_mult

def update_lr(optimizer, lr_mult, base_lrs):
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group['lr'] = base_lr * lr_mult

def reduce_tensor(tensor, world_size):
    if world_size > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt
    return tensor

def train_epoch(model, train_loader, optimizer, scaler, device, rank, world_size,
                global_step, max_steps, warmup_ratio, warmdown_ratio, final_lr_frac,
                base_lrs, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
                use_wandb, pbar, val_check_interval, val_loader, tokenizer, limit_val_batches):
    model.train()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / accumulate_grad_batches

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr_mult = get_lr_schedule(optimizer, global_step, max_steps, warmup_ratio, warmdown_ratio, final_lr_frac)
            update_lr(optimizer, lr_mult, base_lrs)

            global_step += 1

            if global_step % log_every_n_steps == 0:
                loss_reduced = reduce_tensor(loss * accumulate_grad_batches, world_size)

                if rank == 0:
                    warmup_steps = int(warmup_ratio * max_steps)
                    warmdown_start = max_steps - int(warmdown_ratio * max_steps)
                    if global_step < warmup_steps:
                        phase = 0
                    elif global_step < warmdown_start:
                        phase = 1
                    else:
                        phase = 2

                    log_dict = {
                        'train/loss': loss_reduced.item(),
                        'lr/head': optimizer.param_groups[0]['lr'],
                        'lr/embeddings': optimizer.param_groups[1]['lr'],
                        'lr/other': optimizer.param_groups[2]['lr'],
                        'opt/phase': phase,
                        'opt/step_progress': global_step / max_steps,
                        'global_step': global_step
                    }

                    if torch.cuda.is_available():
                        log_dict.update({
                            'memory/allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                            'memory/reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                            'memory/max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
                        })

                    if use_wandb:
                        wandb.log(log_dict)

                    pbar.set_postfix({'loss': f"{loss_reduced.item():.4f}", 'lr': f"{optimizer.param_groups[2]['lr']:.2e}"})

            pbar.update(1)

            if global_step % val_check_interval == 0:
                validate(model, val_loader, device, rank, world_size, global_step,
                        tokenizer, use_wandb, limit_val_batches)
                model.train()  # Set back to training mode after validation

            if global_step >= max_steps:
                break

    return global_step

def validate(model, val_loader, device, rank, world_size, global_step, tokenizer,
             use_wandb, limit_val_batches):
    model.eval()
    total_loss = 0.0
    total_ppl = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= limit_val_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            ppl = torch.exp(loss)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()

            total_loss += loss.item()
            total_ppl += ppl.item()
            total_acc += acc.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ppl = total_ppl / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0

    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_ppl_tensor = torch.tensor(avg_ppl, device=device)
    avg_acc_tensor = torch.tensor(avg_acc, device=device)

    avg_loss_reduced = reduce_tensor(avg_loss_tensor, world_size).item()
    avg_ppl_reduced = reduce_tensor(avg_ppl_tensor, world_size).item()
    avg_acc_reduced = reduce_tensor(avg_acc_tensor, world_size).item()

    if rank == 0:
        log_dict = {
            'val/loss': avg_loss_reduced,
            'val/perplexity': avg_ppl_reduced,
            'val/accuracy': avg_acc_reduced,
            'global_step': global_step
        }

        from core.validation import run_world_knowledge_validation, run_sentence_completion

        base_model = model.module if isinstance(model, DDP) else model

        world_knowledge_results = run_world_knowledge_validation(
            base_model,
            tokenizer,
            device=device
        )

        if 'metrics' in world_knowledge_results:
            for k, v in world_knowledge_results['metrics'].items():
                log_dict[f"val/{k}"] = v

        sentence_completions = run_sentence_completion(
            base_model,
            tokenizer,
            device=device
        )

        if use_wandb:
            wandb.log(log_dict)

            if sentence_completions:
                sentence_completion_data = [
                    [global_step, item['prompt'], item['completion']]
                    for item in sentence_completions
                ]
                wandb.log({
                    "sentence_completions": wandb.Table(
                        columns=["step", "prompt", "completion"],
                        data=sentence_completion_data
                    )
                })

        print0(f"\nValidation - Loss: {avg_loss_reduced:.4f}, PPL: {avg_ppl_reduced:.4f}, Acc: {avg_acc_reduced:.4f}")

    return avg_loss_reduced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TinyGPT model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--fast_dev_run', type=int, default=0, help='Run a quick test with N batches')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    max_steps = args.max_steps
    val_check_interval = 250

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    n_layers = 20
    max_seq_len = 128
    batch_size = args.batch_size
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )

    base_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0

    lr_scale = (dim / 768) ** -0.5
    print0(f"\n=== Optimizer Configuration ===")
    print0(f"Base LR: {base_lr}")
    print0(f"LR scale factor: {lr_scale:.4f}")
    print0(f"Head LR: {0.004 * lr_scale * base_lr:.6f}")
    print0(f"Embedding LR: {0.2 * lr_scale * base_lr:.6f}")
    print0(f"Other LR: {0.02 * base_lr:.6f}")
    print0(f"Weight decay: {weight_decay}")
    print0(f"Warmup ratio: {warmup_ratio}")
    print0(f"Warmdown ratio: {warmdown_ratio}")

    warmup_steps = int(warmup_ratio * max_steps)
    warmdown_steps = int(warmdown_ratio * max_steps)
    constant_steps = max_steps - warmup_steps - warmdown_steps
    print0(f"Warmup steps: {warmup_steps}")
    print0(f"Constant steps: {constant_steps}")
    print0(f"Warmdown steps: {warmdown_steps}")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        accelerator = 'gpu'
        num_gpus = torch.cuda.device_count()
        if rank == 0:
            print0(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        device = torch.device('cpu')
        accelerator = 'cpu'
        if rank == 0:
            print0("No GPU detected, using CPU")

    print0('adding model to device')
    model = model.to(device)
    print0('model added to device')
    print('world size ', world_size)

    if world_size > 1:
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed, about to broadcast test")

        # Test if NCCL collectives work at all
        test_tensor = torch.ones(1, device=device) * rank
        dist.all_reduce(test_tensor)
        print(f"[Rank {rank}] all_reduce result: {test_tensor.item()}")

        print(f"[Rank {rank}] About to wrap DDP")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] DDP complete")

    embedding_params = list(model.module.tok_emb.parameters() if isinstance(model, DDP) else model.tok_emb.parameters()) + \
                       list(model.module.pos_emb.parameters() if isinstance(model, DDP) else model.pos_emb.parameters())
    head_params = list(model.module.head.parameters() if isinstance(model, DDP) else model.head.parameters())
    other_params = list(model.module.blocks.parameters() if isinstance(model, DDP) else model.blocks.parameters()) + \
                   list(model.module.ln_f.parameters() if isinstance(model, DDP) else model.ln_f.parameters())

    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': 0.004 * lr_scale * base_lr},
        {'params': embedding_params, 'lr': 0.2 * lr_scale * base_lr},
        {'params': other_params, 'lr': 0.02 * base_lr}
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    scaler = torch.amp.GradScaler('cuda', enabled=(accelerator == 'gpu'))

    train_loader, val_loader = create_dataloaders(
        'data/base_data',
        'data/base_data',
        tokenizer,
        batch_size,
        max_seq_len,
        num_workers=4,
        val_sequences=1000,
        rank=rank,
        world_size=world_size
    )
    print0('loaders created')

    use_wandb = rank == 0
    if use_wandb:
        print('im gonna use wandb for logging here!')
        wandb.init(
            project="llm-training",
            name="test",
            dir=str(logs_dir),
            config={
                "vocab_size": tokenizer.vocab_size,
                "dim": dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "max_seq_len": max_seq_len,
                "batch_size": batch_size,
                "seq_length": max_seq_len,
                "base_lr": base_lr,
                "lr_scale": lr_scale,
                "head_lr": 0.004 * lr_scale * base_lr,
                "embedding_lr": 0.2 * lr_scale * base_lr,
                "other_lr": 0.02 * base_lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "warmdown_ratio": warmdown_ratio,
                "final_lr_frac": final_lr_frac,
                "max_steps": max_steps,
                "accelerator": accelerator,
                "devices": world_size,
            }
        )

    accumulate_grad_batches = 4
    gradient_clip_val = 1.0
    log_every_n_steps = 10
    limit_val_batches = batch_size

    global_step = 0
    pbar = tqdm(total=max_steps, disable=(rank != 0), desc="Training")

    if args.fast_dev_run > 0:
        max_steps = args.fast_dev_run

    while global_step < max_steps:
        print0('starting training!')
        global_step = train_epoch(
            model, train_loader, optimizer, scaler, device, rank, world_size,
            global_step, max_steps, warmup_ratio, warmdown_ratio, final_lr_frac,
            base_lrs, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
            use_wandb, pbar, val_check_interval, val_loader, tokenizer, limit_val_batches
        )

    pbar.close()

    if use_wandb:
        wandb.finish()

    cleanup_distributed()
