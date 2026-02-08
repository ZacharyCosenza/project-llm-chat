import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
from core.validation import run_world_knowledge_validation, run_sentence_completion
import math

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
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
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
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device(f'cuda:{local_rank}')
        )

    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    if world_size > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt
    return tensor


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, checkpoint_dir, rank):
    """Save training checkpoint. Only rank 0 saves to avoid conflicts."""
    if rank != 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_model = model.module if isinstance(model, DDP) else model

    checkpoint = {
        'global_step': global_step,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }

    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    print0(f"Saved checkpoint to {latest_path} (step {global_step})")


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        return latest_path

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None

    def get_step(path):
        try:
            return int(path.stem.split('_')[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and return the global step."""
    print0(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    base_model = model.module if isinstance(model, DDP) else model
    base_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    global_step = checkpoint['global_step']
    print0(f"Resumed from step {global_step}")

    return global_step

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
                global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
                use_wandb, pbar, val_check_interval, val_loader, tokenizer, limit_val_batches,
                warmup_steps, checkpoint_dir=None):
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
            scheduler.step()

            global_step += 1

            if global_step % log_every_n_steps == 0:
                loss_reduced = reduce_tensor(loss * accumulate_grad_batches, world_size)

                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    if global_step < warmup_steps:
                        phase = 0  # warmup
                    else:
                        phase = 1  # cosine decay

                    log_dict = {
                        'train/loss': loss_reduced.item(),
                        'lr': current_lr,
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

                    pbar.set_postfix({'loss': f"{loss_reduced.item():.4f}", 'lr': f"{current_lr:.2e}"})

            pbar.update(1)

            if global_step % val_check_interval == 0:
                validate(model, val_loader, device, rank, world_size, global_step,
                        tokenizer, use_wandb, limit_val_batches)
                model.train()

                if checkpoint_dir:
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, checkpoint_dir, rank)
                    if world_size > 1:
                        dist.barrier()

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

        if torch.cuda.is_available():
            log_dict.update({
                'memory/val_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory/val_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'memory/val_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            })

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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    return avg_loss_reduced

def create_optimizer_and_scheduler(model, peak_lr, min_lr, weight_decay, warmup_steps, max_steps):
    """Create AdamW optimizer with proper weight decay exclusion and cosine schedule with warmup."""
    # Separate parameters: no weight decay for bias and LayerNorm
    no_decay = ["bias", "ln", "layernorm", "layer_norm"]

    decay_params = []
    no_decay_params = []

    base_model = model.module if isinstance(model, DDP) else model

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Cosine schedule with linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return optimizer, scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TinyGPT model with GPT-2 standard optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--fast_dev_run', type=int, default=0, help='Run a quick test with N batches')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file or run directory to resume from')
    parser.add_argument('--peak_lr', type=float, default=6e-4, help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate (10x below peak)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.01, help='Warmup ratio (1-2% of total steps)')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    max_steps = args.max_steps
    val_check_interval = 50

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    n_layers = 20
    max_seq_len = 2048
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

    # GPT-2 standard optimizer config
    peak_lr = args.peak_lr
    min_lr = args.min_lr
    weight_decay = args.weight_decay
    warmup_steps = int(args.warmup_ratio * max_steps)

    print0(f"\n=== GPT-2 Standard Optimizer Configuration ===")
    print0(f"Peak LR: {peak_lr}")
    print0(f"Min LR: {min_lr}")
    print0(f"Weight decay: {weight_decay}")
    print0(f"Betas: (0.9, 0.95)")
    print0(f"Eps: 1e-8")
    print0(f"Warmup steps: {warmup_steps} ({args.warmup_ratio*100:.1f}% of {max_steps})")
    print0(f"Cosine decay steps: {max_steps - warmup_steps}")

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

    effective_batch_size = batch_size * world_size
    print0(f"\n=== Training Configuration ===")
    print0(f"World size: {world_size}")
    print0(f"Batch size per GPU: {batch_size}")
    print0(f"Effective batch size: {effective_batch_size}")
    print0(f"Tokens per step: {effective_batch_size * max_seq_len:,}")

    print0('adding model to device')
    model = model.to(device)
    print0('model added to device')

    if world_size > 1:
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed, about to broadcast test")

        test_tensor = torch.ones(1, device=device) * rank
        dist.all_reduce(test_tensor)
        print(f"[Rank {rank}] all_reduce result: {test_tensor.item()}")

        print(f"[Rank {rank}] About to wrap DDP")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] DDP complete")

    # Create optimizer with proper weight decay exclusion
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, peak_lr, min_lr, weight_decay, warmup_steps, max_steps
    )

    # Count parameters in each group
    base_model = model.module if isinstance(model, DDP) else model
    decay_count = sum(p.numel() for n, p in base_model.named_parameters()
                      if not any(nd in n.lower() for nd in ["bias", "ln", "layernorm", "layer_norm"]))
    no_decay_count = sum(p.numel() for n, p in base_model.named_parameters()
                         if any(nd in n.lower() for nd in ["bias", "ln", "layernorm", "layer_norm"]))
    print0(f"Parameters with weight decay: {decay_count:,}")
    print0(f"Parameters without weight decay: {no_decay_count:,}")

    scaler = torch.amp.GradScaler('cuda', enabled=(accelerator == 'gpu'))

    train_loader, val_loader = create_dataloaders(
        'data/base_data',
        'data/base_data',
        tokenizer,
        batch_size,
        max_seq_len,
        num_workers=0,
        val_sequences=1000,
        rank=rank,
        world_size=world_size
    )
    print0('loaders created')

    use_wandb = rank == 0
    wandb_run_id = None
    run_name = f"{world_size}gpu_bs{effective_batch_size}_gpt2opt"

    if use_wandb:
        print('Initializing wandb with GPT-2 standard config')
        wandb_run_id = wandb.util.generate_id()
        run_dir = logs_dir / wandb_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project="llm-training",
            name=run_name,
            id=wandb_run_id,
            dir=str(run_dir),
            resume="allow",
            config={
                "optimizer": "GPT-2 Standard",
                "vocab_size": tokenizer.vocab_size,
                "dim": dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "max_seq_len": max_seq_len,
                "batch_size_per_gpu": batch_size,
                "num_gpus": world_size,
                "effective_batch_size": effective_batch_size,
                "tokens_per_step": effective_batch_size * max_seq_len,
                "peak_lr": peak_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
                "betas": "(0.9, 0.95)",
                "eps": 1e-8,
                "warmup_steps": warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "max_steps": max_steps,
                "accelerator": accelerator,
            }
        )

    # Broadcast wandb run ID to all ranks for consistent checkpoint directory
    if world_size > 1:
        if rank == 0:
            run_id_tensor = torch.zeros(64, dtype=torch.uint8, device=device)
            if wandb_run_id:
                run_id_bytes = wandb_run_id.encode('utf-8')[:64]
                run_id_tensor[:len(run_id_bytes)] = torch.tensor(list(run_id_bytes), dtype=torch.uint8)
        else:
            run_id_tensor = torch.zeros(64, dtype=torch.uint8, device=device)
        dist.broadcast(run_id_tensor, src=0)
        wandb_run_id = bytes(run_id_tensor.cpu().tolist()).decode('utf-8').rstrip('\x00')

    if wandb_run_id:
        checkpoint_dir = logs_dir / wandb_run_id / "checkpoints"
    else:
        checkpoint_dir = logs_dir / "no_wandb" / "checkpoints"

    accumulate_grad_batches = max(1, 4 // world_size)
    gradient_clip_val = 1.0
    log_every_n_steps = 1
    limit_val_batches = batch_size

    print0(f"Gradient accumulation steps: {accumulate_grad_batches}")
    print0(f"Gradient clip: {gradient_clip_val}")
    print0(f"Checkpoints will be saved to: {checkpoint_dir}")

    global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            ckpt_subdir = resume_path / "checkpoints"
            if ckpt_subdir.exists():
                checkpoint_path = find_latest_checkpoint(ckpt_subdir)
            else:
                checkpoint_path = find_latest_checkpoint(resume_path)
        else:
            checkpoint_path = resume_path

        if checkpoint_path and checkpoint_path.exists():
            global_step = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device)
        else:
            print0(f"No checkpoint found at {args.resume}, starting from scratch")

    pbar = tqdm(total=max_steps, initial=global_step, disable=(rank != 0), desc="Training")

    if args.fast_dev_run > 0:
        max_steps = args.fast_dev_run

    while global_step < max_steps:
        print0('starting training!')
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
            global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
            use_wandb, pbar, val_check_interval, val_loader, tokenizer, limit_val_batches,
            warmup_steps, checkpoint_dir=checkpoint_dir
        )

    pbar.close()
    print0(f"Training complete at step {global_step}")

    if use_wandb:
        wandb.finish()

    cleanup_distributed()
