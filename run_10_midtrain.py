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
from bisect import bisect
import random
import json
from itertools import cycle
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import os
import wandb
from tqdm import tqdm
from core.validation import run_world_knowledge_validation, run_sentence_completion, run_core_eval
import math

# ---- Mode-dependent training configs ----

TRAIN_CONFIGS = {
    'pretrain': {
        'peak_lr': 6e-4,
        'min_lr': 6e-5,
        'weight_decay': 0.1,
        'warmup_ratio': 0.01,
        'datasets': {'fineweb': 1.0},
        'loss_fn': 'cross_entropy',
    },
    'midtrain': {
        'peak_lr': 3e-4,
        'min_lr': 3e-5,
        'weight_decay': 0.1,
        'warmup_ratio': 0.005,
        'datasets': {
            'fineweb': 0.30,
            'smoltalk': 0.4844,
            'ultrachat_gen': 0.119,
            'ultrachat_sft': 0.0966,
        },
        'loss_fn': 'cross_entropy',
    },
}

# ---- Datasets ----

class StreamingParquetDataset(IterableDataset):
    """Streams FineWeb-Edu parquet files, packing text into fixed-length chunks."""
    def __init__(self, files, tokenizer, seq_length=2048,
                 rank=0, world_size=1, shuffle=False, max_sequences=None,
                 bos_token_id=None):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.max_sequences = max_sequences
        self.bos_token_id = bos_token_id

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

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
                        if self.bos_token_id is not None:
                            chunk = [self.bos_token_id] + chunk[:-1]
                        yield {
                            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                            'labels': torch.tensor(chunk[1:], dtype=torch.long)
                        }
                        sequences_yielded += 1
                        if self.max_sequences and sequences_yielded >= self.max_sequences:
                            return


class ConversationStreamingDataset(IterableDataset):
    """Streams conversation parquets, formatting each conversation with special tokens
    and padding to seq_length (no cross-conversation packing)."""
    def __init__(self, files, tokenizer, seq_length=2048,
                 rank=0, world_size=1, shuffle=False, max_sequences=None,
                 source_filter=None, pad_token_id=None):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.max_sequences = max_sequences
        self.source_filter = source_filter
        self.pad_token_id = pad_token_id

        # Cache special token IDs
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.role_ids = {
            'user': tokenizer.convert_tokens_to_ids('<|user|>'),
            'assistant': tokenizer.convert_tokens_to_ids('<|assistant|>'),
            'system': tokenizer.convert_tokens_to_ids('<|system|>'),
        }

    def _format_conversation(self, messages):
        """Convert messages list to token IDs with special tokens.
        Returns: [BOS, role, content..., role, content..., EOS]"""
        tokens = [self.bos_token_id]
        for msg in messages:
            role_id = self.role_ids.get(msg['role'])
            if role_id is not None:
                tokens.append(role_id)
            content_tokens = self.tokenizer.encode(msg['content'], add_special_tokens=False)
            tokens.extend(content_tokens)
        tokens.append(self.eos_token_id)
        return tokens

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)

        sequences_yielded = 0
        target_len = self.seq_length + 1  # +1 for input/label shift

        for file_idx in cycle(range(len(files))) if self.max_sequences else range(len(files)):
            if file_idx % total_shards != shard_id:
                continue

            filepath = files[file_idx]
            pf = pq.ParquetFile(filepath)

            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=['messages', 'source'])
                messages_col = table['messages'].to_pylist()
                source_col = table['source'].to_pylist()

                for messages, source in zip(messages_col, source_col):
                    # Filter by source if specified
                    if self.source_filter is not None and source != self.source_filter:
                        continue

                    # Skip empty conversations
                    if not messages:
                        continue

                    tokens = self._format_conversation(messages)

                    # Truncate or pad to target_len
                    if len(tokens) > target_len:
                        tokens = tokens[:target_len]
                    else:
                        tokens = tokens + [self.pad_token_id] * (target_len - len(tokens))

                    yield {
                        'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                        'labels': torch.tensor(tokens[1:], dtype=torch.long)
                    }
                    sequences_yielded += 1
                    if self.max_sequences and sequences_yielded >= self.max_sequences:
                        return


class MixedStreamingDataset(IterableDataset):
    """Mixes multiple datasets by sampling with specified probabilities."""
    def __init__(self, datasets_with_names, probabilities, seed=42, rank=0):
        assert len(datasets_with_names) == len(probabilities)
        self.datasets_with_names = datasets_with_names
        self.probabilities = probabilities
        self.seed = seed
        self.rank = rank

        # Precompute cumulative probabilities
        total = sum(probabilities)
        self.cumulative = []
        cum = 0.0
        for p in probabilities:
            cum += p / total
            self.cumulative.append(cum)

    def __iter__(self):
        rng = random.Random(self.seed + self.rank)
        iters = [iter(ds) for _, ds in self.datasets_with_names]

        while True:
            r = rng.random()
            idx = bisect(self.cumulative, r)
            idx = min(idx, len(iters) - 1)
            yield next(iters[idx])


# ---- Collate ----

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


# ---- Dataloaders ----

def create_dataloaders(mode_train, tokenizer, batch_size, seq_length,
                       num_workers, val_sequences, rank, world_size,
                       bos_token_id=None, pad_token_id=None):
    """Create train loader and per-dataset val loaders based on mode_train.

    Returns:
        train_loader: DataLoader
        val_loaders: dict of {name: DataLoader}
    """
    config = TRAIN_CONFIGS[mode_train]
    dataset_config = config['datasets']

    # FineWeb files (used in all modes)
    fineweb_dir = Path('data/base_data')
    fineweb_files = sorted(fineweb_dir.glob("*.parquet"))
    fineweb_val_files = fineweb_files[:1]
    fineweb_train_files = fineweb_files[1:]

    val_loaders = {}

    if mode_train == 'pretrain':
        print0(f"Data: {len(fineweb_train_files)} FineWeb train shards, {len(fineweb_val_files)} val shards")

        train_dataset = StreamingParquetDataset(
            fineweb_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, bos_token_id=bos_token_id
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Single val loader
        fineweb_val = StreamingParquetDataset(
            fineweb_val_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=False, max_sequences=val_sequences // world_size, bos_token_id=bos_token_id
        )
        val_loaders['fineweb'] = DataLoader(
            fineweb_val, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True
        )

    elif mode_train == 'midtrain':
        conv_dir = Path('data/conversation_data')
        conv_files = sorted(conv_dir.glob("*.parquet"))
        conv_val_files = conv_files[:1]
        conv_train_files = conv_files[1:]

        print0(f"Data: {len(fineweb_train_files)} FineWeb + {len(conv_train_files)} conversation train shards")
        print0(f"  Mix: {' | '.join(f'{k}={v:.1%}' for k, v in dataset_config.items())}")

        # Training: 4 sub-datasets mixed
        fineweb_train = StreamingParquetDataset(
            fineweb_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, bos_token_id=bos_token_id
        )
        smoltalk_train = ConversationStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, source_filter='smoltalk', pad_token_id=pad_token_id
        )
        ultrachat_gen_train = ConversationStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, source_filter='ultrachat_gen', pad_token_id=pad_token_id
        )
        ultrachat_sft_train = ConversationStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, source_filter='ultrachat_sft', pad_token_id=pad_token_id
        )

        datasets_with_names = [
            ('fineweb', fineweb_train),
            ('smoltalk', smoltalk_train),
            ('ultrachat_gen', ultrachat_gen_train),
            ('ultrachat_sft', ultrachat_sft_train),
        ]
        probabilities = [dataset_config[name] for name, _ in datasets_with_names]

        train_dataset = MixedStreamingDataset(
            datasets_with_names, probabilities, seed=42, rank=rank
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Per-dataset val loaders
        num_val_datasets = len(dataset_config)
        val_seqs_per_dataset = val_sequences // world_size // num_val_datasets

        # FineWeb val (packed)
        fineweb_val = StreamingParquetDataset(
            fineweb_val_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=False, max_sequences=val_seqs_per_dataset, bos_token_id=bos_token_id
        )
        val_loaders['fineweb'] = DataLoader(
            fineweb_val, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True
        )

        # Conversation val loaders (padded)
        for source in ['smoltalk', 'ultrachat_gen', 'ultrachat_sft']:
            conv_val = ConversationStreamingDataset(
                conv_val_files, tokenizer, seq_length, rank=rank, world_size=world_size,
                shuffle=False, max_sequences=val_seqs_per_dataset,
                source_filter=source, pad_token_id=pad_token_id
            )
            val_loaders[source] = DataLoader(
                conv_val, batch_size=batch_size, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=True
            )

    return train_loader, val_loaders


# ---- Distributed helpers ----

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))

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


# ---- Checkpointing ----

def save_checkpoint(model, global_step, checkpoint_dir, rank):
    if rank != 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    old_checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

    base_model = model.module if isinstance(model, DDP) else model
    path = checkpoint_dir / f"checkpoint_{global_step}.pt"
    torch.save({'model_state_dict': base_model.state_dict()}, path)
    print0(f"Saved checkpoint to {path} (step {global_step})")

    for old_ckpt in old_checkpoints:
        if old_ckpt != path:
            old_ckpt.unlink()

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    def get_step(path):
        try:
            return int(path.stem.split('_')[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]

def load_checkpoint(checkpoint_path, model, scheduler, device):
    print0(f"Loading checkpoint from {checkpoint_path}")

    global_step = int(Path(checkpoint_path).stem.split('_')[-1])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    base_model = model.module if isinstance(model, DDP) else model
    base_model.load_state_dict(checkpoint['model_state_dict'])

    for _ in range(global_step):
        scheduler.step()

    print0(f"Resumed from step {global_step}")
    return global_step

def resize_embeddings(model, new_vocab_size):
    base_model = model.module if isinstance(model, DDP) else model
    old_vocab = base_model.tok_emb.num_embeddings
    if old_vocab >= new_vocab_size:
        return

    dim = base_model.tok_emb.embedding_dim
    old_tok = base_model.tok_emb.weight.data.clone()
    old_head = base_model.head.weight.data.clone()

    base_model.tok_emb = nn.Embedding(new_vocab_size, dim).to(old_tok.device)
    base_model.head = nn.Linear(dim, new_vocab_size, bias=False).to(old_head.device)

    base_model.tok_emb.weight.data[:old_vocab] = old_tok
    base_model.head.weight.data[:old_vocab] = old_head

    print0(f"Resized embeddings: {old_vocab} -> {new_vocab_size} (+{new_vocab_size - old_vocab} tokens)")


# ---- Loss function ----

def compute_loss(logits, labels, mode_train, pad_token_id=None):
    """Mode-dependent loss. Same for pretrain/midtrain (cross_entropy), extensible for SFT/RL."""
    ignore_index = pad_token_id if pad_token_id is not None else -100
    if TRAIN_CONFIGS[mode_train]['loss_fn'] == 'cross_entropy':
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               ignore_index=ignore_index)
    raise ValueError(f"Unknown loss_fn: {TRAIN_CONFIGS[mode_train]['loss_fn']}")


# ---- Training ----

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
                global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
                use_wandb, pbar, val_check_interval, val_loaders, tokenizer, limit_val_batches,
                warmup_steps, mode_train, checkpoint_dir=None, core_examples=27, pad_token_id=None):
    model.train()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            logits = model(input_ids)
            loss = compute_loss(logits, labels, mode_train, pad_token_id)
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
                    phase = 0 if global_step < warmup_steps else 1

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
                validate(model, val_loaders, device, rank, world_size, global_step,
                        tokenizer, use_wandb, limit_val_batches, mode_train,
                        core_examples, pad_token_id)
                model.train()

                if checkpoint_dir:
                    save_checkpoint(model, global_step, checkpoint_dir, rank)
                    if world_size > 1:
                        dist.barrier()

            if global_step >= max_steps:
                break

    return global_step


# ---- Validation ----

def validate(model, val_loaders, device, rank, world_size, global_step, tokenizer,
             use_wandb, limit_val_batches, mode_train, core_examples=27, pad_token_id=None):
    """Run per-dataset validation and CORE eval."""
    model.eval()
    log_dict = {'global_step': global_step}

    # Per-dataset loss and perplexity
    for name, loader in val_loaders.items():
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= limit_val_batches:
                    break

                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    logits = model(input_ids)
                    loss = compute_loss(logits, labels, mode_train, pad_token_id)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        avg_loss_reduced = reduce_tensor(torch.tensor(avg_loss, device=device), world_size).item()
        avg_ppl_reduced = reduce_tensor(torch.tensor(avg_ppl, device=device), world_size).item()

        log_dict[f'val/{name}/loss'] = avg_loss_reduced
        log_dict[f'val/{name}/perplexity'] = avg_ppl_reduced

        print0(f"  val/{name} - Loss: {avg_loss_reduced:.4f}, PPL: {avg_ppl_reduced:.4f}")

    if rank == 0:
        if torch.cuda.is_available():
            log_dict.update({
                'memory/val_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory/val_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'memory/val_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            })

        # Run qualitative + CORE eval once (not per dataset)
        base_model = model.module if isinstance(model, DDP) else model
        run_world_knowledge_validation(base_model, tokenizer, device=device)
        sentence_completions = run_sentence_completion(base_model, tokenizer, device=device)
        core_results = run_core_eval(base_model, tokenizer, device=device, num_examples=core_examples)

        if core_results:
            log_dict['core/metric'] = core_results['core_metric']
            log_dict['core/raw_accuracy'] = core_results['raw_accuracy']
            for label, centered in core_results['per_task_centered'].items():
                log_dict[f'core_tasks/{label}'] = centered

        if use_wandb:
            wandb.log(log_dict)
            if sentence_completions:
                wandb.log({
                    "sentence_completions": wandb.Table(
                        columns=["step", "prompt", "completion"],
                        data=[[global_step, item['prompt'], item['completion']] for item in sentence_completions]
                    )
                })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---- Optimizer ----

def create_optimizer_and_scheduler(model, peak_lr, min_lr, weight_decay, warmup_steps, max_steps):
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

    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95), eps=1e-8
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    return optimizer, scheduler


# ---- Main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--fast_dev_run', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--mode', type=str, default='pretrain', choices=['test', 'pretrain', 'midtrain'])
    parser.add_argument('--core_examples', type=int, default=27)
    parser.add_argument('--use_special_tokens', action='store_true')
    args = parser.parse_args()

    # Validate mode requirements
    if args.mode == 'midtrain' and not args.use_special_tokens:
        raise ValueError("--mode midtrain requires --use_special_tokens")

    is_training = args.mode in ('pretrain', 'midtrain')
    mode_train = args.mode if is_training else 'pretrain'  # test mode uses pretrain config for data

    rank, world_size, local_rank = setup_distributed()

    # Load mode-dependent config
    train_config = TRAIN_CONFIGS[mode_train]
    peak_lr = train_config['peak_lr']
    min_lr = train_config['min_lr']
    weight_decay = train_config['weight_decay']

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    max_steps = args.max_steps
    val_check_interval = 50
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if args.use_special_tokens:
        special_tokens = {
            'bos_token': '<|beginoftext|>',
            'pad_token': '<|pad|>',
            'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|system|>']
        }
        tokenizer.add_special_tokens(special_tokens)
        print0(f"Added special tokens: vocab {50257} -> {len(tokenizer)}")
        print0(f"  bos={tokenizer.bos_token_id}, pad={tokenizer.pad_token_id}, "
               f"eos={tokenizer.eos_token_id}")

    n_layers = 20
    max_seq_len = 2048
    batch_size = args.batch_size
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)
    warmup_steps = int(train_config['warmup_ratio'] * max_steps)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size, dim=dim,
        n_layers=n_layers, n_heads=n_heads, max_seq_len=max_seq_len
    )

    print0(f"\n=== Configuration ===")
    print0(f"Mode: {args.mode}")
    print0(f"Peak LR: {peak_lr}, Min LR: {min_lr}, Weight decay: {weight_decay}")
    print0(f"Warmup steps: {warmup_steps} ({train_config['warmup_ratio']*100:.1f}% of {max_steps})")
    print0(f"Datasets: {' | '.join(f'{k}={v:.1%}' for k, v in train_config['datasets'].items())}")
    print0(f"Loss: {train_config['loss_fn']}")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        accelerator = 'gpu'
    else:
        device = torch.device('cpu')
        accelerator = 'cpu'

    effective_batch_size = batch_size * world_size
    print0(f"World size: {world_size}, Batch size per GPU: {batch_size}, Effective: {effective_batch_size}")
    print0(f"Tokens per step: {effective_batch_size * max_seq_len:,}")
    print0(f"Special tokens: {'ON' if args.use_special_tokens else 'OFF'}")

    model = model.to(device)

    if world_size > 1:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, peak_lr, min_lr, weight_decay, warmup_steps, max_steps
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(accelerator == 'gpu'))

    bos_token_id = tokenizer.bos_token_id if args.use_special_tokens else None
    pad_token_id = tokenizer.pad_token_id if args.use_special_tokens else None

    train_loader, val_loaders = create_dataloaders(
        mode_train, tokenizer, batch_size, max_seq_len,
        num_workers=0, val_sequences=1000, rank=rank, world_size=world_size,
        bos_token_id=bos_token_id, pad_token_id=pad_token_id
    )

    use_wandb = rank == 0 and is_training
    wandb_run_id = None
    run_name = f"{mode_train}_{world_size}gpu_bs{effective_batch_size}"

    if use_wandb:
        wandb_run_id = wandb.util.generate_id()
        run_dir = logs_dir / wandb_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project="llm-training", name=run_name, id=wandb_run_id,
            dir=str(run_dir), resume="allow",
            config={
                "mode": args.mode,
                "vocab_size": len(tokenizer), "dim": dim,
                "n_layers": n_layers, "n_heads": n_heads, "max_seq_len": max_seq_len,
                "batch_size_per_gpu": batch_size, "num_gpus": world_size,
                "effective_batch_size": effective_batch_size,
                "tokens_per_step": effective_batch_size * max_seq_len,
                "peak_lr": peak_lr, "min_lr": min_lr, "weight_decay": weight_decay,
                "warmup_steps": warmup_steps, "max_steps": max_steps,
                "use_special_tokens": args.use_special_tokens,
                "dataset_mix": train_config['datasets'],
                "loss_fn": train_config['loss_fn'],
            }
        )

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

    global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            ckpt_subdir = resume_path / "checkpoints"
            checkpoint_path = find_latest_checkpoint(ckpt_subdir if ckpt_subdir.exists() else resume_path)
        else:
            checkpoint_path = resume_path

        if checkpoint_path and checkpoint_path.exists():
            global_step = load_checkpoint(checkpoint_path, model, scheduler, device)
        else:
            print0(f"No checkpoint found at {args.resume}, starting from scratch")

    if args.use_special_tokens:
        resize_embeddings(model, len(tokenizer))
        model = model.to(device)

    if not is_training:
        if global_step == 0:
            print0("WARNING: No checkpoint loaded. Running test on untrained model.")

        print0(f"\n=== Test Mode (loaded step {global_step}) ===")

        # Run per-dataset validation
        validate(model, val_loaders, device, rank, world_size, global_step,
                tokenizer, False, limit_val_batches, mode_train,
                args.core_examples, pad_token_id)

        cleanup_distributed()
        exit(0)

    if args.fast_dev_run > 0:
        max_steps = global_step + args.fast_dev_run

    pbar = tqdm(total=max_steps, initial=global_step, disable=(rank != 0), desc="Training")

    while global_step < max_steps:
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
            global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
            use_wandb, pbar, val_check_interval, val_loaders, tokenizer, limit_val_batches,
            warmup_steps, mode_train, checkpoint_dir=checkpoint_dir,
            core_examples=args.core_examples, pad_token_id=pad_token_id
        )

    pbar.close()
    print0(f"Training complete at step {global_step}")

    if use_wandb:
        wandb.finish()

    cleanup_distributed()
