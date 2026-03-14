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
from itertools import cycle
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import os
import copy
import math
import wandb
from tqdm import tqdm
from core.validation import run_world_knowledge_validation, run_sentence_completion, run_core_eval, run_conversation_validation

TRAIN_CONFIGS = {
    'pretrain': {
        'peak_lr': 6e-4,
        'min_lr': 6e-5,
        'weight_decay': 0.1,
        'warmup_ratio': 0.01,
        'datasets': {'fineweb': 1.0},
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
    },
    'sft': {
        'peak_lr': 2e-5,
        'min_lr': 2e-6,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'datasets': {
            'smoltalk': 0.65,
            'ultrachat_gen': 0.20,
            'ultrachat_sft': 0.15,
        },
    },
    'sft-lima': {
        'peak_lr': 1e-5,
        'min_lr': 2e-6,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'datasets': {'lima': 1.0},
    },
    'sft-benchmark': {
        'peak_lr': 5e-6,
        'min_lr': 1e-6,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'datasets': {
            'mmlu': 0.97,
            'gsm8k': 0.03
            },
    },
    'sft-topic-switch': {
        'peak_lr': 1e-5,
        'min_lr': 2e-6,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'datasets': {'topic_switch': 1.0},
    },
    'dpo': {
        'peak_lr':      1e-6,
        'min_lr':       1e-7,
        'weight_decay': 0.0,
        'warmup_ratio': 0.1,
        'dpo_beta':     0.1,
        'datasets': {
            'ultrafeedback': 0.6,
            'hh_rlhf':       0.4,
        },
    },
}


# ---- Shared tokenization ----

def tokenize_conversation(messages, tokenizer, mask_policy='assistant_only'):
    """Tokenize a conversation into (token_ids, loss_mask).

    mask_policy='all':
        Every token gets loss_mask=1. Used for pretrain/midtrain packed sequences.
    mask_policy='assistant_only':
        loss_mask=1 for: <|assistant|> role token, assistant content, the <|user|>
        token immediately following an assistant turn (turn-end signal), and EOS
        after the final assistant turn. Everything else (BOS, system, user context)
        gets loss_mask=0. Used for SFT and for DPO log-prob computation.

    Returns (tokens: list[int], loss_mask: list[int]).
    """
    bos_id  = tokenizer.bos_token_id
    eos_id  = tokenizer.eos_token_id
    role_ids = {
        'user':      tokenizer.convert_tokens_to_ids('<|user|>'),
        'assistant': tokenizer.convert_tokens_to_ids('<|assistant|>'),
        'system':    tokenizer.convert_tokens_to_ids('<|system|>'),
    }

    tokens = [bos_id]
    mask   = [0]

    prev_role = None
    for msg in messages:
        role    = msg.get('role', 'user')
        role_id = role_ids.get(role)
        is_asst = (role == 'assistant')

        if mask_policy == 'all':
            role_mask    = 1
            content_mask = 1
        else:  # 'assistant_only'
            is_turn_end  = (role == 'user' and prev_role == 'assistant')
            role_mask    = 1 if (is_asst or is_turn_end) else 0
            content_mask = 1 if is_asst else 0

        if role_id is not None:
            tokens.append(role_id)
            mask.append(role_mask)

        content_ids = tokenizer.encode(msg.get('content', ''), add_special_tokens=False)
        tokens.extend(content_ids)
        mask.extend([content_mask] * len(content_ids))
        prev_role = role

    tokens.append(eos_id)
    mask.append(1 if (mask_policy == 'all' or prev_role == 'assistant') else 0)

    return tokens, mask


# ---- SFT Datasets ----

class PackedStreamingDataset(IterableDataset):
    """Streams parquet files, tokenizes content, and packs into fixed-length chunks.

    Supports two data formats:
    - 'text': Raw text column (FineWeb-Edu). Tokenizes text, appends EOS.
    - 'conversation': Messages column with source. Formats as [BOS, role, content..., EOS].
      Optionally filters by source field.

    Every yielded batch includes a 'loss_mask' tensor (float, same shape as labels) controlled
    by mask_policy, which also determines the packing strategy:
    - 'all': linear stream-packing (no padding, no boundary alignment); loss on every token.
      Used for pretrain and midtrain.
    - 'assistant_only': conversation-boundary-aligned packing; each chunk always starts at the
      beginning of a conversation to avoid mid-conversation context gaps. Conversations longer
      than seq_length are truncated; short chunks are padded with pad_token_id (loss_mask=0).
      Loss only on: <|assistant|> role token, assistant content, the <|user|> token immediately
      following an assistant turn (turn-end signal), and EOS after the final assistant turn.
    """
    def __init__(self, files, tokenizer, seq_length=2048,
                 rank=0, world_size=1, shuffle=False, max_sequences=None,
                 data_format='text', source_filter=None, mask_policy='all'):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.max_sequences = max_sequences
        self.data_format = data_format
        self.source_filter = source_filter
        self.mask_policy = mask_policy

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def _tokenize_text(self, text):
        """Tokenize raw text, append EOS. Returns (tokens, loss_mask)."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.eos_token_id)
        mask = [1] * len(tokens)
        return tokens, mask

    def _tokenize_conversation(self, messages):
        return tokenize_conversation(messages, self.tokenizer, self.mask_policy)

    def _iter_tokens_from_file(self, filepath):
        """Yield (tokens, loss_mask) pairs from a single parquet file."""
        pf = pq.ParquetFile(filepath)

        if self.data_format == 'text':
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=['text'])
                for text in table['text'].to_pylist():
                    yield self._tokenize_text(text)

        elif self.data_format == 'conversation':
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=['messages', 'source'])
                messages_col = table['messages'].to_pylist()
                source_col = table['source'].to_pylist()

                for messages, source in zip(messages_col, source_col):
                    if self.source_filter is not None and source != self.source_filter:
                        continue
                    if not messages:
                        continue
                    yield self._tokenize_conversation(messages)

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

        # Compute which files belong to this shard.
        # If there are fewer files than shards (e.g. 1-file benchmark datasets with world_size=4),
        # fall back to modding by the number of files so every rank gets at least one file.
        effective_shards = min(total_shards, len(files)) if files else total_shards
        effective_shard_id = shard_id % effective_shards if effective_shards else 0
        my_file_indices = [i for i in range(len(files)) if i % effective_shards == effective_shard_id]
        if not my_file_indices:
            return

        file_iter = cycle(my_file_indices) if self.max_sequences else iter(my_file_indices)

        if self.mask_policy == 'assistant_only':
            yield from self._yield_bos_aligned(files, file_iter)
        else:
            yield from self._yield_packed(files, file_iter)

    def _yield_packed(self, files, file_iter):
        """Linear stream-packing: extend a token buffer and emit fixed-length chunks.

        Tokens from consecutive documents/conversations flow into a single deque with no
        padding and no respect for boundaries. loss_mask[i] = 1 iff labels[i] (chunk_t[i+1])
        is a token the model trains on (always true here — mask_policy='all').
        """
        buffer_t = deque()
        buffer_m = deque()
        sequences_yielded = 0

        for file_idx in file_iter:
            for tokens, mask in self._iter_tokens_from_file(files[file_idx]):
                buffer_t.extend(tokens)
                buffer_m.extend(mask)

                while len(buffer_t) >= self.seq_length + 1:
                    chunk_t = [buffer_t.popleft() for _ in range(self.seq_length + 1)]
                    chunk_m = [buffer_m.popleft() for _ in range(self.seq_length + 1)]
                    yield {
                        'input_ids': torch.tensor(chunk_t[:-1], dtype=torch.long),
                        'labels':    torch.tensor(chunk_t[1:],  dtype=torch.long),
                        'loss_mask': torch.tensor(chunk_m[1:],  dtype=torch.float),
                    }
                    sequences_yielded += 1
                    if self.max_sequences and sequences_yielded >= self.max_sequences:
                        return

    def _yield_bos_aligned(self, files, file_iter):
        """Conversation-boundary-aligned packing for SFT (mask_policy='assistant_only').

        Each chunk starts at the beginning of a conversation so the model always has full
        context for any assistant turn it trains on. When the next conversation would overflow
        the current chunk, the chunk is padded to seq_length+1 with pad_token_id (loss_mask=0)
        and yielded. Conversations longer than seq_length are truncated.
        """
        buffer_t = []
        buffer_m = []
        sequences_yielded = 0

        for file_idx in file_iter:
            for tokens, mask in self._iter_tokens_from_file(files[file_idx]):
                if len(tokens) > self.seq_length + 1:
                    tokens = tokens[:self.seq_length + 1]
                    mask = mask[:self.seq_length + 1]

                if buffer_t and len(buffer_t) + len(tokens) > self.seq_length + 1:
                    pad_len = self.seq_length + 1 - len(buffer_t)
                    chunk_t = buffer_t + [self.pad_token_id] * pad_len
                    chunk_m = buffer_m + [0] * pad_len
                    buffer_t, buffer_m = [], []
                    yield {
                        'input_ids': torch.tensor(chunk_t[:-1], dtype=torch.long),
                        'labels':    torch.tensor(chunk_t[1:],  dtype=torch.long),
                        'loss_mask': torch.tensor(chunk_m[1:],  dtype=torch.float),
                    }
                    sequences_yielded += 1
                    if self.max_sequences and sequences_yielded >= self.max_sequences:
                        return

                buffer_t.extend(tokens)
                buffer_m.extend(mask)

                if len(buffer_t) == self.seq_length + 1:
                    yield {
                        'input_ids': torch.tensor(buffer_t[:-1], dtype=torch.long),
                        'labels':    torch.tensor(buffer_t[1:],  dtype=torch.long),
                        'loss_mask': torch.tensor(buffer_m[1:],  dtype=torch.float),
                    }
                    buffer_t, buffer_m = [], []
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
            try:
                yield next(iters[idx])
            except StopIteration:
                # Sub-dataset exhausted — restart it. Both yield next() calls must be inside
                # try/except because PEP 479 converts any StopIteration escaping a generator
                # into RuntimeError.
                _, ds = self.datasets_with_names[idx]
                iters[idx] = iter(ds)
                try:
                    yield next(iters[idx])
                except StopIteration:
                    pass  # dataset truly empty, skip this draw


# ---- DPO Datasets ----

class DPOModel(nn.Module):
    """Policy model + frozen reference copy held in one object.

    The reference is stored via object.__setattr__, bypassing nn.Module's submodule
    registry. This means DDP only sees and synchronises the policy parameters — the
    reference is a plain Python attribute, invisible to DDP and the optimiser.

    After loading an SFT checkpoint into the policy, call init_reference_from_policy()
    to snapshot the weights into the reference before any DPO updates begin.
    """
    def __init__(self, vocab_size, dim, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.policy = TinyGPT(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                               n_heads=n_heads, max_seq_len=max_seq_len)
        # Stored outside nn.Module's registry → invisible to DDP and optimiser
        object.__setattr__(self, '_ref', TinyGPT(vocab_size=vocab_size, dim=dim,
                                                  n_layers=n_layers, n_heads=n_heads,
                                                  max_seq_len=max_seq_len))
        for p in self._ref.parameters():
            p.requires_grad_(False)
        self._ref.eval()

    def to(self, *args, **kwargs):
        # nn.Module.to() only moves registered submodules; move _ref manually.
        super().to(*args, **kwargs)
        self._ref.to(*args, **kwargs)
        return self

    def init_reference_from_policy(self):
        """Snapshot policy weights into the frozen reference. Call once after loading
        the SFT checkpoint, before the first optimiser step."""
        self._ref.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        for p in self._ref.parameters():
            p.requires_grad_(False)
        self._ref.eval()

    def forward(self, input_ids):
        return self.policy(input_ids)

    @torch.no_grad()
    def ref_forward(self, input_ids):
        return self._ref(input_ids)

class DPOStreamingDataset(IterableDataset):
    """Stream chosen/rejected preference pairs from a list of parquet shards.

    Each row must have:
      chosen   — list[{role, content}]  (full conversation with preferred response)
      rejected — list[{role, content}]  (full conversation with dispreferred response)

    Yields dicts: chosen_ids, chosen_mask, rejected_ids, rejected_mask — all [seq_length].
    Sequences are truncated or right-padded to seq_length with pad_token_id (mask=0).
    """
    def __init__(self, files, tokenizer, seq_length, rank=0, world_size=1,
                 split='train', val_fraction=0.1, max_sequences=None):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.val_fraction = val_fraction
        self.max_sequences = max_sequences
        self._pad_id = tokenizer.pad_token_id
        if not self.files:
            raise ValueError('DPOStreamingDataset: no parquet files provided')

    def _pad(self, token_ids, loss_mask):
        n = self.seq_length
        if len(token_ids) >= n:
            return token_ids[:n], loss_mask[:n]
        pad = n - len(token_ids)
        return token_ids + [self._pad_id] * pad, loss_mask + [0] * pad

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        # Shard across DDP ranks and dataloader workers; fall back if fewer
        # files than shards (same logic as PackedStreamingDataset).
        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id
        effective_shards = min(total_shards, len(self.files)) if self.files else total_shards
        effective_shard_id = shard_id % effective_shards if effective_shards else 0
        my_files = [f for i, f in enumerate(self.files) if i % effective_shards == effective_shard_id]
        if not my_files:
            return

        file_iter = cycle(my_files) if self.split == 'train' else iter(my_files)
        sequences_yielded = 0

        for shard in file_iter:
            try:
                rows = pq.read_table(shard).to_pylist()
            except Exception as e:
                print0(f'Error loading {shard}: {e}')
                continue

            split_idx = max(1, int(self.val_fraction * len(rows)))
            rows = rows[:split_idx] if self.split == 'val' else rows[split_idx:]
            random.shuffle(rows)

            for row in rows:
                c = tokenize_conversation(row['chosen'],   self.tokenizer, 'assistant_only')
                r = tokenize_conversation(row['rejected'], self.tokenizer, 'assistant_only')
                if c is None or r is None:
                    continue

                c_ids, c_mask = self._pad(*c)
                r_ids, r_mask = self._pad(*r)

                # Skip examples where no assistant tokens survived truncation
                if sum(c_mask) == 0 or sum(r_mask) == 0:
                    continue

                yield {
                    'chosen_ids':    torch.tensor(c_ids,  dtype=torch.long),
                    'chosen_mask':   torch.tensor(c_mask, dtype=torch.long),
                    'rejected_ids':  torch.tensor(r_ids,  dtype=torch.long),
                    'rejected_mask': torch.tensor(r_mask, dtype=torch.long),
                }
                sequences_yielded += 1
                if self.max_sequences and sequences_yielded >= self.max_sequences:
                    return  # val dataset — stop after quota


# ---- Collate ----

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'labels':    torch.stack([x['labels']    for x in batch]),
        'loss_mask': torch.stack([x['loss_mask'] for x in batch]),
    }


# ---- SFT Dataloaders ----

def create_dataloaders(mode_train, tokenizer, batch_size, seq_length,
                       num_workers, val_sequences, rank, world_size):
    """Create train loader and per-dataset val loaders based on mode_train.

    Returns:
        train_loader: DataLoader
        val_loaders: dict of {name: DataLoader}
    """
    config = TRAIN_CONFIGS[mode_train]
    dataset_config = config['datasets']

    fineweb_dir = Path('data/base_data')
    fineweb_files = sorted(fineweb_dir.glob("*.parquet"))
    fineweb_val_files = fineweb_files[:1]
    fineweb_train_files = fineweb_files[1:]

    val_loaders = {}

    if mode_train == 'pretrain':
        print0(f"Data: {len(fineweb_train_files)} FineWeb train shards, {len(fineweb_val_files)} val shards")

        train_dataset = PackedStreamingDataset(
            fineweb_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='text'
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Val loaders use rank=0, world_size=1 so all ranks eval the same data (correct reduce_tensor)
        fineweb_val = PackedStreamingDataset(
            fineweb_val_files, tokenizer, seq_length, rank=0, world_size=1,
            shuffle=False, max_sequences=val_sequences, data_format='text'
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

        fineweb_train = PackedStreamingDataset(
            fineweb_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='text'
        )
        smoltalk_train = PackedStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter='smoltalk'
        )
        ultrachat_gen_train = PackedStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter='ultrachat_gen'
        )
        ultrachat_sft_train = PackedStreamingDataset(
            conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter='ultrachat_sft'
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

        num_val_datasets = len(dataset_config)
        val_seqs_per_dataset = val_sequences // num_val_datasets

        fineweb_val = PackedStreamingDataset(
            fineweb_val_files, tokenizer, seq_length, rank=0, world_size=1,
            shuffle=False, max_sequences=val_seqs_per_dataset, data_format='text'
        )
        val_loaders['fineweb'] = DataLoader(
            fineweb_val, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True
        )

        for source in ['smoltalk', 'ultrachat_gen', 'ultrachat_sft']:
            conv_val = PackedStreamingDataset(
                conv_val_files, tokenizer, seq_length, rank=0, world_size=1,
                shuffle=False, max_sequences=val_seqs_per_dataset,
                data_format='conversation', source_filter=source
            )
            val_loaders[source] = DataLoader(
                conv_val, batch_size=batch_size, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=True
            )

    elif mode_train == 'sft':
        conv_dir = Path('data/conversation_data')
        conv_files = sorted(conv_dir.glob("*.parquet"))
        conv_val_files = conv_files[:1]
        conv_train_files = conv_files[1:]

        sources = list(dataset_config.keys())
        probabilities = list(dataset_config.values())
        print0(f"Data: {len(conv_train_files)} conversation train shards (SFT, assistant-only loss)")
        print0(f"  Mix: {' | '.join(f'{k}={v:.1%}' for k, v in dataset_config.items())}")

        datasets_with_names = [
            (src, PackedStreamingDataset(
                conv_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
                shuffle=True, data_format='conversation', source_filter=src,
                mask_policy='assistant_only',
            ))
            for src in sources
        ]
        train_dataset = MixedStreamingDataset(datasets_with_names, probabilities, seed=42, rank=rank)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        val_seqs_per_dataset = val_sequences // len(sources)
        for src in sources:
            conv_val = PackedStreamingDataset(
                conv_val_files, tokenizer, seq_length, rank=0, world_size=1,
                shuffle=False, max_sequences=val_seqs_per_dataset,
                data_format='conversation', source_filter=src,
                mask_policy='assistant_only',
            )
            val_loaders[src] = DataLoader(
                conv_val, batch_size=batch_size, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=True
            )

    elif mode_train == 'sft-lima':
        lima_dir = Path('data/lima_data')
        lima_files = sorted(lima_dir.glob("*.parquet"))
        if not lima_files:
            raise FileNotFoundError("No LIMA parquet files found in data/lima_data/. "
                                    "Run: python -m core.dataset --lima")

        print0(f"Data: LIMA ({len(lima_files)} file, 100% assistant-only loss)")

        train_dataset = PackedStreamingDataset(
            lima_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Val on the same file (dataset is too small to hold out a separate split)
        lima_val = PackedStreamingDataset(
            lima_files, tokenizer, seq_length, rank=0, world_size=1,
            shuffle=False, max_sequences=val_sequences,
            data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )
        val_loaders['lima'] = DataLoader(
            lima_val, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True
        )

    elif mode_train == 'sft-benchmark':
        mmlu_dir   = Path('data/mmlu_data')
        gsm8k_dir  = Path('data/gsm8k_data')
        mmlu_train  = sorted(mmlu_dir.glob("train.parquet"))
        mmlu_test   = sorted(mmlu_dir.glob("test.parquet"))
        gsm8k_train = sorted(gsm8k_dir.glob("train.parquet"))
        gsm8k_test  = sorted(gsm8k_dir.glob("test.parquet"))

        missing = []
        if not mmlu_train:  missing.append("data/mmlu_data/train.parquet  (run: python -m core.dataset --mmlu)")
        if not gsm8k_train: missing.append("data/gsm8k_data/train.parquet  (run: python -m core.dataset --gsm8k)")
        if missing:
            raise FileNotFoundError("Missing benchmark data:\n  " + "\n  ".join(missing))

        print0(f"Data: MMLU train ({len(mmlu_train)} file), GSM8K train ({len(gsm8k_train)} file) "
               f"— weights {' '.join(f'{k}:{v:.2f}' for k, v in dataset_config.items())}, assistant-only loss")

        mmlu_ds = PackedStreamingDataset(
            mmlu_train, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )
        gsm8k_ds = PackedStreamingDataset(
            gsm8k_train, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )

        weights = dataset_config
        train_dataset = MixedStreamingDataset(
            [('mmlu', mmlu_ds), ('gsm8k', gsm8k_ds)],
            probabilities=[weights['mmlu'], weights['gsm8k']],
            rank=rank,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        if mmlu_test:
            mmlu_val = PackedStreamingDataset(
                mmlu_test, tokenizer, seq_length, rank=0, world_size=1,
                shuffle=False, max_sequences=val_sequences // 2,
                data_format='conversation', source_filter=None,
                mask_policy='assistant_only',
            )
            val_loaders['mmlu'] = DataLoader(
                mmlu_val, batch_size=batch_size, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=True
            )
        if gsm8k_test:
            gsm8k_val = PackedStreamingDataset(
                gsm8k_test, tokenizer, seq_length, rank=0, world_size=1,
                shuffle=False, max_sequences=val_sequences // 2,
                data_format='conversation', source_filter=None,
                mask_policy='assistant_only',
            )
            val_loaders['gsm8k'] = DataLoader(
                gsm8k_val, batch_size=batch_size, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=True
            )

    elif mode_train == 'sft-topic-switch':
        ts_dir = Path('data/topic_switch_data')
        ts_files = sorted(ts_dir.glob("*.parquet"))
        if not ts_files:
            raise FileNotFoundError("No topic-switch parquet files found in data/topic_switch_data/. "
                                    "Run: python -m core.dataset --topic-switch")
        # First shard → val, rest → train. If only one shard, use it for both.
        ts_val_files   = ts_files[:1]
        ts_train_files = ts_files[1:] if len(ts_files) > 1 else ts_files

        print0(f"Data: topic-switch ({len(ts_train_files)} train shard(s), {len(ts_val_files)} val shard(s), assistant-only loss)")

        train_dataset = PackedStreamingDataset(
            ts_train_files, tokenizer, seq_length, rank=rank, world_size=world_size,
            shuffle=True, data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        ts_val = PackedStreamingDataset(
            ts_val_files, tokenizer, seq_length, rank=0, world_size=1,
            shuffle=False, max_sequences=val_sequences,
            data_format='conversation', source_filter=None,
            mask_policy='assistant_only',
        )
        val_loaders['topic_switch'] = DataLoader(
            ts_val, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loaders


# ---- DPO Dataloaders ----

def create_dpo_dataloaders(mode, batch_size, tokenizer, seq_length, val_sequences, rank, world_size):
    """Create train and val loaders for DPO modes.

    Returns:
        train_loaders: dict of {source: DataLoader}
        val_loaders:   dict of {source: DataLoader}

    Each shard is split row-wise: first 10% of rows → val, remaining 90% → train.
    All shards are used for both splits. val_sequences caps how many val examples are
    evaluated per step (per source).
    """
    config = TRAIN_CONFIGS[mode]

    train_loaders, val_loaders = {}, {}
    val_seqs_per_source = val_sequences // len(config['datasets'])

    for source in config['datasets']:
        data_dirs = {'ultrafeedback': 'data/ultrafeedback_data', 'hh_rlhf': 'data/hh_rlhf_data'}
        data_dir = Path(data_dirs[source])
        shards   = sorted(data_dir.glob('*.parquet'))

        if not shards:
            raise FileNotFoundError(
                f'No parquet files in {data_dir}. '
                f'Run: python -m core.dataset --{source.replace("_", "-")}'
            )

        print0(f"  DPO {source}: {len(shards)} shards (90% train / 10% val per shard)")

        train_loaders[source] = DataLoader(
            DPOStreamingDataset(shards, tokenizer, seq_length, rank=rank, world_size=world_size,
                                split='train'),
            batch_size=batch_size, num_workers=0
        )
        val_loaders[source] = DataLoader(
            DPOStreamingDataset(shards, tokenizer, seq_length, rank=0, world_size=1,
                                split='val', max_sequences=val_seqs_per_source),
            batch_size=batch_size, num_workers=0
        )

    return train_loaders, val_loaders


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

    # For DPOModel save only the policy weights — same format as SFT checkpoints,
    # so DPO checkpoints can be loaded back into a plain TinyGPT for evaluation.
    if isinstance(base_model, DPOModel):
        state_dict = base_model.policy.state_dict()
    else:
        state_dict = base_model.state_dict()

    path = checkpoint_dir / f"checkpoint_{global_step}.pt"
    torch.save({'model_state_dict': state_dict}, path)
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

def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint. Returns the global step.

    Handles vocab size mismatches between checkpoint and current model.
    Old checkpoints (pre-special-tokens, vocab=50257) are loaded by temporarily
    resizing the model to the checkpoint's vocab; the caller must call
    resize_embeddings() afterwards to expand to the current tokenizer vocab.
    New checkpoints (with special tokens, vocab=50262) load without any resize.

    For DPOModel, weights are loaded into the policy submodule.
    """
    print0(f"Loading checkpoint from {checkpoint_path}")

    global_step = int(Path(checkpoint_path).stem.split('_')[-1])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_state = checkpoint['model_state_dict']

    base_model = model.module if isinstance(model, DDP) else model
    target = base_model.policy if isinstance(base_model, DPOModel) else base_model

    if 'tok_emb.weight' in saved_state:
        saved_vocab = saved_state['tok_emb.weight'].shape[0]
        current_vocab = target.tok_emb.num_embeddings
        if saved_vocab != current_vocab:
            print0(f"Vocab size mismatch: checkpoint={saved_vocab}, model={current_vocab}. "
                   f"Temporarily resizing to load checkpoint (resize_embeddings will expand afterwards).")
            dim = saved_state['tok_emb.weight'].shape[1]
            target.tok_emb = nn.Embedding(saved_vocab, dim).to(device)
            target.head = nn.Linear(dim, saved_vocab, bias=False).to(device)

    target.load_state_dict(saved_state)
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


# ---- Loss functions ----

def compute_loss(logits, labels, loss_mask):
    """Masked token-level cross-entropy.

    loss_mask is a float tensor matching labels: 1.0 where the token contributes
    to the loss, 0.0 where it is masked out. For pretrain/midtrain the mask is all
    ones (equivalent to standard mean cross-entropy). For SFT the mask covers only
    assistant-generated tokens. Uses sum/count rather than mean to avoid dividing by
    zero when a packed chunk contains no unmasked tokens.
    """
    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
    ).view(labels.shape)
    return (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)


def get_sequence_logprobs(logits, token_ids, loss_mask):
    """Sum of per-token log probs over masked (assistant) positions.

    logits:    [B, T, V]  logits (bfloat16 or float32)
    token_ids: [B, T]     ground-truth token IDs
    loss_mask: [B, T]     1 for assistant positions, 0 elsewhere
    Returns:   [B]        per-sequence summed log probs (float32)
    """
    shift_logits = logits[:, :-1, :]          # [B, T-1, V]
    shift_ids = token_ids[:, 1:]              # [B, T-1]
    shift_mask = loss_mask[:, 1:].float()     # [B, T-1]

    # Process one sequence at a time so both forward and backward of log_softmax are
    # bounded to [T-1, V] (~400 MB) rather than [B, T-1, V] (~3.5 GB). Autograd frees
    # each slice's saved tensors before moving to the next.
    logps = []
    for i in range(shift_logits.size(0)):
        nll_i = F.cross_entropy(shift_logits[i], shift_ids[i], reduction='none')  # [T-1]
        logps.append((-nll_i.float() * shift_mask[i]).sum())
    return torch.stack(logps)  # [B]


def compute_dpo_loss(policy_logits, ref_logits,
                     chosen_ids, rejected_ids,
                     chosen_mask, rejected_mask,
                     beta):
    """Standard DPO loss (Rafailov et al. 2023, eq. 7).

    policy_logits / ref_logits: [2B, T, V] for the concatenated [chosen | rejected] batch.
    Chosen examples occupy the first B rows, rejected the last B.

    Returns (loss, chosen_reward_mean, rejected_reward_mean, reward_accuracy).
    reward = beta * (log π_θ − log π_ref).  reward_accuracy = mean(chosen_r > rejected_r).
    """
    B = chosen_ids.size(0)

    pol_c, pol_r = policy_logits[:B], policy_logits[B:]

    ref_c_logps = get_sequence_logprobs(ref_logits[:B], chosen_ids, chosen_mask)
    ref_r_logps = get_sequence_logprobs(ref_logits[B:], rejected_ids, rejected_mask)
    del ref_logits  # free before policy forward-graph is retained for backward

    pol_c_logps = get_sequence_logprobs(pol_c, chosen_ids, chosen_mask)
    pol_r_logps = get_sequence_logprobs(pol_r, rejected_ids, rejected_mask)

    chosen_rewards = beta * (pol_c_logps - ref_c_logps)
    rejected_rewards = beta * (pol_r_logps - ref_r_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    with torch.no_grad():
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards.mean().detach(), rejected_rewards.mean().detach(), reward_acc


# ---- Training ----

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
                global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
                use_wandb, pbar, val_check_interval, val_loaders, tokenizer, limit_val_batches,
                warmup_steps, checkpoint_dir=None):
    model.train()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            logits = model(input_ids)
            loss = compute_loss(logits, labels, loss_mask)
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
                        tokenizer, use_wandb, limit_val_batches)
                model.train()

                if checkpoint_dir:
                    save_checkpoint(model, global_step, checkpoint_dir, rank)
                    if world_size > 1:
                        dist.barrier()

            if global_step >= max_steps:
                break

    return global_step


def train_dpo_epoch(model, train_loaders, dataset_weights, optimizer, scheduler, scaler,
                    device, rank, world_size, global_step, max_steps, accumulate_grad_batches,
                    gradient_clip_val, log_every_n_steps, use_wandb, pbar, val_check_interval,
                    val_loaders, tokenizer, dpo_beta, limit_val_batches, checkpoint_dir=None):
    model.train()
    optimizer.zero_grad()

    base_model = model.module if isinstance(model, DDP) else model
    iterators = {src: iter(ldr) for src, ldr in train_loaders.items()}
    batch_idx = 0

    while global_step < max_steps:
        source = random.choices(list(dataset_weights.keys()),
                                weights=list(dataset_weights.values()))[0]
        try:
            batch = next(iterators[source])
        except StopIteration:
            iterators[source] = iter(train_loaders[source])
            batch = next(iterators[source])

        chosen_ids = batch['chosen_ids'].to(device)
        chosen_mask = batch['chosen_mask'].to(device)
        rejected_ids = batch['rejected_ids'].to(device)
        rejected_mask = batch['rejected_mask'].to(device)

        # Concatenate chosen + rejected into one batch: 2 forward passes instead of 4
        combined_ids = torch.cat([chosen_ids, rejected_ids], dim=0)  # [2B, T]

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            policy_logits = model(combined_ids)

        ref_logits = base_model.ref_forward(combined_ids)

        loss, chosen_r, rejected_r, reward_acc = compute_dpo_loss(
            policy_logits, ref_logits,
            chosen_ids, rejected_ids, chosen_mask, rejected_mask, dpo_beta
        )
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
                loss_val = loss.item() * accumulate_grad_batches
                loss_reduced = reduce_tensor(torch.tensor(loss_val, device=device), world_size)
                acc_reduced = reduce_tensor(reward_acc, world_size)

                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log_dict = {
                        'train/dpo_loss':       loss_reduced.item(),
                        'train/chosen_reward':  chosen_r.item(),
                        'train/rejected_reward':rejected_r.item(),
                        'train/reward_margin':  (chosen_r - rejected_r).item(),
                        'train/reward_accuracy':acc_reduced.item(),
                        'lr': current_lr,
                        'global_step': global_step,
                    }

                    if torch.cuda.is_available():
                        log_dict.update({
                            'memory/allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                            'memory/reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                            'memory/max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                        })

                    if use_wandb:
                        wandb.log(log_dict)

                    pbar.set_postfix({
                        'dpo_loss': f"{loss_reduced.item():.4f}",
                        'acc': f"{acc_reduced.item():.3f}",
                        'lr': f"{current_lr:.2e}",
                    })

            pbar.update(1)

            if global_step % val_check_interval == 0:
                validate_dpo(model, val_loaders, device, rank, world_size, global_step,
                             tokenizer, dpo_beta, use_wandb, limit_val_batches)
                model.train()

                if checkpoint_dir:
                    save_checkpoint(model, global_step, checkpoint_dir, rank)
                    if world_size > 1:
                        dist.barrier()

            if global_step >= max_steps:
                break

        batch_idx += 1

    return global_step


# ---- Validation ----

def validate(model, val_loaders, device, rank, world_size, global_step, tokenizer,
             use_wandb, limit_val_batches):
    """Run per-dataset validation and CORE eval."""
    model.eval()
    log_dict = {'global_step': global_step}

    for name, loader in val_loaders.items():
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= limit_val_batches:
                    break

                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                loss_mask = batch['loss_mask'].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    logits = model(input_ids)
                    loss = compute_loss(logits, labels, loss_mask)

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
        conversation_completions = run_conversation_validation(base_model, tokenizer, device=device)
        core_results = run_core_eval(base_model, tokenizer, device=device, num_examples=500)

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
            if conversation_completions:
                wandb.log({
                    "conversation_validation": wandb.Table(
                        columns=["step", "group", "desc", "system", "user", "response"],
                        data=[
                            [global_step, item['group'], item['desc'],
                             item['system'], item['user'], item['response']]
                            for item in conversation_completions
                        ]
                    )
                })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def validate_dpo(model, val_loaders, device, rank, world_size, global_step, tokenizer,
                 dpo_beta, use_wandb, limit_val_batches):
    """DPO validation: per-dataset DPO loss + reward accuracy, then qualitative evals."""
    model.eval()
    base_model = model.module if isinstance(model, DDP) else model
    log_dict = {'global_step': global_step}

    for name, loader in val_loaders.items():
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= limit_val_batches:
                    break

                chosen_ids = batch['chosen_ids'].to(device)
                chosen_mask = batch['chosen_mask'].to(device)
                rejected_ids = batch['rejected_ids'].to(device)
                rejected_mask = batch['rejected_mask'].to(device)
                combined_ids = torch.cat([chosen_ids, rejected_ids], dim=0)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    policy_logits = model(combined_ids)

                ref_logits = base_model.ref_forward(combined_ids)

                loss, _, _, reward_acc = compute_dpo_loss(
                    policy_logits, ref_logits,
                    chosen_ids, rejected_ids, chosen_mask, rejected_mask, dpo_beta
                )

                total_loss += loss.item()
                total_acc += reward_acc.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0

        avg_loss_reduced = reduce_tensor(torch.tensor(avg_loss, device=device), world_size).item()
        avg_acc_reduced = reduce_tensor(torch.tensor(avg_acc, device=device), world_size).item()

        log_dict[f'val/{name}/dpo_loss'] = avg_loss_reduced
        log_dict[f'val/{name}/reward_accuracy'] = avg_acc_reduced

        print0(f"  val/{name} - DPO Loss: {avg_loss_reduced:.4f}, Reward Acc: {avg_acc_reduced:.3f}")

    if rank == 0:
        if torch.cuda.is_available():
            log_dict.update({
                'memory/val_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory/val_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'memory/val_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            })

        run_world_knowledge_validation(base_model.policy, tokenizer, device=device)
        sentence_completions = run_sentence_completion(base_model.policy, tokenizer, device=device)
        conversation_completions = run_conversation_validation(base_model.policy, tokenizer, device=device)
        core_results = run_core_eval(base_model.policy, tokenizer, device=device, num_examples=500)

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
            if conversation_completions:
                wandb.log({
                    "conversation_validation": wandb.Table(
                        columns=["step", "group", "desc", "system", "user", "response"],
                        data=[
                            [global_step, item['group'], item['desc'],
                             item['system'], item['user'], item['response']]
                            for item in conversation_completions
                        ]
                    )
                })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---- Optimizer ----

def create_optimizer_and_scheduler(model, opt_config):
    peak_lr = opt_config['peak_lr']
    min_lr = opt_config['min_lr']
    weight_decay = opt_config['weight_decay']
    warmup_steps = opt_config['warmup_steps']
    max_steps = opt_config['max_steps']

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


# ---- Entry point ----

def run(args):
    is_dpo = args.mode.startswith('dpo')
    is_training = args.mode != 'test'
    mode_train = args.mode if is_training else 'pretrain'

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        'bos_token': '<|beginoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|system|>']
    })
    print0(f"Added special tokens: vocab {50257} -> {len(tokenizer)}")
    print0(f"  bos={tokenizer.bos_token_id}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")

    train_config = TRAIN_CONFIGS[mode_train]
    peak_lr = train_config['peak_lr']
    min_lr = train_config['min_lr']
    weight_decay = train_config['weight_decay']
    dpo_beta = train_config.get('dpo_beta', None)
    max_steps = args.max_steps
    batch_size = args.batch_size

    n_layers = 20
    max_seq_len = 2048
    dim = n_layers * 64
    n_heads = max(1, (dim + 127) // 128)
    effective_batch_size = batch_size * world_size
    sft_modes = ('sft', 'sft-lima', 'sft-benchmark', 'sft-topic-switch')

    print0(f"\n=== Configuration ===")
    print0(f"Mode: {args.mode}")
    print0(f"Peak LR: {peak_lr}, Min LR: {min_lr}, Weight decay: {weight_decay}" +
           (f", Beta: {dpo_beta}" if is_dpo else ""))
    print0(f"Datasets: {' | '.join(f'{k}={v:.1%}' for k, v in train_config['datasets'].items())}")
    if not is_dpo:
        print0(f"Mask policy: {'assistant_only' if mode_train in sft_modes else 'all'}")
    print0(f"World size: {world_size}, Batch size per GPU: {batch_size}, Effective: {effective_batch_size}")

    if is_dpo:
        model = DPOModel(vocab_size=len(tokenizer), dim=dim,
                         n_layers=n_layers, n_heads=n_heads, max_seq_len=max_seq_len).to(device)
    else:
        model = TinyGPT(vocab_size=len(tokenizer), dim=dim,
                        n_layers=n_layers, n_heads=n_heads, max_seq_len=max_seq_len).to(device)

    # Load checkpoint BEFORE DDP wrapping and resize
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            ckpt_subdir = resume_path / "checkpoints"
            checkpoint_path = find_latest_checkpoint(ckpt_subdir if ckpt_subdir.exists() else resume_path)
        else:
            checkpoint_path = resume_path

        if checkpoint_path and checkpoint_path.exists():
            global_step = load_checkpoint(checkpoint_path, model, device)
        else:
            print0(f"No checkpoint found at {args.resume}, starting from scratch")

    remaining_steps = max_steps - global_step
    warmup_steps = int(train_config['warmup_ratio'] * remaining_steps)
    print0(f"Warmup steps: {warmup_steps} ({train_config['warmup_ratio']*100:.1f}% of {remaining_steps} remaining)")

    # Resize embeddings BEFORE DDP wrapping (avoids stale parameter references)
    resize_target = model.policy if is_dpo else model
    resize_embeddings(resize_target, len(tokenizer))
    resize_target.to(device)

    if is_dpo:
        # Snapshot policy → frozen reference (must happen after resize, before DDP)
        model.init_reference_from_policy()
        print0("Reference policy initialised from loaded policy weights.")

    if world_size > 1:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # create_optimizer_and_scheduler only sees policy params via named_parameters()
    # because _ref is not a registered submodule
    opt_config = {
        'peak_lr': peak_lr,
        'min_lr': min_lr,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'max_steps': remaining_steps,
    }
    optimizer, scheduler = create_optimizer_and_scheduler(model, opt_config)
    scaler = torch.amp.GradScaler('cuda', enabled=(accelerator == 'gpu'))

    if is_dpo:
        train_loaders, val_loaders = create_dpo_dataloaders(
            args.mode, batch_size, tokenizer, max_seq_len, val_sequences=1000, rank=rank, world_size=world_size
        )
    else:
        train_loader, val_loaders = create_dataloaders(
            mode_train, tokenizer, batch_size, max_seq_len,
            num_workers=0, val_sequences=1000, rank=rank, world_size=world_size
        )

    use_wandb = rank == 0 and is_training
    wandb_run_id = None
    run_name = f"{args.mode}_{world_size}gpu_bs{effective_batch_size}"
    if is_dpo:
        run_name += f"_beta{dpo_beta}"

    if use_wandb:
        wandb_run_id = wandb.util.generate_id()
        run_dir = logs_dir / wandb_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        wandb_config = {
            "mode": args.mode,
            "vocab_size": len(tokenizer), "dim": dim,
            "n_layers": n_layers, "n_heads": n_heads, "max_seq_len": max_seq_len,
            "batch_size_per_gpu": batch_size, "num_gpus": world_size,
            "effective_batch_size": effective_batch_size,
            "peak_lr": peak_lr, "min_lr": min_lr, "weight_decay": weight_decay,
            "warmup_steps": warmup_steps, "max_steps": max_steps,
            "dataset_mix": train_config['datasets'],
        }
        if is_dpo:
            wandb_config['dpo_beta'] = dpo_beta
        else:
            wandb_config['tokens_per_step'] = effective_batch_size * max_seq_len
            wandb_config['mask_policy'] = "assistant_only" if mode_train in sft_modes else "all"

        wandb.init(
            project="llm-training", name=run_name, id=wandb_run_id,
            dir=str(run_dir), resume="allow", config=wandb_config
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

    checkpoint_dir = logs_dir / (wandb_run_id or "no_wandb") / "checkpoints"
    accumulate_grad_batches = max(1, 4 // world_size)
    gradient_clip_val = 1.0
    log_every_n_steps = 1
    limit_val_batches = batch_size
    val_check_interval = args.eval_interval

    if not is_training:
        if global_step == 0:
            print0("WARNING: No checkpoint loaded. Running test on untrained model.")
        print0(f"\n=== Test Mode (loaded step {global_step}) ===")
        validate(model, val_loaders, device, rank, world_size, global_step,
                 tokenizer, False, limit_val_batches)
        cleanup_distributed()
        return

    if args.fast_dev_run > 0:
        max_steps = global_step + args.fast_dev_run

    pbar = tqdm(total=max_steps, initial=global_step, disable=(rank != 0),
                desc="DPO Training" if is_dpo else "Training")

    while global_step < max_steps:
        if is_dpo:
            global_step = train_dpo_epoch(
                model, train_loaders, train_config['datasets'], optimizer, scheduler, scaler,
                device, rank, world_size, global_step, max_steps, accumulate_grad_batches,
                gradient_clip_val, log_every_n_steps, use_wandb, pbar, val_check_interval,
                val_loaders, tokenizer, dpo_beta, limit_val_batches,
                checkpoint_dir=checkpoint_dir
            )
        else:
            global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, rank, world_size,
                global_step, max_steps, accumulate_grad_batches, gradient_clip_val, log_every_n_steps,
                use_wandb, pbar, val_check_interval, val_loaders, tokenizer, limit_val_batches,
                warmup_steps, checkpoint_dir=checkpoint_dir
            )

    pbar.close()
    print0(f"Training complete at step {global_step}")

    save_checkpoint(model, global_step, checkpoint_dir, rank)
    if world_size > 1:
        dist.barrier()

    if use_wandb:
        wandb.finish()

    cleanup_distributed()


# ---- Main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--max_steps', type=int, default=89200)
    parser.add_argument('--fast_dev_run', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--mode', type=str, default='dpo', choices=[
        'test', 'pretrain', 'midtrain',
        'sft', 'sft-lima', 'sft-benchmark', 'sft-topic-switch', 'dpo',
    ])
    parser.add_argument('--eval_interval', type=int, default=200)
    args = parser.parse_args()

    run(args)
