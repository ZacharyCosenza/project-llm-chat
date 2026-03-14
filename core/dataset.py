"""
Utilities for dataset download: FineWeb-Edu (pretraining), SmolTalk+UltraChat (midtraining), eval bundle.
See `repackage_data_reference.py` for dataset preparation details.
"""

import os
import re
import time
import random
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
import argparse

# Dataset config
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822
# Use absolute path relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'base_data')
os.makedirs(DATA_DIR, exist_ok=True)

def shard_path(index):
    """Get filepath for shard index."""
    return os.path.join(DATA_DIR, f"shard_{index:05d}.parquet")

def list_parquet_files(data_dir=None):
    """Return full paths to all parquet files in data_dir."""
    data_dir = data_dir or DATA_DIR
    files = sorted(f for f in os.listdir(data_dir) 
                   if f.endswith('.parquet') and not f.endswith('.tmp'))
    return [os.path.join(data_dir, f) for f in files]

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate dataset in row_group batches.
    split: "train" (all but last file) or "val" (last file only)
    start/step: for DDP (e.g., start=rank, step=world_size)
    """
    assert split in ["train", "val"]
    paths = list_parquet_files()
    paths = paths[:-1] if split == "train" else paths[-1:]
    
    for filepath in paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            yield pf.read_row_group(rg_idx).column('text').to_pylist()

def download_single_file(index):
    """Download shard with exponential backoff retry."""
    filepath = shard_path(index)
    if os.path.exists(filepath):
        print(f"Skipping shard_{index:05d} (exists)")
        return True

    url = f"{BASE_URL}/shard_{index:05d}.parquet"
    print(f"Downloading shard_{index:05d}...")

    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"✓ shard_{index:05d}")
            return True
            
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/5 failed: {e}")
            for path in [temp_path, filepath]:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
            
            if attempt < 5:
                wait = 2 ** attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
    
    print(f"✗ shard_{index:05d} failed after 5 attempts")
    return False

CONVERSATION_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'conversation_data')
SHARD_SIZE = 50_000  # rows per parquet shard

def download_conversation_data():
    """Download SmolTalk + UltraChat 200k and save as parquet shards.

    Both datasets use the same schema: messages column with [{content, role}] dicts.
    Combined ~1.5M conversations for midtraining.
    """
    from datasets import load_dataset, concatenate_datasets
    import pyarrow as pa

    conv_dir = os.path.abspath(CONVERSATION_DATA_DIR)
    if os.path.exists(conv_dir) and any(f.endswith('.parquet') for f in os.listdir(conv_dir)):
        existing = [f for f in os.listdir(conv_dir) if f.endswith('.parquet')]
        print(f"Conversation data already exists at {conv_dir} ({len(existing)} shards)")
        return True

    os.makedirs(conv_dir, exist_ok=True)

    print("Downloading SmolTalk (all, train split)...")
    smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    print(f"  SmolTalk: {len(smoltalk)} conversations")

    print("Downloading UltraChat 200k (train_sft + train_gen)...")
    ultrachat_sft = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultrachat_gen = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    print(f"  UltraChat: {len(ultrachat_sft)} (sft) + {len(ultrachat_gen)} (gen)")

    # Normalize: keep only 'messages' column + add 'source' for traceability
    def normalize_dataset(ds, source_name):
        ds = ds.map(lambda x: {"source": source_name}, num_proc=4)
        return ds.select_columns(["messages", "source"])

    smoltalk = normalize_dataset(smoltalk, "smoltalk")
    ultrachat_sft = normalize_dataset(ultrachat_sft, "ultrachat_sft")
    ultrachat_gen = normalize_dataset(ultrachat_gen, "ultrachat_gen")

    combined = concatenate_datasets([smoltalk, ultrachat_sft, ultrachat_gen])
    combined = combined.shuffle(seed=42)
    print(f"Combined: {len(combined)} conversations")

    # Save as parquet shards
    num_shards = (len(combined) + SHARD_SIZE - 1) // SHARD_SIZE
    for i in range(num_shards):
        start = i * SHARD_SIZE
        end = min(start + SHARD_SIZE, len(combined))
        shard = combined.select(range(start, end))
        shard_path = os.path.join(conv_dir, f"shard_{i:05d}.parquet")
        shard.to_parquet(shard_path)
        print(f"  Saved shard_{i:05d}.parquet ({end - start} rows)")

    print(f"\nDone! {num_shards} shards saved to {conv_dir}")
    return True


LIMA_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'lima_data')

def download_lima():
    """Download GAIR/lima and save as a single parquet file.

    LIMA contains ~1K high-quality human-curated conversations.
    The raw dataset uses a flat `conversations` list (alternating user/assistant strings).
    We convert to the standard `messages` format: [{"role": ..., "content": ...}]
    and add source='lima' for compatibility with PackedStreamingDataset's source_filter.

    Uses HuggingFace datasets-server parquet API to bypass the deprecated dataset
    loading script (GAIR/lima uses an old-style lima.py that datasets 3.x no longer runs).

    Output: data/lima_data/lima.parquet
    """
    import pyarrow.parquet as pq_raw
    from datasets import Dataset

    lima_dir = os.path.abspath(LIMA_DATA_DIR)
    lima_path = os.path.join(lima_dir, "lima.parquet")

    if os.path.exists(lima_path):
        print(f"LIMA already exists at {lima_path}")
        return True

    os.makedirs(lima_dir, exist_ok=True)

    # GAIR/lima is a gated dataset — requires a HuggingFace token.
    # 1. Go to https://huggingface.co/datasets/GAIR/lima and accept the terms.
    # 2. Create a token at https://huggingface.co/settings/tokens
    # 3. Set it: export HF_TOKEN=hf_...
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "GAIR/lima is a gated dataset. Set HF_TOKEN to your HuggingFace token.\n"
            "  1. Accept terms at: https://huggingface.co/datasets/GAIR/lima\n"
            "  2. Create a token at: https://huggingface.co/settings/tokens\n"
            "  3. export HF_TOKEN=hf_..."
        )
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Ask the HuggingFace datasets-server for auto-converted parquet URLs
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=GAIR/lima"
    print("Fetching LIMA parquet URLs from HuggingFace datasets server...")
    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    parquet_urls = [f["url"] for f in resp.json()["parquet_files"] if f["split"] == "train"]
    print(f"  Found {len(parquet_urls)} parquet file(s)")

    # Download each shard to a temp file, read with pyarrow, then delete
    import pyarrow as pa
    tables = []
    for url in parquet_urls:
        tmp = lima_path + ".tmp"
        print(f"  Downloading {url.split('/')[-1]}...")
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tables.append(pq_raw.read_table(tmp))
        os.unlink(tmp)

    raw = pa.concat_tables(tables)
    print(f"  LIMA: {len(raw)} conversations")

    # Convert flat alternating string list → [{role, content}] dicts
    roles = ["user", "assistant"]
    conversations_col = raw.column("conversations").to_pylist()
    messages_list = [
        [{"role": roles[i % 2], "content": text} for i, text in enumerate(turns)]
        for turns in conversations_col
    ]

    # Save using HuggingFace Dataset.to_parquet for schema consistency with SmolTalk/UltraChat
    out = Dataset.from_dict({"messages": messages_list, "source": ["lima"] * len(messages_list)})
    out.to_parquet(lima_path)
    print(f"  Saved {len(out)} conversations to {lima_path}")
    return True


MMLU_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'mmlu_data')

def download_mmlu():
    """Download cais/mmlu and save as train/test parquet files.

    auxiliary_train: ~100K examples across 57 knowledge domains (no subject field).
    all/test: ~14K examples with subject metadata — used for evaluation.

    Stored as multiple-choice conversations:
      user:      "Multiple Choice question: {question}\\n- {choice_text}=A\\n..."
      assistant: full text of the correct choice (e.g. "Paris")
    Extra columns: source='mmlu', subject (empty string for auxiliary_train).

    Output: data/mmlu_data/train.parquet, data/mmlu_data/test.parquet
    """
    from datasets import Dataset

    mmlu_dir = os.path.abspath(MMLU_DATA_DIR)
    train_path = os.path.join(mmlu_dir, "train.parquet")
    test_path  = os.path.join(mmlu_dir, "test.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"MMLU already exists at {mmlu_dir}")
        return True

    os.makedirs(mmlu_dir, exist_ok=True)

    letters = ["A", "B", "C", "D"]

    def _fetch_parquet_rows(dataset, config, split):
        """Download parquet files via datasets-server API and return list of row dicts."""
        import pyarrow as pa
        import pyarrow.parquet as pq_raw

        api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset}&config={config}&split={split}"
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        urls = [f["url"] for f in resp.json()["parquet_files"]]
        print(f"  Found {len(urls)} parquet file(s) for {config}/{split}")

        tables = []
        for url in urls:
            tmp = os.path.join(mmlu_dir, "_tmp.parquet")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tables.append(pq_raw.read_table(tmp))
            os.unlink(tmp)

        # Unify column order across shards before concat (some splits differ)
        if len(tables) > 1:
            cols = tables[0].schema.names
            tables = [t.select(cols) for t in tables]
        table = pa.concat_tables(tables)
        rows = table.to_pylist()
        print(f"  Columns: {table.schema.names}")
        # If data is nested under a single key (e.g. {"train": {...}}), flatten it
        if rows and len(table.schema.names) == 1 and isinstance(rows[0][table.schema.names[0]], dict):
            nested_key = table.schema.names[0]
            print(f"  Flattening nested key '{nested_key}': inner keys = {list(rows[0][nested_key].keys())}")
            rows = [row[nested_key] for row in rows]
        return rows

    def format_rows(rows):
        """Convert raw MMLU rows → {messages, source, subject} dicts."""
        out = []
        for ex in rows:
            # choices: list column or individual A/B/C/D columns
            if "choices" in ex:
                choices = ex["choices"]
            else:
                choices = [ex.get(l, ex.get(f"option_{l.lower()}", "")) for l in letters]
            # answer: int index or letter string → resolve to the answer text
            ans = ex.get("answer", ex.get("correct_answer", ex.get("label", 0)))
            ans_idx = ans if isinstance(ans, int) else letters.index(str(ans))
            answer_text = choices[ans_idx]
            choice_lines = "\n".join(f"- {choices[i]}={letters[i]}" for i in range(len(choices)))
            user_content = f"Multiple Choice question: {ex['question']}\n{choice_lines}"
            out.append({
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": answer_text},
                    {"role": "user",      "content": ""},  # trailing USR turn-end signal
                ],
                "source":  "mmlu",
                "subject": ex.get("subject", ""),
            })
        return out

    print("Downloading MMLU auxiliary_train (~100K)...")
    train_rows = _fetch_parquet_rows("cais%2Fmmlu", "auxiliary_train", "train")
    random.shuffle(train_rows)
    train_formatted = format_rows(train_rows)
    train_out = Dataset.from_list(train_formatted)
    train_out.to_parquet(train_path)
    print(f"  Saved {len(train_out)} examples to {train_path}")

    print("Downloading MMLU test split (~14K)...")
    test_rows = _fetch_parquet_rows("cais%2Fmmlu", "all", "test")
    test_formatted = format_rows(test_rows)
    test_out = Dataset.from_list(test_formatted)
    test_out.to_parquet(test_path)
    print(f"  Saved {len(test_out)} examples to {test_path}")

    return True


GSM8K_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'gsm8k_data')

def download_gsm8k():
    """Download openai/gsm8k and save as train/test parquet files.

    ~8K train / ~1.3K test grade-school math problems with step-by-step
    chain-of-thought solutions. Answers follow the '#### <number>' marker.

    Stored as conversations:
      user:      math word problem
      assistant: full step-by-step solution ending in '#### <answer>'

    Output: data/gsm8k_data/train.parquet, data/gsm8k_data/test.parquet
    """
    from datasets import load_dataset

    gsm8k_dir = os.path.abspath(GSM8K_DATA_DIR)
    train_path = os.path.join(gsm8k_dir, "train.parquet")
    test_path  = os.path.join(gsm8k_dir, "test.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"GSM8K already exists at {gsm8k_dir}")
        return True

    os.makedirs(gsm8k_dir, exist_ok=True)

    def format_example(example):
        return {
            "messages": [
                {"role": "user",      "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
                {"role": "user",      "content": ""},  # trailing USR turn-end signal
            ],
            "source": "gsm8k",
        }

    print("Downloading GSM8K...")
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    test_ds  = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  GSM8K: {len(train_ds)} train, {len(test_ds)} test")

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(format_example,  remove_columns=test_ds.column_names)

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f"  Saved to {gsm8k_dir}")

    return True


CUSTOM_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'custom_data')
TOPIC_SWITCH_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'topic_switch_data')

# p_sample weights for each source in the topic-switch dataset.
# Uniform by default; adjust to oversample/undersample specific sources.
TOPIC_SWITCH_P_SAMPLE = {'mmlu': 1/3, 'gsm8k': 1/3, 'smoltalk': 1/3}


def _ts_load_conversations(paths, source_filter=None):
    """Load messages lists from parquet files, optionally filtering by 'source' column."""
    out = []
    for path in paths:
        pf = pq.ParquetFile(path)
        cols = ['messages', 'source'] if source_filter else ['messages']
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=cols)
            msgs_list = table['messages'].to_pylist()
            if source_filter:
                src_list = table['source'].to_pylist()
                for msgs, src in zip(msgs_list, src_list):
                    if src == source_filter:
                        out.append(msgs)
            else:
                out.extend(msgs_list)
    return out


def _ts_strip_trailing_empty_user(messages):
    """Remove the trailing empty user turn used as a turn-end signal in MMLU/GSM8K."""
    if messages and messages[-1]['role'] == 'user' and messages[-1]['content'] == '':
        return messages[:-1]
    return messages


def _ts_get_ua_pairs(messages):
    """Extract (user_msg, assistant_msg) pairs from a conversation, skipping system turns."""
    turns = [m for m in messages if m['role'] != 'system']
    pairs = []
    for i in range(0, len(turns) - 1, 2):
        if turns[i]['role'] == 'user' and turns[i + 1]['role'] == 'assistant':
            pairs.append((turns[i], turns[i + 1]))
    return pairs


def _ts_estimate_tokens(messages):
    """Rough token estimate: ~4 chars/token + 1 role token per message + 2 (BOS + EOS)."""
    return sum(max(1, len(m['content']) // 4) + 1 for m in messages) + 2


def _ts_flush_shard(rows, output_dir, shard_idx):
    from datasets import Dataset
    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
    Dataset.from_list(rows).to_parquet(path)
    print(f"  Saved shard_{shard_idx:05d}.parquet ({len(rows)} rows)")


def generate_topic_switch_dataset(
    num_conversations=100_000,
    p_resume=0.5,
    p_sample=None,
    p_system_prompt=0.5,
    seq_length=2048,
    seed=42,
):
    """Generate a topic-switching SFT dataset mixing MMLU, GSM8K, and SmolTalk.

    Each generated conversation is drawn from a single source (selected by p_sample).
    Turns come from different conversations within that source to teach topic switching.
    p_resume is the probability of continuing with the next turn from the current
    source conversation; (1 - p_resume) causes a switch to a new one.

    System prompt is either "You are a helpful assistant." or absent, chosen randomly
    with probability p_system_prompt per conversation.

    Output: data/topic_switch_data/shard_*.parquet
    Schema: messages (list of {role, content}), source='topic_switch'
    Compatible with PackedStreamingDataset(data_format='conversation').
    """
    random.seed(seed)

    if p_sample is None:
        p_sample = TOPIC_SWITCH_P_SAMPLE.copy()

    sources = list(p_sample.keys())
    total_weight = sum(p_sample.values())
    weights = [p_sample[s] / total_weight for s in sources]

    # --- Load source pools ---
    print("Loading source conversation pools...")
    source_pools = {}

    if 'mmlu' in sources:
        raw = _ts_load_conversations([os.path.join(MMLU_DATA_DIR, 'train.parquet')])
        source_pools['mmlu'] = [_ts_strip_trailing_empty_user(m) for m in raw if m]
        print(f"  MMLU: {len(source_pools['mmlu'])} conversations")

    if 'gsm8k' in sources:
        raw = _ts_load_conversations([os.path.join(GSM8K_DATA_DIR, 'train.parquet')])
        source_pools['gsm8k'] = [_ts_strip_trailing_empty_user(m) for m in raw if m]
        print(f"  GSM8K: {len(source_pools['gsm8k'])} conversations")

    if 'smoltalk' in sources:
        conv_paths = sorted(
            os.path.join(CONVERSATION_DATA_DIR, f)
            for f in os.listdir(CONVERSATION_DATA_DIR)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        )
        raw = _ts_load_conversations(conv_paths, source_filter='smoltalk')
        source_pools['smoltalk'] = raw
        print(f"  SmolTalk: {len(source_pools['smoltalk'])} conversations")

    output_dir = os.path.abspath(TOPIC_SWITCH_DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_conversations} conversations → {output_dir}")

    rows = []
    shard_idx = 0

    for conv_i in range(num_conversations):
        source = random.choices(sources, weights=weights, k=1)[0]
        pool = source_pools[source]

        messages = []
        if random.random() < p_system_prompt:
            messages.append({'role': 'system', 'content': 'You are a helpful assistant.'})

        # Seed the first source conversation
        curr_conv = random.choice(pool)
        curr_pairs = _ts_get_ua_pairs(curr_conv)
        pair_idx = 0

        while True:
            # Replenish if current conversation is exhausted
            while pair_idx >= len(curr_pairs):
                curr_conv = random.choice(pool)
                curr_pairs = _ts_get_ua_pairs(curr_conv)
                pair_idx = 0

            user_msg, asst_msg = curr_pairs[pair_idx]
            pair_idx += 1

            # Stop if next pair would exceed the token budget
            if _ts_estimate_tokens(messages + [user_msg, asst_msg]) > seq_length:
                break

            messages.append(user_msg)
            messages.append(asst_msg)

            # p_resume: continue with next pair, or switch to a new conversation
            if random.random() >= p_resume:
                curr_conv = random.choice(pool)
                curr_pairs = _ts_get_ua_pairs(curr_conv)
                pair_idx = 0

        # Guard: skip if no turns were generated (can only happen if a single pair > budget)
        if not any(m['role'] == 'assistant' for m in messages):
            continue

        rows.append({'messages': messages, 'source': 'topic_switch'})

        if len(rows) >= SHARD_SIZE:
            _ts_flush_shard(rows, output_dir, shard_idx)
            shard_idx += 1
            rows = []

        if (conv_i + 1) % 10_000 == 0:
            print(f"  {conv_i + 1}/{num_conversations} generated")

    if rows:
        _ts_flush_shard(rows, output_dir, shard_idx)

    print(f"Done! {num_conversations} conversations saved to {output_dir}")

def ingest_custom(json_path, source_name='custom'):
    """Convert a JSON conversation file to parquet for SFT training.

    Input format — a JSON array of conversation objects:

        [
          {
            "system": "Optional system prompt.",   ← optional
            "messages": [
              ["u", "User turn text"],
              ["a", "Assistant turn text"],
              ...
            ]
          },
          ...
        ]

    Role codes: "u" = user, "a" = assistant.
    Rules enforced:
      - messages must be a non-empty list of ["u"|"a", text] pairs
      - first message must be "u" (user)
      - last message must be "a" (assistant)
      - roles must strictly alternate u/a
    Output: data/custom_data/{source_name}.parquet
    """
    import json
    from datasets import Dataset

    ROLE_MAP = {'u': 'user', 'a': 'assistant'}

    custom_dir = os.path.abspath(CUSTOM_DATA_DIR)
    os.makedirs(custom_dir, exist_ok=True)
    out_path = os.path.join(custom_dir, f"{source_name}.parquet")

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array of conversation objects.")

    rows, skipped = [], 0
    for i, conv in enumerate(data, 1):
        tag = f"[conv {i}]"

        sys_content = conv.get('system', '').strip()
        raw_msgs = conv.get('messages', [])

        if not raw_msgs:
            print(f"  {tag} empty messages — skipped")
            skipped += 1; continue

        # Validate and expand
        ok = True
        messages = []
        if sys_content:
            messages.append({"role": "system", "content": sys_content})

        prev_role = None
        for j, item in enumerate(raw_msgs):
            if not (isinstance(item, list) and len(item) == 2 and item[0] in ROLE_MAP):
                print(f"  {tag} message {j+1}: expected [\"u\"|\"a\", text] — skipped")
                ok = False; break
            role = ROLE_MAP[item[0]]
            if prev_role == role:
                print(f"  {tag} message {j+1}: consecutive '{role}' turns — skipped")
                ok = False; break
            messages.append({"role": role, "content": str(item[1])})
            prev_role = role

        if not ok:
            skipped += 1; continue

        user_asst = [m for m in messages if m['role'] != 'system']
        if not user_asst or user_asst[0]['role'] != 'user':
            print(f"  {tag} first non-system message must be 'u' — skipped")
            skipped += 1; continue
        if user_asst[-1]['role'] != 'assistant':
            print(f"  {tag} last message must be 'a' — skipped")
            skipped += 1; continue

        rows.append({"messages": messages, "source": source_name})

    if not rows:
        raise ValueError(f"No valid conversations found in {json_path}")

    Dataset.from_list(rows).to_parquet(out_path)
    print(f"  Saved {len(rows)} conversations ({skipped} skipped) → {out_path}")
    return True


HUMANEVAL_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'humaneval_data')

def download_humaneval():
    """Download openai/openai_humaneval and save as a test parquet file.

    164 Python function-completion problems (eval only — no train split).
    The prompt contains the function signature and docstring.
    The canonical solution is the complete function body.

    Stored as conversations:
      user:      function prompt (signature + docstring)
      assistant: prompt + canonical_solution (full runnable function)
    Extra columns: source='humaneval', task_id, entry_point, test (assert block).

    Output: data/humaneval_data/test.parquet
    """
    from datasets import load_dataset

    humaneval_dir = os.path.abspath(HUMANEVAL_DATA_DIR)
    test_path = os.path.join(humaneval_dir, "test.parquet")

    if os.path.exists(test_path):
        print(f"HumanEval already exists at {humaneval_dir}")
        return True

    os.makedirs(humaneval_dir, exist_ok=True)

    def format_example(example):
        # Assistant content = full function (prompt already contains the def line)
        return {
            "messages": [
                {"role": "user",      "content": example["prompt"]},
                {"role": "assistant", "content": example["prompt"] + example["canonical_solution"]},
            ],
            "source":      "humaneval",
            "task_id":     example["task_id"],
            "entry_point": example["entry_point"],
            "test":        example["test"],
        }

    print("Downloading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"  HumanEval: {len(ds)} problems")

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds.to_parquet(test_path)
    print(f"  Saved to {test_path}")

    return True


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'eval_data')

def download_eval_bundle():
    """Download and extract the CORE eval bundle from S3."""
    eval_dir = os.path.abspath(EVAL_DIR)
    if os.path.exists(eval_dir):
        print(f"Eval bundle already exists at {eval_dir}")
        return True

    import zipfile
    import tempfile
    import shutil

    print(f"Downloading eval bundle from {EVAL_BUNDLE_URL}...")
    for attempt in range(1, 4):
        try:
            response = requests.get(EVAL_BUNDLE_URL, stream=True, timeout=60)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                tmp_path = tmp.name
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        tmp.write(chunk)

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(tmp_path, 'r') as zf:
                    zf.extractall(tmpdir)
                extracted = os.path.join(tmpdir, "eval_bundle")
                shutil.move(extracted, eval_dir)

            os.unlink(tmp_path)
            print(f"Eval bundle extracted to {eval_dir}")
            return True

        except (requests.RequestException, IOError, zipfile.BadZipFile) as e:
            print(f"Attempt {attempt}/3 failed: {e}")
            for p in [tmp_path]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if attempt < 3:
                wait = 2 ** attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    print("Failed to download eval bundle")
    return False

ULTRAFEEDBACK_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data', 'ultrafeedback_data')
HH_RLHF_DATA_DIR       = os.path.join(_SCRIPT_DIR, '..', 'data', 'hh_rlhf_data')


def download_ultrafeedback():
    """Download HuggingFaceH4/ultrafeedback_binarized and save as DPO parquet shards.

    Schema: chosen (list[{role,content}]), rejected (list[{role,content}]), source='ultrafeedback'
    Split used: train_prefs (~61K examples). chosen/rejected are already in messages format.

    Output: data/ultrafeedback_data/shard_*.parquet
    """
    from datasets import load_dataset
    import pyarrow as pa

    out_dir = os.path.abspath(ULTRAFEEDBACK_DATA_DIR)
    if os.path.exists(out_dir) and any(f.endswith('.parquet') for f in os.listdir(out_dir)):
        existing = [f for f in os.listdir(out_dir) if f.endswith('.parquet')]
        print(f"UltraFeedback already exists at {out_dir} ({len(existing)} shards)")
        return True

    os.makedirs(out_dir, exist_ok=True)

    print("Downloading HuggingFaceH4/ultrafeedback_binarized (train_prefs)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    print(f"  {len(ds)} preference pairs")

    # Keep only chosen, rejected; add source tag
    def normalize(row):
        return {
            "chosen":   row["chosen"],
            "rejected": row["rejected"],
            "source":   "ultrafeedback",
        }

    ds = ds.map(normalize, remove_columns=ds.column_names, num_proc=4)

    num_shards = (len(ds) + SHARD_SIZE - 1) // SHARD_SIZE
    for i in range(num_shards):
        start = i * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(ds))
        shard = ds.select(range(start, end))
        shard_path = os.path.join(out_dir, f"shard_{i:05d}.parquet")
        shard.to_parquet(shard_path)
        print(f"  Saved shard_{i:05d}.parquet ({end - start} rows)")

    print(f"\nDone! {num_shards} shards saved to {out_dir}")
    return True


def _parse_hh_rlhf_text(text):
    """Parse HH-RLHF raw text into a messages list [{role, content}]."""
    role_map = {"Human": "user", "Assistant": "assistant"}
    matches = re.findall(
        r'\n\n(Human|Assistant): (.*?)(?=\n\n(?:Human|Assistant): |$)',
        text,
        re.DOTALL,
    )
    return [{"role": role_map[r], "content": c.strip()} for r, c in matches if c.strip()]


def download_hh_rlhf():
    """Download Anthropic/hh-rlhf (helpful + harmless splits) and save as DPO parquet shards.

    Schema: chosen (list[{role,content}]), rejected (list[{role,content}]), source='hh_rlhf'
    Parses the raw Human/Assistant text format into messages lists.
    Splits used: helpful-base train + harmless-base train (~86K total).

    Output: data/hh_rlhf_data/shard_*.parquet
    """
    from datasets import load_dataset, concatenate_datasets
    import pyarrow as pa

    out_dir = os.path.abspath(HH_RLHF_DATA_DIR)
    if os.path.exists(out_dir) and any(f.endswith('.parquet') for f in os.listdir(out_dir)):
        existing = [f for f in os.listdir(out_dir) if f.endswith('.parquet')]
        print(f"HH-RLHF already exists at {out_dir} ({len(existing)} shards)")
        return True

    os.makedirs(out_dir, exist_ok=True)

    print("Downloading Anthropic/hh-rlhf (helpful-base + harmless-base, train)...")
    helpful  = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base",  split="train")
    harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    print(f"  helpful-base: {len(helpful)}  harmless-base: {len(harmless)}")

    combined = concatenate_datasets([helpful, harmless])

    def normalize(row):
        chosen_msgs   = _parse_hh_rlhf_text(row["chosen"])
        rejected_msgs = _parse_hh_rlhf_text(row["rejected"])
        # Skip malformed rows (empty parse)
        if not chosen_msgs or not rejected_msgs:
            return {"chosen": None, "rejected": None, "source": "hh_rlhf"}
        return {
            "chosen":   chosen_msgs,
            "rejected": rejected_msgs,
            "source":   "hh_rlhf",
        }

    combined = combined.map(normalize, remove_columns=combined.column_names, num_proc=4)
    # Drop rows where parse failed
    combined = combined.filter(lambda x: x["chosen"] is not None, num_proc=4)
    print(f"  {len(combined)} valid pairs after parsing")

    num_shards = (len(combined) + SHARD_SIZE - 1) // SHARD_SIZE
    for i in range(num_shards):
        start = i * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(combined))
        shard = combined.select(range(start, end))
        shard_path = os.path.join(out_dir, f"shard_{i:05d}.parquet")
        shard.to_parquet(shard_path)
        print(f"  Saved shard_{i:05d}.parquet ({end - start} rows)")

    print(f"\nDone! {num_shards} shards saved to {out_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu shards and/or eval data")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
                        help="Number of shards (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--eval", action="store_true",
                        help="Download CORE eval bundle")
    parser.add_argument("--conversation", action="store_true",
                        help="Download SmolTalk + UltraChat conversation data for midtraining")
    parser.add_argument("--lima", action="store_true",
                        help="Download GAIR/lima high-quality SFT data (~1K conversations)")
    parser.add_argument("--mmlu", action="store_true",
                        help="Download MMLU auxiliary_train + test (~100K train, ~14K test)")
    parser.add_argument("--gsm8k", action="store_true",
                        help="Download GSM8K math reasoning dataset (~8K train, ~1.3K test)")
    parser.add_argument("--humaneval", action="store_true",
                        help="Download HumanEval coding benchmark (164 problems, eval only)")
    parser.add_argument("--custom", metavar="JSON_PATH",
                        help="Ingest a JSON conversation file into data/custom_data/")
    parser.add_argument("--source", default="custom",
                        help="Source name tag written to the 'source' column (default: 'custom')")
    parser.add_argument("--topic-switch", action="store_true",
                        help="Generate topic-switching SFT dataset (MMLU + GSM8K + SmolTalk, 100K conversations)")
    parser.add_argument("--ultrafeedback", action="store_true",
                        help="Download UltraFeedback binarized DPO pairs (~61K chosen/rejected pairs)")
    parser.add_argument("--hh-rlhf", action="store_true",
                        help="Download Anthropic HH-RLHF DPO pairs (~86K helpful+harmless pairs)")
    args = parser.parse_args()

    if args.topic_switch:
        generate_topic_switch_dataset()
    elif args.ultrafeedback:
        download_ultrafeedback()
    elif args.hh_rlhf:
        download_hh_rlhf()
    elif args.custom:
        ingest_custom(args.custom, source_name=args.source)
    elif args.lima:
        download_lima()
    elif args.mmlu:
        download_mmlu()
    elif args.gsm8k:
        download_gsm8k()
    elif args.humaneval:
        download_humaneval()
    elif args.conversation:
        download_conversation_data()
    elif args.eval:
        download_eval_bundle()
    else:
        num = (MAX_SHARD + 1) if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
        print(f"Downloading {num} shards to {DATA_DIR} ({args.num_workers} workers)\n")

        with Pool(args.num_workers) as pool:
            results = pool.map(download_single_file, range(num))

        print(f"\nDone! {sum(results)}/{num} shards downloaded")