"""
Utilities for dataset download: FineWeb-Edu (pretraining), SmolTalk+UltraChat (midtraining), eval bundle.
See `repackage_data_reference.py` for dataset preparation details.
"""

import os
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
      user:      "Multiple Choice question: {question}\\n- {A}=A\\n..."
      assistant: "A"  (single letter)
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
    args = parser.parse_args()

    if args.lima:
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