"""
Utilities for dataset download: FineWeb-Edu (pretraining), SmolTalk+UltraChat (midtraining), eval bundle.
See `repackage_data_reference.py` for dataset preparation details.
"""

import os
import time
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
    args = parser.parse_args()

    if args.conversation:
        download_conversation_data()
    elif args.eval:
        download_eval_bundle()
    else:
        num = (MAX_SHARD + 1) if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
        print(f"Downloading {num} shards to {DATA_DIR} ({args.num_workers} workers)\n")

        with Pool(args.num_workers) as pool:
            results = pool.map(download_single_file, range(num))

        print(f"\nDone! {sum(results)}/{num} shards downloaded")