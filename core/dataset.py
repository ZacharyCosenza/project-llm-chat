"""
Utilities for FineWeb-Edu dataset: iterate parquet files, download on demand.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, 
                        help="Number of shards (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Parallel workers")
    args = parser.parse_args()

    num = (MAX_SHARD + 1) if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    print(f"Downloading {num} shards to {DATA_DIR} ({args.num_workers} workers)\n")
    
    with Pool(args.num_workers) as pool:
        results = pool.map(download_single_file, range(num))
    
    print(f"\nDone! {sum(results)}/{num} shards downloaded")