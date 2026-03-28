# ZAC-GPT-2: A ~500M Parameter LLM From Scratch

A full multi-stage LLM training pipeline — pre-training through online RL — implemented in pure PyTorch with DDP multi-GPU support. Built as a learning project to understand large-scale training, preference optimization, and RL from human feedback at a practical scale.

**Training stages**: Pre-train → Mid-train → SFT → DPO → Code RL
**Model**: ~500M parameter GPT-2 style transformer
**Target hardware**: 4× H100/H200 (80 GB) on a single node

---

## Quick Start

```bash
# 1. Environment setup (Ubuntu cloud GPU pod, CUDA 12.4)
bash startup.sh

# 2. Download datasets
python -m core.dataset -n 200 -w 4       # FineWeb-Edu (pre-training)
python -m core.dataset --conversation     # SmolTalk + UltraChat (mid-train/SFT)
python -m core.dataset --lima             # LIMA (SFT)
python -m core.dataset --mmlu             # MMLU (SFT benchmark)
python -m core.dataset --gsm8k            # GSM8K (SFT math)
python -m core.dataset --ultrafeedback    # UltraFeedback (DPO)
python -m core.dataset --hh-rlhf          # Anthropic HH-RLHF (DPO)
python -m core.dataset --humaneval        # HumanEval (code RL)
python -m core.dataset --mbpp             # MBPP (code RL)
python -m core.dataset --calculator       # Calculator reasoning (code RL)
python -m core.dataset --eval             # CORE benchmark eval bundle

# 3. Train (example: 4 GPUs, DPO mode)
./run_pytorch.sh 4

# 4. Chat with the model
python -m core.chat
```

---

## 1. Datasets and Loss Functions

Each training stage uses a different dataset mix and loss function. The progression is designed so that later stages refine the behavior learned in earlier ones without catastrophically forgetting.

### Datasets by Stage

| Stage | Datasets | Mix |
|---|---|---|
| Pre-train | FineWeb-Edu (~10.8B tokens) | 100% |
| Mid-train | FineWeb-Edu + SmolTalk + UltraChat-Gen + UltraChat-SFT | 30/48/12/10 |
| SFT | SmolTalk + UltraChat-Gen + UltraChat-SFT | 65/20/15 |
| SFT-LIMA | GAIR/LIMA (~1K curated conversations) | 100% |
| SFT-benchmark | MMLU + GSM8K | 97/3 |
| DPO | UltraFeedback + HH-RLHF | 60/40 |
| Code RL | Calculator + HumanEval + MBPP | 33/33/33 |

**FineWeb-Edu** (~170 parquet shards, ~56M tokens each): high-quality web text filtered for educational content. Used raw in pre-training; blended in mid-training to prevent forgetting.

**SmolTalk / UltraChat**: multi-turn conversation datasets. SmolTalk is denser and more curated; UltraChat is larger and more varied. Both use the `messages` column format with `role`/`content` pairs.

**LIMA**: 1,000 hand-curated high-quality conversations. Used for a short fine-tuning pass after main SFT to sharpen instruction-following.

**MMLU / GSM8K**: structured benchmark formats. MMLU is multiple-choice questions across 57 subjects. GSM8K is grade-school math with chain-of-thought solutions. Both include a trailing `{"role": "user", "content": ""}` message so `tokenize_conversation` emits the `<|user|>` turn-end token after the answer — making single-turn examples structurally identical to multi-turn conversation interiors.

**UltraFeedback / HH-RLHF**: preference pair datasets. Each example has a `chosen` response (preferred by human raters) and a `rejected` response (dispreferred). ~61K and ~86K pairs respectively. Preferred responses trend ~50–120 characters longer on average — a consistent signal across 147K examples is enough to shift generation behavior.

**HumanEval / MBPP**: Python coding benchmarks. HumanEval has 164 problems with function signatures and docstrings; MBPP has ~870 problems with plain text descriptions and test assertions. Both include test harnesses used at reward time.

**CalculatorDataset**: infinite programmatic generator in `core/dataset.py`. Produces three problem types sampled uniformly: (1) arithmetic chains (+, -, ×, ÷, **, √), (2) statistics (mean/std/median/variance), (3) unit conversions (miles↔km, °C↔°F, kg↔lb). Ground truth is computed by trusted local `eval()`. Train and val sets use different RNG seeds so problems never overlap.

### Packing Strategies

How tokens are arranged into fixed-length chunks affects what the loss function sees.

**Linear stream-packing** (`mask_policy='all'`, pre-train and mid-train): tokens from consecutive documents flow into a deque; 2049-token chunks are popped regardless of document boundaries. No padding, no alignment — maximum token utilization. The model trains on every token including across document boundaries, which is fine at this stage because the goal is next-token prediction, not conversation structure.

**BOS-aligned packing** (`mask_policy='assistant_only'`, SFT): each chunk must start at a conversation boundary. When the next conversation won't fit in the remaining space, the chunk is padded with `<|pad|>` (loss_mask=0) and yielded. Conversations longer than `seq_length` are truncated.

```
Pre-train:  [tok1 tok2 | tok3 tok4 tok5 | tok6 tok7]  ← boundaries ignored
SFT:        [CONV_A (padded) ...................]
            [CONV_B | CONV_C (padded) ..........]      ← always starts at conversation start
```

This matters for SFT because the loss is only computed on assistant tokens — if a conversation starts mid-chunk, the model may be trained on assistant tokens without their preceding user context.

**No packing** (Code RL): each item is a single prompt → generation pair. Lengths vary per item; no need to pack.

### Loss Functions

**Causal LM loss** (pre-train and mid-train):

```
L = CrossEntropy(logits[:-1], tokens[1:])
```

Every token predicts the next. No masking. `compute_loss` normalizes by `loss_mask.sum().clamp(min=1)` to handle chunks that are all padding.

**Masked SFT loss** (SFT modes):

The loss is identical in form but `loss_mask=1` only for:
- The `<|assistant|>` role token at the start of each assistant turn
- All assistant content tokens
- The `<|user|>` turn-end token immediately after an assistant turn
- EOS after the final assistant turn

Everything else — BOS, system prompt, user messages — has `loss_mask=0` and contributes zero to the gradient. This teaches the model to generate assistant responses without memorizing user inputs or system prompts.

**DPO loss** (Rafailov et al. 2023):

```
L_DPO = -logsigmoid( β × [log π_θ(y_w|x) - log π_ref(y_w|x)]
                    - β × [log π_θ(y_l|x) - log π_ref(y_l|x)] )
```

Where `y_w` = chosen response, `y_l` = rejected response, `β=0.1` controls divergence from the reference. The loss pushes the policy to assign higher log-probability to chosen responses and lower log-probability to rejected responses, measured relative to a frozen reference model. The reference model is the SFT checkpoint before any DPO updates.

Each example is a chosen/rejected pair, so the effective forward batch per GPU is `[2B, T]`. With `batch_size=9` per GPU, this is equivalent to `batch_size=18` in SFT for memory purposes.

`get_sequence_logprobs` computes the per-sequence log-probability sum for each item in the batch. It processes one sequence at a time to keep peak memory at `[T-1, V]` rather than `[B, T-1, V]`:

```python
def get_sequence_logprobs(logits, token_ids, loss_mask):
    shift_logits = logits[:, :-1, :]
    shift_ids    = token_ids[:, 1:]
    shift_mask   = loss_mask[:, 1:].float()
    logps = []
    for i in range(shift_logits.size(0)):
        nll_i = F.cross_entropy(shift_logits[i], shift_ids[i], reduction='none')
        logps.append((-nll_i.float() * shift_mask[i]).sum())
    return torch.stack(logps)
```

**REINFORCE loss** (Code RL):

```
L = (per_token_CE × loss_mask × discounted_returns).sum() / loss_mask.sum()
```

`discounted_returns[t] = γ^(n_tokens-1-t) × advantage` (γ=0.99). The last generated token receives the full advantage; earlier tokens receive slightly less. `advantages = rewards - rewards.mean()` (batch baseline subtraction to reduce variance).

### Reward Functions (Code RL)

Rewards come from executing the model's generated Python code. Partial credit is given to encourage the model to produce executable code even before it produces correct answers.

| Outcome | Calculator | Code task (HumanEval/MBPP) |
|---|---|---|
| Code block executes | +0.3 | +0.3 |
| Numeric output produced | +0.2 | — |
| Correct answer | +0.5 | — |
| All assertions pass | — | +0.7 |

For the calculator task, the model is expected to output a ` ```python ``` ` block whose `stdout` matches the expected numeric answer. For code tasks, the model's function body is assembled with a test harness (`test_prefix + code + test_suffix`) and executed.

If no code block is found in a calculator response, `PythonExecutor` falls back to parsing the last number from the plain-text response — a small reward signal that encourages numeric output even without code formatting.

### Validation Splits and Metrics

How train/val splits are constructed varies by dataset type:

- **Pretrain/SFT**: first shard → validation, remaining shards → training
- **DPO**: row-level split per shard — first `(1 - val_fraction) × N` rows go to train, last `val_fraction × N` to val; `val_fraction=0.1` across all shards gives ~10% of total pairs for validation
- **HumanEval**: 20% index holdout (no official train split exists)
- **MBPP**: official HuggingFace train/test splits
- **Calculator**: different RNG seeds for train and val — problems never overlap

Validation runs every `eval_interval` steps. Rank 0 runs all evals; metric tensors from training are averaged across ranks via `reduce_tensor` before logging.

| Metric | Description |
|---|---|
| `val/{name}/loss` | Per-dataset validation loss |
| `val/{name}/perplexity` | `exp(loss)` |
| `core/metric` | CORE benchmark (mean centered accuracy) |
| `train/dpo_loss` | DPO negative log-sigmoid of reward margin |
| `train/reward_accuracy` | DPO fraction where chosen reward > rejected reward |
| `train/rl_loss` | Code RL REINFORCE loss |
| `train/mean_reward` | Code RL mean execution reward per batch |
| `train/exec_rate` | Code RL fraction of generated code that executes |

**CORE Benchmark** (DCLM paper): 500 examples across three task types:
- *Multiple choice*: same context, different continuations → lowest mean loss wins
- *Schema*: different contexts, same continuation → lowest mean loss wins
- *Language modeling*: argmax predictions must exactly match the continuation

Score = mean centered accuracy: `(acc - random_baseline) / (1 - random_baseline)`. Positive = above chance.

---

## 2. Model and Optimizer

### Architecture: TinyGPT (`core/models.py`)

A GPT-2 style decoder implemented using PyTorch's `TransformerEncoderLayer` with a causal mask applied manually. Pre-norm architecture (`norm_first=True`).

```
Input → tok_emb + pos_emb
      → N × TransformerEncoderLayer (d_ff = dim × 4, norm_first=True)
      → LayerNorm
      → Linear head (dim → vocab_size)
```

| Hyperparameter | Value | Formula |
|---|---|---|
| Vocab size | 50,262 | GPT-2 base (50,257) + 5 special tokens |
| `dim` | 1,280 | `n_layers × 64` |
| `n_layers` | 20 | |
| `n_heads` | 10 | `max(1, (dim+127) // 128)` |
| `d_ff` | 5,120 | `dim × 4` |
| `max_seq_len` | 2,048 | |
| Parameters | ~500M | |

**Special tokens** (always-on, added to GPT-2 tokenizer):

| Token | Role |
|---|---|
| `<\|beginoftext\|>` | Starts every sequence (BOS) |
| `<\|pad\|>` | Padding in batches |
| `<\|user\|>` | User turn marker; also emitted by the model as a turn-end signal after each assistant turn |
| `<\|assistant\|>` | Assistant turn marker |
| `<\|system\|>` | System prompt marker |

**FLOP estimate** (used for budget planning):
```
FLOPs/token = 6 × (params − embedding_params) + 12 × L × heads × head_dim × T
```

**Generation**: `model.predict(idx, max_new_tokens, temperature, top_k)` — autoregressive with temperature scaling and top-k filtering. Sliding window crops to `max_seq_len` when context grows long.

### DPOModel: Hidden Reference

DPO requires a frozen reference copy of the model. It must be invisible to DDP (so gradients don't try to synchronize it across ranks) and invisible to the optimizer (so it doesn't get updated).

```python
class DPOModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.policy = TinyGPT(...)          # registered — DDP + optimizer see this
        object.__setattr__(self, '_ref',    # NOT registered — invisible to DDP/optimizer
            TinyGPT(...))
        for p in self._ref.parameters():
            p.requires_grad_(False)
```

`object.__setattr__` bypasses `nn.Module.__setattr__`, which would normally register the child as a submodule. Because `_ref` is a plain Python attribute, DDP's `named_parameters()` traversal never finds it. The `.to()` method is overridden manually to move `_ref` to the correct device, since `nn.Module.to()` only moves registered submodules.

Initialization order matters:
1. Build `DPOModel` (allocates both `policy` and `_ref`)
2. Load SFT checkpoint into `policy`
3. Resize embeddings on `policy`
4. `init_reference_from_policy()` — deep-copy `policy` → `_ref` before any DPO updates
5. Wrap with DDP

### Optimizer

AdamW with two parameter groups:

```python
no_decay = ["bias", "ln", "layernorm", "layer_norm"]
# group 1: weight matrices — weight_decay applied
# group 2: biases + LayerNorm params — no weight decay (standard practice)
optimizer = torch.optim.AdamW(
    [{"params": decay_params,    "weight_decay": weight_decay},
     {"params": no_decay_params, "weight_decay": 0.0}],
    lr=peak_lr, betas=(0.9, 0.95), eps=1e-8
)
```

Gradient clipping at 1.0 applied before every optimizer step via `torch.nn.utils.clip_grad_norm_`.

**Gradient accumulation**:

```python
accumulate_grad_batches = max(1, 4 // world_size)
```

With 4 GPUs: `4 // 4 = 1` — no accumulation, each step is a real optimizer step. With 2 GPUs: `4 // 2 = 2` — accumulate over 2 batches before stepping. This keeps the effective global batch size constant regardless of how many GPUs are available. Gradients are divided by `accumulate_grad_batches` before accumulation so the scale is equivalent to a single step with the full batch.

### LR Schedule

Linear warmup → cosine decay to `min_lr`:

```
LinearLR(start_factor=1e-2, ..., total_iters=warmup_steps)
    then
CosineAnnealingLR(T_max=remaining_steps - warmup_steps, eta_min=min_lr)
```

**Resume-aware warmup**: warmup and cosine are computed over `remaining_steps = max_steps - global_step`, not total `max_steps`. If resuming from step 5,000 with `max_steps=10,000`, the scheduler runs over 5,000 remaining steps with warmup proportional to that remainder. This prevents re-running a warmup that already happened.

| Stage | Peak LR | Min LR | Weight Decay | Warmup |
|---|---|---|---|---|
| Pre-train | 6e-4 | 6e-5 | 0.1 | 1% |
| Mid-train | 3e-4 | 3e-5 | 0.1 | 0.5% |
| SFT | 2e-5 | 2e-6 | 0.01 | 5% |
| SFT-LIMA | 1e-5 | 2e-6 | 0.01 | 5% |
| SFT-benchmark | 5e-6 | 1e-6 | 0.01 | 5% |
| DPO | 1e-6 | 1e-7 | 0.0 | 10% |
| Code RL | 5e-6 | 5e-7 | 0.01 | 5% |

DPO uses zero weight decay — the policy should stay close to the reference, and weight decay would pull weights toward zero rather than toward the SFT prior.

---

## 3. Multi-GPU Setup

This was the hardest part of the project. Cloud GPU pods (RunPod, Lambda) require specific NCCL configuration to prevent silent hangs and crashes.

### The Problem

NCCL's default communication strategy assumes high-bandwidth interconnects: NVLink for intra-node GPU-to-GPU transfers, and InfiniBand for inter-node transfers. Cloud instances typically have neither. Without explicit configuration, NCCL probes for these hardware features at startup and either hangs indefinitely at the first `all_reduce` or fails silently — the processes appear to run but no gradients are synchronized.

### NCCL Flags (`run_pytorch.sh`)

All flags are set before `torchrun` in `run_pytorch.sh`:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node="$N_GPUS" run_13_code.py [args]
```

**`NCCL_P2P_DISABLE=1`**
Disables peer-to-peer GPU memory access (direct GPU-to-GPU copies without going through the CPU). P2P requires NVLink; on cloud instances without it, NCCL attempts the transfer anyway and hangs waiting for hardware that isn't there. This is typically the flag that causes the training process to freeze completely at the very first gradient synchronization.

**`NCCL_IB_DISABLE=1`**
Disables InfiniBand transport. InfiniBand is a high-speed network interconnect used in HPC clusters. Cloud VMs don't have IB hardware, but NCCL probes for it via the `ibverbs` library at initialization. Without this flag, NCCL may print confusing warnings, select IB as the preferred transport, and then fail when the first actual transfer is attempted.

**`NCCL_SOCKET_IFNAME=lo`**
Forces NCCL to use the loopback network interface (`lo`, i.e. `127.0.0.1`) for socket-based communication. For single-node multi-GPU training, all processes are on the same machine and loopback is the correct interface — it avoids NCCL picking the wrong external NIC or a virtual interface that might have packet filtering or routing issues.

**`NCCL_NET=Socket`**
Forces NCCL to use plain TCP socket transport rather than its default selection logic. Without this, NCCL may select a faster-looking but misconfigured transport (IB, shared memory with P2P, a network plugin). Socket is the most universally compatible option and has negligible overhead for single-node training where compute is the bottleneck.

**`NCCL_NET_PLUGIN=none`**
Disables NCCL network plugins (e.g. aws-ofi-nccl, custom vendor plugins). Cloud instances sometimes have these plugins installed but misconfigured. This flag ensures NCCL doesn't attempt to load or use them.

**`NCCL_DEBUG=WARN`**
NCCL's default log level (`INFO`) produces a large volume of output per process per step, which obscures training logs. `WARN` shows only actual warnings and errors. Setting this to `INFO` temporarily is useful when debugging a new hang or communication error.

Together these flags mean: "don't try anything fancy, just use plain sockets over loopback." The overhead compared to NVLink or P2P is negligible when GPU compute is the bottleneck.

### `torchrun`

```bash
torchrun --standalone --nproc_per_node=N run_13_code.py [args]
```

`--standalone` handles the rendezvous internally — no external etcd store, c10d coordinator, or separate master process is needed. Each GPU gets its own process with `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` set automatically via environment variables.

### DDP Initialization in Code

```python
def setup_distributed():
    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device(f'cuda:{local_rank}')
        )
    return rank, world_size, local_rank
```

`local_rank` is the GPU index on this machine (0–3 for 4 GPUs). `rank` is the global process index (same as `local_rank` for single-node). Passing `device_id` to `init_process_group` ensures NCCL binds to the correct GPU before any CUDA allocations happen on that device.

After loading the checkpoint and resizing embeddings:

```python
if world_size > 1:
    dist.barrier()   # ensure all ranks have identical weights before DDP copies them
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

The `dist.barrier()` before wrapping is important — it ensures all ranks have completed checkpoint loading and the embedding resize before DDP broadcasts weights. Without it, ranks can proceed at different speeds and end up with different initial weights.

### Sharding the Dataloader

Each rank must see a different, non-overlapping slice of the data. `PackedStreamingDataset` and `DPOStreamingDataset` both implement the same sharding logic inside `__iter__`:

```python
worker_info = torch.utils.data.get_worker_info()
if worker_info is None:
    num_workers, worker_id = 1, 0
else:
    num_workers, worker_id = worker_info.num_workers, worker_info.id

total_shards     = world_size * num_workers
shard_id         = rank * num_workers + worker_id
effective_shards = min(total_shards, len(files))    # fallback for small datasets
effective_shard_id = shard_id % effective_shards
my_files = [f for i, f in enumerate(files) if i % effective_shards == effective_shard_id]
```

The `effective_shards` fallback is needed for small datasets that have fewer parquet files than `world_size`. Without it, some ranks would receive an empty file list and immediately return from `__iter__`, then block forever at the next `all_reduce` while other ranks are still generating batches.

**Val loaders always use `rank=0, world_size=1`**: validation runs on rank 0 only, seeing all the val data. Training loaders use each rank's actual rank/world_size. Metrics from training are averaged across ranks via `reduce_tensor`:

```python
def reduce_tensor(tensor, world_size):
    if world_size > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
    return rt
```

**`num_workers=0`**: all DataLoaders use `num_workers=0`. With PyTorch `IterableDataset`, each worker spawns a subprocess that independently continues the same infinite generator — without explicit offset handling, multiple workers on the same rank produce duplicate data. Beyond correctness, worker processes on cloud instances caused persistent crashes during DDP initialization. Setting `num_workers=0` runs the data pipeline on the main process and eliminates this instability entirely.

### GPU Memory Monitoring

Understanding memory usage is critical for multi-GPU training, where OOM errors manifest differently than on single-GPU setups and can be difficult to diagnose. PyTorch exposes three distinct measurements:

```python
torch.cuda.memory_allocated()     # bytes actively held by live tensors
torch.cuda.memory_reserved()      # bytes held by the caching allocator (includes free fragments)
torch.cuda.max_memory_allocated()  # high-water mark since last reset
```

**`memory_allocated`** reflects the true working set — tensors currently referenced by Python objects. If this is close to the GPU's physical memory, you are near the OOM boundary.

**`memory_reserved`** is always ≥ `memory_allocated`. PyTorch's caching allocator holds freed memory in a pool rather than returning it to CUDA immediately (because `cudaMalloc`/`cudaFree` are expensive). The gap between reserved and allocated is fragmented free memory — unavailable to other processes but reusable by the current process. A large gap is normal during training; it closes when `torch.cuda.empty_cache()` is called, but this should not be done in the training loop because it defeats the allocator's purpose.

**`max_memory_allocated`** is the peak since the last `reset_peak_memory_stats()` call. It is more useful than the instantaneous value for diagnosing OOM because OOMs happen at the peak, not at an arbitrary sampling point. `reset_peak_memory_stats()` is called at the end of each validation step so the reported peak reflects the most recent training window.

**Using peak memory to diagnose OOM bugs**: the peak typically occurs at one of three points:
1. *Forward pass* — activations for all layers are live simultaneously (worst case: `[B, T, dim]` per layer × N layers)
2. *Backward pass* — gradients are accumulated, activation graph is still held; peaks above forward
3. *Optimizer step* — AdamW maintains `m` and `v` tensors per parameter in fp32 (~2× model size)

DPO added a second forward pass (reference model) and surfaced several non-obvious memory pitfalls. All were diagnosed by comparing `max_memory_allocated` before and after each code change:

- **Dtype casts create copies**: `logits.float()` while the bf16 original is still in scope doubles the tensor's footprint from `[B, T, V] × 2 bytes` to `[B, T, V] × (2+4) bytes`. Fix: upcast only after gathering down to `[B, T]`.
- **`log_softmax` materialises `[B, T, V]`**: calling `F.log_softmax` then `.gather()` computes and stores a value per vocabulary token — 99.99% of which is immediately discarded. Fix: gather the target logit first, then compute `log_z` via `logsumexp` over the original logits.
- **`logsumexp` backward also allocates `[B, T, V]`**: the backward of `logsumexp` computes `softmax(input)` — the same size as the input. Fix: process one sequence at a time so autograd frees each `[T-1, V]` slice before the next. Peak drops from ~3.5 GB to ~400 MB.
- **No-grad before grad**: a backward-tracked forward pass holds its full activation graph (~40 GB for policy) until `loss.backward()` completes. Running the reference forward pass (needing ~17 GB working memory) after the policy forward OOMs because the activation graph is already occupying the GPU. Fix: always run no-grad computations first — their working memory is freed immediately when the call returns.
