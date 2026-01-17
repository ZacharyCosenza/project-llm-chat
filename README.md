# Project LLM Chat

Training experiments for small-scale GPT models using PyTorch Lightning.

## Experiments

**Run 1 ([run_1_base.py](run_1_base.py))**: Baseline implementation with a small TinyGPT model (2 layers, 64 dim, 128 seq len) using standard AdamW optimizer with uniform learning rate. Validates basic training pipeline, multi-GPU support, and WandB integration.

**Run 2 ([run_2_opt.py](run_2_opt.py))**: Scaled-up model (20 layers, 1280 dim, 2048 seq len) with advanced optimization techniques including differentiated learning rates for different parameter groups (head/embeddings/other), learning rate scheduling (warmup/constant/warmdown), and FLOPs tracking. Demonstrates how layer-specific learning rates and proper scheduling improve training of larger models. Uses tuned AdamW hyperparameters (β=(0.8, 0.95), eps=1e-10) and provides CLI for easy experimentation.

NCCL_P2P_DISABLE=1