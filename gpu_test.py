import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# ensure to run pkill -9 python

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'INFO'  # See what NCCL is doing
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"[Rank {rank}] Running basic DDP example on rank {rank}.", flush=True)
    
    setup(rank, world_size)
    print(f"[Rank {rank}] Process group initialized", flush=True)
    
    # Check GPU memory
    free_mem, total_mem = torch.cuda.mem_get_info(rank)
    print(f"[Rank {rank}] GPU {rank} free memory: {free_mem / 1e9:.2f} GB / {total_mem / 1e9:.2f} GB", flush=True)
    
    # Set seed for consistency
    torch.manual_seed(42)
    print(f"[Rank {rank}] Creating model...", flush=True)
    
    model = ToyModel().to(rank)
    print(f"[Rank {rank}] Model moved to GPU {rank}", flush=True)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] Model has {num_params:,} parameters", flush=True)
    
    print(f"[Rank {rank}] Starting DDP wrap (this does parameter broadcast)...", flush=True)
    ddp_model = DDP(model, device_ids=[rank])
    print(f"[Rank {rank}] *** DDP WRAP COMPLETE ***", flush=True)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    print(f"[Rank {rank}] Starting forward pass...", flush=True)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(rank))  # Make sure input is on GPU!
    print(f"[Rank {rank}] Forward pass complete", flush=True)
    
    labels = torch.randn(20, 5).to(rank)
    print(f"[Rank {rank}] Starting backward pass...", flush=True)
    loss_fn(outputs, labels).backward()
    print(f"[Rank {rank}] Backward pass complete", flush=True)
    
    print(f"[Rank {rank}] Starting optimizer step...", flush=True)
    optimizer.step()
    print(f"[Rank {rank}] Optimizer step complete", flush=True)
    
    cleanup()
    print(f"[Rank {rank}] Finished running basic DDP example.", flush=True)


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = torch.accelerator.current_accelerator()
    # configure map_location properly
    map_location = {f'{acc}:0': f'{acc}:{rank}'}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")

if __name__ == "__main__":
    n_gpus = torch.accelerator.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)