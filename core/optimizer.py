"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor
from functools import partial
from core.utils import get_dist_info

warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR

def get_lr_multiplier(it, num_iterations):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton–Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0,1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton–Schulz iterations for the orthogonalization
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params}) # sort to ensure consistent / deterministic ordering
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the Muon update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()


def setup_optimizers(model, model_dim=None, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """
        Setup optimizers for a model with separate handling for embeddings, transformer layers, and output head.

        Args:
            model: The model to optimize (can be a LightningModule or regular nn.Module)
            model_dim: Model dimension for LR scaling. If None, attempts to infer from model.
            unembedding_lr: Learning rate for the output/unembedding layer
            embedding_lr: Learning rate for the embedding layer
            matrix_lr: Learning rate for transformer matrix parameters (optimized with Muon)
            weight_decay: Weight decay for AdamW optimizer
        """
        # Try to get the actual model if wrapped in LightningModule
        actual_model = model.model if hasattr(model, 'model') else model

        # Infer model_dim if not provided
        if model_dim is None:
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'dim'):
                model_dim = model.hparams.dim
            elif hasattr(actual_model, 'tok_emb'):
                model_dim = actual_model.tok_emb.embedding_dim
            else:
                model_dim = 768  # default fallback

        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into 3 groups (matrix, embedding, head)
        # Adapt to TinyGPT structure: tok_emb, pos_emb, blocks, ln_f, head
        matrix_params = []
        embedding_params = []
        lm_head_params = []

        if hasattr(actual_model, 'blocks'):
            # TinyGPT-style model
            matrix_params = list(actual_model.blocks.parameters())
            if hasattr(actual_model, 'ln_f'):
                matrix_params.extend(list(actual_model.ln_f.parameters()))
            if hasattr(actual_model, 'tok_emb'):
                embedding_params.extend(list(actual_model.tok_emb.parameters()))
            if hasattr(actual_model, 'pos_emb'):
                embedding_params.extend(list(actual_model.pos_emb.parameters()))
            if hasattr(actual_model, 'head'):
                lm_head_params = list(actual_model.head.parameters())
        elif hasattr(actual_model, 'transformer'):
            # Original GPT-style model structure
            if hasattr(actual_model.transformer, 'h'):
                matrix_params = list(actual_model.transformer.h.parameters())
            if hasattr(actual_model.transformer, 'wte'):
                embedding_params = list(actual_model.transformer.wte.parameters())
            if hasattr(actual_model, 'lm_head'):
                lm_head_params = list(actual_model.lm_head.parameters())

        total_params = len(list(actual_model.parameters()))
        found_params = len(matrix_params) + len(embedding_params) + len(lm_head_params)
        if rank == 0:
            print(f"Parameter groups: {len(matrix_params)} matrix, {len(embedding_params)} embedding, {len(lm_head_params)} head")
            print(f"Total params: {total_params}, Found: {found_params}")

        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers