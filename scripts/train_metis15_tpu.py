#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from metis_mamba import MetisMambaConfig, cosine_lr
from metis_mamba.checkpoints import atomic_torch_save
from metis_mamba.optim import build_optimizer_from_args


@dataclass
class Runtime:
    device: torch.device
    device_kind: str
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_xla: bool
    xm: Any | None = None


ROUTER_OVERRIDE_CHOICES = ("learned", "force_balanced", "uniform_random")
LOSS_MODE_CHOICES = ("real_ce", "dummy_loss")
CE_LOGITS_DTYPE_CHOICES = ("float32", "bfloat16", "model")
CE_IMPL_CHOICES = ("cross_entropy", "manual_logsumexp")
ATTENTION_MODE_CHOICES = ("real", "identity")
ATTENTION_KERNEL_CHOICES = ("eager", "eager_gqa", "sdpa")
MOE_MODE_CHOICES = ("real", "identity", "local_only", "dense_ffn")
DISPATCH_PACK_IMPL_CHOICES = (
    "index_add",
    "one_hot",
    "gather",
    "one_hot_gather",
    "sort_pack",
    "balanced_static",
    "group_static",
    "group_static_gather",
    "dense_all_experts",
)
GRAD_SYNC_MODE_CHOICES = ("all_reduce", "all_reduce_staged", "expert_only", "none")
ACTIVATION_CHECKPOINT_CHOICES = ("none", "layers", "attention", "moe", "attention_moe")
BALANCED_STATIC_LAYOUT_CHOICES = ("strided", "indexed")
BALANCED_STATIC_ROUTER_WEIGHT_CHOICES = ("uniform", "learned")
BALANCED_STATIC_ROUTER_INPUT_CHOICES = ("latent", "hidden")
EXPERT_ACTIVATION_SAFETY_CHOICES = ("clamp", "none")
STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will save latest.pt after the current optimizer step.", flush=True)


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_runtime(device_arg: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    pjrt_device = os.environ.get("PJRT_DEVICE", "").upper()
    wants_xla = device_arg == "xla" or pjrt_device == "TPU"
    if wants_xla:
        import torch_xla.core.xla_model as xm
        try:
            import torch_xla.runtime as xr
        except Exception:
            xr = None

        if distributed:
            import torch_xla.distributed.xla_backend  # noqa: F401

            if not dist.is_initialized():
                try:
                    dist.init_process_group("xla", init_method="xla://")
                except Exception:
                    dist.init_process_group("xla")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        elif xr is not None:
            try:
                world_size = int(xr.world_size())
                rank = int(xr.global_ordinal())
                local_rank = int(xr.local_ordinal())
                distributed = world_size > 1
            except Exception:
                pass
        device = xm.xla_device()
        try:
            device_kind = xm.xla_device_kind()
        except Exception:
            device_kind = "xla"
        return Runtime(
            device=device,
            device_kind=device_kind,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_xla=True,
            xm=xm,
        )
    if distributed:
        raise RuntimeError("The TPU trainer only supports distributed training through --device xla.")
    return Runtime(
        device=torch.device(device_arg),
        device_kind=device_arg,
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_xla=False,
        xm=None,
    )


def cleanup_runtime(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()


def mark_step(runtime: Runtime) -> None:
    if runtime.is_xla:
        runtime.xm.mark_step()


def wait_device(runtime: Runtime) -> None:
    if runtime.is_xla:
        runtime.xm.wait_device_ops()


class XlaAllToAllAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        split_dimension: int,
        concat_dimension: int,
        split_count: int,
    ) -> torch.Tensor:
        import torch_xla.core.xla_model as xm

        ctx.split_dimension = int(split_dimension)
        ctx.concat_dimension = int(concat_dimension)
        ctx.split_count = int(split_count)
        return xm.all_to_all(
            value,
            ctx.split_dimension,
            ctx.concat_dimension,
            ctx.split_count,
            pin_layout=False,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        import torch_xla.core.xla_model as xm

        grad_input = xm.all_to_all(
            grad_output.contiguous(),
            ctx.concat_dimension,
            ctx.split_dimension,
            ctx.split_count,
            pin_layout=False,
        )
        return grad_input, None, None, None


def xla_all_to_all(runtime: Runtime, value: torch.Tensor) -> torch.Tensor:
    if not runtime.is_xla or runtime.world_size == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        return XlaAllToAllAutograd.apply(value, 0, 0, runtime.world_size)
    return runtime.xm.all_to_all(value, 0, 0, runtime.world_size, pin_layout=False)


class XlaAllGatherAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, dim: int, rank: int, world_size: int) -> torch.Tensor:
        import torch_xla.core.xla_model as xm

        ctx.dim = int(dim)
        ctx.rank = int(rank)
        ctx.world_size = int(world_size)
        ctx.local_size = int(value.shape[ctx.dim])
        return xm.all_gather(value, dim=ctx.dim, pin_layout=False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        import torch_xla.core.xla_model as xm

        start = ctx.rank * ctx.local_size
        grad_slice = grad_output.narrow(ctx.dim, start, ctx.local_size).contiguous()
        grad_slice = xm.all_reduce(
            xm.REDUCE_SUM,
            grad_slice,
            pin_layout=False,
        )
        return grad_slice, None, None, None


def xla_all_gather(runtime: Runtime, value: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    if not runtime.is_xla or runtime.world_size == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        return XlaAllGatherAutograd.apply(value, dim, runtime.rank, runtime.world_size)
    return runtime.xm.all_gather(value, dim=dim, pin_layout=False)


def xla_all_reduce_gradients(
    runtime: Runtime,
    model: nn.Module,
    *,
    sharded_name_fragment: str = ".moe.experts.",
    mode: str = "all_reduce",
    bucket_mb: float = 0.0,
) -> None:
    if runtime.world_size <= 1 or mode == "none":
        return
    replicated_grads: list[torch.Tensor] = []

    def reduce_grads_in_place(grads: list[torch.Tensor]) -> None:
        if not grads:
            return
        reduced = runtime.xm.all_reduce(
            runtime.xm.REDUCE_SUM,
            grads,
            scale=1.0 / float(runtime.world_size),
            pin_layout=False,
        )
        if reduced is None:
            return
        if torch.is_tensor(reduced):
            if len(grads) != 1:
                raise RuntimeError("XLA all_reduce returned one tensor for multiple gradient inputs.")
            grads[0].copy_(reduced)
            return
        for grad, reduced_grad in zip(grads, reduced, strict=True):
            grad.copy_(reduced_grad)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if sharded_name_fragment in name:
            param.grad.mul_(1.0 / float(runtime.world_size))
        elif mode in {"all_reduce", "all_reduce_staged"}:
            replicated_grads.append(param.grad)
    if mode in {"all_reduce", "all_reduce_staged"} and replicated_grads:
        staged = mode == "all_reduce_staged"
        if staged and bucket_mb <= 0:
            raise ValueError("--grad-sync-mode all_reduce_staged requires --grad-sync-bucket-mb > 0.")
        if bucket_mb <= 0:
            reduce_grads_in_place(replicated_grads)
            return
        bucket_limit = max(1, int(bucket_mb * 1024 * 1024))
        bucket: list[torch.Tensor] = []
        bucket_bytes = 0
        max_chunk_elems_by_dtype: dict[torch.dtype, int] = {}

        def flush_bucket() -> None:
            nonlocal bucket, bucket_bytes
            if bucket:
                reduce_grads_in_place(bucket)
                if staged:
                    mark_step(runtime)
                bucket = []
                bucket_bytes = 0

        def max_chunk_elems(grad: torch.Tensor) -> int:
            cached = max_chunk_elems_by_dtype.get(grad.dtype)
            if cached is not None:
                return cached
            value = max(1, bucket_limit // max(grad.element_size(), 1))
            max_chunk_elems_by_dtype[grad.dtype] = value
            return value

        for grad in replicated_grads:
            grad_bytes = grad.numel() * grad.element_size()
            if grad_bytes > bucket_limit:
                flush_bucket()
                flat_grad = grad.view(-1)
                for chunk in flat_grad.split(max_chunk_elems(grad)):
                    reduce_grads_in_place([chunk])
                    if staged:
                        mark_step(runtime)
                continue
            if bucket and bucket_bytes + grad_bytes > bucket_limit:
                flush_bucket()
            bucket.append(grad)
            bucket_bytes += grad_bytes
        flush_bucket()


def xla_reduce_tensor(
    runtime: Runtime,
    value: torch.Tensor,
    *,
    reduce_type: str = "sum",
    scale: float = 1.0,
) -> torch.Tensor:
    if runtime.world_size <= 1:
        return value
    if reduce_type not in {"sum", "max"}:
        raise ValueError(f"Unsupported reduce_type: {reduce_type}")
    if runtime.is_xla:
        op = runtime.xm.REDUCE_SUM if reduce_type == "sum" else runtime.xm.REDUCE_MAX
        return runtime.xm.all_reduce(
            op,
            value,
            scale=scale,
            pin_layout=False,
        )
    if dist.is_initialized():
        reduced = value.clone()
        op = dist.ReduceOp.SUM if reduce_type == "sum" else dist.ReduceOp.MAX
        dist.all_reduce(reduced, op=op)
        if scale != 1.0 and reduce_type == "sum":
            reduced.mul_(scale)
        return reduced
    return value


def grad_norm_diagnostics(
    runtime: Runtime,
    model: nn.Module,
    *,
    sharded_name_fragment: str = ".moe.experts.",
) -> torch.Tensor:
    local_common_sq = torch.zeros((), device=runtime.device, dtype=torch.float32)
    local_expert_sq = torch.zeros((), device=runtime.device, dtype=torch.float32)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        if sharded_name_fragment in name:
            local_expert_sq = local_expert_sq + grad_sq
        else:
            local_common_sq = local_common_sq + grad_sq
    common_sq = xla_reduce_tensor(runtime, local_common_sq, reduce_type="sum", scale=1.0)
    expert_sq = xla_reduce_tensor(runtime, local_expert_sq, reduce_type="sum", scale=1.0)
    return torch.stack((common_sq.sqrt(), expert_sq.sqrt()))


def capture_param_delta_refs(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    preferred = (
        "layers.0.attn.qkv_proj.weight",
        "layers.0.attn.o_proj.weight",
        "layers.0.moe.router.weight",
        "layers.0.moe.down_proj.weight",
        "layers.0.moe.up_proj.weight",
        "layers.0.moe.shared.0.up.weight",
        "layers.0.moe.experts.0.up.weight",
        "layers.0.moe.experts.0.down.weight",
        "lm_head.weight",
    )
    params = dict(model.named_parameters())
    refs: list[tuple[str, torch.nn.Parameter]] = []
    for name in preferred:
        param = params.get(name)
        if param is not None and param.requires_grad:
            refs.append((name, param))
    if refs:
        return refs
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            refs.append((name, param))
            if len(refs) >= 4:
                break
    return refs


def clone_param_refs(refs: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, torch.Tensor]]:
    return [(name, param.detach().clone()) for name, param in refs]


def param_delta_diagnostics(
    refs: list[tuple[str, torch.nn.Parameter]],
    before: list[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    before_by_name = {name: tensor for name, tensor in before}
    values: list[torch.Tensor] = []
    for name, param in refs:
        previous = before_by_name.get(name)
        if previous is None:
            continue
        delta = (param.detach().float() - previous.float()).norm()
        current = param.detach().float().norm().clamp_min(1e-12)
        values.append(delta / current)
    if not values:
        device = refs[0][1].device if refs else torch.device("cpu")
        return torch.zeros((0,), device=device)
    return torch.stack(values)


def preinitialize_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Materialize optimizer tensors before the first XLA train graph."""
    for group in optimizer.param_groups:
        mode = group.get("optimizer", "adamw")
        amsgrad = bool(group.get("amsgrad", False))
        use_device_step = bool(group.get("capturable", False) or group.get("fused", False))
        use_master_weights = bool(group.get("master_weights", False))
        for param in group["params"]:
            if param.requires_grad is False:
                continue
            state = optimizer.state[param]
            if state:
                if use_master_weights and "master_param" not in state:
                    state["master_param"] = param.detach().float().clone()
                continue
            if mode == "muon":
                state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                if use_master_weights:
                    state["master_param"] = param.detach().float().clone()
                continue
            if mode != "adamw":
                continue
            if str(param.device).startswith("xla") and "optimizer" in group:
                use_device_step = True
            step_device = param.device if use_device_step else torch.device("cpu")
            state_dtype = torch.float32 if "optimizer" in group else param.dtype
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(param, dtype=state_dtype, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=state_dtype, memory_format=torch.preserve_format)
            if use_master_weights:
                state["master_param"] = param.detach().float().clone()
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(param, dtype=state_dtype, memory_format=torch.preserve_format)


def activation_checkpoint(runtime: Runtime, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if runtime.is_xla:
        from torch_xla.utils.checkpoint import checkpoint
    else:
        from torch.utils.checkpoint import checkpoint

    return checkpoint(module, hidden_states, use_reentrant=True)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normed.to(dtype) * self.weight.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float, max_seq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("s,d->sd", positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        del device
        return self.cos_cached[:seq_len].to(dtype=dtype), self.sin_cached[:seq_len].to(dtype=dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * repeats, seq_len, head_dim)


class XlaSelfAttention(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, attention_kernel: str, qk_clip_enabled: bool) -> None:
        super().__init__()
        self.attention_kernel = attention_kernel
        self.qk_clip_enabled = qk_clip_enabled
        self.num_heads = int(config.n_heads)
        self.num_kv_heads = int(config.n_kv_heads)
        self.head_dim = int(config.head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(config.d_model, self.q_dim + (2 * self.kv_dim), bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_dim, config.d_model, bias=config.attention_bias)
        self.rotary = RotaryEmbedding(self.head_dim, base=float(config.rope_theta), max_seq_len=int(config.block_size))
        causal = torch.triu(
            torch.full((int(config.block_size), int(config.block_size)), -10000.0, dtype=torch.float32),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer("qk_clip_max_logits", torch.zeros(self.num_heads, dtype=torch.float32), persistent=False)

    def reset_qk_clip_stats(self) -> None:
        self.qk_clip_max_logits = torch.zeros_like(self.qk_clip_max_logits)

    def _record_qk_clip_scores(self, query: torch.Tensor, key: torch.Tensor, seq_len: int) -> None:
        if not self.qk_clip_enabled:
            return
        batch_size = query.shape[0]
        q = query.float().view(batch_size, self.num_kv_heads, self.num_kv_groups, seq_len, self.head_dim)
        k = key.float().unsqueeze(2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        self._record_qk_clip_max_from_scores(scores, seq_len)

    def _record_qk_clip_max_from_scores(self, scores: torch.Tensor, seq_len: int) -> None:
        if not self.qk_clip_enabled:
            return
        scores = scores.float()
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device))
        if scores.ndim == 5:
            scores = scores.masked_fill(~causal.view(1, 1, 1, seq_len, seq_len), float("-inf"))
            max_logits = scores.amax(dim=(0, 3, 4)).reshape(self.num_heads)
        else:
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), float("-inf"))
            max_logits = scores.amax(dim=(0, 2, 3)).reshape(self.num_heads)
        max_logits = torch.nan_to_num(max_logits, nan=0.0, posinf=0.0, neginf=0.0)
        self.qk_clip_max_logits = torch.maximum(
            self.qk_clip_max_logits.to(device=max_logits.device),
            max_logits.detach(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _hidden = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(seq_len, hidden_states.device, hidden_states.dtype)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        if self.attention_kernel == "sdpa":
            self._record_qk_clip_scores(query, key, seq_len)
            key = repeat_kv(key, self.num_kv_groups)
            value = repeat_kv(value, self.num_kv_groups)
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )
        elif self.attention_kernel == "eager_gqa":
            query = query.view(batch_size, self.num_kv_heads, self.num_kv_groups, seq_len, self.head_dim)
            scores = torch.matmul(query, key.unsqueeze(2).transpose(-1, -2)) * self.scale
            self._record_qk_clip_max_from_scores(scores, seq_len)
            causal = self.causal_mask[:seq_len, :seq_len].view(1, 1, 1, seq_len, seq_len)
            scores = scores.float() + causal
            weights = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
            out = torch.matmul(weights, value.unsqueeze(2))
            out = out.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        else:
            key = repeat_kv(key, self.num_kv_groups)
            value = repeat_kv(value, self.num_kv_groups)
            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            self._record_qk_clip_max_from_scores(scores, seq_len)
            causal = self.causal_mask[:seq_len, :seq_len]
            scores = scores.float() + causal.view(1, 1, seq_len, seq_len)
            weights = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
            out = torch.matmul(weights, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_dim)
        return self.o_proj(out)


class SquaredReluExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        bias: bool,
        activation_safety: str,
    ) -> None:
        super().__init__()
        self.up = nn.Linear(in_features, hidden_features, bias=bias)
        self.down = nn.Linear(hidden_features, out_features, bias=bias)
        self.activation_safety = activation_safety

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.up(hidden_states)
        if self.activation_safety == "clamp":
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-64.0, max=64.0)
        elif self.activation_safety != "none":
            raise ValueError(f"Unsupported expert activation safety mode: {self.activation_safety}")
        hidden = F.relu(hidden).square()
        return self.down(hidden)


class StaticExpertParallelMoE(nn.Module):
    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        runtime: Runtime,
        layer_idx: int,
        capacity_factor: float,
        capacity: int | None,
        dispatch_pack_impl: str,
        router_override: str,
        moe_mode: str,
        track_route_metrics: bool,
        balanced_static_layout: str,
        balanced_static_router_weights: str,
        balanced_static_router_input: str,
        expert_activation_safety: str,
    ) -> None:
        super().__init__()
        if not config.uses_single_latent_moe:
            raise ValueError("TPU Metis trainer currently supports the single-latent MoE contract.")
        if int(config.moe_num_experts) % int(runtime.world_size) != 0:
            raise ValueError("moe_num_experts must divide runtime world_size for TPU expert parallel.")
        self.runtime = runtime
        self.layer_idx = layer_idx
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.shared_experts = int(config.moe_shared_experts)
        self.local_num_experts = self.num_experts // max(1, int(runtime.world_size))
        self.local_expert_start = int(runtime.rank) * self.local_num_experts
        self.router_temperature = float(config.moe_router_temperature)
        self.router_score = str(config.moe_router_score)
        self.capacity_factor = float(capacity_factor)
        self.explicit_capacity = capacity
        self.dispatch_pack_impl = dispatch_pack_impl
        self.router_override = router_override
        self.moe_mode = moe_mode
        self.track_route_metrics = bool(track_route_metrics)
        self.balanced_static_layout = balanced_static_layout
        self.balanced_static_router_weights = balanced_static_router_weights
        self.balanced_static_router_input = balanced_static_router_input
        self.balanced_static_route_offset = int(self.layer_idx * max(1, self.top_k)) % max(1, self.num_experts)
        self.capacity_alignment = max(1, int(config.moe_capacity_alignment))
        self.aux_loss_coef = float(config.moe_aux_loss_coef)
        self.balance_strategy = str(config.moe_balance_strategy)
        self.balance_bias_update_rate = float(config.moe_balance_bias_update_rate)
        self.balance_bias_clamp = float(config.moe_balance_bias_clamp)
        self.balance_scale_by_token_fraction = bool(config.moe_balance_scale_by_token_fraction)
        self.balance_update_scale = 1.0
        self.routed_dim = int(config.moe_routed_latent_size)
        self.d_model = int(config.d_model)
        self.down_proj = nn.Linear(config.d_model, self.routed_dim, bias=False)
        self.up_proj = nn.Linear(self.routed_dim, config.d_model, bias=False)
        if self.balanced_static_router_input not in BALANCED_STATIC_ROUTER_INPUT_CHOICES:
            raise ValueError(
                f"Unsupported balanced static router input: {self.balanced_static_router_input}"
            )
        router_in_features = self.d_model if self.balanced_static_router_input == "hidden" else self.routed_dim
        self.router = nn.Linear(router_in_features, self.num_experts, bias=True)
        self.register_buffer("balance_bias", torch.zeros(self.num_experts), persistent=True)
        self.dense_ffn = SquaredReluExpert(
            config.d_model,
            config.intermediate_size,
            config.d_model,
            bias=config.mlp_bias,
            activation_safety=expert_activation_safety,
        )
        self.shared = nn.ModuleList(
            [
                SquaredReluExpert(
                    config.d_model,
                    config.moe_expert_intermediate_size,
                    config.d_model,
                    bias=config.mlp_bias,
                    activation_safety=expert_activation_safety,
                )
                for _ in range(self.shared_experts)
            ]
        )
        self.experts = nn.ModuleList(
            [
                SquaredReluExpert(
                    self.routed_dim,
                    config.moe_expert_intermediate_size,
                    self.routed_dim,
                    bias=config.mlp_bias,
                    activation_safety=expert_activation_safety,
                )
                for _ in range(self.local_num_experts)
            ]
        )
        self.last_aux_loss: torch.Tensor | None = None
        self.last_valid_assignments: torch.Tensor | None = None
        self.last_route_counts: torch.Tensor | None = None
        self.last_dest_counts: torch.Tensor | None = None
        self.last_dest_valid: torch.Tensor | None = None
        self.last_capacity: torch.Tensor | None = None
        self.last_total_assignments: torch.Tensor | None = None
        self.last_dropped_assignments: torch.Tensor | None = None

    def reset_local_experts(self, init_std: float, base_seed: int) -> None:
        layer_offset = (int(self.layer_idx) + 1) * 1_000_003
        for local_idx, expert in enumerate(self.experts):
            global_expert = self.local_expert_start + local_idx
            seed = int(base_seed) + layer_offset + global_expert
            for module in (expert.up, expert.down):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                nn.init.normal_(module.weight, mean=0.0, std=init_std, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _capacity(self, assignments: int) -> int:
        if self.explicit_capacity is not None and self.explicit_capacity > 0:
            capacity = int(self.explicit_capacity)
        else:
            per_rank = math.ceil(float(assignments) / float(max(1, self.runtime.world_size)))
            capacity = int(math.ceil(per_rank * max(self.capacity_factor, 1.0)))
        if capacity % self.capacity_alignment:
            capacity += self.capacity_alignment - (capacity % self.capacity_alignment)
        return max(1, min(capacity, assignments))

    def _zero_aux(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def _override_route(self, latent_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _latent_dim = latent_tokens.shape
        row_ids = torch.arange(batch_size * seq_len, device=latent_tokens.device, dtype=torch.long)
        slot_ids = torch.arange(self.top_k, device=latent_tokens.device, dtype=torch.long)
        if self.router_override == "force_balanced":
            expert_ids = (row_ids.view(-1, 1) * self.top_k + slot_ids.view(1, -1)) % self.num_experts
        elif self.router_override == "uniform_random":
            layer_salt = (int(self.layer_idx) + 1) * 2_654_435_761
            hashed = (
                (row_ids.view(-1, 1) + 1) * 1_103_515_245
                + (slot_ids.view(1, -1) + 17) * 12_345
                + layer_salt
            )
            expert_ids = torch.remainder(hashed, self.num_experts)
        else:
            raise ValueError(f"Unknown router override: {self.router_override}")
        weights = latent_tokens.new_full(expert_ids.shape, 1.0 / float(max(self.top_k, 1)))
        return (
            expert_ids.view(batch_size, seq_len, self.top_k),
            weights.view(batch_size, seq_len, self.top_k),
            self._zero_aux(latent_tokens),
        )

    @torch.no_grad()
    def _route_counts_tensor(self, topk_indices: torch.Tensor) -> torch.Tensor:
        flat = topk_indices.reshape(-1).to(torch.long)
        if self.runtime.is_xla:
            return F.one_hot(flat, num_classes=self.num_experts).to(torch.float32).sum(dim=0)
        return torch.bincount(flat, minlength=self.num_experts).to(torch.float32)

    @torch.no_grad()
    def _record_route_counts(self, topk_indices: torch.Tensor) -> None:
        self.last_route_counts = self._route_counts_tensor(topk_indices)

    @torch.no_grad()
    def _maybe_update_balance_bias(self, topk_indices: torch.Tensor) -> None:
        if (
            not self.training
            or self.router_override != "learned"
            or self.balance_strategy != "aux_loss_free_bias"
            or self.balance_bias_update_rate <= 0.0
        ):
            return
        update_rate = self.balance_bias_update_rate
        if self.balance_scale_by_token_fraction:
            update_rate *= self.balance_update_scale
        if update_rate <= 0.0:
            return
        counts = self._route_counts_tensor(topk_indices).to(dtype=torch.float32)
        if self.runtime.world_size > 1 and self.runtime.is_xla:
            counts = self.runtime.xm.all_reduce(
                self.runtime.xm.REDUCE_SUM,
                counts,
                pin_layout=False,
            )
        elif self.runtime.world_size > 1 and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        load = counts / counts.sum().clamp_min(1.0)
        target = 1.0 / float(self.num_experts)
        direction = torch.where(load > target, -1.0, 1.0)
        self.balance_bias.add_(direction.to(dtype=self.balance_bias.dtype) * update_rate)
        self.balance_bias.clamp_(min=-self.balance_bias_clamp, max=self.balance_bias_clamp)

    def _route(
        self,
        latent_tokens: torch.Tensor,
        hidden_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.router_override != "learned":
            topk_indices, topk_weights, aux_loss = self._override_route(latent_tokens)
            self._record_route_counts(topk_indices)
            return topk_indices, topk_weights, aux_loss
        router_tokens = latent_tokens
        if self.balanced_static_router_input == "hidden":
            if hidden_tokens is None:
                raise RuntimeError("hidden router input requires hidden tokens.")
            router_tokens = hidden_tokens
        logits = self.router(router_tokens).float() / self.router_temperature
        if self.router_score == "softmax":
            probs = torch.softmax(logits, dim=-1)
            selection_scores = logits
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, self.num_experts).to(logits.dtype)
            _values, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = logits.gather(-1, topk_indices)
            topk_weights = torch.softmax(topk_scores, dim=-1)
        else:
            probs = torch.sigmoid(logits)
            selection_scores = probs
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, self.num_experts).to(probs.dtype)
            _values, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = probs.gather(-1, topk_indices)
            topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_load = probs.mean(dim=(0, 1))
        uniform = expert_load.new_full(expert_load.shape, 1.0 / float(self.num_experts))
        aux_loss = self.num_experts * torch.mean((expert_load - uniform).pow(2))
        if self.track_route_metrics:
            self._record_route_counts(topk_indices)
        self._maybe_update_balance_bias(topk_indices)
        return topk_indices, topk_weights.to(dtype=latent_tokens.dtype), aux_loss

    def _static_balanced_indices(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _latent_dim = latent_tokens.shape
        row_ids = torch.arange(batch_size * seq_len, device=latent_tokens.device, dtype=torch.long)
        slot_ids = torch.arange(self.top_k, device=latent_tokens.device, dtype=torch.long)
        route_offset = int(self.balanced_static_route_offset) % max(1, self.num_experts)
        expert_ids = torch.remainder(
            row_ids.view(-1, 1) * self.top_k + slot_ids.view(1, -1) + route_offset,
            self.num_experts,
        )
        return expert_ids.view(batch_size, seq_len, self.top_k)

    def _route_static_balanced_weights(
        self,
        latent_tokens: torch.Tensor,
        hidden_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_indices = self._static_balanced_indices(latent_tokens)
        if self.balanced_static_router_weights == "uniform":
            weights = latent_tokens.new_full(topk_indices.shape, 1.0 / float(max(self.top_k, 1)))
            return topk_indices, weights, self._zero_aux(latent_tokens)
        if self.balanced_static_router_weights != "learned":
            raise ValueError(f"Unsupported balanced static router weights: {self.balanced_static_router_weights}")

        router_tokens = latent_tokens
        if self.balanced_static_router_input == "hidden":
            if hidden_tokens is None:
                raise RuntimeError("balanced_static hidden router input requires hidden tokens.")
            router_tokens = hidden_tokens
        logits = self.router(router_tokens).float() / self.router_temperature
        if self.router_score == "softmax":
            balance_probs = torch.softmax(logits, dim=-1)
            gathered = logits.gather(-1, topk_indices)
            weights = torch.softmax(gathered, dim=-1)
        else:
            router_scores = torch.sigmoid(logits)
            balance_probs = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            gathered = router_scores.gather(-1, topk_indices)
            weights = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_load = balance_probs.mean(dim=(0, 1))
        uniform = expert_load.new_full(expert_load.shape, 1.0 / float(self.num_experts))
        aux_loss = self.num_experts * torch.mean((expert_load - uniform).pow(2))
        if self.track_route_metrics:
            self._record_route_counts(topk_indices)
        return topk_indices, weights.to(dtype=latent_tokens.dtype), aux_loss

    def _group_static_expert_ids(self, device: torch.device) -> torch.Tensor:
        if self.top_k <= 0 or self.num_experts % self.top_k != 0:
            raise ValueError(
                "group_static dispatch requires moe_num_experts to divide evenly by moe_top_k "
                f"(experts={self.num_experts}, top_k={self.top_k})."
            )
        num_groups = self.num_experts // self.top_k
        group_ids = torch.arange(num_groups, device=device, dtype=torch.long)
        slot_ids = torch.arange(self.top_k, device=device, dtype=torch.long)
        group_offset = int(self.layer_idx) % max(1, num_groups)
        expert_groups = torch.remainder(group_ids + group_offset, num_groups)
        return (expert_groups.view(num_groups, 1) * self.top_k) + slot_ids.view(1, self.top_k)

    def _route_group_static(
        self,
        latent_tokens: torch.Tensor,
        hidden_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.router_override != "learned":
            raise ValueError("group_static routing requires --router-override learned.")
        router_tokens = latent_tokens
        if self.balanced_static_router_input == "hidden":
            if hidden_tokens is None:
                raise RuntimeError("group_static hidden router input requires hidden tokens.")
            router_tokens = hidden_tokens
        batch_size, seq_len, _hidden = router_tokens.shape
        expert_ids_by_group = self._group_static_expert_ids(router_tokens.device)
        num_groups = int(expert_ids_by_group.shape[0])
        group_offset = int(self.layer_idx) % max(1, num_groups)
        slot_ids = torch.arange(self.top_k, device=router_tokens.device, dtype=torch.long)

        logits = self.router(router_tokens).float() / self.router_temperature
        if self.router_score == "softmax":
            probs = torch.softmax(logits, dim=-1)
            grouped_logits = torch.roll(
                logits.view(batch_size, seq_len, num_groups, self.top_k),
                shifts=-group_offset,
                dims=2,
            )
            selection_scores = grouped_logits
            if self.balance_strategy == "aux_loss_free_bias":
                grouped_bias = torch.roll(
                    self.balance_bias.view(num_groups, self.top_k).to(logits.dtype),
                    shifts=-group_offset,
                    dims=0,
                )
                selection_scores = selection_scores + grouped_bias.view(1, 1, num_groups, self.top_k)
            group_scores = selection_scores.mean(dim=-1)
            group_ids = torch.argmax(group_scores, dim=-1)
            gathered_logits = torch.gather(
                grouped_logits,
                2,
                group_ids.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 1, self.top_k),
            )
            weights = torch.softmax(gathered_logits.squeeze(2), dim=-1)
        else:
            router_scores = torch.sigmoid(logits)
            grouped_scores = torch.roll(
                router_scores.view(batch_size, seq_len, num_groups, self.top_k),
                shifts=-group_offset,
                dims=2,
            )
            selection_scores = grouped_scores
            if self.balance_strategy == "aux_loss_free_bias":
                grouped_bias = torch.roll(
                    self.balance_bias.view(num_groups, self.top_k).to(router_scores.dtype),
                    shifts=-group_offset,
                    dims=0,
                )
                selection_scores = selection_scores + grouped_bias.view(1, 1, num_groups, self.top_k)
            group_scores = selection_scores.mean(dim=-1)
            group_ids = torch.argmax(group_scores, dim=-1)
            gathered_scores = torch.gather(
                grouped_scores,
                2,
                group_ids.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 1, self.top_k),
            )
            gathered_scores = gathered_scores.squeeze(2)
            weights = gathered_scores / gathered_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            probs = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        grouped_probs = torch.roll(
            probs.view(batch_size, seq_len, num_groups, self.top_k),
            shifts=-group_offset,
            dims=2,
        ).sum(dim=-1)
        group_load = grouped_probs.mean(dim=(0, 1))
        uniform = group_load.new_full(group_load.shape, 1.0 / float(num_groups))
        aux_loss = num_groups * torch.mean((group_load - uniform).pow(2))
        picked_expert_groups = torch.remainder(group_ids.to(torch.long) + group_offset, num_groups)
        picked_expert_ids = picked_expert_groups.unsqueeze(-1) * int(self.top_k) + slot_ids.view(1, 1, self.top_k)
        if self.track_route_metrics:
            self._record_route_counts(picked_expert_ids)
        self._maybe_update_balance_bias(picked_expert_ids)
        return (
            group_ids.reshape(batch_size * seq_len),
            picked_expert_ids,
            weights.to(dtype=latent_tokens.dtype),
            aux_loss,
        )

    def _select_for_rank(
        self,
        flat_hidden: torch.Tensor,
        flat_weights: torch.Tensor,
        flat_owners: torch.Tensor,
        flat_local_experts: torch.Tensor,
        flat_source_rows: torch.Tensor,
        flat_assignment_ids: torch.Tensor,
        dest_rank: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = flat_owners == int(dest_rank)
        valid_float = valid.to(dtype=torch.float32)
        slots_float = torch.cumsum(valid_float, dim=0) - 1.0
        selected_valid = (valid & (slots_float < float(capacity))).to(dtype=flat_hidden.dtype)
        slots = slots_float.clamp(min=0.0, max=float(capacity - 1)).to(torch.long)

        if self.dispatch_pack_impl == "gather":
            assignment_count = int(flat_hidden.shape[0])
            positions = torch.arange(assignment_count, device=flat_hidden.device, dtype=torch.long)
            invalid_keys = assignment_count + positions
            sort_keys = torch.where(valid, positions, invalid_keys)
            picked = torch.argsort(sort_keys, dim=0)[:capacity]
            picked_keys = torch.gather(sort_keys, 0, picked)
            valid_packed = (picked_keys < assignment_count).to(dtype=flat_weights.dtype)
            hidden = torch.gather(
                flat_hidden,
                0,
                picked.view(capacity, 1).expand(capacity, int(flat_hidden.shape[-1])),
            ) * valid_packed.to(dtype=flat_hidden.dtype).unsqueeze(-1)
            weights = torch.gather(flat_weights, 0, picked) * valid_packed
            local_experts_float = (
                torch.gather(flat_local_experts, 0, picked).to(dtype=torch.float32)
                * valid_packed.to(dtype=torch.float32)
            )
            source_rows_float = (
                torch.gather(flat_source_rows, 0, picked).to(dtype=torch.float32)
                * valid_packed.to(dtype=torch.float32)
            )
            assignment_ids_float = (
                torch.gather(flat_assignment_ids, 0, picked).to(dtype=torch.float32)
                * valid_packed.to(dtype=torch.float32)
            )
        elif self.dispatch_pack_impl in {"one_hot", "one_hot_gather"}:
            slot_ids = torch.arange(capacity, device=flat_hidden.device, dtype=slots.dtype).view(capacity, 1)
            selector_f32 = ((slots.view(1, -1) == slot_ids).to(torch.float32) * selected_valid.view(1, -1).to(torch.float32))
            hidden = torch.matmul(selector_f32.to(dtype=flat_hidden.dtype), flat_hidden)
            weights = torch.matmul(selector_f32.to(dtype=flat_weights.dtype), flat_weights.unsqueeze(-1)).squeeze(-1)
            local_experts_float = torch.matmul(
                selector_f32,
                flat_local_experts.to(dtype=torch.float32).unsqueeze(-1),
            ).squeeze(-1)
            source_rows_float = torch.matmul(
                selector_f32,
                flat_source_rows.to(dtype=torch.float32).unsqueeze(-1),
            ).squeeze(-1)
            assignment_ids_float = torch.matmul(
                selector_f32,
                flat_assignment_ids.to(dtype=torch.float32).unsqueeze(-1),
            ).squeeze(-1)
            valid_packed = selector_f32.sum(dim=-1).to(dtype=flat_weights.dtype)
        else:
            hidden = flat_hidden.new_zeros((capacity, flat_hidden.shape[-1]))
            hidden = hidden.index_add(0, slots, flat_hidden * selected_valid.unsqueeze(-1))
            weights = flat_weights.new_zeros((capacity,))
            weights = weights.index_add(0, slots, flat_weights * selected_valid)
            local_experts_float = flat_hidden.new_zeros((capacity,))
            local_experts_float = local_experts_float.index_add(
                0,
                slots,
                flat_local_experts.to(dtype=flat_hidden.dtype) * selected_valid,
            )
            source_rows_float = flat_hidden.new_zeros((capacity,))
            source_rows_float = source_rows_float.index_add(
                0,
                slots,
                flat_source_rows.to(dtype=flat_hidden.dtype) * selected_valid,
            )
            assignment_ids_float = flat_hidden.new_zeros((capacity,))
            assignment_ids_float = assignment_ids_float.index_add(
                0,
                slots,
                flat_assignment_ids.to(dtype=flat_hidden.dtype) * selected_valid,
            )
            valid_packed = flat_weights.new_zeros((capacity,))
            valid_packed = valid_packed.index_add(0, slots, selected_valid)
        return (
            hidden,
            weights,
            local_experts_float.to(torch.int32),
            source_rows_float.to(torch.long),
            assignment_ids_float.to(torch.long),
            valid_packed.clamp(max=1.0),
        )

    def _dispatch_static_ep(
        self,
        latent_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_rows, latent_dim = latent_states.shape
        assignments = num_rows * self.top_k
        capacity = self._capacity(assignments)
        flat_hidden = latent_states.unsqueeze(1).expand(num_rows, self.top_k, latent_dim).reshape(assignments, latent_dim)
        flat_weights = topk_weights.reshape(assignments)
        flat_experts = topk_indices.reshape(assignments).to(torch.long)
        flat_owners = torch.div(flat_experts, self.local_num_experts, rounding_mode="floor")
        flat_owners = flat_owners.clamp(min=0, max=max(0, self.runtime.world_size - 1))
        flat_local_experts = flat_experts - (flat_owners * self.local_num_experts)
        flat_assignment_ids = torch.arange(assignments, device=latent_states.device, dtype=torch.long)
        flat_source_rows = (
            flat_assignment_ids // self.top_k
        )

        if self.dispatch_pack_impl == "sort_pack":
            flat_assignment_ids_i32 = torch.arange(assignments, device=latent_states.device, dtype=torch.int32)
            flat_source_rows_i32 = flat_assignment_ids_i32 // self.top_k
            positions_i32 = torch.arange(assignments, device=latent_states.device, dtype=torch.int32)
            flat_owners_i32 = flat_owners.to(torch.int32)
            sort_keys = flat_owners_i32 * assignments + positions_i32
            order = torch.argsort(sort_keys, dim=0)
            sorted_positions_i32 = torch.gather(positions_i32, 0, order)
            if self.runtime.is_xla:
                owner_counts = F.one_hot(
                    flat_owners.to(torch.long),
                    num_classes=self.runtime.world_size,
                ).to(torch.float32).sum(dim=0).to(torch.int32)
            else:
                owner_counts = torch.bincount(
                    flat_owners.to(torch.long),
                    minlength=self.runtime.world_size,
                ).to(torch.int32)
            owner_starts = torch.cumsum(owner_counts, dim=0) - owner_counts
            slot_ids = torch.arange(capacity, device=latent_states.device, dtype=torch.int32).view(1, capacity)
            picked_offsets = owner_starts.view(self.runtime.world_size, 1) + slot_ids
            picked_offsets = picked_offsets.clamp(min=0, max=max(0, assignments - 1))
            picked = torch.gather(
                sorted_positions_i32,
                0,
                picked_offsets.reshape(-1).to(torch.long),
            ).view(self.runtime.world_size, capacity)
            send_valid_tensor = (slot_ids < owner_counts.view(self.runtime.world_size, 1)).to(dtype=flat_weights.dtype)
            picked_flat = picked.reshape(-1).to(torch.long)
            flat_valid = send_valid_tensor.reshape(-1)
            send_hidden_tensor = torch.gather(
                flat_hidden,
                0,
                picked_flat.view(-1, 1).expand(self.runtime.world_size * capacity, latent_dim),
            ).view(self.runtime.world_size, capacity, latent_dim)
            send_hidden_tensor = (send_hidden_tensor * flat_valid.to(dtype=flat_hidden.dtype).view(
                self.runtime.world_size,
                capacity,
                1,
            )).contiguous()
            send_weights_tensor = (
                torch.gather(flat_weights, 0, picked_flat).view(self.runtime.world_size, capacity)
                * send_valid_tensor
            ).contiguous()
            send_local_tensor = (
                torch.gather(flat_local_experts, 0, picked_flat).view(self.runtime.world_size, capacity)
                * send_valid_tensor.to(dtype=flat_local_experts.dtype)
            ).to(torch.int32).contiguous()
            send_source_tensor = (
                torch.gather(flat_source_rows_i32, 0, picked_flat).view(self.runtime.world_size, capacity)
                * send_valid_tensor.to(dtype=torch.int32)
            ).to(torch.int32).contiguous()
            send_assignment_tensor = (
                torch.gather(flat_assignment_ids_i32, 0, picked_flat).view(self.runtime.world_size, capacity)
                * send_valid_tensor.to(dtype=torch.int32)
            ).to(torch.int32).contiguous()
        else:
            send_hidden: list[torch.Tensor] = []
            send_weights: list[torch.Tensor] = []
            send_local_experts: list[torch.Tensor] = []
            send_source_rows: list[torch.Tensor] = []
            send_assignment_ids: list[torch.Tensor] = []
            send_valid: list[torch.Tensor] = []
            for dest_rank in range(self.runtime.world_size):
                hidden, weights, local_experts, source_rows, assignment_ids, valid = self._select_for_rank(
                    flat_hidden,
                    flat_weights,
                    flat_owners,
                    flat_local_experts,
                    flat_source_rows,
                    flat_assignment_ids,
                    dest_rank,
                    capacity,
                )
                send_hidden.append(hidden)
                send_weights.append(weights)
                send_local_experts.append(local_experts)
                send_source_rows.append(source_rows)
                send_assignment_ids.append(assignment_ids)
                send_valid.append(valid)

            send_hidden_tensor = torch.stack(send_hidden, dim=0).contiguous()
            send_weights_tensor = torch.stack(send_weights, dim=0).contiguous()
            send_local_tensor = torch.stack(send_local_experts, dim=0).to(torch.int32).contiguous()
            send_valid_tensor = torch.stack(send_valid, dim=0).contiguous()
            send_source_tensor = torch.stack(send_source_rows, dim=0).contiguous()
            send_assignment_tensor = torch.stack(send_assignment_ids, dim=0).contiguous()
        self.last_capacity = torch.tensor(float(capacity), device=latent_states.device)
        self.last_total_assignments = torch.tensor(float(assignments), device=latent_states.device)
        if self.runtime.is_xla:
            self.last_dest_counts = F.one_hot(
                flat_owners.to(torch.long),
                num_classes=self.runtime.world_size,
            ).to(torch.float32).sum(dim=0)
        else:
            self.last_dest_counts = torch.bincount(
                flat_owners.to(torch.long),
                minlength=self.runtime.world_size,
            ).to(dtype=torch.float32)
        self.last_dest_valid = send_valid_tensor.sum(dim=1).to(dtype=torch.float32)

        recv_hidden = xla_all_to_all(self.runtime, send_hidden_tensor)
        recv_weights = xla_all_to_all(self.runtime, send_weights_tensor)
        recv_local = xla_all_to_all(self.runtime, send_local_tensor)
        recv_valid = xla_all_to_all(self.runtime, send_valid_tensor)
        flat_recv = recv_hidden.reshape(self.runtime.world_size * capacity, latent_dim)
        flat_recv_weights = recv_weights.reshape(self.runtime.world_size * capacity)
        flat_recv_local = recv_local.reshape(self.runtime.world_size * capacity)
        flat_recv_valid = recv_valid.reshape(self.runtime.world_size * capacity)

        flat_out = flat_recv.new_zeros(flat_recv.shape)
        for local_idx, expert in enumerate(self.experts):
            expert_out = expert(flat_recv)
            mask = ((flat_recv_local == local_idx).to(dtype=flat_recv.dtype) * flat_recv_valid * flat_recv_weights)
            flat_out = flat_out + expert_out * mask.unsqueeze(-1)

        recv_out = flat_out.view(self.runtime.world_size, capacity, latent_dim).contiguous()
        send_back = xla_all_to_all(self.runtime, recv_out)
        flat_send_back = send_back.reshape(-1, latent_dim)
        if self.dispatch_pack_impl == "sort_pack":
            source_rows = send_source_tensor.reshape(-1).to(torch.long)
            flat_valid = send_valid_tensor.reshape(-1).to(dtype=flat_send_back.dtype)
            routed = latent_states.new_zeros((num_rows, latent_dim))
            routed = routed.index_add(0, source_rows, flat_send_back * flat_valid.unsqueeze(-1))
        elif self.dispatch_pack_impl in {"gather", "one_hot_gather"}:
            flat_assignment_ids = send_assignment_tensor.reshape(-1).to(torch.int32)
            flat_valid = send_valid_tensor.reshape(-1) > 0.5
            slot_count = int(flat_assignment_ids.shape[0])
            invalid_keys = (
                assignments + torch.arange(slot_count, device=latent_states.device, dtype=torch.int32)
            ).to(torch.int32)
            sort_keys = torch.where(flat_valid, flat_assignment_ids, invalid_keys)
            order = torch.argsort(sort_keys, dim=0)
            sorted_keys = torch.gather(sort_keys, 0, order)
            sorted_out = torch.gather(
                flat_send_back,
                0,
                order.view(slot_count, 1).expand(slot_count, latent_dim),
            )
            query = torch.arange(assignments, device=latent_states.device, dtype=torch.int32)
            found_pos = torch.searchsorted(sorted_keys, query, right=False)
            found_pos = found_pos.clamp(min=0, max=max(0, slot_count - 1)).to(torch.long)
            found_keys = torch.gather(sorted_keys, 0, found_pos)
            found = (found_keys == query).to(dtype=latent_states.dtype)
            assignment_out = torch.gather(
                sorted_out,
                0,
                found_pos.view(assignments, 1).expand(assignments, latent_dim),
            ) * found.unsqueeze(-1)
            routed = assignment_out.view(num_rows, self.top_k, latent_dim).sum(dim=1)
        elif self.dispatch_pack_impl == "one_hot":
            source_rows = send_source_tensor.reshape(-1)
            row_ids = torch.arange(num_rows, device=latent_states.device, dtype=source_rows.dtype).view(num_rows, 1)
            combine = (source_rows.view(1, -1) == row_ids).to(dtype=latent_states.dtype)
            routed = torch.matmul(combine, flat_send_back)
        else:
            routed = latent_states.new_zeros((num_rows, latent_dim))
            routed = routed.index_add(0, send_source_tensor.reshape(-1), flat_send_back)
        self.last_valid_assignments = send_valid_tensor.sum()
        self.last_dropped_assignments = self.last_total_assignments - self.last_valid_assignments
        return routed

    def _dispatch_force_balanced_static_indexed(
        self,
        latent_states: torch.Tensor,
        flat_assignment_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.router_override != "force_balanced":
            raise ValueError("balanced_static dispatch requires --router-override force_balanced.")
        if self.local_num_experts != 1 or self.runtime.world_size != self.num_experts:
            raise ValueError(
                "balanced_static dispatch currently requires one local expert per rank "
                f"(world_size={self.runtime.world_size}, num_experts={self.num_experts}, "
                f"local_num_experts={self.local_num_experts})."
            )
        num_rows, latent_dim = latent_states.shape
        assignments = num_rows * self.top_k
        if assignments % self.runtime.world_size != 0:
            raise ValueError(
                "balanced_static dispatch requires total assignments to divide evenly across ranks "
                f"(assignments={assignments}, world_size={self.runtime.world_size})."
            )

        capacity = assignments // self.runtime.world_size
        world = self.runtime.world_size
        device = latent_states.device
        dest_ids = torch.arange(world, device=device, dtype=torch.long).view(world, 1)
        slot_ids = torch.arange(capacity, device=device, dtype=torch.long).view(1, capacity)
        route_offset = int(self.balanced_static_route_offset) % world
        assignment_base = torch.remainder(dest_ids - route_offset, world)
        assignment_by_dest = assignment_base + (slot_ids * world)
        source_rows = torch.div(assignment_by_dest.reshape(-1), self.top_k, rounding_mode="floor")
        send_hidden = latent_states.index_select(0, source_rows).view(world, capacity, latent_dim).contiguous()

        recv_hidden = xla_all_to_all(self.runtime, send_hidden)
        flat_recv = recv_hidden.reshape(world * capacity, latent_dim)
        flat_out = self.experts[0](flat_recv)
        recv_out = flat_out.view(world, capacity, latent_dim).contiguous()
        send_back = xla_all_to_all(self.runtime, recv_out)

        flat_send_back = send_back.reshape(world * capacity, latent_dim)
        assignment_ids = torch.arange(assignments, device=device, dtype=torch.long)
        expert_destinations = torch.remainder(assignment_ids + route_offset, world)
        packed_positions = expert_destinations * capacity + torch.div(
            assignment_ids,
            world,
            rounding_mode="floor",
        )
        assignment_out = flat_send_back.index_select(0, packed_positions)
        if flat_assignment_weights is None:
            assignment_out = assignment_out * (1.0 / float(max(1, self.top_k)))
        else:
            assignment_out = assignment_out * flat_assignment_weights.to(dtype=assignment_out.dtype).view(assignments, 1)
        routed = assignment_out.view(num_rows, self.top_k, latent_dim).sum(dim=1)

        if self.track_route_metrics:
            total = torch.tensor(float(assignments), device=device)
            per_dest = torch.full((world,), float(capacity), device=device, dtype=torch.float32)
            self.last_capacity = torch.tensor(float(capacity), device=device)
            self.last_total_assignments = total
            self.last_valid_assignments = total
            self.last_dropped_assignments = total.new_zeros(())
            self.last_dest_counts = per_dest
            self.last_dest_valid = per_dest
            self.last_route_counts = per_dest.clone()
        return routed

    def _dispatch_force_balanced_static_strided(
        self,
        latent_states: torch.Tensor,
        flat_assignment_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.router_override != "force_balanced":
            raise ValueError("balanced_static dispatch requires --router-override force_balanced.")
        if self.local_num_experts != 1 or self.runtime.world_size != self.num_experts:
            raise ValueError(
                "balanced_static dispatch currently requires one local expert per rank "
                f"(world_size={self.runtime.world_size}, num_experts={self.num_experts}, "
                f"local_num_experts={self.local_num_experts})."
            )
        if (
            self.runtime.world_size % self.top_k != 0
        ):
            return self._dispatch_force_balanced_static_indexed(latent_states, flat_assignment_weights)

        num_rows, latent_dim = latent_states.shape
        assignments = num_rows * self.top_k
        if assignments % self.runtime.world_size != 0:
            raise ValueError(
                "balanced_static dispatch requires total assignments to divide evenly across ranks "
                f"(assignments={assignments}, world_size={self.runtime.world_size})."
            )

        world = self.runtime.world_size
        capacity = assignments // world
        rows_per_capacity_slot = world // self.top_k
        if num_rows != capacity * rows_per_capacity_slot:
            return self._dispatch_force_balanced_static_indexed(latent_states)

        grouped = latent_states.view(capacity, rows_per_capacity_slot, latent_dim)
        send_hidden = (
            grouped.transpose(0, 1)
            .unsqueeze(1)
            .expand(rows_per_capacity_slot, self.top_k, capacity, latent_dim)
            .reshape(world, capacity, latent_dim)
            .contiguous()
        )
        route_offset = int(self.balanced_static_route_offset) % world
        if route_offset:
            send_hidden = torch.roll(send_hidden, shifts=route_offset, dims=0)

        recv_hidden = xla_all_to_all(self.runtime, send_hidden)
        flat_recv = recv_hidden.reshape(world * capacity, latent_dim)
        flat_out = self.experts[0](flat_recv)
        recv_out = flat_out.view(world, capacity, latent_dim).contiguous()
        send_back = xla_all_to_all(self.runtime, recv_out)
        if route_offset:
            send_back = torch.roll(send_back, shifts=-route_offset, dims=0)

        assignment_out = send_back.transpose(0, 1).reshape(num_rows, self.top_k, latent_dim)
        if flat_assignment_weights is None:
            assignment_out = assignment_out * (1.0 / float(max(1, self.top_k)))
        else:
            assignment_out = assignment_out * flat_assignment_weights.to(dtype=assignment_out.dtype).view(
                num_rows,
                self.top_k,
                1,
            )
        routed = assignment_out.sum(dim=1)

        if self.track_route_metrics:
            total = torch.tensor(float(assignments), device=latent_states.device)
            per_dest = torch.full((world,), float(capacity), device=latent_states.device, dtype=torch.float32)
            self.last_capacity = torch.tensor(float(capacity), device=latent_states.device)
            self.last_total_assignments = total
            self.last_valid_assignments = total
            self.last_dropped_assignments = total.new_zeros(())
            self.last_dest_counts = per_dest
            self.last_dest_valid = per_dest
            self.last_route_counts = per_dest.clone()
        return routed

    def _dispatch_force_balanced_static(
        self,
        latent_states: torch.Tensor,
        flat_assignment_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.balanced_static_layout == "strided":
            return self._dispatch_force_balanced_static_strided(latent_states, flat_assignment_weights)
        if self.balanced_static_layout == "indexed":
            return self._dispatch_force_balanced_static_indexed(latent_states, flat_assignment_weights)
        raise ValueError(f"Unsupported balanced static layout: {self.balanced_static_layout}")

    def _group_capacity(self, num_rows: int, num_groups: int) -> int:
        if self.explicit_capacity is not None and self.explicit_capacity > 0:
            capacity = int(self.explicit_capacity)
        else:
            per_group = math.ceil(float(num_rows) / float(max(1, num_groups)))
            capacity = int(math.ceil(per_group * max(self.capacity_factor, 1.0)))
        alignment = max(1, min(int(self.capacity_alignment), 32))
        if capacity % alignment:
            capacity += alignment - (capacity % alignment)
        return max(1, min(capacity, num_rows))

    def _dispatch_group_static(
        self,
        latent_states: torch.Tensor,
        flat_group_ids: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.router_override != "learned":
            raise ValueError("group_static dispatch requires --router-override learned.")
        if self.local_num_experts != 1 or self.runtime.world_size != self.num_experts:
            raise ValueError(
                "group_static dispatch currently requires one expert per rank "
                f"(world_size={self.runtime.world_size}, num_experts={self.num_experts}, "
                f"local_num_experts={self.local_num_experts})."
            )
        if self.runtime.world_size % self.top_k != 0:
            raise ValueError(
                "group_static dispatch requires world_size to divide evenly by top_k "
                f"(world_size={self.runtime.world_size}, top_k={self.top_k})."
            )

        num_rows, latent_dim = latent_states.shape
        world = self.runtime.world_size
        num_groups = world // self.top_k
        capacity = self._group_capacity(num_rows, num_groups)
        device = latent_states.device
        row_ids = torch.arange(num_rows, device=device, dtype=torch.long)
        slot_ids = torch.arange(capacity, device=device, dtype=torch.long).view(capacity, 1)
        expert_ids_by_group = self._group_static_expert_ids(device)

        send_hidden: list[torch.Tensor | None] = [None] * world
        send_weights: list[torch.Tensor | None] = [None] * world
        group_counts: list[torch.Tensor] = []
        group_valid_counts: list[torch.Tensor] = []
        group_offset = int(self.layer_idx) % max(1, num_groups)
        for group_idx in range(num_groups):
            valid = flat_group_ids == int(group_idx)
            valid_float = valid.to(dtype=torch.float32)
            if self.dispatch_pack_impl == "group_static_gather":
                invalid_keys = num_rows + row_ids
                sort_keys = torch.where(valid, row_ids, invalid_keys)
                picked = torch.argsort(sort_keys, dim=0)[:capacity]
                picked_keys = torch.gather(sort_keys, 0, picked)
                group_valid = (picked_keys < num_rows).to(dtype=latent_states.dtype)
                group_hidden = torch.gather(
                    latent_states,
                    0,
                    picked.view(capacity, 1).expand(capacity, latent_dim),
                ) * group_valid.unsqueeze(-1)
            else:
                slots_float = torch.cumsum(valid_float, dim=0) - 1.0
                selected_valid = (valid & (slots_float < float(capacity))).to(dtype=latent_states.dtype)
                slots = slots_float.clamp(min=0.0, max=float(capacity - 1)).to(torch.long)
                selector_f32 = (
                    (slots.view(1, -1) == slot_ids).to(torch.float32)
                    * selected_valid.view(1, -1).to(torch.float32)
                )
                group_hidden = torch.matmul(selector_f32.to(dtype=latent_states.dtype), latent_states)
                group_valid = selector_f32.sum(dim=-1).to(dtype=latent_states.dtype)
            group_counts.append(valid_float.sum().to(dtype=torch.float32))
            group_valid_counts.append(group_valid.to(dtype=torch.float32).sum())
            for expert_slot in range(self.top_k):
                dest_rank = ((group_idx + group_offset) % num_groups) * self.top_k + expert_slot
                send_hidden[dest_rank] = group_hidden
                if self.dispatch_pack_impl == "group_static_gather":
                    send_weights[dest_rank] = torch.gather(
                        flat_expert_weights[:, expert_slot],
                        0,
                        picked,
                    ) * group_valid.to(dtype=flat_expert_weights.dtype)
                else:
                    send_weights[dest_rank] = torch.matmul(
                        selector_f32.to(dtype=flat_expert_weights.dtype),
                        flat_expert_weights[:, expert_slot].unsqueeze(-1),
                    ).squeeze(-1)

        if any(value is None for value in send_hidden):
            raise RuntimeError("group_static internal error: not every destination rank received a packet.")
        send_hidden_tensor = torch.stack([value for value in send_hidden if value is not None], dim=0).contiguous()
        send_weights_tensor = torch.stack([value for value in send_weights if value is not None], dim=0).contiguous()

        recv_hidden = xla_all_to_all(self.runtime, send_hidden_tensor)
        recv_weights = xla_all_to_all(self.runtime, send_weights_tensor)
        flat_recv = recv_hidden.reshape(world * capacity, latent_dim)
        flat_recv_weights = recv_weights.reshape(world * capacity)
        flat_out = self.experts[0](flat_recv) * flat_recv_weights.to(dtype=flat_recv.dtype).unsqueeze(-1)
        recv_out = flat_out.view(world, capacity, latent_dim).contiguous()
        send_back = xla_all_to_all(self.runtime, recv_out)

        flat_send_back = send_back.reshape(world * capacity, latent_dim)
        row_slot = latent_states.new_zeros((num_rows,), dtype=torch.float32)
        row_valid = latent_states.new_zeros((num_rows,), dtype=torch.float32)
        for group_idx in range(num_groups):
            valid = flat_group_ids == int(group_idx)
            valid_float = valid.to(dtype=torch.float32)
            slots_float = torch.cumsum(valid_float, dim=0) - 1.0
            in_capacity = (valid & (slots_float < float(capacity))).to(dtype=torch.float32)
            row_slot = torch.where(valid, slots_float, row_slot)
            row_valid = torch.where(valid, in_capacity, row_valid)

        row_slot_long = row_slot.clamp(min=0.0, max=float(capacity - 1)).to(torch.long)
        expert_slot_ids = torch.arange(self.top_k, device=device, dtype=torch.long).view(1, self.top_k)
        dest_base = (
            torch.remainder(flat_group_ids.to(torch.long) + int(group_offset), int(num_groups))
            * int(self.top_k)
        )
        gather_indices = (
            (dest_base.view(num_rows, 1) + expert_slot_ids) * int(capacity)
            + row_slot_long.view(num_rows, 1)
        ).reshape(num_rows * self.top_k)
        gathered = torch.gather(
            flat_send_back,
            0,
            gather_indices.view(num_rows * self.top_k, 1).expand(num_rows * self.top_k, latent_dim),
        )
        routed = (
            gathered.view(num_rows, self.top_k, latent_dim)
            * row_valid.to(dtype=latent_states.dtype).view(num_rows, 1, 1)
        ).sum(dim=1)

        if self.track_route_metrics:
            total = torch.tensor(float(num_rows * self.top_k), device=device)
            dest_counts = torch.stack(
                [value for value in group_counts for _slot in range(self.top_k)]
            ).to(device=device, dtype=torch.float32)
            dest_valid = torch.stack(
                [value for value in group_valid_counts for _slot in range(self.top_k)]
            ).to(device=device, dtype=torch.float32)
            self.last_capacity = torch.tensor(float(capacity), device=device)
            self.last_total_assignments = total
            self.last_valid_assignments = dest_valid.sum()
            self.last_dropped_assignments = total - self.last_valid_assignments
            self.last_dest_counts = dest_counts
            self.last_dest_valid = dest_valid
            self.last_route_counts = dest_counts
        return routed

    def _dispatch_dense_all_experts(
        self,
        latent_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.router_override != "learned":
            raise ValueError("dense_all_experts dispatch requires --router-override learned.")
        if self.local_num_experts != 1 or self.runtime.world_size != self.num_experts:
            raise ValueError(
                "dense_all_experts dispatch currently requires one expert per rank "
                f"(world_size={self.runtime.world_size}, num_experts={self.num_experts}, "
                f"local_num_experts={self.local_num_experts})."
            )
        num_rows, latent_dim = latent_states.shape
        world = self.runtime.world_size
        send_hidden = latent_states.unsqueeze(0).expand(world, num_rows, latent_dim).contiguous()
        recv_hidden = xla_all_to_all(self.runtime, send_hidden)
        expert_input = recv_hidden.reshape(world * num_rows, latent_dim)
        expert_out = self.experts[0](expert_input).view(world, num_rows, latent_dim).contiguous()
        gathered = xla_all_to_all(self.runtime, expert_out)
        all_expert_out = gathered.transpose(0, 1).contiguous()
        selected = torch.gather(
            all_expert_out,
            1,
            topk_indices.to(torch.long).view(num_rows, self.top_k, 1).expand(num_rows, self.top_k, latent_dim),
        )
        routed = (selected * topk_weights.to(dtype=selected.dtype).view(num_rows, self.top_k, 1)).sum(dim=1)

        if self.track_route_metrics:
            total = torch.tensor(float(num_rows * self.top_k), device=latent_states.device)
            per_dest = torch.full((world,), float(num_rows), device=latent_states.device, dtype=torch.float32)
            self.last_capacity = torch.tensor(float(num_rows), device=latent_states.device)
            self.last_total_assignments = total
            self.last_valid_assignments = total
            self.last_dropped_assignments = total.new_zeros(())
            self.last_dest_counts = per_dest
            self.last_dest_valid = per_dest
            if self.last_route_counts is None:
                self.last_route_counts = per_dest.clone()
        return routed

    def _dispatch_local(self, latent_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(latent_states)
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(latent_states)
            mask = (topk_indices == expert_idx).to(dtype=latent_states.dtype)
            weights = (topk_weights * mask).sum(dim=-1)
            out = out + expert_out * weights.unsqueeze(-1)
        self.last_valid_assignments = torch.full((), float(topk_indices.numel()), device=latent_states.device)
        self.last_capacity = torch.tensor(float(topk_indices.numel()), device=latent_states.device)
        self.last_total_assignments = torch.tensor(float(topk_indices.numel()), device=latent_states.device)
        self.last_dropped_assignments = self.last_valid_assignments.new_zeros(())
        self.last_dest_counts = self.last_valid_assignments.reshape(1).to(dtype=torch.float32)
        self.last_dest_valid = self.last_valid_assignments.reshape(1).to(dtype=torch.float32)
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe_mode == "identity":
            zero = hidden_states.new_zeros(())
            self.last_aux_loss = zero
            self.last_valid_assignments = zero
            self.last_capacity = zero
            self.last_total_assignments = zero
            self.last_dropped_assignments = zero
            self.last_route_counts = hidden_states.new_zeros((self.num_experts,), dtype=torch.float32)
            self.last_dest_counts = hidden_states.new_zeros((max(1, self.runtime.world_size),), dtype=torch.float32)
            self.last_dest_valid = hidden_states.new_zeros((max(1, self.runtime.world_size),), dtype=torch.float32)
            return torch.zeros_like(hidden_states)
        if self.moe_mode == "dense_ffn":
            zero = hidden_states.new_zeros(())
            self.last_aux_loss = zero
            self.last_valid_assignments = zero
            self.last_capacity = zero
            self.last_total_assignments = zero
            self.last_dropped_assignments = zero
            self.last_route_counts = hidden_states.new_zeros((self.num_experts,), dtype=torch.float32)
            self.last_dest_counts = hidden_states.new_zeros((max(1, self.runtime.world_size),), dtype=torch.float32)
            self.last_dest_valid = hidden_states.new_zeros((max(1, self.runtime.world_size),), dtype=torch.float32)
            return self.dense_ffn(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_size)
        latent = self.down_proj(flat_hidden)
        latent_tokens = latent.reshape(batch_size, seq_len, self.routed_dim)
        hidden_tokens = flat_hidden.reshape(batch_size, seq_len, hidden_size)
        if (
            self.moe_mode == "real"
            and self.runtime.world_size > 1
            and self.dispatch_pack_impl == "balanced_static"
            and self.router_override == "force_balanced"
        ):
            _topk_indices, topk_weights, aux_loss = self._route_static_balanced_weights(latent_tokens, hidden_tokens)
            self.last_aux_loss = aux_loss
            routed_latent = self._dispatch_force_balanced_static(latent, topk_weights.reshape(-1))
        elif (
            self.moe_mode == "real"
            and self.runtime.world_size > 1
            and self.dispatch_pack_impl in {"group_static", "group_static_gather"}
            and self.router_override == "learned"
        ):
            flat_group_ids, _topk_indices, topk_weights, aux_loss = self._route_group_static(latent_tokens, hidden_tokens)
            self.last_aux_loss = aux_loss
            routed_latent = self._dispatch_group_static(
                latent,
                flat_group_ids,
                topk_weights.reshape(batch_size * seq_len, self.top_k),
            )
        else:
            topk_indices, topk_weights, aux_loss = self._route(latent_tokens, hidden_tokens)
            self.last_aux_loss = aux_loss
            flat_topk_indices = topk_indices.reshape(batch_size * seq_len, self.top_k)
            flat_topk_weights = topk_weights.reshape(batch_size * seq_len, self.top_k)
            if (
                self.moe_mode == "real"
                and self.runtime.world_size > 1
                and self.dispatch_pack_impl == "dense_all_experts"
                and self.router_override == "learned"
            ):
                routed_latent = self._dispatch_dense_all_experts(latent, flat_topk_indices, flat_topk_weights)
            elif self.moe_mode == "local_only":
                routed_latent = self._dispatch_local(latent, flat_topk_indices % self.local_num_experts, flat_topk_weights)
            elif self.runtime.world_size > 1:
                routed_latent = self._dispatch_static_ep(latent, flat_topk_indices, flat_topk_weights)
            else:
                routed_latent = self._dispatch_local(latent, flat_topk_indices, flat_topk_weights)
        output = torch.zeros_like(flat_hidden)
        if self.shared:
            shared = torch.zeros_like(flat_hidden)
            for expert in self.shared:
                shared = shared + expert(flat_hidden)
            output = output + shared / float(len(self.shared))
        output = output + self.up_proj(routed_latent)
        return output.reshape(batch_size, seq_len, hidden_size)


class XlaMetisBlock(nn.Module):
    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        runtime: Runtime,
        layer_idx: int,
        capacity_factor: float,
        capacity: int | None,
        dispatch_pack_impl: str,
        router_override: str,
        attention_mode: str,
        attention_kernel: str,
        qk_clip_enabled: bool,
        activation_checkpointing: str,
        moe_mode: str,
        track_route_metrics: bool,
        balanced_static_layout: str,
        balanced_static_router_weights: str,
        balanced_static_router_input: str,
        expert_activation_safety: str,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.attention_mode = attention_mode
        self.activation_checkpointing = activation_checkpointing
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = XlaSelfAttention(
            config,
            attention_kernel=attention_kernel,
            qk_clip_enabled=qk_clip_enabled,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = StaticExpertParallelMoE(
            config,
            runtime=runtime,
            layer_idx=layer_idx,
            capacity_factor=capacity_factor,
            capacity=capacity,
            dispatch_pack_impl=dispatch_pack_impl,
            router_override=router_override,
            moe_mode=moe_mode,
            track_route_metrics=track_route_metrics,
            balanced_static_layout=balanced_static_layout,
            balanced_static_router_weights=balanced_static_router_weights,
            balanced_static_router_input=balanced_static_router_input,
            expert_activation_safety=expert_activation_safety,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.attention_mode == "real":
            attn_input = self.attn_norm(hidden_states)
            if self.activation_checkpointing in {"attention", "attention_moe"} and self.training:
                attn_out = activation_checkpoint(self.runtime, self.attn, attn_input)
            else:
                attn_out = self.attn(attn_input)
            hidden_states = hidden_states + attn_out
        moe_input = self.ffn_norm(hidden_states)
        if self.activation_checkpointing in {"moe", "attention_moe"} and self.training:
            moe_out = activation_checkpoint(self.runtime, self.moe, moe_input)
        else:
            moe_out = self.moe(moe_input)
        hidden_states = hidden_states + moe_out
        return hidden_states


class XlaMetisForCausalLM(nn.Module):
    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        runtime: Runtime,
        capacity_factor: float,
        capacity: int | None,
        dispatch_pack_impl: str,
        router_override: str,
        attention_mode: str,
        attention_kernel: str,
        qk_clip_enabled: bool,
        moe_mode: str,
        loss_mode: str,
        ce_logits_dtype: str,
        ce_impl: str,
        activation_checkpointing: str,
        activation_checkpoint_layer_interval: int,
        track_route_metrics: bool,
        balanced_static_layout: str,
        balanced_static_router_weights: str,
        balanced_static_router_input: str,
        expert_activation_safety: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.loss_mode = loss_mode
        self.ce_logits_dtype = ce_logits_dtype
        self.ce_impl = ce_impl
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpoint_layer_interval = max(1, int(activation_checkpoint_layer_interval))
        self.balanced_static_router_weights = balanced_static_router_weights
        self.fast_balanced_static_metrics = (
            not track_route_metrics
            and dispatch_pack_impl == "balanced_static"
            and router_override == "force_balanced"
        )
        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                XlaMetisBlock(
                    config,
                    runtime=runtime,
                    layer_idx=layer_idx,
                    capacity_factor=capacity_factor,
                    capacity=capacity,
                    dispatch_pack_impl=dispatch_pack_impl,
                    router_override=router_override,
                    attention_mode=attention_mode,
                    attention_kernel=attention_kernel,
                    qk_clip_enabled=qk_clip_enabled,
                    activation_checkpointing=activation_checkpointing,
                    moe_mode=moe_mode,
                    track_route_metrics=track_route_metrics,
                    balanced_static_layout=balanced_static_layout,
                    balanced_static_router_weights=balanced_static_router_weights,
                    balanced_static_router_input=balanced_static_router_input,
                    expert_activation_safety=expert_activation_safety,
                )
                for layer_idx in range(config.n_layer)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.padded_vocab_size, bias=False)
        self.apply(self._init_weights)
        for layer in self.layers:
            layer.moe.reset_local_experts(config.initializer_range, config.moe_expert_parallel_init_seed)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def reset_qk_clip_stats(self) -> None:
        for layer in self.layers:
            layer.attn.reset_qk_clip_stats()

    def qk_clip_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            params.append(layer.attn.qkv_proj.weight)
            if layer.attn.qkv_proj.bias is not None:
                params.append(layer.attn.qkv_proj.bias)
        return params

    @torch.no_grad()
    def apply_qk_clip(self, *, threshold: float, alpha: float, runtime: Runtime) -> torch.Tensor:
        if threshold <= 0:
            return torch.zeros(3, device=runtime.device, dtype=torch.float32)
        threshold_t = torch.tensor(float(threshold), device=runtime.device, dtype=torch.float32)
        total_scaled = torch.zeros((), device=runtime.device, dtype=torch.float32)
        global_max = torch.zeros((), device=runtime.device, dtype=torch.float32)
        min_scale = torch.ones((), device=runtime.device, dtype=torch.float32)
        for layer in self.layers:
            attn = layer.attn
            max_logits = attn.qk_clip_max_logits.to(device=runtime.device, dtype=torch.float32)
            if runtime.world_size > 1:
                max_logits = xla_reduce_tensor(runtime, max_logits, reduce_type="max", scale=1.0)
            eta = torch.clamp(threshold_t / max_logits.clamp_min(threshold_t), max=1.0)
            query_scales = torch.pow(eta, float(alpha))
            key_scales = torch.pow(
                eta.view(attn.num_kv_heads, attn.num_kv_groups).min(dim=1).values,
                1.0 - float(alpha),
            )
            q_weight = attn.qkv_proj.weight[: attn.q_dim]
            k_weight = attn.qkv_proj.weight[attn.q_dim : attn.q_dim + attn.kv_dim]
            q_bias = attn.qkv_proj.bias[: attn.q_dim] if attn.qkv_proj.bias is not None else None
            k_bias = (
                attn.qkv_proj.bias[attn.q_dim : attn.q_dim + attn.kv_dim]
                if attn.qkv_proj.bias is not None
                else None
            )
            for head_idx in range(attn.num_heads):
                row_start = head_idx * attn.head_dim
                row_end = row_start + attn.head_dim
                q_weight[row_start:row_end].mul_(query_scales[head_idx])
                if q_bias is not None:
                    q_bias[row_start:row_end].mul_(query_scales[head_idx])
            for kv_head_idx in range(attn.num_kv_heads):
                row_start = kv_head_idx * attn.head_dim
                row_end = row_start + attn.head_dim
                k_weight[row_start:row_end].mul_(key_scales[kv_head_idx])
                if k_bias is not None:
                    k_bias[row_start:row_end].mul_(key_scales[kv_head_idx])
            global_max = torch.maximum(global_max, max_logits.max())
            min_scale = torch.minimum(min_scale, torch.minimum(query_scales.min(), key_scales.min()))
            total_scaled = total_scaled + (eta < 0.9999).float().sum()
            attn.reset_qk_clip_stats()
        return torch.stack((global_max, min_scale, total_scaled))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            checkpoint_layer = (
                self.activation_checkpointing == "layers"
                and self.training
                and (layer_idx % self.activation_checkpoint_layer_interval == 0)
            )
            if checkpoint_layer:
                hidden_states = activation_checkpoint(self.runtime, layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        reference = hidden_states
        logits: torch.Tensor | None = None
        lm_loss = reference.new_zeros(())
        loss = reference.new_zeros(())
        moe_aux = reference.new_zeros(())
        if labels is not None and self.loss_mode == "dummy_loss":
            lm_loss = hidden_states.float().square().mean().to(dtype=hidden_states.dtype)
            loss = lm_loss
        elif labels is not None:
            logits = self.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :].contiguous()
            if self.ce_logits_dtype == "float32":
                shift_logits = shift_logits.float()
            elif self.ce_logits_dtype == "bfloat16":
                shift_logits = shift_logits.to(dtype=torch.bfloat16)
            elif self.ce_logits_dtype != "model":
                raise ValueError(f"Unsupported CE logits dtype: {self.ce_logits_dtype}")
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
            flat_labels = shift_labels.view(-1)
            if self.ce_impl == "cross_entropy":
                lm_loss = F.cross_entropy(flat_logits, flat_labels)
            elif self.ce_impl == "manual_logsumexp":
                row_max = torch.max(flat_logits, dim=-1).values
                centered = flat_logits - row_max.unsqueeze(-1)
                log_denom = torch.log(torch.exp(centered).sum(dim=-1))
                target_logits = torch.gather(flat_logits, 1, flat_labels.view(-1, 1)).squeeze(1)
                lm_loss = (log_denom + row_max - target_logits).float().mean()
            else:
                raise ValueError(f"Unsupported CE implementation: {self.ce_impl}")
            if self.fast_balanced_static_metrics and self.balanced_static_router_weights != "learned":
                moe_aux = lm_loss.new_zeros(())
            else:
                moe_aux_terms = [
                    layer.moe.last_aux_loss
                    for layer in self.layers
                    if layer.moe.last_aux_loss is not None
                ]
                moe_aux = torch.stack(moe_aux_terms).mean() if moe_aux_terms else lm_loss.new_zeros(())
            loss = lm_loss + (moe_aux * float(self.config.moe_aux_loss_coef))
        if self.fast_balanced_static_metrics:
            valid_assignment_total = reference.new_tensor(
                float(len(self.layers) * input_ids.numel() * int(self.config.moe_top_k)),
                dtype=torch.float32,
            )
        else:
            valid_assignments = [
                layer.moe.last_valid_assignments
                for layer in self.layers
                if layer.moe.last_valid_assignments is not None
            ]
            valid_assignment_total = torch.stack(valid_assignments).sum() if valid_assignments else reference.new_zeros(())
        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "moe_aux_loss": moe_aux,
            "valid_assignments": valid_assignment_total,
        }


def apply_overrides(config: MetisMambaConfig, args: argparse.Namespace) -> None:
    overrides = {
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_expert_intermediate_size": args.moe_expert_intermediate_size,
        "moe_routed_latent_size": args.moe_routed_latent_size,
        "moe_router_latent_size": args.moe_router_latent_size,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.tie_embeddings is not None:
        config.tie_embeddings = bool(args.tie_embeddings)
    config.training_mode = "static_dense_pretrain"
    config.mor_enabled = False
    config.mor_train_router = False
    config.mor_runtime_mode = "disabled"
    config.low_precision_mode = "none"
    config.torch_dtype = "bfloat16"
    config.attention_backend = "eager"
    config.native_gqa_attention = False
    config.te_dot_product_attention = False
    config.ffn_type = "single_latent_moe"
    config.moe_backend = "torch_bmm"
    config.moe_dispatch_mode = "bucketed"
    config.moe_expert_parallel_size = int(args.world_size_override or int(os.environ.get("WORLD_SIZE", "1")))
    config.moe_single_latent_router_input = str(args.balanced_static_router_input)
    if args.disable_moe_balance_update:
        config.moe_balance_bias_update_rate = 0.0
    if args.moe_balance_bias_update_rate is not None:
        config.moe_balance_bias_update_rate = float(args.moe_balance_bias_update_rate)
    config.moe_memory_efficient_permutation = True
    config.moe_permute_fusion = False
    config.moe_fused_combine = True
    config.validate()


def load_config(args: argparse.Namespace) -> tuple[MetisMambaConfig, dict[str, Any]]:
    manifest = json.loads(Path(args.manifest).read_text())
    config = MetisMambaConfig.from_dict(manifest["model"])
    apply_overrides(config, args)
    return config, manifest


def load_memmap(data_dir: Path) -> tuple[np.memmap, np.dtype]:
    meta = json.loads((data_dir / "meta.json").read_text())
    dtype = np.dtype(meta["dtype"])
    return np.memmap(data_dir / "train.bin", dtype=dtype, mode="r"), dtype


def get_batch(
    *,
    data: np.memmap | None,
    rng: np.random.Generator,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data is None:
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device, dtype=torch.long)
        return x, x.clone()
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("Dataset is too small for the selected block size.")
    positions = rng.integers(0, max_start, size=(batch_size,), dtype=np.int64)
    offsets = positions[:, None] + np.arange(block_size, dtype=np.int64)[None, :]
    arr = np.asarray(data[offsets], dtype=np.int64)
    x = torch.from_numpy(arr).contiguous().to(device)
    return x, x.clone()


def advance_batch_rng_for_resume(
    *,
    data: np.memmap | None,
    rng: np.random.Generator,
    batch_size: int,
    block_size: int,
    grad_accum_steps: int,
    resume_step: int,
) -> int:
    if data is None or resume_step <= 0:
        return 0
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("Dataset is too small for the selected block size.")
    total_draws = int(resume_step) * int(grad_accum_steps) * int(batch_size)
    remaining = total_draws
    chunk_size = 1_000_000
    while remaining > 0:
        draws = min(remaining, chunk_size)
        rng.integers(0, max_start, size=(draws,), dtype=np.int64)
        remaining -= draws
    return total_draws


def rank_checkpoint_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1 or rank == 0:
        return path
    return path.with_name(f"{path.stem}.rank{rank:03d}{path.suffix}")


def compact_rank_checkpoint_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return path
    return path.with_name(f"{path.stem}.rank{rank:03d}{path.suffix}")


def is_sharded_expert_state_name(name: str) -> bool:
    return ".moe.experts." in name


def tensor_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_to_cpu(item) for item in value)
    return value


def filtered_model_state_dict(model: nn.Module, *, sharded: bool) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if is_sharded_expert_state_name(key) == sharded
    }


def cpu_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def optimizer_param_id_to_name(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[int, str]:
    params_to_names = {param: name for name, param in model.named_parameters()}
    state_dict = optimizer.state_dict()
    mapping: dict[int, str] = {}
    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"], strict=True):
        for param, saved_id in zip(group["params"], saved_group["params"], strict=True):
            name = params_to_names.get(param)
            if name is not None:
                mapping[int(saved_id)] = name
    return mapping


def filtered_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    sharded: bool,
) -> dict[str, Any]:
    state_dict = optimizer.state_dict()
    id_to_name = optimizer_param_id_to_name(model, optimizer)
    keep_ids = {
        param_id
        for param_id, name in id_to_name.items()
        if is_sharded_expert_state_name(name) == sharded
    }
    return {
        "state": {
            param_id: tensor_tree_to_cpu(state)
            for param_id, state in state_dict["state"].items()
            if int(param_id) in keep_ids
        },
        "param_groups": tensor_tree_to_cpu(state_dict["param_groups"]),
    }


def cpu_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return tensor_tree_to_cpu(optimizer.state_dict())


def merge_optimizer_state_dicts(common: dict[str, Any], shard: dict[str, Any] | None) -> dict[str, Any]:
    merged = {
        "state": dict(common.get("state", {})),
        "param_groups": common["param_groups"],
    }
    if shard is not None:
        merged["state"].update(shard.get("state", {}))
    return merged


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: MetisMambaConfig,
    step: int,
    tokens_seen: int,
    args: argparse.Namespace,
    runtime: Runtime,
    batch_rng_state: dict[str, Any] | None = None,
) -> None:
    checkpoint_start_s = time.perf_counter()
    wait_device(runtime)
    checkpoint_wait_s = time.perf_counter()
    saved_entries: list[tuple[Path, int]] = []
    if runtime.rank == 0:
        common_payload = {
            "model_family": config.model_type,
            "checkpoint_format": "tpu_ep_common_and_rank_experts_v1",
            "model_state_dict": (
                filtered_model_state_dict(model, sharded=False)
                if runtime.world_size > 1
                else cpu_model_state_dict(model)
            ),
            "optimizer_state_dict": (
                filtered_optimizer_state_dict(model, optimizer, sharded=False)
                if runtime.world_size > 1
                else cpu_optimizer_state_dict(optimizer)
            ),
            "scheduler_state_dict": tensor_tree_to_cpu(scheduler.state_dict()),
            "model_config": config.to_dict(),
            "step": int(step),
            "total_tokens_seen": int(tokens_seen),
            "batch_rng_state": batch_rng_state,
            "train_args": vars(args) | {
                "world_size": runtime.world_size,
                "expert_parallel": runtime.world_size > 1,
                "expert_parallel_size": runtime.world_size,
                "device_kind": runtime.device_kind,
                "trainer": "train_metis15_tpu.py",
            },
        }
        common_size = atomic_torch_save(path, common_payload)
        saved_entries.append((path, common_size))

    if runtime.world_size > 1:
        shard_path = compact_rank_checkpoint_path(path, runtime.rank, runtime.world_size)
        shard_payload = {
            "model_family": config.model_type,
            "checkpoint_format": "tpu_ep_rank_experts_v1",
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "model_state_dict": filtered_model_state_dict(model, sharded=True),
            "optimizer_state_dict": filtered_optimizer_state_dict(model, optimizer, sharded=True),
            "step": int(step),
            "total_tokens_seen": int(tokens_seen),
        }
        shard_size = atomic_torch_save(shard_path, shard_payload)
        saved_entries.append((shard_path, shard_size))

    checkpoint_write_s = time.perf_counter()
    if runtime.distributed and dist.is_initialized():
        dist.barrier()
    checkpoint_barrier_s = time.perf_counter()

    total_size = sum(size for _path, size in saved_entries)
    print(
        f"rank {runtime.rank:03d} saved compact checkpoint step={step} "
        f"local_write={total_size / (1024 ** 3):.2f}GiB "
        f"wait_s={checkpoint_wait_s - checkpoint_start_s:.3f} "
        f"write_s={checkpoint_write_s - checkpoint_wait_s:.3f} "
        f"barrier_s={checkpoint_barrier_s - checkpoint_write_s:.3f} "
        f"total_s={checkpoint_barrier_s - checkpoint_start_s:.3f} "
        f"files={[entry_path.name for entry_path, _size in saved_entries]}",
        flush=True,
    )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def load_training_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    runtime: Runtime,
) -> tuple[int, int, dict[str, Any] | None]:
    base_path = path / "latest.pt" if path.is_dir() else path
    if not base_path.exists():
        legacy_path = rank_checkpoint_path(base_path, runtime.rank, runtime.world_size)
        raise FileNotFoundError(f"Checkpoint is missing: {base_path} (legacy rank path: {legacy_path}).")

    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_format") in {
        "tpu_ep_common_and_rank_experts_v1",
        "neuron_ep_common_and_rank_experts_v1",
    }:
        model_state = dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint["optimizer_state_dict"]
        if runtime.world_size > 1:
            shard_path = compact_rank_checkpoint_path(base_path, runtime.rank, runtime.world_size)
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Rank {runtime.rank} expert checkpoint shard is missing: {shard_path}. "
                    "Compact TPU checkpoints require latest.pt plus latest.rankNNN.pt files."
                )
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if shard.get("checkpoint_format") not in {
                "tpu_ep_rank_experts_v1",
                "neuron_ep_rank_experts_v1",
            }:
                raise ValueError(f"Unexpected checkpoint shard format in {shard_path}.")
            model_state.update(shard["model_state_dict"])
            optimizer_state = merge_optimizer_state_dicts(optimizer_state, shard.get("optimizer_state_dict"))
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        move_optimizer_state_to_device(optimizer, runtime.device)
        return (
            int(checkpoint.get("step", 0)),
            int(checkpoint.get("total_tokens_seen", 0)),
            checkpoint.get("batch_rng_state"),
        )

    load_path = rank_checkpoint_path(base_path, runtime.rank, runtime.world_size)
    if load_path != base_path:
        if not load_path.exists():
            raise FileNotFoundError(
                f"Rank {runtime.rank} checkpoint is missing: {load_path}. "
                "For legacy expert-parallel TPU checkpoints, keep every rank shard next to latest.pt."
            )
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, runtime.device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("total_tokens_seen", 0)),
        checkpoint.get("batch_rng_state"),
    )


def reset_resume_optimizer_lrs(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    base_lrs: list[float],
    lr_scale: float,
) -> list[float]:
    target_lrs = [float(lr) * float(lr_scale) for lr in base_lrs]
    if len(target_lrs) != len(optimizer.param_groups):
        raise ValueError(
            "Cannot reset optimizer LR on resume: saved optimizer param group count "
            f"{len(optimizer.param_groups)} does not match current group count {len(target_lrs)}."
        )
    for group, lr in zip(optimizer.param_groups, target_lrs, strict=True):
        group["lr"] = lr
    scheduler.base_lrs = list(base_lrs)
    scheduler._last_lr = list(target_lrs)  # noqa: SLF001 - LambdaLR exposes no public setter.
    return target_lrs


def finite_check(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad.detach()).all().item()):
            raise FloatingPointError(f"non-finite gradient: {name}")
        if not bool(torch.isfinite(param.detach()).all().item()):
            raise FloatingPointError(f"non-finite parameter: {name}")


def _zero_like_device(model: XlaMetisForCausalLM, shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    device = model.embed_tokens.weight.device
    return torch.zeros(shape, device=device, dtype=dtype)


@torch.no_grad()
def collect_expert_histograms(
    model: XlaMetisForCausalLM,
    runtime: Runtime,
) -> list[dict[str, Any]]:
    if not model.layers:
        return []
    num_layers = len(model.layers)
    num_experts = model.layers[0].moe.num_experts
    world_size = max(1, runtime.world_size)
    counts_rows: list[torch.Tensor] = []
    dest_rows: list[torch.Tensor] = []
    valid_dest_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    dropped_rows: list[torch.Tensor] = []
    assignment_rows: list[torch.Tensor] = []
    capacity_rows: list[torch.Tensor] = []
    for layer in model.layers:
        moe = layer.moe
        counts_rows.append(
            moe.last_route_counts.to(dtype=torch.float32)
            if moe.last_route_counts is not None
            else _zero_like_device(model, (num_experts,))
        )
        dest_counts = (
            moe.last_dest_counts.to(dtype=torch.float32)
            if moe.last_dest_counts is not None
            else _zero_like_device(model, (world_size,))
        )
        dest_valid = (
            moe.last_dest_valid.to(dtype=torch.float32)
            if moe.last_dest_valid is not None
            else _zero_like_device(model, (world_size,))
        )
        if dest_counts.numel() != world_size:
            padded = _zero_like_device(model, (world_size,))
            padded[: min(world_size, dest_counts.numel())] = dest_counts[: min(world_size, dest_counts.numel())]
            dest_counts = padded
        if dest_valid.numel() != world_size:
            padded = _zero_like_device(model, (world_size,))
            padded[: min(world_size, dest_valid.numel())] = dest_valid[: min(world_size, dest_valid.numel())]
            dest_valid = padded
        dest_rows.append(dest_counts)
        valid_dest_rows.append(dest_valid)
        valid_rows.append(
            moe.last_valid_assignments.to(dtype=torch.float32).reshape(())
            if moe.last_valid_assignments is not None
            else _zero_like_device(model, ())
        )
        dropped_rows.append(
            moe.last_dropped_assignments.to(dtype=torch.float32).reshape(())
            if moe.last_dropped_assignments is not None
            else _zero_like_device(model, ())
        )
        assignment_rows.append(
            moe.last_total_assignments.to(dtype=torch.float32).reshape(())
            if moe.last_total_assignments is not None
            else _zero_like_device(model, ())
        )
        capacity_rows.append(
            moe.last_capacity.to(dtype=torch.float32).reshape(())
            if moe.last_capacity is not None
            else _zero_like_device(model, ())
        )

    counts = torch.stack(counts_rows, dim=0)
    dest_counts = torch.stack(dest_rows, dim=0)
    dest_valid = torch.stack(valid_dest_rows, dim=0)
    valid = torch.stack(valid_rows, dim=0)
    dropped = torch.stack(dropped_rows, dim=0)
    assignments = torch.stack(assignment_rows, dim=0)
    capacities = torch.stack(capacity_rows, dim=0)

    if runtime.world_size > 1 and runtime.is_xla:
        for tensor in (counts, dest_counts, dest_valid, valid, dropped, assignments):
            reduced = runtime.xm.all_reduce(runtime.xm.REDUCE_SUM, tensor, pin_layout=False)
            tensor.copy_(reduced)
    elif runtime.world_size > 1 and dist.is_initialized():
        for tensor in (counts, dest_counts, dest_valid, valid, dropped, assignments):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    wait_device(runtime)
    counts_cpu = counts.detach().cpu()
    dest_cpu = dest_counts.detach().cpu()
    dest_valid_cpu = dest_valid.detach().cpu()
    valid_cpu = valid.detach().cpu()
    dropped_cpu = dropped.detach().cpu()
    assignments_cpu = assignments.detach().cpu()
    capacities_cpu = capacities.detach().cpu()
    rows: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        layer_counts = counts_cpu[layer_idx]
        mean = float(layer_counts.float().mean().item()) if layer_counts.numel() else 0.0
        max_count = float(layer_counts.float().max().item()) if layer_counts.numel() else 0.0
        rows.append(
            {
                "layer": layer_idx,
                "counts": [int(value) for value in layer_counts.tolist()],
                "dest_counts": [int(value) for value in dest_cpu[layer_idx].tolist()],
                "dest_valid": [int(value) for value in dest_valid_cpu[layer_idx].tolist()],
                "max_mean": max_count / max(mean, 1.0),
                "valid": int(valid_cpu[layer_idx].item()),
                "dropped": int(dropped_cpu[layer_idx].item()),
                "assignments": int(assignments_cpu[layer_idx].item()),
                "capacity": int(capacities_cpu[layer_idx].item()),
            }
        )
    return rows


def make_profile_totals() -> dict[str, float]:
    return {
        "data_s": 0.0,
        "fwd_bwd_s": 0.0,
        "grad_sync_s": 0.0,
        "optim_s": 0.0,
        "qk_clip_s": 0.0,
        "mark_s": 0.0,
        "finite_s": 0.0,
        "log_wait_s": 0.0,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google TPU/PJRT static expert-parallel Metis-1.5 trainer.")
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["xla", "cpu"], default="xla")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--perf-warmup-steps",
        type=int,
        default=0,
        help="Exclude the first N optimizer steps from throughput/profile intervals.",
    )
    parser.add_argument("--constant-lr", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw", "muon-adamw", "hybrid_muon_adamw"], default="adamw")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument(
        "--xla-stable-adamw",
        action="store_true",
        help="Use the custom AdamW loop with on-device step counters to avoid step-dependent XLA recompiles.",
    )
    parser.add_argument("--hybrid-adamw-impl", choices=["loop", "foreach"], default="loop")
    parser.add_argument(
        "--optimizer-master-weights",
        action="store_true",
        help="Maintain FP32 master parameters for BF16/FP16 training weights and copy them back after each optimizer step.",
    )
    parser.add_argument("--muon-beta", type=float, default=None)
    parser.add_argument("--muon-ns-steps", type=int, default=None)
    parser.add_argument("--muon-lr-scale", type=float, default=None)
    parser.add_argument(
        "--muon-scale-mode",
        choices=["original", "match_rms_adamw"],
        default=None,
        help="Shape scaling for Muon updates; match_rms_adamw follows the scalable Moonshot/Kimi recipe.",
    )
    parser.add_argument("--muon-include-routed-experts", action="store_true")
    parser.add_argument(
        "--preinit-optimizer-state",
        action="store_true",
        help="Materialize optimizer state before step 1 so TPU/XLA compiles the steady-state optimizer graph.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--mark-step-each-microbatch",
        action="store_true",
        help=(
            "On XLA, execute each gradient-accumulation microbatch as its own graph after backward. "
            "This keeps accumulation from compiling one oversized multi-microbatch graph."
        ),
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--override-optimizer-lr-on-resume",
        action="store_true",
        help=(
            "After loading a checkpoint, reset optimizer param-group learning rates to the "
            "current command-line schedule. This prevents spot resumes from silently inheriting "
            "stale LR values from exploratory checkpoints."
        ),
    )
    parser.add_argument("--expert-capacity-factor", type=float, default=4.0)
    parser.add_argument("--expert-capacity", type=int, default=None)
    parser.add_argument(
        "--dispatch-pack-impl",
        choices=DISPATCH_PACK_IMPL_CHOICES,
        default="index_add",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--debug-finite", action="store_true")
    parser.add_argument(
        "--debug-grad-norm-interval",
        type=int,
        default=0,
        help="When >0, log global common/expert gradient norms at this step interval.",
    )
    parser.add_argument(
        "--debug-param-delta-interval",
        type=int,
        default=0,
        help="When >0, log relative parameter update magnitudes for representative tensors.",
    )
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Reuse one sampled batch for every step; intended for overfit/correctness canaries only.",
    )
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--local-log-metrics",
        action="store_true",
        help="Log rank-local loss metrics instead of global averages. Intended only for low-overhead debugging.",
    )
    parser.add_argument("--log-expert-histograms", action="store_true")
    parser.add_argument("--router-override", choices=ROUTER_OVERRIDE_CHOICES, default="learned")
    parser.add_argument("--loss-mode", choices=LOSS_MODE_CHOICES, default="real_ce")
    parser.add_argument(
        "--ce-logits-dtype",
        choices=CE_LOGITS_DTYPE_CHOICES,
        default="float32",
        help="Dtype used for the shifted logits passed to cross entropy; float32 matches the original trainer.",
    )
    parser.add_argument(
        "--ce-impl",
        choices=CE_IMPL_CHOICES,
        default="cross_entropy",
        help="Cross entropy implementation; manual_logsumexp is a BF16-friendly exact CE fallback.",
    )
    parser.add_argument("--attention-mode", choices=ATTENTION_MODE_CHOICES, default="real")
    parser.add_argument("--attention-kernel", choices=ATTENTION_KERNEL_CHOICES, default="eager")
    parser.add_argument(
        "--qk-clip-threshold",
        type=float,
        default=100.0,
        help="Kimi/Moonshot-style MuonClip threshold for max scaled attention logits; <=0 disables QK clipping.",
    )
    parser.add_argument(
        "--qk-clip-alpha",
        type=float,
        default=0.5,
        help="Split factor for Q/K rescaling: query scales by eta^alpha and key by eta^(1-alpha).",
    )
    parser.add_argument(
        "--qk-clip-interval",
        type=int,
        default=1,
        help="Apply QK clipping every N optimizer steps after the Muon/AdamW update; 1 matches MuonClip.",
    )
    parser.add_argument(
        "--qk-clip-warmup-steps",
        type=int,
        default=0,
        help="Delay QK clipping until this optimizer step; use only for ablations.",
    )
    parser.add_argument("--moe-mode", choices=MOE_MODE_CHOICES, default="real")
    parser.add_argument("--grad-sync-mode", choices=GRAD_SYNC_MODE_CHOICES, default="all_reduce")
    parser.add_argument(
        "--grad-sync-bucket-mb",
        type=float,
        default=0.0,
        help="Bucket replicated XLA gradient all-reduces to reduce collective scratch memory; 0 means one all-reduce.",
    )
    parser.add_argument("--activation-checkpointing", choices=ACTIVATION_CHECKPOINT_CHOICES, default="none")
    parser.add_argument("--activation-checkpoint-layer-interval", type=int, default=1)
    parser.add_argument("--balanced-static-layout", choices=BALANCED_STATIC_LAYOUT_CHOICES, default="indexed")
    parser.add_argument(
        "--balanced-static-router-weights",
        choices=BALANCED_STATIC_ROUTER_WEIGHT_CHOICES,
        default="uniform",
        help=(
            "For balanced_static dispatch, keep fixed balanced expert destinations but choose combine weights "
            "uniformly or from the learned router over that static expert set."
        ),
    )
    parser.add_argument(
        "--balanced-static-router-input",
        choices=BALANCED_STATIC_ROUTER_INPUT_CHOICES,
        default="hidden",
        help=(
            "Input representation for the single-latent MoE router. hidden follows the LatentMoE design where "
            "gating sees the full model state while only routed payloads move through latent space."
        ),
    )
    parser.add_argument("--expert-activation-safety", choices=EXPERT_ACTIVATION_SAFETY_CHOICES, default="clamp")
    parser.add_argument("--moe-balance-bias-update-rate", type=float, default=None)
    parser.add_argument("--disable-moe-balance-update", action="store_true")
    parser.add_argument("--world-size-override", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--moe-num-experts", type=int, default=None)
    parser.add_argument("--moe-top-k", type=int, default=None)
    parser.add_argument("--moe-expert-intermediate-size", type=int, default=None)
    parser.add_argument("--moe-routed-latent-size", type=int, default=None)
    parser.add_argument("--moe-router-latent-size", type=int, default=None)
    tie_group = parser.add_mutually_exclusive_group()
    tie_group.add_argument(
        "--tie-embeddings",
        dest="tie_embeddings",
        action="store_true",
        default=None,
        help="Force tied input/output embeddings, overriding the manifest.",
    )
    tie_group.add_argument(
        "--untie-embeddings",
        dest="tie_embeddings",
        action="store_false",
        help="Force separate input embeddings and LM head, overriding the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    runtime = setup_runtime(args.device)
    try:
        config, manifest = load_config(args)
        if config.moe_num_experts % runtime.world_size != 0:
            raise ValueError(
                f"moe_num_experts={config.moe_num_experts} must divide world_size={runtime.world_size}."
            )
        if args.dispatch_pack_impl == "balanced_static":
            if args.router_override != "force_balanced":
                raise ValueError("balanced_static dispatch requires --router-override force_balanced.")
            if config.moe_num_experts != runtime.world_size:
                raise ValueError(
                    "balanced_static dispatch requires exactly one expert per rank "
                    f"(moe_num_experts={config.moe_num_experts}, world_size={runtime.world_size})."
                )
        if args.dispatch_pack_impl in {"group_static", "group_static_gather"}:
            if args.router_override != "learned":
                raise ValueError(f"{args.dispatch_pack_impl} dispatch requires --router-override learned.")
            if config.moe_num_experts != runtime.world_size:
                raise ValueError(
                    f"{args.dispatch_pack_impl} dispatch requires exactly one expert per rank "
                    f"(moe_num_experts={config.moe_num_experts}, world_size={runtime.world_size})."
                )
            if runtime.world_size % config.moe_top_k != 0:
                raise ValueError(
                    f"{args.dispatch_pack_impl} dispatch requires world_size to divide evenly by top_k "
                    f"(world_size={runtime.world_size}, top_k={config.moe_top_k})."
                )
        if args.dispatch_pack_impl == "dense_all_experts":
            if args.router_override != "learned":
                raise ValueError("dense_all_experts dispatch requires --router-override learned.")
            if config.moe_num_experts != runtime.world_size:
                raise ValueError(
                    "dense_all_experts dispatch requires exactly one expert per rank "
                    f"(moe_num_experts={config.moe_num_experts}, world_size={runtime.world_size})."
                )
        if args.activation_checkpoint_layer_interval < 1:
            raise ValueError("--activation-checkpoint-layer-interval must be >= 1.")
        if args.grad_sync_bucket_mb < 0:
            raise ValueError("--grad-sync-bucket-mb must be >= 0.")
        if args.perf_warmup_steps < 0:
            raise ValueError("--perf-warmup-steps must be >= 0.")
        if args.qk_clip_interval < 1:
            raise ValueError("--qk-clip-interval must be >= 1.")
        if not 0.0 <= args.qk_clip_alpha <= 1.0:
            raise ValueError("--qk-clip-alpha must be between 0 and 1.")
        random.seed(args.seed + runtime.rank)
        np.random.seed(args.seed + runtime.rank)
        torch.manual_seed(args.seed)
        model_dtype = torch.bfloat16 if runtime.is_xla else torch.float32
        model = XlaMetisForCausalLM(
            config,
            runtime=runtime,
            capacity_factor=args.expert_capacity_factor,
            capacity=args.expert_capacity,
            dispatch_pack_impl=args.dispatch_pack_impl,
            router_override=args.router_override,
            attention_mode=args.attention_mode,
            attention_kernel=args.attention_kernel,
            qk_clip_enabled=args.qk_clip_threshold > 0,
            moe_mode=args.moe_mode,
            loss_mode=args.loss_mode,
            ce_logits_dtype=args.ce_logits_dtype,
            ce_impl=args.ce_impl,
            activation_checkpointing=args.activation_checkpointing,
            activation_checkpoint_layer_interval=args.activation_checkpoint_layer_interval,
            track_route_metrics=(
                args.log_expert_histograms
                or args.dispatch_pack_impl != "balanced_static"
                or args.router_override != "force_balanced"
            ),
            balanced_static_layout=args.balanced_static_layout,
            balanced_static_router_weights=args.balanced_static_router_weights,
            balanced_static_router_input=args.balanced_static_router_input,
            expert_activation_safety=args.expert_activation_safety,
        ).to(device=runtime.device, dtype=model_dtype)
        torch.manual_seed(args.seed + runtime.rank)
        data = None
        if not args.synthetic_data:
            if not args.data_dir:
                raise ValueError("--data-dir is required unless --synthetic-data is used.")
            data, _dtype = load_memmap(Path(args.data_dir))
        rng = np.random.default_rng(args.seed + runtime.rank)
        optimizer, optimizer_summary = build_optimizer_from_args(model, args, manifest.get("optimizer"))
        current_base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if args.preinit_optimizer_state and not args.resume_from:
            preinitialize_optimizer_state(optimizer)
        lr_lambda = (
            (lambda _step: 1.0)
            if args.constant_lr
            else (lambda step: cosine_lr(step, max_steps=args.max_steps, warmup_steps=args.warmup_steps))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        tokens_seen = 0
        resume_step = 0
        batch_rng_state = None
        if args.resume_from:
            resume_step, tokens_seen, batch_rng_state = load_training_checkpoint(
                path=Path(args.resume_from),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                runtime=runtime,
            )
            if args.override_optimizer_lr_on_resume:
                reset_lrs = reset_resume_optimizer_lrs(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    base_lrs=current_base_lrs,
                    lr_scale=lr_lambda(resume_step),
                )
            else:
                reset_lrs = None
            if runtime.rank == 0:
                print(
                    f"Resumed TPU checkpoint from {args.resume_from} "
                    f"at step={resume_step} tokens_seen={tokens_seen:,}",
                    flush=True,
                )
                if reset_lrs is not None:
                    joined_lrs = ",".join(f"{lr:.6e}" for lr in reset_lrs)
                    print(
                        "Reset optimizer learning rates after resume "
                        f"from current args/schedule: {joined_lrs}",
                        flush=True,
                    )
        if batch_rng_state is not None:
            rng.bit_generator.state = batch_rng_state
            if runtime.rank == 0:
                print("Restored data sampler state from checkpoint.", flush=True)
        else:
            skipped_rng_draws = advance_batch_rng_for_resume(
                data=data,
                rng=rng,
                batch_size=args.batch_size,
                block_size=config.block_size,
                grad_accum_steps=args.grad_accum_steps,
                resume_step=resume_step,
            )
            if skipped_rng_draws and runtime.rank == 0:
                print(
                    f"Advanced data sampler by {skipped_rng_draws:,} draws for resume_step={resume_step}.",
                    flush=True,
                )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokens_per_step = args.batch_size * args.grad_accum_steps * config.block_size * runtime.world_size
        expected_valid_assignments = (
            args.batch_size * config.block_size * config.moe_top_k * config.n_layer * runtime.world_size
        )
        if runtime.rank == 0:
            (out_dir / "train_config.json").write_text(
                json.dumps(
                    vars(args)
                    | {
                        "runtime_world_size": runtime.world_size,
                        "device_kind": runtime.device_kind,
                        "trainer_moe_backend": "tpu_static_ep",
                        "model_config": config.to_dict(),
                        "manifest_name": manifest.get("name"),
                        "optimizer_summary": optimizer_summary.to_dict() if optimizer_summary is not None else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            audit = config.param_application_audit()
            local_params = sum(param.numel() for param in model.parameters())
            print("TPU Metis launch", flush=True)
            print(f"  device: {runtime.device} ({runtime.device_kind})", flush=True)
            print(f"  world_size: {runtime.world_size}", flush=True)
            print(f"  local batch size: {args.batch_size}", flush=True)
            print(f"  grad accum steps: {args.grad_accum_steps}", flush=True)
            print(f"  tokens per optimizer step: {tokens_per_step}", flush=True)
            print(f"  expected valid assignments/logged step: {expected_valid_assignments}", flush=True)
            print(f"  config: layers={config.n_layer} d_model={config.d_model} block={config.block_size}", flush=True)
            print(
                f"  moe: experts={config.moe_num_experts} local={config.moe_num_experts // runtime.world_size} "
                f"top_k={config.moe_top_k} latent={config.moe_routed_latent_size} hidden={config.moe_expert_intermediate_size}",
                flush=True,
            )
            print(f"  local params/rank: {local_params:,}", flush=True)
            print(f"  rough param apps/token: {audit.get('rough_total_param_apps_per_token'):,}", flush=True)
            print(f"  static capacity factor: {args.expert_capacity_factor}", flush=True)
            print(f"  dispatch pack impl: {args.dispatch_pack_impl}", flush=True)
            print(f"  single latent router input: {args.balanced_static_router_input}", flush=True)
            if args.dispatch_pack_impl == "balanced_static":
                print(f"  balanced static layout: {args.balanced_static_layout}", flush=True)
                print("  balanced static layer stagger: enabled", flush=True)
                print(f"  balanced static router weights: {args.balanced_static_router_weights}", flush=True)
                print(f"  balanced static router input: {args.balanced_static_router_input}", flush=True)
            print(
                "  ablations: "
                f"router={args.router_override} loss={args.loss_mode} "
                f"ce_impl={args.ce_impl} ce_logits_dtype={args.ce_logits_dtype} "
                f"attention={args.attention_mode} attention_kernel={args.attention_kernel} "
                f"moe={args.moe_mode}",
                flush=True,
            )
            print(
                f"  qk clip: threshold={args.qk_clip_threshold:g} "
                f"alpha={args.qk_clip_alpha:g} interval={args.qk_clip_interval} "
                f"warmup_steps={args.qk_clip_warmup_steps}",
                flush=True,
            )
            print(f"  grad sync mode: {args.grad_sync_mode}", flush=True)
            print(f"  grad sync bucket MB: {args.grad_sync_bucket_mb:g}", flush=True)
            print(f"  optimizer: {args.optimizer}", flush=True)
            if optimizer_summary is not None:
                print(
                    f"  hybrid optimizer: muon_params={optimizer_summary.muon_params:,} "
                    f"adamw_params={optimizer_summary.adamw_params:,} "
                    f"routed_experts_muon={optimizer_summary.routed_experts_muon} "
                    f"adamw_impl={optimizer_summary.adamw_impl} "
                    f"master_weights={optimizer_summary.master_weights}",
                    flush=True,
                )
                for group_summary in optimizer_summary.groups:
                    print(
                        f"    optimizer_group {group_summary.name}: "
                        f"mode={group_summary.optimizer} tensors={group_summary.tensor_count} "
                        f"params={group_summary.param_count:,}",
                        flush=True,
                    )
            print(f"  grad clip: {args.grad_clip}", flush=True)
            print(f"  preinit optimizer state: {args.preinit_optimizer_state}", flush=True)
            print(f"  expert activation safety: {args.expert_activation_safety}", flush=True)
            print(f"  perf warmup steps: {args.perf_warmup_steps}", flush=True)
            print(
                f"  activation checkpointing: {args.activation_checkpointing} "
                f"(layer_interval={args.activation_checkpoint_layer_interval})",
                flush=True,
            )
            print(
                f"  moe balance: strategy={config.moe_balance_strategy} "
                f"bias_update_rate={config.moe_balance_bias_update_rate:.3e}",
                flush=True,
            )
            print("  trainer MoE backend: tpu_static_ep", flush=True)

        interval_profile = make_profile_totals()
        interval_tokens = 0
        interval_updates = 0
        interval_step_times: list[float] = []
        interval_start = time.perf_counter()
        fixed_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if args.fixed_batch:
            fixed_batches = [
                get_batch(
                    data=data,
                    rng=rng,
                    batch_size=args.batch_size,
                    block_size=config.block_size,
                    vocab_size=config.padded_vocab_size,
                    device=runtime.device,
                )
                for _ in range(args.grad_accum_steps)
            ]
            if runtime.rank == 0:
                print(
                    f"fixed_batch enabled: reusing {len(fixed_batches)} microbatch(es) for correctness canary",
                    flush=True,
                )
        param_delta_refs = capture_param_delta_refs(model)
        param_delta_names = [name for name, _param in param_delta_refs]

        for step in range(resume_step + 1, args.max_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = None
            step_lm = None
            step_aux = None
            step_valid = None
            step_grad_norms = None
            step_param_before = None
            step_param_delta = None
            step_qk_clip_stats = None
            if args.qk_clip_threshold > 0:
                model.reset_qk_clip_stats()
            for micro_idx in range(args.grad_accum_steps):
                t0 = time.perf_counter()
                if fixed_batches is None:
                    xb, yb = get_batch(
                        data=data,
                        rng=rng,
                        batch_size=args.batch_size,
                        block_size=config.block_size,
                        vocab_size=config.padded_vocab_size,
                        device=runtime.device,
                    )
                else:
                    xb, yb = fixed_batches[micro_idx]
                interval_profile["data_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                out = model(xb, labels=yb)
                (out["loss"] / float(args.grad_accum_steps)).backward()
                interval_profile["fwd_bwd_s"] += time.perf_counter() - t0
                if args.mark_step_each_microbatch and args.grad_accum_steps > 1:
                    t0 = time.perf_counter()
                    mark_step(runtime)
                    interval_profile["mark_s"] += time.perf_counter() - t0
                step_loss = out["loss"].detach() if step_loss is None else step_loss + out["loss"].detach()
                step_lm = out["lm_loss"].detach() if step_lm is None else step_lm + out["lm_loss"].detach()
                step_aux = out["moe_aux_loss"].detach() if step_aux is None else step_aux + out["moe_aux_loss"].detach()
                step_valid = (
                    out["valid_assignments"].detach()
                    if step_valid is None
                        else step_valid + out["valid_assignments"].detach()
                    )
            t0 = time.perf_counter()
            xla_all_reduce_gradients(
                runtime,
                model,
                mode=args.grad_sync_mode,
                bucket_mb=args.grad_sync_bucket_mb,
            )
            if args.debug_grad_norm_interval > 0 and step % args.debug_grad_norm_interval == 0:
                step_grad_norms = grad_norm_diagnostics(runtime, model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.debug_param_delta_interval > 0 and step % args.debug_param_delta_interval == 0:
                step_param_before = clone_param_refs(param_delta_refs)
            interval_profile["grad_sync_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            optimizer.step()
            if not args.constant_lr:
                scheduler.step()
            interval_profile["optim_s"] += time.perf_counter() - t0
            if (
                args.qk_clip_threshold > 0
                and step >= args.qk_clip_warmup_steps
                and step % args.qk_clip_interval == 0
            ):
                t0 = time.perf_counter()
                step_qk_clip_stats = model.apply_qk_clip(
                    threshold=args.qk_clip_threshold,
                    alpha=args.qk_clip_alpha,
                    runtime=runtime,
                )
                sync_master_params = getattr(optimizer, "sync_master_params_from_model", None)
                if sync_master_params is not None:
                    sync_master_params(model.qk_clip_parameters())
                interval_profile["qk_clip_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            mark_step(runtime)
            interval_profile["mark_s"] += time.perf_counter() - t0
            if step_param_before is not None:
                wait_device(runtime)
                step_param_delta = param_delta_diagnostics(param_delta_refs, step_param_before)
            step_elapsed = time.perf_counter() - step_start
            if args.debug_finite:
                t0 = time.perf_counter()
                wait_device(runtime)
                finite_check(model)
                interval_profile["finite_s"] += time.perf_counter() - t0
            scale = 1.0 / float(args.grad_accum_steps)
            step_loss = step_loss * scale
            step_lm = step_lm * scale
            step_aux = step_aux * scale
            step_valid = step_valid * scale
            tokens_seen += tokens_per_step

            if step <= args.perf_warmup_steps:
                if step == args.perf_warmup_steps and runtime.rank == 0:
                    print(f"perf_warmup_complete step={step}", flush=True)
                if step == args.perf_warmup_steps:
                    interval_profile = make_profile_totals()
                    interval_tokens = 0
                    interval_updates = 0
                    interval_step_times = []
                    interval_start = time.perf_counter()
                continue

            interval_updates += 1
            interval_tokens += args.batch_size * args.grad_accum_steps * config.block_size
            interval_step_times.append(step_elapsed)

            should_log = args.log_interval > 0 and (step % args.log_interval == 0 or step == args.max_steps)
            if should_log:
                t0 = time.perf_counter()
                local_metrics = torch.stack(
                    [
                        step_loss,
                        step_lm,
                        step_aux,
                        step_valid,
                    ]
                )
                if not args.local_log_metrics and runtime.world_size > 1:
                    global_losses = xla_reduce_tensor(
                        runtime,
                        local_metrics[:3],
                        scale=1.0 / float(runtime.world_size),
                    )
                    global_valid = xla_reduce_tensor(runtime, local_metrics[3], scale=1.0)
                    local_metrics = torch.cat((global_losses, global_valid.reshape(1)))
                values = local_metrics.detach().cpu().float().tolist()
                grad_norm_values = (
                    step_grad_norms.detach().cpu().float().tolist()
                    if step_grad_norms is not None
                    else None
                )
                param_delta_values = (
                    step_param_delta.detach().cpu().float().tolist()
                    if step_param_delta is not None
                    else None
                )
                qk_clip_values = (
                    step_qk_clip_stats.detach().cpu().float().tolist()
                    if step_qk_clip_stats is not None
                    else None
                )
                interval_profile["log_wait_s"] += time.perf_counter() - t0
                elapsed = max(time.perf_counter() - interval_start, 1e-6)
                total_updates = max(float(interval_updates), 1.0)
                total_tokens = interval_tokens * runtime.world_size
                p50_step_s = percentile(interval_step_times, 50.0)
                p95_step_s = percentile(interval_step_times, 95.0)
                if runtime.rank == 0:
                    global_tokens_per_s = total_tokens / elapsed
                    print(
                        f"step {step:6d} | loss {values[0]:.4f} | "
                        f"lm {values[1]:.4f} | moe_aux {values[2]:.5f} | "
                        f"valid_assign {values[3]:,.0f} | "
                        f"tok/s {global_tokens_per_s:,.0f} | step_s {elapsed / total_updates:.3f} | "
                        f"lr {scheduler.get_last_lr()[0]:.6e} | tok_seen {tokens_seen / 1e9:.4f}B",
                        flush=True,
                    )
                    if args.profile_components:
                        denom = max(float(total_updates), 1.0)
                        print(
                            "profile_components "
                            f"step={step} "
                            f"data_s={interval_profile['data_s'] / denom:.4f} "
                            f"fwd_bwd_s={interval_profile['fwd_bwd_s'] / denom:.4f} "
                            f"grad_sync_s={interval_profile['grad_sync_s'] / denom:.4f} "
                            f"optim_s={interval_profile['optim_s'] / denom:.4f} "
                            f"qk_clip_s={interval_profile['qk_clip_s'] / denom:.4f} "
                            f"mark_s={interval_profile['mark_s'] / denom:.4f} "
                            f"finite_s={interval_profile['finite_s'] / denom:.4f} "
                            f"log_wait_s={interval_profile['log_wait_s'] / denom:.4f} "
                            f"p50_step_s={p50_step_s:.4f} "
                            f"p95_step_s={p95_step_s:.4f}",
                            flush=True,
                        )
                    if grad_norm_values is not None:
                        print(
                            "grad_norms "
                            f"step={step} "
                            f"common={grad_norm_values[0]:.6e} "
                            f"expert={grad_norm_values[1]:.6e}",
                            flush=True,
                        )
                    if param_delta_values is not None:
                        joined = " ".join(
                            f"{name}={value:.6e}"
                            for name, value in zip(param_delta_names, param_delta_values, strict=False)
                        )
                        print(f"param_delta step={step} {joined}", flush=True)
                    if qk_clip_values is not None:
                        print(
                            "qk_clip "
                            f"step={step} "
                            f"max_logit={qk_clip_values[0]:.3f} "
                            f"min_scale={qk_clip_values[1]:.6f} "
                            f"scaled_heads={qk_clip_values[2]:.0f}",
                            flush=True,
                        )
                if args.log_expert_histograms:
                    histogram_rows = collect_expert_histograms(model, runtime)
                    if runtime.rank == 0:
                        for row in histogram_rows:
                            drop_frac = row["dropped"] / max(row["assignments"], 1)
                            print(
                                "expert_hist "
                                f"step={step} layer={row['layer']:02d} "
                                f"max_mean={row['max_mean']:.3f} "
                                f"valid={row['valid']:,}/{row['assignments']:,} "
                                f"dropped={row['dropped']:,} drop_frac={drop_frac:.4f} "
                                f"capacity={row['capacity']} "
                                f"dest={row['dest_counts']} dest_valid={row['dest_valid']} "
                                f"counts={row['counts']}",
                                flush=True,
                            )
                interval_profile = make_profile_totals()
                interval_tokens = 0
                interval_updates = 0
                interval_step_times = []
                interval_start = time.perf_counter()

            if (
                not args.skip_checkpoint
                and args.checkpoint_interval > 0
                and (step % args.checkpoint_interval == 0 or step == args.max_steps or STOP_REQUESTED)
            ):
                save_checkpoint(
                    path=out_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    step=step,
                    tokens_seen=tokens_seen,
                    args=args,
                    runtime=runtime,
                    batch_rng_state=rng.bit_generator.state,
                )
            if STOP_REQUESTED:
                if runtime.rank == 0:
                    print(f"stop_requested step={step}; exiting after checkpoint.", flush=True)
                break
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
