from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from torch._dynamo import disable as dynamo_disable
except Exception:  # pragma: no cover - optional torch.compile integration
    def dynamo_disable(fn):
        return fn

from .config import MetisMambaConfig
from .fp8 import (
    build_dot_product_attention,
    build_fp8_block_recipe,
    build_grouped_linear,
    build_layernorm_mlp,
    build_linear,
    build_mxfp8_recipe,
    build_rmsnorm,
    build_swiglu_activation,
    fp8_autocast_context,
    fp8_disabled_context,
    is_transformer_engine_module,
)
from .moe_kernels import (
    bucket_dispatch,
    bucket_dispatch_counts,
    capacity_bucket_dispatch,
    fused_swiglu,
    reverse_weighted_combine,
    reverse_unweighted_combine,
    triton_moe_kernels_available,
    unweighted_unpermute,
    weighted_unpermute,
)


def _dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_world_size(group=None) -> int:
    return dist.get_world_size(group=group) if _dist_available() else 1


def _dist_rank(group=None) -> int:
    return dist.get_rank(group=group) if _dist_available() else 0


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, input_splits: tuple[int, ...], output_splits: tuple[int, ...], group):
        ctx.input_splits = tuple(int(value) for value in input_splits)
        ctx.output_splits = tuple(int(value) for value in output_splits)
        ctx.group = group
        output_rows = int(sum(ctx.output_splits))
        output = input_tensor.new_empty((output_rows, *input_tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=list(ctx.output_splits),
            input_split_sizes=list(ctx.input_splits),
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input_rows = int(sum(ctx.input_splits))
        grad_input = grad_output.new_empty((grad_input_rows, *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.input_splits),
            input_split_sizes=list(ctx.output_splits),
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all_autograd(
    input_tensor: torch.Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group=None,
) -> torch.Tensor:
    return _AllToAllSingle.apply(input_tensor, tuple(input_splits), tuple(output_splits), group)


@torch.no_grad()
def _all_to_all_no_grad(
    input_tensor: torch.Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group=None,
) -> torch.Tensor:
    output = input_tensor.new_empty((int(sum(output_splits)), *input_tensor.shape[1:]))
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


@torch.no_grad()
def _exchange_all_to_all_splits(send_counts: torch.Tensor, *, group=None) -> tuple[list[int], list[int]]:
    world_size = _dist_world_size(group)
    rank = _dist_rank(group)
    send_counts = send_counts.to(dtype=torch.int64).contiguous()
    gathered = [torch.empty_like(send_counts) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts, group=group)
    count_matrix = torch.stack(gathered, dim=0)
    input_splits = [int(value) for value in send_counts.detach().cpu().tolist()]
    output_splits = [int(count_matrix[source_rank, rank].item()) for source_rank in range(world_size)]
    return input_splits, output_splits


@lru_cache(maxsize=1)
def _load_liger_fused_linear_ce():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore

    return LigerFusedLinearCrossEntropyLoss


@lru_cache(maxsize=1)
def _load_liger_silu_mul():
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # type: ignore

    return LigerSiLUMulFunction


def _torch_swiglu(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate, up = gate_up.split(intermediate_size, dim=-1)
    return F.silu(gate) * up


@lru_cache(maxsize=1)
def _load_compiled_swiglu():
    mode = os.environ.get("METIS_SWIGLU_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    return torch.compile(_torch_swiglu, mode=mode, fullgraph=True)


def _nvtx_range(name: str):
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def _apply_swiglu(
    gate_up: torch.Tensor,
    intermediate_size: int,
    swiglu_module: nn.Module | None,
    *,
    surface: str,
) -> torch.Tensor:
    if swiglu_module is not None:
        return swiglu_module(gate_up)
    enabled_surfaces = {
        item.strip().lower()
        for item in os.environ.get("METIS_TRITON_SWIGLU_SURFACES", "grouped_experts").split(",")
        if item.strip()
    }
    swiglu_impl = os.environ.get("METIS_SWIGLU_IMPL", "torch").strip().lower()
    if (
        swiglu_impl in {"compile", "compiled", "torch_compile"}
        and gate_up.is_cuda
        and ("all" in enabled_surfaces or surface.lower() in enabled_surfaces)
    ):
        return _load_compiled_swiglu()(gate_up, int(intermediate_size))
    if (
        swiglu_impl == "liger"
        and gate_up.is_cuda
        and ("all" in enabled_surfaces or surface.lower() in enabled_surfaces)
    ):
        gate, up = gate_up.split(intermediate_size, dim=-1)
        return _load_liger_silu_mul().apply(gate.contiguous(), up.contiguous())
    if (
        gate_up.is_cuda
        and triton_moe_kernels_available()
        and swiglu_impl == "triton"
        and os.environ.get("METIS_DISABLE_TRITON_SWIGLU", "1").strip().lower() not in {"1", "true", "yes", "on"}
        and ("all" in enabled_surfaces or surface.lower() in enabled_surfaces)
    ):
        hidden = fused_swiglu(gate_up, int(intermediate_size))
        if os.environ.get("METIS_DEBUG_SWIGLU_FINITE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            if not torch.isfinite(hidden).all().item():
                finite_gate_up = torch.isfinite(gate_up).all().item()
                print(
                    "nonfinite_swiglu "
                    f"surface={surface} gate_up_finite={finite_gate_up} "
                    f"gate_up_min={float(gate_up.float().nan_to_num().amin().item()):.6e} "
                    f"gate_up_max={float(gate_up.float().nan_to_num().amax().item()):.6e} "
                    f"hidden_min={float(hidden.float().nan_to_num().amin().item()):.6e} "
                    f"hidden_max={float(hidden.float().nan_to_num().amax().item()):.6e}",
                    flush=True,
                )
        return hidden
    return _torch_swiglu(gate_up, intermediate_size)


@lru_cache(maxsize=1)
def _load_flash_attention_3():
    errors: list[str] = []
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_func  # type: ignore

        return flash_attn_func
    except ImportError as exc:
        errors.append(str(exc))

    try:
        import flash_attn_interface  # type: ignore

        flash_attn_func = getattr(flash_attn_interface, "flash_attn_func", None)
        if flash_attn_func is not None:
            return flash_attn_func
        errors.append("flash_attn_interface imported but does not expose flash_attn_func")
    except ImportError as exc:
        errors.append(str(exc))

    try:
        from hopper.flash_attn_interface import flash_attn_func  # type: ignore

        return flash_attn_func
    except ImportError as exc:
        errors.append(str(exc))

    raise RuntimeError(
        "FlashAttention-3 is not available. Install the official Dao-AILab hopper package "
        "(cd flash-attention/hopper && python setup.py install). "
        f"Import errors: {' | '.join(errors)}"
    )


@lru_cache(maxsize=1)
def _flash_attention_3_accepts_dropout_p() -> bool:
    try:
        return "dropout_p" in signature(_load_flash_attention_3()).parameters
    except (TypeError, ValueError):
        return True


@lru_cache(maxsize=1)
def _sdpa_accepts_enable_gqa() -> bool:
    try:
        return "enable_gqa" in signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


def _is_hopper_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 9


@dataclass
class MetisCausalLMOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None
    route_probs: torch.Tensor | None = None
    chosen_depths: torch.Tensor | None = None
    route_aux_loss: torch.Tensor | None = None
    moe_aux_loss: torch.Tensor | None = None
    mean_depth: torch.Tensor | None = None
    active_token_ratios: torch.Tensor | None = None


@dataclass
class MetisRewardOutput:
    rewards: torch.Tensor
    loss: torch.Tensor | None = None
    route_aux_loss: torch.Tensor | None = None
    moe_aux_loss: torch.Tensor | None = None
    mean_depth: torch.Tensor | None = None
    active_token_ratios: torch.Tensor | None = None


class MetisRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class MetisLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        use_fp8: bool,
        low_precision_allowed: bool = True,
        local_low_precision_recipe=None,
        force_bf16: bool = False,
    ) -> None:
        super().__init__()
        self.local_low_precision_recipe = local_low_precision_recipe
        self.force_bf16 = force_bf16
        self.impl = build_linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_fp8=use_fp8 and not force_bf16,
            low_precision_allowed=low_precision_allowed and not force_bf16,
        )
        self.uses_transformer_engine = is_transformer_engine_module(self.impl)

    @property
    def weight(self):
        return self.impl.weight

    @weight.setter
    def weight(self, value) -> None:
        self.impl.weight = value

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        if self.uses_transformer_engine:
            context = (
                fp8_disabled_context()
                if self.force_bf16
                else fp8_autocast_context(
                    enabled=self.local_low_precision_recipe is not None,
                    recipe=self.local_low_precision_recipe,
                )
            )
            with context:
                return self.impl(hidden_states, is_first_microbatch=is_first_microbatch)
        return self.impl(hidden_states)


class _TorchGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        init_std: float,
    ) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_gemms, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_gemms, out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_splits: list[int],
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        del is_first_microbatch
        outputs: list[torch.Tensor] = []
        start = 0
        for expert_index, split_size in enumerate(m_splits):
            split_size = int(split_size)
            if split_size <= 0:
                continue
            chunk = hidden_states.narrow(0, start, split_size)
            bias = None if self.bias is None else self.bias[expert_index]
            outputs.append(F.linear(chunk, self.weight[expert_index], bias))
            start += split_size
        if start != hidden_states.shape[0]:
            raise RuntimeError(
                f"GroupedLinear m_splits sum mismatch: consumed {start} rows from {hidden_states.shape[0]} rows."
            )
        if not outputs:
            return hidden_states.new_zeros((0, self.out_features))
        return torch.cat(outputs, dim=0)


class _AtenGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        init_std: float,
        disable_autocast: bool = True,
    ) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.disable_autocast = disable_autocast
        self.weight = nn.Parameter(torch.empty(num_gemms, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_gemms, out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "_grouped_mm")

    def forward(
        self,
        hidden_states: torch.Tensor,
        splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        del is_first_microbatch
        if hidden_states.numel() == 0:
            return hidden_states.new_zeros((0, self.out_features))
        if self.bias is not None and self.bias.requires_grad:
            raise RuntimeError("torch_grouped MoE backend only supports bias=False while training.")
        if isinstance(splits, torch.Tensor):
            offsets = torch.cumsum(
                splits.to(device=hidden_states.device, dtype=torch.int32),
                dim=0,
                dtype=torch.int32,
            )
        else:
            offsets = torch.tensor(splits, device=hidden_states.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        if offsets.numel() != self.num_gemms:
            raise RuntimeError(
                f"GroupedLinear split count mismatch: got {offsets.numel()} splits for {self.num_gemms} experts."
            )
        if not isinstance(splits, torch.Tensor):
            total = int(offsets[-1].item()) if offsets.numel() else 0
        else:
            total = hidden_states.shape[0]
        if total != hidden_states.shape[0]:
            raise RuntimeError(
                f"GroupedLinear split sum mismatch: split total {total} for {hidden_states.shape[0]} rows."
            )
        bias = None if self.bias is None else self.bias
        grouped_input = hidden_states.to(dtype=self.weight.dtype).contiguous()
        autocast_context = (
            torch.amp.autocast(device_type=hidden_states.device.type, enabled=False)
            if self.disable_autocast and hidden_states.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            return torch._grouped_mm(grouped_input, self.weight, offsets, bias)


class _TorchBmmGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        init_std: float,
    ) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_gemms, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_gemms, out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        del is_first_microbatch
        if isinstance(splits, torch.Tensor):
            split_values = [int(value) for value in splits.detach().cpu().tolist()]
        else:
            split_values = [int(value) for value in splits]
        if len(split_values) != self.num_gemms:
            raise RuntimeError(
                f"BMM grouped split count mismatch: got {len(split_values)} splits for {self.num_gemms} experts."
            )
        total = int(sum(split_values))
        if total != hidden_states.shape[0]:
            raise RuntimeError(
                f"BMM grouped split sum mismatch: split total {total} for {hidden_states.shape[0]} rows."
            )
        if total == 0:
            return hidden_states.new_zeros((0, self.out_features))
        capacity = max(split_values)
        if capacity <= 0:
            return hidden_states.new_zeros((0, self.out_features))

        padded = hidden_states.new_zeros((self.num_gemms, capacity, self.in_features))
        out = hidden_states.new_empty((total, self.out_features))
        src_offset = 0
        for expert_idx, split_size in enumerate(split_values):
            if split_size <= 0:
                continue
            padded[expert_idx, :split_size].copy_(hidden_states[src_offset : src_offset + split_size])
            src_offset += split_size
        with torch.amp.autocast(device_type=hidden_states.device.type, enabled=False):
            bmm_out = torch.bmm(padded.to(dtype=self.weight.dtype), self.weight)
            if self.bias is not None:
                bmm_out = bmm_out + self.bias[:, None, :]
        dst_offset = 0
        for expert_idx, split_size in enumerate(split_values):
            if split_size <= 0:
                continue
            out[dst_offset : dst_offset + split_size].copy_(bmm_out[expert_idx, :split_size])
            dst_offset += split_size
        return out


class _TorchLoopedGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        init_std: float,
    ) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_gemms, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_gemms, out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        del is_first_microbatch
        if isinstance(splits, torch.Tensor):
            split_values = [int(value) for value in splits.detach().cpu().tolist()]
        else:
            split_values = [int(value) for value in splits]
        if len(split_values) != self.num_gemms:
            raise RuntimeError(
                f"Looped grouped split count mismatch: got {len(split_values)} splits for {self.num_gemms} experts."
            )
        total = int(sum(split_values))
        if total != hidden_states.shape[0]:
            raise RuntimeError(
                f"Looped grouped split sum mismatch: split total {total} for {hidden_states.shape[0]} rows."
            )
        if total == 0:
            return hidden_states.new_zeros((0, self.out_features))
        outputs: list[torch.Tensor] = []
        offset = 0
        autocast_context = (
            torch.amp.autocast(device_type=hidden_states.device.type, enabled=False)
            if hidden_states.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            for expert_idx, split_size in enumerate(split_values):
                split_size = int(split_size)
                if split_size <= 0:
                    continue
                expert_input = hidden_states.narrow(0, offset, split_size).to(dtype=self.weight.dtype).contiguous()
                expert_output = expert_input.matmul(self.weight[expert_idx])
                if self.bias is not None:
                    expert_output = expert_output + self.bias[expert_idx]
                outputs.append(expert_output)
                offset += split_size
        if not outputs:
            return hidden_states.new_zeros((0, self.out_features))
        return torch.cat(outputs, dim=0)


class MetisGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        use_fp8: bool,
        init_std: float,
        low_precision_allowed: bool = True,
        local_low_precision_recipe=None,
        force_bf16: bool = False,
        backend: str = "te_grouped",
    ) -> None:
        super().__init__()
        self.local_low_precision_recipe = local_low_precision_recipe
        self.force_bf16 = force_bf16

        def init_method(weight: torch.Tensor) -> None:
            nn.init.normal_(weight, mean=0.0, std=init_std)

        grouped_impl = None
        if backend in {"torch_grouped", "torch_grouped_safe"}:
            if not _AtenGroupedLinear.is_available():
                raise RuntimeError("moe_backend='torch_grouped' requires torch._grouped_mm.")
            grouped_impl = _AtenGroupedLinear(
                num_gemms,
                in_features,
                out_features,
                bias=bias,
                init_std=init_std,
            )
        elif backend == "torch_bmm":
            grouped_impl = _TorchBmmGroupedLinear(
                num_gemms,
                in_features,
                out_features,
                bias=bias,
                init_std=init_std,
            )
        elif backend == "torch_looped":
            grouped_impl = _TorchLoopedGroupedLinear(
                num_gemms,
                in_features,
                out_features,
                bias=bias,
                init_std=init_std,
            )
        elif backend == "te_grouped":
            grouped_impl = build_grouped_linear(
                num_gemms=num_gemms,
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                use_fp8=use_fp8 and not force_bf16,
                low_precision_allowed=True if force_bf16 else low_precision_allowed,
                init_method=init_method,
            )
        else:
            raise RuntimeError(f"GroupedLinear backend {backend!r} is not implemented.")
        if grouped_impl is None:
            grouped_impl = _TorchGroupedLinear(
                num_gemms,
                in_features,
                out_features,
                bias=bias,
                init_std=init_std,
            )
        self.impl = grouped_impl
        self.uses_transformer_engine = is_transformer_engine_module(self.impl)

    @torch.no_grad()
    def reset_expert_parameters(self, expert_seeds: list[int], *, init_std: float) -> bool:
        weight = getattr(self.impl, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return False
        if weight.dim() < 1 or weight.shape[0] != len(expert_seeds):
            return False
        for expert_idx, seed in enumerate(expert_seeds):
            generator_device = weight.device if weight.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed))
            nn.init.normal_(weight[expert_idx], mean=0.0, std=init_std, generator=generator)
        bias = getattr(self.impl, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias.zero_()
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        if self.uses_transformer_engine:
            context = (
                fp8_disabled_context()
                if self.force_bf16
                else fp8_autocast_context(
                    enabled=self.local_low_precision_recipe is not None,
                    recipe=self.local_low_precision_recipe,
                )
            )
            with context:
                if isinstance(m_splits, torch.Tensor):
                    m_splits = [int(value) for value in m_splits.detach().cpu().tolist()]
                return self.impl(
                    hidden_states,
                    m_splits,
                    is_first_microbatch=is_first_microbatch,
                )
        return self.impl(hidden_states, m_splits, is_first_microbatch=is_first_microbatch)


def _nvfp4_linear_precision_kwargs(
    config: MetisMambaConfig,
    surface: str,
    *,
    use_fp8: bool,
) -> dict[str, Any]:
    precision = config.nvfp4_surface_precision(surface)
    kwargs: dict[str, Any] = {
        "low_precision_allowed": not (config.low_precision_mode == "nvfp4" and precision == "bf16"),
    }
    if config.low_precision_mode == "nvfp4" and precision == "bf16":
        kwargs["force_bf16"] = True
    if config.low_precision_mode == "nvfp4" and precision == "mxfp8" and use_fp8:
        kwargs["local_low_precision_recipe"] = build_mxfp8_recipe()
    if config.low_precision_mode == "nvfp4" and precision == "fp8_block" and use_fp8:
        kwargs["local_low_precision_recipe"] = build_fp8_block_recipe()
    return kwargs


def _precision_kwargs_for_value(
    config: MetisMambaConfig,
    precision: str,
    *,
    use_fp8: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "low_precision_allowed": not (config.low_precision_mode == "nvfp4" and precision == "bf16"),
    }
    if precision == "bf16" and config.low_precision_mode in {"fp8", "nvfp4"}:
        kwargs["force_bf16"] = True
        kwargs["low_precision_allowed"] = False
    if config.low_precision_mode == "nvfp4" and precision == "mxfp8" and use_fp8:
        kwargs["local_low_precision_recipe"] = build_mxfp8_recipe()
    if config.low_precision_mode == "nvfp4" and precision == "fp8_block" and use_fp8:
        kwargs["local_low_precision_recipe"] = build_fp8_block_recipe()
    return kwargs


def build_rms_norm_module(hidden_size: int, *, eps: float, use_fp8: bool) -> nn.Module:
    te_norm = build_rmsnorm(hidden_size=hidden_size, eps=eps, use_fp8=use_fp8)
    if te_norm is not None:
        return te_norm
    return MetisRMSNorm(hidden_size=hidden_size, eps=eps)


class MetisRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self._cached_seq_len = 0
        self._cached_dtype: torch.dtype | None = None
        self._cached_device: torch.device | None = None
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _can_use_arange_cache(self, position_ids: torch.Tensor) -> bool:
        return (
            position_ids.dim() == 2
            and position_ids.shape[1] > 0
            and position_ids.stride(0) == 0
            and position_ids.stride(1) == 1
        )

    def _cached_arange(self, seq_len: int, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_cos is not None
            and self._cached_sin is not None
            and self._cached_seq_len >= seq_len
            and self._cached_dtype == dtype
            and self._cached_device == device
        ):
            cos = self._cached_cos[:seq_len]
            sin = self._cached_sin[:seq_len]
            if os.environ.get("METIS_ROTARY_CACHE_CLONE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                cos = cos.clone()
                sin = sin.clone()
            return cos, sin
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("l,d->ld", pos, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cached_seq_len = seq_len
        self._cached_dtype = dtype
        self._cached_device = device
        self._cached_cos = emb.cos().to(dtype=dtype)
        self._cached_sin = emb.sin().to(dtype=dtype)
        return self._cached_cos, self._cached_sin

    def forward(self, position_ids: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._can_use_arange_cache(position_ids)
            and os.environ.get("METIS_DISABLE_ROTARY_CACHE", "0").strip().lower() not in {"1", "true", "yes", "on"}
        ):
            cos, sin = self._cached_arange(position_ids.shape[1], dtype=dtype, device=position_ids.device)
            return cos.unsqueeze(0).expand(position_ids.shape[0], -1, -1), sin.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
        pos = position_ids.to(self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("bl,d->bld", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., ::2]
    second = x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


def build_causal_mask(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    if attention_mask is None:
        return causal
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape (batch, seq_len).")
    expanded = causal.expand(batch_size, 1, seq_len, seq_len).clone()
    key_padding = ~attention_mask.to(torch.bool)
    return expanded.masked_fill(key_padding[:, None, None, :], torch.finfo(dtype).min)


def pack_active_tokens(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    pad_multiple: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    lengths = active_mask.sum(dim=-1)
    max_active = int(lengths.max().item())
    if max_active == 0:
        return None
    if pad_multiple > 1 and (max_active % pad_multiple) != 0:
        max_active = max_active + (pad_multiple - (max_active % pad_multiple))
    batch_size, _seq_len, hidden_size = hidden_states.shape
    packed_hidden_rows: list[torch.Tensor] = []
    packed_position_rows: list[torch.Tensor] = []
    packed_mask_rows: list[torch.Tensor] = []
    packed_index_rows: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        indices = torch.nonzero(active_mask[batch_index], as_tuple=False).squeeze(-1)
        active_count = int(indices.numel())
        selected_hidden = hidden_states[batch_index].index_select(0, indices) if active_count > 0 else hidden_states.new_zeros((0, hidden_size))
        selected_positions = position_ids[batch_index].index_select(0, indices) if active_count > 0 else position_ids.new_zeros((0,))
        selected_mask = torch.ones(active_count, dtype=torch.bool, device=active_mask.device)
        if active_count < max_active:
            pad_amount = max_active - active_count
            selected_hidden = torch.cat(
                [selected_hidden, hidden_states.new_zeros((pad_amount, hidden_size))],
                dim=0,
            )
            selected_positions = torch.cat(
                [selected_positions, position_ids.new_zeros((pad_amount,))],
                dim=0,
            )
            indices = torch.cat(
                [indices, position_ids.new_full((pad_amount,), -1)],
                dim=0,
            )
            selected_mask = torch.cat(
                [selected_mask, torch.zeros(pad_amount, dtype=torch.bool, device=active_mask.device)],
                dim=0,
            )
        packed_hidden_rows.append(selected_hidden)
        packed_position_rows.append(selected_positions)
        packed_mask_rows.append(selected_mask)
        packed_index_rows.append(indices)
    packed_hidden = torch.stack(packed_hidden_rows, dim=0)
    packed_positions = torch.stack(packed_position_rows, dim=0)
    packed_mask = torch.stack(packed_mask_rows, dim=0)
    packed_indices = torch.stack(packed_index_rows, dim=0)
    return packed_hidden, packed_positions, packed_mask, packed_indices


def scatter_active_tokens(
    full_hidden_states: torch.Tensor,
    packed_hidden_states: torch.Tensor,
    packed_mask: torch.Tensor,
    packed_indices: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    scatter_indices = packed_indices.clamp_min(0).unsqueeze(-1).expand_as(packed_hidden_states)
    scatter_src = packed_hidden_states * packed_mask.unsqueeze(-1).to(dtype=packed_hidden_states.dtype)
    scattered = torch.zeros_like(full_hidden_states).scatter_add(1, scatter_indices, scatter_src)
    return torch.where(active_mask.unsqueeze(-1), scattered, full_hidden_states)


class MetisSelfAttention(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool) -> None:
        super().__init__()
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = float(config.attention_dropout)
        self.attention_backend = config.attention_backend
        self.softmax_scale = self.head_dim ** -0.5
        self.native_gqa_attention = bool(config.native_gqa_attention)
        self.debug_attention_backend = config.debug_attention_backend
        self._debug_backend_printed = False
        self._fa3_native_gqa_failed = False
        self.perf_counters: dict[str, int] | None = None

        self.qkv_proj = MetisLinear(
            self.hidden_size,
            self.q_dim + (2 * self.kv_dim),
            bias=config.attention_bias,
            use_fp8=use_fp8,
            **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
        )
        self.o_proj = MetisLinear(
            self.q_dim,
            self.hidden_size,
            bias=config.attention_bias,
            use_fp8=use_fp8,
            **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
        )
        self.rotary_emb = MetisRotaryEmbedding(self.head_dim, base=config.rope_theta)
        self.te_attention = None
        if config.te_dot_product_attention:
            self.te_attention = build_dot_product_attention(
                num_attention_heads=self.num_heads,
                num_gqa_groups=self.num_kv_heads,
                head_dim=self.head_dim,
                attention_dropout=self.dropout,
                softmax_scale=self.softmax_scale,
                use_fp8=use_fp8,
            )
            if use_fp8 and self.te_attention is None:
                raise RuntimeError(
                    "TE dot-product attention was requested, but this Transformer Engine build "
                    "does not expose a compatible DotProductAttention module for Metis GQA."
                )

    def _should_use_flash_attention_3(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> bool:
        if self.attention_backend not in {"auto", "flash_attention_3"}:
            return False
        if attention_mask is not None:
            return False
        if query_states.device.type != "cuda":
            return False
        if query_states.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if not (_is_hopper_device(query_states.device) and _is_hopper_device(key_states.device)):
            return False
        if query_states.shape[-1] > 256 or key_states.shape[-1] > 256:
            return False
        return value_states.dtype == query_states.dtype

    def _should_use_te_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> bool:
        if self.te_attention is None:
            return False
        if attention_mask is not None:
            return False
        if query_states.device.type != "cuda":
            return False
        if query_states.dtype not in {torch.float16, torch.bfloat16}:
            return False
        return value_states.dtype == query_states.dtype and key_states.shape[1] == self.num_kv_heads

    def _bump_counter(self, name: str, amount: int = 1) -> None:
        if self.perf_counters is not None:
            self.perf_counters[name] = self.perf_counters.get(name, 0) + amount

    def _debug_backend_once(self, backend: str, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        if not self.debug_attention_backend or self._debug_backend_printed:
            return
        print(
            f"MetisSelfAttention backend={backend} shape={tuple(hidden_states.shape)} "
            f"mask={'none' if attention_mask is None else tuple(attention_mask.shape)}",
            flush=True,
        )
        self._debug_backend_printed = True

    def _flash_attention_3(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        flash_attn_func = _load_flash_attention_3()
        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()
        kwargs: dict[str, Any] = {
            "softmax_scale": self.softmax_scale,
            "causal": True,
        }
        if _flash_attention_3_accepts_dropout_p():
            kwargs["dropout_p"] = self.dropout if self.training else 0.0
        elif self.training and self.dropout:
            raise RuntimeError("This FlashAttention-3 build does not expose dropout_p; set attention_dropout=0.0.")
        attn_output = flash_attn_func(q, k, v, **kwargs)
        return attn_output.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv_states = self.qkv_proj(hidden_states, is_first_microbatch=is_first_microbatch)
        query_states, key_states, value_states = qkv_states.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids, dtype=hidden_states.dtype)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        if attention_mask is not None:
            self._bump_counter("attention_mask_passed_calls")

        if self._should_use_te_attention(query_states, key_states, value_states, attention_mask):
            self._bump_counter("te_attention_calls")
            self._debug_backend_once("te_dot_product_attention", hidden_states, attention_mask)
            attn_output = self.te_attention(
                query_states.transpose(1, 2).contiguous(),
                key_states.transpose(1, 2).contiguous(),
                value_states.transpose(1, 2).contiguous(),
                attn_mask_type="causal",
                qkv_format="bshd",
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = attn_output.transpose(1, 2)
        elif self._should_use_flash_attention_3(query_states, key_states, value_states, attention_mask):
            self._bump_counter("fa3_calls")
            self._debug_backend_once("flash_attention_3", hidden_states, attention_mask)
            fa3_key_states = key_states
            fa3_value_states = value_states
            if not self.native_gqa_attention or self._fa3_native_gqa_failed:
                fa3_key_states = repeat_kv(key_states, self.num_kv_groups)
                fa3_value_states = repeat_kv(value_states, self.num_kv_groups)
            try:
                attn_output = self._flash_attention_3(query_states, fa3_key_states, fa3_value_states)
            except RuntimeError:
                if fa3_key_states.shape[1] == query_states.shape[1]:
                    raise
                self._fa3_native_gqa_failed = True
                attn_output = self._flash_attention_3(
                    query_states,
                    repeat_kv(key_states, self.num_kv_groups),
                    repeat_kv(value_states, self.num_kv_groups),
                )
        elif hasattr(F, "scaled_dot_product_attention"):
            self._bump_counter("sdpa_calls")
            self._debug_backend_once("sdpa", hidden_states, attention_mask)
            sdpa_key_states = key_states
            sdpa_value_states = value_states
            sdpa_kwargs: dict[str, Any] = {}
            if key_states.shape[1] != query_states.shape[1]:
                if self.native_gqa_attention and _sdpa_accepts_enable_gqa():
                    sdpa_kwargs["enable_gqa"] = True
                else:
                    sdpa_key_states = repeat_kv(key_states, self.num_kv_groups)
                    sdpa_value_states = repeat_kv(value_states, self.num_kv_groups)
            attn_mask = None
            if attention_mask is not None:
                attn_mask = build_causal_mask(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=hidden_states.device,
                    dtype=query_states.dtype,
                    attention_mask=attention_mask,
                )
            attn_output = F.scaled_dot_product_attention(
                query_states,
                sdpa_key_states,
                sdpa_value_states,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None,
                scale=self.softmax_scale,
                **sdpa_kwargs,
            )
        else:
            self._bump_counter("eager_attention_calls")
            self._debug_backend_once("eager", hidden_states, attention_mask)
            key_states = repeat_kv(key_states, self.num_kv_groups)
            value_states = repeat_kv(value_states, self.num_kv_groups)
            scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale
            if attention_mask is not None:
                scores = scores + build_causal_mask(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=hidden_states.device,
                    dtype=scores.dtype,
                    attention_mask=attention_mask,
                )
            else:
                scores = scores + torch.triu(
                    torch.full(
                        (seq_len, seq_len),
                        torch.finfo(scores.dtype).min,
                        device=scores.device,
                        dtype=scores.dtype,
                    ),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
            weights = torch.softmax(scores.float(), dim=-1).to(dtype=query_states.dtype)
            if self.dropout > 0.0 and self.training:
                weights = F.dropout(weights, p=self.dropout, training=True)
            attn_output = torch.matmul(weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output, is_first_microbatch=is_first_microbatch)


class MetisSwiGLU(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MetisLinear(
            config.d_model,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
            use_fp8=use_fp8,
        )
        self.down_proj = MetisLinear(
            config.intermediate_size,
            config.d_model,
            bias=config.mlp_bias,
            use_fp8=use_fp8,
        )
        self.swiglu = build_swiglu_activation(use_fp8=use_fp8)

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states, is_first_microbatch=is_first_microbatch)
        hidden = _apply_swiglu(gate_up, self.intermediate_size, self.swiglu, surface="dense")
        return self.down_proj(hidden, is_first_microbatch=is_first_microbatch)


class MetisTELayerNormSwiGLU(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool) -> None:
        super().__init__()
        self.impl = build_layernorm_mlp(
            hidden_size=config.d_model,
            ffn_hidden_size=config.intermediate_size,
            eps=1e-6,
            bias=config.mlp_bias,
            use_fp8=use_fp8,
        )
        if self.impl is None:
            raise RuntimeError("TE fused MLP requires Transformer Engine FP8 modules.")

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        return self.impl(hidden_states, is_first_microbatch=is_first_microbatch)


class MetisHeadExpert(nn.Module):
    def __init__(
        self,
        head_dim: int,
        intermediate_size: int,
        *,
        bias: bool,
        use_fp8: bool,
        activation: str = "swiglu",
        precision_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.activation = activation
        precision_kwargs = precision_kwargs or {}
        self.gate_up_proj = MetisLinear(
            head_dim,
            (2 * intermediate_size) if activation == "swiglu" else intermediate_size,
            bias=bias,
            use_fp8=use_fp8,
            **precision_kwargs,
        )
        self.down_proj = MetisLinear(
            intermediate_size,
            head_dim,
            bias=bias,
            use_fp8=use_fp8,
            **precision_kwargs,
        )
        self.swiglu = build_swiglu_activation(use_fp8=use_fp8) if activation == "swiglu" else None

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states, is_first_microbatch=is_first_microbatch)
        if self.activation == "squared_relu":
            gate_up = torch.nan_to_num(gate_up, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-64.0, max=64.0)
            hidden = F.relu(gate_up).square()
        else:
            hidden = _apply_swiglu(gate_up, self.intermediate_size, self.swiglu, surface="single_expert")
        return self.down_proj(hidden, is_first_microbatch=is_first_microbatch)


class MetisGroupedHeadExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        head_dim: int,
        intermediate_size: int,
        *,
        bias: bool,
        use_fp8: bool,
        init_std: float,
        activation: str = "swiglu",
        precision_kwargs: dict[str, Any] | None = None,
        backend: str = "te_grouped",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.sanitize_grouped_outputs = backend == "torch_grouped_safe"
        self.sync_grouped_outputs = self.sanitize_grouped_outputs and os.environ.get(
            "METIS_TORCH_GROUPED_SAFE_SYNC",
            "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        precision_kwargs = precision_kwargs or {}
        self.gate_up_proj = MetisGroupedLinear(
            num_experts,
            head_dim,
            (2 * intermediate_size) if activation == "swiglu" else intermediate_size,
            bias=bias,
            use_fp8=use_fp8,
            init_std=init_std,
            backend=backend,
            **precision_kwargs,
        )
        self.down_proj = MetisGroupedLinear(
            num_experts,
            intermediate_size,
            head_dim,
            bias=bias,
            use_fp8=use_fp8,
            init_std=init_std,
            backend=backend,
            **precision_kwargs,
        )
        self.swiglu = build_swiglu_activation(use_fp8=use_fp8) if activation == "swiglu" else None

    @torch.no_grad()
    def reset_expert_parameters(self, expert_seeds: list[int], *, init_std: float) -> None:
        reset_gate = self.gate_up_proj.reset_expert_parameters(expert_seeds, init_std=init_std)
        reset_down = self.down_proj.reset_expert_parameters(expert_seeds, init_std=init_std)
        if not (reset_gate and reset_down):
            raise RuntimeError("Expert-parallel deterministic expert init requires resettable grouped linear weights.")

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        m_splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
        valid_positions: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(
            hidden_states,
            m_splits,
            is_first_microbatch=is_first_microbatch,
        )
        if self.sanitize_grouped_outputs:
            gate_up = torch.nan_to_num(gate_up, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-64.0, max=64.0)
            if self.sync_grouped_outputs and gate_up.device.type == "cuda":
                torch.cuda.synchronize(gate_up.device)
        if valid_positions is not None:
            valid_mask = torch.zeros(gate_up.shape[0], device=gate_up.device, dtype=torch.bool)
            valid_mask.index_fill_(0, valid_positions, True)
            gate_up = gate_up.masked_fill(~valid_mask.unsqueeze(-1), 0)
        if self.activation == "squared_relu":
            gate_up = torch.nan_to_num(gate_up, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-64.0, max=64.0)
            hidden = F.relu(gate_up).square()
        else:
            hidden = _apply_swiglu(gate_up, self.intermediate_size, self.swiglu, surface="grouped_experts")
        if routing_weights is not None:
            if routing_weights.shape[0] != hidden.shape[0]:
                raise RuntimeError(
                    f"routing_weights rows {routing_weights.shape[0]} do not match grouped expert rows {hidden.shape[0]}."
                )
            hidden = hidden * routing_weights.to(dtype=hidden.dtype).unsqueeze(-1)
        output = self.down_proj(
            hidden,
            m_splits,
            is_first_microbatch=is_first_microbatch,
        )
        if self.sanitize_grouped_outputs:
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-64.0, max=64.0)
            if self.sync_grouped_outputs and output.device.type == "cuda":
                torch.cuda.synchronize(output.device)
        return output

    @dynamo_disable
    def _forward_dynamo_disabled(
        self,
        hidden_states: torch.Tensor,
        m_splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
        valid_positions: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            m_splits,
            is_first_microbatch=is_first_microbatch,
            valid_positions=valid_positions,
            routing_weights=routing_weights,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_splits: list[int] | torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
        valid_positions: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if os.environ.get("METIS_DYNAMO_DISABLE_GROUPED_EXPERTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return self._forward_dynamo_disabled(
                hidden_states,
                m_splits,
                is_first_microbatch=is_first_microbatch,
                valid_positions=valid_positions,
                routing_weights=routing_weights,
            )
        return self._forward_impl(
            hidden_states,
            m_splits,
            is_first_microbatch=is_first_microbatch,
            valid_positions=valid_positions,
            routing_weights=routing_weights,
        )


class MetisMultiHeadLatentMoE(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool, layer_idx: int | None = None) -> None:
        super().__init__()
        if not config.uses_moe:
            raise ValueError("MetisMultiHeadLatentMoE requires config.ffn_type='multi_head_latent_moe'.")
        self.num_heads = config.moe_num_heads
        self.head_dim = config.moe_head_dim
        self.routed_dim = config.moe_effective_routed_dim
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.shared_experts = config.moe_shared_experts
        requested_ep_size = int(config.moe_expert_parallel_size)
        runtime_world_size = _dist_world_size()
        runtime_rank = _dist_rank()
        self.expert_parallel_enabled = bool(requested_ep_size > 1 and runtime_world_size > 1)
        self.expert_parallel_size = requested_ep_size if self.expert_parallel_enabled else 1
        self.expert_parallel_rank = runtime_rank if self.expert_parallel_enabled else 0
        self.expert_parallel_group = dist.group.WORLD if self.expert_parallel_enabled else None
        if self.expert_parallel_enabled:
            if requested_ep_size != runtime_world_size:
                raise RuntimeError(
                    "Metis expert parallel currently uses the full torch.distributed WORLD group; "
                    f"got moe_expert_parallel_size={requested_ep_size} with WORLD_SIZE={runtime_world_size}."
                )
            if self.num_experts % self.expert_parallel_size != 0:
                raise RuntimeError("moe_num_experts must divide evenly across expert-parallel ranks.")
        self.local_num_experts = self.num_experts // self.expert_parallel_size
        self.local_expert_start = self.expert_parallel_rank * self.local_num_experts
        self.local_expert_end = self.local_expert_start + self.local_num_experts
        self.router_temperature = float(config.moe_router_temperature)
        self.router_score = config.moe_router_score
        self.router_override = str(config.moe_router_override)
        self.balance_strategy = config.moe_balance_strategy
        self.balance_bias_update_rate = float(config.moe_balance_bias_update_rate)
        self.balance_bias_clamp = float(config.moe_balance_bias_clamp)
        self.balance_scale_by_token_fraction = bool(config.moe_balance_scale_by_token_fraction)
        self.balance_update_scale = 1.0
        self.use_fp8 = use_fp8
        self.dispatch_mode = config.moe_dispatch_mode
        self.moe_backend = config.moe_backend
        self.static_capacity = int(config.moe_static_capacity)
        self.capacity_factor = float(config.moe_capacity_factor)
        self.capacity_alignment = int(config.moe_capacity_alignment)
        self.overflow_mode = config.moe_overflow_mode
        self.fused_combine = bool(config.moe_fused_combine)
        self.graphable = bool(config.moe_graphable)
        self.memory_efficient_permutation = bool(config.moe_memory_efficient_permutation)
        self.token_dispatcher_type = str(config.moe_token_dispatcher_type)
        self.flex_dispatcher_backend = str(config.moe_flex_dispatcher_backend)
        self.overlap_expert_parallel_comm = bool(config.moe_overlap_expert_parallel_comm)
        self.delay_wgrad_compute = bool(config.moe_delay_wgrad_compute)
        self.router_fusion = bool(config.moe_router_fusion)
        self.permute_fusion = bool(config.moe_permute_fusion)
        expert_precision = config.expert_precision_for_layer(layer_idx)
        self.grouped_m_split_alignment = (
            int(config.fp8_pad_multiple) if use_fp8 and expert_precision != "bf16" else 1
        )
        self.grouped_m_split_min_m = 0
        if self.moe_backend == "torch_grouped_safe":
            self.grouped_m_split_min_m = max(8, int(config.moe_torch_grouped_min_m))
        expert_precision_kwargs = _precision_kwargs_for_value(config, expert_precision, use_fp8=use_fp8)
        self.latent_proj = MetisLinear(
            self.head_dim,
            config.moe_router_latent_size,
            bias=False,
            use_fp8=use_fp8,
            **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
        )
        self.routed_down_proj: MetisLinear | None = None
        self.routed_up_proj: MetisLinear | None = None
        if self.routed_dim != self.head_dim:
            self.routed_down_proj = MetisLinear(
                self.head_dim,
                self.routed_dim,
                bias=False,
                use_fp8=use_fp8,
                **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
            )
            self.routed_up_proj = MetisLinear(
                self.routed_dim,
                self.head_dim,
                bias=False,
                use_fp8=use_fp8,
                **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
            )
        self.expert_embeddings = nn.Parameter(
            torch.empty(self.num_heads, self.num_experts, config.moe_router_latent_size)
        )
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=config.initializer_range)
        self.expert_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_experts))
        self.register_buffer(
            "balance_bias",
            torch.zeros(self.num_heads, self.num_experts),
            persistent=True,
        )
        self.grouped_experts: MetisGroupedHeadExperts | None = None
        self.experts: nn.ModuleList | None = None
        if self.moe_backend not in {"te_grouped", "torch_grouped", "torch_grouped_safe", "torch_bmm", "torch_looped"}:
            raise RuntimeError(
                f"MoE backend {self.moe_backend!r} is not wired yet. "
                "Use moe_backend='te_grouped', 'torch_grouped', 'torch_grouped_safe', 'torch_bmm', or "
                "'torch_looped' with moe_dispatch_mode='bucketed'."
            )
        if self.dispatch_mode in {"grouped", "bucketed"}:
            self.grouped_experts = MetisGroupedHeadExperts(
                self.local_num_experts,
                self.routed_dim,
                config.moe_expert_intermediate_size,
                bias=config.mlp_bias,
                use_fp8=use_fp8,
                init_std=config.initializer_range,
                activation=config.moe_activation,
                precision_kwargs=expert_precision_kwargs,
                backend=self.moe_backend,
            )
            if self.expert_parallel_enabled:
                base_seed = int(config.moe_expert_parallel_init_seed)
                layer_offset = 0 if layer_idx is None else (int(layer_idx) + 1) * 1_000_003
                expert_seeds = [
                    base_seed + layer_offset + global_expert
                    for global_expert in range(self.local_expert_start, self.local_expert_end)
                ]
                self.grouped_experts.reset_expert_parameters(
                    expert_seeds,
                    init_std=config.initializer_range,
                )
        else:
            self.experts = nn.ModuleList(
                [
                    MetisHeadExpert(
                        self.routed_dim,
                        config.moe_expert_intermediate_size,
                        bias=config.mlp_bias,
                        use_fp8=use_fp8,
                        activation=config.moe_activation,
                        precision_kwargs=expert_precision_kwargs,
                    )
                    for _ in range(self.num_experts)
                ]
            )
        self.shared = nn.ModuleList(
            [
                MetisHeadExpert(
                    self.head_dim,
                    config.moe_expert_intermediate_size,
                    bias=config.mlp_bias,
                    use_fp8=use_fp8,
                    activation=config.moe_activation,
                    precision_kwargs=expert_precision_kwargs,
                )
                for _ in range(self.shared_experts)
            ]
        )
        self.last_aux_loss: torch.Tensor | None = None
        self.perf_counters: dict[str, int] | None = None

    def _bump_counter(self, name: str, amount: int = 1) -> None:
        if self.perf_counters is not None:
            self.perf_counters[name] = self.perf_counters.get(name, 0) + amount

    def _record_expert_load_stats(self, tokens_per_expert: list[int] | torch.Tensor) -> None:
        if self.perf_counters is None:
            return
        if isinstance(tokens_per_expert, torch.Tensor):
            counts = [int(value) for value in tokens_per_expert.detach().cpu().tolist()]
        else:
            counts = [int(value) for value in tokens_per_expert]
        if not counts:
            return
        sorted_counts = sorted(counts)
        p95_index = min(len(sorted_counts) - 1, max(0, math.ceil(0.95 * len(sorted_counts)) - 1))
        self._bump_counter("moe_expert_load_reports")
        self._bump_counter("moe_expert_empty_count", sum(1 for count in counts if count <= 0))
        self._bump_counter("moe_expert_min_rows_sum", sorted_counts[0])
        self._bump_counter("moe_expert_p95_rows_sum", sorted_counts[p95_index])
        self._bump_counter("moe_expert_max_rows_sum", sorted_counts[-1])

    def set_balance_update_scale(self, scale: float) -> None:
        self.balance_update_scale = max(0.0, float(scale))

    def _override_topk(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids = torch.arange(batch_size * seq_len * num_heads, device=device, dtype=torch.long)
        slot_ids = torch.arange(self.top_k, device=device, dtype=torch.long)
        if self.router_override == "force_balanced":
            expert_ids = (row_ids.view(-1, 1) * self.top_k + slot_ids.view(1, -1)) % self.num_experts
        elif self.router_override == "uniform_random":
            hashed = (
                (row_ids.view(-1, 1) + 1) * 1_103_515_245
                + (slot_ids.view(1, -1) + 17) * 12_345
                + (num_heads * 97_531)
            )
            expert_ids = torch.remainder(hashed, self.num_experts)
        else:
            raise ValueError(f"Unknown MoE router override: {self.router_override}")
        weights = torch.full(expert_ids.shape, 1.0 / float(max(self.top_k, 1)), device=device, dtype=dtype)
        return (
            expert_ids.view(batch_size, seq_len, num_heads, self.top_k),
            weights.view(batch_size, seq_len, num_heads, self.top_k),
        )

    def _route(
        self,
        hidden_heads: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, _head_dim = hidden_heads.shape
        if self.router_override != "learned":
            topk_indices, topk_weights = self._override_topk(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                device=hidden_heads.device,
                dtype=hidden_heads.dtype,
            )
            flat_indices = topk_indices.reshape(batch_size * seq_len * num_heads, self.top_k)
            flat_weights = topk_weights.reshape(batch_size * seq_len * num_heads, self.top_k)
            return flat_indices, flat_weights, hidden_heads.new_zeros(()), hidden_heads.new_empty(())
        router_input = hidden_heads.reshape(batch_size * seq_len * num_heads, self.head_dim)
        original_rows = router_input.shape[0]
        pad_rows = 0
        if self.use_fp8 and original_rows % 8 != 0:
            pad_rows = 8 - (original_rows % 8)
            router_input = F.pad(router_input, (0, 0, 0, pad_rows))
        router_latents = self.latent_proj(router_input, is_first_microbatch=is_first_microbatch)
        if pad_rows:
            router_latents = router_latents[:original_rows]
        router_latents = router_latents.reshape(batch_size, seq_len, num_heads, -1)
        router_latents = F.normalize(router_latents.float(), dim=-1).to(dtype=hidden_heads.dtype)
        expert_embeddings = F.normalize(self.expert_embeddings.float(), dim=-1).to(dtype=hidden_heads.dtype)
        router_logits = torch.einsum("bshd,hnd->bshn", router_latents, expert_embeddings)
        router_logits = (router_logits + self.expert_bias.view(1, 1, num_heads, self.num_experts)) / self.router_temperature
        if self.router_score == "sigmoid":
            router_scores = torch.sigmoid(router_logits)
            selection_scores = router_scores
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, num_heads, self.num_experts)
            _topk_selection_scores, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = router_scores.gather(-1, topk_indices)
            topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            balance_probs = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            balance_probs = torch.softmax(router_logits, dim=-1)
            selection_scores = router_logits
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, num_heads, self.num_experts)
            _topk_selection_scores, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = router_logits.gather(-1, topk_indices)
            topk_weights = torch.softmax(topk_scores, dim=-1)
        expert_load = balance_probs.float().mean(dim=(0, 1, 2))
        uniform = expert_load.new_full(expert_load.shape, 1.0 / float(self.num_experts))
        aux_loss = self.num_experts * torch.mean((expert_load - uniform).pow(2))
        if self.balance_strategy == "aux_loss_free_bias":
            self._maybe_update_balance_bias(topk_indices)
        flat_indices = topk_indices.reshape(batch_size * seq_len * num_heads, self.top_k)
        flat_weights = topk_weights.reshape(batch_size * seq_len * num_heads, self.top_k)
        return flat_indices, flat_weights, aux_loss, router_logits

    @torch.no_grad()
    def _maybe_update_balance_bias(self, topk_indices: torch.Tensor) -> None:
        if not self.training or self.balance_bias_update_rate <= 0.0:
            return
        update_rate = self.balance_bias_update_rate
        if self.balance_scale_by_token_fraction:
            update_rate *= self.balance_update_scale
        if update_rate <= 0.0:
            return
        counts = torch.zeros(
            self.num_heads,
            self.num_experts,
            device=topk_indices.device,
            dtype=torch.float32,
        )
        flat_heads = torch.arange(self.num_heads, device=topk_indices.device).view(1, 1, self.num_heads, 1)
        flat_heads = flat_heads.expand_as(topk_indices).reshape(-1)
        counts.index_put_(
            (flat_heads, topk_indices.reshape(-1)),
            torch.ones(topk_indices.numel(), device=topk_indices.device, dtype=torch.float32),
            accumulate=True,
        )
        if _dist_available() and counts.device.type == "cuda":
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.expert_parallel_group)
        load = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        target = 1.0 / float(self.num_experts)
        direction = torch.where(load > target, -1.0, 1.0)
        self.balance_bias.add_(direction.to(dtype=self.balance_bias.dtype) * update_rate)
        self.balance_bias.clamp_(min=-self.balance_bias_clamp, max=self.balance_bias_clamp)

    def _expanded_row_indices(self, num_rows: int, device: torch.device) -> torch.Tensor:
        return (
            torch.arange(num_rows, device=device, dtype=torch.long)
            .view(num_rows, 1)
            .expand(num_rows, self.top_k)
            .reshape(-1)
        )

    def _dispatch_routed_loop(
        self,
        routed_heads: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> torch.Tensor:
        if self.experts is None:
            raise RuntimeError("Loop MoE dispatch requested, but routed expert modules are not initialized.")
        routed_output = torch.zeros_like(routed_heads)
        if self.dispatch_mode == "sorted_loop":
            flat_experts = topk_indices.reshape(-1)
            flat_weights = topk_weights.reshape(-1)
            tokens_per_expert = torch.bincount(flat_experts, minlength=self.num_experts).tolist()
            order = torch.argsort(flat_experts)
            sorted_rows = torch.div(order, self.top_k, rounding_mode="floor")
            sorted_weights = flat_weights.index_select(0, order)
            offset = 0
            for expert_index, split_size in enumerate(tokens_per_expert):
                split_size = int(split_size)
                if split_size <= 0:
                    continue
                row_indices = sorted_rows.narrow(0, offset, split_size)
                expert_input = routed_heads.index_select(0, row_indices)
                expert_output = self.experts[expert_index](
                    expert_input,
                    is_first_microbatch=is_first_microbatch,
                )
                weights = sorted_weights.narrow(0, offset, split_size).to(dtype=expert_output.dtype).unsqueeze(-1)
                routed_output.index_add_(0, row_indices, expert_output * weights)
                self._bump_counter("moe_routed_expert_dispatches")
                offset += split_size
            return routed_output

        for expert_index, expert in enumerate(self.experts):
            selected = topk_indices == expert_index
            if not bool(selected.any().item()):
                continue
            row_indices, slot_indices = torch.nonzero(selected, as_tuple=True)
            expert_input = routed_heads.index_select(0, row_indices)
            expert_output = expert(expert_input, is_first_microbatch=is_first_microbatch)
            weights = topk_weights[row_indices, slot_indices].to(dtype=expert_output.dtype).unsqueeze(-1)
            routed_output.index_add_(0, row_indices, expert_output * weights)
            self._bump_counter("moe_routed_expert_dispatches")
        return routed_output

    def _padded_grouped_assignments(
        self,
        x_perm: torch.Tensor,
        row_indices: torch.Tensor,
        weights: torch.Tensor,
        tokens_per_expert: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor] | None:
        total_assignments = int(sum(tokens_per_expert))
        if total_assignments == 0:
            return None
        expert_count = max(1, len(tokens_per_expert))
        split_alignment = max(1, self.grouped_m_split_alignment)
        padded_splits: list[int] | None = None
        if self.capacity_factor > 0.0:
            mean_tokens = total_assignments / float(expert_count)
            capacity = int(math.ceil(mean_tokens * self.capacity_factor))
            capacity_alignment = max(1, self.capacity_alignment, split_alignment)
            if capacity % capacity_alignment:
                capacity += capacity_alignment - (capacity % capacity_alignment)
            if capacity > 0 and max(tokens_per_expert) <= capacity:
                padded_splits = [capacity] * expert_count
                self._bump_counter("moe_capacity_padded_dispatches")
            else:
                self._bump_counter("moe_capacity_overflow_fallbacks")

        split_floor = max(split_alignment, self.grouped_m_split_min_m)
        if padded_splits is None and split_floor > 1:
            padded_splits = []
            for split_size in tokens_per_expert:
                split_size = int(split_size)
                padded_split_size = max(split_size, split_floor)
                if padded_split_size % split_floor:
                    padded_split_size += split_floor - (padded_split_size % split_floor)
                if padded_split_size == split_size:
                    padded_splits.append(split_size)
                else:
                    padded_splits.append(padded_split_size)

        if padded_splits is None or padded_splits == tokens_per_expert:
            return None

        padded_rows = int(sum(padded_splits))
        padded_x = x_perm.new_zeros((padded_rows, x_perm.shape[-1]))
        padded_weights = weights.new_zeros((padded_rows,))
        valid_positions = row_indices.new_empty((total_assignments,))
        src_offset = 0
        dst_offset = 0
        for split_size, padded_split_size in zip(tokens_per_expert, padded_splits, strict=True):
            split_size = int(split_size)
            if split_size <= 0:
                dst_offset += int(padded_split_size)
                continue
            dst_slice = slice(dst_offset, dst_offset + split_size)
            src_slice = slice(src_offset, src_offset + split_size)
            padded_x[dst_slice] = x_perm[src_slice]
            padded_weights[dst_slice] = weights[src_slice]
            valid_positions[src_slice] = torch.arange(
                dst_offset,
                dst_offset + split_size,
                device=x_perm.device,
                dtype=row_indices.dtype,
            )
            src_offset += split_size
            dst_offset += int(padded_split_size)
        self._bump_counter("moe_capacity_padded_tokens", padded_rows - total_assignments)
        return padded_x, row_indices, padded_weights, padded_splits, valid_positions

    def _dispatch_routed_grouped(
        self,
        routed_heads: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> torch.Tensor:
        if self.grouped_experts is None:
            raise RuntimeError("Grouped MoE dispatch requested, but grouped experts are not initialized.")
        num_rows = topk_indices.shape[0]
        flat_experts = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        with _nvtx_range("moe_grouped_sort"):
            tokens_per_expert = [
                int(value) for value in torch.bincount(flat_experts, minlength=self.num_experts).tolist()
            ]
            order = torch.argsort(flat_experts)
        self._record_expert_load_stats(tokens_per_expert)
        with _nvtx_range("moe_grouped_gather"):
            row_indices = torch.div(order, self.top_k, rounding_mode="floor")
            weights = flat_weights.index_select(0, order)
            x_perm = routed_heads.index_select(0, row_indices)
        valid_positions: torch.Tensor | None = None
        padded = self._padded_grouped_assignments(
            x_perm,
            row_indices,
            weights,
            tokens_per_expert,
        )
        if padded is not None:
            x_perm, row_indices, weights, tokens_per_expert, valid_positions = padded

        with _nvtx_range("moe_grouped_experts"):
            y_perm = self.grouped_experts(
                x_perm,
                tokens_per_expert,
                is_first_microbatch=is_first_microbatch,
                valid_positions=valid_positions,
                routing_weights=weights if self.memory_efficient_permutation else None,
        )
        if valid_positions is not None:
            y_perm = y_perm.index_select(0, valid_positions)
            if not self.memory_efficient_permutation:
                weights = weights.index_select(0, valid_positions)
        with _nvtx_range("moe_grouped_unpermute"):
            routed_output = torch.zeros_like(routed_heads)
            if self.memory_efficient_permutation:
                routed_output.index_add_(0, row_indices, y_perm)
                self._bump_counter("moe_memory_efficient_permute_calls")
            else:
                routed_output.index_add_(0, row_indices, y_perm * weights.to(dtype=y_perm.dtype).unsqueeze(-1))
        self._bump_counter("moe_grouped_expert_dispatches")
        self._bump_counter("moe_grouped_assignments", int(row_indices.numel()))
        self._bump_counter("moe_routed_expert_dispatches", sum(1 for count in tokens_per_expert if count > 0))
        return routed_output

    def _dispatch_routed_expert_parallel(
        self,
        routed_heads: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> torch.Tensor:
        if self.grouped_experts is None:
            raise RuntimeError("Expert-parallel MoE requires grouped expert modules.")
        if not self.expert_parallel_enabled:
            return self._dispatch_routed_bucketed(
                routed_heads,
                topk_indices,
                topk_weights,
                is_first_microbatch=is_first_microbatch,
            )
        if not _dist_available() or routed_heads.device.type != "cuda":
            raise RuntimeError("Expert-parallel MoE requires initialized CUDA torch.distributed training.")

        num_rows = int(topk_indices.shape[0])
        hidden_dim = int(routed_heads.shape[-1])
        flat_experts = topk_indices.reshape(-1).to(dtype=torch.long)
        flat_weights = topk_weights.reshape(-1)
        total_assignments = int(flat_experts.numel())
        if total_assignments == 0:
            return torch.zeros_like(routed_heads)

        with _nvtx_range("moe_ep_pack"):
            owner_ranks = torch.div(flat_experts, self.local_num_experts, rounding_mode="floor")
            owner_ranks = owner_ranks.clamp_(min=0, max=self.expert_parallel_size - 1)
            local_expert_ids = flat_experts - (owner_ranks * self.local_num_experts)
            send_counts_tensor = torch.bincount(owner_ranks, minlength=self.expert_parallel_size).to(
                device=routed_heads.device,
                dtype=torch.int64,
            )
            input_splits, output_splits = _exchange_all_to_all_splits(
                send_counts_tensor,
                group=self.expert_parallel_group,
            )
            send_order = torch.argsort(owner_ranks)
            source_rows = torch.div(
                torch.arange(total_assignments, device=routed_heads.device, dtype=torch.long),
                self.top_k,
                rounding_mode="floor",
            )
            send_rows = source_rows.index_select(0, send_order)
            send_hidden = routed_heads.index_select(0, send_rows).contiguous()
            send_local_expert_ids = local_expert_ids.index_select(0, send_order).contiguous()
            send_weights = flat_weights.index_select(0, send_order)

        with _nvtx_range("moe_ep_all_to_all_in"):
            recv_hidden = _all_to_all_autograd(
                send_hidden,
                input_splits=input_splits,
                output_splits=output_splits,
                group=self.expert_parallel_group,
            )
            recv_weights = (
                _all_to_all_autograd(
                    send_weights.contiguous(),
                    input_splits=input_splits,
                    output_splits=output_splits,
                    group=self.expert_parallel_group,
                )
                if self.memory_efficient_permutation
                else None
            )
            recv_local_expert_ids = _all_to_all_no_grad(
                send_local_expert_ids,
                input_splits=input_splits,
                output_splits=output_splits,
                group=self.expert_parallel_group,
            )

        self._bump_counter("moe_ep_all_to_all_calls", 3 if self.memory_efficient_permutation else 2)
        self._bump_counter("moe_ep_send_assignments", int(send_hidden.shape[0]))
        self._bump_counter("moe_ep_recv_assignments", int(recv_hidden.shape[0]))
        self._bump_counter("moe_grouped_assignments", total_assignments)

        with _nvtx_range("moe_ep_local_experts"):
            if recv_hidden.shape[0] == 0:
                recv_output = recv_hidden.new_zeros((0, hidden_dim))
                local_counts: list[int] = [0] * self.local_num_experts
            else:
                expert_order = torch.argsort(recv_local_expert_ids)
                local_routing_weights = (
                    recv_weights.index_select(0, expert_order).contiguous()
                    if self.memory_efficient_permutation and recv_weights is not None
                    else None
                )
                local_counts = [
                    int(value)
                    for value in torch.bincount(
                        recv_local_expert_ids,
                        minlength=self.local_num_experts,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ]
                self._record_expert_load_stats(local_counts)
                local_x = recv_hidden.index_select(0, expert_order)
                valid_positions: torch.Tensor | None = None
                padded = self._padded_grouped_assignments(
                    local_x,
                    torch.arange(local_x.shape[0], device=local_x.device, dtype=torch.long),
                    local_routing_weights
                    if local_routing_weights is not None
                    else torch.ones(local_x.shape[0], device=local_x.device, dtype=local_x.dtype),
                    local_counts,
                )
                if padded is not None:
                    local_x, _row_indices, padded_weights, local_counts, valid_positions = padded
                    if self.memory_efficient_permutation:
                        local_routing_weights = padded_weights
                local_y = self.grouped_experts(
                    local_x,
                    local_counts,
                    is_first_microbatch=is_first_microbatch,
                    valid_positions=valid_positions,
                    routing_weights=local_routing_weights if self.memory_efficient_permutation else None,
                )
                if valid_positions is not None:
                    local_y = local_y.index_select(0, valid_positions)
                recv_output = recv_hidden.new_empty((recv_hidden.shape[0], hidden_dim))
                recv_output.index_copy_(0, expert_order, local_y)

        with _nvtx_range("moe_ep_all_to_all_out"):
            send_back = _all_to_all_autograd(
                recv_output,
                input_splits=output_splits,
                output_splits=input_splits,
                group=self.expert_parallel_group,
            )

        with _nvtx_range("moe_ep_combine"):
            routed_output = torch.zeros_like(routed_heads)
            if self.memory_efficient_permutation:
                routed_output.index_add_(0, send_rows, send_back)
                self._bump_counter("moe_memory_efficient_permute_calls")
            else:
                routed_output.index_add_(
                    0,
                    send_rows,
                    send_back * send_weights.to(dtype=send_back.dtype).unsqueeze(-1),
                )

        self._bump_counter("moe_grouped_expert_dispatches")
        self._bump_counter("moe_routed_expert_dispatches", sum(1 for count in local_counts if count > 0))
        return routed_output

    def _dispatch_routed_bucketed(
        self,
        routed_heads: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> torch.Tensor:
        if self.grouped_experts is None:
            raise RuntimeError("Bucketed MoE dispatch requested, but grouped experts are not initialized.")
        if not self.permute_fusion or not triton_moe_kernels_available() or not routed_heads.is_cuda:
            return self._dispatch_routed_grouped(
                routed_heads,
                topk_indices,
                topk_weights,
                is_first_microbatch=is_first_microbatch,
            )
        num_rows = topk_indices.shape[0]
        using_fixed_capacity = False
        with _nvtx_range("moe_bucket_dispatch"):
            if self.static_capacity > 0 or self.capacity_factor > 0.0:
                if self.static_capacity > 0:
                    capacity = int(self.static_capacity)
                    capacity_alignment = max(1, self.capacity_alignment)
                    if capacity % capacity_alignment:
                        capacity += capacity_alignment - (capacity % capacity_alignment)
                    self._bump_counter("moe_static_capacity_dispatches")
                else:
                    mean_tokens = float(topk_indices.numel()) / float(self.num_experts)
                    capacity = int(math.ceil(mean_tokens * self.capacity_factor))
                    capacity_alignment = max(1, self.capacity_alignment)
                    if capacity % capacity_alignment:
                        capacity += capacity_alignment - (capacity % capacity_alignment)
                check_overflow = self.overflow_mode != "drop"
                x_perm, assignment_ids, reverse_positions, weights, tokens_per_expert, overflow = capacity_bucket_dispatch(
                    routed_heads,
                    topk_indices,
                    topk_weights,
                    num_experts=self.num_experts,
                    capacity=capacity,
                    check_overflow=check_overflow,
                )
                if overflow and self.overflow_mode == "error":
                    raise RuntimeError(
                        f"MoE static capacity overflowed: capacity={capacity}, "
                        f"assignments={int(topk_indices.numel())}, experts={self.num_experts}."
                    )
                if overflow and self.overflow_mode == "fallback":
                    self._bump_counter("moe_capacity_overflow_fallbacks")
                    x_perm, assignment_ids, reverse_positions, weights, tokens_per_expert = bucket_dispatch(
                        routed_heads,
                        topk_indices,
                        topk_weights,
                        num_experts=self.num_experts,
                    )
                else:
                    using_fixed_capacity = True
                    self._bump_counter("moe_capacity_padded_dispatches")
                    self._bump_counter("moe_capacity_padded_tokens", int(sum(tokens_per_expert)) - int(topk_indices.numel()))
            else:
                if self.moe_backend in {"torch_grouped", "torch_grouped_safe", "torch_bmm"}:
                    x_perm, assignment_ids, reverse_positions, weights, tokens_per_expert = bucket_dispatch_counts(
                        routed_heads,
                        topk_indices,
                        topk_weights,
                        num_experts=self.num_experts,
                    )
                else:
                    x_perm, assignment_ids, reverse_positions, weights, tokens_per_expert = bucket_dispatch(
                        routed_heads,
                        topk_indices,
                        topk_weights,
                        num_experts=self.num_experts,
                    )

        valid_positions: torch.Tensor | None = None
        self._record_expert_load_stats(tokens_per_expert)
        if isinstance(tokens_per_expert, torch.Tensor) and self.moe_backend in {"torch_grouped_safe", "torch_bmm"}:
            tokens_per_expert = [int(value) for value in tokens_per_expert.detach().cpu().tolist()]
        if (
            not using_fixed_capacity
            and self.capacity_factor <= 0.0
            and not isinstance(tokens_per_expert, torch.Tensor)
        ):
            padded = self._padded_grouped_assignments(
                x_perm,
                torch.div(assignment_ids, self.top_k, rounding_mode="floor"),
                weights,
                tokens_per_expert,
            )
            if padded is not None:
                x_perm, _row_indices, weights, tokens_per_expert, valid_positions = padded

        with _nvtx_range("moe_grouped_experts"):
            y_perm = self.grouped_experts(
                x_perm,
                tokens_per_expert,
                is_first_microbatch=is_first_microbatch,
                valid_positions=valid_positions,
                routing_weights=weights if self.memory_efficient_permutation else None,
        )
        if valid_positions is not None:
            y_perm = y_perm.index_select(0, valid_positions)
            if not self.memory_efficient_permutation:
                weights = weights.index_select(0, valid_positions)
        with _nvtx_range("moe_bucket_unpermute"):
            if self.memory_efficient_permutation and self.fused_combine and valid_positions is None:
                routed_output = reverse_unweighted_combine(
                    y_perm,
                    reverse_positions,
                    output_rows=num_rows,
                    top_k=self.top_k,
                )
                self._bump_counter("moe_memory_efficient_permute_calls")
                self._bump_counter("moe_fused_combine_calls")
            elif self.memory_efficient_permutation:
                routed_output = unweighted_unpermute(
                    y_perm,
                    assignment_ids,
                    output_rows=num_rows,
                    top_k=self.top_k,
                )
                self._bump_counter("moe_memory_efficient_permute_calls")
            elif self.fused_combine and valid_positions is None:
                routed_output = reverse_weighted_combine(
                    y_perm,
                    reverse_positions,
                    topk_weights,
                    output_rows=num_rows,
                    top_k=self.top_k,
                )
                self._bump_counter("moe_fused_combine_calls")
            else:
                routed_output = weighted_unpermute(
                    y_perm,
                    assignment_ids,
                    weights,
                    output_rows=num_rows,
                    top_k=self.top_k,
                )
        self._bump_counter("moe_grouped_expert_dispatches")
        self._bump_counter("moe_grouped_assignments", int(assignment_ids.numel()))
        if isinstance(tokens_per_expert, torch.Tensor):
            self._bump_counter("moe_routed_expert_dispatches", self.num_experts)
        else:
            self._bump_counter("moe_routed_expert_dispatches", sum(1 for count in tokens_per_expert if count > 0))
        return routed_output

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        self._bump_counter("moe_forward_calls")
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_heads = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        flat_heads = hidden_heads.reshape(batch_size * seq_len * self.num_heads, self.head_dim)
        routed_heads = flat_heads
        if self.routed_down_proj is not None:
            routed_heads = self.routed_down_proj(flat_heads, is_first_microbatch=is_first_microbatch)
        topk_indices, topk_weights, aux_loss, _router_logits = self._route(
            hidden_heads,
            is_first_microbatch=is_first_microbatch,
        )
        self.last_aux_loss = aux_loss
        self._bump_counter("moe_router_calls")
        output = torch.zeros_like(flat_heads)
        if self.shared_experts:
            with _nvtx_range("moe_shared_experts"):
                shared_output = torch.zeros_like(flat_heads)
                for expert in self.shared:
                    shared_output = shared_output + expert(flat_heads, is_first_microbatch=is_first_microbatch)
                    self._bump_counter("moe_shared_expert_dispatches")
                output = output + (shared_output / float(self.shared_experts))

        with _nvtx_range(f"moe_routed_{self.dispatch_mode}"):
            if self.expert_parallel_enabled:
                routed_output = self._dispatch_routed_expert_parallel(
                    routed_heads,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            elif self.dispatch_mode == "bucketed":
                routed_output = self._dispatch_routed_bucketed(
                    routed_heads,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            elif self.dispatch_mode == "grouped":
                routed_output = self._dispatch_routed_grouped(
                    routed_heads,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            else:
                routed_output = self._dispatch_routed_loop(
                    routed_heads,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )

        if self.routed_up_proj is not None:
            output = output + self.routed_up_proj(routed_output, is_first_microbatch=is_first_microbatch)
        else:
            output = output + routed_output

        return output.reshape(batch_size, seq_len, hidden_size)


class MetisSingleLatentMoE(MetisMultiHeadLatentMoE):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool, layer_idx: int | None = None) -> None:
        super().__init__(config, use_fp8=use_fp8, layer_idx=layer_idx)
        if not config.uses_single_latent_moe:
            raise ValueError("MetisSingleLatentMoE requires config.ffn_type='single_latent_moe'.")
        if self.num_heads != 1:
            raise ValueError("MetisSingleLatentMoE routes one latent vector per token; moe_num_heads must be 1.")
        if self.routed_down_proj is None or self.routed_up_proj is None:
            raise ValueError("MetisSingleLatentMoE requires moe_routed_latent_size < d_model.")

        del self.latent_proj
        del self.expert_embeddings
        del self.expert_bias
        self.single_latent_router_input = str(config.moe_single_latent_router_input)
        router_input_dim = config.d_model if self.single_latent_router_input == "hidden" else self.routed_dim
        self.router = MetisLinear(
            router_input_dim,
            self.num_experts,
            bias=True,
            use_fp8=use_fp8,
            **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
        )
        shared_precision_kwargs = _precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8)
        self.shared = nn.ModuleList(
            [
                MetisHeadExpert(
                    config.d_model,
                    config.moe_expert_intermediate_size,
                    bias=config.mlp_bias,
                    use_fp8=use_fp8,
                    activation=config.moe_activation,
                    precision_kwargs=shared_precision_kwargs,
                )
                for _ in range(self.shared_experts)
            ]
        )

    def _route_latents(
        self,
        latent_states: torch.Tensor,
        *,
        is_first_microbatch: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _latent_dim = latent_states.shape
        if self.router_override != "learned":
            topk_indices, topk_weights = self._override_topk(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=1,
                device=latent_states.device,
                dtype=latent_states.dtype,
            )
            flat_indices = topk_indices.reshape(batch_size * seq_len, self.top_k)
            flat_weights = topk_weights.reshape(batch_size * seq_len, self.top_k)
            return flat_indices, flat_weights, latent_states.new_zeros(()), latent_states.new_empty(())
        router_logits = self.router(latent_states, is_first_microbatch=is_first_microbatch).float()
        router_logits = router_logits / self.router_temperature
        if self.router_score == "sigmoid":
            router_scores = torch.sigmoid(router_logits)
            selection_scores = router_scores
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, self.num_experts)
            _topk_selection_scores, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = router_scores.gather(-1, topk_indices)
            topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            balance_probs = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            balance_probs = torch.softmax(router_logits, dim=-1)
            selection_scores = router_logits
            if self.balance_strategy == "aux_loss_free_bias":
                selection_scores = selection_scores + self.balance_bias.view(1, 1, self.num_experts)
            _topk_selection_scores, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            topk_scores = router_logits.gather(-1, topk_indices)
            topk_weights = torch.softmax(topk_scores, dim=-1)

        expert_load = balance_probs.float().mean(dim=(0, 1))
        uniform = expert_load.new_full(expert_load.shape, 1.0 / float(self.num_experts))
        aux_loss = self.num_experts * torch.mean((expert_load - uniform).pow(2))
        if self.balance_strategy == "aux_loss_free_bias":
            self._maybe_update_balance_bias(topk_indices.unsqueeze(2))

        flat_indices = topk_indices.reshape(batch_size * seq_len, self.top_k)
        flat_weights = topk_weights.to(dtype=latent_states.dtype).reshape(batch_size * seq_len, self.top_k)
        return flat_indices, flat_weights, aux_loss, router_logits

    def forward(self, hidden_states: torch.Tensor, *, is_first_microbatch: bool | None = None) -> torch.Tensor:
        self._bump_counter("moe_forward_calls")
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_size)
        with _nvtx_range("single_latent_moe_down"):
            latent_states = self.routed_down_proj(flat_hidden, is_first_microbatch=is_first_microbatch)
        latent_tokens = latent_states.reshape(batch_size, seq_len, self.routed_dim)
        router_tokens = hidden_states if self.single_latent_router_input == "hidden" else latent_tokens
        topk_indices, topk_weights, aux_loss, _router_logits = self._route_latents(
            router_tokens,
            is_first_microbatch=is_first_microbatch,
        )
        self.last_aux_loss = aux_loss
        self._bump_counter("moe_router_calls")

        output = torch.zeros_like(flat_hidden)
        if self.shared_experts:
            with _nvtx_range("moe_shared_experts"):
                shared_output = torch.zeros_like(flat_hidden)
                for expert in self.shared:
                    shared_output = shared_output + expert(flat_hidden, is_first_microbatch=is_first_microbatch)
                    self._bump_counter("moe_shared_expert_dispatches")
                output = output + (shared_output / float(self.shared_experts))

        with _nvtx_range(f"single_latent_moe_routed_{self.dispatch_mode}"):
            if self.expert_parallel_enabled:
                routed_latent = self._dispatch_routed_expert_parallel(
                    latent_states,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            elif self.dispatch_mode == "bucketed":
                routed_latent = self._dispatch_routed_bucketed(
                    latent_states,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            elif self.dispatch_mode == "grouped":
                routed_latent = self._dispatch_routed_grouped(
                    latent_states,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )
            else:
                routed_latent = self._dispatch_routed_loop(
                    latent_states,
                    topk_indices,
                    topk_weights,
                    is_first_microbatch=is_first_microbatch,
                )

        with _nvtx_range("single_latent_moe_up"):
            output = output + self.routed_up_proj(routed_latent, is_first_microbatch=is_first_microbatch)
        return output.reshape(batch_size, seq_len, hidden_size)


class MetisTransformerBlock(nn.Module):
    def __init__(self, config: MetisMambaConfig, *, use_fp8: bool, layer_idx: int | None = None) -> None:
        super().__init__()
        self.attn_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
        self.self_attn = MetisSelfAttention(config, use_fp8=use_fp8)
        self.use_te_fused_mlp = bool(config.te_fused_mlp and use_fp8)
        if self.use_te_fused_mlp:
            self.mlp = MetisTELayerNormSwiGLU(config, use_fp8=use_fp8)
            self.ffn_norm = None
        else:
            self.ffn_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
            if config.uses_single_latent_moe:
                self.mlp = MetisSingleLatentMoE(config, use_fp8=use_fp8, layer_idx=layer_idx)
            elif config.uses_moe:
                self.mlp = MetisMultiHeadLatentMoE(config, use_fp8=use_fp8, layer_idx=layer_idx)
            else:
                self.mlp = MetisSwiGLU(config, use_fp8=use_fp8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        if self.use_te_fused_mlp:
            hidden_states = hidden_states + self.mlp(hidden_states, is_first_microbatch=is_first_microbatch)
        else:
            hidden_states = hidden_states + self.mlp(
                self.ffn_norm(hidden_states),
                is_first_microbatch=is_first_microbatch,
            )
        return hidden_states


class MetisMoRModel(nn.Module):
    COUNTER_KEYS = (
        "static_dense_forward_calls",
        "dynamic_token_mor_forward_calls",
        "static_sequence_mor_forward_calls",
        "static_block_mor_forward_calls",
        "router_calls",
        "pack_active_tokens_calls",
        "scatter_active_tokens_calls",
        "attention_mask_passed_calls",
        "fa3_calls",
        "sdpa_calls",
        "eager_attention_calls",
        "te_attention_calls",
        "moe_forward_calls",
        "moe_router_calls",
        "moe_shared_expert_dispatches",
        "moe_routed_expert_dispatches",
        "moe_grouped_expert_dispatches",
        "moe_grouped_assignments",
        "moe_capacity_padded_dispatches",
        "moe_capacity_padded_tokens",
        "moe_capacity_overflow_fallbacks",
        "moe_static_capacity_dispatches",
        "moe_fused_combine_calls",
        "moe_memory_efficient_permute_calls",
        "moe_ep_all_to_all_calls",
        "moe_ep_send_assignments",
        "moe_ep_recv_assignments",
        "moe_expert_load_reports",
        "moe_expert_empty_count",
        "moe_expert_min_rows_sum",
        "moe_expert_p95_rows_sum",
        "moe_expert_max_rows_sum",
    )

    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        use_fp8: bool = False,
        fp8_recipe=None,
        fp8_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_fp8 = use_fp8
        self.fp8_recipe = fp8_recipe
        self.fp8_group = fp8_group
        self.training_routing_mode = os.environ.get("METIS_MOR_TRAIN_ROUTING_MODE", "token_pack").strip().lower()
        if self.training_routing_mode not in {"token_pack", "dense_active"}:
            raise ValueError(
                "METIS_MOR_TRAIN_ROUTING_MODE must be one of: token_pack, dense_active."
            )
        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [MetisTransformerBlock(config, use_fp8=use_fp8, layer_idx=layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.final_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
        self.router_norm: nn.Module | None = None
        self.router_up: MetisLinear | None = None
        self.router_out: nn.Linear | None = None
        self.sequence_router_norm: nn.Module | None = None
        self.sequence_router_up: MetisLinear | None = None
        self.sequence_router_out: nn.Linear | None = None
        self.block_router_norm: nn.Module | None = None
        self.block_router_up: MetisLinear | None = None
        self.block_router_out: nn.Linear | None = None

        if config.uses_dynamic_token_mor:
            self.router_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
            self.router_up = MetisLinear(
                config.d_model,
                config.mor_router_hidden_dim,
                bias=True,
                use_fp8=use_fp8,
            )
            self.router_out = nn.Linear(config.mor_router_hidden_dim, config.mor_max_depth)
        elif config.uses_static_sequence_mor:
            self.sequence_router_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
            self.sequence_router_up = MetisLinear(
                config.d_model,
                config.mor_router_hidden_dim,
                bias=True,
                use_fp8=use_fp8,
            )
            self.sequence_router_out = nn.Linear(config.mor_router_hidden_dim, config.mor_max_depth)
        elif config.uses_static_block_mor:
            self.block_router_norm = build_rms_norm_module(config.d_model, eps=1e-6, use_fp8=use_fp8)
            self.block_router_up = MetisLinear(
                config.d_model,
                config.mor_router_hidden_dim,
                bias=True,
                use_fp8=use_fp8,
            )
            self.block_router_out = nn.Linear(config.mor_router_hidden_dim, config.mor_max_depth)

        self.perf_counters: dict[str, int] | None = {}
        self._moe_aux_accum: torch.Tensor | None = None
        self.reset_perf_counters()

    def reset_perf_counters(self) -> None:
        if os.environ.get("METIS_DISABLE_PERF_COUNTERS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            self.perf_counters = None
        else:
            self.perf_counters = {key: 0 for key in self.COUNTER_KEYS}
        for layer in self.layers:
            layer.self_attn.perf_counters = self.perf_counters
            if hasattr(layer.mlp, "perf_counters"):
                layer.mlp.perf_counters = self.perf_counters

    def get_perf_counters(self) -> dict[str, int]:
        return {} if self.perf_counters is None else dict(self.perf_counters)

    def _bump_counter(self, name: str, amount: int = 1) -> None:
        if self.perf_counters is not None:
            self.perf_counters[name] = self.perf_counters.get(name, 0) + amount

    def _build_position_ids(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

    def _static_attention_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if self.config.attention_mask_mode == "causal_none":
            return None
        return attention_mask

    def _zero_aux(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_zeros(())

    def _reset_moe_aux(self, hidden_states: torch.Tensor) -> None:
        self._moe_aux_accum = self._zero_aux(hidden_states)

    def _record_moe_aux(self, hidden_states: torch.Tensor) -> None:
        if self._moe_aux_accum is None:
            self._moe_aux_accum = self._zero_aux(hidden_states)
        layer_aux = []
        for layer in self.layers:
            aux_loss = getattr(layer.mlp, "last_aux_loss", None)
            if aux_loss is not None:
                layer_aux.append(aux_loss)
        if layer_aux:
            self._moe_aux_accum = self._moe_aux_accum + torch.stack(layer_aux).mean()

    def _take_moe_aux(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._moe_aux_accum is None:
            return self._zero_aux(hidden_states)
        aux_loss = self._moe_aux_accum
        self._moe_aux_accum = None
        return aux_loss

    def _set_moe_balance_update_scale(self, scale: float) -> None:
        for layer in self.layers:
            if hasattr(layer.mlp, "set_balance_update_scale"):
                layer.mlp.set_balance_update_scale(scale)

    def _run_shared_stack(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        is_first_microbatch: bool | None = None,
        moe_balance_update_scale: float = 1.0,
    ) -> torch.Tensor:
        self._set_moe_balance_update_scale(moe_balance_update_scale)
        with fp8_autocast_context(
            enabled=self.use_fp8,
            recipe=self.fp8_recipe,
            fp8_group=self.fp8_group,
        ):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    is_first_microbatch=is_first_microbatch,
                )
        self._record_moe_aux(hidden_states)
        self._set_moe_balance_update_scale(1.0)
        return hidden_states

    def _run_router_mlp(
        self,
        norm: nn.Module,
        up: MetisLinear,
        out: nn.Linear,
        hidden_states: torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        original_leading_shape = hidden_states.shape[:-1]
        hidden_size = hidden_states.shape[-1]
        router_input = hidden_states.reshape(-1, hidden_size)
        original_rows = router_input.shape[0]
        pad_rows = 0
        if self.use_fp8 and original_rows % 8 != 0:
            pad_rows = 8 - (original_rows % 8)
            router_input = F.pad(router_input, (0, 0, 0, pad_rows))
        with fp8_autocast_context(
            enabled=self.use_fp8,
            recipe=self.fp8_recipe,
            fp8_group=self.fp8_group,
        ):
            router_hidden = F.silu(
                up(
                    norm(router_input),
                    is_first_microbatch=is_first_microbatch,
                )
            )
        router_logits = out(router_hidden)
        if pad_rows:
            router_logits = router_logits[:original_rows]
        return router_logits.reshape(*original_leading_shape, router_logits.shape[-1])

    def _route_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        is_first_microbatch: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.router_norm is None or self.router_up is None or self.router_out is None:
            raise RuntimeError("Dynamic token MoR was requested, but the token router is disabled in this config.")
        self._bump_counter("router_calls")
        router_logits = self._run_router_mlp(
            self.router_norm,
            self.router_up,
            self.router_out,
            hidden_states,
            is_first_microbatch=is_first_microbatch,
        )
        soft_route_probs = torch.softmax(router_logits / self.config.mor_router_temperature, dim=-1)
        if self.training:
            hard_route_probs = F.gumbel_softmax(
                router_logits,
                tau=self.config.mor_router_temperature,
                hard=True,
                dim=-1,
            )
        else:
            hard_route_probs = torch.zeros_like(soft_route_probs)
            hard_route_probs.scatter_(
                -1,
                torch.argmax(soft_route_probs, dim=-1, keepdim=True),
                1.0,
            )
        chosen_depths = torch.argmax(hard_route_probs, dim=-1) + 1
        if attention_mask is not None:
            chosen_depths = chosen_depths * attention_mask.to(torch.long)
        depth_values = torch.arange(
            1,
            self.config.mor_max_depth + 1,
            device=hidden_states.device,
            dtype=soft_route_probs.dtype,
        )
        expected_depth = torch.sum(soft_route_probs * depth_values, dim=-1)
        if attention_mask is not None:
            valid_tokens = attention_mask.to(torch.bool)
            mean_depth = expected_depth.masked_select(valid_tokens).mean()
            router_probs_for_reg = soft_route_probs.masked_select(valid_tokens.unsqueeze(-1)).view(-1, self.config.mor_max_depth)
        else:
            mean_depth = expected_depth.mean()
            router_probs_for_reg = soft_route_probs.reshape(-1, self.config.mor_max_depth)
        route_aux_loss = (mean_depth - self.config.mor_target_avg_depth).pow(2)
        if self.config.mor_router_entropy_coef > 0:
            entropy = -(router_probs_for_reg * router_probs_for_reg.clamp_min(1e-8).log()).sum(dim=-1).mean()
            max_entropy = torch.log(hidden_states.new_tensor(float(self.config.mor_max_depth))).clamp_min(1e-6)
            route_aux_loss = route_aux_loss - (self.config.mor_router_entropy_coef * (entropy / max_entropy))
        if self.config.mor_router_z_loss_coef > 0:
            z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2)
            if attention_mask is not None:
                z_loss = z_loss.masked_select(attention_mask.to(torch.bool))
            route_aux_loss = route_aux_loss + (self.config.mor_router_z_loss_coef * z_loss.mean().to(route_aux_loss.dtype))
        return soft_route_probs, hard_route_probs, chosen_depths, route_aux_loss

    def _default_static_capacities(self, item_count: int, configured_depth2: int, configured_depth3: int) -> tuple[int, int]:
        if configured_depth2 > 0 or configured_depth3 > 0:
            depth2 = min(item_count, configured_depth2)
            depth3 = min(depth2, configured_depth3)
            return depth2, depth3
        extra_items = max(0, int(round(item_count * max(self.config.mor_target_avg_depth - 1.0, 0.0))))
        depth3 = min(item_count, int(round(extra_items * 0.375)))
        depth2 = min(item_count, max(0, extra_items - depth3))
        depth3 = min(depth2, depth3)
        return depth2, depth3

    def _topk_indices(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        return torch.topk(scores, k=min(k, scores.numel()), sorted=False).indices

    def _static_route_aux_loss(self, hidden_states: torch.Tensor, mean_depth: torch.Tensor) -> torch.Tensor:
        if not self.config.mor_train_router:
            return self._zero_aux(hidden_states)
        return (mean_depth - self.config.mor_target_avg_depth).pow(2)

    def _forward_static_dense(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        is_first_microbatch: bool | None,
    ) -> dict[str, Any]:
        self._bump_counter("static_dense_forward_calls")
        hidden_states = self.embed_tokens(input_ids)
        self._reset_moe_aux(hidden_states)
        position_ids = self._build_position_ids(input_ids, position_ids)
        current_hidden = self._run_shared_stack(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        final_hidden = self.final_norm(current_hidden)
        one = hidden_states.new_ones(())
        return {
            "hidden_states": [final_hidden],
            "final_hidden": final_hidden,
            "route_probs": None,
            "chosen_depths": None,
            "route_aux_loss": self._zero_aux(hidden_states),
            "moe_aux_loss": self._take_moe_aux(hidden_states),
            "mean_depth": one,
            "active_token_ratios": hidden_states.new_ones(1),
        }

    def _forward_static_sequence_mor(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        is_first_microbatch: bool | None,
    ) -> dict[str, Any]:
        if self.sequence_router_norm is None or self.sequence_router_up is None or self.sequence_router_out is None:
            raise RuntimeError("Static sequence MoR was requested, but the sequence router is disabled in this config.")
        self._bump_counter("static_sequence_mor_forward_calls")
        hidden_states = self.embed_tokens(input_ids)
        self._reset_moe_aux(hidden_states)
        batch_size, _seq_len, _hidden_size = hidden_states.shape
        position_ids = self._build_position_ids(input_ids, position_ids)
        stack_attention_mask = self._static_attention_mask(attention_mask)
        current_hidden = self._run_shared_stack(
            hidden_states,
            attention_mask=stack_attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        step_hidden_states = [self.final_norm(current_hidden)]

        router_repr = current_hidden.detach().mean(dim=1)
        self._bump_counter("router_calls")
        router_logits = self._run_router_mlp(
            self.sequence_router_norm,
            self.sequence_router_up,
            self.sequence_router_out,
            router_repr,
            is_first_microbatch=is_first_microbatch,
        )
        route_probs = torch.softmax(router_logits / self.config.mor_router_temperature, dim=-1)
        depth2_cap, depth3_cap = self._default_static_capacities(
            batch_size,
            self.config.mor_depth2_capacity_sequences,
            self.config.mor_depth3_capacity_sequences,
        )

        chosen_depths = torch.ones(batch_size, device=input_ids.device, dtype=torch.long)
        if self.config.mor_max_depth >= 2 and depth2_cap > 0:
            depth2_scores = router_logits[:, 1:].amax(dim=-1)
            depth2_indices = self._topk_indices(depth2_scores, depth2_cap)
            selected_hidden = current_hidden.index_select(0, depth2_indices)
            selected_position_ids = position_ids.index_select(0, depth2_indices)
            updated_hidden = self._run_shared_stack(
                selected_hidden,
                attention_mask=None,
                position_ids=selected_position_ids,
                is_first_microbatch=is_first_microbatch,
            )
            current_hidden = current_hidden.index_copy(0, depth2_indices, updated_hidden)
            chosen_depths = chosen_depths.index_fill(0, depth2_indices, 2)
            step_hidden_states.append(self.final_norm(current_hidden))

            if self.config.mor_max_depth >= 3 and depth3_cap > 0:
                depth3_scores = router_logits.index_select(0, depth2_indices)[:, 2]
                local_depth3 = self._topk_indices(depth3_scores, min(depth3_cap, depth2_indices.numel()))
                depth3_indices = depth2_indices.index_select(0, local_depth3)
                selected_hidden = current_hidden.index_select(0, depth3_indices)
                selected_position_ids = position_ids.index_select(0, depth3_indices)
                updated_hidden = self._run_shared_stack(
                    selected_hidden,
                    attention_mask=None,
                    position_ids=selected_position_ids,
                    is_first_microbatch=is_first_microbatch,
                )
                current_hidden = current_hidden.index_copy(0, depth3_indices, updated_hidden)
                chosen_depths = chosen_depths.index_fill(0, depth3_indices, 3)
                step_hidden_states.append(self.final_norm(current_hidden))

        final_hidden = self.final_norm(current_hidden)
        mean_depth = hidden_states.new_tensor(1.0 + (float(depth2_cap) / batch_size) + (float(depth3_cap) / batch_size))
        active_ratios = hidden_states.new_tensor([1.0, float(depth2_cap) / batch_size, float(depth3_cap) / batch_size])
        return {
            "hidden_states": step_hidden_states,
            "final_hidden": final_hidden,
            "route_probs": route_probs,
            "chosen_depths": chosen_depths,
            "route_aux_loss": self._static_route_aux_loss(hidden_states, mean_depth),
            "moe_aux_loss": self._take_moe_aux(hidden_states),
            "mean_depth": mean_depth,
            "active_token_ratios": active_ratios,
        }

    def _forward_static_block_mor(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        is_first_microbatch: bool | None,
    ) -> dict[str, Any]:
        if self.block_router_norm is None or self.block_router_up is None or self.block_router_out is None:
            raise RuntimeError("Static block MoR was requested, but the block router is disabled in this config.")
        self._bump_counter("static_block_mor_forward_calls")
        hidden_states = self.embed_tokens(input_ids)
        self._reset_moe_aux(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        block_size = self.config.mor_block_size
        if seq_len % block_size != 0:
            raise ValueError(f"static_block_mor requires seq_len divisible by mor_block_size ({seq_len} % {block_size}).")
        num_blocks = seq_len // block_size
        total_blocks = batch_size * num_blocks
        position_ids = self._build_position_ids(input_ids, position_ids)
        stack_attention_mask = self._static_attention_mask(attention_mask)
        current_hidden = self._run_shared_stack(
            hidden_states,
            attention_mask=stack_attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        step_hidden_states = [self.final_norm(current_hidden)]

        blocks = current_hidden.detach().contiguous().view(batch_size, num_blocks, block_size, hidden_size)
        router_repr = blocks.mean(dim=2).reshape(total_blocks, hidden_size)
        self._bump_counter("router_calls")
        router_logits = self._run_router_mlp(
            self.block_router_norm,
            self.block_router_up,
            self.block_router_out,
            router_repr,
            is_first_microbatch=is_first_microbatch,
        )
        route_probs = torch.softmax(router_logits / self.config.mor_router_temperature, dim=-1).view(
            batch_size,
            num_blocks,
            self.config.mor_max_depth,
        )
        depth2_cap, depth3_cap = self._default_static_capacities(
            total_blocks,
            self.config.mor_depth2_capacity_blocks,
            self.config.mor_depth3_capacity_blocks,
        )

        flat_hidden = current_hidden.contiguous().view(batch_size, num_blocks, block_size, hidden_size).reshape(
            total_blocks,
            block_size,
            hidden_size,
        )
        flat_position_ids = position_ids.contiguous().view(batch_size, num_blocks, block_size).reshape(total_blocks, block_size)
        chosen_depths = torch.ones(total_blocks, device=input_ids.device, dtype=torch.long)

        if self.config.mor_max_depth >= 2 and depth2_cap > 0:
            depth2_scores = router_logits[:, 1:].amax(dim=-1)
            depth2_indices = self._topk_indices(depth2_scores, depth2_cap)
            selected_hidden = flat_hidden.index_select(0, depth2_indices)
            selected_position_ids = flat_position_ids.index_select(0, depth2_indices)
            updated_hidden = self._run_shared_stack(
                selected_hidden,
                attention_mask=None,
                position_ids=selected_position_ids,
                is_first_microbatch=is_first_microbatch,
            )
            flat_hidden = flat_hidden.index_copy(0, depth2_indices, updated_hidden)
            chosen_depths = chosen_depths.index_fill(0, depth2_indices, 2)
            current_hidden = flat_hidden.view(batch_size, num_blocks, block_size, hidden_size).reshape(
                batch_size,
                seq_len,
                hidden_size,
            )
            step_hidden_states.append(self.final_norm(current_hidden))

            if self.config.mor_max_depth >= 3 and depth3_cap > 0:
                depth3_scores = router_logits.index_select(0, depth2_indices)[:, 2]
                local_depth3 = self._topk_indices(depth3_scores, min(depth3_cap, depth2_indices.numel()))
                depth3_indices = depth2_indices.index_select(0, local_depth3)
                selected_hidden = flat_hidden.index_select(0, depth3_indices)
                selected_position_ids = flat_position_ids.index_select(0, depth3_indices)
                updated_hidden = self._run_shared_stack(
                    selected_hidden,
                    attention_mask=None,
                    position_ids=selected_position_ids,
                    is_first_microbatch=is_first_microbatch,
                )
                flat_hidden = flat_hidden.index_copy(0, depth3_indices, updated_hidden)
                chosen_depths = chosen_depths.index_fill(0, depth3_indices, 3)
                current_hidden = flat_hidden.view(batch_size, num_blocks, block_size, hidden_size).reshape(
                    batch_size,
                    seq_len,
                    hidden_size,
                )
                step_hidden_states.append(self.final_norm(current_hidden))

        final_hidden = self.final_norm(
            flat_hidden.view(batch_size, num_blocks, block_size, hidden_size).reshape(batch_size, seq_len, hidden_size)
        )
        mean_depth = hidden_states.new_tensor(
            1.0 + (float(depth2_cap) / total_blocks) + (float(depth3_cap) / total_blocks)
        )
        active_ratios = hidden_states.new_tensor(
            [1.0, float(depth2_cap) / total_blocks, float(depth3_cap) / total_blocks]
        )
        return {
            "hidden_states": step_hidden_states,
            "final_hidden": final_hidden,
            "route_probs": route_probs,
            "chosen_depths": chosen_depths.view(batch_size, num_blocks),
            "route_aux_loss": self._static_route_aux_loss(hidden_states, mean_depth),
            "moe_aux_loss": self._take_moe_aux(hidden_states),
            "mean_depth": mean_depth,
            "active_token_ratios": active_ratios,
        }

    def _forward_dynamic_token_mor(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        is_first_microbatch: bool | None,
    ) -> dict[str, Any]:
        self._bump_counter("dynamic_token_mor_forward_calls")
        hidden_states = self.embed_tokens(input_ids)
        self._reset_moe_aux(hidden_states)
        position_ids = self._build_position_ids(input_ids, position_ids)
        valid_tokens = attention_mask.to(torch.bool) if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        soft_route_probs, _hard_route_probs, chosen_depths, route_aux_loss = self._route_tokens(
            hidden_states,
            attention_mask=attention_mask,
            is_first_microbatch=is_first_microbatch,
        )

        step_hidden_states: list[torch.Tensor] = []
        active_ratios: list[torch.Tensor] = []
        current_hidden = hidden_states
        valid_count = valid_tokens.sum().clamp_min(1)
        use_dense_active = (
            self.training
            and (
                self.training_routing_mode == "dense_active"
                or self.config.disable_token_packing
                or self.config.disable_token_scatter
            )
        )
        for step_index in range(1, self.config.mor_max_depth + 1):
            active_mask = (chosen_depths >= step_index) & valid_tokens
            active_ratio = active_mask.sum().to(hidden_states.dtype) / valid_count.to(hidden_states.dtype)
            active_ratios.append(active_ratio)
            moe_balance_update_scale = float(active_ratio.detach().item()) if self.training else 1.0
            if use_dense_active:
                updated_hidden = self._run_shared_stack(
                    current_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    is_first_microbatch=is_first_microbatch,
                    moe_balance_update_scale=moe_balance_update_scale,
                )
                current_hidden = torch.where(active_mask.unsqueeze(-1), updated_hidden, current_hidden)
            elif bool(active_mask.all().item()):
                current_hidden = self._run_shared_stack(
                    current_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    is_first_microbatch=is_first_microbatch,
                    moe_balance_update_scale=moe_balance_update_scale,
                )
            else:
                self._bump_counter("pack_active_tokens_calls")
                packed = pack_active_tokens(
                    current_hidden,
                    position_ids,
                    active_mask,
                    pad_multiple=self.config.fp8_pad_multiple if self.use_fp8 else 1,
                )
                if packed is not None:
                    packed_hidden, packed_positions, packed_mask, packed_indices = packed
                    updated_hidden = self._run_shared_stack(
                        packed_hidden,
                        attention_mask=packed_mask,
                        position_ids=packed_positions,
                        is_first_microbatch=is_first_microbatch,
                        moe_balance_update_scale=moe_balance_update_scale,
                    )
                    if self.config.disable_token_scatter:
                        current_hidden = updated_hidden
                    else:
                        self._bump_counter("scatter_active_tokens_calls")
                        current_hidden = scatter_active_tokens(
                            current_hidden,
                            updated_hidden,
                            packed_mask,
                            packed_indices,
                            active_mask,
                        )
            step_hidden_states.append(self.final_norm(current_hidden))

        if self.config.disable_depth_stack:
            final_hidden = step_hidden_states[-1]
        else:
            stacked_hidden = torch.stack(step_hidden_states, dim=2)
            if self.training:
                final_hidden = torch.sum(
                    stacked_hidden * soft_route_probs.unsqueeze(-1).to(stacked_hidden.dtype),
                    dim=2,
                )
            else:
                gather_index = (chosen_depths.clamp_min(1) - 1).unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    1,
                    stacked_hidden.size(-1),
                )
                final_hidden = torch.gather(stacked_hidden, 2, gather_index).squeeze(2)

        depth_values = torch.arange(
            1,
            self.config.mor_max_depth + 1,
            device=hidden_states.device,
            dtype=soft_route_probs.dtype,
        )
        mean_depth = torch.sum(soft_route_probs * depth_values, dim=-1)
        if attention_mask is not None:
            mean_depth = mean_depth.masked_select(valid_tokens).mean()
        else:
            mean_depth = mean_depth.mean()

        return {
            "hidden_states": step_hidden_states,
            "final_hidden": final_hidden,
            "route_probs": soft_route_probs,
            "chosen_depths": chosen_depths,
            "route_aux_loss": route_aux_loss,
            "moe_aux_loss": self._take_moe_aux(hidden_states),
            "mean_depth": mean_depth,
            "active_token_ratios": torch.stack(active_ratios),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        is_first_microbatch: bool | None = None,
    ) -> dict[str, Any]:
        if self.config.training_mode == "static_dense_pretrain" or not self.config.mor_enabled:
            return self._forward_static_dense(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                is_first_microbatch=is_first_microbatch,
            )
        if self.config.training_mode == "static_sequence_mor":
            return self._forward_static_sequence_mor(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                is_first_microbatch=is_first_microbatch,
            )
        if self.config.training_mode == "static_block_mor":
            return self._forward_static_block_mor(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                is_first_microbatch=is_first_microbatch,
            )
        return self._forward_dynamic_token_mor(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )


class MetisMoRLMHeadModel(nn.Module):
    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        use_fp8: bool = False,
        fp8_recipe=None,
        fp8_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_fp8 = use_fp8
        self.fp8_recipe = fp8_recipe
        self.fp8_group = fp8_group
        self.backbone = MetisMoRModel(
            config,
            use_fp8=use_fp8,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
        )
        self.lm_head = MetisLinear(
            config.d_model,
            config.padded_vocab_size,
            bias=False,
            use_fp8=use_fp8,
            **_precision_kwargs_for_value(config, "bf16", use_fp8=use_fp8),
        )
        self.fused_linear_ce = None
        if config.lm_loss_impl == "liger_fused_linear_ce":
            self.fused_linear_ce = _load_liger_fused_linear_ce()(ignore_index=-100)
        self.model_family = config.model_type
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.tie_weights()

    def reset_perf_counters(self) -> None:
        self.backbone.reset_perf_counters()

    def get_perf_counters(self) -> dict[str, int]:
        return self.backbone.get_perf_counters()

    def tie_weights(self) -> None:
        self.lm_head.impl.weight = self.backbone.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, MetisLinear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module.impl, "bias", None) is not None:
                nn.init.zeros_(module.impl.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        is_first_microbatch: bool | None = None,
        return_logits: bool = True,
        **_: Any,
    ) -> MetisCausalLMOutput:
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        final_hidden = outputs["final_hidden"]
        logits = None
        loss = None
        lm_loss = None

        if labels is not None and self.fused_linear_ce is not None:
            shift_hidden = final_hidden[:, :-1, :].contiguous().view(-1, final_hidden.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            with _nvtx_range("fused_linear_ce"):
                lm_loss = self.fused_linear_ce(self.lm_head.weight, shift_hidden, shift_labels)
        else:
            with _nvtx_range("lm_head"):
                with fp8_autocast_context(
                    enabled=self.use_fp8,
                    recipe=self.fp8_recipe,
                    fp8_group=self.fp8_group,
                ):
                    logits = self.lm_head(
                        final_hidden,
                        is_first_microbatch=is_first_microbatch,
                    )
            if labels is not None:
                with _nvtx_range("cross_entropy"):
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    lm_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

        if labels is not None:
            if lm_loss is None:
                raise RuntimeError("Language-model loss was not computed despite labels being provided.")
            if return_logits and logits is None:
                with _nvtx_range("lm_head_for_logits"):
                    with fp8_autocast_context(
                        enabled=self.use_fp8,
                        recipe=self.fp8_recipe,
                        fp8_group=self.fp8_group,
                    ):
                        logits = self.lm_head(
                            final_hidden,
                            is_first_microbatch=is_first_microbatch,
                        )
            elif not return_logits:
                logits = None
            route_aux_loss = outputs.get("route_aux_loss")
            moe_aux_loss = outputs.get("moe_aux_loss")
            aux_loss = lm_loss.new_zeros(())
            if route_aux_loss is not None:
                aux_loss = aux_loss + (route_aux_loss * self.config.mor_router_aux_loss_coef)
            if moe_aux_loss is not None:
                aux_loss = aux_loss + (moe_aux_loss * self.config.moe_aux_loss_coef)
            loss = lm_loss + aux_loss

        return MetisCausalLMOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            hidden_states=outputs["hidden_states"],
            route_probs=outputs["route_probs"],
            chosen_depths=outputs["chosen_depths"],
            route_aux_loss=outputs["route_aux_loss"],
            moe_aux_loss=outputs.get("moe_aux_loss"),
            mean_depth=outputs["mean_depth"],
            active_token_ratios=outputs["active_token_ratios"],
        )


class MetisMoRRewardModel(nn.Module):
    def __init__(
        self,
        config: MetisMambaConfig,
        *,
        use_fp8: bool = False,
        fp8_recipe=None,
        fp8_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_fp8 = use_fp8
        self.fp8_recipe = fp8_recipe
        self.fp8_group = fp8_group
        self.backbone = MetisMoRModel(
            config,
            use_fp8=use_fp8,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
        )
        self.score_head = nn.Linear(config.d_model, 1, bias=False)
        self.model_family = f"{config.model_type}_reward_model"
        self.apply(self._init_weights)

    def reset_perf_counters(self) -> None:
        self.backbone.reset_perf_counters()

    def get_perf_counters(self) -> dict[str, int]:
        return self.backbone.get_perf_counters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, MetisLinear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module.impl, "bias", None) is not None:
                nn.init.zeros_(module.impl.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        is_first_microbatch: bool | None = None,
        **_: Any,
    ) -> MetisRewardOutput:
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_first_microbatch=is_first_microbatch,
        )
        hidden_states = outputs["final_hidden"]
        batch_size = hidden_states.size(0)
        if attention_mask is None:
            last_indices = torch.full(
                (batch_size,),
                hidden_states.size(1) - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            last_indices = attention_mask.to(torch.long).sum(dim=-1).clamp_min(1) - 1
        pooled_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_indices]
        rewards = self.score_head(pooled_hidden).squeeze(-1)
        return MetisRewardOutput(
            rewards=rewards,
            route_aux_loss=outputs["route_aux_loss"],
            moe_aux_loss=outputs.get("moe_aux_loss"),
            mean_depth=outputs["mean_depth"],
            active_token_ratios=outputs["active_token_ratios"],
        )
