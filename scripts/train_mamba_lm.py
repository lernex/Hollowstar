from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from metis_mamba import MetisMambaConfig, build_model, cosine_lr, parse_torch_dtype
from metis_mamba.checkpoint_compat import filter_state_dict_for_model
from metis_mamba.checkpoints import atomic_torch_save
from metis_mamba.fp8 import (
    build_fp8_recipe,
    build_nvfp4_recipe,
    transformer_engine_is_available,
    transformer_engine_runtime_supports_fp8_block_scaling,
    transformer_engine_runtime_supports_mxfp8,
    transformer_engine_runtime_supports_nvfp4,
    transformer_engine_supports_fp8_block_scaling,
    transformer_engine_supports_mxfp8,
    transformer_engine_supports_nvfp4,
)
from metis_mamba.optim import OptimizerBuildSummary, build_optimizer_from_args


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_runtime(requested: str | None) -> tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Metis training requires CUDA devices.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return device, distributed, rank, local_rank, world_size

    return choose_device(requested), distributed, rank, local_rank, world_size


def cleanup_runtime(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def aggregate_interval_stats(
    *,
    loss_total: float,
    updates: int,
    tokens: int,
    device: torch.device,
    distributed: bool,
) -> tuple[float, int, int]:
    if not distributed:
        return loss_total, updates, tokens
    stats = torch.tensor(
        [loss_total, float(updates), float(tokens)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item()), int(stats[2].item())


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _check_tensor_finite(name: str, tensor: torch.Tensor, *, step: int, micro_step: int | None = None) -> None:
    finite = torch.isfinite(tensor.detach())
    if bool(finite.all().item()):
        return
    data = tensor.detach().float()
    safe = data.nan_to_num(posinf=0.0, neginf=0.0)
    nonfinite = int((~finite).sum().item())
    prefix = f"step={step}"
    if micro_step is not None:
        prefix += f" micro_step={micro_step}"
    print(
        "nonfinite_tensor "
        f"{prefix} name={name} dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        f"nonfinite={nonfinite} finite_min={float(safe.amin().item()):.6e} "
        f"finite_max={float(safe.amax().item()):.6e}",
        flush=True,
    )
    raise FloatingPointError(f"Non-finite tensor detected: {name} at {prefix}")


def _check_model_finite(
    model: torch.nn.Module,
    *,
    kind: str,
    step: int,
    micro_step: int | None = None,
) -> None:
    for name, param in model.named_parameters():
        target = param.grad if kind.startswith("grad") else param
        if target is None:
            continue
        _check_tensor_finite(f"{kind}:{name}", target, step=step, micro_step=micro_step)


def _check_optimizer_state_finite(
    optimizer: torch.optim.Optimizer,
    param_name_map: dict[torch.nn.Parameter, str],
    *,
    step: int,
) -> None:
    for param, state in optimizer.state.items():
        param_name = param_name_map.get(param, "<unnamed_param>")
        for state_name, value in state.items():
            if isinstance(value, torch.Tensor):
                _check_tensor_finite(f"optimizer:{param_name}:{state_name}", value, step=step)


def _log_model_absmax(
    model: torch.nn.Module,
    *,
    kind: str,
    step: int,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    rows: list[tuple[float, str, tuple[int, ...], torch.dtype]] = []
    for name, param in model.named_parameters():
        target = param.grad if kind.startswith("grad") else param
        if target is None:
            continue
        data = target.detach().float().abs().nan_to_num(posinf=float("inf"), neginf=float("inf"))
        rows.append((float(data.amax().item()), name, tuple(target.shape), target.dtype))
    rows.sort(key=lambda row: row[0], reverse=True)
    for rank, (absmax, name, shape, dtype) in enumerate(rows[:top_k], start=1):
        print(
            f"tensor_absmax step={step} kind={kind} rank={rank} "
            f"name={name} dtype={dtype} shape={shape} absmax={absmax:.6e}",
            flush=True,
        )


def _iter_tensors(value: object):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _install_forward_nonfinite_hooks(
    model: torch.nn.Module,
    state: dict[str, int | None],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return handles

    def hook_for(name: str):
        def _hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            step = int(state.get("step") or -1)
            micro_step = state.get("micro_step")
            for index, tensor in enumerate(_iter_tensors(output)):
                _check_tensor_finite(
                    f"forward_hook:{name}:{index}",
                    tensor,
                    step=step,
                    micro_step=None if micro_step is None else int(micro_step),
                )

        return _hook

    targets: list[tuple[str, torch.nn.Module]] = []
    seen_module_ids: set[int] = set()

    def add_target(name: str, module: torch.nn.Module) -> None:
        module_id = id(module)
        if module_id in seen_module_ids:
            return
        seen_module_ids.add(module_id)
        targets.append((name, module))

    embed_tokens = getattr(backbone, "embed_tokens", None)
    if isinstance(embed_tokens, torch.nn.Module):
        add_target("backbone.embed_tokens", embed_tokens)
    for layer_idx, layer in enumerate(getattr(backbone, "layers", [])):
        for child_name in ("attn_norm", "self_attn", "ffn_norm", "mlp"):
            child = getattr(layer, child_name, None)
            if isinstance(child, torch.nn.Module):
                add_target(f"backbone.layers.{layer_idx}.{child_name}", child)
                if child_name in {"self_attn", "mlp"}:
                    for sub_name, submodule in child.named_modules():
                        if sub_name:
                            add_target(f"backbone.layers.{layer_idx}.{child_name}.{sub_name}", submodule)
    final_norm = getattr(backbone, "final_norm", None)
    if isinstance(final_norm, torch.nn.Module):
        add_target("backbone.final_norm", final_norm)
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module):
        add_target("lm_head", lm_head)

    for name, module in targets:
        handles.append(module.register_forward_hook(hook_for(name)))
    return handles


def broadcast_pair(
    first: float,
    second: float,
    *,
    device: torch.device,
    distributed: bool,
) -> tuple[float, float]:
    if not distributed:
        return first, second
    payload = torch.tensor([first, second], device=device, dtype=torch.float64)
    dist.broadcast(payload, src=0)
    return float(payload[0].item()), float(payload[1].item())


def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def build_gate_schedule(
    *,
    stage_manifest: dict,
    block_size: int,
    local_batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    start_step: int = 0,
    tokens_already_seen: int = 0,
) -> dict[int, str]:
    tokens_per_step = local_batch_size * grad_accum_steps * block_size * world_size
    schedule: dict[int, str] = {}
    for gate in stage_manifest.get("gates", []):
        target_tokens = int(gate["tokens"])
        if target_tokens <= tokens_already_seen:
            continue
        label = str(gate.get("label", f"gate_{target_tokens}"))
        step = start_step + max(1, math.ceil((target_tokens - tokens_already_seen) / tokens_per_step))
        schedule[step] = label
    return schedule


def infer_checkpoint_tokens_seen(
    checkpoint: dict,
    *,
    fallback_block_size: int,
    fallback_world_size: int,
    fallback_batch_size: int,
    fallback_grad_accum_steps: int,
) -> int:
    if "total_tokens_seen" in checkpoint:
        return int(checkpoint["total_tokens_seen"])
    step = int(checkpoint.get("step", 0))
    train_args = checkpoint.get("train_args") or {}
    block_size = int(
        train_args.get(
            "block_size",
            checkpoint.get("model_config", {}).get("block_size", fallback_block_size),
        )
    )
    world_size = int(train_args.get("world_size", fallback_world_size))
    batch_size = int(train_args.get("batch_size", fallback_batch_size))
    grad_accum_steps = int(train_args.get("grad_accum_steps", fallback_grad_accum_steps))
    return step * world_size * batch_size * grad_accum_steps * block_size


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_enable_cuda_speedups(
    device: torch.device,
    *,
    matmul_precision: str | None,
    tf32: bool,
) -> None:
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True


def maybe_mark_compile_step(enabled: bool) -> None:
    if enabled and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def nvtx_range(name: str, device: torch.device):
    if device.type == "cuda" and hasattr(torch.cuda, "nvtx"):
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def unwrap_model(model):
    target = model.module if isinstance(model, DDP) else model
    return getattr(target, "_orig_mod", target)


def is_sharded_expert_parameter(name: str) -> bool:
    lowered = name.lower()
    return ".mlp.grouped_experts." in lowered or ".mlp.experts." in lowered


def expert_parallel_active(
    config: MetisMambaConfig,
    *,
    distributed: bool,
    world_size: int,
) -> bool:
    return bool(distributed and world_size > 1 and config.uses_moe and int(config.moe_expert_parallel_size) > 1)


@torch.no_grad()
def broadcast_replicated_parameters(model: torch.nn.Module, *, device: torch.device, distributed: bool) -> None:
    if not distributed:
        return
    with nvtx_range("sync_replicated_params", device):
        for name, param in model.named_parameters():
            if is_sharded_expert_parameter(name):
                continue
            dist.broadcast(param.data, src=0)
        for _name, buffer in model.named_buffers():
            dist.broadcast(buffer.data, src=0)


@torch.no_grad()
def all_reduce_replicated_gradients(
    model: torch.nn.Module,
    *,
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> None:
    if not distributed or world_size <= 1:
        return
    with nvtx_range("sync_replicated_grads", device):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if is_sharded_expert_parameter(name):
                param.grad.div_(float(world_size))
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(float(world_size))


def rank_checkpoint_path(path: Path, *, rank: int, expert_parallel: bool) -> Path:
    if not expert_parallel or rank == 0:
        return path
    return path.with_name(f"{path.stem}.rank{rank:03d}{path.suffix}")


def checkpoint_was_expert_parallel(checkpoint: dict) -> bool:
    train_args = checkpoint.get("train_args") or {}
    model_config = checkpoint.get("model_config") or {}
    size = train_args.get("expert_parallel_size", model_config.get("moe_expert_parallel_size", 1))
    try:
        return int(size) > 1
    except (TypeError, ValueError):
        return False


def load_rank_checkpoint(path: Path, *, rank: int, expert_parallel: bool) -> tuple[dict, Path] | tuple[None, Path]:
    load_path = rank_checkpoint_path(path, rank=rank, expert_parallel=expert_parallel)
    if load_path.exists():
        return torch.load(load_path, map_location="cpu"), load_path
    if not path.exists():
        return None, load_path
    checkpoint = torch.load(path, map_location="cpu")
    if expert_parallel and rank != 0 and checkpoint_was_expert_parallel(checkpoint):
        raise FileNotFoundError(
            f"Missing expert-parallel checkpoint shard for rank {rank}: {load_path}. "
            f"The coordinator checkpoint {path} only contains rank 0's routed expert shard."
        )
    return checkpoint, path


def reset_perf_counters(model) -> None:
    target = unwrap_model(model)
    if hasattr(target, "reset_perf_counters"):
        target.reset_perf_counters()


def get_perf_counters(model) -> dict[str, int]:
    target = unwrap_model(model)
    if hasattr(target, "get_perf_counters"):
        return target.get_perf_counters()
    return {}


def apply_training_mode_overrides(config: MetisMambaConfig, training_mode: str | None) -> None:
    if not training_mode:
        return
    config.training_mode = training_mode
    if training_mode == "static_dense_pretrain":
        config.mor_enabled = False
        config.mor_train_router = False
        config.mor_runtime_mode = "disabled"
        config.mor_max_depth = 1
        config.mor_target_avg_depth = 1.0
        config.mor_router_aux_loss_coef = 0.0
        config.attention_mask_mode = "causal_none"
        config.disable_depth_stack = True
        config.disable_token_packing = True
        config.disable_token_scatter = True
    elif training_mode == "static_sequence_mor":
        config.mor_enabled = True
        config.mor_train_router = False
        config.mor_runtime_mode = "static_sequence"
        config.mor_target_avg_depth = 1.4
        config.mor_router_aux_loss_coef = 0.0
        if config.mor_depth2_capacity_sequences <= 0:
            config.mor_depth2_capacity_sequences = 10
        if config.mor_depth3_capacity_sequences <= 0:
            config.mor_depth3_capacity_sequences = 6
        config.attention_mask_mode = "causal_none"
        config.disable_depth_stack = False
        config.disable_token_packing = True
        config.disable_token_scatter = True
    elif training_mode == "static_block_mor":
        config.mor_enabled = True
        config.mor_train_router = False
        config.mor_runtime_mode = "static_block"
        config.mor_target_avg_depth = 1.4
        config.mor_router_aux_loss_coef = 0.0
        if config.mor_depth2_capacity_blocks <= 0:
            config.mor_depth2_capacity_blocks = 80
        if config.mor_depth3_capacity_blocks <= 0:
            config.mor_depth3_capacity_blocks = 48
        config.attention_mask_mode = "causal_none"
        config.disable_depth_stack = False
        config.disable_token_packing = True
        config.disable_token_scatter = True
    elif training_mode == "dynamic_token_mor":
        config.mor_enabled = True
        config.mor_train_router = True
        config.mor_runtime_mode = "dynamic_token"
        if config.mor_max_depth < 2:
            config.mor_max_depth = 3
        if config.mor_target_avg_depth <= 1.0:
            config.mor_target_avg_depth = 1.5
        if config.mor_router_aux_loss_coef <= 0.0:
            config.mor_router_aux_loss_coef = 0.01
        config.attention_mask_mode = "auto"
        config.disable_depth_stack = False
        config.disable_token_packing = False
        config.disable_token_scatter = False


def apply_stage_training_overrides(
    config: MetisMambaConfig,
    *,
    train_stage: str,
    stage_manifest: dict,
    explicit_training_mode: str | None,
) -> None:
    stage_training_mode = stage_manifest.get("training_mode")
    if explicit_training_mode is None and isinstance(stage_training_mode, str):
        apply_training_mode_overrides(config, stage_training_mode)
    mor = stage_manifest.get("mor", {})
    if not isinstance(mor, dict):
        return
    field_map = {
        "enabled": "mor_enabled",
        "train_router": "mor_train_router",
        "runtime_mode": "mor_runtime_mode",
        "max_depth": "mor_max_depth",
        "router_hidden_dim": "mor_router_hidden_dim",
        "router_temperature": "mor_router_temperature",
        "router_aux_loss_coef": "mor_router_aux_loss_coef",
        "router_entropy_coef": "mor_router_entropy_coef",
        "router_z_loss_coef": "mor_router_z_loss_coef",
        "target_avg_depth": "mor_target_avg_depth",
        "disable_depth_stack": "disable_depth_stack",
        "disable_token_packing": "disable_token_packing",
        "disable_token_scatter": "disable_token_scatter",
        "moe_balance_scale_by_token_fraction": "moe_balance_scale_by_token_fraction",
    }
    for source_key, config_key in field_map.items():
        if source_key in mor:
            setattr(config, config_key, mor[source_key])
    if train_stage == "continued_pretrain" and explicit_training_mode is None and not config.uses_dynamic_token_mor:
        apply_training_mode_overrides(config, "dynamic_token_mor")


def update_mor_schedule(config: MetisMambaConfig, stage_manifest: dict, tokens_seen: int) -> None:
    if not config.uses_dynamic_token_mor:
        return
    mor = stage_manifest.get("mor", {})
    if not isinstance(mor, dict):
        return
    warmup_tokens = int(mor.get("target_avg_depth_warmup_tokens", 0) or 0)
    end_depth = float(mor.get("target_avg_depth_end", mor.get("target_avg_depth", config.mor_target_avg_depth)))
    start_depth = float(mor.get("target_avg_depth_start", end_depth))
    if warmup_tokens > 0:
        progress = min(max(float(tokens_seen) / float(warmup_tokens), 0.0), 1.0)
        config.mor_target_avg_depth = start_depth + ((end_depth - start_depth) * progress)
    else:
        config.mor_target_avg_depth = end_depth
    aux_end = float(mor.get("router_aux_loss_coef_end", mor.get("router_aux_loss_coef", config.mor_router_aux_loss_coef)))
    aux_start = float(mor.get("router_aux_loss_coef_start", aux_end))
    if warmup_tokens > 0:
        progress = min(max(float(tokens_seen) / float(warmup_tokens), 0.0), 1.0)
        config.mor_router_aux_loss_coef = aux_start + ((aux_end - aux_start) * progress)
    else:
        config.mor_router_aux_loss_coef = aux_end


def resolve_precision(
    args: argparse.Namespace,
    device: torch.device,
    config: MetisMambaConfig,
) -> tuple[torch.dtype, bool, object | None, str]:
    if bool(args.nvfp4) and bool(args.fp8):
        raise RuntimeError("Choose one low-precision recipe: --nvfp4 or --fp8, not both.")
    if args.nvfp4:
        if device.type != "cuda":
            raise RuntimeError("NVFP4 training requires CUDA.")
        if not transformer_engine_is_available():
            raise RuntimeError(
                "NVFP4 was requested, but Transformer Engine is not installed. "
                "Install a Blackwell-capable transformer_engine[pytorch] build."
            )
        if not transformer_engine_supports_nvfp4():
            raise RuntimeError(
                "NVFP4 was requested, but this Transformer Engine build does not expose "
                "NVFP4BlockScaling. Use TE 2.14+ / a Blackwell-capable container."
            )
        if not transformer_engine_runtime_supports_nvfp4(
            disable_rht=args.nvfp4_disable_rht,
            disable_2d_quantization=args.nvfp4_disable_2d_quantization,
            disable_stochastic_rounding=args.nvfp4_disable_stochastic_rounding,
        ):
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                "NVFP4 was requested, and Transformer Engine exposes NVFP4BlockScaling, "
                f"but this GPU/runtime combination is not safe for Metis training yet "
                f"(cuda capability={capability}). On RTX PRO 6000 / SM120, the default TE "
                "NVFP4 recipe fails exact Metis GEMM smoke tests in the Hadamard/RHT path. "
                "Use the SM120-safe recipe flags: --nvfp4-disable-rht "
                "--nvfp4-disable-2d-quantization --nvfp4-disable-stochastic-rounding."
            )
        config.low_precision_mode = "nvfp4"
        if config.nvfp4_uses_mxfp8 and not transformer_engine_supports_mxfp8():
            raise RuntimeError(
                "Metis-1.5 is configured to use MXFP8 for selected NVFP4 stability surfaces, "
                "but this Transformer Engine build does not expose MXFP8BlockScaling."
            )
        if config.nvfp4_uses_mxfp8 and not transformer_engine_runtime_supports_mxfp8():
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                "Metis-1.5 is configured to use MXFP8 for selected NVFP4 stability surfaces, "
                f"but Transformer Engine does not support MXFP8 on this runtime "
                f"(cuda capability={capability}). Use fp8_block or BF16 on RTX PRO 6000 / SM120 for now."
            )
        if config.nvfp4_uses_fp8_block and not transformer_engine_supports_fp8_block_scaling():
            raise RuntimeError(
                "Metis-1.5 is configured to use Float8BlockScaling for selected NVFP4 stability "
                "surfaces, but this Transformer Engine build does not expose Float8BlockScaling."
            )
        if config.nvfp4_uses_fp8_block and not transformer_engine_runtime_supports_fp8_block_scaling():
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                "Metis-1.5 is configured to use Float8BlockScaling for selected NVFP4 stability "
                f"surfaces, but Transformer Engine does not support it on this runtime "
                f"(cuda capability={capability}). Use CUDA 12.9+/TE main or set those surfaces to BF16."
            )
        if args.dtype not in {"bf16", "bfloat16"}:
            raise RuntimeError("NVFP4 training keeps master weights in BF16. Use --dtype bf16 with --nvfp4.")
        recipe = build_nvfp4_recipe(
            disable_rht=args.nvfp4_disable_rht,
            disable_2d_quantization=args.nvfp4_disable_2d_quantization,
            disable_stochastic_rounding=args.nvfp4_disable_stochastic_rounding,
        )
        if config.nvfp4_uses_mxfp8:
            return torch.bfloat16, True, recipe, "NVFP4 compute with MXFP8 stability linears and BF16 master weights"
        if config.nvfp4_uses_fp8_block:
            return torch.bfloat16, True, recipe, "NVFP4 compute with Float8BlockScaling stability linears and BF16 master weights"
        return torch.bfloat16, True, recipe, "NVFP4 compute with BF16 master weights"
    fp8_enabled = bool(args.fp8)
    if fp8_enabled:
        if device.type != "cuda":
            raise RuntimeError("FP8 training requires CUDA.")
        if not transformer_engine_is_available():
            raise RuntimeError(
                "FP8 was requested, but Transformer Engine is not installed. "
                "Install transformer_engine[pytorch] in the GPU training environment."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("FP8 training requires CUDA.")
        major, _minor = torch.cuda.get_device_capability(device)
        if major < 9:
            raise RuntimeError("FP8 training in this repo is only enabled for Hopper-class GPUs (H100/H800).")
        if args.dtype not in {"bf16", "bfloat16"}:
            raise RuntimeError("FP8 training keeps master weights in BF16. Use --dtype bf16 with --fp8.")
        recipe = build_fp8_recipe(
            format_name=args.fp8_format,
            margin=args.fp8_margin,
            amax_history_len=args.fp8_amax_history_len,
            amax_compute_algo=args.fp8_amax_compute_algo,
            fp8_dpa=bool(args.fp8_dpa or config.fp8_dpa),
            fp8_mha=bool(args.fp8_mha or config.fp8_mha),
        )
        config.low_precision_mode = "fp8"
        return torch.bfloat16, True, recipe, "FP8 compute with BF16 master weights"
    config.low_precision_mode = "none"
    return parse_torch_dtype(args.dtype), False, None, args.dtype.upper()


def get_batch(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("Dataset is too small for the selected block size.")
    positions = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
    offsets = positions[:, None] + np.arange(block_size, dtype=np.int64)[None, :]
    batch = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64))
    x = batch.contiguous()
    # MetisMoRLMHeadModel shifts labels internally, so labels must stay aligned
    # with input_ids here. Pre-shifting would train a two-token-ahead objective.
    y = x.clone()
    non_blocking = device.type == "cuda"
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


class CudaBatchPrefetcher:
    def __init__(
        self,
        data: np.memmap,
        *,
        batch_size: int,
        block_size: int,
        device: torch.device,
        depth: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CudaBatchPrefetcher requires a CUDA device.")
        max_start = len(data) - block_size
        if max_start <= 0:
            raise ValueError("Dataset is too small for the selected block size.")
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.depth = max(1, int(depth))
        self.max_start = max_start
        self.offset_base = np.arange(block_size, dtype=np.int64)
        self.queue: queue.Queue[torch.Tensor | BaseException] = queue.Queue(maxsize=self.depth)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="metis-cuda-batch-prefetch", daemon=True)
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self.worker.start()
        self._schedule_next()

    def _make_cpu_batch(self) -> torch.Tensor:
        positions = np.random.randint(0, self.max_start, size=(self.batch_size,), dtype=np.int64)
        offsets = positions[:, None] + self.offset_base[None, :]
        arr = np.asarray(self.data[offsets], dtype=np.int32)
        x_cpu = torch.empty((self.batch_size, self.block_size), dtype=torch.int32, pin_memory=True)
        x_cpu.copy_(torch.from_numpy(arr), non_blocking=False)
        return x_cpu

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                batch = self._make_cpu_batch()
            except BaseException as exc:  # propagated through next()
                self.queue.put(exc)
                return
            self.queue.put(batch)

    def _next_cpu_batch(self) -> torch.Tensor:
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def _schedule_next(self) -> None:
        x_cpu = self._next_cpu_batch()
        with torch.cuda.stream(self.stream):
            x = x_cpu.to(self.device, non_blocking=True)
            # MetisMoRLMHeadModel shifts labels internally, so labels stay aligned.
            # Keep CPU staging narrow and widen labels after the asynchronous copy.
            y = x.to(torch.long)
        self.next_batch = (x, y)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if self.next_batch is None:
            raise RuntimeError("CUDA prefetcher has no scheduled batch.")
        batch = self.next_batch
        self._schedule_next()
        return batch

    def close(self) -> None:
        self.stop_event.set()
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self.worker.join(timeout=1.0)


@torch.no_grad()
def estimate_loss(
    model,
    *,
    train_data: np.memmap,
    val_data: np.memmap,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    compiled: bool,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            maybe_mark_compile_step(compiled)
            output = model(xb, labels=yb, is_first_microbatch=True, return_logits=False)
            tracked_loss = output.lm_loss if output.lm_loss is not None else output.loss
            losses[index] = float(tracked_loss.item())
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    total_tokens_seen: int,
    best_val_loss: float,
    train_args: dict,
    elapsed_seconds: float,
    rank: int = 0,
    expert_parallel: bool = False,
) -> None:
    save_path = rank_checkpoint_path(path, rank=rank, expert_parallel=expert_parallel)
    size_bytes = atomic_torch_save(
        save_path,
        {
            "model_family": getattr(model, "model_family", model.config.model_type),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config.to_dict(),
            "step": step,
            "total_tokens_seen": total_tokens_seen,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    if rank == 0 or expert_parallel:
        print(
            f"saved checkpoint {save_path.name} | step {step:6d} | size {size_bytes / (1024 ** 3):.2f} GiB",
            flush=True,
        )


def estimate_tflops(tokens_per_second: float, param_count: int) -> float:
    return (6.0 * float(param_count) * tokens_per_second) / 1e12


def estimate_active_param_count(config: MetisMambaConfig, mean_depth: float) -> int:
    return config.estimate_active_params(mean_depth)


def build_optimizer(
    model,
    args: argparse.Namespace,
    optimizer_manifest: dict[str, object],
) -> tuple[torch.optim.Optimizer, OptimizerBuildSummary | None]:
    return build_optimizer_from_args(model, args, optimizer_manifest)


def preinitialize_adamw_state(optimizer: torch.optim.Optimizer) -> int:
    initialized = 0
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is None or not getattr(param, "requires_grad", False):
                continue
            state = optimizer.state[param]
            if state:
                continue
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            initialized += 1
    return initialized


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Metis decoder base model on memmap token data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    parser.add_argument("--train-stage", choices=["pretrain", "continued_pretrain"], default="pretrain")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument(
        "--training-mode",
        choices=["dynamic_token_mor", "static_dense_pretrain", "static_sequence_mor", "static_block_mor"],
        default=None,
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-format", choices=["HYBRID", "E4M3"], default="HYBRID")
    parser.add_argument("--fp8-margin", type=int, default=0)
    parser.add_argument("--fp8-amax-history-len", type=int, default=16)
    parser.add_argument("--fp8-amax-compute-algo", default="max")
    parser.add_argument("--fp8-dpa", action="store_true")
    parser.add_argument("--fp8-mha", action="store_true")
    parser.add_argument("--fp8-expert-precision", choices=["fp8", "bf16"], default=None)
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--nvfp4-disable-rht", action="store_true")
    parser.add_argument("--nvfp4-disable-2d-quantization", action="store_true")
    parser.add_argument("--nvfp4-disable-stochastic-rounding", action="store_true")
    parser.add_argument("--nvfp4-final-expert-layers", type=int, default=None)
    parser.add_argument("--nvfp4-final-expert-precision", choices=["bf16", "mxfp8", "fp8_block", "nvfp4"], default=None)
    parser.add_argument(
        "--moe-backend",
        choices=[
            "te_grouped",
            "torch_grouped",
            "torch_grouped_safe",
            "torch_bmm",
            "torch_looped",
            "cudnnfe",
            "triton",
            "cutlass",
        ],
        default=None,
    )
    parser.add_argument("--moe-dispatch-mode", choices=["loop", "sorted_loop", "grouped", "bucketed"], default=None)
    parser.add_argument("--moe-static-capacity", type=int, default=None)
    parser.add_argument("--moe-capacity-factor", type=float, default=None)
    parser.add_argument("--moe-capacity-alignment", type=int, default=None)
    parser.add_argument("--moe-torch-grouped-min-m", type=int, default=None)
    parser.add_argument("--moe-overflow-mode", choices=["fallback", "drop", "error"], default=None)
    parser.add_argument("--moe-router-override", choices=["learned", "force_balanced", "uniform_random"], default=None)
    parser.add_argument("--moe-expert-parallel-size", type=int, default=None)
    parser.add_argument("--moe-memory-efficient-permutation", action="store_true")
    parser.add_argument("--moe-overlap-expert-parallel-comm", action="store_true")
    parser.add_argument("--moe-delay-wgrad-compute", action="store_true")
    parser.add_argument("--moe-router-fusion", action="store_true")
    parser.add_argument("--disable-moe-permute-fusion", action="store_true")
    parser.add_argument("--disable-moe-fused-combine", action="store_true")
    parser.add_argument("--disable-moe-aux", action="store_true")
    parser.add_argument("--disable-moe-balance-update", action="store_true")
    parser.add_argument("--moe-graphable", action="store_true")
    parser.add_argument("--fp8-pad-multiple", type=int, default=None)
    parser.add_argument("--te-fused-mlp", action="store_true")
    parser.add_argument("--te-dot-product-attention", action="store_true")
    parser.add_argument("--disable-native-gqa-attention", action="store_true")
    parser.add_argument(
        "--lm-loss-impl",
        choices=["standard", "liger_fused_linear_ce"],
        default="standard",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh-optimizer-on-resume", action="store_true")
    parser.add_argument("--fresh-scheduler-on-resume", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default=None)
    parser.add_argument("--muon-beta", type=float, default=None)
    parser.add_argument("--muon-ns-steps", type=int, default=None)
    parser.add_argument("--muon-lr-scale", type=float, default=None)
    parser.add_argument("--muon-include-routed-experts", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--hybrid-adamw-impl", choices=["loop", "foreach"], default=None)
    parser.add_argument("--prefetch-batches", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--allow-low-precision-compile", action="store_true")
    parser.add_argument("--debug-nonfinite", action="store_true")
    parser.add_argument("--debug-forward-nonfinite-hooks", action="store_true")
    parser.add_argument("--debug-absmax-top-k", type=int, default=0)
    parser.add_argument("--retain-standard-ce-logits", action="store_true")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    checkpoint_requested = False

    def request_checkpoint(signum, _frame) -> None:
        nonlocal checkpoint_requested
        checkpoint_requested = True
        print(f"Received signal {signum}; will save latest.pt after the current optimizer step.", flush=True)

    signal.signal(signal.SIGUSR1, request_checkpoint)

    device, distributed, rank, local_rank, world_size = setup_runtime(args.device)
    prefetcher: CudaBatchPrefetcher | None = None
    try:
        seed = args.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        manifest = json.loads(Path(args.manifest).read_text())
        config = MetisMambaConfig.from_dict(manifest["model"])
        stage_manifest = manifest.get(args.train_stage)
        if not isinstance(stage_manifest, dict):
            raise ValueError(f"Manifest does not contain train stage {args.train_stage!r}.")
        optimizer_manifest = manifest.get("optimizer", {})
        if not isinstance(optimizer_manifest, dict):
            optimizer_manifest = {}
        if args.optimizer is None:
            args.optimizer = str(optimizer_manifest.get("name", "adamw")).lower().replace("-", "_")
        apply_stage_training_overrides(
            config,
            train_stage=args.train_stage,
            stage_manifest=stage_manifest,
            explicit_training_mode=args.training_mode,
        )
        apply_training_mode_overrides(config, args.training_mode)
        update_mor_schedule(config, stage_manifest, tokens_seen=0)
        config.te_fused_mlp = bool(args.te_fused_mlp)
        config.te_dot_product_attention = bool(args.te_dot_product_attention)
        config.native_gqa_attention = not bool(args.disable_native_gqa_attention)
        config.lm_loss_impl = args.lm_loss_impl
        if args.moe_backend is not None:
            config.moe_backend = args.moe_backend
        if args.moe_dispatch_mode is not None:
            config.moe_dispatch_mode = args.moe_dispatch_mode
        if args.moe_static_capacity is not None:
            config.moe_static_capacity = args.moe_static_capacity
        if args.moe_capacity_factor is not None:
            config.moe_capacity_factor = args.moe_capacity_factor
        if args.moe_capacity_alignment is not None:
            config.moe_capacity_alignment = args.moe_capacity_alignment
        if args.moe_torch_grouped_min_m is not None:
            config.moe_torch_grouped_min_m = args.moe_torch_grouped_min_m
        if args.moe_overflow_mode is not None:
            config.moe_overflow_mode = args.moe_overflow_mode
        if args.moe_router_override is not None:
            config.moe_router_override = args.moe_router_override
        if args.moe_expert_parallel_size is not None:
            config.moe_expert_parallel_size = args.moe_expert_parallel_size
        if args.moe_memory_efficient_permutation:
            config.moe_memory_efficient_permutation = True
        if args.moe_overlap_expert_parallel_comm:
            config.moe_overlap_expert_parallel_comm = True
        if args.moe_delay_wgrad_compute:
            config.moe_delay_wgrad_compute = True
        if args.moe_router_fusion:
            config.moe_router_fusion = True
        if args.disable_moe_permute_fusion:
            config.moe_permute_fusion = False
        if int(config.moe_expert_parallel_size) > 1 and world_size == 1:
            if is_main_process(rank):
                print(
                    "Warning: moe_expert_parallel_size is set but WORLD_SIZE=1; "
                    "disabling expert parallel for this local run.",
                    flush=True,
                )
            config.moe_expert_parallel_size = 1
        if args.disable_moe_fused_combine:
            config.moe_fused_combine = False
        if args.disable_moe_aux:
            config.moe_aux_loss_coef = 0.0
        if args.disable_moe_balance_update:
            config.moe_balance_bias_update_rate = 0.0
        if args.moe_graphable:
            config.moe_graphable = True
            config.moe_dispatch_mode = "bucketed"
            config.moe_overflow_mode = "drop"
            config.moe_fused_combine = True
        if args.fp8_pad_multiple is not None:
            config.fp8_pad_multiple = args.fp8_pad_multiple
        if args.nvfp4_final_expert_layers is not None:
            config.nvfp4_final_expert_layers = args.nvfp4_final_expert_layers
        if args.nvfp4_final_expert_precision is not None:
            config.nvfp4_final_expert_precision = args.nvfp4_final_expert_precision
        if args.fp8_expert_precision is not None:
            config.fp8_expert_precision = args.fp8_expert_precision
        config.validate()
        expert_parallel = expert_parallel_active(
            config,
            distributed=distributed,
            world_size=world_size,
        )
        if expert_parallel and int(config.moe_expert_parallel_size) != world_size:
            raise RuntimeError(
                "Metis expert parallel currently maps one expert-parallel rank per process. "
                f"Launch with WORLD_SIZE={config.moe_expert_parallel_size}, got {world_size}."
            )
        param_audit = config.param_application_audit()
        rough_param_apps = int(param_audit["rough_total_param_apps_per_token"])
        if config.uses_single_latent_moe and rough_param_apps > 450_000_000:
            raise RuntimeError(
                "Metis-1.5 single_latent_moe compute audit exceeded the H100 launch gate: "
                f"rough_total_param_apps_per_token={rough_param_apps:,} > 450,000,000. "
                "Reduce latent_dim, top_k, expert_hidden, shared experts, or layer count before training."
            )
        manifest_world_size = int(manifest.get("hardware", {}).get("world_size", world_size))
        if manifest_world_size != world_size and is_main_process(rank):
            print(
                f"Warning: manifest hardware.world_size={manifest_world_size} but runtime world_size={world_size}.",
                flush=True,
            )
        model_dtype, fp8_enabled, fp8_recipe, precision_label = resolve_precision(args, device, config)
        maybe_enable_cuda_speedups(
            device,
            matmul_precision=args.matmul_precision,
            tf32=args.tf32,
        )

        data_dir = Path(args.data_dir)
        meta = load_meta(data_dir)
        dtype = np.dtype(meta["dtype"])
        train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
        val_data = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")
        if args.prefetch_batches > 0 and device.type == "cuda":
            prefetcher = CudaBatchPrefetcher(
                train_data,
                batch_size=args.batch_size,
                block_size=config.block_size,
                device=device,
                depth=args.prefetch_batches,
            )

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(
            config,
            use_fp8=fp8_enabled,
            fp8_recipe=fp8_recipe,
            fp8_group=dist.group.WORLD if distributed and fp8_enabled else None,
        )
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if args.init_checkpoint:
            init_path = Path(args.init_checkpoint)
            init_checkpoint, resolved_init_path = load_rank_checkpoint(
                init_path,
                rank=rank,
                expert_parallel=expert_parallel,
            )
            if init_checkpoint is None:
                raise FileNotFoundError(f"Init checkpoint is missing: {init_path}")
            model_state, conversions = filter_state_dict_for_model(model, init_checkpoint["model_state_dict"])
            model.load_state_dict(
                model_state,
                strict=False,
            )
            if is_main_process(rank) and conversions:
                print(f"Converted legacy checkpoint modules for fused layout: {len(conversions)}", flush=True)
        model.to(device=device, dtype=model_dtype)
        param_name_map = {param: name for name, param in model.named_parameters()}
        forward_debug_state: dict[str, int | None] = {"step": None, "micro_step": None}
        forward_debug_handles: list[torch.utils.hooks.RemovableHandle] = []
        if args.debug_forward_nonfinite_hooks:
            if args.compile and is_main_process(rank):
                print("Forward non-finite hooks requested; disabling torch.compile for hook fidelity.", flush=True)
            args.compile = False
            forward_debug_handles = _install_forward_nonfinite_hooks(model, forward_debug_state)
            if is_main_process(rank):
                print(f"Forward non-finite hooks installed: {len(forward_debug_handles)} modules.", flush=True)
        compile_enabled = (
            args.compile
            and device.type == "cuda"
            and not distributed
            and (not fp8_enabled or bool(args.allow_low_precision_compile))
        )
        if args.compile and distributed and is_main_process(rank):
            print("Distributed run detected; disabling torch.compile for stability.", flush=True)
        if args.compile and fp8_enabled and not args.allow_low_precision_compile and is_main_process(rank):
            print("Transformer Engine low-precision run detected; disabling torch.compile for stability.", flush=True)
        if args.compile and fp8_enabled and args.allow_low_precision_compile and is_main_process(rank):
            print("Experimental: torch.compile enabled for Transformer Engine low-precision run.", flush=True)
        train_model = torch.compile(model, mode=args.compile_mode) if compile_enabled else model
        if expert_parallel:
            broadcast_replicated_parameters(model, device=device, distributed=distributed)
        if distributed and not expert_parallel:
            train_model = DDP(train_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        if is_main_process(rank):
            print(f"Using device: {device}")
            print(f"World size: {world_size}")
            print(f"Model family: {model.model_family}")
            print(f"Train stage: {args.train_stage}")
            print(f"Training mode: {config.training_mode}")
            print(f"TE fused MLP: {config.te_fused_mlp}")
            print(f"TE dot-product attention: {config.te_dot_product_attention}")
            print(f"Native GQA attention: {config.native_gqa_attention}")
            print(f"LM loss impl: {config.lm_loss_impl}")
            print(f"MoE backend: {config.moe_backend}")
            print(f"MoE dispatch: {config.moe_dispatch_mode}")
            print(f"MoE router override: {config.moe_router_override}")
            print(
                f"MoE expert parallel: {config.moe_expert_parallel_size} "
                f"({'enabled' if expert_parallel else 'disabled'})",
                flush=True,
            )
            if config.uses_moe:
                print(
                    "MoE optimization profile: "
                    f"dispatcher={config.moe_token_dispatcher_type} "
                    f"flex_backend={config.moe_flex_dispatcher_backend} "
                    f"memory_efficient_permutation={config.moe_memory_efficient_permutation} "
                    f"overlap_ep_comm={config.moe_overlap_expert_parallel_comm} "
                    f"delay_wgrad={config.moe_delay_wgrad_compute} "
                    f"router_fusion={config.moe_router_fusion} "
                    f"permute_fusion={config.moe_permute_fusion}",
                    flush=True,
                )
            if config.moe_static_capacity > 0:
                print(
                    f"MoE static capacity: {config.moe_static_capacity} | "
                    f"overflow={config.moe_overflow_mode} | fused_combine={config.moe_fused_combine}",
                    flush=True,
                )
            if config.moe_capacity_factor > 0:
                print(
                    f"MoE capacity padding: factor {config.moe_capacity_factor:.3f} | "
                    f"alignment {config.moe_capacity_alignment}",
                    flush=True,
                )
            print(f"Low precision mode: {config.low_precision_mode}")
            if config.low_precision_mode == "fp8":
                print(f"FP8 precision map: experts={config.fp8_expert_precision}", flush=True)
            if config.low_precision_mode == "nvfp4":
                print(
                    "NVFP4 precision map: "
                    f"embeddings={'bf16' if config.nvfp4_keep_embeddings_bf16 else 'nvfp4'}, "
                    f"qkv={config.nvfp4_surface_precision('qkv')}, "
                    f"latent_moe_projections={config.nvfp4_surface_precision('latent_moe_projection')}, "
                    f"lm_head={config.nvfp4_surface_precision('lm_head')}, "
                    f"final_expert_layers={config.nvfp4_final_expert_layers}x{config.nvfp4_final_expert_precision}",
                    flush=True,
                )
            print(f"MoR enabled: {config.mor_enabled} | runtime: {config.mor_runtime_mode}")
            print(f"Estimated params: {config.estimate_params():,}")
            print("Metis compute audit:", flush=True)
            for key in (
                "ffn_type",
                "d_model",
                "latent_dim",
                "num_experts",
                "expert_parallel_size",
                "local_experts_per_rank",
                "top_k",
                "routing_units_per_token",
                "expert_hidden",
                "moe_activation",
                "moe_memory_efficient_permutation",
                "moe_token_dispatcher_type",
                "moe_flex_dispatcher_backend",
                "moe_overlap_expert_parallel_comm",
                "moe_delay_wgrad_compute",
                "moe_router_fusion",
                "moe_permute_fusion",
                "expert_param_apps_per_assignment",
                "routed_expert_params_per_rank",
                "routed_expert_param_apps_per_layer",
                "shared_expert_param_apps_per_layer",
                "latent_projection_apps_per_layer",
                "router_projection_and_match_apps_per_layer",
                "attention_apps_per_layer",
                "rough_total_param_apps_per_token",
                "estimated_params_per_expert_parallel_rank",
                "estimated_train_flops_per_token",
            ):
                if key in param_audit:
                    value = param_audit[key]
                    rendered = f"{value:,}" if isinstance(value, int) else str(value)
                    print(f"  {key}: {rendered}", flush=True)
            print(f"Attention backend: {config.attention_backend}")
            print(f"Attention mask mode: {config.attention_mask_mode}")
            print(f"CUDA batch prefetch: {args.prefetch_batches if prefetcher is not None else 0}")
            print(f"Precision path: {precision_label}")
            startup_param_count = config.estimate_params()
            active_param_count = config.estimate_active_params(config.mor_target_avg_depth if config.uses_mor else 1.0)
            print(
                "450 TFLOP/s token targets: "
                f"total-param {450e12 / (6.0 * max(startup_param_count, 1)):,.0f} tok/s | "
                f"active-param {450e12 / (6.0 * max(active_param_count, 1)):,.0f} tok/s",
                flush=True,
            )
            if args.init_checkpoint:
                init_display = str(resolved_init_path) if "resolved_init_path" in locals() else args.init_checkpoint
                print(f"Initialized weights from checkpoint: {init_display}")
            print(
                f"MoR depth budget: 1..{config.mor_max_depth} recursions "
                f"(max effective depth {config.effective_layer_count}, target {config.target_effective_layer_count:.1f})"
            )
            if config.uses_dynamic_token_mor:
                mor_stage = stage_manifest.get("mor", {})
                if isinstance(mor_stage, dict):
                    print(
                        "Dynamic MoR schedule: "
                        f"target {mor_stage.get('target_avg_depth_start', config.mor_target_avg_depth)}"
                        f"->{mor_stage.get('target_avg_depth_end', mor_stage.get('target_avg_depth', config.mor_target_avg_depth))} "
                        f"over {int(mor_stage.get('target_avg_depth_warmup_tokens', 0) or 0) / 1e9:.3f}B tokens | "
                        f"router_aux {config.mor_router_aux_loss_coef:.4f} | "
                        f"entropy {config.mor_router_entropy_coef:.4f} | "
                        f"z {config.mor_router_z_loss_coef:.6f}",
                        flush=True,
                    )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process(rank):
            train_config = vars(args) | {
                "world_size": world_size,
                "expert_parallel": expert_parallel,
                "expert_parallel_size": int(config.moe_expert_parallel_size),
            }
            (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2) + "\n", encoding="utf-8")
        barrier(distributed)

        optimizer, optimizer_summary = build_optimizer(model, args, optimizer_manifest)
        if os.environ.get("METIS_PREINIT_ADAMW_STATE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            initialized_states = preinitialize_adamw_state(optimizer)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if is_main_process(rank):
                print(f"Preinitialized AdamW state tensors: {initialized_states}", flush=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_lr(
                step,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
            ),
        )
        if is_main_process(rank):
            if optimizer_summary is None:
                print("Optimizer: AdamW", flush=True)
            else:
                print(
                    "Optimizer: Muon-AdamW hybrid | "
                    f"muon {optimizer_summary.muon_params:,} params | "
                    f"adamw {optimizer_summary.adamw_params:,} params | "
                    f"adamw_impl={optimizer_summary.adamw_impl} | "
                    f"routed_experts_muon={optimizer_summary.routed_experts_muon}",
                    flush=True,
                )
                for group in optimizer_summary.groups:
                    sample = ", ".join(group.sample_names[:3])
                    print(
                        f"  {group.name}: {group.optimizer} | "
                        f"{group.tensor_count} tensors | {group.param_count:,} params | {sample}",
                        flush=True,
                    )

        latest_checkpoint_path = out_dir / "latest.pt"
        start_step = 0
        best_val_loss = math.inf
        previous_elapsed = 0.0
        checkpoint_tokens_seen: int | None = None

        resume_checkpoint: dict | None = None
        resume_checkpoint_path = latest_checkpoint_path
        if args.resume:
            resume_checkpoint, resume_checkpoint_path = load_rank_checkpoint(
                latest_checkpoint_path,
                rank=rank,
                expert_parallel=expert_parallel,
            )

        if args.resume and resume_checkpoint is not None:
            checkpoint = resume_checkpoint
            model_state, conversions = filter_state_dict_for_model(model, checkpoint["model_state_dict"])
            model.load_state_dict(
                model_state,
                strict=False,
            )
            if is_main_process(rank) and conversions:
                print(f"Converted legacy checkpoint modules for fused layout: {len(conversions)}", flush=True)
            if args.fresh_optimizer_on_resume:
                if is_main_process(rank):
                    print(
                        "Skipping optimizer state restore by request; continuing with fresh optimizer state.",
                        flush=True,
                    )
            else:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    optimizer_to_device(optimizer, device)
                except (KeyError, RuntimeError, ValueError) as exc:
                    if is_main_process(rank):
                        print(
                            "Warning: could not restore optimizer state after model layout change; "
                            f"continuing with a fresh optimizer state. Reason: {exc}",
                            flush=True,
                        )
            if args.fresh_scheduler_on_resume:
                if is_main_process(rank):
                    print(
                        "Skipping scheduler state restore by request; continuing with fresh scheduler state.",
                        flush=True,
                    )
            else:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except (KeyError, RuntimeError, ValueError) as exc:
                    if is_main_process(rank):
                        print(
                            "Warning: could not restore scheduler state; continuing with current scheduler state. "
                            f"Reason: {exc}",
                            flush=True,
                        )
            start_step = int(checkpoint.get("step", 0))
            checkpoint_tokens_seen = infer_checkpoint_tokens_seen(
                checkpoint,
                fallback_block_size=config.block_size,
                fallback_world_size=world_size,
                fallback_batch_size=args.batch_size,
                fallback_grad_accum_steps=args.grad_accum_steps,
            )
            best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
            previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
            model.to(device=device, dtype=model_dtype)
            if is_main_process(rank):
                print(
                    f"Resuming from {resume_checkpoint_path} at step {start_step} "
                    f"({checkpoint_tokens_seen / 1e9:.3f}B tokens seen)",
                    flush=True,
                )
            if expert_parallel:
                broadcast_replicated_parameters(model, device=device, distributed=distributed)
        barrier(distributed)

        tokens_per_step = args.batch_size * args.grad_accum_steps * config.block_size * world_size
        total_tokens_seen = checkpoint_tokens_seen if checkpoint_tokens_seen is not None else start_step * tokens_per_step
        target_train_tokens = int(stage_manifest.get("target_train_tokens", 0))
        if checkpoint_tokens_seen is not None and target_train_tokens > total_tokens_seen:
            token_adjusted_max_steps = start_step + math.ceil((target_train_tokens - total_tokens_seen) / tokens_per_step)
            if token_adjusted_max_steps > args.max_steps:
                if is_main_process(rank):
                    print(
                        f"Adjusting max_steps from {args.max_steps} to {token_adjusted_max_steps} "
                        "to preserve target tokens after resume batch-shape change.",
                        flush=True,
                    )
                args.max_steps = token_adjusted_max_steps

        if start_step >= args.max_steps:
            if is_main_process(rank):
                print(f"Checkpoint already reached step {start_step}, which is >= max_steps={args.max_steps}.")
            return

        wall_start_time = time.time()
        interval_start_time = time.perf_counter()
        interval_loss = 0.0
        interval_route_aux = 0.0
        interval_moe_aux = 0.0
        interval_mean_depth = 0.0
        last_active_ratios: list[float] = []
        last_active_ratios_tensor: torch.Tensor | None = None
        interval_updates = 0
        interval_tokens = 0
        async_metrics = _env_flag("METIS_ASYNC_METRICS", "0")
        interval_loss_tensor: torch.Tensor | None = None
        interval_route_aux_tensor: torch.Tensor | None = None
        interval_moe_aux_tensor: torch.Tensor | None = None
        interval_mean_depth_tensor: torch.Tensor | None = None
        gate_schedule = build_gate_schedule(
            stage_manifest=stage_manifest,
            block_size=config.block_size,
            local_batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            world_size=world_size,
            start_step=start_step,
            tokens_already_seen=total_tokens_seen,
        )
        if is_main_process(rank):
            if gate_schedule:
                gate_lines = ", ".join(f"{label}@step{step}" for step, label in sorted(gate_schedule.items()))
                print(f"Pretrain gates: {gate_lines}", flush=True)
            else:
                print("Pretrain gates: all configured gates already passed.", flush=True)
        param_count = config.estimate_params()
        counter_model = train_model
        reset_perf_counters(counter_model)

        for step in range(start_step + 1, args.max_steps + 1):
            update_mor_schedule(config, stage_manifest, tokens_seen=total_tokens_seen)
            with nvtx_range("optimizer_zero_grad", device):
                optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_route_aux = 0.0
            running_moe_aux = 0.0
            running_mean_depth = 0.0
            running_loss_tensor: torch.Tensor | None = None
            running_route_aux_tensor: torch.Tensor | None = None
            running_moe_aux_tensor: torch.Tensor | None = None
            running_mean_depth_tensor: torch.Tensor | None = None
            running_active_ratios: torch.Tensor | None = None
            running_active_ratio_updates = 0
            step_token_count = 0

            for micro_step in range(args.grad_accum_steps):
                with nvtx_range("batch_fetch", device):
                    if prefetcher is not None:
                        xb, yb = prefetcher.next()
                    else:
                        xb, yb = get_batch(
                            train_data,
                            batch_size=args.batch_size,
                            block_size=config.block_size,
                            device=device,
                        )
                sync_context = (
                    train_model.no_sync()
                    if distributed and not expert_parallel and micro_step < args.grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    maybe_mark_compile_step(compile_enabled)
                    with nvtx_range("forward", device):
                        forward_debug_state["step"] = step
                        forward_debug_state["micro_step"] = micro_step
                        output = train_model(
                            xb,
                            labels=yb,
                            is_first_microbatch=(micro_step == 0),
                            return_logits=bool(
                                (args.debug_nonfinite or args.retain_standard_ce_logits)
                                and config.lm_loss_impl == "standard"
                            ),
                        )
                        loss = output.loss
                        if args.debug_nonfinite:
                            if output.hidden_states:
                                _check_tensor_finite(
                                    "forward:final_hidden",
                                    output.hidden_states[-1],
                                    step=step,
                                    micro_step=micro_step,
                                )
                            if output.logits is not None:
                                _check_tensor_finite("forward:logits", output.logits, step=step, micro_step=micro_step)
                            if output.lm_loss is not None:
                                _check_tensor_finite("loss_component:lm_loss", output.lm_loss, step=step, micro_step=micro_step)
                            if output.moe_aux_loss is not None:
                                _check_tensor_finite(
                                    "loss_component:moe_aux_loss",
                                    output.moe_aux_loss,
                                    step=step,
                                    micro_step=micro_step,
                                )
                            if output.route_aux_loss is not None:
                                _check_tensor_finite(
                                    "loss_component:route_aux_loss",
                                    output.route_aux_loss,
                                    step=step,
                                    micro_step=micro_step,
                                )
                            _check_tensor_finite("loss", loss, step=step, micro_step=micro_step)
                        if not torch.isfinite(loss.detach()):
                            raise FloatingPointError(
                                f"Non-finite loss at step {step}, micro_step {micro_step}: {loss.detach().float().item()}"
                            )
                    with nvtx_range("backward", device):
                        (loss / args.grad_accum_steps).backward()
                    if args.debug_nonfinite:
                        _check_model_finite(model, kind="grad_after_backward", step=step, micro_step=micro_step)
                    if args.debug_absmax_top_k > 0:
                        _log_model_absmax(
                            model,
                            kind=f"grad_after_backward_micro_{micro_step}",
                            step=step,
                            top_k=args.debug_absmax_top_k,
                        )
                tracked_loss = output.lm_loss if output.lm_loss is not None else output.loss
                if async_metrics:
                    tracked_loss_detached = tracked_loss.detach()
                    running_loss_tensor = (
                        tracked_loss_detached
                        if running_loss_tensor is None
                        else running_loss_tensor + tracked_loss_detached
                    )
                else:
                    running_loss += float(tracked_loss.item())
                if output.route_aux_loss is not None:
                    if async_metrics:
                        route_aux_detached = output.route_aux_loss.detach()
                        running_route_aux_tensor = (
                            route_aux_detached
                            if running_route_aux_tensor is None
                            else running_route_aux_tensor + route_aux_detached
                        )
                    else:
                        running_route_aux += float(output.route_aux_loss.item())
                if output.moe_aux_loss is not None:
                    if async_metrics:
                        moe_aux_detached = output.moe_aux_loss.detach()
                        running_moe_aux_tensor = (
                            moe_aux_detached
                            if running_moe_aux_tensor is None
                            else running_moe_aux_tensor + moe_aux_detached
                        )
                    else:
                        running_moe_aux += float(output.moe_aux_loss.item())
                if output.mean_depth is not None:
                    if async_metrics:
                        mean_depth_detached = output.mean_depth.detach()
                        running_mean_depth_tensor = (
                            mean_depth_detached
                            if running_mean_depth_tensor is None
                            else running_mean_depth_tensor + mean_depth_detached
                        )
                    else:
                        running_mean_depth += float(output.mean_depth.item())
                if output.active_token_ratios is not None:
                    active_ratios = output.active_token_ratios.detach().float()
                    if running_active_ratios is None:
                        running_active_ratios = torch.zeros_like(active_ratios)
                    running_active_ratios = running_active_ratios + active_ratios
                    running_active_ratio_updates += 1
                step_token_count += xb.numel()

            if expert_parallel:
                all_reduce_replicated_gradients(
                    model,
                    device=device,
                    distributed=distributed,
                    world_size=world_size,
                )
            if args.grad_clip > 0:
                with nvtx_range("grad_clip", device):
                    if distributed and not expert_parallel:
                        torch.nn.utils.clip_grad_norm_(train_model.module.parameters(), args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if args.debug_nonfinite:
                    _check_model_finite(model, kind="grad_after_clip", step=step)
                if args.debug_absmax_top_k > 0:
                    _log_model_absmax(model, kind="grad_after_clip", step=step, top_k=args.debug_absmax_top_k)
            with nvtx_range("optimizer_step", device):
                optimizer.step()
                scheduler.step()
            if args.debug_nonfinite:
                _check_model_finite(model, kind="param_after_step", step=step)
                _check_optimizer_state_finite(optimizer, param_name_map, step=step)
            if args.debug_absmax_top_k > 0:
                _log_model_absmax(model, kind="param_after_step", step=step, top_k=args.debug_absmax_top_k)

            if async_metrics:
                scale = 1.0 / float(args.grad_accum_steps)
                mean_loss_tensor = (
                    running_loss_tensor * scale
                    if running_loss_tensor is not None
                    else torch.zeros((), device=device, dtype=torch.float32)
                )
                interval_loss_tensor = (
                    mean_loss_tensor
                    if interval_loss_tensor is None
                    else interval_loss_tensor + mean_loss_tensor
                )
                if running_route_aux_tensor is not None:
                    mean_route_aux_tensor = running_route_aux_tensor * scale
                    interval_route_aux_tensor = (
                        mean_route_aux_tensor
                        if interval_route_aux_tensor is None
                        else interval_route_aux_tensor + mean_route_aux_tensor
                    )
                if running_moe_aux_tensor is not None:
                    mean_moe_aux_tensor = running_moe_aux_tensor * scale
                    interval_moe_aux_tensor = (
                        mean_moe_aux_tensor
                        if interval_moe_aux_tensor is None
                        else interval_moe_aux_tensor + mean_moe_aux_tensor
                    )
                if running_mean_depth_tensor is not None:
                    mean_depth_tensor = running_mean_depth_tensor * scale
                    interval_mean_depth_tensor = (
                        mean_depth_tensor
                        if interval_mean_depth_tensor is None
                        else interval_mean_depth_tensor + mean_depth_tensor
                    )
            else:
                mean_loss = running_loss / args.grad_accum_steps
                mean_route_aux = running_route_aux / args.grad_accum_steps
                mean_moe_aux = running_moe_aux / args.grad_accum_steps
                mean_depth = running_mean_depth / args.grad_accum_steps
                interval_loss += mean_loss
                interval_route_aux += mean_route_aux
                interval_moe_aux += mean_moe_aux
                interval_mean_depth += mean_depth
            if running_active_ratios is not None and running_active_ratio_updates > 0:
                active_ratio_snapshot = (running_active_ratios / float(running_active_ratio_updates)).detach()
                if async_metrics:
                    last_active_ratios_tensor = active_ratio_snapshot
                else:
                    last_active_ratios = active_ratio_snapshot.cpu().tolist()
            interval_updates += 1
            interval_tokens += step_token_count
            total_tokens_seen += step_token_count * world_size

            if args.log_interval > 0 and step % args.log_interval == 0:
                interval_elapsed = max(time.perf_counter() - interval_start_time, 1e-6)
                if async_metrics:
                    interval_loss = float(
                        (interval_loss_tensor if interval_loss_tensor is not None else torch.zeros((), device=device)).item()
                    )
                    interval_route_aux = float(
                        (
                            interval_route_aux_tensor
                            if interval_route_aux_tensor is not None
                            else torch.zeros((), device=device)
                        ).item()
                    )
                    interval_moe_aux = float(
                        (
                            interval_moe_aux_tensor
                            if interval_moe_aux_tensor is not None
                            else torch.zeros((), device=device)
                        ).item()
                    )
                    interval_mean_depth = float(
                        (
                            interval_mean_depth_tensor
                            if interval_mean_depth_tensor is not None
                            else torch.zeros((), device=device)
                        ).item()
                    )
                total_loss, total_updates, total_tokens = aggregate_interval_stats(
                    loss_total=interval_loss,
                    updates=interval_updates,
                    tokens=interval_tokens,
                    device=device,
                    distributed=distributed,
                )
                if is_main_process(rank):
                    if async_metrics and last_active_ratios_tensor is not None:
                        last_active_ratios = last_active_ratios_tensor.cpu().tolist()
                    global_updates = max(total_updates // world_size, 1) if distributed else max(total_updates, 1)
                    avg_step_seconds = interval_elapsed / global_updates
                    tokens_per_second = total_tokens / interval_elapsed
                    est_tflops = estimate_tflops(tokens_per_second, param_count)
                    active_param_count = estimate_active_param_count(
                        config,
                        interval_mean_depth / max(interval_updates, 1),
                    )
                    est_active_tflops = estimate_tflops(tokens_per_second, active_param_count)
                    counters = get_perf_counters(counter_model)
                    counter_line = (
                        f"fa3 {counters.get('fa3_calls', 0)} | "
                        f"sdpa {counters.get('sdpa_calls', 0)} | "
                        f"mask {counters.get('attention_mask_passed_calls', 0)} | "
                        f"mor_router {counters.get('router_calls', 0)} | "
                        f"moe_router {counters.get('moe_router_calls', 0)} | "
                        f"moe_grouped {counters.get('moe_grouped_expert_dispatches', 0)} | "
                        f"assign {counters.get('moe_grouped_assignments', 0)} | "
                        f"ep_a2a {counters.get('moe_ep_all_to_all_calls', 0)} | "
                        f"ep_send {counters.get('moe_ep_send_assignments', 0)} | "
                        f"ep_recv {counters.get('moe_ep_recv_assignments', 0)} | "
                        f"static_cap {counters.get('moe_static_capacity_dispatches', 0)} | "
                        f"fused_combine {counters.get('moe_fused_combine_calls', 0)} | "
                        f"mem_perm {counters.get('moe_memory_efficient_permute_calls', 0)} | "
                        f"cap_pad {counters.get('moe_capacity_padded_dispatches', 0)} | "
                        f"cap_over {counters.get('moe_capacity_overflow_fallbacks', 0)} | "
                        f"pad_tok {counters.get('moe_capacity_padded_tokens', 0)} | "
                        f"pack {counters.get('pack_active_tokens_calls', 0)} | "
                        f"scatter {counters.get('scatter_active_tokens_calls', 0)}"
                    )
                    load_reports = max(counters.get("moe_expert_load_reports", 0), 1)
                    expert_line = ""
                    if counters.get("moe_expert_load_reports", 0) > 0:
                        expert_line = (
                            f" | expert_empty {counters.get('moe_expert_empty_count', 0)} | "
                            "expert_rows "
                            f"{counters.get('moe_expert_min_rows_sum', 0) / load_reports:.0f}/"
                            f"{counters.get('moe_expert_p95_rows_sum', 0) / load_reports:.0f}/"
                            f"{counters.get('moe_expert_max_rows_sum', 0) / load_reports:.0f}"
                        )
                    active_line = ""
                    if last_active_ratios:
                        active_line = " | active " + ",".join(f"{ratio:.2f}" for ratio in last_active_ratios)
                    print(
                        f"step {step:6d} | train {total_loss / max(total_updates, 1):.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.6e} | "
                        f"depth {interval_mean_depth / max(interval_updates, 1):.2f}/{config.mor_target_avg_depth:.2f} | "
                        f"route_aux {interval_route_aux / max(interval_updates, 1):.4f} | "
                        f"moe_aux {interval_moe_aux / max(interval_updates, 1):.4f} | "
                        f"tok/s {tokens_per_second:,.0f} | "
                        f"step_s {avg_step_seconds:.2f} | "
                        f"tok_seen {total_tokens_seen / 1e9:.3f}B | "
                        f"est_tflops {est_tflops:.2f} | "
                        f"est_active_tflops {est_active_tflops:.2f} | "
                        f"{counter_line}"
                        f"{expert_line}"
                        f"{active_line}",
                        flush=True,
                    )
                    reset_perf_counters(counter_model)
                interval_start_time = time.perf_counter()
                interval_loss = 0.0
                interval_route_aux = 0.0
                interval_moe_aux = 0.0
                interval_mean_depth = 0.0
                last_active_ratios = []
                last_active_ratios_tensor = None
                interval_updates = 0
                interval_tokens = 0
                interval_loss_tensor = None
                interval_route_aux_tensor = None
                interval_moe_aux_tensor = None
                interval_mean_depth_tensor = None

            should_eval = (
                (step == args.max_steps and not args.skip_final_eval)
                or (args.eval_interval > 0 and step % args.eval_interval == 0)
            )
            should_checkpoint = checkpoint_requested or (step == args.max_steps and not args.skip_final_checkpoint) or (
                args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0
            )
            if distributed:
                checkpoint_flag = torch.tensor([1 if should_checkpoint else 0], device=device, dtype=torch.int32)
                dist.all_reduce(checkpoint_flag, op=dist.ReduceOp.MAX)
                should_checkpoint = bool(int(checkpoint_flag.item()))

            if should_eval:
                if expert_parallel:
                    losses = estimate_loss(
                        model,
                        train_data=train_data,
                        val_data=val_data,
                        eval_iters=args.eval_iters,
                        batch_size=args.batch_size,
                        block_size=config.block_size,
                        device=device,
                        compiled=compile_enabled,
                    )
                    metrics = torch.tensor(
                        [float(losses["train"]), float(losses["val"])],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    metrics.div_(float(world_size))
                    train_loss, val_loss = float(metrics[0].item()), float(metrics[1].item())
                elif is_main_process(rank):
                    losses = estimate_loss(
                        model,
                        train_data=train_data,
                        val_data=val_data,
                        eval_iters=args.eval_iters,
                        batch_size=args.batch_size,
                        block_size=config.block_size,
                        device=device,
                        compiled=compile_enabled,
                    )
                    train_loss, val_loss = losses["train"], losses["val"]
                else:
                    train_loss, val_loss = 0.0, 0.0
                if not expert_parallel:
                    train_loss, val_loss = broadcast_pair(
                        train_loss,
                        val_loss,
                        device=device,
                        distributed=distributed,
                    )
                if is_main_process(rank):
                    print(
                        f"step {step:6d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                        f"ppl {math.exp(val_loss):.2f}",
                        flush=True,
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if is_main_process(rank) or expert_parallel:
                        save_checkpoint(
                            out_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            total_tokens_seen=total_tokens_seen,
                            best_val_loss=best_val_loss,
                            train_args=vars(args)
                            | {
                                "world_size": world_size,
                                "expert_parallel": expert_parallel,
                                "expert_parallel_size": int(config.moe_expert_parallel_size),
                            },
                            elapsed_seconds=previous_elapsed + (time.time() - wall_start_time),
                            rank=rank,
                            expert_parallel=expert_parallel,
                        )
                barrier(distributed)

            if should_checkpoint:
                if is_main_process(rank) or expert_parallel:
                    save_checkpoint(
                        latest_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        total_tokens_seen=total_tokens_seen,
                        best_val_loss=best_val_loss,
                        train_args=vars(args)
                        | {
                            "world_size": world_size,
                            "expert_parallel": expert_parallel,
                            "expert_parallel_size": int(config.moe_expert_parallel_size),
                        },
                        elapsed_seconds=previous_elapsed + (time.time() - wall_start_time),
                        rank=rank,
                        expert_parallel=expert_parallel,
                    )
                    checkpoint_requested = False
                barrier(distributed)

            gate_label = gate_schedule.get(step)
            if gate_label is not None:
                if is_main_process(rank) or expert_parallel:
                    save_checkpoint(
                        out_dir / f"{gate_label}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        total_tokens_seen=total_tokens_seen,
                        best_val_loss=best_val_loss,
                        train_args=vars(args)
                        | {
                            "world_size": world_size,
                            "expert_parallel": expert_parallel,
                            "expert_parallel_size": int(config.moe_expert_parallel_size),
                        },
                        elapsed_seconds=previous_elapsed + (time.time() - wall_start_time),
                        rank=rank,
                        expert_parallel=expert_parallel,
                    )
                barrier(distributed)
    finally:
        if prefetcher is not None:
            prefetcher.close()
        cleanup_runtime(distributed)


if __name__ == "__main__":
    main()
