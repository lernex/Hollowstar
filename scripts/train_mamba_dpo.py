from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer

from metis_mamba import MetisMambaConfig, build_model, cosine_lr, parse_torch_dtype
from metis_mamba.checkpoints import atomic_torch_save
from metis_mamba.fp8 import (
    build_fp8_recipe,
    build_nvfp4_recipe,
    transformer_engine_is_available,
    transformer_engine_runtime_supports_fp8_block_scaling,
    transformer_engine_runtime_supports_nvfp4,
    transformer_engine_supports_fp8_block_scaling,
    transformer_engine_supports_mxfp8,
    transformer_engine_supports_nvfp4,
)
from metis_mamba.optim import build_optimizer_from_args


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
            raise RuntimeError("Distributed Metis DPO training requires CUDA devices.")
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
    accuracy_total: float,
    updates: int,
    micro_batches: int,
    tokens: int,
    device: torch.device,
    distributed: bool,
) -> tuple[float, float, int, int, int]:
    if not distributed:
        return loss_total, accuracy_total, updates, micro_batches, tokens
    stats = torch.tensor(
        [loss_total, accuracy_total, float(updates), float(micro_batches), float(tokens)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), float(stats[1].item()), int(stats[2].item()), int(stats[3].item()), int(stats[4].item())


def broadcast_pair(first: float, second: float, *, device: torch.device, distributed: bool) -> tuple[float, float]:
    if not distributed:
        return first, second
    payload = torch.tensor([first, second], device=device, dtype=torch.float64)
    dist.broadcast(payload, src=0)
    return float(payload[0].item()), float(payload[1].item())


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def filter_state_dict_for_model(model, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    allowed = set(model.state_dict().keys())
    return {name: tensor for name, tensor in state_dict.items() if name in allowed}


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


def resolve_precision(args: argparse.Namespace, device: torch.device) -> tuple[torch.dtype, bool, object | None, str]:
    if bool(args.nvfp4) and bool(args.fp8):
        raise RuntimeError("Choose one low-precision recipe: --nvfp4 or --fp8, not both.")
    if args.nvfp4:
        if device.type != "cuda":
            raise RuntimeError("NVFP4 DPO training requires CUDA.")
        if not transformer_engine_is_available() or not transformer_engine_supports_nvfp4():
            raise RuntimeError("NVFP4 requires a Blackwell-capable Transformer Engine build with NVFP4BlockScaling.")
        if not transformer_engine_runtime_supports_nvfp4(
            disable_rht=args.nvfp4_disable_rht,
            disable_2d_quantization=args.nvfp4_disable_2d_quantization,
            disable_stochastic_rounding=args.nvfp4_disable_stochastic_rounding,
        ):
            raise RuntimeError(
                "This GPU/runtime rejects the requested NVFP4 recipe. "
                "On RTX PRO 6000 / SM120, use --nvfp4-disable-rht "
                "--nvfp4-disable-2d-quantization --nvfp4-disable-stochastic-rounding."
            )
        if args.dtype not in {"bf16", "bfloat16"}:
            raise RuntimeError("NVFP4 training keeps master weights in BF16. Use --dtype bf16 with --nvfp4.")
        recipe = build_nvfp4_recipe(
            disable_rht=args.nvfp4_disable_rht,
            disable_2d_quantization=args.nvfp4_disable_2d_quantization,
            disable_stochastic_rounding=args.nvfp4_disable_stochastic_rounding,
        )
        return torch.bfloat16, True, recipe, "NVFP4 compute with BF16 master weights"
    fp8_enabled = bool(args.fp8)
    if fp8_enabled:
        if device.type != "cuda":
            raise RuntimeError("FP8 DPO training requires CUDA.")
        if not transformer_engine_is_available():
            raise RuntimeError(
                "FP8 was requested, but Transformer Engine is not installed. "
                "Install transformer_engine[pytorch] in the GPU training environment."
            )
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
        )
        return torch.bfloat16, True, recipe, "FP8 compute with BF16 master weights"
    return parse_torch_dtype(args.dtype), False, None, args.dtype.upper()


def build_preference_sequence(prompt: str, response: str) -> tuple[str, str]:
    prompt = prompt.strip()
    if prompt.endswith("Assistant:"):
        return prompt, response.strip()
    return f"{prompt}\nAssistant:", response.strip()


def tokenize_completion(
    *,
    prompt: str,
    response: str,
    tokenizer: Tokenizer,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text, answer_text = build_preference_sequence(prompt, response)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    answer_ids = tokenizer.encode(" " + answer_text, add_special_tokens=False).ids

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    input_ids: list[int] = []
    labels: list[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    input_ids.extend(prompt_ids)
    labels.extend([-100] * len(prompt_ids))
    input_ids.extend(answer_ids)
    labels.extend(answer_ids)

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id)

    input_ids = input_ids[-max_length:]
    labels = labels[-max_length:]
    if all(value == -100 for value in labels):
        return {"input_ids": [], "labels": []}
    return {"input_ids": input_ids, "labels": labels}


def tokenize_pair_example(example: dict, *, tokenizer: Tokenizer, max_length: int) -> dict[str, Any]:
    chosen = tokenize_completion(
        prompt=str(example["prompt"]),
        response=str(example["chosen"]),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    rejected = tokenize_completion(
        prompt=str(example["prompt"]),
        response=str(example["rejected"]),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if not chosen["input_ids"] or not rejected["input_ids"]:
        return {
            "chosen_input_ids": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_labels": [],
            "pair_weight": 1.0,
        }
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_labels": rejected["labels"],
        "pair_weight": float(example.get("pair_weight", 1.0)),
    }


def collate_sequences(
    input_key: str,
    label_key: str,
    batch: list[dict[str, Any]],
    pad_token_id: int,
    *,
    pad_to_multiple: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item[input_key]) for item in batch)
    if pad_to_multiple and (max_len % pad_to_multiple) != 0:
        max_len = max_len + (pad_to_multiple - (max_len % pad_to_multiple))

    input_ids = []
    labels = []
    attention_masks = []
    for item in batch:
        pad_len = max_len - len(item[input_key])
        input_ids.append(item[input_key] + ([pad_token_id] * pad_len))
        labels.append(item[label_key] + ([-100] * pad_len))
        attention_masks.append(([1] * len(item[input_key])) + ([0] * pad_len))
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.bool),
    )


def collate_pair_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    *,
    pad_to_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_ids, chosen_labels, chosen_mask = collate_sequences(
        "chosen_input_ids",
        "chosen_labels",
        batch,
        pad_token_id,
        pad_to_multiple=pad_to_multiple,
    )
    rejected_ids, rejected_labels, rejected_mask = collate_sequences(
        "rejected_input_ids",
        "rejected_labels",
        batch,
        pad_token_id,
        pad_to_multiple=pad_to_multiple,
    )
    weights = torch.tensor([float(item["pair_weight"]) for item in batch], dtype=torch.float32)
    return chosen_ids, chosen_labels, chosen_mask, rejected_ids, rejected_labels, rejected_mask, weights


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    score_mode: str,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].contiguous()
    loss_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * loss_mask
    if score_mode == "sum":
        return token_log_probs.sum(dim=-1)
    lengths = loss_mask.sum(dim=-1).clamp_min(1)
    return token_log_probs.sum(dim=-1) / lengths


@torch.no_grad()
def evaluate(
    policy_model,
    reference_model,
    loader: DataLoader,
    device: torch.device,
    *,
    compiled: bool,
    beta: float,
    score_mode: str,
) -> tuple[float, float]:
    policy_model.eval()
    reference_model.eval()
    losses = []
    accuracies = []
    for batch in loader:
        chosen_ids, chosen_labels, chosen_mask, rejected_ids, rejected_labels, rejected_mask, weights = batch
        non_blocking = device.type == "cuda"
        chosen_ids = chosen_ids.to(device, non_blocking=non_blocking)
        chosen_labels = chosen_labels.to(device, non_blocking=non_blocking)
        chosen_mask = chosen_mask.to(device, non_blocking=non_blocking)
        rejected_ids = rejected_ids.to(device, non_blocking=non_blocking)
        rejected_labels = rejected_labels.to(device, non_blocking=non_blocking)
        rejected_mask = rejected_mask.to(device, non_blocking=non_blocking)
        weights = weights.to(device, non_blocking=non_blocking)

        maybe_mark_compile_step(compiled)
        policy_chosen = policy_model(chosen_ids, attention_mask=chosen_mask, is_first_microbatch=True)
        policy_rejected = policy_model(rejected_ids, attention_mask=rejected_mask, is_first_microbatch=True)
        reference_chosen = reference_model(chosen_ids, attention_mask=chosen_mask, is_first_microbatch=True)
        reference_rejected = reference_model(rejected_ids, attention_mask=rejected_mask, is_first_microbatch=True)

        policy_chosen_logp = sequence_log_probs(policy_chosen.logits, chosen_labels, score_mode=score_mode)
        policy_rejected_logp = sequence_log_probs(policy_rejected.logits, rejected_labels, score_mode=score_mode)
        ref_chosen_logp = sequence_log_probs(reference_chosen.logits, chosen_labels, score_mode=score_mode)
        ref_rejected_logp = sequence_log_probs(reference_rejected.logits, rejected_labels, score_mode=score_mode)

        preference_margin = (policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
        loss = (-F.logsigmoid(beta * preference_margin) * weights).mean()
        losses.append(float(loss.item()))
        accuracies.append(float((preference_margin > 0).float().mean().item()))
    policy_model.train()
    return sum(losses) / max(len(losses), 1), sum(accuracies) / max(len(accuracies), 1)


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_args: dict,
) -> None:
    size_bytes = atomic_torch_save(
        path,
        {
            "model_family": getattr(model, "model_family", model.config.model_type),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
        },
    )
    print(
        f"saved checkpoint {path.name} | step {global_step:5d} | epoch {epoch:3d} | size {size_bytes / (1024 ** 3):.2f} GiB",
        flush=True,
    )


def estimate_tflops(tokens_per_second: float, param_count: int) -> float:
    return (6.0 * float(param_count) * tokens_per_second) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Metis-1.5 DPO policy on pairwise preference data.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--sequence-score-mode", choices=["mean", "sum"], default="mean")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-format", choices=["HYBRID", "E4M3"], default="HYBRID")
    parser.add_argument("--fp8-margin", type=int, default=0)
    parser.add_argument("--fp8-amax-history-len", type=int, default=16)
    parser.add_argument("--fp8-amax-compute-algo", default="max")
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--nvfp4-disable-rht", action="store_true")
    parser.add_argument("--nvfp4-disable-2d-quantization", action="store_true")
    parser.add_argument("--nvfp4-disable-stochastic-rounding", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default="adamw")
    parser.add_argument("--muon-beta", type=float, default=None)
    parser.add_argument("--muon-ns-steps", type=int, default=None)
    parser.add_argument("--muon-lr-scale", type=float, default=None)
    parser.add_argument("--muon-include-routed-experts", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device, distributed, rank, local_rank, world_size = setup_runtime(args.device)
    try:
        seed = args.seed + rank
        random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model_dtype, fp8_enabled, fp8_recipe, precision_label = resolve_precision(args, device)
        maybe_enable_cuda_speedups(
            device,
            matmul_precision=args.matmul_precision,
            tf32=args.tf32,
        )

        policy_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
        reference_checkpoint = torch.load(args.reference_checkpoint, map_location="cpu")
        config = MetisMambaConfig.from_dict(policy_checkpoint["model_config"])
        if args.nvfp4:
            config.low_precision_mode = "nvfp4"
            if config.nvfp4_uses_mxfp8 and not transformer_engine_supports_mxfp8():
                raise RuntimeError(
                    "This checkpoint requests MXFP8 stability linears under NVFP4, "
                    "but this Transformer Engine build does not expose MXFP8BlockScaling."
                )
            if config.nvfp4_uses_fp8_block and not transformer_engine_supports_fp8_block_scaling():
                raise RuntimeError(
                    "This checkpoint requests Float8BlockScaling stability linears under NVFP4, "
                    "but this Transformer Engine build does not expose Float8BlockScaling."
                )
            if config.nvfp4_uses_fp8_block and not transformer_engine_runtime_supports_fp8_block_scaling():
                raise RuntimeError(
                    "This GPU/runtime does not support Float8BlockScaling stability linears. "
                    "Use the CUDA 13.2 + Transformer Engine main Blackwell image."
                )
        elif args.fp8:
            config.low_precision_mode = "fp8"
        else:
            config.low_precision_mode = "none"
        policy_model = build_model(
            config,
            use_fp8=fp8_enabled,
            fp8_recipe=fp8_recipe,
            fp8_group=dist.group.WORLD if distributed and fp8_enabled else None,
        )
        reference_model = build_model(config)
        policy_model.load_state_dict(
            filter_state_dict_for_model(policy_model, policy_checkpoint["model_state_dict"]),
            strict=False,
        )
        reference_model.load_state_dict(
            filter_state_dict_for_model(reference_model, reference_checkpoint["model_state_dict"]),
            strict=False,
        )
        policy_model.to(device=device, dtype=model_dtype)
        reference_model.to(device=device, dtype=model_dtype)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

        compile_enabled = args.compile and device.type == "cuda" and not distributed and not fp8_enabled
        if args.compile and distributed and is_main_process(rank):
            print("Distributed run detected; disabling torch.compile for stability.", flush=True)
        if args.compile and fp8_enabled and is_main_process(rank):
            print("Transformer Engine low-precision run detected; disabling torch.compile for stability.", flush=True)
        train_model = torch.compile(policy_model, mode=args.compile_mode) if compile_enabled else policy_model
        if distributed:
            train_model = DDP(train_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        if is_main_process(rank):
            print(f"Using device: {device}")
            print(f"World size: {world_size}")
            print(f"Loaded policy checkpoint: {args.base_checkpoint}")
            print(f"Loaded reference checkpoint: {args.reference_checkpoint}")
            print(f"Attention backend: {config.attention_backend}")
            print(f"Precision path: {precision_label}")
            print(
                f"MoR depth budget: 1..{config.mor_max_depth} recursions "
                f"(max effective depth {config.effective_layer_count}, target {config.target_effective_layer_count:.1f})"
            )

        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        pad_token_id = tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            raise ValueError("Tokenizer is missing <pad> token.")

        train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
        val_dataset = load_dataset("json", data_files=args.val_jsonl, split="train")
        train_dataset = train_dataset.map(
            lambda row: tokenize_pair_example(row, tokenizer=tokenizer, max_length=args.max_length),
            remove_columns=train_dataset.column_names,
            desc="Tokenizing Metis-1.5 DPO train",
        ).filter(lambda row: len(row["chosen_input_ids"]) > 0 and len(row["rejected_input_ids"]) > 0)
        val_dataset = val_dataset.map(
            lambda row: tokenize_pair_example(row, tokenizer=tokenizer, max_length=args.max_length),
            remove_columns=val_dataset.column_names,
            desc="Tokenizing Metis-1.5 DPO val",
        ).filter(lambda row: len(row["chosen_input_ids"]) > 0 and len(row["rejected_input_ids"]) > 0)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("DPO tokenization produced an empty train or val dataset.")

        pin_memory = device.type == "cuda"
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=pin_memory,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=lambda batch: collate_pair_batch(
                batch,
                pad_token_id,
                pad_to_multiple=config.fp8_pad_multiple if fp8_enabled else None,
            ),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=lambda batch: collate_pair_batch(
                batch,
                pad_token_id,
                pad_to_multiple=config.fp8_pad_multiple if fp8_enabled else None,
            ),
        )

        updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
        total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
        optimizer, optimizer_summary = build_optimizer_from_args(policy_model, args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_lr(
                step,
                max_steps=total_updates,
                warmup_steps=args.warmup_steps,
            ),
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process(rank):
            train_config = vars(args) | {"world_size": world_size}
            (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2) + "\n", encoding="utf-8")
            if optimizer_summary is not None:
                print(
                    "Optimizer: Muon-AdamW hybrid | "
                    f"muon {optimizer_summary.muon_params:,} params | "
                    f"adamw {optimizer_summary.adamw_params:,} params | "
                    f"routed_experts_muon={optimizer_summary.routed_experts_muon}",
                    flush=True,
                )
            else:
                print("Optimizer: AdamW", flush=True)
        barrier(distributed)

        latest_checkpoint_path = out_dir / "latest.pt"
        best_val_loss = math.inf
        global_step = 0
        start_epoch = 1

        if args.resume and latest_checkpoint_path.exists():
            resume_checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
            policy_model.load_state_dict(
                filter_state_dict_for_model(policy_model, resume_checkpoint["model_state_dict"]),
                strict=False,
            )
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            global_step = int(resume_checkpoint.get("global_step", 0))
            best_val_loss = float(resume_checkpoint.get("best_val_loss", math.inf))
            policy_model.to(device=device, dtype=model_dtype)
            if is_main_process(rank):
                print(f"Resuming DPO training from {latest_checkpoint_path} after epoch {start_epoch - 1}")
        barrier(distributed)

        if global_step >= total_updates:
            if is_main_process(rank):
                print(
                    f"Checkpoint already reached global_step {global_step}, "
                    f"which is >= total_updates={total_updates}."
                )
            return

        param_count = config.estimate_params()
        total_tokens_seen = 0
        epoch = start_epoch
        while global_step < total_updates:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            interval_start_time = time.perf_counter()
            interval_loss = 0.0
            interval_accuracy = 0.0
            interval_micro_batches = 0
            interval_updates = 0
            interval_tokens = 0
            step_token_count = 0

            for step_in_epoch, batch in enumerate(train_loader, start=1):
                (
                    chosen_ids,
                    chosen_labels,
                    chosen_mask,
                    rejected_ids,
                    rejected_labels,
                    rejected_mask,
                    pair_weights,
                ) = batch
                non_blocking = device.type == "cuda"
                chosen_ids = chosen_ids.to(device, non_blocking=non_blocking)
                chosen_labels = chosen_labels.to(device, non_blocking=non_blocking)
                chosen_mask = chosen_mask.to(device, non_blocking=non_blocking)
                rejected_ids = rejected_ids.to(device, non_blocking=non_blocking)
                rejected_labels = rejected_labels.to(device, non_blocking=non_blocking)
                rejected_mask = rejected_mask.to(device, non_blocking=non_blocking)
                pair_weights = pair_weights.to(device, non_blocking=non_blocking)
                step_token_count += chosen_ids.numel() + rejected_ids.numel()

                should_step = step_in_epoch % args.grad_accum_steps == 0 or step_in_epoch == len(train_loader)
                sync_context = (
                    train_model.no_sync()
                    if distributed and not should_step
                    else contextlib.nullcontext()
                )
                with sync_context:
                    maybe_mark_compile_step(compile_enabled)
                    policy_chosen = train_model(
                        chosen_ids,
                        attention_mask=chosen_mask,
                        is_first_microbatch=((step_in_epoch - 1) % args.grad_accum_steps == 0),
                    )
                    policy_rejected = train_model(
                        rejected_ids,
                        attention_mask=rejected_mask,
                        is_first_microbatch=False,
                    )
                    with torch.no_grad():
                        reference_chosen = reference_model(
                            chosen_ids,
                            attention_mask=chosen_mask,
                            is_first_microbatch=True,
                        )
                        reference_rejected = reference_model(
                            rejected_ids,
                            attention_mask=rejected_mask,
                            is_first_microbatch=True,
                        )

                    policy_chosen_logp = sequence_log_probs(
                        policy_chosen.logits,
                        chosen_labels,
                        score_mode=args.sequence_score_mode,
                    )
                    policy_rejected_logp = sequence_log_probs(
                        policy_rejected.logits,
                        rejected_labels,
                        score_mode=args.sequence_score_mode,
                    )
                    ref_chosen_logp = sequence_log_probs(
                        reference_chosen.logits,
                        chosen_labels,
                        score_mode=args.sequence_score_mode,
                    )
                    ref_rejected_logp = sequence_log_probs(
                        reference_rejected.logits,
                        rejected_labels,
                        score_mode=args.sequence_score_mode,
                    )

                    preference_margin = (
                        (policy_chosen_logp - policy_rejected_logp)
                        - (ref_chosen_logp - ref_rejected_logp)
                    )
                    dpo_loss = (-F.logsigmoid(args.dpo_beta * preference_margin) * pair_weights).mean()
                    aux_loss = dpo_loss.new_zeros(())
                    if policy_chosen.route_aux_loss is not None and policy_rejected.route_aux_loss is not None:
                        aux_loss = aux_loss + (
                            0.5 * (policy_chosen.route_aux_loss + policy_rejected.route_aux_loss)
                            * config.mor_router_aux_loss_coef
                        )
                    if policy_chosen.moe_aux_loss is not None and policy_rejected.moe_aux_loss is not None:
                        aux_loss = aux_loss + (
                            0.5 * (policy_chosen.moe_aux_loss + policy_rejected.moe_aux_loss)
                            * config.moe_aux_loss_coef
                        )
                    loss = dpo_loss + aux_loss
                    (loss / args.grad_accum_steps).backward()

                interval_loss += float(dpo_loss.item())
                interval_accuracy += float((preference_margin > 0).float().mean().item())
                interval_micro_batches += 1

                if should_step:
                    if args.grad_clip > 0:
                        if distributed:
                            torch.nn.utils.clip_grad_norm_(train_model.module.parameters(), args.grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    interval_updates += 1
                    interval_tokens += step_token_count
                    total_tokens_seen += step_token_count * world_size

                    if args.log_interval > 0 and global_step % args.log_interval == 0:
                        interval_elapsed = max(time.perf_counter() - interval_start_time, 1e-6)
                        total_loss, total_accuracy, total_updates_logged, total_micro_batches, total_tokens = aggregate_interval_stats(
                            loss_total=interval_loss,
                            accuracy_total=interval_accuracy,
                            updates=interval_updates,
                            micro_batches=interval_micro_batches,
                            tokens=interval_tokens,
                            device=device,
                            distributed=distributed,
                        )
                        if is_main_process(rank):
                            global_updates = (
                                max(total_updates_logged // world_size, 1)
                                if distributed
                                else max(total_updates_logged, 1)
                            )
                            avg_step_seconds = interval_elapsed / global_updates
                            tokens_per_second = total_tokens / interval_elapsed
                            est_tflops = estimate_tflops(tokens_per_second, param_count)
                            print(
                                f"epoch {epoch:3d} | step {global_step:5d} | "
                                f"dpo {total_loss / max(total_micro_batches, 1):.4f} | "
                                f"acc {total_accuracy / max(total_micro_batches, 1):.3f} | "
                                f"lr {scheduler.get_last_lr()[0]:.6e} | "
                                f"seq_tok/s {tokens_per_second:,.0f} | "
                                f"step_s {avg_step_seconds:.2f} | "
                                f"seq_tok_seen {total_tokens_seen / 1e9:.3f}B | "
                                f"est_tflops {est_tflops:.2f}",
                                flush=True,
                            )
                        interval_start_time = time.perf_counter()
                        interval_loss = 0.0
                        interval_accuracy = 0.0
                        interval_micro_batches = 0
                        interval_updates = 0
                        interval_tokens = 0
                    step_token_count = 0

                    if global_step >= total_updates:
                        break

                    should_eval = global_step % args.eval_interval == 0
                    should_checkpoint = global_step % args.checkpoint_interval == 0

                    if should_eval:
                        if is_main_process(rank):
                            val_loss, val_accuracy = evaluate(
                                policy_model,
                                reference_model,
                                val_loader,
                                device,
                                compiled=compile_enabled,
                                beta=args.dpo_beta,
                                score_mode=args.sequence_score_mode,
                            )
                        else:
                            val_loss, val_accuracy = 0.0, 0.0
                        val_loss, val_accuracy = broadcast_pair(
                            val_loss,
                            val_accuracy,
                            device=device,
                            distributed=distributed,
                        )
                        if is_main_process(rank):
                            print(
                                f"epoch {epoch:3d} | step {global_step:5d} | "
                                f"val {val_loss:.4f} | acc {val_accuracy:.3f}",
                                flush=True,
                            )
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                save_checkpoint(
                                    out_dir / "best.pt",
                                    model=policy_model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    global_step=global_step,
                                    best_val_loss=best_val_loss,
                                    train_args=vars(args) | {"world_size": world_size},
                                )
                        barrier(distributed)

                    if should_checkpoint:
                        if is_main_process(rank):
                            save_checkpoint(
                                latest_checkpoint_path,
                                model=policy_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                best_val_loss=best_val_loss,
                                train_args=vars(args) | {"world_size": world_size},
                            )
                        barrier(distributed)

            if is_main_process(rank):
                val_loss, val_accuracy = evaluate(
                    policy_model,
                    reference_model,
                    val_loader,
                    device,
                    compiled=compile_enabled,
                    beta=args.dpo_beta,
                    score_mode=args.sequence_score_mode,
                )
            else:
                val_loss, val_accuracy = 0.0, 0.0
            val_loss, val_accuracy = broadcast_pair(
                val_loss,
                val_accuracy,
                device=device,
                distributed=distributed,
            )
            if is_main_process(rank):
                print(
                    f"epoch {epoch:3d} | step {global_step:5d} | "
                    f"val {val_loss:.4f} | acc {val_accuracy:.3f}",
                    flush=True,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        out_dir / "best.pt",
                        model=policy_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        train_args=vars(args) | {"world_size": world_size},
                    )
                save_checkpoint(
                    latest_checkpoint_path,
                    model=policy_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    train_args=vars(args) | {"world_size": world_size},
                )
            barrier(distributed)
            epoch += 1
    finally:
        cleanup_runtime(distributed)


if __name__ == "__main__":
    main()
