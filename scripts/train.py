from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinylm import GPTConfig, GPTLanguageModel


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[dtype_name]


def maybe_enable_cuda_speedups(
    device: torch.device,
    matmul_precision: str | None,
    tf32: bool,
) -> None:
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def build_optimizer(
    model: GPTLanguageModel,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    use_fused = bool(args.fused_adamw and device.type == "cuda")
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=use_fused,
    )


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the selected block size.")
    positions = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.asarray(data[pos : pos + block_size], dtype=np.int64)) for pos in positions]
    )
    y = torch.stack(
        [torch.from_numpy(np.asarray(data[pos + 1 : pos + 1 + block_size], dtype=np.int64)) for pos in positions]
    )
    non_blocking = device.type == "cuda"
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    autocast_context,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            with autocast_context():
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_val_loss: float,
    train_args: dict,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "model_config": model.config.to_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT on tokenized TinyStories.")
    parser.add_argument("--data-dir", default="data/tinystories_bpe")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--out-dir", default="checkpoints/default")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")
    maybe_enable_cuda_speedups(device, args.matmul_precision, args.tf32)

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)
    dtype = np.dtype(meta["dtype"])

    train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model = GPTLanguageModel(config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    latest_checkpoint_path = out_dir / "latest.pt"
    best_val_loss = math.inf
    start_step = 0
    previous_elapsed = 0.0

    if args.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"Resuming from {latest_checkpoint_path} at step {start_step}")

    if start_step >= args.max_steps:
        print(
            f"Checkpoint already reached step {start_step}, "
            f"which is >= max_steps={args.max_steps}. Nothing to do."
        )
        return

    optimizer = build_optimizer(model, device, args)
    amp_dtype = parse_dtype(args.dtype)
    autocast_enabled = args.dtype != "fp32" and device.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and args.dtype == "fp16"))

    if args.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)

    def autocast_context():
        if autocast_enabled:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    train_model = torch.compile(model, mode=args.compile_mode) if args.compile else model

    wall_start_time = time.time()
    last_log_time = wall_start_time

    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(args.grad_accum_steps):
            xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
            with autocast_context():
                _, loss = train_model(xb, yb)
            running_loss += loss.item()
            scaler.scale(loss / args.grad_accum_steps).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == 1:
            now = time.time()
            dt = max(now - last_log_time, 1e-9)
            steps_since_log = step - max(start_step, step - args.log_interval)
            tokens = steps_since_log * args.grad_accum_steps * args.batch_size * args.block_size
            if step == start_step + 1:
                tokens = args.grad_accum_steps * args.batch_size * args.block_size
            tok_per_sec = tokens / dt
            avg_loss = running_loss / args.grad_accum_steps
            elapsed_minutes = (previous_elapsed + (now - wall_start_time)) / 60
            print(
                f"step {step:5d} | train_loss {avg_loss:.4f} | "
                f"tok/s {tok_per_sec:,.0f} | elapsed {elapsed_minutes:.1f} min"
            )
            last_log_time = now

        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(
                train_model,
                train_data,
                val_data,
                args.eval_iters,
                args.batch_size,
                args.block_size,
                device,
                autocast_context,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]
            elapsed_seconds = previous_elapsed + (time.time() - wall_start_time)
            print(
                f"eval step {step:5d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | ppl {math.exp(val_loss):.2f}"
            )
            save_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scaler,
                step,
                best_val_loss,
                vars(args),
                elapsed_seconds,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_val_loss,
                    vars(args),
                    elapsed_seconds,
                )


if __name__ == "__main__":
    main()
