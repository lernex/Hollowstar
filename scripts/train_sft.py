from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinylm import GPTConfig, GPTLanguageModel


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def collate_batch(
    batch: list[dict],
    pad_token_id: int,
    block_size: int,
    pad_to_block_size: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    trimmed_batch = []
    for item in batch:
        input_ids = item["input_ids"][-block_size:]
        labels = item["labels"][-block_size:]
        trimmed_batch.append({"input_ids": input_ids, "labels": labels})

    max_len = block_size if pad_to_block_size else max(len(item["input_ids"]) for item in trimmed_batch)
    input_ids = []
    labels = []
    for item in trimmed_batch:
        seq = item["input_ids"]
        tgt = item["labels"]
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_token_id] * pad_len)
        labels.append(tgt + [-100] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        non_blocking = device.type == "cuda"
        xb = xb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)
        with autocast_context():
            _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for chat-style data.")
    parser.add_argument("--base-checkpoint", default="checkpoints/default/best.pt")
    parser.add_argument("--train-data", default="data/chat_sft/train.pt")
    parser.add_argument("--val-data", default="data/chat_sft/val.pt")
    parser.add_argument("--out-dir", default="checkpoints/chat_sft")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--pad-to-block-size", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    maybe_enable_cuda_speedups(device, args.matmul_precision, args.tf32)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Using device: {device}")
    print(f"Loaded base checkpoint: {args.base_checkpoint}")

    train_examples = torch.load(args.train_data)
    val_examples = torch.load(args.val_data)
    if not train_examples:
        raise ValueError("Training set is empty.")
    if not val_examples:
        raise ValueError("Validation set is empty.")

    pad_token_id = 0
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id, config.block_size, args.pad_to_block_size),
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id, config.block_size, args.pad_to_block_size),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    latest_checkpoint_path = out_dir / "latest.pt"
    best_val_loss = math.inf
    global_step = 0
    start_epoch = 1

    if args.resume and latest_checkpoint_path.exists():
        resume_checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_val_loss = float(resume_checkpoint.get("best_val_loss", math.inf))
        print(f"Resuming SFT from {latest_checkpoint_path} after epoch {start_epoch - 1}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}, "
            f"which is >= epochs={args.epochs}. Nothing to do."
        )
        return

    optimizer = build_optimizer(model, device, args)
    amp_dtype = parse_dtype(args.dtype)
    autocast_enabled = args.dtype != "fp32" and device.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and args.dtype == "fp16"))

    if args.resume and latest_checkpoint_path.exists():
        resume_checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)

    def autocast_context():
        if autocast_enabled:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    train_model = torch.compile(model, mode=args.compile_mode) if args.compile else model

    for epoch in range(start_epoch, args.epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, (xb, yb) in enumerate(train_loader, start=1):
            non_blocking = device.type == "cuda"
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            with autocast_context():
                _, loss = train_model(xb, yb)
            running_loss += loss.item()
            scaler.scale(loss / args.grad_accum_steps).backward()

            if step_in_epoch % args.grad_accum_steps == 0 or step_in_epoch == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(train_model, val_loader, device, autocast_context)
        print(
            f"epoch {epoch:3d} | step {global_step:4d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | ppl {math.exp(val_loss):.2f}"
        )

        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            best_val_loss,
            vars(args),
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                vars(args),
            )


if __name__ == "__main__":
    main()
