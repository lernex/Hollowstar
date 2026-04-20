from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metis_mamba import MetisMambaConfig, build_model, cosine_lr, parse_torch_dtype


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_batch(
    data: np.memmap,
    *,
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
    model,
    *,
    train_data: np.memmap,
    val_data: np.memmap,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            logits = model(xb).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            losses[index] = loss.item()
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
    best_val_loss: float,
    train_args: dict,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "metis_mamba2_hybrid",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config.to_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def build_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=bool(args.fused_adamw and torch.cuda.is_available()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Metis Mamba2-hybrid base model on memmap token data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="configs/metis13_manifest.json")
    parser.add_argument("--out-dir", required=True)
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
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    config = MetisMambaConfig.from_dict(manifest["model"])
    device = choose_device(args.device)
    model_dtype = parse_torch_dtype(args.dtype)

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)
    dtype = np.dtype(meta["dtype"])
    train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")

    model = build_model(config)
    model.to(device=device, dtype=model_dtype)
    print(f"Using device: {device}")
    print(f"Model family: {model.model_family}")
    print(f"Estimated params: {config.estimate_params():,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr(
            step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
        ),
    )

    latest_checkpoint_path = out_dir / "latest.pt"
    start_step = 0
    best_val_loss = math.inf
    previous_elapsed = 0.0

    if args.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        model.to(device=device, dtype=model_dtype)
        print(f"Resuming from {latest_checkpoint_path} at step {start_step}")

    if start_step >= args.max_steps:
        print(f"Checkpoint already reached step {start_step}, which is >= max_steps={args.max_steps}.")
        return

    wall_start_time = time.time()
    interval_loss = 0.0
    interval_updates = 0

    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(args.grad_accum_steps):
            xb, yb = get_batch(
                train_data,
                batch_size=args.batch_size,
                block_size=config.block_size,
                device=device,
            )
            logits = model(xb).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        mean_loss = running_loss / args.grad_accum_steps
        interval_loss += mean_loss
        interval_updates += 1

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"step {step:6d} | train {interval_loss / max(interval_updates, 1):.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.6e}",
                flush=True,
            )
            interval_loss = 0.0
            interval_updates = 0

        should_eval = step == args.max_steps or (args.eval_interval > 0 and step % args.eval_interval == 0)
        should_checkpoint = step == args.max_steps or (
            args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0
        )

        if should_eval:
            losses = estimate_loss(
                model,
                train_data=train_data,
                val_data=val_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=config.block_size,
                device=device,
            )
            print(
                f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.2f}",
                flush=True,
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_val_loss=best_val_loss,
                    train_args=vars(args),
                    elapsed_seconds=previous_elapsed + (time.time() - wall_start_time),
                )

        if should_checkpoint:
            save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_loss=best_val_loss,
                train_args=vars(args),
                elapsed_seconds=previous_elapsed + (time.time() - wall_start_time),
            )


if __name__ == "__main__":
    main()
