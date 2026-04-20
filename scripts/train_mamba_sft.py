from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from metis_mamba import MetisMambaConfig, build_model, cosine_lr, parse_torch_dtype


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message['role'].capitalize()}: {message['content'].strip()}")
    return "\n".join(parts)


def tokenize_example(
    example: dict,
    *,
    tokenizer: Tokenizer,
    max_length: int,
) -> dict[str, list[int]]:
    messages = example["messages"]
    if len(messages) < 2:
        return {"input_ids": [], "labels": []}

    prompt = build_prompt(messages[:-1]) + "\nAssistant:"
    assistant = " " + messages[-1]["content"].strip()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    assistant_ids = tokenizer.encode(assistant, add_special_tokens=False).ids

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    input_ids: list[int] = []
    labels: list[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    input_ids.extend(prompt_ids)
    labels.extend([-100] * len(prompt_ids))

    input_ids.extend(assistant_ids)
    labels.extend(assistant_ids)

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id)

    input_ids = input_ids[-max_length:]
    labels = labels[-max_length:]
    if all(value == -100 for value in labels):
        return {"input_ids": [], "labels": []}
    return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch: list[dict], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        non_blocking = device.type == "cuda"
        xb = xb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)
        logits = model(xb).logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "metis_mamba2_hybrid",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "train_args": train_args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Metis-1.3 JSONL chat data.")
    parser.add_argument("--base-checkpoint", required=True)
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
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    model_dtype = parse_torch_dtype(args.dtype)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    config = MetisMambaConfig.from_dict(checkpoint["model_config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=model_dtype)
    print(f"Using device: {device}")
    print(f"Loaded base checkpoint: {args.base_checkpoint}")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing <pad> token.")

    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    val_dataset = load_dataset("json", data_files=args.val_jsonl, split="train")
    train_dataset = train_dataset.map(
        lambda row: tokenize_example(row, tokenizer=tokenizer, max_length=args.max_length),
        remove_columns=train_dataset.column_names,
        desc="Tokenizing Metis-1.3 train SFT",
    ).filter(lambda row: len(row["input_ids"]) > 0)
    val_dataset = val_dataset.map(
        lambda row: tokenize_example(row, tokenizer=tokenizer, max_length=args.max_length),
        remove_columns=val_dataset.column_names,
        desc="Tokenizing Metis-1.3 val SFT",
    ).filter(lambda row: len(row["input_ids"]) > 0)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("SFT tokenization produced an empty train or val dataset.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )

    total_updates = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * math.ceil(args.epochs))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=bool(args.fused_adamw and torch.cuda.is_available()),
    )
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
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    latest_checkpoint_path = out_dir / "latest.pt"
    best_val_loss = math.inf
    global_step = 0
    start_epoch = 1
    target_epochs = int(math.ceil(args.epochs))

    if args.resume and latest_checkpoint_path.exists():
        resume_checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_val_loss = float(resume_checkpoint.get("best_val_loss", math.inf))
        model.to(device=device, dtype=model_dtype)
        print(f"Resuming SFT from {latest_checkpoint_path} after epoch {start_epoch - 1}")

    if start_epoch > target_epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}, which is >= epochs={target_epochs}.")
        return

    for epoch in range(start_epoch, target_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        interval_loss = 0.0
        interval_updates = 0

        for step_in_epoch, (xb, yb) in enumerate(train_loader, start=1):
            non_blocking = device.type == "cuda"
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            logits = model(xb).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
                ignore_index=-100,
            )
            (loss / args.grad_accum_steps).backward()
            interval_loss += loss.item()

            if step_in_epoch % args.grad_accum_steps == 0 or step_in_epoch == len(train_loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                interval_updates += 1

                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    print(
                        f"epoch {epoch:3d} | step {global_step:5d} | "
                        f"train {interval_loss / max(interval_updates, 1):.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.6e}",
                        flush=True,
                    )
                    interval_loss = 0.0
                    interval_updates = 0

                should_eval = global_step % args.eval_interval == 0
                should_checkpoint = global_step % args.checkpoint_interval == 0

                if should_eval:
                    val_loss = evaluate(model, val_loader, device)
                    print(
                        f"epoch {epoch:3d} | step {global_step:5d} | "
                        f"val {val_loss:.4f} | ppl {math.exp(val_loss):.2f}",
                        flush=True,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            out_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            train_args=vars(args),
                        )

                if should_checkpoint:
                    save_checkpoint(
                        latest_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        train_args=vars(args),
                    )

        val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d} | step {global_step:5d} | "
            f"val {val_loss:.4f} | ppl {math.exp(val_loss):.2f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                train_args=vars(args),
            )

        save_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            train_args=vars(args),
        )


if __name__ == "__main__":
    main()
