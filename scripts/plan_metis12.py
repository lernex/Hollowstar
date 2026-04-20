from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_stage_plan(
    *,
    train_examples: int,
    epochs: float,
    global_batch_size: int,
    warmup_ratio: float,
    lr: float,
) -> dict:
    steps = max(1, math.ceil((train_examples * epochs) / global_batch_size))
    warmup_steps = max(1, int(round(steps * warmup_ratio)))
    return {
        "train_examples": train_examples,
        "epochs": epochs,
        "global_batch_size": global_batch_size,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the derived Metis-1.2 stage plan from prepared data.")
    parser.add_argument("--manifest", default="configs/metis12_manifest.json")
    parser.add_argument("--pretrain-meta", default="data/metis12_base/meta.json")
    parser.add_argument("--chat-meta", default="data/metis12_chat_sft/meta.json")
    parser.add_argument("--reasoning-meta", default="data/metis12_reasoning_sft/meta.json")
    parser.add_argument("--output-path", default="artifacts/metis12_plan.json")
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest))
    pretrain_meta = load_json(Path(args.pretrain_meta))
    chat_meta = load_json(Path(args.chat_meta))
    reasoning_meta = load_json(Path(args.reasoning_meta))

    seq_len = manifest["model"]["block_size"]
    target_tokens = manifest["pretrain"]["target_train_tokens"]
    global_batch_size = manifest["pretrain"]["global_batch_size"]
    tokens_per_step = global_batch_size * seq_len
    base_steps = max(1, math.ceil(target_tokens / tokens_per_step))
    base_warmup = max(1, int(round(base_steps * manifest["pretrain"]["warmup_ratio"])))

    chat_plan = build_stage_plan(
        train_examples=int(chat_meta["train_examples"]),
        epochs=float(manifest["chat_sft"]["epochs"]),
        global_batch_size=int(manifest["chat_sft"]["global_batch_size"]),
        warmup_ratio=float(manifest["chat_sft"]["warmup_ratio"]),
        lr=float(manifest["chat_sft"]["base_lr"]),
    )
    reasoning_plan = build_stage_plan(
        train_examples=int(reasoning_meta["train_examples"]),
        epochs=float(manifest["reasoning_sft"]["epochs"]),
        global_batch_size=int(manifest["reasoning_sft"]["global_batch_size"]),
        warmup_ratio=float(manifest["reasoning_sft"]["warmup_ratio"]),
        lr=float(manifest["reasoning_sft"]["base_lr"]),
    )

    output = {
        "name": manifest["name"],
        "model": manifest["model"],
        "pretrain": {
            "target_train_tokens": target_tokens,
            "prepared_train_tokens": int(pretrain_meta["train_tokens"]),
            "prepared_val_tokens": int(pretrain_meta["val_tokens"]),
            "global_batch_size": global_batch_size,
            "local_batch_size": int(manifest["pretrain"]["local_batch_size"]),
            "tokens_per_step": tokens_per_step,
            "steps": base_steps,
            "warmup_steps": base_warmup,
            "lr": float(manifest["pretrain"]["base_lr"]),
            "weight_decay": float(manifest["pretrain"]["weight_decay"]),
        },
        "chat_sft": chat_plan,
        "reasoning_sft": reasoning_plan,
        "release_repos": manifest["release"]["repos"],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

