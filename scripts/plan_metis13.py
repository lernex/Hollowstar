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


def build_token_stage_plan(
    *,
    target_train_tokens: int,
    seq_len: int,
    local_batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    warmup_ratio: float,
    lr: float,
    weight_decay: float,
    gates: list[dict] | None = None,
) -> dict:
    global_batch_size = local_batch_size * grad_accum_steps * world_size
    tokens_per_step = global_batch_size * seq_len
    steps = max(1, math.ceil(target_train_tokens / tokens_per_step))
    warmup_steps = max(1, int(round(steps * warmup_ratio)))
    plan = {
        "target_train_tokens": target_train_tokens,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "world_size": world_size,
        "tokens_per_step": tokens_per_step,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    if gates:
        plan["gates"] = [
            {
                "label": str(gate.get("label", f"gate_{gate['tokens']}")),
                "tokens": int(gate["tokens"]),
                "step": max(1, math.ceil(int(gate["tokens"]) / tokens_per_step)),
            }
            for gate in gates
        ]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the derived Metis stage plan from prepared data.")
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    parser.add_argument("--pretrain-meta", default="data/metis15_base/meta.json")
    parser.add_argument("--chat-meta", default="data/metis15_chat_sft/meta.json")
    parser.add_argument("--reasoning-meta", default="data/metis15_reasoning_sft/meta.json")
    parser.add_argument("--output-path", default="artifacts/metis15_plan.json")
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest))
    pretrain_meta = load_json(Path(args.pretrain_meta))
    chat_meta = load_json(Path(args.chat_meta))
    reasoning_meta = load_json(Path(args.reasoning_meta))

    seq_len = manifest["model"]["block_size"]
    hardware = manifest.get("hardware", {})
    world_size = int(hardware.get("world_size", 1))

    pretrain_plan = build_token_stage_plan(
        target_train_tokens=int(manifest["pretrain"]["target_train_tokens"]),
        seq_len=seq_len,
        local_batch_size=int(manifest["pretrain"]["local_batch_size"]),
        grad_accum_steps=int(manifest["pretrain"]["grad_accum_steps"]),
        world_size=world_size,
        warmup_ratio=float(manifest["pretrain"]["warmup_ratio"]),
        lr=float(manifest["pretrain"]["base_lr"]),
        weight_decay=float(manifest["pretrain"]["weight_decay"]),
        gates=manifest["pretrain"].get("gates"),
    )

    chat_plan = build_stage_plan(
        train_examples=int(chat_meta["train_examples"]),
        epochs=float(manifest["chat_sft"]["epochs"]),
        global_batch_size=(
            int(manifest["chat_sft"]["local_batch_size"])
            * int(manifest["chat_sft"]["grad_accum_steps"])
            * world_size
        ),
        warmup_ratio=float(manifest["chat_sft"]["warmup_ratio"]),
        lr=float(manifest["chat_sft"]["base_lr"]),
    )
    reasoning_plan = build_stage_plan(
        train_examples=int(reasoning_meta["train_examples"]),
        epochs=float(manifest["reasoning_sft"]["epochs"]),
        global_batch_size=(
            int(manifest["reasoning_sft"]["local_batch_size"])
            * int(manifest["reasoning_sft"]["grad_accum_steps"])
            * world_size
        ),
        warmup_ratio=float(manifest["reasoning_sft"]["warmup_ratio"]),
        lr=float(manifest["reasoning_sft"]["base_lr"]),
    )

    continued_pretrain_plan = None
    if "continued_pretrain" in manifest:
        continued_pretrain_plan = build_token_stage_plan(
            target_train_tokens=int(manifest["continued_pretrain"]["target_train_tokens"]),
            seq_len=seq_len,
            local_batch_size=int(manifest["continued_pretrain"]["local_batch_size"]),
            grad_accum_steps=int(manifest["continued_pretrain"]["grad_accum_steps"]),
            world_size=world_size,
            warmup_ratio=float(manifest["continued_pretrain"]["warmup_ratio"]),
            lr=float(manifest["continued_pretrain"]["base_lr"]),
            weight_decay=float(manifest["continued_pretrain"]["weight_decay"]),
            gates=manifest["continued_pretrain"].get("gates"),
        )

    preference_plan = None
    if "preference_optimization" in manifest:
        preference_plan = build_stage_plan(
            train_examples=int(manifest["preference_optimization"]["target_pairs"]),
            epochs=float(manifest["preference_optimization"]["epochs"]),
            global_batch_size=(
                int(manifest["preference_optimization"]["local_batch_size"])
                * int(manifest["preference_optimization"]["grad_accum_steps"])
                * world_size
            ),
            warmup_ratio=float(manifest["preference_optimization"]["warmup_ratio"]),
            lr=float(manifest["preference_optimization"]["base_lr"]),
        )

    output = {
        "name": manifest["name"],
        "model": manifest["model"],
        "hardware": hardware,
        "pretrain": pretrain_plan | {
            "prepared_train_tokens": int(pretrain_meta["train_tokens"]),
            "prepared_val_tokens": int(pretrain_meta["val_tokens"]),
        },
        "chat_sft": chat_plan,
        "reasoning_sft": reasoning_plan,
        "release_repos": manifest["release"]["repos"]
    }
    if continued_pretrain_plan is not None:
        output["continued_pretrain"] = continued_pretrain_plan
    if preference_plan is not None:
        output["preference_optimization"] = preference_plan

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
