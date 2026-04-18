from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


ROLE_PREFIX = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hf_split(dataset_name: str, split: str, limit: int | None):
    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    return list(dataset)


def load_hf_stream(dataset_name: str, split: str, limit: int | None):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if limit is None:
        raise ValueError("Streaming mode requires an explicit limit.")
    rows = []
    for index, row in enumerate(islice(dataset, limit), start=1):
        rows.append(row)
        if index == 1 or index % 5000 == 0:
            print(f"Loaded chat examples from HF: {index}/{limit}", flush=True)
    return rows


def split_name(example: dict, index: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"

    payload = None
    for key in ("id", "prompt", "instruction"):
        if key in example and example[key]:
            payload = str(example[key])
            break
    if payload is None:
        payload = json.dumps(
            example.get("messages") or example.get("conversations") or example,
            sort_keys=True,
            ensure_ascii=False,
        )
    digest = hashlib.sha1(f"{index}:{payload}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def normalize_messages(example: dict) -> list[dict[str, str]]:
    if "messages" in example:
        return example["messages"]
    if "prompt" in example and "response" in example:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    if "instruction" in example and "output" in example:
        instruction = example["instruction"].strip()
        extra_input = example.get("input", "").strip()
        prompt = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["output"]},
        ]
    raise ValueError("Each example must contain either messages or prompt/response.")


def encode_conversation(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> dict | None:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    input_ids: list[int] = []
    labels: list[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    saw_assistant = False
    last_role = None

    for message in messages:
        role = message["role"].strip().lower()
        if role not in ROLE_PREFIX:
            raise ValueError(f"Unsupported role: {role}")
        text = f"{ROLE_PREFIX[role]}: {message['content'].strip()}\n"
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        input_ids.extend(token_ids)
        if role == "assistant":
            labels.extend(token_ids)
            saw_assistant = True
        else:
            labels.extend([-100] * len(token_ids))
        last_role = role

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id if last_role == "assistant" else -100)

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    if not saw_assistant or not any(label != -100 for label in labels):
        return None

    return {"input_ids": input_ids, "labels": labels}


def write_dataset_from_examples(
    raw_examples,
    output_path: Path,
    tokenizer: Tokenizer,
    max_length: int,
    split_name: str,
) -> tuple[int, int]:
    processed = []
    skipped = 0
    total = len(raw_examples)
    for index, example in enumerate(tqdm(raw_examples, desc=f"Encoding {split_name}", total=total), start=1):
        item = encode_conversation(
            tokenizer=tokenizer,
            messages=normalize_messages(example),
            max_length=max_length,
        )
        if item is None:
            skipped += 1
            continue
        processed.append(item)
        if index == 1 or index % 5000 == 0:
            print(
                f"Encoded {split_name} chat examples: {index}/{total} | "
                f"kept={len(processed)} skipped={skipped}",
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed, output_path)
    return len(processed), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chat SFT data from JSONL or Hugging Face datasets.")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--train-jsonl", default="examples/chat/train.jsonl")
    parser.add_argument("--val-jsonl", default="examples/chat/val.jsonl")
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-train-split", default="train_sft")
    parser.add_argument("--hf-val-split", default="test_sft")
    parser.add_argument("--hf-limit", type=int, default=None)
    parser.add_argument("--hf-train-limit", type=int, default=None)
    parser.add_argument("--hf-val-limit", type=int, default=None)
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--output-dir", default="data/chat_sft")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    output_dir = Path(args.output_dir)

    if args.hf_dataset:
        if args.hf_split:
            loader = load_hf_stream if args.hf_streaming else load_hf_split
            examples = loader(args.hf_dataset, args.hf_split, args.hf_limit)
            train_examples = []
            val_examples = []
            for index, example in enumerate(examples, start=1):
                target = val_examples if split_name(example, index, args.val_ratio) == "val" else train_examples
                target.append(example)
            source = {
                "hf_dataset": args.hf_dataset,
                "hf_split": args.hf_split,
                "hf_limit": args.hf_limit,
                "hf_streaming": args.hf_streaming,
                "val_ratio": args.val_ratio,
            }
        else:
            train_examples = load_hf_split(args.hf_dataset, args.hf_train_split, args.hf_train_limit)
            val_examples = load_hf_split(args.hf_dataset, args.hf_val_split, args.hf_val_limit)
            source = {
                "hf_dataset": args.hf_dataset,
                "hf_train_split": args.hf_train_split,
                "hf_val_split": args.hf_val_split,
                "hf_train_limit": args.hf_train_limit,
                "hf_val_limit": args.hf_val_limit,
                "hf_streaming": False,
            }
    else:
        train_examples = load_jsonl(Path(args.train_jsonl))
        val_examples = load_jsonl(Path(args.val_jsonl))
        source = {
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
        }

    train_count, train_skipped = write_dataset_from_examples(
        train_examples,
        output_dir / "train.pt",
        tokenizer,
        args.max_length,
        "train",
    )
    val_count, val_skipped = write_dataset_from_examples(
        val_examples,
        output_dir / "val.pt",
        tokenizer,
        args.max_length,
        "val",
    )

    metadata = {
        "tokenizer_path": args.tokenizer_path,
        "max_length": args.max_length,
        "train_examples": train_count,
        "val_examples": val_count,
        "train_skipped": train_skipped,
        "val_skipped": val_skipped,
    }
    metadata.update(source)
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
