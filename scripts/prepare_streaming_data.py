from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from data_mixture import DatasetMixture


def split_name(doc_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def iter_jsonl_rows(
    path: str | Path,
    *,
    text_field: str,
    id_field: str,
    source_field: str,
    max_docs: int | None,
) -> Iterator[dict[str, str]]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_docs is not None and count >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            count += 1
            yield {
                "source": str(row.get(source_field, "jsonl")),
                "doc_id": str(row.get(id_field, count)),
                "text": str(row.get(text_field, "")),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare binary token files from a streaming text dataset.")
    parser.add_argument("--mixture-config", default=None)
    parser.add_argument("--jsonl-path", default=None)
    parser.add_argument("--jsonl-text-field", default="text")
    parser.add_argument("--jsonl-id-field", default="doc_id")
    parser.add_argument("--jsonl-source-field", default="source")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/metis_base")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--target-total-tokens", type=int, default=None)
    parser.add_argument("--target-train-tokens", type=int, default=None)
    parser.add_argument("--target-val-tokens", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    counts = {
        "train_docs": 0,
        "val_docs": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }
    source_counts: dict[str, dict[str, int]] = {}

    target_train_tokens = args.target_train_tokens
    target_val_tokens = args.target_val_tokens
    if args.target_total_tokens is not None and target_train_tokens is None and target_val_tokens is None:
        target_val_tokens = int(round(args.target_total_tokens * args.val_ratio))
        target_train_tokens = max(args.target_total_tokens - target_val_tokens, 0)
    elif target_train_tokens is not None and target_val_tokens is None:
        target_val_tokens = int(round(target_train_tokens * args.val_ratio / max(1 - args.val_ratio, 1e-9)))

    token_target_mode = target_train_tokens is not None or target_val_tokens is not None

    if args.jsonl_path:
        mixture = None
        row_iterator = iter_jsonl_rows(
            args.jsonl_path,
            text_field=args.jsonl_text_field,
            id_field=args.jsonl_id_field,
            source_field=args.jsonl_source_field,
            max_docs=args.max_docs,
        )
        progress_total = args.max_docs
    elif args.mixture_config:
        if args.max_docs is None:
            raise ValueError("--max-docs is required when using --mixture-config.")
        mixture = DatasetMixture(args.mixture_config, total_examples=args.max_docs)
        row_iterator = iter(mixture)
        progress_total = args.max_docs
    else:
        mixture = None
        dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config,
            split=args.split,
            streaming=True,
        )
        if args.max_docs is not None:
            dataset = islice(dataset, args.max_docs)
        row_iterator = (
            {
                "source": args.dataset_config or args.dataset_name,
                "doc_id": str(row.get(args.id_column, index)),
                "text": row.get(args.text_column, ""),
            }
            for index, row in enumerate(dataset, start=1)
        )
        progress_total = args.max_docs

    batch_rows: list[dict[str, str]] = []

    def flush_batch() -> None:
        nonlocal batch_rows
        if not batch_rows:
            return

        encoded_batch = tokenizer.encode_batch([row["text"] for row in batch_rows])
        for row, encoding in zip(batch_rows, encoded_batch):
            ids = np.asarray(encoding.ids, dtype=dtype)
            if ids.size == 0:
                continue

            split = row["split"]
            if split == "train" and target_train_tokens is not None and counts["train_tokens"] >= target_train_tokens:
                continue
            if split == "val" and target_val_tokens is not None and counts["val_tokens"] >= target_val_tokens:
                continue

            source_name = row["source"]
            source_bucket = source_counts.setdefault(
                source_name,
                {"train_docs": 0, "val_docs": 0, "train_tokens": 0, "val_tokens": 0},
            )

            if split == "train":
                if target_train_tokens is not None:
                    remaining = target_train_tokens - counts["train_tokens"]
                    ids = ids[: max(remaining, 0)]
                    if ids.size == 0:
                        continue
                ids.tofile(train_handle)
                counts["train_docs"] += 1
                counts["train_tokens"] += int(ids.size)
                source_bucket["train_docs"] += 1
                source_bucket["train_tokens"] += int(ids.size)
            else:
                if target_val_tokens is not None:
                    remaining = target_val_tokens - counts["val_tokens"]
                    ids = ids[: max(remaining, 0)]
                    if ids.size == 0:
                        continue
                ids.tofile(val_handle)
                counts["val_docs"] += 1
                counts["val_tokens"] += int(ids.size)
                source_bucket["val_docs"] += 1
                source_bucket["val_tokens"] += int(ids.size)

        batch_rows = []

    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for index, row in enumerate(
            tqdm(row_iterator, total=progress_total, desc="Streaming documents"),
            start=1,
        ):
            if token_target_mode:
                train_done = target_train_tokens is not None and counts["train_tokens"] >= target_train_tokens
                val_done = target_val_tokens is not None and counts["val_tokens"] >= target_val_tokens
                if train_done and val_done:
                    break

            text = row.get("text", "")
            if not text or not text.strip():
                continue

            doc_id = str(row.get("doc_id"))
            split = split_name(doc_id, args.val_ratio)
            batch_rows.append(
                {
                    "source": str(row.get("source", args.dataset_config or args.dataset_name)),
                    "doc_id": doc_id,
                    "split": split,
                    "text": text,
                }
            )

            if len(batch_rows) >= args.encode_batch_size:
                flush_batch()

            if index == 1 or index % args.progress_interval == 0:
                print(
                    "Prepared docs "
                    f"{index}/{args.max_docs} | "
                    f"train_docs={counts['train_docs']} val_docs={counts['val_docs']} | "
                    f"train_tokens={counts['train_tokens']:,} val_tokens={counts['val_tokens']:,}",
                    flush=True,
                )

        flush_batch()

    meta = {
        "source_mode": "jsonl" if args.jsonl_path else ("mixture" if mixture is not None else "single_dataset"),
        "mixture_config": args.mixture_config,
        "jsonl_path": args.jsonl_path,
        "tokenizer_path": args.tokenizer_path,
        "text_column": args.text_column,
        "id_column": args.id_column,
        "max_docs": args.max_docs,
        "val_ratio": args.val_ratio,
        "target_total_tokens": args.target_total_tokens,
        "target_train_tokens": target_train_tokens,
        "target_val_tokens": target_val_tokens,
        "vocab_size": vocab_size,
        "dtype": str(np.dtype(dtype)),
        "encode_batch_size": args.encode_batch_size,
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "source_counts": source_counts,
    }
    if args.jsonl_path:
        meta["jsonl_text_field"] = args.jsonl_text_field
        meta["jsonl_id_field"] = args.jsonl_id_field
        meta["jsonl_source_field"] = args.jsonl_source_field
    elif mixture is not None:
        meta["mixture"] = mixture.summary()
    else:
        meta["dataset_name"] = args.dataset_name
        meta["dataset_config"] = args.dataset_config
        meta["split"] = args.split
    meta.update(counts)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
