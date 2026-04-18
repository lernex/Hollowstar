from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from data_mixture import DatasetMixture


def split_name(doc_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare binary token files from a streaming text dataset.")
    parser.add_argument("--mixture-config", default=None)
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/metis_base")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-docs", type=int, default=60000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
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

    if args.mixture_config:
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
        row_iterator = (
            {
                "source": args.dataset_config or args.dataset_name,
                "doc_id": str(row.get(args.id_column, index)),
                "text": row.get(args.text_column, ""),
            }
            for index, row in enumerate(islice(dataset, args.max_docs), start=1)
        )
        progress_total = args.max_docs

    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for row in tqdm(row_iterator, total=progress_total, desc="Streaming documents"):
            text = row.get("text", "")
            if not text or not text.strip():
                continue

            doc_id = str(row.get("doc_id"))
            source_name = str(row.get("source", args.dataset_config or args.dataset_name))
            split = split_name(doc_id, args.val_ratio)
            ids = np.asarray(tokenizer.encode(text).ids, dtype=dtype)
            if ids.size == 0:
                continue

            source_bucket = source_counts.setdefault(
                source_name,
                {"train_docs": 0, "val_docs": 0, "train_tokens": 0, "val_tokens": 0},
            )
            if split == "train":
                ids.tofile(train_handle)
                counts["train_docs"] += 1
                counts["train_tokens"] += int(ids.size)
                source_bucket["train_docs"] += 1
                source_bucket["train_tokens"] += int(ids.size)
            else:
                ids.tofile(val_handle)
                counts["val_docs"] += 1
                counts["val_tokens"] += int(ids.size)
                source_bucket["val_docs"] += 1
                source_bucket["val_tokens"] += int(ids.size)

    meta = {
        "source_mode": "mixture" if mixture is not None else "single_dataset",
        "mixture_config": args.mixture_config,
        "tokenizer_path": args.tokenizer_path,
        "text_column": args.text_column,
        "id_column": args.id_column,
        "max_docs": args.max_docs,
        "val_ratio": args.val_ratio,
        "vocab_size": vocab_size,
        "dtype": str(np.dtype(dtype)),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "source_counts": source_counts,
    }
    if mixture is not None:
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
