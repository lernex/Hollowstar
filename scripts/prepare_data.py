from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def token_count(dataset, tokenizer: Tokenizer, text_column: str, batch_size: int) -> int:
    total = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc="Counting tokens"):
        rows = dataset[start : start + batch_size][text_column]
        encodings = tokenizer.encode_batch(rows)
        total += sum(len(enc.ids) for enc in encodings)
    return total


def write_split(
    dataset,
    tokenizer: Tokenizer,
    output_path: Path,
    text_column: str,
    batch_size: int,
    dtype: np.dtype,
) -> tuple[int, int]:
    total_tokens = token_count(dataset, tokenizer, text_column, batch_size)
    mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))

    write_index = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Writing {output_path.name}"):
        rows = dataset[start : start + batch_size][text_column]
        encodings = tokenizer.encode_batch(rows)
        flat_ids = np.concatenate([np.asarray(enc.ids, dtype=dtype) for enc in encodings])
        mmap[write_index : write_index + len(flat_ids)] = flat_ids
        write_index += len(flat_ids)

    mmap.flush()
    return len(dataset), total_tokens


def load_split(dataset_name: str, split: str, limit: int | None):
    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize TinyStories into binary files.")
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/tinystories_bpe")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(args.dataset_name, args.train_split, args.train_limit)
    val_ds = load_split(args.dataset_name, args.val_split, args.val_limit)

    train_rows, train_tokens = write_split(
        train_ds,
        tokenizer,
        output_dir / "train.bin",
        args.text_column,
        args.batch_size,
        dtype,
    )
    val_rows, val_tokens = write_split(
        val_ds,
        tokenizer,
        output_dir / "val.bin",
        args.text_column,
        args.batch_size,
        dtype,
    )

    meta = {
        "dataset_name": args.dataset_name,
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": vocab_size,
        "dtype": str(np.dtype(dtype)),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_bin": str(output_dir / "train.bin"),
        "val_bin": str(output_dir / "val.bin"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

