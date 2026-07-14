from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data_mixture import DatasetMixture
from jsonl_artifacts import iter_jsonl_records
from normalized_shard_mixture import NormalizedShardMixture


def log_progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild a tokenizer sample JSONL from a dataset mixture.")
    parser.add_argument("--mixture-config", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--jsonl-dir", default=None)
    parser.add_argument("--jsonl-glob", default="*.jsonl*")
    parser.add_argument("--jsonl-text-field", default="text")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    if not args.mixture_config and not args.jsonl_dir:
        raise ValueError("Provide either --mixture-config or --jsonl-dir.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_counts: dict[str, int] = {}

    if args.mixture_config:
        if args.normalized_root:
            mixture = NormalizedShardMixture(
                args.mixture_config,
                args.normalized_root,
                total_examples=args.max_samples,
                glob_pattern=args.jsonl_glob,
            )
        else:
            mixture = DatasetMixture(args.mixture_config, total_examples=args.max_samples)
        source_counts = {spec.name: 0 for spec in mixture.sources}
        row_iterator = iter(mixture)
    else:
        mixture = None
        row_iterator = iter_jsonl_records(
            jsonl_dir=args.jsonl_dir,
            glob_pattern=args.jsonl_glob,
            max_rows=args.max_samples,
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for yielded, example in enumerate(row_iterator, start=1):
            if yielded > args.max_samples:
                break
            source_name = str(example.get("source", "jsonl"))
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            if mixture is None:
                payload = {
                    "source": source_name,
                    "doc_id": str(example.get("doc_id", yielded)),
                    "text": str(example.get(args.jsonl_text_field, "")),
                }
            else:
                payload = example
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if yielded == 1 or yielded % args.progress_interval == 0:
                log_progress(
                    f"Tokenizer sample docs written: {yielded}/{args.max_samples} | "
                    f"source={source_name}"
                )

    meta = {
        "mixture_config": args.mixture_config,
        "jsonl_dir": args.jsonl_dir,
        "normalized_root": args.normalized_root,
        "jsonl_glob": args.jsonl_glob,
        "jsonl_text_field": args.jsonl_text_field,
        "max_samples": args.max_samples,
        "output_path": str(output_path),
        "source_counts": source_counts,
    }
    if mixture is not None:
        meta["mixture"] = mixture.summary()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_progress(f"Saved tokenizer sample to {output_path}")
    log_progress(json.dumps(meta, indent=2))
    # Explicitly terminate after writing outputs. Large streaming/HF iterators have
    # previously crashed during Python teardown on remote pods even after success.
    os._exit(0)


if __name__ == "__main__":
    main()
