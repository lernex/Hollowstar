from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data_mixture import DatasetMixture


def log_progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild a tokenizer sample JSONL from a dataset mixture.")
    parser.add_argument("--mixture-config", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mixture = DatasetMixture(args.mixture_config, total_examples=args.max_samples)
    source_counts = {spec.name: 0 for spec in mixture.sources}

    with output_path.open("w", encoding="utf-8") as handle:
        for yielded, example in enumerate(mixture, start=1):
            source_counts[example["source"]] += 1
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            if yielded == 1 or yielded % args.progress_interval == 0:
                log_progress(
                    f"Tokenizer sample docs written: {yielded}/{args.max_samples} | "
                    f"source={example['source']}"
                )

    meta = {
        "mixture_config": args.mixture_config,
        "max_samples": args.max_samples,
        "output_path": str(output_path),
        "source_counts": source_counts,
        "mixture": mixture.summary(),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_progress(f"Saved tokenizer sample to {output_path}")
    log_progress(json.dumps(meta, indent=2))
    # Explicitly terminate after writing outputs. Large streaming/HF iterators have
    # previously crashed during Python teardown on remote pods even after success.
    os._exit(0)


if __name__ == "__main__":
    main()
