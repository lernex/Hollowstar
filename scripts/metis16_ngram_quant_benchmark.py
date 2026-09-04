#!/usr/bin/env python3
"""Benchmark Metis N-gram tables at BF16, FP8 E4M3, and NVFP4.

The input may be an ablation ``state.pt`` checkpoint, its containing step
directory, a model checkpoint carrying ``model``/``model_state_dict``, or a
rank-two tensor saved directly with ``torch.save``.  A model checkpoint also
supplies the learned N-gram fusion projection, allowing the report to measure
error after the sixteen retrieved rows are concatenated and projected.

This command does not mutate the checkpoint and does not claim model-loss
parity.  For the latter, load the model normally and use
``metis_training.ngram_quantization.compare_model_ngram_losses``; its context
changes only packed table lookup and restores BF16 even if the probe fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from metis_training.ngram_quantization import (
    NGramQuantizationSpec,
    benchmark_table_collection,
    checkpoint_ngram_tensors,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolate N-gram table storage, lookup, and projection quantization error"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file, checkpoint step directory, or directly saved rank-two table",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device (default: cuda when available)",
    )
    parser.add_argument(
        "--checkpoint-owner",
        default=None,
        help=(
            "Production tensor-chunk table owner (for example tables-ep-0000); "
            "the first table owner is used by default"
        ),
    )
    parser.add_argument(
        "--formats",
        default="bf16,fp8_e4m3,nvfp4",
        help="Comma-separated subset of bf16,fp8_e4m3,nvfp4",
    )
    parser.add_argument(
        "--fp8-block-size",
        type=int,
        default=64,
        help="E4M3 scale block within each row; zero uses one table-wide scale",
    )
    parser.add_argument(
        "--nvfp4-rounding",
        choices=("nearest", "stochastic"),
        default="nearest",
    )
    parser.add_argument("--lookup-rows", type=int, default=8_192)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--timed-iterations", type=int, default=20)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=16_062_026)
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=None,
        help="Deterministic prefix for smoke tests; omit for exact full-table storage",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path; stdout always receives the complete report",
    )
    return parser


def _specs(args: argparse.Namespace) -> list[NGramQuantizationSpec]:
    requested = [value.strip() for value in args.formats.split(",") if value.strip()]
    specs: list[NGramQuantizationSpec] = []
    for name in requested:
        if name == "fp8_e4m3":
            specs.append(
                NGramQuantizationSpec(name, block_size=args.fp8_block_size, seed=args.seed)
            )
        elif name == "nvfp4":
            specs.append(
                NGramQuantizationSpec(
                    name,
                    block_size=16,
                    rounding=args.nvfp4_rounding,
                    seed=args.seed,
                )
            )
        else:
            specs.append(NGramQuantizationSpec(name, seed=args.seed))
    if not specs:
        raise SystemExit("--formats selected no formats")
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    tables, projection = checkpoint_ngram_tensors(
        Path(args.checkpoint),
        checkpoint_owner=args.checkpoint_owner,
    )
    original_rows = {name: int(weight.shape[0]) for name, weight in tables.items()}
    if args.max_table_rows is not None:
        if args.max_table_rows <= 0:
            raise SystemExit("--max-table-rows must be positive")
        tables = {
            name: weight[: min(args.max_table_rows, weight.shape[0])]
            for name, weight in tables.items()
        }
    tables = {name: weight.to(device) for name, weight in tables.items()}
    if projection is not None:
        projection = projection.to(device)
    report = benchmark_table_collection(
        tables,
        formats=_specs(args),
        projection_weight=projection,
        lookup_rows=args.lookup_rows,
        warmup_iterations=args.warmup_iterations,
        timed_iterations=args.timed_iterations,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
    )
    report["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
    report["checkpoint_owner"] = args.checkpoint_owner or "automatic-first-table-owner"
    report["sampled_table_rows"] = {
        name: int(weight.shape[0]) for name, weight in tables.items()
    }
    report["full_checkpoint_table_rows"] = original_rows
    report["full_tables_benchmarked"] = report["sampled_table_rows"] == original_rows
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + f".partial-{os.getpid()}")
        staging.write_text(encoded + "\n", encoding="utf-8")
        staging.replace(target)
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
