#!/usr/bin/env python3
"""Measure tokenizer_train cost before committing the full-size job.

`tokenizer_train` is the longest single-node stage in the Metis-1.6 CPU build
and the only one that cannot be made parallel: BPE learns its merges strictly
one after another. Its runtime and, more importantly, its peak memory are not
predictable from the input size, because the trainer holds a table of *distinct*
word types plus pair bookkeeping rather than the corpus itself. A code- and
math-heavy mixture produces far more distinct types per gigabyte than plain
English does.

This script trains the real production tokenizer contract (byte-level BPE,
65,536 entries, the manifest's special tokens) on a ladder of subsample sizes,
records wall time and peak RSS for each, fits a power law, and extrapolates to
the manifest's full `sample_target_bytes`. It answers two questions before a
multi-hour job is submitted:

  1. roughly how long will tokenizer_train take, and
  2. will it fit in node memory, or die near the end and have to start over.

Each size runs in a fresh subprocess so peak RSS is measured per size rather
than accumulated. Nothing is written into the build state; this is a preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_data.config import load_profile, repository_root  # noqa: E402
from metis_data.manifest import validate_manifest  # noqa: E402


DEFAULT_LADDER_GB = (1.0, 2.0, 5.0, 10.0)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, the BSDs and macOS report bytes.
    return int(peak) if platform.system() == "Darwin" else int(peak) * 1024


def _corpus_files(profile: dict[str, Any]) -> tuple[str, list[Path]]:
    """Prefer the built sample shards, fall back to the eligible corpus."""

    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    parts = root / directories["tokenizer"] / "sample-parts"
    if parts.is_dir():
        files = sorted(path for path in parts.glob("task-*.jsonl.zst") if path.is_file())
        if files:
            return "sample-parts", files
    eligible = root / directories["eligible"] / "final"
    files = sorted(
        path
        for path in eligible.glob("**/*.jsonl*")
        if path.is_file() and not path.name.endswith(".incomplete")
    )
    if not files:
        raise SystemExit(
            f"No tokenizer input found under {parts} or {eligible}. Run the sample "
            "stages first, or point --profile at a build that has them."
        )
    return "eligible/final", files


def _iter_texts(files: list[Path], budget_bytes: int) -> Iterator[str]:
    """Yield documents until the UTF-8 budget is met, striding across shards.

    Striding rather than reading shard 0 to exhaustion keeps the subsample
    representative of the full source mixture, which is what determines the
    distinct-type count the trainer has to hold.
    """

    from metis_data.stage_runner import _iter_rows

    consumed = 0
    exhausted = set()
    handles = {index: _iter_rows(path) for index, path in enumerate(files)}
    while consumed < budget_bytes and len(exhausted) < len(files):
        for index in range(len(files)):
            if index in exhausted:
                continue
            try:
                row = next(handles[index])
            except StopIteration:
                exhausted.add(index)
                continue
            text = str(row.get("text", ""))
            if not text:
                continue
            consumed += len(text.encode("utf-8"))
            yield text
            if consumed >= budget_bytes:
                return


def _train_once(profile_name: str, budget_bytes: int, vocabulary: int) -> dict[str, Any]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    _path, profile = load_profile(profile_name)
    manifest_path = Path(profile["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repository_root() / manifest_path
    manifest = validate_manifest(manifest_path).require_valid()
    special_tokens = list(manifest["tokenizer"]["special_tokens"])
    _label, files = _corpus_files(profile)

    # Time a pure read of the same budget first. The trainer pulls its corpus
    # through a Python generator holding the GIL, so this separates "cost of
    # feeding the trainer" from "cost of the sequential merge loop".
    read_started = time.monotonic()
    read_bytes = 0
    documents = 0
    for text in _iter_texts(files, budget_bytes):
        read_bytes += len(text.encode("utf-8"))
        documents += 1
    read_seconds = time.monotonic() - read_started

    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    train_started = time.monotonic()
    tokenizer.train_from_iterator(_iter_texts(files, budget_bytes), trainer=trainer, length=None)
    train_seconds = time.monotonic() - train_started

    return {
        "requested_bytes": budget_bytes,
        "corpus_bytes": read_bytes,
        "documents": documents,
        "vocabulary_size": len(tokenizer.get_vocab()),
        "read_seconds": round(read_seconds, 2),
        "train_seconds": round(train_seconds, 2),
        "merge_seconds": round(max(0.0, train_seconds - read_seconds), 2),
        "peak_rss_bytes": _peak_rss_bytes(),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "<unset>"),
    }


def _power_law(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares fit of y = a * x**b in log space."""

    usable = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(x) for x, _ in usable]
    ys = [math.log(y) for _, y in usable]
    n = len(usable)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return math.exp(mean_y - slope * mean_x), slope


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000 or unit == "TB":
            return f"{value:,.1f}{unit}"
        value /= 1000
    return f"{value:,.1f}TB"


def _human_seconds(value: float) -> str:
    if value < 90:
        return f"{value:,.0f}s"
    if value < 5400:
        return f"{value / 60:,.1f}m"
    return f"{value / 3600:,.1f}h"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="portage-cpu")
    parser.add_argument(
        "--sizes-gb",
        default=",".join(str(size) for size in DEFAULT_LADDER_GB),
        help="Comma-separated subsample sizes in GB",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=480.0,
        help="Usable node memory the full job must fit inside",
    )
    parser.add_argument("--vocabulary", type=int, default=0, help="Defaults to the manifest value")
    parser.add_argument("--report", default="", help="Where to write the JSON report")
    parser.add_argument("--single-size-bytes", type=int, default=0, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    _path, profile = load_profile(arguments.profile)
    manifest_path = Path(profile["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repository_root() / manifest_path
    manifest = validate_manifest(manifest_path).require_valid()
    vocabulary = arguments.vocabulary or int(
        manifest["tokenizer"]["vocabulary_size_including_special_tokens"]
    )
    full_target = int(manifest["tokenizer"]["sample_target_bytes"])

    if arguments.single_size_bytes:
        # Child mode: one size, fresh address space, peak RSS is this size alone.
        print(json.dumps(_train_once(arguments.profile, arguments.single_size_bytes, vocabulary)))
        return 0

    sizes = [int(float(value) * 1_000_000_000) for value in arguments.sizes_gb.split(",") if value.strip()]
    if len(sizes) < 2:
        raise SystemExit("Give at least two sizes so the extrapolation has a slope")
    label, files = _corpus_files(profile)
    print(f"tokenizer_train preflight  profile={arguments.profile}  source={label} ({len(files):,} files)")
    print(f"vocabulary={vocabulary:,}  full sample_target_bytes={_human_bytes(full_target)}")
    print(f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM', '<unset>')}\n")

    measurements: list[dict[str, Any]] = []
    print(f"{'corpus':>10}{'docs':>12}{'read':>9}{'train':>9}{'merge':>9}{'peak RSS':>12}")
    print("-" * 61)
    for size in sizes:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--profile", arguments.profile,
             "--vocabulary", str(vocabulary),
             "--single-size-bytes", str(size)],
            check=True, capture_output=True, text=True,
        )
        measured = json.loads(result.stdout.strip().splitlines()[-1])
        measurements.append(measured)
        print(
            f"{_human_bytes(measured['corpus_bytes']):>10}"
            f"{measured['documents']:>12,}"
            f"{_human_seconds(measured['read_seconds']):>9}"
            f"{_human_seconds(measured['train_seconds']):>9}"
            f"{_human_seconds(measured['merge_seconds']):>9}"
            f"{_human_bytes(measured['peak_rss_bytes']):>12}"
        )
        if measured["corpus_bytes"] < size * 0.95:
            print(f"  note: source exhausted at {_human_bytes(measured['corpus_bytes'])}; "
                  "larger points are not independent")
            break

    print()
    report: dict[str, Any] = {
        "schema": "metis.tokenizer-train-preflight/v1",
        "profile": arguments.profile,
        "vocabulary_size": vocabulary,
        "full_target_bytes": full_target,
        "memory_budget_bytes": int(arguments.memory_budget_gb * 1_000_000_000),
        "measurements": measurements,
    }

    memory_fit = _power_law([(m["corpus_bytes"], m["peak_rss_bytes"]) for m in measurements])
    time_fit = _power_law([(m["corpus_bytes"], m["train_seconds"]) for m in measurements])
    budget_bytes = int(arguments.memory_budget_gb * 1_000_000_000)

    if memory_fit:
        scale, exponent = memory_fit
        predicted_memory = scale * full_target**exponent
        report["memory_exponent"] = round(exponent, 3)
        report["predicted_peak_rss_bytes"] = int(predicted_memory)
        verdict = "FITS" if predicted_memory < budget_bytes else "WILL NOT FIT"
        report["memory_verdict"] = verdict
        print(f"peak RSS scales as bytes^{exponent:.2f}")
        print(f"  predicted at {_human_bytes(full_target)}: {_human_bytes(predicted_memory)} "
              f"vs {_human_bytes(budget_bytes)} budget  ->  {verdict}")
        if predicted_memory >= budget_bytes:
            affordable = (budget_bytes / scale) ** (1 / exponent) if exponent > 0 else 0
            report["largest_affordable_bytes"] = int(affordable)
            print(f"  largest sample that fits the budget: ~{_human_bytes(affordable)}")
    if time_fit:
        scale, exponent = time_fit
        predicted_seconds = scale * full_target**exponent
        report["time_exponent"] = round(exponent, 3)
        report["predicted_train_seconds"] = int(predicted_seconds)
        print(f"train time scales as bytes^{exponent:.2f}")
        print(f"  predicted at {_human_bytes(full_target)}: {_human_seconds(predicted_seconds)}")

    print("\nExtrapolating several orders of magnitude from small samples is indicative, "
          "not exact.\nTreat a near-budget memory prediction as a failure.")

    if arguments.report:
        destination = Path(arguments.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nreport: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
