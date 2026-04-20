from __future__ import annotations

import gzip
import io
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import requests
from datasets import load_dataset


SOFTWARE_HERITAGE_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
USER_AGENT = "metis-data-pipeline/1.1"
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTIBLANK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    streaming: bool = True
    weight: float = 1.0
    text_column: str = "text"
    id_column: str | None = None
    loader: str = "text"
    blob_id_column: str = "blob_id"
    min_chars: int = 0
    max_chars: int | None = None
    min_alpha_ratio: float | None = None
    max_repeat_char_run: int | None = 48
    max_line_length: int | None = None
    max_url_count: int | None = None
    normalize_whitespace: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=raw["name"],
            dataset_name=raw["dataset_name"],
            dataset_config=raw.get("dataset_config"),
            split=raw.get("split", "train"),
            streaming=bool(raw.get("streaming", True)),
            weight=float(raw.get("weight", 1.0)),
            text_column=raw.get("text_column", "text"),
            id_column=raw.get("id_column"),
            loader=raw.get("loader", "text"),
            blob_id_column=raw.get("blob_id_column", "blob_id"),
            min_chars=int(raw.get("min_chars", 0)),
            max_chars=raw.get("max_chars"),
            min_alpha_ratio=raw.get("min_alpha_ratio"),
            max_repeat_char_run=raw.get("max_repeat_char_run", 48),
            max_line_length=raw.get("max_line_length"),
            max_url_count=raw.get("max_url_count"),
            normalize_whitespace=bool(raw.get("normalize_whitespace", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "streaming": self.streaming,
            "weight": self.weight,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "loader": self.loader,
            "blob_id_column": self.blob_id_column,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_alpha_ratio": self.min_alpha_ratio,
            "max_repeat_char_run": self.max_repeat_char_run,
            "max_line_length": self.max_line_length,
            "max_url_count": self.max_url_count,
            "normalize_whitespace": self.normalize_whitespace,
        }


@dataclass
class SourceState:
    spec: SourceSpec
    iterator: Any
    attempted_rows: int = 0
    emitted_examples: int = 0
    skipped_examples: int = 0
    fetch_errors: int = 0

    def next_example(self, fetcher: "SoftwareHeritageFetcher") -> dict[str, str] | None:
        while True:
            try:
                row = next(self.iterator)
            except StopIteration:
                return None

            self.attempted_rows += 1
            try:
                text = extract_text(row, self.spec, fetcher)
            except Exception:
                self.fetch_errors += 1
                continue

            if not text:
                self.skipped_examples += 1
                continue

            stripped = clean_text(text, self.spec)
            if len(stripped) < self.spec.min_chars:
                self.skipped_examples += 1
                continue

            if self.spec.max_chars is not None:
                stripped = stripped[: self.spec.max_chars].strip()
                if not stripped:
                    self.skipped_examples += 1
                    continue

            if not passes_quality_filters(stripped, self.spec):
                self.skipped_examples += 1
                continue

            self.emitted_examples += 1
            return {
                "source": self.spec.name,
                "doc_id": extract_doc_id(row, self.spec, self.attempted_rows),
                "text": stripped,
            }


class SoftwareHeritageFetcher:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, blob_id: str) -> str:
        last_error: Exception | None = None
        url = SOFTWARE_HERITAGE_URL.format(blob_id=blob_id)
        for attempt in range(5):
            try:
                response = self.session.get(url, timeout=45)
                response.raise_for_status()
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_handle:
                    return gz_handle.read().decode("utf-8", errors="ignore")
            except (OSError, requests.RequestException) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Failed to fetch Software Heritage blob {blob_id}") from last_error


class DatasetMixture:
    def __init__(self, config_path: str | Path, total_examples: int, seed: int | None = None) -> None:
        self.config_path = str(Path(config_path))
        self.raw_config = json.loads(Path(config_path).read_text())
        raw_sources = self.raw_config.get("sources")
        if not raw_sources:
            raise ValueError("Mixture config must include a non-empty `sources` list.")

        self.sources = [SourceSpec.from_dict(item) for item in raw_sources]
        if any(spec.weight <= 0 for spec in self.sources):
            raise ValueError("All source weights must be positive.")

        self.total_examples = int(total_examples)
        self.seed = int(self.raw_config.get("seed", 42) if seed is None else seed)
        self.planned_counts = build_planned_counts(
            [spec.weight for spec in self.sources],
            self.total_examples,
        )
        self.fetcher = SoftwareHeritageFetcher()
        self.states = [SourceState(spec=spec, iterator=iter(load_source_dataset(spec))) for spec in self.sources]

    def __iter__(self):
        for source_index in iter_schedule(self.planned_counts, self.seed):
            example = self.states[source_index].next_example(self.fetcher)
            if example is None:
                continue
            yield example

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "seed": self.seed,
            "total_examples_requested": self.total_examples,
            "planned_counts": {
                self.sources[index].name: count for index, count in sorted(self.planned_counts.items())
            },
            "sources": [spec.to_dict() for spec in self.sources],
            "source_stats": {
                state.spec.name: {
                    "attempted_rows": state.attempted_rows,
                    "emitted_examples": state.emitted_examples,
                    "skipped_examples": state.skipped_examples,
                    "fetch_errors": state.fetch_errors,
                }
                for state in self.states
            },
        }


def load_source_dataset(spec: SourceSpec):
        return load_dataset(
            spec.dataset_name,
            name=spec.dataset_config,
            split=spec.split,
            streaming=spec.streaming,
            trust_remote_code=True,
        )


def build_planned_counts(weights: list[float], total_examples: int) -> dict[int, int]:
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights]
    raw_counts = [weight * total_examples for weight in normalized]
    counts = [int(count) for count in raw_counts]
    remainder = total_examples - sum(counts)
    ranked = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1

    return {index: count for index, count in enumerate(counts)}


def clean_text(text: str, spec: SourceSpec) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    if spec.normalize_whitespace:
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = MULTISPACE_RE.sub(" ", text)
        text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    alpha = sum(char.isalpha() for char in meaningful)
    return alpha / len(meaningful)


def max_repeat_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    for previous, current_char in zip(text, text[1:]):
        if current_char == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def url_count(text: str) -> int:
    return text.count("http://") + text.count("https://") + text.count("www.")


def passes_quality_filters(text: str, spec: SourceSpec) -> bool:
    if spec.min_alpha_ratio is not None and alpha_ratio(text) < float(spec.min_alpha_ratio):
        return False
    if spec.max_repeat_char_run is not None and max_repeat_char_run(text) > int(spec.max_repeat_char_run):
        return False
    if spec.max_url_count is not None and url_count(text) > int(spec.max_url_count):
        return False
    if spec.max_line_length is not None:
        if any(len(line) > int(spec.max_line_length) for line in text.splitlines()):
            return False
    return True


def iter_schedule(planned_counts: dict[int, int], seed: int):
    remaining = dict(planned_counts)
    total_remaining = sum(remaining.values())
    rng = Random(seed)
    source_indices = sorted(remaining)
    while total_remaining > 0:
        pick = rng.randrange(total_remaining)
        cumulative = 0
        for index in source_indices:
            count = remaining[index]
            if count <= 0:
                continue
            cumulative += count
            if pick < cumulative:
                remaining[index] -= 1
                total_remaining -= 1
                yield index
                break


def extract_doc_id(row: dict[str, Any], spec: SourceSpec, fallback_index: int) -> str:
    if spec.id_column and row.get(spec.id_column) is not None:
        return str(row[spec.id_column])
    if spec.loader == "software_heritage_blob" and row.get(spec.blob_id_column):
        return str(row[spec.blob_id_column])
    return f"{spec.name}:{fallback_index}"


def extract_text(row: dict[str, Any], spec: SourceSpec, fetcher: SoftwareHeritageFetcher) -> str:
    if spec.loader == "text":
        return str(row.get(spec.text_column, ""))
    if spec.loader == "software_heritage_blob":
        blob_id = row.get(spec.blob_id_column)
        if not blob_id:
            return ""
        return fetcher.fetch(str(blob_id))
    raise ValueError(f"Unsupported source loader: {spec.loader}")
