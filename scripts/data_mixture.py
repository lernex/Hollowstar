from __future__ import annotations

import gzip
import io
import json
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

            stripped = text.strip()
            if len(stripped) < self.spec.min_chars:
                self.skipped_examples += 1
                continue

            if self.spec.max_chars is not None:
                stripped = stripped[: self.spec.max_chars].strip()
                if not stripped:
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
        self.schedule = build_schedule([spec.weight for spec in self.sources], self.total_examples, self.seed)
        self.fetcher = SoftwareHeritageFetcher()
        self.states = [SourceState(spec=spec, iterator=iter(load_source_dataset(spec))) for spec in self.sources]
        self.planned_counts = dict(Counter(self.schedule))

    def __iter__(self):
        for source_index in self.schedule:
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
    )


def build_schedule(weights: list[float], total_examples: int, seed: int) -> list[int]:
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

    schedule: list[int] = []
    for index, count in enumerate(counts):
        schedule.extend([index] * count)

    rng = Random(seed)
    rng.shuffle(schedule)
    return schedule


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
