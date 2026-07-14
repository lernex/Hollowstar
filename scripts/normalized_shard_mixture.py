from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from data_mixture import SourceSpec, build_planned_counts, iter_schedule, load_source_specs_from_config
from jsonl_artifacts import iter_jsonl_records


@dataclass
class LocalSourceState:
    spec: SourceSpec
    iterator: Iterator[dict[str, Any]]
    emitted_examples: int = 0

    def next_example(self) -> dict[str, str] | None:
        while True:
            try:
                row = next(self.iterator)
            except StopIteration:
                return None

            text = str(row.get("text", "")).strip()
            if not text:
                continue

            self.emitted_examples += 1
            return {
                "source": str(row.get("source", self.spec.name)),
                "doc_id": str(row.get("doc_id", f"{self.spec.name}:{self.emitted_examples}")),
                "text": text,
            }


class NormalizedShardMixture:
    def __init__(
        self,
        config_path: str | Path,
        normalized_root: str | Path,
        total_examples: int,
        *,
        seed: int | None = None,
        glob_pattern: str = "shard-*.jsonl.zst",
    ) -> None:
        self.config_path = str(Path(config_path))
        self.normalized_root = Path(normalized_root)
        self.raw_config = json.loads(Path(config_path).read_text())
        self.sources = load_source_specs_from_config(self.raw_config)
        self.total_examples = int(total_examples)
        self.seed = int(self.raw_config.get("seed", 42) if seed is None else seed)
        self.glob_pattern = glob_pattern
        self.planned_counts = build_planned_counts(
            [spec.weight for spec in self.sources],
            self.total_examples,
        )

        self.states: list[LocalSourceState] = []
        for spec in self.sources:
            source_dir = self.normalized_root / spec.name
            iterator = iter_jsonl_records(
                jsonl_dir=source_dir,
                glob_pattern=self.glob_pattern,
            )
            self.states.append(LocalSourceState(spec=spec, iterator=iterator))
        self.dead_sources: set[int] = set()

    def same_bucket_indices(self, source_index: int) -> list[int]:
        bucket = self.sources[source_index].bucket
        if not bucket:
            return [source_index]
        return [
            index
            for index, spec in enumerate(self.sources)
            if spec.bucket == bucket
        ]

    def __iter__(self):
        for source_index in iter_schedule(self.planned_counts, self.seed):
            for candidate_index in self.same_bucket_indices(source_index):
                if candidate_index in self.dead_sources:
                    continue
                example = self.states[candidate_index].next_example()
                if example is None:
                    self.dead_sources.add(candidate_index)
                    continue
                yield example
                break

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "normalized_root": str(self.normalized_root),
            "seed": self.seed,
            "total_examples_requested": self.total_examples,
            "planned_counts": {
                self.sources[index].name: count for index, count in sorted(self.planned_counts.items())
            },
            "sources": [spec.to_dict() for spec in self.sources],
            "source_stats": {
                state.spec.name: {
                    "emitted_examples": state.emitted_examples,
                    "source_dir": str(self.normalized_root / state.spec.name),
                    "exhausted": index in self.dead_sources,
                }
                for index, state in enumerate(self.states)
            },
        }
