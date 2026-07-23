from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from metis_data.decontaminate import ContaminationIndex


POLICY_FIELDS = (
    "ngram_size",
    "minimum_matching_ngrams",
    "short_ngram_size",
    "minimum_short_matching_ngrams",
    "code_ngram_size",
    "minimum_code_matching_ngrams",
    "code_skeleton_ngram_size",
    "minimum_code_skeleton_matching_ngrams",
    "maximum_shingle_rows",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_contamination_inputs(
    directory: Path,
    index: ContaminationIndex,
    holdouts: Iterable[str | dict[str, Any]],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, value in enumerate(holdouts):
        if isinstance(value, str):
            records.append(
                {
                    "id": f"fragment-{position}",
                    "text": value,
                    "metadata": {"holdout_row_id": f"row-{position}"},
                }
            )
        else:
            records.append(value)
    holdout_path = directory / "holdouts.jsonl"
    holdout_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    policy = {
        "fail_closed_if_unavailable": True,
        "semantic_dedup": False,
        **{field: int(getattr(index, field)) for field in POLICY_FIELDS},
    }
    registry = {
        "schema": "metis.contamination-registry/v2",
        "policy": policy,
        "benchmarks": [
            {
                "id": "fixture",
                "family": "test",
                "repo_id": "test/fixture",
                "revision": "0" * 40,
                "config": "default",
                "split": "test",
                "use": "evaluation_only",
            }
        ],
    }
    registry_path = directory / "eval-holdouts.yaml"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "schema": "metis.holdout-bundle/v4",
        "extractor_version": "test-fixture/v1",
        "output": str(holdout_path),
        "output_size": holdout_path.stat().st_size,
        "output_sha256": _sha256(holdout_path),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "benchmark_registry_count": 1,
        "job_count": 1,
        "jobs": [
            {
                "benchmark": "fixture",
                "family": "test",
                "job": "default:test",
                "source_rows": max(1, len(records)),
                "fragments": max(1, len(records)),
            }
        ],
        "total_fragments": max(1, len(records)),
        "training_use": "forbidden",
    }
    (directory / "HOLDOUTS.json").write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry_path
