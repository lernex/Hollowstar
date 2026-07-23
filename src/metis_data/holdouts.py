from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import load_yaml, repository_root
from .dedup import canonical_text
from .state import StateStore, utc_now


HOLDOUT_BUNDLE_SCHEMA = "metis.holdout-bundle/v4"
HOLDOUT_EXTRACTOR_VERSION = "metis-benchmark-fragments-2026-07-22-v3"


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


FRAGMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "query": (
        "question", "prompt", "problem", "input", "question_content", "query", "instruction",
        "turns", "messages", "description", "title",
    ),
    "context": (
        "context", "passage", "article", "story", "documents", "supporting_facts", "paragraphs",
        "background", "scenario", "fact1", "fact2", "fact3", "facts", "table", "pre_text", "post_text",
    ),
    "choices": (
        "choices", "options", "endings", "candidates", "incorrect_answers", "distractors",
    ),
    "answer": (
        "answer", "answers", "solution", "canonical_solution", "target", "output", "response",
        "reference_answer", "ideal", "label_text", "rationale", "explanation",
    ),
    "code": (
        "code", "starter_code", "completion", "patch", "diff", "test", "tests", "test_list",
        "public_tests", "private_tests", "entry_point", "setup_code", "test_code",
    ),
}


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        value = value.strip()
        if value:
            yield value
    elif isinstance(value, (int, float, bool)):
        yield str(value)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)):
                text = str(item).strip()
                if text:
                    yield f"{key}: {text}"
            else:
                yield from _strings(item)


def _benchmark_fragments(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    seen: set[str] = set()
    used_fields: set[str] = set()
    for kind, fields in FRAGMENT_FIELDS.items():
        for field in fields:
            if field not in row:
                continue
            used_fields.add(field)
            for text in _strings(row[field]):
                normalized = canonical_text(text)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    yield kind, text

    # Unknown benchmark schemas fail safely toward broader exclusion. Binary
    # image/audio payloads and opaque numeric metadata are deliberately ignored.
    for field, value in row.items():
        if field in used_fields or field.lower() in {
            "id", "idx", "index", "image", "images", "audio", "video", "label", "score", "metadata"
        }:
            continue
        for text in _strings(value):
            normalized = canonical_text(text)
            if len(normalized) >= 24 and normalized not in seen:
                seen.add(normalized)
                yield "other", text


def _normalized_benchmark_rows(
    entry: dict[str, Any], job: str, index: int, row: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    row_id = row.get("id", row.get("idx", row.get("index", index)))
    holdout_row_id = hashlib.sha256(
        json.dumps(
            [entry["id"], job, index, str(row_id)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    for fragment_index, (kind, text) in enumerate(_benchmark_fragments(row)):
        yield {
            "id": f"{entry['id']}:{job}:{row_id}:{fragment_index}",
            "text": text,
            "metadata": {
                "task": entry["id"],
                "job": job,
                "source_row": str(row_id),
                "holdout_row_id": holdout_row_id,
                "fragment_kind": kind,
                "evaluation_only": True,
            },
        }


def _records_from_bytes(name: str, payload: bytes) -> Iterator[dict[str, Any]]:
    lower = name.lower()
    if lower.endswith(".jsonl"):
        for line in payload.decode("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        return
    if lower.endswith(".json"):
        value = json.loads(payload.decode("utf-8"))
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows = value.get("data", value.get("rows", value))
            if isinstance(rows, list):
                yield from (row for row in rows if isinstance(row, dict))
            else:
                yield value
        return
    if lower.endswith(".csv"):
        yield from csv.DictReader(io.StringIO(payload.decode("utf-8")))
        return
    if lower.endswith(".parquet"):
        table = pq.read_table(io.BytesIO(payload))
        yield from table.to_pylist()


def _records_from_file(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if member.lower().endswith((".jsonl", ".json", ".csv", ".parquet")):
                    yield from _records_from_bytes(member, archive.read(member))
        return
    yield from _records_from_bytes(path.name, path.read_bytes())


def _benchmark_jobs(entry: dict[str, Any]) -> Iterator[tuple[str | None, str]]:
    if entry.get("jobs"):
        for job in entry["jobs"]:
            yield job.get("config"), str(job["split"])
        return
    configs = entry.get("configs") or [entry.get("config")]
    splits = entry.get("splits") or [entry.get("split")]
    for config in configs:
        for split in splits:
            if split:
                yield config, str(split)


def _benchmark_job_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if entry.get("files"):
        return [
            {"job": str(filename), "filename": str(filename)}
            for filename in entry["files"]
        ]
    return [
        {
            "job": f"{config or 'default'}:{split}",
            "config": config,
            "split": split,
        }
        for config, split in _benchmark_jobs(entry)
    ]


def _benchmark_source_rows(
    entry: dict[str, Any],
    job: dict[str, Any],
    cache_dir: Path,
) -> Iterator[tuple[int, dict[str, Any]]]:
    filename = job.get("filename")
    if filename:
        path = Path(
            hf_hub_download(
                repo_id=entry["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=entry["revision"],
                cache_dir=cache_dir,
            )
        )
        for index, row in enumerate(_records_from_file(path)):
            yield index, row
        return
    config = job.get("config")
    split = str(job["split"])
    dataset = load_dataset(
        entry["repo_id"],
        name=config,
        split=split,
        revision=entry["revision"],
        streaming=True,
        cache_dir=str(cache_dir),
    )
    for index, row in enumerate(dataset):
        yield index, dict(row)


def _benchmark_rows(entry: dict[str, Any], cache_dir: Path) -> Iterator[dict[str, Any]]:
    for job in _benchmark_job_specs(entry):
        job_name = str(job["job"])
        for index, row in _benchmark_source_rows(entry, job, cache_dir):
            yield from _normalized_benchmark_rows(entry, job_name, index, row)


def prepare_holdouts(profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    root = Path(profile["storage"]["lustre_root"])
    output_dir = root / profile["storage"]["directories"]["contamination"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "holdouts.jsonl"
    cache_dir = root / profile["runtime"].get("hf_home", "cache/huggingface") / "eval-holdouts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "HOLDOUTS.json"
    registry_path = repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
    registry_sha256 = _sha256_file(registry_path)
    registry = load_yaml(registry_path)
    policy = registry.get("policy", {})
    family_labels = sorted(
        {str(benchmark["family"]) for benchmark in registry["benchmarks"]}
    )
    expected_jobs = sum(
        len(_benchmark_job_specs(benchmark))
        for benchmark in registry["benchmarks"]
    )
    expected_contract = {
        "benchmark registries": (
            policy.get("expected_benchmark_registries"),
            len(registry["benchmarks"]),
        ),
        "family labels": (
            policy.get("expected_family_labels"),
            len(family_labels),
        ),
        "expanded jobs": (policy.get("expected_jobs"), expected_jobs),
    }
    for label, (expected, actual) in expected_contract.items():
        if expected is not None and int(expected) != actual:
            raise RuntimeError(
                f"Fail-closed: holdout registry declares {expected} {label}, found {actual}"
            )
    if report_path.exists() and output.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        existing_jobs = existing.get("jobs")
        if (
            existing.get("schema") == HOLDOUT_BUNDLE_SCHEMA
            and existing.get("extractor_version") == HOLDOUT_EXTRACTOR_VERSION
            and existing.get("registry_sha256") == registry_sha256
            and int(existing.get("benchmark_registry_count", -1))
            == len(registry["benchmarks"])
            and int(existing.get("family_label_count", -1)) == len(family_labels)
            and int(existing.get("job_count", -1)) == expected_jobs
            and isinstance(existing_jobs, list)
            and len(existing_jobs) == expected_jobs
            and all(
                int(job.get("source_rows", 0)) > 0
                and int(job.get("fragments", 0)) > 0
                for job in existing_jobs
                if isinstance(job, dict)
            )
            and all(isinstance(job, dict) for job in existing_jobs)
            and int(existing.get("output_size", -1)) == output.stat().st_size
            and existing.get("output_sha256") == _sha256_file(output)
        ):
            return existing
        raise RuntimeError(
            "Existing evaluation holdout bundle does not match the pinned benchmark "
            "registry/extractor/job inventory; preserve it for audit and build a new data release"
        )
    if report_path.exists() != output.exists():
        raise RuntimeError(
            "Evaluation holdout bundle/report is incomplete; preserve the state and resume a new release"
        )
    counts: dict[str, int] = {}
    job_reports: list[dict[str, Any]] = []
    temporary = output.with_suffix(".jsonl.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        for benchmark in registry["benchmarks"]:
            benchmark_fragments = 0
            for job in _benchmark_job_specs(benchmark):
                job_name = str(job["job"])
                source_rows = 0
                fragments = 0
                for index, row in _benchmark_source_rows(benchmark, job, cache_dir):
                    source_rows += 1
                    for normalized in _normalized_benchmark_rows(
                        benchmark, job_name, index, row
                    ):
                        handle.write(
                            json.dumps(
                                normalized, ensure_ascii=False, sort_keys=True
                            )
                            + "\n"
                        )
                        fragments += 1
                if source_rows == 0:
                    raise RuntimeError(
                        "Fail-closed: evaluation holdout job "
                        f"{benchmark['id']}:{job_name} produced zero source rows"
                    )
                if fragments == 0:
                    raise RuntimeError(
                        "Fail-closed: evaluation holdout job "
                        f"{benchmark['id']}:{job_name} produced zero indexable fragments"
                    )
                benchmark_fragments += fragments
                job_reports.append(
                    {
                        "benchmark": benchmark["id"],
                        "family": benchmark["family"],
                        "job": job_name,
                        "source_rows": source_rows,
                        "fragments": fragments,
                    }
                )
            if benchmark_fragments == 0:
                raise RuntimeError(
                    f"Fail-closed: evaluation holdout {benchmark['id']} has no configured jobs"
                )
            counts[benchmark["id"]] = benchmark_fragments
    if len(job_reports) != expected_jobs:
        raise RuntimeError(
            f"Fail-closed: prepared {len(job_reports)} holdout jobs, expected {expected_jobs}"
        )
    temporary.replace(output)
    payload = {
        "schema": HOLDOUT_BUNDLE_SCHEMA,
        "extractor_version": HOLDOUT_EXTRACTOR_VERSION,
        "created_at": utc_now(),
        "output": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha256,
        "benchmarks": counts,
        "benchmark_registry_count": len(registry["benchmarks"]),
        "family_label_count": len(family_labels),
        "family_labels": family_labels,
        "job_count": len(job_reports),
        "jobs": job_reports,
        "total_fragments": sum(counts.values()),
        "total_source_rows": sum(int(job["source_rows"]) for job in job_reports),
        "training_use": "forbidden",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.complete("holdouts", "task-000000", payload)
    return payload
