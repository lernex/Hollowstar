from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import load_yaml, repository_root
from .state import StateStore, utc_now


QUESTION_KEYS = ("question", "prompt", "problem", "input", "question_content", "query")
ANSWER_KEYS = ("answer", "solution", "canonical_solution", "target", "output", "response")


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_stringify(item)}" for key, item in value.items())
    return ""


def _first(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            value = _stringify(row[key]).strip()
            if value:
                return value
    return ""


def _normalized_benchmark_row(entry: dict[str, Any], config: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    question = _first(row, QUESTION_KEYS)
    answer = _first(row, ANSWER_KEYS)
    choices = row.get("choices") or row.get("options")
    if choices and answer.isdigit():
        choice_list = list(choices.values()) if isinstance(choices, dict) else list(choices)
        choice_index = int(answer)
        if 0 <= choice_index < len(choice_list):
            answer = _stringify(choice_list[choice_index])
    if not question:
        question = _stringify({key: value for key, value in row.items() if key not in ANSWER_KEYS})
    if not answer:
        answer = question or _stringify(row)
    return {
        "id": f"{entry['id']}:{config}:{index}",
        "text": answer,
        "metadata": {"query": question, "task": entry["id"], "evaluation_only": True},
    }


def _benchmark_rows(entry: dict[str, Any], cache_dir: Path):
    if entry.get("files"):
        for filename in entry["files"]:
            path = Path(
                hf_hub_download(
                    repo_id=entry["repo_id"],
                    filename=filename,
                    repo_type="dataset",
                    revision=entry["revision"],
                    cache_dir=cache_dir,
                )
            )
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if line.strip():
                        yield _normalized_benchmark_row(entry, filename, index, json.loads(line))
        return
    configs = entry.get("configs") or [entry.get("config")]
    for config in configs:
        dataset = load_dataset(
            entry["repo_id"],
            name=config,
            split=entry["split"],
            revision=entry["revision"],
            streaming=True,
        )
        for index, row in enumerate(dataset):
            yield _normalized_benchmark_row(entry, f"{config}:{entry['split']}", index, dict(row))


def prepare_holdouts(profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    root = Path(profile["storage"]["lustre_root"])
    output_dir = root / profile["storage"]["directories"]["contamination"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "holdouts.jsonl"
    cache_dir = root / profile["runtime"].get("hf_home", "cache/huggingface") / "eval-holdouts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "HOLDOUTS.json"
    if report_path.exists() and output.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    registry = load_yaml(repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml")
    counts: dict[str, int] = {}
    temporary = output.with_suffix(".jsonl.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        for benchmark in registry["benchmarks"]:
            count = 0
            for row in _benchmark_rows(benchmark, cache_dir):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            if count == 0:
                raise RuntimeError(f"Fail-closed: evaluation holdout {benchmark['id']} produced zero rows")
            counts[benchmark["id"]] = count
    temporary.replace(output)
    payload = {
        "schema": "metis.holdout-bundle/v1",
        "created_at": utc_now(),
        "output": str(output),
        "benchmarks": counts,
        "total_rows": sum(counts.values()),
        "training_use": "forbidden",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.complete("holdouts", "task-000000", payload)
    return payload
