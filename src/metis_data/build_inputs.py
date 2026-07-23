from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .download import sha256_file
from .state import StateStore, utc_now


INDEX_ROLES = {"source_index", "metadata_index", "retrieval_index"}


def _stable_id(record: dict[str, Any]) -> str:
    value = "\0".join(
        str(record.get(key, ""))
        for key in ("source_id", "kind", "local_path", "sha256", "revision", "repo_path")
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expanded_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("payload_role") in INDEX_ROLES:
        return []
    if record.get("kind") == "remote_source_plan" or record.get("materialized") is False:
        raise RuntimeError(f"Unresolved acquisition plan cannot become a build input: {record.get('source_id')}")
    if record.get("kind") == "materialized_dataset":
        output: list[dict[str, Any]] = []
        for shard in record.get("shards", []):
            output.append(
                {
                    "kind": "materialized_jsonl",
                    "source_id": record["source_id"],
                    "driver": record.get("driver"),
                    "revision": record.get("revision"),
                    "repo_path": Path(shard["path"]).name,
                    "local_path": shard["path"],
                    "size": shard["size"],
                    "sha256": shard["sha256"],
                    "payload_role": "training_records",
                }
            )
        return output
    if record.get("local_path"):
        return [{**record, "payload_role": record.get("payload_role", "training_records")}]
    return []


def prepare_build_inputs(profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    """Freeze one independently normalizable task per materialized training file."""

    root = Path(profile["storage"]["lustre_root"]).resolve()
    lock = state.read("sources.lock.json")
    if not lock:
        raise RuntimeError("sources.lock.json is missing")
    handoff = state.read("ACQUISITION_READY.json", default={})
    recorded_root = Path(str(handoff.get("lustre_root", root))).expanduser().resolve()
    inputs: list[dict[str, Any]] = []
    for task in lock.get("download_tasks", []):
        completion = state.read("completed", "download", f"{task['task_id']}.json")
        if not completion:
            raise RuntimeError(f"Acquisition task is incomplete: {task['task_id']}")
        for output in completion.get("files", []):
            for record in _expanded_records(output):
                raw_path = Path(str(record["local_path"])).expanduser()
                if raw_path.is_absolute():
                    path = raw_path.resolve()
                    try:
                        path.relative_to(root)
                    except ValueError:
                        try:
                            path = (root / path.relative_to(recorded_root)).resolve()
                        except ValueError:
                            pass
                else:
                    path = (root / raw_path).resolve()
                try:
                    relative = path.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(f"Build input is outside the configured Lustre root: {path}") from exc
                if not path.is_file():
                    raise RuntimeError(f"Build input is missing: {path}")
                size = path.stat().st_size
                if record.get("size") is not None and size != int(record["size"]):
                    raise RuntimeError(f"Build input size changed: {path}")
                digest = str(record.get("sha256") or sha256_file(path))
                canonical = {
                    **record,
                    "local_path": str(path),
                    "relative_path": str(relative),
                    "size": size,
                    "sha256": digest,
                }
                canonical["input_id"] = _stable_id(canonical)
                inputs.append(canonical)
    inputs.sort(key=lambda row: (str(row.get("source_id")), str(row.get("relative_path")), row["input_id"]))
    if not inputs:
        raise RuntimeError("Acquisition contains no training-record files")
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for record in inputs:
        local_path = str(record["local_path"])
        input_id = str(record["input_id"])
        if local_path in seen_paths:
            raise RuntimeError(f"Acquisition contains a duplicate training-record path: {local_path}")
        if input_id in seen_ids:
            raise RuntimeError(f"Acquisition contains a duplicate training input ID: {input_id}")
        seen_paths.add(local_path)
        seen_ids.add(input_id)
    by_source: dict[str, int] = {}
    for record in inputs:
        source_id = str(record.get("source_id") or "")
        if not source_id:
            raise RuntimeError(f"Build input has no source_id: {record['relative_path']}")
        by_source[source_id] = by_source.get(source_id, 0) + 1
    expected_sources = {
        str(source["id"])
        for source in lock.get("sources", [])
        if isinstance(source, dict) and source.get("id")
    }
    missing_sources = sorted(expected_sources - set(by_source))
    if missing_sources:
        raise RuntimeError(
            "Acquisition has no materialized training-record payload for source(s): "
            + ", ".join(missing_sources)
        )
    payload = {
        "schema": "metis.build-inputs/v1",
        "created_at": utc_now(),
        "release": lock.get("release"),
        "inputs": inputs,
        "input_count": len(inputs),
        "input_bytes": sum(int(item["size"]) for item in inputs),
        "files_by_source": by_source,
        "expected_sources": sorted(expected_sources),
    }
    existing = state.read("build.inputs.json")
    if existing:
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
        comparable_new = {key: value for key, value in payload.items() if key != "created_at"}
        if comparable_existing != comparable_new:
            raise RuntimeError(
                "Frozen build.inputs.json differs from current acquisition; create a new data release or review the drift"
            )
        return existing
    state.write("build.inputs.json", payload=payload)
    return payload


def build_input_count(state: StateStore) -> int:
    payload = state.read("build.inputs.json")
    if not payload:
        raise RuntimeError("build.inputs.json is missing; freeze the acquisition handoff before submitting Rhea")
    count = int(payload.get("input_count", len(payload.get("inputs", []))))
    if count <= 0:
        raise RuntimeError("build.inputs.json contains no inputs")
    return count
