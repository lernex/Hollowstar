from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .download import sha256_file
from .state import StateStore, utc_now


INDEX_ROLES = {"source_index", "metadata_index", "retrieval_index"}


def _stable_id(record: dict[str, Any]) -> str:
    keys = ["source_id", "kind", "local_path", "sha256", "revision", "repo_path"]
    # Only a split input is identified by its part. An unsplit one has to hash
    # exactly as it did before splitting existed, or every frozen
    # build.inputs.json in flight stops matching the acquisition it describes.
    if int(record.get("part_count", 1) or 1) > 1:
        keys += ["part_index", "part_count"]
    value = "\0".join(str(record.get(key, "")) for key in keys)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_oversized(records: list[dict[str, Any]], maximum_bytes: int) -> list[dict[str, Any]]:
    """Bound how much of the corpus any one task can be handed.

    One acquired file becomes one task and therefore one shard for the whole
    build, so a stage cannot finish before its largest single file does. Nothing
    inside a shard is parallel. On the 1.6 corpus the ten largest files were
    6.9GB against a 0.2GB median, and decontamination sat on them for a day and
    a half after the rest of the corpus was done.

    Splitting is by record position, not by byte offset: a member of a gzip
    stream cannot be seeked to, and a document must never be divided. Part p of
    n keeps the records where ``index % n == p``, so the parts are disjoint,
    together cover every record exactly once, and need no pre-scan to determine
    where to cut. Each part re-reads the container and discards the records it
    does not own, which costs a decompression pass -- around ninety seconds
    against the tens of hours the imbalance costs.
    """

    if maximum_bytes <= 0:
        return records
    output: list[dict[str, Any]] = []
    for record in records:
        size = int(record.get("size") or 0)
        parts = -(-size // maximum_bytes) if size > maximum_bytes else 1
        if parts <= 1:
            output.append({**record, "part_index": 0, "part_count": 1})
            continue
        for part_index in range(parts):
            output.append({**record, "part_index": part_index, "part_count": parts})
    return output


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
                inputs.append(canonical)
    # Split before identity is assigned: a part is its own task, so it needs its
    # own input_id.
    maximum_input_bytes = int(
        profile.get("storage", {}).get("maximum_input_bytes", 0)
    )
    inputs = _split_oversized(inputs, maximum_input_bytes)
    for canonical in inputs:
        canonical["input_id"] = _stable_id(canonical)
    inputs.sort(
        key=lambda row: (
            str(row.get("source_id")),
            str(row.get("relative_path")),
            int(row.get("part_index", 0)),
            row["input_id"],
        )
    )
    if not inputs:
        raise RuntimeError("Acquisition contains no training-record files")
    seen_paths: set[tuple[str, int]] = set()
    seen_ids: set[str] = set()
    for record in inputs:
        # A split file legitimately appears once per part, so the path alone is
        # no longer the identity; the part is.
        local_path = (str(record["local_path"]), int(record.get("part_index", 0)))
        input_id = str(record["input_id"])
        if local_path in seen_paths:
            raise RuntimeError(f"Acquisition contains a duplicate training-record path: {local_path[0]}")
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
        "input_bytes": sum(
            int(item["size"])
            for item in {str(row["local_path"]): row for row in inputs}.values()
        ),
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
