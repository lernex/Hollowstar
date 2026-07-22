from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .state import StateStore, utc_now


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_hf_file(item: dict[str, Any], *, root: Path, cache_dir: Path) -> dict[str, Any]:
    source_root = root / "raw" / item["source_id"] / item["repo_id"].replace("/", "--") / item["revision"]
    source_root.mkdir(parents=True, exist_ok=True)
    local_path = Path(
        hf_hub_download(
            repo_id=item["repo_id"],
            filename=item["path"],
            repo_type="dataset",
            revision=item["revision"],
            cache_dir=cache_dir,
            local_dir=source_root,
        )
    )
    actual_size = local_path.stat().st_size
    if int(item.get("size", actual_size)) != actual_size:
        raise RuntimeError(f"Size mismatch for {item['repo_id']}:{item['path']}: {actual_size} != {item.get('size')}")
    actual_sha = sha256_file(local_path)
    expected_sha = item.get("lfs_sha256")
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError(f"Checksum mismatch for {item['repo_id']}:{item['path']}")
    return {
        "kind": "hf_file",
        "source_id": item["source_id"],
        "repo_id": item["repo_id"],
        "revision": item["revision"],
        "repo_path": item["path"],
        "local_path": str(local_path),
        "size": actual_size,
        "sha256": actual_sha,
    }


def run_download_task(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    lustre = Path(profile["storage"]["lustre_root"])
    state_root = lustre / profile["storage"]["directories"]["state"]
    state = StateStore(state_root)
    lock = state.read("sources.lock.json")
    if lock is None:
        raise RuntimeError("sources.lock.json is missing; run `metisctl resolve` first")
    try:
        task = lock["download_tasks"][task_index]
    except IndexError as exc:
        raise ValueError(f"Unknown download task index {task_index}") from exc
    task_id = task["task_id"]
    if state.is_complete("download", task_id):
        return state.read("completed", "download", f"{task_id}.json")

    safety_bytes = int(float(profile["storage"].get("safety_free_tb", 0)) * 1_000_000_000_000)
    planned_bytes = int(task.get("planned_bytes", 0))
    free_bytes = shutil.disk_usage(lustre).free
    if free_bytes - planned_bytes < safety_bytes:
        raise RuntimeError(
            f"Insufficient free space for {task_id}: {free_bytes:,} free, {planned_bytes:,} planned, "
            f"{safety_bytes:,} safety reserve"
        )

    cache_dir = lustre / profile["runtime"].get("hf_home", "cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    with state.task_lock("download", task_id):
        for item in task["items"]:
            if item["kind"] == "hf_file":
                outputs.append(_download_hf_file(item, root=lustre, cache_dir=cache_dir))
            elif item["kind"] == "builder":
                from .source_builders import run_source_builder

                outputs.append(run_source_builder(item, profile=profile, root=lustre))
            else:
                raise ValueError(f"Unknown download item kind {item['kind']}")
        result = {
            "task_id": task_id,
            "task_index": task_index,
            "completed_at": utc_now(),
            "files": outputs,
            "downloaded_bytes": sum(int(item.get("size", item.get("downloaded_bytes", 0))) for item in outputs),
        }
        state.complete("download", task_id, result)
        return result
