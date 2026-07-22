from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, get_token

from .manifest import candidate_plan, matches_any, total_phase_tokens
from .state import StateStore, utc_now


def _stable_key(seed: int, repo_id: str, path: str) -> str:
    return hashlib.sha256(f"{seed}\0{repo_id}\0{path}".encode()).hexdigest()


def _iter_repo_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    patterns: Iterable[str],
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in api.list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=True, expand=True):
        path = getattr(item, "path", "")
        if not path or not matches_any(path, patterns):
            continue
        size = int(getattr(item, "size", 0) or 0)
        if size <= 0:
            continue
        lfs = getattr(item, "lfs", None)
        lfs_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append(
            {
                "path": path,
                "size": size,
                "blob_id": getattr(item, "blob_id", None),
                "lfs_sha256": lfs_sha256,
            }
        )
    return files


def _select_files(
    files: list[dict[str, Any]],
    *,
    seed: int,
    repo_id: str,
    target_bytes: int,
    take_all: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    ordered = sorted(files, key=lambda item: _stable_key(seed, repo_id, item["path"]))
    if take_all:
        return ordered, False
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for item in ordered:
        selected.append(item)
        selected_bytes += item["size"]
        if selected_bytes >= target_bytes:
            break
    return selected, selected_bytes < target_bytes


def _hf_accesses(source: dict[str, Any]) -> list[dict[str, Any]]:
    access = source["access"]
    if access.get("type") == "huggingface":
        return [access]
    if access.get("type") == "repository_index":
        return [
            {
                "type": "huggingface",
                "repo_id": component["repo_id"],
                "revision": component["revision"],
                "gated": component.get("gated", False),
                "allow_patterns": ["**/*.parquet", "**/*.jsonl", "**/*.jsonl.zst"],
                "take_all": True,
            }
            for component in access.get("components", [])
        ]
    return []


def resolve_sources(manifest: dict[str, Any], profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    api = HfApi(token=get_token())
    seed = int(manifest.get("selection", {}).get("seed", 0))
    plan_by_source = {row["id"]: row for row in candidate_plan(manifest)["sources"]}
    resolved_sources: list[dict[str, Any]] = []
    all_download_items: list[dict[str, Any]] = []

    for source in manifest["sources"]:
        source_id = source["id"]
        source_plan = plan_by_source[source_id]
        hf_accesses = _hf_accesses(source)
        resolved: dict[str, Any] = {
            "id": source_id,
            "driver": source["acquisition"]["driver"],
            "final_exposure_tokens": total_phase_tokens(source),
            "candidate_tokens": source_plan["candidate_tokens"],
            "planned_download_bytes": source_plan["planned_download_bytes"],
            "repositories": [],
        }
        if hf_accesses:
            per_repo_target = max(1, source_plan["planned_download_bytes"] // len(hf_accesses))
            for access in hf_accesses:
                repo_id = access["repo_id"]
                revision = access["revision"]
                info = api.dataset_info(repo_id, revision=revision, timeout=60)
                if info.sha != revision:
                    raise RuntimeError(f"Pinned revision drift for {repo_id}: expected {revision}, resolved {info.sha}")
                files = _iter_repo_files(
                    api,
                    repo_id,
                    revision,
                    access.get("allow_patterns", ["**/*.parquet", "**/*.jsonl*"]),
                )
                selected, short = _select_files(
                    files,
                    seed=seed,
                    repo_id=repo_id,
                    target_bytes=per_repo_target,
                    take_all=bool(access.get("take_all")),
                )
                resolved_repo = {
                    "repo_id": repo_id,
                    "revision": revision,
                    "gated": access.get("gated", False),
                    "selected_bytes": sum(item["size"] for item in selected),
                    "available_bytes": sum(item["size"] for item in files),
                    "candidate_shortfall_at_byte_estimate": short,
                    "files": selected,
                }
                resolved["repositories"].append(resolved_repo)
                for item in selected:
                    all_download_items.append(
                        {
                            "kind": "hf_file",
                            "source_id": source_id,
                            "repo_id": repo_id,
                            "revision": revision,
                            **item,
                        }
                    )
            if source["acquisition"]["driver"] == "repository_index":
                # The Hugging Face payload is an immutable retrieval index, not
                # repository source code.  Keep a separate unresolved item in
                # the lock so neither status nor normalization can count index
                # rows as materialized training documents.
                all_download_items.append(
                    {
                        "kind": "builder",
                        "source_id": source_id,
                        "driver": "repository_index",
                        "access": source["access"],
                        "planned_download_bytes": source_plan["planned_download_bytes"],
                        "candidate_tokens": source_plan["candidate_tokens"],
                    }
                )
        elif source["access"].get("type") == "derived":
            resolved["derived_from"] = source["access"].get("parents", [])
            # A parent reference describes a derivation recipe; it is not a
            # payload.  Record it as an unresolved materialization task until a
            # tested builder writes actual canonical records.
            all_download_items.append(
                {
                    "kind": "builder",
                    "source_id": source_id,
                    "driver": "derived_after_download",
                    "access": source["access"],
                    "planned_download_bytes": source_plan["planned_download_bytes"],
                    "candidate_tokens": source_plan["candidate_tokens"],
                }
            )
        else:
            all_download_items.append(
                {
                    "kind": "builder",
                    "source_id": source_id,
                    "driver": source["acquisition"]["driver"],
                    "access": source["access"],
                    "planned_download_bytes": source_plan["planned_download_bytes"],
                    "candidate_tokens": source_plan["candidate_tokens"],
                }
            )
        resolved_sources.append(resolved)

    target_task_bytes = int(profile.get("scheduler", {}).get("download", {}).get("target_bytes_per_task", 20_000_000_000))
    tasks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for item in all_download_items:
        item_bytes = int(item.get("size", item.get("planned_download_bytes", 0)))
        if current and current_bytes + item_bytes > target_task_bytes:
            tasks.append({"items": current, "planned_bytes": current_bytes})
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
        if item.get("kind") == "builder" or current_bytes >= target_task_bytes:
            tasks.append({"items": current, "planned_bytes": current_bytes})
            current = []
            current_bytes = 0
    if current:
        tasks.append({"items": current, "planned_bytes": current_bytes})
    for index, task in enumerate(tasks):
        task["task_index"] = index
        task["task_id"] = f"download-{index:06d}"

    lock = {
        "schema": "metis.source-lock/v1",
        "release": manifest["release"],
        "resolved_at": utc_now(),
        "source_manifest": manifest["_path"],
        "sources": resolved_sources,
        "download_tasks": tasks,
    }
    state.write("sources.lock.json", payload=lock)
    return lock
