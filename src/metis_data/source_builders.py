from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import requests
import yaml

from .config import repository_root
from .state import atomic_json, utc_now


def _write_remote_plan(item: dict[str, Any], root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output = root / "raw" / item["source_id"] / "REMOTE_SOURCE.json"
    atomic_json(output, {**payload, "source_id": item["source_id"], "created_at": utc_now()})
    return {
        "kind": "remote_source_plan",
        "source_id": item["source_id"],
        "local_path": str(output),
        "size": output.stat().st_size,
        "materialized": False,
        "ready_for_training_build": False,
    }


def _common_crawl(item: dict[str, Any], root: Path) -> dict[str, Any]:
    response = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=60)
    response.raise_for_status()
    available = {entry["id"]: entry for entry in response.json()}
    crawls = [
        *item["access"]["crawls"],
        *item["access"].get("reserve_crawls", []),
    ]
    missing = [crawl for crawl in crawls if crawl not in available]
    if missing:
        raise RuntimeError(f"Common Crawl releases are unavailable: {missing}")
    return _write_remote_plan(
        item,
        root,
        {
            "driver": "common_crawl_ranges",
            "crawls": [available[crawl] for crawl in crawls],
            "url_index": item["access"]["url_index"],
            "warc_root": item["access"]["warc_root"],
            "candidate_tokens": item["candidate_tokens"],
            "instructions": "Query the Parquet URL Index in bulk, then fetch only selected WARC byte ranges.",
        },
    )


def _registry_plan(item: dict[str, Any], root: Path) -> dict[str, Any]:
    registry = (repository_root() / item["access"]["registry"]).resolve()
    if not registry.exists():
        raise FileNotFoundError(f"Registry not found: {registry}")
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    resolved_repositories = []
    for repo in payload.get("repositories", []):
        result = subprocess.run(
            ["git", "ls-remote", repo["url"], "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        revision = result.stdout.split()[0]
        resolved_repositories.append({**repo, "revision": revision})
    return _write_remote_plan(
        item,
        root,
        {
            "driver": item["driver"],
            "registry": str(registry),
            "registry_payload": payload,
            "resolved_repositories": resolved_repositories,
            "candidate_tokens": item["candidate_tokens"],
        },
    )


def _github_plan(item: dict[str, Any], root: Path) -> dict[str, Any]:
    access = item["access"]
    return _write_remote_plan(
        item,
        root,
        {
            "driver": item["driver"],
            "cutoff_start": access["cutoff_start"],
            "cutoff_end": access["cutoff_end"],
            "candidate_tokens": item["candidate_tokens"],
            "selection_inputs": [
                "GH Archive activity events",
                "repository license at pinned commit",
                "non-fork canonical repository",
                "recent accepted commits and release activity",
            ],
        },
    )


def _repository_index_plan(item: dict[str, Any], root: Path) -> dict[str, Any]:
    return _write_remote_plan(
        item,
        root,
        {
            "driver": "repository_index",
            "components": item["access"].get("components", []),
            "candidate_tokens": item["candidate_tokens"],
            "instructions": (
                "Resolve every accepted repo, pinned commit, and path from the downloaded metadata; "
                "fetch the source blob; verify license, content hash, and repository filters; then emit canonical records."
            ),
        },
    )


def _derived_plan(item: dict[str, Any], root: Path) -> dict[str, Any]:
    return _write_remote_plan(
        item,
        root,
        {
            "driver": "derived_after_download",
            "parents": item["access"].get("parents", []),
            "candidate_tokens": item["candidate_tokens"],
            "instructions": (
                "Run the source-specific, provenance-preserving derivation after all parents have passed "
                "normalization, global deduplication, quality filtering, and contamination removal."
            ),
        },
    )


def run_source_builder(item: dict[str, Any], *, profile: dict[str, Any], root: Path) -> dict[str, Any]:
    driver = item["driver"]
    if driver == "common_crawl_ranges":
        return _common_crawl(item, root)
    if driver in {"canonical_web", "canonical_git", "canonical_http"}:
        return _registry_plan(item, root)
    if driver in {"github_repositories", "github_discussions"}:
        return _github_plan(item, root)
    if driver == "repository_index":
        return _repository_index_plan(item, root)
    if driver == "derived_after_download":
        return _derived_plan(item, root)
    raise RuntimeError(f"No acquisition builder is registered for driver {driver!r}")
