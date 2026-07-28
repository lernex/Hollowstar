from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .freshweb import (
    MAXIMUM_INDEX_SCAN_WORKERS,
    FreshWebOptions,
    materialize_freshweb,
)
from .state import StateStore, utc_now


FRESHWEB_ROUTE_SETTINGS: dict[str, dict[str, Any]] = {
    "general_web": {
        "estimated_tokens_per_document": 1_200,
        "selection_oversample": 1.75,
        "minimum_characters": 500,
        "allowed_categories": (
            "official_docs",
            "government",
            "education",
            "technical",
            "science",
            "software",
            "reporting",
            "general",
        ),
        "category_weights": (
            ("official_docs", 0.16),
            ("government", 0.14),
            ("education", 0.14),
            ("technical", 0.13),
            ("science", 0.10),
            ("software", 0.10),
            ("reporting", 0.13),
            ("general", 0.10),
        ),
    },
    "software_docs": {
        "estimated_tokens_per_document": 800,
        "selection_oversample": 3.00,
        "minimum_characters": 300,
        "allowed_categories": ("official_docs", "software", "technical"),
        "category_weights": (
            ("official_docs", 0.50),
            ("government", 0.00),
            ("education", 0.00),
            ("technical", 0.20),
            ("science", 0.00),
            ("software", 0.30),
            ("reporting", 0.00),
            ("general", 0.00),
        ),
    },
    "fresh_science": {
        "estimated_tokens_per_document": 1_800,
        "selection_oversample": 5.00,
        "minimum_characters": 700,
        "allowed_categories": ("science", "education", "government", "reporting"),
        "category_weights": (
            ("official_docs", 0.00),
            ("government", 0.10),
            ("education", 0.15),
            ("technical", 0.00),
            ("science", 0.55),
            ("software", 0.00),
            ("reporting", 0.20),
            ("general", 0.00),
        ),
    },
    "official_docs": {
        "estimated_tokens_per_document": 1_200,
        "selection_oversample": 3.00,
        "minimum_characters": 350,
        "allowed_categories": (
            "official_docs",
            "government",
            "education",
            "science",
            "technical",
        ),
        "category_weights": (
            ("official_docs", 0.50),
            ("government", 0.20),
            ("education", 0.10),
            ("technical", 0.10),
            ("science", 0.10),
            ("software", 0.00),
            ("reporting", 0.00),
            ("general", 0.00),
        ),
    },
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_path_under_root(path_value: Any, root: Path) -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Completed acquisition output escaped the Lustre root: {path}") from exc
    return path


def _validate_completion_outputs(value: Any, *, root: Path) -> int:
    """Cheaply reconcile a completion marker before a resume skips its task.

    SHA-256 verification remains an explicit parallel Rhea gate. Here we check
    identity, containment, existence, and recorded byte sizes so a deleted or
    truncated shard is detected before the supervisor spends days on later
    waves.
    """

    validated = 0
    if isinstance(value, list):
        return sum(_validate_completion_outputs(item, root=root) for item in value)
    if not isinstance(value, dict):
        return 0
    if value.get("kind") == "remote_source_plan" or value.get("materialized") is False:
        raise RuntimeError("Completed acquisition task contains an unresolved remote plan")
    if value.get("ready_for_training_build") is False:
        raise RuntimeError("Completed acquisition output is not build-ready")
    if value.get("kind") == "materialized_dataset":
        receipt_value = value.get("receipt")
        shards = value.get("shards")
        if not receipt_value or not isinstance(shards, list) or not shards:
            raise RuntimeError("Completed materialized dataset is missing its receipt or shards")
        receipt = _require_path_under_root(receipt_value, root)
        if not receipt.is_file():
            raise RuntimeError(f"Completed materialization receipt is missing: {receipt}")
        try:
            json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Completed materialization receipt is unreadable: {receipt}") from exc
        validated += 1
        for shard in shards:
            if not isinstance(shard, dict) or not shard.get("path"):
                raise RuntimeError("Completed materialized dataset contains an invalid shard record")
            shard_path = _require_path_under_root(shard["path"], root)
            if not shard_path.is_file():
                raise RuntimeError(f"Completed materialized shard is missing: {shard_path}")
            expected_size = shard.get("size")
            if expected_size is not None and shard_path.stat().st_size != int(expected_size):
                raise RuntimeError(f"Completed materialized shard size changed: {shard_path}")
            validated += 1
        return validated
    local_path = value.get("local_path")
    if local_path:
        path = _require_path_under_root(local_path, root)
        if not path.is_file():
            raise RuntimeError(f"Completed acquisition output is missing or is not a file: {path}")
        expected_size = value.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise RuntimeError(f"Completed acquisition output size changed: {path}")
        validated += 1
    for key in ("files", "outputs", "artifacts"):
        validated += _validate_completion_outputs(value.get(key), root=root)
    return validated


def validate_download_completion(
    profile: dict[str, Any], task: dict[str, Any], completion: dict[str, Any]
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    expected_signature = task.get("task_sha256")
    if expected_signature and completion.get("task_sha256") != expected_signature:
        raise RuntimeError(f"Completion marker does not match the immutable task: {task_id}")
    validated = _validate_completion_outputs(completion.get("files"), root=Path(profile["storage"]["lustre_root"]))
    if validated <= 0:
        raise RuntimeError(f"Completion marker has no materialized outputs: {task_id}")
    return completion


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
        "payload_role": item.get("payload_role", "training_records"),
        "candidate_token_estimate": int(item.get("candidate_token_estimate", 0)),
        "candidate_estimator": item.get("candidate_estimator"),
    }


def freshweb_options_for_item(item: dict[str, Any], profile: dict[str, Any]) -> FreshWebOptions:
    access = item.get("access")
    if not isinstance(access, dict):
        raise ValueError("common_crawl_ranges item requires an access mapping")
    route = str(access.get("route") or "")
    try:
        route_settings = FRESHWEB_ROUTE_SETTINGS[route]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Common Crawl route {route!r}; expected one of {sorted(FRESHWEB_ROUTE_SETTINGS)}"
        ) from exc
    license_contract = item.get("license")
    if not isinstance(license_contract, dict) or not license_contract.get("status"):
        raise ValueError("common_crawl_ranges item requires a source license contract")

    acquisition = profile.get("acquisition", {})
    runtime = profile.get("runtime", {})
    common_crawl = acquisition.get("common_crawl", {})
    if not isinstance(common_crawl, dict):
        raise ValueError("profile acquisition.common_crawl must be a mapping")
    lane_workers = max(
        1,
        int(acquisition.get("lane_max_workers", {}).get("common_crawl", 1)),
    )
    process_workers = min(
        10,
        max(1, 10 // lane_workers),
        max(1, int(common_crawl.get("range_workers", acquisition.get("max_workers", 10)))),
    )
    max_records = common_crawl.get("max_records_per_partition")
    # The URL-index scan is CPU-bound pure Python over billions of rows, so it
    # is the throughput lever for Common Crawl acquisition.  Each worker also
    # fetches its own partition, so this bounds concurrent index requests too.
    # Default conservatively: acquisition runs on a shared login host.
    index_scan_workers = min(
        MAXIMUM_INDEX_SCAN_WORKERS,
        max(
            1,
            int(
                common_crawl.get(
                    "index_scan_workers", max(1, min(8, (os.cpu_count() or 2) // 2))
                )
            ),
        ),
    )
    scratch_root = common_crawl.get("state_scratch_root") or runtime.get("state_scratch_root")
    options = FreshWebOptions(
        collinfo_url=str(common_crawl.get("collinfo_url", FreshWebOptions.collinfo_url)),
        data_root=str(access.get("warc_root") or common_crawl.get("data_root") or FreshWebOptions.data_root),
        opt_out_csv_url=str(common_crawl.get("opt_out_csv_url", FreshWebOptions.opt_out_csv_url)),
        max_records_per_partition=int(max_records) if max_records is not None else None,
        route=route,
        allowed_categories=tuple(route_settings["allowed_categories"]),
        estimated_tokens_per_document=int(route_settings["estimated_tokens_per_document"]),
        selection_oversample=float(route_settings["selection_oversample"]),
        category_weights=tuple(route_settings["category_weights"]),
        require_english=True,
        require_reusable_open_license=(
            license_contract["status"] == "per_record_required"
        ),
        minimum_characters=int(route_settings["minimum_characters"]),
        shard_count=int(common_crawl.get("shard_count", 128)),
        max_workers=process_workers,
        request_timeout_seconds=float(runtime.get("request_timeout_seconds", 120.0)),
        max_retries=int(runtime.get("download_retries", 8)),
        retry_base_seconds=float(common_crawl.get("retry_base_seconds", 1.0)),
        coalesce_gap_bytes=int(common_crawl.get("coalesce_gap_bytes", 64 * 1024)),
        maximum_span_bytes=int(common_crawl.get("maximum_span_bytes", 8 * 1024 * 1024)),
        parquet_batch_rows=int(common_crawl.get("parquet_batch_rows", 65_536)),
        # All four freshness routes share the same immutable URL-index cache.
        # Retain it through acquisition so the large index corpus is downloaded
        # once rather than once per route.
        keep_index_files=bool(common_crawl.get("keep_index_files", True)),
        user_agent=str(common_crawl.get("user_agent", FreshWebOptions.user_agent)),
        seed=f"metis-freshweb-2026-v1:{route}",
        freshness_cutoff_start=(
            str(access["cutoff_start"]) if access.get("cutoff_start") else None
        ),
        freshness_cutoff_end=(
            str(access["cutoff_end"]) if access.get("cutoff_end") else None
        ),
        index_scan_workers=index_scan_workers,
        state_scratch_root=str(scratch_root) if scratch_root else None,
        state_checkpoint_seconds=float(
            common_crawl.get("state_checkpoint_seconds", 300.0)
        ),
        state_cache_mib=max(2, int(common_crawl.get("state_cache_mib", 1_024))),
    )
    options.validate()
    return options


def _materialize_common_crawl(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path
) -> dict[str, Any]:
    route = str(item.get("access", {}).get("route") or "")
    license_contract = item.get("license")
    if not isinstance(license_contract, dict) or not license_contract.get("status"):
        raise ValueError("common_crawl_ranges item requires a source license contract")
    license_required = license_contract["status"] == "per_record_required"
    cache_directory = str(profile.get("storage", {}).get("directories", {}).get("cache", "cache"))
    cache_root = (root / cache_directory / "common-crawl").resolve()
    try:
        cache_root.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Configured Common Crawl cache must be beneath the Lustre root") from exc
    common_crawl = profile.get("acquisition", {}).get("common_crawl", {})
    reserve_multiplier = float(common_crawl.get("final_opt_out_reserve_multiplier", 1.0))
    if reserve_multiplier < 1.0 or reserve_multiplier > 1.25:
        raise ValueError(
            "acquisition.common_crawl.final_opt_out_reserve_multiplier must be between 1.0 and 1.25"
        )
    locked_candidate_target = int(item.get("candidate_tokens", 0))
    materialization_item = item
    if reserve_multiplier != 1.0:
        materialization_item = {
            **item,
            # The immutable source lock remains the minimum handoff contract.
            # A small deterministic reserve absorbs publisher opt-outs that
            # arrive during a multi-day acquisition; the final handoff recount
            # still fails closed against the original target.
            "candidate_tokens": math.ceil(locked_candidate_target * reserve_multiplier),
            "locked_candidate_tokens": locked_candidate_target,
            "final_opt_out_reserve_multiplier": reserve_multiplier,
        }
    maximum_selection_round = int(common_crawl.get("maximum_selection_round", 8))
    if maximum_selection_round < 0:
        raise ValueError("acquisition.common_crawl.maximum_selection_round must be non-negative")
    reserve_crawls = [
        str(crawl)
        for crawl in materialization_item.get("access", {}).get(
            "reserve_crawls", []
        )
    ]
    materialization_tiers: list[tuple[str, dict[str, Any]]] = [
        ("primary", materialization_item)
    ]
    if reserve_crawls:
        primary_crawls = [
            str(crawl)
            for crawl in materialization_item.get("access", {}).get("crawls", [])
        ]
        merged_crawls = list(dict.fromkeys([*primary_crawls, *reserve_crawls]))
        materialization_tiers.append(
            (
                "cold_reserve",
                {
                    **materialization_item,
                    "access": {
                        **materialization_item["access"],
                        "crawls": merged_crawls,
                    },
                },
            )
        )
    receipt: dict[str, Any] = {}
    activated_tier = "primary"
    active_materialization_item = materialization_item
    for tier_name, tier_item in materialization_tiers:
        activated_tier = tier_name
        active_materialization_item = tier_item
        freshweb_options = freshweb_options_for_item(tier_item, profile)
        while True:
            receipt = materialize_freshweb(
                tier_item,
                root=root,
                cache_root=cache_root,
                options=freshweb_options,
            )
            if receipt.get("candidate_target_met") and receipt.get(
                "ready_for_training_build"
            ):
                break
            current_round = int(receipt.get("selection_round", 0))
            next_round = int(receipt.get("next_selection_round", current_round))
            if (
                receipt.get("automatic_widening") is not True
                or next_round <= current_round
                or next_round > maximum_selection_round
            ):
                break
        if receipt.get("candidate_target_met") and receipt.get(
            "ready_for_training_build"
        ):
            break
    if not receipt.get("candidate_target_met") or not receipt.get("ready_for_training_build"):
        qualified_tokens = int(
            receipt.get(
                "estimated_license_eligible_tokens",
                receipt.get("estimated_materialized_tokens", 0),
            )
        )
        raise RuntimeError(
            f"Common Crawl route {route!r} for {item['source_id']} did not meet its "
            f"candidate target with license qualification: {qualified_tokens:,} of "
            f"{int(receipt.get('candidate_token_target', active_materialization_item['candidate_tokens'])):,} "
            "estimated tokens"
        )
    if license_required and (
        receipt.get("license_eligibility", {}).get("required_for_candidate_target") is not True
        or int(receipt.get("estimated_license_eligible_tokens", 0))
        < int(receipt.get("candidate_token_target", materialization_item["candidate_tokens"]))
    ):
        raise RuntimeError(
            f"Common Crawl route {route!r} for {item['source_id']} did not meet its "
            "per-record license-eligible candidate target"
        )
    shards = receipt.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"Common Crawl route {route!r} for {item['source_id']} produced no shards")
    documents = Path(str(receipt.get("local_path") or "")).resolve()
    receipt_path = documents.parent / "ACQUISITION_RECEIPT.json"
    if not documents.is_dir() or not receipt_path.is_file():
        raise RuntimeError(f"Common Crawl materialization receipt is missing for {item['source_id']}")
    try:
        persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Common Crawl materialization receipt is unreadable for {item['source_id']}") from exc
    if (
        persisted_receipt.get("candidate_target_met") is not True
        or persisted_receipt.get("ready_for_training_build") is not True
        or persisted_receipt.get("shards") != shards
        or (
            license_required
            and (
                persisted_receipt.get("license_eligibility", {}).get(
                    "required_for_candidate_target"
                )
                is not True
                or int(persisted_receipt.get("estimated_license_eligible_tokens", 0))
                < int(persisted_receipt.get("candidate_token_target", 0))
            )
        )
    ):
        raise RuntimeError(
            f"Common Crawl materialization receipt is not handoff-ready for {item['source_id']}"
        )
    return {
        "kind": "materialized_dataset",
        "source_id": item["source_id"],
        "driver": "common_crawl_ranges",
        "route": route,
        "local_path": str(documents.parent),
        "receipt": str(receipt_path),
        "shards": shards,
        "size": sum(int(shard["size"]) for shard in shards),
        "records": int(receipt.get("warc", {}).get("records_extracted", 0)),
        "estimated_tokens": int(receipt.get("estimated_materialized_tokens", 0)),
        "estimated_license_eligible_tokens": int(
            receipt.get("estimated_license_eligible_tokens", 0)
        ),
        "candidate_token_estimate": int(
            receipt.get("estimated_license_eligible_tokens", 0)
            if license_required
            else receipt.get("estimated_materialized_tokens", 0)
        ),
        "candidate_estimator": (
            "license_eligible_extracted_text_bytes_divided_by_4"
            if license_required
            else "extracted_text_bytes_divided_by_4"
        ),
        "materialized": True,
        "ready_for_training_build": True,
        "replacement_reserve_tier": activated_tier,
        "reserve_crawls_activated": (
            reserve_crawls if activated_tier == "cold_reserve" else []
        ),
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
        completion = state.read("completed", "download", f"{task_id}.json")
        return validate_download_completion(profile, task, completion)

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
                if item.get("driver") == "common_crawl_ranges":
                    outputs.append(_materialize_common_crawl(item, profile=profile, root=lustre))
                    continue
                if item.get("driver") in {"github_repositories", "github_discussions"}:
                    from .acquisition.github import materialize_github

                    outputs.append(materialize_github(item, profile=profile, root=lustre))
                    continue
                from .materializers import SUPPORTED_MATERIALIZER_DRIVERS, run_production_materializer

                if item.get("driver") in SUPPORTED_MATERIALIZER_DRIVERS:
                    outputs.extend(run_production_materializer(item, profile=profile, root=lustre))
                else:
                    from .source_builders import run_source_builder

                    outputs.append(run_source_builder(item, profile=profile, root=lustre))
            else:
                raise ValueError(f"Unknown download item kind {item['kind']}")
        result = {
            "task_id": task_id,
            "task_index": task_index,
            "task_sha256": task.get("task_sha256"),
            "completed_at": utc_now(),
            "files": outputs,
            "downloaded_bytes": sum(int(item.get("size", item.get("downloaded_bytes", 0))) for item in outputs),
        }
        state.complete("download", task_id, result)
        return result
