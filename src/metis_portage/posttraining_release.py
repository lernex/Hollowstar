from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from metis_data.ngram_canonical import validate_canonical_id_sidecar

from .config import PortageConfig
from .distributed import DistributedContext
from .util import atomic_write_json, file_sha256, json_sha256, read_json, utc_now


UMBRELLA_SCHEMA = "metis.posttraining-release-umbrella/v1"
INDEX_SCHEMA = "metis.posttraining-release-index/v1"
SEALED_SCHEMA = "metis.sealed-artifact/v1"
PREFLIGHT_SCHEMA = "metis.posttraining-release-preflight/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _contract(config: PortageConfig) -> dict[str, Any]:
    payload = yaml.safe_load(config.posttraining_contract.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "metis.posttraining-pipeline/v1"
    ):
        raise RuntimeError("Post-training contract has the wrong schema")
    return payload


def _requirements(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    stages = contract.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("Post-training contract has no stages")
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("enabled") is not True:
            continue
        stage_id = str(stage.get("id", ""))
        if not stage_id or stage_id in result:
            raise RuntimeError("Post-training stages must have unique non-empty ids")
        rows: dict[str, dict[str, Any]] = {}
        requirements = stage.get("requirements", [])
        if not isinstance(requirements, list):
            raise RuntimeError(f"{stage_id} requirements must be a list")
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                raise RuntimeError("Post-training requirement must be a mapping")
            name = str(requirement.get("name", ""))
            environment = str(requirement.get("env", ""))
            schema = str(requirement.get("schema", ""))
            if not name or name in rows or not environment or not schema:
                raise RuntimeError(
                    f"{stage_id} requirements need unique names, env, and schema"
                )
            rows[name] = dict(requirement)
        result[stage_id] = rows
    return result


def _safe_relative_path(
    anchor: Path,
    raw: Any,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise RuntimeError(f"{label} must be a safe relative path")
    root = anchor.parent.resolve()
    unresolved = root / raw
    if unresolved.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its release-index root") from exc
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise RuntimeError(f"{label} is missing or unsafe: {resolved}")
    if not must_exist and resolved.exists() and (
        resolved.is_symlink() or not resolved.is_file()
    ):
        raise RuntimeError(f"{label} future output is already unsafe: {resolved}")
    return resolved


def _safe_relative_directory(
    anchor: Path,
    raw: Any,
    *,
    label: str,
) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise RuntimeError(f"{label} must be a safe relative path")
    root = anchor.parent.resolve()
    unresolved = root / raw
    if unresolved.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its release-index root") from exc
    if resolved.exists() and (
        resolved.is_symlink() or not resolved.is_dir()
    ):
        raise RuntimeError(
            f"{label} must be absent or an existing directory: {resolved}"
        )
    return resolved


def _deep_verification(
    *,
    umbrella_path: Path,
    umbrella: Mapping[str, Any],
    contract_sha256: str,
    verify_payload_hashes: bool = False,
) -> dict[str, Any]:
    pointer = umbrella.get("deep_verification")
    if not isinstance(pointer, Mapping) or set(pointer) != {
        "path",
        "sha256",
        "receipt_sha256",
    }:
        raise RuntimeError("Post-training umbrella omits deep verification")
    path = _safe_relative_path(
        umbrella_path,
        pointer.get("path"),
        label="post-training deep verification",
        must_exist=True,
    )
    if (
        not _SHA256.fullmatch(str(pointer.get("sha256", "")))
        or not _SHA256.fullmatch(str(pointer.get("receipt_sha256", "")))
        or file_sha256(path) != pointer["sha256"]
    ):
        raise RuntimeError("Post-training deep-verification pointer changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    files = receipt.get("files") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema")
        != "metis.posttraining-release-deep-verification/v1"
        or receipt.get("posttraining_contract_sha256") != contract_sha256
        or receipt.get("receipt_sha256")
        != json_sha256(receipt, omit=("receipt_sha256",))
        or receipt.get("receipt_sha256") != pointer["receipt_sha256"]
        or receipt.get("complete") is not True
        or not isinstance(files, list)
        or not files
        or int(receipt.get("file_count", -1)) != len(files)
    ):
        raise RuntimeError("Post-training deep-verification receipt is invalid")
    release_root = umbrella_path.parent.resolve()
    seen: set[str] = set()
    total_bytes = 0
    for raw_record in files:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Deep-verification file record is invalid")
        relative = raw_record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _SHA256.fullmatch(str(raw_record.get("sha256", "")))
        ):
            raise RuntimeError("Deep-verification file record is unsafe")
        seen.add(relative)
        file_path = (release_root / relative).resolve()
        try:
            file_path.relative_to(release_root)
        except ValueError as exc:
            raise RuntimeError("Deep-verification payload escaped release") from exc
        if (
            not file_path.is_file()
            or file_path.is_symlink()
            or file_path.stat().st_size != int(raw_record.get("bytes", -1))
        ):
            raise RuntimeError(
                f"Deep-verification payload changed: {relative}"
            )
        if (
            verify_payload_hashes
            and file_sha256(file_path) != raw_record["sha256"]
        ):
            raise RuntimeError(
                f"Deep-verification payload hash changed: {relative}"
            )
        total_bytes += file_path.stat().st_size
    if int(receipt.get("total_bytes", -1)) != total_bytes:
        raise RuntimeError("Deep-verification total byte count changed")
    return {
        "path": str(path),
        "file_sha256": pointer["sha256"],
        "receipt_sha256": pointer["receipt_sha256"],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "inventory_sha256": json_sha256(files),
    }


def verify_posttraining_release_distributed(
    *,
    preflight_path: str | Path,
    output_path: str | Path,
    receipt_directory: str | Path,
    context: DistributedContext,
) -> dict[str, Any] | None:
    """Hash every sealed static post-training payload exactly once."""

    preflight_source = Path(preflight_path).expanduser().resolve()
    preflight = read_json(preflight_source)
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("ok") is not True
        or preflight.get("preflight_sha256")
        != json_sha256(preflight, omit=("preflight_sha256",))
    ):
        raise RuntimeError("Post-training preflight is invalid before deep audit")
    deep = preflight.get("deep_verification")
    if not isinstance(deep, Mapping):
        raise RuntimeError("Post-training preflight omits deep verification")
    deep_path = Path(str(deep.get("path", ""))).expanduser().resolve()
    if (
        not deep_path.is_file()
        or deep_path.is_symlink()
        or file_sha256(deep_path) != deep.get("file_sha256")
    ):
        raise RuntimeError("Post-training deep-verification receipt changed")
    receipt = read_json(deep_path)
    files = receipt.get("files")
    if (
        receipt.get("schema")
        != "metis.posttraining-release-deep-verification/v1"
        or receipt.get("receipt_sha256")
        != json_sha256(receipt, omit=("receipt_sha256",))
        or receipt.get("receipt_sha256") != deep.get("receipt_sha256")
        or receipt.get("complete") is not True
        or not isinstance(files, list)
        or int(receipt.get("file_count", -1)) != len(files)
        or json_sha256(files) != deep.get("inventory_sha256")
    ):
        raise RuntimeError("Post-training deep-verification inventory is invalid")
    release_root = deep_path.parent.resolve()
    assigned = [
        (index, row)
        for index, row in enumerate(files)
        if index % context.world_size == context.rank
    ]
    verified: list[int] = []
    bytes_hashed = 0
    for index, raw_record in assigned:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Post-training deep-verification record is invalid")
        relative = raw_record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("Post-training deep-verification path is unsafe")
        path = (release_root / relative).resolve()
        try:
            path.relative_to(release_root)
        except ValueError as exc:
            raise RuntimeError(
                "Post-training deep-verification payload escaped its release"
            ) from exc
        expected_bytes = int(raw_record.get("bytes", -1))
        expected_sha = str(raw_record.get("sha256", ""))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or not _SHA256.fullmatch(expected_sha)
            or file_sha256(path) != expected_sha
        ):
            raise RuntimeError(
                f"Post-training deep-verification payload changed: {relative}"
            )
        verified.append(index)
        bytes_hashed += expected_bytes
    rank_receipt: dict[str, Any] = {
        "schema": "metis.portage-posttraining-release-rank-receipt/v1",
        "rank": context.rank,
        "world_size": context.world_size,
        "record_indices": verified,
        "bytes_hashed": bytes_hashed,
        "preflight_sha256": preflight["preflight_sha256"],
        "deep_verification_receipt_sha256": receipt["receipt_sha256"],
        "inventory_sha256": deep["inventory_sha256"],
        "verified_at": utc_now(),
    }
    rank_receipt["receipt_sha256"] = json_sha256(rank_receipt)
    rank_root = Path(receipt_directory).expanduser().resolve()
    atomic_write_json(
        rank_root / f"rank-{context.rank:05d}.json",
        rank_receipt,
    )
    if context.initialized:
        import torch.distributed as dist

        gathered: list[dict[str, Any] | None] = [None] * context.world_size
        dist.all_gather_object(gathered, rank_receipt)
    else:
        gathered = [rank_receipt]
    if not context.is_root:
        return None
    concrete = [row for row in gathered if isinstance(row, dict)]
    indices = [
        int(index)
        for row in concrete
        for index in row.get("record_indices", [])
    ]
    if (
        len(concrete) != context.world_size
        or sorted(indices) != list(range(len(files)))
        or len(indices) != len(set(indices))
        or sum(int(row.get("bytes_hashed", -1)) for row in concrete)
        != int(receipt.get("total_bytes", -1))
    ):
        raise RuntimeError(
            "Distributed post-training verification did not cover each byte once"
        )
    for rank, row in enumerate(sorted(concrete, key=lambda item: int(item["rank"]))):
        if (
            int(row.get("rank", -1)) != rank
            or int(row.get("world_size", -1)) != context.world_size
            or row.get("receipt_sha256")
            != json_sha256(row, omit=("receipt_sha256",))
            or row.get("preflight_sha256") != preflight["preflight_sha256"]
            or row.get("deep_verification_receipt_sha256")
            != receipt["receipt_sha256"]
            or row.get("inventory_sha256") != deep["inventory_sha256"]
        ):
            raise RuntimeError("Post-training rank verification receipt is invalid")
    marker: dict[str, Any] = {
        "schema": "metis.portage-posttraining-release-verification/v1",
        "preflight_path": str(preflight_source),
        "preflight_file_sha256": file_sha256(preflight_source),
        "preflight_sha256": preflight["preflight_sha256"],
        "deep_verification_path": str(deep_path),
        "deep_verification_file_sha256": deep["file_sha256"],
        "deep_verification_receipt_sha256": receipt["receipt_sha256"],
        "inventory_sha256": deep["inventory_sha256"],
        "world_size": context.world_size,
        "file_count": len(files),
        "total_bytes": int(receipt["total_bytes"]),
        "rank_receipt_sha256s": [
            str(row["receipt_sha256"])
            for row in sorted(concrete, key=lambda item: int(item["rank"]))
        ],
        "verified_at": utc_now(),
        "ok": True,
    }
    marker["marker_sha256"] = json_sha256(marker)
    atomic_write_json(Path(output_path).expanduser().resolve(), marker)
    return marker


def _sealed_manifest(
    *,
    index_path: Path,
    entry: Mapping[str, Any],
    expected_schema: str,
    tokenizer_sha256: str | None,
    requirement: Mapping[str, Any] | None,
    family: str | None,
) -> tuple[Path, dict[str, Any], str]:
    path = _safe_relative_path(
        index_path,
        entry.get("path", entry.get("manifest")),
        label="sealed post-training manifest",
        must_exist=True,
    )
    expected_file_hash = str(entry.get("sha256", "")).lower()
    if (
        not _SHA256.fullmatch(expected_file_hash)
        or file_sha256(path) != expected_file_hash
    ):
        raise RuntimeError(f"Post-training index file hash mismatch: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid sealed post-training manifest: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("envelope_schema") != SEALED_SCHEMA
        or payload.get("schema") != expected_schema
        or payload.get("complete") is not True
    ):
        raise RuntimeError(
            f"{path} must be a complete {SEALED_SCHEMA} / {expected_schema}"
        )
    manifest_sha = json_sha256(payload, omit=("manifest_sha256",))
    if (
        payload.get("manifest_sha256") != manifest_sha
        or entry.get("manifest_sha256") != manifest_sha
    ):
        raise RuntimeError(f"Sealed post-training manifest self-hash mismatch: {path}")
    if tokenizer_sha256 is not None and payload.get("tokenizer_sha256") != tokenizer_sha256:
        raise RuntimeError(f"Post-training tokenizer lineage mismatch: {path}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Sealed post-training manifest has no payload files: {path}")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Invalid payload record in {path}")
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise RuntimeError(f"Unsafe or duplicate sealed payload in {path}")
        seen.add(relative)
        payload_path = (path.parent / relative).resolve()
        try:
            payload_path.relative_to(path.parent.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Sealed payload escapes its artifact root: {payload_path}"
            ) from exc
        if payload_path.is_symlink() or not payload_path.is_file():
            raise RuntimeError(f"Sealed payload is missing or a symlink: {payload_path}")
        if payload_path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"Sealed payload size drifted: {payload_path}")
        if not _SHA256.fullmatch(str(record.get("sha256", ""))):
            raise RuntimeError(f"Sealed payload lacks a SHA-256: {payload_path}")
    if requirement is not None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"{path} has no metadata mapping")
        if requirement.get("family_bound") is True and metadata.get("family") != family:
            raise RuntimeError(f"{path} is not bound to {family}")
        generated = requirement.get("generated_from_stage")
        if generated is not None and metadata.get("generated_from_stage") != generated:
            raise RuntimeError(
                f"{path} was not generated from the declared {generated} stage"
            )
        if requirement.get("checkpoint_bound") is True and not _SHA256.fullmatch(
            str(metadata.get("generated_from_checkpoint_sha256", ""))
        ):
            raise RuntimeError(f"{path} has no future-stage checkpoint binding")
        for requirement_key, metadata_key in (
            ("minimum_records", "records"),
            ("minimum_tokens", "tokens"),
            ("minimum_source_instructions", "source_instruction_count"),
        ):
            if requirement_key in requirement and int(
                metadata.get(metadata_key, -1)
            ) < int(requirement[requirement_key]):
                raise RuntimeError(
                    f"{path} does not meet {requirement_key}"
                )
        if "maximum_tokens" in requirement and int(metadata.get("tokens", -1)) > int(
            requirement["maximum_tokens"]
        ):
            raise RuntimeError(f"{path} exceeds its token budget")
        required_metadata = requirement.get("required_metadata", {})
        if not isinstance(required_metadata, Mapping):
            raise RuntimeError("required_metadata must be a mapping")
        for field, expected in required_metadata.items():
            if metadata.get(field) != expected:
                raise RuntimeError(
                    f"{path} metadata {field}={metadata.get(field)!r}, "
                    f"expected {expected!r}"
                )
    return path, payload, manifest_sha


def _validate_deferred(
    *,
    index_path: Path,
    record: Mapping[str, Any],
    stage_id: str,
    requirement_name: str,
) -> dict[str, Any]:
    output = _safe_relative_path(
        index_path,
        record.get("manifest"),
        label=f"deferred output {stage_id}.{requirement_name}",
        must_exist=False,
    )
    hook = record.get("generation_hook")
    if not isinstance(hook, Mapping):
        raise RuntimeError(
            f"Deferred {stage_id}.{requirement_name} has no generation_hook"
        )
    receipt = _safe_relative_path(
        index_path,
        hook.get("receipt"),
        label=f"deferred receipt {stage_id}.{requirement_name}",
        must_exist=False,
    )
    rank_receipts = _safe_relative_directory(
        index_path,
        hook.get("rank_receipts"),
        label=f"deferred rank receipts {stage_id}.{requirement_name}",
    )
    executable = _safe_relative_path(
        index_path,
        hook.get("executable"),
        label=f"deferred executable {stage_id}.{requirement_name}",
        must_exist=True,
    )
    executable_sha = str(hook.get("executable_sha256", "")).lower()
    if (
        not _SHA256.fullmatch(executable_sha)
        or file_sha256(executable) != executable_sha
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(
            f"Deferred executable hash/mode failed: {stage_id}.{requirement_name}"
        )
    args = hook.get("args", [])
    timeout = int(hook.get("timeout_seconds", 0))
    execution = hook.get("execution")
    if not isinstance(execution, Mapping):
        raise RuntimeError(
            f"Deferred hook has no execution protocol: {stage_id}.{requirement_name}"
        )
    protocol = execution.get("protocol")
    if protocol == "distributed_family_v1":
        valid_execution = set(execution) == {"protocol"}
    elif protocol == "rank0_only_v1":
        valid_execution = (
            set(execution)
            == {"protocol", "nodes", "tasks", "gpus_per_task"}
            and int(execution.get("nodes", 0)) == 1
            and int(execution.get("tasks", 0)) == 1
            and int(execution.get("gpus_per_task", -1)) in {0, 1}
        )
    else:
        valid_execution = False
    if (
        not isinstance(args, list)
        or not all(isinstance(item, str) for item in args)
        or not 1 <= timeout <= 7 * 24 * 60 * 60
        or len({output, receipt, rank_receipts, executable}) != 4
        or not valid_execution
    ):
        raise RuntimeError(
            f"Deferred hook contract is invalid: {stage_id}.{requirement_name}"
        )
    return {
        "state": "deferred",
        "output": str(output),
        "receipt": str(receipt),
        "rank_receipts": str(rank_receipts),
        "executable": str(executable),
        "executable_sha256": executable_sha,
        "args": list(args),
        "timeout_seconds": timeout,
        "execution": dict(execution),
    }


def _validate_base_tokenizer_binding(
    *,
    config: PortageConfig,
    tokenizer_manifest: Path,
    tokenizer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the post-training tokenizer bytes to the verified base release."""

    base_root = config.release_root.resolve()
    descriptor_path = base_root / "RELEASE.json"
    descriptor = read_json(descriptor_path)
    artifacts = descriptor.get("artifacts")
    if (
        descriptor.get("schema") != "metis.data-release/v2"
        or descriptor.get("release_sha256")
        != json_sha256(descriptor, omit=("release_sha256",))
        or not isinstance(artifacts, Mapping)
    ):
        raise RuntimeError(
            "Base RELEASE.json is not a self-hashed Metis data release"
        )

    def artifact(name: str) -> Path:
        raw = artifacts.get(name)
        if (
            not isinstance(raw, str)
            or not raw
            or Path(raw).is_absolute()
            or ".." in Path(raw).parts
        ):
            raise RuntimeError(f"Base release artifact {name} has an unsafe path")
        unresolved = base_root / raw
        if unresolved.is_symlink():
            raise RuntimeError(f"Base release artifact {name} may not be a symlink")
        path = unresolved.resolve()
        try:
            path.relative_to(base_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Base release artifact {name} escapes its root"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"Base release artifact {name} is missing")
        return path

    base_tokenizer = artifact("tokenizer")
    canonical_map = artifact("ngram_canonical_map")
    canonical_ids = artifact("ngram_canonical_ids")
    base_tokenizer_sha = file_sha256(base_tokenizer)
    canonical_map_file_sha = file_sha256(canonical_map)
    canonical_ids_sha = file_sha256(canonical_ids)
    if (
        descriptor.get("tokenizer_sha256") != base_tokenizer_sha
        or descriptor.get("ngram_canonical_map_manifest_sha256")
        != canonical_map_file_sha
        or descriptor.get("ngram_canonical_ids_sha256")
        != canonical_ids_sha
    ):
        raise RuntimeError(
            "Base release tokenizer/canonical bytes differ from RELEASE.json"
        )
    canonical_descriptor, _canonical_values = validate_canonical_id_sidecar(
        manifest_path=canonical_map,
        binary_path=canonical_ids,
        tokenizer_path=base_tokenizer,
        expected_vocabulary_size=65_536,
        expected_manifest_sha256=descriptor.get(
            "ngram_canonical_map_self_sha256"
        ),
        expected_binary_sha256=canonical_ids_sha,
        recompute_from_tokenizer=True,
    )

    metadata = tokenizer_payload.get("metadata")
    files = tokenizer_payload.get("files")
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise RuntimeError("Post-training tokenizer has no sealed metadata/files")
    tokenizer_file = metadata.get("tokenizer_file")
    if not isinstance(tokenizer_file, str):
        raise RuntimeError("Post-training tokenizer omits metadata.tokenizer_file")
    copied_tokenizer = _safe_relative_path(
        tokenizer_manifest,
        tokenizer_file,
        label="post-training tokenizer bytes",
        must_exist=True,
    )
    records = {
        str(record.get("path")): record
        for record in files
        if isinstance(record, Mapping)
    }
    expected_metadata = {
        "vocabulary_size": 65_536,
        "tokenizer_sha256": base_tokenizer_sha,
        "base_release_sha256": descriptor["release_sha256"],
        "base_release_tokenizer_path": base_tokenizer.relative_to(
            base_root
        ).as_posix(),
        "base_release_tokenizer_sha256": base_tokenizer_sha,
        "base_release_canonical_map_path": canonical_map.relative_to(
            base_root
        ).as_posix(),
        "base_release_canonical_ids_path": canonical_ids.relative_to(
            base_root
        ).as_posix(),
        "ngram_canonical_map_self_sha256": canonical_descriptor[
            "manifest_sha256"
        ],
        "ngram_canonical_ids_sha256": canonical_ids_sha,
    }
    tokenizer_record = records.get(tokenizer_file)
    if (
        any(metadata.get(field) != value for field, value in expected_metadata.items())
        or not isinstance(tokenizer_record, Mapping)
        or tokenizer_record.get("sha256") != base_tokenizer_sha
        or file_sha256(copied_tokenizer) != base_tokenizer_sha
    ):
        raise RuntimeError(
            "Post-training tokenizer is not byte/path-bound to the base release"
        )
    return {
        "base_release_descriptor": str(descriptor_path.resolve()),
        "base_release_descriptor_file_sha256": file_sha256(descriptor_path),
        "base_release_sha256": descriptor["release_sha256"],
        "base_tokenizer": str(base_tokenizer),
        "base_tokenizer_sha256": base_tokenizer_sha,
        "sealed_tokenizer": str(copied_tokenizer),
        "sealed_tokenizer_sha256": base_tokenizer_sha,
        "canonical_map": str(canonical_map),
        "canonical_map_file_sha256": canonical_map_file_sha,
        "canonical_map_self_sha256": canonical_descriptor["manifest_sha256"],
        "canonical_ids": str(canonical_ids),
        "canonical_ids_sha256": canonical_ids_sha,
    }


def _family_index(
    *,
    config: PortageConfig,
    umbrella_path: Path,
    family: str,
    entry: Mapping[str, Any],
    requirements: Mapping[str, Mapping[str, Mapping[str, Any]]],
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_path = _safe_relative_path(
        umbrella_path,
        entry.get("path"),
        label=f"{family} release index",
        must_exist=True,
    )
    expected_file_hash = str(entry.get("sha256", "")).lower()
    if (
        not _SHA256.fullmatch(expected_file_hash)
        or file_sha256(index_path) != expected_file_hash
    ):
        raise RuntimeError(f"{family} release-index file hash mismatch")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {family} release index") from exc
    if (
        not isinstance(index, dict)
        or index.get("schema") != INDEX_SCHEMA
        or index.get("family") != family
        or index.get("pipeline_sha256") != contract_sha256
        or index.get("index_sha256")
        != json_sha256(index, omit=("index_sha256",))
        or entry.get("index_sha256") != index.get("index_sha256")
    ):
        raise RuntimeError(f"{family} release-index lineage is invalid")
    tokenizer_entry = index.get("tokenizer_manifest")
    if not isinstance(tokenizer_entry, Mapping):
        raise RuntimeError(f"{family} release index omits tokenizer_manifest pins")
    tokenizer_path, tokenizer_payload, tokenizer_sha = _sealed_manifest(
        index_path=index_path,
        entry=tokenizer_entry,
        expected_schema="metis.tokenizer/v1",
        tokenizer_sha256=None,
        requirement=None,
        family=None,
    )
    tokenizer_binding = _validate_base_tokenizer_binding(
        config=config,
        tokenizer_manifest=tokenizer_path,
        tokenizer_payload=tokenizer_payload,
    )
    stage_records = index.get("requirements")
    if not isinstance(stage_records, Mapping):
        raise RuntimeError(f"{family} release index has no requirements mapping")
    if set(stage_records) != set(requirements):
        raise RuntimeError(
            f"{family} release index stage coverage differs from the contract"
        )
    records: dict[str, dict[str, Any]] = {}
    for stage_id, stage_requirements in requirements.items():
        indexed_stage = stage_records.get(stage_id)
        if not isinstance(indexed_stage, Mapping):
            raise RuntimeError(f"{family} release index omits stage {stage_id}")
        if set(indexed_stage) != set(stage_requirements):
            raise RuntimeError(
                f"{family} release index requirement coverage differs for "
                f"{stage_id}"
            )
        records[stage_id] = {}
        for name, requirement in stage_requirements.items():
            record = indexed_stage.get(name)
            if (
                not isinstance(record, Mapping)
                or record.get("schema") != requirement["schema"]
            ):
                raise RuntimeError(
                    f"{family} release index omits or mis-types {stage_id}.{name}"
                )
            state = record.get("state")
            checkpoint_bound = requirement.get("checkpoint_bound") is True
            if checkpoint_bound and state != "deferred":
                raise RuntimeError(
                    f"{family} {stage_id}.{name} is checkpoint-bound and "
                    "must be deferred"
                )
            if not checkpoint_bound and state != "sealed":
                raise RuntimeError(
                    f"{family} {stage_id}.{name} is not checkpoint-bound and "
                    "must be sealed"
                )
            if state == "sealed":
                path, _, manifest_sha = _sealed_manifest(
                    index_path=index_path,
                    entry=record,
                    expected_schema=str(requirement["schema"]),
                    tokenizer_sha256=(
                        tokenizer_sha
                        if requirement.get("tokenizer_bound", True)
                        else None
                    ),
                    requirement=requirement,
                    family=family,
                )
                records[stage_id][name] = {
                    "state": "sealed",
                    "path": str(path),
                    "manifest_sha256": manifest_sha,
                }
            elif state == "deferred":
                records[stage_id][name] = _validate_deferred(
                    index_path=index_path,
                    record=record,
                    stage_id=stage_id,
                    requirement_name=name,
                )
            else:
                raise RuntimeError(
                    f"{family} {stage_id}.{name} must be sealed or deferred"
                )
    return (
        {
            "path": str(index_path),
            "file_sha256": expected_file_hash,
            "index_sha256": index["index_sha256"],
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_manifest_sha256": tokenizer_sha,
            "tokenizer_binding": tokenizer_binding,
            "requirements": records,
        },
        index,
    )


def _failure_report(
    *,
    config: PortageConfig,
    requirements: Mapping[str, Mapping[str, Mapping[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    missing_requirements = [
        f"{stage_id}.{name}"
        for stage_id, rows in requirements.items()
        for name in rows
    ]
    index_path = config.posttraining_release_index
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "index_path": str(index_path),
        "index_file_sha256": (
            file_sha256(index_path) if index_path.is_file() else None
        ),
        "posttraining_contract_path": str(config.posttraining_contract),
        "posttraining_contract_sha256": file_sha256(config.posttraining_contract),
        "family_indexes": {},
        "bindings": {},
        "missing": {
            family.name: ["tokenizer_manifest", *missing_requirements]
            for family in config.families
        },
        "errors": errors,
        "ok": False,
    }


def inspect_posttraining_release_index(config: PortageConfig) -> dict[str, Any]:
    """Validate the umbrella and both backend-native family release indexes.

    Sealed records are verified immediately.  Checkpoint-bound data may instead
    be represented by a hash-pinned deferred generation hook; the backend binds
    its output receipt to the live parent checkpoint before consuming it.
    """

    contract = _contract(config)
    requirements = _requirements(contract)
    index_path = config.posttraining_release_index
    if not index_path.is_file() or index_path.is_symlink():
        return _failure_report(
            config=config,
            requirements=requirements,
            errors=[
                "post-training release umbrella is missing; Rhea must seal "
                "static inputs and pin generators for checkpoint-bound inputs"
            ],
        )
    try:
        umbrella = json.loads(index_path.read_text(encoding="utf-8"))
        contract_sha = file_sha256(config.posttraining_contract)
        if (
            not isinstance(umbrella, dict)
            or umbrella.get("schema") != UMBRELLA_SCHEMA
            or umbrella.get("posttraining_contract_sha256") != contract_sha
            or umbrella.get("umbrella_sha256")
            != json_sha256(umbrella, omit=("umbrella_sha256",))
        ):
            raise RuntimeError(
                f"Post-training umbrella must be self-hashed {UMBRELLA_SCHEMA}"
            )
        deep_verification = _deep_verification(
            umbrella_path=index_path,
            umbrella=umbrella,
            contract_sha256=contract_sha,
        )
        family_entries = umbrella.get("families")
        if not isinstance(family_entries, Mapping):
            raise RuntimeError("Post-training umbrella requires a families mapping")
        family_indexes: dict[str, dict[str, Any]] = {}
        tokenizer_lineage: str | None = None
        for family in config.families:
            entry = family_entries.get(family.name)
            if not isinstance(entry, Mapping):
                raise RuntimeError(
                    f"Post-training umbrella omits family {family.name}"
                )
            row, _index = _family_index(
                config=config,
                umbrella_path=index_path,
                family=family.name,
                entry=entry,
                requirements=requirements,
                contract_sha256=contract_sha,
            )
            if tokenizer_lineage is None:
                tokenizer_lineage = str(row["tokenizer_manifest_sha256"])
            elif row["tokenizer_manifest_sha256"] != tokenizer_lineage:
                raise RuntimeError(
                    "Praxis and Logos release indexes use different tokenizers"
                )
            family_indexes[family.name] = row
        resolved_umbrella = str(index_path.resolve())
        report: dict[str, Any] = {
            "schema": PREFLIGHT_SCHEMA,
            "created_at": utc_now(),
            "index_path": resolved_umbrella,
            "index_file_sha256": file_sha256(index_path),
            "umbrella_sha256": umbrella["umbrella_sha256"],
            "posttraining_contract_path": str(config.posttraining_contract),
            "posttraining_contract_sha256": contract_sha,
            "tokenizer_manifest_sha256": tokenizer_lineage,
            "deep_verification": deep_verification,
            "family_indexes": family_indexes,
            "bindings": {
                family.name: {
                    "METIS_POSTTRAINING_RELEASE_INDEX": resolved_umbrella,
                }
                for family in config.families
            },
            "missing": {},
            "errors": [],
            "ok": True,
        }
        report["preflight_sha256"] = json_sha256(report)
        return report
    except Exception as exc:
        return _failure_report(
            config=config,
            requirements=requirements,
            errors=[f"{type(exc).__name__}: {exc}"],
        )


def require_posttraining_release_index(report: Mapping[str, Any]) -> None:
    if report.get("ok") is not True:
        raise RuntimeError(
            "Post-training release preflight failed before allocation. "
            f"missing={report.get('missing', {})}; "
            f"errors={report.get('errors', [])}; "
            f"index={report.get('index_path')}"
        )


def validate_posttraining_preflight(
    report: Mapping[str, Any],
    *,
    config: PortageConfig,
) -> None:
    index = Path(str(report.get("index_path", "")))
    family_indexes = report.get("family_indexes")
    deep = report.get("deep_verification")
    if (
        report.get("schema") != PREFLIGHT_SCHEMA
        or report.get("ok") is not True
        or report.get("preflight_sha256")
        != json_sha256(report, omit=("preflight_sha256",))
        or not index.is_file()
        or index.is_symlink()
        or file_sha256(index) != report.get("index_file_sha256")
        or file_sha256(config.posttraining_contract)
        != report.get("posttraining_contract_sha256")
        or not isinstance(family_indexes, Mapping)
        or not isinstance(deep, Mapping)
    ):
        raise RuntimeError("Post-training release preflight is invalid or stale")
    umbrella = json.loads(index.read_text(encoding="utf-8"))
    observed_deep = _deep_verification(
        umbrella_path=index,
        umbrella=umbrella,
        contract_sha256=str(report["posttraining_contract_sha256"]),
    )
    if observed_deep != dict(deep):
        raise RuntimeError("Post-training deep verification changed after preflight")
    for family in config.families:
        row = family_indexes.get(family.name)
        if not isinstance(row, Mapping):
            raise RuntimeError("Post-training preflight omits a family index")
        path = Path(str(row.get("path", "")))
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != row.get("file_sha256")
        ):
            raise RuntimeError(
                f"Post-training {family.name} index changed after preflight"
            )
        tokenizer_binding = row.get("tokenizer_binding")
        if not isinstance(tokenizer_binding, Mapping):
            raise RuntimeError(
                f"Post-training {family.name} tokenizer binding is missing"
            )
        descriptor_path = Path(
            str(tokenizer_binding.get("base_release_descriptor", ""))
        )
        base_tokenizer = Path(
            str(tokenizer_binding.get("base_tokenizer", ""))
        )
        sealed_tokenizer = Path(
            str(tokenizer_binding.get("sealed_tokenizer", ""))
        )
        canonical_map = Path(
            str(tokenizer_binding.get("canonical_map", ""))
        )
        canonical_ids = Path(
            str(tokenizer_binding.get("canonical_ids", ""))
        )
        expected_descriptor = (config.release_root / "RELEASE.json").resolve()
        if (
            descriptor_path.resolve() != expected_descriptor
            or any(
                not candidate.is_file() or candidate.is_symlink()
                for candidate in (
                    descriptor_path,
                    base_tokenizer,
                    sealed_tokenizer,
                    canonical_map,
                    canonical_ids,
                )
            )
            or file_sha256(descriptor_path)
            != tokenizer_binding.get("base_release_descriptor_file_sha256")
            or file_sha256(base_tokenizer)
            != tokenizer_binding.get("base_tokenizer_sha256")
            or file_sha256(sealed_tokenizer)
            != tokenizer_binding.get("sealed_tokenizer_sha256")
            or file_sha256(canonical_map)
            != tokenizer_binding.get("canonical_map_file_sha256")
            or file_sha256(canonical_ids)
            != tokenizer_binding.get("canonical_ids_sha256")
        ):
            raise RuntimeError(
                f"Post-training {family.name} tokenizer/base-release bytes "
                "changed after preflight"
            )


def environment_for_family(
    report: Mapping[str, Any],
    family: str,
    *,
    config: PortageConfig | None = None,
) -> dict[str, str]:
    require_posttraining_release_index(report)
    if config is not None:
        validate_posttraining_preflight(report, config=config)
    row = report.get("bindings", {}).get(family)
    if not isinstance(row, Mapping) or set(row) != {
        "METIS_POSTTRAINING_RELEASE_INDEX"
    }:
        raise RuntimeError(f"Post-training index has no validated {family} binding")
    return {str(key): str(value) for key, value in row.items()}
