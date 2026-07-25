from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from metis_data.ngram_canonical import validate_canonical_id_sidecar

from .config import PortageConfig, load_portage_config
from .posttraining_release import (
    INDEX_SCHEMA,
    SEALED_SCHEMA,
    UMBRELLA_SCHEMA,
    _contract,
    _requirements,
    inspect_posttraining_release_index,
    require_posttraining_release_index,
)
from .util import atomic_write_json, file_sha256, json_sha256, utc_now


BUILD_SPEC_SCHEMA = "metis.posttraining-release-build/v1"
DEEP_RECEIPT_SCHEMA = "metis.posttraining-release-deep-verification/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHASE_TOKENS = {
    "phase_a": 700_000_000_000,
    "phase_b": 250_000_000_000,
    "phase_c": 50_000_000_000,
}
_BUILTIN_HOOK_NAME = "metis16-posttraining-materialize"
_BUILTIN_ADAPTER_SOURCES = {
    "deepseek_dpd_pilot": ("deepseek_dpd_pilot", "deepseek_teacher"),
    "deepseek_dpd": ("deepseek_dpd", "deepseek_teacher"),
    "specialist_reasoning": ("specialist_reasoning", "stem_verifier"),
    "specialist_code": ("specialist_code", "code_verifier"),
    "specialist_knowledge": (
        "specialist_knowledge",
        "knowledge_verifier",
    ),
    "specialist_writing": ("specialist_writing", "writing_verifier"),
    "specialist_agentic": (
        "specialist_agentic",
        "agentic_environment",
    ),
    "opd_consolidation": (
        "opd_consolidation",
        "opd_generation_adapter",
    ),
    "preference_alignment": (
        "pairwise_reward_model",
        "pairwise_preference_data",
    ),
    "evaluation": ("evaluation", "evaluation_suite"),
    "publish_gate": ("evaluation", "evaluation_suite"),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _safe_path(
    root: Path,
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
        raise RuntimeError(f"{label} must be a safe path relative to the release root")
    candidate = root / raw
    if candidate.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the post-training release root") from exc
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise RuntimeError(f"{label} is missing or unsafe: {resolved}")
    if not must_exist and resolved.exists() and (
        not resolved.is_file() or resolved.is_symlink()
    ):
        raise RuntimeError(f"{label} future output is already unsafe: {resolved}")
    return resolved


def _safe_directory(
    root: Path,
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
        raise RuntimeError(f"{label} must be a safe path relative to the release root")
    candidate = root / raw
    if candidate.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the post-training release root") from exc
    if resolved.exists() and (
        resolved.is_symlink() or not resolved.is_dir()
    ):
        raise RuntimeError(
            f"{label} must be absent or an existing directory: {resolved}"
        )
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _release_artifact(
    release_root: Path,
    artifacts: Mapping[str, Any],
    name: str,
) -> Path:
    raw = artifacts.get(name)
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise RuntimeError(f"Base release artifact {name} has an unsafe path")
    unresolved = release_root / raw
    if unresolved.is_symlink():
        raise RuntimeError(f"Base release artifact {name} may not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(release_root)
    except ValueError as exc:
        raise RuntimeError(f"Base release artifact {name} escapes its root") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Base release artifact {name} is missing: {path}")
    return path


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.partial"
    )
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o640)
    finally:
        temporary.unlink(missing_ok=True)


def _install_builtin_generation_hook(
    *,
    config: PortageConfig,
    release_root: Path,
) -> Path:
    repository = Path(
        getattr(
            config,
            "repository",
            Path(__file__).resolve().parents[2],
        )
    ).resolve()
    source = (
        repository / "ops" / _BUILTIN_HOOK_NAME
    ).resolve()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(
            f"Built-in post-training generation hook is missing: {source}"
        )
    destination = (
        release_root / "hooks" / _BUILTIN_HOOK_NAME
    ).resolve()
    try:
        destination.relative_to(release_root)
    except ValueError as exc:
        raise RuntimeError(
            "Built-in generation hook destination escapes the release"
        ) from exc
    _copy_atomic(source, destination)
    destination.chmod(0o750)
    return destination


def _sealed_base_tokenizer(
    *,
    config: PortageConfig,
    release_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Seal an exact copy of the immutable Rhea tokenizer for post-training."""

    base_root = config.release_root.resolve()
    descriptor_path = base_root / "RELEASE.json"
    descriptor = _read_json(descriptor_path, label="base data release")
    if (
        descriptor.get("schema") != "metis.data-release/v2"
        or descriptor.get("release") != "metis-1.6-data-r1"
        or descriptor.get("release_sha256")
        != json_sha256(descriptor, omit=("release_sha256",))
        or int(descriptor.get("target_tokens", -1)) != 1_000_000_000_000
        or descriptor.get("phase_tokens") != _PHASE_TOKENS
        or descriptor.get("token_dtype") != "uint16"
        or descriptor.get("token_endianness") != "little"
        or descriptor.get("verification", {}).get("ok") is not True
    ):
        raise RuntimeError(
            "Base RELEASE.json is not the verified immutable Metis-1.6 release"
        )
    artifacts = _mapping(descriptor.get("artifacts"), "base release artifacts")
    tokenizer = _release_artifact(base_root, artifacts, "tokenizer")
    canonical_map = _release_artifact(
        base_root, artifacts, "ngram_canonical_map"
    )
    canonical_ids = _release_artifact(
        base_root, artifacts, "ngram_canonical_ids"
    )
    tokenizer_sha = file_sha256(tokenizer)
    if (
        descriptor.get("tokenizer_sha256") != tokenizer_sha
        or descriptor.get("ngram_canonical_map_manifest_sha256")
        != file_sha256(canonical_map)
        or descriptor.get("ngram_canonical_ids_sha256")
        != file_sha256(canonical_ids)
    ):
        raise RuntimeError("Base release tokenizer/canonical artifact hashes changed")
    canonical_descriptor, _canonical_values = validate_canonical_id_sidecar(
        manifest_path=canonical_map,
        binary_path=canonical_ids,
        tokenizer_path=tokenizer,
        expected_vocabulary_size=65_536,
        expected_manifest_sha256=descriptor.get(
            "ngram_canonical_map_self_sha256"
        ),
        expected_binary_sha256=descriptor.get("ngram_canonical_ids_sha256"),
        recompute_from_tokenizer=False,
    )
    tokenizer_contract = _mapping(
        descriptor.get("tokenizer_contract"), "base tokenizer contract"
    )
    if (
        tokenizer_contract.get("tokenizer_sha256") != tokenizer_sha
        or tokenizer_contract.get("ngram_canonical_map_self_sha256")
        != canonical_descriptor.get("manifest_sha256")
        or tokenizer_contract.get("ngram_canonical_ids_sha256")
        != descriptor.get("ngram_canonical_ids_sha256")
    ):
        raise RuntimeError("Base tokenizer contract and canonical sidecar disagree")

    artifact_root = release_root / "tokenizer"
    copied_tokenizer = artifact_root / "tokenizer.json"
    _copy_atomic(tokenizer, copied_tokenizer)
    payload: dict[str, Any] = {
        "envelope_schema": SEALED_SCHEMA,
        "schema": "metis.tokenizer/v1",
        "complete": True,
        "files": [
            {
                "path": copied_tokenizer.name,
                "bytes": copied_tokenizer.stat().st_size,
                "sha256": tokenizer_sha,
            }
        ],
        "metadata": {
            "vocabulary_size": 65_536,
            "tokenizer_file": copied_tokenizer.name,
            "tokenizer_sha256": tokenizer_sha,
            "base_release_sha256": descriptor["release_sha256"],
            "base_release_tokenizer_path": tokenizer.relative_to(
                base_root
            ).as_posix(),
            "base_release_tokenizer_sha256": tokenizer_sha,
            "base_release_canonical_map_path": canonical_map.relative_to(
                base_root
            ).as_posix(),
            "base_release_canonical_ids_path": canonical_ids.relative_to(
                base_root
            ).as_posix(),
            "ngram_canonical_map_self_sha256": canonical_descriptor[
                "manifest_sha256"
            ],
            "ngram_canonical_ids_sha256": descriptor[
                "ngram_canonical_ids_sha256"
            ],
        },
    }
    payload["manifest_sha256"] = json_sha256(payload)
    manifest_path = artifact_root / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return (
        {
            "state": "sealed",
            "schema": "metis.tokenizer/v1",
            "path": _relative(release_root, manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifest_sha256": payload["manifest_sha256"],
        },
        manifest_path,
        payload,
    )


def _sealed_record(
    *,
    root: Path,
    raw: Mapping[str, Any],
    expected_schema: str,
    label: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = _safe_path(
        root,
        raw.get("manifest", raw.get("path")),
        label=f"{label} manifest",
        must_exist=True,
    )
    payload = _read_json(manifest_path, label=f"{label} manifest")
    observed_self_hash = json_sha256(payload, omit=("manifest_sha256",))
    if (
        payload.get("envelope_schema") != SEALED_SCHEMA
        or payload.get("schema") != expected_schema
        or payload.get("complete") is not True
        or payload.get("manifest_sha256") != observed_self_hash
    ):
        raise RuntimeError(
            f"{label} must be a complete self-hashed {SEALED_SCHEMA} / "
            f"{expected_schema}"
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{label} manifest has no payload files")
    return (
        {
            "state": "sealed",
            "schema": expected_schema,
            "path": _relative(root, manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifest_sha256": observed_self_hash,
        },
        manifest_path,
        payload,
    )


def _validate_generation_adapter(
    *,
    manifest_path: Path,
    payload: Mapping[str, Any],
    stage_id: str,
    requirement_name: str,
) -> Path:
    metadata = _mapping(
        payload.get("metadata"), f"{stage_id}.{requirement_name} metadata"
    )
    adapter = _mapping(
        metadata.get("generation_adapter"),
        f"{stage_id}.{requirement_name} generation_adapter",
    )
    stages = adapter.get("stages")
    requirements = adapter.get("requirements")
    args = adapter.get("args")
    if (
        metadata.get("generation_adapter_present") is not True
        or adapter.get("schema") != "metis.generation-adapter/v1"
        or adapter.get("adapter_sha256")
        != json_sha256(adapter, omit=("adapter_sha256",))
        or adapter.get("output_envelope_schema") != SEALED_SCHEMA
        or not isinstance(stages, list)
        or stage_id not in stages
        or not all(isinstance(item, str) for item in stages)
        or not isinstance(requirements, list)
        or requirement_name not in requirements
        or not all(isinstance(item, str) for item in requirements)
        or not isinstance(args, list)
        or not all(isinstance(item, str) for item in args)
        or not _SHA256.fullmatch(
            str(adapter.get("executable_sha256", ""))
        )
    ):
        raise RuntimeError(
            f"{stage_id}.{requirement_name} has no valid sealed generation adapter"
        )
    executable = _safe_path(
        manifest_path.parent.resolve(),
        adapter.get("executable"),
        label=f"{stage_id}.{requirement_name} adapter executable",
        must_exist=True,
    )
    declared = {
        str(row.get("path")): row
        for row in payload.get("files", [])
        if isinstance(row, Mapping)
    }
    relative = executable.relative_to(
        manifest_path.parent.resolve()
    ).as_posix()
    record = declared.get(relative)
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != adapter.get("executable_sha256")
        or file_sha256(executable) != adapter.get("executable_sha256")
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(
            f"{stage_id}.{requirement_name} adapter executable is not sealed"
        )
    return executable


def _deferred_record(
    *,
    root: Path,
    raw: Mapping[str, Any],
    expected_schema: str,
    label: str,
) -> tuple[dict[str, Any], Path]:
    output = _safe_path(
        root,
        raw.get("manifest"),
        label=f"{label} future manifest",
        must_exist=False,
    )
    hook = _mapping(raw.get("generation_hook"), f"{label} generation_hook")
    executable = _safe_path(
        root,
        hook.get("executable"),
        label=f"{label} executable",
        must_exist=True,
    )
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"{label} executable is not executable")
    receipt = _safe_path(
        root,
        hook.get("receipt"),
        label=f"{label} future receipt",
        must_exist=False,
    )
    rank_receipts = _safe_directory(
        root,
        hook.get("rank_receipts"),
        label=f"{label} future rank-receipt directory",
    )
    args = hook.get("args", [])
    timeout_seconds = int(hook.get("timeout_seconds", 0))
    execution = hook.get("execution")
    valid_execution = bool(
        isinstance(execution, Mapping)
        and (
            (
                execution.get("protocol") == "distributed_family_v1"
                and set(execution) == {"protocol"}
            )
            or (
                execution.get("protocol") == "rank0_only_v1"
                and set(execution)
                == {"protocol", "nodes", "tasks", "gpus_per_task"}
                and int(execution.get("nodes", 0)) == 1
                and int(execution.get("tasks", 0)) == 1
                and int(execution.get("gpus_per_task", -1)) in {0, 1}
            )
        )
    )
    if (
        not isinstance(args, list)
        or not all(isinstance(item, str) for item in args)
        or not 1 <= timeout_seconds <= 7 * 24 * 60 * 60
        or not valid_execution
        or len({output, receipt, rank_receipts, executable}) != 4
    ):
        raise RuntimeError(
            f"{label} hook requires string args, bounded timeout, and execution "
            "protocol distributed_family_v1 or rank0_only_v1"
        )
    return (
        {
            "state": "deferred",
            "schema": expected_schema,
            "manifest": _relative(root, output),
            "generation_hook": {
                "executable": _relative(root, executable),
                "executable_sha256": file_sha256(executable),
                "args": list(args),
                "timeout_seconds": timeout_seconds,
                "execution": dict(execution),
                "receipt": _relative(root, receipt),
                "rank_receipts": _relative(root, rank_receipts),
            },
        },
        executable,
    )


def _payload_inventory(
    *,
    root: Path,
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    executables: Sequence[Path],
    workers: int,
) -> list[dict[str, Any]]:
    paths: dict[str, tuple[Path, int | None, str | None, str]] = {}
    for manifest_path, payload in manifests:
        key = _relative(root, manifest_path)
        paths[key] = (
            manifest_path,
            manifest_path.stat().st_size,
            file_sha256(manifest_path),
            "manifest",
        )
        for raw_record in payload["files"]:
            record = _mapping(raw_record, f"{key} payload record")
            raw_relative = record.get("path")
            if (
                not isinstance(raw_relative, str)
                or not raw_relative
                or Path(raw_relative).is_absolute()
                or ".." in Path(raw_relative).parts
            ):
                raise RuntimeError(f"{key} contains an unsafe payload path")
            payload_path = (manifest_path.parent / raw_relative).resolve()
            try:
                payload_path.relative_to(root)
                payload_path.relative_to(manifest_path.parent.resolve())
            except ValueError as exc:
                raise RuntimeError(f"{key} payload escapes its artifact root") from exc
            if payload_path.is_symlink() or not payload_path.is_file():
                raise RuntimeError(f"{key} payload is missing or a symlink: {payload_path}")
            expected_bytes = int(record.get("bytes", -1))
            expected_sha = str(record.get("sha256", "")).lower()
            if expected_bytes < 0 or not _SHA256.fullmatch(expected_sha):
                raise RuntimeError(f"{key} payload has invalid size/hash metadata")
            relative = _relative(root, payload_path)
            existing = paths.get(relative)
            row = (payload_path, expected_bytes, expected_sha, "payload")
            if existing is not None and existing[:3] != row[:3]:
                raise RuntimeError(f"Conflicting payload declarations for {relative}")
            paths[relative] = row
    for executable in executables:
        relative = _relative(root, executable)
        row = (
            executable,
            executable.stat().st_size,
            file_sha256(executable),
            "generation_executable",
        )
        existing = paths.get(relative)
        if existing is not None and existing[:3] != row[:3]:
            raise RuntimeError(f"Conflicting executable declaration for {relative}")
        paths[relative] = row

    def verify(
        item: tuple[str, tuple[Path, int | None, str | None, str]]
    ) -> dict[str, Any]:
        relative, (path, expected_bytes, expected_sha, kind) = item
        observed_bytes = path.stat().st_size
        if expected_bytes is not None and observed_bytes != expected_bytes:
            raise RuntimeError(f"Deep verification size mismatch: {relative}")
        observed_sha = file_sha256(path)
        if expected_sha is not None and observed_sha != expected_sha:
            raise RuntimeError(f"Deep verification SHA-256 mismatch: {relative}")
        return {
            "path": relative,
            "kind": kind,
            "bytes": observed_bytes,
            "sha256": observed_sha,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sorted(
            executor.map(verify, sorted(paths.items())),
            key=lambda row: row["path"],
        )


def build_posttraining_release(
    *,
    config: PortageConfig,
    spec_path: str | Path,
    workers: int = 8,
) -> dict[str, Any]:
    if workers < 1 or workers > 64:
        raise ValueError("workers must be in [1, 64]")
    umbrella_path = config.posttraining_release_index
    root = umbrella_path.parent.resolve()
    if root == config.lustre_root or not root.is_relative_to(config.lustre_root):
        raise RuntimeError("Post-training release root must be a safe Lustre child")
    root.mkdir(parents=True, exist_ok=True)
    spec_source = Path(spec_path).expanduser().resolve()
    spec = _read_json(spec_source, label="post-training build spec")
    if spec.get("schema") != BUILD_SPEC_SCHEMA:
        raise RuntimeError(f"Build spec must use schema {BUILD_SPEC_SCHEMA}")
    contract = _contract(config)
    requirements = _requirements(contract)
    contract_sha = file_sha256(config.posttraining_contract)
    if spec.get("posttraining_contract_sha256") != contract_sha:
        raise RuntimeError("Build spec is stale for the post-training contract")

    if "tokenizer_manifest" in spec:
        raise RuntimeError(
            "Build specs may not supply a tokenizer envelope; it is derived "
            "directly from the verified base data release"
        )
    tokenizer_record, tokenizer_path, tokenizer_payload = _sealed_base_tokenizer(
        config=config,
        release_root=root,
    )
    tokenizer_sha = str(tokenizer_record["manifest_sha256"])
    family_specs = _mapping(spec.get("families"), "families")
    expected_families = {"praxis", "logos"}
    if set(family_specs) != expected_families:
        raise RuntimeError("Build spec must contain exactly praxis and logos")

    indexes: dict[str, dict[str, Any]] = {}
    manifests: list[tuple[Path, Mapping[str, Any]]] = [
        (tokenizer_path, tokenizer_payload)
    ]
    executables: list[Path] = []
    for family in sorted(expected_families):
        family_spec = _mapping(family_specs[family], f"{family} spec")
        stage_specs = _mapping(
            family_spec.get("requirements"), f"{family}.requirements"
        )
        if set(stage_specs) != set(requirements):
            raise RuntimeError(
                f"{family} stage coverage differs from the post-training contract"
            )
        stage_records: dict[str, dict[str, Any]] = {}
        sealed_requirements: dict[
            tuple[str, str], tuple[Path, Mapping[str, Any]]
        ] = {}
        for stage_id, required_rows in requirements.items():
            supplied = _mapping(
                stage_specs[stage_id], f"{family}.{stage_id}"
            )
            if set(supplied) != set(required_rows):
                missing = sorted(set(required_rows) - set(supplied))
                extra = sorted(set(supplied) - set(required_rows))
                raise RuntimeError(
                    f"{family}.{stage_id} requirement coverage mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            stage_records[stage_id] = {}
            for name, requirement in required_rows.items():
                raw = _mapping(
                    supplied[name], f"{family}.{stage_id}.{name}"
                )
                state = raw.get("state")
                label = f"{family}.{stage_id}.{name}"
                checkpoint_bound = requirement.get("checkpoint_bound") is True
                if checkpoint_bound and state != "deferred":
                    raise RuntimeError(
                        f"{label} is checkpoint-bound and must use a deferred generator"
                    )
                if not checkpoint_bound and state != "sealed":
                    raise RuntimeError(
                        f"{label} is not checkpoint-bound and must be sealed "
                        "before the Portage allocation"
                    )
                if state == "sealed":
                    record, path, payload = _sealed_record(
                        root=root,
                        raw=raw,
                        expected_schema=str(requirement["schema"]),
                        label=label,
                    )
                    if (
                        requirement.get("tokenizer_bound", True)
                        and payload.get("tokenizer_sha256") != tokenizer_sha
                    ):
                        raise RuntimeError(f"{label} tokenizer lineage is stale")
                    stage_records[stage_id][name] = record
                    manifests.append((path, payload))
                    sealed_requirements[(stage_id, name)] = (
                        path,
                        payload,
                    )
                elif state == "deferred":
                    record, executable = _deferred_record(
                        root=root,
                        raw=raw,
                        expected_schema=str(requirement["schema"]),
                        label=label,
                    )
                    stage_records[stage_id][name] = record
                    executables.append(executable)
                else:
                    raise RuntimeError(f"{label} state must be sealed or deferred")
        for generated_stage, adapter_source in _BUILTIN_ADAPTER_SOURCES.items():
            if generated_stage not in stage_records:
                continue
            if not any(
                record.get("state") == "deferred"
                for record in stage_records[generated_stage].values()
            ):
                continue
            source = sealed_requirements.get(adapter_source)
            if source is None:
                raise RuntimeError(
                    f"{family}.{generated_stage} has no sealed adapter source "
                    f"{adapter_source[0]}.{adapter_source[1]}"
                )
            adapter_path, adapter_payload = source
            executables.append(
                _validate_generation_adapter(
                    manifest_path=adapter_path,
                    payload=adapter_payload,
                    stage_id=generated_stage,
                    requirement_name=next(
                        name
                        for name, record in stage_records[
                            generated_stage
                        ].items()
                        if record.get("state") == "deferred"
                    ),
                )
            )
        index: dict[str, Any] = {
            "schema": INDEX_SCHEMA,
            "family": family,
            "pipeline_sha256": contract_sha,
            "tokenizer_manifest": {
                key: value
                for key, value in tokenizer_record.items()
                if key != "state"
            },
            "requirements": stage_records,
        }
        index["index_sha256"] = json_sha256(index)
        index_path = root / f"{family.upper()}_RELEASE_INDEX.json"
        atomic_write_json(index_path, index)
        indexes[family] = {
            "path": _relative(root, index_path),
            "sha256": file_sha256(index_path),
            "index_sha256": index["index_sha256"],
        }

    inventory = _payload_inventory(
        root=root,
        manifests=manifests,
        executables=executables,
        workers=workers,
    )
    deep_receipt: dict[str, Any] = {
        "schema": DEEP_RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "posttraining_contract_sha256": contract_sha,
        "build_spec_sha256": file_sha256(spec_source),
        "tokenizer_manifest_sha256": tokenizer_sha,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "complete": True,
    }
    deep_receipt["receipt_sha256"] = json_sha256(deep_receipt)
    deep_path = root / "DEEP_VERIFICATION.json"
    atomic_write_json(deep_path, deep_receipt)

    umbrella: dict[str, Any] = {
        "schema": UMBRELLA_SCHEMA,
        "created_at": utc_now(),
        "posttraining_contract_sha256": contract_sha,
        "families": indexes,
        "deep_verification": {
            "path": _relative(root, deep_path),
            "sha256": file_sha256(deep_path),
            "receipt_sha256": deep_receipt["receipt_sha256"],
        },
    }
    umbrella["umbrella_sha256"] = json_sha256(umbrella)
    atomic_write_json(umbrella_path, umbrella)
    report = inspect_posttraining_release_index(config)
    require_posttraining_release_index(report)
    return {
        "schema": "metis.posttraining-release-build-result/v1",
        "release_root": str(root),
        "umbrella": str(umbrella_path),
        "umbrella_sha256": umbrella["umbrella_sha256"],
        "deep_verification": str(deep_path),
        "deep_verification_receipt_sha256": deep_receipt["receipt_sha256"],
        "family_indexes": indexes,
        "preflight_sha256": report["preflight_sha256"],
        "ok": True,
    }


def posttraining_build_template(config: PortageConfig) -> dict[str, Any]:
    """Return a complete, convention-based build specification to fill on Rhea."""

    requirements = _requirements(_contract(config))
    release_root = config.posttraining_release_index.parent.resolve()
    builtin_hook = _install_builtin_generation_hook(
        config=config,
        release_root=release_root,
    )
    builtin_hook_relative = _relative(release_root, builtin_hook)
    families: dict[str, Any] = {}
    for family in ("praxis", "logos"):
        stage_rows: dict[str, Any] = {}
        for stage_id, required_rows in requirements.items():
            stage_rows[stage_id] = {}
            for name, requirement in required_rows.items():
                if requirement.get("checkpoint_bound") is True:
                    generated_root = f"generated/{family}/{stage_id}/{name}"
                    adapter_source = _BUILTIN_ADAPTER_SOURCES.get(stage_id)
                    use_builtin = adapter_source is not None
                    if use_builtin:
                        adapter_stage, adapter_requirement = adapter_source
                        hook_args = [
                            "--adapter-stage",
                            adapter_stage,
                            "--adapter-requirement",
                            adapter_requirement,
                        ]
                        hook_executable = builtin_hook_relative
                    else:
                        hook_args = []
                        hook_executable = f"hooks/{stage_id}-{name}.py"
                    stage_rows[stage_id][name] = {
                        "state": "deferred",
                        "manifest": f"{generated_root}/MANIFEST.json",
                        "generation_hook": {
                            "executable": hook_executable,
                            "args": hook_args,
                            "timeout_seconds": (
                                345_600 if use_builtin else 86_400
                            ),
                            "execution": {
                                "protocol": "distributed_family_v1"
                            },
                            "receipt": (
                                f"{generated_root}/GENERATION_RECEIPT.json"
                            ),
                            "rank_receipts": (
                                f"{generated_root}/rank-receipts"
                            ),
                        },
                    }
                else:
                    stage_rows[stage_id][name] = {
                        "state": "sealed",
                        "manifest": (
                            f"artifacts/{stage_id}/{name}/MANIFEST.json"
                        ),
                    }
        families[family] = {"requirements": stage_rows}
    return {
        "schema": BUILD_SPEC_SCHEMA,
        "posttraining_contract_sha256": file_sha256(
            config.posttraining_contract
        ),
        "families": families,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal and byte-verify the Rhea post-training release consumed by "
            "the autonomous Portage campaign."
        )
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec")
    mode.add_argument("--write-template")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_portage_config(args.config)
    if args.write_template:
        output = Path(args.write_template).expanduser().resolve()
        atomic_write_json(output, posttraining_build_template(config))
        print(
            json.dumps(
                {
                    "schema": "metis.posttraining-release-template-result/v1",
                    "path": str(output),
                    "sha256": file_sha256(output),
                    "ok": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = build_posttraining_release(
        config=config,
        spec_path=str(args.spec),
        workers=args.workers,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
