from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from metis_data.ngram_canonical import validate_canonical_id_sidecar

from .distributed import DistributedContext
from .util import atomic_write_json, file_sha256, json_sha256, read_json, utc_now


PHASE_TOKENS = {
    "phase_a": 700_000_000_000,
    "phase_b": 250_000_000_000,
    "phase_c": 50_000_000_000,
}
METADATA_HASH_FIELDS = {
    "verification": "verification_file_sha256",
    "selection": "selection_sha256",
    "token_count_contract": "token_count_contract_sha256",
    "filter_chain": "filter_chain_sha256",
    "tokenizer": "tokenizer_sha256",
    "ngram_canonical_map": "ngram_canonical_map_manifest_sha256",
    "ngram_canonical_ids": "ngram_canonical_ids_sha256",
    "source_lock": "source_lock_sha256",
    "build_inputs": "build_inputs_sha256",
    "manifest": "manifest_sha256",
    "license_ledger": "license_ledger_sha256",
    "shard_manifest": "shard_manifest_sha256",
}


def _safe_artifact(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RuntimeError(f"Release artifact {label!r} has an unsafe path")
    unresolved = root / value
    if unresolved.is_symlink():
        raise RuntimeError(f"Release artifact {label!r} may not be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Release artifact {label!r} escapes the release root") from exc
    if not resolved.is_file():
        raise RuntimeError(f"Release artifact {label!r} is missing: {resolved}")
    return resolved


def _unsigned_hash(value: dict[str, Any], field: str) -> str:
    return json_sha256(value, omit=(field,))


def _metadata_and_rows(
    release_root: str | Path,
    training_contract_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]], dict[str, str]]:
    root = Path(release_root).expanduser().resolve()
    contract_path = Path(training_contract_path).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Verified data release is missing: {root}")
    release_path = root / "RELEASE.json"
    if release_path.is_symlink() or not release_path.is_file():
        raise RuntimeError(f"Release contract is missing or a symlink: {release_path}")
    release = read_json(release_path)
    if (
        release.get("schema") != "metis.data-release/v2"
        or release.get("release") != "metis-1.6-data-r1"
        or release.get("release_sha256") != _unsigned_hash(release, "release_sha256")
        or release.get("token_dtype") != "uint16"
        or release.get("token_endianness") != "little"
        or int(release.get("target_tokens", -1)) != 1_000_000_000_000
        or release.get("phase_tokens") != PHASE_TOKENS
        or release.get("verification", {}).get("ok") is not True
    ):
        raise RuntimeError("RELEASE.json does not satisfy the immutable Metis-1.6 contract")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "metis.pretraining-contract/v1"
        or contract.get("data_release") != release.get("release")
        or int(contract.get("total_train_tokens", -1)) != 1_000_000_000_000
        or contract.get("token_dtype") != "uint16"
        or int(contract.get("tokenizer_vocabulary_size", -1)) != 65_536
    ):
        raise RuntimeError("Training contract and immutable release disagree")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("RELEASE.json is missing its artifact inventory")
    artifact_paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, hash_field in METADATA_HASH_FIELDS.items():
        path = _safe_artifact(root, artifacts.get(name), label=name)
        observed = file_sha256(path)
        expected = str(release.get(hash_field, ""))
        if observed != expected:
            raise RuntimeError(f"Release metadata hash mismatch for {name}: {path}")
        artifact_paths[name] = path
        hashes[name] = observed
    tokenizer_contract = release.get("tokenizer_contract", {})
    for name, hash_field in (
        ("tokenizer_vocab", "vocab_sha256"),
        ("tokenizer_release", "tokenizer_release_sha256"),
        ("tokenizer_validation", "tokenizer_validation_sha256"),
    ):
        path = _safe_artifact(root, artifacts.get(name), label=name)
        observed = file_sha256(path)
        if observed != tokenizer_contract.get(hash_field):
            raise RuntimeError(f"Tokenizer release hash mismatch for {name}")
        hashes[name] = observed
    if (
        hashes["ngram_canonical_map"]
        != tokenizer_contract.get("ngram_canonical_map_manifest_sha256")
        or hashes["ngram_canonical_ids"]
        != tokenizer_contract.get("ngram_canonical_ids_sha256")
        or release.get("ngram_canonical_map_self_sha256")
        != tokenizer_contract.get("ngram_canonical_map_self_sha256")
    ):
        raise RuntimeError(
            "Canonical-ID release hashes disagree with the tokenizer contract"
        )
    canonical_descriptor, _canonical_ids = validate_canonical_id_sidecar(
        manifest_path=artifact_paths["ngram_canonical_map"],
        binary_path=artifact_paths["ngram_canonical_ids"],
        tokenizer_path=artifact_paths["tokenizer"],
        expected_vocabulary_size=65_536,
        expected_manifest_sha256=tokenizer_contract.get(
            "ngram_canonical_map_self_sha256"
        ),
        expected_binary_sha256=tokenizer_contract.get(
            "ngram_canonical_ids_sha256"
        ),
        recompute_from_tokenizer=False,
    )
    if (
        tokenizer_contract.get("ngram_canonicalization_algorithm")
        != canonical_descriptor.get("algorithm")
        or tokenizer_contract.get("ngram_canonical_dtype") != "uint16"
        or tokenizer_contract.get("ngram_canonical_endianness") != "little"
        or int(tokenizer_contract.get("ngram_canonical_entry_count", -1))
        != 65_536
        or int(tokenizer_contract.get("ngram_canonical_vocabulary_size", -1))
        != int(canonical_descriptor.get("canonical_vocabulary_size", -2))
    ):
        raise RuntimeError("Canonical-ID tokenizer contract is invalid")
    verification = read_json(artifact_paths["verification"])
    if (
        verification != release["verification"]
        or verification.get("schema") != "metis.verification/v2"
        or verification.get("verification_sha256")
        != _unsigned_hash(verification, "verification_sha256")
    ):
        raise RuntimeError("Released verification receipt is invalid")
    rows = [
        json.loads(line)
        for line in artifact_paths["shard_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or {int(row.get("task_index", -1)) for row in rows} != set(range(len(rows))):
        raise RuntimeError("Shard task indices are not unique and contiguous")
    totals = {phase: 0 for phase in PHASE_TOKENS}
    seen_paths: set[Path] = set()
    for row in rows:
        phase = str(row.get("phase", ""))
        if phase not in totals:
            raise RuntimeError(f"Unknown shard phase: {phase!r}")
        task_index = int(row["task_index"])
        phase_index = int(row.get("phase_index", -1))
        binary = _safe_artifact(root, row.get("binary"), label=f"binary[{task_index}]")
        index = _safe_artifact(root, row.get("index"), label=f"index[{task_index}]")
        phase_root = (root / {"phase_a": "phase-a", "phase_b": "phase-b", "phase_c": "phase-c"}[phase]).resolve()
        try:
            binary.relative_to(phase_root)
            index.relative_to(phase_root)
        except ValueError as exc:
            raise RuntimeError(f"Shard {task_index} escapes its phase directory") from exc
        if (
            binary.name != f"shard-{phase_index:05d}.bin"
            or index.name != f"shard-{phase_index:05d}.index.jsonl"
            or binary in seen_paths
            or index in seen_paths
        ):
            raise RuntimeError(f"Shard {task_index} filename or uniqueness contract failed")
        seen_paths.update((binary, index))
        tokens = int(row.get("tokens", -1))
        if tokens < 1 or binary.stat().st_size != tokens * 2:
            raise RuntimeError(f"Shard {task_index} byte size is not uint16-exact")
        totals[phase] += tokens
    if totals != PHASE_TOKENS:
        raise RuntimeError(f"Shard phase token totals are wrong: {totals}")
    binding = {
        "release_json_sha256": file_sha256(release_path),
        "release_self_sha256": str(release["release_sha256"]),
        "training_contract_sha256": file_sha256(contract_path),
        "shard_manifest_sha256": hashes["shard_manifest"],
        "shard_inventory_sha256": json_sha256(
            [
                {
                    "task_index": int(row["task_index"]),
                    "phase": str(row["phase"]),
                    "binary": str(row["binary"]),
                    "binary_sha256": str(row["binary_sha256"]),
                    "index": str(row["index"]),
                    "index_sha256": str(row["index_sha256"]),
                    "tokens": int(row["tokens"]),
                }
                for row in sorted(rows, key=lambda item: int(item["task_index"]))
            ]
        ),
    }
    return release, contract, artifact_paths["shard_manifest"], rows, binding


def verify_release_distributed(
    *,
    release_root: str | Path,
    training_contract_path: str | Path,
    output_path: str | Path,
    receipt_directory: str | Path,
    context: DistributedContext,
) -> dict[str, Any] | None:
    """Hash every released token/index shard exactly once across all ranks."""

    root = Path(release_root).expanduser().resolve()
    contract_path = Path(training_contract_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    receipts = Path(receipt_directory).expanduser().resolve()
    release, _contract, shard_manifest_path, rows, binding = _metadata_and_rows(
        root,
        contract_path,
    )
    assigned = [
        row
        for row in rows
        if int(row["task_index"]) % context.world_size == context.rank
    ]
    verified_rows: list[int] = []
    bytes_hashed = 0
    for row in assigned:
        task_index = int(row["task_index"])
        binary = _safe_artifact(root, row["binary"], label=f"binary[{task_index}]")
        index = _safe_artifact(root, row["index"], label=f"index[{task_index}]")
        if file_sha256(binary) != row["binary_sha256"]:
            raise RuntimeError(f"Binary shard hash mismatch at task {task_index}: {binary}")
        if file_sha256(index) != row["index_sha256"]:
            raise RuntimeError(f"Index shard hash mismatch at task {task_index}: {index}")
        bytes_hashed += binary.stat().st_size + index.stat().st_size
        verified_rows.append(task_index)
    receipt: dict[str, Any] = {
        "schema": "metis.portage-release-rank-receipt/v1",
        "rank": context.rank,
        "world_size": context.world_size,
        "task_indices": verified_rows,
        "bytes_hashed": bytes_hashed,
        "release_json_sha256": binding["release_json_sha256"],
        "shard_manifest_sha256": binding["shard_manifest_sha256"],
        "verified_at": utc_now(),
    }
    receipt["receipt_sha256"] = json_sha256(receipt)
    receipt_path = receipts / f"rank-{context.rank:05d}.json"
    atomic_write_json(receipt_path, receipt)

    if context.initialized:
        import torch.distributed as dist

        gathered: list[dict[str, Any] | None] = [None] * context.world_size
        dist.all_gather_object(gathered, receipt)
    else:
        gathered = [receipt]
    if not context.is_root:
        return None
    concrete = [item for item in gathered if isinstance(item, dict)]
    if len(concrete) != context.world_size:
        raise RuntimeError("Not every release-verification rank returned a receipt")
    observed_tasks = [
        int(task)
        for item in concrete
        for task in item.get("task_indices", [])
    ]
    if (
        sorted(observed_tasks) != list(range(len(rows)))
        or len(observed_tasks) != len(set(observed_tasks))
    ):
        raise RuntimeError("Distributed release verification did not cover every shard exactly once")
    for item in concrete:
        if (
            item.get("receipt_sha256") != json_sha256(item, omit=("receipt_sha256",))
            or item.get("release_json_sha256") != binding["release_json_sha256"]
            or item.get("shard_manifest_sha256") != binding["shard_manifest_sha256"]
        ):
            raise RuntimeError(f"Invalid release receipt from rank {item.get('rank')}")
    marker: dict[str, Any] = {
        "schema": "metis.portage-release-verification/v1",
        "release_root": str(root),
        "release_json_sha256": binding["release_json_sha256"],
        "release_self_sha256": binding["release_self_sha256"],
        "training_contract_path": str(contract_path),
        "training_contract_sha256": binding["training_contract_sha256"],
        "shard_manifest_path": str(shard_manifest_path),
        "shard_manifest_sha256": binding["shard_manifest_sha256"],
        "shard_inventory_sha256": binding["shard_inventory_sha256"],
        "world_size": context.world_size,
        "shard_count": len(rows),
        "total_tokens": int(release["target_tokens"]),
        "phase_tokens": release["phase_tokens"],
        "verified_at": utc_now(),
        "task_receipt_sha256s": [
            str(item["receipt_sha256"])
            for item in sorted(concrete, key=lambda row: int(row["rank"]))
        ],
    }
    marker["marker_sha256"] = json_sha256(marker)
    atomic_write_json(output, marker)
    return marker


def validate_release_marker(
    marker_path: str | Path,
    *,
    expected_release_root: str | Path,
    expected_contract_path: str | Path,
    expected_posttraining_preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    marker = read_json(marker_path)
    root = Path(expected_release_root).expanduser().resolve()
    contract = Path(expected_contract_path).expanduser().resolve()
    if (
        marker.get("schema") != "metis.portage-release-verification/v1"
        or marker.get("marker_sha256") != json_sha256(marker, omit=("marker_sha256",))
        or Path(marker.get("release_root", "")).resolve() != root
        or Path(marker.get("training_contract_path", "")).resolve() != contract
        or int(marker.get("total_tokens", -1)) != 1_000_000_000_000
        or marker.get("phase_tokens") != PHASE_TOKENS
    ):
        raise RuntimeError("Distributed release marker contract is invalid")
    release_path = root / "RELEASE.json"
    if (
        file_sha256(release_path) != marker["release_json_sha256"]
        or file_sha256(contract) != marker["training_contract_sha256"]
        or file_sha256(marker["shard_manifest_path"]) != marker["shard_manifest_sha256"]
    ):
        raise RuntimeError("Distributed release marker is stale")
    if expected_posttraining_preflight_path is not None:
        preflight_path = (
            Path(expected_posttraining_preflight_path).expanduser().resolve()
        )
        posttraining = marker.get("posttraining")
        if (
            not isinstance(posttraining, dict)
            or posttraining.get("schema")
            != "metis.portage-posttraining-release-verification/v1"
            or posttraining.get("ok") is not True
            or posttraining.get("marker_sha256")
            != json_sha256(posttraining, omit=("marker_sha256",))
            or Path(str(posttraining.get("preflight_path", ""))).resolve()
            != preflight_path
            or not preflight_path.is_file()
            or preflight_path.is_symlink()
            or file_sha256(preflight_path)
            != posttraining.get("preflight_file_sha256")
        ):
            raise RuntimeError(
                "Distributed post-training release marker is invalid or stale"
            )
        preflight = read_json(preflight_path)
        if (
            preflight.get("preflight_sha256")
            != posttraining.get("preflight_sha256")
            or preflight.get("preflight_sha256")
            != json_sha256(preflight, omit=("preflight_sha256",))
        ):
            raise RuntimeError(
                "Post-training deep audit is bound to a different preflight"
            )
    return marker


def validate_release_fast(
    release_root: str | Path,
    training_contract_path: str | Path,
) -> dict[str, Any]:
    """Validate release metadata, exact shard inventory, paths, and byte sizes.

    This intentionally does not hash the 2 TB token payload.  The distributed
    ``verify_release_distributed`` gate performs that work after submission.
    """

    release, _contract, shard_manifest, rows, binding = _metadata_and_rows(
        release_root,
        training_contract_path,
    )
    return {
        "schema": "metis.portage-release-fast-check/v1",
        "release_root": str(Path(release_root).expanduser().resolve()),
        "release_json_sha256": binding["release_json_sha256"],
        "release_self_sha256": binding["release_self_sha256"],
        "training_contract_sha256": binding["training_contract_sha256"],
        "shard_manifest_path": str(shard_manifest),
        "shard_manifest_sha256": binding["shard_manifest_sha256"],
        "shard_inventory_sha256": binding["shard_inventory_sha256"],
        "shard_count": len(rows),
        "total_tokens": int(release["target_tokens"]),
        "phase_tokens": release["phase_tokens"],
        "checked_at": utc_now(),
        "ok": True,
    }
