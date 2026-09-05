from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .common import digest_json, read_receipt, sha256_file, under_root, write_receipt
from .dedup_locks import metadata_lock
from .dedup_storage import storage_namespace


PREPARED_SCHEMAS = {"metis17.prepared-object/v1", "metis17.prepared-chunk/v1"}


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Receipt proof must contain a lowercase SHA-256 digest")
    return value


def _read_snapshot(
    path: Path, expected_sha256: str | None = None, expected_file_sha256: str | None = None,
):
    data = path.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    payload = read_receipt(path)
    payload_sha256 = digest_json(payload)
    sealed = {**payload, "receipt_sha256": payload_sha256}
    if json.loads(data) != sealed:
        raise ValueError("Receipt changed while pinning its generation")
    if expected_sha256 is not None and payload_sha256 != _digest(expected_sha256):
        raise ValueError(
            "Prepared stage receipt digest differs from its canonical seal; "
            "use receipt_file_sha256 to pin the full JSON file"
        )
    if expected_file_sha256 is not None and checksum != _digest(expected_file_sha256):
        raise ValueError("receipt_file_sha256 full-file checksum mismatch")
    return payload, checksum, data


def _archive(
    directory: Path | None, payload: dict[str, Any], checksum: str, data: bytes,
    working_budget: Any = None,
):
    if directory is None:
        return None
    path = directory / checksum[:2] / f"{checksum}.json"
    if working_budget is not None and not path.exists():
        blob = directory.parent / "receipt-blobs" / checksum[:2] / checksum
        path = blob / "receipt.json"
        with metadata_lock(directory.parent / "locks" / f"receipt-{checksum}"):
            with storage_namespace(working_budget, "receipt-blob", blob) as quota:
                if path.exists():
                    if sha256_file(path) != checksum:
                        raise ValueError("Archived stage receipt checksum mismatch")
                else:
                    quota.write_bytes(path, data)
                    path.chmod(0o444)
    elif path.exists():
        if sha256_file(path) != checksum:
            raise ValueError("Archived stage receipt checksum mismatch")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{checksum}.{uuid.uuid4().hex}.part")
        try:
            with pending.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, path)
            path.chmod(0o444)
        finally:
            pending.unlink(missing_ok=True)
    descriptor = {
        "path": str(path), "sha256": checksum, "payload_sha256": digest_json(payload),
    }
    payload_sha = descriptor["payload_sha256"]
    locators = directory.parent / "receipt-locators" if working_budget is not None else directory / "payloads"
    write_receipt(locators / payload_sha[:2] / f"{payload_sha}.json", descriptor)
    return descriptor


def _relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Prepared receipt paths must be nonempty release-relative paths")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Prepared receipt paths must remain below the release root")
    return path


def _release_root(receipt: dict[str, Any], path: Path) -> tuple[Path, Path]:
    relative = _relative(receipt.get("receipt_path"))
    if path.parent.name == "accepted-receipts":
        if path.name != f"{digest_json(receipt)}.json":
            raise ValueError("Accepted receipt snapshot filename must match its canonical seal")
        return path.parent.parent, path
    root = path
    for _ in relative.parts:
        root = root.parent
    if (root / relative).resolve() == path:
        return root, path
    if (
        path.name == "PREP_COMPLETE.json" and relative.name == "ELIGIBLE.json"
        and relative.parent.parent.name == "eligible"
    ):
        root = path.parent
        for _ in relative.parent.parent.parent.parts:
            root = root.parent
        sealed = under_root(root, str(relative))
        if read_receipt(sealed) == receipt:
            return root, sealed
    raise ValueError("Stage receipt path does not identify its declared release generation")


def _completion_proofs(
    receipt: dict[str, Any], root: Path, archive_dir: Path | None, working_budget: Any = None,
):
    if receipt["schema"] != "metis17.prepared-chunk/v1":
        return []
    proof = receipt.get("object_completion")
    if receipt.get("object_complete") is not True or not isinstance(proof, dict):
        raise ValueError("Eligible chunk requires sealed raw-object completion evidence")
    completion_path = under_root(root, str(_relative(proof.get("path"))))
    manifest, checksum, data = _read_snapshot(completion_path)
    if digest_json(manifest) != _digest(proof.get("receipt_sha256")):
        raise ValueError("Raw-object completion receipt hash mismatch")
    if (
        manifest.get("schema") != "metis17.normalized-object/v1"
        or manifest.get("status") != "NORMALIZED" or manifest.get("reblock_complete") is not True
        or any(manifest.get(name) != receipt.get(name)
               for name in ("source_id", "object_id", "normalization_fingerprint"))
    ):
        raise ValueError("Eligible chunk has an incomplete or foreign raw-object completion receipt")
    index = receipt.get("chunk_index")
    chunks, ready_paths = manifest.get("chunks", []), manifest.get("chunk_receipts", [])
    if (
        type(index) is not int or index < 0 or index >= len(chunks)
        or len(ready_paths) != len(chunks)
    ):
        raise ValueError("Object completion does not cover this chunk")
    chunk = chunks[index]
    if (
        chunk.get("chunk_id") != receipt.get("chunk_id")
        or chunk.get("path") != receipt.get("base_chunk")
        or chunk.get("ready_receipt") != receipt.get("base_chunk_receipt")
        or ready_paths[index] != receipt.get("base_chunk_receipt")
    ):
        raise ValueError("Object completion references a different base chunk")
    ready_path = under_root(root, str(_relative(receipt.get("base_chunk_receipt"))))
    ready, ready_sha, ready_data = _read_snapshot(ready_path)
    if (
        ready.get("schema") != "metis17.base-chunk/v1"
        or ready.get("status") != "NORMALIZED_CHUNK_READY"
        or ready.get("chunk") != chunk or ready.get("chunk_index") != index
        or ready.get("chunk_id") != receipt.get("chunk_id")
        or any(ready.get(name) != receipt.get(name)
               for name in ("source_id", "object_id", "normalization_fingerprint"))
        or digest_json(ready) != receipt.get("inputs", {}).get("base_chunk_receipt_sha256")
    ):
        raise ValueError("Base chunk receipt differs from the eligible generation")
    return [
        item for item in (
            _archive(archive_dir, manifest, checksum, data, working_budget),
            _archive(archive_dir, ready, ready_sha, ready_data, working_budget),
        ) if item is not None
    ]


def eligible_stage(
    path: Path, *, expected_sha256: str | None = None, archive_dir: Path | None = None,
    source_path: Path | None = None, expected_file_sha256: str | None = None,
    working_budget: Any = None,
) -> dict[str, Any]:
    path = path.resolve()
    receipt, checksum, data = _read_snapshot(
        path, expected_sha256=expected_sha256, expected_file_sha256=expected_file_sha256,
    )
    if (
        receipt.get("schema") not in PREPARED_SCHEMAS or receipt.get("status") != "ELIGIBLE"
        or receipt.get("eligible") is not True or receipt.get("training_ready") is not True
        or receipt.get("pending_reasons", []) != []
    ):
        raise ValueError("Only a finalized eligible stage receipt can enter authoritative dedup")
    inventory = receipt.get("chunks")
    if not isinstance(inventory, list):
        raise ValueError("Eligible receipt must enumerate chunks, never screened/normalized fallbacks")
    total, seen = 0, set()
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Malformed eligible chunk inventory")
        relative = str(_relative(item.get("path")))
        if relative in seen:
            raise ValueError("Duplicate eligible inventory paths")
        seen.add(relative)
        for name in ("records", "byte_count"):
            if type(item.get(name)) is not int or item[name] < 0:
                raise ValueError(f"Eligible inventory requires a nonnegative {name}")
        _digest(item.get("sha256"))
        total += item["records"]
    if type(receipt.get("eligible_documents")) is not int or total != receipt["eligible_documents"]:
        raise ValueError("Eligible receipt inventory does not cover its declared rows")
    source_path = path if source_path is None else source_path.resolve()
    root, canonical_path = _release_root(receipt, source_path)
    if canonical_path != source_path and sha256_file(canonical_path) != checksum:
        canonical_path = source_path
    proofs = _completion_proofs(receipt, root, archive_dir, working_budget)
    snapshot = _archive(archive_dir, receipt, checksum, data, working_budget)
    return {
        "receipt": receipt, "root": root, "path": canonical_path,
        "origin_path": root / _relative(receipt.get("receipt_path")),
        "sha256": digest_json(receipt), "file_sha256": checksum, "payload_sha256": digest_json(receipt),
        "snapshot": snapshot, "completion_proofs": proofs,
        "inventory": {under_root(root, item["path"]): item for item in inventory},
    }
