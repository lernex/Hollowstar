from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PHASE_DIRECTORIES = {"phase_a": "phase-a", "phase_b": "phase-b", "phase_c": "phase-c"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_release(release_root: str | Path, contract_path: str | Path) -> dict[str, Any]:
    root = Path(release_root).expanduser().resolve()
    contract = yaml.safe_load(Path(contract_path).expanduser().resolve().read_text(encoding="utf-8"))
    release_path = root / "RELEASE.json"
    if not release_path.exists():
        raise RuntimeError(f"Verified data release is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != "metis.data-release/v1":
        raise RuntimeError("Unexpected data release schema")
    if release.get("release") != contract.get("data_release"):
        raise RuntimeError(f"Data release mismatch: {release.get('release')} != {contract.get('data_release')}")
    if int(release.get("target_tokens", 0)) != int(contract.get("total_train_tokens", 0)):
        raise RuntimeError("Training contract and release token targets differ")
    if not release.get("verification", {}).get("ok"):
        raise RuntimeError("Training refuses an unverified data release")
    tokenizer_path = root / release["artifacts"]["tokenizer"]
    if _sha256(tokenizer_path) != release["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer hash does not match RELEASE.json")
    shard_manifest_path = root / release["artifacts"]["shard_manifest"]
    expected_manifest_sha = release["verification"]["shard_manifest_sha256"]
    if _sha256(shard_manifest_path) != expected_manifest_sha:
        raise RuntimeError("Shard manifest hash does not match RELEASE.json")
    shard_rows = [json.loads(line) for line in shard_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(shard_rows) != int(release["verification"]["shards"]):
        raise RuntimeError("Shard manifest row count does not match verification report")
    for row in shard_rows:
        binary = root / row["binary"]
        index = root / row["index"]
        if binary.stat().st_size != int(row["tokens"]) * 2:
            raise RuntimeError(f"Shard byte size mismatch: {binary}")
        if _sha256(binary) != row["binary_sha256"] or _sha256(index) != row["index_sha256"]:
            raise RuntimeError(f"Shard or index hash mismatch: {binary}")

    phase_rows = []
    expected_cursor = 0
    for phase in contract["phases"]:
        phase_id = phase["id"]
        start = int(phase["start_token"])
        end = int(phase["end_token_exclusive"])
        if start != expected_cursor or end <= start:
            raise RuntimeError(f"Training phases are not contiguous at {phase_id}")
        expected = int(release["phase_tokens"][phase_id])
        if end - start != expected:
            raise RuntimeError(f"Phase token mismatch for {phase_id}: {end - start:,} != {expected:,}")
        phase_root = root / phase["data_directory"]
        binaries = sorted(phase_root.glob("shard-*.bin"))
        indices = sorted(phase_root.glob("shard-*.index.jsonl"))
        if len(binaries) != len(indices) or not binaries:
            raise RuntimeError(f"Phase {phase_id} has incomplete shard/index pairs")
        bytes_on_disk = sum(path.stat().st_size for path in binaries)
        if bytes_on_disk != expected * 2:
            raise RuntimeError(f"Phase {phase_id} uint16 byte count mismatch")
        phase_rows.append({"id": phase_id, "tokens": expected, "shards": len(binaries), "bytes": bytes_on_disk})
        expected_cursor = end
    if expected_cursor != int(contract["total_train_tokens"]):
        raise RuntimeError("Phase boundaries do not end at total_train_tokens")
    return {
        "ok": True,
        "release": release["release"],
        "release_root": str(root),
        "tokenizer": str(tokenizer_path),
        "sequence_length": int(contract["sequence_length"]),
        "phases": phase_rows,
        "total_tokens": expected_cursor,
    }


def phase_for_token(contract_path: str | Path, token_cursor: int) -> str:
    contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    for phase in contract["phases"]:
        if int(phase["start_token"]) <= token_cursor < int(phase["end_token_exclusive"]):
            return str(phase["id"])
    raise ValueError(f"Token cursor {token_cursor:,} is outside the training schedule")
