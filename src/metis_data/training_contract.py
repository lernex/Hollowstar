from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

from .manifest import validate_manifest
from .ngram_canonical import validate_canonical_id_sidecar
from .tokenizer import tokenizer_split_digits_setting, tokenizer_splits_digits


PHASE_DIRECTORIES = {
    "phase_a": "phase-a",
    "phase_b": "phase-b",
    "phase_c": "phase-c",
}


def phase_tokens_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    """The per-phase totals the release must hit, taken from the manifest.

    Was a module constant pinned to 700B/250B/50B. That made the schedule a
    property of the validator rather than of the release being validated, so
    refitting the mix to measured supply -- which is the normal outcome of
    acquiring less than planned -- failed here, at verify, after the entire
    corpus had been built. The manifest is the single source of truth and
    validate_manifest already checks it is internally consistent.
    """

    phases = manifest["schedule"]["phases"]
    return {
        phase: int(phases[phase]["unique_tokens"]) + int(phases[phase]["replay_tokens"])
        for phase in PHASE_DIRECTORIES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_digests(rows: list[tuple[Path, str]]) -> list[tuple[Path, bool]]:
    """Check many independent digests with the node rather than one core.

    Validating the release re-hashes every shard binary and index: 806 shards
    and about 2TB. Measured on Portage from the running stage, one stream did
    289 MB/s, which is an hour and a half of a 192-core node with the rest of
    the build waiting on it. hashlib releases the GIL around each buffer, so
    threads are enough and nothing needs pickling; the caller still raises on
    the first mismatch in manifest order, so the failure is identical.
    """

    if not rows:
        return []
    if int(os.environ.get("METIS_TASKS_PER_JOB", "1") or 1) > 1:
        return [(path, _sha256(path) == expected) for path, expected in rows]
    workers = max(1, min(32, len(rows), os.cpu_count() or 1))
    if workers == 1:
        return [(path, _sha256(path) == expected) for path, expected in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(lambda row: _sha256(row[0]), rows))
    return [
        (path, digest == expected)
        for (path, expected), digest in zip(rows, digests)
    ]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"Manifest bundle may not contain symlinks: {candidate}")
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _artifact(
    root: Path,
    artifacts: dict[str, Any],
    name: str,
    *,
    directory: bool = False,
) -> Path:
    raw = artifacts.get(name)
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise RuntimeError(f"Release artifact {name!r} is not a safe relative path")
    unresolved = root / raw
    if unresolved.is_symlink():
        raise RuntimeError(f"Release artifact {name!r} may not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Release artifact {name!r} escapes the release root") from exc
    if directory:
        if not path.is_dir():
            raise RuntimeError(f"Release artifact directory is missing: {path}")
    elif not path.is_file():
        raise RuntimeError(f"Release artifact is missing: {path}")
    return path


def _validate_continued_pretraining(contract: dict) -> None:
    """The base model is the 1T checkpoint plus continued pretraining.

    Continued pretraining owns a separate corpus, a separate token counter, and
    a separate learning-rate segment.  Its exposure is deliberately excluded
    from ``total_train_tokens`` and from the verified 1T release, so this only
    checks that the split is declared and that nobody has folded the
    long-context corpus into the pretraining phases.
    """

    boundary = list(contract.get("base_model_complete_after") or ())
    if boundary != ["phase_c", "continued_pretraining"]:
        raise RuntimeError(
            "Pretraining contract must declare the base model complete only "
            "after phase_c and continued_pretraining"
        )
    continued = contract.get("continued_pretraining")
    if not isinstance(continued, dict):
        raise RuntimeError("Pretraining contract is missing continued_pretraining")
    if continued.get("in_pretraining_release") is not False:
        raise RuntimeError(
            "Continued-pretraining data must stay outside the 1T pretraining release"
        )
    if continued.get("data_env") != "METIS_CONTEXT_EXTENSION_DATA":
        raise RuntimeError(
            "Continued pretraining must consume its own sealed long-context corpus"
        )
    if int(continued.get("token_budget", -1)) != 18_000_000_000:
        raise RuntimeError("Continued-pretraining exposure must be exactly 18B tokens")
    if any(
        int(phase.get("end_token_exclusive", 0)) > int(contract["total_train_tokens"])
        for phase in contract.get("phases", ())
    ):
        raise RuntimeError(
            "Pretraining phases must not absorb continued-pretraining exposure"
        )


def _unsigned_hash(payload: dict[str, Any], field: str) -> str:
    return _json_sha256({key: value for key, value in payload.items() if key != field})


def _manifest_contract_sha256(manifest: dict[str, Any]) -> str:
    def public(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): public(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [public(item) for item in value]
        return value

    return _json_sha256(public(manifest))


def validate_training_release(
    release_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(release_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Verified data release directory is missing: {root}")
    contract = yaml.safe_load(
        Path(contract_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    if (
        contract.get("schema") != "metis.pretraining-contract/v1"
        or contract.get("require_verified_release") is not True
        or contract.get("token_dtype") != "uint16"
        or int(contract.get("tokenizer_vocabulary_size", -1)) != 65_536
    ):
        raise RuntimeError("Pretraining contract is not the Metis-1.6 uint16/65,536 contract")

    release_path = root / "RELEASE.json"
    if not release_path.is_file():
        raise RuntimeError(f"Verified data release is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != "metis.data-release/v2":
        raise RuntimeError("Unexpected data release schema")
    if release.get("release_sha256") != _unsigned_hash(release, "release_sha256"):
        raise RuntimeError("RELEASE.json failed its self-hash check")
    if release.get("release") != contract.get("data_release"):
        raise RuntimeError(
            f"Data release mismatch: {release.get('release')} != {contract.get('data_release')}"
        )
    if (
        release.get("token_dtype") != "uint16"
        or release.get("token_endianness") != "little"
    ):
        raise RuntimeError("Release is not explicitly little-endian uint16")
    if int(release.get("target_tokens", 0)) != int(
        contract.get("total_train_tokens", 0)
    ):
        raise RuntimeError(
            "Training contract and release token targets differ: "
            f"release {int(release.get('target_tokens', 0)):,} against contract "
            f"{int(contract.get('total_train_tokens', 0)):,}"
        )
    _validate_continued_pretraining(contract)
    if not release.get("verification", {}).get("ok"):
        raise RuntimeError("Training refuses an unverified data release")

    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("RELEASE.json is missing its artifact map")
    verification_path = _artifact(root, artifacts, "verification")
    selection_path = _artifact(root, artifacts, "selection")
    token_count_path = _artifact(root, artifacts, "token_count_contract")
    filter_chain_path = _artifact(root, artifacts, "filter_chain")
    tokenizer_path = _artifact(root, artifacts, "tokenizer")
    tokenizer_vocab_path = _artifact(root, artifacts, "tokenizer_vocab")
    tokenizer_release_path = _artifact(root, artifacts, "tokenizer_release")
    tokenizer_validation_path = _artifact(root, artifacts, "tokenizer_validation")
    ngram_canonical_manifest_path = _artifact(
        root, artifacts, "ngram_canonical_map"
    )
    ngram_canonical_ids_path = _artifact(root, artifacts, "ngram_canonical_ids")
    source_lock_path = _artifact(root, artifacts, "source_lock")
    build_inputs_path = _artifact(root, artifacts, "build_inputs")
    manifest_path = _artifact(root, artifacts, "manifest")
    manifest_bundle = _artifact(root, artifacts, "manifest_bundle", directory=True)
    license_ledger_path = _artifact(root, artifacts, "license_ledger")
    shard_manifest_path = _artifact(root, artifacts, "shard_manifest")

    expected_hashes = (
        (verification_path, release.get("verification_file_sha256"), "verification"),
        (selection_path, release.get("selection_sha256"), "selection"),
        (token_count_path, release.get("token_count_contract_sha256"), "token-count contract"),
        (filter_chain_path, release.get("filter_chain_sha256"), "filter chain"),
        (tokenizer_path, release.get("tokenizer_sha256"), "tokenizer"),
        (
            ngram_canonical_manifest_path,
            release.get("ngram_canonical_map_manifest_sha256"),
            "N-gram canonical-ID manifest",
        ),
        (
            ngram_canonical_ids_path,
            release.get("ngram_canonical_ids_sha256"),
            "N-gram canonical-ID binary",
        ),
        (source_lock_path, release.get("source_lock_sha256"), "source lock"),
        (build_inputs_path, release.get("build_inputs_sha256"), "build inputs"),
        (manifest_path, release.get("manifest_sha256"), "data manifest"),
        (license_ledger_path, release.get("license_ledger_sha256"), "license ledger"),
        (shard_manifest_path, release.get("shard_manifest_sha256"), "shard manifest"),
    )
    for path, expected, label in expected_hashes:
        if _sha256(path) != expected:
            raise RuntimeError(f"{label.title()} hash does not match RELEASE.json")
    if _tree_sha256(manifest_bundle) != release.get("manifest_bundle_sha256"):
        raise RuntimeError("Manifest bundle hash does not match RELEASE.json")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification != release["verification"]
        or verification.get("schema") != "metis.verification/v2"
        or verification.get("verification_sha256")
        != _unsigned_hash(verification, "verification_sha256")
    ):
        raise RuntimeError("Embedded verification contract is invalid")
    if (
        verification.get("manifest_sha256") != release.get("manifest_sha256")
        or verification.get("manifest_contract_sha256")
        != release.get("manifest_contract_sha256")
        or verification.get("source_lock_sha256") != release.get("source_lock_sha256")
        or verification.get("build_inputs_sha256") != release.get("build_inputs_sha256")
        or verification.get("selection_sha256") != release.get("selection_sha256")
        or verification.get("token_count_contract_sha256")
        != release.get("token_count_contract_sha256")
        or verification.get("license_ledger_sha256")
        != release.get("license_ledger_sha256")
        or verification.get("shard_manifest_sha256")
        != release.get("shard_manifest_sha256")
    ):
        raise RuntimeError("Verification and release provenance hashes disagree")

    released_manifest = validate_manifest(manifest_path).require_valid()
    schedule = released_manifest["schedule"]
    expected_phase_tokens = phase_tokens_from_manifest(released_manifest)
    if release.get("phase_tokens") != expected_phase_tokens:
        raise RuntimeError(
            "Release phase schedule does not match the bundled manifest: "
            f"{release.get('phase_tokens')} against {expected_phase_tokens}"
        )
    selection_contract = verification.get("selection_contract", {})
    expected_unique = int(schedule["unique_target_tokens"])
    expected_replay = int(schedule["replay_target_tokens"])
    if (
        int(selection_contract.get("unique_tokens", -1)) != expected_unique
        or int(selection_contract.get("replay_tokens", -1)) != expected_replay
        or int(verification.get("packed_unique_tokens", -1)) != expected_unique
        or int(verification.get("packed_replay_tokens", -1)) != expected_replay
    ):
        raise RuntimeError(
            "Verified release does not match the manifest schedule: expected "
            f"{expected_unique:,} unique plus {expected_replay:,} replay, found "
            f"{int(selection_contract.get('unique_tokens', -1)):,} selected and "
            f"{int(verification.get('packed_unique_tokens', -1)):,} packed unique"
        )

    if (
        released_manifest.get("release") != release.get("release")
        or _manifest_contract_sha256(released_manifest)
        != release.get("manifest_contract_sha256")
        or int(schedule["target_tokens"]) != int(release["target_tokens"])
        or verification.get("source_phase_tokens")
        != {
            source["id"]: {
                phase: int(source["phase_tokens"].get(phase, 0))
                for phase in PHASE_DIRECTORIES
            }
            for source in released_manifest["sources"]
        }
    ):
        raise RuntimeError("Bundled data manifest does not match verified source/phase quotas")
    ledger_rows = [
        json.loads(line)
        for line in license_ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger_by_source = {
        str(row.get("source_id", "")): row for row in ledger_rows
    }
    if (
        len(ledger_by_source) != len(ledger_rows)
        or set(ledger_by_source)
        != {str(source["id"]) for source in released_manifest["sources"]}
    ):
        raise RuntimeError("License ledger does not contain exactly one row per source")
    for source in released_manifest["sources"]:
        row = ledger_by_source[str(source["id"])]
        observed = row.get("observed_license_tokens", {})
        observed_total = sum(int(value) for value in observed.values())
        expected_total = sum(int(value) for value in source["phase_tokens"].values())
        if (
            row.get("license_status") != source["license"]["status"]
            or row.get("license_expression") != source["license"]["expression"]
            or row.get("training_recipe_disposition") != "verified_for_training"
            or observed_total != expected_total
        ):
            raise RuntimeError(
                f"License ledger does not reconcile source {source['id']}"
            )

    token_count = json.loads(token_count_path.read_text(encoding="utf-8"))
    if (
        token_count.get("schema") != "metis.token-count-set/v1"
        or token_count.get("contract_sha256")
        != _unsigned_hash(token_count, "contract_sha256")
    ):
        raise RuntimeError("Token-count contract failed its self-hash check")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("schema") != "metis.selection-release/v2"
        or selection.get("token_count_contract_sha256")
        != release.get("token_count_contract_sha256")
    ):
        raise RuntimeError("Selection is not bound to the released token-count contract")

    tokenizer_contract = release.get("tokenizer_contract", {})
    if tokenizer_contract != verification.get("tokenizer_contract"):
        raise RuntimeError("Tokenizer contract differs between release and verification")
    if (
        token_count.get("tokenizer_contract") != tokenizer_contract
        or selection.get("tokenizer_contract") != tokenizer_contract
    ):
        raise RuntimeError(
            "Token-count/selection artifacts are not bound to the released tokenizer"
        )
    tokenizer_hashes = (
        (tokenizer_path, "tokenizer_sha256"),
        (tokenizer_vocab_path, "vocab_sha256"),
        (tokenizer_release_path, "tokenizer_release_sha256"),
        (tokenizer_validation_path, "tokenizer_validation_sha256"),
        (
            ngram_canonical_manifest_path,
            "ngram_canonical_map_manifest_sha256",
        ),
        (ngram_canonical_ids_path, "ngram_canonical_ids_sha256"),
    )
    for path, field in tokenizer_hashes:
        if _sha256(path) != tokenizer_contract.get(field):
            raise RuntimeError(f"Tokenizer artifact hash mismatch: {path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    if (
        len(vocab) != 65_536
        or len(set(ids)) != 65_536
        or set(ids) != set(range(65_536))
        or tokenizer_contract.get("token_dtype") != "uint16"
        or tokenizer_contract.get("endianness") != "little"
        or int(tokenizer_contract.get("vocabulary_size", -1)) != 65_536
        or int(tokenizer_contract.get("maximum_id", -1)) != 65_535
        or tokenizer_contract.get("contract_sha256")
        != _unsigned_hash(tokenizer_contract, "contract_sha256")
    ):
        raise RuntimeError("Released tokenizer violates the 65,536-ID uint16 contract")
    if json.loads(tokenizer_vocab_path.read_text(encoding="utf-8")) != vocab:
        raise RuntimeError("Released vocab.json does not match tokenizer.json")
    tokenizer_release = json.loads(tokenizer_release_path.read_text(encoding="utf-8"))
    tokenizer_validation = json.loads(tokenizer_validation_path.read_text(encoding="utf-8"))
    canonical_descriptor, _canonical_ids = validate_canonical_id_sidecar(
        manifest_path=ngram_canonical_manifest_path,
        binary_path=ngram_canonical_ids_path,
        tokenizer_path=tokenizer_path,
        expected_vocabulary_size=65_536,
        expected_manifest_sha256=tokenizer_contract.get(
            "ngram_canonical_map_self_sha256"
        ),
        expected_binary_sha256=tokenizer_contract.get(
            "ngram_canonical_ids_sha256"
        ),
        recompute_from_tokenizer=True,
    )
    expected_special_tokens = {
        str(token): tokenizer.token_to_id(str(token))
        for token in released_manifest["tokenizer"]["special_tokens"]
    }
    expected_split_digits = tokenizer_split_digits_setting(
        released_manifest["tokenizer"]
    )
    reported_split_digits = tokenizer_release.get("split_digits")
    contracted_split_digits = tokenizer_contract.get("split_digits")
    if (
        tokenizer_release.get("tokenizer_sha256") != _sha256(tokenizer_path)
        or tokenizer_release.get("uint16_safe") is not True
        or int(tokenizer_release.get("vocabulary_size", -1)) != 65_536
        or tokenizer_release.get("special_tokens") != expected_special_tokens
        or tokenizer_splits_digits(tokenizer) is not expected_split_digits
        or (
            reported_split_digits is not None
            and reported_split_digits is not expected_split_digits
        )
        or (expected_split_digits and reported_split_digits is not True)
        or (
            contracted_split_digits is not None
            and contracted_split_digits is not expected_split_digits
        )
        or (expected_split_digits and contracted_split_digits is not True)
        or tokenizer_release.get("ngram_canonical_ids_manifest_sha256")
        != canonical_descriptor.get("manifest_sha256")
        or tokenizer_release.get("ngram_canonical_ids_sha256")
        != canonical_descriptor.get("binary_sha256")
        or tokenizer_release.get("ngram_canonicalization_algorithm")
        != canonical_descriptor.get("algorithm")
        or int(tokenizer_release.get("ngram_canonical_vocabulary_size", -1))
        != int(canonical_descriptor.get("canonical_vocabulary_size", -2))
        or tokenizer_contract.get("ngram_canonical_dtype") != "uint16"
        or tokenizer_contract.get("ngram_canonical_endianness") != "little"
        or int(tokenizer_contract.get("ngram_canonical_entry_count", -1))
        != 65_536
        or int(tokenizer_contract.get("ngram_canonical_vocabulary_size", -1))
        != int(canonical_descriptor.get("canonical_vocabulary_size", -2))
        or any(token_id is None for token_id in expected_special_tokens.values())
        or tokenizer_validation.get("ok") is not True
    ):
        raise RuntimeError("Released tokenizer reports do not attest the tokenizer")

    filter_chain = json.loads(filter_chain_path.read_text(encoding="utf-8"))
    if (
        filter_chain.get("schema") != "metis.filter-chain/v1"
        or filter_chain.get("filter_chain_sha256")
        != _unsigned_hash(filter_chain, "filter_chain_sha256")
    ):
        raise RuntimeError("Filtering/decontamination receipt failed its self-hash check")

    shard_rows = [
        json.loads(line)
        for line in shard_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(shard_rows) != int(verification["shards"]):
        raise RuntimeError("Shard manifest row count does not match verification report")
    expected_tasks = set(range(len(shard_rows)))
    if {int(row.get("task_index", -1)) for row in shard_rows} != expected_tasks:
        raise RuntimeError("Shard task indices are not unique and contiguous")
    listed_binaries: dict[str, set[Path]] = {phase: set() for phase in PHASE_DIRECTORIES}
    listed_indices: dict[str, set[Path]] = {phase: set() for phase in PHASE_DIRECTORIES}
    listed_phase_indices: dict[str, set[int]] = {
        phase: set() for phase in PHASE_DIRECTORIES
    }
    phase_row_tokens = {phase: 0 for phase in PHASE_DIRECTORIES}
    pending_digests: list[tuple[Path, str]] = []
    for row in shard_rows:
        phase = str(row.get("phase", ""))
        if phase not in PHASE_DIRECTORIES:
            raise RuntimeError(f"Shard manifest contains unknown phase {phase!r}")
        binary_raw = str(row.get("binary", ""))
        index_raw = str(row.get("index", ""))
        if Path(binary_raw).is_absolute() or Path(index_raw).is_absolute():
            raise RuntimeError("Shard manifest paths must be relative to release root")
        binary = (root / binary_raw).resolve()
        index = (root / index_raw).resolve()
        phase_root = (root / PHASE_DIRECTORIES[phase]).resolve()
        try:
            binary.relative_to(phase_root)
            index.relative_to(phase_root)
        except ValueError as exc:
            raise RuntimeError(f"Shard path escapes phase directory: {binary}") from exc
        phase_index = int(row.get("phase_index", -1))
        if (
            binary.name != f"shard-{phase_index:05d}.bin"
            or index.name != f"shard-{phase_index:05d}.index.jsonl"
        ):
            raise RuntimeError(f"Shard filenames do not match phase index {phase_index}")
        tokens = int(row["tokens"])
        if binary.stat().st_size != tokens * 2:
            raise RuntimeError(f"Shard or index hash mismatch: {binary}")
        pending_digests.append((binary, str(row["binary_sha256"])))
        pending_digests.append((index, str(row["index_sha256"])))
        if binary in listed_binaries[phase] or index in listed_indices[phase]:
            raise RuntimeError(f"Duplicate shard artifact in manifest: {binary}")
        if phase_index in listed_phase_indices[phase]:
            raise RuntimeError(
                f"Duplicate phase shard index {phase_index} in {phase}"
            )
        listed_binaries[phase].add(binary)
        listed_indices[phase].add(index)
        listed_phase_indices[phase].add(phase_index)
        phase_row_tokens[phase] += tokens
    for path, matched in _verify_digests(pending_digests):
        if not matched:
            raise RuntimeError(f"Shard or index hash mismatch: {path}")
    if phase_row_tokens != expected_phase_tokens:
        raise RuntimeError(
            "Shard manifest token totals do not match the manifest schedule: "
            f"{dict(phase_row_tokens)} against {expected_phase_tokens}"
        )

    phases = contract.get("phases", [])
    if [phase.get("id") for phase in phases] != list(PHASE_DIRECTORIES):
        raise RuntimeError("Training phases must be ordered phase_a, phase_b, phase_c")
    phase_rows = []
    expected_cursor = 0
    for phase in phases:
        phase_id = str(phase["id"])
        if phase.get("data_directory") != PHASE_DIRECTORIES[phase_id]:
            raise RuntimeError(f"Training phase directory is not canonical for {phase_id}")
        start = int(phase["start_token"])
        end = int(phase["end_token_exclusive"])
        expected = expected_phase_tokens[phase_id]
        if start != expected_cursor or end - start != expected:
            raise RuntimeError(f"Training phase boundary mismatch for {phase_id}")
        phase_root = (root / PHASE_DIRECTORIES[phase_id]).resolve()
        actual_binaries = set(phase_root.glob("shard-*.bin"))
        actual_indices = set(phase_root.glob("shard-*.index.jsonl"))
        if (
            actual_binaries != listed_binaries[phase_id]
            or actual_indices != listed_indices[phase_id]
            or not actual_binaries
            or listed_phase_indices[phase_id]
            != set(range(len(listed_phase_indices[phase_id])))
        ):
            raise RuntimeError(f"Phase {phase_id} on-disk shard inventory is not exact")
        bytes_on_disk = sum(path.stat().st_size for path in actual_binaries)
        if bytes_on_disk != expected * 2:
            raise RuntimeError(f"Phase {phase_id} uint16 byte count mismatch")
        phase_rows.append(
            {
                "id": phase_id,
                "tokens": expected,
                "shards": len(actual_binaries),
                "bytes": bytes_on_disk,
            }
        )
        expected_cursor = end
    if expected_cursor != int(contract["total_train_tokens"]):
        raise RuntimeError("Phase boundaries do not end at total_train_tokens")
    if int(contract.get("ordering", {}).get("seed", -1)) != int(
        selection_contract.get("selection_seed", -2)
    ):
        raise RuntimeError("Training shard-order seed differs from selection seed")
    return {
        "ok": True,
        "release": release["release"],
        "release_root": str(root),
        "tokenizer": str(tokenizer_path),
        "ngram_canonical_map": str(ngram_canonical_manifest_path),
        "ngram_canonical_ids": str(ngram_canonical_ids_path),
        "token_dtype": "uint16",
        "token_endianness": "little",
        "sequence_length": int(contract["sequence_length"]),
        "phases": phase_rows,
        "shards": [
            {
                "task_index": int(row["task_index"]),
                "phase": row["phase"],
                "binary": str((root / row["binary"]).resolve()),
                "index": str((root / row["index"]).resolve()),
                "tokens": int(row["tokens"]),
            }
            for row in sorted(shard_rows, key=lambda item: int(item["task_index"]))
        ],
        "total_tokens": expected_cursor,
    }


def phase_for_token(contract_path: str | Path, token_cursor: int) -> str:
    contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    for phase in contract["phases"]:
        if int(phase["start_token"]) <= token_cursor < int(phase["end_token_exclusive"]):
            return str(phase["id"])
    raise ValueError(f"Token cursor {token_cursor:,} is outside the training schedule")
