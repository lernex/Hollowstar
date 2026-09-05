from __future__ import annotations

import contextlib
import copy
import hashlib
import heapq
import io
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from . import tokenizer as tk
from .common import atomic_json, canonical_json, digest_json, read_receipt, sha256_file, under_root, utc_now


REQUIRED_CATEGORIES = ("web", "code", "math", "science", "multilingual")
PRODUCTION_SAMPLE_BYTES = 150_000_000_000
PRODUCTION_CATEGORY_MINIMUM = 1_000_000_000
CURRENT_MAX_WORKING_BYTES = 2_000_000_000_000
CURRENT_MAX_RAW_BYTES = 400_000_000_000
ReserveOutput = Callable[[Mapping[str, Any]], ContextManager[Mapping[str, Any]]]
QuotaOpener = Callable[..., Any]
_CONTROL_LIMIT = 32 * 1024**2


@dataclass(frozen=True)
class _QuotaAccess:
    quota: Any
    ceiling: int

    def __call__(self, path: Path, mode: str = "wb") -> Any:
        return self.quota.open(path, mode=mode)


def _unlink_output(path: Path, opener: QuotaOpener | None) -> None:
    if not path.exists():
        return
    if isinstance(opener, _QuotaAccess):
        opener.quota.unlink(path)
    else:
        path.unlink()


def _replace_output(source: Path, destination: Path, opener: QuotaOpener | None) -> None:
    if isinstance(opener, _QuotaAccess):
        opener.quota.replace(source, destination)
    else:
        os.replace(source, destination)


def _remove_output_stage(path: Path, opener: QuotaOpener | None) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Invalid tokenizer output staging directory")
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("Unexpected file type in tokenizer output staging")
        _unlink_output(child, opener)
    path.rmdir()


def _output_stage(parent: Path, name: str, opener: QuotaOpener | None) -> Path:
    if parent.is_symlink():
        raise ValueError("Tokenizer output directory may not be redirected by a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / name
    _remove_output_stage(stage, opener)
    stage.mkdir()
    if isinstance(opener, _QuotaAccess):
        opener.quota.reserve(opener.ceiling)
    return stage


@dataclass(frozen=True)
class _Config:
    production: bool
    target: int
    minimums: dict[str, int]
    source_minimums: dict[str, int]
    language_minimums: dict[str, int]
    special_tokens: list[str]
    vocabulary_size: int
    minimum_frequency: int
    max_document_bytes: int
    scratch_bytes: int
    sample_output_bytes: int
    model_output_bytes: int
    id_output_bytes: int
    max_working_bytes: int
    raw_reservation_bytes: int
    max_candidates: int
    max_chunks: int
    max_events_per_step: int
    max_attempts: int
    batch_size: int
    seed: str

    @property
    def overshoot_limit(self) -> int:
        return len(REQUIRED_CATEGORIES) * self.max_document_bytes


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _minimum_mapping(value: Any, name: str, target: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or len(value) > 1024:
        raise ValueError(f"{name} must be a bounded mapping")
    result = {}
    for key, minimum in value.items():
        if not isinstance(key, str) or not key.strip() or len(key.encode("utf-8")) > 256:
            raise ValueError(f"{name} requires nonempty bounded source/language names")
        result[key] = _integer(minimum, f"{name}.{key}")
    if sum(result.values()) > target:
        raise ValueError(f"{name} exceeds the training sample target")
    return dict(sorted(result.items()))


def _config(
    run: Mapping[str, Any], test_mode: bool, *, live_limits: Mapping[str, Any] | None = None,
) -> _Config:
    if type(test_mode) is not bool:
        raise ValueError("test_mode must be a boolean")
    config = run.get("config", run)
    tokenizer = config["tokenizer"]
    production = tokenizer.get("production", True)
    if type(production) is not bool or production is test_mode:
        raise ValueError("Production is mandatory unless both test_mode=True and tokenizer.production=False")
    specials = tokenizer.get("special_tokens")
    if (
        not isinstance(specials, list) or len(specials) != 7
        or any(not isinstance(token, str) or not token or len(token.encode("utf-8")) > 256 for token in specials)
        or len(set(specials)) != 7
        or any(any(character.isnumeric() for character in token) for token in specials)
    ):
        raise ValueError("RUN tokenizer.special_tokens must contain exactly seven unique literal, digit-free strings")
    if tokenizer.get("split_digits") is not True:
        raise ValueError("RUN must explicitly enable global split_digits=true")
    if tokenizer.get("dtype", tokenizer.get("token_dtype", "<u4")) not in {"<u4", "uint32"}:
        raise ValueError("Tokenizer IDs must be little-endian uint32")
    if tokenizer.get("byte_order", "little") != "little":
        raise ValueError("Tokenizer IDs must be little-endian uint32")
    vocabulary = _integer(tokenizer.get("vocabulary_size", 131_072), "vocabulary_size", minimum=263)
    target = _integer(tokenizer.get("sample_target_bytes", PRODUCTION_SAMPLE_BYTES), "sample_target_bytes")
    minimum = tokenizer.get(
        "minimum_category_bytes",
        tokenizer.get("min_sample_bytes_per_category", PRODUCTION_CATEGORY_MINIMUM),
    )
    minimums = (
        {category: _integer(minimum[category], f"minimum_category_bytes.{category}") for category in REQUIRED_CATEGORIES}
        if isinstance(minimum, dict)
        else {category: _integer(minimum, "minimum_category_bytes") for category in REQUIRED_CATEGORIES}
    )
    if production and (
        vocabulary != 131_072 or target != PRODUCTION_SAMPLE_BYTES
        or min(minimums.values()) < PRODUCTION_CATEGORY_MINIMUM
    ):
        raise ValueError(
            "Production requires vocab131072, the full 150e9-byte training target, and >=1e9 bytes "
            "in every required category; a frozen 160GB RUN needs an explicit 150GB tokenizer recipe"
        )
    if not production and vocabulary >= 131_072:
        raise ValueError("Explicit test mode must use a nonproduction vocabulary below 131072")
    if sum(minimums.values()) > target:
        raise ValueError("Required category minimums exceed the intended sample target")
    source_minimums = _minimum_mapping(
        tokenizer.get("required_source_minimum_bytes", {}), "required_source_minimum_bytes", target,
    )
    language_minimums = _minimum_mapping(
        tokenizer.get("required_language_minimum_bytes", {}), "required_language_minimum_bytes", target,
    )
    limits = config.get("limits", {}) if live_limits is None else live_limits
    working = _integer(limits.get("max_working_bytes", CURRENT_MAX_WORKING_BYTES), "max_working_bytes")
    raw = _integer(limits.get("max_raw_bytes", CURRENT_MAX_RAW_BYTES), "max_raw_bytes")
    confirmation = limits.get("capacity_confirmation", "pending")
    if confirmation not in {"pending", "administrator-confirmed", "unlimited"}:
        raise ValueError("Unrecognized storage-capacity confirmation")
    if confirmation == "pending" and (working > CURRENT_MAX_WORKING_BYTES or raw > CURRENT_MAX_RAW_BYTES):
        raise ValueError("This run may not expand the current 400GB raw / 2TB working bounds")
    if raw > 200_000_000_000_000 or working <= raw:
        raise ValueError("Tokenizer capacity must fit the 200 TB plan and reserve derived space")
    max_document = _integer(tokenizer.get("max_sample_document_bytes", 16 * 1024**2), "max_sample_document_bytes")
    if production and max_document * len(REQUIRED_CATEGORIES) > 1_000_000_000:
        raise ValueError("Whole-document production target overshoot must remain bounded by 1e9 bytes")
    sample_output = _integer(tokenizer.get("max_sample_output_bytes", 256 * 1024**3), "max_sample_output_bytes", minimum=65_536)
    model_output = _integer(tokenizer.get("max_model_output_bytes", 256 * 1024**2), "max_model_output_bytes", minimum=65_536)
    id_output = _integer(tokenizer.get("max_id_partition_output_bytes", 64 * 1024**3), "max_id_partition_output_bytes")
    if max(sample_output, model_output, id_output) > working:
        raise ValueError("Configured tokenizer output budgets exceed max_working_bytes")
    seed = tokenizer.get("sample_seed", "metis1.7-tokenizer-v1")
    if not isinstance(seed, str) or not seed:
        raise ValueError("sample_seed must be a nonempty string")
    return _Config(
        production, target, minimums, source_minimums, language_minimums, list(specials), vocabulary,
        _integer(tokenizer.get("minimum_frequency", 2), "minimum_frequency"),
        max_document,
        _integer(tokenizer.get("max_scratch_bytes", 64 * 1024**3), "max_scratch_bytes", minimum=262_144),
        sample_output, model_output, id_output, working, raw,
        _integer(tokenizer.get("max_candidate_documents", 100_000_000), "max_candidate_documents"),
        _integer(tokenizer.get("max_input_paths", 65_536), "max_input_paths"),
        _integer(tokenizer.get("max_events_per_step", 32), "max_events_per_step"),
        _integer(tokenizer.get("max_sample_attempts", 16), "max_sample_attempts"),
        _integer(tokenizer.get("batch_size", 256), "batch_size"), seed,
    )


def _seal(
    path: Path, payload: Mapping[str, Any], *, limit: int = _CONTROL_LIMIT,
    opener: QuotaOpener | None = None,
) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = digest_json(result)
    serialized = json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(serialized) > limit:
        raise ValueError("Bounded tokenizer control/receipt output limit exceeded")
    if opener is None:
        atomic_json(path, result)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.incomplete")
        try:
            with opener(partial, mode="wb") as stream:
                stream.write(serialized)
                stream.flush()
            tk._sync_file(partial)
            _replace_output(partial, path, opener)
        finally:
            _unlink_output(partial, opener)
    tk._sync_directory(path.parent)
    return result


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _CONTROL_LIMIT:
        raise ValueError("Tokenizer receipt exceeds its bounded control-file size")
    payload = read_receipt(path)
    payload["receipt_sha256"] = digest_json(payload)
    return payload


def _read_run(root: Path) -> tuple[dict[str, Any], str]:
    with (root / "RUN.json").open("rb") as stream:
        data = stream.read(_CONTROL_LIMIT + 1)
    if len(data) > _CONTROL_LIMIT:
        raise ValueError("Frozen RUN exceeds the bounded configuration size")
    run = json.loads(data)
    if not isinstance(run, dict):
        raise ValueError("Frozen RUN must be an object")
    if "receipt_sha256" in run and run.pop("receipt_sha256") != digest_json(run):
        raise ValueError("Frozen RUN canonical receipt seal mismatch")
    return run, hashlib.sha256(data).hexdigest()


def freeze_tokenizer_recipe(root: Path, tokenizer_settings: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze complete tokenizer settings without changing the acquisition RUN.

    The target counts usable training text only, not a separate heldout pool.
    Nonproduction recipes still require test_mode=True when running the pipeline.
    """
    root = Path(root).resolve()
    if not isinstance(tokenizer_settings, Mapping):
        raise ValueError("Tokenizer recipe settings must be a mapping")
    settings = json.loads(canonical_json(dict(tokenizer_settings)))
    run, run_sha = _read_run(root)
    config = run.get("config", run)
    live_limits = read_receipt(root / "limits.json") if (root / "limits.json").exists() else None
    _config(
        {**config, "tokenizer": settings}, settings.get("production", True) is False,
        live_limits=live_limits,
    )
    payload = {
        "schema": "metis17.tokenizer-recipe/v1", "run_sha256": run_sha,
        "tokenizer": settings,
    }
    path = under_root(root, "tokenizer/RECIPE.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tk._locked(path.parent / ".recipe.lock", blocking=False):
        if path.exists():
            existing = _read(path)
            if _payload(existing) != payload:
                raise ValueError("Immutable tokenizer recipe changed")
            return existing
        if sha256_file(root / "RUN.json") != run_sha:
            raise ValueError("Frozen RUN changed while sealing the tokenizer recipe")
        result = _seal(path, payload)
        path.chmod(0o444)
        return result


def _recipe_config(
    root: Path, run: Mapping[str, Any], run_sha: str, test_mode: bool,
    *, live_limits: Mapping[str, Any] | None = None,
) -> tuple[_Config, str | None]:
    path = under_root(root, "tokenizer/RECIPE.json")
    recipe_sha = None
    if path.exists():
        recipe = _read(path)
        if recipe.get("schema") != "metis17.tokenizer-recipe/v1" or recipe.get("run_sha256") != run_sha:
            raise ValueError("Tokenizer recipe does not match the exact frozen RUN.json")
        if not isinstance(recipe.get("tokenizer"), dict):
            raise ValueError("Tokenizer recipe requires explicit tokenizer settings")
        run = {**run.get("config", run), "tokenizer": recipe["tokenizer"]}
        recipe_sha = recipe["receipt_sha256"]
    return _config(run, test_mode, live_limits=live_limits), recipe_sha


def _payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def _save(work: Path, state: dict[str, Any]) -> dict[str, Any]:
    state = _payload(state)
    path = work / "STATE.json"
    if path.exists():
        previous = _read(path)
        if _payload(previous) == state:
            return previous
    state["updated_at"] = utc_now()
    return _seal(path, state)


def _initial(
    run_sha: str, production: bool, generation: str, recipe_sha: str | None = None,
    *, recipe_bound: bool = True,
) -> dict[str, Any]:
    return {
        "schema": "metis17.tokenizer-pipeline/v1", "run_sha256": run_sha,
        "recipe_sha256": recipe_sha,
        "recipe_bound": recipe_bound,
        "generation": generation, "generation_descriptor_sha256": "",
        "production": production, "status": "WAITING", "activity": "waiting_for_eligible_data",
        "stage_receipts": {}, "chunks": {}, "event_cursors": {}, "next_host": 0,
        "admitted_rows": 0, "admitted_characters": {category: 0 for category in REQUIRED_CATEGORIES},
        "admitted_source_characters": {}, "admitted_language_characters": {},
        "inventory_bytes": 0, "sample_attempts": 0,
        "ignored_other_generation_events": 0, "ignored_unscoped_events": 0,
        "id_cache": {"status": "DEFERRED", "reason": "tokenizer_not_frozen"},
    }


def _generation(root: Path, generation: str) -> dict[str, Any]:
    tk._digest(generation, "eligibility generation")
    descriptor = _read(root / "preparation" / "generations" / f"{generation}.json")
    if descriptor.get("generation", generation) != generation:
        raise ValueError("Eligibility generation descriptor does not match its immutable filename")
    return descriptor


def _event_batch(root: Path, state: Mapping[str, Any], maximum: int) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    paths = sorted((root / "events" / "eligible").glob("*.jsonl"))
    if len(paths) > 256:
        raise ValueError("Eligible-event host count exceeds its explicit bound")
    cursors = copy.deepcopy(state["event_cursors"])
    events = []
    if not paths:
        return events, cursors, 0
    start = state.get("next_host", 0) % len(paths)
    ordered = paths[start:] + paths[:start]
    with contextlib.ExitStack() as stack:
        streams = {}
        for path in ordered:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Eligible event logs must remain under the release root")
            key = str(path.relative_to(root))
            stat = path.stat()
            cursor = cursors.setdefault(key, {"offset": 0, "device": stat.st_dev, "inode": stat.st_ino})
            if (
                cursor["device"] != stat.st_dev or cursor["inode"] != stat.st_ino
                or stat.st_size < cursor["offset"]
            ):
                raise ValueError(f"Eligible event log was replaced or truncated: {key}")
            stream = stack.enter_context(path.open("rb"))
            if cursor.get("last_line_bytes"):
                stream.seek(cursor["offset"] - cursor["last_line_bytes"])
                if hashlib.sha256(stream.read(cursor["last_line_bytes"])).hexdigest() != cursor["last_line_sha256"]:
                    raise ValueError(f"Eligible event-log checkpoint changed: {key}")
            stream.seek(cursor["offset"])
            streams[key] = stream
        exhausted: set[str] = set()
        last_host = start
        while len(events) < maximum and len(exhausted) < len(streams):
            for path in ordered:
                key = str(path.relative_to(root))
                if key in exhausted:
                    continue
                stream = streams[key]
                line = stream.readline(1024 * 1024 + 1)
                if len(line) > 1024 * 1024:
                    raise ValueError("Eligible event line exceeds its bounded size")
                if not line or not line.endswith(b"\n"):
                    exhausted.add(key)
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Eligible events must be JSON objects")
                events.append(event)
                cursors[key].update({
                    "offset": stream.tell(), "last_line_bytes": len(line),
                    "last_line_sha256": hashlib.sha256(line).hexdigest(),
                })
                last_host = paths.index(path)
                if len(events) == maximum:
                    break
        return events, cursors, (last_host + 1) % len(paths) if events else state.get("next_host", 0)


def _completion_proof(root: Path, receipt: Mapping[str, Any], object_id: str) -> dict[str, Any]:
    proof = receipt.get("object_completion")
    if receipt.get("object_complete") is not True or not isinstance(proof, dict):
        raise ValueError("Production eligibility requires object_complete and a sealed object-completion proof")
    relative = proof.get("receipt_path", proof.get("path", proof.get("snapshot", proof.get("relative_path", proof.get("manifest_path")))))
    if not isinstance(relative, str):
        raise ValueError("Object-completion proof must reference a sealed receipt path")
    path = under_root(root, relative)
    body = _read(path)
    canonical = proof.get("receipt_sha256", proof.get("payload_sha256", proof.get("stage_receipt_sha256")))
    file_hash = proof.get("file_sha256", proof.get("sha256", proof.get("manifest_sha256")))
    if canonical is None and file_hash is None:
        raise ValueError("Object-completion proof has no explicitly pinned checksum")
    if canonical is not None and canonical != body["receipt_sha256"]:
        raise ValueError("Object-completion canonical seal mismatch")
    if file_hash is not None and file_hash != sha256_file(path):
        raise ValueError("Object-completion file checksum mismatch")
    if body.get("object_id") != object_id:
        raise ValueError("Object-completion proof belongs to another object")
    if body.get("object_complete") is False or body.get("complete") is False:
        raise ValueError("Object-completion proof is explicitly incomplete")
    schema = str(body.get("schema", ""))
    object_schema = "object" in schema and "chunk" not in schema
    terminal_status = body.get("status") in {"COMPLETE", "OBJECT_COMPLETE", "NORMALIZED_OBJECT_COMPLETE", "NORMALIZATION_COMPLETE"}
    normalized_object = body.get("status") == "NORMALIZED" and object_schema and "normal" in schema
    if not (
        body.get("object_complete") is True
        or object_schema and (
            any(body.get(field) is True for field in ("complete", "eof", "eof_reached"))
            or terminal_status or normalized_object
        )
    ):
        raise ValueError("Object-completion receipt does not positively prove EOF completion")
    if receipt.get("schema") == "metis17.prepared-chunk/v1":
        from .dedup_receipts import _completion_proofs

        _completion_proofs(dict(receipt), root, None)
    return {"path": relative, "stage_receipt_sha256": body["receipt_sha256"]}


def _stage_event(root: Path, event: Mapping[str, Any], production: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = tk._digest(event.get("stage_receipt_sha256"), "stage_receipt_sha256")
    relative = event.get("receipt_path")
    if not isinstance(relative, str):
        raise ValueError("Eligible event requires a root-relative receipt_path")
    receipt = _read(under_root(root, relative))
    if receipt["receipt_sha256"] != expected:
        raise ValueError("Eligible event requires the canonical stage seal, not receipt-file SHA")
    if receipt.get("schema") not in {"metis17.prepared-object/v1", "metis17.prepared-chunk/v1"}:
        raise ValueError("Unsupported eligible preparation receipt schema")
    if (
        receipt.get("status") != "ELIGIBLE" or receipt.get("eligible") is not True
        or receipt.get("training_ready") is not True or receipt.get("pending_reasons")
    ):
        raise ValueError("Pending, filtered, normalized, or quarantined records cannot enter tokenizer orchestration")
    for key in ("object_id", "source_id"):
        if not isinstance(event.get(key), str) or not event[key] or receipt.get(key) != event[key]:
            raise ValueError(f"Eligible event and receipt disagree on {key}")
    for field in ("generation", "eligibility_generation"):
        if field in receipt and receipt[field] != event.get("generation"):
            raise ValueError("Eligible event and receipt disagree on their eligibility generation")
    proof = {
        "receipt_path": relative, "stage_receipt_sha256": expected,
        "object_id": event["object_id"], "source_id": event["source_id"],
        "generation": event.get("generation"),
    }
    if production or receipt["schema"] == "metis17.prepared-chunk/v1":
        proof["object_completion"] = _completion_proof(root, receipt, event["object_id"])
    if not isinstance(receipt.get("chunks"), list):
        raise ValueError("Only an explicit eligible chunks inventory may be admitted")
    if any(not isinstance(chunk, dict) for chunk in receipt["chunks"]):
        raise ValueError("Malformed eligible chunk inventory")
    paths = [chunk.get("path") for chunk in receipt["chunks"]]
    if any(not isinstance(path, str) or not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("Eligible chunk inventory requires distinct nonempty paths")
    records = sum(_integer(chunk.get("records"), "chunk.records", minimum=0) for chunk in receipt["chunks"])
    if receipt.get("eligible_documents", records) != records:
        raise ValueError("Eligible chunk inventory does not match its declared record coverage")
    if "eligible_documents" in event and _integer(event["eligible_documents"], "event eligible_documents", minimum=0) != records:
        raise ValueError("Eligible event document count disagrees with its sealed receipt")
    for field in ("chunk_id", "input_documents"):
        if field in event and field in receipt and event[field] != receipt[field]:
            raise ValueError(f"Eligible event and receipt disagree on {field}")
    return receipt, proof


def _chunk_inventory(
    root: Path, work: Path, chunk: Mapping[str, Any], proof: Mapping[str, Any], config: _Config,
    *, maximum_output_bytes: int,
) -> tuple[str, dict[str, Any]]:
    relative = chunk.get("path")
    if not isinstance(relative, str):
        raise ValueError("Eligible chunk paths must be root-relative")
    expected = tk._digest(chunk.get("sha256"), "prepared Parquet SHA")
    byte_count = _integer(chunk.get("byte_count"), "chunk.byte_count", minimum=0)
    rows = _integer(chunk.get("records"), "chunk.records", minimum=0)
    identity = {"path": relative, "sha256": expected, "byte_count": byte_count, "rows": rows}
    key = digest_json(identity)
    fragment = work / "inventory" / f"{key}.json"
    source = under_root(root, relative)
    if fragment.exists():
        if fragment.stat().st_size > maximum_output_bytes:
            raise ValueError("Bounded tokenizer inventory metadata budget exhausted")
        cached = _read(fragment)
        if cached.get("source") != identity:
            raise ValueError("Corrupt tokenizer inventory identity")
        if any(cached["proof"].get(key) != proof.get(key) for key in ("source_id", "object_id", "generation")):
            raise ValueError("Cached tokenizer inventory belongs to another eligibility proof")
        if source.stat().st_size != byte_count:
            raise ValueError("Admitted prepared Parquet changed")
        return key, cached
    if source.stat().st_size != byte_count or sha256_file(source) != expected:
        raise ValueError("Eligible prepared Parquet byte count or hash mismatch")
    parquet = tk._prepared_file(source)
    if parquet.metadata.num_rows != rows:
        raise ValueError("Eligible Parquet record coverage mismatch")
    characters = {category: 0 for category in REQUIRED_CATEGORIES}
    category_rows = {category: 0 for category in REQUIRED_CATEGORIES}
    source_characters = {source_id: 0 for source_id in config.source_minimums}
    language_characters = {language: 0 for language in config.language_minimums}
    observed = 0
    total_characters = 0
    columns = [field.name for field in tk.PREPARED_SCHEMA if field.name != "text"]
    max_metadata_bytes = 0
    for batch in parquet.iter_batches(batch_size=config.batch_size, columns=columns):
        for row in batch.to_pylist():
            tk._checked_metadata(row)
            if row["source_id"] != proof["source_id"] or row["object_id"] != proof["object_id"]:
                raise ValueError("Prepared row identity does not match its admitted stage receipt")
            tk._digest(row["content_hash"], "prepared content_hash")
            count = _integer(row["character_count"], "character_count", minimum=0)
            total_characters += count
            if not isinstance(row["doc_id"], str) or len(row["doc_id"].encode("utf-8")) > 4096:
                raise ValueError("Prepared document ID exceeds the bounded metadata contract")
            max_metadata_bytes = max(max_metadata_bytes, sum(
                len(value.encode("utf-8")) if isinstance(value, str) else 8 for value in row.values()
            ))
            if row["category"] in characters:
                characters[row["category"]] += count
                category_rows[row["category"]] += 1
                if row["source_id"] in source_characters:
                    source_characters[row["source_id"]] += count
                if row["language"] in language_characters:
                    language_characters[row["language"]] += count
            observed += 1
    if observed != rows:
        raise ValueError("Prepared metadata scan dropped source rows")
    payload = {
        "schema": "metis17.tokenizer-inventory/v1", "source": identity, "proof": dict(proof),
        "characters": characters, "category_rows": category_rows, "max_metadata_row_bytes": max_metadata_bytes,
        "source_characters": source_characters, "language_characters": language_characters,
        "total_characters": total_characters,
    }
    return key, _seal(fragment, payload, limit=maximum_output_bytes)


def _admit(
    root: Path, work: Path, state: dict[str, Any], events: Sequence[Mapping[str, Any]], config: _Config,
) -> None:
    for event in events:
        if event.get("generation") != state["generation"]:
            field = "ignored_unscoped_events" if event.get("generation") is None else "ignored_other_generation_events"
            state[field] += 1
            continue
        stage_hash = tk._digest(event.get("stage_receipt_sha256"), "stage_receipt_sha256")
        if stage_hash in state["stage_receipts"]:
            continue
        receipt, proof = _stage_event(root, event, config.production)
        for chunk in receipt["chunks"]:
            if chunk["path"] in state["chunks"]:
                if state["chunks"][chunk["path"]]["sha256"] != chunk["sha256"]:
                    raise ValueError("An admitted immutable Parquet path was reused with different bytes")
                continue
            if len(state["chunks"]) >= config.max_chunks:
                raise ValueError("Tokenizer input inventory exceeds max_input_paths")
            key, inventory = _chunk_inventory(
                root, work, chunk, proof, config,
                maximum_output_bytes=min(_CONTROL_LIMIT, 256 * 1024**2 - state["inventory_bytes"]),
            )
            fragment = work / "inventory" / f"{key}.json"
            state["inventory_bytes"] += fragment.stat().st_size
            if state["inventory_bytes"] > 256 * 1024**2:
                raise ValueError("Bounded tokenizer inventory metadata budget exhausted")
            state["chunks"][chunk["path"]] = {"key": key, "sha256": chunk["sha256"]}
            state["admitted_rows"] += inventory["source"]["rows"]
            for category in REQUIRED_CATEGORIES:
                state["admitted_characters"][category] += inventory["characters"][category]
            for field in ("source", "language"):
                totals = state.setdefault(f"admitted_{field}_characters", {})
                for key, count in inventory.get(f"{field}_characters", {}).items():
                    totals[key] = totals.get(key, 0) + count
        state["stage_receipts"][stage_hash] = proof
        if len(state["stage_receipts"]) > config.max_chunks * 2:
            raise ValueError("Tokenizer stage-proof inventory exceeds its bound")


def _inputs(root: Path, work: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = []
    for relative, item in sorted(state["chunks"].items()):
        inventory = _read(work / "inventory" / f"{item['key']}.json")
        source = inventory["source"]
        if source["path"] != relative or source["sha256"] != item["sha256"]:
            raise ValueError("Tokenizer inventory checkpoint does not match its immutable fragment")
        inputs.append({**source, "path": str(under_root(root, relative))})
    return inputs


def _partition_inputs(
    root: Path, work: Path, event: Mapping[str, Any], config: _Config,
    *, paths: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    receipt, proof = _stage_event(root, event, config.production)
    inventory = {chunk["path"]: chunk for chunk in receipt["chunks"]}
    selected = set(inventory) if paths is None else paths
    if not selected.issubset(inventory):
        raise ValueError("Token partition contains paths outside its explicit eligibility receipt")
    if len(selected) > config.max_chunks:
        raise ValueError("Token partition inventory exceeds max_input_paths")
    seal = proof["stage_receipt_sha256"]
    directory = work / "partition-inputs" / seal[:2] / seal
    result = {}
    for relative in sorted(selected):
        key, _ = _chunk_inventory(
            root, directory, inventory[relative], proof, config, maximum_output_bytes=_CONTROL_LIMIT,
        )
        result[relative] = {
            "sha256": inventory[relative]["sha256"],
            "inventory_path": str((directory / "inventory" / f"{key}.json").relative_to(work)),
        }
    return result


def _admit_latest(
    root: Path, work: Path, state: dict[str, Any], events: Sequence[Mapping[str, Any]], config: _Config,
) -> None:
    # Frozen sample state must not become a catalogue of the entire 200 TB run.
    # The last bounded batch remains available to the existing dispatch API;
    # concurrent/later dispatch can instead supply its own immutable stage proof.
    latest: dict[str, dict[str, Any]] = {}
    seen = set()
    for event in events:
        if event.get("generation") != state["generation"]:
            field = "ignored_unscoped_events" if event.get("generation") is None else "ignored_other_generation_events"
            state[field] += 1
            continue
        seal = tk._digest(event.get("stage_receipt_sha256"), "stage_receipt_sha256")
        if seal in seen:
            continue
        seen.add(seal)
        for path, item in _partition_inputs(root, work, event, config).items():
            previous = latest.get(path, state["chunks"].get(path))
            if previous is not None and previous["sha256"] != item["sha256"]:
                raise ValueError("An admitted immutable Parquet path was reused with different bytes")
            latest[path] = item
            if len(latest) > config.max_chunks:
                raise ValueError("Recent token partition inventory exceeds max_input_paths")
    state["recent_partition_inputs"] = latest


def _reservation_request(work: Path, state: Mapping[str, Any], config: _Config, kind: str, key: str, amount: int) -> dict[str, Any]:
    return {
        "schema": "metis17.tokenizer-reservation/v1",
        "reservation_id": digest_json({
            "run_sha256": state["run_sha256"], "generation": state["generation"], "kind": kind, "key": key,
        }),
        "kind": kind, "requested_bytes": amount, "max_working_bytes": config.max_working_bytes,
        "generation": state["generation"],
        "raw_reservation_bytes": config.raw_reservation_bytes, "max_total_bytes": config.max_working_bytes,
        "output_root": str(work), "production": config.production,
    }


@contextlib.contextmanager
def _reserved(request: Mapping[str, Any], reserve_output: ReserveOutput | None) -> Iterator[dict[str, Any]]:
    if reserve_output is None:
        raise ValueError("A parent atomic output-reservation provider is required before allocating tokenizer output")
    with reserve_output(request) as grant:
        if not isinstance(grant, Mapping):
            raise ValueError("Output reservation must yield a budget certificate")
        reserved = _integer(grant.get("reserved_bytes"), "reserved_bytes")
        used = _integer(grant.get("used_working_bytes"), "used_working_bytes", minimum=0)
        other = _integer(grant.get("other_reserved_bytes"), "other_reserved_bytes", minimum=0)
        raw = _integer(grant.get("raw_reservation_bytes", request["raw_reservation_bytes"]), "raw_reservation_bytes")
        if (
            raw != request["raw_reservation_bytes"]
            or reserved < request["requested_bytes"]
            or used + other + reserved + raw > request["max_total_bytes"]
            or grant.get("reservation_id", request["reservation_id"]) != request["reservation_id"]
        ):
            raise ValueError("Output reservation exceeds max_working_bytes including the full raw reservation")
        yield {**dict(grant), "reservation_id": request["reservation_id"], "raw_reservation_bytes": raw}


@contextlib.contextmanager
def _allocation(
    root: Path, directory: Path, request: Mapping[str, Any],
    reserve_output: ReserveOutput | None, working_budget: Any | None,
) -> Iterator[tuple[dict[str, Any], QuotaOpener | None]]:
    if reserve_output is not None:
        if working_budget is not None:
            raise ValueError("Choose one authoritative tokenizer output-budget provider")
        with _reserved(request, reserve_output) as grant:
            yield grant, None
        return
    if working_budget is None:
        if not request["production"]:
            raise ValueError("Test output requires an explicit reservation provider or WorkingBudget")
        from .storage import WorkingBudget

        working_budget = WorkingBudget(root)
    snapshot = _budget_snapshot(working_budget, request["max_total_bytes"], request["raw_reservation_bytes"])
    namespace = "{}{}:{}".format(
        request["kind"], "_test" if not request["production"] else "", request["generation"],
    )
    with working_budget.quota(namespace, directory) as quota:
        if quota.reserve(request["requested_bytes"]) < request["requested_bytes"]:
            raise ValueError("WorkingBudget did not authorize the full tokenizer output ceiling")
        yield {
            "provider": "WorkingBudget", "namespace": namespace,
            "directory": str(directory.relative_to(root)), "reservation_mode": "fixed_ceiling_and_hard_writes",
            "growth_cap_bytes": request["requested_bytes"], "max_total_bytes": request["max_total_bytes"],
            "raw_reservation_bytes": request["raw_reservation_bytes"],
            "policy_and_metadata_reserve_bytes": snapshot["policy_and_metadata_reserve_bytes"],
        }, _QuotaAccess(quota, request["requested_bytes"])


def _budget_snapshot(
    working_budget: Any, maximum_total: int, raw_reservation: int = CURRENT_MAX_RAW_BYTES,
) -> dict[str, Any]:
    snapshot = dict(working_budget.snapshot())
    raw = _integer(snapshot.get("max_raw_bytes"), "WorkingBudget max_raw_bytes")
    total = _integer(snapshot.get("max_working_bytes"), "WorkingBudget max_working_bytes")
    metadata = _integer(snapshot.get("policy_and_metadata_reserve_bytes"), "WorkingBudget metadata reserve", minimum=0)
    derived = _integer(snapshot.get("derived_limit_bytes"), "WorkingBudget derived limit", minimum=0)
    if raw != raw_reservation or total > maximum_total or derived > total - raw - metadata:
        raise ValueError("WorkingBudget must protect the full raw and metadata reserves inside the approved total")
    return snapshot


class _BoundedFile(io.RawIOBase):
    def __init__(self, path: Path, maximum: int, *, opener: QuotaOpener | None = None) -> None:
        self.path = path
        self.stream = path.open("xb") if opener is None else opener(path, mode="wb")
        self.maximum = maximum

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.stream.tell()

    def write(self, data: bytes) -> int:
        if self.stream.tell() + len(data) > self.maximum:
            raise ValueError("Reserved tokenizer output byte limit exhausted")
        return self.stream.write(data)

    def flush(self) -> None:
        if not self.stream.closed:
            self.stream.flush()

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
            tk._sync_file(self.path)
        super().close()


def _scratch_workspace(scratch: Path, work: Path, state: Mapping[str, Any], config: _Config) -> Path:
    scratch = tk._local_scratch(scratch, work, max(config.scratch_bytes, config.model_output_bytes))
    local = scratch / "tokenizer-pipeline"
    identity = {
        "output": str(work), "run_sha256": state["run_sha256"],
        "recipe_sha256": state.get("recipe_sha256"),
    }
    marker = local / "OWNER.json"
    if local.is_symlink():
        raise ValueError("Tokenizer pipeline scratch may not be a symlink")
    if not marker.exists() or _read(marker).get("identity") != identity:
        tk._stage(scratch, "tokenizer-pipeline")
        _seal(marker, {"identity": identity})
    return local


def _sample_gate(sample: Mapping[str, Any], config: _Config) -> bool:
    selected = _integer(sample.get("selected_bytes"), "selected_bytes", minimum=0)
    coverage = sample.get("coverage", {})
    return (
        sample.get("ready") is True and sample.get("target_met") is True
        and _integer(sample.get("training_bytes", selected), "training_bytes", minimum=0) == selected
        and _integer(sample.get("heldout_bytes", 0), "heldout_bytes", minimum=0) == 0
        and config.target <= selected <= config.target + config.overshoot_limit
        and sample.get("target_bytes") == config.target
        and sample.get("overshoot_bytes", selected - config.target) == selected - config.target
        and sample.get("overshoot_limit_bytes", config.overshoot_limit) == config.overshoot_limit
        and all(
            _integer(coverage.get(category, {}).get("selected_bytes", 0), "selected category bytes", minimum=0)
            >= config.minimums[category]
            for category in REQUIRED_CATEGORIES
        )
        and all(
            _integer(sample.get(f"{field}_coverage", {}).get(key, {}).get("selected_bytes", 0),
                     f"selected {field} bytes", minimum=0) >= minimum
            for field, minimums in (("source", config.source_minimums), ("language", config.language_minimums))
            for key, minimum in minimums.items()
        )
    )


def _sample_inputs(root: Path, work: Path, state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    inputs = _inputs(root, work, state)
    return inputs, _sample_plan_id(state, inputs)


def _sample_plan_id(state: Mapping[str, Any], inputs: Sequence[Mapping[str, Any]]) -> str:
    return digest_json({
        "run_sha256": state["run_sha256"], "generation": state["generation"],
        "recipe_sha256": state.get("recipe_sha256"),
        "generation_descriptor_sha256": state["generation_descriptor_sha256"], "inputs": inputs,
    })


def _batched_rows(cursor: sqlite3.Cursor, batch_size: int) -> Iterator[sqlite3.Row]:
    while batch := cursor.fetchmany(batch_size):
        yield from batch


def _build_sample(
    root: Path, work: Path, state: Mapping[str, Any], config: _Config,
    scratch: Path, inputs: list[dict[str, Any]], sample_id: str,
    *, opener: QuotaOpener | None = None,
) -> tuple[dict[str, Any], Path | None]:
    destination = work / "samples" / sample_id
    if destination.exists():
        sample = tk._validate_sample(destination)
        if sample["identity"].get("pipeline_run_sha256") != state["run_sha256"]:
            raise ValueError("Existing tokenizer sample belongs to a different frozen RUN")
        if not _sample_gate(sample, config):
            raise ValueError("Existing frozen sample does not meet the full production target")
        return sample, destination
    local = _scratch_workspace(scratch, work, state, config)
    for suffix in ("", "-journal", "-wal", "-shm"):
        (local / f"candidates.sqlite3{suffix}").unlink(missing_ok=True)
    database = tk._local_database(local / "candidates.sqlite3", config.scratch_bytes)
    stage = _output_stage(work / "samples", ".sample-incomplete", opener)
    try:
        database.executescript(
            """
            CREATE TABLE candidates (
                category TEXT NOT NULL, rank TEXT NOT NULL, content_hash TEXT NOT NULL,
                source_number INTEGER NOT NULL, source_row INTEGER NOT NULL,
                doc_id TEXT NOT NULL, utf8_bytes INTEGER NOT NULL,
                source_id TEXT NOT NULL, language TEXT NOT NULL,
                PRIMARY KEY(category, rank, content_hash), UNIQUE(category, content_hash)
            ) WITHOUT ROWID;
            CREATE TABLE selected (
                source_number INTEGER NOT NULL, source_row INTEGER NOT NULL,
                category TEXT NOT NULL, rank TEXT NOT NULL, content_hash TEXT NOT NULL,
                doc_id TEXT NOT NULL, utf8_bytes INTEGER NOT NULL,
                PRIMARY KEY(source_number, source_row)
            ) WITHOUT ROWID;
            """
        )
        coverage = {
            category: {
                "minimum_bytes": config.minimums[category], "available_unique_bytes": 0,
                "available_unique_documents": 0, "selected_bytes": 0, "selected_documents": 0,
                "oversized_documents": 0, "duplicate_documents": 0,
            }
            for category in REQUIRED_CATEGORIES
        }
        required_coverage = {
            field: {
                key: {"minimum_bytes": minimum, "available_unique_bytes": 0, "selected_bytes": 0}
                for key, minimum in minimums.items()
            }
            for field, minimums in (("source_id", config.source_minimums), ("language", config.language_minimums))
        }
        for field, requirements in required_coverage.items():
            if requirements:
                database.execute(f"CREATE INDEX candidates_{field} ON candidates({field}, rank, content_hash)")
        candidate_count = records = uncapped = 0
        for source_number, source in enumerate(inputs):
            parquet = tk._prepared_file(Path(source["path"]))
            source_row = 0
            for batch in parquet.iter_batches(batch_size=config.batch_size, columns=tk.PREPARED_SCHEMA.names):
                with database:
                    for row in batch.to_pylist():
                        byte_count = tk._checked_row(row)
                        category = row["category"]
                        if category not in coverage:
                            uncapped += 1
                        elif byte_count > config.max_document_bytes:
                            coverage[category]["oversized_documents"] += 1
                        elif byte_count:
                            rank = digest_json({"seed": config.seed, "stratum": category, "content_hash": row["content_hash"]})
                            cursor = database.execute(
                                "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (category, rank, row["content_hash"], source_number, source_row, row["doc_id"],
                                 byte_count, row["source_id"], row["language"]),
                            )
                            if cursor.rowcount:
                                candidate_count += 1
                                coverage[category]["available_unique_bytes"] += byte_count
                                coverage[category]["available_unique_documents"] += 1
                                for field, requirements in required_coverage.items():
                                    if row[field] in requirements:
                                        requirements[row[field]]["available_unique_bytes"] += byte_count
                                if candidate_count > config.max_candidates:
                                    raise ValueError("Tokenizer sampler exceeded its explicit candidate-document bound")
                            else:
                                coverage[category]["duplicate_documents"] += 1
                        source_row += 1
                        records += 1
            if source_row != source["rows"]:
                raise ValueError("Tokenizer sample input row coverage changed")
            tk._unchanged(source)
        available = sum(values["available_unique_bytes"] for values in coverage.values())
        deficient = [
            category for category in REQUIRED_CATEGORIES
            if coverage[category]["available_unique_bytes"] < config.minimums[category]
        ]
        missing_sources = [
            key for key, values in required_coverage["source_id"].items()
            if values["available_unique_bytes"] < values["minimum_bytes"]
        ]
        missing_languages = [
            key for key, values in required_coverage["language"].items()
            if values["available_unique_bytes"] < values["minimum_bytes"]
        ]
        if available < config.target or deficient or missing_sources or missing_languages:
            return {
                "ready": False, "target_met": False, "available_unique_bytes": available,
                "coverage": coverage, "missing_categories": deficient,
                "target_bytes": config.target, "records_scanned": records,
                "source_coverage": required_coverage["source_id"], "language_coverage": required_coverage["language"],
                "missing_sources": missing_sources, "missing_languages": missing_languages,
            }, None
        streams = {
            category: _batched_rows(database.execute(
                "SELECT * FROM candidates WHERE category=? ORDER BY rank, content_hash", (category,),
            ), config.batch_size)
            for category in REQUIRED_CATEGORIES
        }
        total = 0

        def select(row: sqlite3.Row) -> bool:
            nonlocal total
            inserted = database.execute(
                "INSERT OR IGNORE INTO selected VALUES (?, ?, ?, ?, ?, ?, ?)",
                tuple(row[key] for key in (
                    "source_number", "source_row", "category", "rank", "content_hash", "doc_id", "utf8_bytes",
                )),
            )
            if not inserted.rowcount:
                return False
            category = row["category"]
            coverage[category]["selected_bytes"] += row["utf8_bytes"]
            coverage[category]["selected_documents"] += 1
            for field, requirements in required_coverage.items():
                if row[field] in requirements:
                    requirements[row[field]]["selected_bytes"] += row["utf8_bytes"]
            total += row["utf8_bytes"]
            return True

        def take(category: str) -> bool:
            for row in streams[category]:
                if select(row):
                    return True
            return False

        with database:
            for field, requirements in required_coverage.items():
                for key, values in requirements.items():
                    candidates = _batched_rows(database.execute(
                        f"SELECT * FROM candidates WHERE {field}=? ORDER BY rank, content_hash", (key,),
                    ), config.batch_size)
                    while values["selected_bytes"] < values["minimum_bytes"]:
                        row = next(candidates, None)
                        if row is None:
                            raise ValueError("Candidate inventory lost required source/language coverage")
                        select(row)
            for category in REQUIRED_CATEGORIES:
                while coverage[category]["selected_bytes"] < config.minimums[category]:
                    if not take(category):
                        raise ValueError("Candidate inventory lost required category coverage")
            queue = [(coverage[category]["selected_bytes"], category) for category in REQUIRED_CATEGORIES]
            heapq.heapify(queue)
            while total < config.target and queue:
                _bytes, category = heapq.heappop(queue)
                if take(category):
                    heapq.heappush(queue, (coverage[category]["selected_bytes"], category))
        if not config.target <= total <= config.target + config.overshoot_limit:
            raise ValueError("Tokenizer sample target/whole-document overshoot invariant failed")
        policy = tk._sample_policy(
            {category: config.target + config.overshoot_limit for category in REQUIRED_CATEGORIES},
            REQUIRED_CATEGORIES, ("category",), config.minimums, config.seed,
        )
        manifest = {
            "schema": "metis17.tokenizer-sample/v1", "created_at": utc_now(),
            "identity": {
                "inputs": inputs, "sampling_policy": policy,
                "selection": "required-source-language-minima-then-category-balanced-hash-ranks/v1",
                "pipeline_run_sha256": state["run_sha256"],
                "recipe_sha256": state.get("recipe_sha256"),
                "generation": state["generation"],
                "generation_descriptor_sha256": state["generation_descriptor_sha256"],
                "stage_receipts": list(state["stage_receipts"].values()),
                "required_source_minimum_bytes": config.source_minimums,
                "required_language_minimum_bytes": config.language_minimums,
            },
            "production": config.production, "sample_id": sample_id,
            "ready": True, "target_met": True, "target_bytes": config.target,
            "overshoot_bytes": total - config.target, "overshoot_limit_bytes": config.overshoot_limit,
            "maximum_document_bytes": config.max_document_bytes,
            "selected_bytes": total,
            "training_bytes": total, "heldout_bytes": 0,
            "selected_documents": sum(values["selected_documents"] for values in coverage.values()),
            "missing_strata": [], "coverage": coverage, "records_scanned": records,
            "source_coverage": required_coverage["source_id"], "language_coverage": required_coverage["language"],
            "uncapped_documents": uncapped, "candidate_documents": candidate_count,
            "samples_path": "samples.parquet", "samples_sha256": "0" * 64,
        }
        receipt_allowance = len(json.dumps({**manifest, "receipt_sha256": "0" * 64}, indent=2, sort_keys=True).encode()) + 4096
        if receipt_allowance > _CONTROL_LIMIT or receipt_allowance >= config.sample_output_bytes:
            raise ValueError("Tokenizer sample receipt exceeds its reserved output budget")
        sink = _BoundedFile(
            stage / "samples.parquet", config.sample_output_bytes - receipt_allowance, opener=opener,
        )
        try:
            with pq.ParquetWriter(sink, tk.SAMPLE_SCHEMA, compression="zstd") as writer:
                cursor = database.execute("SELECT * FROM selected ORDER BY source_number, source_row")
                while selected := cursor.fetchmany(config.batch_size):
                    rows = [
                        {
                            "source_number": row["source_number"], "source_shard": inputs[row["source_number"]]["path"],
                            "source_row": row["source_row"], "doc_id": row["doc_id"],
                            "content_hash": row["content_hash"], "stratum": row["category"],
                            "rank": row["rank"], "utf8_bytes": row["utf8_bytes"],
                        }
                        for row in selected
                    ]
                    writer.write_table(pa.Table.from_pylist(rows, schema=tk.SAMPLE_SCHEMA))
        finally:
            sink.close()
        manifest["samples_sha256"] = sha256_file(stage / "samples.parquet")
        result = _seal(stage / "SAMPLE_RECEIPT.json", manifest, limit=receipt_allowance, opener=opener)
        tk._validate_sample(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tk._publish(stage, destination)
        return result, destination
    finally:
        database.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            (local / f"candidates.sqlite3{suffix}").unlink(missing_ok=True)
        if stage.exists():
            _remove_output_stage(stage, opener)


def _attempt_due(state: Mapping[str, Any], config: _Config) -> bool:
    characters = state["admitted_characters"]
    if any(characters[category] * 4 < config.minimums[category] for category in REQUIRED_CATEGORIES):
        return False
    if sum(characters.values()) * 4 < config.target:
        return False
    for field, minimums in (("source", config.source_minimums), ("language", config.language_minimums)):
        if any(state.get(f"admitted_{field}_characters", {}).get(key, 0) * 4 < minimum
               for key, minimum in minimums.items()):
            return False
    previous = state.get("last_sample_attempt")
    if previous is None:
        return True
    if previous["inventory_sha256"] == digest_json(state["chunks"]):
        return False
    if previous.get("missing_sources") or previous.get("missing_languages"):
        return all(
            state.get(f"admitted_{field}_characters", {}).get(key, 0)
            >= previous[f"next_{field}_characters"][key]
            for field, plural in (("source", "sources"), ("language", "languages"))
            for key in previous.get(f"missing_{plural}", [])
        )
    deficient = previous.get("missing_categories", [])
    if deficient:
        return all(characters[category] >= previous["next_category_characters"][category] for category in deficient)
    return sum(characters.values()) >= previous["next_total_characters"]


def _next_threshold(observed_characters: int, measured_bytes: int, wanted_bytes: int) -> int:
    if measured_bytes <= 0:
        return max(observed_characters + 1, observed_characters * 2)
    projected = (observed_characters * wanted_bytes + measured_bytes - 1) // measured_bytes
    return max(observed_characters + 1, min(observed_characters * 2, projected))


def _sample_state(work: Path, state: dict[str, Any], config: _Config, sample: Mapping[str, Any], path: Path) -> None:
    if not _sample_gate(sample, config):
        raise ValueError("Tokenizer training cannot start from category minima without the full target")
    if (
        sample["identity"].get("generation") != state["generation"]
        or sample["identity"].get("generation_descriptor_sha256") != state["generation_descriptor_sha256"]
        or sample["identity"].get("pipeline_run_sha256") != state["run_sha256"]
        or sample["identity"].get("recipe_sha256") != state.get("recipe_sha256")
        or sample.get("sample_id") != path.name
        or sample.get("sample_id") != _sample_plan_id(state, sample["identity"]["inputs"])
    ):
        raise ValueError("Ready sample does not match the frozen RUN and eligibility generation")
    state.update({
        "status": "SAMPLE_READY", "activity": "awaiting_training_step",
        "sample": {
            "path": str(path.relative_to(work)), "receipt_sha256": sample["receipt_sha256"],
            "selected_bytes": sample["selected_bytes"], "selected_documents": sample["selected_documents"],
            "target_bytes": config.target, "overshoot_bytes": sample["overshoot_bytes"],
        },
    })
    state.pop("error", None)
    state.pop("pending_sample", None)
    state.pop("sampling_failure_inventory", None)


def _read_planned_sample(
    work: Path, state: Mapping[str, Any], config: _Config, *, verify_metadata: bool = True,
) -> tuple[dict[str, Any], Path]:
    directory = under_root(work, state["sample"]["path"])
    sample = tk._validate_sample(directory) if verify_metadata else _read(directory / "SAMPLE_RECEIPT.json")
    if sample["receipt_sha256"] != state["sample"]["receipt_sha256"] or not _sample_gate(sample, config):
        raise ValueError("Frozen tokenizer sample changed or does not meet the full target")
    if sample.get("production") is not config.production:
        raise ValueError("A test sample cannot be used to freeze a production tokenizer")
    if sample["identity"].get("pipeline_run_sha256") != state["run_sha256"]:
        raise ValueError("Tokenizer sample belongs to a different RUN")
    if sample["identity"].get("recipe_sha256") != state.get("recipe_sha256"):
        raise ValueError("Tokenizer sample belongs to a different recipe")
    if (
        sample["identity"].get("generation") != state["generation"]
        or sample["identity"].get("generation_descriptor_sha256") != state["generation_descriptor_sha256"]
        or sample.get("sample_id") != directory.name
        or sample.get("sample_id") != _sample_plan_id(state, sample["identity"]["inputs"])
    ):
        raise ValueError("Tokenizer sample belongs to a different or changed eligibility generation")
    return sample, directory


def _model_matches(directory: Path, config: _Config, sample: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = tk.load_tokenizer17(directory, production=config.production)
    release = _read(directory / tk.TOKENIZER_RELEASE)
    if (
        release.get("production") is not config.production
        or release.get("requested_vocabulary_size") != config.vocabulary_size
        or release.get("minimum_frequency") != config.minimum_frequency
        or release.get("special_tokens") != {token: index for index, token in enumerate(config.special_tokens)}
        or release.get("eos_token") != config.special_tokens[0]
        or release.get("training", {}).get("documents") != sample["selected_documents"]
        or release.get("training", {}).get("utf8_bytes") != sample["selected_bytes"]
        or tokenizer.get_vocab_size(with_added_tokens=True) != release["vocabulary_size"]
    ):
        raise ValueError("Tokenizer artifact does not match the intended full sample and explicit seven-token recipe")
    return release


def _frozen_model(work: Path, state: Mapping[str, Any], config: _Config, sample: Mapping[str, Any]) -> dict[str, Any]:
    release = _model_matches(work / "artifact", config, sample)
    provenance = _read(work / "TRAINING_PROVENANCE.json")
    if (
        release["tokenizer_sha256"] != state["tokenizer_sha256"]
        or release["receipt_sha256"] != state["tokenizer_release_sha256"]
        or provenance["receipt_sha256"] != state["training_provenance_sha256"]
        or provenance.get("recipe_sha256") != state.get("recipe_sha256")
    ):
        raise ValueError("Frozen tokenizer artifact, release, or training provenance changed")
    return release


def _publish_model(
    local: Path, destination: Path, maximum: int, *, opener: QuotaOpener | None = None,
) -> None:
    names = ("tokenizer.json", tk.TOKENIZER_RELEASE)
    if destination.is_symlink():
        raise ValueError("Tokenizer artifact directory cannot be redirected by a symlink")
    sizes = sum((local / name).stat().st_size for name in names)
    if sizes > maximum:
        raise ValueError("Tokenizer artifact exceeds its reserved durable output budget")
    if (destination / tk.TOKENIZER_RELEASE).exists():
        raise ValueError("Refusing to overwrite an immutable tokenizer artifact")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        _unlink_output(destination / name, opener)
    stage = _output_stage(destination, ".artifact-incomplete", opener)
    try:
        remaining = maximum
        for name in names:
            target = _BoundedFile(stage / name, remaining, opener=opener)
            try:
                with (local / name).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
            finally:
                target.close()
            remaining -= (stage / name).stat().st_size
        for name in names:
            _replace_output(stage / name, destination / name, opener)
        tk._sync_directory(stage)
        tk._sync_directory(destination)
    finally:
        if stage.exists():
            _remove_output_stage(stage, opener)
            tk._sync_directory(destination)


def _train(
    work: Path, state: dict[str, Any], config: _Config, scratch: Path,
    reserve_output: ReserveOutput | None, working_budget: Any | None, root: Path,
) -> dict[str, Any]:
    sample, sample_dir = _read_planned_sample(
        work, state, config,
        verify_metadata=state.get("validated_training_sample") != state["sample"]["receipt_sha256"],
    )
    state["validated_training_sample"] = sample["receipt_sha256"]
    destination = work / "artifact"
    planned = state.get("training_started_for") == sample["receipt_sha256"]
    if (destination / tk.TOKENIZER_RELEASE).exists():
        if not planned and state.get("activity") != "trained":
            raise ValueError("Refusing an unmanaged tokenizer artifact without a planned training transition")
        release = _model_matches(destination, config, sample)
    else:
        if (
            not planned
            and any((destination / name).exists() for name in ("tokenizer.json", ".artifact-incomplete"))
        ):
            raise ValueError("Refusing an unmanaged partial tokenizer artifact")
        if state.get("training_failure_sample") == sample["receipt_sha256"]:
            raise ValueError("The frozen sample already failed training; repeated polling will not retrain it")
        request = _reservation_request(work, state, config, "tokenizer_model", sample["receipt_sha256"], config.model_output_bytes)
        state["reservation_request"] = request
        with _allocation(root, destination, request, reserve_output, working_budget) as (grant, opener):
            state.update({
                "status": "SAMPLE_READY", "activity": "training", "reservation": grant,
                "training_started_for": sample["receipt_sha256"],
            })
            _save(work, state)
            scratch = tk._local_scratch(scratch, work, max(config.scratch_bytes, config.model_output_bytes))
            with tk._locked(scratch / ".tokenizer-worker.lock", blocking=False):
                local = _scratch_workspace(scratch, work, state, config)
                training_key = {
                    "run_sha256": state["run_sha256"], "sample_sha256": sample["receipt_sha256"],
                    "recipe_sha256": state.get("recipe_sha256"),
                }
                marker = local / "TRAINING.json"
                artifact = local / "artifact"
                if marker.exists() and _read(marker).get("identity") != training_key:
                    if artifact.exists():
                        shutil.rmtree(artifact)
                    marker.unlink()
                if not marker.exists():
                    if artifact.exists():
                        raise ValueError("Node-local tokenizer artifact has no training provenance")
                    _seal(marker, {"identity": training_key})
                if not artifact.exists():
                    try:
                        tk.train_tokenizer17(
                            tk.iter_tokenizer_sample17(sample_dir, production=config.production, batch_size=config.batch_size),
                            artifact, vocabulary_size=config.vocabulary_size,
                            special_tokens=config.special_tokens, minimum_frequency=config.minimum_frequency,
                            production=config.production,
                        )
                    except Exception:
                        state["training_failure_sample"] = sample["receipt_sha256"]
                        raise
                release = _model_matches(artifact, config, sample)
                _publish_model(artifact, destination, config.model_output_bytes - 4096, opener=opener)
    provenance = {
        "schema": "metis17.tokenizer-training-provenance/v1",
        "run_sha256": state["run_sha256"], "production": config.production,
        "recipe_sha256": state.get("recipe_sha256"),
        "generation": state["generation"],
        "generation_descriptor_sha256": state["generation_descriptor_sha256"],
        "sample_receipt_sha256": sample["receipt_sha256"],
        "sample_path": state["sample"]["path"], "selected_bytes": sample["selected_bytes"],
        "training_bytes": sample["selected_bytes"], "heldout_bytes": 0,
        "target_bytes": config.target, "overshoot_bytes": sample["overshoot_bytes"],
        "tokenizer_sha256": release["tokenizer_sha256"],
        "tokenizer_release_sha256": release["receipt_sha256"],
    }
    path = work / "TRAINING_PROVENANCE.json"
    if path.exists() and _payload(_read(path)) != provenance:
        raise ValueError("Tokenizer training provenance changed")
    if not path.exists():
        _seal(path, provenance, limit=4096)
    state.update({
        "status": "TRAINED", "activity": "trained", "tokenizer_path": "artifact",
        "tokenizer_sha256": release["tokenizer_sha256"],
        "tokenizer_release_sha256": release["receipt_sha256"],
        "training_provenance_sha256": _read(path)["receipt_sha256"],
        "id_cache": {
            "status": "READY_FOR_PARTITIONS", "reason": "generation_is_explicitly_reserved_per_partition",
            "api": "tokenize_ready_partition",
        },
    })
    state.pop("error", None)
    return _save(work, state)


def run_tokenizer_step(
    root: Path, *, scratch_dir: Path, generation: str,
    eligible_events: Iterable[Mapping[str, Any]] | None = None,
    reserve_output: ReserveOutput | None = None, working_budget: Any | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Advance one sealed WAITING -> SAMPLE_READY -> TRAINED step, or report BLOCKED.

    RUN.json contains tokenizer settings and limits.max_working_bytes. Production
    target/minimum floors cannot be reduced. Test mode requires two explicit
    latches and publishes only under tokenizer/test, never the production path.
    generation is an explicit canonical policy-generation hash; its sealed
    preparation/generations/<hash>.json descriptor and tokenizer state are pinned
    independently of every other generation. Superseded/unscoped events are
    counted and skipped before any receipt or Parquet is opened.
    Events supplied explicitly must be an incremental bounded batch. Otherwise
    root/events/eligible/*.jsonl is consumed using append-only byte checkpoints.
    After training, only the latest batch is retained for partition dispatch;
    pass an explicit stage receipt to tokenize_ready_partition for older batches.
    A separately frozen RECIPE.json overrides tokenizer settings, never RUN.json.
    reserve_output(request) must hold an atomic parent reservation until context
    exit and yield reserved_bytes, used_working_bytes, other_reserved_bytes.
    Used/other working bytes EXCLUDE raw: the current sealed limits reserve the
    entire approved raw ceiling inside total working capacity, never on top.
    By default production uses WorkingBudget(root).quota().open for every sample
    and model data write. working_budget may supply that already-created object;
    reserve_output is an alternative atomic whole-upper-bound reservation API.
    """
    root = Path(root).resolve()
    tk._digest(generation, "eligibility generation")
    base = root / "tokenizer" / "test" if test_mode else root / "tokenizer"
    work = under_root(root, str((base / "generations" / generation).relative_to(root)))
    work.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] | None = None
    lock = tk._locked(work / ".pipeline.lock", blocking=False)
    acquired = False
    try:
        lock.__enter__()
        acquired = True
        with (root / "RUN.json").open("rb") as run_stream:
            run_path = root / "RUN.json"
            run_bytes = run_stream.read(_CONTROL_LIMIT + 1)
            if len(run_bytes) > _CONTROL_LIMIT:
                raise ValueError("Frozen RUN exceeds the bounded configuration size")
            run_sha = hashlib.sha256(run_bytes).hexdigest()
            run = json.loads(run_bytes)
            if "receipt_sha256" in run:
                seal = run.pop("receipt_sha256")
                if seal != digest_json(run):
                    raise ValueError("Frozen RUN canonical receipt seal mismatch")
            state = _read(work / "STATE.json") if (work / "STATE.json").exists() else None
            if state is not None and state.get("run_sha256") != run_sha:
                raise ValueError("Frozen RUN.json changed; start a new run instead of mutating tokenizer inputs")
            live_limits = read_receipt(root / "limits.json") if (root / "limits.json").exists() else None
            config, recipe_sha = _recipe_config(root, run, run_sha, test_mode, live_limits=live_limits)
            if state is None:
                state = _initial(run_sha, config.production, generation, recipe_sha)
            if state.get("recipe_bound", True) and state.get("recipe_sha256") != recipe_sha:
                raise ValueError("Frozen tokenizer recipe changed or disappeared after generation initialization")
            state["recipe_sha256"], state["recipe_bound"] = recipe_sha, True
            descriptor = _generation(root, generation)
            if (
                state.get("generation") != generation
                or state.get("generation_descriptor_sha256") not in {"", descriptor["receipt_sha256"]}
            ):
                raise ValueError("Frozen eligibility generation descriptor changed")
            state["generation_descriptor_sha256"] = descriptor["receipt_sha256"]
            if state.get("production") is not config.production:
                raise ValueError("Tokenizer state production mode does not match frozen RUN")
            candidate = copy.deepcopy(state)
            if eligible_events is None:
                events, cursors, next_host = _event_batch(root, candidate, config.max_events_per_step)
            else:
                events = []
                for event in eligible_events:
                    events.append(dict(event))
                    if len(events) > config.max_events_per_step:
                        raise ValueError("Explicit eligible_events must be a bounded incremental batch; no events were checkpointed")
                cursors, next_host = candidate["event_cursors"], candidate["next_host"]
            pending = state.get("pending_sample")
            closed_sample = state.get("sample") or (
                pending and (work / "samples" / pending["sample_id"]).exists()
            )
            if state.get("tokenizer_sha256"):
                sample, _directory = _read_planned_sample(work, state, config, verify_metadata=False)
                _frozen_model(work, state, config, sample)
            if closed_sample:
                _admit_latest(root, work, candidate, events, config)
            else:
                _admit(root, work, candidate, events, config)
            candidate["event_cursors"], candidate["next_host"] = cursors, next_host
            if len(canonical_json(_payload(candidate)).encode("utf-8")) > _CONTROL_LIMIT - 65_536:
                raise ValueError("Tokenizer checkpoint inventory exceeds its control-output bound")
            state = candidate
            if state.get("tokenizer_sha256"):
                state.update({"status": "TRAINED", "activity": "tokenizer_frozen"})
                state.pop("error", None)
                return _save(work, state)
            if state.get("sample"):
                return _train(work, state, config, Path(scratch_dir), reserve_output, working_budget, root)
            pending = state.get("pending_sample")
            if pending:
                directory = work / "samples" / pending["sample_id"]
                if directory.exists():
                    sample = tk._validate_sample(directory)
                    _sample_state(work, state, config, sample, directory)
                    return _save(work, state)
            if state.get("sampling_failure_inventory") == digest_json(state["chunks"]):
                raise ValueError("The same immutable inventory already failed sampling; polling will not rescan it")
            if not _attempt_due(state, config):
                state.update({
                    "status": "WAITING",
                    "activity": "waiting_for_measured_sample_growth" if state.get("last_sample_attempt") else "waiting_for_full_target_and_coverage",
                    "target_bytes": config.target, "required_category_bytes": config.minimums,
                    "required_source_bytes": config.source_minimums,
                    "required_language_bytes": config.language_minimums,
                    "available_bytes_are_metadata_bounds_not_training_credits": True,
                })
                state.pop("error", None)
                return _save(work, state)
            if state["sample_attempts"] >= config.max_attempts:
                raise ValueError("Bounded tokenizer sampling-attempt budget exhausted")
            inputs, sample_id = _sample_inputs(root, work, state)
            request = _reservation_request(work, state, config, "tokenizer_sample", sample_id, config.sample_output_bytes)
            state["reservation_request"] = request
            with _allocation(root, work / "samples", request, reserve_output, working_budget) as (grant, opener):
                scratch = tk._local_scratch(Path(scratch_dir), work, config.scratch_bytes)
                with tk._locked(scratch / ".tokenizer-worker.lock", blocking=False):
                    state["sample_attempts"] += 1
                    state.update({
                        "status": "WAITING", "activity": "sampling",
                        "pending_sample": {"sample_id": sample_id}, "reservation": grant,
                    })
                    _save(work, state)
                    sample, directory = _build_sample(
                        root, work, state, config, scratch, inputs, sample_id, opener=opener,
                    )
            if directory is not None:
                _sample_state(work, state, config, sample, directory)
                return _save(work, state)
            characters = state["admitted_characters"]
            state["last_sample_attempt"] = {
                "inventory_sha256": digest_json(state["chunks"]),
                "available_unique_bytes": sample["available_unique_bytes"],
                "missing_categories": sample["missing_categories"], "coverage": sample["coverage"],
                "next_total_characters": _next_threshold(sum(characters.values()), sample["available_unique_bytes"], config.target),
                "next_category_characters": {
                    category: _next_threshold(
                        characters[category], sample["coverage"][category]["available_unique_bytes"],
                        config.minimums[category],
                    )
                    for category in REQUIRED_CATEGORIES
                },
                "missing_sources": sample["missing_sources"],
                "missing_languages": sample["missing_languages"],
                **{
                    f"next_{field}_characters": {
                        key: state.get(f"admitted_{field}_characters", {}).get(key, 0) + max(
                            1, (minimum - sample[f"{field}_coverage"][key]["available_unique_bytes"] + 3) // 4,
                        )
                        for key, minimum in minimums.items()
                    }
                    for field, minimums in (("source", config.source_minimums), ("language", config.language_minimums))
                },
            }
            state.update({"status": "WAITING", "activity": "waiting_for_measured_sample_growth"})
            state.pop("pending_sample", None)
            return _save(work, state)
    except Exception as error:
        if "already held" in str(error):
            if acquired and state is not None:
                state.update({
                    "status": "SAMPLE_READY" if state.get("sample") else "WAITING",
                    "activity": "waiting_for_node_local_scratch",
                })
                return _save(work, state)
            path = work / "STATE.json"
            if path.exists():
                return _read(path)
            return _seal(work / "BUSY.json", {
                "schema": "metis17.tokenizer-pipeline-busy/v1", "status": "WAITING",
                "activity": "another_tokenizer_step_is_running",
            })
        if state is None:
            state = _initial(
                sha256_file(root / "RUN.json") if (root / "RUN.json").is_file() else "",
                not test_mode, generation, recipe_bound=False,
            )
        if state.get("activity") == "sampling":
            state["sampling_failure_inventory"] = digest_json(state["chunks"])
        state.update({"status": "BLOCKED", "error": {"type": type(error).__name__, "message": str(error)[:4096]}})
        try:
            return _save(work, state)
        except (OSError, ValueError, TypeError):
            return _seal(work / "BLOCKED.json", {
                "schema": "metis17.tokenizer-pipeline-error/v1", "status": "BLOCKED",
                "run_sha256": state["run_sha256"], "error": state["error"],
            })
    finally:
        if acquired:
            lock.__exit__(None, None, None)


def _validated_pipeline_state(
    root: Path, work: Path, generation: str, test_mode: bool,
) -> tuple[dict[str, Any], _Config]:
    state = _read(work / "STATE.json")
    descriptor = _generation(root, generation)
    if state.get("generation") != generation or state.get("generation_descriptor_sha256") != descriptor["receipt_sha256"]:
        raise ValueError("Tokenizer generation does not match its frozen descriptor")
    run, run_sha = _read_run(root)
    if run_sha != state["run_sha256"]:
        raise ValueError("Frozen RUN.json changed")
    live_limits = read_receipt(root / "limits.json") if (root / "limits.json").exists() else None
    config, recipe_sha = _recipe_config(root, run, run_sha, test_mode, live_limits=live_limits)
    if state.get("recipe_sha256") != recipe_sha:
        raise ValueError("Frozen tokenizer recipe changed or disappeared")
    if state.get("production") is not config.production:
        raise ValueError("Tokenizer state production mode does not match its frozen recipe")
    return state, config


def tokenizer_status(root: Path, *, generation: str, test_mode: bool = False) -> dict[str, Any]:
    """Read verified status without checkpoints, locks, scratch, or reservations.

    Selected UTF-8 bytes come from the pinned training-sample receipt, never
    inventory estimates. A trained artifact is loaded and validated on every
    call, without rescanning the sample's Parquet metadata or corpus text.
    """
    result: dict[str, Any] = {
        "schema": "metis17.tokenizer-status/v1", "generation": generation,
        "status": "WAITING", "ready": False, "production": not test_mode,
        "activity": "waiting_for_tokenizer_state", "target_bytes": None,
        "selected_utf8_bytes": 0, "selected_documents": 0, "overshoot_bytes": 0,
        "vocabulary_size": None, "run_sha256": None, "recipe_sha256": None,
        "state_sha256": None, "generation_descriptor_sha256": None, "sample_sha256": None,
        "tokenizer_sha256": None, "tokenizer_release_sha256": None,
        "training_provenance_sha256": None,
    }
    try:
        root = Path(root).resolve()
        tk._digest(generation, "eligibility generation")
        base = root / "tokenizer" / "test" if test_mode else root / "tokenizer"
        work = under_root(root, str((base / "generations" / generation).relative_to(root)))
        if not (work / "STATE.json").exists():
            return result
        state, config = _validated_pipeline_state(root, work, generation, test_mode)
        result.update({
            "status": state["status"], "activity": state.get("activity"), "production": config.production,
            "target_bytes": config.target, "run_sha256": state["run_sha256"],
            "recipe_sha256": state.get("recipe_sha256"), "state_sha256": state["receipt_sha256"],
            "generation_descriptor_sha256": state["generation_descriptor_sha256"],
        })
        if state.get("error"):
            result["error"] = state["error"]
        sample = None
        if state.get("sample"):
            sample, _directory = _read_planned_sample(work, state, config, verify_metadata=False)
            result.update({
                "selected_utf8_bytes": sample["selected_bytes"], "selected_documents": sample["selected_documents"],
                "overshoot_bytes": sample["overshoot_bytes"], "sample_sha256": sample["receipt_sha256"],
            })
        if state.get("tokenizer_sha256") or state["status"] == "TRAINED":
            if sample is None:
                raise ValueError("Frozen tokenizer has no validated training sample")
            release = _frozen_model(work, state, config, sample)
            result.update({
                "vocabulary_size": release["vocabulary_size"], "tokenizer_sha256": release["tokenizer_sha256"],
                "tokenizer_release_sha256": release["receipt_sha256"],
                "training_provenance_sha256": state["training_provenance_sha256"],
                "ready": state["status"] == "TRAINED",
            })
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result.update({
            "status": "BLOCKED", "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)[:4096]},
        })
        return result


def tokenize_ready_partition(
    root: Path, input_paths: Sequence[Path], *, scratch_dir: Path, partition_id: str, generation: str,
    reserve_output: ReserveOutput | None = None, working_budget: Any | None = None,
    limits: tk.TokenCacheLimits17 | None = None, test_mode: bool = False,
    stage_receipt_path: Path | str | None = None, stage_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Explicit post-freeze ID generation; retain stable partition IDs for replay.

    Each partition is caller-assigned, bounded and separately reserved. Independent
    partitions can duplicate caching; count/pack use existing offset receipts.
    WorkingQuota.reserve authorizes the complete upper bound before invoking the
    existing path-writing engine; reconcile accounts the actual result on exit.
    The returned wrapper provides partition_root and the core tokenization receipt.
    An explicit stage_receipt_path plus canonical stage_receipt_sha256 admits
    only this partition, without appending to frozen sampling state. Otherwise
    paths must belong to the sample inventory or the latest admitted event batch.
    """
    root = Path(root).resolve()
    tk._digest(generation, "eligibility generation")
    base = root / "tokenizer" / "test" if test_mode else root / "tokenizer"
    work = under_root(root, str((base / "generations" / generation).relative_to(root)))
    state, config = _validated_pipeline_state(root, work, generation, test_mode)
    if state.get("status") != "TRAINED":
        raise ValueError("Token-ID output requires the truly frozen tokenizer from the unchanged RUN")
    sample, _sample_dir = _read_planned_sample(work, state, config, verify_metadata=False)
    _frozen_model(work, state, config, sample)
    limits = tk.TokenCacheLimits17() if limits is None else limits
    limits.validate()
    paths = [Path(path).resolve() for path in input_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Token partition inputs must be distinct")
    if not paths or len(paths) > min(config.max_chunks, limits.max_input_paths):
        raise ValueError("Token partition requires a bounded nonempty input inventory")
    relative_paths = {str(path.relative_to(root)) for path in paths}
    if (stage_receipt_path is None) != (stage_receipt_sha256 is None):
        raise ValueError("Explicit partition admission requires both stage receipt path and SHA256")
    direct = None
    if stage_receipt_path is not None:
        receipt_path = under_root(root, str(stage_receipt_path))
        receipt = _read(receipt_path)
        direct = _partition_inputs(root, work, {
            "receipt_path": str(receipt_path.relative_to(root)),
            "stage_receipt_sha256": stage_receipt_sha256, "generation": generation,
            "source_id": receipt.get("source_id"), "object_id": receipt.get("object_id"),
        }, config, paths=relative_paths)
    records = metadata_budget = characters = 0
    for path in paths:
        relative = str(path.relative_to(root))
        item = (
            direct.get(relative) if direct is not None
            else state["chunks"].get(relative, state.get("recent_partition_inputs", {}).get(relative))
        )
        if item is None:
            raise ValueError("Token partition contains a path not admitted by an immutable ELIGIBLE receipt")
        inventory_path = item.get("inventory_path", f"inventory/{item.get('key')}.json")
        inventory = _read(under_root(work, inventory_path))
        if (
            inventory["source"]["path"] != relative
            or inventory["source"]["sha256"] != item["sha256"]
            or sha256_file(path) != inventory["source"]["sha256"]
        ):
            raise ValueError("Admitted token partition input changed")
        records += inventory["source"]["rows"]
        if "total_characters" in inventory:
            characters += inventory["total_characters"]
        else:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=config.batch_size, columns=["character_count"]):
                characters += sum(_integer(row["character_count"], "character_count", minimum=0) for row in batch.to_pylist())
        metadata_budget += inventory["source"]["rows"] * (
            4 * (inventory["max_metadata_row_bytes"] + len(str(path).encode()) + 512) + 4096
        )
    if records > limits.max_documents:
        raise ValueError("Partition physical record count exceeds its explicitly bounded metadata allocation")
    # The reservation includes raw IDs, source metadata, cache offsets, and receipts.
    maximum = min(limits.max_token_bytes, characters * 16) + metadata_budget + limits.max_shards * 8192 + 64 * 1024**2
    if maximum > config.id_output_bytes:
        raise ValueError("Token partition cannot fit its configured ID-output budget")
    key = digest_json({"partition_id": partition_id, "inputs": [str(path) for path in paths], "tokenizer": state["tokenizer_sha256"]})
    request = _reservation_request(work, state, config, "tokenizer_ids", key, maximum)
    if (work / "ids").is_symlink():
        raise ValueError("Token-ID namespace may not be redirected by a symlink")
    output = under_root(work, f"ids/{partition_id}")
    session = tk.TokenizationSession17(
        output, work / "artifact", scratch_dir=scratch_dir, partition_id=partition_id,
        production=config.production, limits=limits,
    )

    def execute(opener: QuotaOpener | None) -> dict[str, Any]:
        with session:
            receipt = session.tokenize_parquet(paths, batch_size=config.batch_size)
        return _seal(output / "PARTITION_RECEIPT.json", {
            "schema": "metis17.tokenized-partition/v1", "generation": generation,
            "partition_id": partition_id, "partition_root": str(output.relative_to(root)),
            "tokenizer_sha256": state["tokenizer_sha256"], "tokenization": receipt,
            "recipe_sha256": state.get("recipe_sha256"),
            "stage_receipt_sha256": stage_receipt_sha256,
        }, opener=opener)

    if reserve_output is not None:
        if working_budget is not None:
            raise ValueError("Choose one authoritative token-ID budget provider")
        with _reserved(request, reserve_output):
            return execute(None)
    if working_budget is None:
        if not config.production:
            raise ValueError("Test token-ID output requires an explicit budget provider")
        from .storage import WorkingBudget

        working_budget = WorkingBudget(root)
    _budget_snapshot(working_budget, config.max_working_bytes, config.raw_reservation_bytes)
    namespace = "tokenizer_ids:" + digest_json({
        "generation": generation, "partition_id": partition_id, "production": config.production,
    })
    with working_budget.quota(namespace, output) as quota:
        existing = _namespace_bytes(output, limits.max_shards * 8 + limits.max_input_paths * 8 + 512)
        ceiling = existing + maximum
        if ceiling > config.id_output_bytes:
            raise ValueError("Existing partition plus bounded growth exceeds the configured ID-output ceiling")
        if quota.reserve(ceiling) < ceiling:
            raise ValueError("WorkingBudget did not authorize the complete token-ID output bound")
        try:
            return execute(_QuotaAccess(quota, ceiling))
        finally:
            quota.reconcile()


def _namespace_bytes(directory: Path, max_files: int) -> int:
    total = files = 0
    for parent, directories, names in os.walk(directory, followlinks=False):
        if any((Path(parent) / name).is_symlink() for name in directories + names):
            raise ValueError("Token-ID namespace may not contain symlinks")
        for name in names:
            files += 1
            if files > max_files:
                raise ValueError("Token-ID namespace exceeds its bounded file inventory")
            total += (Path(parent) / name).stat().st_size
    return total
