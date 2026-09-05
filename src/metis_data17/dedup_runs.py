from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import contextlib
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .common import canonical_json, digest_json, read_receipt, sha256_file, under_root
from .dedup_locks import metadata_lock
from .dedup_receipts import eligible_stage
from .dedup_storage import storage_descriptor


HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
PREP_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("content_hash", pa.string()), ("dedup_hash", pa.string()),
    ("source_id", pa.string()), ("object_id", pa.string()), ("text", pa.string()),
    ("metadata_json", pa.string()), ("priority", pa.int32()), ("quality_score", pa.float64()),
    ("language", pa.string()), ("category", pa.string()), ("character_count", pa.int64()),
])
REFERENCE_SCHEMA = pa.schema([
    field for field in PREP_SCHEMA if field.name != "text"
] + [
    pa.field("occurrence_id", pa.string()), pa.field("prepared_path", pa.string()),
    pa.field("prepared_row", pa.int64()), pa.field("prepared_sha256", pa.string()),
    pa.field("stage_receipt_sha256", pa.string()),
])
WINNER_SCHEMA = pa.schema(list(REFERENCE_SCHEMA) + [pa.field("occurrence_count", pa.int64())])
METADATA_COLUMNS = [name for name in PREP_SCHEMA.names if name != "text"]
MAX_METADATA_BYTES = 65_536
MAX_SOURCE_METADATA_BYTES = 128_000_000
METADATA_REFERENCE = "_metis17_metadata_reference"
COMPARATOR = "priority-desc,known-quality-desc,score-desc,stable-evidence-asc/v1"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def bucket_for(dedup_hash: str, bucket_count: int) -> int:
    """Partition by the full digest, independently of worker/rank assignments."""
    positive_integer(bucket_count, "bucket_count")
    return int(validate_hash(dedup_hash, "dedup_hash"), 16) % bucket_count


def winner_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    priority, quality = row["priority"], row["quality_score"]
    if type(priority) is not int or not -(1 << 31) <= priority < (1 << 31):
        raise ValueError("priority must be int32")
    if (
        isinstance(quality, bool) or not isinstance(quality, (int, float))
        or not math.isfinite(quality) or (quality < 0 and quality != -1)
    ):
        raise ValueError("quality_score must be finite and nonnegative, or -1 (unknown)")
    return (
        -priority, quality == -1, -quality if quality != -1 else 0,
        row["doc_id"], row["content_hash"], row["source_id"], row["object_id"],
        row["prepared_sha256"], row["prepared_row"],
        row["prepared_path"], row["stage_receipt_sha256"],
    )


def occurrence_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return row["dedup_hash"], row["occurrence_id"]


def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*occurrence_key(row), *winner_key(row))


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value: {value}")


def _metadata_object(value: Any) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, str):
        raise ValueError("metadata_json must be a JSON object string")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_SOURCE_METADATA_BYTES:
        raise ValueError("Prepared metadata exceeds the canonical record bound")
    parsed = json.loads(value, parse_constant=_reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must encode an object")
    return parsed, encoded


def _index_metadata(value: str, artifact: Mapping[str, Any], index: int) -> str:
    parsed, encoded = _metadata_object(value)
    if len(encoded) <= MAX_METADATA_BYTES and METADATA_REFERENCE not in parsed:
        return value
    # Real PDF metadata can contain full alternate text (pre_sa_key and
    # no_references). Keep its verified location, not copies in every LSM run.
    return canonical_json({METADATA_REFERENCE: {
        "schema": "metis17.prepared-metadata-reference/v1",
        "path": artifact["path"], "row": index, "prepared_sha256": artifact["sha256"],
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(), "metadata_bytes": len(encoded),
    }})


@lru_cache(maxsize=16)
def _verified_metadata_source(path: str, checksum: str, stamp: tuple[int, ...]) -> None:
    if sha256_file(Path(path)) != checksum:
        raise ValueError("Prepared metadata source checksum changed")


def read_reference_metadata(reference_row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve bounded index metadata without reading the document text column."""
    parsed, _ = _metadata_object(reference_row["metadata_json"])
    if METADATA_REFERENCE not in parsed:
        return parsed
    item = parsed[METADATA_REFERENCE]
    if (
        set(parsed) != {METADATA_REFERENCE} or not isinstance(item, dict)
        or item.get("schema") != "metis17.prepared-metadata-reference/v1"
        or item.get("path") != reference_row["prepared_path"]
        or item.get("row") != reference_row["prepared_row"]
        or item.get("prepared_sha256") != reference_row["prepared_sha256"]
        or type(item.get("row")) is not int or item["row"] < 0
    ):
        raise ValueError("Metadata reference disagrees with its immutable prepared row")
    validate_hash(item.get("metadata_sha256"), "metadata_sha256")
    path = Path(item["path"])
    stat = path.stat()
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    _verified_metadata_source(str(path), item["prepared_sha256"], stamp)
    offset = item["row"]
    with contextlib.closing(pq.ParquetFile(path)) as parquet:
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if offset < count:
                table = parquet.read_row_group(group, columns=["metadata_json"], use_threads=False)
                value = table.column("metadata_json")[offset].as_py()
                metadata, encoded = _metadata_object(value)
                if len(encoded) != item["metadata_bytes"] or hashlib.sha256(encoded).hexdigest() != item["metadata_sha256"]:
                    raise ValueError("Referenced metadata bytes differ from their seal")
                return metadata
            offset -= count
    raise ValueError("Metadata reference row is outside its prepared shard")


def validate_prepared_schema(schema: pa.Schema) -> None:
    for field in PREP_SCHEMA:
        if schema.get_field_index(field.name) < 0 or schema.field(field.name).type != field.type:
            raise ValueError(f"Prepared Parquet requires {field.name}: {field.type}")


def reference(row: dict[str, Any], artifact: Mapping[str, Any], index: int) -> dict[str, Any]:
    for name in ("doc_id", "source_id", "object_id", "language", "category"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"Nonempty prepared {name} required")
    for name in ("content_hash", "dedup_hash"):
        validate_hash(row.get(name), name)
    size = row.get("character_count")
    if type(size) is not int or size < 0:
        raise ValueError("character_count must be a nonnegative int64")
    metadata = _index_metadata(row.get("metadata_json"), artifact, index)
    result = {name: row[name] for name in METADATA_COLUMNS}
    result["metadata_json"] = metadata
    result.update(
        occurrence_id=digest_json([
            row["source_id"], row["object_id"], row["doc_id"], row["content_hash"],
        ]),
        prepared_path=artifact["path"], prepared_row=index,
        prepared_sha256=artifact["sha256"],
        stage_receipt_sha256=artifact["stage_receipt_sha256"],
    )
    winner_key(result)
    return result


def prepare_inputs(
    parquet_paths: Sequence[Path], *, stage_receipt_path: Path | None,
    stage_receipt_sha256: str | None, receipt_snapshot_dir: Path | None = None,
    stage_proofs: list[dict[str, Any]] | None = None,
    stage_receipt_file_sha256: str | None = None,
    working_budget: Any = None,
) -> list[dict[str, Any]]:
    paths = sorted({Path(path).resolve() for path in parquet_paths}, key=str)
    if len(paths) != len(parquet_paths):
        raise ValueError("Duplicate Parquet paths in one input batch")
    def load(path: Path) -> dict[str, Any]:
        actual = path
        file_sha = stage_receipt_file_sha256
        if file_sha is not None:
            validate_hash(file_sha, "stage_receipt_file_sha256")
        # Only mutable compatibility pointers need cached-generation replay.
        replay_pointer = path.name == "PREP_COMPLETE.json"
        if file_sha is not None and receipt_snapshot_dir is not None and replay_pointer:
            for archived in (
                receipt_snapshot_dir.parent / "receipt-blobs" / file_sha[:2] / file_sha / "receipt.json",
                receipt_snapshot_dir / file_sha[:2] / f"{file_sha}.json",
            ):
                if archived.is_file():
                    actual = archived
                    break
        elif stage_receipt_sha256 is not None and receipt_snapshot_dir is not None and replay_pointer:
            checksum = validate_hash(stage_receipt_sha256, "stage_receipt_sha256")
            pointer = receipt_snapshot_dir.parent / "receipt-locators" / checksum[:2] / f"{checksum}.json"
            if not pointer.exists():
                pointer = receipt_snapshot_dir / "payloads" / checksum[:2] / f"{checksum}.json"
            if pointer.is_file():
                archived = read_receipt(pointer)
                actual = Path(archived["path"]).resolve()
                if archived["payload_sha256"] != checksum or not any(
                    actual.is_relative_to(base)
                    for base in (receipt_snapshot_dir, receipt_snapshot_dir.parent / "receipt-blobs")
                ):
                    raise ValueError("Archived canonical receipt reference differs from its seal")
                if sha256_file(actual) != archived["sha256"]:
                    raise ValueError("Archived receipt snapshot checksum mismatch")
        stage = eligible_stage(
            actual, expected_sha256=stage_receipt_sha256,
            archive_dir=receipt_snapshot_dir, source_path=path, expected_file_sha256=file_sha,
            working_budget=working_budget,
        )
        if stage["receipt"]["schema"] == "metis17.prepared-chunk/v1" and (
            stage_receipt_path is None or stage_receipt_sha256 is None
        ):
            raise ValueError("Production chunks require explicit stage_receipt_path and canonical stage_receipt_sha256")
        if stage_proofs is not None:
            stage_proofs.append({
                "path": str(stage["path"]), "sha256": stage["sha256"],
                "origin_path": str(stage["origin_path"]),
                "file_sha256": stage["file_sha256"],
                "payload_sha256": stage["payload_sha256"], "snapshot": stage["snapshot"],
                "completion_proofs": stage["completion_proofs"],
            })
        return stage

    receipt_paths: dict[Path, Path] = {}
    if stage_receipt_path is not None:
        receipt_path = Path(stage_receipt_path).resolve()
        receipt_paths = {path: receipt_path for path in paths}
        if not paths:
            load(receipt_path)
    else:
        if (stage_receipt_sha256 is not None or stage_receipt_file_sha256 is not None) and not paths:
            raise ValueError("An empty generation needs its explicit stage_receipt_path")
        for path in paths:
            candidates = [path.parent / name for name in ("ELIGIBLE.json", "PREP_COMPLETE.json")]
            if path.parent.parent.name == "eligible":
                candidates.append(path.parent.parent.parent / "PREP_COMPLETE.json")
            found = [candidate for candidate in candidates if candidate.is_file()]
            if path.parent / "ELIGIBLE.json" in found:
                found = [path.parent / "ELIGIBLE.json"]
            if len(found) != 1:
                raise ValueError("Pass an explicit finalized stage_receipt_path for these shards")
            receipt_paths[path] = found[0]
    receipts: dict[Path, dict[str, Any]] = {}
    output = []
    for path in paths:
        receipt_path = receipt_paths[path]
        if receipt_path not in receipts:
            receipts[receipt_path] = load(receipt_path)
        stage = receipts[receipt_path]
        receipt = stage["receipt"]
        if path not in stage["inventory"]:
            raise ValueError(f"Input shard is not attested by the eligible receipt: {path}")
        attested = stage["inventory"][path]
        before = path.stat()
        if before.st_size != attested["byte_count"]:
            raise ValueError(f"Prepared shard byte count mismatch: {path}")
        checksum = sha256_file(path)
        if checksum != validate_hash(attested.get("sha256"), "prepared sha256"):
            raise ValueError(f"Prepared shard checksum mismatch: {path}")
        with contextlib.closing(pq.ParquetFile(path)) as parquet:
            validate_prepared_schema(parquet.schema_arrow)
            rows = parquet.metadata.num_rows
        if rows != attested["records"]:
            raise ValueError(f"Prepared shard row count mismatch: {path}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Prepared shard changed during verification: {path}")
        output.append({
            "path": str(path), "sha256": checksum, "bytes": after.st_size, "rows": rows,
            "input_id": digest_json({"path": str(path), "sha256": checksum}),
            "stage_receipt_path": str(stage["path"]), "stage_receipt_sha256": stage["sha256"],
            "stage_receipt_origin": str(stage["origin_path"]),
            "stage_receipt_file_sha256": stage["file_sha256"],
            "stage_receipt_payload_sha256": stage["payload_sha256"],
            "stage_receipt_snapshot": stage["snapshot"], "completion_proofs": stage["completion_proofs"],
            "source_id": receipt.get("source_id"), "object_id": receipt.get("object_id"),
        })
    return output


def input_contract(
    inputs: Sequence[Mapping[str, Any]], stage_proofs: Sequence[Mapping[str, Any]],
    stage_receipt_file_sha256: str | None,
) -> dict[str, Any]:
    keys = (
        "path", "sha256", "bytes", "rows", "input_id", "stage_receipt_path", "stage_receipt_sha256",
        "source_id", "object_id",
    )
    return {
        "inputs": [{
            **{key: item[key] for key in keys},
            "stage_receipt_path": item.get("stage_receipt_origin", item["stage_receipt_path"]),
        } for item in inputs],
        "stage_receipts": [{
            "path": proof.get("origin_path", proof["path"]), "sha256": proof["sha256"],
        } for proof in stage_proofs],
        "stage_receipt_file_sha256": stage_receipt_file_sha256,
    }


def receipt_file_pin(preferred: str | None, compatibility: str | None) -> str | None:
    if preferred is not None and compatibility is not None and preferred != compatibility:
        raise ValueError("Conflicting receipt_file_sha256 and stage_receipt_file_sha256 pins")
    return preferred if preferred is not None else compatibility


def prepared_rows(
    artifact: Mapping[str, Any], batch_size: int, *, with_text: bool = False,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    columns = PREP_SCHEMA.names if with_text else METADATA_COLUMNS
    row_number = 0
    with contextlib.closing(pq.ParquetFile(artifact["path"])) as parquet:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                text = row.get("text") if with_text else None
                for name in ("source_id", "object_id"):
                    if artifact.get(name) is not None and row.get(name) != artifact[name]:
                        raise ValueError(f"Prepared {name} differs from its finalized generation")
                if with_text and (
                    not isinstance(text, str)
                    or hashlib.sha256(text.encode("utf-8")).hexdigest() != row["content_hash"]
                    or len(text) != row["character_count"]
                ):
                    raise ValueError("Prepared text does not match its content hash/character count")
                yield reference(row, artifact, row_number), text
                row_number += 1
    if row_number != artifact["rows"]:
        raise ValueError("Prepared Parquet row coverage changed during ingestion")


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int, *, quota: Any = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.schema, self.batch_size = path, schema, batch_size
        self.buffer: list[dict[str, Any]] = []
        self.stream = quota.open(path, "wb") if quota is not None else None
        try:
            self.writer = pq.ParquetWriter(self.stream if self.stream is not None else path,
                                           schema, compression="zstd")
        except BaseException:
            if self.stream is not None:
                self.stream.close()
            raise
        self.closed = False
        self.failed = False
        self.count = 0

    def append(self, row: Mapping[str, Any]) -> None:
        if self.closed or self.failed:
            raise RuntimeError("Cannot append to a closed or failed Parquet sink")
        self.buffer.append(dict(row))
        self.count += 1
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            rows, self.buffer = self.buffer, []
            try:
                self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
            except BaseException:
                # Arrow cannot safely resume a row group after its output stream failed.
                self.failed = True
                raise

    def close(self) -> None:
        if self.closed:
            return
        try:
            try:
                if not self.failed:
                    self.flush()
            finally:
                self.writer.close()
        finally:
            self.closed = True
            if self.stream is not None:
                self.stream.close()
        with self.path.open("rb") as stream:
            os.fsync(stream.fileno())

    def artifact(self, root: Path) -> dict[str, Any]:
        return {
            "path": str(self.path.relative_to(root)), "rows": self.count,
            "bytes": self.path.stat().st_size, "sha256": sha256_file(self.path),
        }


def read_artifact(
    root: Path, artifact: Mapping[str, Any], *, winners: bool = False, batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    path = under_root(root, artifact["path"])
    if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Dedup metadata artifact checksum mismatch: {path}")
    expected_schema = WINNER_SCHEMA if winners else REFERENCE_SCHEMA
    with contextlib.closing(pq.ParquetFile(path)) as parquet:
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError("Unexpected dedup metadata schema (text must never enter a metadata run)")
        seen, previous = 0, None
        for batch in parquet.iter_batches(batch_size=batch_size, columns=expected_schema.names):
            for row in batch.to_pylist():
                key = (row["dedup_hash"],) if winners else sort_key(row)
                if previous is not None and key < previous:
                    raise ValueError("Dedup metadata run is not sorted")
                previous = key
                seen += 1
                yield row
        if seen != artifact["rows"]:
            raise ValueError("Dedup metadata row coverage does not match its receipt")


def merge_rows(
    root: Path, runs: Sequence[Mapping[str, Any]], *, batch_size: int = 4096,
    winners: bool = False,
) -> Iterator[dict[str, Any]]:
    with contextlib.ExitStack() as readers:
        streams = [
            readers.enter_context(contextlib.closing(read_artifact(
                root, run["winners" if winners else "occurrences"],
                winners=winners, batch_size=batch_size,
            )))
            for run in runs
        ]
        yield from heapq.merge(*streams, key=(lambda row: row["dedup_hash"]) if winners else sort_key)


def canonical_rows(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for _, group in groupby(rows, occurrence_key):
        yield next(group)


def write_run(
    root: Path, prefix: Path, rows: Iterable[dict[str, Any]], *,
    bucket: int, weight: int, batch_size: int, preserve_deliveries: bool,
    quota: Any = None, storage_kind: str = "exact-batch",
) -> dict[str, Any]:
    occurrences = ParquetSink(prefix.with_suffix(".occurrences.parquet"), REFERENCE_SCHEMA, batch_size, quota=quota)
    try:
        winners = ParquetSink(prefix.with_suffix(".winners.parquet"), WINNER_SCHEMA, batch_size, quota=quota)
    except BaseException:
        occurrences.close()
        raise
    current_hash, last_occurrence, best = None, None, None
    count, canonical_count, raw_characters, winner_characters = 0, 0, 0, 0
    previous = None
    try:
        for row in rows:
            key = sort_key(row)
            if previous is not None and key < previous:
                raise ValueError("Attempted to write an unsorted dedup run")
            previous = key
            if preserve_deliveries or occurrence_key(row) != last_occurrence:
                occurrences.append(row)
            if occurrence_key(row) == last_occurrence:
                continue
            if row["dedup_hash"] != current_hash:
                if best is not None:
                    winners.append({**best, "occurrence_count": count})
                    winner_characters += best["character_count"]
                current_hash, best, count = row["dedup_hash"], row, 0
            elif winner_key(row) < winner_key(best):
                best = row
            count += 1
            canonical_count += 1
            raw_characters += row["character_count"]
            last_occurrence = occurrence_key(row)
        if best is not None:
            winners.append({**best, "occurrence_count": count})
            winner_characters += best["character_count"]
    finally:
        try:
            close_rows = getattr(rows, "close", None)
            if close_rows is not None:
                close_rows()
        finally:
            try:
                occurrences.close()
            finally:
                winners.close()
    result = {
        "bucket": bucket, "weight": weight, "level": max(0, weight.bit_length() - 1),
        "occurrences": occurrences.artifact(root), "winners": winners.artifact(root),
        "canonical_occurrences": canonical_count, "unique_winners": winners.count,
        "raw_characters": raw_characters, "winner_characters": winner_characters,
    }
    storage = storage_descriptor(root, quota, storage_kind)
    if storage is not None:
        result["storage"] = storage
    result["run_id"] = digest_json(result)
    return result
