"""Restartable per-object preparation; normalized data is not training-ready.

All chunk paths in receipts are relative to ``config["root"]``. Consumers must
use the eligible ``chunks`` inventory, not glob the normalization directory.
``apply_eligibility`` can re-filter a sealed normalization receipt without ever
opening or decoding the raw object. A fork-based coordinator should first call
``prepare_runtime`` so workers inherit the already-verified policy mappings.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import re
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from metis_data import datatrove_blocks, decontaminate, freshweb, normalization_evidence, quality

from . import optout17, prep_policy, prep_readers
from .acquisition import CapacityPending
from .common import (
    ObjectSpec, RawReceipt, canonical_json, digest_json, read_receipt,
    sha256_file, under_root, utc_now, write_receipt,
)
from .prep_policy import decide_eligibility, load_eligibility_policy
from .prep_readers import PreparationError, extract_documents, iter_source_rows, normalize_document
from .storage import WorkingBudget, WorkingQuota


NORMALIZATION_VERSION = "metis17.normalized-object/v1"
PREPARATION_VERSION = "metis17.prepared-object/v1"
BASE_CHUNK_VERSION = "metis17.base-chunk/v1"
FILTERED_CHUNK_VERSION = "metis17.filtered-chunk/v1"
CANONICAL_COLUMNS = (
    "doc_id", "content_hash", "dedup_hash", "source_id", "object_id", "text",
    "metadata_json", "priority", "quality_score", "language", "category", "character_count",
)
_IMPORTED_CODE = {
    module.__name__: sha256_file(Path(module.__file__))
    for module in (prep_readers, prep_policy, optout17, quality, normalization_evidence,
                   datatrove_blocks, decontaminate, freshweb)
}


@contextmanager
def _object_lock(path: Path) -> Iterable[None]:
    import fcntl
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def canonical_schema() -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field(name, pa.int32() if name == "priority" else
                 pa.float64() if name == "quality_score" else
                 pa.int64() if name == "character_count" else pa.string(), nullable=False)
        for name in CANONICAL_COLUMNS
    ])


def _settings(spec: ObjectSpec, output_dir: Path, config: Mapping[str, Any]) -> tuple[Path, Path, int, int, int]:
    root = Path(config["root"]).expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output == root or not output.is_relative_to(root):
        raise PreparationError(spec, "preparation_output_must_be_under_release_root")
    chunk_bytes = config.get("output_chunk_bytes", 128_000_000)
    batch_size = config.get("batch_size", 256)
    maximum = config.get("maximum_record_bytes", 128_000_000)
    if any(type(value) is not int or value < 1 for value in (chunk_bytes, batch_size, maximum)):
        raise PreparationError(spec, "positive_preparation_bounds_required")
    if type(spec.priority) is not int or not -(2**31) <= spec.priority < 2**31:
        raise PreparationError(spec, "priority_must_fit_int32")
    if spec.adapter not in prep_readers.ADAPTERS:
        raise PreparationError(spec, "unsupported_adapter")
    if spec.wire_format not in prep_readers.FORMATS:
        raise PreparationError(spec, "unsupported_wire_format")
    defaults = spec.policy.get("metadata", {})
    if not isinstance(defaults, Mapping):
        raise PreparationError(spec, "source_metadata_defaults_must_be_a_mapping")
    if set(defaults) & prep_readers._PER_RECORD_DEFAULTS or any(str(key).endswith("_passed") for key in defaults):
        raise PreparationError(spec, "per_record_evidence_cannot_be_a_source_default")
    output.mkdir(parents=True, exist_ok=True)
    return root, output, chunk_bytes, batch_size, maximum


def _verify_raw(spec: ObjectSpec, raw: RawReceipt, root: Path) -> Path:
    if not isinstance(raw, RawReceipt) or raw.object_id != spec.object_id or raw.source_id != spec.source_id:
        raise PreparationError(spec, "raw_receipt_identity_mismatch")
    if type(raw.byte_count) is not int or raw.byte_count < 0:
        raise PreparationError(spec, "invalid_raw_receipt_byte_count")
    try:
        path = under_root(root, raw.relative_path)
    except ValueError:
        raise PreparationError(spec, "raw_path_escapes_release_root") from None
    if not path.is_file() or path.stat().st_size != raw.byte_count:
        raise PreparationError(spec, "raw_receipt_size_mismatch")
    if spec.expected_bytes is not None and raw.byte_count != spec.expected_bytes:
        raise PreparationError(spec, "raw_catalogue_size_mismatch")
    if spec.expected_sha256 is not None and raw.sha256 != spec.expected_sha256:
        raise PreparationError(spec, "raw_catalogue_digest_mismatch")
    if sha256_file(path) != raw.sha256:
        raise PreparationError(spec, "raw_receipt_digest_mismatch")
    return path


class _ChunkWriter:
    def __init__(self, directory: Path, destination: Path, root: Path,
                 chunk_bytes: int, batch_size: int,
                 on_chunk: Callable[[Path, dict[str, Any]], None] | None = None,
                 quota: WorkingQuota | None = None) -> None:
        self.directory, self.destination, self.root = directory, destination, root
        self.chunk_bytes, self.batch_size = chunk_bytes, batch_size
        self.buffer: list[dict[str, Any]] = []
        self.chunk_size = 0
        self.records = 0
        self.writer: Any = None
        self.quota, self.stream = quota, None
        self.failed = False
        self.failure: CapacityPending | OSError | None = None
        self.path: Path | None = None
        self.artifacts: list[dict[str, Any]] = []
        self.total_records = 0
        self.schema = canonical_schema()
        self.on_chunk = on_chunk
        self.first_identity: tuple[str, str] | None = None
        self.last_identity: tuple[str, str] | None = None
        self.source_rows = 0
        self.last_source_row: int | None = None

    def _check_failed(self) -> None:
        if self.failure is not None:
            raise self.failure
        if self.failed:
            raise RuntimeError("Cannot reuse a failed or aborted Parquet writer")

    def add(self, record: Mapping[str, Any], *, source_row: int | None = None) -> None:
        self._check_failed()
        size = sum(len(value.encode("utf-8")) for value in record.values() if isinstance(value, str)) + 32
        if self.records and self.chunk_size + size > self.chunk_bytes:
            self._close()
        if source_row is None:
            source_row = json.loads(record["metadata_json"])["row_index"]
        if source_row != self.last_source_row:
            self.source_rows += 1
            self.last_source_row = source_row
        identity = (record["doc_id"], record["metadata_json"])
        if self.first_identity is None:
            self.first_identity = identity
        self.last_identity = identity
        self.buffer.append(dict(record))
        self.records += 1
        self.total_records += 1
        self.chunk_size += size
        if len(self.buffer) >= self.batch_size or self.chunk_size >= self.chunk_bytes:
            self._flush()
        if self.chunk_size >= self.chunk_bytes:
            self._close()

    def _flush(self) -> None:
        self._check_failed()
        if not self.buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Arrow's partially advanced row-group encoder cannot safely be retried
        # after output failure, even when the Python buffer is still available.
        rows, self.buffer = self.buffer, []
        self.failed = True
        try:
            if self.writer is None:
                self.path = self.directory / f"part-{len(self.artifacts):06d}.parquet"
                self.stream = self.quota.open(self.path) if self.quota is not None else None
                self.writer = pq.ParquetWriter(self.stream if self.stream is not None else self.path,
                                               self.schema, compression="zstd",
                                               use_dictionary=False, write_statistics=False)
            table = pa.Table.from_pylist(rows, schema=self.schema)
            self.writer.write_table(table, row_group_size=self.batch_size)
        except (CapacityPending, OSError) as exc:
            self.failure = exc
            raise
        self.failed = False

    def _close(self) -> None:
        self._flush()
        if self.writer is None:
            return
        self.failed = True
        try:
            self.writer.close()
        except (CapacityPending, OSError) as exc:
            self.failure = exc
            raise
        finally:
            self.writer = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None
        assert self.path is not None
        with self.path.open("rb") as stream:
            os.fsync(stream.fileno())
        assert self.first_identity is not None and self.last_identity is not None
        first = json.loads(self.first_identity[1])
        last = json.loads(self.last_identity[1])
        artifact = {
            "path": str((self.destination / self.path.name).relative_to(self.root)),
            "byte_count": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
            "records": self.records,
            "source_rows": self.source_rows,
            "uncompressed_bytes": self.chunk_size,
            "document_start": self.total_records - self.records,
            "document_stop": self.total_records,
            "row_start": first["row_index"],
            "row_end": last["row_index"],
            "first_component": first["component"],
            "last_component": last["component"],
            "first_doc_id": self.first_identity[0],
            "last_doc_id": self.last_identity[0],
        }
        if self.on_chunk is not None:
            self.on_chunk(self.path, artifact)
        self.artifacts.append(artifact)
        self.records = self.chunk_size = 0
        self.first_identity = self.last_identity = None
        self.source_rows = 0
        self.last_source_row = None
        self.failed = False

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return self.artifacts

    def abort(self) -> None:
        self.failed = True
        self.buffer.clear()
        try:
            if self.writer is not None:
                self.writer.close()
        except CapacityPending:
            # This is cleanup of an already failed generation, not a commit.
            pass
        finally:
            self.writer = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None


class _DecisionWriter:
    def __init__(self, directory: Path, destination: Path, root: Path,
                 quota: WorkingQuota | None = None) -> None:
        self.path = directory / "decisions.jsonl"
        self.destination, self.root = destination, root
        self.stream = (
            io.TextIOWrapper(quota.open(self.path), encoding="utf-8")
            if quota is not None else self.path.open("w", encoding="utf-8")
        )
        self.count = 0

    def add(self, *, row: Any, component: str, reason: str, quarantine: bool,
            doc_id: str | None = None) -> None:
        self.stream.write(canonical_json({
            "row": row, "component": component, "doc_id": doc_id,
            "reason": reason, "disposition": "quarantine" if quarantine else "reject",
        }) + "\n")
        self.count += 1

    def finish(self) -> dict[str, Any]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return {
            "path": str((self.destination / self.path.name).relative_to(self.root)),
            "records": self.count, "byte_count": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
        }

    def abort(self) -> None:
        try:
            self.stream.close()
        except CapacityPending:
            pass


class _ReadyChunkPublisher:
    def __init__(self, spec: ObjectSpec, raw: RawReceipt, root: Path,
                 destination: Path, inputs: dict[str, Any],
                 quota: WorkingQuota | None = None) -> None:
        self.spec, self.raw, self.root, self.destination = spec, raw, root, destination
        self.inputs = inputs
        self.quota = quota
        self.fingerprint = digest_json(inputs)
        self.receipts: list[str] = []
        if not destination.resolve().is_relative_to(root):
            raise PreparationError(spec, "reblock_destination_escapes_release_root")
        destination.mkdir(parents=True, exist_ok=True)
        self.progress("SCANNING")

    def progress(self, status: str, *, reason: str | None = None) -> None:
        write_receipt(self.destination / "REBLOCK_STATUS.json", {
            "schema": "metis17.reblock-status/v1", "status": status,
            "source_id": self.spec.source_id, "object_id": self.spec.object_id,
            "normalization_fingerprint": self.fingerprint,
            "committed_chunks": len(self.receipts), "reason": reason, "updated_at": utc_now(),
        })

    def __call__(self, path: Path, artifact: dict[str, Any]) -> None:
        index = len(self.receipts)
        ready_path = self.destination / path.with_suffix(".READY.json").name
        artifact["chunk_id"] = digest_json({
            "normalization_fingerprint": self.fingerprint,
            "chunk_index": index, "sha256": artifact["sha256"],
        })
        artifact["ready_receipt"] = str(ready_path.relative_to(self.root))
        value = {
            "schema": BASE_CHUNK_VERSION, "status": "NORMALIZED_CHUNK_READY",
            "eligible": False, "training_ready": False,
            "source_id": self.spec.source_id, "object_id": self.spec.object_id,
            "spec": self.spec.to_dict(), "raw": self.raw.to_dict(),
            "raw_verification": "sha256_verified_before_decode",
            "normalization_fingerprint": self.fingerprint,
            "normalization_inputs": self.inputs,
            "chunk_id": artifact["chunk_id"], "chunk_index": index, "chunk": dict(artifact),
            "record_range": [artifact["document_start"], artifact["document_stop"]],
            "row_range": [artifact["row_start"], artifact["row_end"]],
            "object_manifest": str((self.destination / "NORMALIZED.json").relative_to(self.root)),
            "requires_object_completion": True,
        }
        final_path = self.destination / path.name
        if ready_path.exists():
            previous = read_receipt(ready_path)
            raw_fields = ("object_id", "source_id", "sha256", "byte_count")
            if all(previous.get("raw", {}).get(key) == value["raw"][key] for key in raw_fields):
                value["raw"] = previous["raw"]
            if previous != value:
                raise PreparationError(self.spec, "published_chunk_replay_mismatch", artifact["row_start"])
            _validated_artifact(self.root, artifact, self.spec)
            if self.quota is not None:
                self.quota.unlink(path)
            else:
                path.unlink()
        else:
            if final_path.exists():
                _validated_artifact(self.root, artifact, self.spec)
                if self.quota is not None:
                    self.quota.unlink(path)
                else:
                    path.unlink()
            else:
                if self.quota is not None:
                    self.quota.replace(path, final_path)
                else:
                    os.replace(path, final_path)
            final_path.chmod(0o444)
            write_receipt(ready_path, value)
            ready_path.chmod(0o444)
        self.receipts.append(artifact["ready_receipt"])
        self.progress("SCANNING")

    def verify_inventory(self, chunks: list[dict[str, Any]]) -> None:
        expected = {Path(path).name for path in self.receipts}
        actual = {path.name for path in self.destination.glob("part-*.READY.json")}
        parquet = {path.name for path in self.destination.glob("part-*.parquet")}
        if actual != expected or parquet != {Path(chunk["path"]).name for chunk in chunks}:
            raise PreparationError(self.spec, "reblock_chunk_inventory_mismatch")
        position = 0
        for chunk in chunks:
            if chunk["document_start"] != position or chunk["document_stop"] - position != chunk["records"]:
                raise PreparationError(self.spec, "reblock_document_range_mismatch")
            position = chunk["document_stop"]


def _normalization_policy(spec: ObjectSpec) -> dict[str, Any]:
    return {key: spec.policy.get(key) for key in
            ("category", "language", "generated", "metadata", "quality_score_field")}


def _check_normalization_contract(spec: ObjectSpec, inputs: Mapping[str, Any]) -> None:
    if (
        inputs.get("object_id") != spec.object_id
        or inputs.get("adapter") != spec.adapter
        or inputs.get("wire_format") != spec.wire_format
        or inputs.get("source_policy") != _normalization_policy(spec)
        or inputs.get("code") != _stage_code("normalization")
    ):
        raise PreparationError(spec, "normalization_contract_changed")


def _compute_stage_code(stage: str) -> dict[str, Any]:
    functions = [_ChunkWriter, _DecisionWriter, canonical_schema, _settings,
                 _validated_artifact, _validate_stage, _generation,
                 _normalization_policy, _check_normalization_contract]
    modules = [prep_readers, normalization_evidence, freshweb]
    if stage == "normalization":
        functions.extend([_normalize, _verify_raw, _ReadyChunkPublisher])
    else:
        functions.extend([_canonical_rows, _filter_output])
        if stage == "chunk_eligibility":
            functions.extend([_load_base_chunk, _prepare_chunk, _chunk_admission,
                              _object_completion, _cached_object_manifest])
        else:
            functions.append(_apply)
        modules = [prep_policy, optout17, quality, normalization_evidence, decontaminate, datatrove_blocks, freshweb]
    return {
        "functions_sha256": hashlib.sha256(
            "\n".join(inspect.getsource(function) for function in functions).encode()
        ).hexdigest(),
        "modules": {module.__name__: _IMPORTED_CODE[module.__name__] for module in modules},
    }


def _stage_code(stage: str) -> dict[str, Any]:
    return _FROZEN_STAGE_CODE[stage]


def _validated_artifact(root: Path, artifact: Mapping[str, Any], spec: ObjectSpec) -> Path:
    try:
        path = under_root(root, str(artifact["path"]))
    except (ValueError, KeyError):
        raise PreparationError(spec, "invalid_preparation_artifact_path") from None
    if not path.is_file() or path.stat().st_size != artifact.get("byte_count"):
        raise PreparationError(spec, "preparation_artifact_size_mismatch")
    if sha256_file(path) != artifact.get("sha256"):
        raise PreparationError(spec, "preparation_artifact_digest_mismatch")
    return path


def _validate_stage(root: Path, receipt: Mapping[str, Any], spec: ObjectSpec) -> None:
    if receipt.get("source_id") != spec.source_id or receipt.get("object_id") != spec.object_id:
        raise PreparationError(spec, "preparation_receipt_identity_mismatch")
    for artifact in receipt.get("chunks", []):
        _validated_artifact(root, artifact, spec)
    if receipt.get("decisions"):
        _validated_artifact(root, receipt["decisions"], spec)


def _generation(output: Path, stage: str, fingerprint: str,
                quota: WorkingQuota | None = None) -> tuple[Path, Path]:
    destination = output / stage / fingerprint
    destination.parent.mkdir(parents=True, exist_ok=True)
    removed = False
    for abandoned in destination.parent.glob(f".{fingerprint}.*.partial"):
        if re.fullmatch(rf"\.{fingerprint}\.[0-9a-f]{{32}}\.partial", abandoned.name):
            if abandoned.is_symlink():
                raise ValueError("Incomplete preparation directory must not be a symlink")
            shutil.rmtree(abandoned)
            removed = True
    if removed and quota is not None:
        quota.reconcile()
    staging = destination.with_name(f".{fingerprint}.{uuid.uuid4().hex}.partial")
    staging.mkdir()
    return destination, staging


def _normalize(
    spec: ObjectSpec, raw: RawReceipt, output: Path, config: Mapping[str, Any],
    *, publish_chunks: bool = False, quota: WorkingQuota | None = None,
) -> dict[str, Any]:
    root, output, chunk_bytes, batch_size, maximum = _settings(spec, output, config)
    path = _verify_raw(spec, raw, root)
    inputs = {
        "object_id": spec.object_id, "raw_sha256": raw.sha256, "raw_bytes": raw.byte_count,
        "wire_format": spec.wire_format, "adapter": spec.adapter,
        "source_policy": _normalization_policy(spec),
        "priority": spec.priority,
        "code": _stage_code("normalization"),
        "layout": {"output_chunk_bytes": chunk_bytes, "batch_size": batch_size,
                   "maximum_record_bytes": maximum, "publish_chunks": publish_chunks},
    }
    fingerprint = digest_json(inputs)
    destination = output / "normalized" / fingerprint
    receipt_path = destination / "NORMALIZED.json"
    if receipt_path.is_file():
        receipt = read_receipt(receipt_path)
        if receipt.get("normalization_fingerprint") != fingerprint:
            raise PreparationError(spec, "normalization_fingerprint_mismatch")
        _validate_stage(root, receipt, spec)
        write_receipt(output / "NORMALIZED.json", receipt)
        if publish_chunks:
            for chunk in receipt["chunks"]:
                _load_base_chunk(root / chunk["path"], config)
            write_receipt(output / "REBLOCK_COMPLETE.json", receipt)
        return receipt
    if destination.exists() and not publish_chunks:
        raise PreparationError(spec, "unsealed_normalization_generation")
    destination, staging = _generation(output, "normalized", fingerprint, quota)
    publisher = _ReadyChunkPublisher(spec, raw, root, destination, inputs, quota) if publish_chunks else None
    writer = _ChunkWriter(staging, destination, root, chunk_bytes, batch_size, on_chunk=publisher, quota=quota)
    decisions = _DecisionWriter(staging, destination, root, quota)
    rejected: Counter[str] = Counter()
    input_rows = input_documents = 0
    try:
        for item in iter_source_rows(spec, path, batch_size=batch_size, maximum_record_bytes=maximum):
            input_rows += 1
            for document in extract_documents(item, spec):
                input_documents += 1
                reason = document.rejection
                if reason is None and (document.text is None or not document.text.strip()):
                    reason = "empty_text"
                if reason:
                    rejected[reason] += 1
                    decisions.add(row=item.index, component=document.component, reason=reason, quarantine=False)
                    continue
                record = normalize_document(document, spec, item.index)
                if len(record["text"].encode("utf-8")) > maximum or len(record["metadata_json"].encode("utf-8")) > maximum:
                    raise PreparationError(spec, "canonical_record_exceeds_bound", item.index)
                writer.add(record, source_row=item.index)
        chunks = writer.finish()
        if input_documents != writer.total_records + sum(rejected.values()):
            raise PreparationError(spec, "normalization_coverage_mismatch")
        if publisher is not None:
            publisher.verify_inventory(chunks)
        receipt = {
            "schema": NORMALIZATION_VERSION, "status": "NORMALIZED",
            "eligible": False, "training_ready": False,
            "source_id": spec.source_id, "object_id": spec.object_id,
            "normalization_fingerprint": fingerprint, "inputs": inputs,
            "raw": raw.to_dict(), "created_at": utc_now(),
            "receipt_path": str(receipt_path.relative_to(root)),
            "chunks": chunks, "decisions": decisions.finish(),
            "input_rows": input_rows, "input_documents": input_documents,
            "normalized_documents": writer.total_records,
            "rejected": dict(rejected), "quarantined": {},
            "empty": writer.total_records == 0,
            "empty_reason": "empty_input" if not input_rows else
                            "all_documents_rejected_during_extraction" if not writer.total_records else None,
        }
        if publisher is not None:
            receipt["chunk_receipts"] = publisher.receipts
            receipt["reblock_complete"] = True
            final_decisions = destination / "decisions.jsonl"
            if final_decisions.exists():
                _validated_artifact(root, receipt["decisions"], spec)
                if quota is not None:
                    quota.unlink(staging / "decisions.jsonl")
                else:
                    (staging / "decisions.jsonl").unlink()
            else:
                if quota is not None:
                    quota.replace(staging / "decisions.jsonl", final_decisions)
                else:
                    os.replace(staging / "decisions.jsonl", final_decisions)
                final_decisions.chmod(0o444)
            write_receipt(receipt_path, receipt)
            publisher.progress("COMPLETE")
            write_receipt(output / "REBLOCK_COMPLETE.json", receipt)
        else:
            write_receipt(staging / "NORMALIZED.json", receipt)
            os.replace(staging, destination)
        write_receipt(output / "NORMALIZED.json", receipt)
        return receipt
    except BaseException as exc:
        if publisher is not None:
            publisher.progress("FAILED", reason=exc.reason if isinstance(exc, PreparationError) else type(exc).__name__)
        raise
    finally:
        writer.abort()
        decisions.abort()
        if staging.exists():
            shutil.rmtree(staging)


def _canonical_rows(root: Path, normalized: Mapping[str, Any], spec: ObjectSpec,
                    batch_size: int) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq
    count = 0
    for chunk in normalized["chunks"]:
        path = under_root(root, chunk["path"])
        chunk_count = 0
        first_identity = last_identity = None
        with pq.ParquetFile(path) as parquet:
            if not parquet.schema_arrow.equals(canonical_schema(), check_metadata=False):
                raise PreparationError(spec, "canonical_parquet_schema_mismatch")
            for batch in parquet.iter_batches(batch_size=batch_size):
                for record in batch.to_pylist():
                    if record["source_id"] != spec.source_id or record["object_id"] != spec.object_id:
                        raise PreparationError(spec, "canonical_record_identity_mismatch")
                    identity = (record["doc_id"], record["metadata_json"])
                    if first_identity is None:
                        first_identity = identity
                    last_identity = identity
                    chunk_count += 1
                    yield record
        if chunk_count != chunk["records"]:
            raise PreparationError(spec, "canonical_chunk_count_mismatch")
        if first_identity is not None and "row_start" in chunk:
            assert last_identity is not None
            first = json.loads(first_identity[1])
            last = json.loads(last_identity[1])
            if (
                (chunk["row_start"], chunk["first_component"], chunk["first_doc_id"])
                != (first["row_index"], first["component"], first_identity[0])
                or (chunk["row_end"], chunk["last_component"], chunk["last_doc_id"])
                != (last["row_index"], last["component"], last_identity[0])
            ):
                raise PreparationError(spec, "canonical_chunk_row_range_mismatch")
        count += chunk_count
    if count != normalized["normalized_documents"]:
        raise PreparationError(spec, "canonical_object_count_mismatch")


def _filter_output(
    spec: ObjectSpec, normalized: Mapping[str, Any], root: Path,
    destination: Path, staging: Path, chunk_bytes: int, batch_size: int, policy: Any,
    quota: WorkingQuota | None = None,
) -> dict[str, Any]:
    writer = _ChunkWriter(staging, destination, root, chunk_bytes, batch_size, quota=quota)
    decisions = _DecisionWriter(staging, destination, root, quota)
    rejected = Counter(normalized["rejected"])
    quarantined = Counter(normalized["quarantined"])
    counters: Counter[str] = Counter()
    seen = 0
    try:
        for record in _canonical_rows(root, normalized, spec, batch_size):
            seen += 1
            accepted, reason, quarantine, coverage = decide_eligibility(record, spec, policy)
            counters.update(coverage)
            if accepted is not None:
                writer.add(accepted)
            else:
                assert reason is not None
                (quarantined if quarantine else rejected)[reason] += 1
                metadata = json.loads(record["metadata_json"])
                decisions.add(row=metadata.get("row_index"), component=metadata["component"],
                              doc_id=record["doc_id"], reason=reason, quarantine=quarantine)
        chunks = writer.finish()
        if (
            seen != normalized["normalized_documents"]
            or normalized["input_documents"] !=
            writer.total_records + sum(rejected.values()) + sum(quarantined.values())
        ):
            raise PreparationError(spec, "eligibility_coverage_mismatch")
        return {
            "chunks": chunks, "decisions": decisions.finish(),
            "eligible_documents": writer.total_records, "accepted_documents": writer.total_records,
            "rejected": dict(rejected), "quarantined": dict(quarantined),
            "policy_coverage": dict(counters),
            "acceptance_fraction": writer.total_records / max(1, normalized["input_documents"]),
            "empty": writer.total_records == 0,
            "empty_reason": "empty_input" if not normalized["input_rows"] else
                            "all_documents_rejected_or_quarantined" if not writer.total_records else None,
        }
    finally:
        writer.abort()
        decisions.abort()


def _apply(
    spec: ObjectSpec, normalized: Mapping[str, Any], output: Path, config: Mapping[str, Any],
    quota: WorkingQuota | None = None,
) -> dict[str, Any]:
    root, output, chunk_bytes, batch_size, _ = _settings(spec, output, config)
    if normalized.get("schema") != NORMALIZATION_VERSION or normalized.get("status") != "NORMALIZED":
        raise PreparationError(spec, "sealed_normalization_receipt_required")
    path = under_root(root, str(normalized.get("receipt_path", "")))
    if read_receipt(path) != normalized:
        raise PreparationError(spec, "normalization_receipt_changed")
    _check_normalization_contract(spec, normalized["inputs"])
    _validate_stage(root, normalized, spec)
    policy = load_eligibility_policy(spec, config)
    inputs = {
        "normalization_fingerprint": normalized["normalization_fingerprint"],
        "normalization_receipt_sha256": digest_json(normalized),
        "policy": policy.descriptor,
        "code": _stage_code("eligibility"),
        "layout": {"output_chunk_bytes": chunk_bytes, "batch_size": batch_size},
    }
    fingerprint = digest_json(inputs)
    common = {
        "schema": PREPARATION_VERSION, "source_id": spec.source_id, "object_id": spec.object_id,
        "preparation_fingerprint": fingerprint, "eligibility_fingerprint": fingerprint,
        "normalization_fingerprint": normalized["normalization_fingerprint"],
        "normalization_receipt": normalized["receipt_path"],
        "normalized_chunks": normalized["chunks"], "inputs": inputs,
        "input_rows": normalized["input_rows"], "input_documents": normalized["input_documents"],
        "normalized_documents": normalized["normalized_documents"],
        "raw_retained": True,
    }
    if policy.pending:
        receipt = {
            **common, "status": "NORMALIZED_PENDING_DECONTAMINATION",
            "eligible": False, "training_ready": False, "pending_reasons": list(policy.pending),
            "chunks": [], "eligible_documents": 0, "accepted_documents": 0,
            "rejected": normalized["rejected"], "quarantined": normalized["quarantined"],
            "empty": normalized["empty"], "empty_reason": normalized["empty_reason"],
        }
        write_receipt(output / "PREP_COMPLETE.json", receipt)
        return receipt
    destination = output / "eligible" / fingerprint
    receipt_path = destination / "ELIGIBLE.json"
    if receipt_path.is_file():
        receipt = read_receipt(receipt_path)
        if receipt.get("eligibility_fingerprint") != fingerprint:
            raise PreparationError(spec, "eligibility_fingerprint_mismatch")
        _validate_stage(root, receipt, spec)
        write_receipt(output / "PREP_COMPLETE.json", receipt)
        return receipt
    if destination.exists():
        raise PreparationError(spec, "unsealed_eligibility_generation")
    destination, staging = _generation(output, "eligible", fingerprint, quota)
    try:
        outcome = _filter_output(spec, normalized, root, destination, staging, chunk_bytes, batch_size, policy, quota)
        receipt = {
            **common, **outcome, "status": "ELIGIBLE", "eligible": True, "training_ready": True,
            "pending_reasons": [],
            "created_at": utc_now(), "receipt_path": str(receipt_path.relative_to(root)),
        }
        write_receipt(staging / "ELIGIBLE.json", receipt)
        os.replace(staging, destination)
        write_receipt(output / "PREP_COMPLETE.json", receipt)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_base_chunk(
    base_chunk_path: Path, config: Mapping[str, Any],
) -> tuple[ObjectSpec, dict[str, Any], Path]:
    root = Path(config["root"]).expanduser().resolve()
    supplied = under_root(root, str(base_chunk_path))
    ready_path = under_root(root, str(
        supplied if supplied.name.endswith(".READY.json") else supplied.with_suffix(".READY.json")
    ))
    ready = read_receipt(ready_path)
    if ready.get("schema") != BASE_CHUNK_VERSION or ready.get("status") != "NORMALIZED_CHUNK_READY":
        raise ValueError("A sealed NORMALIZED_CHUNK_READY receipt is required")
    spec = ObjectSpec.from_dict(ready["spec"])
    override = config.get("object_spec")
    if override is not None:
        current = ObjectSpec.from_dict(override.to_dict() if isinstance(override, ObjectSpec) else override)
        if current.object_id != spec.object_id or current.source_id != spec.source_id:
            raise PreparationError(spec, "base_chunk_source_override_mismatch")
        spec = current
    artifact = ready["chunk"]
    inputs = ready["normalization_inputs"]
    raw = RawReceipt.from_dict(ready["raw"])
    index = ready["chunk_index"]
    if (
        ready.get("object_id") != spec.object_id or ready.get("source_id") != spec.source_id
        or raw.object_id != spec.object_id or raw.source_id != spec.source_id
        or raw.sha256 != inputs.get("raw_sha256") or raw.byte_count != inputs.get("raw_bytes")
        or (spec.expected_sha256 is not None and spec.expected_sha256 != raw.sha256)
        or (spec.expected_bytes is not None and spec.expected_bytes != raw.byte_count)
        or ready.get("raw_verification") != "sha256_verified_before_decode"
    ):
        raise PreparationError(spec, "base_chunk_raw_identity_mismatch")
    _check_normalization_contract(spec, inputs)
    if digest_json(inputs) != ready["normalization_fingerprint"] or inputs["layout"].get("publish_chunks") is not True:
        raise PreparationError(spec, "base_chunk_extraction_contract_mismatch")
    if (
        type(index) is not int or index < 0
        or any(type(artifact.get(key)) is not int for key in
               ("records", "source_rows", "document_start", "document_stop", "row_start", "row_end"))
        or artifact["records"] < 1 or artifact["document_start"] < 0
        or not 1 <= artifact["source_rows"] <= artifact["records"]
        or artifact["document_stop"] - artifact["document_start"] != artifact["records"]
        or artifact["row_start"] < 1 or artifact["row_end"] < artifact["row_start"]
        or ready["record_range"] != [artifact["document_start"], artifact["document_stop"]]
        or ready["row_range"] != [artifact["row_start"], artifact["row_end"]]
    ):
        raise PreparationError(spec, "base_chunk_range_contract_mismatch")
    expected_id = digest_json({
        "normalization_fingerprint": ready["normalization_fingerprint"],
        "chunk_index": index, "sha256": artifact["sha256"],
    })
    if ready.get("chunk_id") != expected_id or artifact.get("chunk_id") != expected_id:
        raise PreparationError(spec, "base_chunk_identity_mismatch")
    path = under_root(root, artifact["path"])
    if (
        path.name != f"part-{index:06d}.parquet"
        or ready_path != path.with_suffix(".READY.json")
        or (supplied != ready_path and supplied != path)
        or under_root(root, artifact["ready_receipt"]) != ready_path
        or under_root(root, ready["object_manifest"]) != path.parent / "NORMALIZED.json"
    ):
        raise PreparationError(spec, "base_chunk_path_contract_mismatch")
    _validated_artifact(root, artifact, spec)
    return spec, ready, path


@lru_cache(maxsize=16)
def _cached_object_manifest(path: str, stamp: tuple[int, ...]) -> dict[str, Any]:
    return read_receipt(Path(path))


def _object_completion(
    spec: ObjectSpec, ready: Mapping[str, Any], root: Path,
) -> dict[str, str] | None:
    path = under_root(root, ready["object_manifest"])
    if not path.is_file():
        return None
    stat = path.stat()
    manifest = _cached_object_manifest(str(path), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    index = ready["chunk_index"]
    chunks = manifest.get("chunks", [])
    if (
        manifest.get("schema") != NORMALIZATION_VERSION or manifest.get("status") != "NORMALIZED"
        or manifest.get("reblock_complete") is not True
        or manifest.get("normalization_fingerprint") != ready["normalization_fingerprint"]
        or manifest.get("object_id") != spec.object_id or manifest.get("source_id") != spec.source_id
        or manifest.get("inputs") != ready["normalization_inputs"]
        or index >= len(chunks) or chunks[index] != ready["chunk"]
        or len(manifest.get("chunk_receipts", [])) != len(chunks)
        or manifest["chunk_receipts"][index] != ready["chunk"]["ready_receipt"]
    ):
        raise PreparationError(spec, "object_completion_contract_mismatch")
    return {"path": ready["object_manifest"], "receipt_sha256": digest_json(manifest)}


def _chunk_admission(
    spec: ObjectSpec, ready: Mapping[str, Any], filtered: Mapping[str, Any], root: Path, output: Path,
) -> dict[str, Any]:
    completion = _object_completion(spec, ready, root)
    complete = completion is not None
    receipt_path = (
        under_root(root, filtered["receipt_path"]).with_name("ELIGIBLE.json")
        if complete else output / "PREP_COMPLETE.json"
    )
    result = {
        **filtered, "schema": "metis17.prepared-chunk/v1",
        "status": "ELIGIBLE" if complete else "ELIGIBLE_PENDING_OBJECT_COMPLETION",
        "eligible": complete, "training_ready": complete,
        "object_complete": complete, "requires_object_completion": True,
        "object_completion": completion,
        "pending_reasons": [] if complete else ["object_reblock_incomplete"],
        "screened_chunks": filtered["chunks"],
        "screened_documents": filtered["accepted_documents"],
        "chunks": filtered["chunks"] if complete else [],
        "eligible_documents": filtered["accepted_documents"] if complete else 0,
        "screening_receipt": filtered["receipt_path"],
        "receipt_path": str(receipt_path.relative_to(root)),
    }
    if complete:
        if receipt_path.exists():
            if read_receipt(receipt_path) != result:
                raise PreparationError(spec, "immutable_eligible_chunk_receipt_changed")
        else:
            write_receipt(receipt_path, result)
            receipt_path.chmod(0o444)
    write_receipt(output / "PREP_COMPLETE.json", result)
    return result


def _prepare_chunk(
    spec: ObjectSpec, ready: Mapping[str, Any], output: Path, config: Mapping[str, Any],
    quota: WorkingQuota | None = None,
) -> dict[str, Any]:
    root, output, chunk_bytes, batch_size, _ = _settings(spec, output, config)
    policy = load_eligibility_policy(spec, config)
    artifact = ready["chunk"]
    inputs = {
        "base_chunk_receipt_sha256": digest_json(ready),
        "normalization_fingerprint": ready["normalization_fingerprint"],
        "policy": policy.descriptor, "code": _stage_code("chunk_eligibility"),
        "layout": {"output_chunk_bytes": chunk_bytes, "batch_size": batch_size},
    }
    fingerprint = digest_json(inputs)
    common = {
        "schema": "metis17.prepared-chunk/v1", "source_id": spec.source_id, "object_id": spec.object_id,
        "chunk_id": ready["chunk_id"], "chunk_index": ready["chunk_index"],
        "base_chunk": artifact["path"], "base_chunk_receipt": artifact["ready_receipt"],
        "object_manifest": ready["object_manifest"], "normalization_fingerprint": ready["normalization_fingerprint"],
        "normalization_receipt": artifact["ready_receipt"],
        "preparation_fingerprint": fingerprint, "eligibility_fingerprint": fingerprint,
        "normalized_chunks": [artifact], "inputs": inputs, "input_scope": "base_chunk",
        "record_range": ready["record_range"], "row_range": ready["row_range"],
        "input_rows": artifact["source_rows"],
        "input_documents": artifact["records"], "normalized_documents": artifact["records"],
        "raw_retained": True, "requires_object_completion": True,
    }
    if policy.pending:
        result = {
            **common, "status": "NORMALIZED_PENDING_DECONTAMINATION",
            "eligible": False, "training_ready": False, "pending_reasons": list(policy.pending),
            "chunks": [], "eligible_documents": 0, "accepted_documents": 0,
            "rejected": {}, "quarantined": {}, "empty": False, "empty_reason": None,
            "receipt_path": str((output / "PREP_COMPLETE.json").relative_to(root)),
        }
        write_receipt(output / "PREP_COMPLETE.json", result)
        return result
    destination = output / "eligible" / fingerprint
    receipt_path = destination / "FILTERED.json"
    if receipt_path.is_file():
        filtered = read_receipt(receipt_path)
        if (
            filtered.get("schema") != FILTERED_CHUNK_VERSION
            or filtered.get("status") != "FILTERED"
            or filtered.get("eligibility_fingerprint") != fingerprint
            or filtered.get("chunk_id") != ready["chunk_id"]
        ):
            raise PreparationError(spec, "filtered_chunk_contract_mismatch")
        _validate_stage(root, filtered, spec)
        return _chunk_admission(spec, ready, filtered, root, output)
    if destination.exists():
        raise PreparationError(spec, "unsealed_chunk_eligibility_generation")
    destination, staging = _generation(output, "eligible", fingerprint, quota)
    normalized = {
        "chunks": [artifact], "normalized_documents": artifact["records"],
        "input_documents": artifact["records"], "input_rows": common["input_rows"],
        "rejected": {}, "quarantined": {},
    }
    try:
        outcome = _filter_output(spec, normalized, root, destination, staging, chunk_bytes, batch_size, policy, quota)
        filtered = {
            **common, **outcome, "schema": FILTERED_CHUNK_VERSION, "status": "FILTERED",
            "eligible": False, "training_ready": False, "filtering_complete": True,
            "created_at": utc_now(), "receipt_path": str(receipt_path.relative_to(root)),
        }
        write_receipt(staging / "FILTERED.json", filtered)
        os.replace(staging, destination)
        return _chunk_admission(spec, ready, filtered, root, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _storage_quota(
    config: Mapping[str, Any], namespace: str, directory: Path,
) -> Any:
    enforce = config.get("enforce_storage_budget", False)
    if type(enforce) is not bool:
        raise ValueError("enforce_storage_budget must be a boolean")
    if enforce:
        budget = WorkingBudget(Path(config["root"]))
        relative = str(directory.expanduser().resolve().relative_to(budget.root))
        return budget.quota(f"{namespace}:{digest_json(relative)}", directory)
    return nullcontext(None)


def normalize_object(
    spec: ObjectSpec, raw: RawReceipt, output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify this RAW_READY object and publish reusable, ineligible base chunks."""
    _, output, *_ = _settings(spec, output_dir, config)
    with _object_lock(output / ".prepare.lock"), \
            _storage_quota(config, f"normalization:{spec.object_id}", output / "normalized") as quota:
        return _normalize(spec, raw, output, config, quota=quota)


def reblock_object(
    spec: ObjectSpec, raw: RawReceipt, output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish immutable ``part-*.READY.json`` receipts during one raw scan.

    Each receipt binds a Parquet chunk, raw SHA, extraction policy, and ranges.
    ``REBLOCK_COMPLETE.json`` seals object coverage only after successful EOF.
    A retry replays the raw stream once, verifies already-published chunks, and
    never overwrites or deletes them.
    """
    _, output, *_ = _settings(spec, output_dir, config)
    with _object_lock(output / ".prepare.lock"), \
            _storage_quota(config, f"normalization:{spec.object_id}", output / "normalized") as quota:
        return _normalize(spec, raw, output, config, publish_chunks=True, quota=quota)


def prepare_chunk(
    base_chunk_path: Path, output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter one ready Parquet chunk without opening raw or sibling chunks.

    ``base_chunk_path`` may name the Parquet file or its READY receipt. Work is
    isolated below ``output_dir/chunks/<chunk_id>``; there is no producer lock.
    ``config["object_spec"]`` optionally supplies current source policy/priority
    with the same extraction contract. Before object completion, results are
    screened but not training-ready; calling again promotes cached screening
    without repeating quality/decontamination once coverage is sealed. Only
    then is immutable ``ELIGIBLE.json`` published beside the eligible shards;
    ``PREP_COMPLETE.json`` remains an alias of that exact receipt.
    """
    spec, ready, _ = _load_base_chunk(base_chunk_path, config)
    _, output, *_ = _settings(spec, output_dir, config)
    namespace = output / "chunks" / ready["chunk_id"]
    namespace.mkdir(parents=True, exist_ok=True)
    with _object_lock(namespace / ".prepare.lock"), \
            _storage_quota(config, f"eligibility:{ready['chunk_id']}", namespace) as quota:
        return _prepare_chunk(spec, ready, namespace, config, quota)


def prepare_runtime(config: Mapping[str, Any], *, require_ready: bool = False) -> None:
    """Preload policy mappings once before forking; absent policies stay pending."""
    prep_policy.prepare_runtime(config, require_ready=require_ready)


def apply_eligibility(
    spec: ObjectSpec, normalized: Mapping[str, Any], output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply current pinned policy to saved Parquet, with no access to raw bytes."""
    _, output, *_ = _settings(spec, output_dir, config)
    with _object_lock(output / ".prepare.lock"), \
            _storage_quota(config, f"eligibility:{spec.object_id}", output / "eligible") as quota:
        return _apply(spec, normalized, output, config, quota)


def prepare_object(
    spec: ObjectSpec, raw: RawReceipt, output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare one verified object, publishing either pending or eligible output."""
    _, output, *_ = _settings(spec, output_dir, config)
    with _object_lock(output / ".prepare.lock"):
        with _storage_quota(config, f"normalization:{spec.object_id}", output / "normalized") as quota:
            normalized = _normalize(spec, raw, output, config, quota=quota)
        with _storage_quota(config, f"eligibility:{spec.object_id}", output / "eligible") as quota:
            return _apply(spec, normalized, output, config, quota)


_FROZEN_STAGE_CODE = {
    stage: _compute_stage_code(stage) for stage in ("normalization", "eligibility", "chunk_eligibility")
}
