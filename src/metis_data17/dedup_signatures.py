"""Scoped candidate signatures; no irreversible near/span/code removals.

MinHash uses case-preserving five-token shingles and bounded NumPy permutation
blocks. Code/math additionally retain whitespace in their shingles and use raw
span hashes: the 1.6 code tokenizer discards significant Python indentation, so
its normalized file/unit digests are not reused. Only prose uses the existing
sentence-span helper. Every comparison key includes language, snapshot and
semantic namespace. Scope closure, candidate resolution and quality-priority
selection must precede any final near/span/code removal view.
"""
from __future__ import annotations

import contextlib
import hashlib
import re
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa

from metis_data.span_dedup import iter_span_signatures

from .common import digest_json, read_receipt, sha256_file, under_root, write_receipt
from .dedup_runs import (
    REFERENCE_SCHEMA, ParquetSink, input_contract, positive_integer, prepare_inputs, prepared_rows,
    receipt_file_pin,
)
from .dedup_storage import bind_working_budget, quota_receipt, quota_rmtree, storage_namespace


WORDS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SEMANTIC_TOKENS = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
NEAR_SCHEMA = pa.schema(list(REFERENCE_SCHEMA) + [
    pa.field("scope_id", pa.string()), pa.field("snapshot", pa.string()),
    pa.field("semantic_namespace", pa.string()), pa.field("minhash", pa.binary()),
    pa.field("shingle_count", pa.int64()), pa.field("token_policy", pa.string()),
])
SPAN_SCHEMA = pa.schema([
    ("scope_id", pa.string()), ("signature", pa.string()), ("kind", pa.string()),
    ("offset", pa.int64()), ("extent", pa.int64()), ("occurrence_id", pa.string()),
    ("doc_id", pa.string()), ("content_hash", pa.string()), ("prepared_path", pa.string()),
    ("prepared_row", pa.int64()), ("prepared_sha256", pa.string()),
    ("priority", pa.int32()), ("quality_score", pa.float64()),
])


def _scope_value(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 1024:
        raise ValueError(f"An explicit bounded {name} is required for comparison scope")
    return value


def scope_id(language: str, snapshot: str, semantic_namespace: str) -> str:
    return digest_json({
        "language": _scope_value(language, "language"),
        "snapshot": _scope_value(snapshot, "snapshot"),
        "semantic_namespace": _scope_value(semantic_namespace, "semantic_namespace"),
    })


def _shingles(text: str, *, semantic: bool, n_grams: int) -> Iterator[int]:
    tokens: deque[str] = deque(maxlen=n_grams)
    tokenizer = SEMANTIC_TOKENS if semantic else WORDS
    for match in tokenizer.finditer(text):
        tokens.append(match[0])
        if len(tokens) == n_grams:
            digest = hashlib.blake2b(digest_size=8)
            for token in tokens:
                data = token.encode("utf-8")
                digest.update(len(data).to_bytes(8, "little"))
                digest.update(data)
            yield int.from_bytes(digest.digest(), "little")


def minhash_signature(
    text: str, *, semantic: bool = False, n_grams: int = 5,
    num_perm: int = 128, seed: int = 16062026, shingle_batch_size: int = 256,
) -> tuple[bytes, int]:
    """Return real MinHash values, using uint64 overflow then Mersenne reduction.

    A short document has zero shingles and an empty signature, not an all-MAX
    signature which could spuriously cluster every short document together.
    """
    for value, name in ((n_grams, "n_grams"), (num_perm, "num_perm"),
                        (shingle_batch_size, "shingle_batch_size")):
        positive_integer(value, name)
    if type(seed) is not int or not 0 <= seed < (1 << 32):
        raise ValueError("seed must be uint32")
    prime = np.uint64((1 << 61) - 1)
    random = np.random.RandomState(seed)
    coefficients = random.randint(1, int(prime), size=num_perm, dtype=np.uint64)
    offsets = random.randint(0, int(prime), size=num_perm, dtype=np.uint64)
    minimum = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    values: list[int] = []
    count = 0

    def update() -> None:
        nonlocal minimum
        hashes = np.asarray(values, dtype=np.uint64)
        permutations = (hashes[:, None] * coefficients + offsets) % prime
        minimum = np.minimum(minimum, permutations.min(axis=0))
        values.clear()

    for value in _shingles(text, semantic=semantic, n_grams=n_grams):
        values.append(value)
        count += 1
        if len(values) == shingle_batch_size:
            update()
    if values:
        update()
    return (minimum.astype("<u8", copy=False).tobytes() if count else b""), count


def minhash_bands(signature: bytes, comparison_scope: str, *, rows_per_band: int = 8) -> Iterator[str]:
    positive_integer(rows_per_band, "rows_per_band")
    width = 8 * rows_per_band
    if len(signature) % width:
        raise ValueError("MinHash signature size must be divisible by band width")
    for offset in range(0, len(signature), width):
        yield hashlib.sha256(
            comparison_scope.encode("utf-8") + offset.to_bytes(8, "little") + signature[offset:offset + width]
        ).hexdigest()


def _span_records(text: str, category: str, scope: str, *, raw_window: int) -> Iterator[dict[str, Any]]:
    if category in {"code", "math"}:
        kind = f"{category}.file.raw"
        yield {
            "kind": kind, "offset": 0, "extent": len(text),
            "signature": digest_json([scope, kind, hashlib.sha256(text.encode("utf-8")).hexdigest()]),
        }
        kind = f"{category}.window.raw"
        for offset in range(0, len(text), max(1, raw_window // 2)):
            span = text[offset:offset + raw_window]
            yield {
                "kind": kind, "offset": offset, "extent": len(span),
                "signature": digest_json([scope, kind, hashlib.sha256(span.encode("utf-8")).hexdigest()]),
            }
    else:
        for signature in iter_span_signatures(text):
            yield {
                "kind": "prose.three-sentences.normalized", "offset": signature.sentence_start,
                "extent": 3, "signature": digest_json([scope, "prose.span", signature.digest.hex()]),
            }


def generate_signatures(
    parquet_paths: Sequence[Path], output_dir: Path, *, batch_id: str,
    snapshot: str, semantic_namespace: str, language: str | None = None,
    batch_size: int = 256, num_perm: int = 128, n_grams: int = 5, raw_window: int = 512,
    stage_receipt_path: Path | None = None, stage_receipt_sha256: str | None = None,
    stage_receipt_file_sha256: str | None = None,
    receipt_file_sha256: str | None = None,
    working_budget: Any = None,
) -> dict[str, Any]:
    """Generate bounded, immutable near and span/code candidate artifacts.

    ``language=None`` explicitly uses each prepared row's language, never a
    global mixed-language namespace. In-flight signatures have no authority to
    remove documents or grant training credits.
    ``stage_receipt_sha256`` pins the canonical receipt seal; the separate
    ``receipt_file_sha256`` optionally pins its original JSON bytes
    (``stage_receipt_file_sha256`` is retained as a compatibility spelling).
    Production chunks require the explicit receipt path and canonical seal.
    ``working_budget`` meters streams before writes and is auto-enabled under
    production RUN/limits. It must be a WorkingBudget, not a whole-index quota.
    """
    from .dedup import _batch_key, _claimed_input, _lock

    for value, name in ((batch_size, "batch_size"), (num_perm, "num_perm"),
                        (n_grams, "n_grams"), (raw_window, "raw_window")):
        positive_integer(value, name)
    stage_receipt_file_sha256 = receipt_file_pin(receipt_file_sha256, stage_receipt_file_sha256)
    _scope_value(snapshot, "snapshot")
    _scope_value(semantic_namespace, "semantic_namespace")
    if language is not None:
        _scope_value(language, "language")
    root = Path(output_dir).resolve()
    working_budget = bind_working_budget(root, working_budget)
    stage_proofs: list[dict[str, Any]] = []
    inputs = prepare_inputs(
        parquet_paths, stage_receipt_path=stage_receipt_path,
        stage_receipt_sha256=stage_receipt_sha256,
        receipt_snapshot_dir=root / "receipts", stage_proofs=stage_proofs,
        stage_receipt_file_sha256=stage_receipt_file_sha256,
        working_budget=working_budget,
    )
    config = {
        "schema": "metis17.scoped-signatures/v1", "snapshot": snapshot,
        "semantic_namespace": semantic_namespace, "language": language,
        "language_policy": "prepared_column" if language is None else "require_match",
        "num_perm": num_perm, "n_grams": n_grams, "seed": 16062026, "raw_window": raw_window,
        "minhash_algorithm": "blake2b64,uint64-affine-overflow,mersenne61/v1",
        "code_math_policy": "raw-whitespace-and-case-preserving/v1",
        "stage_receipt_hash": "canonical-payload-sha256/v1",
    }
    scope_root = root / "scopes" / digest_json(config)
    identity = digest_json({
        **input_contract(inputs, stage_proofs, stage_receipt_file_sha256), "config": config,
    })
    batch_root = scope_root / "batches" / _batch_key(batch_id)
    commit = batch_root / "COMMITTED.json"
    with _lock(scope_root, f"batch-{_batch_key(batch_id)}"), \
            storage_namespace(working_budget, "signature-batch", batch_root) as quota:
        if commit.exists():
            manifest = read_receipt(commit)
            if manifest["input_sha256"] != identity:
                raise ValueError("Signature batch_id was reused with different inputs")
            for artifact in manifest["artifacts"]:
                if sha256_file(under_root(root, artifact["path"])) != artifact["sha256"]:
                    raise ValueError("Committed signature artifact checksum mismatch")
            return manifest
        intent = {"input_sha256": identity, "batch_id": batch_id}
        intent_path = batch_root / "INTENT.json"
        if intent_path.exists() and read_receipt(intent_path) != intent:
            raise ValueError("Signature batch_id was reused with different inputs")
        quota_receipt(quota, intent_path, intent)
        for abandoned in batch_root.glob("attempt-*"):
            if abandoned.is_dir():
                quota_rmtree(quota, abandoned)
        with contextlib.ExitStack() as locks:
            new_inputs = []
            by_id = {item["input_id"]: item for item in inputs}
            for input_id in sorted(by_id):
                locks.enter_context(_lock(scope_root, f"input-{input_id}"))
                claim = scope_root / "inputs" / input_id[:2] / f"{input_id}.json"
                if not _claimed_input(root, claim, input_id):
                    new_inputs.append(by_id[input_id])
            work = batch_root / f"attempt-{uuid.uuid4().hex}"
            near = ParquetSink(work / "near.parquet", NEAR_SCHEMA, batch_size, quota=quota)
            try:
                spans = ParquetSink(work / "spans.parquet", SPAN_SCHEMA, batch_size, quota=quota)
            except BaseException:
                near.close()
                raise
            documents, shingled, code_documents = 0, 0, 0
            published = False
            try:
                try:
                    for artifact in new_inputs:
                        with contextlib.closing(prepared_rows(artifact, batch_size, with_text=True)) as rows:
                            for row, text in rows:
                                if language is not None and row["language"] != language:
                                    raise ValueError("Prepared language differs from explicit signature scope")
                                semantic = row["category"] in {"code", "math"}
                                scope = digest_json({
                                    "declared_scope": scope_id(row["language"], snapshot, semantic_namespace),
                                    "category": row["category"], "semantic_tokens": semantic,
                                })
                                signature, count = minhash_signature(
                                    text, semantic=semantic, num_perm=num_perm, n_grams=n_grams,
                                )
                                near.append({
                                    **row, "scope_id": scope, "snapshot": snapshot,
                                    "semantic_namespace": semantic_namespace, "minhash": signature,
                                    "shingle_count": count,
                                    "token_policy": "raw-semantic" if semantic else "case-preserving-words",
                                })
                                for span in _span_records(text, row["category"], scope, raw_window=raw_window):
                                    spans.append({
                                        **{name: row[name] for name in SPAN_SCHEMA.names if name in row},
                                        **span, "scope_id": scope,
                                    })
                                documents += 1
                                shingled += bool(count)
                                code_documents += row["category"] == "code"
                finally:
                    try:
                        near.close()
                    finally:
                        spans.close()
                manifest = {
                    "schema": "metis17.signature-batch/v1", "status": "complete",
                    "batch_id": batch_id, "input_sha256": identity, "config": config,
                    "stage_receipts": stage_proofs,
                    "inputs": inputs, "admitted_inputs": sorted(item["input_id"] for item in new_inputs),
                    "documents": documents, "documents_with_minhash": shingled,
                    "code_documents": code_documents, "span_signatures": spans.count,
                    "artifacts": [near.artifact(root), spans.artifact(root)],
                    "signatures_complete": True, "decisions_complete": False,
                    "decisions_status": "queued_until_comparison_scope_closes",
                    "survivor_comparator": "metis_data17.dedup.winner_key",
                    "source_text_deleted": False,
                }
                for item in new_inputs:
                    input_id = item["input_id"]
                    write_receipt(scope_root / "inputs" / input_id[:2] / f"{input_id}.json", {
                        "input_id": input_id, "sha256": item["sha256"], "commit": str(commit.relative_to(root)),
                        "commit_sha256": digest_json(manifest),
                    })
                quota_receipt(quota, commit, manifest)
                published = True
            finally:
                if not published and not commit.exists():
                    quota_rmtree(quota, work)
    return manifest


def signature_status(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    batches, documents, spans, artifacts = 0, 0, 0, 0
    for path in sorted((root / "scopes").glob("*/batches/*/COMMITTED.json")):
        manifest = read_receipt(path)
        for artifact in manifest["artifacts"]:
            file = under_root(root, artifact["path"])
            if file.stat().st_size != artifact["bytes"] or sha256_file(file) != artifact["sha256"]:
                raise ValueError("Signature metadata artifact checksum mismatch")
            artifacts += 1
        batches += 1
        documents += manifest["documents"]
        spans += manifest["span_signatures"]
    return {
        "schema": "metis17.signature-status/v1", "batches": batches,
        "documents": documents, "span_signatures": spans, "artifacts": artifacts,
        "decisions_complete": False, "decisions_status": "queued_until_comparison_scope_closes",
    }
