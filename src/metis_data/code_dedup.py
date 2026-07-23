from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import shutil
import re
import struct
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator

import numpy as np

from .external_sort import external_sort_records, iter_fixed_records
from .state import atomic_json


LOCKFILES = {
    "bun.lock", "bun.lockb", "cargo.lock", "composer.lock", "gemfile.lock",
    "go.sum", "package-lock.json", "pipfile.lock", "pnpm-lock.yaml",
    "poetry.lock", "uv.lock", "yarn.lock",
}
VENDORED_PARTS = {
    ".bundle", ".next", ".nuxt", "bower_components", "build", "deps", "dist",
    "generated", "node_modules", "packages.lock.json", "target", "third_party",
    "third-party", "vendor", "vendors",
}
GENERATED_SUFFIXES = {
    ".designer.cs", ".g.cs", ".generated.cs", ".min.css", ".min.js",
    ".pb.cc", ".pb.go", ".pb.h", ".pb.py",
}
BENCHMARK_REPOSITORY_MARKERS = {
    "apps-solutions", "codeforces-solutions", "gsm8k-solutions", "humaneval-solutions",
    "leetcode-solutions", "livecodebench-solutions", "mbpp-solutions", "swe-bench-solutions",
}
BENCHMARK_REPOSITORY_RE = re.compile(
    r"(?:^|[/_.-])(?:aime|bigbench(?:hard)?|code[-_]?contests|cruxeval|ds[-_]?1000|gpqa|gsm8k|"
    r"humaneval(?:plus)?|legalbench|livecodebench|mathvista|mbpp(?:plus)?|mmlu(?:[-_]?pro)?|"
    r"olympiadbench|swe[-_]?bench|wmdp)(?:$|[/_.-])"
)
CODE_TOKEN_RE = re.compile(
    r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|'''
    r'''[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|'''
    r'''===|!==|==|!=|<=|>=|=>|->|::|\+\+|--|&&|\|\||<<|>>|\*\*|[{}()\[\];,.?:+\-*/%&|^~<>=!])'''
)
BRACE_FUNCTION_RE = re.compile(
    r"(?m)^[ \t]*(?:[A-Za-z_$][\w$<>:\[\],*&? ]+[ \t]+)?"
    r"[A-Za-z_$][\w$]*[ \t]*\([^;{}\n]{0,500}\)[ \t]*(?:const[ \t]*)?\{"
)
SIGNATURE_DTYPE = np.dtype(
    [("digest", "S16"), ("rank", "<u4"), ("doc", "<u4"), ("priority", "<u2"),
     ("kind", "u1"), ("weight", "<u4"), ("tie", "<u8")]
)
REMOVAL_DTYPE = np.dtype([("doc", "<u4"), ("kind", "u1"), ("weight", "<u4")])
SIGNATURE_RECORD = struct.Struct("<16sIIHBIQ")
REMOVAL_RECORD = struct.Struct("<IBI")
SIGNATURE_SCHEMA = "metis.code-signatures/v2"
REMOVAL_SCHEMA = "metis.code-removals/v2"
KIND_FILE = 0
KIND_UNIT = 1


def _metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, ""):
            return value
    upstream = metadata.get("upstream_metadata")
    if isinstance(upstream, dict):
        for name in names:
            value = upstream.get(name)
            if value not in (None, ""):
                return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def code_hygiene_reason(text: str, metadata: dict[str, Any]) -> str | None:
    """Reject repository material that should never enter structural dedup.

    The checks consume canonical metadata emitted by the repository materializers,
    while retaining conservative path/text fallbacks for packaged code datasets.
    """

    if str(metadata.get("category", "")) != "code":
        return None
    if _truthy(_metadata_value(metadata, "is_fork", "fork", "repository_is_fork")):
        return "repository_fork"
    if _truthy(_metadata_value(metadata, "is_mirror", "mirror", "repository_is_mirror")):
        return "repository_mirror"
    if _truthy(_metadata_value(metadata, "vendored", "is_vendored")):
        return "vendored_file"
    if _truthy(_metadata_value(metadata, "generated_file", "is_generated")):
        return "generated_file"
    if _truthy(_metadata_value(metadata, "benchmark_solution", "is_benchmark_solution")):
        return "benchmark_solution_repository"

    raw_path = str(_metadata_value(metadata, "repo_path", "path", "file_path", "source_file") or "")
    normalized_path = raw_path.replace("\\", "/").lower().lstrip("./")
    path = PurePosixPath(normalized_path)
    if path.name in LOCKFILES:
        return "lockfile"
    if any(part in VENDORED_PARTS for part in path.parts):
        return "vendored_or_build_tree"
    if any(normalized_path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
        return "generated_or_minified_path"
    header = text[:4_000].lower()
    if "@generated" in header or "auto-generated" in header or "automatically generated" in header:
        return "generated_file_header"

    repository = str(
        _metadata_value(metadata, "repository", "repo_name", "repository_name", "repository_url") or ""
    ).lower()
    if any(marker in repository for marker in BENCHMARK_REPOSITORY_MARKERS) or BENCHMARK_REPOSITORY_RE.search(repository):
        return "benchmark_solution_repository"

    lines = text.splitlines()
    if lines:
        longest = max(len(line) for line in lines)
        average = sum(len(line) for line in lines) / len(lines)
        if longest > 20_000 or (average > 500 and len(lines) >= 8):
            return "probable_minified_or_blob"
    compact = re.sub(r"\s+", "", text[:200_000])
    if len(compact) >= 20_000:
        base64ish = sum(character.isalnum() or character in "+/=" for character in compact)
        if base64ish / len(compact) > 0.985:
            return "probable_encoded_blob"
    return None


def code_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in CODE_TOKEN_RE.finditer(text))


def _python_function_spans(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return []
    lines = text.splitlines(keepends=True)
    spans: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "end_lineno"):
            spans.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    return spans


def _brace_function_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in BRACE_FUNCTION_RE.finditer(text):
        start = match.start()
        opening = text.find("{", match.start(), match.end())
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(opening, min(len(text), opening + 1_000_000)):
            character = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'", "`"}:
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start : index + 1])
                    break
    return spans


def structural_units(text: str, metadata: dict[str, Any], *, block_tokens: int = 96) -> dict[bytes, int]:
    """Return exact function/block fingerprints and token weights.

    This is deliberately structural exact matching, not embeddings or semantic
    similarity. A file is removed only when copied units dominate it.
    """

    path = str(_metadata_value(metadata, "repo_path", "path", "file_path", "source_file") or "").lower()
    spans = _python_function_spans(text) if path.endswith(".py") else _brace_function_spans(text)
    units: dict[bytes, int] = {}
    for span in spans:
        tokens = code_tokens(span)
        if len(tokens) < 32:
            continue
        digest = hashlib.blake2b("\0".join(tokens).encode(), digest_size=16).digest()
        units[digest] = max(units.get(digest, 0), len(tokens))

    tokens = code_tokens(text)
    if len(tokens) >= block_tokens:
        for offset in range(0, len(tokens) - block_tokens + 1, block_tokens):
            block = tokens[offset : offset + block_tokens]
            digest = hashlib.blake2b("\0".join(block).encode(), digest_size=16).digest()
            units[digest] = max(units.get(digest, 0), len(block))
    return units


def normalized_file_digest(text: str) -> bytes:
    tokens = code_tokens(text)
    return hashlib.blake2b("\0".join(tokens).encode(), digest_size=16).digest()


def write_code_signatures(
    documents: Iterable[Any], output_root: Path, *, rank: int, finder_workers: int, block_tokens: int = 96
) -> dict[str, Any]:
    if finder_workers < 1:
        raise ValueError("finder_workers must be positive")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".code-rank-{rank:06d}-", dir=output_root))
    handles: OrderedDict[int, BinaryIO] = OrderedDict()
    bucket_counts: dict[int, int] = {}

    def write(bucket: int, row: tuple[Any, ...]) -> None:
        handle = handles.pop(bucket, None)
        if handle is None:
            if len(handles) >= 32:
                _, oldest = handles.popitem(last=False)
                oldest.close()
            handle = (stage / f"{bucket:04d}.sig").open("ab")
        handles[bucket] = handle
        handle.write(SIGNATURE_RECORD.pack(*row))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    documents_seen = 0
    signatures = 0
    try:
        for doc_index, document in enumerate(documents):
            if str(document.metadata.get("category", "")) != "code":
                continue
            documents_seen += 1
            priority = max(1, min(65535, int(document.metadata.get("priority", 1))))
            tie = int.from_bytes(hashlib.blake2b(str(document.id).encode(), digest_size=8).digest(), "little")
            file_digest = normalized_file_digest(str(document.text))
            bucket = int.from_bytes(file_digest[:8], "little") % finder_workers
            write(bucket, (file_digest, rank, doc_index, priority, KIND_FILE, 0, tie))
            signatures += 1
            for digest, weight in structural_units(
                str(document.text), document.metadata, block_tokens=block_tokens
            ).items():
                write(
                    int.from_bytes(digest[:8], "little") % finder_workers,
                    (digest, rank, doc_index, priority, KIND_UNIT, weight, tie),
                )
                signatures += 1
        while handles:
            _, handle = handles.popitem(last=False)
            handle.close()
        bucket_manifest: dict[str, dict[str, Any]] = {}
        for bucket in range(finder_workers):
            destination = output_root / f"{bucket:04d}" / f"{rank:06d}.sig"
            source = stage / f"{bucket:04d}.sig"
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                bucket_manifest[str(bucket)] = {
                    "records": bucket_counts[bucket],
                    "size": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            else:
                destination.unlink(missing_ok=True)
        report = {
            "schema": SIGNATURE_SCHEMA,
            "rank": rank,
            "finder_workers": finder_workers,
            "record_size": SIGNATURE_RECORD.size,
            "documents": documents_seen,
            "signatures": signatures,
            "buckets": bucket_manifest,
        }
        atomic_json(output_root / "_manifests" / f"{rank:06d}.json", report)
        return report
    finally:
        while handles:
            _, handle = handles.popitem(last=False)
            handle.close()
        shutil.rmtree(stage, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_manifests(
    signature_root: Path, finder_workers: int | None, expected_ranks: int | None
) -> list[dict[str, Any]]:
    paths = sorted((signature_root / "_manifests").glob("*.json"))
    expected_ranks = len(paths) if expected_ranks is None else expected_ranks
    if {path.name for path in paths} != {f"{rank:06d}.json" for rank in range(expected_ranks)}:
        raise RuntimeError("Code signature rank manifests are incomplete")
    manifests = []
    for rank, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != SIGNATURE_SCHEMA
            or int(payload.get("rank", -1)) != rank
            or int(payload.get("record_size", -1)) != SIGNATURE_RECORD.size
            or (finder_workers is not None and int(payload.get("finder_workers", -1)) != finder_workers)
        ):
            raise RuntimeError(f"Invalid code signature manifest: {path}")
        for raw_bucket, record in payload.get("buckets", {}).items():
            candidate = signature_root / f"{int(raw_bucket):04d}" / f"{rank:06d}.sig"
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(record.get("size", -1))
                or candidate.stat().st_size != int(record.get("records", -1)) * SIGNATURE_RECORD.size
                or _sha256_file(candidate) != record.get("sha256")
            ):
                raise RuntimeError(f"Code signature partition missing or corrupt: {candidate}")
        manifests.append(payload)
    return manifests


def find_code_duplicates(
    signature_root: Path,
    removal_root: Path,
    *,
    bucket: int,
    finder_workers: int | None = None,
    expected_ranks: int | None = None,
    temporary_directory: Path | None = None,
) -> dict[str, Any]:
    signature_root = Path(signature_root)
    removal_root = Path(removal_root)
    manifests = _code_manifests(signature_root, finder_workers, expected_ranks)
    expected_ranks = len(manifests)
    paths = [
        signature_root / f"{bucket:04d}" / f"{rank:06d}.sig"
        for rank, manifest in enumerate(manifests)
        if str(bucket) in manifest.get("buckets", {})
    ]
    by_rank: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    groups = 0
    current: tuple[bytes, int] | None = None
    keeper: tuple[int, int] | None = None
    seen_nodes: set[tuple[int, int]] = set()
    for digest, rank, document, priority, kind, weight, tie in external_sort_records(
        paths,
        record=SIGNATURE_RECORD,
        key=lambda row: (row[0], row[4], -row[3], row[6], row[1], row[2]),
        temporary_directory=Path(temporary_directory or (removal_root / ".sort-work")),
    ):
        group = (bytes(digest), int(kind))
        node = (int(rank), int(document))
        if group != current:
            current = group
            keeper = node
            seen_nodes = {node}
            continue
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        if len(seen_nodes) == 2:
            groups += 1
        assert keeper is not None
        by_rank[node[0]].append((node[1], int(kind), int(weight)))

    folder = removal_root / f"{bucket:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".code-remove-{bucket:04d}-", dir=folder))
    outputs: dict[str, dict[str, Any]] = {}
    try:
        for rank in range(expected_ranks):
            removals = sorted(set(by_rank.get(rank, [])))
            path = stage / f"{rank:06d}.remove"
            with path.open("wb") as handle:
                for removal in removals:
                    handle.write(REMOVAL_RECORD.pack(*removal))
            outputs[str(rank)] = {
                "records": len(removals),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        for stale in folder.glob("*.remove"):
            stale.unlink()
        for path in stage.glob("*.remove"):
            path.replace(folder / path.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    report = {
        "schema": REMOVAL_SCHEMA,
        "bucket": bucket,
        "finder_workers": finder_workers if finder_workers is not None else -1,
        "expected_ranks": expected_ranks,
        "groups": groups,
        "removals": sum(len(set(items)) for items in by_rank.values()),
        "outputs": outputs,
    }
    atomic_json(removal_root / "_manifests" / f"{bucket:04d}.json", report)
    return report


def load_code_removals(removal_root: Path, *, rank: int, finder_workers: int) -> tuple[set[int], dict[int, int]]:
    files: set[int] = set()
    weights: dict[int, int] = defaultdict(int)
    for bucket in range(finder_workers):
        manifest_path = removal_root / "_manifests" / f"{bucket:04d}.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing code removal manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest.get("outputs", {}).get(str(rank))
        if manifest.get("schema") != REMOVAL_SCHEMA or not isinstance(record, dict):
            raise RuntimeError(f"Invalid code removal manifest: {manifest_path}")
        path = removal_root / f"{bucket:04d}" / f"{rank:06d}.remove"
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or path.stat().st_size != int(record.get("records", -1)) * REMOVAL_RECORD.size
            or _sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"Code removal output missing or corrupt: {path}")
        for doc, kind, weight in iter_fixed_records(path, REMOVAL_RECORD):
            doc = int(doc)
            if int(kind) == KIND_FILE:
                files.add(doc)
            else:
                weights[doc] += int(weight)
    return files, dict(weights)


def build_code_structural_filter(
    removal_root: Path,
    *,
    finder_workers: int,
    duplicate_fraction: float,
    block_tokens: int = 96,
    exclusion_writer: Any = None,
) -> Any:
    from datatrove.pipeline.base import PipelineStep

    class CodeStructuralFilter(PipelineStep):
        name = "Metis exact code function/block deduplication"
        type = "CODE-DEDUP"

        def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Iterator[Any]:
            file_removals, duplicate_weights = load_code_removals(
                removal_root, rank=rank, finder_workers=finder_workers
            )
            with exclusion_writer if exclusion_writer else contextlib.nullcontext() as writer:
                for doc_index, document in enumerate(data):
                    if str(document.metadata.get("category", "")) != "code":
                        yield document
                        continue
                    reason = None
                    if doc_index in file_removals:
                        reason = "normalized_file_duplicate"
                    else:
                        total = sum(
                            structural_units(
                                str(document.text), document.metadata, block_tokens=block_tokens
                            ).values()
                        )
                        duplicated = duplicate_weights.get(doc_index, 0)
                        if total and duplicated / total >= duplicate_fraction:
                            reason = "function_or_block_duplicate"
                    if reason:
                        document.metadata["filter_reason"] = reason
                        if writer:
                            writer.write(document, rank)
                    else:
                        yield document

    return CodeStructuralFilter()
