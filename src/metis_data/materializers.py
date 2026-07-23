from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator
from urllib.parse import quote, urlparse

import pyarrow.parquet as pq
import requests
import yaml
import zstandard as zstd

from .code_dedup import code_hygiene_reason
from .config import repository_root
from .manifest import load_manifest
from .repository_license import (
    DEFAULT_REPOSITORY_LICENSE_ALLOWLIST,
    classify_repository_archive,
)
from .state import atomic_json, utc_now


SUPPORTED_MATERIALIZER_DRIVERS = {
    "canonical_git",
    "canonical_http",
    "canonical_web",
    "repository_index",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
TEXT_SUFFIXES = {
    ".asm", ".bash", ".c", ".cc", ".cfg", ".clj", ".cljs", ".cmake", ".coffee",
    ".cpp", ".cs", ".css", ".cu", ".cuh", ".dart", ".ex", ".exs", ".f", ".f03",
    ".f08", ".f90", ".f95", ".fs", ".fsx", ".go", ".graphql", ".h", ".hh",
    ".hpp", ".hs", ".html", ".java", ".jl", ".js", ".json", ".jsx", ".kt",
    ".kts", ".lean", ".lua", ".m", ".md", ".mjs", ".ml", ".mli", ".mm",
    ".pas", ".php", ".pl", ".proto", ".ps1", ".py", ".r", ".rb", ".rs",
    ".scala", ".scss", ".sh", ".sol", ".sql", ".swift", ".tex", ".tf", ".toml",
    ".ts", ".tsx", ".txt", ".v", ".vb", ".vue", ".xml", ".yaml", ".yml", ".zig",
}
REPOSITORY_INDEX_SCHEMA = "repository_index_codeload/v2"
REPOSITORY_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
GITHUB_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REPOSITORY_NOISY_PARTS = {
    ".git",
    ".idea",
    ".next",
    ".nuxt",
    ".tox",
    ".venv",
    "__pycache__",
    "bower_components",
    "build",
    "coverage",
    "deps",
    "dist",
    "generated",
    "node_modules",
    "target",
    "third_party",
    "third-party",
    "vendor",
    "vendors",
    "venv",
}


class MaterializationError(RuntimeError):
    """A source cannot be materialized without weakening its declared contract."""


class TransientMaterializationError(MaterializationError):
    """Acquisition should stop and safely resume after external state recovers."""


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized[:80] or _stable_id(value)[:16]


def _source_for_item(item: dict[str, Any], profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(profile.get("manifest", "manifests/metis-1.6.yaml"))
    if not manifest_path.is_absolute():
        manifest_path = repository_root() / manifest_path
    manifest = load_manifest(manifest_path)
    try:
        source = next(source for source in manifest["sources"] if source["id"] == item["source_id"])
    except StopIteration as exc:
        raise MaterializationError(f"Unknown source id {item['source_id']!r} in materializer item") from exc
    return manifest, source


def _output_record(
    source_id: str,
    path: Path,
    *,
    materializer: str,
    records: int,
    revision: str,
    text_bytes: int,
) -> dict[str, Any]:
    return {
        "kind": "materialized_jsonl",
        "source_id": source_id,
        "local_path": str(path),
        "repo_path": path.name,
        "revision": revision,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "records": records,
        "text_bytes": int(text_bytes),
        "candidate_token_estimate": int(text_bytes) // 4,
        "candidate_estimator": "accepted_utf8_text_bytes_divided_by_4",
        "materializer": materializer,
        "materialized": True,
        "ready_for_training_build": True,
    }


class _ShardWriter:
    """Atomically writes a bounded work unit as one or more JSONL.zst shards."""

    def __init__(
        self,
        output_root: Path,
        *,
        source_id: str,
        unit_id: str,
        materializer: str,
        revision: str,
        target_uncompressed_bytes: int,
    ) -> None:
        self.output_root = output_root
        self.source_id = source_id
        self.unit_id = _safe_name(unit_id)
        self.materializer = materializer
        self.revision = revision
        self.target_uncompressed_bytes = max(1_000_000, target_uncompressed_bytes)
        self.temporary_root = output_root / ".incomplete" / self.unit_id
        shutil.rmtree(self.temporary_root, ignore_errors=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self._raw: Any = None
        self._compressed: Any = None
        self._text: Any = None
        self._bytes = 0
        self._records = 0
        self._part_records = 0
        self._part = 0
        self._part_text_bytes = 0
        self._temporary_paths: list[tuple[Path, int, int]] = []

    def _open(self) -> None:
        path = self.temporary_root / f"{self.unit_id}-{self._part:05d}.jsonl.zst"
        self._raw = path.open("wb")
        self._compressed = zstd.ZstdCompressor(level=6).stream_writer(self._raw)
        self._text = io.TextIOWrapper(self._compressed, encoding="utf-8")
        self._part_records = 0
        self._part_text_bytes = 0

    def _close_part(self) -> None:
        if self._text is None:
            return
        path = Path(self._raw.name)
        self._text.flush()
        self._text.close()
        self._text = self._compressed = self._raw = None
        if self._part_records:
            self._temporary_paths.append((path, self._part_records, self._part_text_bytes))
        else:
            path.unlink(missing_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if self._text is not None and self._bytes and self._bytes + len(encoded) > self.target_uncompressed_bytes:
            self._close_part()
            self._part += 1
            self._bytes = 0
        if self._text is None:
            self._open()
        self._text.write(encoded.decode("utf-8"))
        self._bytes += len(encoded)
        self._records += 1
        self._part_records += 1
        self._part_text_bytes += len(str(row.get("text", "")).encode("utf-8"))

    def finish(self) -> tuple[list[dict[str, Any]], int]:
        self._close_part()
        outputs: list[dict[str, Any]] = []
        self.output_root.mkdir(parents=True, exist_ok=True)
        for temporary, records, text_bytes in self._temporary_paths:
            final = self.output_root / temporary.name
            os.replace(temporary, final)
            outputs.append(
                _output_record(
                    self.source_id,
                    final,
                    materializer=self.materializer,
                    records=records,
                    revision=self.revision,
                    text_bytes=text_bytes,
                )
            )
        shutil.rmtree(self.temporary_root, ignore_errors=True)
        return outputs, self._records

    def abort(self) -> None:
        with contextlib.suppress(Exception):
            self._close_part()
        shutil.rmtree(self.temporary_root, ignore_errors=True)


def _marker_path(output_root: Path, unit_id: str) -> Path:
    return output_root / ".markers" / f"{_safe_name(unit_id)}.json"


def _load_completed_unit(output_root: Path, unit_id: str, signature: str) -> list[dict[str, Any]] | None:
    marker = _marker_path(output_root, unit_id)
    if not marker.exists():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("input_signature") != signature:
        raise MaterializationError(
            f"Restart marker drift for {unit_id}: upstream selection changed; remove {marker} only after review"
        )
    outputs = payload.get("outputs", [])
    for output in outputs:
        path = Path(output["local_path"])
        if not path.exists() or path.stat().st_size != int(output["size"]):
            raise MaterializationError(f"Completed materializer output is missing or truncated: {path}")
        if _sha256_file(path) != output["sha256"]:
            raise MaterializationError(f"Completed materializer output checksum changed: {path}")
    return outputs


def _commit_unit(output_root: Path, unit_id: str, signature: str, outputs: list[dict[str, Any]], report: dict[str, Any]) -> None:
    atomic_json(
        _marker_path(output_root, unit_id),
        {
            "schema": "metis.materializer-unit/v1",
            "unit_id": unit_id,
            "input_signature": signature,
            "completed_at": utc_now(),
            "outputs": outputs,
            "report": report,
        },
    )


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 3600) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise MaterializationError(f"Command failed ({' '.join(command[:4])}): {(detail or ['unknown error'])[-1]}") from exc
    return result.stdout


def _resolve_git_revision(url: str, requested: str | None) -> str:
    if requested and HEX40.fullmatch(requested):
        return requested
    ref = requested or "HEAD"
    output = _run(["git", "ls-remote", url, ref], timeout=180)
    matches = [line.split()[0] for line in output.splitlines() if line.strip()]
    if not matches or not HEX40.fullmatch(matches[0]):
        raise MaterializationError(f"Unable to resolve a full Git revision for {url} ref {ref!r}")
    return matches[0]


def _registry_path(access: dict[str, Any]) -> Path:
    configured = access.get("registry")
    if not configured:
        raise MaterializationError("Canonical source is missing access.registry")
    path = (repository_root() / str(configured)).resolve()
    if not path.exists():
        raise MaterializationError(f"Canonical registry does not exist: {path}")
    return path


def _git_registry_entries(
    item: dict[str, Any], manifest: dict[str, Any], source: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    if payload.get("schema") != "metis.canonical-git-registry/v1":
        raise MaterializationError(
            f"{source['id']}: canonical_git requires a metis.canonical-git-registry/v1 registry; "
            f"{source['access']['registry']} is {payload.get('schema')!r}"
        )
    selector = source["access"].get("selector")
    groups = payload.get("groups", {})
    if selector:
        entries = groups.get(selector)
        if not isinstance(entries, list):
            raise MaterializationError(
                f"{source['id']}: selector {selector!r} is not declared under registry.groups; "
                "add a source-specific list of Git repository ids"
            )
        by_id = {entry.get("id"): entry for entry in payload.get("repositories", [])}
        selected = [by_id.get(value) if isinstance(value, str) else value for value in entries]
        if any(not isinstance(entry, dict) for entry in selected):
            raise MaterializationError(f"{source['id']}: registry group {selector!r} references an unknown repository")
    else:
        same_registry = [
            candidate["id"]
            for candidate in manifest["sources"]
            if candidate.get("acquisition", {}).get("driver") == "canonical_git"
            and candidate.get("access", {}).get("registry") == source["access"].get("registry")
        ]
        if len(same_registry) > 1:
            raise MaterializationError(
                f"{source['id']}: registry is shared by {same_registry} but none has a selector. "
                "Declare registry.groups and access.selector so formal and systems files cannot be silently mixed"
            )
        selected = payload.get("repositories")
        if not isinstance(selected, list) or not selected:
            raise MaterializationError(f"{source['id']}: canonical Git registry contains no repositories")

    if payload.get("resolve_revisions_on_first_run") is not False:
        raise MaterializationError(
            f"{source['id']}: canonical Git registry must disable first-run revision resolution"
        )
    for entry in selected:
        revision = str(entry.get("revision") or "")
        if not HEX40.fullmatch(revision):
            raise MaterializationError(
                f"{source['id']}: canonical Git entry {entry.get('id')!r} must pin a full 40-hex revision"
            )
    return list(selected)


def _language_for_path(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "text"


def _materialize_git_repository(
    *,
    source: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    work_root: Path,
    target_shard_bytes: int,
    maximum_file_bytes: int,
) -> list[dict[str, Any]]:
    repo_id = str(entry.get("id", ""))
    url = str(entry.get("url", ""))
    license_name = str(entry.get("license", "")).strip()
    if not repo_id or not url.startswith("https://") or not license_name:
        raise MaterializationError(
            f"{source['id']}: canonical Git entry must include id, HTTPS url, and an explicit license: {entry}"
        )
    revision = _resolve_git_revision(url, str(entry.get("revision") or entry.get("ref") or "HEAD"))
    signature = _stable_id("canonical_git/v1", source["id"], url, revision, license_name, maximum_file_bytes)
    unit_id = f"git-{repo_id}-{revision[:12]}"
    completed = _load_completed_unit(output_root, unit_id, signature)
    if completed is not None:
        return completed

    checkout = work_root / _safe_name(f"{repo_id}-{revision[:12]}")
    shutil.rmtree(checkout, ignore_errors=True)
    checkout.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "--quiet"], cwd=checkout)
    _run(["git", "remote", "add", "origin", url], cwd=checkout)
    _run(["git", "fetch", "--quiet", "--depth", "1", "origin", revision], cwd=checkout, timeout=7200)
    _run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout, timeout=7200)
    writer = _ShardWriter(
        output_root,
        source_id=source["id"],
        unit_id=unit_id,
        materializer="canonical_git/v1",
        revision=revision,
        target_uncompressed_bytes=target_shard_bytes,
    )
    rejected = {"non_text": 0, "too_large": 0, "hygiene": 0, "decode": 0}
    try:
        for path in sorted(candidate for candidate in checkout.rglob("*") if candidate.is_file()):
            relative = path.relative_to(checkout)
            if ".git" in relative.parts:
                continue
            if path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES:
                rejected["non_text"] += 1
                continue
            size = path.stat().st_size
            if size <= 0 or size > maximum_file_bytes:
                rejected["too_large"] += 1
                continue
            try:
                raw = path.read_bytes()
                if b"\0" in raw:
                    rejected["non_text"] += 1
                    continue
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                rejected["decode"] += 1
                continue
            metadata = {
                "repository": repo_id,
                "repository_url": url,
                "repo_path": relative.as_posix(),
                "path": relative.as_posix(),
                "commit_id": revision,
                "license": license_name,
                "canonical_url": f"{url.removesuffix('.git')}/blob/{revision}/{quote(relative.as_posix())}",
                "language": _language_for_path(relative),
                "generated_file_probability": 0.0,
                "retrieval_date": utc_now(),
            }
            if code_hygiene_reason(text, {**metadata, "category": "code"}):
                rejected["hygiene"] += 1
                continue
            writer.write(
                {
                    "id": _stable_id(source["id"], repo_id, revision, relative.as_posix()),
                    "text": text,
                    "metadata": metadata,
                }
            )
        outputs, records = writer.finish()
    except BaseException:
        writer.abort()
        raise
    finally:
        shutil.rmtree(checkout, ignore_errors=True)
    if records == 0:
        raise MaterializationError(f"{source['id']}: repository {repo_id} produced zero accepted text files")
    _commit_unit(output_root, unit_id, signature, outputs, {"records": records, "rejected": rejected})
    return outputs


def _materialize_canonical_git(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path, manifest: dict[str, Any], source: dict[str, Any]
) -> list[dict[str, Any]]:
    registry = _registry_path(item["access"])
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    entries = _git_registry_entries(item, manifest, source, payload)
    output_root = root / "raw" / source["id"] / "materialized"
    work_root = root / profile["runtime"].get("temp_dir", "cache/tmp") / "materializers" / source["id"]
    work_root.mkdir(parents=True, exist_ok=True)
    target_shard_bytes = int(profile.get("acquisition", {}).get("materializer_shard_bytes", 512_000_000))
    maximum_file_bytes = int(profile.get("acquisition", {}).get("maximum_repository_file_bytes", 2_000_000))
    outputs: list[dict[str, Any]] = []
    for entry in entries:
        outputs.extend(
            _materialize_git_repository(
                source=source,
                entry=entry,
                output_root=output_root,
                work_root=work_root,
                target_shard_bytes=target_shard_bytes,
                maximum_file_bytes=maximum_file_bytes,
            )
        )
    return outputs


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif tag.lower() in {"p", "br", "li", "pre", "code", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
        elif tag.lower() in {"p", "li", "pre", "code", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"[ \t]+", " ", line).strip() for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def _request(session: requests.Session, url: str, *, stream: bool = False, timeout: int = 900) -> requests.Response:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise MaterializationError(f"Only explicit HTTPS acquisition URLs are allowed, got {url!r}")
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(url, stream=stream, timeout=timeout)
            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                reset = response.headers.get("X-RateLimit-Reset", "unknown")
                raise TransientMaterializationError(
                    f"GitHub API rate limit exhausted while fetching {url}; reset epoch={reset}. "
                    "The work unit is restartable; resume after the reset or use an approved higher-quota token"
                )
            if response.status_code in {429, 500, 502, 503, 504}:
                retry = min(60, int(response.headers.get("Retry-After", 2 ** attempt)))
                time.sleep(retry)
                continue
            response.raise_for_status()
            return response
        except MaterializationError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(30, 2 ** attempt))
    raise MaterializationError(f"HTTPS acquisition failed for {url}: {last_error}")


def _registry_web_entries(source: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    selector = source["access"].get("selector")
    schema = payload.get("schema")
    if schema == "metis.canonical-web-registry/v1":
        entries = payload.get("seeds", [])
        if selector == "stable_specs":
            entries = [entry for entry in entries if entry.get("stable")]
        elif selector and selector != payload.get("selection"):
            raise MaterializationError(
                f"{source['id']}: selector {selector!r} is not represented by this canonical-web registry"
            )
        return entries
    if schema in {"metis.reference-registry/v1", "metis.science-registry/v1"}:
        if not selector:
            raise MaterializationError(f"{source['id']}: {schema} requires access.selector")
        entries = payload.get("sources", {}).get(selector)
        if not isinstance(entries, list):
            raise MaterializationError(f"{source['id']}: registry selector {selector!r} does not exist")
        return entries
    raise MaterializationError(f"{source['id']}: unsupported canonical registry schema {schema!r}")


def _artifact_rows(path: Path, artifact: dict[str, Any]) -> Iterator[dict[str, Any]]:
    record_format = str(artifact.get("record_format", "")).lower()
    if record_format == "text":
        yield {"text": path.read_text(encoding="utf-8")}
        return
    if record_format == "parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            yield from batch.to_pylist()
        return
    if record_format in {"jsonl", "jsonl.gz", "jsonl.zst"}:
        if record_format == "jsonl.gz":
            handle: Any = gzip.open(path, "rt", encoding="utf-8")
        elif record_format == "jsonl.zst":
            raw = path.open("rb")
            handle = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        else:
            handle = path.open("r", encoding="utf-8")
        with handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return
    raise MaterializationError(
        f"Artifact {artifact.get('url')} has unsupported/missing record_format; "
        "declare one of text, parquet, jsonl, jsonl.gz, or jsonl.zst"
    )


def _download_artifact(session: requests.Session, url: str, path: Path, expected_sha: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise MaterializationError(f"Only explicit HTTPS artifact URLs are allowed, got {url!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise MaterializationError(f"Direct artifact {url} must declare a lowercase SHA-256 checksum")
    if path.exists() and _sha256_file(path) == expected_sha:
        return
    part = path.with_suffix(path.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    existing = part.stat().st_size if part.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    response: requests.Response | None = None
    for attempt in range(6):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=900)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(60, int(response.headers.get("Retry-After", 2 ** attempt))))
                continue
            if response.status_code not in {200, 206}:
                response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 5:
                raise MaterializationError(f"Canonical artifact download failed for {url}: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))
    if response is None or response.status_code not in {200, 206}:
        raise MaterializationError(f"Canonical artifact retries were exhausted for {url}")
    mode = "ab" if response.status_code == 206 and existing else "wb"
    with part.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    if _sha256_file(part) != expected_sha:
        raise MaterializationError(f"SHA-256 mismatch for canonical artifact {url}")
    os.replace(part, path)


def _materialize_web_entry(
    *, source: dict[str, Any], entry: dict[str, Any], output_root: Path, target_shard_bytes: int
) -> list[dict[str, Any]]:
    mode = str(entry.get("fetch_mode") or entry.get("mode") or "")
    license_name = str(entry.get("license") or entry.get("license_expression") or "").strip()
    if not mode:
        raise MaterializationError(
            f"{source['id']} entry {entry.get('id')!r} is a discovery/landing URL, not an acquisition contract. "
            "Declare fetch_mode plus an explicit license; use single_page only for one document, or add typed artifacts/sitemap rules"
        )
    if not license_name:
        raise MaterializationError(f"{source['id']} entry {entry.get('id')!r} is missing an explicit record license")
    entry_id = str(entry.get("id") or _stable_id(entry)[:16])
    unit_id = f"web-{entry_id}"
    signature = _stable_id("canonical_web_http/v1", source["id"], json.dumps(entry, sort_keys=True))
    completed = _load_completed_unit(output_root, unit_id, signature)
    if completed is not None:
        return completed
    session = requests.Session()
    session.headers["User-Agent"] = "MetisDataFactory/1.6 (+research corpus acquisition)"
    writer = _ShardWriter(
        output_root,
        source_id=source["id"],
        unit_id=unit_id,
        materializer="canonical_web_http/v1",
        revision=signature,
        target_uncompressed_bytes=target_shard_bytes,
    )
    try:
        if mode == "single_page":
            url = str(entry.get("url", ""))
            response = _request(session, url, timeout=120)
            if len(response.content) > 25_000_000:
                raise MaterializationError(f"single_page response exceeds 25MB for {url}; use an explicit artifact")
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                parser = _VisibleTextParser()
                parser.feed(response.text)
                text = parser.text()
            elif content_type.startswith("text/"):
                text = response.text
            else:
                raise MaterializationError(f"single_page requires HTML/text, got {content_type!r} for {url}")
            writer.write(
                {
                    "id": _stable_id(source["id"], url, signature),
                    "text": text,
                    "metadata": {
                        "canonical_url": url,
                        "license": license_name,
                        "version": entry.get("version"),
                        "publication_date": entry.get("publication_date"),
                        "capture_date": utc_now(),
                        "retrieval_date": utc_now(),
                        "current_version": entry.get("version") in {"current", "latest", "stable"},
                    },
                }
            )
        elif mode == "artifacts":
            artifacts = entry.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise MaterializationError(f"{source['id']} entry {entry_id!r} declares artifacts mode without artifacts")
            artifact_root = output_root.parent / "artifacts" / _safe_name(entry_id)
            for artifact in artifacts:
                url = str(artifact.get("url", ""))
                expected_sha = str(artifact.get("sha256", ""))
                filename = str(artifact.get("filename") or Path(urlparse(url).path).name)
                if not filename:
                    raise MaterializationError(f"Artifact URL has no filename: {url}")
                local = artifact_root / _safe_name(filename)
                _download_artifact(session, url, local, expected_sha)
                text_field = str(artifact.get("text_field", "text"))
                id_field = str(artifact.get("id_field", "id"))
                for row_index, row in enumerate(_artifact_rows(local, artifact)):
                    text = row.get(text_field)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    row_license = str(row.get(str(artifact.get("license_field", "license")), "") or license_name)
                    if not row_license:
                        continue
                    writer.write(
                        {
                            "id": str(row.get(id_field) or _stable_id(source["id"], url, row_index)),
                            "text": text,
                            "metadata": {
                                "canonical_url": str(row.get(str(artifact.get("url_field", "url")), "") or url),
                                "license": row_license,
                                "version": artifact.get("version") or entry.get("version"),
                                "publication_date": row.get(str(artifact.get("publication_date_field", "publication_date"))),
                                "capture_date": utc_now(),
                                "retrieval_date": utc_now(),
                                "source_artifact_sha256": expected_sha,
                            },
                        }
                    )
        else:
            raise MaterializationError(
                f"{source['id']} entry {entry_id!r} uses unsupported fetch_mode {mode!r}; "
                "supported bounded modes are single_page and artifacts"
            )
        outputs, records = writer.finish()
    except BaseException:
        writer.abort()
        raise
    if records == 0:
        raise MaterializationError(f"{source['id']} entry {entry_id!r} produced zero licensed text records")
    _commit_unit(output_root, unit_id, signature, outputs, {"records": records, "entry": entry_id})
    return outputs


def _materialize_canonical_web_http(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    registry = _registry_path(item["access"])
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    entries = _registry_web_entries(source, payload)
    if not entries:
        raise MaterializationError(f"{source['id']}: registry selection contains no entries")
    output_root = root / "raw" / source["id"] / "materialized"
    target_shard_bytes = int(profile.get("acquisition", {}).get("materializer_shard_bytes", 512_000_000))
    outputs: list[dict[str, Any]] = []
    for entry in entries:
        outputs.extend(
            _materialize_web_entry(
                source=source,
                entry=entry,
                output_root=output_root,
                target_shard_bytes=target_shard_bytes,
            )
        )
    return outputs


class _CodeloadArchiveClient:
    """Unauthenticated, resumable client for one archive per repo+commit."""

    def __init__(
        self,
        cache_root: Path,
        *,
        retries: int = 6,
        timeout: int = 900,
        retain_archives: bool = False,
    ) -> None:
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.retries = max(0, int(retries))
        self.timeout = int(timeout)
        self.retain_archives = bool(retain_archives)
        self.session = requests.Session()
        # Never inherit a GitHub credential into codeload requests.  Public
        # archives do not need one, and a token would make this path both
        # rate-limit-bound and unnecessarily secret-bearing.
        self.session.headers.clear()
        self.session.headers.update(
            {
                "User-Agent": "MetisDataFactory/1.6 (+public codeload archive hydration)",
                "Accept-Encoding": "identity",
            }
        )

    def _archive_paths(self, repo: str, commit: str) -> tuple[Path, Path]:
        repository_key = repo.replace("/", "--")
        archive = self.cache_root / repository_key / f"{commit}.tar.gz"
        return archive, archive.with_suffix(archive.suffix + ".sha256.json")

    def _verified_existing(self, archive: Path, receipt: Path, url: str) -> dict[str, Any] | None:
        if not archive.exists():
            return None
        actual_sha256 = _sha256_file(archive)
        if receipt.exists():
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                payload.get("url") != url
                or int(payload.get("size", -1)) != archive.stat().st_size
                or payload.get("sha256") != actual_sha256
            ):
                raise MaterializationError(f"Cached codeload archive checksum/provenance changed: {archive}")
        else:
            payload = {
                "schema": "metis.repository-archive/v1",
                "url": url,
                "size": archive.stat().st_size,
                "sha256": actual_sha256,
                "retrieved_at": utc_now(),
            }
            atomic_json(receipt, payload)
        return {"path": archive, "receipt_path": receipt, **payload}

    def fetch(self, repo: str, commit: str) -> dict[str, Any]:
        repo = _normalize_repository_name(repo)
        commit = _normalize_repository_commit(commit)
        owner, repository = repo.split("/", 1)
        url = (
            f"https://codeload.github.com/{quote(owner, safe='')}/"
            f"{quote(repository, safe='')}/tar.gz/{commit}"
        )
        archive, receipt = self._archive_paths(repo, commit)
        existing = self._verified_existing(archive, receipt, url)
        if existing is not None:
            return existing
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(archive.suffix + ".partial")
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                response = self.session.get(url, headers=headers, stream=True, timeout=self.timeout)
                if response.status_code == 404:
                    response.close()
                    raise MaterializationError("repository_archive_unavailable")
                if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                    response.close()
                    raise TransientMaterializationError(f"transient_codeload_http_{response.status_code}")
                if offset and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    offset = 0
                elif offset and response.status_code != 206:
                    response.raise_for_status()
                else:
                    response.raise_for_status()
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, archive)
                payload = {
                    "schema": "metis.repository-archive/v1",
                    "url": url,
                    "size": archive.stat().st_size,
                    "sha256": _sha256_file(archive),
                    "retrieved_at": utc_now(),
                }
                atomic_json(receipt, payload)
                return {"path": archive, "receipt_path": receipt, **payload}
            except MaterializationError as exc:
                last_error = exc
                if not isinstance(exc, TransientMaterializationError):
                    raise
            except (OSError, requests.RequestException) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(60, 2**attempt))
        raise TransientMaterializationError(
            f"codeload archive retries exhausted for {repo}@{commit}: {last_error}"
        )

    def release(self, archive_record: dict[str, Any]) -> None:
        if self.retain_archives:
            return
        archive = Path(str(archive_record["path"]))
        receipt_value = archive_record.get("receipt_path")
        receipt = (
            Path(str(receipt_value))
            if receipt_value
            else archive.with_suffix(archive.suffix + ".sha256.json")
        )
        archive.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            archive.parent.rmdir()


def _normalize_repository_name(value: Any) -> str:
    repo = str(value or "").strip().strip("/")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if repo.lower().startswith(prefix):
            repo = repo[len(prefix) :]
            break
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = repo.strip("/")
    if not GITHUB_REPOSITORY_RE.fullmatch(repo):
        raise MaterializationError("invalid_or_non_github_repository")
    return repo.lower()


def _normalize_repository_commit(value: Any) -> str:
    commit = str(value or "").strip().lower()
    if not REPOSITORY_COMMIT_RE.fullmatch(commit):
        raise MaterializationError("invalid_repository_commit")
    return commit


def _normalize_repository_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip().lstrip("/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise MaterializationError("invalid_repository_path")
    lowered = {part.lower() for part in path.parts}
    if lowered & REPOSITORY_NOISY_PARTS:
        raise MaterializationError("vendored_or_build_tree")
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name.lower() not in {
        "dockerfile",
        "makefile",
        "readme",
    }:
        raise MaterializationError("unsupported_repository_file_type")
    return path.as_posix()


def _metadata_files(root: Path, source: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    files: list[tuple[dict[str, Any], Path]] = []
    for component in source["access"].get("components", []):
        component_root = root / "raw" / source["id"] / component["repo_id"].replace("/", "--") / component["revision"]
        if not component_root.exists():
            raise MaterializationError(
                f"{source['id']}: downloaded metadata component is missing at {component_root}; resolve/download it first"
            )
        for path in sorted(component_root.rglob("*")):
            lower = path.name.lower()
            if path.is_file() and (
                lower.endswith(".parquet") or lower.endswith(".jsonl") or lower.endswith(".jsonl.gz") or lower.endswith(".jsonl.zst")
            ):
                files.append((component, path))
    if not files:
        raise MaterializationError(
            f"{source['id']}: repository-index components contain no supported Parquet/JSONL metadata files"
        )
    return files


def _metadata_units(component: dict[str, Any], path: Path) -> Iterator[tuple[str, str, Iterator[dict[str, Any]]]]:
    component_key = f"{component['repo_id']}@{component['revision']}:{path.name}"
    metadata_sha256 = _sha256_file(path)
    if path.name.lower().endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            unit_id = f"index-{_stable_id(component_key)[:12]}-rg-{group:05d}"

            def rows(group_index: int = group, metadata_path: Path = path) -> Iterator[dict[str, Any]]:
                # Open lazily so sorting thousands of work units does not retain
                # one file descriptor per metadata shard, and stream batches so
                # a very large row group cannot dominate login-node memory.
                parquet_file = pq.ParquetFile(metadata_path)
                for batch in parquet_file.iter_batches(batch_size=65_536, row_groups=[group_index]):
                    yield from batch.to_pylist()

            signature = _stable_id("repository_index/v1", component_key, metadata_sha256, group)
            yield unit_id, signature, rows()
        return

    def json_rows() -> Iterator[dict[str, Any]]:
        lower = path.name.lower()
        if lower.endswith(".jsonl.gz"):
            handle: Any = gzip.open(path, "rt", encoding="utf-8")
        elif lower.endswith(".jsonl.zst"):
            raw = path.open("rb")
            handle = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        else:
            handle = path.open("r", encoding="utf-8")
        with handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row

    yield (
        f"index-{_stable_id(component_key)[:12]}",
        _stable_id("repository_index/v1", component_key, metadata_sha256),
        json_rows(),
    )


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def _index_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _repository_index_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=600, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    row = connection.execute("SELECT value FROM settings WHERE key='schema'").fetchone()
    if row and str(row[0]) != REPOSITORY_INDEX_SCHEMA:
        connection.close()
        raise MaterializationError(
            f"Repository request index schema drift at {path}: found {row[0]!r}, "
            f"expected {REPOSITORY_INDEX_SCHEMA!r}; preserve the old database for audit, then rebuild it"
        )
    connection.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES('schema', ?)",
        (REPOSITORY_INDEX_SCHEMA,),
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata_units ("
        "unit_key BLOB PRIMARY KEY, unit_id TEXT NOT NULL UNIQUE, signature TEXT NOT NULL, component TEXT NOT NULL, "
        "rows_seen INTEGER NOT NULL, requests_inserted INTEGER NOT NULL, completed_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS repo_commits ("
        "repo_key BLOB PRIMARY KEY, repo TEXT NOT NULL, commit_id TEXT NOT NULL, rank_key TEXT NOT NULL, "
        "UNIQUE(repo, commit_id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS repo_commits_rank ON repo_commits(rank_key, repo, commit_id)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS requested_paths ("
        "repo_key BLOB NOT NULL, rel_path TEXT NOT NULL, language TEXT, index_license TEXT, "
        "metadata_unit_key BLOB NOT NULL, PRIMARY KEY(repo_key, rel_path)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS repository_state ("
        "repo TEXT NOT NULL, commit_id TEXT NOT NULL, request_signature TEXT NOT NULL, "
        "status TEXT NOT NULL, reason TEXT, archive_sha256 TEXT, archive_size INTEGER, "
        "license TEXT, license_file TEXT, content_manifest_sha256 TEXT, "
        "accepted_text_bytes INTEGER NOT NULL DEFAULT 0, "
        "batch_id TEXT, completed_at TEXT NOT NULL, PRIMARY KEY(repo, commit_id)) WITHOUT ROWID"
    )
    state_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(repository_state)")
    }
    if "batch_id" not in state_columns:
        connection.execute("ALTER TABLE repository_state ADD COLUMN batch_id TEXT")
    if "content_manifest_sha256" not in state_columns:
        connection.execute(
            "ALTER TABLE repository_state ADD COLUMN content_manifest_sha256 TEXT"
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS output_batches ("
        "batch_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL UNIQUE, signature TEXT NOT NULL, "
        "accepted_text_bytes INTEGER NOT NULL, repositories INTEGER NOT NULL, completed_at TEXT NOT NULL)"
    )
    return connection


def _metadata_request(
    row: dict[str, Any],
    *,
    unit_key: bytes,
) -> tuple[tuple[bytes, str, str, str, str | None, str | None, bytes] | None, str | None]:
    if _index_truthy(_field(row, "is_fork", "fork", "repository_is_fork")):
        return None, "repository_fork"
    if _index_truthy(_field(row, "is_mirror", "mirror", "repository_is_mirror")):
        return None, "repository_mirror"
    try:
        repo = _normalize_repository_name(
            _field(row, "repo", "repository", "repo_name", "repository_name", "repository_url")
        )
        commit = _normalize_repository_commit(_field(row, "commit_id", "commit", "revision", "sha"))
        rel_path = _normalize_repository_path(_field(row, "rel_path", "path", "file_path"))
    except MaterializationError as exc:
        return None, str(exc)
    language_value = _field(row, "language", "lang")
    language = str(language_value).strip() if language_value not in (None, "") else None
    # Index licenses are retained solely as provenance. They never authorize
    # the underlying file; the pinned archive must carry an accepted root
    # license file before any requested path is emitted.
    index_license_value = _field(row, "license", "repo_license", "spdx_license")
    index_license = str(index_license_value).strip() if index_license_value not in (None, "") else None
    return (
        hashlib.sha256(f"{repo}\0{commit}".encode("utf-8")).digest()[:16],
        repo,
        commit,
        rel_path,
        language,
        index_license,
        unit_key,
    ), None


def _ingest_repository_metadata(
    connection: sqlite3.Connection,
    *,
    source: dict[str, Any],
    metadata_files: Iterable[tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    """Group all index rows on disk before fetching a single source archive."""

    counters: dict[str, int] = {}
    units_completed = 0
    units_reused = 0
    for component, path in metadata_files:
        component_name = f"{component['repo_id']}@{component['revision']}:{path.name}"
        for unit_id, signature, rows in _metadata_units(component, path):
            unit_key = hashlib.sha256(unit_id.encode("utf-8")).digest()[:16]
            existing = connection.execute(
                "SELECT signature FROM metadata_units WHERE unit_id=?", (unit_id,)
            ).fetchone()
            if existing:
                if str(existing[0]) != signature:
                    raise MaterializationError(
                        f"{source['id']}: metadata unit signature drift for {unit_id}; "
                        "the pinned metadata or local file changed"
                    )
                units_reused += 1
                continue
            rows_seen = 0
            requests_inserted = 0
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO metadata_units("
                    "unit_key, unit_id, signature, component, rows_seen, requests_inserted, completed_at"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (unit_key, unit_id, signature, component_name, 0, 0, ""),
                )
                request_batch: list[tuple[bytes, str, str | None, str | None, bytes]] = []
                repo_batch: dict[bytes, tuple[bytes, str, str, str]] = {}

                def flush() -> None:
                    nonlocal requests_inserted
                    if not request_batch:
                        return
                    connection.executemany(
                        "INSERT OR IGNORE INTO repo_commits(repo_key, repo, commit_id, rank_key) VALUES(?,?,?,?)",
                        repo_batch.values(),
                    )
                    before_requests = connection.total_changes
                    connection.executemany(
                        "INSERT OR IGNORE INTO requested_paths("
                        "repo_key, rel_path, language, index_license, metadata_unit_key"
                        ") VALUES(?,?,?,?,?)",
                        request_batch,
                    )
                    requests_inserted += connection.total_changes - before_requests
                    request_batch.clear()
                    repo_batch.clear()

                for row in rows:
                    rows_seen += 1
                    request, reason = _metadata_request(
                        row,
                        unit_key=unit_key,
                    )
                    if request is None:
                        assert reason is not None
                        counters[reason] = counters.get(reason, 0) + 1
                        continue
                    repo_key, repo, commit, rel_path, language, index_license, request_unit_key = request
                    request_batch.append(
                        (repo_key, rel_path, language, index_license, request_unit_key)
                    )
                    repo_batch[repo_key] = (
                        repo_key,
                        repo,
                        commit,
                        _stable_id(source["id"], repo, commit),
                    )
                    if len(request_batch) >= 10_000:
                        flush()
                flush()
                connection.execute(
                    "UPDATE metadata_units SET rows_seen=?, requests_inserted=?, completed_at=? "
                    "WHERE unit_key=?",
                    (rows_seen, requests_inserted, utc_now(), unit_key),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            units_completed += 1
    request_count = int(connection.execute("SELECT COUNT(*) FROM requested_paths").fetchone()[0])
    repository_count = int(connection.execute("SELECT COUNT(*) FROM repo_commits").fetchone()[0])
    if request_count == 0:
        raise MaterializationError(
            f"{source['id']}: repository-index metadata yielded no valid GitHub repo/commit/path requests"
        )
    return {
        "metadata_units_completed": units_completed,
        "metadata_units_reused": units_reused,
        "repositories": repository_count,
        "requested_paths": request_count,
        "rejected_rows": counters,
    }


def _repository_requests(
    connection: sqlite3.Connection,
    repo_key: bytes,
    repo: str,
    commit: str,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for rel_path, language, index_license, component, metadata_unit in connection.execute(
        "SELECT p.rel_path, p.language, p.index_license, u.component, u.unit_id "
        "FROM requested_paths p JOIN metadata_units u ON u.unit_key=p.metadata_unit_key "
        "WHERE p.repo_key=? ORDER BY p.rel_path",
        (repo_key,),
    ):
        row = {
            "rel_path": str(rel_path),
            "language": str(language) if language is not None else None,
            "index_license": str(index_license) if index_license is not None else None,
            "metadata_component": str(component),
            "metadata_unit": str(metadata_unit),
        }
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    signature = _stable_id(
        REPOSITORY_INDEX_SCHEMA,
        repo,
        commit,
        len(rows),
        digest.hexdigest(),
    )
    return rows, signature


def _archive_member_relative_path(member_name: str) -> str | None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        return None
    return PurePosixPath(*path.parts[1:]).as_posix()


def _release_repository_archive(client: Any, archive_record: dict[str, Any]) -> None:
    release = getattr(client, "release", None)
    if callable(release):
        release(archive_record)


def _materialize_repository_unit(
    *,
    source: dict[str, Any],
    repo: str,
    commit: str,
    requests_for_repo: Iterable[dict[str, Any]],
    unit_id: str,
    signature: str,
    output_root: Path,
    client: Any,
    target_shard_bytes: int,
    maximum_file_bytes: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    completed = _load_completed_unit(output_root, unit_id, signature)
    if completed is not None:
        text_bytes = sum(int(output.get("text_bytes", 0)) for output in completed)
        report = {
            "archive_sha256": next(
                (str(output["repository_archive_sha256"]) for output in completed if output.get("repository_archive_sha256")),
                None,
            ),
            "archive_size": next(
                (int(output["repository_archive_size"]) for output in completed if output.get("repository_archive_size")),
                None,
            ),
            "license": next((str(output["license"]) for output in completed if output.get("license")), None),
            "license_file": next(
                (str(output["license_file"]) for output in completed if output.get("license_file")),
                None,
            ),
            "content_manifest_sha256": next(
                (
                    str(output["repository_content_manifest_sha256"])
                    for output in completed
                    if output.get("repository_content_manifest_sha256")
                ),
                None,
            ),
            "resumed": True,
        }
        return completed, text_bytes, report
    request_rows = list(requests_for_repo)
    if not request_rows:
        raise MaterializationError("repository_has_no_requested_paths")
    requested = {str(row["rel_path"]): row for row in request_rows}
    archive_record = client.fetch(repo, commit)
    archive_path = Path(archive_record["path"])
    archive_sha256 = str(archive_record.get("sha256") or _sha256_file(archive_path))
    archive_size = int(archive_record.get("size") or archive_path.stat().st_size)
    try:
        with tarfile.open(archive_path, mode="r|gz") as bundle:
            license_name, license_file = classify_repository_archive(bundle)
    except (OSError, tarfile.TarError) as exc:
        _release_repository_archive(client, archive_record)
        raise MaterializationError("invalid_repository_archive") from exc
    if (
        license_name not in DEFAULT_REPOSITORY_LICENSE_ALLOWLIST
        or not license_file
    ):
        _release_repository_archive(client, archive_record)
        raise MaterializationError("repository_license_not_allowlisted")
    writer = _ShardWriter(
        output_root,
        source_id=source["id"],
        unit_id=unit_id,
        materializer=f"{REPOSITORY_INDEX_SCHEMA}-spool",
        revision=signature,
        target_uncompressed_bytes=target_shard_bytes,
    )
    rejected: dict[str, int] = {}
    accepted_text_bytes = 0
    retrieved_at = utc_now()
    try:
        try:
            bundle = tarfile.open(archive_path, mode="r|gz")
        except (OSError, tarfile.TarError) as exc:
            raise MaterializationError("invalid_repository_archive") from exc
        with bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                rel_path = _archive_member_relative_path(member.name)
                if rel_path is None or rel_path not in requested:
                    continue
                row = requested[rel_path]
                if member.size <= 0 or member.size > maximum_file_bytes:
                    rejected["source_blob_empty_or_too_large"] = (
                        rejected.get("source_blob_empty_or_too_large", 0) + 1
                    )
                    continue
                extracted = bundle.extractfile(member)
                if extracted is None:
                    rejected["source_blob_unavailable"] = rejected.get("source_blob_unavailable", 0) + 1
                    continue
                raw = extracted.read()
                if not raw or b"\0" in raw:
                    rejected["binary_or_empty_source_blob"] = rejected.get("binary_or_empty_source_blob", 0) + 1
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    rejected["non_utf8_source_blob"] = rejected.get("non_utf8_source_blob", 0) + 1
                    continue
                metadata = {
                    "repository": repo,
                    "repository_url": f"https://github.com/{repo}",
                    "repo_path": rel_path,
                    "path": rel_path,
                    "commit_id": commit,
                    "commit_reference_kind": "full" if len(commit) == 40 else "abbreviated",
                    "license": license_name,
                    "license_file": license_file,
                    "license_basis": "repository-root-license-file-at-pinned-archive",
                    "index_license": row.get("index_license"),
                    "language": row.get("language") or _language_for_path(Path(rel_path)),
                    "generated_file_probability": 0.0,
                    "canonical_url": f"https://github.com/{repo}/blob/{commit}/{quote(rel_path)}",
                    "retrieval_date": retrieved_at,
                    "source_archive_url": archive_record.get("url"),
                    "source_archive_sha256": archive_sha256,
                    "source_archive_size": archive_size,
                    "source_content_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_index_component": row.get("metadata_component"),
                    "source_index_unit": row.get("metadata_unit"),
                }
                reason = code_hygiene_reason(text, {**metadata, "category": "code"})
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                writer.write(
                    {
                        "id": _stable_id(source["id"], repo, commit, rel_path),
                        "text": text,
                        "metadata": metadata,
                    }
                )
                accepted_text_bytes += len(text.encode("utf-8"))
        outputs, records = writer.finish()
    except BaseException:
        writer.abort()
        _release_repository_archive(client, archive_record)
        raise
    if records == 0:
        _release_repository_archive(client, archive_record)
        raise MaterializationError("repository_archive_no_eligible_requested_files")
    # Keep restart accounting exact when one row group spans multiple shards:
    # the unit total belongs on only one output, not every output.
    content_manifest_sha256 = hashlib.sha256(
        "\n".join(str(output["sha256"]) for output in outputs).encode("ascii")
    ).hexdigest()
    for output_index, output in enumerate(outputs):
        output["kind"] = "repository_spool"
        output["ready_for_training_build"] = False
        output["text_bytes"] = accepted_text_bytes if output_index == 0 else 0
        output["candidate_token_estimate"] = (
            accepted_text_bytes // 4 if output_index == 0 else 0
        )
        output["candidate_estimator"] = "accepted_utf8_text_bytes_divided_by_4"
        output["repository_archive_sha256"] = archive_sha256
        output["repository_archive_size"] = archive_size
        output["repository"] = repo
        output["commit_id"] = commit
        output["license"] = license_name
        output["license_file"] = license_file
        output["repository_content_manifest_sha256"] = content_manifest_sha256
    report = {
        "records": records,
        "requested_paths": len(requested),
        "text_bytes": accepted_text_bytes,
        "rejected": rejected,
        "repository": repo,
        "commit_id": commit,
        "archive_url": archive_record.get("url"),
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "license": license_name,
        "license_file": license_file,
        "content_manifest_sha256": content_manifest_sha256,
    }
    _commit_unit(output_root, unit_id, signature, outputs, report)
    _release_repository_archive(client, archive_record)
    return outputs, accepted_text_bytes, report


def _iter_spool_records(path: Path) -> Iterator[dict[str, Any]]:
    raw = path.open("rb")
    with io.TextIOWrapper(
        zstd.ZstdDecompressor().stream_reader(raw),
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise MaterializationError(f"Invalid repository spool record in {path}")
            yield row


def _delete_repository_spool(
    spool_root: Path,
    unit_id: str,
    spool_outputs: Iterable[dict[str, Any]],
) -> None:
    for output in spool_outputs:
        Path(str(output["local_path"])).unlink(missing_ok=True)
    _marker_path(spool_root, unit_id).unlink(missing_ok=True)


class _RepositoryOutputBatch:
    """Compacts checksum-verified per-repository spools into durable shards."""

    def __init__(
        self,
        *,
        source: dict[str, Any],
        output_root: Path,
        spool_root: Path,
        sequence: int,
        target_shard_bytes: int,
        maximum_repositories: int,
    ) -> None:
        self.source = source
        self.output_root = output_root
        self.spool_root = spool_root
        self.sequence = sequence
        self.batch_id = f"repository-batch-{sequence:06d}"
        self.target_shard_bytes = target_shard_bytes
        self.maximum_repositories = maximum_repositories
        self.entries: list[dict[str, Any]] = []
        self.text_bytes = 0

    def add(
        self,
        *,
        repo: str,
        commit: str,
        request_signature: str,
        unit_id: str,
        spool_outputs: list[dict[str, Any]],
        text_bytes: int,
        report: dict[str, Any],
    ) -> None:
        self.entries.append(
            {
                "repo": repo,
                "commit": commit,
                "request_signature": request_signature,
                "unit_id": unit_id,
                "spool_outputs": spool_outputs,
                "text_bytes": int(text_bytes),
                "report": report,
            }
        )
        self.text_bytes += int(text_bytes)

    def should_commit(self, *, target_text_bytes: int, committed_text_bytes: int) -> bool:
        return bool(
            self.entries
            and (
                self.text_bytes >= self.target_shard_bytes
                or len(self.entries) >= self.maximum_repositories
                or (
                    target_text_bytes
                    and committed_text_bytes + self.text_bytes >= target_text_bytes
                )
            )
        )

    def _signature(self) -> str:
        digest = hashlib.sha256()
        for entry in self.entries:
            digest.update(str(entry["repo"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(entry["commit"]).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(entry["request_signature"]).encode("ascii"))
            digest.update(b"\0")
            for output in entry["spool_outputs"]:
                digest.update(str(output["sha256"]).encode("ascii"))
                digest.update(b"\0")
        return _stable_id(
            f"{REPOSITORY_INDEX_SCHEMA}-aggregate/v1",
            self.source["id"],
            self.sequence,
            len(self.entries),
            self.text_bytes,
            digest.hexdigest(),
        )

    def commit(self) -> tuple[list[dict[str, Any]], str]:
        if not self.entries:
            return [], ""
        signature = self._signature()
        completed = _load_completed_unit(self.output_root, self.batch_id, signature)
        if completed is None:
            writer = _ShardWriter(
                self.output_root,
                source_id=self.source["id"],
                unit_id=self.batch_id,
                materializer=f"{REPOSITORY_INDEX_SCHEMA}-aggregate/v1",
                revision=signature,
                target_uncompressed_bytes=self.target_shard_bytes,
            )
            try:
                for entry in self.entries:
                    for spool in entry["spool_outputs"]:
                        for row in _iter_spool_records(Path(str(spool["local_path"]))):
                            writer.write(row)
                outputs, records = writer.finish()
            except BaseException:
                writer.abort()
                raise
            if not outputs or records == 0:
                raise MaterializationError(
                    f"{self.batch_id}: repository aggregation produced no records"
                )
            for output_index, output in enumerate(outputs):
                output["text_bytes"] = self.text_bytes if output_index == 0 else 0
                output["candidate_token_estimate"] = (
                    self.text_bytes // 4 if output_index == 0 else 0
                )
                output["candidate_estimator"] = "accepted_utf8_text_bytes_divided_by_4"
                output["batch_id"] = self.batch_id
                output["repository_count"] = len(self.entries)
            _commit_unit(
                self.output_root,
                self.batch_id,
                signature,
                outputs,
                {
                    "records": records,
                    "repositories": len(self.entries),
                    "text_bytes": self.text_bytes,
                },
            )
            completed = outputs
        return completed, signature

    def cleanup_spools(self) -> None:
        for entry in self.entries:
            _delete_repository_spool(
                self.spool_root,
                str(entry["unit_id"]),
                entry["spool_outputs"],
            )


def _cleanup_committed_repository_spools(
    connection: sqlite3.Connection,
    spool_root: Path,
) -> None:
    marker_root = spool_root / ".markers"
    if not marker_root.exists():
        return
    for marker in marker_root.glob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            report = payload.get("report") or {}
            repo = str(report.get("repository") or "")
            commit = str(report.get("commit_id") or "")
            state = connection.execute(
                "SELECT status FROM repository_state WHERE repo=? AND commit_id=?",
                (repo, commit),
            ).fetchone()
            if not state or str(state[0]) != "accepted":
                continue
            for output in payload.get("outputs", []):
                Path(str(output["local_path"])).unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Do not erase an unparseable marker.  The regular checksum path
            # will fail closed if that repository ever needs to resume.
            continue


def _materialize_repository_index(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path, source: dict[str, Any]
) -> list[dict[str, Any]]:
    output_root = root / "raw" / source["id"] / "materialized"
    cache_root = root / "raw" / source["id"] / "repository-index-cache"
    spool_root = cache_root / "spool"
    runtime = profile.get("runtime", {})
    acquisition = profile.get("acquisition", {})
    spool_shard_bytes = int(acquisition.get("materializer_shard_bytes", 512_000_000))
    output_shard_bytes = max(
        1_000_000,
        int(acquisition.get("repository_output_shard_bytes", 4_000_000_000)),
    )
    maximum_repositories_per_batch = max(
        1,
        int(acquisition.get("repository_max_repositories_per_batch", 8_192)),
    )
    maximum_file_bytes = int(acquisition.get("maximum_repository_file_bytes", 2_000_000))
    bytes_per_token = float(acquisition.get("repository_code_bytes_per_token", 3.5))
    repository_workers = max(
        1,
        min(
            16,
            int(acquisition.get("repository_index_workers", acquisition.get("max_workers", 8))),
        ),
    )
    target_text_bytes = int(int(item.get("candidate_tokens", 0)) * bytes_per_token)
    outputs: list[dict[str, Any]] = []
    accepted_text_bytes = 0
    connection = _repository_index_connection(cache_root / "requests.sqlite3")
    rejected_repositories: dict[str, int] = {}
    accepted_repositories = 0
    resumed_repositories = 0
    try:
        index_report = _ingest_repository_metadata(
            connection,
            source=source,
            metadata_files=_metadata_files(root, source),
        )
        missing_batch_states = int(
            connection.execute(
                "SELECT COUNT(*) FROM repository_state s "
                "LEFT JOIN output_batches b ON b.batch_id=s.batch_id "
                "WHERE s.status='accepted' AND b.batch_id IS NULL"
            ).fetchone()[0]
        )
        if missing_batch_states:
            raise MaterializationError(
                f"{source['id']}: {missing_batch_states} accepted repository states are not attached "
                "to a checksum-verified output batch"
            )
        next_batch_sequence = 0
        for batch_id, sequence, signature, text_bytes, repositories in connection.execute(
            "SELECT batch_id, sequence, signature, accepted_text_bytes, repositories "
            "FROM output_batches ORDER BY sequence"
        ):
            completed = _load_completed_unit(output_root, str(batch_id), str(signature))
            if completed is None:
                raise MaterializationError(
                    f"{source['id']}: committed output batch marker is missing: {batch_id}"
                )
            outputs.extend(completed)
            accepted_text_bytes += int(text_bytes)
            accepted_repositories += int(repositories)
            resumed_repositories += int(repositories)
            next_batch_sequence = max(next_batch_sequence, int(sequence) + 1)
        _cleanup_committed_repository_spools(connection, spool_root)

        work: list[tuple[str, str, str, str, list[dict[str, Any]]]] = []
        current_batch = _RepositoryOutputBatch(
            source=source,
            output_root=output_root,
            spool_root=spool_root,
            sequence=next_batch_sequence,
            target_shard_bytes=output_shard_bytes,
            maximum_repositories=maximum_repositories_per_batch,
        )

        def candidate_target_met() -> bool:
            return bool(target_text_bytes and accepted_text_bytes >= target_text_bytes)

        def commit_current_batch() -> None:
            nonlocal accepted_text_bytes, accepted_repositories, resumed_repositories
            nonlocal next_batch_sequence, current_batch
            if not current_batch.entries:
                return
            batch_outputs, batch_signature = current_batch.commit()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO output_batches("
                    "batch_id, sequence, signature, accepted_text_bytes, repositories, completed_at"
                    ") VALUES(?,?,?,?,?,?)",
                    (
                        current_batch.batch_id,
                        current_batch.sequence,
                        batch_signature,
                        current_batch.text_bytes,
                        len(current_batch.entries),
                        utc_now(),
                    ),
                )
                for entry in current_batch.entries:
                    report = entry["report"]
                    connection.execute(
                        "INSERT INTO repository_state("
                        "repo, commit_id, request_signature, status, reason, archive_sha256, archive_size, "
                        "license, license_file, content_manifest_sha256, accepted_text_bytes, batch_id, completed_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(repo, commit_id) DO UPDATE SET "
                        "request_signature=excluded.request_signature, status=excluded.status, reason=NULL, "
                        "archive_sha256=excluded.archive_sha256, archive_size=excluded.archive_size, "
                        "license=excluded.license, license_file=excluded.license_file, "
                        "content_manifest_sha256=excluded.content_manifest_sha256, "
                        "accepted_text_bytes=excluded.accepted_text_bytes, batch_id=excluded.batch_id, "
                        "completed_at=excluded.completed_at",
                        (
                            entry["repo"],
                            entry["commit"],
                            entry["request_signature"],
                            "accepted",
                            None,
                            report.get("archive_sha256"),
                            report.get("archive_size"),
                            report.get("license"),
                            report.get("license_file"),
                            report.get("content_manifest_sha256"),
                            entry["text_bytes"],
                            current_batch.batch_id,
                            utc_now(),
                        ),
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            outputs.extend(batch_outputs)
            accepted_text_bytes += current_batch.text_bytes
            accepted_repositories += len(current_batch.entries)
            resumed_repositories += sum(
                int(bool(entry["report"].get("resumed")))
                for entry in current_batch.entries
            )
            current_batch.cleanup_spools()
            next_batch_sequence += 1
            current_batch = _RepositoryOutputBatch(
                source=source,
                output_root=output_root,
                spool_root=spool_root,
                sequence=next_batch_sequence,
                target_shard_bytes=output_shard_bytes,
                maximum_repositories=maximum_repositories_per_batch,
            )

        def process_batch(
            executor: ThreadPoolExecutor,
            batch: list[tuple[str, str, str, str, list[dict[str, Any]]]],
        ) -> None:
            futures: list[tuple[tuple[str, str, str, str, list[dict[str, Any]]], Future[Any]]] = []
            for request in batch:
                repo, commit, signature, unit_id, request_rows = request
                client = _CodeloadArchiveClient(
                    cache_root / "archives",
                    retries=int(runtime.get("download_retries", 6)),
                    timeout=int(runtime.get("request_timeout_seconds", 900)),
                    retain_archives=bool(acquisition.get("retain_repository_archives", False)),
                )
                future = executor.submit(
                    _materialize_repository_unit,
                    source=source,
                    repo=repo,
                    commit=commit,
                    requests_for_repo=request_rows,
                    unit_id=unit_id,
                    signature=signature,
                    output_root=spool_root,
                    client=client,
                    target_shard_bytes=spool_shard_bytes,
                    maximum_file_bytes=maximum_file_bytes,
                )
                futures.append((request, future))
            transient: TransientMaterializationError | None = None
            for request, future in futures:
                repo, commit, signature, _unit_id, _request_rows = request
                try:
                    spool_outputs, text_bytes, unit_report = future.result()
                except TransientMaterializationError as exc:
                    if not candidate_target_met():
                        transient = transient or exc
                    continue
                except (MaterializationError, OSError, tarfile.TarError) as exc:
                    reason = str(exc) or type(exc).__name__
                    connection.execute(
                        "INSERT INTO repository_state("
                        "repo, commit_id, request_signature, status, reason, accepted_text_bytes, completed_at"
                        ") VALUES(?,?,?,?,?,?,?) "
                        "ON CONFLICT(repo, commit_id) DO UPDATE SET "
                        "request_signature=excluded.request_signature, status=excluded.status, "
                        "reason=excluded.reason, accepted_text_bytes=0, batch_id=NULL, "
                        "completed_at=excluded.completed_at",
                        (repo, commit, signature, "rejected", reason, 0, utc_now()),
                    )
                    rejected_repositories[reason] = rejected_repositories.get(reason, 0) + 1
                    continue
                if candidate_target_met():
                    _delete_repository_spool(spool_root, _unit_id, spool_outputs)
                    continue
                current_batch.add(
                    repo=repo,
                    commit=commit,
                    request_signature=signature,
                    unit_id=_unit_id,
                    spool_outputs=spool_outputs,
                    text_bytes=text_bytes,
                    report=unit_report,
                )
                if current_batch.should_commit(
                    target_text_bytes=target_text_bytes,
                    committed_text_bytes=accepted_text_bytes,
                ):
                    commit_current_batch()
            if transient is not None:
                raise transient

        with ThreadPoolExecutor(
            max_workers=repository_workers,
            thread_name_prefix="metis-repository-index",
        ) as executor:
            repository_rows = connection.execute(
                "SELECT repo_key, repo, commit_id FROM repo_commits ORDER BY rank_key, repo, commit_id"
            )
            for repo_key, repo, commit in repository_rows:
                if candidate_target_met():
                    break
                repo = str(repo)
                commit = str(commit)
                request_rows, signature = _repository_requests(connection, bytes(repo_key), repo, commit)
                state = connection.execute(
                    "SELECT request_signature, status, reason, batch_id FROM repository_state "
                    "WHERE repo=? AND commit_id=?",
                    (repo, commit),
                ).fetchone()
                if state and str(state[0]) != signature:
                    raise MaterializationError(
                        f"{source['id']}: grouped request drift for {repo}@{commit}; "
                        "preserve the old index for audit and rebuild"
                    )
                if state and str(state[1]) == "rejected":
                    reason = str(state[2] or "repository_rejected")
                    rejected_repositories[reason] = rejected_repositories.get(reason, 0) + 1
                    continue
                if state and str(state[1]) == "accepted":
                    if not state[3]:
                        raise MaterializationError(
                            f"{source['id']}: accepted repository has no output batch: {repo}@{commit}"
                        )
                    continue
                work.append(
                    (
                        repo,
                        commit,
                        signature,
                        f"repo-{_stable_id(repo, commit)[:24]}",
                        request_rows,
                    )
                )
                if len(work) >= repository_workers:
                    process_batch(executor, work)
                    work = []
            if work and not candidate_target_met():
                process_batch(executor, work)
        if current_batch.entries:
            commit_current_batch()
    finally:
        connection.close()
    report = {
        "schema": "metis.repository-index-materialization/v2",
        "source_id": source["id"],
        "completed_at": utc_now(),
        "request_index": index_report,
        "candidate_token_target": int(item.get("candidate_tokens", 0)),
        "bytes_per_token_estimator": bytes_per_token,
        "candidate_text_byte_target": target_text_bytes,
        "accepted_text_bytes": accepted_text_bytes,
        "estimated_candidate_tokens": int(accepted_text_bytes / bytes_per_token) if bytes_per_token else 0,
        "candidate_target_met": not target_text_bytes or accepted_text_bytes >= target_text_bytes,
        "accepted_repositories": accepted_repositories,
        "resumed_repositories": resumed_repositories,
        "rejected_repositories": rejected_repositories,
        "repository_workers": repository_workers,
        "output_shard_target_bytes": output_shard_bytes,
        "maximum_repositories_per_batch": maximum_repositories_per_batch,
        "durable_output_files": len(outputs),
        "archive_transport": "public-codeload-no-github-api",
        "repository_archives_retained": bool(acquisition.get("retain_repository_archives", False)),
    }
    atomic_json(output_root / "MATERIALIZATION_REPORT.json", report)
    if not outputs:
        raise MaterializationError(f"{source['id']}: repository-index hydration produced no licensed source files")
    if target_text_bytes and accepted_text_bytes < target_text_bytes:
        raise MaterializationError(
            f"{source['id']}: repository-index metadata exhausted at {accepted_text_bytes:,} accepted text bytes "
            f"(~{int(accepted_text_bytes / bytes_per_token):,} estimated tokens), below the "
            f"{target_text_bytes:,}-byte / {int(item.get('candidate_tokens', 0)):,}-token candidate target; "
            "expand the pinned metadata or revise the reviewed byte/token estimator"
        )
    for output_index, output in enumerate(outputs):
        output["candidate_token_estimate"] = (
            int(accepted_text_bytes / bytes_per_token) if output_index == 0 else 0
        )
        output["candidate_estimator"] = "accepted_code_bytes_divided_by_profile_ratio"
    return outputs


def run_production_materializer(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    driver = str(item.get("driver", ""))
    if driver not in SUPPORTED_MATERIALIZER_DRIVERS:
        raise MaterializationError(f"No production materializer is registered for driver {driver!r}")
    manifest, source = _source_for_item(item, profile)
    if driver == "canonical_git":
        return _materialize_canonical_git(item, profile=profile, root=root, manifest=manifest, source=source)
    if driver in {"canonical_web", "canonical_http"}:
        return _materialize_canonical_web_http(item, profile=profile, root=root, source=source)
    if driver == "repository_index":
        return _materialize_repository_index(item, profile=profile, root=root, source=source)
    raise AssertionError(driver)
