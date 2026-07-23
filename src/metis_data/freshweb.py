"""Bounded, restartable materialization of a curated Common Crawl slice.

This module deliberately has no scheduler or manifest integration.  The public
``materialize_freshweb`` function accepts the source-lock item already used by
the acquisition layer and writes an independently auditable raw-data release.

The implementation keeps network selection cheap and exact:

* query the bulk Parquet URL Index rather than the rate-limited CDX service;
* apply the live Common Crawl opt-out registry before retaining coordinates;
* deterministically bound every Parquet partition;
* resolve canonical-URL and payload-digest duplicates before WARC retrieval;
* coalesce nearby byte ranges and validate every returned WARC member; and
* append independent zstd frames to a fixed number of JSONL shards, with
  committed offsets in SQLite so an interrupted append can be rolled back.

It intentionally does not perform semantic deduplication.  Global normalized
exact/MinHash deduplication and final-token accounting remain downstream jobs.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import heapq
import io
import ipaddress
import json
import math
import os
import random
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .state import atomic_json, utc_now


FRESHWEB_SCHEMA = "metis.freshweb-materialization/v2"
FRESHWEB_PROGRESS_SCHEMA = "metis.freshweb-progress/v1"
URL_INDEX_INTEGRITY_SCHEMA = "metis.common-crawl-url-index-cache/v1"
EXTRACTOR_VERSION = "metis-warc-html-v3"
OPT_OUT_PARSER_VERSION = "metis-common-crawl-opt-out-v2"
MAX_DECOMPRESSED_RECORD_BYTES = 64 * 1024 * 1024
MAX_DECODED_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
DEFAULT_DATA_ROOT = "https://data.commoncrawl.org/"
DEFAULT_OPT_OUT_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1uavIZ-Y2ew-Vj7d0_pY67SGD2jcUCsZZ5rPy1qEyPzg/export?format=csv&gid=0"
)

INDEX_COLUMNS = (
    "url",
    "url_surtkey",
    "url_host_name",
    "url_host_registered_domain",
    "url_host_private_domain",
    "url_host_registry_suffix",
    "url_protocol",
    "url_path",
    "url_query",
    "fetch_time",
    "fetch_status",
    "fetch_redirect",
    "content_digest",
    "content_mime_type",
    "content_mime_detected",
    "content_charset",
    "content_languages",
    "content_truncated",
    "warc_record_id",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "warc_segment",
)

REQUIRED_INDEX_COLUMNS = {
    "url",
    "fetch_time",
    "fetch_status",
    "content_digest",
    "content_mime_type",
    "content_truncated",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
}

TRACKING_QUERY_KEYS = {
    "_ga",
    "_gl",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

REJECTED_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:"
    r"account|accounts|auth|cart|checkout|comments?|feed|login|logout|register|"
    r"search|signin|signup|sitemap|tag|tags|wp-admin"
    r")(?:/|$)|"
    r"(?:^|[?&])(?:page|paged|replytocom|session|sid)=|"
    r"\.(?:7z|apk|avi|bin|bmp|css|dmg|docx?|epub|exe|gif|gz|ico|iso|jar|jpe?g|"
    r"m4a|mkv|mov|mp3|mp4|mpeg|ogg|pdf|png|pptx?|rar|svg|tar|tiff?|webm|webp|"
    r"woff2?|xlsx?|xml|xz|zip)(?:$|[?#])",
    re.IGNORECASE,
)

DATE_PATH_PATTERN = re.compile(r"/(?:20(?:2[5-9]|[3-9][0-9]))/(?:0?[1-9]|1[0-2])(?:/|$)")

DEFAULT_CATEGORY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("official_docs", 0.20),
    ("government", 0.15),
    ("education", 0.15),
    ("technical", 0.15),
    ("science", 0.10),
    ("software", 0.10),
    ("reporting", 0.10),
    ("general", 0.05),
)

CATEGORY_PRIORITY = {
    "official_docs": 100,
    "government": 98,
    "education": 96,
    "science": 94,
    "software": 92,
    "technical": 90,
    "reporting": 88,
    "general": 70,
}

ROUTE_ALLOWED_CATEGORIES: dict[str, frozenset[str]] = {
    "general_web": frozenset(CATEGORY_PRIORITY),
    "software_docs": frozenset({"official_docs", "software", "technical"}),
    "fresh_science": frozenset({"science", "education", "government", "reporting"}),
    "official_docs": frozenset(
        {"official_docs", "government", "education", "science", "technical"}
    ),
}


@dataclass(frozen=True)
class FreshWebOptions:
    """Operational bounds and endpoints for ``materialize_freshweb``.

    ``max_records_per_partition`` is primarily a site-safety override.  When
    omitted, the cap is calculated from the source's candidate-token target,
    the number of URL-index partitions, and ``estimated_tokens_per_document``.
    """

    collinfo_url: str = DEFAULT_COLLINFO_URL
    data_root: str = DEFAULT_DATA_ROOT
    opt_out_csv_url: str = DEFAULT_OPT_OUT_CSV_URL
    max_records_per_partition: int | None = None
    route: str = "general_web"
    allowed_categories: tuple[str, ...] = tuple(CATEGORY_PRIORITY)
    estimated_tokens_per_document: int = 1_200
    selection_oversample: float = 1.50
    category_weights: tuple[tuple[str, float], ...] = DEFAULT_CATEGORY_WEIGHTS
    require_english: bool = True
    require_reusable_open_license: bool = False
    minimum_characters: int = 500
    shard_count: int = 128
    max_workers: int = 10
    request_timeout_seconds: float = 120.0
    max_retries: int = 8
    retry_base_seconds: float = 1.0
    coalesce_gap_bytes: int = 64 * 1024
    maximum_span_bytes: int = 8 * 1024 * 1024
    parquet_batch_rows: int = 65_536
    keep_index_files: bool = False
    user_agent: str = "MetisData/1.6 (+https://github.com/lernex-ai)"
    seed: str = "metis-freshweb-2026-v1"
    allow_domains: tuple[str, ...] = ()
    deny_domains: tuple[str, ...] = ()
    freshness_cutoff_start: str | None = None
    freshness_cutoff_end: str | None = None

    def validate(self) -> None:
        if self.max_records_per_partition is not None and self.max_records_per_partition <= 0:
            raise ValueError("max_records_per_partition must be positive")
        if self.estimated_tokens_per_document <= 0:
            raise ValueError("estimated_tokens_per_document must be positive")
        if self.route not in ROUTE_ALLOWED_CATEGORIES:
            raise ValueError(f"FreshWeb route must be one of {sorted(ROUTE_ALLOWED_CATEGORIES)}")
        if (
            not self.allowed_categories
            or len(self.allowed_categories) != len(set(self.allowed_categories))
            or not set(self.allowed_categories).issubset(CATEGORY_PRIORITY)
        ):
            raise ValueError("allowed_categories must be a non-empty unique category subset")
        if set(self.allowed_categories) != ROUTE_ALLOWED_CATEGORIES[self.route]:
            raise ValueError(f"allowed_categories do not match the fail-closed contract for {self.route}")
        if not math.isfinite(self.selection_oversample) or self.selection_oversample < 1:
            raise ValueError("selection_oversample must be at least 1")
        if self.minimum_characters < 1:
            raise ValueError("minimum_characters must be positive")
        if self.shard_count < 1:
            raise ValueError("shard_count must be positive")
        if not 1 <= self.max_workers <= 10:
            raise ValueError("max_workers must be between 1 and 10 for polite Common Crawl access")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if self.request_timeout_seconds <= 0 or self.retry_base_seconds < 0:
            raise ValueError("HTTP timeout and retry bounds are invalid")
        if self.coalesce_gap_bytes < 0 or self.maximum_span_bytes < 1:
            raise ValueError("range-coalescing limits are invalid")
        if self.parquet_batch_rows < 1:
            raise ValueError("parquet_batch_rows must be positive")
        if not self.user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        invalid_domains = [
            value
            for value in (*self.allow_domains, *self.deny_domains)
            if not _normalise_host(value)
        ]
        if invalid_domains:
            raise ValueError(f"invalid configured domains: {invalid_domains}")
        if bool(self.freshness_cutoff_start) != bool(self.freshness_cutoff_end):
            raise ValueError("freshness cutoff start and end must be configured together")
        if self.freshness_cutoff_start and self.freshness_cutoff_end:
            try:
                start = datetime.strptime(self.freshness_cutoff_start, "%Y-%m-%d").date()
                end = datetime.strptime(self.freshness_cutoff_end, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError("freshness cutoffs must use YYYY-MM-DD") from exc
            if start > end:
                raise ValueError("freshness cutoff start must not be after the end")
        weights = dict(self.category_weights)
        if len(self.category_weights) != len(CATEGORY_PRIORITY) or set(weights) != set(CATEGORY_PRIORITY):
            raise ValueError(f"category_weights must cover {sorted(CATEGORY_PRIORITY)}")
        if (
            any(not math.isfinite(value) or value < 0 for value in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError("category weights must be non-negative with a positive total")
        if any(weights[category] > 0 for category in CATEGORY_PRIORITY if category not in self.allowed_categories):
            raise ValueError("disallowed FreshWeb categories must have zero selection weight")
        if any(weights[category] <= 0 for category in self.allowed_categories):
            raise ValueError("every allowed FreshWeb category must have positive selection weight")


@dataclass(frozen=True)
class ResolvedIndexPartition:
    crawl: str
    relative_path: str
    url: str
    listing_sha256: str


@dataclass(frozen=True)
class OptOutUrlRule:
    host: str
    path: str
    query: str | None
    path_prefix: bool = False
    query_prefix: bool = False


@dataclass(frozen=True)
class OptOutPolicy:
    domains: frozenset[str]
    url_paths: frozenset[tuple[str, str]]
    snapshot_sha256: str
    last_updated: str | None
    url_rules: tuple[OptOutUrlRule, ...] = ()
    input_entries: int = 0
    unparsed_entries: int = 0

    def reason(self, url: str) -> str | None:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return "invalid_url"
        host = _normalise_host(parsed.hostname or "")
        if not host:
            return "invalid_url"
        labels = host.split(".")
        for index in range(len(labels)):
            if ".".join(labels[index:]) in self.domains:
                return "common_crawl_opt_out_domain"
        path = _normalise_path(parsed.path)
        if (host, path) in self.url_paths:
            return "common_crawl_opt_out_url"
        query = _normalise_query(parsed.query)
        for rule in self.url_rules:
            if host != rule.host and not host.endswith("." + rule.host):
                continue
            if rule.path_prefix:
                if not path.startswith(rule.path):
                    continue
            elif path != rule.path and not path.startswith(rule.path.rstrip("/") + "/"):
                continue
            if rule.query is None:
                return "common_crawl_opt_out_url"
            if rule.query_prefix and query.startswith(rule.query):
                return "common_crawl_opt_out_url"
            if not rule.query_prefix and query == rule.query:
                return "common_crawl_opt_out_url"
        return None


@dataclass(frozen=True)
class RangeSpan:
    start: int
    end: int
    records: tuple[dict[str, Any], ...]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(order=True)
class _HeapCandidate:
    key: tuple[int, str, str, int, str, int]
    candidate: dict[str, Any] = field(compare=False)


class FreshWebError(RuntimeError):
    """Base class for fail-closed FreshWeb acquisition failures."""


class PermanentHttpError(FreshWebError):
    """An HTTP response which should stop rather than be retried."""


def _normalise_host(host: str) -> str:
    host = host.strip().strip(".").lower()
    if not host:
        return ""
    try:
        normalized = host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    try:
        ipaddress.ip_address(normalized.strip("[]"))
        return normalized.strip("[]")
    except ValueError:
        pass
    if len(normalized) > 253:
        return ""
    labels = normalized.split(".")
    if len(labels) < 2 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not re.fullmatch(r"[a-z0-9-]+", label)
        for label in labels
    ):
        return ""
    return normalized


def _normalise_path(path: str) -> str:
    path = re.sub(r"/{2,}", "/", path or "/")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/") or "/"
    return path


def _normalise_query(query: str) -> str:
    if not query:
        return ""
    try:
        values = parse_qsl(query, keep_blank_values=True, strict_parsing=False)
    except ValueError:
        return query
    values.sort()
    return urlencode(values, doseq=True)


def canonicalize_url(url: str) -> str:
    """Return a conservative canonical URL suitable for exact URL dedup."""

    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("unsupported URL scheme")
    host = _normalise_host(parsed.hostname or "")
    if not host:
        raise ValueError("URL has no valid host")
    display_host = f"[{host}]" if ":" in host else host
    port = parsed.port
    if port and not ((parsed.scheme.lower() == "http" and port == 80) or (parsed.scheme.lower() == "https" and port == 443)):
        netloc = f"{display_host}:{port}"
    else:
        netloc = display_host
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS and not key.lower().startswith("utm_")
    ]
    query.sort()
    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            _normalise_path(parsed.path),
            urlencode(query, doseq=True),
            "",
        )
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _HttpClient:
    def __init__(self, options: FreshWebOptions, provided_session: Any | None = None) -> None:
        self.options = options
        self.provided_session = provided_session
        self.local = threading.local()

    def _session(self) -> Any:
        if self.provided_session is not None:
            return self.provided_session
        session = getattr(self.local, "session", None)
        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - production dependency guard
                raise RuntimeError("FreshWeb acquisition requires requests") from exc
            session = requests.Session()
            session.headers.update({"User-Agent": self.options.user_agent, "Accept-Encoding": "identity"})
            self.local.session = session
        return session

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
        expected: frozenset[int] = frozenset({200}),
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.options.max_retries):
            try:
                response = self._session().get(
                    url,
                    headers=dict(headers or {}),
                    stream=stream,
                    timeout=self.options.request_timeout_seconds,
                )
                if response.status_code in expected:
                    return response
                if response.status_code == 403:
                    response.close()
                    raise PermanentHttpError(
                        f"Common Crawl returned HTTP 403 for {url}; stop rather than bypassing or retrying"
                    )
                if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                    status = response.status_code
                    response.close()
                    raise PermanentHttpError(f"Unexpected HTTP {status} for {url}")
                retry_after = response.headers.get("Retry-After")
                response.close()
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = self.options.retry_base_seconds * (2**attempt) * random.uniform(0.5, 1.5)
                time.sleep(min(delay, 60.0))
            except PermanentHttpError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self.options.max_retries:
                    break
                delay = self.options.retry_base_seconds * (2**attempt) * random.uniform(0.5, 1.5)
                time.sleep(min(delay, 60.0))
        raise FreshWebError(f"Failed HTTP GET after {self.options.max_retries} attempts: {url}") from last_error

    def bytes(self, url: str) -> bytes:
        response = self.get(url)
        try:
            return bytes(response.content)
        finally:
            response.close()


def _opt_out_token(value: str) -> tuple[str, str | OptOutUrlRule] | None:
    """Parse one whitespace/comma-delimited registry token conservatively."""

    value = value.strip().lstrip("-*•").strip().strip("<>()[]{}\"'").rstrip(".,;")
    if not value or "@" in value:
        return None
    value = value.replace(r"\.", ".").replace(r"\?", "?")
    has_scheme = value.lower().startswith(("http://", "https://"))
    candidate = value if has_scheme else f"https://{value}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    raw_host = (parsed.hostname or "").removeprefix("*.")
    host = _normalise_host(raw_host)
    if not host:
        return None
    raw_path = parsed.path or "/"
    raw_query = parsed.query
    path_prefix = "*" in raw_path
    query_prefix = "*" in raw_query
    raw_path = raw_path.split("*", 1)[0].rstrip("$") or "/"
    raw_query = raw_query.split("*", 1)[0].rstrip("$")
    path = _normalise_path(raw_path)
    query = _normalise_query(raw_query) if parsed.query else None
    if path == "/" and query is None:
        return "domain", host
    return "url", OptOutUrlRule(
        host=host,
        path=path,
        query=query,
        path_prefix=path_prefix,
        query_prefix=query_prefix,
    )


def _opt_out_entries(value: str) -> list[tuple[str, str | OptOutUrlRule]]:
    # The official sheet contains both one-entry-per-line rows and long
    # comma/semicolon/space-delimited publisher lists.  Tokenizing first avoids
    # silently treating an entire publisher list as one malformed hostname.
    entries: list[tuple[str, str | OptOutUrlRule]] = []
    for token in re.split(r"[,;\s]+", value):
        parsed = _opt_out_token(token)
        if parsed is not None:
            entries.append(parsed)
    return entries


def _opt_out_entry(value: str) -> tuple[str, str] | None:
    """Backward-compatible single-entry parser used by older callers."""

    entries = _opt_out_entries(value)
    if len(entries) != 1:
        return None
    kind, normalized = entries[0]
    if kind == "domain":
        return kind, str(normalized)
    assert isinstance(normalized, OptOutUrlRule)
    return kind, urlunsplit(("https", normalized.host, normalized.path, normalized.query or "", ""))


def parse_opt_out_registry(payload: bytes) -> OptOutPolicy:
    """Parse the official registry CSV, failing closed on schema drift."""

    text = payload.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][:3] != ["Publisher/Requester", "Date of notice", "List of domains/URLs"]:
        raise FreshWebError("Common Crawl opt-out registry schema changed")
    domains: set[str] = set()
    url_paths: set[tuple[str, str]] = set()
    url_rules: set[OptOutUrlRule] = set()
    last_updated: str | None = None
    nonempty_entries = 0
    parsed_entries = 0
    unparsed_entries = 0
    for row in rows[1:]:
        if row and row[0].strip().upper().startswith("LAST UPDATED:"):
            last_updated = row[0].split(":", 1)[1].strip()
        if len(row) < 3:
            continue
        for line in row[2].splitlines():
            line = line.strip()
            if not line:
                continue
            nonempty_entries += 1
            entries = _opt_out_entries(line)
            if not entries:
                if line.strip().lower() not in {"n/a", "na", "none"} and not line.strip().lower().startswith(
                    "additional"
                ):
                    unparsed_entries += 1
                continue
            parsed_entries += 1
            for kind, normalized in entries:
                if kind == "domain":
                    domains.add(str(normalized))
                else:
                    assert isinstance(normalized, OptOutUrlRule)
                    url_rules.add(normalized)
                    if normalized.query is None and not normalized.path_prefix:
                        url_paths.add((normalized.host, normalized.path))
    if nonempty_entries and not parsed_entries:
        raise FreshWebError("Common Crawl opt-out registry contained no parseable entries")
    if nonempty_entries and parsed_entries / nonempty_entries < 0.95:
        raise FreshWebError(
            "Common Crawl opt-out registry parse coverage fell below 95%; refusing schema drift"
        )
    return OptOutPolicy(
        domains=frozenset(domains),
        url_paths=frozenset(url_paths),
        snapshot_sha256=_sha256_bytes(payload),
        last_updated=last_updated,
        url_rules=tuple(sorted(url_rules, key=lambda item: (item.host, item.path, item.query or ""))),
        input_entries=nonempty_entries,
        unparsed_entries=unparsed_entries,
    )


def _snapshot_opt_out(base_root: Path, http: _HttpClient, url: str) -> tuple[OptOutPolicy, Path]:
    payload = http.bytes(url)
    policy = parse_opt_out_registry(payload)
    snapshot = base_root / "compliance" / f"opt-out-{policy.snapshot_sha256}.csv"
    if not snapshot.exists():
        _atomic_bytes(snapshot, payload)
    atomic_json(
        snapshot.with_suffix(".json"),
        {
            "schema": "metis.common-crawl-opt-out/v1",
            "parser_version": OPT_OUT_PARSER_VERSION,
            "source": url,
            "retrieved_at": utc_now(),
            "sha256": policy.snapshot_sha256,
            "last_updated": policy.last_updated,
            "domains": len(policy.domains),
            "url_paths": len(policy.url_paths),
            "url_rules": len(policy.url_rules),
            "input_entries": policy.input_entries,
            "unparsed_entries": policy.unparsed_entries,
        },
    )
    return policy, snapshot


def snapshot_common_crawl_opt_out(
    root: str | Path,
    *,
    url: str = DEFAULT_OPT_OUT_CSV_URL,
    options: FreshWebOptions | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    """Atomically snapshot and parse the current Common Crawl opt-out registry.

    The returned hash-addressed CSV is suitable as a final-normalization input.
    Callers should bind the returned SHA-256 in their handoff or release state;
    this helper deliberately does not mutate pipeline stages itself.
    """

    options = options or FreshWebOptions()
    options.validate()
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = _HttpClient(options, session).bytes(url)
    policy = parse_opt_out_registry(payload)
    snapshot_path = destination / f"common-crawl-opt-out-{policy.snapshot_sha256}.csv"
    normalized_rules = {
        "schema": "metis.common-crawl-opt-out-rules/v1",
        "parser_version": OPT_OUT_PARSER_VERSION,
        "source_sha256": policy.snapshot_sha256,
        "domains": sorted(policy.domains),
        "url_paths": [
            {"host": host, "path": path} for host, path in sorted(policy.url_paths)
        ],
        "url_rules": [asdict(rule) for rule in policy.url_rules],
    }
    rules_payload = (
        json.dumps(normalized_rules, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    rules_sha256 = _sha256_bytes(rules_payload)
    rules_path = destination / f"common-crawl-opt-out-rules-{rules_sha256}.json"
    metadata_path = destination / (
        f"common-crawl-opt-out-{policy.snapshot_sha256}-{OPT_OUT_PARSER_VERSION}.json"
    )
    if snapshot_path.exists():
        if _sha256_file(snapshot_path) != policy.snapshot_sha256:
            raise FreshWebError(f"Existing opt-out snapshot checksum failed: {snapshot_path}")
    else:
        _atomic_bytes(snapshot_path, payload)
    if rules_path.exists():
        if _sha256_file(rules_path) != rules_sha256:
            raise FreshWebError(f"Existing normalized opt-out rules checksum failed: {rules_path}")
    else:
        _atomic_bytes(rules_path, rules_payload)
    metadata = {
        "schema": "metis.common-crawl-opt-out/v1",
        "source": url,
        "retrieved_at": utc_now(),
        "path": str(snapshot_path),
        "sha256": policy.snapshot_sha256,
        "parser_version": OPT_OUT_PARSER_VERSION,
        "rules_path": str(rules_path),
        "rules_sha256": rules_sha256,
        "last_updated": policy.last_updated,
        "domains": len(policy.domains),
        "url_paths": len(policy.url_paths),
        "url_rules": len(policy.url_rules),
        "input_entries": policy.input_entries,
        "unparsed_entries": policy.unparsed_entries,
    }
    if not metadata_path.exists():
        atomic_json(metadata_path, metadata)
    else:
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FreshWebError(
                f"Existing opt-out snapshot metadata is unreadable: {metadata_path}"
            ) from exc
        if (
            existing.get("sha256") != policy.snapshot_sha256
            or existing.get("path") != str(snapshot_path)
            or existing.get("parser_version") != OPT_OUT_PARSER_VERSION
            or existing.get("rules_path") != str(rules_path)
            or existing.get("rules_sha256") != rules_sha256
        ):
            raise FreshWebError(f"Existing opt-out snapshot metadata is invalid: {metadata_path}")
        metadata = existing
    pointer = {**metadata, "metadata_path": str(metadata_path), "checked_at": utc_now()}
    atomic_json(destination / "LATEST_COMMON_CRAWL_OPT_OUT.json", pointer)
    return pointer


def resolve_common_crawl_paths(
    crawls: Sequence[str],
    *,
    http: _HttpClient,
    data_root: str,
    collinfo_url: str,
) -> tuple[list[ResolvedIndexPartition], dict[str, Any], dict[str, bytes]]:
    """Resolve and validate the WARC-subset Parquet listings for each crawl."""

    try:
        collections = json.loads(http.bytes(collinfo_url))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FreshWebError("Invalid Common Crawl collection manifest") from exc
    by_id = {entry.get("id"): entry for entry in collections if isinstance(entry, dict) and entry.get("id")}
    missing = [crawl for crawl in crawls if crawl not in by_id]
    if missing:
        raise FreshWebError(f"Common Crawl releases are unavailable: {missing}")
    partitions: list[ResolvedIndexPartition] = []
    listings: dict[str, bytes] = {}
    for crawl in crawls:
        listing_url = urljoin(data_root.rstrip("/") + "/", f"crawl-data/{crawl}/cc-index-table.paths.gz")
        compressed = http.bytes(listing_url)
        try:
            listing = gzip.decompress(compressed).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise FreshWebError(f"Invalid URL-index path listing for {crawl}") from exc
        listing_sha = _sha256_bytes(compressed)
        listings[crawl] = compressed
        relative_paths = sorted(
            line.strip()
            for line in listing.splitlines()
            if f"crawl={crawl}/subset=warc/" in line
        )
        if not relative_paths:
            raise FreshWebError(f"No subset=warc URL-index partitions were listed for {crawl}")
        if len(relative_paths) != len(set(relative_paths)):
            raise FreshWebError(f"Duplicate URL-index paths were listed for {crawl}")
        for relative in relative_paths:
            expected_prefix = f"cc-index/table/cc-main/warc/crawl={crawl}/subset=warc/"
            if (
                relative.startswith("/")
                or ".." in Path(relative).parts
                or not relative.startswith(expected_prefix)
                or not relative.endswith(".parquet")
            ):
                raise FreshWebError(f"Unsafe Common Crawl path in {crawl}: {relative}")
            partitions.append(
                ResolvedIndexPartition(
                    crawl=crawl,
                    relative_path=relative,
                    url=urljoin(data_root.rstrip("/") + "/", relative),
                    listing_sha256=listing_sha,
                )
            )
    resolved = {
        "collinfo_url": collinfo_url,
        "collections": [by_id[crawl] for crawl in crawls],
        "partition_count": len(partitions),
        "listing_sha256": {crawl: _sha256_bytes(payload) for crawl, payload in listings.items()},
    }
    return partitions, resolved, listings


def _matches_domain(host: str, domains: Sequence[str]) -> bool:
    normalized = _normalise_host(host)
    for domain in domains:
        normalized_domain = _normalise_host(domain)
        if normalized_domain and (
            normalized == normalized_domain or normalized.endswith("." + normalized_domain)
        ):
            return True
    return False


def _fresh_category(row: Mapping[str, Any], canonical_url: str) -> str:
    parsed = urlsplit(canonical_url)
    host = parsed.hostname or ""
    path = parsed.path.lower()
    registry_suffix = str(row.get("url_host_registry_suffix") or "").lower()
    host_tokens = set(re.split(r"[.\-_]", host))
    path_tokens = set(filter(None, re.split(r"[/._\-]", path)))
    all_tokens = host_tokens | path_tokens
    if (
        registry_suffix in {"gov", "gov.uk", "gc.ca", "gouv.fr"}
        or host.endswith(".gov")
        or ".gov." in host
    ):
        return "government"
    if registry_suffix in {"edu", "ac.uk", "edu.au", "ac.jp", "edu.cn"} or host.endswith(".edu"):
        return "education"
    if all_tokens & {"docs", "documentation", "developer", "developers", "api", "apis", "reference", "manual"}:
        return "official_docs"
    if all_tokens & {"standards", "standard", "spec", "specification", "rfc"}:
        return "official_docs"
    if all_tokens & {"research", "science", "scientific", "laboratory", "journal", "preprint", "publication"}:
        return "science"
    if all_tokens & {"changelog", "release", "releases", "engineering", "software", "package", "migration"}:
        return "software"
    if "blog" in all_tokens or all_tokens & {"technical", "technology", "architecture"}:
        return "technical"
    if "news" in all_tokens or DATE_PATH_PATTERN.search(path):
        return "reporting"
    return "general"


def _timestamp(value: Any) -> tuple[str, int]:
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("invalid fetch_time") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return moment.isoformat(), int(moment.timestamp() * 1_000_000)


def _record_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)


def metadata_candidate(
    row: Mapping[str, Any],
    *,
    crawl: str,
    source_id: str,
    policy: OptOutPolicy,
    options: FreshWebOptions,
) -> tuple[dict[str, Any] | None, str | None]:
    """Apply fail-closed metadata gates and return a normalized coordinate."""

    try:
        if int(row.get("fetch_status") or 0) != 200:
            return None, "http_status"
        if row.get("content_truncated") not in (None, "", False, 0):
            return None, "content_truncated"
        digest = str(row.get("content_digest") or "").lower().removeprefix("sha1:")
        if not digest or not re.fullmatch(r"[a-z2-7]{32}", digest):
            return None, "missing_or_invalid_digest"
        sent_mime = str(row.get("content_mime_type") or "").lower().split(";", 1)[0].strip()
        detected_mime = str(row.get("content_mime_detected") or "").lower().split(";", 1)[0].strip()
        accepted_mimes = {"text/html", "application/xhtml+xml", "text/plain"}
        if sent_mime not in accepted_mimes and detected_mime not in accepted_mimes:
            return None, "mime"
        languages = [part.strip().lower() for part in str(row.get("content_languages") or "").split(",") if part.strip()]
        if options.require_english and "eng" not in languages:
            return None, "language"
        original_url = str(row.get("url") or "")
        canonical_url = canonicalize_url(original_url)
        parsed = urlsplit(canonical_url)
        host = _normalise_host(parsed.hostname or "")
        if options.allow_domains and not _matches_domain(host, options.allow_domains):
            return None, "domain_not_allowed"
        if options.deny_domains and _matches_domain(host, options.deny_domains):
            return None, "domain_denied"
        # Compliance matching uses the captured URL, before training-oriented
        # canonicalization removes tracking-style query keys.
        opt_out_reason = policy.reason(original_url)
        if opt_out_reason:
            return None, opt_out_reason
        if len(canonical_url) > 2_048 or REJECTED_PATH_PATTERN.search(canonical_url):
            return None, "url_pattern"
        offset = int(row.get("warc_record_offset"))
        length = int(row.get("warc_record_length"))
        if offset < 0 or length < 256 or length > min(16 * 1024 * 1024, options.maximum_span_bytes):
            return None, "range_bounds"
        warc_filename = str(row.get("warc_filename") or "")
        if (
            not warc_filename.startswith(f"crawl-data/{crawl}/")
            or not warc_filename.endswith(".warc.gz")
            or ".." in Path(warc_filename).parts
        ):
            return None, "warc_path"
        fetch_time, fetch_epoch = _timestamp(row.get("fetch_time"))
        capture_date = fetch_time[:10]
        if options.freshness_cutoff_start and capture_date < options.freshness_cutoff_start:
            return None, "capture_before_freshness_cutoff"
        if options.freshness_cutoff_end and capture_date > options.freshness_cutoff_end:
            return None, "capture_after_freshness_cutoff"
    except (TypeError, ValueError, OverflowError):
        return None, "malformed_metadata"
    category = _fresh_category(row, canonical_url)
    if category not in options.allowed_categories:
        return None, "route_category"
    sample_hash = hashlib.sha256(
        f"{options.seed}\0{crawl}\0{canonical_url}\0{digest}".encode("utf-8")
    ).hexdigest()
    return (
        {
            "source_id": source_id,
            "crawl": crawl,
            "url": original_url,
            "canonical_url": canonical_url,
            "host": host,
            "registered_domain": str(row.get("url_host_registered_domain") or host),
            "private_domain": str(row.get("url_host_private_domain") or ""),
            "fetch_time": fetch_time,
            "capture_date": capture_date,
            "fetch_epoch": fetch_epoch,
            "content_digest": digest.upper(),
            "content_mime_type": sent_mime or detected_mime,
            "content_mime_detected": detected_mime or sent_mime,
            "content_charset": str(row.get("content_charset") or ""),
            "content_languages": languages,
            "warc_record_id": _record_id(row.get("warc_record_id")),
            "warc_filename": warc_filename,
            "warc_record_offset": offset,
            "warc_record_length": length,
            "warc_segment": str(row.get("warc_segment") or ""),
            "fresh_category": category,
            "route": options.route,
            "priority": CATEGORY_PRIORITY[category],
            "full_body_likelihood": int(length >= 1_024),
            "sample_hash": sample_hash,
            "opt_out_snapshot_sha256": policy.snapshot_sha256,
        },
        None,
    )


def _allocate(capacity: int, weights: Sequence[tuple[str, float]]) -> dict[str, int]:
    total = sum(weight for _, weight in weights)
    raw = {name: capacity * weight / total for name, weight in weights}
    result = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = capacity - sum(result.values())
    for name in sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))[:remainder]:
        result[name] += 1
    return result


def select_partition_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    crawl: str,
    source_id: str,
    policy: OptOutPolicy,
    options: FreshWebOptions,
    capacity: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select a taxonomy-balanced bounded partition."""

    quotas = _allocate(capacity, options.category_weights)
    category_heaps: dict[str, list[_HeapCandidate]] = defaultdict(list)
    fallback_heap: list[_HeapCandidate] = []
    rejections: Counter[str] = Counter()
    scanned = eligible = 0

    def retain(heap: list[_HeapCandidate], limit: int, candidate: dict[str, Any]) -> None:
        if limit <= 0:
            return
        sample = int(candidate["sample_hash"], 16)
        item = _HeapCandidate(
            key=(
                -sample,
                candidate["canonical_url"],
                candidate["content_digest"],
                int(candidate["fetch_epoch"]),
                candidate["warc_filename"],
                int(candidate["warc_record_offset"]),
            ),
            candidate=candidate,
        )
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    for row in rows:
        scanned += 1
        candidate, reason = metadata_candidate(
            row,
            crawl=crawl,
            source_id=source_id,
            policy=policy,
            options=options,
        )
        if candidate is None:
            rejections[reason or "unknown"] += 1
            continue
        eligible += 1
        retain(category_heaps[candidate["fresh_category"]], quotas[candidate["fresh_category"]], candidate)
        retain(fallback_heap, capacity, candidate)

    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for heap in category_heaps.values():
        for item in heap:
            candidate = item.candidate
            selected_by_key[(candidate["canonical_url"], candidate["content_digest"])] = candidate
    for item in sorted(fallback_heap, reverse=True):
        if len(selected_by_key) >= capacity:
            break
        candidate = item.candidate
        selected_by_key.setdefault((candidate["canonical_url"], candidate["content_digest"]), candidate)
    selected = sorted(selected_by_key.values(), key=lambda item: (item["sample_hash"], item["canonical_url"]))
    return selected, {
        "scanned": scanned,
        "eligible": eligible,
        "selected": len(selected),
        "rejections": dict(sorted(rejections.items())),
        "selected_categories": dict(sorted(Counter(item["fresh_category"] for item in selected).items())),
    }


def select_exact_candidates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure exact URL-then-digest winner selection used by tests and tools."""

    def winner_key(item: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            int(item.get("priority", 0)),
            int(item.get("full_body_likelihood", 0)),
            int(item.get("fetch_epoch", 0)),
            int(item.get("warc_record_length", 0)),
            -int(str(item.get("sample_hash", "f" * 64)), 16),
        )

    urls: dict[str, dict[str, Any]] = {}
    for value in rows:
        item = dict(value)
        key = str(item["canonical_url"])
        if key not in urls or winner_key(item) > winner_key(urls[key]):
            urls[key] = item
    digests: dict[str, dict[str, Any]] = {}
    for item in urls.values():
        key = str(item["content_digest"])
        if key not in digests or winner_key(item) > winner_key(digests[key]):
            digests[key] = item
    return sorted(digests.values(), key=lambda item: (item["warc_filename"], item["warc_record_offset"]))


def coalesce_ranges(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_gap: int,
    maximum_span: int,
) -> list[RangeSpan]:
    """Coalesce nearby record-level gzip members into bounded HTTP ranges."""

    if maximum_gap < 0 or maximum_span < 1:
        raise ValueError("invalid range coalescing limits")
    ordered = sorted((dict(record) for record in records), key=lambda item: int(item["warc_record_offset"]))
    spans: list[RangeSpan] = []
    current: list[dict[str, Any]] = []
    start = end = -1
    for record in ordered:
        record_start = int(record["warc_record_offset"])
        record_end = record_start + int(record["warc_record_length"]) - 1
        if record_end < record_start:
            raise ValueError("invalid record range")
        can_merge = current and record_start <= end + maximum_gap + 1 and record_end - start + 1 <= maximum_span
        if not current or can_merge:
            if not current:
                start = record_start
            end = max(end, record_end)
            current.append(record)
            continue
        spans.append(RangeSpan(start=start, end=end, records=tuple(current)))
        current = [record]
        start, end = record_start, record_end
    if current:
        spans.append(RangeSpan(start=start, end=end, records=tuple(current)))
    return spans


def _parse_headers(payload: bytes) -> tuple[dict[str, str], bytes]:
    marker = b"\r\n\r\n"
    index = payload.find(marker)
    if index < 0:
        marker = b"\n\n"
        index = payload.find(marker)
    if index < 0:
        raise FreshWebError("record headers are incomplete")
    header_bytes = payload[:index]
    body = payload[index + len(marker) :]
    lines = header_bytes.decode("latin-1").replace("\r\n", "\n").split("\n")
    headers: dict[str, str] = {":status-line": lines[0].strip() if lines else ""}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        key = name.strip().lower()
        value = value.strip()
        if key in headers:
            headers[key] += ", " + value
        else:
            headers[key] = value
    return headers, body


def _gzip_decompress_bounded(payload: bytes, limit: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
        output = handle.read(limit + 1)
    if len(output) > limit:
        raise FreshWebError(f"Compressed payload expands beyond {limit} bytes")
    return output


def _zlib_decompress_bounded(payload: bytes, *, wbits: int, limit: int) -> bytes:
    import zlib

    decompressor = zlib.decompressobj(wbits)
    output = decompressor.decompress(payload, limit + 1)
    if len(output) > limit or decompressor.unconsumed_tail:
        raise FreshWebError(f"Compressed payload expands beyond {limit} bytes")
    output += decompressor.flush(limit + 1 - len(output))
    if len(output) > limit:
        raise FreshWebError(f"Compressed payload expands beyond {limit} bytes")
    if not decompressor.eof:
        raise FreshWebError("Compressed payload ended before its end-of-stream marker")
    return output


def _decode_http_body(body: bytes, headers: Mapping[str, str]) -> bytes:
    encoding = headers.get("content-encoding", "").lower().strip()
    if not encoding:
        return body
    if encoding == "gzip":
        return _zlib_decompress_bounded(body, wbits=16 + 15, limit=MAX_DECODED_BODY_BYTES)
    if encoding == "deflate":
        import zlib

        try:
            return _zlib_decompress_bounded(body, wbits=15, limit=MAX_DECODED_BODY_BYTES)
        except zlib.error:
            return _zlib_decompress_bounded(body, wbits=-15, limit=MAX_DECODED_BODY_BYTES)
    if encoding == "br":
        try:
            import brotli
        except ImportError as exc:
            raise FreshWebError("Brotli HTTP payload requires brotli") from exc
        output = brotli.decompress(body)
        if len(output) > MAX_DECODED_BODY_BYTES:
            raise FreshWebError(
                f"Compressed payload expands beyond {MAX_DECODED_BODY_BYTES} bytes"
            )
        return output
    if encoding in {"zstd", "zstandard"}:
        try:
            import zstandard
        except ImportError as exc:
            raise FreshWebError("Zstandard HTTP payload requires zstandard") from exc
        return zstandard.ZstdDecompressor().decompress(
            body, max_output_size=MAX_DECODED_BODY_BYTES
        )
    raise FreshWebError(f"Unsupported HTTP content encoding: {encoding}")


def _charset(content_type: str, index_charset: str) -> str:
    match = re.search(r"charset\s*=\s*['\"]?([^;'\"\s]+)", content_type, flags=re.IGNORECASE)
    return (match.group(1) if match else index_charset or "utf-8").strip()


CANONICAL_META_KEYS = {"canonical", "og:url", "twitter:url"}
LICENSE_META_KEYS = {
    "license",
    "dc.license",
    "dcterms.license",
    "dc.rights",
    "dcterms.rights",
    "rights",
}
VERSION_META_KEYS = {
    "version",
    "softwareversion",
    "software:version",
    "docsearch:version",
    "docs:version",
    "dcterms.hasversion",
    "dcterms.isversionof",
    "release",
    "revision",
    "git:commit",
}
PUBLICATION_META_KEYS = {
    "article:published_time",
    "article:publication_date",
    "citation_date",
    "citation_publication_date",
    "date",
    "datepublished",
    "dc.date",
    "dc.date.issued",
    "dcterms.date",
    "dcterms.issued",
    "og:published_time",
    "publishdate",
    "pubdate",
}
ENGLISH_COMMON_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "in",
    "is",
    "of",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


def _bounded_evidence_value(value: Any, limit: int = 2_048) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _evidence(source: str, key: str, value: Any) -> dict[str, str] | None:
    normalized = _bounded_evidence_value(value)
    if not normalized:
        return None
    return {"source": source, "key": key, "value": normalized}


class _StructuredHTMLExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "aside", "form"}
    BLOCK_TAGS = {"article", "blockquote", "dd", "div", "dl", "dt", "main", "p", "section", "table", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.pre_depth = 0
        self.inline_code_depth = 0
        self.robot_directives: set[str] = set()
        self.tag_counts: Counter[str] = Counter()
        self.html_languages: set[str] = set()
        self.canonical_evidence: list[dict[str, str]] = []
        self.license_evidence: list[dict[str, str]] = []
        self.version_evidence: list[dict[str, str]] = []
        self.publication_evidence: list[dict[str, str]] = []
        self.titles: list[str] = []
        self.headings: list[str] = []
        self._capture_stack: list[dict[str, Any]] = []
        self._jsonld_parts: list[str] | None = None
        self._jsonld_characters = 0
        self.jsonld_documents: list[str] = []
        self.code_characters = 0

    @staticmethod
    def _meta_key(attributes: Mapping[str, str]) -> str:
        return (
            attributes.get("property")
            or attributes.get("name")
            or attributes.get("itemprop")
            or attributes.get("http-equiv")
            or ""
        ).strip().lower()

    @staticmethod
    def _append(target: list[dict[str, str]], source: str, key: str, value: Any) -> None:
        if len(target) >= 128:
            return
        item = _evidence(source, key, value)
        if item is not None and item not in target:
            target.append(item)

    def _newline(self) -> None:
        if not self.parts or not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag == "script" and attributes.get("type", "").lower().split(";", 1)[0] == "application/ld+json":
            self._jsonld_parts = []
            self._jsonld_characters = 0
            return
        if tag == "html" and attributes.get("lang"):
            self.html_languages.add(attributes["lang"].strip().lower())
        if tag == "meta":
            key = self._meta_key(attributes)
            content = attributes.get("content", "")
            if key in {"robots", "googlebot", "ccbot"}:
                self.robot_directives.update(
                    token.strip().lower()
                    for token in re.split(r"[,\s]+", content)
                    if token.strip()
                )
            if key == "content-language":
                self.html_languages.update(
                    token.strip().lower()
                    for token in re.split(r"[,\s]+", content)
                    if token.strip()
                )
            if key in CANONICAL_META_KEYS:
                self._append(self.canonical_evidence, "html_meta", key, content)
            if key in LICENSE_META_KEYS:
                self._append(self.license_evidence, "html_meta", key, content)
            if key in VERSION_META_KEYS:
                self._append(self.version_evidence, "html_meta", key, content)
            if key in PUBLICATION_META_KEYS:
                self._append(self.publication_evidence, "html_meta", key, content)
        if tag in {"link", "a"}:
            rels = {token.lower() for token in re.split(r"\s+", attributes.get("rel", "")) if token}
            href = attributes.get("href", "")
            if "canonical" in rels:
                self._append(self.canonical_evidence, "html_link", "canonical", href)
            if "license" in rels:
                self._append(self.license_evidence, f"html_{tag}", "license", href)
        for key in ("data-version", "data-doc-version", "data-release", "data-revision"):
            if attributes.get(key):
                self._append(self.version_evidence, "html_attribute", key, attributes[key])
        if tag == "time" and attributes.get("datetime"):
            hints = " ".join(
                (
                    attributes.get("itemprop", ""),
                    attributes.get("class", ""),
                    attributes.get("name", ""),
                )
            ).lower()
            if any(token in hints for token in ("publish", "issued", "article", "date")):
                self._append(
                    self.publication_evidence,
                    "html_time",
                    "datetime",
                    attributes["datetime"],
                )
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        self.tag_counts[tag] += 1
        if tag in {"title", "h1"} and len(self._capture_stack) < 64:
            self._capture_stack.append({"tag": tag, "parts": [], "characters": 0})
        if tag in self.BLOCK_TAGS or tag in {"br", "hr"}:
            self._newline()
        if re.fullmatch(r"h[1-6]", tag):
            self._newline()
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self._newline()
            self.parts.append("- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "pre":
            self._newline()
            self.parts.append("```\n")
            self.pre_depth += 1
        elif tag == "code" and not self.pre_depth:
            self.parts.append("`")
            self.inline_code_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script" and self._jsonld_parts is not None:
            payload = "".join(self._jsonld_parts).strip()
            if payload and len(self.jsonld_documents) < 8:
                self.jsonld_documents.append(payload[:1_000_000])
            self._jsonld_parts = None
            return
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        for index in range(len(self._capture_stack) - 1, -1, -1):
            capture = self._capture_stack[index]
            if capture["tag"] == tag:
                value = _bounded_evidence_value(" ".join(capture["parts"]))
                del self._capture_stack[index]
                if value:
                    target = self.titles if tag == "title" else self.headings
                    if len(target) < 64:
                        target.append(value)
                break
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
            self.parts.append("\n```\n")
        elif tag == "code" and self.inline_code_depth and not self.pre_depth:
            self.inline_code_depth -= 1
            self.parts.append("`")
        elif tag in self.BLOCK_TAGS or tag == "li" or re.fullmatch(r"h[1-6]", tag):
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._jsonld_parts is not None:
            remaining = 1_000_000 - self._jsonld_characters
            if remaining > 0:
                value = data[:remaining]
                self._jsonld_parts.append(value)
                self._jsonld_characters += len(value)
            return
        if self.skip_depth or not data:
            return
        for capture in self._capture_stack:
            if capture["characters"] < 2_048:
                value = data[: 2_048 - capture["characters"]]
                capture["parts"].append(value)
                capture["characters"] += len(value)
        if self.pre_depth:
            self.code_characters += len(data)
            self.parts.append(data)
        else:
            if self.inline_code_depth:
                self.code_characters += len(data)
            collapsed = re.sub(r"\s+", " ", data)
            if collapsed.strip():
                if self.parts and not self.parts[-1].endswith((" ", "\n", "`")):
                    self.parts.append(" ")
                self.parts.append(collapsed.strip())

    def text(self) -> str:
        lines: list[str] = []
        in_fence = False
        for raw in "".join(self.parts).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if raw.strip() == "```":
                line = "```"
                in_fence = not in_fence
            elif in_fence:
                line = raw.rstrip()
            else:
                line = re.sub(r"[ \t]+", " ", raw).strip().strip("|").strip()
            if line or (lines and lines[-1] and not in_fence):
                lines.append(line)
        compact: list[str] = []
        for line in lines:
            if not line and (not compact or not compact[-1]):
                continue
            compact.append(line)
        return "\n".join(compact).strip()


def _append_unique_evidence(
    target: list[dict[str, Any]], source: str, key: str, value: Any
) -> None:
    if len(target) >= 128:
        return
    item = _evidence(source, key, value)
    if item is not None and item not in target:
        target.append(item)


def _jsonld_scalar_values(value: Any) -> Iterator[str]:
    if isinstance(value, (str, int, float)):
        yield str(value)
    elif isinstance(value, list):
        for item in value:
            yield from _jsonld_scalar_values(item)
    elif isinstance(value, Mapping):
        for key in ("@id", "url", "name"):
            if key in value:
                yield from _jsonld_scalar_values(value[key])


def _jsonld_url_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _jsonld_url_values(item)
    elif isinstance(value, Mapping):
        for key in ("@id", "url"):
            if key in value:
                yield from _jsonld_url_values(value[key])


def _collect_jsonld_evidence(
    value: Any,
    *,
    canonical: list[dict[str, Any]],
    licenses: list[dict[str, Any]],
    versions: list[dict[str, Any]],
    publications: list[dict[str, Any]],
    depth: int = 0,
) -> None:
    if depth > 32:
        return
    if isinstance(value, list):
        for item in value:
            _collect_jsonld_evidence(
                item,
                canonical=canonical,
                licenses=licenses,
                versions=versions,
                publications=publications,
                depth=depth,
            )
        return
    if not isinstance(value, Mapping):
        return
    types = {
        str(item).lower()
        for item in (
            value.get("@type") if isinstance(value.get("@type"), list) else [value.get("@type")]
        )
        if item
    }
    for raw_key, child in value.items():
        key = str(raw_key).lower().replace("_", "")
        if key == "mainentityofpage" or (
            key == "url"
            and bool(types & {"article", "webpage", "techarticle", "scholarlyarticle", "creativework"})
        ):
            for scalar in _jsonld_url_values(child):
                _append_unique_evidence(canonical, "json_ld", str(raw_key), scalar)
        if key in {"license", "copyrightnotice"}:
            for scalar in _jsonld_scalar_values(child):
                _append_unique_evidence(licenses, "json_ld", str(raw_key), scalar)
        if key in {"version", "softwareversion", "release", "revision"}:
            for scalar in _jsonld_scalar_values(child):
                _append_unique_evidence(versions, "json_ld", str(raw_key), scalar)
        if key in {"datepublished", "publicationdate", "dateissued"}:
            for scalar in _jsonld_scalar_values(child):
                _append_unique_evidence(publications, "json_ld", str(raw_key), scalar)
        _collect_jsonld_evidence(
            child,
            canonical=canonical,
            licenses=licenses,
            versions=versions,
            publications=publications,
            depth=depth + 1,
        )


def _http_link_evidence(
    header: str,
    canonical: list[dict[str, Any]],
    licenses: list[dict[str, Any]],
) -> None:
    for match in re.finditer(
        r"<([^>]+)>\s*;[^,]*?\brel\s*=\s*[\"']?([^\"';,]+)",
        header,
        flags=re.IGNORECASE,
    ):
        href, relation = match.groups()
        rels = {token.lower() for token in relation.split()}
        if "canonical" in rels:
            _append_unique_evidence(canonical, "http_link", "canonical", href)
        if "license" in rels:
            _append_unique_evidence(licenses, "http_link", "license", href)


def _resolve_url_evidence(entries: Sequence[Mapping[str, Any]], base_url: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in entries:
        item = dict(raw)
        value = str(item.get("value") or "")
        looks_like_url = bool(
            value.lower().startswith(("http://", "https://", "//", "/", "./", "../"))
            or re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/|$)", value)
        )
        if looks_like_url:
            try:
                resolved = urljoin(base_url, value)
                if urlsplit(resolved).scheme.lower() in {"http", "https"}:
                    item["resolved_url"] = canonicalize_url(resolved)
            except (TypeError, ValueError):
                pass
        if item not in output:
            output.append(item)
    return output


def _normalise_publication_date(value: Any) -> str | None:
    text = _bounded_evidence_value(value, 256)
    if not text:
        return None
    iso_match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if iso_match:
        year, month, day = (int(part) for part in iso_match.groups())
        try:
            moment = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None
        if 1800 <= moment.year <= datetime.now(timezone.utc).year + 1:
            return moment.date().isoformat()
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(candidate)
    except ValueError:
        moment = None
    if moment is not None and 1800 <= moment.year <= datetime.now(timezone.utc).year + 1:
        return moment.date().isoformat()
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            moment = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if 1800 <= moment.year <= datetime.now(timezone.utc).year + 1:
            return moment.date().isoformat()
    return None


def _derive_version_evidence(
    explicit: Sequence[Mapping[str, Any]],
    *,
    titles: Sequence[str],
    headings: Sequence[str],
    urls: Sequence[str],
) -> list[dict[str, Any]]:
    output = [dict(item) for item in explicit]
    version_token = re.compile(
        r"\b(?:version|release|revision|rev|v)\s*[:=_-]?\s*"
        r"(?:\d+(?:\.\d+){0,4}|latest|stable|current)\b",
        flags=re.IGNORECASE,
    )
    documentation_version = re.compile(r"\b\d+\.\d+(?:\.\d+){0,3}\b")
    for source, values in (("html_title", titles), ("html_h1", headings)):
        for value in values:
            match = version_token.search(value)
            if match is None and re.search(
                r"\b(?:api|docs?|documentation|manual|reference)\b", value, flags=re.IGNORECASE
            ):
                match = documentation_version.search(value)
            if match is not None:
                _append_unique_evidence(output, source, "version_text", match.group(0))
    url_pattern = re.compile(
        r"/(?:v\d+(?:\.\d+){0,4}|version(?:s)?[/_-][^/?#]+|"
        r"releases?[/_-][^/?#]+|latest|stable|current)(?:/|$)|"
        r"/(?:docs?|api|reference|manual)/\d+(?:\.\d+){1,3}(?:/|$)|"
        r"[?&](?:version|release|v)=[^&#]+",
        flags=re.IGNORECASE,
    )
    for value in urls:
        match = url_pattern.search(value)
        if match is not None:
            _append_unique_evidence(output, "url", "version_path", match.group(0))
    return output


def _evaluate_open_license(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    negative = re.compile(
        r"(?:by[-_ /]?(?:nc|nd)|non[- ]?commercial|no[- ]?derivatives?|all rights reserved)",
        flags=re.IGNORECASE,
    )
    positive_rules = {
        "cc_by_or_cc_by_sa": re.compile(
            r"creativecommons\.org/licenses/(?:by|by-sa)/|\bcc\s*[- ]?by(?:\s*[- ]?sa)?\b|"
            r"creative commons attribution(?:-sharealike)?",
            flags=re.IGNORECASE,
        ),
        "cc0_or_public_domain": re.compile(
            r"creativecommons\.org/publicdomain/(?:zero|mark)/|\bcc\s*0\b|\bpublic domain\b",
            flags=re.IGNORECASE,
        ),
        "open_source_or_open_government": re.compile(
            r"\bmit license\b|\bapache license(?:,? version)? 2(?:\.0)?\b|"
            r"\bbsd (?:2|3)[- ]clause\b|\bmpl 2\.0\b|\bopen government licen[cs]e\b|"
            r"\bopen data commons attribution licen[cs]e\b",
            flags=re.IGNORECASE,
        ),
    }
    matches: list[dict[str, Any]] = []
    for item in evidence:
        candidate = " ".join(
            str(item.get(key) or "") for key in ("value", "resolved_url")
        )
        if negative.search(candidate):
            continue
        for rule, pattern in positive_rules.items():
            if pattern.search(candidate):
                matches.append({"rule": rule, "evidence": dict(item)})
                break
    return {
        "policy": "explicit-reusable-open-license-v1",
        "decision": bool(matches),
        "matches": matches,
    }


def _structural_quality(
    text: str,
    parser: _StructuredHTMLExtractor | None,
    *,
    mime: str,
) -> dict[str, Any]:
    characters = len(text)
    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text, flags=re.UNICODE)
    alphabetic = sum(character.isalpha() for character in text)
    replacement = text.count("�")
    tags = dict(sorted((parser.tag_counts if parser is not None else {}).items()))
    code_characters = parser.code_characters if parser is not None else 0
    line_count = len(text.splitlines())
    score = 0
    score += 2 if len(words) >= 80 else int(len(words) >= 40)
    if tags.get("main", 0) or tags.get("article", 0):
        score += 2
    elif tags.get("p", 0) >= 2 or line_count >= 5:
        score += 1
    score += int(sum(tags.get(f"h{level}", 0) for level in range(1, 7)) > 0)
    score += int(tags.get("p", 0) >= 2)
    score += int(
        code_characters >= 80
        or tags.get("li", 0) > 0
        or tags.get("table", 0) > 0
    )
    alphabetic_ratio = alphabetic / max(1, characters)
    replacement_ratio = replacement / max(1, characters)
    non_whitespace_ratio = sum(not character.isspace() for character in text) / max(1, characters)
    score += int(alphabetic_ratio >= 0.35 or code_characters >= 80)
    score += int(replacement_ratio <= 0.005)
    score += int(non_whitespace_ratio >= 0.55)
    passed = (
        score >= 5
        and replacement_ratio <= 0.02
        and (alphabetic_ratio >= 0.20 or code_characters >= 100)
        and (len(words) >= 25 or code_characters >= 100)
    )
    return {
        "policy": "freshweb-structural-quality-v1",
        "passed": passed,
        "score": score,
        "mime": mime,
        "characters": characters,
        "words": len(words),
        "lines": line_count,
        "alphabetic_ratio": round(alphabetic_ratio, 6),
        "replacement_character_ratio": round(replacement_ratio, 8),
        "non_whitespace_ratio": round(non_whitespace_ratio, 6),
        "code_characters": code_characters,
        "tag_counts": tags,
    }


def _english_evidence(
    text: str,
    parser: _StructuredHTMLExtractor | None,
    http_headers: Mapping[str, str],
    coordinate: Mapping[str, Any],
) -> dict[str, Any]:
    raw_index_languages = coordinate.get("content_languages", [])
    if isinstance(raw_index_languages, str):
        index_languages = [
            token.strip().lower() for token in raw_index_languages.split(",") if token.strip()
        ]
    else:
        index_languages = [str(value).lower() for value in raw_index_languages]
    html_languages = sorted(parser.html_languages) if parser is not None else []
    http_languages = [
        token.strip().lower()
        for token in re.split(r"[,\s]+", http_headers.get("content-language", ""))
        if token.strip()
    ]
    words = {value.lower() for value in re.findall(r"[A-Za-z]+", text)}
    common_hits = sorted(words & ENGLISH_COMMON_WORDS)
    alphabetic = [character for character in text if character.isalpha()]
    latin_ratio = (
        sum(character.isascii() and character.isalpha() for character in alphabetic)
        / max(1, len(alphabetic))
    )
    reasons = []
    if "eng" in index_languages:
        reasons.append("url_index_eng")
    if any(value == "en" or value.startswith("en-") for value in html_languages):
        reasons.append("html_lang_en")
    if any(value == "en" or value.startswith("en-") for value in http_languages):
        reasons.append("http_content_language_en")
    if latin_ratio >= 0.85 and len(common_hits) >= 3:
        reasons.append("latin_text_with_common_english_words")
    return {
        "policy": "freshweb-english-evidence-v1",
        "decision": bool(reasons),
        "reasons": reasons,
        "url_index_languages": index_languages,
        "html_languages": html_languages,
        "http_content_languages": http_languages,
        "common_word_hits": common_hits,
        "latin_letter_ratio": round(latin_ratio, 6),
    }


def _derive_page_evidence(
    *,
    parser: _StructuredHTMLExtractor | None,
    http_headers: Mapping[str, str],
    coordinate: Mapping[str, Any],
    text: str,
    mime: str,
) -> dict[str, Any]:
    canonical: list[dict[str, Any]] = list(parser.canonical_evidence if parser else [])
    licenses: list[dict[str, Any]] = list(parser.license_evidence if parser else [])
    versions: list[dict[str, Any]] = list(parser.version_evidence if parser else [])
    publications: list[dict[str, Any]] = list(parser.publication_evidence if parser else [])
    _http_link_evidence(http_headers.get("link", ""), canonical, licenses)
    for key in ("license", "x-license", "x-content-license"):
        if http_headers.get(key):
            _append_unique_evidence(licenses, "http_header", key, http_headers[key])
    for key in ("version", "x-version", "x-document-version", "api-version"):
        if http_headers.get(key):
            _append_unique_evidence(versions, "http_header", key, http_headers[key])
    for key in ("publication-date", "x-publication-date", "x-published-time"):
        if http_headers.get(key):
            _append_unique_evidence(publications, "http_header", key, http_headers[key])
    jsonld_parse_errors = 0
    if parser is not None:
        for payload in parser.jsonld_documents:
            try:
                decoded = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                jsonld_parse_errors += 1
                continue
            _collect_jsonld_evidence(
                decoded,
                canonical=canonical,
                licenses=licenses,
                versions=versions,
                publications=publications,
            )
    base_url = str(coordinate.get("url") or coordinate.get("canonical_url") or "")
    canonical = _resolve_url_evidence(canonical, base_url)
    licenses = _resolve_url_evidence(licenses, base_url)
    version_urls = [base_url, str(coordinate.get("canonical_url") or "")]
    version_urls.extend(
        str(item["resolved_url"]) for item in canonical if item.get("resolved_url")
    )
    versions = _derive_version_evidence(
        versions,
        titles=parser.titles if parser else (),
        headings=parser.headings if parser else (),
        urls=version_urls,
    )
    normalized_publications: list[dict[str, Any]] = []
    for item in publications:
        value = dict(item)
        normalized = _normalise_publication_date(value.get("value"))
        if normalized:
            value["normalized_date"] = normalized
        if value not in normalized_publications:
            normalized_publications.append(value)
    publication_date = next(
        (str(item["normalized_date"]) for item in normalized_publications if item.get("normalized_date")),
        None,
    )
    license_evaluation = _evaluate_open_license(licenses)
    return {
        "canonical": canonical,
        "declared_canonical_url": next(
            (str(item["resolved_url"]) for item in canonical if item.get("resolved_url")),
            None,
        ),
        "licenses": licenses,
        "license_evaluation": license_evaluation,
        "versions": versions,
        "publications": normalized_publications,
        "publication_date": publication_date,
        "document_title": parser.titles[0] if parser and parser.titles else None,
        "structural_quality": _structural_quality(text, parser, mime=mime),
        "english": _english_evidence(text, parser, http_headers, coordinate),
        "jsonld_parse_errors": jsonld_parse_errors,
    }


def _extract_warc_member(
    compressed: bytes,
    coordinate: Mapping[str, Any],
    *,
    options: FreshWebOptions,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        record = _gzip_decompress_bounded(compressed, MAX_DECOMPRESSED_RECORD_BYTES)
        warc_headers, block_and_trailer = _parse_headers(record)
        if not warc_headers.get(":status-line", "").startswith("WARC/"):
            return None, "not_warc"
        record_type = warc_headers.get("warc-type", "").lower()
        if record_type == "revisit":
            return None, "bodyless_revisit"
        if record_type != "response":
            return None, "not_response"
        block_length = int(warc_headers.get("content-length", len(block_and_trailer)))
        if block_length < 0 or block_length > len(block_and_trailer):
            return None, "warc_block_length"
        block = block_and_trailer[:block_length]
        http_headers, body = _parse_headers(block)
        status_line = http_headers.get(":status-line", "")
        if not re.match(r"^HTTP/\d(?:\.\d)?\s+200(?:\s|$)", status_line):
            return None, "warc_http_status"
        expected_digest = str(coordinate["content_digest"]).upper().removeprefix("SHA1:")
        header_digest = warc_headers.get("warc-payload-digest", "").upper().removeprefix("SHA1:")
        if not header_digest or header_digest != expected_digest:
            return None, "warc_digest_header"
        target = warc_headers.get("warc-target-uri", "")
        if target:
            try:
                if canonicalize_url(target) != coordinate["canonical_url"]:
                    return None, "warc_target_uri"
            except ValueError:
                return None, "warc_target_uri"
        # WARC-Payload-Digest covers the stored HTTP entity body.  Validate it
        # before interpreting a remaining HTTP Content-Encoding.
        actual_digest = base64.b32encode(hashlib.sha1(body).digest()).decode("ascii").rstrip("=")
        if actual_digest != expected_digest:
            return None, "warc_payload_digest"
        try:
            body = _decode_http_body(body, http_headers)
        except MemoryError:
            raise
        except Exception as exc:
            raise FreshWebError("HTTP entity decoding failed") from exc
        content_type = http_headers.get("content-type", str(coordinate.get("content_mime_type") or ""))
        mime = content_type.lower().split(";", 1)[0].strip()
        charset = _charset(content_type, str(coordinate.get("content_charset") or ""))
        try:
            decoded = body.decode(charset, errors="replace")
        except LookupError:
            charset = "utf-8"
            decoded = body.decode(charset, errors="replace")
        robots: set[str] = {
            token.strip().lower()
            for token in re.split(r"[,\s]+", http_headers.get("x-robots-tag", ""))
            if token.strip()
        }
        parser: _StructuredHTMLExtractor | None = None
        if mime in {"text/html", "application/xhtml+xml"}:
            parser = _StructuredHTMLExtractor()
            parser.feed(decoded)
            parser.close()
            text = parser.text()
            robots |= parser.robot_directives
        elif mime == "text/plain":
            text = "\n".join(line.rstrip() for line in decoded.replace("\r\n", "\n").split("\n")).strip()
        else:
            return None, "extracted_mime"
        if robots & {"noai", "noml", "notrain", "noarchive"}:
            return None, "publisher_machine_learning_opt_out"
        if len(text) < options.minimum_characters:
            return None, "too_short"
        if coordinate.get("route") and coordinate.get("route") != options.route:
            return None, "route_mismatch"
        page_evidence = _derive_page_evidence(
            parser=parser,
            http_headers=http_headers,
            coordinate=coordinate,
            text=text,
            mime=mime,
        )
        if not page_evidence["structural_quality"]["passed"]:
            return None, "structural_quality"
        if options.require_english and not page_evidence["english"]["decision"]:
            return None, "english_evidence"
        if (
            options.require_reusable_open_license
            and not page_evidence["license_evaluation"]["decision"]
        ):
            return None, "missing_reusable_open_license"
        if options.route == "fresh_science":
            if not page_evidence["publication_date"]:
                return None, "missing_publication_date"
            if (
                options.freshness_cutoff_start
                and page_evidence["publication_date"] < options.freshness_cutoff_start
            ) or (
                options.freshness_cutoff_end
                and page_evidence["publication_date"] > options.freshness_cutoff_end
            ):
                return None, "publication_outside_freshness_cutoff"
        if options.route in {"software_docs", "official_docs"} and not page_evidence["versions"]:
            return None, "missing_version_evidence"
    except (FreshWebError, OSError, EOFError, ValueError, TypeError, UnicodeError):
        return None, "malformed_warc"

    raw_sha = _sha256_bytes(compressed)
    document_id = hashlib.sha256(
        f"freshweb\0{coordinate['content_digest']}".encode("ascii")
    ).hexdigest()
    metadata = {
        key: coordinate.get(key)
        for key in (
            "source_id",
            "crawl",
            "url",
            "canonical_url",
            "host",
            "registered_domain",
            "private_domain",
            "fetch_time",
            "capture_date",
            "content_digest",
            "content_mime_type",
            "content_mime_detected",
            "content_charset",
            "content_languages",
            "warc_record_id",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
            "warc_segment",
            "fresh_category",
            "route",
            "priority",
            "opt_out_snapshot_sha256",
        )
    }
    metadata.update(
        {
            "fresh": True,
            "category": "web",
            "raw_warc_member_sha256": raw_sha,
            "warc_payload_digest_verified": True,
            "extractor_version": EXTRACTOR_VERSION,
            "extracted_charset": charset,
            "declared_canonical_url": page_evidence["declared_canonical_url"],
            "canonical_evidence": page_evidence["canonical"],
            "license_evidence": page_evidence["licenses"],
            "license_evaluation": page_evidence["license_evaluation"],
            "license_status": (
                "explicit_reusable_open"
                if page_evidence["license_evaluation"]["decision"]
                else "per_record_required"
            ),
            "version_evidence": page_evidence["versions"],
            "publication_evidence": page_evidence["publications"],
            "publication_date": page_evidence["publication_date"],
            "document_title": page_evidence["document_title"],
            "structural_quality": page_evidence["structural_quality"],
            "english_evidence": page_evidence["english"],
            "jsonld_parse_errors": page_evidence["jsonld_parse_errors"],
        }
    )
    return {
        "id": document_id,
        "text": text,
        "metadata": metadata,
    }, None


def _range_content(response: Any, start: int, end: int, url: str) -> bytes:
    if response.status_code != 206:
        raise PermanentHttpError(f"Range request returned HTTP {response.status_code}, not 206: {url}")
    content_range = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(?:\d+|\*)", content_range, flags=re.IGNORECASE)
    if not match or int(match.group(1)) != start or int(match.group(2)) != end:
        raise FreshWebError(f"Invalid Content-Range for {url}: {content_range!r}")
    content = bytes(response.content)
    expected = end - start + 1
    if len(content) != expected:
        raise FreshWebError(f"Range length mismatch for {url}: {len(content)} != {expected}")
    return content


def _fetch_warc_group(
    filename: str,
    coordinates: Sequence[dict[str, Any]],
    *,
    http: _HttpClient,
    options: FreshWebOptions,
) -> dict[str, Any]:
    url = urljoin(options.data_root.rstrip("/") + "/", filename)
    output: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    downloaded_bytes = 0
    spans = coalesce_ranges(
        coordinates,
        maximum_gap=options.coalesce_gap_bytes,
        maximum_span=options.maximum_span_bytes,
    )
    for span in spans:
        response = http.get(
            url,
            headers={"Range": f"bytes={span.start}-{span.end}", "Accept-Encoding": "identity"},
            expected=frozenset({206}),
        )
        try:
            payload = _range_content(response, span.start, span.end, url)
        finally:
            response.close()
        downloaded_bytes += len(payload)
        for coordinate in span.records:
            relative = int(coordinate["warc_record_offset"]) - span.start
            length = int(coordinate["warc_record_length"])
            compressed = payload[relative : relative + length]
            if len(compressed) != length:
                rejections["coalesced_slice_bounds"] += 1
                continue
            document, reason = _extract_warc_member(
                compressed,
                coordinate,
                options=options,
            )
            if document is None:
                rejections[reason or "unknown"] += 1
            else:
                output.append(document)
    return {
        "filename": filename,
        "documents": output,
        "planned_records": len(coordinates),
        "downloaded_bytes": downloaded_bytes,
        "range_requests": len(spans),
        "rejections": dict(sorted(rejections.items())),
    }


def _url_index_integrity_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".integrity.json")


def _url_index_partial_integrity_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".partial.integrity.json")


def _validate_parquet_footer(path: Path) -> dict[str, Any]:
    """Validate Parquet magic, footer metadata, schema, and URL-index columns."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("FreshWeb URL-index processing requires pyarrow") from exc
    try:
        size = path.stat().st_size
        if size < 12:
            raise FreshWebError("file is too small to contain a Parquet footer")
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                raise FreshWebError("missing leading Parquet magic")
            handle.seek(-4, os.SEEK_END)
            if handle.read(4) != b"PAR1":
                raise FreshWebError("missing trailing Parquet magic")
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        missing = REQUIRED_INDEX_COLUMNS - available
        if missing:
            raise FreshWebError(
                f"URL-index partition is missing required columns: {sorted(missing)}"
            )
        metadata = parquet.metadata
        if metadata is None or metadata.serialized_size <= 0:
            raise FreshWebError("Parquet footer metadata is empty")
        return {
            "num_rows": int(metadata.num_rows),
            "num_row_groups": int(metadata.num_row_groups),
            "serialized_size": int(metadata.serialized_size),
        }
    except FreshWebError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise FreshWebError(f"Invalid URL-index Parquet footer: {path}") from exc
    except Exception as exc:
        # PyArrow uses several implementation-specific exception classes for
        # corrupt Thrift/footer metadata.  Normalize them at this trust boundary.
        raise FreshWebError(f"Invalid URL-index Parquet footer: {path}") from exc


def _read_integrity_sidecar(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshWebError(f"URL-index integrity sidecar is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise FreshWebError(f"URL-index integrity sidecar is not an object: {path}")
    return value


def _verify_cached_url_index(url: str, destination: Path) -> Path:
    sidecar_path = _url_index_integrity_path(destination)
    sidecar = _read_integrity_sidecar(sidecar_path)
    try:
        content_length = int(sidecar["content_length"])
        expected_size = int(sidecar["size"])
        etag = str(sidecar["etag"]).strip()
        expected_sha256 = str(sidecar["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FreshWebError(f"URL-index integrity sidecar is incomplete: {sidecar_path}") from exc
    if (
        sidecar.get("schema") != URL_INDEX_INTEGRITY_SCHEMA
        or sidecar.get("url") != url
        or content_length <= 0
        or expected_size != content_length
        or not etag
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise FreshWebError(f"URL-index integrity sidecar is invalid: {sidecar_path}")
    if not destination.is_file() or destination.stat().st_size != expected_size:
        raise FreshWebError(f"URL-index cache length mismatch: {destination}")
    if _sha256_file(destination) != expected_sha256:
        raise FreshWebError(f"URL-index cache checksum mismatch: {destination}")
    footer = _validate_parquet_footer(destination)
    if sidecar.get("parquet_footer") != footer:
        raise FreshWebError(f"URL-index Parquet footer changed: {destination}")
    return destination


def _discard_cached_url_index(destination: Path) -> None:
    for path in (
        destination,
        _url_index_integrity_path(destination),
        destination.with_name(destination.name + ".partial"),
        _url_index_partial_integrity_path(destination),
    ):
        path.unlink(missing_ok=True)


def _download_resumable_unlocked(http: _HttpClient, url: str, destination: Path) -> Path:
    if destination.exists():
        try:
            return _verify_cached_url_index(url, destination)
        except FreshWebError:
            _discard_cached_url_index(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial_sidecar_path = _url_index_partial_integrity_path(destination)
    partial_sidecar: dict[str, Any] | None = None
    start = partial.stat().st_size if partial.exists() else 0
    if start:
        try:
            partial_sidecar = _read_integrity_sidecar(partial_sidecar_path)
            if (
                partial_sidecar.get("schema") != URL_INDEX_INTEGRITY_SCHEMA
                or partial_sidecar.get("url") != url
                or not str(partial_sidecar.get("etag") or "").strip()
                or int(partial_sidecar.get("content_length", 0)) < start
            ):
                raise FreshWebError("partial URL-index identity mismatch")
        except (FreshWebError, TypeError, ValueError):
            partial.unlink(missing_ok=True)
            partial_sidecar_path.unlink(missing_ok=True)
            partial_sidecar = None
            start = 0

    headers = {"Accept-Encoding": "identity"}
    expected = frozenset({200})
    if start:
        headers["Range"] = f"bytes={start}-"
        headers["If-Range"] = str(partial_sidecar["etag"])
        expected = frozenset({200, 206})
    response = http.get(url, headers=headers, stream=True, expected=expected)
    try:
        etag = str(response.headers.get("ETag") or "").strip()
        content_length_value = str(response.headers.get("Content-Length") or "").strip()
        if not etag:
            raise FreshWebError(f"URL-index response omitted ETag: {url}")
        if not content_length_value.isdigit() or int(content_length_value) <= 0:
            raise FreshWebError(f"URL-index response omitted a valid Content-Length: {url}")
        response_length = int(content_length_value)

        mode = "ab"
        if start and response.status_code == 206:
            content_range = str(response.headers.get("Content-Range") or "")
            match = re.fullmatch(
                rf"bytes\s+{start}-(\d+)/(\d+)", content_range, flags=re.IGNORECASE
            )
            if not match:
                raise FreshWebError(f"Invalid resumed Content-Range for {url}: {content_range!r}")
            end = int(match.group(1))
            total_length = int(match.group(2))
            if end < start or response_length != end - start + 1:
                raise FreshWebError(f"Resumed Content-Length disagrees with Content-Range: {url}")
            if (
                etag != str(partial_sidecar["etag"])
                or total_length != int(partial_sidecar["content_length"])
            ):
                raise FreshWebError(f"URL-index ETag or length changed during resume: {url}")
        elif response.status_code == 200:
            mode = "wb"
            start = 0
            total_length = response_length
        else:
            raise FreshWebError(f"Cannot resume {url}: HTTP {response.status_code}")

        atomic_json(
            partial_sidecar_path,
            {
                "schema": URL_INDEX_INTEGRITY_SCHEMA,
                "url": url,
                "content_length": total_length,
                "etag": etag,
            },
        )
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        response.close()

    if partial.stat().st_size != total_length:
        raise FreshWebError(
            f"URL-index download length mismatch for {url}: "
            f"{partial.stat().st_size} != {total_length}"
        )
    try:
        footer = _validate_parquet_footer(partial)
    except FreshWebError:
        # A transfer can have the advertised byte length yet still contain a
        # corrupt Parquet footer.  Do not leave a full-size partial that would
        # produce an invalid ``Range: bytes=<length>-`` request on retry.
        partial.unlink(missing_ok=True)
        partial_sidecar_path.unlink(missing_ok=True)
        raise
    sha256 = _sha256_file(partial)
    os.replace(partial, destination)
    atomic_json(
        _url_index_integrity_path(destination),
        {
            "schema": URL_INDEX_INTEGRITY_SCHEMA,
            "url": url,
            "content_length": total_length,
            "etag": etag,
            "size": destination.stat().st_size,
            "sha256": sha256,
            "parquet_footer": footer,
            "retrieved_at": utc_now(),
        },
    )
    partial_sidecar_path.unlink(missing_ok=True)
    return destination


def _download_resumable(http: _HttpClient, url: str, destination: Path) -> Path:
    """Download once across concurrent route processes and resume partials."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Portage and Rhea are Unix
        raise RuntimeError("FreshWeb shared-cache locking requires a Unix fcntl implementation") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _download_resumable_unlocked(http, url, destination)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _iter_parquet(path: Path, batch_rows: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("FreshWeb URL-index processing requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    missing = REQUIRED_INDEX_COLUMNS - available
    if missing:
        raise FreshWebError(f"URL-index partition is missing required columns: {sorted(missing)}")
    columns = [name for name in INDEX_COLUMNS if name in available]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS partitions (
            relative_path TEXT PRIMARY KEY,
            crawl TEXT NOT NULL,
            selection_capacity INTEGER NOT NULL DEFAULT 0,
            scanned INTEGER NOT NULL,
            eligible INTEGER NOT NULL,
            selected INTEGER NOT NULL,
            rejections_json TEXT NOT NULL,
            categories_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS url_winners (
            canonical_url TEXT PRIMARY KEY,
            content_digest TEXT NOT NULL,
            priority INTEGER NOT NULL,
            full_body_likelihood INTEGER NOT NULL,
            fetch_epoch INTEGER NOT NULL,
            warc_record_length INTEGER NOT NULL,
            sample_hash TEXT NOT NULL,
            warc_filename TEXT NOT NULL,
            warc_record_offset INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS digest_winners (
            content_digest TEXT PRIMARY KEY,
            canonical_url TEXT NOT NULL,
            priority INTEGER NOT NULL,
            full_body_likelihood INTEGER NOT NULL,
            fetch_epoch INTEGER NOT NULL,
            warc_record_length INTEGER NOT NULL,
            sample_hash TEXT NOT NULL,
            warc_filename TEXT NOT NULL,
            warc_record_offset INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fetch_groups (
            group_id TEXT PRIMARY KEY,
            warc_filename TEXT NOT NULL,
            planned_records INTEGER NOT NULL,
            extracted_records INTEGER NOT NULL,
            estimated_tokens INTEGER NOT NULL,
            license_eligible_tokens INTEGER NOT NULL DEFAULT 0,
            downloaded_bytes INTEGER NOT NULL,
            range_requests INTEGER NOT NULL,
            rejections_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shard_offsets (
            shard_id INTEGER PRIMARY KEY,
            committed_offset INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS digest_winners_warc_position_idx
            ON digest_winners(warc_filename, warc_record_offset);
        """
    )
    partition_columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(partitions)")
    }
    if "selection_capacity" not in partition_columns:
        connection.execute(
            "ALTER TABLE partitions ADD COLUMN selection_capacity INTEGER NOT NULL DEFAULT 0"
        )
    fetch_columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(fetch_groups)")
    }
    if "license_eligible_tokens" not in fetch_columns:
        connection.execute(
            "ALTER TABLE fetch_groups ADD COLUMN license_eligible_tokens INTEGER NOT NULL DEFAULT 0"
        )
    connection.commit()
    return connection


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


URL_UPSERT = """
INSERT INTO url_winners(
    canonical_url, content_digest, priority, full_body_likelihood, fetch_epoch,
    warc_record_length, sample_hash, warc_filename, warc_record_offset, payload_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(canonical_url) DO UPDATE SET
    content_digest=excluded.content_digest,
    priority=excluded.priority,
    full_body_likelihood=excluded.full_body_likelihood,
    fetch_epoch=excluded.fetch_epoch,
    warc_record_length=excluded.warc_record_length,
    sample_hash=excluded.sample_hash,
    warc_filename=excluded.warc_filename,
    warc_record_offset=excluded.warc_record_offset,
    payload_json=excluded.payload_json
WHERE
    excluded.priority > url_winners.priority OR
    (excluded.priority = url_winners.priority AND excluded.full_body_likelihood > url_winners.full_body_likelihood) OR
    (excluded.priority = url_winners.priority AND excluded.full_body_likelihood = url_winners.full_body_likelihood
        AND excluded.fetch_epoch > url_winners.fetch_epoch) OR
    (excluded.priority = url_winners.priority AND excluded.full_body_likelihood = url_winners.full_body_likelihood
        AND excluded.fetch_epoch = url_winners.fetch_epoch AND excluded.warc_record_length > url_winners.warc_record_length) OR
    (excluded.priority = url_winners.priority AND excluded.full_body_likelihood = url_winners.full_body_likelihood
        AND excluded.fetch_epoch = url_winners.fetch_epoch AND excluded.warc_record_length = url_winners.warc_record_length
        AND excluded.sample_hash < url_winners.sample_hash)
"""


def _candidate_values(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["canonical_url"],
        candidate["content_digest"],
        int(candidate["priority"]),
        int(candidate["full_body_likelihood"]),
        int(candidate["fetch_epoch"]),
        int(candidate["warc_record_length"]),
        candidate["sample_hash"],
        candidate["warc_filename"],
        int(candidate["warc_record_offset"]),
        json.dumps(candidate, sort_keys=True, separators=(",", ":")),
    )


def _build_digest_winners(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute("DELETE FROM digest_winners")
        connection.execute(
            """
            INSERT INTO digest_winners(
                content_digest, canonical_url, priority, full_body_likelihood, fetch_epoch,
                warc_record_length, sample_hash, warc_filename, warc_record_offset, payload_json
            )
            SELECT content_digest, canonical_url, priority, full_body_likelihood, fetch_epoch,
                   warc_record_length, sample_hash, warc_filename, warc_record_offset, payload_json
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY content_digest
                    ORDER BY priority DESC, full_body_likelihood DESC, fetch_epoch DESC,
                             warc_record_length DESC, sample_hash ASC
                ) AS winner_rank
                FROM url_winners
            )
            WHERE winner_rank = 1
            """
        )
        _set_metadata(connection, "digest_winners_ready", "true")


def _iter_fetch_groups(connection: sqlite3.Connection) -> Iterator[tuple[str, list[dict[str, Any]], str]]:
    # Keyset pagination leaves no SQLite read cursor open across ``yield``.  The
    # caller commits each fetched WARC group on the same connection, so holding
    # one giant ordered SELECT here would make long resumes fragile.
    last_filename = ""
    while True:
        row = connection.execute(
            """
            SELECT warc_filename
            FROM digest_winners
            WHERE warc_filename > ?
            ORDER BY warc_filename
            LIMIT 1
            """,
            (last_filename,),
        ).fetchone()
        if row is None:
            return
        filename = str(row["warc_filename"])
        payload_rows = connection.execute(
            """
            SELECT payload_json
            FROM digest_winners
            WHERE warc_filename = ?
            ORDER BY warc_record_offset
            """,
            (filename,),
        ).fetchall()
        coordinates = [json.loads(value["payload_json"]) for value in payload_rows]
        identity = "\n".join(
            f"{item['content_digest']}:{item['warc_record_offset']}:{item['warc_record_length']}"
            for item in coordinates
        )
        group_id = hashlib.sha256(f"{filename}\n{identity}".encode("utf-8")).hexdigest()
        last_filename = filename
        if not connection.execute("SELECT 1 FROM fetch_groups WHERE group_id = ?", (group_id,)).fetchone():
            yield filename, coordinates, group_id


def _recover_shards(connection: sqlite3.Connection, shard_root: Path, shard_count: int) -> None:
    shard_root.mkdir(parents=True, exist_ok=True)
    for shard_id in range(shard_count):
        path = shard_root / f"part-{shard_id:05d}.jsonl.zst"
        row = connection.execute(
            "SELECT committed_offset FROM shard_offsets WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        committed = int(row[0]) if row else 0
        if path.exists():
            actual = path.stat().st_size
            if actual < committed:
                raise FreshWebError(f"Shard {path} is shorter than its committed offset")
            if actual > committed:
                with path.open("r+b") as handle:
                    handle.truncate(committed)
                    handle.flush()
                    os.fsync(handle.fileno())
        elif committed:
            raise FreshWebError(f"Committed shard is missing: {path}")


def _compress_jsonl(records: Sequence[dict[str, Any]]) -> bytes:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("FreshWeb output requires zstandard") from exc
    payload = b"".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )
    return zstandard.ZstdCompressor(level=3, write_checksum=True).compress(payload)


def _commit_group(
    connection: sqlite3.Connection,
    shard_root: Path,
    shard_count: int,
    group_id: str,
    result: Mapping[str, Any],
) -> None:
    per_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    estimated_tokens = 0
    license_eligible_tokens = 0
    for document in result["documents"]:
        shard_id = int(document["id"][:16], 16) % shard_count
        per_shard[shard_id].append(document)
        document_tokens = max(1, len(document["text"]) // 4)
        estimated_tokens += document_tokens
        if bool(
            document.get("metadata", {})
            .get("license_evaluation", {})
            .get("decision")
        ):
            license_eligible_tokens += document_tokens
    new_offsets: dict[int, int] = {}
    for shard_id in sorted(per_shard):
        path = shard_root / f"part-{shard_id:05d}.jsonl.zst"
        frame = _compress_jsonl(per_shard[shard_id])
        with path.open("ab") as handle:
            handle.write(frame)
            handle.flush()
            os.fsync(handle.fileno())
            new_offsets[shard_id] = handle.tell()
    with connection:
        for shard_id, offset in new_offsets.items():
            connection.execute(
                """
                INSERT INTO shard_offsets(shard_id, committed_offset) VALUES (?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET committed_offset=excluded.committed_offset
                """,
                (shard_id, offset),
            )
        connection.execute(
            """
            INSERT INTO fetch_groups(
                group_id, warc_filename, planned_records, extracted_records, estimated_tokens,
                license_eligible_tokens, downloaded_bytes, range_requests, rejections_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                result["filename"],
                int(result["planned_records"]),
                len(result["documents"]),
                estimated_tokens,
                license_eligible_tokens,
                int(result["downloaded_bytes"]),
                int(result["range_requests"]),
                json.dumps(result["rejections"], sort_keys=True),
            ),
        )


def _bounded_fetch(
    connection: sqlite3.Connection,
    shard_root: Path,
    *,
    http: _HttpClient,
    options: FreshWebOptions,
) -> None:
    _recover_shards(connection, shard_root, options.shard_count)
    groups = _iter_fetch_groups(connection)
    if options.max_workers == 1:
        for filename, coordinates, group_id in groups:
            result = _fetch_warc_group(filename, coordinates, http=http, options=options)
            _commit_group(connection, shard_root, options.shard_count, group_id, result)
        return

    import concurrent.futures

    pending: deque[tuple[str, concurrent.futures.Future[dict[str, Any]]]] = deque()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=options.max_workers, thread_name_prefix="freshweb-warc"
    ) as pool:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < options.max_workers * 2 and not exhausted:
                try:
                    filename, coordinates, group_id = next(groups)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(
                    (
                        group_id,
                        pool.submit(_fetch_warc_group, filename, coordinates, http=http, options=options),
                    )
                )
            if pending:
                group_id, future = pending.popleft()
                _commit_group(connection, shard_root, options.shard_count, group_id, future.result())


def _receipt(
    connection: sqlite3.Connection,
    *,
    run_root: Path,
    source_id: str,
    candidate_tokens: int,
    policy: OptOutPolicy,
    resolved: Mapping[str, Any],
    options: FreshWebOptions,
    fingerprint: str,
) -> dict[str, Any]:
    partition_totals = connection.execute(
        "SELECT COALESCE(SUM(scanned),0), COALESCE(SUM(eligible),0), COALESCE(SUM(selected),0), COUNT(*) FROM partitions"
    ).fetchone()
    fetch_totals = connection.execute(
        """
        SELECT COALESCE(SUM(planned_records),0), COALESCE(SUM(extracted_records),0),
               COALESCE(SUM(estimated_tokens),0), COALESCE(SUM(license_eligible_tokens),0),
               COALESCE(SUM(downloaded_bytes),0), COALESCE(SUM(range_requests),0), COUNT(*)
        FROM fetch_groups
        """
    ).fetchone()
    metadata_rejections: Counter[str] = Counter()
    for row in connection.execute("SELECT rejections_json FROM partitions"):
        metadata_rejections.update(json.loads(row[0]))
    extraction_rejections: Counter[str] = Counter()
    for row in connection.execute("SELECT rejections_json FROM fetch_groups"):
        extraction_rejections.update(json.loads(row[0]))
    shards = []
    for path in sorted((run_root / "documents").glob("*.jsonl.zst")):
        if not path.stat().st_size:
            continue
        shards.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    estimated_tokens = int(fetch_totals[2])
    license_eligible_tokens = int(fetch_totals[3])
    target_met = estimated_tokens >= candidate_tokens and (
        not options.require_reusable_open_license
        or license_eligible_tokens >= candidate_tokens
    )
    return {
        "schema": FRESHWEB_SCHEMA,
        "status": "complete" if target_met else "retryable_shortfall",
        "source_id": source_id,
        "fingerprint": fingerprint,
        "route": options.route,
        "allowed_categories": list(options.allowed_categories),
        "route_requirements": {
            "structural_quality_evidence": True,
            "english_evidence": options.require_english,
            "reusable_open_license": options.require_reusable_open_license,
            "publication_date": options.route == "fresh_science",
            "version_evidence": options.route in {"software_docs", "official_docs"},
        },
        "completed_at": utc_now(),
        "candidate_token_target": candidate_tokens,
        "estimated_materialized_tokens": estimated_tokens,
        "estimated_license_eligible_tokens": license_eligible_tokens,
        "license_eligibility": {
            "required_for_candidate_target": options.require_reusable_open_license,
            "estimated_tokens": license_eligible_tokens,
            "target_tokens": candidate_tokens,
            "target_met": (
                license_eligible_tokens >= candidate_tokens
                if options.require_reusable_open_license
                else True
            ),
        },
        "candidate_target_met": target_met,
        "ready_for_normalization": target_met,
        # In the acquisition handoff vocabulary, this means the materialized
        # canonical records are ready to enter the CPU normalization/dedup
        # build.  It does not claim these are final packed token shards.
        "ready_for_training_build": target_met,
        "semantic_deduplication": False,
        "crawls": [entry["id"] for entry in resolved["collections"]],
        "url_index_partitions": {
            "resolved": int(resolved["partition_count"]),
            "completed": int(partition_totals[3]),
            "rows_scanned": int(partition_totals[0]),
            "rows_eligible": int(partition_totals[1]),
            "rows_selected": int(partition_totals[2]),
        },
        "exact_selection": {
            "canonical_url_winners": int(connection.execute("SELECT COUNT(*) FROM url_winners").fetchone()[0]),
            "content_digest_winners": int(connection.execute("SELECT COUNT(*) FROM digest_winners").fetchone()[0]),
        },
        "warc": {
            "groups_completed": int(fetch_totals[6]),
            "records_planned": int(fetch_totals[0]),
            "records_extracted": int(fetch_totals[1]),
            "downloaded_bytes": int(fetch_totals[4]),
            "range_requests": int(fetch_totals[5]),
        },
        "metadata_rejections": dict(sorted(metadata_rejections.items())),
        "extraction_rejections": dict(sorted(extraction_rejections.items())),
        "opt_out_registry": {
            "parser_version": OPT_OUT_PARSER_VERSION,
            "sha256": policy.snapshot_sha256,
            "last_updated": policy.last_updated,
            "domains": len(policy.domains),
            "url_paths": len(policy.url_paths),
            "url_rules": len(policy.url_rules),
            "input_entries": policy.input_entries,
            "unparsed_entries": policy.unparsed_entries,
        },
        "extractor_version": EXTRACTOR_VERSION,
        "options": asdict(options),
        "shards": shards,
        "local_path": str(run_root / "documents"),
    }


def _load_verified_receipt(path: Path, *, fingerprint: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshWebError(f"Existing FreshWeb receipt is unreadable: {path}") from exc
    if (
        receipt.get("schema") != FRESHWEB_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("fingerprint") != fingerprint
        or receipt.get("candidate_target_met") is not True
        or receipt.get("ready_for_training_build") is not True
    ):
        raise FreshWebError(f"Existing FreshWeb receipt has an invalid identity: {path}")
    document_root = (path.parent / "documents").resolve()
    for shard in receipt.get("shards", []):
        try:
            shard_path = Path(str(shard["path"])).resolve()
            shard_path.relative_to(document_root)
            expected_size = int(shard["size"])
            expected_sha256 = str(shard["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FreshWebError(f"Existing FreshWeb receipt contains an invalid shard entry: {path}") from exc
        if not shard_path.is_file() or shard_path.stat().st_size != expected_size:
            raise FreshWebError(f"Existing FreshWeb shard is missing or has the wrong size: {shard_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or _sha256_file(shard_path) != expected_sha256:
            raise FreshWebError(f"Existing FreshWeb shard checksum failed: {shard_path}")
    return receipt


def _retry_selection_round(path: Path, *, fingerprint: str) -> int:
    if not path.exists():
        return 0
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
        selection_round = int(progress["next_selection_round"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise FreshWebError(f"FreshWeb retry state is unreadable: {path}") from exc
    if (
        progress.get("schema") != FRESHWEB_PROGRESS_SCHEMA
        or progress.get("status") != "retryable_shortfall"
        or progress.get("fingerprint") != fingerprint
        or selection_round < 0
    ):
        raise FreshWebError(f"FreshWeb retry state has an invalid identity: {path}")
    return selection_round


def _prepare_index_selection_round(
    connection: sqlite3.Connection,
    *,
    selection_round: int,
    policy_sha256: str = "",
) -> None:
    """Start one exact, crash-resumable URL-index selection round.

    A larger taxonomy quota is not guaranteed to be a strict superset of the
    preceding round's quota in every category.  Clear the old winner set once,
    then mark every partition pending so an interrupted round resumes only the
    partitions that have not yet been rescanned at the new capacity.
    """

    value = f"{selection_round}:{policy_sha256}"
    if _metadata_value(connection, "index_selection_round") == value:
        return
    with connection:
        connection.execute("DELETE FROM digest_winners")
        connection.execute("DELETE FROM url_winners")
        connection.execute("UPDATE partitions SET selection_capacity = 0")
        _set_metadata(connection, "index_selection_round", value)


def _prepare_fetch_selection_round(
    connection: sqlite3.Connection,
    *,
    selection_round: int,
    policy_sha256: str = "",
) -> None:
    """Reset materialized output once when the exact winner set is widened.

    Existing documents may no longer be the priority winner after a larger
    deterministic URL-index sample is admitted.  Clearing the committed fetch
    ledger makes the next shard recovery truncate old frames and rebuild the
    selected set exactly, while a crash after this transaction remains
    restartable within the same round.
    """

    value = f"{selection_round}:{policy_sha256}"
    if _metadata_value(connection, "fetch_selection_round") == value:
        return
    with connection:
        connection.execute("DELETE FROM fetch_groups")
        connection.execute("DELETE FROM shard_offsets")
        _set_metadata(connection, "fetch_selection_round", value)


def materialize_freshweb(
    source: Mapping[str, Any],
    *,
    root: str | Path,
    cache_root: str | Path | None = None,
    options: FreshWebOptions | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    """Materialize a bounded Common Crawl FreshWeb source.

    Parameters
    ----------
    source:
        A resolved source-lock item containing ``source_id``, ``candidate_tokens``
        and ``access.crawls``.  The existing manifest's ``url_index`` and
        ``warc_root`` values are accepted but the HTTP endpoint in ``options``
        is used for unauthenticated acquisition.
    root:
        The assigned Metis data root on Lustre.  Source output is written below
        ``raw/<source_id>/freshweb``; reusable URL-index files stay in the
        separately bounded cache.
    cache_root:
        Optional shared Common Crawl cache beneath ``root``.  Separate route
        materializers can safely reuse it; per-file advisory locks prevent
        concurrent writers from racing.
    options:
        Site bounds and injectable endpoints.  Defaults cap network concurrency
        at Common Crawl's documented polite value of ten.
    session:
        Optional requests-compatible session, intended for focused tests.  Use
        ``max_workers=1`` with a non-thread-safe injected session.

    Returns
    -------
    dict
        An immutable acquisition receipt when the qualified candidate target is
        met, otherwise retryable progress state. Exact final-token accounting
        remains a downstream tokenizer job.
    """

    options = options or FreshWebOptions()
    options.validate()
    source_id = str(source.get("source_id") or source.get("id") or "").strip()
    if not source_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id):
        raise ValueError("FreshWeb source_id is missing or unsafe")
    candidate_tokens = int(source.get("candidate_tokens") or 0)
    if candidate_tokens <= 0:
        raise ValueError("FreshWeb candidate_tokens must be positive")
    license_contract = source.get("license")
    if not isinstance(license_contract, Mapping):
        raise ValueError("FreshWeb source license mapping is missing")
    license_status = str(license_contract.get("status") or "").strip()
    if not license_status:
        raise ValueError("FreshWeb source license status is missing")
    options = replace(
        options,
        require_reusable_open_license=license_status == "per_record_required",
    )
    options.validate()
    access = source.get("access")
    if not isinstance(access, Mapping):
        raise ValueError("FreshWeb source access mapping is missing")
    source_route = str(access.get("route") or options.route)
    if source_route != options.route:
        raise ValueError(
            f"FreshWeb source route {source_route!r} does not match options route {options.route!r}"
        )
    crawls = tuple(str(value) for value in access.get("crawls", ()))
    if not crawls or len(set(crawls)) != len(crawls):
        raise ValueError("FreshWeb crawls must be a non-empty unique sequence")
    if any(not re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", crawl) for crawl in crawls):
        raise ValueError("FreshWeb contains an invalid Common Crawl identifier")

    assigned_root = Path(root).expanduser().resolve()
    shared_cache_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else assigned_root / "cache" / "common-crawl"
    )
    try:
        shared_cache_root.relative_to(assigned_root)
    except ValueError as exc:
        raise ValueError("FreshWeb cache_root must be beneath the assigned data root") from exc
    base_root = assigned_root / "raw" / source_id / "freshweb"
    base_root.mkdir(parents=True, exist_ok=True)
    http = _HttpClient(options, session)
    policy, opt_out_snapshot = _snapshot_opt_out(base_root, http, options.opt_out_csv_url)
    partitions, resolved, listings = resolve_common_crawl_paths(
        crawls,
        http=http,
        data_root=options.data_root,
        collinfo_url=options.collinfo_url,
    )
    fingerprint_payload = {
        "schema": FRESHWEB_SCHEMA,
        "extractor_version": EXTRACTOR_VERSION,
        "source_id": source_id,
        "candidate_tokens": candidate_tokens,
        "crawls": crawls,
        "license": dict(license_contract),
        "options": asdict(options),
        # The live registry is deliberately not part of run identity. It is
        # rebound below and forces an in-place winner/output rebuild when it
        # changes, so a multi-day widening resume does not strand a new run.
        "opt_out_policy": "live-common-crawl-registry-revalidated-on-every-run",
        "listings": resolved["listing_sha256"],
    }
    fingerprint = _sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    run_root = base_root / "runs" / fingerprint
    receipt_path = run_root / "ACQUISITION_RECEIPT.json"
    completed_round: int | None = None
    if receipt_path.exists():
        existing_receipt = _load_verified_receipt(receipt_path, fingerprint=fingerprint)
        if (
            existing_receipt.get("opt_out_registry", {}).get("sha256")
            == policy.snapshot_sha256
        ):
            return existing_receipt
        completed_round = int(existing_receipt.get("selection_round", 0))
        receipt_path.unlink()
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "ACQUISITION_PROGRESS.json"
    selection_round = (
        completed_round
        if completed_round is not None
        else _retry_selection_round(progress_path, fingerprint=fingerprint)
    )
    atomic_json(run_root / "SOURCE.json", dict(fingerprint_payload))
    atomic_json(
        run_root / "ACTIVE_OPT_OUT.json",
        {
            "schema": "metis.freshweb-active-opt-out/v1",
            "source": options.opt_out_csv_url,
            "sha256": policy.snapshot_sha256,
            "last_updated": policy.last_updated,
            "selection_round": selection_round,
            "checked_at": utc_now(),
        },
    )
    atomic_json(run_root / "CRAWLS.json", dict(resolved))
    compliance = run_root / "compliance"
    compliance.mkdir(parents=True, exist_ok=True)
    if not (compliance / opt_out_snapshot.name).exists():
        _atomic_bytes(compliance / opt_out_snapshot.name, opt_out_snapshot.read_bytes())
    listings_root = run_root / "url-index" / "listings"
    for crawl, payload in listings.items():
        _atomic_bytes(listings_root / f"{crawl}.paths.gz", payload)

    connection = _connect_state(run_root / "state.sqlite3")
    try:
        existing_fingerprint = _metadata_value(connection, "fingerprint")
        if existing_fingerprint and existing_fingerprint != fingerprint:
            raise FreshWebError("FreshWeb state fingerprint mismatch")
        with connection:
            _set_metadata(connection, "fingerprint", fingerprint)
            _set_metadata(connection, "opt_out_sha256", policy.snapshot_sha256)

        derived_capacity = math.ceil(
            candidate_tokens
            / options.estimated_tokens_per_document
            / len(partitions)
            * options.selection_oversample
        )
        base_partition_capacity = options.max_records_per_partition or max(1, derived_capacity)
        partition_capacity = (
            base_partition_capacity
            if options.max_records_per_partition is not None
            else base_partition_capacity * (2**selection_round)
        )
        _prepare_index_selection_round(
            connection,
            selection_round=selection_round,
            policy_sha256=policy.snapshot_sha256,
        )
        index_cache = shared_cache_root / "url-index"
        for partition in partitions:
            previous = connection.execute(
                "SELECT selection_capacity FROM partitions WHERE relative_path = ?",
                (partition.relative_path,),
            ).fetchone()
            if previous is not None and int(previous["selection_capacity"]) >= partition_capacity:
                continue
            local_path = (
                index_cache
                / partition.crawl
                / partition.listing_sha256
                / Path(partition.relative_path).name
            )
            _download_resumable(http, partition.url, local_path)
            selected, stats = select_partition_candidates(
                _iter_parquet(local_path, options.parquet_batch_rows),
                crawl=partition.crawl,
                source_id=source_id,
                policy=policy,
                options=options,
                capacity=partition_capacity,
            )
            with connection:
                connection.executemany(URL_UPSERT, (_candidate_values(candidate) for candidate in selected))
                connection.execute(
                    """
                    INSERT INTO partitions(
                        relative_path, crawl, selection_capacity, scanned, eligible, selected,
                        rejections_json, categories_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                        crawl=excluded.crawl,
                        selection_capacity=excluded.selection_capacity,
                        scanned=excluded.scanned,
                        eligible=excluded.eligible,
                        selected=excluded.selected,
                        rejections_json=excluded.rejections_json,
                        categories_json=excluded.categories_json
                    """,
                    (
                        partition.relative_path,
                        partition.crawl,
                        partition_capacity,
                        stats["scanned"],
                        stats["eligible"],
                        stats["selected"],
                        json.dumps(stats["rejections"], sort_keys=True),
                        json.dumps(stats["selected_categories"], sort_keys=True),
                    ),
                )
            if not options.keep_index_files:
                local_path.unlink(missing_ok=True)
                _url_index_integrity_path(local_path).unlink(missing_ok=True)
        completed_partitions = int(
            connection.execute(
                "SELECT COUNT(*) FROM partitions WHERE selection_capacity >= ?",
                (partition_capacity,),
            ).fetchone()[0]
        )
        if completed_partitions != len(partitions):
            raise FreshWebError(
                f"URL-index partition state is incomplete: {completed_partitions} of {len(partitions)}"
            )
        with connection:
            _set_metadata(connection, "selection_round", str(selection_round))
            _set_metadata(connection, "selection_capacity", str(partition_capacity))
        _build_digest_winners(connection)
        _prepare_fetch_selection_round(
            connection,
            selection_round=selection_round,
            policy_sha256=policy.snapshot_sha256,
        )
        _bounded_fetch(connection, run_root / "documents", http=http, options=options)
        receipt = _receipt(
            connection,
            run_root=run_root,
            source_id=source_id,
            candidate_tokens=candidate_tokens,
            policy=policy,
            resolved=resolved,
            options=options,
            fingerprint=fingerprint,
        )
        receipt["selection_round"] = selection_round
        receipt["selection_capacity_per_partition"] = partition_capacity
        if receipt["candidate_target_met"]:
            atomic_json(receipt_path, receipt)
            progress_path.unlink(missing_ok=True)
            atomic_json(
                base_root / "LATEST.json",
                {
                    "fingerprint": fingerprint,
                    "status": "complete",
                    "receipt": str(receipt_path),
                },
            )
            return receipt

        saturated_partitions = int(
            connection.execute(
                "SELECT COUNT(*) FROM partitions WHERE selected >= selection_capacity"
            ).fetchone()[0]
        )
        can_widen = (
            options.max_records_per_partition is None and saturated_partitions > 0
        )
        next_selection_round = selection_round + 1 if can_widen else selection_round
        progress = {
            **receipt,
            "schema": FRESHWEB_PROGRESS_SCHEMA,
            "status": "retryable_shortfall",
            "fingerprint": fingerprint,
            "next_selection_round": next_selection_round,
            "automatic_widening": can_widen,
            "saturated_partitions": saturated_partitions,
            "retry_guidance": (
                "rerun the same acquisition task to double the deterministic per-partition sample"
                if can_widen
                else "increase max_records_per_partition in a reviewed profile and create a new run fingerprint"
            ),
            "checked_at": utc_now(),
        }
        atomic_json(progress_path, progress)
        atomic_json(
            base_root / "LATEST.json",
            {
                "fingerprint": fingerprint,
                "status": "retryable_shortfall",
                "progress": str(progress_path),
            },
        )
        return progress
    finally:
        connection.close()


__all__ = [
    "FreshWebError",
    "FreshWebOptions",
    "materialize_freshweb",
    "snapshot_common_crawl_opt_out",
]
