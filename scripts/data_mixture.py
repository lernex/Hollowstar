from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import pyarrow.parquet as pq
import requests
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url


SOFTWARE_HERITAGE_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
PG19_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
PG19_METADATA_URL = PG19_ASSET_ROOT_URL + "metadata.csv"
USER_AGENT = "metis-data-pipeline/1.1"
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTIBLANK_RE = re.compile(r"\n{3,}")


def hf_request_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    streaming: bool = True
    weight: float = 1.0
    text_column: str = "text"
    id_column: str | None = None
    loader: str = "text"
    blob_id_column: str = "blob_id"
    min_chars: int = 0
    max_chars: int | None = None
    min_alpha_ratio: float | None = None
    max_repeat_char_run: int | None = 48
    max_line_length: int | None = None
    max_url_count: int | None = None
    normalize_whitespace: bool = True
    text_template: str | None = None
    repo_parquet_prefix: str | None = None
    trust_remote_code: bool = False
    bucket: str | None = None
    target_tokens: int | None = None
    max_tokens: int | None = None
    filters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=raw["name"],
            dataset_name=raw["dataset_name"],
            dataset_config=raw.get("dataset_config"),
            split=raw.get("split", "train"),
            streaming=bool(raw.get("streaming", True)),
            weight=float(raw.get("weight", 1.0)),
            text_column=raw.get("text_column", "text"),
            id_column=raw.get("id_column"),
            loader=raw.get("loader", "text"),
            blob_id_column=raw.get("blob_id_column", "blob_id"),
            min_chars=int(raw.get("min_chars", 0)),
            max_chars=raw.get("max_chars"),
            min_alpha_ratio=raw.get("min_alpha_ratio"),
            max_repeat_char_run=raw.get("max_repeat_char_run", 48),
            max_line_length=raw.get("max_line_length"),
            max_url_count=raw.get("max_url_count"),
            normalize_whitespace=bool(raw.get("normalize_whitespace", True)),
            text_template=raw.get("text_template"),
            repo_parquet_prefix=raw.get("repo_parquet_prefix"),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            bucket=raw.get("bucket"),
            target_tokens=raw.get("target_tokens"),
            max_tokens=raw.get("max_tokens"),
            filters=raw.get("filters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "streaming": self.streaming,
            "weight": self.weight,
            "text_column": self.text_column,
            "id_column": self.id_column,
            "loader": self.loader,
            "blob_id_column": self.blob_id_column,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_alpha_ratio": self.min_alpha_ratio,
            "max_repeat_char_run": self.max_repeat_char_run,
            "max_line_length": self.max_line_length,
            "max_url_count": self.max_url_count,
            "normalize_whitespace": self.normalize_whitespace,
            "text_template": self.text_template,
            "repo_parquet_prefix": self.repo_parquet_prefix,
            "trust_remote_code": self.trust_remote_code,
            "bucket": self.bucket,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "filters": self.filters,
        }


def source_weight(raw: dict[str, Any]) -> float:
    for key in ("weight", "target_tokens", "target_examples", "target_pairs"):
        value = raw.get(key)
        if value is not None:
            return float(value)
    return 1.0


def flattened_sources(raw_config: dict[str, Any]) -> list[dict[str, Any]]:
    global_filters = dict(raw_config.get("global_filters") or {})
    raw_sources = raw_config.get("sources")
    if raw_sources:
        flattened = []
        for source in raw_sources:
            item = dict(source)
            item.setdefault("weight", source_weight(item))
            filters = dict(global_filters)
            filters.update(item.get("filters") or {})
            if filters:
                item["filters"] = filters
            flattened.append(item)
        return flattened

    buckets = raw_config.get("buckets") or []
    flattened: list[dict[str, Any]] = []
    for bucket in buckets:
        bucket_name = bucket.get("name") or bucket.get("bucket")
        for source in bucket.get("sources") or []:
            item = dict(source)
            item.setdefault("bucket", bucket_name)
            item.setdefault("weight", source_weight(item))
            filters = dict(global_filters)
            filters.update(item.get("filters") or {})
            if filters:
                item["filters"] = filters
            if bucket.get("fallback_policy") and "fallback_policy" not in item:
                item["fallback_policy"] = bucket["fallback_policy"]
            flattened.append(item)

    if flattened:
        return flattened
    raise ValueError("Mixture config must include either `sources` or non-empty `buckets`.")


def load_source_specs_from_config(raw_config: dict[str, Any]) -> list[SourceSpec]:
    sources = flattened_sources(raw_config)
    if any(source_weight(spec) <= 0 for spec in sources):
        raise ValueError("All source weights must be positive.")
    return [SourceSpec.from_dict(item) for item in sources]


@dataclass
class SourceState:
    spec: SourceSpec
    iterator: Any
    attempted_rows: int = 0
    emitted_examples: int = 0
    skipped_examples: int = 0
    fetch_errors: int = 0

    def next_example(self, fetcher: "SoftwareHeritageFetcher") -> dict[str, str] | None:
        while True:
            try:
                row = next(self.iterator)
            except StopIteration:
                return None

            self.attempted_rows += 1
            if not passes_row_filters(row, self.spec):
                self.skipped_examples += 1
                continue

            try:
                text = extract_text(row, self.spec, fetcher)
            except Exception:
                self.fetch_errors += 1
                continue

            if not text:
                self.skipped_examples += 1
                continue

            stripped = clean_text(text, self.spec)
            if len(stripped) < self.spec.min_chars:
                self.skipped_examples += 1
                continue

            if self.spec.max_chars is not None:
                stripped = stripped[: self.spec.max_chars].strip()
                if not stripped:
                    self.skipped_examples += 1
                    continue

            if not passes_quality_filters(stripped, self.spec):
                self.skipped_examples += 1
                continue

            self.emitted_examples += 1
            return {
                "source": self.spec.name,
                "doc_id": extract_doc_id(row, self.spec, self.attempted_rows),
                "text": stripped,
            }


class SoftwareHeritageFetcher:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, blob_id: str) -> str:
        last_error: Exception | None = None
        url = SOFTWARE_HERITAGE_URL.format(blob_id=blob_id)
        for attempt in range(5):
            try:
                response = self.session.get(url, timeout=45)
                response.raise_for_status()
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_handle:
                    return gz_handle.read().decode("utf-8", errors="ignore")
            except (OSError, requests.RequestException) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Failed to fetch Software Heritage blob {blob_id}") from last_error


class DatasetMixture:
    def __init__(self, config_path: str | Path, total_examples: int, seed: int | None = None) -> None:
        self.config_path = str(Path(config_path))
        self.raw_config = json.loads(Path(config_path).read_text())
        self.sources = load_source_specs_from_config(self.raw_config)

        self.total_examples = int(total_examples)
        self.seed = int(self.raw_config.get("seed", 42) if seed is None else seed)
        self.planned_counts = build_planned_counts(
            [spec.weight for spec in self.sources],
            self.total_examples,
        )
        self.fetcher = SoftwareHeritageFetcher()
        self.states = [SourceState(spec=spec, iterator=iter(load_source_dataset(spec))) for spec in self.sources]
        self.dead_sources: set[int] = set()

    def same_bucket_indices(self, source_index: int) -> list[int]:
        bucket = self.sources[source_index].bucket
        if not bucket:
            return [source_index]
        return [
            index
            for index, spec in enumerate(self.sources)
            if spec.bucket == bucket
        ]

    def __iter__(self):
        for source_index in iter_schedule(self.planned_counts, self.seed):
            for candidate_index in self.same_bucket_indices(source_index):
                if candidate_index in self.dead_sources:
                    continue
                example = self.states[candidate_index].next_example(self.fetcher)
                if example is None:
                    self.dead_sources.add(candidate_index)
                    continue
                yield example
                break

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "seed": self.seed,
            "total_examples_requested": self.total_examples,
            "planned_counts": {
                self.sources[index].name: count for index, count in sorted(self.planned_counts.items())
            },
            "sources": [spec.to_dict() for spec in self.sources],
            "source_stats": {
                state.spec.name: {
                    "attempted_rows": state.attempted_rows,
                    "emitted_examples": state.emitted_examples,
                    "skipped_examples": state.skipped_examples,
                    "fetch_errors": state.fetch_errors,
                    "exhausted": index in self.dead_sources,
                }
                for index, state in enumerate(self.states)
            },
        }


def truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def infer_repo_parquet_prefix(spec: SourceSpec) -> str | None:
    if spec.repo_parquet_prefix:
        return spec.repo_parquet_prefix
    if spec.dataset_name == "nvidia/Nemotron-CC-v2":
        return "High-Quality/"
    if spec.dataset_name == "EssentialAI/essential-web-v1.0":
        return "data/"
    if spec.dataset_name == "Zyphra/Zyda-2" and spec.dataset_config == "sample-100BT":
        return "sample/100BT/"
    if spec.dataset_name == "Zyphra/Zyda-2":
        return "data/"
    if spec.dataset_name == "openbmb/Ultra-FineWeb":
        return "data/ultrafineweb_en_v1_4/"
    if spec.dataset_name == "HuggingFaceFW/finewiki" and spec.dataset_config in {"en", "enwiki"}:
        return "data/enwiki/"
    if spec.dataset_name == "HuggingFaceFW/finepdfs":
        return "data/eng_Latn/train/"
    if spec.dataset_name == "LLM360/MegaMath":
        return "megamath-web-pro/"
    if spec.dataset_name == "PleIAs/common_corpus":
        return "common_corpus_"
    if spec.dataset_name == "bigscience-data/roots_en_no_code_stackexchange":
        return "data/"
    if spec.dataset_name == "HuggingFaceFW/fineweb-edu" and spec.dataset_config and spec.dataset_config.startswith("sample-"):
        suffix = spec.dataset_config.removeprefix("sample-")
        return f"sample/{suffix}/"
    if spec.dataset_config:
        return f"{spec.dataset_config}/"
    if spec.dataset_name in {"HuggingFaceTB/dclm-edu", "epfml/FineWeb-HQ"}:
        return "data/"
    return None


def parquet_cache_limit_bytes() -> int:
    raw = os.environ.get("METIS_PARQUET_CACHE_LIMIT_GB", "0").strip()
    if not raw:
        return 0
    return max(0, int(float(raw) * (1024**3)))


def parquet_prefetch_count() -> int:
    raw = os.environ.get("METIS_PARQUET_PREFETCH_COUNT", "8").strip()
    return max(1, int(raw or "8"))


def parquet_cache_root() -> Path:
    base = os.environ.get("METIS_PARQUET_CACHE_ROOT")
    if base:
        root = Path(base)
    else:
        root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "metis_parquet_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def shard_content_length(session: requests.Session, url: str) -> int:
    try:
        response = session.head(url, allow_redirects=True, timeout=(20, 120))
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length:
            return int(length)
    except requests.RequestException:
        pass
    return 0


@dataclass
class DownloadedParquetShard:
    repo_path: str
    local_path: Path
    size_bytes: int


class ParquetShardPrefetcher:
    def __init__(self, *, spec: SourceSpec, parquet_files: list[str]) -> None:
        self.spec = spec
        self.parquet_files = parquet_files
        self.session = requests.Session()
        self.session.headers.update(hf_request_headers())
        self.cache_limit_bytes = parquet_cache_limit_bytes()
        self.prefetch_count = parquet_prefetch_count()
        self.queue: deque[DownloadedParquetShard] = deque()
        self.queue_bytes = 0
        self.index = 0
        self.closed = False
        self.error: Exception | None = None
        cache_dir = parquet_cache_root() / f"{spec.name}_pid{os.getpid()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._producer, name=f"metis-prefetch-{spec.name}", daemon=True)
        self.thread.start()

    def _producer(self) -> None:
        try:
            while True:
                with self.condition:
                    while (
                        self.index < len(self.parquet_files)
                        and (
                            len(self.queue) >= self.prefetch_count
                            or (
                                self.cache_limit_bytes > 0
                                and self.queue_bytes >= self.cache_limit_bytes
                            )
                        )
                    ):
                        self.condition.wait()

                    if self.index >= len(self.parquet_files):
                        self.closed = True
                        self.condition.notify_all()
                        return

                    repo_path = self.parquet_files[self.index]
                    self.index += 1

                url = hf_hub_url(repo_id=self.spec.dataset_name, filename=repo_path, repo_type="dataset")
                expected_size = shard_content_length(self.session, url)

                with self.condition:
                    while (
                        self.queue
                        and self.cache_limit_bytes > 0
                        and expected_size > 0
                        and self.queue_bytes + expected_size > self.cache_limit_bytes
                    ):
                        self.condition.wait()

                local_path = self.cache_dir / repo_path.replace("/", "__")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"[{self.spec.name}] Prefetching parquet shard {self.index}/{len(self.parquet_files)}: {repo_path}",
                    flush=True,
                )
                downloaded_bytes = self._download_with_retries(url, local_path)

                with self.condition:
                    self.queue.append(
                        DownloadedParquetShard(
                            repo_path=repo_path,
                            local_path=local_path,
                            size_bytes=downloaded_bytes,
                        )
                    )
                    self.queue_bytes += downloaded_bytes
                    self.condition.notify_all()
        except Exception as exc:  # pragma: no cover - defensive path for live network failures
            with self.condition:
                self.error = exc
                self.closed = True
                self.condition.notify_all()

    def __iter__(self):
        try:
            while True:
                with self.condition:
                    while not self.queue and not self.closed and self.error is None:
                        self.condition.wait()
                    if self.error is not None:
                        raise RuntimeError(
                            f"Parquet prefetch failed for source {self.spec.name}"
                        ) from self.error
                    if not self.queue and self.closed:
                        return
                    shard = self.queue.popleft()

                parquet = pq.ParquetFile(shard.local_path)
                try:
                    for batch in parquet.iter_batches(batch_size=1024):
                        for row in batch.to_pylist():
                            yield row
                finally:
                    shard.local_path.unlink(missing_ok=True)
                    with self.condition:
                        self.queue_bytes = max(0, self.queue_bytes - shard.size_bytes)
                        self.condition.notify_all()
        finally:
            self.thread.join(timeout=1)
            for leftover in list(self.cache_dir.glob("*")):
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)

    def _download_with_retries(self, url: str, local_path: Path) -> int:
        last_error: Exception | None = None
        for attempt in range(5):
            downloaded_bytes = 0
            try:
                local_path.unlink(missing_ok=True)
                with self.session.get(url, stream=True, timeout=(30, 900)) as response:
                    response.raise_for_status()
                    with local_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                downloaded_bytes += len(chunk)
                if downloaded_bytes > 0:
                    return downloaded_bytes
            except requests.RequestException as exc:
                last_error = exc
                local_path.unlink(missing_ok=True)
                sleep_for = min(2 ** attempt, 16)
                print(
                    f"[{self.spec.name}] Parquet shard download retry {attempt + 1}/5 after {type(exc).__name__}: {url}",
                    flush=True,
                )
                time.sleep(sleep_for)
        raise RuntimeError(f"Failed to download parquet shard after retries: {url}") from last_error


def iter_remote_parquet_rows(spec: SourceSpec):
    prefix = infer_repo_parquet_prefix(spec)
    configured_prefixes = tuple((spec.filters or {}).get("parquet_prefixes") or ())
    prefixes = configured_prefixes or ((prefix,) if prefix else ())
    if not prefixes:
        raise ValueError(f"No parquet prefix configured for incremental source {spec.name}")

    parquet_files = repo_files_matching(
        spec,
        suffixes=(".parquet",),
        prefixes=prefixes,
    )
    parquet_files = partitioned_files(parquet_files, spec)
    if not parquet_files:
        if (spec.filters or {}).get("_allow_empty_partition"):
            return
        print(
            f"[{spec.name}] No parquet shard files found under prefixes {prefixes!r}; "
            "falling back to HF streaming.",
            flush=True,
        )
        yield from load_dataset(
            spec.dataset_name,
            name=spec.dataset_config,
            split=spec.split,
            streaming=True,
            trust_remote_code=spec.trust_remote_code,
        )
        return
    yield from ParquetShardPrefetcher(spec=spec, parquet_files=parquet_files)


def repo_files_matching(
    spec: SourceSpec,
    *,
    suffixes: tuple[str, ...],
    prefixes: tuple[str, ...] = (),
) -> list[str]:
    api = HfApi(token=get_token())
    matches: list[str] = []
    if prefixes:
        for prefix in prefixes:
            try:
                tree = api.list_repo_tree(
                    spec.dataset_name,
                    repo_type="dataset",
                    path_in_repo=prefix.rstrip("/"),
                    recursive=True,
                )
                matches.extend(
                    item.path
                    for item in tree
                    if item.path.endswith(suffixes) and item.path.startswith(prefix)
                )
            except Exception:
                repo_files = api.list_repo_files(spec.dataset_name, repo_type="dataset")
                matches.extend(
                    path
                    for path in repo_files
                    if path.endswith(suffixes) and path.startswith(prefix)
                )
    else:
        repo_files = api.list_repo_files(spec.dataset_name, repo_type="dataset")
        matches.extend(path for path in repo_files if path.endswith(suffixes))
    return filter_repo_files(sorted(set(matches)), spec)


def normalize_file_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def filter_repo_files(files: list[str], spec: SourceSpec) -> list[str]:
    filters = spec.filters or {}
    include_terms = [
        normalize_file_match_text(str(term))
        for term in filters.get("file_name_include_any", [])
        if str(term).strip()
    ]
    exclude_terms = [
        normalize_file_match_text(str(term))
        for term in filters.get("file_name_exclude_any", [])
        if str(term).strip()
    ]
    if not include_terms and not exclude_terms:
        return files

    kept: list[str] = []
    for path in files:
        match_text = normalize_file_match_text(path)
        if include_terms and not any(term in match_text for term in include_terms):
            continue
        if exclude_terms and any(term in match_text for term in exclude_terms):
            continue
        kept.append(path)

    dropped = len(files) - len(kept)
    print(
        f"[{spec.name}] File-name filters kept {len(kept)}/{len(files)} files "
        f"(dropped {dropped}).",
        flush=True,
    )
    return kept


def partitioned_files(files: list[str], spec: SourceSpec) -> list[str]:
    filters = spec.filters or {}
    partition_count = int(filters.get("file_partition_count") or 1)
    partition_index = int(filters.get("file_partition_index") or 0)
    if partition_count <= 1:
        return files
    if partition_index < 0 or partition_index >= partition_count:
        raise ValueError(
            f"Invalid file partition {partition_index}/{partition_count} for source {spec.name}"
        )
    return [
        path
        for file_index, path in enumerate(files)
        if file_index % partition_count == partition_index
    ]


def iter_remote_jsonl_rows(spec: SourceSpec, prefixes: tuple[str, ...]):
    files = list((spec.filters or {}).get("jsonl_files") or [])
    if not files:
        files = repo_files_matching(spec, suffixes=(".jsonl",), prefixes=prefixes)
    files = partitioned_files(files, spec)
    if not files:
        if (spec.filters or {}).get("_allow_empty_partition"):
            return
        raise RuntimeError(f"No jsonl files found for source {spec.name} with prefixes {prefixes}")

    session = requests.Session()
    session.headers.update(hf_request_headers())
    for file_index, repo_path in enumerate(files, start=1):
        print(
            f"[{spec.name}] Streaming jsonl shard {file_index}/{len(files)}: {repo_path}",
            flush=True,
        )
        url = hf_hub_url(repo_id=spec.dataset_name, filename=repo_path, repo_type="dataset")
        with session.get(url, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            for line_index, line in enumerate(response.iter_lines(decode_unicode=True), start=1):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row.setdefault("id", f"{repo_path}:{line_index}")
                yield row


def iter_remote_jsonl_zst_rows(
    spec: SourceSpec,
    prefixes: tuple[str, ...],
    *,
    suffixes: tuple[str, ...] = (".jsonl.zst",),
):
    files = repo_files_matching(spec, suffixes=suffixes, prefixes=prefixes)
    files = partitioned_files(files, spec)
    if not files:
        if (spec.filters or {}).get("_allow_empty_partition"):
            return
        raise RuntimeError(f"No jsonl.zst files found for source {spec.name} with prefixes {prefixes}")

    session = requests.Session()
    session.headers.update(hf_request_headers())
    dctx = zstd.ZstdDecompressor()
    for file_index, repo_path in enumerate(files, start=1):
        print(
            f"[{spec.name}] Streaming zstd jsonl shard {file_index}/{len(files)}: {repo_path}",
            flush=True,
        )
        url = hf_hub_url(repo_id=spec.dataset_name, filename=repo_path, repo_type="dataset")
        with session.get(url, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            with dctx.stream_reader(response.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                for line_index, line in enumerate(text_stream, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.setdefault("id", f"{repo_path}:{line_index}")
                    yield row


def iter_remote_json_gz_rows(spec: SourceSpec, prefixes: tuple[str, ...] = ()):
    files = repo_files_matching(spec, suffixes=(".json.gz", ".jsonl.gz"), prefixes=prefixes)
    files = partitioned_files(files, spec)
    if not files:
        if (spec.filters or {}).get("_allow_empty_partition"):
            return
        raise RuntimeError(f"No gzip json files found for source {spec.name}")

    session = requests.Session()
    session.headers.update(hf_request_headers())
    chunk_long_documents = bool((spec.filters or {}).get("chunk_long_documents"))
    target_chars = int((spec.filters or {}).get("chunk_target_chars") or max(spec.min_chars, 4096))
    max_chars = int((spec.filters or {}).get("chunk_max_chars") or max(target_chars * 2, target_chars + 512))
    for file_index, repo_path in enumerate(files, start=1):
        print(
            f"[{spec.name}] Streaming gzip json shard {file_index}/{len(files)}: {repo_path}",
            flush=True,
        )
        url = hf_hub_url(repo_id=spec.dataset_name, filename=repo_path, repo_type="dataset")
        with session.get(url, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as gz_handle:
                text_stream = io.TextIOWrapper(gz_handle, encoding="utf-8", errors="ignore")
                for line_index, line in enumerate(text_stream, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.setdefault("id", f"{repo_path}:{line_index}")
                    if chunk_long_documents:
                        text = str(row.get(spec.text_column, ""))
                        for chunk_index, chunk in enumerate(
                            iter_text_chunks(
                                text,
                                min_chars=spec.min_chars,
                                target_chars=target_chars,
                                max_chars=max_chars,
                            ),
                            start=1,
                        ):
                            chunked_row = dict(row)
                            chunked_row[spec.text_column] = chunk
                            chunked_row["id"] = f"{row['id']}:{chunk_index}"
                            yield chunked_row
                        continue
                    yield row


def iter_remote_text_file_rows(spec: SourceSpec):
    files = repo_files_matching(spec, suffixes=(".txt",), prefixes=("data/",))
    files = partitioned_files(files, spec)
    if not files:
        if (spec.filters or {}).get("_allow_empty_partition"):
            return
        raise RuntimeError(f"No text files found for source {spec.name}")

    session = requests.Session()
    session.headers.update(hf_request_headers())
    target_chars = max(spec.min_chars, 2048)
    max_chars = max(target_chars * 2, target_chars + 512)
    for file_index, repo_path in enumerate(files, start=1):
        print(
            f"[{spec.name}] Fetching text file {file_index}/{len(files)}: {repo_path}",
            flush=True,
        )
        url = hf_hub_url(repo_id=spec.dataset_name, filename=repo_path, repo_type="dataset")
        response = session.get(url, timeout=(30, 900))
        response.raise_for_status()
        text = response.content.decode("utf-8", errors="ignore")
        title = Path(repo_path).stem
        for chunk_index, chunk in enumerate(
            iter_text_chunks(
                text,
                min_chars=spec.min_chars,
                target_chars=target_chars,
                max_chars=max_chars,
            ),
            start=1,
        ):
            yield {
                spec.text_column: chunk,
                "id": f"{repo_path}:{chunk_index}",
                "title": title,
            }


def load_pg19_titles(session: requests.Session) -> dict[str, str]:
    try:
        response = session.get(PG19_METADATA_URL, timeout=(30, 300))
        response.raise_for_status()
    except requests.RequestException:
        return {}

    titles: dict[str, str] = {}
    reader = csv.DictReader(io.StringIO(response.text))
    for row in reader:
        book_id = str(row.get("book_id", "")).strip()
        if not book_id:
            continue
        title = (
            str(row.get("short_book_title") or "").strip()
            or str(row.get("book_title") or "").strip()
            or book_id
        )
        titles[book_id] = title
    return titles


def iter_text_chunks(
    text: str,
    *,
    min_chars: int,
    target_chars: int = 2048,
    max_chars: int = 4096,
) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    def split_long_piece(piece: str) -> list[str]:
        piece = piece.strip()
        if len(piece) <= max_chars:
            return [piece] if piece else []
        chunks: list[str] = []
        start = 0
        while start < len(piece):
            end = min(len(piece), start + max_chars)
            if end < len(piece):
                pivot = piece.rfind(" ", start + min_chars, end)
                if pivot != -1 and pivot - start >= min_chars:
                    end = pivot
            chunk = piece[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
            while start < len(piece) and piece[start].isspace():
                start += 1
        return chunks

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_chars = 0

    def flush_current() -> None:
        nonlocal current_parts, current_chars
        if not current_parts:
            return
        chunk = "\n\n".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_parts = []
        current_chars = 0

    for paragraph in paragraphs:
        pieces = split_long_piece(paragraph)
        for piece in pieces:
            addition = len(piece) + (2 if current_parts else 0)
            if current_parts and current_chars + addition > max_chars:
                flush_current()
            current_parts.append(piece)
            current_chars += len(piece) + (2 if len(current_parts) > 1 else 0)
            if current_chars >= target_chars:
                flush_current()

    flush_current()
    if not chunks and len(normalized) >= min_chars:
        return [normalized]
    return [chunk for chunk in chunks if len(chunk) >= min_chars]


def iter_pg19_rows(spec: SourceSpec):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    split_files_path = hf_hub_download(
        repo_id=spec.dataset_name,
        repo_type="dataset",
        filename=f"data/{spec.split}_files.txt",
    )
    with open(split_files_path, "r", encoding="utf-8") as handle:
        split_files = [line.strip() for line in handle if line.strip()]
    split_files = partitioned_files(split_files, spec)
    if not split_files and (spec.filters or {}).get("_allow_empty_partition"):
        return

    titles = load_pg19_titles(session)
    target_chars = max(spec.min_chars, 2048)
    max_chars = max(target_chars * 2, target_chars + 512)

    for file_index, file_path in enumerate(split_files, start=1):
        if file_index <= 3 or file_index % 100 == 0:
            print(
                f"[{spec.name}] Fetching PG19 book {file_index}/{len(split_files)}: {file_path}",
                flush=True,
            )

        url = PG19_ASSET_ROOT_URL + file_path.lstrip("/")
        response = session.get(url, timeout=(30, 900))
        response.raise_for_status()
        book_text = response.content.decode("utf-8", errors="ignore")

        book_id = Path(file_path).stem
        title = titles.get(book_id, book_id)
        for chunk_index, chunk in enumerate(
            iter_text_chunks(
                book_text,
                min_chars=max(spec.min_chars, 1200),
                target_chars=target_chars,
                max_chars=max_chars,
            ),
            start=1,
        ):
            yield {
                spec.text_column: chunk,
                "short_book_title": f"{title} [chunk {chunk_index}]",
                "book_id": book_id,
                "book_path": file_path,
                "chunk_index": chunk_index,
            }


def load_source_dataset(spec: SourceSpec):
    force_local = truthy_env("METIS_LOCAL_DATASETS")
    incremental = truthy_env("METIS_INCREMENTAL_HF_SHARDS")
    parquet_prefix = infer_repo_parquet_prefix(spec)
    has_configured_parquet_prefixes = bool((spec.filters or {}).get("parquet_prefixes"))
    if force_local and incremental and spec.dataset_name == "LLM360/TxT360" and spec.loader == "text":
        prefixes = tuple((spec.filters or {}).get("jsonl_prefixes") or ("v1.1/TxT360_BestOfWeb/",))
        print(
            f"[{spec.name}] Using direct TxT360 jsonl loader for prefixes {prefixes}.",
            flush=True,
        )
        return iter_remote_jsonl_rows(spec, prefixes)
    if force_local and incremental and spec.dataset_name == "mlfoundations/dclm-baseline-1.0" and spec.loader == "text":
        prefixes = tuple((spec.filters or {}).get("jsonl_zst_prefixes") or ("global-shard_",))
        print(
            f"[{spec.name}] Using direct DCLM zstd-jsonl loader for prefixes {prefixes}.",
            flush=True,
        )
        return iter_remote_jsonl_zst_rows(spec, prefixes)
    if force_local and incremental and spec.dataset_name == "allenai/peS2o" and spec.loader == "text":
        prefixes = tuple((spec.filters or {}).get("jsonl_zst_prefixes") or ("data/v3/",))
        print(
            f"[{spec.name}] Using direct peS2o zstd-jsonl loader for prefixes {prefixes}.",
            flush=True,
        )
        return iter_remote_jsonl_zst_rows(spec, prefixes, suffixes=(".zst", ".jsonl.zst"))
    if force_local and incremental and spec.dataset_name == "deepmind/pg19" and spec.loader == "text":
        print(f"[{spec.name}] Using direct PG19 book-file loader.", flush=True)
        return iter_pg19_rows(spec)
    if force_local and incremental and spec.dataset_name == "crumb/openstax-text" and spec.loader == "text":
        print(f"[{spec.name}] Using direct OpenStax text-file loader.", flush=True)
        return iter_remote_text_file_rows(spec)
    if (
        force_local
        and incremental
        and spec.dataset_name
        in {
            "common-pile/project_gutenberg_filtered",
            "common-pile/pre_1929_books_filtered",
            "common-pile/pre_1929_books",
        }
        and spec.loader == "text"
    ):
        print(f"[{spec.name}] Using direct Common Pile gzip-json loader.", flush=True)
        return iter_remote_json_gz_rows(spec)
    if force_local and incremental and spec.dataset_name == "EleutherAI/proof-pile-2" and spec.loader == "text":
        if "arxiv" in spec.name:
            prefixes = ("arxiv/train/",)
        elif "math" in spec.name or "proof" in spec.name:
            prefixes = (
                "open-web-math/train/",
                "algebraic-stack/train/lean_proofsteps",
                "algebraic-stack/train/isa_proofsteps",
            )
        else:
            prefixes = ("arxiv/train/", "open-web-math/train/", "algebraic-stack/train/")
        print(
            f"[{spec.name}] Using direct Proof-Pile-2 zstd-jsonl loader for prefixes {prefixes}.",
            flush=True,
        )
        return iter_remote_jsonl_zst_rows(spec, prefixes)
    if force_local and incremental and (parquet_prefix or has_configured_parquet_prefixes) and spec.loader == "text":
        print(f"[{spec.name}] Using incremental parquet-shard loader.", flush=True)
        return iter_remote_parquet_rows(spec)
    if force_local:
        print(
            f"[{spec.name}] No incremental local shard loader configured; "
            "falling back to HF streaming to avoid a full local corpus download.",
            flush=True,
        )
    dataset = load_dataset(
        spec.dataset_name,
        name=spec.dataset_config,
        split=spec.split,
        streaming=spec.streaming,
        trust_remote_code=spec.trust_remote_code,
    )
    filters = spec.filters or {}
    shard_count = int(filters.get("dataset_shard_count") or 1)
    shard_index = int(filters.get("dataset_shard_index") or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"Invalid dataset shard {shard_index}/{shard_count} for source {spec.name}"
            )
        if not hasattr(dataset, "shard"):
            raise RuntimeError(f"Dataset object for {spec.name} does not support sharding")
        try:
            dataset = dataset.shard(num_shards=shard_count, index=shard_index, contiguous=False)
        except (IndexError, ValueError) as exc:
            if filters.get("_allow_empty_partition"):
                print(
                    f"[{spec.name}] Dataset shard {shard_index}/{shard_count} is empty "
                    f"({type(exc).__name__}); treating partition as exhausted.",
                    flush=True,
                )
                return iter(())
            raise
    return dataset


def build_planned_counts(weights: list[float], total_examples: int) -> dict[int, int]:
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights]
    raw_counts = [weight * total_examples for weight in normalized]
    counts = [int(count) for count in raw_counts]
    remainder = total_examples - sum(counts)
    ranked = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1

    return {index: count for index, count in enumerate(counts)}


def clean_text(text: str, spec: SourceSpec) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    if spec.normalize_whitespace:
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = MULTISPACE_RE.sub(" ", text)
        text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    alpha = sum(char.isalpha() for char in meaningful)
    return alpha / len(meaningful)


def latin_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    latin_letters = sum(("a" <= char.lower() <= "z") for char in letters)
    return latin_letters / len(letters)


def max_repeat_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    for previous, current_char in zip(text, text[1:]):
        if current_char == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def url_count(text: str) -> int:
    return text.count("http://") + text.count("https://") + text.count("www.")


def nested_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def value_matches(value: Any, allowed: Any) -> bool:
    if not isinstance(allowed, (list, tuple, set)):
        allowed = [allowed]
    if isinstance(value, str):
        lowered = value.lower()
        return any(str(candidate).lower() == lowered for candidate in allowed)
    return value in allowed


def passes_row_filters(row: dict[str, Any], spec: SourceSpec) -> bool:
    filters = spec.filters or {}
    for path, minimum in (filters.get("row_numeric_min") or {}).items():
        value = nested_get(row, path)
        try:
            if value is None or float(value) < float(minimum):
                return False
        except (TypeError, ValueError):
            return False
    for path, maximum in (filters.get("row_numeric_max") or {}).items():
        value = nested_get(row, path)
        try:
            if value is None or float(value) > float(maximum):
                return False
        except (TypeError, ValueError):
            return False
    for path, allowed in (filters.get("row_allowed_values") or {}).items():
        if not value_matches(nested_get(row, path), allowed):
            return False
    for path, disallowed in (filters.get("row_disallowed_values") or {}).items():
        value = nested_get(row, path)
        if value is not None and value_matches(value, disallowed):
            return False
    for any_filter in filters.get("row_any_numeric_min", []) or []:
        paths = any_filter.get("paths", [])
        minimum = float(any_filter.get("min", 0))
        matched = False
        for path in paths:
            value = nested_get(row, path)
            try:
                if value is not None and float(value) >= minimum:
                    matched = True
                    break
            except (TypeError, ValueError):
                continue
        if not matched:
            return False
    for any_filter in filters.get("row_any_label_contains", []) or []:
        paths = any_filter.get("paths", [])
        needles = [str(item).lower() for item in any_filter.get("contains", [])]
        if not needles:
            continue
        matched = False
        for path in paths:
            value = nested_get(row, path)
            if value is None:
                continue
            text = str(value).lower()
            if any(needle in text for needle in needles):
                matched = True
                break
        if not matched:
            return False
    return True


def passes_quality_filters(text: str, spec: SourceSpec) -> bool:
    filters = spec.filters or {}
    wants_english = (
        str(filters.get("language", "")).lower() in {"en", "english"}
        or str(filters.get("script", "")).lower() == "latn"
        or filters.get("english_confidence_min") is not None
    )
    if wants_english and latin_letter_ratio(text) < float(filters.get("latin_letter_ratio_min", 0.65)):
        return False
    if spec.min_alpha_ratio is not None and alpha_ratio(text) < float(spec.min_alpha_ratio):
        return False
    if spec.max_repeat_char_run is not None and max_repeat_char_run(text) > int(spec.max_repeat_char_run):
        return False
    if spec.max_url_count is not None and url_count(text) > int(spec.max_url_count):
        return False
    if spec.max_line_length is not None:
        if any(len(line) > int(spec.max_line_length) for line in text.splitlines()):
            return False
    return True


def iter_schedule(planned_counts: dict[int, int], seed: int):
    remaining = dict(planned_counts)
    total_remaining = sum(remaining.values())
    rng = Random(seed)
    source_indices = sorted(remaining)
    while total_remaining > 0:
        pick = rng.randrange(total_remaining)
        cumulative = 0
        for index in source_indices:
            count = remaining[index]
            if count <= 0:
                continue
            cumulative += count
            if pick < cumulative:
                remaining[index] -= 1
                total_remaining -= 1
                yield index
                break


def extract_doc_id(row: dict[str, Any], spec: SourceSpec, fallback_index: int) -> str:
    if spec.id_column and row.get(spec.id_column) is not None:
        return str(row[spec.id_column])
    if spec.loader == "software_heritage_blob" and row.get(spec.blob_id_column):
        return str(row[spec.blob_id_column])
    return f"{spec.name}:{fallback_index}"


def extract_text(row: dict[str, Any], spec: SourceSpec, fetcher: SoftwareHeritageFetcher) -> str:
    if spec.loader == "text":
        return str(row.get(spec.text_column, ""))
    if spec.loader == "template":
        if not spec.text_template:
            return ""
        cooked_row = {key: "" if value is None else str(value) for key, value in row.items()}
        try:
            return spec.text_template.format(**cooked_row)
        except KeyError:
            return ""
    if spec.loader == "software_heritage_blob":
        blob_id = row.get(spec.blob_id_column)
        if not blob_id:
            return ""
        return fetcher.fetch(str(blob_id))
    raise ValueError(f"Unsupported source loader: {spec.loader}")
