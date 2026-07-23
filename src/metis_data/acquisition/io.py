from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
import zstandard as zstd

from ..state import atomic_json, utc_now


def sha256_path(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.integrity.json")


def _remove_cached_download(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
    _validation_receipt_path(path).unlink(missing_ok=True)


def ensure_validated_download(
    client: "RetrySession",
    url: str,
    destination: Path,
    *,
    validator_id: str,
    validator: Callable[[Path], None],
    validation_attempts: int = 2,
) -> Path:
    """Return a checksum-bound, format-validated cached download.

    A non-empty cache entry is not evidence that a prior transfer reached a
    valid gzip/tar boundary.  The integrity sidecar binds the validated bytes,
    size, and validator version.  Every reuse checks the byte checksum before
    trusting the sidecar.  A missing, stale, corrupt, or format-invalid cache
    entry is deleted and downloaded again.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = _validation_receipt_path(destination)
    attempts = max(1, int(validation_attempts))

    for attempt in range(attempts):
        if destination.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
            actual_size = destination.stat().st_size
            actual_sha256 = sha256_path(destination) if actual_size > 0 else ""
            receipt_is_known = receipt.get("schema") == "metis.validated-download/v1"
            receipt_is_current = (
                receipt_is_known
                and receipt.get("validator") == validator_id
                and receipt.get("url") == url
            )
            if receipt_is_known and receipt.get("url") != url:
                _remove_cached_download(destination)
            elif receipt_is_current:
                if (
                    int(receipt.get("size", -1)) == actual_size
                    and receipt.get("sha256") == actual_sha256
                ):
                    return destination
                # Once a cache entry has been validated, any byte drift is a
                # corruption signal even if the replacement happens to be a
                # syntactically valid archive. Fetch the pinned upstream object.
                _remove_cached_download(destination)
            else:
                try:
                    if actual_size <= 0:
                        raise RuntimeError("cached download is empty")
                    validator(destination)
                except Exception:
                    _remove_cached_download(destination)
                else:
                    atomic_json(
                        receipt_path,
                        {
                            "schema": "metis.validated-download/v1",
                            "validator": validator_id,
                            "url": url,
                            "size": actual_size,
                            "sha256": actual_sha256,
                        },
                    )
                    return destination

        try:
            downloaded = client.download(url, destination)
            if downloaded.resolve() != destination.resolve():
                raise RuntimeError(
                    f"Download client returned an unexpected cache path: {downloaded}"
                )
            if not destination.is_file() or destination.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded cache entry is empty: {destination}")
            validator(destination)
            actual_size = destination.stat().st_size
            actual_sha256 = sha256_path(destination)
            atomic_json(
                receipt_path,
                {
                    "schema": "metis.validated-download/v1",
                    "validator": validator_id,
                    "url": url,
                    "size": actual_size,
                    "sha256": actual_sha256,
                },
            )
            return destination
        except Exception:
            _remove_cached_download(destination)
            if attempt + 1 >= attempts:
                raise

    raise AssertionError("validated download retry loop exited unexpectedly")


def reset_incomplete_materialization(output_dir: Path) -> None:
    """Remove only an output that has no immutable completion receipt.

    Materializer-local databases and compressed writers do not share a single
    transaction.  Rebuilding an incomplete month from validated caches is
    deterministic and prevents a dedup database from getting ahead of durable
    shards after a process or host crash.
    """

    receipt = output_dir / "ACQUISITION_RECEIPT.json"
    if receipt.exists():
        raise RuntimeError(
            f"Refusing to reset a materialization with a completion receipt: {output_dir}"
        )
    if output_dir.is_symlink():
        raise RuntimeError(f"Refusing to reset a symlinked materialization path: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def stable_fraction(*parts: object) -> float:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class RetrySession:
    """Small HTTP client with bounded retries, jitter, and resumable downloads."""

    def __init__(
        self,
        *,
        retries: int = 8,
        timeout: int = 900,
        user_agent: str = "Metis-1.6 data acquisition (contact: data@lernex.net)",
    ) -> None:
        self.retries = max(0, int(retries))
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "identity"})

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self.timeout)
        last: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(method, url, timeout=timeout, **kwargs)
                if response.status_code not in {403, 408, 409, 425, 429, 500, 502, 503, 504}:
                    return response
                rate_limit_delay: float | None = None
                retry_after = str(response.headers.get("Retry-After") or "").strip()
                if retry_after.isdigit():
                    rate_limit_delay = float(retry_after) + 1.0
                elif (
                    response.status_code == 403
                    and str(response.headers.get("X-RateLimit-Remaining") or "").strip() == "0"
                ):
                    reset = str(response.headers.get("X-RateLimit-Reset") or "").strip()
                    if reset.isdigit():
                        rate_limit_delay = max(1.0, float(reset) - time.time() + 2.0)
                if rate_limit_delay is not None and attempt < self.retries:
                    response.close()
                    # GitHub's primary limit normally resets within an hour.
                    # A larger delay is almost certainly malformed metadata or
                    # an unsuitable credential and must fail rather than hang.
                    if rate_limit_delay > 3_700:
                        raise RuntimeError(
                            f"Refusing implausible {rate_limit_delay:.0f}s HTTP rate-limit delay for {url}"
                        )
                    time.sleep(rate_limit_delay)
                    last = RuntimeError(
                        f"HTTP {response.status_code} rate limit from {url}; waited {rate_limit_delay:.0f}s"
                    )
                    continue
                if response.status_code == 403 and attempt >= 2:
                    response.raise_for_status()
                response.close()
                last = RuntimeError(f"HTTP {response.status_code} from {url}")
            except (requests.RequestException, OSError) as exc:
                last = exc
            if attempt < self.retries:
                ceiling = min(90.0, 1.5 * (2**attempt))
                time.sleep(random.uniform(0.0, ceiling))
        assert last is not None
        raise RuntimeError(f"HTTP request failed after {self.retries + 1} attempts: {url}: {last}") from last

    def download(self, url: str, destination: Path, *, expected_size: int | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = self.request("GET", url, headers=headers, stream=True)
        if offset and response.status_code == 200:
            partial.unlink(missing_ok=True)
            offset = 0
        elif offset and response.status_code != 206:
            response.raise_for_status()
        else:
            response.raise_for_status()
        if offset:
            content_range = str(response.headers.get("Content-Range") or "")
            if not content_range.lower().startswith(f"bytes {offset}-"):
                response.close()
                raise RuntimeError(
                    f"Invalid resume Content-Range for {url}: expected offset {offset}, "
                    f"received {content_range!r}"
                )
        content_length = response.headers.get("Content-Length")
        expected_transfer_size = (
            offset + int(content_length)
            if content_length is not None and str(content_length).isdigit()
            else None
        )
        mode = "ab" if offset else "wb"
        try:
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            response.close()
        if expected_transfer_size is not None and partial.stat().st_size != expected_transfer_size:
            raise RuntimeError(
                f"Truncated download for {url}: {partial.stat().st_size:,} != "
                f"{expected_transfer_size:,} bytes from Content-Length"
            )
        if expected_size is not None and partial.stat().st_size != int(expected_size):
            raise RuntimeError(
                f"Download size mismatch for {url}: {partial.stat().st_size:,} != {int(expected_size):,}"
            )
        os.replace(partial, destination)
        return destination


class JsonlShardWriter:
    """Atomic, compressed record writer that avoids tiny-file pressure on Lustre."""

    def __init__(self, output_dir: Path, *, prefix: str = "part", target_uncompressed_bytes: int = 4_000_000_000) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.target_uncompressed_bytes = int(target_uncompressed_bytes)
        self.shards: list[dict[str, Any]] = []
        self.records = 0
        self.text_characters = 0
        self._index = 0
        self._raw: Any = None
        self._compressed: Any = None
        self._text: Any = None
        self._temporary: Path | None = None
        self._uncompressed_bytes = 0
        self._shard_records = 0

    def _open(self) -> None:
        self._temporary = self.output_dir / f".{self.prefix}-{self._index:06d}.jsonl.zst.partial"
        self._raw = self._temporary.open("wb")
        self._compressed = zstd.ZstdCompressor(level=6, threads=0).stream_writer(self._raw, closefd=False)
        self._text = io.TextIOWrapper(self._compressed, encoding="utf-8")
        self._uncompressed_bytes = 0
        self._shard_records = 0

    def write(self, record: dict[str, Any]) -> None:
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if self._text is None:
            self._open()
        if self._shard_records and self._uncompressed_bytes + len(encoded) > self.target_uncompressed_bytes:
            self._close_shard()
            self._open()
        self._text.write(encoded.decode("utf-8"))
        self._uncompressed_bytes += len(encoded)
        self._shard_records += 1
        self.records += 1
        self.text_characters += len(str(record.get("text", "")))

    def _close_shard(self) -> None:
        if self._text is None or self._temporary is None:
            return
        self._text.flush()
        self._text.detach()
        self._compressed.flush(zstd.FLUSH_FRAME)
        self._compressed.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        destination = self.output_dir / f"{self.prefix}-{self._index:06d}.jsonl.zst"
        os.replace(self._temporary, destination)
        self.shards.append(
            {
                "path": str(destination),
                "size": destination.stat().st_size,
                "sha256": sha256_path(destination),
                "records": self._shard_records,
                "uncompressed_bytes": self._uncompressed_bytes,
            }
        )
        self._index += 1
        self._raw = self._compressed = self._text = self._temporary = None

    def close(self) -> list[dict[str, Any]]:
        self._close_shard()
        return self.shards

    def abort(self) -> None:
        """Close an incomplete writer and remove only its temporary shard."""

        try:
            if self._text is not None:
                self._text.close()
        except Exception:
            pass
        try:
            if self._compressed is not None:
                self._compressed.close()
        except Exception:
            pass
        try:
            if self._raw is not None and not self._raw.closed:
                self._raw.close()
        except Exception:
            pass
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
        self._raw = self._compressed = self._text = self._temporary = None

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
            return
        self.abort()

    def __del__(self) -> None:
        # Best-effort protection for materializers interrupted by an exception
        # outside a context manager. A process crash is handled by rebuilding an
        # unreceipted output directory on the next invocation.
        try:
            self.abort()
        except Exception:
            pass


def complete_materialization(
    output_dir: Path,
    *,
    source_id: str,
    driver: str,
    writer: JsonlShardWriter,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    shards = writer.close()
    if not shards:
        raise RuntimeError(f"{driver} materializer for {source_id} produced no records")
    payload = {
        "schema": "metis.acquisition-receipt/v1",
        "source_id": source_id,
        "driver": driver,
        "completed_at": utc_now(),
        "records": writer.records,
        "text_characters": writer.text_characters,
        "estimated_tokens": writer.text_characters // 4,
        "shards": shards,
        **receipt,
    }
    completion = output_dir / "ACQUISITION_RECEIPT.json"
    atomic_json(completion, payload)
    return materialized_result(completion)


def materialized_result(receipt_path: Path) -> dict[str, Any]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    shards = payload.get("shards", [])
    for shard in shards:
        path = Path(shard["path"])
        if not path.exists() or path.stat().st_size != int(shard["size"]):
            raise RuntimeError(f"Materialized shard is missing or has changed: {path}")
    return {
        "kind": "materialized_dataset",
        "source_id": payload["source_id"],
        "driver": payload["driver"],
        "local_path": str(receipt_path.parent),
        "receipt": str(receipt_path),
        "shards": shards,
        "size": sum(int(shard["size"]) for shard in shards),
        "records": int(payload.get("records", 0)),
        "estimated_tokens": int(payload.get("estimated_tokens", 0)),
        "materialized": True,
        "ready_for_training_build": True,
    }


def existing_materialization(output_dir: Path) -> dict[str, Any] | None:
    receipt = output_dir / "ACQUISITION_RECEIPT.json"
    return materialized_result(receipt) if receipt.exists() else None


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for line in text:
                    if line.strip():
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            yield payload
