from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

import requests
from huggingface_hub import get_token

from .common import ObjectSpec, RawReceipt, atomic_json, read_receipt, sha256_file, under_root, utc_now, write_receipt


class CapacityPending(RuntimeError):
    """Intake has reached its explicitly approved or bounded capacity."""


class DownloadFailure(RuntimeError):
    """A content object could not be acquired with its integrity contract."""


class DownloadPaused(RuntimeError):
    """A requested shutdown retained the verified-resumable partial object."""


def receipt_path(root: Path, object_id: str) -> Path:
    return root / "ready" / object_id[:2] / f"{object_id}.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def file_lock(path: Path, *, timeout: float = 60) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "nonce": uuid.uuid4().hex}
    deadline = time.monotonic() + timeout
    while True:
        reap = path.with_name(path.name + ".reaper")
        if reap.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Lock recovery is busy: {path}")
            time.sleep(0.1)
            continue
        try:
            path.mkdir()
            atomic_json(path / "owner.json", owner)
            break
        except FileExistsError:
            owner_path = path / "owner.json"
            if owner_path.is_file():
                try:
                    previous = json.loads(owner_path.read_text())
                except FileNotFoundError:
                    # The holder can release between is_file and open.
                    continue
                if previous["host"] == owner["host"] and not _pid_alive(int(previous["pid"])):
                    try:
                        reap.mkdir()
                    except FileExistsError:
                        continue
                    try:
                        # Serialize recovery and re-read ownership so two
                        # waiters cannot rename a newly acquired live lock.
                        if not owner_path.is_file():
                            continue
                        current = json.loads(owner_path.read_text())
                        if current != previous or _pid_alive(int(current["pid"])):
                            continue
                        stale = path.with_name(f"{path.name}.dead-{uuid.uuid4().hex}")
                        path.rename(stale)
                        (stale / "owner.json").unlink()
                        stale.rmdir()
                    finally:
                        reap.rmdir()
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Live or unverified-owner lock is busy: {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        current = json.loads((path / "owner.json").read_text())
        if current != owner:
            raise RuntimeError(f"Lock ownership changed unexpectedly: {path}")
        (path / "owner.json").unlink()
        path.rmdir()


class IntakeBudget:
    def __init__(self, root: Path, limits: Mapping[str, Any]) -> None:
        self.root = root
        self.limits = limits
        self.path = root / "state" / "intake-budget.json"
        self.lock = root / "locks" / "intake-budget.lock"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema": "metis17.intake-budget/v1",
                "raw_bytes": 0,
                "network_payload_bytes": 0,
                "sources": {},
                "inflight": {},
            }
        return read_receipt(self.path)

    def _recover_published(self, state: dict[str, Any]) -> None:
        for object_id, reservation in list(state["inflight"].items()):
            path = receipt_path(self.root, object_id)
            if not path.exists():
                continue
            receipt = RawReceipt.from_dict(read_receipt(path))
            if receipt.object_id != object_id or receipt.source_id != reservation["source_id"]:
                raise RuntimeError("Published receipt differs from its budget reservation")
            state["raw_bytes"] += receipt.byte_count
            source = state["sources"].setdefault(receipt.source_id, {"bytes": 0, "objects": 0})
            source["bytes"] += receipt.byte_count
            source["objects"] += 1
            del state["inflight"][object_id]

    def reserve(self, spec: ObjectSpec, size: int) -> None:
        if size < 0:
            raise ValueError("Cannot reserve negative bytes")
        with file_lock(self.lock):
            state = self._read()
            self._recover_published(state)
            if receipt_path(self.root, spec.object_id).exists():
                write_receipt(self.path, state)
                return
            existing = state["inflight"].get(spec.object_id)
            other = sum(int(v["bytes"]) for k, v in state["inflight"].items() if k != spec.object_id)
            ceiling = int(self.limits["max_raw_bytes"])
            if state["raw_bytes"] + other + size > ceiling:
                raise CapacityPending(f"Raw intake ceiling reached: {ceiling} bytes")
            source_cap = int(spec.policy.get("source_budget_bytes", ceiling))
            source_reserved = sum(
                int(v["bytes"])
                for k, v in state["inflight"].items()
                if k != spec.object_id and v["source_id"] == spec.source_id
            )
            source_done = int(state["sources"].get(spec.source_id, {}).get("bytes", 0))
            if source_done + source_reserved + size > source_cap:
                raise CapacityPending(f"Source ceiling reached: {spec.source_id}")
            factor = int(self.limits.get("working_reservation_factor", 4))
            fixed = int(self.limits.get("policy_and_metadata_reserve_bytes", 20_000_000_000))
            if factor * (state["raw_bytes"] + other + size) + fixed > int(self.limits["max_working_bytes"]):
                raise CapacityPending("Bounded working-storage reservation is exhausted")
            free = shutil.disk_usage(self.root).free
            if free < factor * size + int(self.limits.get("filesystem_free_floor_bytes", 100_000_000_000)):
                raise CapacityPending("Filesystem free-space safety floor reached")
            if existing and existing["source_id"] != spec.source_id:
                raise RuntimeError("Object reservation source changed")
            state["inflight"][spec.object_id] = {
                "bytes": size,
                "source_id": spec.source_id,
                "host": socket.gethostname(),
                "pid": os.getpid(),
            }
            write_receipt(self.path, state)

    def finish(self, spec: ObjectSpec, receipt: RawReceipt, transferred: int) -> None:
        with file_lock(self.lock):
            state = self._read()
            self._recover_published(state)
            path = receipt_path(self.root, spec.object_id)
            if not path.exists():
                reservation = state["inflight"].get(spec.object_id)
                if not reservation or receipt.byte_count > int(reservation["bytes"]):
                    raise RuntimeError("Object completion has no sufficient reservation")
                write_receipt(path, {**receipt.to_dict(), "spec": spec.to_dict()})
                self._recover_published(state)
            else:
                old = RawReceipt.from_dict(read_receipt(path))
                if old.sha256 != receipt.sha256 or old.byte_count != receipt.byte_count:
                    raise RuntimeError("Conflicting content for an already published object")
            state["network_payload_bytes"] += transferred
            state["updated_at"] = utc_now()
            write_receipt(self.path, state)

    def failed_attempt(self, spec: ObjectSpec, transferred: int, *, release: bool) -> None:
        with file_lock(self.lock):
            state = self._read()
            self._recover_published(state)
            state["network_payload_bytes"] += transferred
            if release:
                state["inflight"].pop(spec.object_id, None)
            state["updated_at"] = utc_now()
            write_receipt(self.path, state)


def _headers(url: str, offset: int, etag: str | None) -> dict[str, str]:
    headers = {"User-Agent": "Metis-1.7-content-acquisition", "Accept-Encoding": "identity"}
    if urlsplit(url).hostname == "huggingface.co":
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if etag:
            headers["If-Range"] = etag
    return headers


def _response_size(response: requests.Response, offset: int) -> tuple[int | None, bool]:
    if response.status_code == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or int(match[1]) != offset or int(match[2]) < offset:
            raise DownloadFailure("Invalid partial-content range")
        if int(match[2]) >= int(match[3]):
            raise DownloadFailure("Partial-content range exceeds object length")
        return int(match[3]), True
    if response.status_code != 200:
        raise DownloadFailure(f"Object request returned HTTP {response.status_code}")
    length = response.headers.get("Content-Length")
    return (int(length) if length is not None else None), False


def download_object(
    spec: ObjectSpec,
    root: Path,
    limits: Mapping[str, Any],
    *,
    attempts: int = 8,
    timeout: float = 90,
    session: requests.Session | None = None,
    stop_event: threading.Event | None = None,
) -> RawReceipt:
    ready = receipt_path(root, spec.object_id)
    if ready.exists():
        receipt = RawReceipt.from_dict(read_receipt(ready))
        path = under_root(root, receipt.relative_path)
        if receipt.object_id != spec.object_id or receipt.source_id != spec.source_id:
            raise DownloadFailure("Ready receipt identity mismatch")
        if not path.is_file() or path.stat().st_size != receipt.byte_count:
            raise DownloadFailure("Published raw object is missing or changed")
        if sha256_file(path) != receipt.sha256:
            raise DownloadFailure("Published raw object failed its checksum")
        return receipt
    suffix = {
        "parquet": ".parquet",
        "jsonl_gzip": ".jsonl.gz",
        "jsonl_zstd": ".jsonl.zst",
        "json_zstd": ".json.zst",
        "raw_jsonl": ".jsonl",
        "warc_wet_gzip": ".warc.wet.gz",
        "warc_gzip": ".warc.gz",
        "xml_bzip2": ".xml.bz2",
    }.get(spec.wire_format)
    if suffix is None:
        raise ValueError(f"Unsupported wire format: {spec.wire_format}")
    target = root / "raw" / spec.object_id[:2] / f"{spec.object_id}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    transfer_state = partial.with_name(partial.name + ".json")
    progress = root / "transfers" / spec.object_id[:2] / f"{spec.object_id}.json"
    budget = IntakeBudget(root, limits)
    http = session or requests.Session()
    owns_session = session is None
    try:
        with file_lock(root / "locks" / "objects" / f"{spec.object_id}.lock", timeout=2):
            if ready.exists():
                return RawReceipt.from_dict(read_receipt(ready))
            if transfer_state.exists():
                pending = read_receipt(transfer_state)
                recoverable = target if target.exists() else partial
                if pending.get("completed_sha256") and recoverable.exists():
                    if sha256_file(recoverable) != pending["completed_sha256"]:
                        raise DownloadFailure("Completed unpublished object changed")
                    if recoverable.stat().st_size != int(pending["completed_bytes"]):
                        raise DownloadFailure("Completed unpublished object length changed")
                    budget.reserve(spec, recoverable.stat().st_size)
                    if recoverable != target:
                        recoverable.replace(target)
                    recovered = RawReceipt(
                        spec.object_id, spec.source_id, str(target.relative_to(root)),
                        target.stat().st_size, pending["completed_sha256"],
                        socket.gethostname(), utc_now(),
                    )
                    budget.finish(spec, recovered, 0)
                    transfer_state.unlink()
                    return recovered
            for attempt in range(attempts):
                transferred = 0
                previous = read_receipt(transfer_state) if transfer_state.exists() else {}
                offset = partial.stat().st_size if partial.exists() else 0
                started = time.monotonic()
                response: requests.Response | None = None
                try:
                    if stop_event is not None and stop_event.is_set():
                        raise DownloadPaused("Acquisition shutdown requested")
                    response = http.get(
                        spec.url,
                        headers=_headers(spec.url, offset, previous.get("etag")),
                        stream=True,
                        allow_redirects=True,
                        timeout=(30, timeout),
                    )
                    if response.status_code in {401, 403}:
                        raise PermissionError(f"Source access denied: {spec.source_id}; HTTP {response.status_code}")
                    if response.status_code in {429, 500, 502, 503, 504}:
                        retry_after = response.headers.get("Retry-After", "")
                        wait = min(900, int(retry_after)) if retry_after.isdigit() else min(120, 2 ** (attempt + 1))
                        response.close()
                        time.sleep(wait)
                        raise requests.ConnectionError(f"Retryable HTTP {response.status_code}")
                    total, resumed = _response_size(response, offset)
                    if spec.expected_bytes is not None and total is not None and total != spec.expected_bytes:
                        raise DownloadFailure("Publisher length differs from pinned object size")
                    known_total = spec.expected_bytes if spec.expected_bytes is not None else total
                    reserved = known_total if known_total is not None else int(limits.get("max_unknown_object_bytes", 32_000_000_000))
                    budget.reserve(spec, reserved)
                    if not resumed:
                        offset = 0
                    write_receipt(transfer_state, {
                        "object_id": spec.object_id,
                        "etag": response.headers.get("ETag"),
                        "expected_bytes": known_total,
                    })
                    digest = hashlib.sha256()
                    md5 = hashlib.md5()
                    if offset:
                        with partial.open("rb") as existing:
                            for block in iter(lambda: existing.read(8 * 1024 * 1024), b""):
                                digest.update(block)
                                md5.update(block)
                    last_report = 0.0
                    with partial.open("ab" if offset else "wb") as output:
                        for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if stop_event is not None and stop_event.is_set():
                                transferred += len(block)
                                raise DownloadPaused("Acquisition shutdown requested")
                            if not block:
                                continue
                            transferred += len(block)
                            if offset + transferred > reserved:
                                raise DownloadFailure("Object exceeded reserved byte ceiling")
                            output.write(block)
                            digest.update(block)
                            md5.update(block)
                            now = time.monotonic()
                            if now - last_report >= 5:
                                atomic_json(progress, {
                                    "object_id": spec.object_id,
                                    "source_id": spec.source_id,
                                    "host": socket.gethostname(),
                                    "pid": os.getpid(),
                                    "status": "downloading",
                                    "bytes_present": offset + transferred,
                                    "expected_bytes": known_total,
                                    "bytes_this_attempt": transferred,
                                    "bytes_per_second": transferred / max(0.001, now - started),
                                    "updated_at": utc_now(),
                                })
                                last_report = now
                        output.flush()
                        os.fsync(output.fileno())
                    actual_size = partial.stat().st_size
                    if known_total is not None and actual_size != known_total:
                        raise DownloadFailure("Object ended before its expected byte count")
                    actual_sha = digest.hexdigest()
                    if spec.expected_sha256 and actual_sha != spec.expected_sha256:
                        raise DownloadFailure("Object SHA-256 differs from publisher")
                    expected_md5 = spec.policy.get("expected_md5")
                    if expected_md5 and md5.hexdigest() != expected_md5:
                        raise DownloadFailure("Object MD5 differs from publisher")
                    write_receipt(transfer_state, {
                        "object_id": spec.object_id,
                        "etag": response.headers.get("ETag"),
                        "expected_bytes": known_total,
                        "completed_bytes": actual_size,
                        "completed_sha256": actual_sha,
                    })
                    os.replace(partial, target)
                    receipt = RawReceipt(
                        object_id=spec.object_id,
                        source_id=spec.source_id,
                        relative_path=str(target.relative_to(root)),
                        byte_count=actual_size,
                        sha256=actual_sha,
                        download_host=socket.gethostname(),
                        completed_at=utc_now(),
                    )
                    budget.finish(spec, receipt, transferred)
                    transfer_state.unlink(missing_ok=True)
                    atomic_json(progress, {
                        **receipt.to_dict(),
                        "host": receipt.download_host,
                        "status": "raw_ready",
                        "updated_at": utc_now(),
                    })
                    return receipt
                except DownloadPaused:
                    budget.failed_attempt(spec, transferred, release=not partial.exists())
                    atomic_json(progress, {
                        "object_id": spec.object_id, "source_id": spec.source_id,
                        "host": socket.gethostname(), "status": "paused",
                        "bytes_present": partial.stat().st_size if partial.exists() else 0,
                        "updated_at": utc_now(),
                    })
                    raise
                except (requests.RequestException, DownloadFailure) as exc:
                    corrupt = isinstance(exc, DownloadFailure)
                    if corrupt:
                        partial.unlink(missing_ok=True)
                        transfer_state.unlink(missing_ok=True)
                    budget.failed_attempt(
                        spec, transferred,
                        release=corrupt or not partial.exists(),
                    )
                    atomic_json(progress, {
                        "object_id": spec.object_id,
                        "source_id": spec.source_id,
                        "host": socket.gethostname(),
                        "status": "retrying" if attempt + 1 < attempts else "failed",
                        "error_type": type(exc).__name__,
                        "attempt": attempt + 1,
                        "updated_at": utc_now(),
                    })
                    if attempt + 1 == attempts:
                        raise DownloadFailure(f"Acquisition failed after {attempts} attempts: {spec.object_id}") from exc
                    time.sleep(min(60, 2 ** attempt))
                finally:
                    if response is not None:
                        response.close()
    finally:
        if owns_session:
            http.close()
    raise DownloadFailure("Download loop ended without a receipt")
