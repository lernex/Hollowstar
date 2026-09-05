from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from metis_data.state import atomic_json, utc_now


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def under_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Artifact path must be below the release root: {relative!r}")
    return path


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    source_id: str
    url: str
    revision: str
    relative_key: str
    wire_format: str
    adapter: str
    priority: int
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        url: str,
        revision: str,
        relative_key: str,
        wire_format: str,
        adapter: str,
        priority: int,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
        policy: Mapping[str, Any] | None = None,
    ) -> ObjectSpec:
        for name, value in (
            ("source_id", source_id), ("revision", revision), ("relative_key", relative_key),
            ("url", url), ("wire_format", wire_format), ("adapter", adapter),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Nonempty string required for {name}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id):
            raise ValueError("Source identifier is not a safe catalogue name")
        if type(priority) is not int:
            raise ValueError("Priority must be an integer, not a coerced string/float")
        if policy is not None and not isinstance(policy, Mapping):
            raise ValueError("Object policy must be a mapping")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Acquisition requires an explicit HTTPS object URL without credentials")
        if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes < 0):
            raise ValueError("Expected byte count cannot be negative")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            raise ValueError("Expected SHA-256 must be a lowercase hexadecimal digest")
        md5 = (policy or {}).get("expected_md5")
        if md5 is not None and (not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5)):
            raise ValueError("Expected MD5 must be a lowercase hexadecimal digest")
        identity = {
            "source_id": source_id,
            "url": url,
            "revision": revision,
            "relative_key": relative_key,
            "expected_sha256": expected_sha256,
            "expected_md5": md5,
        }
        return cls(
            object_id=digest_json(identity),
            source_id=source_id,
            url=url,
            revision=revision,
            relative_key=relative_key,
            wire_format=wire_format,
            adapter=adapter,
            priority=priority,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            policy=dict(policy or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObjectSpec:
        spec = cls.create(
            source_id=value["source_id"],
            url=value["url"],
            revision=value["revision"],
            relative_key=value["relative_key"],
            wire_format=value["wire_format"],
            adapter=value["adapter"],
            priority=value["priority"],
            expected_bytes=value.get("expected_bytes"),
            expected_sha256=value.get("expected_sha256"),
            policy=value.get("policy"),
        )
        if spec.object_id != value.get("object_id"):
            raise ValueError("Object manifest identity does not match its content")
        return spec


@dataclass(frozen=True)
class RawReceipt:
    object_id: str
    source_id: str
    relative_path: str
    byte_count: int
    sha256: str
    download_host: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RawReceipt:
        result = cls(
            object_id=str(value["object_id"]),
            source_id=str(value["source_id"]),
            relative_path=str(value["relative_path"]),
            byte_count=int(value["byte_count"]),
            sha256=str(value["sha256"]),
            download_host=str(value["download_host"]),
            completed_at=str(value["completed_at"]),
        )
        if result.byte_count < 0 or not re.fullmatch(r"[0-9a-f]{64}", result.sha256):
            raise ValueError("Malformed raw-object receipt")
        return result


def write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = digest_json(payload)
    atomic_json(path, payload)


def read_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Receipt is not an object: {path}")
    expected = payload.pop("receipt_sha256", None)
    if expected != digest_json(payload):
        raise ValueError(f"Receipt hash mismatch: {path}")
    return payload
