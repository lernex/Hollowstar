from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile

from .acquisition import file_lock
from .common import ObjectSpec, digest_json, read_receipt, utc_now, write_receipt


FORMATS = (
    (".warc.wet.gz", "warc_wet_gzip"),
    (".warc.gz", "warc_gzip"),
    (".jsonl.gz", "jsonl_gzip"),
    (".jsonl.zst", "jsonl_zstd"),
    (".json.zst", "json_zstd"),
    (".parquet", "parquet"),
    (".jsonl", "raw_jsonl"),
    (".xml.bz2", "xml_bzip2"),
)


def wire_format(key: str) -> str:
    for suffix, name in FORMATS:
        if key.endswith(suffix):
            return name
    raise ValueError(f"No admitted content format for object: {key}")


def _match(path: str, patterns: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or ("/**/" in pattern and fnmatch.fnmatchcase(path, pattern.replace("/**/", "/")))
        for pattern in patterns
    )


class CatalogueWriter:
    def __init__(self, root: Path, source: Mapping[str, Any], *, page_size: int = 256) -> None:
        if not isinstance(source.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", source["id"]):
            raise ValueError("Source identifier is not a safe catalogue name")
        self.root = root
        self.source = dict(source)
        self.source_hash = digest_json(source)
        self.directory = root / "catalogue" / f"{source['id']}-{self.source_hash[:16]}"
        self.page_size = page_size
        self.page: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = []
        self.seen: set[str] = set()
        self.bytes_known = 0
        self.unknown_sizes = 0
        self.bootstrap_groups: set[str] = set()

    def add(self, spec: ObjectSpec) -> None:
        if spec.object_id in self.seen:
            raise RuntimeError(f"Duplicate object in source catalogue: {spec.relative_key}")
        self.seen.add(spec.object_id)
        if spec.expected_bytes is None:
            self.unknown_sizes += 1
        else:
            self.bytes_known += spec.expected_bytes
        group = str(spec.policy.get("admission_group", spec.source_id))
        value = spec.to_dict()
        value["policy"]["bootstrap"] = group not in self.bootstrap_groups
        self.bootstrap_groups.add(group)
        self.page.append(value)
        if len(self.page) >= self.page_size:
            self.flush()

    def flush(self) -> None:
        if not self.page:
            return
        payload = {
            "schema": "metis17.object-page/v1",
            "source_id": self.source["id"],
            "source_hash": self.source_hash,
            "page": len(self.pages),
            "objects": self.page,
        }
        destination = self.directory / f"page-{len(self.pages):06d}.json"
        if destination.exists():
            if read_receipt(destination) != payload:
                raise RuntimeError(f"Immutable catalogue page changed: {destination}")
        else:
            write_receipt(destination, payload)
        self.pages.append({"path": str(destination.relative_to(self.root)), "objects": len(self.page)})
        self.page = []

    def seal(self) -> dict[str, Any]:
        self.flush()
        expected = self.source.get("expected_inventory_bytes")
        if expected is not None and (self.unknown_sizes or self.bytes_known != int(expected)):
            raise RuntimeError(
                f"Source inventory mismatch for {self.source['id']}: "
                f"{self.bytes_known} known bytes; expected {expected}; {self.unknown_sizes} unknown objects"
            )
        expected_count = self.source.get("expected_objects")
        if expected_count is not None and len(self.seen) != int(expected_count):
            raise RuntimeError(f"Source object coverage mismatch: {self.source['id']}")
        if not self.seen:
            raise RuntimeError(f"Source catalogue contains no content objects: {self.source['id']}")
        payload = {
            "schema": "metis17.source-catalogue/v1",
            "source": self.source,
            "source_hash": self.source_hash,
            "objects": len(self.seen),
            "known_bytes": self.bytes_known,
            "unknown_size_objects": self.unknown_sizes,
            "pages": self.pages,
        }
        path = self.directory / "SOURCE_COMPLETE.json"
        if path.exists() and read_receipt(path) != payload:
            raise RuntimeError("Completed source catalogue changed")
        write_receipt(path, payload)
        return payload


def _policy_for(source: Mapping[str, Any], key: str) -> tuple[dict[str, Any], int]:
    policy = dict(source["policy"])
    priority = int(source["priority"])
    group = source["id"]
    for override in source.get("path_policies", []):
        if _match(key, override["patterns"]):
            policy.update(override.get("policy", {}))
            priority = int(override.get("priority", priority))
            group = f"{source['id']}:{override['name']}"
            break
    policy["source_budget_bytes"] = int(source["budget_bytes"])
    policy["admission_group"] = str(group)
    return policy, priority


def _resolve_hf(root: Path, source: Mapping[str, Any], writer: CatalogueWriter) -> None:
    api = HfApi()
    repo = str(source["repo"])
    revision = str(source["revision"])
    info = api.dataset_info(repo, revision=revision, timeout=60)
    if info.sha != revision:
        raise RuntimeError(f"HF revision did not resolve exactly: {repo}")
    seen_paths: set[str] = set()
    for prefix in source.get("prefixes", [None]):
        for item in api.list_repo_tree(
            repo,
            repo_type="dataset",
            revision=revision,
            path_in_repo=prefix,
            recursive=True,
            expand=False,
        ):
            if not isinstance(item, RepoFile):
                continue
            key = item.path
            if key in seen_paths:
                raise RuntimeError(f"Overlapping catalogue prefixes: {repo}:{key}")
            seen_paths.add(key)
            if not _match(key, source["allow_patterns"]) or _match(key, source.get("deny_patterns", [])):
                continue
            if item.size < 0:
                raise RuntimeError(f"Unresolved payload object size: {repo}:{key}")
            lfs = item.lfs
            checksum = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            policy, priority = _policy_for(source, key)
            policy["repo_id"] = repo
            writer.add(ObjectSpec.create(
                source_id=str(source["id"]),
                url=hf_hub_url(repo, key, repo_type="dataset", revision=revision),
                revision=revision,
                relative_key=key,
                wire_format=wire_format(key),
                adapter=str(source["adapter"]),
                priority=priority,
                expected_bytes=item.size,
                expected_sha256=checksum,
                policy=policy,
            ))


def _cached_metadata(root: Path, url: str, *, maximum_bytes: int = 10_000_000) -> bytes:
    key = hashlib.sha256(url.encode()).hexdigest()
    path = root / "metadata" / key
    if path.exists():
        return path.read_bytes()
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(128 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise RuntimeError("Catalogue metadata exceeds its explicit bound")
            chunks.append(chunk)
    data = b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return data


def _resolve_hplt(root: Path, source: Mapping[str, Any], writer: CatalogueWriter) -> None:
    data = _cached_metadata(root, str(source["manifest_url"]))
    digest = hashlib.sha256(data).hexdigest()
    if digest != source["manifest_sha256"]:
        raise RuntimeError("HPLT catalogue differs from its reviewed manifest")
    records = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    names: set[str] = set()
    for record in records:
        name = str(record["name"])
        if name in names:
            raise RuntimeError("HPLT language is duplicated in the catalogue")
        names.add(name)
        if source["selection"] == "english-wds10" and name != "eng_Latn":
            continue
        if source["selection"] == "nonenglish" and name == "eng_Latn":
            continue
        md5_data = _cached_metadata(root, str(record["md5"])).decode("utf-8")
        checksums: dict[str, str] = {}
        for line in md5_data.split("\n"):
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{32})\s+\*?(.+)", line)
            if not match:
                raise RuntimeError(f"Unrecognized HPLT checksum entry for {name}")
            checksums[Path(match[2]).name] = match[1].lower()
        urls = sorted(record["urls"], key=lambda url: (-int(Path(urlsplit(url).path).name.split("_")[0]), str(url)))
        for url in urls:
            parsed = urlsplit(str(url))
            if parsed.scheme != "https" or parsed.hostname != "data.hplt-project.org":
                raise RuntimeError("HPLT manifest contains an unapproved origin")
            filename = Path(parsed.path).name
            bucket = int(filename.split("_", 1)[0])
            if source["selection"] == "english-wds10" and bucket != 10:
                continue
            if filename not in checksums:
                raise RuntimeError(f"HPLT object has no published checksum: {name}/{filename}")
            policy, _ = _policy_for(source, filename)
            policy.update({
                "language": name,
                "expected_md5": checksums[filename],
                "publisher_wds_bucket": bucket,
                "admission_group": f"{source['id']}:{name}",
                "metadata": {"language": name, "publisher_wds_bucket": bucket},
            })
            writer.add(ObjectSpec.create(
                source_id=str(source["id"]),
                url=str(url),
                revision=f"{digest}:md5:{checksums[filename]}",
                relative_key=f"{name}/{filename}",
                wire_format="jsonl_zstd",
                adapter="text",
                priority=int(source["priority"]) + bucket,
                policy=policy,
            ))


def _resolve_cc(root: Path, source: Mapping[str, Any], writer: CatalogueWriter) -> None:
    data = _cached_metadata(root, str(source["manifest_url"]))
    digest = hashlib.sha256(data).hexdigest()
    expected = source.get("manifest_sha256")
    if expected and digest != expected:
        raise RuntimeError("Common Crawl object list changed")
    lines = gzip.decompress(data).decode("utf-8").split("\n")
    prefix = str(source["key_prefix"])
    for key in lines:
        if not key:
            continue
        if not key.startswith(prefix) or ".." in Path(key).parts or "://" in key:
            raise RuntimeError("Invalid object key in Common Crawl path list")
        policy, priority = _policy_for(source, key)
        writer.add(ObjectSpec.create(
            source_id=str(source["id"]),
            url="https://data.commoncrawl.org/" + key,
            revision=digest,
            relative_key=key,
            wire_format=wire_format(key),
            adapter=str(source["adapter"]),
            priority=priority,
            policy=policy,
        ))


def resolve_source(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    writer = CatalogueWriter(root, source)
    complete = writer.directory / "SOURCE_COMPLETE.json"
    if complete.exists():
        return read_receipt(complete)
    with file_lock(root / "locks" / "catalogue" / f"{source['id']}.lock", timeout=2):
        active = root / "catalogue" / "active" / f"{source['id']}.json"
        descriptor = {
            "source_id": source["id"],
            "source_hash": writer.source_hash,
            "directory": str(writer.directory.relative_to(root)),
        }
        if active.exists() and read_receipt(active) != descriptor:
            raise RuntimeError("Active source definition changed; create a separate immutable batch")
        write_receipt(active, descriptor)
        kind = source["kind"]
        if kind == "hf":
            _resolve_hf(root, source, writer)
        elif kind == "hplt":
            _resolve_hplt(root, source, writer)
        elif kind == "cc":
            _resolve_cc(root, source, writer)
        else:
            raise ValueError(f"No content-only source resolver for kind: {kind}")
        return writer.seal()


def catalogue_objects(root: Path, *, origins: set[str] | None = None) -> Iterable[ObjectSpec]:
    seen: set[str] = set()
    for active in sorted((root / "catalogue" / "active").glob("*.json")):
        descriptor = read_receipt(active)
        directory = (root / descriptor["directory"]).resolve()
        if not directory.is_relative_to((root / "catalogue").resolve()):
            raise RuntimeError("Catalogue directory escapes the release")
        for page in sorted(directory.glob("page-*.json")):
            value = read_receipt(page)
            if value.get("schema") != "metis17.object-page/v1" or value["source_hash"] != descriptor["source_hash"]:
                raise RuntimeError("Unrecognized or mismatched catalogue page")
            for record in value["objects"]:
                spec = ObjectSpec.from_dict(record)
                if spec.object_id in seen:
                    raise RuntimeError("Object appears more than once in active catalogues")
                seen.add(spec.object_id)
                if origins is None or urlsplit(spec.url).hostname in origins:
                    yield spec
