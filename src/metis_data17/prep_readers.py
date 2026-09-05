"""Offline, bounded object readers and loss-conscious source adapters."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping
from urllib.parse import urlsplit

from metis_data.freshweb import FreshWebError, _charset, _decode_http_body, _normalise_host, _parse_headers
from metis_data.normalization_evidence import _normalise_license

from .common import ObjectSpec, canonical_json


ADAPTER_VERSION = "metis17.inline-adapters/v1"
FORMATS = frozenset({
    "parquet", "raw_jsonl", "jsonl", "jsonl_gzip", "jsonl_zstd", "json_zstd",
    "warc_wet_gzip", "warc_gzip", "xml_bzip2",
})
ADAPTERS = frozenset({
    "text", "content", "latex", "messages", "conversation", "stack", "stack_files",
    "wet", "warc", "cc_news", "wikipedia", "wikipedia_xml", "french_science",
    "frenchscience",
})
TEXT_FIELDS = ("text", "content", "latex", "code", "body", "document", "wikitext")
STRUCTURED_FIELDS = ("messages", "conversations", "tools")
LICENSE_FIELDS = (
    "license", "license_name", "repo_license", "spdx_license", "detected_licenses",
    "all_licenses", "licenses", "max_stars_repo_licenses",
)
_POINTER_FIELDS = frozenset({
    "blob_id", "swh_id", "swhid", "content_id", "content_path", "storage_blob",
    "blob_url", "download_url", "hexsha", "blob_sha", "content_sha256",
})
_PER_RECORD_DEFAULTS = frozenset({
    "verification_passed", "verified", "execution_passed", "parser_or_compiler_passed",
    "source_document_id", "structurally_complete", "reading_order_passed",
    "equation_integrity_passed", "license", "detected_licenses", "licenses",
})


class PreparationError(RuntimeError):
    """An object-local failure whose message never includes source payloads."""

    def __init__(self, spec: ObjectSpec, reason: str, row: int | str = "-") -> None:
        self.reason = reason
        super().__init__(f"source={spec.source_id} object={spec.object_id} row={row}: {reason}")


@dataclass(frozen=True)
class SourceRow:
    index: int
    value: Mapping[str, Any]
    rejection: str | None = None


@dataclass(frozen=True)
class Document:
    component: str
    text: str | None
    metadata: dict[str, Any]
    rejection: str | None = None
    quarantine: str | None = None


class _FrameCheckedZstd:
    """Check frame termination without a second decompression pass.

    python-zstandard's stream_reader accepts a truncated final frame as EOF.
    Tracking block boundaries on the compressed input closes that gap while
    keeping its bounded, native streaming decompressor.
    """

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.pending = bytearray()
        self.state = "magic"
        self.remaining = 0
        self.checksum = False
        self.last = False
        self.frames = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(131075 if size < 0 else min(size, 131075))
        if data:
            self._feed(data)
        else:
            self.finish()
        return data

    def _feed(self, data: bytes) -> None:
        self.pending.extend(data)
        offset = 0
        while True:
            available = len(self.pending) - offset
            if self.state in {"body", "skip", "header", "checksum"}:
                used = min(available, self.remaining)
                self.remaining -= used
                offset += used
                if self.remaining:
                    break
                if self.state == "header":
                    self.state = "block"
                elif self.state == "body" and not self.last:
                    self.state = "block"
                elif self.state == "body" and self.checksum:
                    self.state, self.remaining = "checksum", 4
                else:
                    self.state = "magic"
                continue
            needed = {"magic": 4, "descriptor": 1, "block": 3, "skip_size": 4}[self.state]
            if available < needed:
                break
            value = int.from_bytes(self.pending[offset:offset + needed], "little")
            offset += needed
            if self.state == "magic":
                if value == 0xFD2FB528:
                    self.frames += 1
                    self.state = "descriptor"
                elif value & 0xFFFFFFF0 == 0x184D2A50:
                    self.state = "skip_size"
                else:
                    raise ValueError("invalid_zstandard_frame")
            elif self.state == "descriptor":
                if value & 0x18:
                    raise ValueError("reserved_zstandard_frame_bits")
                single_segment = bool(value & 0x20)
                size_flag = value >> 6
                content_size_bytes = (1 if single_segment else 0, 2, 4, 8)[size_flag]
                dictionary_bytes = (0, 1, 2, 4)[value & 3]
                self.checksum = bool(value & 4)
                self.remaining = int(not single_segment) + content_size_bytes + dictionary_bytes
                self.state = "header"
            elif self.state == "skip_size":
                self.state, self.remaining = "skip", value
            else:
                self.last = bool(value & 1)
                block_type = (value >> 1) & 3
                if block_type == 3:
                    raise ValueError("reserved_zstandard_block")
                self.remaining = 1 if block_type == 1 else value >> 3
                self.state = "body"
        del self.pending[:offset]

    def finish(self) -> None:
        if self.pending or self.state != "magic" or not self.frames:
            raise EOFError("truncated_zstandard_frame")


def _validate_fields(spec: ObjectSpec, names: set[str], row: int | str) -> None:
    if spec.adapter in {"stack", "stack_files"}:
        valid = "files" in names
    elif spec.adapter in {"messages", "conversation"}:
        valid = bool(names & {"messages", "conversations"})
    else:
        valid = bool(names & (set(TEXT_FIELDS) | {"messages", "conversations", "files", "pages"}))
    if not valid and not names & _POINTER_FIELDS:
        raise PreparationError(spec, "invalid_payload_schema", row)


def _json_rows(stream: BinaryIO, spec: ObjectSpec, maximum: int) -> Iterator[SourceRow]:
    index = 0
    while line := stream.readline(maximum + 1):
        index += 1
        if len(line) > maximum:
            raise PreparationError(spec, "json_record_exceeds_bound", index)
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8-sig" if index == 1 else "utf-8"))
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise PreparationError(spec, "invalid_json_record", index) from None
        if isinstance(value, list):
            value = {"messages": value}
        if not isinstance(value, dict):
            raise PreparationError(spec, "json_record_is_not_an_object", index)
        yield SourceRow(index, value)


def _warc_rows(stream: BinaryIO, spec: ObjectSpec, maximum: int) -> Iterator[SourceRow]:
    index = 0
    while True:
        first = stream.readline(65537)
        if not first:
            return
        if first in {b"\n", b"\r\n"}:
            continue
        index += 1
        if not re.fullmatch(rb"WARC/1\.[01]\r?\n", first):
            raise PreparationError(spec, "invalid_warc_header", index)
        headers: dict[str, str] = {}
        header_bytes = len(first)
        while True:
            line = stream.readline(65537)
            header_bytes += len(line)
            if not line or header_bytes > 65536:
                raise PreparationError(spec, "truncated_or_oversized_warc_header", index)
            if line in {b"\r\n", b"\n"}:
                break
            if b":" not in line:
                raise PreparationError(spec, "invalid_warc_header_field", index)
            name, value = line.decode("latin-1").rstrip("\r\n").split(":", 1)
            name = name.lower()
            if name in headers:
                raise PreparationError(spec, "duplicate_warc_header_field", index)
            headers[name] = value.strip()
        length = headers.get("content-length", "")
        if not length.isdecimal():
            raise PreparationError(spec, "invalid_warc_content_length", index)
        size = int(length)
        if size > maximum:
            raise PreparationError(spec, "warc_record_exceeds_bound", index)
        body = stream.read(size)
        if len(body) != size:
            raise PreparationError(spec, "truncated_warc_record", index)
        separator = stream.readline(3)
        if separator not in {b"\r\n", b"\n"}:
            raise PreparationError(spec, "invalid_warc_record_separator", index)
        metadata: dict[str, Any] = {
            "url": headers.get("warc-target-uri"),
            "warc_record_id": headers.get("warc-record-id"),
            "capture_date": headers.get("warc-date"),
            "warc_type": headers.get("warc-type"),
            "language": headers.get("warc-identified-content-language"),
        }
        record_type = headers.get("warc-type")
        if spec.wire_format == "warc_wet_gzip":
            if record_type != "conversion":
                yield SourceRow(index, metadata, "warc_not_conversion")
                continue
            try:
                text = body.decode("utf-8")
            except UnicodeError:
                raise PreparationError(spec, "invalid_wet_utf8", index) from None
            metadata["text"] = text
            metadata["extraction"] = {"method": "wet_conversion", "encoding": "utf-8"}
            yield SourceRow(index, metadata)
            continue
        if record_type != "response":
            reason = "warc_bodyless_revisit" if record_type == "revisit" else "warc_not_response"
            yield SourceRow(index, metadata, reason)
            continue
        try:
            http_headers, body = _parse_headers(body)
            if not re.match(r"^HTTP/\d(?:\.\d)?\s+200(?:\s|$)", http_headers[":status-line"]):
                yield SourceRow(index, metadata, "warc_http_status")
                continue
            transfer = http_headers.get("transfer-encoding", "").lower()
            if transfer:
                if transfer != "chunked":
                    raise PreparationError(spec, "unsupported_http_transfer_encoding", index)
                body = _unchunk(body, spec, index)
            body = _decode_http_body(body, http_headers)
            mime = http_headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if mime not in {"text/html", "application/xhtml+xml", "text/plain"}:
                yield SourceRow(index, metadata, "warc_nontext_mime")
                continue
            encoding = _charset(http_headers.get("content-type", ""), "")
            text = body.decode(encoding)
        except (UnicodeError, LookupError):
            raise PreparationError(spec, "unsupported_or_invalid_http_encoding", index) from None
        except (ValueError, OSError, EOFError, FreshWebError):
            raise PreparationError(spec, "invalid_http_payload", index) from None
        robots = set(re.split(r"[,\s]+", http_headers.get("x-robots-tag", "").lower()))
        if mime != "text/plain":
            try:
                text, html_meta = _html_text(text)
            except ValueError:
                raise PreparationError(spec, "invalid_html_payload", index) from None
            metadata.update(html_meta)
            robots.update(html_meta.get("robots", []))
        metadata["text"] = text
        metadata["publisher_machine_learning_opt_out"] = bool(
            robots & {"noai", "noml", "notrain", "noarchive"}
        )
        metadata["extraction"] = {"method": "offline_html" if mime != "text/plain" else "http_text",
                                  "encoding": encoding}
        yield SourceRow(index, metadata)


def _unchunk(body: bytes, spec: ObjectSpec, index: int) -> bytes:
    stream = io.BytesIO(body)
    result = bytearray()
    while True:
        line = stream.readline(1025)
        if not re.fullmatch(rb"[0-9a-fA-F]+(?:;[^\r\n]*)?\r\n", line):
            raise PreparationError(spec, "invalid_http_chunk_size", index)
        size = int(line.split(b";", 1)[0], 16)
        if not size:
            while trailer := stream.readline():
                if trailer == b"\r\n":
                    if stream.read():
                        raise PreparationError(spec, "http_chunk_trailing_data", index)
                    return bytes(result)
                if b":" not in trailer:
                    break
            raise PreparationError(spec, "truncated_http_chunk_trailer", index)
        value = stream.read(size)
        if len(value) != size or stream.read(2) != b"\r\n":
            raise PreparationError(spec, "truncated_http_chunk", index)
        result.extend(value)


def _html_text(html: str) -> tuple[str, dict[str, Any]]:
    from lxml import etree
    from lxml import html as lxml_html

    if not html.strip():
        return "", {}
    try:
        tree = lxml_html.fromstring(html, parser=lxml_html.HTMLParser(no_network=True, remove_comments=True))
    except etree.ParserError:
        raise ValueError("invalid_html_document") from None
    metadata: dict[str, Any] = {"robots": []}
    if tree.get("lang"):
        metadata["language"] = tree.get("lang")
    for meta in tree.iter("meta"):
        if str(meta.get("name", "")).lower() in {"robots", "ccbot"}:
            metadata["robots"].extend(re.split(r"[,\s]+", str(meta.get("content", "")).lower()))
    for link in tree.iter("link"):
        if "canonical" in link.get("rel", "").lower().split() and link.get("href"):
            metadata["declared_canonical_url"] = link.get("href")
    for script in tree.iter("script"):
        if script.get("type", "").split(";", 1)[0].strip().lower() in {"math/tex", "math/asciimath"}:
            script.tag = "span"
    for unwanted in list(tree.iter("script", "style", "noscript", "svg", "canvas", "nav", "footer", "form")):
        unwanted.drop_tree()
    parts: list[str] = []
    blocks = {"p", "div", "article", "section", "main", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def walk(node: Any) -> None:
        if node.tag == "pre":
            parts.extend(["\n", "".join(node.itertext()), "\n"])
            return
        if node.tag in blocks or node.tag == "br":
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if node.tag in blocks:
            parts.append("\n")

    bodies = tree.xpath("//body")
    walk(bodies[0] if bodies else tree)
    return "".join(parts), metadata


def _xml_rows(stream: BinaryIO, spec: ObjectSpec, maximum: int) -> Iterator[SourceRow]:
    events = ET.iterparse(stream, events=("start", "end"))
    try:
        _, root = next(events)
    except StopIteration:
        raise PreparationError(spec, "empty_xml_container") from None
    if root.tag.rsplit("}", 1)[-1] != "mediawiki":
        raise PreparationError(spec, "invalid_wikipedia_xml_schema")
    index = 0
    for event, page in events:
        if event != "end" or page.tag.rsplit("}", 1)[-1] != "page":
            continue
        index += 1
        children = {child.tag.rsplit("}", 1)[-1]: child for child in page}
        ns = children.get("ns")
        if ns is None or ns.text is None:
            raise PreparationError(spec, "wikipedia_missing_namespace", index)
        if ns.text != "0":
            yield SourceRow(index, {}, "wikipedia_nonarticle_namespace")
        else:
            revisions = [child for child in page if child.tag.rsplit("}", 1)[-1] == "revision"]
            if len(revisions) != 1:
                raise PreparationError(spec, "wikipedia_requires_current_revision", index)
            revision = {child.tag.rsplit("}", 1)[-1]: child for child in revisions[0]}
            text_node = revision.get("text")
            if text_node is None:
                raise PreparationError(spec, "wikipedia_missing_text_element", index)
            text = text_node.text or ""
            if len(text.encode("utf-8")) > maximum:
                raise PreparationError(spec, "wikipedia_record_exceeds_bound", index)
            title = children.get("title")
            page_id = children.get("id")
            timestamp = revision.get("timestamp")
            revision_id = revision.get("id")
            yield SourceRow(index, {
                "wikitext": text,
                "title": title.text if title is not None else None,
                "page_id": page_id.text if page_id is not None else None,
                "revision_id": revision_id.text if revision_id is not None else None,
                "timestamp": timestamp.text if timestamp is not None else None,
                "namespace": 0,
                "extraction": {"method": "wikipedia_current_wikitext", "markup_preserved": True},
            })
        page.clear()
        root.clear()


def iter_source_rows(
    spec: ObjectSpec, path: Path, *, batch_size: int, maximum_record_bytes: int,
) -> Iterator[SourceRow]:
    if spec.wire_format not in FORMATS:
        raise PreparationError(spec, "unsupported_wire_format")
    if spec.adapter not in ADAPTERS:
        raise PreparationError(spec, "unsupported_adapter")
    row_index: int | str = "-"
    try:
        if spec.wire_format == "parquet":
            import pyarrow.parquet as pq
            with pq.ParquetFile(path) as parquet:
                _validate_fields(spec, set(parquet.schema_arrow.names), "schema")
                row_index = 0
                for batch in parquet.iter_batches(batch_size=batch_size):
                    for value in batch.to_pylist():
                        row_index += 1
                        yield SourceRow(row_index, value)
            return
        with path.open("rb") as raw:
            magic = raw.read(4)
            raw.seek(0)
            if spec.wire_format in {"jsonl_gzip", "warc_wet_gzip", "warc_gzip"}:
                if not magic.startswith(b"\x1f\x8b"):
                    raise PreparationError(spec, "invalid_gzip_container")
                with gzip.GzipFile(fileobj=raw) as stream:
                    rows = (_warc_rows(stream, spec, maximum_record_bytes)
                            if spec.wire_format.startswith("warc") else
                            _json_rows(stream, spec, maximum_record_bytes))
                    for item in rows:
                        row_index = item.index
                        yield item
            elif spec.wire_format in {"jsonl_zstd", "json_zstd"}:
                import zstandard
                checked = _FrameCheckedZstd(raw)
                try:
                    with zstandard.ZstdDecompressor().stream_reader(
                        checked, read_across_frames=True,
                    ) as decompressed:
                        with io.BufferedReader(decompressed) as stream:
                            for item in _json_rows(stream, spec, maximum_record_bytes):
                                row_index = item.index
                                yield item
                except zstandard.ZstdError:
                    raise PreparationError(spec, "invalid_zstandard_payload", row_index) from None
                checked.finish()
            elif spec.wire_format == "xml_bzip2":
                if not magic.startswith(b"BZh"):
                    raise PreparationError(spec, "invalid_bzip2_container")
                with bz2.BZ2File(raw) as stream:
                    for item in _xml_rows(stream, spec, maximum_record_bytes):
                        row_index = item.index
                        yield item
            else:
                for item in _json_rows(raw, spec, maximum_record_bytes):
                    row_index = item.index
                    yield item
    except PreparationError:
        raise
    except (OSError, EOFError, ValueError, ET.ParseError, RecursionError, zlib.error):
        raise PreparationError(spec, "object_parser_or_truncation_error", row_index) from None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return {"binary_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(nested) for nested in value]
    raise ValueError("unsupported_metadata_value")


def _decode_field(value: Any, spec: ObjectSpec, index: int, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, RecursionError):
            raise PreparationError(spec, f"invalid_json_encoded_{field}", index) from None
    return value


def _lineage(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _lineage(nested) for key, nested in value.items()}
        lineage = [result[key] for key in ("source_dataset", "hf_dataset_name", "seed_source",
                                         "dataset", "dataset_name", "seed_dataset", "upstream_dataset")
                   if result.get(key) is not None]
        if lineage:
            result["source_dataset"] = lineage[0] if len(lineage) == 1 else lineage
        return result
    if isinstance(value, list):
        return [_lineage(nested) for nested in value]
    return value


def published_http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or parsed.username is not None or parsed.password is not None
            or not _normalise_host(parsed.hostname or "")
        ):
            return None
        _ = parsed.port
    except ValueError:
        return None
    return value


def _hplt_language_evidence(
    direct: Mapping[str, Any], result: dict[str, Any], evidence: list[dict[str, Any]],
    spec: ObjectSpec, index: int,
) -> None:
    labels = direct.get("lang")
    probabilities = direct.get("prob")
    if not isinstance(labels, list):
        return
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise PreparationError(spec, "invalid_hplt_language_labels", index)
    result["language_labels"] = list(labels)
    if probabilities is None:
        if len(labels) == 1:
            result["language"] = labels[0]
            evidence.append({"field": "language", "method": "single_published_language", "source_field": "lang"})
        return
    if (
        not isinstance(probabilities, list) or len(probabilities) != len(labels)
        or any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
               for value in probabilities)
    ):
        raise PreparationError(spec, "invalid_hplt_language_probability_vector", index)
    result["language_probabilities"] = list(probabilities)
    if labels:
        selected = max(range(len(labels)), key=lambda position: probabilities[position])
        result["language"] = labels[selected]
        result["language_probability"] = probabilities[selected]
        evidence.append({
            "field": "language", "method": "maximum_published_language_probability",
            "source_field": "lang", "probability_source_field": "prob", "selected_index": selected,
        })


def record_metadata(row: Mapping[str, Any], spec: ObjectSpec, index: int) -> dict[str, Any]:
    excluded = set(TEXT_FIELDS) | set(STRUCTURED_FIELDS) | {"files", "pages"}
    upstream = _json_value({key: value for key, value in row.items() if key not in excluded})
    result: dict[str, Any] = {"upstream": upstream, "row_index": index}
    direct = dict(upstream)
    for key in ("meta", "metadata"):
        if row.get(key) is not None:
            decoded = _decode_field(row[key], spec, index, key)
            if not isinstance(decoded, (Mapping, list)):
                raise PreparationError(spec, f"{key}_is_not_a_metadata_container", index)
            result[key] = _json_value(decoded)
            if isinstance(decoded, Mapping):
                for name, value in decoded.items():
                    direct.setdefault(str(name), _json_value(value))
    for name, value in direct.items():
        if name not in {"upstream", "normalization_evidence", "row_index", "source_defaults"}:
            result.setdefault(name, value)
    defaults = spec.policy.get("metadata", {})
    if not isinstance(defaults, Mapping):
        raise PreparationError(spec, "source_metadata_defaults_must_be_a_mapping")
    if set(defaults) & _PER_RECORD_DEFAULTS or any(str(k).endswith("_passed") for k in defaults):
        raise PreparationError(spec, "per_record_evidence_cannot_be_a_source_default")
    result["source_defaults"] = _json_value(defaults)
    evidence = []
    for key, value in defaults.items():
        if key not in result or result[key] is None:
            result[key] = _json_value(value)
            evidence.append({"field": key, "method": "explicit_source_policy_default"})
    result.pop("license", None)
    for field in LICENSE_FIELDS:
        value = direct.get(field)
        if value is not None and (license_value := _normalise_license(value)):
            if license_value.lower().replace("-", "").replace("_", "") in {"noassertion", "unlicensed"}:
                continue
            result["license"] = license_value
            evidence.append({"field": "license", "method": "upstream_per_record_license",
                             "source_field": field})
            break
    if not result.get("license") and isinstance(direct.get("file_info"), Mapping):
        license_value = _normalise_license(direct["file_info"].get("detected_licenses"))
        if license_value:
            result["license"] = license_value
            evidence.append({"field": "license", "method": "upstream_per_record_license",
                             "source_field": "file_info.detected_licenses"})
    for key in tuple(result):
        if key.endswith("_passed") and type(result[key]) is not bool:
            del result[key]
    is_hplt = spec.source_id.lower().startswith("hplt")
    url_aliases = ("canonical_url", "url", "source_url", "original_url")
    if is_hplt:
        url_aliases += ("u",)
    for canonical, aliases in {
        "language": ("language", "lang", "doc_language"),
        "canonical_url": url_aliases,
    }.items():
        for alias in aliases:
            if isinstance(direct.get(alias), str) and direct[alias]:
                result[canonical] = direct[alias]
                evidence.append({"field": canonical, "method": "upstream_field", "source_field": alias})
                break
    if is_hplt:
        _hplt_language_evidence(direct, result, evidence, spec, index)
        if isinstance(direct.get("crawl_id"), str) and direct["crawl_id"]:
            result["crawl"] = direct["crawl_id"]
            evidence.append({"field": "crawl", "method": "upstream_field", "source_field": "crawl_id"})
    if spec.source_id.startswith("nemotron_cc") and not published_http_url(result.get("canonical_url")):
        result["document_url_status"] = "not_published"
    result["normalization_evidence"] = evidence
    return _lineage(result)


def _structured_text(row: Mapping[str, Any], spec: ObjectSpec, index: int) -> str:
    key = "messages" if "messages" in row else "conversations"
    messages = _decode_field(row[key], spec, index, key)
    if not isinstance(messages, list):
        raise PreparationError(spec, "messages_are_not_an_array", index)
    if not messages:
        return ""
    if any(not isinstance(message, Mapping) for message in messages):
        raise PreparationError(spec, "message_is_not_an_object", index)
    if any(not ({"content", "value", "tool_calls", "function_call"} & set(message)) for message in messages):
        raise PreparationError(spec, "message_has_no_payload", index)
    payload = {key: messages}
    if row.get("tools") is not None:
        tools = _decode_field(row["tools"], spec, index, "tools")
        if not isinstance(tools, (list, dict)):
            raise PreparationError(spec, "tools_have_invalid_schema", index)
        payload["tools"] = tools
    return canonical_json(_json_value(payload))


def _pointer_text(text: str) -> bool:
    stripped = text.strip()
    return bool(
        (stripped.splitlines()[0:1] == ["version https://git-lfs.github.com/spec/v1"]
         and re.search(r"(?m)^oid sha256:[0-9a-fA-F]{64}\s*$", stripped)
         and re.search(r"(?m)^size \d+\s*$", stripped))
        or re.fullmatch(r"swh:1:cnt:[0-9a-fA-F]{40}(?:;[^\s]*)?", stripped)
    )


def extract_documents(item: SourceRow, spec: ObjectSpec) -> Iterator[Document]:
    if item.rejection:
        yield Document("row", None, {"row_index": item.index}, rejection=item.rejection)
        return
    row = item.value
    _validate_fields(spec, set(row), item.index)
    try:
        metadata = record_metadata(row, spec, item.index)
        if row.get("files") is not None:
            files = _decode_field(row["files"], spec, item.index, "files")
            if not isinstance(files, list):
                raise PreparationError(spec, "files_are_not_an_array", item.index)
            if not files:
                yield Document("row", None, metadata, rejection="empty_file_list")
            for position, value in enumerate(files):
                if not isinstance(value, Mapping):
                    raise PreparationError(spec, "file_component_is_not_an_object", item.index)
                file_metadata = record_metadata(value, spec, item.index)
                combined = {**metadata, **file_metadata, "repository_metadata": metadata,
                            "file_index": position, "file_count": len(files)}
                if not file_metadata.get("license") and metadata.get("license"):
                    combined["license"] = metadata["license"]
                if any(value.get(field) is not None and not isinstance(value[field], str) for field in TEXT_FIELDS):
                    raise PreparationError(spec, "file_text_is_not_a_string", item.index)
                text = next((value[field] for field in TEXT_FIELDS if isinstance(value.get(field), str)
                             and value[field].strip()),
                            "" if any(isinstance(value.get(field), str) for field in TEXT_FIELDS) else None)
                reason = "pointer_only_code" if text is None or (text and _pointer_text(text)) else None
                yield Document(f"files/{position}", text, combined, rejection=reason)
            return
        french = spec.adapter in {"frenchscience", "french_science"} or "frenchscience" in re.sub(
            r"[^a-z0-9]", "", spec.source_id.lower(),
        )
        if french and row.get("pages") is not None:
            pages = _decode_field(row["pages"], spec, item.index, "pages")
            if not isinstance(pages, list):
                raise PreparationError(spec, "pages_are_not_an_array", item.index)
            texts = []
            for page in pages:
                text = page if isinstance(page, str) else next(
                    (page[key] for key in TEXT_FIELDS if isinstance(page, Mapping) and isinstance(page.get(key), str)), None,
                )
                if text is None:
                    raise PreparationError(spec, "page_has_no_inline_text", item.index)
                texts.append(text)
            text = "\n".join(texts)
            metadata["assembly_page_count"] = len(texts)
        elif spec.adapter in {"messages", "conversation"} or (
            not any(isinstance(row.get(key), str) and row[key].strip() for key in TEXT_FIELDS)
            and any(key in row for key in ("messages", "conversations"))
        ):
            text = _structured_text(row, spec, item.index)
            metadata["extraction"] = {"method": "ordered_conversation_json", "tools_executed": False}
        else:
            fields = (spec.adapter,) + tuple(key for key in TEXT_FIELDS if key != spec.adapter)
            text = next((row[key] for key in fields if isinstance(row.get(key), str) and row[key].strip()),
                        "" if any(isinstance(row.get(key), str) for key in fields) else None)
            if "tools" in row and row["tools"] is not None:
                metadata["tools"] = _decode_field(row["tools"], spec, item.index, "tools")
        if text is None:
            if set(row) & _POINTER_FIELDS:
                yield Document("row", None, metadata, rejection="pointer_only_code")
                return
            if any(key in row and row[key] is not None for key in TEXT_FIELDS):
                raise PreparationError(spec, "inline_text_is_not_a_string", item.index)
            yield Document("row", None, metadata, rejection="empty_text")
            return
        quarantine = None
        if french:
            completeness = [metadata.get(key) for key in ("document_complete", "is_complete", "pages_complete")]
            complete = any(value is True for value in completeness)
            page_count = metadata.get("page_count", metadata.get("num_pages"))
            complete = complete or (type(page_count) is int and page_count == 1)
            if "pages" in row and type(page_count) is int and page_count > 0:
                complete = complete or page_count == metadata.get("assembly_page_count")
            if any(value is False for value in completeness):
                complete = False
            if not complete:
                quarantine = "source_local_assembly_pending"
        yield Document("row", text, metadata,
                       rejection="pointer_only_code" if _pointer_text(text) else None,
                       quarantine=quarantine)
    except PreparationError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise PreparationError(spec, "invalid_adapter_metadata", item.index) from None


def normalize_document(document: Document, spec: ObjectSpec, row_index: int) -> dict[str, Any]:
    assert document.text is not None
    original = document.text
    text = original.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        encoded = text.encode("utf-8")
        original_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    except UnicodeError:
        raise PreparationError(spec, "invalid_unicode_text", row_index) from None
    metadata = dict(document.metadata)
    metadata["component"] = document.component
    metadata["normalization"] = {
        "version": ADAPTER_VERSION,
        "original_content_sha256": original_hash,
        "newlines_changed": "\r" in original,
        "leading_bom_removed": original.startswith("\ufeff"),
        "unicode_normalization": "none",
        "whitespace_and_case": "preserved",
    }
    if document.quarantine:
        metadata["admission_block"] = document.quarantine
    score = None
    score_field = spec.policy.get("quality_score_field")
    fields = (str(score_field),) if score_field else ("quality_score", "educational_score", "math_score")
    for field in fields:
        if metadata.get(field) is not None:
            value = metadata[field]
            if isinstance(value, bool):
                raise PreparationError(spec, "invalid_publisher_quality_score", row_index)
            try:
                score = float(value)
            except (TypeError, ValueError):
                raise PreparationError(spec, "invalid_publisher_quality_score", row_index) from None
            if not math.isfinite(score) or score == -1:
                raise PreparationError(spec, "invalid_publisher_quality_score", row_index)
            metadata["quality_score_evidence"] = {"field": field, "method": "published_value"}
            metadata["quality_score"] = score
            break
    metadata["quality_score_status"] = "published" if score is not None else "unknown"
    language = metadata.get("language") or spec.policy.get("language", "und")
    if not isinstance(language, str):
        raise PreparationError(spec, "invalid_language_label", row_index)
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        metadata_json = canonical_json(metadata)
        metadata_json.encode("utf-8")
    except UnicodeError:
        raise PreparationError(spec, "invalid_unicode_metadata", row_index) from None
    return {
        "doc_id": hashlib.sha256(f"{spec.object_id}\0{row_index}\0{document.component}".encode()).hexdigest(),
        "content_hash": digest,
        "dedup_hash": digest,
        "source_id": spec.source_id,
        "object_id": spec.object_id,
        "text": text,
        "metadata_json": metadata_json,
        "priority": spec.priority,
        "quality_score": score if score is not None else -1.0,
        "language": language if language not in {"any", "*", ""} else "und",
        "category": str(spec.policy.get("category", "unknown")),
        "character_count": len(text),
    }
