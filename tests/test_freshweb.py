from __future__ import annotations

import base64
import dataclasses
import gzip
import hashlib
import inspect
import itertools
import json
import re
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from metis_data import freshweb
from metis_data.freshweb import (
    CATEGORY_PRIORITY,
    DEFAULT_CATEGORY_WEIGHTS,
    INDEX_COLUMNS,
    REQUIRED_INDEX_COLUMNS,
    FreshWebError,
    FreshWebOptions,
    OptOutPolicy,
    PrefilterTally,
    _HttpClient,
    _connect_state,
    _download_resumable,
    _prefilter_index_batch,
    _prepare_index_selection_round,
    _extract_warc_member,
    _url_index_integrity_path,
    canonicalize_url,
    coalesce_ranges,
    materialize_freshweb,
    metadata_candidate,
    parse_opt_out_registry,
    select_exact_candidates,
    select_partition_candidates,
    snapshot_common_crawl_opt_out,
)
from metis_data.download import (
    FRESHWEB_ROUTE_SETTINGS,
    _materialize_common_crawl,
    freshweb_options_for_item,
    run_download_task,
)
from metis_data.handoff import _iter_output_records
from metis_data.state import StateStore


def _payload_digest(payload: bytes) -> str:
    return base64.b32encode(hashlib.sha1(payload).digest()).decode("ascii").rstrip("=")


def _warc_member(
    url: str,
    text: str,
    record_id: str,
    *,
    html: str | None = None,
    response_headers: tuple[tuple[str, str], ...] = (),
) -> tuple[bytes, str]:
    body = (
        html
        or (
            "<!doctype html><html lang='en'><head><title>Metis source</title>"
            "<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'></head>"
            f"<body><nav>boilerplate menu</nav><main><h1>Current guide</h1><p>{text}</p>"
            "<pre>def verified_example(value):\n    return value + 1</pre></main>"
            "<footer>boilerplate footer</footer></body></html>"
        )
    ).encode("utf-8")
    digest = _payload_digest(body)
    http = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html; charset=UTF-8\r\n"
        + b"".join(f"{name}: {value}\r\n".encode("utf-8") for name, value in response_headers)
        + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )
    warc = (
        b"WARC/1.0\r\n"
        b"WARC-Type: response\r\n"
        b"WARC-Date: 2026-06-18T12:00:00Z\r\n"
        + f"WARC-Record-ID: <urn:uuid:{record_id}>\r\n".encode("ascii")
        + f"WARC-Target-URI: {url}\r\n".encode("utf-8")
        + f"WARC-Payload-Digest: sha1:{digest}\r\n".encode("ascii")
        + b"WARC-Identified-Payload-Type: text/html\r\n"
        + f"Content-Length: {len(http)}\r\n".encode("ascii")
        + b"Content-Type: application/http; msgtype=response\r\n\r\n"
        + http
        + b"\r\n\r\n"
    )
    return gzip.compress(warc, mtime=0), digest


def _coordinate(url: str, digest: str, *, route: str, category: str) -> dict[str, object]:
    return {
        "source_id": f"fixture-{route}",
        "crawl": "CC-MAIN-2026-25",
        "url": url,
        "canonical_url": canonicalize_url(url),
        "host": urlsplit(url).hostname,
        "registered_domain": urlsplit(url).hostname,
        "private_domain": urlsplit(url).hostname,
        "fetch_time": "2026-06-18T12:00:00+00:00",
        "capture_date": "2026-06-18",
        "content_digest": digest,
        "content_mime_type": "text/html",
        "content_mime_detected": "text/html",
        "content_charset": "UTF-8",
        "content_languages": ["eng"],
        "warc_record_id": "fixture-record",
        "warc_filename": "crawl-data/CC-MAIN-2026-25/segments/1/warc/fixture.warc.gz",
        "warc_record_offset": 0,
        "warc_record_length": 1,
        "warc_segment": "1",
        "fresh_category": category,
        "route": route,
        "priority": 100,
        "opt_out_snapshot_sha256": "f" * 64,
    }


class _FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        self.server.counts[path] += 1  # type: ignore[attr-defined]
        payload = self.server.files.get(path)  # type: ignore[attr-defined]
        if payload is None:
            self.send_error(404)
            return
        range_header = self.headers.get("Range")
        if range_header:
            unit, value = range_header.split("=", 1)
            if unit != "bytes":
                self.send_error(400)
                return
            start_text, end_text = value.split("-", 1)
            start = int(start_text)
            end = int(end_text) if end_text else len(payload) - 1
            if start >= len(payload) or end < start:
                self.send_response(416)
                self.end_headers()
                return
            end = min(end, len(payload) - 1)
            response = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
        else:
            response = payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(response)))
        self.send_header("ETag", f'"{hashlib.sha256(payload).hexdigest()}"')
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FixtureServer:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        self.server.files = files  # type: ignore[attr-defined]
        self.server.counts = Counter()  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_FixtureServer":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def root(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/"

    @property
    def counts(self) -> Counter[str]:
        return self.server.counts  # type: ignore[attr-defined]


class FreshWebUnitTests(unittest.TestCase):
    def test_canonicalization_opt_out_and_exact_selection(self) -> None:
        self.assertEqual(
            canonicalize_url("HTTPS://Example.COM:443/a//b/?utm_source=x&b=2&a=1#fragment"),
            "https://example.com/a/b?a=1&b=2",
        )
        registry = parse_opt_out_registry(
            b"Publisher/Requester,Date of notice,List of domains/URLs\n"
            b"LAST UPDATED: 2026 Jul 21,,\n"
            b"Publisher,2026-07-21,example.com\n"
            b"Person,2026-07-21,https://allowed.test/private/?x=1\n"
            b'Publisher group,2026-07-21,"first.test, second.test/private/*"\n'
        )
        self.assertEqual(registry.reason("https://sub.example.com/page"), "common_crawl_opt_out_domain")
        self.assertEqual(registry.reason("https://allowed.test/private?x=1"), "common_crawl_opt_out_url")
        self.assertIsNone(registry.reason("https://allowed.test/private?different=1"))
        self.assertIsNone(registry.reason("https://allowed.test/public"))
        self.assertEqual(registry.reason("https://www.first.test/anything"), "common_crawl_opt_out_domain")
        self.assertEqual(
            registry.reason("https://second.test/private/nested"), "common_crawl_opt_out_url"
        )

        base = {
            "priority": 90,
            "full_body_likelihood": 1,
            "warc_record_length": 2_000,
            "sample_hash": "8" * 64,
            "warc_filename": "crawl-data/a.warc.gz",
            "warc_record_offset": 0,
        }
        rows = [
            {**base, "canonical_url": "https://a.test/page", "content_digest": "A" * 32, "fetch_epoch": 1},
            {**base, "canonical_url": "https://a.test/page", "content_digest": "B" * 32, "fetch_epoch": 2},
            {
                **base,
                "canonical_url": "https://mirror.test/page",
                "content_digest": "B" * 32,
                "fetch_epoch": 3,
                "priority": 70,
            },
        ]
        selected = select_exact_candidates(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["canonical_url"], "https://a.test/page")
        self.assertEqual(selected[0]["content_digest"], "B" * 32)

    def test_coalescing_respects_gap_and_span_limits(self) -> None:
        records = [
            {"warc_record_offset": 0, "warc_record_length": 100},
            {"warc_record_offset": 120, "warc_record_length": 80},
            {"warc_record_offset": 1_000, "warc_record_length": 100},
        ]
        spans = coalesce_ranges(records, maximum_gap=32, maximum_span=500)
        self.assertEqual([(span.start, span.end, len(span.records)) for span in spans], [(0, 199, 2), (1_000, 1_099, 1)])

    def test_url_index_cache_repairs_corrupt_footer_using_integrity_sidecar(self) -> None:
        row = {
            "url": "https://docs.example.org/guide",
            "fetch_time": "2026-06-18T00:00:00Z",
            "fetch_status": 200,
            "content_digest": "A" * 32,
            "content_mime_type": "text/html",
            "content_truncated": None,
            "warc_filename": "crawl-data/CC-MAIN-2026-25/segments/1/warc/a.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": 512,
        }
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.parquet"
            pq.write_table(pa.Table.from_pylist([row]), source_path)
            payload = source_path.read_bytes()
            with _FixtureServer({"/index.parquet": payload}) as server:
                options = FreshWebOptions(
                    max_workers=1,
                    retry_base_seconds=0,
                    request_timeout_seconds=10,
                )
                http = _HttpClient(options)
                destination = Path(temporary) / "cache" / "index.parquet"
                _download_resumable(http, server.root + "index.parquet", destination)
                sidecar_path = _url_index_integrity_path(destination)
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.assertEqual(sidecar["content_length"], len(payload))
                self.assertTrue(sidecar["etag"])
                self.assertEqual(sidecar["size"], len(payload))
                self.assertEqual(server.counts["/index.parquet"], 1)

                corrupt = destination.read_bytes()[:-4] + b"NOPE"
                destination.write_bytes(corrupt)
                # Preserve a self-consistent size/hash sidecar so the dedicated
                # Parquet-footer validation, not only checksum validation, must
                # detect and repair this cache entry.
                sidecar["sha256"] = hashlib.sha256(corrupt).hexdigest()
                sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
                _download_resumable(http, server.root + "index.parquet", destination)

                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(server.counts["/index.parquet"], 2)

    def test_widened_selection_round_clears_stale_winners_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            connection = _connect_state(Path(temporary) / "state.sqlite3")
            try:
                _prepare_index_selection_round(connection, selection_round=0)
                with connection:
                    connection.execute(
                        """
                        INSERT INTO partitions(
                            relative_path, crawl, selection_capacity, scanned, eligible,
                            selected, rejections_json, categories_json
                        ) VALUES ('part.parquet', 'CC-MAIN-2026-25', 1, 1, 1, 1, '{}', '{}')
                        """
                    )
                    connection.execute(
                        """
                        INSERT INTO url_winners(
                            canonical_url, content_digest, priority, full_body_likelihood,
                            fetch_epoch, warc_record_length, sample_hash, warc_filename,
                            warc_record_offset, payload_json
                        ) VALUES (
                            'https://stale.example/', 'STALE', 1, 1, 1, 1,
                            'aaaaaaaa', 'stale.warc.gz', 0, '{}'
                        )
                        """
                    )

                _prepare_index_selection_round(connection, selection_round=1)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM url_winners").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT selection_capacity FROM partitions"
                    ).fetchone()[0],
                    0,
                )

                # Re-entering the same round is a crash resume, so already
                # rescanned candidates must remain committed.
                with connection:
                    connection.execute(
                        "UPDATE partitions SET selection_capacity = 2"
                    )
                    connection.execute(
                        """
                        INSERT INTO url_winners(
                            canonical_url, content_digest, priority, full_body_likelihood,
                            fetch_epoch, warc_record_length, sample_hash, warc_filename,
                            warc_record_offset, payload_json
                        ) VALUES (
                            'https://current.example/', 'CURRENT', 1, 1, 1, 1,
                            'bbbbbbbb', 'current.warc.gz', 0, '{}'
                        )
                        """
                    )
                _prepare_index_selection_round(connection, selection_round=1)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM url_winners").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT selection_capacity FROM partitions"
                    ).fetchone()[0],
                    2,
                )
            finally:
                connection.close()

    def test_route_category_gate_rejects_out_of_scope_pages(self) -> None:
        policy = OptOutPolicy(
            domains=frozenset(),
            url_paths=frozenset(),
            snapshot_sha256="f" * 64,
            last_updated="2026 Jul 21",
        )
        options = freshweb_options_for_item(
            {
                "source_id": "software",
                "license": {"status": "per_record_required"},
                "access": {
                    "route": "software_docs",
                    "cutoff_start": "2025-01-01",
                    "cutoff_end": "2026-06-30",
                },
            },
            {},
        )

        def row(url: str) -> dict[str, object]:
            parsed = urlsplit(url)
            return {
                "url": url,
                "url_host_registered_domain": parsed.hostname,
                "url_host_registry_suffix": (parsed.hostname or "").split(".")[-1],
                "fetch_time": "2026-06-18T12:00:00Z",
                "fetch_status": 200,
                "content_digest": "A" * 32,
                "content_mime_type": "text/html",
                "content_mime_detected": "text/html",
                "content_charset": "UTF-8",
                "content_languages": "eng",
                "content_truncated": None,
                "warc_filename": "crawl-data/CC-MAIN-2026-25/segments/1/warc/a.warc.gz",
                "warc_record_offset": 0,
                "warc_record_length": 1_000,
            }

        rejected, reason = metadata_candidate(
            row("https://example.com/careers/current"),
            crawl="CC-MAIN-2026-25",
            source_id="software",
            policy=policy,
            options=options,
        )
        self.assertIsNone(rejected)
        self.assertEqual(reason, "route_category")
        accepted, reason = metadata_candidate(
            row("https://example.com/docs/api/v2/guide"),
            crawl="CC-MAIN-2026-25",
            source_id="software",
            policy=policy,
            options=options,
        )
        self.assertIsNone(reason)
        self.assertEqual(accepted["fresh_category"], "official_docs")
        self.assertEqual(accepted["route"], "software_docs")
        self.assertEqual(accepted["capture_date"], "2026-06-18")
        stale = row("https://example.com/docs/api/v2/guide")
        stale["fetch_time"] = "2024-12-31T23:59:59Z"
        rejected, reason = metadata_candidate(
            stale,
            crawl="CC-MAIN-2026-25",
            source_id="software",
            policy=policy,
            options=options,
        )
        self.assertIsNone(rejected)
        self.assertEqual(reason, "capture_before_freshness_cutoff")

    def test_science_and_documentation_routes_require_auditable_evidence(self) -> None:
        science_options = freshweb_options_for_item(
            {
                "source_id": "science",
                "license": {"status": "per_record_required"},
                "access": {
                    "route": "fresh_science",
                    "cutoff_start": "2025-01-01",
                    "cutoff_end": "2026-06-30",
                },
            },
            {},
        )
        science_text = (
            "The research team reports a reproducible experiment with methods, controls, "
            "measurements, uncertainty, results, and a careful technical discussion. " * 20
        )
        science_url = "https://research.example.edu/publication/new-study"
        science_html = (
            "<!doctype html><html lang='en'><head>"
            "<title>Reproducible materials study</title>"
            "<link rel='canonical' href='/publication/new-study'>"
            "<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'>"
            "<meta name='citation_publication_date' content='2026-05-14'>"
            "</head><body><main><article><h1>Reproducible materials study</h1>"
            f"<p>{science_text}</p><p>{science_text}</p>"
            "</article></main></body></html>"
        )
        member, digest = _warc_member(
            science_url,
            science_text,
            "00000000-0000-0000-0000-000000000101",
            html=science_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(science_url, digest, route="fresh_science", category="science"),
            options=science_options,
        )
        self.assertIsNone(reason)
        metadata = document["metadata"]
        self.assertEqual(metadata["publication_date"], "2026-05-14")
        self.assertTrue(metadata["license_evaluation"]["decision"])
        self.assertEqual(metadata["license_evidence"][0]["source"], "html_link")
        self.assertEqual(
            metadata["license_evidence"][0]["value"],
            "https://creativecommons.org/licenses/by/4.0/",
        )
        self.assertEqual(
            metadata["declared_canonical_url"],
            "https://research.example.edu/publication/new-study",
        )
        self.assertTrue(metadata["structural_quality"]["passed"])
        self.assertIn("html_lang_en", metadata["english_evidence"]["reasons"])

        missing_license_html = science_html.replace(
            "<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'>", ""
        )
        member, digest = _warc_member(
            science_url,
            science_text,
            "00000000-0000-0000-0000-000000000102",
            html=missing_license_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(science_url, digest, route="fresh_science", category="science"),
            options=science_options,
        )
        self.assertIsNone(document)
        self.assertEqual(reason, "missing_reusable_open_license")

        missing_date_html = science_html.replace(
            "<meta name='citation_publication_date' content='2026-05-14'>", ""
        )
        member, digest = _warc_member(
            science_url,
            science_text,
            "00000000-0000-0000-0000-000000000103",
            html=missing_date_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(science_url, digest, route="fresh_science", category="science"),
            options=science_options,
        )
        self.assertIsNone(document)
        self.assertEqual(reason, "missing_publication_date")

        stale_date_html = science_html.replace("2026-05-14", "2024-12-31")
        member, digest = _warc_member(
            science_url,
            science_text,
            "00000000-0000-0000-0000-000000000104",
            html=stale_date_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(science_url, digest, route="fresh_science", category="science"),
            options=science_options,
        )
        self.assertIsNone(document)
        self.assertEqual(reason, "publication_outside_freshness_cutoff")

        docs_options = freshweb_options_for_item(
            {
                "source_id": "docs",
                "license": {"status": "per_record_required"},
                "access": {"route": "official_docs"},
            },
            {},
        )
        docs_url = "https://developer.example.com/docs/guide"
        docs_text = (
            "This API documentation explains the current request format, response fields, "
            "error behavior, migration details, examples, and compatibility guarantees. " * 15
        )
        docs_html = (
            "<!doctype html><html lang='en'><head><title>Example API documentation</title>"
            "<meta name='docsearch:version' content='3.2'>"
            "<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'>"
            "<link rel='canonical' href='/docs/v3.2/guide'></head>"
            f"<body><main><h1>Example API documentation</h1><p>{docs_text}</p>"
            f"<p>{docs_text}</p></main></body></html>"
        )
        member, digest = _warc_member(
            docs_url,
            docs_text,
            "00000000-0000-0000-0000-000000000104",
            html=docs_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(docs_url, digest, route="official_docs", category="official_docs"),
            options=docs_options,
        )
        self.assertIsNone(reason)
        self.assertEqual(document["metadata"]["version_evidence"][0], {
            "source": "html_meta",
            "key": "docsearch:version",
            "value": "3.2",
        })

        unlicensed_docs_html = docs_html.replace(
            "<link rel='license' href='https://creativecommons.org/licenses/by/4.0/'>",
            "",
        )
        member, digest = _warc_member(
            docs_url,
            docs_text,
            "00000000-0000-0000-0000-000000000106",
            html=unlicensed_docs_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(docs_url, digest, route="official_docs", category="official_docs"),
            options=docs_options,
        )
        self.assertIsNone(document)
        self.assertEqual(reason, "missing_reusable_open_license")

        no_version_html = docs_html.replace(
            "<meta name='docsearch:version' content='3.2'>", ""
        ).replace("/docs/v3.2/guide", "/docs/guide")
        member, digest = _warc_member(
            docs_url,
            docs_text,
            "00000000-0000-0000-0000-000000000105",
            html=no_version_html,
        )
        document, reason = _extract_warc_member(
            member,
            _coordinate(docs_url, digest, route="official_docs", category="official_docs"),
            options=docs_options,
        )
        self.assertIsNone(document)
        self.assertEqual(reason, "missing_version_evidence")

    def test_public_opt_out_snapshot_helper_is_hash_addressed_and_atomic(self) -> None:
        registry = (
            b"Publisher/Requester,Date of notice,List of domains/URLs\n"
            b"LAST UPDATED: 2026 Jul 22,,\n"
            b'Publisher,2026-07-22,"blocked.example, private.example/path/*"\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            with _FixtureServer({"/optout.csv": registry}) as server:
                output = snapshot_common_crawl_opt_out(
                    Path(temporary) / "compliance",
                    url=server.root + "optout.csv",
                    options=FreshWebOptions(max_workers=1),
                )
                self.assertEqual(output["sha256"], hashlib.sha256(registry).hexdigest())
                self.assertEqual(output["domains"], 1)
                self.assertEqual(output["url_rules"], 1)
                self.assertEqual(output["unparsed_entries"], 0)
                self.assertTrue(Path(output["path"]).is_file())
                self.assertTrue(Path(output["rules_path"]).is_file())
                self.assertEqual(
                    hashlib.sha256(Path(output["rules_path"]).read_bytes()).hexdigest(),
                    output["rules_sha256"],
                )
                normalized_rules = json.loads(Path(output["rules_path"]).read_text())
                self.assertEqual(normalized_rules["domains"], ["blocked.example"])
                self.assertEqual(normalized_rules["url_rules"][0]["host"], "private.example")
                self.assertTrue(Path(output["metadata_path"]).is_file())
                pointer = Path(temporary) / "compliance" / "LATEST_COMMON_CRAWL_OPT_OUT.json"
                self.assertEqual(json.loads(pointer.read_text())["sha256"], output["sha256"])
                second = snapshot_common_crawl_opt_out(
                    Path(temporary) / "compliance",
                    url=server.root + "optout.csv",
                    options=FreshWebOptions(max_workers=1),
                )
                self.assertEqual(second["path"], output["path"])


class FreshWebMaterializerTests(unittest.TestCase):
    def test_end_to_end_materialization_is_filtered_exact_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary)
            warc_path = "crawl-data/CC-MAIN-2026-25/segments/1.0/warc/fixture.warc.gz"
            members: list[bytes] = []
            rows: list[dict[str, object]] = []

            def add(url: str, text: str, record_id: str, fetch_time: str, *, digest_override: str | None = None) -> None:
                member, digest = _warc_member(url, text, record_id)
                offset = sum(len(value) for value in members)
                members.append(member)
                parsed = urlsplit(url)
                rows.append(
                    {
                        "url": url,
                        "url_surtkey": parsed.hostname,
                        "url_host_name": parsed.hostname,
                        "url_host_registered_domain": parsed.hostname,
                        "url_host_private_domain": parsed.hostname,
                        "url_host_registry_suffix": (parsed.hostname or "").split(".")[-1],
                        "url_protocol": parsed.scheme,
                        "url_path": parsed.path,
                        "url_query": parsed.query,
                        "fetch_time": fetch_time,
                        "fetch_status": 200,
                        "fetch_redirect": None,
                        "content_digest": digest_override or digest,
                        "content_mime_type": "text/html",
                        "content_mime_detected": "text/html",
                        "content_charset": "UTF-8",
                        "content_languages": "eng",
                        "content_truncated": None,
                        "warc_record_id": record_id,
                        "warc_filename": warc_path,
                        "warc_record_offset": offset,
                        "warc_record_length": len(member),
                        "warc_segment": "1.0",
                    }
                )

            old_text = "This older guide should lose the exact canonical URL decision. " * 4
            current_text = "This is the current authoritative API guide with concrete examples. " * 5
            lesson_text = "This university lesson explains a recent scientific result carefully. " * 5
            blocked_text = "This publisher requested exclusion and must never reach the WARC fetch plan. " * 5
            add("https://docs.example.org/guide?utm_source=old", old_text, "00000000-0000-0000-0000-000000000001", "2026-02-10T00:00:00Z")
            add("https://docs.example.org/guide", current_text, "00000000-0000-0000-0000-000000000002", "2026-06-18T00:00:00Z")
            current_digest = rows[-1]["content_digest"]
            add(
                "https://mirror.test/copied",
                current_text,
                "00000000-0000-0000-0000-000000000003",
                "2026-06-18T01:00:00Z",
                digest_override=str(current_digest),
            )
            add("https://courses.university.edu/lesson", lesson_text, "00000000-0000-0000-0000-000000000004", "2026-06-17T00:00:00Z")
            add("https://blocked.example/article", blocked_text, "00000000-0000-0000-0000-000000000005", "2026-06-17T00:00:00Z")

            parquet_file = fixture_root / "part-00000.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet_file)
            parquet_relative = (
                "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-25/"
                "subset=warc/part-00000-fixture.c000.gz.parquet"
            )
            listing = gzip.compress((parquet_relative + "\n").encode("utf-8"), mtime=0)
            collinfo = json.dumps(
                [
                    {
                        "id": "CC-MAIN-2026-25",
                        "name": "June 2026 Index",
                        "from": "2026-06-05T21:48:11",
                        "to": "2026-06-18T19:32:05",
                    }
                ]
            ).encode("utf-8")
            opt_out = (
                b"Publisher/Requester,Date of notice,List of domains/URLs\n"
                b"LAST UPDATED: 2026 Jul 21,,\n"
                b"Blocked publisher,2026-07-21,blocked.example\n"
            )
            files = {
                "/collinfo.json": collinfo,
                "/optout.csv": opt_out,
                "/crawl-data/CC-MAIN-2026-25/cc-index-table.paths.gz": listing,
                "/" + parquet_relative: parquet_file.read_bytes(),
                "/" + warc_path: b"".join(members),
            }
            with _FixtureServer(files) as server:
                options = FreshWebOptions(
                    collinfo_url=server.root + "collinfo.json",
                    data_root=server.root,
                    opt_out_csv_url=server.root + "optout.csv",
                    max_records_per_partition=20,
                    estimated_tokens_per_document=10,
                    minimum_characters=40,
                    shard_count=2,
                    max_workers=1,
                    coalesce_gap_bytes=8 * 1024,
                    maximum_span_bytes=1024 * 1024,
                )
                source = {
                    "source_id": "metis_freshweb_2026_test",
                    "candidate_tokens": 20,
                    "license": {
                        "status": "per_record_required",
                        "expression": "source-page-terms-plus-Common-Crawl-terms",
                    },
                    "access": {"crawls": ["CC-MAIN-2026-25"]},
                }
                output_root = fixture_root / "lustre"
                receipt = materialize_freshweb(source, root=output_root, options=options)
                self.assertEqual(receipt["status"], "complete")
                self.assertEqual(receipt["url_index_partitions"]["resolved"], 1)
                self.assertEqual(receipt["url_index_partitions"]["completed"], 1)
                self.assertEqual(receipt["metadata_rejections"]["common_crawl_opt_out_domain"], 1)
                self.assertEqual(receipt["exact_selection"]["canonical_url_winners"], 3)
                self.assertEqual(receipt["exact_selection"]["content_digest_winners"], 2)
                self.assertEqual(receipt["warc"]["records_extracted"], 2)
                self.assertEqual(receipt["warc"]["range_requests"], 1)
                self.assertTrue(receipt["candidate_target_met"])
                self.assertTrue(receipt["ready_for_training_build"])
                self.assertTrue(
                    receipt["license_eligibility"]["required_for_candidate_target"]
                )
                self.assertGreaterEqual(
                    receipt["estimated_license_eligible_tokens"],
                    receipt["candidate_token_target"],
                )

                documents: list[dict[str, object]] = []
                for shard in receipt["shards"]:
                    path = Path(shard["path"])
                    with path.open("rb") as handle:
                        with zstandard.ZstdDecompressor().stream_reader(handle) as reader:
                            text = reader.read().decode("utf-8")
                    documents.extend(json.loads(line) for line in text.splitlines() if line)
                self.assertEqual(len(documents), 2)
                urls = {document["metadata"]["canonical_url"] for document in documents}
                self.assertEqual(urls, {"https://docs.example.org/guide", "https://courses.university.edu/lesson"})
                for document in documents:
                    metadata = document["metadata"]
                    self.assertEqual(metadata["extractor_version"], "metis-warc-html-v3")
                    self.assertTrue(metadata["warc_payload_digest_verified"])
                    self.assertEqual(metadata["capture_date"], metadata["fetch_time"][:10])
                    self.assertTrue(metadata["structural_quality"]["passed"])
                    self.assertTrue(metadata["english_evidence"]["decision"])
                    self.assertRegex(metadata["raw_warc_member_sha256"], r"^[0-9a-f]{64}$")
                    self.assertEqual(metadata["opt_out_snapshot_sha256"], receipt["opt_out_registry"]["sha256"])
                    self.assertNotIn("boilerplate menu", document["text"])
                    self.assertNotIn("boilerplate footer", document["text"])

                widening_options = FreshWebOptions(
                    collinfo_url=server.root + "collinfo.json",
                    data_root=server.root,
                    opt_out_csv_url=server.root + "optout.csv",
                    estimated_tokens_per_document=1_000,
                    selection_oversample=1.0,
                    minimum_characters=40,
                    shard_count=2,
                    max_workers=1,
                    coalesce_gap_bytes=8 * 1024,
                    maximum_span_bytes=1024 * 1024,
                    keep_index_files=True,
                )
                widening_source = {
                    "source_id": "metis_freshweb_retryable_test",
                    "candidate_tokens": 150,
                    "license": {
                        "status": "per_record_required",
                        "expression": "source-page-terms-plus-Common-Crawl-terms",
                    },
                    "access": {"crawls": ["CC-MAIN-2026-25"]},
                }
                progress = materialize_freshweb(
                    widening_source,
                    root=output_root,
                    options=widening_options,
                )
                self.assertEqual(progress["status"], "retryable_shortfall")
                self.assertFalse(progress["candidate_target_met"])
                self.assertEqual(progress["selection_round"], 0)
                self.assertEqual(progress["next_selection_round"], 1)
                progress_root = Path(progress["local_path"]).parent
                self.assertFalse((progress_root / "ACQUISITION_RECEIPT.json").exists())
                self.assertTrue((progress_root / "ACQUISITION_PROGRESS.json").exists())

                widened = progress
                for _ in range(4):
                    widened = materialize_freshweb(
                        widening_source,
                        root=output_root,
                        options=widening_options,
                    )
                    if widened["candidate_target_met"]:
                        break
                self.assertTrue(widened["candidate_target_met"])
                self.assertEqual(widened["status"], "complete")
                self.assertGreater(widened["selection_round"], 0)
                self.assertGreaterEqual(
                    widened["estimated_license_eligible_tokens"],
                    widened["candidate_token_target"],
                )
                self.assertTrue((progress_root / "ACQUISITION_RECEIPT.json").is_file())
                self.assertFalse((progress_root / "ACQUISITION_PROGRESS.json").exists())

                warc_requests = server.counts["/" + warc_path]
                damaged_shard = Path(receipt["shards"][0]["path"])
                committed_size = damaged_shard.stat().st_size
                with damaged_shard.open("ab") as handle:
                    handle.write(b"uncommitted-tail")
                (damaged_shard.parent.parent / "ACQUISITION_RECEIPT.json").unlink()
                second = materialize_freshweb(source, root=output_root, options=options)
                self.assertEqual(second["fingerprint"], receipt["fingerprint"])
                self.assertEqual(damaged_shard.stat().st_size, committed_size)

                updated_opt_out = (
                    b"Publisher/Requester,Date of notice,List of domains/URLs\n"
                    b"LAST UPDATED: 2026 Jul 22,,\n"
                    b"Blocked publisher,2026-07-21,blocked.example\n"
                    b"New request,2026-07-22,docs.example.org\n"
                )
                server.server.files["/optout.csv"] = updated_opt_out  # type: ignore[attr-defined]
                refreshed = materialize_freshweb(source, root=output_root, options=options)
                self.assertEqual(refreshed["fingerprint"], receipt["fingerprint"])
                self.assertNotEqual(
                    refreshed["opt_out_registry"]["sha256"],
                    receipt["opt_out_registry"]["sha256"],
                )
                refreshed_documents: list[dict[str, object]] = []
                for shard in refreshed["shards"]:
                    with Path(shard["path"]).open("rb") as handle:
                        with zstandard.ZstdDecompressor().stream_reader(handle) as reader:
                            payload = reader.read().decode("utf-8")
                    refreshed_documents.extend(
                        json.loads(line) for line in payload.splitlines() if line
                    )
                self.assertNotIn(
                    "https://docs.example.org/guide",
                    {
                        document["metadata"]["canonical_url"]
                        for document in refreshed_documents
                    },
                )
                refreshed_warc_requests = server.counts["/" + warc_path]
                self.assertGreaterEqual(refreshed_warc_requests, warc_requests)

                third = materialize_freshweb(source, root=output_root, options=options)
                self.assertEqual(third["fingerprint"], receipt["fingerprint"])
                self.assertEqual(server.counts["/" + warc_path], refreshed_warc_requests)
                with damaged_shard.open("ab") as handle:
                    handle.write(b"post-release-corruption")
                with self.assertRaisesRegex(FreshWebError, "wrong size"):
                    materialize_freshweb(source, root=output_root, options=options)


class FreshWebDownloadDispatchTests(unittest.TestCase):
    def test_all_manifest_routes_have_distinct_bounded_selection_profiles(self) -> None:
        profile = {
            "acquisition": {
                "max_workers": 8,
                "lane_max_workers": {"common_crawl": 2},
            },
            "runtime": {"request_timeout_seconds": 900, "download_retries": 7},
        }
        dominant_category = {
            "general_web": "official_docs",
            "software_docs": "official_docs",
            "fresh_science": "science",
            "official_docs": "official_docs",
        }
        allowed_categories = {
            "general_web": set(FRESHWEB_ROUTE_SETTINGS["general_web"]["allowed_categories"]),
            "software_docs": {"official_docs", "software", "technical"},
            "fresh_science": {"science", "education", "government", "reporting"},
            "official_docs": {"official_docs", "government", "education", "science", "technical"},
        }
        seeds = set()
        for route in FRESHWEB_ROUTE_SETTINGS:
            item = {
                "source_id": f"fresh-{route}",
                "license": {"status": "per_record_required"},
                "access": {
                    "route": route,
                    "warc_root": "https://common-crawl.test/root",
                },
            }
            options = freshweb_options_for_item(item, profile)
            weights = dict(options.category_weights)
            self.assertEqual(max(weights, key=weights.get), dominant_category[route])
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertEqual(set(options.allowed_categories), allowed_categories[route])
            self.assertTrue(
                all(
                    weight == 0
                    for category, weight in weights.items()
                    if category not in allowed_categories[route]
                )
            )
            self.assertEqual(options.data_root, "https://common-crawl.test/root")
            self.assertEqual(options.max_workers, 5)
            self.assertEqual(options.request_timeout_seconds, 900)
            self.assertEqual(options.max_retries, 7)
            self.assertTrue(options.keep_index_files)
            self.assertTrue(options.require_reusable_open_license)
            seeds.add(options.seed)
        self.assertEqual(len(seeds), len(FRESHWEB_ROUTE_SETTINGS))
        with self.assertRaisesRegex(ValueError, "Unsupported Common Crawl route"):
            freshweb_options_for_item(
                {"source_id": "bad", "access": {"route": "unknown"}}, profile
            )

    def test_download_dispatch_returns_handoff_compatible_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "safety_free_tb": 0,
                    "directories": {"state": "state"},
                },
                "runtime": {"hf_home": "cache/huggingface"},
                "acquisition": {
                    "max_workers": 8,
                    "lane_max_workers": {"common_crawl": 2},
                },
            }
            item = {
                "kind": "builder",
                "source_id": "metis-freshdocs-test",
                "driver": "common_crawl_ranges",
                "license": {"status": "per_record_required"},
                "access": {
                    "type": "common_crawl",
                    "route": "official_docs",
                    "warc_root": "https://data.commoncrawl.org/",
                    "crawls": ["CC-MAIN-2026-25"],
                },
                "candidate_tokens": 100,
                "planned_download_bytes": 10,
            }
            state = StateStore(root / "state")
            state.write(
                "sources.lock.json",
                payload={
                    "download_tasks": [
                        {"task_id": "download-000000", "planned_bytes": 10, "items": [item]}
                    ]
                },
            )
            run_root = root / "raw" / item["source_id"] / "freshweb" / "runs" / "fixture"
            documents = run_root / "documents"
            documents.mkdir(parents=True)
            shard = documents / "part-00000.jsonl.zst"
            shard.write_bytes(b"fixture shard")
            shard_record = {
                "path": str(shard),
                "size": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
            receipt = {
                "candidate_target_met": True,
                "candidate_token_target": 100,
                "estimated_materialized_tokens": 120,
                "estimated_license_eligible_tokens": 120,
                "license_eligibility": {
                    "required_for_candidate_target": True,
                    "estimated_tokens": 120,
                    "target_tokens": 100,
                    "target_met": True,
                },
                "ready_for_training_build": True,
                "local_path": str(documents),
                "shards": [shard_record],
                "warc": {"records_extracted": 3},
            }
            receipt_path = run_root / "ACQUISITION_RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch("metis_data.download.materialize_freshweb", return_value=receipt) as materialize:
                result = run_download_task(profile, 0)
            materialize.assert_called_once()
            self.assertEqual(materialize.call_args.args, (item,))
            self.assertEqual(materialize.call_args.kwargs["root"], root)
            self.assertEqual(
                materialize.call_args.kwargs["cache_root"], root / "cache" / "common-crawl"
            )
            self.assertEqual(
                max(dict(materialize.call_args.kwargs["options"].category_weights), key=dict(materialize.call_args.kwargs["options"].category_weights).get),
                "official_docs",
            )
            dataset = result["files"][0]
            self.assertEqual(dataset["kind"], "materialized_dataset")
            self.assertEqual(dataset["receipt"], str(receipt_path))
            self.assertEqual(dataset["shards"], [shard_record])
            self.assertTrue(dataset["ready_for_training_build"])
            self.assertEqual(dataset["estimated_tokens"], 120)
            self.assertEqual(dataset["estimated_license_eligible_tokens"], 120)
            handoff_records = list(_iter_output_records(dataset))
            self.assertEqual(
                [record["kind"] for record in handoff_records],
                ["materialization_receipt", "materialized_shard"],
            )

    def test_download_wrapper_fails_closed_on_candidate_shortfall(self) -> None:
        item = {
            "source_id": "fresh-shortfall",
            "license": {"status": "per_record_required"},
            "access": {"route": "general_web", "crawls": ["CC-MAIN-2026-25"]},
            "candidate_tokens": 100,
        }
        short = {
            "candidate_target_met": False,
            "candidate_token_target": 100,
            "estimated_materialized_tokens": 90,
            "ready_for_training_build": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("metis_data.download.materialize_freshweb", return_value=short):
                with self.assertRaisesRegex(RuntimeError, "did not meet its candidate target"):
                    _materialize_common_crawl(item, profile={}, root=Path(temporary))

    def test_download_wrapper_widens_retryable_shortfall_without_operator_restart(self) -> None:
        item = {
            "source_id": "fresh-widen",
            "license": {"status": "per_record_required"},
            "access": {"route": "general_web", "crawls": ["CC-MAIN-2026-25"]},
            "candidate_tokens": 100,
        }
        short = {
            "candidate_target_met": False,
            "candidate_token_target": 100,
            "estimated_license_eligible_tokens": 90,
            "ready_for_training_build": False,
            "selection_round": 0,
            "next_selection_round": 1,
            "automatic_widening": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "raw" / "fresh-widen" / "freshweb" / "runs" / "test"
            documents = run_root / "documents"
            documents.mkdir(parents=True)
            shard = {"path": str(documents / "part-00000.jsonl.zst"), "size": 0}
            complete = {
                "candidate_target_met": True,
                "candidate_token_target": 100,
                "estimated_materialized_tokens": 120,
                "estimated_license_eligible_tokens": 120,
                "license_eligibility": {"required_for_candidate_target": True},
                "ready_for_training_build": True,
                "local_path": str(documents),
                "shards": [shard],
                "warc": {"records_extracted": 3},
                "selection_round": 1,
            }
            (run_root / "ACQUISITION_RECEIPT.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            profile = {
                "acquisition": {"common_crawl": {"maximum_selection_round": 2}}
            }
            with mock.patch(
                "metis_data.download.materialize_freshweb",
                side_effect=[short, complete],
            ) as materialize:
                result = _materialize_common_crawl(item, profile=profile, root=root)
            self.assertEqual(materialize.call_count, 2)
            self.assertTrue(result["ready_for_training_build"])
            self.assertEqual(result["candidate_token_estimate"], 120)

    def test_download_wrapper_activates_cold_reserve_crawl_after_primary_exhaustion(
        self,
    ) -> None:
        item = {
            "source_id": "fresh-cold-reserve",
            "license": {"status": "per_record_required"},
            "access": {
                "route": "general_web",
                "crawls": ["CC-MAIN-2026-25"],
                "reserve_crawls": ["CC-MAIN-2026-04"],
            },
            "candidate_tokens": 100,
        }
        short = {
            "candidate_target_met": False,
            "candidate_token_target": 100,
            "estimated_license_eligible_tokens": 90,
            "ready_for_training_build": False,
            "selection_round": 0,
            "next_selection_round": 1,
            "automatic_widening": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = (
                root
                / "raw"
                / "fresh-cold-reserve"
                / "freshweb"
                / "runs"
                / "test"
            )
            documents = run_root / "documents"
            documents.mkdir(parents=True)
            shard = {"path": str(documents / "part-00000.jsonl.zst"), "size": 0}
            complete = {
                "candidate_target_met": True,
                "candidate_token_target": 100,
                "estimated_materialized_tokens": 120,
                "estimated_license_eligible_tokens": 120,
                "license_eligibility": {"required_for_candidate_target": True},
                "ready_for_training_build": True,
                "local_path": str(documents),
                "shards": [shard],
                "warc": {"records_extracted": 3},
                "selection_round": 0,
            }
            (run_root / "ACQUISITION_RECEIPT.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            profile = {
                "acquisition": {"common_crawl": {"maximum_selection_round": 0}}
            }
            with mock.patch(
                "metis_data.download.materialize_freshweb",
                side_effect=[short, complete],
            ) as materialize:
                result = _materialize_common_crawl(item, profile=profile, root=root)
            self.assertEqual(materialize.call_count, 2)
            self.assertEqual(
                materialize.call_args_list[0].args[0]["access"]["crawls"],
                ["CC-MAIN-2026-25"],
            )
            self.assertEqual(
                materialize.call_args_list[1].args[0]["access"]["crawls"],
                ["CC-MAIN-2026-25", "CC-MAIN-2026-04"],
            )
            self.assertEqual(result["replacement_reserve_tier"], "cold_reserve")
            self.assertEqual(
                result["reserve_crawls_activated"], ["CC-MAIN-2026-04"]
            )

    def test_download_wrapper_materializes_final_opt_out_reserve(self) -> None:
        item = {
            "source_id": "fresh-reserve",
            "license": {"status": "per_record_required"},
            "access": {"route": "general_web", "crawls": ["CC-MAIN-2026-25"]},
            "candidate_tokens": 100,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "raw" / "fresh-reserve" / "freshweb" / "runs" / "test"
            documents = run_root / "documents"
            documents.mkdir(parents=True)
            shard = {"path": str(documents / "part-00000.jsonl.zst"), "size": 0}
            complete = {
                "candidate_target_met": True,
                "candidate_token_target": 105,
                "estimated_materialized_tokens": 110,
                "estimated_license_eligible_tokens": 110,
                "license_eligibility": {"required_for_candidate_target": True},
                "ready_for_training_build": True,
                "local_path": str(documents),
                "shards": [shard],
                "warc": {"records_extracted": 2},
                "selection_round": 0,
            }
            (run_root / "ACQUISITION_RECEIPT.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            profile = {
                "acquisition": {
                    "common_crawl": {"final_opt_out_reserve_multiplier": 1.05}
                }
            }
            with mock.patch(
                "metis_data.download.materialize_freshweb", return_value=complete
            ) as materialize:
                result = _materialize_common_crawl(item, profile=profile, root=root)
            materialized_item = materialize.call_args.args[0]
            self.assertEqual(materialized_item["candidate_tokens"], 105)
            self.assertEqual(materialized_item["locked_candidate_tokens"], 100)
            self.assertEqual(result["candidate_token_estimate"], 110)


class FreshWebScanTests(unittest.TestCase):
    """The URL-index scan is the multi-day phase; guard its fast paths."""

    crawl = "CC-MAIN-2026-25"

    @staticmethod
    def _options(**overrides: object) -> FreshWebOptions:
        options = FreshWebOptions(
            route="general_web",
            allowed_categories=tuple(CATEGORY_PRIORITY),
            category_weights=DEFAULT_CATEGORY_WEIGHTS,
            max_workers=1,
            retry_base_seconds=0,
            request_timeout_seconds=10,
            **overrides,
        )
        options.validate()
        return options

    @staticmethod
    def _policy() -> OptOutPolicy:
        return OptOutPolicy(
            domains=frozenset({"blocked.example"}),
            url_paths=frozenset({("docs.example.org", "/private")}),
            snapshot_sha256="0" * 64,
            last_updated="2026-07-21",
        )

    def _index_row(self, url: str, **overrides: object) -> dict[str, object]:
        parsed = urlsplit(url)
        row: dict[str, object] = {
            "url": url,
            "url_host_registered_domain": parsed.hostname,
            "url_host_private_domain": parsed.hostname,
            "url_host_registry_suffix": (parsed.hostname or "").rsplit(".", 1)[-1],
            "fetch_time": "2026-06-18T12:00:00Z",
            "fetch_status": 200,
            "content_digest": _payload_digest(url.encode("utf-8")),
            "content_mime_type": "text/html",
            "content_mime_detected": "text/html",
            "content_charset": "UTF-8",
            "content_languages": "eng",
            "content_truncated": None,
            "warc_record_id": "fixture",
            "warc_filename": f"crawl-data/{self.crawl}/segments/1/warc/a.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": 4_096,
            "warc_segment": "1",
        }
        row.update(overrides)
        return row

    _INDEX_SCHEMA = pa.schema(
        [
            ("url", pa.string()),
            ("url_host_registered_domain", pa.string()),
            ("url_host_private_domain", pa.string()),
            ("url_host_registry_suffix", pa.string()),
            ("fetch_time", pa.string()),
            ("fetch_status", pa.int16()),
            ("content_digest", pa.string()),
            ("content_mime_type", pa.string()),
            ("content_mime_detected", pa.string()),
            ("content_charset", pa.string()),
            ("content_languages", pa.string()),
            ("content_truncated", pa.string()),
            ("warc_record_id", pa.string()),
            ("warc_filename", pa.string()),
            ("warc_record_offset", pa.int32()),
            ("warc_record_length", pa.int32()),
            ("warc_segment", pa.string()),
        ]
    )

    def test_vectorized_pre_gate_matches_the_reference_row_gates(self) -> None:
        """The pre-gate must never change the selected set or the audit tally.

        It resolves the leading gates in Arrow, so its agreement with
        ``metadata_candidate`` is what keeps a scanned partition reproducible.
        Values it cannot decide -- non-ASCII, embedded newlines, odd spacing --
        must fall through to the reference gates rather than be guessed at.
        """

        statuses = [200, 0, 404, 301, None, -1]
        truncations = [None, "", "length", "disconnect", "0", " "]
        digests = [
            None, "", "A" * 32, "a" * 32, "sha1:" + "A" * 32, "SHA1:" + "B" * 32,
            "Z" * 32, "1" * 32, "A" * 31, "A" * 33, "A" * 16 + "\n" + "A" * 15,
            "É" * 32, "sha1:", "234567" * 5 + "AB",
        ]
        mimes = [
            None, "", "text/html", "TEXT/HTML", "text/html; charset=utf-8",
            " text/html ", "text/html;", "application/xhtml+xml", "text/plain",
            "application/pdf", "image/png", "text/htm", "TEXT/HTML;CHARSET=X",
            "text/html\n; x", "tëxt/html", ";text/html", "text/html;\ncharset=y",
        ]
        languages = [
            None, "", "eng", "ENG", "eng,fra", "fra,eng", "fra", "engx", "xeng",
            "eng,", ",eng", " eng", "eng ", "eng , fra", "en", "english",
            "eng\n", "ENG,DEU", "zho,eng,spa", "e", "éng", "eng;fra", "deu,fra",
        ]

        rows: list[dict[str, object]] = []
        for status, truncated in itertools.product(statuses, truncations):
            rows.append(
                self._index_row(
                    "https://docs.example.org/guide/topic",
                    fetch_status=status,
                    content_truncated=truncated,
                )
            )
        for digest in digests:
            rows.append(
                self._index_row("https://docs.example.org/guide", content_digest=digest)
            )
        for sent, detected in itertools.product(mimes, mimes):
            rows.append(
                self._index_row(
                    "https://docs.example.org/guide",
                    content_mime_type=sent,
                    content_mime_detected=detected,
                )
            )
        for language in languages:
            rows.append(
                self._index_row(
                    "https://docs.example.org/guide", content_languages=language
                )
            )
        # Gates the pre-filter never claims still have to be attributed by the
        # reference implementation, and only after the earlier gates.
        rows.append(self._index_row("https://blocked.example/article"))
        rows.append(self._index_row("https://docs.example.org/private"))
        rows.append(self._index_row("https://docs.example.org/login"))
        rows.append(self._index_row("https://docs.example.org/a", warc_record_length=8))
        rows.append(
            self._index_row(
                "https://blocked.example/x", content_languages="fra", fetch_status=404
            )
        )

        batch = pa.RecordBatch.from_pylist(rows, schema=self._INDEX_SCHEMA)
        policy = self._policy()
        for require_english in (True, False):
            with self.subTest(require_english=require_english):
                options = self._options(require_english=require_english)
                reference, reference_stats = select_partition_candidates(
                    batch.to_pylist(),
                    crawl=self.crawl,
                    source_id="scan",
                    policy=policy,
                    options=options,
                    capacity=1_000,
                )
                tally = PrefilterTally()
                tally.scanned = batch.num_rows
                kept = _prefilter_index_batch(batch, options=options, tally=tally)
                gated, gated_stats = select_partition_candidates(
                    kept.to_pylist(),
                    crawl=self.crawl,
                    source_id="scan",
                    policy=policy,
                    options=options,
                    capacity=1_000,
                    tally=tally,
                )
                self.assertEqual(reference, gated)
                self.assertEqual(reference_stats, gated_stats)
                self.assertEqual(gated_stats["scanned"], batch.num_rows)
                # The pre-gate has to actually carry its weight, not just agree.
                self.assertLess(kept.num_rows, batch.num_rows)

    def test_scan_projection_covers_every_column_the_gates_read(self) -> None:
        """A narrowed Parquet projection must not silently starve a gate."""

        source = "".join(
            inspect.getsource(function)
            for function in (metadata_candidate, freshweb._fresh_category)
        )
        read = set(re.findall(r"""row(?:\.get\(|\[)["']([a-z_]+)["']""", source))
        self.assertTrue(read)
        self.assertEqual(
            read - set(freshweb.SCAN_INDEX_COLUMNS),
            set(),
            "SCAN_INDEX_COLUMNS omits a column the reference gates read",
        )
        self.assertEqual(set(freshweb.SCAN_INDEX_COLUMNS) - set(INDEX_COLUMNS), set())
        self.assertEqual(REQUIRED_INDEX_COLUMNS - set(freshweb.SCAN_INDEX_COLUMNS), set())

    def test_run_identity_ignores_operational_tuning(self) -> None:
        """Retuning throughput must resume an acquisition, never orphan it.

        A five-crawl URL-index scan runs for days.  If a worker count or a
        scratch path reached the run fingerprint, raising it would strand every
        partition already scanned under the old identity.
        """

        operational = {
            field.name
            for field in dataclasses.fields(FreshWebOptions)
            if field.name not in freshweb.FINGERPRINT_OPTION_FIELDS
        }
        self.assertEqual(
            operational,
            {"index_scan_workers", "state_scratch_root", "state_checkpoint_seconds"},
        )
        selective = {
            field.name for field in dataclasses.fields(FreshWebOptions)
        } - operational
        self.assertEqual(selective, set(freshweb.FINGERPRINT_OPTION_FIELDS))

        def identity(options: FreshWebOptions) -> str:
            values = dataclasses.asdict(options)
            return json.dumps(
                {name: values[name] for name in freshweb.FINGERPRINT_OPTION_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            )

        self.assertEqual(
            identity(self._options()),
            identity(
                self._options(
                    index_scan_workers=8,
                    state_scratch_root="/local/scratch",
                    state_checkpoint_seconds=30.0,
                )
            ),
        )
        # A change that can move the selected set still must move identity.
        self.assertNotEqual(
            identity(self._options()), identity(self._options(seed="other"))
        )

    def _serve_partitions(self, root: Path, partitions: list[list[dict[str, object]]]):
        files: dict[str, bytes] = {}
        relatives: list[str] = []
        for index, rows in enumerate(partitions):
            path = root / f"part-{index:05d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=self._INDEX_SCHEMA), path
            )
            relative = (
                f"cc-index/table/cc-main/warc/crawl={self.crawl}/subset=warc/"
                f"part-{index:05d}-fixture.c000.gz.parquet"
            )
            files["/" + relative] = path.read_bytes()
            relatives.append(relative)
        return files, relatives

    def test_parallel_index_scan_commits_in_partition_order(self) -> None:
        """A parallel scan must select exactly what a serial scan selects.

        ``URL_UPSERT`` keeps the incumbent when two captures tie on every
        ranking key, so the winner depends on commit order.  Results are
        therefore committed in partition order even though they are produced
        out of order -- here the first partition is by far the slowest, so a
        completion-ordered commit would hand the tie to a later partition.
        """

        shared = "https://docs.example.org/shared-guide"
        shared_digest = _payload_digest(b"shared")
        partitions: list[list[dict[str, object]]] = []
        for index in range(4):
            warc = f"crawl-data/{self.crawl}/segments/1/warc/part{index}.warc.gz"
            rows = [
                # The same capture, tied on every ranking key, in every
                # partition: only the WARC coordinate differs.
                self._index_row(
                    shared,
                    content_digest=shared_digest,
                    warc_filename=warc,
                    warc_record_offset=4_096 * index,
                )
            ]
            if index == 0:
                rows.extend(
                    self._index_row(
                        f"https://docs.example.org/filler/{position}",
                        warc_filename=warc,
                        warc_record_offset=8_192 + position,
                    )
                    for position in range(6_000)
                )
            partitions.append(rows)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, relatives = self._serve_partitions(root, partitions)
            with _FixtureServer(files) as server:
                results = {}
                for workers in (1, 3):
                    options = self._options(
                        index_scan_workers=workers, keep_index_files=True
                    )
                    connection = _connect_state(root / f"state-{workers}.sqlite3")
                    try:
                        freshweb._scan_index_partitions(
                            connection,
                            [
                                freshweb._IndexScanJob(
                                    order=order,
                                    relative_path=relative,
                                    crawl=self.crawl,
                                    url=server.root + relative,
                                    local_path=str(
                                        root / f"cache-{workers}" / f"{order}.parquet"
                                    ),
                                    selected_path=str(
                                        root / f"spill-{workers}" / f"{order}.jsonl.zst"
                                    ),
                                    capacity=10_000,
                                    source_id="scan",
                                    policy=self._policy(),
                                    options=options,
                                )
                                for order, relative in enumerate(relatives)
                            ],
                            options=options,
                        )
                        results[workers] = [
                            tuple(row)
                            for row in connection.execute(
                                "SELECT canonical_url, content_digest, warc_filename,"
                                " warc_record_offset FROM url_winners"
                                " ORDER BY canonical_url"
                            )
                        ]
                        self.assertEqual(
                            connection.execute(
                                "SELECT COUNT(*) FROM partitions"
                            ).fetchone()[0],
                            len(relatives),
                        )
                    finally:
                        connection.close()

                self.assertEqual(results[1], results[3])
                winner = next(row for row in results[3] if row[0] == shared)
                self.assertTrue(
                    winner[2].endswith("part0.warc.gz"),
                    f"tie was awarded out of partition order: {winner}",
                )

    def test_index_scan_resumes_at_the_first_unfinished_partition(self) -> None:
        partitions = [
            [self._index_row(f"https://docs.example.org/p{index}/{position}")
             for position in range(4)]
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, relatives = self._serve_partitions(root, partitions)
            with _FixtureServer(files) as server:
                options = self._options(keep_index_files=True)
                jobs = [
                    freshweb._IndexScanJob(
                        order=order,
                        relative_path=relative,
                        crawl=self.crawl,
                        url=server.root + relative,
                        local_path=str(root / "cache" / f"{order}.parquet"),
                        selected_path=str(root / "spill" / f"{order}.jsonl.zst"),
                        capacity=100,
                        source_id="scan",
                        policy=self._policy(),
                        options=options,
                    )
                    for order, relative in enumerate(relatives)
                ]
                connection = _connect_state(root / "state.sqlite3")
                try:
                    freshweb._scan_index_partitions(
                        connection, jobs[:1], options=options
                    )
                    served = sum(server.counts["/" + r] for r in relatives)
                    self.assertEqual(served, 1)

                    # A resume rebuilds the job list from the ledger and must
                    # not re-fetch or re-scan what is already committed.
                    scanned = {
                        str(row["relative_path"])
                        for row in connection.execute(
                            "SELECT relative_path FROM partitions"
                            " WHERE selection_capacity >= ?",
                            (100,),
                        )
                    }
                    self.assertEqual(scanned, {relatives[0]})
                    outstanding = [
                        job for job in jobs if job.relative_path not in scanned
                    ]
                    self.assertEqual(len(outstanding), 2)
                    freshweb._scan_index_partitions(
                        connection, outstanding, options=options
                    )
                    self.assertEqual(
                        sum(server.counts["/" + r] for r in relatives), 3
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM partitions"
                        ).fetchone()[0],
                        3,
                    )
                finally:
                    connection.close()


class FreshWebStateDatabaseTests(unittest.TestCase):
    def test_scratch_ledger_is_published_to_the_durable_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            durable = root / "lustre" / "state.sqlite3"
            scratch = root / "scratch"

            state = freshweb._StateDatabase(
                durable, scratch_root=str(scratch), checkpoint_seconds=0.0, identity="run"
            )
            self.assertNotEqual(state.working_path, durable)
            self.assertFalse(durable.exists())
            with state.connection:
                freshweb._set_metadata(state.connection, "fingerprint", "abc")
            state.checkpoint(force=True)
            self.assertTrue(durable.is_file())
            state.close()

            # A run that resumes on another node seeds from the durable copy.
            elsewhere = freshweb._StateDatabase(
                durable,
                scratch_root=str(root / "other-scratch"),
                checkpoint_seconds=0.0,
                identity="run",
            )
            try:
                self.assertEqual(
                    freshweb._metadata_value(elsewhere.connection, "fingerprint"), "abc"
                )
            finally:
                elsewhere.close()

            # A newer working copy is never rewound by an older durable one.
            resumed = freshweb._StateDatabase(
                durable, scratch_root=str(scratch), checkpoint_seconds=3_600.0, identity="run"
            )
            try:
                with resumed.connection:
                    freshweb._set_metadata(resumed.connection, "fingerprint", "newer")
                resumed.checkpoint(force=True)
            finally:
                resumed.close()
            again = freshweb._StateDatabase(
                durable, scratch_root=str(scratch), checkpoint_seconds=3_600.0, identity="run"
            )
            try:
                self.assertEqual(
                    freshweb._metadata_value(again.connection, "fingerprint"), "newer"
                )
            finally:
                again.close()

    def test_ledger_without_scratch_stays_on_the_durable_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            durable = Path(temporary) / "state.sqlite3"
            state = freshweb._StateDatabase(durable)
            try:
                self.assertEqual(state.working_path, durable)
                self.assertEqual(
                    state.connection.execute("PRAGMA journal_mode").fetchone()[0],
                    "delete",
                )
                state.checkpoint(force=True)
            finally:
                state.close()
            self.assertTrue(durable.is_file())


if __name__ == "__main__":
    unittest.main()
