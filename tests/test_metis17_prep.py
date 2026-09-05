from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import zstandard

from metis_data.datatrove_blocks import save_contamination_index
from metis_data.decontaminate import ContaminationIndex
from metis_data17 import prep, prep_policy
from metis_data17.common import ObjectSpec, RawReceipt, read_receipt, sha256_file, write_receipt
from metis_data17.prep import (
    PreparationError, apply_eligibility, canonical_schema, normalize_object, prepare_object,
    prepare_runtime, prepare_chunk, reblock_object,
)
from tests.contamination_fixtures import write_contamination_inputs


class Metis17PreparationTests(unittest.TestCase):
    HOLDOUT = " ".join(f"benchmarkword{i}" for i in range(45))
    SAFE = "A source document has its own useful explanations and careful measurements 105 220 3340 cm-1."

    def setUp(self) -> None:
        self.root = (Path.cwd() / ".metis17-prep-tests" / uuid.uuid4().hex).resolve()
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.quality = self.root / "quality.yaml"
        self.profiles = {
            "defaults": {"minimum_characters": 1, "reject_secrets": True, "reject_personal_data": True},
            "profiles": {"fixture": {}},
        }
        self._write_profiles()
        self.index_path, self.registry_path = self._index()
        self.config = {
            "root": self.root,
            "quality_profiles_path": self.quality,
            "benchmark_registry": self.registry_path,
            "decontamination_index": self.index_path,
            "opt_out_snapshot": None,
            "output_chunk_bytes": 4096,
            "batch_size": 3,
        }
        self.counter = 0

    def _write_profiles(self) -> None:
        self.quality.write_text(yaml.safe_dump(self.profiles), encoding="utf-8")

    def _index(self, *, directory: str = "policy", holdouts: list[str] | None = None,
               minimum_matching_ngrams: int = 2, contiguous_run_minimum: int = 8,
               minimum_code_matching_ngrams: int = 2) -> tuple[Path, Path]:
        texts = holdouts if holdouts is not None else [self.HOLDOUT]
        index = ContaminationIndex.build(
            texts, ngram_size=13, minimum_matching_ngrams=minimum_matching_ngrams,
            minimum_short_matching_ngrams=0, minimum_code_skeleton_matching_ngrams=0,
            minimum_code_matching_ngrams=minimum_code_matching_ngrams,
            contiguous_run_minimum=contiguous_run_minimum,
        )
        directory_path = self.root / directory
        registry = write_contamination_inputs(directory_path, index, texts)
        path = directory_path / "index.json"
        save_contamination_index(index, path, benchmark_registry_path=registry)
        return path, registry

    def _object(self, rows: list | None = None, *, wire_format: str = "raw_jsonl",
                adapter: str = "text", policy: dict | None = None, payload: bytes | None = None,
                schema: pa.Schema | None = None, source_id: str = "fixture_source") -> tuple[ObjectSpec, RawReceipt, Path]:
        self.counter += 1
        relative = f"raw/object-{self.counter}"
        path = self.root / relative
        path.parent.mkdir(exist_ok=True)
        if payload is None:
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (rows or [])).encode()
            if wire_format == "parquet":
                pq.write_table(pa.Table.from_pylist(rows or [], schema=schema), path, row_group_size=2)
            elif wire_format == "jsonl_gzip":
                path.write_bytes(gzip.compress(data))
            elif wire_format in {"jsonl_zstd", "json_zstd"}:
                path.write_bytes(zstandard.ZstdCompressor(write_checksum=True).compress(data))
            else:
                path.write_bytes(data)
        else:
            path.write_bytes(payload)
        source_policy = {
            "category": "science", "language": "any", "license_mode": "compilation",
            "collection_license": "CC-BY-4.0", "common_crawl_derived": False,
            "generated": False, "quality_profile": "fixture", **(policy or {}),
        }
        spec = ObjectSpec.create(
            source_id=source_id, url=f"https://example.org/objects/{self.counter}",
            revision="pinned-revision", relative_key=path.name, wire_format=wire_format,
            adapter=adapter, priority=97, expected_bytes=path.stat().st_size,
            expected_sha256=sha256_file(path), policy=source_policy,
        )
        raw = RawReceipt(spec.object_id, spec.source_id, relative, path.stat().st_size,
                         sha256_file(path), "fixture-host", "2026-09-04T00:00:00Z")
        return spec, raw, self.root / "prepared" / spec.object_id

    def _rows(self, receipt: dict, *, normalized: bool = False) -> list[dict]:
        key = "normalized_chunks" if normalized else "chunks"
        return [record for chunk in receipt[key]
                for record in pq.read_table(self.root / chunk["path"]).to_pylist()]

    def _opt_out(self) -> Path:
        path = self.root / "opt-out.csv"
        path.write_text(
            "Publisher/Requester,Date of notice,List of domains/URLs\n"
            "Synthetic publisher,2026-09-01,blocked.example\n", encoding="utf-8",
        )
        return path

    def _warc(self, records: list[tuple[str, str, bytes, dict | None]]) -> bytes:
        from warcio.statusandheaders import StatusAndHeaders
        from warcio.warcwriter import WARCWriter
        stream = io.BytesIO()
        writer = WARCWriter(stream, gzip=True)
        for kind, url, body, headers in records:
            http_headers = None
            if headers is not None:
                http_headers = StatusAndHeaders("200 OK", list(headers.items()), protocol="HTTP/1.1")
            record = writer.create_warc_record(url, kind, payload=io.BytesIO(body),
                                               http_headers=http_headers)
            writer.write_record(record)
        return stream.getvalue()

    def test_canonical_schema_exact_hashes_math_digits_and_whitespace(self) -> None:
        text = "\ufeff  Théorème 123456789012345:  α₁ ≠ α₂\r\n\t∫_0^1 x² dx = 1/3  \r\n"
        spec, raw, output = self._object([{"latex": text, "lang": "fra_Latn"}],
                                          adapter="latex", policy={"category": "math"})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["status"], "ELIGIBLE")
        record, = self._rows(result)
        expected = text[1:].replace("\r\n", "\n")
        self.assertEqual(record["text"], expected)
        self.assertEqual(record["content_hash"], hashlib.sha256(expected.encode()).hexdigest())
        self.assertEqual(record["dedup_hash"], record["content_hash"])
        self.assertEqual(record["character_count"], len(expected))
        self.assertEqual(record["priority"], 97)
        self.assertEqual(record["quality_score"], -1.0)
        self.assertEqual(record["language"], "fra_Latn")
        self.assertEqual(pq.read_schema(self.root / result["chunks"][0]["path"]), canonical_schema())
        metadata = json.loads(record["metadata_json"])
        self.assertEqual(metadata["quality_score_status"], "unknown")
        self.assertNotIn("language_probability", metadata)
        self.assertTrue(metadata["normalization"]["newlines_changed"])
        self.assertTrue(metadata["normalization"]["leading_bom_removed"])

    def test_all_inline_formats_parse_each_row_once_across_rowgroups_and_chunks(self) -> None:
        for wire in ("parquet", "raw_jsonl", "jsonl_gzip", "jsonl_zstd", "json_zstd"):
            with self.subTest(wire=wire):
                source = [{"text": self.SAFE + f" ordinal-{i}", "language": "en"} for i in range(23)]
                spec, raw, output = self._object(source, wire_format=wire)
                with patch.object(prep, "iter_source_rows", wraps=prep.iter_source_rows) as reader:
                    result = prepare_object(spec, raw, output, self.config)
                reader.assert_called_once()
                records = self._rows(result)
                self.assertEqual([record["text"] for record in records], [row["text"] for row in source])
                self.assertEqual(len({record["doc_id"] for record in records}), 23)
                self.assertEqual(sum(chunk["records"] for chunk in result["chunks"]), 23)
                self.assertGreater(len(result["chunks"]), 1)
                for chunk in result["chunks"]:
                    parquet = pq.ParquetFile(self.root / chunk["path"])
                    self.assertTrue(all(parquet.metadata.row_group(i).num_rows <= 3
                                        for i in range(parquet.num_row_groups)))
                self.assertEqual(result["input_rows"], 23)
                self.assertEqual(result["input_documents"], 23)

    def test_rows_cross_multiple_rowgroups_in_one_output(self) -> None:
        spec, raw, output = self._object([{"text": f"independent document {i}"} for i in range(11)])
        result = prepare_object(spec, raw, output, {**self.config, "output_chunk_bytes": 1_000_000})
        self.assertEqual(len(result["chunks"]), 1)
        parquet = pq.ParquetFile(self.root / result["chunks"][0]["path"])
        self.assertEqual([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)],
                         [3, 3, 3, 2])

    def test_reblock_publishes_ready_chunks_and_filters_before_raw_scan_finishes(self) -> None:
        source = [{"text": f"{self.SAFE} Ordinal {index}."} for index in range(9)]
        spec, raw, output = self._object(source, wire_format="jsonl_gzip")
        config = {**self.config, "output_chunk_bytes": 1}
        original = prep.iter_source_rows
        early_results = []

        def observing_reader(*args, **kwargs):
            for index, item in enumerate(original(*args, **kwargs)):
                yield item
                if index == 0:
                    ready_path, = output.glob("normalized/*/part-*.READY.json")
                    ready = read_receipt(ready_path)
                    self.assertEqual(ready["raw"]["sha256"], raw.sha256)
                    self.assertEqual(ready["record_range"], [0, 1])
                    self.assertEqual(ready["row_range"], [1, 1])
                    self.assertFalse((self.root / ready["object_manifest"]).exists())
                    self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
                    with patch.object(prep, "_verify_raw", side_effect=AssertionError("chunk read raw")):
                        early = prepare_chunk(ready_path, output, config)
                    self.assertEqual(early["status"], "ELIGIBLE_PENDING_OBJECT_COMPLETION")
                    self.assertFalse(early["training_ready"])
                    self.assertEqual(early["chunks"], [])
                    self.assertEqual(early["screened_documents"], 1)
                    self.assertEqual(len(early["screened_chunks"]), 1)
                    self.assertFalse((self.root / early["screening_receipt"]).with_name("ELIGIBLE.json").exists())
                    self.assertNotIn(source[0]["text"], json.dumps(early))
                    early_results.append((ready_path, early))

        with patch.object(prep, "iter_source_rows", side_effect=observing_reader) as reader:
            manifest = reblock_object(spec, raw, output, config)
        reader.assert_called_once()
        self.assertTrue(manifest["reblock_complete"])
        self.assertEqual(read_receipt(output / "REBLOCK_COMPLETE.json"), manifest)
        self.assertEqual([record["text"] for record in self._rows(manifest)], [row["text"] for row in source])
        self.assertEqual([(chunk["document_start"], chunk["document_stop"]) for chunk in manifest["chunks"]],
                         [(index, index + 1) for index in range(9)])
        self.assertEqual(len(manifest["chunk_receipts"]), 9)
        ready_path, early = early_results[0]
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("chunk read raw")), \
             patch.object(prep, "decide_eligibility", side_effect=AssertionError("chunk filtered twice")):
            promoted = prepare_chunk(ready_path, output, config)
        self.assertEqual(promoted["status"], "ELIGIBLE")
        self.assertTrue(promoted["training_ready"])
        self.assertEqual(promoted["chunks"], early["screened_chunks"])
        self.assertEqual(promoted["screening_receipt"], early["screening_receipt"])
        self.assertTrue(promoted["receipt_path"].endswith("/ELIGIBLE.json"))
        self.assertEqual(read_receipt(self.root / promoted["receipt_path"]), promoted)

    def test_reblocking_and_object_normalization_have_identical_extraction(self) -> None:
        spec, raw, output = self._object([
            {"text": "  Théorème α = 123456789.\r\n\t∫ x² dx = x³/3  \n"},
            {"text": "", "content": self.SAFE},
            {"blob_id": "fixture-pointer"},
        ], wire_format="json_zstd", policy={"category": "math"})
        normalized = normalize_object(spec, raw, output / "atomic", self.config)
        reblocked = reblock_object(spec, raw, output / "streaming", self.config)
        self.assertEqual(self._rows(normalized), self._rows(reblocked))
        for key in ("input_rows", "input_documents", "normalized_documents", "rejected", "quarantined"):
            self.assertEqual(normalized[key], reblocked[key])

    def test_chunk_jobs_cover_every_reblocked_record_exactly_once(self) -> None:
        source = [{"text": f"{self.SAFE} Distinct record {index}."} for index in range(21)]
        source[7]["text"] = self.HOLDOUT
        spec, raw, output = self._object(source, wire_format="jsonl_gzip")
        manifest = reblock_object(spec, raw, output, {**self.config, "output_chunk_bytes": 3000})
        results = []
        for chunk in manifest["chunks"]:
            result = prepare_chunk(self.root / chunk["path"], self.root / "chunk-jobs", self.config)
            self.assertEqual(result["status"], "ELIGIBLE")
            self.assertEqual(result["input_scope"], "base_chunk")
            results.append(result)
        records = [record for result in results for record in self._rows(result)]
        self.assertEqual(len({record["doc_id"] for record in records}), 20)
        self.assertEqual([record["text"] for record in records], [row["text"] for row in source if row["text"] != self.HOLDOUT])
        self.assertEqual(sum(result["input_documents"] for result in results), 21)
        self.assertEqual(sum(result["rejected"].get("benchmark_exact", 0) for result in results), 1)
        self.assertEqual(len({result["receipt_path"] for result in results}), len(results))

    def test_chunk_ranges_preserve_stack_components_across_one_source_row(self) -> None:
        spec, raw, output = self._object([{"files": [
            {"path": f"file-{index}.py", "content": f"def F{index}():\n    return {index}\n"}
            for index in range(6)
        ]}], adapter="stack", policy={"category": "code"})
        config = {**self.config, "output_chunk_bytes": 1}
        manifest = reblock_object(spec, raw, output, config)
        self.assertEqual(manifest["input_rows"], 1)
        self.assertEqual(manifest["input_documents"], 6)
        self.assertEqual([chunk["row_start"] for chunk in manifest["chunks"]], [1] * 6)
        self.assertEqual([chunk["first_component"] for chunk in manifest["chunks"]],
                         [f"files/{index}" for index in range(6)])
        self.assertEqual([chunk["document_start"] for chunk in manifest["chunks"]], list(range(6)))
        results = [prepare_chunk(self.root / chunk["path"], output, config) for chunk in manifest["chunks"]]
        self.assertEqual(sum(result["eligible_documents"] for result in results), 6)

    def test_reblock_failure_retains_immutable_chunks_but_never_seals_incomplete_object(self) -> None:
        payload = (json.dumps({"text": self.SAFE}) + '\n{"text": "INCOMPLETE-SOURCE-RECORD').encode()
        spec, raw, output = self._object(payload=payload)
        config = {**self.config, "output_chunk_bytes": 1}
        with self.assertRaisesRegex(PreparationError, "row=2"):
            reblock_object(spec, raw, output, config)
        ready_path, = output.glob("normalized/*/part-*.READY.json")
        ready = read_receipt(ready_path)
        self.assertTrue((self.root / ready["chunk"]["path"]).is_file())
        self.assertFalse((self.root / ready["object_manifest"]).exists())
        self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
        status = read_receipt(ready_path.parent / "REBLOCK_STATUS.json")
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["reason"], "invalid_json_record")
        result = prepare_chunk(ready_path, self.root / "chunk-jobs", config)
        self.assertEqual(result["status"], "ELIGIBLE_PENDING_OBJECT_COMPLETION")
        self.assertEqual(result["chunks"], [])
        self.assertFalse(result["eligible"])
        self.assertFalse(result["training_ready"])
        self.assertTrue((self.root / raw.relative_path).is_file())

    def test_reblock_retry_replays_once_without_rewriting_or_duplicating_ready_chunks(self) -> None:
        spec, raw, output = self._object([{"text": f"{self.SAFE} Record {index}."} for index in range(8)],
                                          wire_format="jsonl_gzip")
        config = {**self.config, "output_chunk_bytes": 1}
        original = prep.iter_source_rows

        def interrupted(*args, **kwargs):
            for index, item in enumerate(original(*args, **kwargs)):
                if index == 3:
                    raise PreparationError(spec, "injected_transient_interruption", item.index)
                yield item

        with patch.object(prep, "iter_source_rows", side_effect=interrupted), self.assertRaises(PreparationError):
            reblock_object(spec, raw, output, config)
        ready_paths = list(output.glob("normalized/*/part-*.READY.json"))
        self.assertEqual(len(ready_paths), 3)
        before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in ready_paths}
        changed_host = replace(raw, download_host="another-acquisition-host", completed_at="2026-09-05T00:00:00Z")
        with patch.object(prep, "iter_source_rows", wraps=original) as reader:
            manifest = reblock_object(spec, changed_host, output, config)
        reader.assert_called_once()
        self.assertEqual(manifest["normalized_documents"], 8)
        self.assertEqual(len(manifest["chunks"]), 8)
        self.assertEqual(len({record["doc_id"] for record in self._rows(manifest)}), 8)
        self.assertEqual(before, {path: (sha256_file(path), path.stat().st_mtime_ns) for path in ready_paths})
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("completed raw decoded twice")):
            self.assertEqual(reblock_object(spec, changed_host, output, config), manifest)

    def test_prepare_chunk_never_reads_raw_or_other_base_chunks(self) -> None:
        spec, raw, output = self._object([{"text": f"{self.SAFE} Piece {index}."} for index in range(5)])
        manifest = reblock_object(spec, raw, output, {**self.config, "output_chunk_bytes": 1})
        target = self.root / manifest["chunks"][0]["path"]
        forbidden = {self.root / raw.relative_path, *(self.root / chunk["path"] for chunk in manifest["chunks"][1:])}
        original_hash = prep.sha256_file

        def bounded_hash(path):
            self.assertNotIn(Path(path), forbidden)
            return original_hash(path)

        with patch.object(prep, "sha256_file", side_effect=bounded_hash), \
             patch.object(prep, "_verify_raw", side_effect=AssertionError("raw verification in chunk worker")), \
             patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoder in chunk worker")):
            result = prepare_chunk(target, self.root / "chunk-jobs", self.config)
        self.assertEqual(result["eligible_documents"], 1)

    def test_chunk_pending_policy_replays_saved_parquet_without_raw_or_reblocking(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}, {"text": self.HOLDOUT}])
        manifest = reblock_object(spec, raw, output, {**self.config, "output_chunk_bytes": 1_000_000})
        path = self.root / manifest["chunks"][0]["path"]
        pending = prepare_chunk(path, self.root / "chunk-jobs", {**self.config, "decontamination_index": None})
        self.assertEqual(pending["status"], "NORMALIZED_PENDING_DECONTAMINATION")
        self.assertEqual(pending["chunks"], [])
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("raw read")), \
             patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded")):
            result = prepare_chunk(path, self.root / "chunk-jobs", self.config)
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["rejected"], {"benchmark_exact": 1})

    def test_ready_receipt_binds_raw_hash_and_row_range_before_admission(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        manifest = reblock_object(spec, raw, output, self.config)
        path = self.root / manifest["chunk_receipts"][0]
        original = read_receipt(path)
        changed = json.loads(json.dumps(original))
        changed["raw"]["sha256"] = "0" * 64
        write_receipt(path, changed)
        with self.assertRaisesRegex(PreparationError, "base_chunk_raw_identity_mismatch"):
            prepare_chunk(path, self.root / "chunk-jobs", self.config)
        changed = json.loads(json.dumps(original))
        changed["chunk"]["row_start"] = 2
        changed["chunk"]["row_end"] = 2
        changed["row_range"] = [2, 2]
        write_receipt(path, changed)
        with self.assertRaisesRegex(PreparationError, "canonical_chunk_row_range_mismatch"):
            prepare_chunk(path, self.root / "chunk-jobs", self.config)

    def test_chunk_source_priority_override_does_not_reblock_or_redecode(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        manifest = reblock_object(spec, raw, output, self.config)
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("raw read")):
            result = prepare_chunk(self.root / manifest["chunks"][0]["path"], self.root / "chunk-jobs",
                                   {**self.config, "object_spec": replace(spec, priority=123)})
        self.assertEqual(self._rows(result)[0]["priority"], 123)

    def test_chunk_eligible_sidecar_is_immutable_when_current_alias_changes_policy(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        manifest = reblock_object(spec, raw, output, self.config)
        base = self.root / manifest["chunks"][0]["path"]
        first = prepare_chunk(base, self.root / "chunk-jobs", self.config)
        sealed = self.root / first["receipt_path"]
        alias = self.root / "chunk-jobs" / "chunks" / first["chunk_id"] / "PREP_COMPLETE.json"
        self.assertEqual(sealed.name, "ELIGIBLE.json")
        self.assertEqual(sealed.parent, (self.root / first["chunks"][0]["path"]).parent)
        self.assertEqual(read_receipt(alias), read_receipt(sealed))
        original_hash = sha256_file(sealed)
        original_stamp = sealed.stat().st_mtime_ns
        self.profiles["profiles"]["fixture"] = {"minimum_characters": 100_000}
        self._write_profiles()
        updated = prepare_chunk(base, self.root / "chunk-jobs", self.config)
        self.assertEqual(updated["eligible_documents"], 0)
        self.assertNotEqual(first["receipt_path"], updated["receipt_path"])
        self.assertEqual(read_receipt(alias), updated)
        self.assertEqual(read_receipt(sealed), first)
        self.assertEqual(sha256_file(sealed), original_hash)
        self.assertEqual(sealed.stat().st_mtime_ns, original_stamp)

    def test_reblocking_valid_empty_input_publishes_no_fake_chunk(self) -> None:
        spec, raw, output = self._object([], wire_format="jsonl_gzip")
        manifest = reblock_object(spec, raw, output, self.config)
        self.assertTrue(manifest["reblock_complete"])
        self.assertTrue(manifest["empty"])
        self.assertEqual(manifest["empty_reason"], "empty_input")
        self.assertEqual(manifest["chunk_receipts"], [])
        self.assertEqual(list(output.glob("normalized/*/part-*.READY.json")), [])

    def test_chunk_source_row_count_does_not_fabricate_rows_from_range_gaps(self) -> None:
        payload = (
            json.dumps({"text": self.SAFE}) + "\n\n" + json.dumps({"blob_id": "pointer"})
            + "\n" + json.dumps({"text": "Another independent source document."}) + "\n"
        ).encode()
        spec, raw, output = self._object(payload=payload)
        manifest = reblock_object(spec, raw, output, {**self.config, "output_chunk_bytes": 1_000_000})
        chunk, = manifest["chunks"]
        self.assertEqual((chunk["row_start"], chunk["row_end"]), (1, 4))
        self.assertEqual(chunk["source_rows"], 2)
        self.assertEqual(manifest["input_rows"], 3)
        self.assertEqual(manifest["rejected"], {"pointer_only_code": 1})
        result = prepare_chunk(self.root / chunk["path"], self.root / "chunk-jobs", self.config)
        self.assertEqual(result["input_rows"], 2)
        self.assertEqual(result["input_documents"], 2)

    def test_reblocking_does_not_publish_any_chunk_before_raw_verification(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("unverified raw parsed")), \
             self.assertRaises(PreparationError):
            reblock_object(spec, replace(raw, sha256="0" * 64), output, self.config)
        self.assertEqual(list(output.rglob("*.READY.json")), [])

    def test_array_and_json_encoded_messages_tools_preserve_order_without_execution(self) -> None:
        messages = [
            {"role": "user", "content": "Compute 1002003004 without changing any digits."},
            {"role": "assistant", "tool_calls": [{"function": {"name": "never_execute", "arguments": "{}"}}]},
            {"role": "tool", "content": "def F(x):\n    return x + 1\n"},
        ]
        tools = [{"type": "function", "function": {"name": "never_execute", "description": "fixture"}}]
        spec, raw, output = self._object([
            {"messages": messages, "tools": tools},
            {"messages": json.dumps(messages), "tools": json.dumps(tools)},
        ], adapter="messages")
        with patch("socket.create_connection", side_effect=AssertionError("network is forbidden")):
            result = prepare_object(spec, raw, output, self.config)
        records = self._rows(result)
        self.assertEqual(records[0]["text"], records[1]["text"])
        self.assertEqual(json.loads(records[0]["text"]), {"messages": messages, "tools": tools})
        self.assertNotEqual(records[0]["doc_id"], records[1]["doc_id"])

    def test_top_level_json_message_array_is_a_document(self) -> None:
        messages = [{"role": "user", "content": "Une question indépendante."}]
        spec, raw, output = self._object([messages])
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(json.loads(self._rows(result)[0]["text"])["messages"], messages)

    def test_nested_stack_files_keep_order_provenance_and_per_record_licenses(self) -> None:
        spec, raw, output = self._object([{
            "repo_name": "fixture/repo",
            "files": [
                {"path": "Z.py", "content": "def Z():\n    return 42\n", "license": "MIT", "lang": "Python"},
                {"path": "A.py", "content": "def A():\n    return 43\n", "license": "MIT", "lang": "Python"},
                {"path": "unknown.py", "content": "return 0\n"},
                {"path": "restricted.py", "content": "return -1\n", "license": "all rights reserved"},
            ],
        }], adapter="stack", policy={"category": "code", "license_mode": "per_record",
                                    "allowed_licenses": ["MIT"]})
        result = prepare_object(spec, raw, output, self.config)
        records = self._rows(result)
        self.assertEqual([json.loads(record["metadata_json"])["path"] for record in records], ["Z.py", "A.py"])
        self.assertEqual([record["language"] for record in records], ["Python", "Python"])
        metadata = json.loads(records[0]["metadata_json"])
        self.assertEqual(metadata["file_index"], 0)
        self.assertEqual(metadata["repository_metadata"]["repo_name"], "fixture/repo")
        self.assertEqual(result["input_documents"], 4)
        self.assertEqual(result["quarantined"], {"missing_per_record_license": 1})
        self.assertEqual(result["rejected"], {"per_record_license_not_allowed": 1})

    def test_pointer_records_are_rejected_and_inline_code_needs_no_url(self) -> None:
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 12345\n"
        spec, raw, output = self._object([
            {"content": pointer}, {"blob_id": "abc", "swh_id": "swh:1:cnt:" + "a" * 40},
            {"content": "swh:1:cnt:" + "a" * 40},
            {"content": pointer + "ext-0-example sha256:" + "b" * 64 + "\n"},
            {"content": "def f(x):\n    return x ** 2\n"},
        ], adapter="content", policy={"category": "code"})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["rejected"], {"pointer_only_code": 4})
        self.assertEqual(result["eligible_documents"], 1)
        self.assertTrue(self._rows(result)[0]["text"].startswith("def f"))

    def test_string_meta_license_and_genealogy_aliases_are_real_evidence(self) -> None:
        rows = [{"text": f"A disjoint benchmark-derived prompt number {i}.",
                 "meta": json.dumps({key: "test/fixture", "license": "MIT"})}
                for i, key in enumerate(("hf_dataset_name", "seed_source", "dataset"))]
        rows.append({"text": self.SAFE, "meta": json.dumps({"license": ["MIT"], "dataset": "original"})})
        spec, raw, output = self._object(rows, policy={"license_mode": "per_record", "allowed_licenses": ["MIT"]})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["rejected"], {"benchmark_genealogy": 3})
        self.assertEqual(result["eligible_documents"], 1)
        base = self._rows(result, normalized=True)
        metadata = json.loads(base[0]["metadata_json"])
        self.assertEqual(metadata["hf_dataset_name"], "test/fixture")
        self.assertIn("test/fixture", metadata["source_dataset"])
        self.assertEqual(metadata["upstream"]["meta"], rows[0]["meta"])

    def test_compilation_license_never_fills_missing_per_record_license(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}],
                                          policy={"license_mode": "per_record", "allowed_licenses": ["MIT"]})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["quarantined"], {"missing_per_record_license": 1})
        self.assertTrue(result["empty"])
        self.assertEqual(result["acceptance_fraction"], 0.0)
        self.assertEqual(result["empty_reason"], "all_documents_rejected_or_quarantined")

    def test_unknown_per_record_license_is_not_valid_license_evidence(self) -> None:
        spec, raw, output = self._object([
            {"text": self.SAFE, "license": value}
            for value in ("unknown", "none", "NOASSERTION", "", None)
        ], policy={"license_mode": "per_record"})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["quarantined"], {"missing_per_record_license": 5})
        self.assertEqual(result["eligible_documents"], 0)

    def test_string_false_is_not_boolean_verification_evidence(self) -> None:
        self.profiles["profiles"]["fixture"] = {"require_execution_or_static_verification": True}
        self._write_profiles()
        spec, raw, output = self._object([
            {"text": self.SAFE, "verification_passed": "false"},
            {"text": self.SAFE, "verification_passed": True},
        ])
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["rejected"], {"quality_missing_verification_passed": 1})
        self.assertEqual(result["eligible_documents"], 1)

    def test_source_partition_score_has_provenance_not_invented_verification(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}], policy={
            "category": "math", "metadata": {"math_score": 4, "source_quality_evidence": "publisher_4plus_partition"},
        })
        result = prepare_object(spec, raw, output, self.config)
        record, = self._rows(result)
        metadata = json.loads(record["metadata_json"])
        self.assertEqual(record["quality_score"], 4.0)
        self.assertNotIn("verification_passed", metadata)
        self.assertNotIn("equation_integrity_passed", metadata)
        self.assertIn({"field": "math_score", "method": "explicit_source_policy_default"},
                      metadata["normalization_evidence"])

    def test_forbidden_per_record_source_defaults_fail_the_object(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}], policy={"metadata": {"verification_passed": True}})
        with self.assertRaisesRegex(PreparationError, "per_record_evidence_cannot_be_a_source_default"):
            prepare_object(spec, raw, output, self.config)
        self.assertFalse((output / "NORMALIZED.json").exists())

    def test_publisher_scores_are_not_overwritten_by_partition_defaults(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE, "math_score": 5}],
                                          policy={"metadata": {"math_score": 4}})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(self._rows(result)[0]["quality_score"], 5)

    def test_invalid_explicit_score_is_not_fabricated_as_unknown(self) -> None:
        for score in ("not-a-score", float("nan"), float("inf"), -1):
            spec, raw, output = self._object([{"text": self.SAFE, "quality_score": score}])
            with self.subTest(score=score), self.assertRaisesRegex(PreparationError, "invalid_publisher_quality_score"):
                prepare_object(spec, raw, output, self.config)

    def test_unknown_quality_score_is_unknown_not_a_computed_proxy(self) -> None:
        self.profiles["profiles"]["fixture"] = {"math_score_minimum": 4}
        self._write_profiles()
        spec, raw, output = self._object([{"text": self.SAFE}])
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["rejected"], {"quality_missing_math_score": 1})
        self.assertEqual(self._rows(result, normalized=True)[0]["quality_score"], -1)

    def test_missing_index_publishes_only_ineligible_base_then_reuses_it(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}, {"text": self.HOLDOUT}])
        result = prepare_object(spec, raw, output, {**self.config, "decontamination_index": None})
        self.assertEqual(result["status"], "NORMALIZED_PENDING_DECONTAMINATION")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["chunks"], [])
        self.assertEqual(len(self._rows(result, normalized=True)), 2)
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            eligible = prepare_object(spec, raw, output, self.config)
        self.assertEqual(eligible["normalization_fingerprint"], result["normalization_fingerprint"])
        self.assertEqual(eligible["eligible_documents"], 1)
        self.assertEqual(eligible["rejected"], {"benchmark_exact": 1})

    def test_apply_eligibility_never_opens_raw(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        normalized = normalize_object(spec, raw, output, self.config)
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("raw access")), \
             patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decode")):
            result = apply_eligibility(spec, normalized, output, self.config)
        self.assertEqual(result["eligible_documents"], 1)

    def test_restart_and_worker_count_change_are_noops(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        first = prepare_object(spec, raw, output, self.config)
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            second = prepare_object(spec, raw, output, {**self.config, "workers": 999, "global_commit": "changed"})
        self.assertEqual(first, second)
        self.assertEqual(read_receipt(output / "PREP_COMPLETE.json"), second)
        self.assertTrue((self.root / raw.relative_path).is_file())

    def test_runtime_preloads_index_yaml_and_opt_out_once_for_many_objects(self) -> None:
        config = {**self.config, "opt_out_snapshot": self._opt_out()}
        with patch.object(prep_policy, "load_contamination_index",
                          wraps=prep_policy.load_contamination_index) as index_loader, \
             patch.object(prep_policy, "load_yaml", wraps=prep_policy.load_yaml) as yaml_loader, \
             patch.object(prep_policy, "parse_opt_out_registry",
                          wraps=prep_policy.parse_opt_out_registry) as opt_out_loader:
            self.assertIsNone(prepare_runtime(config))
            self.assertIsNone(prepare_runtime(config))
            index_loader.assert_called_once()
            self.assertEqual(yaml_loader.call_count, 2)
            opt_out_loader.assert_called_once()
            with patch.object(prep_policy, "sha256_file", side_effect=AssertionError("policy rehash")):
                for index in range(3):
                    spec, raw, output = self._object([{
                        "text": f"{self.SAFE} Document {index}.",
                    }],
                                                      policy={"common_crawl_derived": True})
                    result = prepare_object(spec, raw, output, config)
                    self.assertEqual(result["eligible_documents"], 1)
            index_loader.assert_called_once()
            self.assertEqual(yaml_loader.call_count, 2)
            opt_out_loader.assert_called_once()

    @unittest.skipUnless(hasattr(os, "fork"), "Policy sharing requires the Linux/Unix fork start method")
    def test_forked_workers_inherit_verified_policy_without_reloading_or_rehashing(self) -> None:
        config = {**self.config, "opt_out_snapshot": self._opt_out()}
        spec, _, _ = self._object([{"text": self.SAFE}], policy={"common_crawl_derived": True})
        prepare_runtime(config)
        parent_policy = prep_policy.load_eligibility_policy(spec, config)
        children = []
        with patch.object(prep_policy, "load_contamination_index", side_effect=AssertionError("index reload")), \
             patch.object(prep_policy, "_cached_yaml", side_effect=AssertionError("yaml reload")), \
             patch.object(prep_policy, "_cached_opt_out", side_effect=AssertionError("opt-out reload")), \
             patch.object(prep_policy, "sha256_file", side_effect=AssertionError("policy rehash")):
            for _ in range(2):
                reader, writer = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(reader)
                    try:
                        policy = prep_policy.load_eligibility_policy(spec, config)
                        result = {
                            "pid": os.getpid(),
                            "same_index": id(policy.index) == id(parent_policy.index),
                            "same_profiles": id(policy.profiles) == id(parent_policy.profiles),
                            "same_registry": id(policy.registry) == id(parent_policy.registry),
                            "same_opt_out": id(policy.opt_out) == id(parent_policy.opt_out),
                            "ready": not policy.pending,
                            "holdout_reason": policy.index.reason(self.HOLDOUT),
                        }
                    except (AssertionError, RuntimeError, ValueError, OSError) as exc:
                        result = {"error": type(exc).__name__}
                    os.write(writer, json.dumps(result).encode())
                    os.close(writer)
                    os._exit(0)
                os.close(writer)
                children.append((pid, reader))
            for pid, reader in children:
                with os.fdopen(reader, "rb") as stream:
                    result = json.loads(stream.read())
                _, status = os.waitpid(pid, 0)
                self.assertEqual(status, 0)
                self.assertNotIn("error", result)
                self.assertNotEqual(result["pid"], os.getpid())
                self.assertTrue(all(result[name] for name in
                                    ("same_index", "same_profiles", "same_registry", "same_opt_out", "ready")))
                self.assertEqual(result["holdout_reason"], "benchmark_exact")

    def test_preloaded_missing_index_stays_pending_until_coordinator_refreshes(self) -> None:
        config = {
            **self.config,
            "decontamination_index": self.root / "late-policy" / "index.json",
            "benchmark_registry": self.root / "late-policy" / "eval-holdouts.yaml",
        }
        spec, raw, output = self._object([{"text": self.SAFE}])
        prepare_runtime(config)
        first = prepare_object(spec, raw, output, config)
        self.assertEqual(first["status"], "NORMALIZED_PENDING_DECONTAMINATION")
        self._index(directory="late-policy")
        with patch.object(prep_policy, "load_contamination_index", side_effect=AssertionError("worker policy load")), \
             patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            pending = prepare_object(spec, raw, output, config)
        self.assertEqual(pending, first)
        prepare_runtime(config)
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            eligible = prepare_object(spec, raw, output, config)
        self.assertEqual(eligible["status"], "ELIGIBLE")
        self.assertEqual(eligible["eligible_documents"], 1)

    def test_mutating_preloaded_artifact_fails_instead_of_reverifying_gigabytes(self) -> None:
        prepare_runtime(self.config)
        path = self.index_path.with_suffix(".exact.npy")
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        spec, raw, output = self._object([{"text": self.SAFE}])
        with patch.object(prep_policy, "load_contamination_index", side_effect=AssertionError("index reload")), \
             self.assertRaisesRegex(PreparationError, "preloaded_policy_artifact_changed"):
            prepare_object(spec, raw, output, self.config)
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_prepare_result_contains_receipts_and_paths_never_document_payloads(self) -> None:
        text = "DO_NOT_RETURN_THIS_DOCUMENT_VIA_IPC. " + self.SAFE
        spec, raw, output = self._object([{"text": text, "meta": {"document_specific": "ROW_METADATA_NOT_FOR_IPC"}}])
        prepare_runtime(self.config)
        result = prepare_object(spec, raw, output, self.config)
        encoded = json.dumps(result)
        self.assertNotIn(text, encoded)
        self.assertNotIn("ROW_METADATA_NOT_FOR_IPC", encoded)
        self.assertNotIn("metadata_json", encoded)
        self.assertTrue(result["chunks"][0]["path"].endswith(".parquet"))

    def test_quality_policy_change_refilters_base_without_decode(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        first = prepare_object(spec, raw, output, self.config)
        self.profiles["profiles"]["fixture"] = {"minimum_characters": 10_000}
        self._write_profiles()
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            second = prepare_object(spec, raw, output, self.config)
        self.assertEqual(first["normalization_fingerprint"], second["normalization_fingerprint"])
        self.assertNotEqual(first["eligibility_fingerprint"], second["eligibility_fingerprint"])
        self.assertEqual(second["rejected"], {"quality_minimum_characters": 1})
        self.assertEqual(second["eligible_documents"], 0)

    def test_unrelated_quality_profile_does_not_invalidate_this_object(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        first = prepare_object(spec, raw, output, self.config)
        self.profiles["profiles"]["unrelated"] = {"minimum_characters": 10_000}
        self._write_profiles()
        self.assertEqual(prepare_object(spec, raw, output, self.config), first)

    def test_priority_change_reuses_saved_base_through_eligibility_api(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        base = normalize_object(spec, raw, output, self.config)
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("raw access")):
            result = apply_eligibility(replace(spec, priority=123), base, output, self.config)
        self.assertEqual(self._rows(result)[0]["priority"], 123)

    def test_common_crawl_requires_real_snapshot_and_applies_it(self) -> None:
        spec, raw, output = self._object([
            {"text": self.SAFE, "url": "https://blocked.example/page"},
            {"text": self.SAFE, "meta": json.dumps({"url": "https://allowed.example/page"})},
        ], policy={"common_crawl_derived": True})
        pending = prepare_object(spec, raw, output, self.config)
        self.assertEqual(pending["pending_reasons"], ["common_crawl_opt_out_snapshot_unavailable"])
        self.assertEqual(pending["chunks"], [])
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw decoded twice")):
            result = prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": self._opt_out()})
        self.assertEqual(result["rejected"], {"final_common_crawl_opt_out_domain": 1})
        self.assertEqual(result["eligible_documents"], 1)

    def test_no_published_url_is_unknown_coverage_not_invented_url_verification(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}], policy={"common_crawl_derived": True})
        config = {**self.config, "opt_out_snapshot": self._opt_out()}
        result = prepare_object(spec, raw, output, config)
        metadata = json.loads(self._rows(result)[0]["metadata_json"])
        self.assertEqual(metadata["opt_out_application"]["coverage"], "no_published_url")
        self.assertEqual(metadata["opt_out_application"]["status"], "UNKNOWN")
        self.assertEqual(metadata["opt_out_application"]["urls_checked"], 0)
        self.assertFalse(metadata["opt_out_application"]["url_required"])
        self.assertEqual(result["policy_coverage"]["opt_out_no_published_url"], 1)
        self.assertFalse(result["inputs"]["policy"]["opt_out_requires_document_url"])
        self.assertEqual(result["inputs"]["policy"]["opt_out_missing_url_action"], "retain_unknown")
        strict_spec = replace(spec, policy={**spec.policy, "require_opt_out_url": True})
        base = read_receipt(self.root / result["normalization_receipt"])
        strict = apply_eligibility(strict_spec, base, output, config)
        self.assertEqual(strict["quarantined"], {"missing_opt_out_url": 1})
        self.assertEqual(strict["chunks"], [])
        self.assertEqual(strict["inputs"]["policy"]["opt_out_missing_url_action"], "quarantine")
        explicit_unknown = replace(spec, policy={**spec.policy, "require_opt_out_url": False})
        retained = apply_eligibility(explicit_unknown, base, output, config)
        self.assertEqual(retained["eligible_documents"], 1)
        self.assertEqual(json.loads(self._rows(retained)[0]["metadata_json"])["opt_out_application"]["status"],
                         "UNKNOWN")

    def test_hplt_u_language_probability_vectors_and_crawl_alias_are_preserved(self) -> None:
        labels = ["ars_Arab", "arb_Arab", "arz_Arab"]
        probabilities = [0.5932, 0.2465, 0.0942]
        shared = {
            "text": "فقرة اختبار اصطناعية توضّح العلاقة α = β + 1234567890.",
            "lang": labels, "prob": probabilities, "crawl_id": "CC-MAIN-2021-04",
            "html_lang": ["ar"], "seg_langs": ["ars_Arab", "arb_Arab"],
            "doc_scores": [0.1, 0.2], "web-register": {"informational": 0.9},
        }
        spec, raw, output = self._object([
            {**shared, "u": "https://allowed.example/article"},
            {**shared, "u": "https://blocked.example/article"},
        ], wire_format="jsonl_zstd", source_id="hplt3_nonenglish", policy={
            "category": "multilingual", "language": "ars_Arab", "common_crawl_derived": True,
            "metadata": {"language": "ars_Arab", "publisher_wds_bucket": 10},
        })
        result = prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": self._opt_out()})
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["rejected"], {"final_common_crawl_opt_out_domain": 1})
        record, = self._rows(result)
        metadata = json.loads(record["metadata_json"])
        self.assertEqual(metadata["canonical_url"], "https://allowed.example/article")
        self.assertEqual(metadata["u"], metadata["canonical_url"])
        self.assertEqual(metadata["lang"], labels)
        self.assertEqual(metadata["prob"], probabilities)
        self.assertEqual(metadata["language_labels"], labels)
        self.assertEqual(metadata["language_probabilities"], probabilities)
        self.assertEqual(metadata["language_probability"], 0.5932)
        self.assertEqual(record["language"], "ars_Arab")
        self.assertEqual(record["quality_score"], -1)
        self.assertEqual(metadata["crawl"], "CC-MAIN-2021-04")
        self.assertEqual(metadata["crawl_id"], metadata["crawl"])
        self.assertEqual(metadata["seg_langs"], shared["seg_langs"])
        self.assertEqual(metadata["html_lang"], shared["html_lang"])
        self.assertIn({"field": "canonical_url", "method": "upstream_field", "source_field": "u"},
                      metadata["normalization_evidence"])

    def test_hplt_language_selection_uses_actual_aligned_probability_without_rescaling(self) -> None:
        spec, raw, output = self._object([{
            "text": self.SAFE,
            "lang": ["arb_Arab", "ars_Arab", "arz_Arab"],
            "prob": [0.2465, 0.5932, 0.0942],
        }], source_id="hplt3_nonenglish", policy={"language": "arb_Arab", "metadata": {"language": "arb_Arab"}})
        result = prepare_object(spec, raw, output, self.config)
        record, = self._rows(result)
        metadata = json.loads(record["metadata_json"])
        self.assertEqual(record["language"], "ars_Arab")
        self.assertEqual(metadata["source_defaults"]["language"], "arb_Arab")
        self.assertEqual(metadata["language_probability"], 0.5932)
        evidence = next(item for item in metadata["normalization_evidence"]
                        if item.get("method") == "maximum_published_language_probability")
        self.assertEqual(evidence["selected_index"], 1)
        self.assertEqual(metadata["lang"], ["arb_Arab", "ars_Arab", "arz_Arab"])

    def test_hplt_absent_probability_stays_unknown_and_malformed_vectors_fail(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE, "lang": ["ars_Arab"], "prob": None}],
                                          source_id="hplt3_nonenglish")
        result = prepare_object(spec, raw, output, self.config)
        record, = self._rows(result)
        self.assertEqual(record["language"], "ars_Arab")
        self.assertNotIn("language_probability", json.loads(record["metadata_json"]))
        for probabilities in ([0.9], [True, 0.4], [0.4, 1.4]):
            spec, raw, output = self._object([{
                "text": self.SAFE, "lang": ["ars_Arab", "arb_Arab"], "prob": probabilities,
            }], source_id="hplt3_nonenglish")
            with self.assertRaisesRegex(PreparationError, "invalid_hplt_language_probability_vector"):
                prepare_object(spec, raw, output, self.config)

    def test_mind_uuid_unknown_coverage_obeys_explicit_source_policy_without_hydration(self) -> None:
        metadata = {
            "category": "MIND", "finemath_int_scores": 4, "finemath_scores": 4.25,
            "nemocurator_int_scores": 4, "nemocurator_scores": 4.1, "models_used": "Phi-4",
            "warc_filename": "crawl-data/CC-MAIN-2021-04/segments/example/warc/example.warc.gz",
            "warc_id": "<urn:uuid:11111111-2222-3333-4444-555555555555>", "warc_type": "response",
        }
        spec, raw, output = self._object([{
            "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "text": self.SAFE + " A citation https://allowed.example/ is not an original document URL.",
            "metadata": metadata,
        }], source_id="nemotron_cc_math", policy={
            "common_crawl_derived": True, "generated": True, "category": "math",
            "metadata": {"math_score": 4, "source_quality_evidence": "publisher_MIND_partition"},
        })
        config = {**self.config, "opt_out_snapshot": self._opt_out()}
        with patch("socket.create_connection", side_effect=AssertionError("must not hydrate lineage")):
            result = prepare_object(spec, raw, output, config)
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["quarantined"], {})
        retained = json.loads(self._rows(result)[0]["metadata_json"])
        self.assertEqual(retained["opt_out_application"]["status"], "UNKNOWN")
        self.assertEqual(retained["opt_out_application"]["urls_checked"], 0)
        self.assertNotIn("canonical_url", retained)
        base = json.loads(self._rows(result, normalized=True)[0]["metadata_json"])
        self.assertEqual(base["document_url_status"], "not_published")
        self.assertEqual(base["id"], "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        self.assertEqual(base["warc_filename"], metadata["warc_filename"])
        self.assertNotIn("canonical_url", base)
        self.assertTrue((self.root / raw.relative_path).is_file())
        normalized = read_receipt(self.root / result["normalization_receipt"])
        strict_spec = replace(spec, policy={**spec.policy, "require_opt_out_url": True})
        with patch.object(prep, "_verify_raw", side_effect=AssertionError("must not refetch raw")):
            strict = apply_eligibility(strict_spec, normalized, output, config)
        self.assertEqual(strict["eligible_documents"], 0)
        self.assertEqual(strict["chunks"], [])
        self.assertEqual(strict["quarantined"], {"missing_opt_out_url": 1})

    def test_generated_cc_rows_with_published_lineage_urls_still_apply_opt_out(self) -> None:
        spec, raw, output = self._object([
            {"id": "uuid-a", "text": self.SAFE, "metadata": {"source_url": "https://allowed.example/original"}},
            {"id": "uuid-b", "text": self.SAFE, "metadata": {"source_url": "https://blocked.example/original"}},
        ], source_id="nemotron_cc_math", policy={"common_crawl_derived": True, "generated": True})
        result = prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": self._opt_out()})
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["rejected"], {"final_common_crawl_opt_out_domain": 1})
        self.assertEqual(result["quarantined"], {})

    def test_collection_url_defaults_and_userinfo_cannot_satisfy_document_opt_out(self) -> None:
        for row, defaults in (
            ({"text": self.SAFE}, {"canonical_url": "https://allowed.example/collection"}),
            ({"text": self.SAFE, "url": "https://reader:example@allowed.example/document"}, {}),
            ({"text": self.SAFE, "id": "https://allowed.example/acquisition-pointer"}, {}),
        ):
            spec, raw, output = self._object([row], policy={
                "common_crawl_derived": True, "require_opt_out_url": True, "metadata": defaults,
            })
            result = prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": self._opt_out()})
            self.assertEqual(result["quarantined"], {"missing_opt_out_url": 1})
            self.assertEqual(result["eligible_documents"], 0)

    def test_empty_opt_out_snapshot_is_not_a_policy(self) -> None:
        snapshot = self.root / "empty.csv"
        snapshot.write_text("Publisher/Requester,Date of notice,List of domains/URLs\n")
        spec, raw, output = self._object([{"text": self.SAFE}], policy={"common_crawl_derived": True})
        with self.assertRaisesRegex(PreparationError, "empty_or_incomplete_common_crawl"):
            prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": snapshot})
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_source_quality_selection_pending_never_marks_raw_web_eligible(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}],
                                          policy={"metadata": {"quality_selection_pending": True}})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["status"], "NORMALIZED_PENDING_DECONTAMINATION")
        self.assertIn("source_quality_selection_pending", result["pending_reasons"])
        self.assertEqual(result["chunks"], [])

    def test_exact_normalized_contiguous_and_code_overlap_use_existing_index(self) -> None:
        code = "def calculate(x):\n    return (x*17)+(x//3)-(x%2)\n"
        index, registry = self._index(directory="overlap-policy", holdouts=[self.HOLDOUT, code])
        config = {**self.config, "decontamination_index": index, "benchmark_registry": registry}
        copied_span = "Independent introduction. " + self.HOLDOUT + " Independent conclusion."
        modified_code = "# new header\n" + code + "# final note\n"
        spec, raw, output = self._object([
            {"text": self.HOLDOUT.upper()}, {"text": copied_span}, {"text": modified_code}, {"text": self.SAFE},
        ])
        result = prepare_object(spec, raw, output, config)
        self.assertEqual(result["rejected"], {
            "benchmark_exact": 1, "benchmark_contiguous_run": 1, "benchmark_code_ngram": 1,
        })
        self.assertEqual(result["eligible_documents"], 1)

    def test_ngram_family_still_applies_when_contiguous_threshold_is_not_reached(self) -> None:
        index, registry = self._index(directory="ngram-policy", contiguous_run_minimum=1000)
        spec, raw, output = self._object([{"text": self.HOLDOUT + " new tail"}])
        result = prepare_object(spec, raw, output, {
            **self.config, "decontamination_index": index, "benchmark_registry": registry,
        })
        self.assertEqual(result["rejected"], {"benchmark_ngram": 1})

    def test_valid_empty_containers_have_explicit_receipts_invalid_schema_does_not(self) -> None:
        for wire in ("raw_jsonl", "jsonl_gzip", "json_zstd", "parquet"):
            with self.subTest(wire=wire):
                spec, raw, output = self._object([], wire_format=wire, schema=pa.schema([("text", pa.string())]))
                result = prepare_object(spec, raw, output, self.config)
                self.assertTrue(result["empty"])
                self.assertEqual(result["empty_reason"], "empty_input")
                self.assertEqual(result["chunks"], [])
                self.assertEqual(result["input_rows"], 0)
        spec, raw, output = self._object([], wire_format="parquet", schema=pa.schema([("unrecognized", pa.string())]))
        with self.assertRaisesRegex(PreparationError, "invalid_payload_schema"):
            prepare_object(spec, raw, output, self.config)
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_invalid_row_schema_is_not_a_valid_empty_corpus(self) -> None:
        for row in ({"unexpected": "data"}, {"text": 123}, {"messages": {"role": "user"}}):
            spec, raw, output = self._object([row])
            with self.subTest(row_keys=tuple(row)), self.assertRaises(PreparationError):
                prepare_object(spec, raw, output, self.config)
            self.assertFalse((output / "NORMALIZED.json").exists())

    def test_parser_failure_names_source_object_row_without_payload_and_leaves_no_partial_success(self) -> None:
        forbidden_sample = "DO-NOT-LOG-THIS-CORPUS-SAMPLE"
        payload = (json.dumps({"text": self.SAFE}) + "\n" + '{"text": "' + forbidden_sample).encode()
        spec, raw, output = self._object(payload=payload)
        with self.assertRaises(PreparationError) as caught:
            prepare_object(spec, raw, output, {**self.config, "output_chunk_bytes": 1})
        message = str(caught.exception)
        self.assertIn("source=fixture_source", message)
        self.assertIn(spec.object_id, message)
        self.assertIn("row=2", message)
        self.assertNotIn(forbidden_sample, message)
        self.assertFalse((output / "PREP_COMPLETE.json").exists())
        self.assertEqual(list(output.rglob("*.parquet")), [])
        self.assertEqual(list(output.rglob("*.partial")), [])
        self.assertTrue((self.root / raw.relative_path).is_file())

    def test_gzip_and_zstandard_truncated_frames_fail_even_after_valid_json(self) -> None:
        data = (json.dumps({"text": self.SAFE}) + "\n").encode()
        for wire, payload in (
            ("jsonl_gzip", gzip.compress(data)[:-4]),
            ("jsonl_zstd", zstandard.ZstdCompressor(write_checksum=True).compress(data)[:-1]),
        ):
            spec, raw, output = self._object(wire_format=wire, payload=payload)
            with self.subTest(wire=wire), self.assertRaises(PreparationError):
                prepare_object(spec, raw, output, self.config)
            self.assertFalse((output / "NORMALIZED.json").exists())

    def test_concatenated_zstandard_frames_cover_every_record(self) -> None:
        compressor = zstandard.ZstdCompressor(write_checksum=True)
        rows = [{"text": f"Independent compressed frame {i}"} for i in range(4)]
        payload = b"".join(compressor.compress((json.dumps(row) + "\n").encode()) for row in rows)
        spec, raw, output = self._object(wire_format="json_zstd", payload=payload)
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual([row["text"] for row in self._rows(result)], [row["text"] for row in rows])

    def test_raw_identity_size_digest_and_root_are_checked_before_decode(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        for broken in (
            replace(raw, object_id="different"), replace(raw, source_id="other"),
            replace(raw, byte_count=raw.byte_count + 1), replace(raw, sha256="0" * 64),
            replace(raw, relative_path="../../outside-release"),
        ):
            with self.subTest(receipt=broken.object_id), \
                 patch.object(prep, "iter_source_rows", side_effect=AssertionError("unverified read")), \
                 self.assertRaises(PreparationError):
                prepare_object(spec, broken, output, self.config)

    def test_unknown_source_defaults_are_invalid_even_for_an_empty_object(self) -> None:
        spec, raw, output = self._object([], policy={"metadata": {"execution_passed": True}})
        with self.assertRaisesRegex(PreparationError, "per_record_evidence_cannot_be_a_source_default"):
            prepare_object(spec, raw, output, self.config)

    def test_reentry_rejects_changed_extraction_policy_but_keeps_sealed_base(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        normalized = normalize_object(spec, raw, output, self.config)
        changed = replace(spec, policy={**spec.policy, "metadata": {"math_score": 5}})
        with self.assertRaisesRegex(PreparationError, "normalization_contract_changed"):
            apply_eligibility(changed, normalized, output, self.config)
        self.assertTrue((self.root / normalized["chunks"][0]["path"]).is_file())

    def test_nested_file_license_and_metadata_array_genealogy_are_preserved(self) -> None:
        spec, raw, output = self._object([
            {"text": self.SAFE, "file_info": {"detected_licenses": ["MIT"]}},
            {"text": "A derived exercise.", "meta": json.dumps([{"seed_source": "test/fixture"}])},
        ], policy={"license_mode": "per_record", "allowed_licenses": ["MIT"]})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["rejected"], {"benchmark_genealogy": 1})

    def test_nonstring_nested_file_text_is_not_pointer_quarantine(self) -> None:
        spec, raw, output = self._object([{"files": [{"content": 123, "path": "bad.py"}]}], adapter="stack")
        with self.assertRaisesRegex(PreparationError, "file_text_is_not_a_string"):
            prepare_object(spec, raw, output, self.config)
        self.assertFalse((output / "NORMALIZED.json").exists())

    def test_empty_primary_field_can_use_a_real_inline_secondary_field(self) -> None:
        spec, raw, output = self._object([{"text": "", "content": self.SAFE}])
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(self._rows(result)[0]["text"], self.SAFE)

    def test_corrupt_zstandard_checksums_are_named_without_source_payloads(self) -> None:
        payload = bytearray(zstandard.ZstdCompressor(write_checksum=True).compress(
            (json.dumps({"text": self.SAFE}) + "\n").encode(),
        ))
        payload[-1] ^= 1
        spec, raw, output = self._object(wire_format="jsonl_zstd", payload=bytes(payload))
        with self.assertRaisesRegex(PreparationError, "invalid_zstandard_payload"):
            prepare_object(spec, raw, output, self.config)
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_preparation_cleans_only_its_abandoned_incomplete_generation(self) -> None:
        output = self.root / "abandoned-output"
        output.mkdir()
        fingerprint = "a" * 64
        abandoned = output / "normalized" / f".{fingerprint}.{'b' * 32}.partial"
        abandoned.mkdir(parents=True)
        (abandoned / "partial.parquet").write_bytes(b"unfinished fixture")
        unrelated = abandoned.parent / ".other-object.partial"
        unrelated.mkdir()
        (unrelated / "keep").write_bytes(b"unrelated fixture")
        destination, staging = prep._generation(output, "normalized", fingerprint)
        self.assertFalse(abandoned.exists())
        self.assertTrue((unrelated / "keep").exists())
        self.assertTrue(staging.is_dir())
        self.assertEqual(destination.name, fingerprint)

    def test_legacy_short_ngram_policy_cannot_be_mislabeled_as_metis17(self) -> None:
        index = ContaminationIndex.build([self.HOLDOUT], contiguous_run_minimum=8)
        directory = self.root / "legacy-policy"
        registry = write_contamination_inputs(directory, index, [self.HOLDOUT])
        path = directory / "index.json"
        save_contamination_index(index, path, benchmark_registry_path=registry)
        spec, raw, output = self._object([{"text": self.SAFE}])
        with self.assertRaisesRegex(PreparationError, "decontamination_policy_not_metis17"):
            prepare_object(spec, raw, output, {
                **self.config, "decontamination_index": path, "benchmark_registry": registry,
            })
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_unsupported_formats_and_adapters_have_no_silent_fallback(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        for changed in (replace(spec, wire_format="zip"), replace(spec, adapter="guess")):
            with self.subTest(adapter=changed.adapter), self.assertRaises(PreparationError):
                prepare_object(changed, raw, output, self.config)

    def test_corrupted_saved_normalization_never_gets_filtered_as_valid(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE}])
        normalized = normalize_object(spec, raw, output, self.config)
        path = self.root / normalized["chunks"][0]["path"]
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaisesRegex(PreparationError, "artifact_digest_mismatch"):
            apply_eligibility(spec, normalized, output, self.config)
        self.assertFalse((output / "PREP_COMPLETE.json").exists())

    def test_json_encoded_metadata_is_not_python_execution(self) -> None:
        spec, raw, output = self._object([{"text": self.SAFE, "meta": "__import__('os').system('echo bad')"}])
        with patch("os.system", side_effect=AssertionError("must never execute")), \
             self.assertRaisesRegex(PreparationError, "invalid_json_encoded_meta"):
            prepare_object(spec, raw, output, self.config)

    def test_french_science_pages_are_quarantined_until_completeness_is_explicit(self) -> None:
        spec, raw, output = self._object([
            {"text": "Une page isolée, pas un document complet.", "page_count": 5},
            {"pages": ["Première page.\n  Équation α = 42.", "Deuxième page."], "page_count": 2},
            {"text": "Un document entier explicitement complet.", "document_complete": True},
            {"text": "Un fragment explicitement incomplet.", "page_count": 1, "document_complete": False},
        ], adapter="french_science", policy={"language": "fr"})
        result = prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["quarantined"], {"source_local_assembly_pending": 2})
        self.assertEqual(result["eligible_documents"], 2)
        self.assertEqual(self._rows(result)[0]["text"], "Première page.\n  Équation α = 42.\nDeuxième page.")

    def test_wet_conversion_records_are_streamed_without_hydrating_references(self) -> None:
        data = self._warc([
            ("metadata", "https://allowed.example/a", b"not a training document", None),
            ("conversion", "https://allowed.example/a", "Un texte français 123456789.\n  α = β".encode(), None),
            ("conversion", "https://blocked.example/b", self.SAFE.encode(), None),
        ])
        spec, raw, output = self._object(wire_format="warc_wet_gzip", adapter="wet", payload=data,
                                          policy={"common_crawl_derived": True})
        result = prepare_object(spec, raw, output, {**self.config, "opt_out_snapshot": self._opt_out()})
        self.assertEqual(result["input_rows"], 3)
        self.assertEqual(result["rejected"], {"warc_not_conversion": 1, "final_common_crawl_opt_out_domain": 1})
        self.assertEqual(self._rows(result)[0]["text"], "Un texte français 123456789.\n  α = β")

    def test_whole_cc_news_warc_extracts_html_code_and_equations_without_network(self) -> None:
        html = (
            '<html lang="fr"><head><title>Une étude</title></head><body>'
            '<p>Contexte scientifique <math><mi>α</mi><mo>=</mo><mn>12345</mn></math>.</p>'
            '<script type="math/tex; mode=display">\\int_0^1 x^2\\,dx = 1/3</script>'
            '<pre>def F(x):\n    return x + 42\n</pre><script>NEVER_EXECUTE</script></body></html>'
        )
        blocked = '<html><head><meta name="robots" content="noai"></head><body>Publisher opts out.</body></html>'
        data = self._warc([
            ("response", "https://allowed.example/article", html.encode(), {"Content-Type": "text/html; charset=utf-8"}),
            ("response", "https://allowed.example/private", blocked.encode(), {"Content-Type": "text/html; charset=utf-8"}),
        ])
        spec, raw, output = self._object(wire_format="warc_gzip", adapter="cc_news", payload=data)
        with patch("socket.create_connection", side_effect=AssertionError("no network")):
            result = prepare_object(spec, raw, output, self.config)
        record, = self._rows(result)
        self.assertIn("α=12345", record["text"])
        self.assertIn("\\int_0^1 x^2\\,dx = 1/3", record["text"])
        self.assertIn("def F(x):\n    return x + 42\n", record["text"])
        self.assertNotIn("NEVER_EXECUTE", record["text"])
        self.assertEqual(record["language"], "fr")
        self.assertEqual(result["rejected"], {"publisher_machine_learning_opt_out": 1})

    def test_wikipedia_current_namespace_zero_retains_wikitext(self) -> None:
        wikitext = "== Théorème ==\n  x² = 12345\n[[Lien|mot]]"
        xml = (
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
            f'<page><title>Math</title><ns>0</ns><id>7</id><revision><id>9</id><text>{wikitext}</text></revision></page>'
            '<page><title>Talk:Math</title><ns>1</ns><id>8</id><revision><text>Discussion</text></revision></page>'
            '</mediawiki>'
        )
        spec, raw, output = self._object(wire_format="xml_bzip2", adapter="wikipedia",
                                          payload=bz2.compress(xml.encode()), policy={"language": "fr"})
        result = prepare_object(spec, raw, output, self.config)
        record, = self._rows(result)
        self.assertEqual(record["text"], wikitext)
        self.assertEqual(result["rejected"], {"wikipedia_nonarticle_namespace": 1})
        self.assertEqual(json.loads(record["metadata_json"])["revision_id"], "9")

    def test_wikipedia_history_and_truncated_xml_are_not_current_objects(self) -> None:
        for xml in (
            '<mediawiki><page><ns>0</ns><revision><text>Old</text></revision><revision><text>New</text></revision></page></mediawiki>',
            '<mediawiki><page><ns>0</ns><revision><text>Truncated',
        ):
            spec, raw, output = self._object(wire_format="xml_bzip2", adapter="wikipedia",
                                              payload=bz2.compress(xml.encode()))
            with self.assertRaises(PreparationError):
                prepare_object(spec, raw, output, self.config)
            self.assertFalse((output / "NORMALIZED.json").exists())


if __name__ == "__main__":
    unittest.main()
