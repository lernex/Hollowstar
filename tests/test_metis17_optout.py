from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from metis_data.datatrove_blocks import save_contamination_index
from metis_data.decontaminate import ContaminationIndex
from metis_data.freshweb import parse_opt_out_registry
from metis_data17 import policy, prep_policy
from metis_data17.common import ObjectSpec, read_receipt, sha256_file, write_receipt
from metis_data17.optout17 import (
    HEADER, PARSER_VERSION, OptOut17Error, load_opt_out_snapshot17,
    parse_opt_out_registry17, snapshot_common_crawl_opt_out17,
)
from metis_data17.prep_readers import (
    _PER_RECORD_DEFAULTS, PreparationError, SourceRow, extract_documents, normalize_document,
)
from tests.contamination_fixtures import write_contamination_inputs


def registry_bytes(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def audited_rows() -> list[list[str]]:
    return [
        ["Bünyamin Tamar", "2026-08-10",
         "github.com/bunyamintamar\nbunyamintamar.github.io/\nmedium.com/@bunyamintamar"],
        ["DCN - Warner Brothers Discovery", "2026-06-04",
         "wbd.com\nboingtv.it\nDCN - Washington Post\nwashingtonpost.com\nnewsweek.com\nkaplan.com"],
        ["News Media Alliance", "2026-04-29",
         "thedodo.com\nthrillist.com Ziff Davis\naberdeen.com\naskmen.com"],
        ["Le Monde", "2024-03-16",
         "lemonde.fr to start (2024-03-16)\nAdditional (2024-04-12): \n"
         "https://www.courrierinternational.com/\nhttps://www.nouvelobs.com/"],
        ["Alliance de la Presse d’Information", "2025-09-03", "n/a"],
    ]


class OptOut17ParsingTests(unittest.TestCase):
    def test_path_at_is_a_scoped_url_not_a_whole_domain(self) -> None:
        payload = registry_bytes([["Fixture publisher", "2026-09-04", "medium.com/@bunyamintamar"]])
        parsed = parse_opt_out_registry17(payload)
        self.assertNotIn("medium.com", parsed.domains)
        self.assertEqual(len(parsed.url_rules), 1)
        self.assertEqual(parsed.reason("https://medium.com/@bunyamintamar"), "common_crawl_opt_out_url")
        self.assertEqual(parsed.reason("https://medium.com/@bunyamintamar/a-post"), "common_crawl_opt_out_url")
        self.assertIsNone(parsed.reason("https://medium.com/@bunyamintamar-other"))
        self.assertIsNone(parsed.reason("https://medium.com/@another-author/a-post"))
        self.assertEqual(parsed.snapshot_sha256, hashlib.sha256(payload).hexdigest())

    def test_query_and_path_at_are_supported_without_allowing_userinfo(self) -> None:
        parsed = parse_opt_out_registry17(registry_bytes([
            ["Fixture", "2026-09-04", "https://host.example/@author/feed?account=@author"],
        ]))
        self.assertEqual(parsed.reason("https://host.example/@author/feed?account=@author"),
                         "common_crawl_opt_out_url")
        for value in (
            "https://reader:example@host.example/@author",
            "https://reader@host.example/path",
            "https://@host.example/path",
            "reader@host.example",
            "allowed.example https://reader:example@host.example/@author",
            "allowed.example https://reader%40host.example/path",
        ):
            with self.subTest(value=value), self.assertRaises(OptOut17Error):
                parse_opt_out_registry17(registry_bytes([["Fixture", "2026-09-04", value]]))

    def test_audited_headings_retain_all_actual_rules_and_original_raw_sha(self) -> None:
        payload = registry_bytes(audited_rows())
        parsed = parse_opt_out_registry17(payload)
        self.assertEqual(parsed.input_entries, parsed.parsed_entries + parsed.non_rule_entries)
        self.assertEqual(parsed.non_rule_entries, 3)
        self.assertEqual(parsed.unparsed_entries, 0)
        self.assertEqual(parsed.snapshot_sha256, hashlib.sha256(payload).hexdigest())
        for domain in ("wbd.com", "washingtonpost.com", "newsweek.com", "kaplan.com", "thrillist.com",
                       "lemonde.fr", "www.courrierinternational.com", "www.nouvelobs.com"):
            with self.subTest(domain=domain):
                self.assertEqual(parsed.reason(f"https://{domain}/article"), "common_crawl_opt_out_domain")
        audit = parsed.audit()
        self.assertEqual(audit["parser_version"], PARSER_VERSION)
        self.assertEqual(audit["source_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual({entry["kind"] for entry in audit["annotations"]}, {
            "embedded_requester_heading", "inline_requester_heading", "inline_notice_date",
            "additional_notice_heading", "explicit_no_rules_placeholder",
        })
        label = next(item for item in audit["annotations"] if item["kind"] == "embedded_requester_heading")
        self.assertEqual(label["previous_entry"], "boingtv.it")
        self.assertEqual(label["next_entry"], "washingtonpost.com")

    def test_dcn_text_is_not_ignored_outside_its_audited_context(self) -> None:
        for row in (
            ["Fixture", "2026-09-04", "DCN - Washington Post"],
            ["DCN - Warner Brothers Discovery", "2026-06-05",
             "wbd.com\nboingtv.it\nDCN - Washington Post\nwashingtonpost.com"],
            ["DCN - Warner Brothers Discovery", "2026-06-04",
             "wbd.com\nother.example\nDCN - Washington Post\nwashingtonpost.com"],
            ["DCN - Warner Brothers Discovery", "2026-06-04",
             "wbd.com\nboingtv.it\nDCN - Washington Post\nother.example"],
            ["DCN - Warner Brothers Discovery", "2026-06-04",
             "wbd.com\nboingtv.it\nDCN - Another Publisher\nwashingtonpost.com"],
        ):
            with self.subTest(publisher=row[0]), self.assertRaises(OptOut17Error):
                parse_opt_out_registry17(registry_bytes([row]))

    def test_inline_annotations_also_require_audited_context(self) -> None:
        for entry in ("thrillist.com Ziff Davis", "lemonde.fr to start (2024-03-16)",
                      "Additional (2024-04-12):", "allowed.example unrecognized-text"):
            with self.subTest(entry=entry), self.assertRaises(OptOut17Error):
                parse_opt_out_registry17(registry_bytes([["Other", "2026-09-04", entry]]))

    def test_no_percentage_based_unknown_rule_allowance(self) -> None:
        lines = [f"publisher-{index}.example" for index in range(100)]
        payload = registry_bytes([["Fixture", "2026-09-04", "\n".join([*lines, "unrecognized-text"])]])
        self.assertEqual(parse_opt_out_registry(payload).unparsed_entries, 1)
        with self.assertRaisesRegex(OptOut17Error, "unrecognized_registry_token"):
            parse_opt_out_registry17(payload)

    def test_legacy_compatible_rules_have_identical_matching_decisions(self) -> None:
        payload = registry_bytes([
            ["Fixture", "2026-09-04",
             "- blocked.example\n*.wild.example/section*\n"
             "https://query.example/search?b=2&a=1\nhttps://plain.example/private\n"
             "one.example,two.example;three.example\nbücher.example/путь"],
            ["LAST UPDATED: 2026-09-04", "", ""],
        ])
        old, new = parse_opt_out_registry(payload), parse_opt_out_registry17(payload)
        self.assertEqual(old.domains, new.domains)
        self.assertEqual(old.url_paths, new.url_paths)
        self.assertEqual(set(old.url_rules), set(new.url_rules))
        self.assertEqual(old.input_entries, new.input_entries)
        self.assertEqual(old.last_updated, new.last_updated)
        for url in (
            "https://sub.blocked.example/document", "http://wild.example/section/one",
            "http://wild.example/elsewhere", "https://query.example/search?a=1&b=2",
            "https://query.example/search?a=9&b=2", "https://plain.example/private/nested",
            "https://plain.example/public", "https://two.example/page",
            "https://bücher.example/путь", "https://allowed.example/",
        ):
            self.assertEqual(new.reason(url), old.reason(url), url)

    def test_empty_malformed_and_unknown_schemas_fail_closed(self) -> None:
        for payload in (
            b"", b"wrong,columns,here\n",
            registry_bytes([]),
            registry_bytes([["Fixture", "2026-09-04", "n/a"]]),
            registry_bytes([["Fixture", "2026-09-04"]]),
            b'Publisher/Requester,Date of notice,List of domains/URLs\n"x,y,"unterminated',
        ):
            with self.subTest(payload_bytes=len(payload)), self.assertRaises(OptOut17Error):
                parse_opt_out_registry17(payload)

    def test_real_ncc_math_partition_defaults_remain_source_evidence_not_row_verification(self) -> None:
        partitions = (
            (100, {"math_score": 4, "source_quality_evidence": "publisher_4plus_partition"}),
            (98, {"math_score": 4, "source_quality_evidence": "publisher_MIND_partition", "generator_family": "Phi-4"}),
            (90, {"math_score": 3, "source_quality_evidence": "publisher_score3_partition"}),
        )
        for priority, metadata in partitions:
            self.assertFalse(set(metadata) & _PER_RECORD_DEFAULTS)
            spec = ObjectSpec.create(
                source_id="nemotron_cc_math", url="https://example.org/pinned.parquet",
                revision="pinned", relative_key="partition/file.parquet", wire_format="parquet",
                adapter="text", priority=priority, policy={
                    "category": "math", "language": "en", "metadata": metadata,
                },
            )
            item = SourceRow(1, {"text": "Let α² = 2. This synthetic fixture preserves its equation."})
            document, = extract_documents(item, spec)
            normalized = normalize_document(document, spec, item.index)
            evidence = json.loads(normalized["metadata_json"])
            self.assertEqual(evidence["source_defaults"], metadata)
            self.assertEqual(evidence["math_score"], metadata["math_score"])
            self.assertIn({"field": "math_score", "method": "explicit_source_policy_default"},
                          evidence["normalization_evidence"])
            self.assertNotIn("verification_passed", evidence)
            self.assertNotIn("parser_or_compiler_passed", evidence)
            measured = SourceRow(2, {"text": item.value["text"], "math_score": 5})
            document, = extract_documents(measured, spec)
            self.assertEqual(normalize_document(document, spec, 2)["quality_score"], 5)


class OptOut17PolicyReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = (Path.cwd() / ".metis-runtime" / "optout17-tests" / uuid.uuid4().hex).resolve()
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.payload = registry_bytes(audited_rows())
        self.snapshot = self.root / "policy" / "opt-out" / "frozen.csv"
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_bytes(self.payload)
        holdout = " ".join(f"benchmarktoken{index}" for index in range(30))
        index = ContaminationIndex.build([holdout], contiguous_run_minimum=8)
        self.source = self.root / "source-index"
        self.registry = write_contamination_inputs(self.source, index, [holdout])
        self.index_path = self.source / "index.json"
        save_contamination_index(index, self.index_path, benchmark_registry_path=self.registry)
        self.current = self.root / "policy" / "CURRENT.json"

    def _current(self) -> dict:
        value = {
            "schema": "metis17.policy-ready/v1",
            "decontamination_index": str(self.index_path),
            "benchmark_registry": str(self.registry),
            "holdout_registry_sha256": sha256_file(self.registry),
            "opt_out_snapshot": str(self.snapshot),
            "opt_out_sha256": sha256_file(self.snapshot),
            "opt_out_unparsed_entries": 2,
        }
        write_receipt(self.current, value)
        return value

    def test_policy_config_reinterprets_old_counts_without_mutating_current_or_raw(self) -> None:
        self._current()
        before = self.current.read_bytes()
        raw_before = self.snapshot.read_bytes()
        result = policy.policy_config(self.root)
        self.assertTrue(result["policy_ready"])
        self.assertEqual(result["opt_out_unparsed_entries"], 0)
        self.assertEqual(result["published_opt_out_unparsed_entries"], 2)
        self.assertEqual(result["opt_out_parser_version"], PARSER_VERSION)
        self.assertEqual(result["opt_out_sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(self.current.read_bytes(), before)
        self.assertEqual(self.snapshot.read_bytes(), raw_before)

    def test_policy_config_never_trusts_a_zero_unknown_count_in_current(self) -> None:
        self.snapshot.write_bytes(registry_bytes([["Fixture", "2026-09-04", "good.example\nunknown-text"]]))
        value = self._current()
        value["opt_out_unparsed_entries"] = 0
        write_receipt(self.current, value)
        before = self.current.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "unrecognized_registry_token"):
            policy.policy_config(self.root)
        self.assertEqual(self.current.read_bytes(), before)

    def test_policy_config_rejects_raw_mutation_and_path_escape(self) -> None:
        value = self._current()
        self.snapshot.write_bytes(self.payload + b"\n")
        with self.assertRaisesRegex(RuntimeError, "raw_snapshot_sha256_mismatch"):
            policy.policy_config(self.root)
        value["opt_out_snapshot"] = "../outside-release.csv"
        write_receipt(self.current, value)
        with self.assertRaises(ValueError):
            policy.policy_config(self.root)

    def test_import_validates_strict_policy_before_copying_arrays_or_publishing_current(self) -> None:
        self._current()
        before = self.current.read_bytes()
        self.snapshot.write_bytes(registry_bytes([["Fixture", "2026-09-04", "allowed.example unknown-text"]]))
        bogus_metadata = {"path": str(self.snapshot), "sha256": sha256_file(self.snapshot), "unparsed_entries": 0}
        with patch.object(policy, "snapshot_common_crawl_opt_out", return_value=bogus_metadata), \
             patch.object(policy, "_copy_verified", side_effect=AssertionError("copied before validation")), \
             self.assertRaisesRegex(RuntimeError, "unrecognized_registry_token"):
            policy.import_policy(self.root, self.source, registry_path=self.registry)
        self.assertEqual(self.current.read_bytes(), before)

    def test_import_reports_strict_audit_and_does_not_rewrite_immutable_index_on_repeat(self) -> None:
        metadata = {"path": str(self.snapshot), "sha256": sha256_file(self.snapshot), "unparsed_entries": 2}
        source_hash = sha256_file(self.index_path)
        with patch.object(policy, "snapshot_common_crawl_opt_out", return_value=metadata):
            first = policy.import_policy(self.root, self.source, registry_path=self.registry)
            derived = Path(first["decontamination_index"])
            index_hash, index_stamp = sha256_file(derived), derived.stat().st_mtime_ns
            second = policy.import_policy(self.root, self.source, registry_path=self.registry)
        self.assertEqual(first["opt_out_unparsed_entries"], 0)
        self.assertEqual(first["opt_out_parser_version"], PARSER_VERSION)
        self.assertEqual(first["opt_out_audit"]["non_rule_entries"], 3)
        self.assertEqual(first["opt_out_sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(second["decontamination_index"], first["decontamination_index"])
        self.assertEqual((sha256_file(derived), derived.stat().st_mtime_ns), (index_hash, index_stamp))
        self.assertEqual(sha256_file(self.index_path), source_hash)
        self.assertEqual(read_receipt(self.current), second)
        self.assertTrue(policy.policy_config(self.root)["policy_ready"])

    def test_strict_snapshot_fetcher_preserves_bytes_and_never_uses_legacy_parser(self) -> None:
        payload = registry_bytes([["Fixture", "2026-09-04", "medium.com/@author"]])
        destination = self.root / "new-snapshot"
        with patch("metis_data17.optout17._HttpClient") as client:
            client.return_value.bytes.return_value = payload
            first = snapshot_common_crawl_opt_out17(destination)
            second = snapshot_common_crawl_opt_out17(destination)
        self.assertEqual(Path(first["path"]).read_bytes(), payload)
        self.assertEqual(first["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(first["unparsed_entries"], 0)
        parsed = load_opt_out_snapshot17(Path(first["path"]), first["sha256"])
        self.assertEqual(parsed.reason("https://medium.com/@author/story"), "common_crawl_opt_out_url")

    def test_preloader_uses_strict_parser_and_binds_its_effective_rule_digest(self) -> None:
        metadata = {"path": str(self.snapshot), "sha256": sha256_file(self.snapshot), "unparsed_entries": 2}
        with patch.object(policy, "snapshot_common_crawl_opt_out", return_value=metadata):
            policy.import_policy(self.root, self.source, registry_path=self.registry)
        quality = self.root / "quality.yaml"
        quality.write_text("defaults: {}\nprofiles:\n  math: {}\n")
        config = {**policy.policy_config(self.root), "quality_profiles_path": quality}
        prep_policy.prepare_runtime(config)
        spec = ObjectSpec.create(
            source_id="fixture", url="https://example.org/pinned.parquet", revision="pinned",
            relative_key="file.parquet", wire_format="parquet", adapter="text", priority=100,
            policy={"quality_profile": "math", "license_mode": "compilation",
                    "collection_license": "fixture", "common_crawl_derived": True},
        )
        ready = prep_policy.load_eligibility_policy(spec, config)
        self.assertFalse(ready.pending)
        self.assertEqual(ready.descriptor["opt_out_parser_version"], PARSER_VERSION)
        self.assertEqual(ready.descriptor["opt_out_effective_rules_sha256"], ready.opt_out.rules_sha256)
        self.assertEqual(ready.opt_out.reason("https://medium.com/@bunyamintamar/post"),
                         "common_crawl_opt_out_url")
        spec = ObjectSpec.create(**{
            "source_id": "noncc", "url": "https://example.org/other.parquet", "revision": "pinned",
            "relative_key": "file.parquet", "wire_format": "parquet", "adapter": "text", "priority": 100,
            "policy": {**spec.policy, "common_crawl_derived": False},
        })
        unrelated = prep_policy.load_eligibility_policy(spec, config)
        self.assertFalse(any(key.startswith("opt_out_") for key in unrelated.descriptor))


if __name__ == "__main__":
    unittest.main()
