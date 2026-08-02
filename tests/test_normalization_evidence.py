from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from metis_data import stage_runner
from metis_data.manifest import load_manifest
from metis_data.normalization_evidence import (
    derive_normalization_evidence,
    extract_training_text,
    final_common_crawl_opt_out_reason,
    load_frozen_common_crawl_opt_out,
)
from metis_data.quality import evaluate_quality
from metis_data.state import StateStore


ROOT = Path(__file__).resolve().parents[1]


class NormalizationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = load_manifest(ROOT / "manifests" / "metis-1.6.yaml")
        cls.sources = {source["id"]: source for source in manifest["sources"]}

    def derive(self, source_id: str, row: dict, *, repo_path: str = "data/shard.parquet"):
        source = self.sources[source_id]
        text = extract_training_text(row)
        metadata = derive_normalization_evidence(
            row,
            source,
            {"repo_path": repo_path, "revision": "fixture-revision"},
            text,
        )
        decision = evaluate_quality(
            text,
            profile_name=source["processing"]["quality_profile"],
            metadata=metadata,
            fail_closed=True,
        )
        return text, metadata, decision

    @staticmethod
    def prose(paragraphs: int = 30, *, prefix: str = "Section") -> str:
        return "\n\n".join(
            (
                f"{prefix} {index}. This document explains the evidence and the method used "
                "to evaluate the result. The analysis is based on the available source, "
                "and it describes how the system works, why the result matters, and what "
                "the reader should understand before applying it in another setting."
            )
            for index in range(paragraphs)
        )

    def test_common_pile_book_maps_nested_license_and_computes_chapter_integrity(self) -> None:
        text = "\n\n".join(
            [
                "Chapter One\n" + self.prose(22, prefix="First chapter section"),
                "Chapter Two\n" + self.prose(22, prefix="Second chapter section"),
                "Chapter Three\n" + self.prose(22, prefix="Third chapter section"),
            ]
        )
        row = {
            "id": "gutenberg-1",
            "text": text,
            # Some Common Pile shards expose this struct as JSON text while
            # others decode it to a mapping; both must preserve the same proof.
            "metadata": json.dumps(
                {
                    "language": "en",
                    "license": "Public Domain",
                    "provenance": "project_gutenberg-0000.json.gz:1",
                    "url": "https://www.gutenberg.org/ebooks/1",
                }
            ),
        }
        _, metadata, decision = self.derive("public_domain_books_gutenberg", row)
        self.assertTrue(decision.keep, decision.reason)
        self.assertEqual(metadata["license"], "Public Domain")
        self.assertTrue(metadata["chapter_integrity_passed"])
        self.assertIn("upstream_metadata", metadata)
        self.assertTrue(
            any(
                item["method"] == "computed_longform_chapter_integrity_v1"
                for item in metadata["normalization_evidence"]
            )
        )

        row["metadata"] = {"language": "en", "url": "https://www.gutenberg.org/ebooks/1"}
        _, metadata, decision = self.derive("public_domain_books_gutenberg", row)
        self.assertNotIn("license", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_license")

        row = {
            "id": "gutenberg-complete-work",
            "text": (
                "The Project Gutenberg Etext of a complete public-domain work.\n\n"
                + self.prose(55, prefix="Article paragraph")
                + "\n\n***"
            ),
            "metadata": {
                "language": "en",
                "license": "Public Domain",
                "title": "A Complete Work Without Chapters",
                "url": "https://www.gutenberg.org/ebooks/2",
            },
        }
        _, metadata, decision = self.derive("public_domain_books_gutenberg", row)
        self.assertTrue(metadata["chapter_integrity_passed"])
        self.assertEqual(
            metadata["chapter_integrity_evidence"]["nonchapter_complete_work"],
            "gutenberg_header_and_terminal_marker",
        )
        self.assertTrue(decision.keep, decision.reason)

    def test_pdf_evidence_is_computed_but_non_english_detector_wins(self) -> None:
        text = self.prose(45)
        valid = {
            "id": "pdf-1",
            "text": text,
            "language": "eng_Latn",
            "full_doc_lid": "eng_Latn",
            "full_doc_lid_score": 0.995,
            "page_ends": [len(text)],
            "url": "https://example.edu/manual.pdf",
        }
        _, metadata, decision = self.derive(
            "finepdfs_edu_english",
            valid,
            repo_path="data/eng_Latn/0000.parquet",
        )
        self.assertTrue(decision.keep, decision.reason)
        self.assertGreaterEqual(metadata["ocr_confidence"], 0.90)
        self.assertTrue(metadata["reading_order_passed"])
        self.assertLessEqual(metadata["repeated_header_footer_fraction"], 0.08)

        mislabeled = dict(valid)
        mislabeled["full_doc_lid"] = "bul_Cyrl"
        mislabeled["full_doc_lid_score"] = 0.99
        _, metadata, decision = self.derive(
            "finepdfs_edu_english",
            mislabeled,
            repo_path="data/eng_Latn/0001.parquet",
        )
        self.assertAlmostEqual(metadata["language_probability"], 0.01)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "language_probability_minimum")

    def test_open_biomedical_paper_requires_explicit_reusable_license(self) -> None:
        text = (
            "# A Carefully Licensed Biomedical Study\n\n"
            + self.prose(18)
            + "\n\nReferences\n"
            + "\n".join(f"[{index}] A relevant study (2020)." for index in range(1, 8))
        )
        row = {
            "id": "PMC1",
            "text": text,
            "metadata": {
                "title": "A Carefully Licensed Biomedical Study",
                "license": (
                    "Creative Commons - Attribution - "
                    "https://creativecommons.org/licenses/by/4.0/"
                ),
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/",
                "provenance": "licensed_pubmed-0000.json.gz:1",
            },
        }
        _, metadata, decision = self.derive("pmc_open_access", row)
        self.assertTrue(decision.keep, decision.reason)
        self.assertTrue(metadata["open_access"])
        self.assertTrue(metadata["title_or_abstract"])

        row["metadata"]["license"] = (
            "Creative Commons Attribution-NonCommercial "
            "https://creativecommons.org/licenses/by-nc/4.0/"
        )
        _, metadata, decision = self.derive("pmc_open_access", row)
        self.assertNotIn("open_access", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_open_access")

    def test_openstax_uses_the_individual_books_retained_license_statement(self) -> None:
        text = (
            "Biology\n\n"
            "Textbook content produced by OpenStax is licensed under a Creative Commons "
            "Attribution 4.0 International License (CC BY 4.0).\n\n"
            + "\n\n".join(
                f"Chapter {number}\n{self.prose(45, prefix=f'Chapter {number} section')}"
                for number in range(1, 6)
            )
        )
        row = {"id": "biology", "text": text}
        _, metadata, decision = self.derive(
            "openstax",
            row,
            repo_path="data/Biology2e-WEB.txt",
        )
        self.assertEqual(metadata["license"], "CC-BY-4.0")
        self.assertTrue(metadata["structurally_complete"])
        self.assertTrue(decision.keep, decision.reason)
        self.assertEqual(self.license_method(metadata), "openstax_in_book_license_statement_v1")

        row["text"] = row["text"].replace(
            "Textbook content produced by OpenStax is licensed under a Creative Commons "
            "Attribution 4.0 International License (CC BY 4.0).\n\n",
            "",
        )
        _, metadata, decision = self.derive(
            "openstax",
            row,
            repo_path="data/Biology2e-WEB.txt",
        )
        # openstax is now corpus-licensed CC-BY-4.0, so a book with no
        # attribution page is still licensed -- 7 of 76 lack the page, and a
        # zero-yield file fails its whole normalization task. What must not be
        # lost is the distinction: where the book states its own licence that
        # is per-book evidence, and where it does not the value comes from the
        # manifest. Recording both under one method would erase that.
        self.assertEqual(metadata["license"], "CC-BY-4.0")
        self.assertEqual(self.license_method(metadata), "pinned_source_manifest_license_v1")

    @staticmethod
    def license_method(metadata: dict) -> str | None:
        for entry in metadata.get("normalization_evidence", []):
            if entry.get("field") == "license":
                return entry.get("method")
        return None

    @staticmethod
    def math_row() -> dict:
        return {
            "id": "math-1",
            "text": (
                "A theorem describes the sum of the first terms in a sequence. "
                "We use the equation $S_n = n(n+1)/2$ and then prove the result "
                "by induction. Therefore the formula is valid for every positive integer.\n\n"
                + NormalizationEvidenceTests.prose(8)
            ),
            "language": "en",
            "language_score": 0.99,
            "url": "https://example.org/math/1",
            "metadata": {"license": "Public Domain"},
        }

    def test_equation_integrity_is_measured(self) -> None:
        _, metadata, decision = self.derive("openwebmath_unique", self.math_row())
        self.assertTrue(decision.keep, decision.reason)
        self.assertTrue(metadata["equation_integrity_passed"])
        self.assertEqual(metadata["math_score"], 3.0)

    def test_missing_per_record_license_stays_closed(self) -> None:
        # Pinned to a source that is still per_record_required. openwebmath is
        # no longer one: it is corpus-licensed ODC-By-1.0, because a Common
        # Crawl derivative has no per-document license to carry. Asserting
        # fail-closed through a source that cannot fail closed would leave the
        # guard untested while still passing.
        row = self.math_row()
        row["metadata"] = {}
        _, metadata, decision = self.derive("proof_pile2_math", row)
        self.assertEqual(
            self.sources["proof_pile2_math"]["license"]["status"], "per_record_required"
        )
        self.assertNotIn("license", metadata)
        # The stage-level per-record-license guard runs immediately before the
        # profile gate; a profile-only decision must not manufacture a license.
        self.assertFalse(decision.keep and bool(metadata.get("license")))

    def test_synthetic_generator_name_is_genealogy_not_verification(self) -> None:
        source_id = "nemotron_specialized_fact_seeking"
        actual_shape = {
            "uuid": "row-1",
            "text": (
                "Which result follows from the source material? The answer is the result "
                "supported by the document. " + self.prose(4)
            ),
            "license": "cc-by-4.0",
            "metadata": {
                "category": "Nemotron-Pretraining-Fact-Seeking",
                "models_used": "Qwen3-30B-A3B-Instruct-2507",
            },
        }
        _, metadata, decision = self.derive(source_id, actual_shape)
        self.assertEqual(
            metadata["genealogy"]["generator_models"],
            ["Qwen3-30B-A3B-Instruct-2507"],
        )
        self.assertNotIn("source_document_id", metadata)
        self.assertNotIn("verification_passed", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_source_document_id")

        verified = dict(actual_shape)
        verified["source_document_id"] = "wiki:Q123"
        verified["verification_passed"] = True
        _, _, decision = self.derive(source_id, verified)
        self.assertTrue(decision.keep, decision.reason)

    def test_reasoning_partition_does_not_imply_programmatic_verification(self) -> None:
        row = {
            "uuid": "reasoning-1",
            "text": "Question: Explain the result.\n\n" + self.prose(6),
            "license": "cc-by-4.0",
            "metadata": {
                "category": "Nemotron-Pretraining-RQA",
                "models_used": "Qwen3-235B-A22B-Thinking-2507",
            },
        }
        _, metadata, decision = self.derive("nemotron_rqa_verified_reasoning", row)
        self.assertIn("genealogy", metadata)
        self.assertNotIn("verification_passed", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_verification_passed")

    def test_government_record_date_is_auditable_version_evidence(self) -> None:
        row = {
            "id": "DOT-OST-1995-557-0008",
            "text": self.prose(8),
            "posted_date": "2000-05-31T04:00:00",
            "metadata": {
                "license": "Public Domain",
                "provenance": "regulations-0000.json.gz:1",
                "url": "https://downloads.regulations.gov/DOT-OST-1995-557-0008/content.doc",
            },
        }
        _, metadata, decision = self.derive("metis_govreference_regulations", row)
        self.assertTrue(decision.keep, decision.reason)
        self.assertEqual(metadata["version"], "2000-05-31T04:00:00")
        self.assertEqual(
            next(
                item["method"]
                for item in metadata["normalization_evidence"]
                if item["field"] == "version"
            ),
            "government_record_edition_date_v1",
        )

    def test_stackexchange_sort_order_is_not_an_answer_score(self) -> None:
        row = {
            "id": "stack-1",
            "text": "How does this system work?\n\n" + self.prose(4),
            "metadata": {
                "license": (
                    "Creative Commons - Attribution Share-Alike - "
                    "https://creativecommons.org/licenses/by-sa/4.0/"
                ),
                "sort": "votes",
                "url": "https://example.stackexchange.com/questions/1",
            },
        }
        _, metadata, decision = self.derive("roots_stackexchange", row)
        self.assertNotIn("answer_score", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_answer_score")

    def test_split_proof_fields_are_rendered_and_checked(self) -> None:
        row = {
            "uuid": "proof-1",
            "problem": (
                "Prove that the sum of two even integers is even. Explain why the "
                "definition applies to each integer and to their sum."
            ),
            "formal_statement": (
                "theorem even_add (a b : Int) (ha : Even a) (hb : Even b) : "
                "Even (a + b) := by\n  rcases ha with ⟨k, rfl⟩\n  "
                "rcases hb with ⟨m, rfl⟩\n  exact ⟨k + m, by ring⟩"
            ),
            "license": "cc-by-4.0",
        }
        text, metadata, decision = self.derive("nemotron_math_proofs", row)
        self.assertIn("Problem:", text)
        self.assertIn("Formal statement:", text)
        self.assertTrue(metadata["statement_and_argument"])
        self.assertTrue(decision.keep, decision.reason)

    @staticmethod
    def freeze_opt_out(root: Path, state: StateStore) -> tuple[Path, str]:
        payload = (
            "Publisher/Requester,Date of notice,List of domains/URLs\n"
            "Publisher,2026-07-20,blocked.example\n"
            "Publisher,2026-07-21,allowed.example/private\n"
        ).encode("utf-8")
        snapshot = root / "contamination" / "common-crawl-opt-out" / "final.csv"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        state.write(
            "ACQUISITION_READY.json",
            payload={
                "common_crawl_opt_out": {
                    "normalization_reapplication_required": True,
                    "artifacts": {
                        "snapshot": {
                            "path": str(snapshot.relative_to(root)),
                            "size": len(payload),
                            "sha256": digest,
                        }
                    },
                }
            },
        )
        return snapshot, digest

    def test_final_common_crawl_opt_out_snapshot_is_verified_and_checks_both_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            snapshot, digest = self.freeze_opt_out(root, state)
            policy = load_frozen_common_crawl_opt_out(root, state)
            self.assertEqual(policy.snapshot_sha256, digest)
            reason, matched = final_common_crawl_opt_out_reason(
                {
                    "text": self.prose(4),
                    "url": "https://safe.example/article",
                    "metadata": {
                        "canonical_url": "https://sub.blocked.example/canonical"
                    },
                },
                policy,
            )
            self.assertEqual(reason, "final_common_crawl_opt_out_domain")
            self.assertEqual(matched, "https://sub.blocked.example/canonical")

            snapshot.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "size changed|SHA-256"):
                load_frozen_common_crawl_opt_out(root, state)

    def test_normalize_task_reapplies_final_common_crawl_opt_out_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            _, digest = self.freeze_opt_out(root, state)
            input_path = root / "raw" / "fresh.jsonl"
            input_path.parent.mkdir()
            rows = [
                {
                    "id": "blocked",
                    "text": self.prose(8, prefix="Blocked document section"),
                    "metadata": {
                        "url": "https://blocked.example/article",
                        "canonical_url": "https://blocked.example/article",
                        "capture_date": "2026-06-01",
                        "license": "Public Domain",
                    },
                },
                {
                    "id": "allowed",
                    "text": self.prose(8, prefix="Allowed document section"),
                    "metadata": {
                        "url": "https://news.allowed.example/article",
                        "canonical_url": "https://news.allowed.example/article",
                        "capture_date": "2026-06-01",
                        "license": "Public Domain",
                    },
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            state.write(
                "build.inputs.json",
                payload={
                    "schema": "metis.build-inputs/v1",
                    "input_count": 1,
                    "inputs": [
                        {
                            # hf_snapshot on purpose: this source is a packaged
                            # Common Crawl extraction, not a live crawl, and the
                            # final opt-out re-check has to reach it anyway. It
                            # used to be gated on the driver, which would exempt
                            # exactly this shape.
                            "source_id": "metis_freshweb_2025",
                            "driver": "hf_snapshot",
                            "local_path": str(input_path),
                            "repo_path": "fresh-00000.jsonl",
                            "revision": "fixture",
                        }
                    ],
                },
            )
            profile = {
                "manifest": str(ROOT / "manifests" / "metis-1.6.yaml"),
                "storage": {
                    "lustre_root": str(root),
                    "directories": {
                        "state": "state",
                        "normalized": "normalized",
                    },
                },
                "gates": {"fail_closed": True},
            }
            report = stage_runner._normalize_task(profile, 0)
            self.assertEqual(report["counts"]["input"], 2)
            self.assertEqual(report["counts"]["accepted"], 1)
            self.assertEqual(report["counts"]["rejected"], 1)
            self.assertEqual(
                report["rejection_reasons"],
                {"final_common_crawl_opt_out_domain": 1},
            )
            self.assertEqual(
                report["common_crawl_opt_out"],
                {"reapplied": True, "snapshot_sha256": digest},
            )
            output = Path(report["output"])
            with output.open("rb") as raw:
                with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                    records = [
                        json.loads(line)
                        for line in io.TextIOWrapper(reader, encoding="utf-8")
                        if line.strip()
                    ]
            self.assertEqual([record["id"] for record in records], ["allowed"])
            self.assertTrue(
                records[0]["metadata"]["final_common_crawl_opt_out_reapplied"]
            )
            self.assertEqual(
                records[0]["metadata"]["final_common_crawl_opt_out_snapshot_sha256"],
                digest,
            )

    def test_normalize_task_fails_closed_when_every_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            input_path = root / "raw" / "low-quality.jsonl"
            input_path.parent.mkdir()
            input_path.write_text(
                json.dumps({"id": "tiny", "text": "too short"}) + "\n",
                encoding="utf-8",
            )
            state.write(
                "build.inputs.json",
                payload={
                    "schema": "metis.build-inputs/v1",
                    "input_count": 1,
                    "inputs": [
                        {
                            "source_id": "fineweb_edu",
                            "driver": "hf_snapshot",
                            "local_path": str(input_path),
                            "repo_path": "low-quality.jsonl",
                            "revision": "fixture",
                        }
                    ],
                },
            )
            profile = {
                "manifest": str(ROOT / "manifests" / "metis-1.6.yaml"),
                "storage": {
                    "lustre_root": str(root),
                    "directories": {"state": "state", "normalized": "normalized"},
                },
                "gates": {"fail_closed": True},
            }
            with self.assertRaisesRegex(RuntimeError, "accepted zero records"):
                stage_runner._normalize_task(profile, 0)
            self.assertFalse(state.is_complete("normalize", "task-000000"))


if __name__ == "__main__":
    unittest.main()


class StackDerivedLicenseTests(unittest.TestCase):
    def test_proof_pile_2_per_record_license_is_read_from_meta(self) -> None:
        """Proof-Pile-2 carries a genuine per-record license inside `meta`.

        `_find` descends into `metadata` but not `meta`, so the whole source
        normalized to zero accepted records under `per_record_required` even
        though every row names its repository licence.
        """

        row = {
            "text": "open import Web.Semantic.DL.Role\n",
            "meta": {
                "hexsha": "3dcbe7dd3386a3c21b79cceb2d381b1a16a4f075",
                "ext": "agda",
                "max_stars_repo_name": "agda/agda-web-semantic",
                "max_stars_repo_licenses": ["MIT"],
            },
        }
        source = {
            "id": "proof_pile2_math",
            "category": "math",
            "license": {"status": "per_record_required", "expression": "per-component"},
            "provenance": {},
            "processing": {"quality_profile": "math_score3_v1"},
        }
        evidence = derive_normalization_evidence(row, source, {}, row["text"])
        self.assertEqual(evidence.get("license"), "MIT")
