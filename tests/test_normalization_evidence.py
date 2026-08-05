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

        # A row that states no licence of its own now falls back to the pinned
        # corpus grant, because this source is `reviewed` rather than
        # `per_record_required`. The per-row value above still wins when it is
        # present -- `_set_evidence` does not overwrite -- so real per-record
        # evidence is preserved and the manifest only fills the silence.
        row["metadata"] = {"language": "en", "url": "https://www.gutenberg.org/ebooks/1"}
        _, metadata, decision = self.derive("public_domain_books_gutenberg", row)
        self.assertEqual(
            metadata["license"],
            self.sources["public_domain_books_gutenberg"]["license"]["expression"],
        )
        self.assertTrue(
            any(
                item["field"] == "license"
                and item["method"] == "pinned_source_manifest_license_v1"
                for item in metadata["normalization_evidence"]
            )
        )
        self.assertTrue(decision.keep, decision.reason)

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

    def test_megamath_probability_score_is_rescaled_to_the_gate_scale(self) -> None:
        # MegaMath-web states math_score as a 0-1 probability while the gate is
        # on FineMath's 0-5 scale, so a top-rated row scored 1.0 against a
        # threshold of 3 and the corpus normalized to zero.
        row = {"text": self.prose(6), "math_score": 0.92, "lang": "en", "lang_score": 0.99}
        _, metadata, _ = self.derive("megamath_unique", row)
        self.assertAlmostEqual(metadata["math_score"], 4.6)

    def test_a_low_probability_megamath_row_still_fails(self) -> None:
        # 0.44 rescales to 2.2, below the threshold of 3. The rescale must not
        # become a way for the low cluster to pass.
        row = {"text": self.prose(6), "math_score": 0.44, "lang": "en", "lang_score": 0.99}
        _, metadata, decision = self.derive("megamath_unique", row)
        self.assertAlmostEqual(metadata["math_score"], 2.2)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "math_score_minimum")

    def test_other_math_sources_keep_their_own_scale(self) -> None:
        # The rescale is scoped to megamath_unique; a 0-5 score elsewhere is
        # read as given.
        row = {"text": self.prose(6), "math_score": 4.0, "lang": "en", "lang_score": 0.99}
        _, metadata, _ = self.derive("openwebmath_unique", row)
        self.assertAlmostEqual(metadata["math_score"], 4.0)

    def test_a_lean_row_without_an_extension_is_read_as_formal(self) -> None:
        # Nemotron-Math-Proofs ships lean.jsonl with no `ext` field; the row
        # declares itself with `formal_statement` and `lean_header` instead.
        row = {
            "problem": "Is a single point topologically connected?",
            "formal_statement": (
                "theorem single_point_preconnected {a : Type*} [TopologicalSpace a] "
                "(x : a) : IsPreconnected ({x} : Set a) := by simp"
            ),
            "lean_header": "import Mathlib",
            "license": "cc-by-4.0",
            "text": (
                "Problem:\nIs a single point topologically connected?\n\n"
                "Formal statement:\ntheorem single_point_preconnected {a : Type*} "
                "[TopologicalSpace a] (x : a) : IsPreconnected ({x} : Set a) := by\n"
                "  simp [IsPreconnected]\n"
            ),
        }
        _, metadata, _ = self.derive("nemotron_math_proofs", row)
        self.assertEqual(metadata["language_probability"], 1.0)
        self.assertEqual(self._language_method(metadata),
                         "natural_language_gate_not_applicable_to_code_v1")

    def test_english_prose_under_proof_v1_is_still_scored_as_prose(self) -> None:
        # The widened test must not exempt an ordinary prose proof, which has
        # no formal-statement field at all. Asserting on the value alone cannot
        # show this -- clean English prose scores exactly 1.0 on its own merits
        # -- so assert on the method that produced it.
        row = {"text": self.prose(12), "license": "cc-by-4.0"}
        _, metadata, _ = self.derive("nemotron_math_proofs", row)
        self.assertEqual(self._language_method(metadata),
                         "computed_english_text_evidence_v1")

    @staticmethod
    def _language_method(metadata: dict) -> str | None:
        for item in metadata.get("normalization_evidence", []):
            if item["field"] == "language_probability":
                return item["method"]
        return None

    def _paginated(self, pages: int = 6) -> tuple[str, list[int]]:
        """A well-ordered document that repeats a running head on every page."""

        header = "Introductory Physics for Engineers -- Chapter Four"
        body = (
            "The derivation proceeds from the conservation law stated above, and "
            "each term is carried through the substitution so that the reader can "
            "follow how the final expression is obtained from the initial one."
        )
        text = ""
        ends: list[int] = []
        for index in range(pages):
            text += f"{header}\n{body}\nPage {index + 1} of {pages}\n"
            ends.append(len(text))
        return text, ends

    def test_a_running_head_no_longer_fails_reading_order(self) -> None:
        # Reading order is about sequence. A textbook that repeats its running
        # head on every page is correctly ordered; with the old compound the
        # repeated edges alone drove `reading_order_passed` to False.
        from metis_data.normalization_evidence import _pdf_evidence

        text, ends = self._paginated()
        evidence = _pdf_evidence(text, {"page_ends": ends})

        self.assertGreater(evidence["repeated_header_footer_fraction"], 0.08)
        self.assertTrue(evidence["reading_order_passed"])

    def test_pagination_is_still_judged_once_by_the_profile(self) -> None:
        # The signal is not discarded, only moved: a document whose page edges
        # are almost entirely boilerplate still exceeds the profile bound.
        from metis_data.normalization_evidence import _pdf_evidence

        text, ends = self._paginated(pages=40)
        evidence = _pdf_evidence(text, {"page_ends": ends})
        self.assertGreater(evidence["repeated_header_footer_fraction"], 0.35)

    def test_scrambled_extraction_still_fails_reading_order(self) -> None:
        # Column bleed shows up as single-token lines and short fragments, and
        # those are what the check now rests on.
        from metis_data.normalization_evidence import _pdf_evidence

        scrambled = "\n".join(["word"] * 60)
        self.assertFalse(_pdf_evidence(scrambled, {})["reading_order_passed"])

    def test_mojibake_still_fails_reading_order(self) -> None:
        from metis_data.normalization_evidence import _pdf_evidence

        text = self.prose(20) + "�" * 400
        self.assertFalse(_pdf_evidence(text, {})["reading_order_passed"])

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
        # Chosen from the manifest rather than named, because naming it has
        # already gone stale twice: openwebmath moved to corpus-licensed
        # ODC-By-1.0 (a Common Crawl derivative has no per-document licence to
        # carry), and proof_pile2_math moved to a corpus-level permissive grant
        # after AlgebraicStack's github-* and *_proofsteps subsets turned out
        # not to carry per-row licences at all. Asserting fail-closed through a
        # source that cannot fail closed leaves the guard untested while still
        # passing, so the source is resolved at run time.
        per_record = sorted(
            source_id
            for source_id, source in self.sources.items()
            if source["license"]["status"] == "per_record_required"
        )
        if not per_record:
            # Every source is now corpus-licensed, so there is no source left
            # that can exercise the guard. The guard itself still stands in
            # stage_runner; this asserts nothing rather than asserting falsely.
            self.skipTest("no per_record_required source remains in the manifest")
        source_id = per_record[0]

        row = self.math_row()
        row["metadata"] = {}
        _, metadata, decision = self.derive(source_id, row)
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
        # The point of this test is unchanged: a generator name is lineage, not
        # a verification result, and the row carries no grounding document. It
        # used to prove that by watching the record be rejected. The rejection
        # was the wrong remedy -- fact-seeking rows never had either field, so
        # the gate discarded 15B tokens without reading one -- but the claim
        # itself must still never be manufactured, which is what is asserted
        # here now that the record is admitted on its lineage and its text.
        self.assertNotIn("source_document_id", metadata)
        self.assertNotIn("verification_passed", metadata)
        self.assertTrue(decision.keep, decision.reason)

        grounded = dict(actual_shape)
        grounded["source_document_id"] = "wiki:Q123"
        _, metadata, decision = self.derive(source_id, grounded)
        self.assertEqual(metadata["source_document_id"], "wiki:Q123")
        self.assertNotIn("verification_passed", metadata)
        self.assertTrue(decision.keep, decision.reason)

    def test_a_synthetic_row_without_a_generator_is_still_rejected(self) -> None:
        # Genealogy is the one provenance field these corpora do ship, so it
        # stays required. Dropping the unshippable fields must not turn the
        # synthetic profiles into a free pass.
        anonymous = {
            "uuid": "row-2",
            "text": "Which result follows? " + self.prose(4),
            "license": "cc-by-4.0",
            "metadata": {"category": "Nemotron-Pretraining-Fact-Seeking"},
        }
        _, metadata, decision = self.derive("nemotron_specialized_fact_seeking", anonymous)
        self.assertNotIn("genealogy", metadata)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "missing_genealogy")

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
        # The subset is named "verified"; the dataset card documents generators
        # and no verification step. Neither the partition name nor the source id
        # may become a verification_passed, whether or not the row is kept.
        self.assertNotIn("verification_passed", metadata)
        self.assertTrue(decision.keep, decision.reason)
        self.assertEqual(
            self.sources["nemotron_rqa_verified_reasoning"]["processing"]["quality_profile"],
            "synthetic_reasoning_v1",
        )

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
        # `sort: votes` is still not a score and must not become one. The
        # profile no longer asks for a score the release does not carry; it
        # asks whether the record poses a question and answers it, which is
        # what the score was standing in for and is visible in the text.
        self.assertNotIn("answer_score", metadata)
        self.assertTrue(metadata["question_and_answer"])
        self.assertTrue(decision.keep, decision.reason)

    def test_a_bare_question_with_no_answer_is_rejected(self) -> None:
        row = {
            "id": "stack-2",
            "text": "How does this system work?",
            "metadata": {
                "license": "Creative Commons - Attribution Share-Alike - "
                "https://creativecommons.org/licenses/by-sa/4.0/",
                "sort": "votes",
            },
        }
        _, metadata, decision = self.derive("roots_stackexchange", row)
        self.assertNotIn("question_and_answer", metadata)
        self.assertFalse(decision.keep)

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

    def test_normalize_task_records_zero_yield_without_stopping_the_build(self) -> None:
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
            # This used to raise. It stopped the build three times on a property
            # of one file rather than a defect -- a 13-byte empty shard, an
            # OpenStax book whose task holds exactly one document, and
            # lean_proofsteps, where proof-step records repeat lines by
            # construction (median repeated_line_fraction 0.795, so 4% pass
            # proof_v1's 0.50). A failed task fails its array element, and
            # afterok then stops all 49 downstream jobs over 12.7MB of 7.58GB.
            #
            # The fact is kept rather than the failure: the task completes, the
            # report carries zero_yield with the rejection histogram, and a
            # genuinely systematic gate error is caught where it is actually
            # visible -- by preflight-profiles before submission, and by
            # minimum_unique_tokens at select.
            payload = stage_runner._normalize_task(profile, 0)
            self.assertTrue(payload["zero_yield"])
            self.assertEqual(payload["counts"]["accepted"], 0)
            self.assertEqual(payload["rejection_reasons"], {"minimum_characters": 1})
            self.assertTrue(state.is_complete("normalize", "task-000000"))


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
