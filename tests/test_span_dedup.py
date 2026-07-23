from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from datatrove.data import Document

from metis_data.span_dedup import (
    build_span_dedup_filter,
    find_repeated_span_candidates,
    find_span_duplicates,
    iter_span_removals,
    iter_span_signatures,
    stable_document_tie,
    write_span_prefilter_signatures,
    write_span_signatures,
)


BOILERPLATE = (
    "Our editorial team reviews every technical guide against the currently published "
    "documentation before release. "
    "Each article includes concrete examples, explicit version information, and links "
    "to the canonical primary sources. "
    "Please report any outdated statement so maintainers can investigate it and publish "
    "a clearly dated correction."
)


def _substantive(prefix: str) -> str:
    return (
        f"{prefix} introduces a carefully scoped subject with enough unique language to "
        "identify this particular source document. "
        f"The opening analysis for {prefix} describes practical constraints, records the relevant "
        "assumptions, and distinguishes observed facts from provisional conclusions. "
        f"{BOILERPLATE} "
        f"After the repeated notice, {prefix} develops a separate implementation example "
        "using measurements gathered from the local system. "
        f"The closing discussion for {prefix} compares failure modes, explains verification checks, and "
        "lists several follow-up questions for future maintainers. "
        f"A final appendix for {prefix} preserves commands, dates, and provenance details needed to "
        "reproduce the result without relying on undocumented context."
    )


class CaptureWriter:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def __enter__(self) -> "CaptureWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def write(self, document: Document, rank: int) -> None:
        snapshot = copy.copy(document)
        snapshot.metadata = dict(document.metadata)
        self.documents.append(snapshot)


def _run_finders(
    signatures: Path,
    removals: Path,
    *,
    finder_workers: int,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
) -> None:
    for bucket in range(finder_workers):
        find_span_duplicates(
            signatures,
            removals,
            bucket=bucket,
            finder_workers=finder_workers,
            sentence_count=sentence_count,
            minimum_span_words=minimum_span_words,
            sqlite_cache_mb=4,
        )


class SpanDedupTests(unittest.TestCase):
    def test_two_pass_prefilter_materializes_only_cross_document_spans(self) -> None:
        repeated_in_one_document = Document(
            text=(
                f"{BOILERPLATE} "
                "A unique bridge explains why this copy remains inside one source document. "
                f"{BOILERPLATE}"
            ),
            id="one-document",
            metadata={"priority": 1},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            candidates = root / "candidates"
            signatures = root / "signatures"
            write_span_prefilter_signatures(
                [repeated_in_one_document],
                compact,
                rank=0,
                finder_workers=2,
            )
            reports = [
                find_repeated_span_candidates(
                    compact,
                    candidates,
                    bucket=bucket,
                    finder_workers=2,
                    total_ranks=1,
                    chunk_records=1,
                    maximum_open_runs=2,
                )
                for bucket in range(2)
            ]
            self.assertEqual(sum(report["candidate_records"] for report in reports), 0)
            full = write_span_signatures(
                [repeated_in_one_document],
                signatures,
                rank=0,
                finder_workers=2,
                candidate_root=candidates,
                total_ranks=1,
            )
            self.assertEqual(full["signatures"], 0)
            self.assertTrue(all(not output["present"] for output in full["outputs"].values()))

    def test_two_pass_external_sort_preserves_priority_and_complete_manifests(self) -> None:
        low = Document(
            text=_substantive("Lower priority material"),
            id="low",
            metadata={"priority": 10},
        )
        high = Document(
            text=_substantive("Premium primary material"),
            id="high",
            metadata={"priority": 100},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            candidates = root / "candidates"
            signatures = root / "signatures"
            removals = root / "removals"
            write_span_prefilter_signatures([low], compact, rank=0, finder_workers=1)
            write_span_prefilter_signatures([high], compact, rank=1, finder_workers=1)
            candidate_report = find_repeated_span_candidates(
                compact,
                candidates,
                bucket=0,
                finder_workers=1,
                total_ranks=2,
                chunk_records=1,
                maximum_open_runs=2,
            )
            self.assertEqual(candidate_report["repeated_digests"], 1)
            self.assertEqual(candidate_report["candidate_records"], 2)
            self.assertGreaterEqual(candidate_report["compact_sort"]["merge_passes"], 2)
            self.assertEqual(set(candidate_report["outputs"]), {"000000", "000001"})

            low_report = write_span_signatures(
                [low],
                signatures,
                rank=0,
                finder_workers=1,
                candidate_root=candidates,
                total_ranks=2,
            )
            high_report = write_span_signatures(
                [high],
                signatures,
                rank=1,
                finder_workers=1,
                candidate_root=candidates,
                total_ranks=2,
            )
            self.assertEqual(low_report["signatures"], 1)
            self.assertEqual(high_report["signatures"], 1)
            finder_report = find_span_duplicates(
                signatures,
                removals,
                bucket=0,
                finder_workers=1,
                total_ranks=2,
                chunk_records=1,
                maximum_open_runs=2,
                sqlite_cache_mb=4,
            )
            self.assertEqual(finder_report["duplicate_groups"], 1)
            self.assertEqual(finder_report["removal_starts"], 1)
            self.assertGreaterEqual(finder_report["signature_sort"]["merge_passes"], 1)
            self.assertEqual(set(finder_report["outputs"]), {"000000", "000001"})
            self.assertEqual(
                [start for _, start, _ in iter_span_removals(
                    removals, rank=0, finder_workers=1
                )],
                [2],
            )
            self.assertEqual(
                list(iter_span_removals(removals, rank=1, finder_workers=1)),
                [],
            )
            self.assertEqual(list(root.rglob("*.sqlite3")), [])

    def test_empty_outputs_are_explicit_and_missing_rank_fails_closed(self) -> None:
        short = Document(
            text="Terms apply. Learn more. Contact us.",
            id="short",
            metadata={"priority": 1},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            candidates = root / "candidates"
            report = write_span_prefilter_signatures(
                [short],
                compact,
                rank=0,
                finder_workers=3,
            )
            self.assertEqual(set(report["outputs"]), {"0000", "0001", "0002"})
            self.assertTrue(all(not output["present"] for output in report["outputs"].values()))
            with self.assertRaisesRegex(RuntimeError, "Incomplete.*inventory"):
                find_repeated_span_candidates(
                    compact,
                    candidates,
                    bucket=0,
                    finder_workers=3,
                    total_ranks=2,
                )

            for bucket in range(3):
                empty_report = find_repeated_span_candidates(
                    compact,
                    candidates,
                    bucket=bucket,
                    finder_workers=3,
                    total_ranks=1,
                    chunk_records=1,
                    maximum_open_runs=2,
                )
                self.assertEqual(empty_report["candidate_records"], 0)
                self.assertEqual(set(empty_report["outputs"]), {"000000"})
                self.assertFalse(empty_report["outputs"]["000000"]["present"])
            on_disk = json.loads(
                (candidates / "_manifests" / "0002.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk["outputs"]["000000"]["records"], 0)

    def test_prefilter_hash_corruption_is_rejected_during_global_sort(self) -> None:
        document = Document(
            text=_substantive("Hash validation source"),
            id="hash-source",
            metadata={"priority": 1},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            write_span_prefilter_signatures([document], compact, rank=0, finder_workers=1)
            path = compact / "0000" / "000000.compact"
            with path.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0xFF]))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                find_repeated_span_candidates(
                    compact,
                    root / "candidates",
                    bucket=0,
                    finder_workers=1,
                    total_ranks=1,
                    chunk_records=1,
                    maximum_open_runs=2,
                )

    def test_higher_priority_document_keeps_shared_span(self) -> None:
        low = Document(
            text=_substantive("Lower priority material"),
            id="low",
            metadata={"priority": 10},
        )
        high = Document(
            text=_substantive("Premium primary material"),
            id="high",
            metadata={"priority": 100},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signatures = root / "signatures"
            removals = root / "removals"
            write_span_signatures([low], signatures, rank=0, finder_workers=4)
            write_span_signatures([high], signatures, rank=1, finder_workers=4)
            _run_finders(signatures, removals, finder_workers=4)

            low_removals = list(iter_span_removals(removals, rank=0, finder_workers=4))
            high_removals = list(iter_span_removals(removals, rank=1, finder_workers=4))
            self.assertEqual([start for _, start, _ in low_removals], [2])
            self.assertEqual(high_removals, [])

    def test_equal_priority_tie_is_stable_across_input_rank_order(self) -> None:
        first = Document(text=_substantive("First source"), id="alpha-source", metadata={"priority": 7})
        second = Document(text=_substantive("Second source"), id="omega-source", metadata={"priority": 7})
        expected_winner = min((first, second), key=lambda document: stable_document_tie(document.id)).id

        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ordered = [second, first] if reverse else [first, second]
                signatures = root / "signatures"
                removals = root / "removals"
                write_span_signatures([ordered[0]], signatures, rank=0, finder_workers=3)
                write_span_signatures([ordered[1]], signatures, rank=1, finder_workers=3)
                _run_finders(signatures, removals, finder_workers=3)
                removed_ranks = {
                    rank
                    for rank in range(2)
                    if list(iter_span_removals(removals, rank=rank, finder_workers=3))
                }
                winning_rank = ({0, 1} - removed_ranks).pop()
                self.assertEqual(ordered[winning_rank].id, expected_winner)

    def test_filter_strips_boilerplate_and_quarantines_original(self) -> None:
        low = Document(
            text=_substantive("Lower priority material"),
            id="low",
            metadata={"priority": 10, "source_id": "low-source"},
        )
        high = Document(
            text=_substantive("Premium primary material"),
            id="high",
            metadata={"priority": 100, "source_id": "high-source"},
        )
        original_low = low.text
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signatures = root / "signatures"
            removals = root / "removals"
            write_span_signatures([low], signatures, rank=0, finder_workers=4)
            write_span_signatures([high], signatures, rank=1, finder_workers=4)
            _run_finders(signatures, removals, finder_workers=4)
            quarantine = CaptureWriter()
            stage = build_span_dedup_filter(
                removals,
                finder_workers=4,
                quarantine_writer=quarantine,
            )
            filtered = list(stage.run([low], rank=0, world_size=2))

            self.assertEqual(len(filtered), 1)
            self.assertNotIn("Our editorial team reviews", filtered[0].text)
            self.assertIn("Lower priority material", filtered[0].text)
            self.assertIn("After the repeated notice", filtered[0].text)
            self.assertEqual(filtered[0].metadata["span_dedup_action"], "modified")
            self.assertEqual(len(quarantine.documents), 1)
            self.assertEqual(quarantine.documents[0].text, original_low)
            self.assertEqual(quarantine.documents[0].metadata["span_dedup_action"], "modified")

    def test_changed_document_below_minimum_is_dropped_and_quarantined(self) -> None:
        low = Document(
            text=f"Short unique introduction for one source. {BOILERPLATE}",
            id="low",
            metadata={"priority": 1},
        )
        high = Document(
            text=f"Different premium introduction for another source. {BOILERPLATE}",
            id="high",
            metadata={"priority": 9},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signatures = root / "signatures"
            removals = root / "removals"
            write_span_signatures([low], signatures, rank=0, finder_workers=2)
            write_span_signatures([high], signatures, rank=1, finder_workers=2)
            _run_finders(signatures, removals, finder_workers=2)
            quarantine = CaptureWriter()
            stage = build_span_dedup_filter(
                removals,
                finder_workers=2,
                quarantine_writer=quarantine,
            )
            self.assertEqual(list(stage.run([low], rank=0, world_size=2)), [])
            self.assertEqual(len(quarantine.documents), 1)
            self.assertEqual(
                quarantine.documents[0].metadata["filter_reason"],
                "repeated_span_below_minimum",
            )

    def test_short_common_three_sentence_text_is_not_signed_or_removed(self) -> None:
        text = "Terms apply. Learn more. Contact us."
        self.assertEqual(list(iter_span_signatures(text)), [])
        left = Document(text=text, id="left", metadata={"priority": 1})
        right = Document(text=text, id="right", metadata={"priority": 2})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signatures = root / "signatures"
            removals = root / "removals"
            write_span_signatures([left], signatures, rank=0, finder_workers=2)
            write_span_signatures([right], signatures, rank=1, finder_workers=2)
            _run_finders(signatures, removals, finder_workers=2)
            self.assertEqual(
                list(iter_span_removals(removals, rank=0, finder_workers=2)),
                [],
            )
            self.assertEqual(
                list(iter_span_removals(removals, rank=1, finder_workers=2)),
                [],
            )


if __name__ == "__main__":
    unittest.main()
