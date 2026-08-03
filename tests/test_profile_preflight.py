from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metis_data.profile_preflight import (
    evaluate_source_sample,
    format_preflight,
    run_profile_preflight,
)


def _prose(paragraphs: int = 6) -> str:
    return "\n\n".join(
        f"Section {n}. This paragraph explains a durable idea in ordinary English "
        f"prose so that the length, alphabetic fraction, and language gates all "
        f"see a realistic document rather than a synthetic stub of repeated text."
        for n in range(paragraphs)
    )


def _paired(rows: list[dict], file_record: dict | None = None):
    """Rows as the readers now yield them: paired with their file record."""

    return iter((file_record or {}, row) for row in rows)


def _source(source_id: str, *, profile: str, category: str = "web", status: str = "reviewed") -> dict:
    return {
        "id": source_id,
        "category": category,
        "license": {"status": status, "expression": "CC-BY-4.0"},
        "provenance": {},
        "processing": {"quality_profile": profile},
    }


class EvaluateSourceSampleTests(unittest.TestCase):
    def test_a_source_meeting_its_profile_is_reported_as_accepted(self) -> None:
        rows = [{"id": f"r{n}", "text": _prose(), "language": "en", "language_score": 0.99}
                for n in range(10)]
        report = evaluate_source_sample(
            _source("good", profile="web_general_v1"), _paired(rows)
        )
        self.assertEqual(report["sampled"], 10)
        self.assertEqual(report["rejections"].get("missing_quality_score", 0), 0)

    def test_missing_evidence_is_reported_by_field_not_swallowed(self) -> None:
        # The whole point: a profile demanding evidence its publisher never
        # emits must surface as a named field, before a Slurm stage discovers it.
        rows = [{"id": f"r{n}", "text": _prose(), "language": "en", "language_score": 0.99}
                for n in range(10)]
        report = evaluate_source_sample(
            _source("synthetic", profile="textbook_synthetic_v1", category="synthetic"),
            _paired(rows),
        )
        self.assertEqual(report["accepted"], 0)
        self.assertIn("missing_source_document_id", report["rejections"])

    def test_a_per_record_licence_gap_is_reported_separately(self) -> None:
        rows = [{"id": "r0", "text": _prose()}]
        report = evaluate_source_sample(
            _source("unlicensed", profile="web_general_v1", status="per_record_required"),
            _paired(rows),
        )
        self.assertEqual(report["rejections"], {"missing_license": 1})


class MultiFileSamplingTests(unittest.TestCase):
    def test_a_source_of_one_document_per_file_is_sampled_across_files(self) -> None:
        # `openstax` ships 76 textbooks as 76 single-record files. Reading only
        # the first file sampled one book and reported the source as `0/1`,
        # which reads as zero-yield rather than as a sample of one.
        from metis_data.profile_preflight import _fixture_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for n in range(5):
                path = root / f"book-{n}.jsonl"
                path.write_text(json.dumps({"id": f"b{n}", "text": _prose()}) + "\n",
                                encoding="utf-8")
                entries.append(({"local_path": str(path)}, path))

            drawn = list(_fixture_rows(entries, 60))

            self.assertEqual(len(drawn), 5)
            self.assertEqual(
                [row["id"] for _, row in drawn], ["b0", "b1", "b2", "b3", "b4"]
            )

    def test_each_row_carries_the_file_record_it_came_from(self) -> None:
        # Evidence derivation reads licence and partition facts off the file
        # record, so a row drawn from the third file must not be handed the
        # first file's record.
        from metis_data.profile_preflight import _fixture_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for n in range(3):
                path = root / f"part-{n}.jsonl"
                path.write_text(json.dumps({"id": f"r{n}", "text": _prose()}) + "\n",
                                encoding="utf-8")
                entries.append(({"marker": n}, path))

            drawn = list(_fixture_rows(entries, 60))

            self.assertEqual([record["marker"] for record, _ in drawn], [0, 1, 2])

    def test_the_row_limit_still_holds_across_files(self) -> None:
        from metis_data.profile_preflight import _fixture_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for n in range(4):
                path = root / f"part-{n}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for r in range(10):
                        handle.write(json.dumps({"id": f"{n}-{r}", "text": _prose()}) + "\n")
                entries.append(({}, path))

            self.assertEqual(len(list(_fixture_rows(entries, 25))), 25)


class RunProfilePreflightTests(unittest.TestCase):
    def _fixture(self, root: Path, rows_by_source: dict[str, list[dict]]) -> Path:
        fixture = root / "fixture"
        fixture.mkdir()
        index = {}
        for source_id, rows in rows_by_source.items():
            with (fixture / f"{source_id}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            index[source_id] = {"rows": len(rows), "bytes": 0, "file_record": {}}
        (fixture / "FIXTURE.json").write_text(json.dumps(index), encoding="utf-8")
        return fixture

    def test_zero_yield_sources_are_named_and_the_sweep_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = [{"id": f"g{n}", "text": _prose(), "language": "en", "language_score": 0.99}
                    for n in range(8)]
            fixture = self._fixture(root, {"keeps": good, "drops": good})
            manifest = {
                "release": "test-release",
                "sources": [
                    _source("keeps", profile="web_general_v1"),
                    _source("drops", profile="textbook_synthetic_v1", category="synthetic"),
                ],
            }
            payload = run_profile_preflight({}, manifest, None, rows=8, fixture=fixture)

            self.assertEqual(payload["zero_yield_sources"], ["drops"])
            # A zero-yield source fails its whole normalization task, so the
            # sweep is a gate rather than a report.
            self.assertFalse(payload["ok"])
            self.assertIn("drops", format_preflight(payload))

    def test_a_source_that_cannot_be_read_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, {"broken": [{"id": "x", "text": _prose()}]})
            (fixture / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
            manifest = {
                "release": "test-release",
                "sources": [_source("broken", profile="web_general_v1")],
            }
            payload = run_profile_preflight({}, manifest, None, rows=8, fixture=fixture)
            self.assertIn("error", payload["sources"][0])
            self.assertFalse(payload["ok"] or payload["sources"][0]["accepted"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
