from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metis_data.stage_runner import (
    _completion_inventory,
    _datatrove_task_counts,
    _stat_total,
)
from metis_data.state import StateStore


REAL_STEPS = [
    {
        "name": "READER: Jsonl",
        "stats": {
            "doc_len": {"total": 21_000_044_756},
            "documents": {"total": 779_046},
        },
    },
    {
        "name": "DECONT: benchmark decontamination",
        "stats": {
            "total": 779_046,
            "dropped_benchmark_short_ngram": 6_062,
            "dropped_benchmark_contiguous_run": 2_241,
        },
    },
    {
        "name": "WRITER: Jsonl",
        "stats": {
            "total": 766_536,
            "doc_len": {"total": 20_614_959_793},
        },
    },
]


def _write(root: Path, rank: int, steps: list) -> Path:
    stats = root / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / f"{rank:05d}.json").write_text(
        json.dumps(steps),
        encoding="utf-8",
    )
    return root


class DatatroveTaskCountsTests(unittest.TestCase):
    def test_reads_the_real_filter_stats_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write(Path(tmp), 2_034, REAL_STEPS)
            counts = _datatrove_task_counts(
                root,
                2_034,
                stage="decontam_filter",
            )
        self.assertEqual(counts["accounting_status"], "complete")
        self.assertEqual(counts["records_in"], 779_046)
        self.assertEqual(counts["records_out"], 766_536)
        self.assertEqual(counts["records_removed"], 12_510)
        self.assertEqual(counts["characters_removed"], 385_084_963)
        self.assertEqual(
            counts["removed_by_reason"],
            {
                "dropped_benchmark_contiguous_run": 2_241,
                "dropped_benchmark_short_ngram": 6_062,
            },
        )

    def test_all_filtered_output_is_recorded_as_total_loss(self) -> None:
        steps = [
            {
                "name": "READER",
                "stats": {
                    "documents": {"total": 500},
                    "doc_len": {"total": 1_000},
                },
            },
            {
                "name": "FILTER",
                "stats": {"dropped_duplicate": 500},
            },
            {
                "name": "WRITER",
                "stats": {
                    "total": 0,
                    "doc_len": {"total": 0},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            counts = _datatrove_task_counts(
                _write(Path(tmp), 0, steps),
                0,
                stage="exact_filter",
            )
        self.assertEqual(counts["records_removed"], 500)
        self.assertEqual(counts["characters_removed"], 1_000)

    def test_signature_stage_never_claims_document_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            counts = _datatrove_task_counts(
                _write(Path(tmp), 0, REAL_STEPS[:-1]),
                0,
                stage="exact_signature",
            )
        self.assertEqual(counts, {})

    def test_missing_and_unreadable_stats_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = _datatrove_task_counts(
                Path(tmp),
                0,
                stage="exact_filter",
            )
            stats = Path(tmp) / "stats"
            stats.mkdir()
            (stats / "00001.json").write_text("{not json", encoding="utf-8")
            unreadable = _datatrove_task_counts(
                Path(tmp),
                1,
                stage="exact_filter",
            )
        self.assertEqual(missing["accounting_status"], "missing")
        self.assertEqual(unreadable["accounting_status"], "unreadable")

    def test_filter_without_writer_is_marked_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            counts = _datatrove_task_counts(
                _write(Path(tmp), 0, REAL_STEPS[:-1]),
                0,
                stage="exact_filter",
            )
        self.assertEqual(counts["accounting_status"], "incomplete")
        self.assertTrue(counts["reader_found"])
        self.assertFalse(counts["writer_found"])
        self.assertNotIn("records_removed", counts)

    def test_missing_required_counters_are_not_reported_as_zero(self) -> None:
        steps = [
            {"name": "READER", "stats": {}},
            {"name": "WRITER", "stats": {}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            counts = _datatrove_task_counts(
                _write(Path(tmp), 0, steps),
                0,
                stage="exact_filter",
            )
        self.assertEqual(counts["accounting_status"], "invalid")
        self.assertEqual(
            counts["missing_or_malformed_counters"],
            ["characters_in", "characters_out", "records_in", "records_out"],
        )

    def test_inconsistent_stats_do_not_report_negative_removal(self) -> None:
        steps = [
            {
                "name": "READER",
                "stats": {
                    "documents": {"total": 5},
                    "doc_len": {"total": 50},
                },
            },
            {
                "name": "WRITER",
                "stats": {
                    "total": 6,
                    "doc_len": {"total": 60},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            counts = _datatrove_task_counts(
                _write(Path(tmp), 0, steps),
                0,
                stage="exact_filter",
            )
        self.assertEqual(counts["accounting_status"], "inconsistent")
        self.assertNotIn("records_removed", counts)

    def test_filter_chain_backfills_legacy_completion_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = StateStore(root / "state")
            contract = "a" * 64
            state.complete(
                "exact_filter",
                "task-000000",
                {
                    "stage": "exact_filter",
                    "task_index": 0,
                    "execution_contract_sha256": contract,
                },
            )
            logs = root / "logs"
            _write(
                logs / "exact_filter" / contract[:24],
                0,
                REAL_STEPS,
            )
            receipt = _completion_inventory(
                state,
                "exact_filter",
                1,
                expected_execution_contract_sha256=contract,
                accounting_logs_root=logs,
            )
            marker = state.read(
                "completed",
                "exact_filter",
                "task-000000.json",
            )
        self.assertEqual(
            marker["counts"]["accounting_status"],
            "complete",
        )
        self.assertEqual(receipt["accounting"]["records_removed"], 12_510)

    def test_filter_chain_rejects_unrecoverable_legacy_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = StateStore(root / "state")
            contract = "b" * 64
            state.complete(
                "exact_filter",
                "task-000000",
                {
                    "stage": "exact_filter",
                    "task_index": 0,
                    "execution_contract_sha256": contract,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "accounting is missing"):
                _completion_inventory(
                    state,
                    "exact_filter",
                    1,
                    expected_execution_contract_sha256=contract,
                    accounting_logs_root=root / "logs",
                )

    def test_filter_chain_rejects_contradictory_complete_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = StateStore(root / "state")
            contract = "c" * 64
            state.complete(
                "exact_filter",
                "task-000000",
                {
                    "stage": "exact_filter",
                    "task_index": 0,
                    "execution_contract_sha256": contract,
                    "counts": {
                        "accounting_status": "complete",
                        "records_in": 100,
                        "records_out": 90,
                        "records_removed": 0,
                        "characters_in": 1_000,
                        "characters_out": 900,
                        "characters_removed": 999,
                    },
                },
            )
            with self.assertRaisesRegex(RuntimeError, "counters disagree"):
                _completion_inventory(
                    state,
                    "exact_filter",
                    1,
                    expected_execution_contract_sha256=contract,
                    accounting_logs_root=root / "logs",
                )

    def test_stat_total_accepts_both_encodings(self) -> None:
        self.assertEqual(_stat_total({"total": 7}), 7)
        self.assertEqual(_stat_total(7), 7)
        self.assertEqual(_stat_total(None), 0)
        self.assertEqual(_stat_total("x"), 0)


if __name__ == "__main__":
    unittest.main()
