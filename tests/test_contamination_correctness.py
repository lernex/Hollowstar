from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data.datatrove_blocks import (
    load_contamination_index,
    save_contamination_index,
)
from metis_data.decontaminate import ContaminationIndex
from metis_data.holdouts import prepare_holdouts
from metis_data.state import StateStore

from tests.contamination_fixtures import write_contamination_inputs


def _fragment(group: str, identifier: str, text: str) -> dict:
    return {
        "id": identifier,
        "text": text,
        "metadata": {"holdout_row_id": group},
    }


class RowAwareContaminationTests(unittest.TestCase):
    def test_threshold_matches_cannot_combine_unrelated_holdout_rows(self) -> None:
        first = "alpha bravo charlie delta echo"
        second = "foxtrot golf hotel india juliet"
        candidate = f"{first} unrelated bridge words separate these fragments {second}"
        common = {
            "ngram_size": 5,
            "minimum_matching_ngrams": 2,
            "short_ngram_size": 50,
            "minimum_short_matching_ngrams": 2,
            "code_ngram_size": 50,
            "minimum_code_matching_ngrams": 2,
            "code_skeleton_ngram_size": 50,
            "minimum_code_skeleton_matching_ngrams": 2,
        }

        unrelated = ContaminationIndex.build(
            [
                _fragment("row-one", "one", first),
                _fragment("row-two", "two", second),
            ],
            **common,
        )
        self.assertIsNone(unrelated.reason(candidate))

        same_row = ContaminationIndex.build(
            [
                _fragment("shared-row", "one", first),
                _fragment("shared-row", "two", second),
            ],
            **common,
        )
        self.assertEqual(same_row.reason(candidate), "benchmark_ngram")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrelated_directory = root / "unrelated"
            unrelated_fragments = [
                _fragment("row-one", "one", first),
                _fragment("row-two", "two", second),
            ]
            write_contamination_inputs(
                unrelated_directory, unrelated, unrelated_fragments
            )
            path = unrelated_directory / "index.json"
            save_contamination_index(unrelated, path)
            disk_index = load_contamination_index(path)
            self.assertIsNone(disk_index.reason(candidate))
            same_directory = root / "same-row"
            same_fragments = [
                _fragment("shared-row", "one", first),
                _fragment("shared-row", "two", second),
            ]
            write_contamination_inputs(same_directory, same_row, same_fragments)
            same_path = same_directory / "index.json"
            save_contamination_index(same_row, same_path)
            self.assertEqual(
                load_contamination_index(same_path).reason(candidate),
                "benchmark_ngram",
            )

    def test_globally_frequent_shingles_are_suppressed(self) -> None:
        common = "shared generic benchmark phrase here"
        index = ContaminationIndex.build(
            [
                _fragment(f"row-{number}", str(number), f"{common} unique {number}")
                for number in range(3)
            ],
            ngram_size=5,
            minimum_matching_ngrams=1,
            short_ngram_size=50,
            code_ngram_size=50,
            code_skeleton_ngram_size=50,
            maximum_shingle_rows=2,
        )
        self.assertGreater(index.suppressed_shingles["ngrams"], 0)
        self.assertIsNone(index.reason(f"prefix {common} suffix"))

    def test_code_skeleton_threshold_is_also_scoped_to_one_row(self) -> None:
        first = "def alpha(x): return x"
        second = "return alpha + beta * 2 - gamma"
        renamed = (
            "def renamed(value): return value\n"
            "return left + right * 9 - offset"
        )
        common = {
            "ngram_size": 50,
            "short_ngram_size": 50,
            "code_ngram_size": 50,
            "code_skeleton_ngram_size": 8,
            "minimum_code_skeleton_matching_ngrams": 2,
        }
        unrelated = ContaminationIndex.build(
            [
                _fragment("code-row-one", "one", first),
                _fragment("code-row-two", "two", second),
            ],
            **common,
        )
        self.assertIsNone(unrelated.reason(renamed))
        same_row = ContaminationIndex.build(
            [
                _fragment("shared-code-row", "one", first),
                _fragment("shared-code-row", "two", second),
            ],
            **common,
        )
        self.assertEqual(
            same_row.reason(renamed),
            "benchmark_code_skeleton_ngram",
        )


class HoldoutJobAccountingTests(unittest.TestCase):
    def _workspace(self, temporary: str) -> tuple[Path, dict, StateStore]:
        root = Path(temporary)
        repository = root / "repository"
        registry = repository / "manifests" / "contamination" / "eval-holdouts.yaml"
        registry.parent.mkdir(parents=True)
        registry.write_text(
            "schema: metis.contamination-registry/v2\n"
            "policy: {semantic_dedup: false}\n"
            "benchmarks:\n"
            "  - id: example\n"
            "    family: reasoning\n"
            "    repo_id: example/benchmark\n"
            "    revision: 0000000000000000000000000000000000000000\n"
            "    jobs:\n"
            "      - {config: first, split: test}\n"
            "      - {config: second, split: validation}\n"
            "    use: evaluation_only\n",
            encoding="utf-8",
        )
        data_root = root / "data"
        profile = {
            "storage": {
                "lustre_root": str(data_root),
                "directories": {
                    "contamination": "contamination",
                    "state": "state",
                },
            },
            "runtime": {"hf_home": "cache/huggingface"},
        }
        state = StateStore(data_root / "state")
        return repository, profile, state

    def test_report_accounts_for_every_expanded_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, profile, state = self._workspace(temporary)

            def rows(_entry: dict, job: dict, _cache: Path):
                yield 0, {"question": f"Question for {job['job']} with enough text"}

            with (
                mock.patch("metis_data.holdouts.repository_root", return_value=repository),
                mock.patch(
                    "metis_data.holdouts._benchmark_source_rows",
                    side_effect=rows,
                ),
            ):
                report = prepare_holdouts(profile, state)

            self.assertEqual(report["benchmark_registry_count"], 1)
            self.assertEqual(report["family_label_count"], 1)
            self.assertEqual(report["job_count"], 2)
            self.assertEqual(
                [job["job"] for job in report["jobs"]],
                ["first:test", "second:validation"],
            )
            self.assertTrue(all(job["source_rows"] == 1 for job in report["jobs"]))
            self.assertTrue(all(job["fragments"] == 1 for job in report["jobs"]))
            holdouts = (
                Path(profile["storage"]["lustre_root"])
                / profile["storage"]["directories"]["contamination"]
                / "holdouts.jsonl"
            )
            records = [
                json.loads(line)
                for line in holdouts.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(record["metadata"].get("holdout_row_id") for record in records)
            )

    def test_any_zero_row_job_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, profile, state = self._workspace(temporary)

            def rows(_entry: dict, job: dict, _cache: Path):
                if job["job"] == "first:test":
                    yield 0, {"question": "A populated benchmark question"}

            with (
                mock.patch("metis_data.holdouts.repository_root", return_value=repository),
                mock.patch(
                    "metis_data.holdouts._benchmark_source_rows",
                    side_effect=rows,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "example:second:validation produced zero source rows",
                ),
            ):
                prepare_holdouts(profile, state)


if __name__ == "__main__":
    unittest.main()
