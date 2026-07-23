from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tests.contamination_fixtures import write_contamination_inputs
from metis_data.datatrove_blocks import (
    CONTAMINATION_INDEX_SCHEMA,
    load_contamination_index,
    save_contamination_index,
)
from metis_data.decontaminate import ContaminationIndex


class ContaminationIndexIntegrityTests(unittest.TestCase):
    SOURCE = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen"
    )

    def _sealed_index(
        self, temporary: str
    ) -> tuple[Path, Path, ContaminationIndex]:
        root = Path(temporary)
        index = ContaminationIndex.build(
            [self.SOURCE],
            short_ngram_size=5,
            code_ngram_size=6,
        )
        registry = write_contamination_inputs(root, index, [self.SOURCE])
        path = root / "index.json"
        save_contamination_index(index, path)
        return path, registry, index

    def test_manifest_seals_every_array_and_benchmark_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _registry, _index = self._sealed_index(temporary)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], CONTAMINATION_INDEX_SCHEMA)
            self.assertEqual(payload["status"], "complete")
            self.assertRegex(payload["manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(payload["arrays_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(payload["inputs_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                set(payload["arrays"]),
                {
                    "exact",
                    "ngram_postings",
                    "short_ngram_postings",
                    "code_ngram_postings",
                    "code_skeleton_ngram_postings",
                },
            )
            for artifact in payload["array_artifacts"].values():
                self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(artifact["size"], 0)
            self.assertRegex(
                payload["inputs"]["holdouts"]["sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertRegex(
                payload["inputs"]["holdout_report"]["sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(
                payload["inputs"]["registry"]["sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertEqual(
                load_contamination_index(path).reason(self.SOURCE),
                "benchmark_exact",
            )

    def test_array_corruption_is_rejected_before_memory_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _registry, _index = self._sealed_index(temporary)
            array = path.with_suffix(".exact.npy")
            payload = bytearray(array.read_bytes())
            payload[-1] ^= 1
            array.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "array exact hash changed"):
                load_contamination_index(path)

    def test_holdout_bundle_mutation_invalidates_the_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _registry, _index = self._sealed_index(temporary)
            holdouts = path.parent / "holdouts.jsonl"
            payload = bytearray(holdouts.read_bytes())
            payload[-2] = ord(" ")
            holdouts.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "holdout bundle hash changed"):
                load_contamination_index(path)

    def test_policy_or_registry_mutation_invalidates_the_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, registry, _index = self._sealed_index(temporary)
            payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
            payload["policy"]["ngram_size"] += 1
            registry.write_text(
                yaml.safe_dump(payload, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "registry (?:size|hash) changed"
            ):
                load_contamination_index(path)

    def test_runtime_registry_must_match_the_sealed_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, registry, _index = self._sealed_index(temporary)
            runtime_registry = yaml.safe_load(registry.read_text(encoding="utf-8"))
            runtime_registry["benchmarks"][0]["id"] = "different"
            with self.assertRaisesRegex(
                RuntimeError, "Runtime benchmark registry does not match"
            ):
                load_contamination_index(
                    path,
                    benchmark_registry=runtime_registry,
                )

    def test_manifest_tampering_and_legacy_indexes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _registry, _index = self._sealed_index(temporary)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["maximum_shingle_rows"] += 1
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest failed integrity"):
                load_contamination_index(path)

            path.write_text(
                json.dumps(
                    {
                        "schema": "metis.contamination-index/v4",
                        "arrays": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "lacks sealed array and benchmark-input provenance"
            ):
                load_contamination_index(path)

    def test_saving_without_a_holdout_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = ContaminationIndex.build([self.SOURCE])
            with self.assertRaisesRegex(RuntimeError, "Holdout report is missing"):
                save_contamination_index(index, Path(temporary) / "index.json")


if __name__ == "__main__":
    unittest.main()
