from __future__ import annotations

import copy
import json
import tempfile
import unittest
import struct
from pathlib import Path

from metis_data.decontaminate import ContaminationIndex
from metis_data.dedup import deduplicate_records
from metis_data.manifest import load_manifest, matches_any, validate_manifest
from metis_data.packing import pack_release
from metis_data.quality import evaluate_quality
from metis_data.selection import build_selection, hamilton_apportion, replay_quotas, unique_quotas
from metis_data.tokenizer import train_tokenizer, validate_tokenizer
from metis_data.training_contract import phase_for_token
from metis_data.datatrove_blocks import build_priority_minhash_removals
from metis_data.source_builders import run_source_builder
from metis_data.slurm import _indices_expression


class ManifestTests(unittest.TestCase):
    def test_production_manifest_is_exact(self) -> None:
        result = validate_manifest()
        self.assertTrue(result.ok, result.errors)
        manifest = result.manifest
        self.assertEqual(manifest["schedule"]["target_tokens"], 1_000_000_000_000)
        self.assertEqual(manifest["freshness_layer"]["target_tokens"], 90_000_000_000)
        self.assertEqual(manifest["tokenizer"]["vocabulary_size_including_special_tokens"], 65_536)
        self.assertEqual(len(manifest["sources"]), 55)
        generated_or_transformed = sum(
            sum(source["phase_tokens"].values())
            for source in manifest["sources"]
            if source["provenance"].get("generated") or source["provenance"].get("transformed")
        )
        self.assertEqual(generated_or_transformed, 97_000_000_000)

    def test_phase_c_contains_no_generated_sources(self) -> None:
        manifest = load_manifest()
        offenders = [
            source["id"]
            for source in manifest["sources"]
            if source["provenance"].get("generated") and source["phase_tokens"].get("phase_c", 0)
        ]
        self.assertEqual(offenders, [])

    def test_pretraining_phase_boundaries_are_token_based(self) -> None:
        contract = Path(__file__).resolve().parents[1] / "configs" / "metis16" / "pretraining.yaml"
        self.assertEqual(phase_for_token(contract, 0), "phase_a")
        self.assertEqual(phase_for_token(contract, 699_999_999_999), "phase_a")
        self.assertEqual(phase_for_token(contract, 700_000_000_000), "phase_b")
        self.assertEqual(phase_for_token(contract, 950_000_000_000), "phase_c")

    def test_hugging_face_double_star_patterns_include_repository_root(self) -> None:
        self.assertTrue(matches_any("data.parquet", ["**/*.parquet"]))
        self.assertTrue(matches_any("nested/data.parquet", ["**/*.parquet"]))
        self.assertFalse(matches_any("data.jsonl", ["**/*.parquet"]))


class QualityAndDedupTests(unittest.TestCase):
    def test_quality_gate_is_fail_closed_and_rejects_secrets(self) -> None:
        missing = evaluate_quality(
            "A sufficiently long educational explanation " * 30,
            profile_name="web_edu_v1",
            metadata={},
        )
        self.assertFalse(missing.keep)
        self.assertEqual(missing.reason, "missing_educational_score")
        secret = evaluate_quality(
            ("A normal technical explanation with credential AKIAABCDEFGHIJKLMNOP inside. " * 20),
            profile_name="web_general_v1",
            metadata={"quality_score": 0.99, "language_probability": 0.99},
        )
        self.assertFalse(secret.keep)
        self.assertEqual(secret.reason, "secret")

    def test_exact_dedup_keeps_higher_priority(self) -> None:
        records = [
            {"doc_id": "low", "text": "The same useful explanation.", "priority": 10},
            {"doc_id": "high", "text": "  THE same useful explanation.  ", "priority": 20},
            {"doc_id": "unique", "text": "A completely different document.", "priority": 5},
        ]
        result = deduplicate_records(records)
        self.assertEqual({row["doc_id"] for row in result.kept}, {"high", "unique"})
        self.assertEqual(result.removed[0]["doc_id"], "low")

    def test_decontamination_removes_exact_and_ngram_matches(self) -> None:
        holdout = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        index = ContaminationIndex.build([holdout], ngram_size=5, minimum_matching_ngrams=2)
        self.assertEqual(index.reason(holdout), "benchmark_exact")
        self.assertEqual(index.reason("prefix one two three four five six seven suffix"), "benchmark_ngram")
        self.assertIsNone(index.reason("independent prose with no benchmark overlap at all"))

    def test_near_dedup_keeps_higher_priority(self) -> None:
        try:
            import datatrove  # noqa: F401
        except ImportError:
            self.skipTest("DataTrove is installed by the Metis-1.6 data runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            duplicates = root / "duplicates"
            removals = root / "removals"
            documents.mkdir()
            duplicates.mkdir()
            (documents / "00000.jsonl").write_text(
                json.dumps({"id": "low", "text": "similar document", "metadata": {"priority": 10}}) + "\n",
                encoding="utf-8",
            )
            (documents / "00001.jsonl").write_text(
                json.dumps({"id": "high", "text": "similar document!", "metadata": {"priority": 20}}) + "\n",
                encoding="utf-8",
            )
            (duplicates / "pairs.dups").write_bytes(struct.pack("<4I", 0, 0, 1, 0))
            report = build_priority_minhash_removals(duplicates, removals, documents, total_tasks=2)
            self.assertEqual(report["removed"], 1)
            self.assertEqual(struct.unpack("<I", (removals / "000000.remove").read_bytes())[0], 0)
            self.assertFalse((removals / "000001.remove").exists())


class AcquisitionTruthTests(unittest.TestCase):
    def test_slurm_range_is_compact_and_rejects_oversized_arrays(self) -> None:
        self.assertEqual(_indices_expression(range(1000), 200, 1000), "0-999%200")
        with self.assertRaises(ValueError):
            _indices_expression(range(1_000_000_000), 200, 1000)

    def test_repository_index_and_derived_recipes_are_not_materialized_payloads(self) -> None:
        items = (
            {
                "source_id": "repository-code",
                "driver": "repository_index",
                "access": {"components": [{"repo_id": "example/index", "revision": "0" * 40}]},
                "candidate_tokens": 100,
            },
            {
                "source_id": "derived-synthetic",
                "driver": "derived_after_download",
                "access": {"parents": ["primary"]},
                "candidate_tokens": 100,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            for item in items:
                result = run_source_builder(item, profile={}, root=Path(temporary))
                self.assertEqual(result["kind"], "remote_source_plan")
                self.assertFalse(result["materialized"])
                self.assertFalse(result["ready_for_training_build"])


class SelectionAndTokenizerTests(unittest.TestCase):
    def test_hamilton_apportion_is_exact(self) -> None:
        apportioned = hamilton_apportion(11, {"a": 5, "b": 3, "c": 2})
        self.assertEqual(sum(apportioned.values()), 11)
        self.assertGreater(apportioned["a"], apportioned["c"])

    def test_tiny_selection_hits_unique_replay_and_shard_contracts(self) -> None:
        manifest = {
            "selection": {"seed": 1, "replay": {"maximum_document_exposures": 4}},
            "schedule": {
                "target_tokens": 100,
                "phases": {
                    "phase_a": {"target_tokens": 60, "replay_tokens": 0},
                    "phase_b": {"target_tokens": 30, "replay_tokens": 10},
                    "phase_c": {"target_tokens": 10, "replay_tokens": 10},
                },
            },
            "sources": [
                {"id": "a", "phase_tokens": {"phase_a": 40, "phase_b": 20, "phase_c": 6}},
                {"id": "b", "phase_tokens": {"phase_a": 20, "phase_b": 10, "phase_c": 4}},
            ],
        }
        records = [
            {"source_id": "a", "doc_id": f"a{i}", "text": "alpha", "token_count": 10, "generated": False}
            for i in range(10)
        ] + [
            {"source_id": "b", "doc_id": f"b{i}", "text": "beta", "token_count": 10, "generated": False}
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = build_selection(
                records,
                manifest=manifest,
                eligible_tokens={"a": 100, "b": 60},
                output_root=Path(temporary),
                shard_tokens=10,
            )
            self.assertEqual(sum(item["target_tokens"] for item in result["shards"]), 100)
            self.assertEqual(len(result["shards"]), 10)
            self.assertEqual(sum(sum(value.values()) for value in result["replay_written"].values()), 20)

    def test_tokenizer_roundtrip_and_uint16_packing(self) -> None:
        corpus = [
            "ordinary English prose with punctuation.",
            "def hello(name: str) -> str:\n    return f'hello {name}'",
            r"For $x^2 + y^2 = z^2$, preserve \LaTeX{}.",
            "Unicode names: José, 李, Δx and emoji 🧪.",
        ] * 50
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = train_tokenizer(
                iter(corpus),
                output_dir=root / "tokenizer",
                vocabulary_size=512,
                special_tokens=["<|endoftext|>", "<|padding|>"],
                minimum_frequency=1,
            )
            self.assertTrue(release["uint16_safe"])
            validation = validate_tokenizer(
                root / "tokenizer" / "tokenizer.json",
                ({"category": "tiny", "text": text} for text in corpus[:4]),
            )
            self.assertTrue(validation["ok"], validation["roundtrip_failures"])
            packed = pack_release(
                [
                    {"phase": "phase_a", "source_id": "tiny", "doc_id": "1", "text": corpus[0]},
                    {"phase": "phase_b", "source_id": "tiny", "doc_id": "2", "text": corpus[1]},
                    {"phase": "phase_c", "source_id": "tiny", "doc_id": "3", "text": corpus[2]},
                ] * 10,
                tokenizer_path=root / "tokenizer" / "tokenizer.json",
                output_root=root / "release",
                phase_targets={"phase_a": 20, "phase_b": 12, "phase_c": 8},
                shard_tokens=10,
            )
            self.assertEqual(packed["target_tokens"], 40)
            for shard in packed["shards"]:
                self.assertEqual(Path(shard["binary"]).stat().st_size, shard["tokens"] * 2)


if __name__ == "__main__":
    unittest.main()
