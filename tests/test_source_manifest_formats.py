from __future__ import annotations

import unittest
from pathlib import Path

from metis_data.config import load_yaml
from metis_data.manifest import matches_any
from metis_data.source_lock import _iter_repo_files, _repo_tree_roots


class SourceManifestFormatTests(unittest.TestCase):
    def _source(self, manifest_name: str, source_id: str) -> dict:
        root = Path(__file__).resolve().parents[1] / "manifests" / "sources"
        payload = load_yaml(root / manifest_name)
        return next(source for source in payload["sources"] if source["id"] == source_id)

    def test_pinned_hugging_face_layouts_match_training_payloads(self) -> None:
        cases = (
            (
                "web.yaml",
                "txt360",
                "v1.1/TxT360_BestOfWeb/cc_1-1/1-1.chunk0_part_000.jsonl",
            ),
            ("math.yaml", "nemotron_math_proofs", "data/lean.jsonl"),
            ("science.yaml", "pes2o", "data/v2/train-00000-of-00020.json.gz"),
        )
        for manifest_name, source_id, pinned_path in cases:
            with self.subTest(source_id=source_id):
                source = self._source(manifest_name, source_id)
                self.assertTrue(matches_any(pinned_path, source["access"]["allow_patterns"]))

    def test_nemotron_math_proofs_uses_publisher_license(self) -> None:
        source = self._source("math.yaml", "nemotron_math_proofs")
        self.assertEqual(source["license"]["expression"], "CC-BY-SA-4.0")

    def test_txt360_excludes_synthetic_and_transformed_v11_partitions(self) -> None:
        source = self._source("web.yaml", "txt360")
        patterns = source["access"]["allow_patterns"]
        self.assertFalse(
            matches_any("v1.1/TxT360_QA/cc_1-1/example.jsonl", patterns)
        )
        self.assertFalse(
            matches_any("v1.1/wikipedia_extended/1-1/example.jsonl", patterns)
        )

    def test_hugging_face_tree_walk_uses_literal_partition_prefixes(self) -> None:
        self.assertEqual(
            _repo_tree_roots(
                (
                    "v1.1/TxT360_BestOfWeb/cc_1-1/*.jsonl",
                    "v1.1/TxT360_BestOfWeb/cc_2-5/*.jsonl",
                )
            ),
            (
                "v1.1/TxT360_BestOfWeb/cc_1-1",
                "v1.1/TxT360_BestOfWeb/cc_2-5",
            ),
        )
        self.assertEqual(_repo_tree_roots(("**/*.parquet",)), (None,))

        class FakeApi:
            def __init__(self) -> None:
                self.roots: list[str | None] = []

            def list_repo_tree(self, *args, path_in_repo=None, **kwargs):
                self.roots.append(path_in_repo)
                return (
                    type(
                        "RepoFile",
                        (),
                        {
                            "path": (
                                "v1.1/TxT360_BestOfWeb/cc_1-1/"
                                "1-1.chunk0_part_000.jsonl"
                            ),
                            "size": 1024,
                            "blob_id": "blob",
                            "lfs": None,
                        },
                    )(),
                )

        api = FakeApi()
        files = _iter_repo_files(
            api,
            "LLM360/TxT360",
            "3" * 40,
            ("v1.1/TxT360_BestOfWeb/cc_1-1/*.jsonl",),
        )
        self.assertEqual(api.roots, ["v1.1/TxT360_BestOfWeb/cc_1-1"])
        self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
