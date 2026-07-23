from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data.final_dedup import content_sha256
from metis_data.selection import _iter_zstd, build_selection
from metis_data.stage_runner import _token_count
from metis_data.state import StateStore
from metis_data.tokenizer import train_tokenizer
from metis_data.training_contract import _artifact


class ReleaseContractIntegrityTests(unittest.TestCase):
    def test_replay_pool_contains_only_previously_selected_token_spans(self) -> None:
        manifest = {
            "selection": {"seed": 7, "replay": {"maximum_document_exposures": 4}},
            "schedule": {
                "target_tokens": 30,
                "phases": {
                    "phase_a": {"target_tokens": 10, "replay_tokens": 0},
                    "phase_b": {"target_tokens": 10, "replay_tokens": 10},
                    "phase_c": {"target_tokens": 10, "replay_tokens": 10},
                },
            },
            "sources": [
                {
                    "id": "source",
                    "phase_tokens": {
                        "phase_a": 10,
                        "phase_b": 10,
                        "phase_c": 10,
                    },
                }
            ],
        }
        record = {
            "source_id": "source",
            "doc_id": "doc",
            "text": "the full document text",
            "token_count": 100,
            "content_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = build_selection(
                [record],
                manifest=manifest,
                eligible_tokens={"source": 100},
                output_root=Path(temporary),
                shard_tokens=10,
            )
            self.assertEqual(result["schema"], "metis.selection-release/v2")
            self.assertEqual(result["unique_tokens"], 10)
            self.assertEqual(result["replay_tokens"], 20)
            for shard in result["shards"]:
                self.assertEqual(
                    hashlib.sha256(Path(shard["path"]).read_bytes()).hexdigest(),
                    shard["sha256"],
                )
                for selected in _iter_zstd(Path(shard["path"])):
                    if selected["replay"]:
                        self.assertLessEqual(
                            int(selected["token_start"])
                            + int(selected["token_count"]),
                            10,
                        )

    def test_token_count_resume_rejects_changed_final_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_root = root / "tokenizer"
            release = train_tokenizer(
                iter(["alpha beta gamma delta"] * 100),
                output_dir=tokenizer_root,
                vocabulary_size=512,
                special_tokens=["<|endoftext|>", "<|padding|>"],
                minimum_frequency=1,
            )
            (tokenizer_root / "TOKENIZER_VALIDATION.json").write_text(
                '{"ok":true}\n',
                encoding="utf-8",
            )
            final_root = root / "eligible" / "final"
            final_root.mkdir(parents=True)
            text = "alpha beta gamma"
            final_path = final_root / "task-000000.jsonl"
            final_path.write_text(
                json.dumps(
                    {
                        "id": "doc",
                        "text": text,
                        "metadata": {
                            "source_id": "source",
                            "category": "web",
                            "final_content_sha256": content_sha256(text).hex(),
                            "license": "CC-BY-4.0",
                            "license_status": "reviewed",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            profile = {
                "manifest": "unused",
                "storage": {
                    "lustre_root": str(root),
                    "final_token_dtype": "uint16",
                    "directories": {
                        "state": "state",
                        "eligible": "eligible",
                        "token_counts": "token-counts",
                        "tokenizer": "tokenizer",
                    },
                },
            }
            state = StateStore(root / "state")
            state.write(
                "build.inputs.json",
                payload={"input_count": 1, "inputs": [{"input_id": "one"}]},
            )
            manifest = {
                "tokenizer": {
                    "vocabulary_size_including_special_tokens": release[
                        "vocabulary_size"
                    ],
                    "special_tokens": ["<|endoftext|>", "<|padding|>"],
                }
            }
            with mock.patch(
                "metis_data.stage_runner._manifest",
                return_value=manifest,
            ):
                report = _token_count(profile, 0)
                self.assertEqual(report["schema"], "metis.token-count-task/v2")
                self.assertEqual(
                    report["output_artifact"]["sha256"],
                    hashlib.sha256(
                        (root / "token-counts" / "task-000000.jsonl.zst").read_bytes()
                    ).hexdigest(),
                )
                final_path.write_text(
                    final_path.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "immutable inputs/tokenizer",
                ):
                    _token_count(profile, 0)

    def test_training_artifact_path_cannot_escape_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root.parent / "outside-release-contract-test"
            outside.write_text("outside", encoding="utf-8")
            self.addCleanup(outside.unlink, missing_ok=True)
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                _artifact(root, {"bad": "../outside-release-contract-test"}, "bad")


if __name__ == "__main__":
    unittest.main()
