from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from metis_ablation.routing_credit_probe import build_parser, held_out_batch
from metis_ablation.sampler import AblationSampleStream
from metis_training.data import TrainingBatch


def sampler(budget=896):
    return AblationSampleStream(
        SimpleNamespace(sequence_length=8),
        budget_tokens=budget,
        block_tokens=16,
        phase_starts={"phase_a": 0, "phase_b": 4096, "phase_c": 6144},
        phase_tokens={"phase_a": 4096, "phase_b": 2048, "phase_c": 1024},
    )


class AblationHoldoutTests(unittest.TestCase):
    def test_every_selected_gap_avoids_every_training_input_and_target(self):
        stream = sampler()
        manifest = stream.describe()
        training = [
            (stream.release_cursor(step), stream.release_cursor(step) + stream.block_tokens)
            for step in range(stream.total_blocks)
        ]
        selected = set()
        for step in range(stream.total_blocks):
            for gap in range(1, 8):
                cursor = stream.evaluation_cursor(step, gap_blocks=gap, window_tokens=8)
                self.assertNotIn(cursor, selected)
                selected.add(cursor)
                for start, last_target in training:
                    self.assertTrue(
                        cursor > last_target or cursor + 8 < start,
                        (step, gap, cursor, start, last_target),
                    )
        self.assertEqual(len(selected), stream.total_blocks * 7)
        self.assertEqual(stream.describe(), manifest)

    def test_first_gap_skips_the_preceding_training_target(self):
        stream = sampler()
        self.assertEqual(stream.evaluation_cursor(0, gap_blocks=1, window_tokens=8), 17)
        self.assertGreater(
            stream.evaluation_cursor(0, gap_blocks=1, window_tokens=8),
            stream.release_cursor(0) + stream.block_tokens,
        )

    def test_boundaries_and_dense_training_budgets_fail_closed(self):
        stream = sampler()
        for gap, length in [(0, 8), (8, 8), (7, 16), (1, 0), (1, 128)]:
            with self.subTest(gap=gap, length=length), self.assertRaises(ValueError):
                stream.evaluation_cursor(0, gap_blocks=gap, window_tokens=length)
        with self.assertRaises(IndexError):
            stream.evaluation_cursor(-1, gap_blocks=2, window_tokens=8)
        with self.assertRaises(IndexError):
            stream.evaluation_cursor(stream.total_blocks, gap_blocks=2, window_tokens=8)
        with self.assertRaises(ValueError):
            sampler(budget=1792).evaluation_cursor(0, gap_blocks=4, window_tokens=8)
        with self.assertRaises(ValueError):
            stream.evaluation_cursor(31, gap_blocks=7, window_tokens=15)
        self.assertEqual(stream.evaluation_cursor(31, gap_blocks=7, window_tokens=14), 4081)

    def test_probe_gap_remains_held_out_after_the_checkpoint_passes_its_cell(self):
        stream = sampler()
        inventory = SimpleNamespace(release_sha256="a" * 64, shard_manifest_sha256="b" * 64)
        identity = {
            "release": {
                "release_sha256": inventory.release_sha256,
                "shard_manifest_sha256": inventory.shard_manifest_sha256,
            },
            "sampler": stream.describe(),
            "model": {"vocab_size": 128},
        }

        def read(*, global_token_cursor, rank, world_size, micro_batch_size):
            self.assertEqual((rank, world_size, micro_batch_size), (0, 1, 1))
            ids = torch.arange(8).reshape(1, 8)
            return TrainingBatch(
                input_ids=ids, labels=ids + 1, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                document_ids=torch.zeros_like(ids), reset_mask=ids.eq(0), canonical_ids=ids,
                phase="phase_a", global_token_cursor=global_token_cursor,
                next_global_token_cursor=global_token_cursor + 8,
                non_padding_tokens=8, supervised_tokens=8,
            )

        with (
            patch("metis_ablation.routing_credit_probe.DeterministicReleaseStream",
                  return_value=SimpleNamespace(batch=read)),
            patch("metis_ablation.routing_credit_probe.build_sample_stream", return_value=stream),
        ):
            batch, metadata = held_out_batch(
                inventory, identity, checkpoint_step=50, step=0,
                sequences=1, sequence_length=8, gap_blocks=3,
            )
        self.assertEqual(batch.global_token_cursor, 49)
        self.assertTrue(metadata["disjoint_from_entire_declared_training_sampler"])
        self.assertEqual(metadata["gap_blocks"], 3)
        with self.assertRaisesRegex(ValueError, "precedes"):
            held_out_batch(
                inventory, identity, checkpoint_step=50, step=0,
                sequences=1, sequence_length=8,
            )

    def test_cli_gap_is_explicit_and_legacy_default_is_unchanged(self):
        parser = build_parser()
        required = ["--checkpoint", "checkpoint", "--release-root", "release", "--output", "out"]
        self.assertEqual(parser.parse_args(required).evaluation_gap_blocks, 0)
        self.assertEqual(
            parser.parse_args([*required, "--evaluation-gap-blocks", "8"]).evaluation_gap_blocks, 8
        )


if __name__ == "__main__":
    unittest.main()
