from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
import uuid

import torch

from metis_ablation.specs import GLOBAL_BATCH_SEQUENCES, GLOBAL_BATCH_TOKENS, spec_by_name
from metis_ablation.train import _parser, train_row


class ExecutionBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.root = Path.cwd() / f".execution-bound-tests-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.spec = replace(
            spec_by_name("more-rm"), apus=1, micro_batch=GLOBAL_BATCH_SEQUENCES,
            grad_accum=1, optimizer_sharding="none",
            config_overrides={
                "d_model": 128, "n_heads": 2, "head_dim": 64, "n_kv_heads": 1,
                "n_layers": 2, "attention_indices": (1,), "latent_dim": 64,
                "expert_intermediate_dim": 32, "n_routed_experts": 8,
                "mamba_ngroups": 1, "mamba_head_dim": 64, "mamba_chunk_size": 16,
                "sequence_length": 64, "final_context_length": 64,
                "context_extension_train_length": 64, "vocab_size": 256,
                "mhc_pass_embedding_dim": 8, "route_feature_dim": 16,
                "memory_dim": 16, "memory_heads": 2, "max_routed_k": 8,
                "target_mean_routed_k": 4.0, "activation_recompute_policy": "none",
                "_ngram_slots_per_head": 257,
            },
        )
        # Match the existing synthetic trainer smoke-test reduction.
        object.__setattr__(self.spec, "micro_batch", 1)
        object.__setattr__(self.spec, "grad_accum", 2)

    def tearDown(self):
        shutil.rmtree(self.root)

    def train(self, name, **changes):
        options = dict(
            output_root=self.root / name, release_root=None,
            budget_tokens=6 * GLOBAL_BATCH_TOKENS, learning_rate=1e-4,
            seed=17, checkpoint_every=0, analysis_every=0, telemetry_every=1,
            max_steps=None, schedule_total_steps=None, device_override="cpu",
            synthetic=True,
        )
        options.update(changes)
        return train_row(self.spec, **options)

    def manifest(self, name):
        return json.loads((self.root / name / self.spec.name / "run.json").read_text())

    def checkpoint_path(self, name, step):
        return self.root / name / self.spec.name / "checkpoints" / f"step-{step:07d}" / "state.pt"

    def checkpoint(self, name, step):
        return torch.load(self.checkpoint_path(name, step), map_location="cpu", weights_only=False)

    def assert_tree_equal(self, left, right, path="payload"):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype, path)
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assert_tree_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (first, second) in enumerate(zip(left, right)):
                self.assert_tree_equal(first, second, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def test_progressive_stops_match_uninterrupted_model_optimizer_rng_and_lr(self):
        uninterrupted = self.train("continuous")
        first = self.train("progressive", stop_after_steps=2)
        first_manifest = self.manifest("progressive")
        first_checkpoint = self.checkpoint("progressive", 2)
        self.assertEqual(first["steps"], 2)
        self.assertEqual(first["planned_total_steps"], 6)
        self.assertTrue(first["stopped_before_planned_end"])
        self.assertEqual(first_manifest["total_steps"], 6)
        self.assertEqual(first_manifest["schedule"]["total_steps"], 6)
        self.assertEqual(first_checkpoint["step"], 2)
        self.assertEqual(first_checkpoint["total_steps"], 6)
        self.assertEqual(first_checkpoint["step_semantics"], "next_unexecuted")

        second = self.train("progressive", stop_after_steps=4)
        self.assertEqual((second["start_step"], second["steps"]), (2, 4))
        self.assertEqual(first_manifest["run_identity_sha256"], self.manifest("progressive")["run_identity_sha256"])
        final = self.train("progressive")
        self.assertEqual((final["start_step"], final["steps"]), (4, 6))
        self.assertEqual(final["final_loss"], uninterrupted["final_loss"])
        self.assertFalse(final["stopped_before_planned_end"])
        self.assert_tree_equal(self.checkpoint("continuous", 6), self.checkpoint("progressive", 6))
        self.assertEqual(first_manifest["run_identity"], self.manifest("progressive")["run_identity"])
        self.assertNotIn("stop_after_steps", json.dumps(first_manifest["run_identity"]))
        campaign = next((self.root / "progressive").glob("CAMPAIGN_IDENTITY-*.json"))
        self.assertNotIn("stop_after_steps", campaign.read_text())

        def rows(name):
            path = self.root / name / self.spec.name / "telemetry/rank-00000.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()]

        left, right = rows("continuous"), rows("progressive")
        self.assertEqual([row["step"] for row in right], list(range(6)))
        for first_row, second_row in zip(left, right):
            for field in ("step", "loss", "learning_rate", "global_supervised_tokens", "depth_histogram"):
                self.assertEqual(first_row[field], second_row[field], field)
        records = list((self.root / "progressive" / self.spec.name / "operational/startups").glob("*.json"))
        self.assertEqual({json.loads(path.read_text())["stop_after_steps"] for path in records}, {None, 2, 4})

    def test_repeating_reached_bound_does_not_rewrite_checkpoint(self):
        self.train("run", stop_after_steps=2)
        path = self.checkpoint_path("run", 2)
        before = (path.stat().st_mtime_ns, path.read_bytes())
        result = self.train("run", stop_after_steps=2)
        self.assertEqual((result["start_step"], result["steps"]), (2, 2))
        self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)

    def test_bound_behind_resume_rejects_before_truncating_progress(self):
        self.train("run", stop_after_steps=2)
        telemetry = self.root / "run" / self.spec.name / "telemetry/rank-00000.jsonl"
        before = telemetry.read_bytes()
        with self.assertRaisesRegex(ValueError, "precedes"):
            self.train("run", stop_after_steps=1)
        self.assertEqual(telemetry.read_bytes(), before)
        self.assertTrue(self.checkpoint_path("run", 2).exists())

    def test_invalid_bounds_and_conflicting_legacy_controls_are_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                self.train(f"invalid-{value}", stop_after_steps=value)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.train("too-large", stop_after_steps=7)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.train("both", stop_after_steps=2, max_steps=4)
        with self.assertRaisesRegex(ValueError, "requires a final checkpoint"):
            self.train("no-final", stop_after_steps=2, final_checkpoint=False)

    def test_legacy_max_steps_still_changes_identity_and_recipe_guards_remain(self):
        result = self.train("legacy", max_steps=2)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(self.manifest("legacy")["total_steps"], 2)
        self.assertEqual(self.checkpoint("legacy", 2)["total_steps"], 2)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.train("legacy", max_steps=4)
        self.train("bounded", stop_after_steps=2)
        with self.assertRaisesRegex(RuntimeError, "Refusing to resume"):
            self.train("bounded", stop_after_steps=4, learning_rate=2e-4)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.train("bounded", stop_after_steps=4, budget_tokens=8 * GLOBAL_BATCH_TOKENS)

    def test_early_stop_records_terminal_telemetry_and_keeps_long_lr_horizon(self):
        result = self.train(
            "scheduled", stop_after_steps=2, schedule_total_steps=10,
            checkpoint_every=2, telemetry_every=100,
        )
        self.assertEqual(result["steps"], 2)
        manifest = self.manifest("scheduled")
        self.assertEqual(manifest["total_steps"], 6)
        self.assertEqual(manifest["schedule"]["total_steps"], 10)
        path = self.root / "scheduled" / self.spec.name / "telemetry/rank-00000.jsonl"
        self.assertEqual([json.loads(line)["step"] for line in path.read_text().splitlines()], [0, 1])
        self.assertTrue(self.checkpoint_path("scheduled", 2).exists())

    def test_cli_stop_bound_is_distinct_from_max_steps(self):
        args = _parser().parse_args([
            "--row", "more-rm", "--output", "unused", "--stop-after-steps", "1000",
        ])
        self.assertEqual(args.stop_after_steps, 1000)
        self.assertIsNone(args.max_steps)

    def test_pilot_checkpoint_retention_preserves_every_progressive_boundary(self):
        self.train("retained", stop_after_steps=2, checkpoint_every=1, keep_all_checkpoints=True)
        original = {step: self.checkpoint_path("retained", step).read_bytes() for step in (1, 2)}
        self.train("retained", stop_after_steps=4, checkpoint_every=1, keep_all_checkpoints=True)
        for step in range(1, 5):
            self.assertTrue(self.checkpoint_path("retained", step).is_file())
        for step, contents in original.items():
            self.assertEqual(self.checkpoint_path("retained", step).read_bytes(), contents)
        self.assertTrue(self.manifest("retained")["keep_all_checkpoints"])
        self.assertNotIn("keep_all_checkpoints", self.manifest("retained")["run_identity"])


if __name__ == "__main__":
    unittest.main()
