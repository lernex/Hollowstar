from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import torch

from metis_ablation.specs import GLOBAL_BATCH_SEQUENCES, spec_by_name
from metis_ablation.train import main
from tests import test_more_execution_bounds as execution_bounds


class CausalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = execution_bounds.ExecutionBoundTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def set_row(self, name):
        self.fixture.spec = replace(
            spec_by_name(name), apus=1, micro_batch=GLOBAL_BATCH_SEQUENCES,
            grad_accum=1, optimizer_sharding="none",
            config_overrides=self.fixture.spec.config_overrides,
        )
        object.__setattr__(self.fixture.spec, "micro_batch", 1)
        object.__setattr__(self.fixture.spec, "grad_accum", 2)

    def test_causal_full_model_training_has_sealed_policy_and_budget_telemetry(self):
        self.set_row("more-core")
        self.fixture.train(
            "causal", compute_allocation_mode="causal", stop_after_steps=2,
            joint_router_exploration=0.5, keep_all_checkpoints=True,
        )
        manifest = self.fixture.manifest("causal")
        self.assertTrue(manifest["model"]["causal_compute_budget"])
        self.assertEqual(manifest["curriculum"]["compute_allocation_mode"], "joint")
        self.assertEqual(manifest["curriculum"]["memory_gate_scale"], 0.0)
        records = [
            json.loads(line) for line in (
                self.fixture.root / "causal/more-core/telemetry/rank-00000.jsonl"
            ).read_text().splitlines()
        ]
        for record in records:
            telemetry = record["telemetry"]
            self.assertGreater(telemetry["global_joint_router_flops"], 0)
            self.assertLessEqual(telemetry["global_joint_budget_fraction"], 1)
            self.assertGreater(record["estimated_hardware_flops"], telemetry["global_joint_model_flops"])
        before = self.fixture.checkpoint("causal", 2)["model"]
        self.fixture.train(
            "causal", compute_allocation_mode="causal", stop_after_steps=3,
            joint_router_exploration=0.5, keep_all_checkpoints=True,
        )
        after = self.fixture.checkpoint("causal", 3)["model"]
        body = [key for key in before if key.startswith("layers.") and isinstance(before[key], torch.Tensor)]
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in body))
        self.assertEqual(manifest["run_identity_sha256"], self.fixture.manifest("causal")["run_identity_sha256"])
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.fixture.train(
                "causal", compute_allocation_mode="causal", stop_after_steps=4,
                joint_router_exploration=0.5, causal_compute_price=0.1,
            )

    def test_lean_reference_keeps_identity_behavior_and_counts_only_terminal_head(self):
        for row in ("loop-fixed", "loop-pathway-frozen"):
            with self.subTest(row=row):
                self.set_row(row)
                self.fixture.train(row, compute_allocation_mode="causal-fixed", stop_after_steps=1)
                manifest = self.fixture.manifest(row)
                self.assertEqual(manifest["curriculum"]["pathway_mode"], spec_by_name(row).pathway_mode)
                self.assertEqual(manifest["curriculum"]["fixed_routed_k"], 4)
                self.assertEqual(manifest["curriculum"]["max_passes"], 2)
                path = self.fixture.root / row / row / "telemetry/rank-00000.jsonl"
                telemetry = json.loads(path.read_text().splitlines()[0])["telemetry"]
                self.assertEqual(telemetry["global_joint_router_flops"], 0)
                self.assertAlmostEqual(telemetry["global_joint_budget_fraction"], 1)
                self.assertEqual(
                    telemetry["global_executed_active_tokens"],
                    2 * telemetry["global_executed_lm_head_tokens"],
                )
                frozen = manifest["optimizer"]["frozen_policy_parameters"]
                self.assertTrue(any(name.startswith("joint_router.") for name in frozen))

    def test_causal_modes_cannot_silently_convert_controls_or_rm(self):
        for row, mode, message in (
            ("loop-fixed", "causal", "silently replace"),
            ("more-core", "causal-fixed", "explicit fixed"),
            ("more-rm", "causal", "Core"),
        ):
            with self.subTest(row=row, mode=mode):
                self.set_row(row)
                with self.assertRaisesRegex(ValueError, message):
                    self.fixture.train(row, compute_allocation_mode=mode, stop_after_steps=1)

    def test_cli_matched_pilot_geometry_preserves_global_batch_and_declared_schedule(self):
        with patch("metis_ablation.train.train_row", return_value={}) as trainer:
            main([
                "--row", "loop-fixed", "--output", "unused",
                "--compute-allocation-mode", "causal-fixed",
                "--diagnostic-apus", "80", "--diagnostic-micro-batch", "6",
                "--stop-after-steps", "100", "--schedule-total-steps", "25429",
                "--keep-all-checkpoints", "--activation-recompute-policy", "none",
            ])
        spec = trainer.call_args.args[0]
        self.assertEqual((spec.apus, spec.micro_batch, spec.grad_accum), (80, 6, 1))
        self.assertEqual(spec.apus * spec.micro_batch * spec.grad_accum, GLOBAL_BATCH_SEQUENCES)
        self.assertIsNone(spec.measured_tokens_per_second)
        self.assertIsNone(trainer.call_args.kwargs["max_steps"])
        self.assertEqual(trainer.call_args.kwargs["schedule_total_steps"], 25429)
        self.assertTrue(trainer.call_args.kwargs["keep_all_checkpoints"])


if __name__ == "__main__":
    unittest.main()
