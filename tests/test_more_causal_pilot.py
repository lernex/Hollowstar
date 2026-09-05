from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from metis_ablation.specs import GLOBAL_BATCH_SEQUENCES, spec_by_name
from metis_ablation.train import _routed_expert_gradient_evidence, main
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
                    2 * telemetry["global_budgeted_lm_head_tokens"],
                )
                self.assertEqual(telemetry["global_executed_lm_head_tokens"], 126)
                self.assertEqual(
                    telemetry["global_lm_head_recompute_flops"],
                    telemetry["global_lm_head_forward_flops"],
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

    def test_terminal_objective_is_sealed_and_runs_only_actual_exit_heads(self):
        self.set_row("more-core")
        self.fixture.train(
            "terminal", compute_allocation_mode="causal", stop_after_steps=2,
            terminal_action_critic=True, terminal_critic_exploration=0.5,
            require_routed_expert_gradients=True,
        )
        manifest = self.fixture.manifest("terminal")
        self.assertTrue(manifest["model"]["terminal_action_critic"])
        self.assertEqual(manifest["model"]["terminal_critic_exploration"], 0.5)
        self.assertTrue(manifest["optimizer"]["require_routed_expert_gradients"])
        path = self.fixture.root / "terminal/more-core/telemetry/rank-00000.jsonl"
        for line in path.read_text().splitlines():
            record = json.loads(line)
            telemetry = record["telemetry"]
            self.assertEqual(telemetry["terminal_action_critic_enabled"], 1)
            self.assertEqual(telemetry["global_executed_lm_head_tokens"], record["global_supervised_tokens"])
            self.assertEqual(telemetry["global_budgeted_lm_head_tokens"], 128)
            self.assertEqual(
                telemetry["global_lm_head_recompute_flops"],
                telemetry["global_lm_head_forward_flops"],
            )
            if record["step"] == 0:
                self.assertGreater(telemetry["routed_expert_gradient_parameters_expected"], 0)
                self.assertEqual(
                    telemetry["routed_expert_gradient_parameters_expected"],
                    telemetry["routed_expert_gradient_parameters_present"],
                )
                self.assertGreater(telemetry["routed_expert_gradient_parameters_nonzero"], 0)
            self.assertLessEqual(telemetry["global_joint_budget_fraction"], 1)
        with self.assertRaisesRegex(ValueError, "explicit causal"):
            self.fixture.train("invalid-terminal", stop_after_steps=1, terminal_action_critic=True)

    def test_reference_bootstrap_is_sealed_and_expires_at_its_absolute_step(self):
        self.set_row("more-core")
        self.fixture.train(
            "bootstrap", compute_allocation_mode="causal", stop_after_steps=1,
            terminal_action_critic=True, terminal_reference_bootstrap_steps=1,
            require_routed_expert_gradients=True,
        )
        first = self.fixture.manifest("bootstrap")
        self.fixture.train(
            "bootstrap", compute_allocation_mode="causal", stop_after_steps=2,
            terminal_action_critic=True, terminal_reference_bootstrap_steps=1,
            require_routed_expert_gradients=True,
        )
        second = self.fixture.manifest("bootstrap")
        self.assertEqual(first["run_identity_sha256"], second["run_identity_sha256"])
        self.assertEqual(first["model"]["terminal_reference_bootstrap_steps"], 1)
        path = self.fixture.root / "bootstrap/more-core/telemetry/rank-00000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(
            [row["telemetry"]["terminal_reference_bootstrap_active"] for row in rows], [1, 0]
        )
        for row in rows:
            self.assertLessEqual(row["telemetry"]["global_joint_budget_fraction"], 1)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.fixture.train(
                "bootstrap", compute_allocation_mode="causal", stop_after_steps=3,
                terminal_action_critic=True, terminal_reference_bootstrap_steps=2,
                require_routed_expert_gradients=True,
            )

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

    def test_quality_launcher_covers_each_permanent_window_exactly_once(self):
        root = self.fixture.root
        binary = root / "bin"
        binary.mkdir()
        programs = {
            "git": '#!/bin/bash\nif [ "$3" = rev-parse ]; then echo pinned; fi\n',
            "srun": '#!/bin/bash\nshift\nfor rank in 0 1 2; do SLURM_LOCALID="$rank" "$@" || exit; done\n',
            "python": (
                f"#!{sys.executable}\nimport json,os,sys\n"
                "print(json.dumps({'argv': sys.argv[1:], 'physical_gpu': os.environ['ROCR_VISIBLE_DEVICES']}))\n"
            ),
        }
        for name, contents in programs.items():
            path = binary / name
            path.write_text(contents)
            path.chmod(0o755)
        for name in ("state.pt", "run.json", "runtime.sh"):
            (root / name).touch()
        environment = {
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "METIS_REPO": str(root), "METIS_EXPECTED_REVISION": "pinned",
            "METIS_RELEASE_ROOT": str(root), "METIS_QUALITY_CHECKPOINT": str(root / "state.pt"),
            "METIS_QUALITY_MANIFEST": str(root / "run.json"),
            "METIS_QUALITY_OUTPUT": str(root / "quality"),
            "METIS_ABLATION_RUNTIME": str(root / "runtime.sh"),
            "SLURM_NTASKS": "3", "SLURM_JOB_NUM_NODES": "1",
        }
        launcher = Path(__file__).resolve().parents[1] / "slurm/ablation/causal-checkpoint-quality.sbatch"
        result = subprocess.run(["bash", str(launcher)], env=environment, text=True, capture_output=True, check=True)
        launches = [json.loads(line) for line in result.stdout.splitlines()]
        commands = [launch["argv"] for launch in launches]
        self.assertEqual(len(commands), 3)
        self.assertEqual([command[command.index("--step") + 1] for command in commands], ["2000", "19000", "25000"])
        self.assertEqual([launch["physical_gpu"] for launch in launches], ["0", "1", "2"])
        self.assertEqual([command[command.index("--device") + 1] for command in commands], ["cuda:0"] * 3)
        for command in commands:
            self.assertIn("--quality-only", command)
            self.assertEqual(command[command.index("--evaluation-gap-blocks") + 1], "8")
            self.assertEqual(command[command.index("--repeat-forwards") + 1], "5")
        for field, value in (("SLURM_NTASKS", "2"), ("SLURM_JOB_NUM_NODES", "3")):
            result = subprocess.run(["bash", str(launcher)], env={**environment, field: value}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    def test_expert_gradient_gate_rejects_disconnected_and_zero_credit(self):
        parameter = torch.nn.Parameter(torch.ones(4))
        model = unittest.mock.Mock()
        model.named_parameters.return_value = [
            ("layers.0.moe.local_experts.gate_up.weight_chunks.0", parameter)
        ]
        with self.assertRaisesRegex(RuntimeError, "missing after backward"):
            _routed_expert_gradient_evidence(model)
        parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "finite nonzero"):
            _routed_expert_gradient_evidence(model)
        parameter.grad.fill_(1)
        evidence = _routed_expert_gradient_evidence(model)
        self.assertEqual(evidence["routed_expert_gradient_parameters_present"], 1)
        self.assertEqual(evidence["routed_expert_gradient_parameters_nonzero"], 1)
        parameter.grad[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite nonzero"):
            _routed_expert_gradient_evidence(model)

    def test_fresh_pilot_cache_is_explicit_and_leaves_existing_cache_untouched(self):
        root = self.fixture.root
        binary = root / "bin"
        binary.mkdir()
        old_cache = root / "existing-cache"
        old_cache.mkdir()
        marker = old_cache / "historical-artifact"
        marker.write_text("preserve")
        runtime = root / "runtime.sh"
        runtime.write_text(
            f'export METIS_NODE_SCRATCH="{root}/private-job"\n'
            f'export TRITON_CACHE_DIR="{old_cache}"\n'
        )
        programs = {
            "git": '#!/bin/bash\nif [ "$3" = rev-parse ]; then echo pinned; fi\n',
            "scontrol": "#!/bin/bash\necho test-node\n",
            "srun": '#!/bin/bash\nshift 2\nSLURM_PROCID=0 SLURM_LOCALID=0 "$@"\n',
            "python": (
                f"#!{sys.executable}\nimport json,os,sys\n"
                "print(json.dumps({'argv': sys.argv[1:], 'cache': os.environ['TRITON_CACHE_DIR']}))\n"
            ),
        }
        for name, contents in programs.items():
            path = binary / name
            path.write_text(contents)
            path.chmod(0o755)
        environment = {
            **os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "METIS_REPO": str(root), "METIS_EXPECTED_REVISION": "pinned",
            "METIS_RELEASE_ROOT": str(root), "METIS_PILOT_OUTPUT": str(root / "output"),
            "METIS_PILOT_EXPERIMENT": "terminal-core", "METIS_PILOT_STOP_AFTER_STEPS": "100",
            "METIS_ABLATION_RUNTIME": str(runtime), "SLURM_NTASKS": "40",
            "SLURM_JOB_NODELIST": "test-node",
        }
        launcher = Path(__file__).resolve().parents[1] / "slurm/ablation/causal-compute-pilot.sbatch"
        for fresh in ("0", "1"):
            result = subprocess.run(
                ["bash", str(launcher)], env={**environment, "METIS_PILOT_FRESH_TRITON_CACHE": fresh},
                text=True, capture_output=True, check=True,
            )
            launch = json.loads(result.stdout)
            expected = old_cache if fresh == "0" else root / "private-job/pilot-triton"
            self.assertEqual(launch["cache"], str(expected))
            self.assertEqual(marker.read_text(), "preserve")
            self.assertIn("--terminal-action-critic", launch["argv"])
            self.assertIn("--stop-after-steps", launch["argv"])


if __name__ == "__main__":
    unittest.main()
