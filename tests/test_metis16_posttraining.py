from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch

from metis_training.posttraining import (
    CHECKPOINT_RECEIPT_SCHEMA,
    PIPELINE_SCHEMA,
    PipelineContractError,
    PostTrainingOrchestrator,
    avg_at_k,
    difficulty_adaptive_length_budget,
    difficulty_adaptive_length_reward,
    evaluate_metric_gate,
    gated_code_efficiency_reward,
    gspo_loss,
    gspo_token_loss,
    load_pipeline,
    masked_causal_cross_entropy,
    strict_on_policy_filter,
    validate_pipeline,
)
from metis_training.posttraining_cli import main as posttraining_cli_main


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "metis16" / "posttraining.yaml"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _hash(value: dict[str, object], omit: str) -> str:
    return hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != omit})
    ).hexdigest()


def _write_sealed(
    root: Path,
    name: str,
    schema: str,
    *,
    metadata: dict[str, object],
    tokenizer_sha256: str | None = None,
) -> Path:
    artifact = root / name
    artifact.mkdir(parents=True)
    payload_path = artifact / "payload.bin"
    payload_path.write_bytes(f"{name}-payload".encode())
    manifest: dict[str, object] = {
        "envelope_schema": "metis.sealed-artifact/v1",
        "schema": schema,
        "complete": True,
        "metadata": metadata,
        "files": [
            {
                "path": payload_path.name,
                "bytes": payload_path.stat().st_size,
                "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            }
        ],
        "manifest_sha256": "",
    }
    if tokenizer_sha256 is not None:
        manifest["tokenizer_sha256"] = tokenizer_sha256
    manifest["manifest_sha256"] = _hash(manifest, "manifest_sha256")
    manifest_path = artifact / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_initial_receipt(root: Path, family: str) -> Path:
    checkpoint_manifest = _write_sealed(
        root,
        f"{family}-base-checkpoint",
        "metis.model-checkpoint/v1",
        metadata={"stage": "base_pretraining"},
    )
    checkpoint_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    receipt: dict[str, object] = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "family": family,
        "stage": "base_pretraining",
        "parent_checkpoint_sha256": None,
        "config_sha256": "base",
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_sha256": checkpoint_payload["manifest_sha256"],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _hash(receipt, "receipt_sha256")
    receipt_path = root / f"{family}-base-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt_path


class PipelineContractTests(unittest.TestCase):
    def test_production_pipeline_is_locked_and_ordered(self) -> None:
        pipeline = load_pipeline(PIPELINE)
        self.assertEqual(pipeline["schema"], PIPELINE_SCHEMA)
        self.assertEqual(pipeline["families"], ["praxis", "logos"])
        self.assertEqual(
            [stage["id"] for stage in pipeline["stages"]],
            [
                "context_extension",
                "cold_start_sft",
                "overall_sft",
                "hybrid_mode_gspo",
                "specialist_reasoning",
                "specialist_code",
                "specialist_knowledge",
                "specialist_writing",
                "specialist_agentic",
                "opd_consolidation",
                "evaluation",
                "publish_gate",
            ],
        )
        self.assertEqual(pipeline["context_extension"]["train_context"], 163_840)
        self.assertEqual(pipeline["context_extension"]["deploy_context"], 131_072)

    def test_staged_context_extension_is_rejected(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        pipeline["context_extension"]["schedule"] = "4096_to_32768_to_131072"
        with self.assertRaisesRegex(PipelineContractError, "one 4,096"):
            validate_pipeline(pipeline)

    def test_context_extension_exposure_is_exactly_eighteen_billion(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        pipeline["context_extension"]["token_budget"] = 12_000_000_001
        with self.assertRaisesRegex(PipelineContractError, "exactly 18B"):
            validate_pipeline(pipeline)
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(
            item
            for item in pipeline["stages"]
            if item["id"] == "context_extension"
        )
        stage["token_budget"] = 12_000_000_001
        with self.assertRaisesRegex(PipelineContractError, "inherit the exact"):
            validate_pipeline(pipeline)

    def test_grpo_or_kl_silently_replacing_gspo_is_rejected(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(
            item
            for item in pipeline["stages"]
            if item["id"] == "specialist_reasoning"
        )
        stage["objective"]["algorithm"] = "grpo"
        with self.assertRaisesRegex(PipelineContractError, "sequence GSPO without KL"):
            validate_pipeline(pipeline)

    def test_missing_direct_answer_mode_is_rejected(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(item for item in pipeline["stages"] if item["id"] == "cold_start_sft")
        stage["answer_modes"] = {"think": 0.5, "think_max": 0.5}
        with self.assertRaisesRegex(
            PipelineContractError, "direct, think, and think_max"
        ):
            validate_pipeline(pipeline)

    def test_think_max_positive_length_reward_is_rejected(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(
            item
            for item in pipeline["stages"]
            if item["id"] == "hybrid_mode_gspo"
        )
        stage["mode_policy"]["think_max_positive_length_reward"] = True
        with self.assertRaisesRegex(
            PipelineContractError, "all three modes"
        ):
            validate_pipeline(pipeline)

    def test_reasoning_mode_may_not_change_more_depth(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        pipeline["reasoning_modes"]["more_invariance"][
            "target_mean_depth_per_mode"
        ]["think_max"] = 2.5
        with self.assertRaisesRegex(
            PipelineContractError, "stay 2.0"
        ):
            validate_pipeline(pipeline)

    def test_evaluation_results_must_be_checkpoint_bound(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(
            item
            for item in pipeline["stages"]
            if item["id"] == "evaluation"
        )
        result = next(
            item
            for item in stage["requirements"]
            if item["name"] == "evaluation_results"
        )
        result["checkpoint_bound"] = False
        with self.assertRaisesRegex(
            PipelineContractError,
            "exact consolidated checkpoint",
        ):
            validate_pipeline(pipeline)

    def test_opd_requires_its_same_tokenizer_generation_adapter(self) -> None:
        pipeline = copy.deepcopy(load_pipeline(PIPELINE))
        stage = next(
            item
            for item in pipeline["stages"]
            if item["id"] == "opd_consolidation"
        )
        stage["requirements"] = [
            item
            for item in stage["requirements"]
            if item["name"] != "opd_generation_adapter"
        ]
        with self.assertRaisesRegex(
            PipelineContractError,
            "same-tokenizer generation adapter",
        ):
            validate_pipeline(pipeline)

    def test_operator_run_routes_to_production_portage_launcher(self) -> None:
        config = object()
        with (
            mock.patch(
                "metis_portage.config.load_portage_config",
                return_value=config,
            ) as load_config,
            mock.patch(
                "metis_portage.launcher.launch",
                return_value={"campaign": "production", "ok": True},
            ) as launch,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = posttraining_cli_main(
                ["run", "--portage-config", "/tmp/portage.yaml"]
            )
        self.assertEqual(code, 0)
        load_config.assert_called_once_with("/tmp/portage.yaml")
        launch.assert_called_once_with(config)
        self.assertIn('"campaign": "production"', output.getvalue())

    def test_legacy_orchestrator_is_only_reachable_by_explicit_command(self) -> None:
        with mock.patch(
            "metis_training.posttraining.main", return_value=7
        ) as legacy:
            code = posttraining_cli_main(
                ["legacy-run", "validate", "--config", "legacy.yaml"]
            )
        self.assertEqual(code, 7)
        legacy.assert_called_once_with(["validate", "--config", "legacy.yaml"])


class ObjectiveTests(unittest.TestCase):
    def test_masked_causal_loss_supervises_only_selected_tokens(self) -> None:
        logits = torch.tensor(
            [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, 1]])
        loss = masked_causal_cross_entropy(
            logits,
            labels,
            torch.tensor([[True, True, False]]),
        )
        loss.backward()
        self.assertLess(float(loss.detach()), 0.1)
        self.assertTrue(torch.all(logits.grad[0, 2] == 0))

    def test_strict_avg_at_16_filter_excludes_boundaries(self) -> None:
        correct = torch.zeros(5, 16)
        correct[1, :2] = 1
        correct[2, :8] = 1
        correct[3, :14] = 1
        correct[4, :] = 1
        rates = avg_at_k(correct)
        mask = strict_on_policy_filter(rates)
        self.assertEqual(mask.tolist(), [False, True, True, True, False])
        explicit = strict_on_policy_filter(torch.tensor([0.10, 0.10001, 0.89999, 0.90]))
        self.assertEqual(explicit.tolist(), [False, True, True, False])

    def test_gspo_uses_sequence_ratio_and_masks_truncation(self) -> None:
        current = torch.zeros(1, 3, 4, requires_grad=True)
        old = torch.zeros_like(current)
        rewards = torch.tensor([[0.0, 1.0, 0.5]])
        mask = torch.ones_like(current, dtype=torch.bool)
        truncated = torch.tensor([[False, False, True]])
        result = gspo_loss(
            current_token_log_probs=current,
            old_token_log_probs=old,
            rewards=rewards,
            response_mask=mask,
            truncated=truncated,
        )
        self.assertEqual(int(result["valid_sequences"]), 2)
        result["loss"].backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(torch.all(current.grad[0, 2] == 0))
        self.assertTrue(torch.any(current.grad[0, :2] != 0))

    def test_gspo_token_supports_turn_level_credit(self) -> None:
        current = torch.zeros(1, 2, 4, requires_grad=True)
        old = torch.zeros_like(current)
        advantages = torch.tensor([[[1.0, 1.0, -0.5, -0.5], [-1.0, -1.0, 0.5, 0.5]]])
        result = gspo_token_loss(
            current_token_log_probs=current,
            old_token_log_probs=old,
            token_advantages=advantages,
            response_mask=torch.ones_like(current, dtype=torch.bool),
        )
        result["loss"].backward()
        self.assertEqual(int(result["valid_sequences"]), 2)
        self.assertEqual(int(result["valid_tokens"]), 8)
        self.assertTrue(torch.any(current.grad != 0))

    def test_code_efficiency_reward_is_gated_on_full_correctness(self) -> None:
        reward = gated_code_efficiency_reward(
            torch.tensor([0.9, 1.0]),
            torch.tensor([100.0, 0.25]),
        )
        self.assertTrue(torch.allclose(reward, torch.tensor([0.9, 1.25])))

    def test_length_reward_never_makes_wrong_short_beat_correct_long(self) -> None:
        pass_rate = torch.tensor([0.8, 0.8])
        correct_mean = torch.tensor([100.0, 100.0])
        budget = difficulty_adaptive_length_budget(
            pass_rate=pass_rate,
            mean_correct_length=correct_mean,
            maximum_length=1000,
        )
        self.assertTrue(torch.allclose(budget, torch.tensor([280.0, 280.0])))
        rewards = difficulty_adaptive_length_reward(
            correctness=torch.tensor([0.0, 1.0]),
            response_lengths=torch.tensor([1.0, 1000.0]),
            pass_rate=pass_rate,
            mean_correct_length=correct_mean,
            maximum_length=1000,
            coefficient=0.05,
        )
        self.assertEqual(float(rewards[0]), 0.0)
        self.assertGreater(float(rewards[1]), float(rewards[0]))

    def test_evaluation_gate_fails_missing_and_regressed_metrics(self) -> None:
        gate = {
            "fail_on_missing_metric": True,
            "metrics": [
                {
                    "name": "score",
                    "comparison": "no_regression_vs_prior",
                    "relative_tolerance": 0.01,
                },
                {"name": "safety", "comparison": "suite_threshold"},
            ],
        }
        result = evaluate_metric_gate(
            {"score": 0.80, "safety": 0.96},
            gate,
            baselines={"prior": {"score": 0.90}},
            suite_thresholds={"safety": {"minimum": 0.95}},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_metrics"][0]["name"], "score")
        missing = evaluate_metric_gate(
            {"score": 0.90},
            gate,
            baselines={"prior": {"score": 0.90}},
            suite_thresholds={"safety": {"minimum": 0.95}},
        )
        self.assertEqual(missing["missing_metrics"], ["safety"])


class OrchestratorTests(unittest.TestCase):
    def test_missing_real_dataset_fails_closed_even_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = _write_sealed(
                root,
                "tokenizer",
                "metis.tokenizer/v1",
                metadata={"vocabulary_size": 65_536},
            )
            initial = _write_initial_receipt(root, "praxis")
            orchestrator = PostTrainingOrchestrator(
                PIPELINE,
                family="praxis",
                state_dir=root / "state",
                initial_checkpoint_receipt=initial,
                environment={"METIS_TOKENIZER_MANIFEST": str(tokenizer)},
            )
            with self.assertRaisesRegex(PipelineContractError, "METIS_CONTEXT_EXTENSION_DATA"):
                orchestrator.run(until="context_extension", dry_run=True)

    def test_completed_stage_resumes_without_relaunching_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = _write_sealed(
                root,
                "tokenizer",
                "metis.tokenizer/v1",
                metadata={"vocabulary_size": 65_536},
            )
            tokenizer_payload = json.loads(tokenizer.read_text(encoding="utf-8"))
            context = _write_sealed(
                root,
                "context",
                "metis.context-extension-data/v1",
                tokenizer_sha256=str(tokenizer_payload["manifest_sha256"]),
                metadata={
                    "tokens": 18_000_000_000,
                    "base_context": 4096,
                    "train_context": 163840,
                    "deploy_context": 131072,
                    "single_jump": True,
                    "long_sequence_fraction": 0.90,
                    "base_sequence_fraction": 0.10,
                    "pretrain_style_fraction": 0.80,
                    "synthetic_long_context_fraction": 0.20,
                    "document_length_histogram_verified": True,
                    "checkpoint_gates": [
                        6_000_000_000,
                        12_000_000_000,
                        18_000_000_000,
                    ],
                    "compact_layout": "metis.compact-causal/v1",
                },
            )
            initial = _write_initial_receipt(root, "praxis")
            counter = root / "backend-count.txt"
            backend = root / "dummy_backend.py"
            backend.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import hashlib
                    import json
                    from pathlib import Path

                    def canonical(value):
                        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--runtime-spec", required=True)
                    args = parser.parse_args()
                    runtime = json.loads(Path(args.runtime_spec).read_text())
                    counter = Path(__import__("os").environ["DUMMY_COUNT"])
                    counter.write_text((counter.read_text() if counter.exists() else "") + "1\\n")
                    root = Path(args.runtime_spec).parent / "checkpoint"
                    root.mkdir()
                    payload = root / "model.bin"
                    payload.write_bytes(b"checkpoint")
                    manifest = {
                        "schema": "metis.distributed-checkpoint/v1",
                        "family": runtime["family"],
                        "extra_state": {
                            "posttraining_stage": runtime["stage"]["id"],
                            "parent_checkpoint_sha256": runtime["parent_checkpoint_sha256"],
                            "stage_config_sha256": runtime["stage_config_sha256"],
                        },
                        "artifacts": [{
                            "path": payload.name,
                            "bytes": payload.stat().st_size,
                            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        }],
                        "checkpoint_sha256": "",
                    }
                    manifest["checkpoint_sha256"] = hashlib.sha256(
                        canonical({k: v for k, v in manifest.items() if k != "checkpoint_sha256"})
                    ).hexdigest()
                    manifest_path = root / "CHECKPOINT.json"
                    manifest_path.write_text(json.dumps(manifest))
                    receipt = {
                        "schema": "metis.checkpoint-receipt/v1",
                        "family": runtime["family"],
                        "stage": runtime["stage"]["id"],
                        "parent_checkpoint_sha256": runtime["parent_checkpoint_sha256"],
                        "config_sha256": runtime["stage_config_sha256"],
                        "checkpoint_manifest": str(manifest_path),
                        "checkpoint_sha256": manifest["checkpoint_sha256"],
                        "receipt_sha256": "",
                    }
                    receipt["receipt_sha256"] = hashlib.sha256(
                        canonical({k: v for k, v in receipt.items() if k != "receipt_sha256"})
                    ).hexdigest()
                    receipt_path = root / "RECEIPT.json"
                    receipt_path.write_text(json.dumps(receipt))
                    output = {
                        "schema": "metis.stage-output/v1",
                        "success": True,
                        "family": runtime["family"],
                        "stage": runtime["stage"]["id"],
                        "parent_checkpoint_sha256": runtime["parent_checkpoint_sha256"],
                        "config_sha256": runtime["stage_config_sha256"],
                        "checkpoint_receipt": str(receipt_path),
                        "receipt_sha256": "",
                    }
                    output["receipt_sha256"] = hashlib.sha256(
                        canonical({k: v for k, v in output.items() if k != "receipt_sha256"})
                    ).hexdigest()
                    Path(runtime["output_receipt"]).write_text(json.dumps(output))
                    """
                ),
                encoding="utf-8",
            )
            environment = {
                "METIS_TOKENIZER_MANIFEST": str(tokenizer),
                "METIS_CONTEXT_EXTENSION_DATA": str(context),
                "DUMMY_COUNT": str(counter),
            }
            arguments = dict(
                pipeline_path=PIPELINE,
                family="praxis",
                state_dir=root / "state",
                initial_checkpoint_receipt=initial,
                backend_command=[sys.executable, str(backend)],
                environment=environment,
            )
            first = PostTrainingOrchestrator(**arguments).run(until="context_extension")
            second = PostTrainingOrchestrator(**arguments).run(until="context_extension")
            self.assertEqual(len(first["completed"]), 1)
            self.assertEqual(first["state_sha256"], second["state_sha256"])
            self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), ["1"])
            output_receipt = Path(first["completed"][0]["output_receipt"])
            output_receipt.write_text(
                output_receipt.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineContractError, "output receipt changed"):
                PostTrainingOrchestrator(**arguments).run(until="context_extension")


if __name__ == "__main__":
    unittest.main()


class BaseModelBoundaryTests(unittest.TestCase):
    """The base model is the 1T checkpoint plus continued pretraining.

    Continued pretraining is contracted by the pretraining side and executed by
    this pipeline.  These tests pin the split so the two declarations cannot
    drift, and so nobody folds the long-context corpus into the 1T release.
    """

    def contract(self) -> dict:
        import yaml

        return yaml.safe_load(
            (ROOT / "configs" / "metis16" / "pretraining.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_base_model_is_not_the_one_trillion_checkpoint(self) -> None:
        contract = self.contract()
        self.assertEqual(
            contract["base_model_complete_after"], ["phase_c", "continued_pretraining"]
        )
        self.assertEqual(
            load_pipeline(PIPELINE)["base_model_boundary"], "context_extension"
        )

    def test_base_model_and_alignment_stages_partition_the_executed_order(self) -> None:
        from metis_training.posttraining import (
            ALIGNMENT_STAGE_IDS,
            BASE_MODEL_STAGE_IDS,
            EXPECTED_STAGE_IDS,
        )

        self.assertEqual(BASE_MODEL_STAGE_IDS + ALIGNMENT_STAGE_IDS, EXPECTED_STAGE_IDS)
        self.assertEqual(BASE_MODEL_STAGE_IDS, ("context_extension",))
        self.assertEqual(ALIGNMENT_STAGE_IDS[0], "cold_start_sft")

    def test_continued_pretraining_corpus_stays_out_of_the_pretraining_release(
        self,
    ) -> None:
        continued = self.contract()["continued_pretraining"]
        self.assertIs(continued["in_pretraining_release"], False)
        self.assertEqual(continued["data_env"], "METIS_CONTEXT_EXTENSION_DATA")
        # The 1T budget and phase boundaries must not absorb the 18B exposure.
        contract = self.contract()
        self.assertEqual(contract["total_train_tokens"], 1_000_000_000_000)
        self.assertEqual(
            max(int(phase["end_token_exclusive"]) for phase in contract["phases"]),
            1_000_000_000_000,
        )
        self.assertEqual(continued["token_budget"], 18_000_000_000)

    def test_divergent_declarations_of_the_base_model_fail_closed(self) -> None:
        from metis_training.posttraining import cross_check_continued_pretraining

        pipeline = load_pipeline(PIPELINE)
        for side, field, value in (
            ("contract", "token_budget", 12_000_000_000),
            ("pipeline", "train_context", 131_072),
            ("contract", "in_pretraining_release", True),
        ):
            with self.subTest(side=side, field=field):
                mutated_pipeline = copy.deepcopy(pipeline)
                mutated_contract = copy.deepcopy(self.contract())
                target = (
                    mutated_contract["continued_pretraining"]
                    if side == "contract"
                    else mutated_pipeline["context_extension"]
                )
                target[field] = value
                with self.assertRaises(PipelineContractError):
                    cross_check_continued_pretraining(
                        mutated_pipeline, mutated_contract
                    )

    def test_allocation_cap_follows_from_the_gate_overshoot_allowance(self) -> None:
        """One sequence per rank is the floor, so the world size is capped."""

        from metis_training.posttraining import cross_check_continued_pretraining

        continued = self.contract()["continued_pretraining"]
        allowance = continued["gate_policy"]["maximum_gate_overshoot_tokens"]
        train_context = continued["train_context"]
        self.assertLessEqual(
            continued["maximum_world_size"] * train_context, allowance
        )
        # The full 512-APU allocation would overshoot its first gate.
        self.assertGreater(512 * train_context, allowance)

        pipeline = load_pipeline(PIPELINE)
        contract = self.contract()
        contract["continued_pretraining"]["maximum_world_size"] = 512
        with self.assertRaises(PipelineContractError):
            cross_check_continued_pretraining(pipeline, contract)
