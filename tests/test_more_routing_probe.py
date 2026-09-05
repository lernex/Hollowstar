from __future__ import annotations

import copy
import ast
import json
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

import torch

from metis_ablation.routing_credit_probe import (
    CapabilityError,
    FrozenRuntimeState,
    RoutingCapture,
    aggregate_pairs,
    build_parser,
    assert_file_unchanged,
    assert_disjoint_windows,
    credit_alignment,
    depth_credit_probe,
    evaluate_in_memory,
    fit_utility_in_memory,
    forward_summary,
    fresh_output_directory,
    held_out_batch,
    identify_batch,
    identify_file,
    infer_run_manifest,
    json_sha256,
    joint_policy_probe,
    load_frozen_model,
    load_utility_artifact,
    plan_cost,
    plan_repeat_statistics,
    repeated_plan_evaluation,
    run_probe,
    repeat_noise,
    select_depth_pairs,
    save_utility_artifact,
    shuffle_plan,
    swap_plan,
    unpack_active,
    teacher_plan,
    validate_checkpoint,
    validate_utility_provenance,
    _select_probe_device,
)
from metis_training.data import TrainingBatch
from metis_training.metrics import estimate_train_flops
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


def tiny_config():
    return replace(Metis16Config.tiny_for_tests(), max_routed_k=4)


def tiny_batch(config, length=8, seed=81):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(config.vocab_size, (2, length), generator=generator)
    labels = inputs.roll(-1, dims=1)
    labels[:, -1] = -100
    reset = torch.zeros_like(inputs, dtype=torch.bool)
    reset[:, 0] = True
    return TrainingBatch(
        input_ids=inputs, canonical_ids=inputs.clone(), labels=labels,
        attention_mask=torch.ones_like(inputs, dtype=torch.bool),
        document_ids=torch.zeros_like(inputs, dtype=torch.int32), reset_mask=reset,
        phase="phase_a", global_token_cursor=100,
        next_global_token_cursor=100 + inputs.numel(),
        non_padding_tokens=inputs.numel(), supervised_tokens=int((labels != -100).sum()),
    )


def fixed_widths(config, depths, width=4):
    widths = torch.full((config.max_passes, config.n_layers, *depths.shape), width, dtype=torch.long)
    active = torch.arange(config.max_passes)[:, None, None] < depths
    return widths * active[:, None]


class OwnedFilesTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".routing-probe-tests-{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root)


class CheckpointQualityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_dense_control_has_zero_expert_work_and_correct_loss(self):
        config = replace(
            tiny_config(), ffn_mode="dense", dense_ffn_intermediate_dim=64,
            n_routed_experts=0, n_shared_experts=0,
        )
        model = Metis16ForCausalLM(config).eval()
        batch = tiny_batch(config)
        curriculum = CurriculumState(continuation_mode="depth_one", stochastic_routing=False)
        result = evaluate_in_memory(model, batch, curriculum, seed=9)
        with torch.no_grad():
            expected = model(
                batch.input_ids, batch.labels, curriculum=curriculum,
                attention_mask=batch.attention_mask, document_ids=batch.document_ids,
                reset_mask=batch.reset_mask, canonical_ids=batch.canonical_ids,
            )
        torch.testing.assert_close(result.output.loss, expected.loss)
        self.assertEqual(result.cost["expert_assignments"], 0)
        self.assertEqual(int(result.widths.sum()), 0)
        self.assertEqual(
            result.cost["nominal_train_flops"],
            estimate_train_flops(config, tokens=batch.non_padding_tokens, observed_mean_passes=1.0),
        )

    def test_frozen_pathway_control_is_assessed_without_changing_its_policy(self):
        config = tiny_config()
        model = Metis16ForCausalLM(config).eval()
        batch = tiny_batch(config)
        curriculum = CurriculumState(
            continuation_mode="fixed_max", max_passes=2, routed_k_mode="fixed",
            fixed_routed_k=4, pathway_mode="frozen", stochastic_routing=False,
        )
        result, statistics = repeated_plan_evaluation(
            model, batch, curriculum, seed=9, runtime_state=FrozenRuntimeState(model),
            repeat_forwards=2, minimum_loss_delta=1e-5,
        )
        self.assertTrue(bool(result.output.chosen_depths.eq(2).all()))
        self.assertEqual(result.cost["expert_assignments"], 2 * config.n_layers * 4 * batch.non_padding_tokens)
        self.assertTrue(statistics["same_complete_plan_every_repeat"])

    def test_quality_only_cannot_silently_refit_a_checkpoint(self):
        args = build_parser().parse_args([
            "--checkpoint", "missing", "--release-root", "missing", "--output", "unused",
            "--quality-only", "--fit-steps", "1",
        ])
        with self.assertRaisesRegex(ValueError, "checkpoint's own trained policy"):
            run_probe(args)


class PackedMappingTests(unittest.TestCase):
    def test_packed_rows_scatter_across_batches_and_padding(self):
        mask = torch.tensor([[True, False, True], [False, True, False]])
        packed = torch.tensor([[[11., 12.], [21., 22.], [31., 32.]]], requires_grad=True)
        result = unpack_active(packed, mask)
        self.assertEqual(tuple(result.shape), (2, 3, 2))
        torch.testing.assert_close(result[mask], packed.squeeze(0))
        self.assertEqual(int(torch.count_nonzero(result[~mask])), 0)
        derivative = torch.autograd.grad(result[1, 1].sum(), packed)[0]
        torch.testing.assert_close(derivative, torch.tensor([[[0., 0.], [0., 0.], [1., 1.]]]))

    def test_full_layout_preserves_active_rows_without_mutation(self):
        mask = torch.tensor([[True, False], [True, True]])
        values = torch.tensor([[1., 999.], [2., 3.]])
        result = unpack_active(values, mask)
        self.assertEqual(result.tolist(), [[1., 0.], [2., 3.]])
        self.assertEqual(values[0, 1], 999)

    def test_empty_layout_and_shape_errors(self):
        empty = unpack_active(torch.empty(1, 0, 2), torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(tuple(empty.shape), (2, 3, 2))
        self.assertEqual(int(torch.count_nonzero(empty)), 0)
        with self.assertRaisesRegex(ValueError, "active token count"):
            unpack_active(torch.ones(1, 2), torch.tensor([[True, False, False]]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            unpack_active(torch.ones(1, 2), torch.ones(1, 2))


class PlanTests(unittest.TestCase):
    def test_shuffle_preserves_complete_cost_and_supervised_pass_counts(self):
        config = tiny_config()
        depths = torch.tensor([[1, 3, 2, 3, 1, 2], [3, 2, 1, 2, 3, 1]])
        mask = torch.ones_like(depths, dtype=torch.bool)
        supervised = mask.clone()
        supervised[:, -1] = False
        widths = fixed_widths(config, depths, 2)
        widths[0, :, :, 0] = 4
        saved_depths, saved_widths = depths.clone(), widths.clone()
        shuffled_depths, shuffled_widths = shuffle_plan(
            depths, widths, mask, seed=91, supervised_mask=supervised,
        )
        self.assertEqual(plan_cost(config, depths, widths), plan_cost(config, shuffled_depths, shuffled_widths))
        for p in range(config.max_passes):
            self.assertEqual(int(((depths > p) & supervised).sum()),
                             int(((shuffled_depths > p) & supervised).sum()))
        self.assertFalse(torch.equal(depths, shuffled_depths) and torch.equal(widths, shuffled_widths))
        torch.testing.assert_close(depths, saved_depths)
        torch.testing.assert_close(widths, saved_widths)
        repeated = shuffle_plan(depths, widths, mask, seed=91, supervised_mask=supervised)
        torch.testing.assert_close(repeated[0], shuffled_depths)
        torch.testing.assert_close(repeated[1], shuffled_widths)

    def test_teacher_explores_depth_and_width_but_bootstraps_at_four(self):
        config = tiny_config()
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[1, -1] = False
        depths, widths = teacher_plan(
            config, mask, max_depth=3, generator=torch.Generator().manual_seed(17),
        )
        self.assertEqual(set(depths[mask].tolist()), {2, 3})
        self.assertTrue(torch.all(widths[0, :, mask] == 4))
        self.assertEqual(int(depths[~mask].sum()), 0)
        self.assertGreater(len(set(widths[1][widths[1] > 0].tolist())), 1)
        plan_cost(config, depths, widths)

    def test_selection_is_deterministic_disjoint_and_document_local(self):
        depths = torch.tensor([[1, 3, 2, 3, 1, 2], [2, 1, 3, 2, 3, 1]])
        mask = torch.ones_like(depths, dtype=torch.bool)
        mask[0, 5] = False
        docs = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        selected = select_depth_pairs(depths, mask, pairs=8, seed=29, document_ids=docs)
        self.assertEqual(selected, select_depth_pairs(depths, mask, pairs=8, seed=29, document_ids=docs))
        self.assertGreater(len(selected), 0)
        used = []
        for pair in selected:
            used.extend((pair.shallow, pair.deep))
            self.assertLess(pair.shallow_depth, pair.deep_depth)
            self.assertEqual(pair.shallow // 6, pair.deep // 6)
            self.assertEqual(docs.flatten()[pair.shallow], docs.flatten()[pair.deep])
            self.assertTrue(mask.flatten()[pair.shallow] and mask.flatten()[pair.deep])
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(select_depth_pairs(torch.ones_like(depths), mask, pairs=8, seed=1), [])

    def test_whole_plan_swap_conserves_each_layer_pass_and_inputs(self):
        config = tiny_config()
        depths = torch.tensor([[1, 3, 2, 3]])
        widths = fixed_widths(config, depths, 2)
        widths[0, :, 0, 0] = 4
        widths[2, :, 0, 1] = 1
        original_depths, original_widths = depths.clone(), widths.clone()
        pair = select_depth_pairs(depths, depths > 0, pairs=1, seed=4)[0]
        swapped_depths, swapped_widths = swap_plan(depths, widths, pair)
        self.assertEqual(plan_cost(config, depths, widths), plan_cost(config, swapped_depths, swapped_widths))
        torch.testing.assert_close(depths, original_depths)
        torch.testing.assert_close(widths, original_widths)
        torch.testing.assert_close(
            swapped_widths.flatten(2)[:, :, pair.shallow],
            widths.flatten(2)[:, :, pair.deep],
        )

    def test_depth_only_fixed_k_swap_conserves_nominal_budget(self):
        config = tiny_config()
        depths = torch.tensor([[1, 2, 3, 1]])
        widths = fixed_widths(config, depths)
        pair = select_depth_pairs(depths, depths > 0, pairs=1, seed=8)[0]
        swapped_depths, swapped_widths = swap_plan(depths, widths, pair)
        torch.testing.assert_close(swapped_widths, fixed_widths(config, swapped_depths))
        self.assertEqual(plan_cost(config, depths, widths), plan_cost(config, swapped_depths, swapped_widths))

    def test_same_average_k_does_not_imply_same_work(self):
        config = tiny_config()
        short = torch.tensor([[1, 1]])
        long = torch.tensor([[3, 3]])
        first = plan_cost(config, short, fixed_widths(config, short, 2))
        second = plan_cost(config, long, fixed_widths(config, long, 2))
        self.assertLess(first["expert_assignments"], second["expert_assignments"])
        self.assertLess(first["nominal_train_flops"], second["nominal_train_flops"])
        bad = fixed_widths(config, short)
        bad[2, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Inactive"):
            plan_cost(config, short, bad)

    def test_alignment_direction_ties_and_aggregation(self):
        self.assertEqual(credit_alignment(-2, -0.1), "aligned")
        self.assertEqual(credit_alignment(-2, 0.1), "opposed")
        self.assertEqual(credit_alignment(0, 0.1), "predicted_tie")
        self.assertEqual(credit_alignment(2, 0), "observed_tie")
        with self.assertRaisesRegex(ValueError, "finite"):
            credit_alignment(float("nan"), 0.1)
        records = [
            {"alignment": "aligned", "global_loss_delta": -0.1},
            {"alignment": "opposed", "global_loss_delta": 0.2},
            {"alignment": "predicted_tie", "global_loss_delta": 0.3},
        ]
        aggregate = aggregate_pairs(records)
        self.assertEqual(aggregate["pairs"], 3)
        self.assertEqual(aggregate["non_tied_pairs"], 2)
        self.assertEqual(aggregate["credit_alignment_fraction"], 0.5)
        self.assertAlmostEqual(aggregate["mean_global_loss_delta"], 0.4 / 3)
        self.assertIsNone(aggregate_pairs([])["credit_alignment_fraction"])

    def test_repeat_noise_excludes_numerically_unresolved_deltas(self):
        noise = repeat_noise([3.0, 3.0001, 3.0002])
        self.assertAlmostEqual(noise["repeat_loss_range"], 0.0002)
        self.assertAlmostEqual(noise["decisive_absolute_loss_delta_threshold"], 0.0006)
        self.assertGreater(noise["decisive_absolute_loss_delta_threshold"], 1e-5)
        self.assertGreaterEqual(
            repeat_noise([3.0, 3.0])["decisive_absolute_loss_delta_threshold"], 1e-5,
        )
        result = aggregate_pairs([
            {"alignment": "aligned", "global_loss_delta": -0.01},
            {"alignment": "below_numerical_noise", "global_loss_delta": 1e-6},
        ])
        self.assertEqual(result["non_tied_pairs"], 1)
        self.assertEqual(result["credit_alignment_fraction"], 1.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            repeat_noise([3.0])

    def test_natural_plan_variability_is_reported_not_treated_as_fixed_noise(self):
        summary = plan_repeat_statistics([
            {"plan_sha256": "a" * 64, "lm_loss": 3.0},
            {"plan_sha256": "b" * 64, "lm_loss": 3.2},
            {"plan_sha256": "a" * 64, "lm_loss": 3.1},
        ], minimum_loss_delta=1e-5)
        self.assertEqual(summary["distinct_complete_plans"], 2)
        self.assertFalse(summary["same_complete_plan_every_repeat"])
        self.assertAlmostEqual(summary["mean_lm_loss"], 3.1)
        self.assertAlmostEqual(summary["sample_standard_deviation"], 0.1)
        self.assertAlmostEqual(summary["repeat_loss_range"], 0.2)
        invalid = aggregate_pairs([
            {"alignment": "reference_replay_mismatch", "global_loss_delta": -0.1},
        ])
        self.assertEqual(invalid["non_tied_pairs"], 0)
        self.assertIsNone(invalid["credit_alignment_fraction"])


class IdentityTests(OwnedFilesTestCase):
    def test_cli_warmup_is_explicit_and_separately_configurable(self):
        required = ["--checkpoint", "state.pt", "--release-root", "release", "--output", "fresh"]
        defaults = build_parser().parse_args(required)
        self.assertEqual(defaults.warmup_forwards, 1)
        self.assertEqual(defaults.fit_steps, 0)
        self.assertEqual(defaults.fit_max_depth, 5)
        disabled = build_parser().parse_args(required + ["--warmup-forwards", "0"])
        self.assertEqual(disabled.warmup_forwards, 0)

    def test_fit_window_overlap_includes_lookahead_and_equal_content(self):
        first = {"tensor_window_sha256": "a" * 64, "global_token_cursor": 100, "next_global_token_cursor": 116}
        second = {"tensor_window_sha256": "b" * 64, "global_token_cursor": 117, "next_global_token_cursor": 133}
        assert_disjoint_windows(first, second)
        with self.assertRaisesRegex(ValueError, "lookahead"):
            assert_disjoint_windows(first, dict(second, global_token_cursor=116))
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_windows(first, dict(second, tensor_window_sha256="a" * 64))

    def test_fresh_directory_refuses_existing_empty_and_nonempty_paths(self):
        output = fresh_output_directory(self.root / "fresh")
        with self.assertRaises(FileExistsError):
            fresh_output_directory(output)
        sentinel = output / "sentinel"
        sentinel.write_text("do not replace")
        with self.assertRaises(FileExistsError):
            fresh_output_directory(output)
        self.assertEqual(sentinel.read_text(), "do not replace")
        link = self.root / "dangling"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            fresh_output_directory(link)

    def test_output_cannot_write_inside_checkpoint_or_release(self):
        protected = self.root / "release"
        protected.mkdir()
        with self.assertRaisesRegex(ValueError, "input directory"):
            fresh_output_directory(protected / "probe", inputs=(protected,))
        self.assertEqual(list(protected.iterdir()), [])

    def test_file_identity_detects_content_and_later_mutation(self):
        source = self.root / "source.json"
        source.write_text('{"version":1}')
        first = identify_file(source)
        self.assertEqual(first, identify_file(source))
        self.assertEqual(len(first["sha256"]), 64)
        assert_file_unchanged(first)
        source.write_text('{"version":100}')
        self.assertNotEqual(first["sha256"], identify_file(source)["sha256"])
        with self.assertRaisesRegex(RuntimeError, "changed"):
            assert_file_unchanged(first)

    def test_checkpoint_identity_validates_without_optimizer_shards(self):
        identity = {
            "model": {"name": "test"}, "curriculum": {"continuation_mode": "budgeted"},
            "sampler": {"block_tokens": 16}, "precision_profile": "bf16",
        }
        manifest = {
            "schema": "more.ablation-run/v1", **identity, "run_identity": identity,
            "run_identity_sha256": json_sha256(identity), "spec": {"name": "test"},
        }
        payload = {
            "schema": "more.ablation-checkpoint/v3",
            "model": {"weight": torch.tensor([1.])}, "spec": {"name": "test"},
            "step": 5000, "step_semantics": "next_unexecuted",
            "run_identity_sha256": manifest["run_identity_sha256"],
            "optimizer_shards": [{"path": "does-not-exist", "rank": 79}],
        }
        self.assertEqual(validate_checkpoint(payload, manifest), identity)
        bad = copy.deepcopy(manifest)
        bad["curriculum"] = {"continuation_mode": "random"}
        with self.assertRaisesRegex(ValueError, "curriculum"):
            validate_checkpoint(payload, bad)
        bad = dict(payload, run_identity_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "identities differ"):
            validate_checkpoint(bad, manifest)
        bad = dict(payload, step_semantics="last_executed")
        with self.assertRaisesRegex(ValueError, "next_unexecuted"):
            validate_checkpoint(bad, manifest)

    def test_manifest_inference_is_structural_not_a_search(self):
        step = self.root / "run" / "checkpoints" / "step-0005000"
        step.mkdir(parents=True)
        self.assertEqual(infer_run_manifest(step), self.root / "run" / "run.json")
        self.assertEqual(infer_run_manifest(step / "state.pt"), self.root / "run" / "run.json")
        with self.assertRaisesRegex(ValueError, "explicitly"):
            infer_run_manifest(self.root / "ambiguous.pt")

    def test_input_identity_never_includes_tokens_and_catches_mask_changes(self):
        batch = tiny_batch(tiny_config())
        original = identify_batch(batch)
        self.assertNotIn("input_ids", original)
        self.assertNotIn("labels", original)
        self.assertEqual(original, identify_batch(batch))
        batch.labels[0, 0] = -100
        self.assertNotEqual(original["tensor_window_sha256"], identify_batch(batch)["tensor_window_sha256"])
        with self.assertRaisesRegex(ValueError, "precedes"):
            held_out_batch(None, {}, checkpoint_step=5000, step=4999, sequences=1, sequence_length=8)

    def test_joint_adapter_rejects_unavailable_router(self):
        model = Metis16ForCausalLM(tiny_config())
        with self.assertRaisesRegex(CapabilityError, "no enabled"):
            load_utility_artifact(
                model, self.root / "absent.pt",
                checkpoint_sha256="0" * 64, run_identity_sha256="1" * 64,
                base_model_config_sha256="2" * 64,
            )

    @unittest.skipUnless(
        "joint_router_hidden_dim" in Metis16Config.__dataclass_fields__,
        "Joint geometry is not present in the independent legacy baseline",
    )
    def test_utility_provenance_separates_train_eval_and_counts_teacher_work(self):
        config = tiny_config()
        training = {
            "tensor_window_sha256": "a" * 64,
            "global_token_cursor": 100, "next_global_token_cursor": 116,
        }
        payload = {
            key: getattr(config, key) for key in
            ("joint_router_hidden_dim", "max_passes", "n_layers", "max_routed_k")
        }
        payload.update({
            "source_revision": "b" * 40, "training_seed": 7,
            "teacher_token_count": 32, "teacher_forward_calls": 2,
            "teacher_backward_calls": 0, "teacher_elapsed_seconds": 0.25,
            "training_windows": [training],
        })
        evaluation = {
            "tensor_window_sha256": "c" * 64,
            "global_token_cursor": 200, "next_global_token_cursor": 216,
        }
        result = validate_utility_provenance(payload, config, evaluation_window=evaluation)
        self.assertEqual(result["teacher_token_count"], 32)
        self.assertEqual(result["training_windows"], [training])
        for start in (100, 108, 116):
            overlapping = dict(evaluation, global_token_cursor=start)
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_utility_provenance(payload, config, evaluation_window=overlapping)
        with self.assertRaisesRegex(ValueError, "teacher_forward_calls"):
            validate_utility_provenance(dict(payload, teacher_forward_calls=0), config)
        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_utility_provenance(dict(payload, n_layers=config.n_layers + 1), config)


@unittest.skipUnless(
    "causal_compute_budget" in Metis16Config.__dataclass_fields__,
    "Causal cost accounting requires the optional causal model",
)
class CausalProbeCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        from metis_training.compute_router import JointComputeCosts

        self.config = replace(
            tiny_config(), joint_compute_router=True, causal_compute_budget=True,
            joint_router_hidden_dim=8, target_mean_routed_k=4.0,
        )
        self.costs = JointComputeCosts.from_config(self.config)
        self.batch = tiny_batch(self.config)
        self.fixed = CurriculumState(
            continuation_mode="fixed_max", routed_k_mode="fixed",
            fixed_routed_k=4, max_passes=2, memory_gate_scale=0.0,
            stochastic_routing=False,
        )

    def test_static_plan_uses_lean_causal_reference_without_invalid_config(self):
        depths = self.batch.attention_mask.long() * 2
        widths = fixed_widths(self.config, depths)
        fixed = plan_cost(self.config, depths, widths, terminal_only=True)
        outcomes = plan_cost(self.config, depths, widths)
        count = self.batch.non_padding_tokens
        self.assertEqual(fixed["nominal_train_flops"], count * self.costs.reference_per_token)
        self.assertEqual(outcomes["nominal_train_flops"] - fixed["nominal_train_flops"],
                         count * self.costs.head_per_token)
        self.assertEqual(fixed["modeled_lm_head_tokens"], count)
        self.assertEqual(outcomes["modeled_lm_head_tokens"], 2 * count)
        legacy = plan_cost(
            replace(self.config, joint_compute_router=False, causal_compute_budget=False),
            depths, widths,
        )
        self.assertEqual(
            legacy["nominal_train_flops"] - fixed["nominal_train_flops"],
            count * (2 * self.costs.removed_policy_per_pass + self.costs.head_per_token),
        )

    def test_fixed_quality_evaluation_and_replays_match_actual_model_ledger(self):
        model = Metis16ForCausalLM(self.config).eval()
        runtime = FrozenRuntimeState(model)
        result = evaluate_in_memory(model, self.batch, self.fixed, seed=7, runtime_state=runtime)
        self.assertEqual(result.cost["nominal_train_flops"], int(result.output.telemetry["joint_model_flops"]))
        self.assertEqual(int(result.output.telemetry["joint_router_flops"]), 0)
        summary = forward_summary(result, self.batch)
        self.assertEqual(summary["nominal_probe_forward_flops"], result.cost["nominal_forward_flops"])
        self.assertTrue(result.cost["terminal_only_head"])
        repeated, statistics = repeated_plan_evaluation(
            model, self.batch, self.fixed, seed=7, runtime_state=runtime,
            repeat_forwards=2, minimum_loss_delta=1e-5,
            force_depth=result.output.chosen_depths, force_routed_k=result.widths,
        )
        self.assertEqual(repeated.cost, result.cost)
        self.assertEqual(statistics["nominal_forward_flops_all_calls"],
                         2 * int(result.output.telemetry["joint_model_flops"]) / 3.0)
        result.output.telemetry["executed_lm_head_tokens"] = torch.tensor(self.batch.supervised_tokens)
        counted = forward_summary(result, self.batch)
        self.assertEqual(counted["lm_head_tokens"], self.batch.supervised_tokens)
        self.assertEqual(counted["lm_head_tokens_basis"], "model_execution_counter")

    def test_joint_quality_counts_critic_once_and_outcome_head_per_active_pass(self):
        model = Metis16ForCausalLM(self.config).eval()
        with torch.no_grad():
            model.joint_router.output.bias[:2].copy_(torch.tensor([2.0, 4.0]))
        joint = replace(
            self.fixed, compute_allocation_mode="joint", continuation_mode="budgeted",
            routed_k_mode="budgeted", max_passes=3, allow_untrained_joint_router=True,
        )
        result, statistics = repeated_plan_evaluation(
            model, self.batch, joint, seed=7, runtime_state=FrozenRuntimeState(model),
            repeat_forwards=2, minimum_loss_delta=1e-5,
        )
        ledger = int(result.output.telemetry["joint_model_flops"])
        critic = int(result.output.telemetry["joint_router_flops"])
        self.assertGreater(critic, 0)
        self.assertEqual(result.cost["nominal_train_flops"] + critic, ledger)
        self.assertFalse(result.cost["terminal_only_head"])
        self.assertEqual(result.cost["modeled_lm_head_tokens"], result.cost["token_passes"])
        self.assertEqual(statistics["nominal_forward_flops_all_calls"], 2 * ledger / 3.0)
        self.assertEqual(forward_summary(result, self.batch)["nominal_probe_forward_flops"], ledger / 3.0)


    @unittest.skipUnless(
        "terminal_action_critic" in Metis16Config.__dataclass_fields__,
        "Terminal critic is unavailable",
    )
    def test_terminal_plan_prepays_one_head_without_double_subtraction(self):
        config = replace(self.config, terminal_action_critic=True)
        costs = type(self.costs).from_config(config)
        depths = self.batch.attention_mask.long() * 2
        widths = fixed_widths(config, depths)
        inferred = plan_cost(config, depths, widths)
        fixed = plan_cost(config, depths, widths, terminal_only=True)
        self.assertEqual(inferred, fixed)
        self.assertEqual(fixed["nominal_train_flops"], self.batch.non_padding_tokens * costs.reference_per_token)
        self.assertEqual(fixed["modeled_lm_head_tokens"], self.batch.non_padding_tokens)
        self.assertEqual(fixed["head_cost_mode"], "prepaid_terminal")
        self.assertTrue(fixed["terminal_only_head"])

    @unittest.skipUnless(
        "terminal_action_critic" in Metis16Config.__dataclass_fields__,
        "Terminal critic is unavailable",
    )
    def test_terminal_fixed_rerouted_and_frozen_quality_match_model_cost(self):
        config = replace(self.config, terminal_action_critic=True)
        for pathway in ("per_pass", "frozen"):
            with self.subTest(pathway=pathway):
                model = Metis16ForCausalLM(config).eval()
                result, statistics = repeated_plan_evaluation(
                    model, self.batch, replace(self.fixed, pathway_mode=pathway),
                    seed=13, runtime_state=FrozenRuntimeState(model),
                    repeat_forwards=2, minimum_loss_delta=1e-5,
                )
                ledger = int(result.output.telemetry["joint_model_flops"])
                self.assertEqual(int(result.output.telemetry["joint_router_flops"]), 0)
                self.assertEqual(result.cost["nominal_train_flops"], ledger)
                self.assertEqual(result.cost["head_cost_mode"], "prepaid_terminal")
                self.assertEqual(statistics["nominal_forward_flops_all_calls"], 2 * ledger / 3.0)

    @unittest.skipUnless(
        "terminal_action_critic" in Metis16Config.__dataclass_fields__,
        "Terminal critic is unavailable",
    )
    def test_terminal_joint_quality_uses_one_head_even_with_joint_curriculum(self):
        config = replace(self.config, terminal_action_critic=True)
        model = Metis16ForCausalLM(config).eval()
        with torch.no_grad():
            model.joint_router.output.bias[1:3].fill_(2.0)
        joint = replace(
            self.fixed, compute_allocation_mode="joint", continuation_mode="budgeted",
            routed_k_mode="budgeted", max_passes=3, allow_untrained_joint_router=True,
        )
        result, statistics = repeated_plan_evaluation(
            model, self.batch, joint, seed=13, runtime_state=FrozenRuntimeState(model),
            repeat_forwards=2, minimum_loss_delta=1e-5,
        )
        ledger = int(result.output.telemetry["joint_model_flops"])
        critic = int(result.output.telemetry["joint_router_flops"])
        self.assertGreater(critic, 0)
        self.assertEqual(result.cost["nominal_train_flops"] + critic, ledger)
        self.assertTrue(result.cost["terminal_only_head"])
        self.assertEqual(result.cost["modeled_lm_head_tokens"], self.batch.non_padding_tokens)
        self.assertEqual(statistics["nominal_forward_flops_all_calls"], 2 * ledger / 3.0)
        self.assertEqual(forward_summary(result, self.batch)["nominal_probe_forward_flops"], ledger / 3.0)


class TinyModelProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(29)
        self.config = tiny_config()
        self.model = Metis16ForCausalLM(self.config)
        self.batch = tiny_batch(self.config)
        self.curriculum = CurriculumState(
            continuation_mode="budgeted", routed_k_mode="fixed",
            fixed_routed_k=4, stochastic_routing=False,
        )

    def test_real_hooks_match_forced_packed_passes_and_no_inputs_change(self):
        depths = torch.tensor([[1, 3, 2, 1, 2, 3, 1, 3], [3, 1, 2, 3, 1, 2, 3, 1]])
        before = identify_batch(self.batch)
        result = evaluate_in_memory(
            self.model, self.batch, self.curriculum, seed=23, force_depth=depths,
        )
        torch.testing.assert_close(result.widths, fixed_widths(self.config, depths))
        self.assertEqual(result.cost["token_passes"], int(depths.sum()))
        self.assertEqual(result.cost["expert_assignments"], int(depths.sum()) * self.config.n_layers * 4)
        self.assertEqual(identify_batch(self.batch), before)
        self.assertTrue(self.model.training)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        summary = forward_summary(result, self.batch)
        self.assertEqual(sum(summary["chosen_depth_histogram"].values()), self.batch.non_padding_tokens)
        self.assertNotIn("final_hidden_state", json.dumps(summary))

    def test_missing_hook_work_fails_instead_of_reporting_partial_metrics(self):
        capture = RoutingCapture(self.model)
        masks = torch.ones(1, *self.batch.input_ids.shape, dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            capture.full_widths(masks)

    def test_summary_prefers_actual_head_rows_and_snapshots_live_replay_counters(self):
        result = evaluate_in_memory(self.model, self.batch, self.curriculum, seed=23)
        telemetry = result.output.telemetry
        self.assertIn("lm_head_forward_rows", telemetry)
        telemetry["executed_lm_head_tokens"] = torch.tensor(999999)
        before = forward_summary(result, self.batch)
        self.assertEqual(before["lm_head_tokens"], int(telemetry["lm_head_forward_rows"]))
        self.assertEqual(before["lm_head_tokens_counter"], "lm_head_forward_rows")
        self.assertEqual(before["lm_head_tokens_basis"], "model_execution_counter")
        self.assertEqual(before["lm_head_forward_flops"], int(telemetry["lm_head_forward_flops"]))
        self.assertEqual(before["lm_head_recompute_rows"], 0)
        self.assertEqual(before["lm_head_recompute_flops"], 0)
        telemetry["lm_head_recompute_rows"].add_(3)
        telemetry["lm_head_recompute_flops"].add_(6 * self.config.vocab_size * self.config.d_model)
        after = forward_summary(result, self.batch)
        self.assertEqual(before["lm_head_recompute_rows"], 0)
        self.assertEqual(after["lm_head_recompute_rows"], 3)
        self.assertEqual(after["lm_head_recompute_flops"], 6 * self.config.vocab_size * self.config.d_model)
        self.assertEqual(before["modeled_total_train_flops"], after["modeled_total_train_flops"])

    def test_summary_legacy_head_counter_and_nominal_fallback_are_explicit(self):
        result = evaluate_in_memory(self.model, self.batch, self.curriculum, seed=23)
        for name in (
            "lm_head_forward_rows", "lm_head_forward_flops",
            "lm_head_recompute_rows", "lm_head_recompute_flops",
        ):
            result.output.telemetry.pop(name, None)
        result.output.telemetry["executed_lm_head_tokens"] = torch.tensor(7)
        legacy = forward_summary(result, self.batch)
        self.assertEqual(legacy["lm_head_tokens"], 7)
        self.assertEqual(legacy["lm_head_tokens_counter"], "executed_lm_head_tokens")
        del result.output.telemetry["executed_lm_head_tokens"]
        fallback = forward_summary(result, self.batch)
        self.assertEqual(fallback["lm_head_tokens"], result.cost["token_passes"])
        self.assertEqual(fallback["lm_head_tokens_basis"], "nominal_nonpadding_fallback")
        self.assertIsNone(fallback["lm_head_tokens_counter"])
        self.assertNotIn("lm_head_forward_rows", fallback)
        self.assertNotIn("lm_head_recompute_flops", fallback)

    def test_device_selection_precedes_backend_construction_and_forward(self):
        self.assertEqual(_select_probe_device(torch.device("cpu")), torch.device("cpu"))
        source = Path(__file__).resolve().parents[1] / "src/metis_ablation/routing_credit_probe.py"
        tree = ast.parse(source.read_text())
        for name, operation in (("load_frozen_model", "build_precision_policy"), ("evaluate_in_memory", "model")):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            selection = next(node for node in calls if node.func.id == "_select_probe_device")
            backend = next(node for node in calls if node.func.id == operation)
            self.assertLess(selection.lineno, backend.lineno)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Nonzero-device placement requires two visible CUDA/ROCm devices",
    )
    def test_nonzero_cuda_is_current_during_construction_and_causal_forward(self):
        previous = torch.cuda.current_device()
        try:
            config = replace(
                self.config, joint_compute_router=True, causal_compute_budget=True,
                joint_router_hidden_dim=8, target_mean_routed_k=4.0,
            )
            source = Metis16ForCausalLM(config)
            torch.cuda.set_device(0)
            model, _ = load_frozen_model(
                config, source.state_dict(), device=torch.device("cuda:1"),
                precision="bf16", checkpoint_precision="bf16", enable_joint=False,
            )
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertTrue(all(parameter.device == torch.device("cuda:1") for parameter in model.parameters()))
            batch = tiny_batch(config).to(torch.device("cuda:1"))
            torch.cuda.set_device(0)
            result = evaluate_in_memory(
                model, batch,
                CurriculumState(
                    compute_allocation_mode="joint", continuation_mode="budgeted",
                    routed_k_mode="budgeted", memory_gate_scale=0.0,
                    stochastic_routing=False, allow_untrained_joint_router=True,
                ),
                seed=23,
            )
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertTrue(torch.isfinite(result.output.loss))
            self.assertEqual(_select_probe_device(torch.device("cuda")), torch.device("cuda:1"))
        finally:
            torch.cuda.set_device(previous)

    def test_full_captured_plan_is_replayed_and_verified_for_noise(self):
        runtime = FrozenRuntimeState(self.model)
        natural, variability = repeated_plan_evaluation(
            self.model, self.batch, self.curriculum, seed=23, runtime_state=runtime,
            repeat_forwards=2, minimum_loss_delta=1e-5,
        )
        forced, noise = repeated_plan_evaluation(
            self.model, self.batch, self.curriculum, seed=23, runtime_state=runtime,
            repeat_forwards=2, minimum_loss_delta=1e-5,
            force_depth=natural.output.chosen_depths, force_routed_k=natural.widths,
        )
        self.assertEqual(variability["repeat_kind"], "natural_policy")
        self.assertEqual(noise["repeat_kind"], "fixed_depth_and_width_plan")
        self.assertEqual(noise["distinct_complete_plans"], 1)
        self.assertEqual(noise["forward_calls"], 2)
        self.assertEqual(forced.cost, natural.cost)
        for record in noise["repeats"]:
            self.assertEqual(record["cost"], natural.cost)
        torch.testing.assert_close(forced.widths, natural.widths, rtol=0, atol=0)

    def test_real_packed_continuation_gradients_and_depth_swaps(self):
        weights = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        runtime = FrozenRuntimeState(self.model)
        reference = evaluate_in_memory(
            self.model, self.batch, self.curriculum, seed=23,
            runtime_state=runtime, continuation_grad=True,
        )
        self.assertGreater(int(reference.continuation_observed.sum()), 0)
        self.assertTrue(torch.isfinite(reference.continuation_gradients).all())
        self.assertEqual(int(torch.count_nonzero(
            reference.continuation_gradients[~reference.continuation_observed],
        )), 0)
        report = depth_credit_probe(
            self.model, self.batch, self.curriculum, pairs=2, seed=23, runtime_state=runtime,
        )
        self.assertGreater(report["aggregate"]["pairs"], 0)
        self.assertLessEqual(report["aggregate"]["pairs"], 2)
        self.assertEqual(report["forward_calls"], 4 + report["aggregate"]["pairs"])
        self.assertEqual(report["deterministic_repeat_noise"]["repeat_count"], 3)
        for pair in report["paired_swaps"]:
            self.assertEqual(pair["nominal_train_flops_delta"], 0)
            self.assertTrue(pair["same_depth_multiset"])
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        json.dumps(report, allow_nan=False)

    def test_bf16_loading_preserves_router_fp32_and_rejects_missing_weights(self):
        model, report = load_frozen_model(
            self.config, self.model.state_dict(), device=torch.device("cpu"),
            precision="bf16", enable_joint=False,
        )
        self.assertEqual(model.embedding.weight.dtype, torch.bfloat16)
        self.assertEqual(model.continuation.output.weight.dtype, torch.float32)
        self.assertEqual(report["effective_profile"], "bf16")
        self.assertEqual(report["optimizer_shards_loaded"], 0)
        self.assertTrue(report["bf16_reference_context"])
        self.assertEqual(report["discarded_checkpoint_state_keys"], [])
        weights = dict(self.model.state_dict())
        del weights["continuation.output.weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_frozen_model(
                self.config, weights, device=torch.device("cpu"),
                precision="bf16", enable_joint=False,
            )
        extra = dict(self.model.state_dict(), unmatched_extra_state=torch.tensor([1]))
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            load_frozen_model(
                self.config, extra, device=torch.device("cpu"),
                precision="bf16", enable_joint=False,
            )


@unittest.skipUnless(
    "joint_compute_router" in Metis16Config.__dataclass_fields__,
    "Utility fitting requires the optional joint-router implementation",
)
class UtilityFittingTests(OwnedFilesTestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        super().setUp()
        torch.manual_seed(712)
        self.base = replace(tiny_config(), target_mean_routed_k=4.0)
        self.config = replace(self.base, joint_compute_router=True)
        self.model = Metis16ForCausalLM(self.config).eval()
        self.curriculum = CurriculumState(
            continuation_mode="budgeted", routed_k_mode="fixed",
            fixed_routed_k=4, stochastic_routing=False,
        )
        self.windows = []
        for index in range(2):
            batch = tiny_batch(self.base, seed=501 + index)
            batch.global_token_cursor = 100 + index * 100
            batch.next_global_token_cursor = batch.global_token_cursor + batch.input_ids.numel()
            self.windows.append((batch, identify_batch(batch)))
        evaluation = tiny_batch(self.base, seed=901)
        evaluation.global_token_cursor = 1000
        evaluation.next_global_token_cursor = 1016
        self.evaluation = identify_batch(evaluation)

    def fit(self):
        return fit_utility_in_memory(
            self.model, self.curriculum, self.windows, steps=2, seed=18,
            learning_rate=0.0003, max_depth=3, evaluation_window=self.evaluation,
        )

    def test_updates_only_utility_head_and_preserves_trained_counter(self):
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        report = self.fit()
        self.assertEqual(report["teacher_forward_calls"], 2)
        self.assertEqual(report["teacher_backward_calls"], 2)
        self.assertEqual(report["teacher_token_count"], 32)
        self.assertGreater(report["observed_target_count"], 0)
        self.assertEqual(int(self.model.joint_router.trained_updates), 2)
        self.assertEqual(report["unknown_continuation_targets"], "unlabeled")
        head_changed = False
        for name, value in self.model.named_parameters():
            if name.startswith("joint_router."):
                head_changed |= not torch.equal(value, before[name])
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertIsNone(value.grad)
        self.assertTrue(head_changed)
        self.assertFalse(self.model.training)
        FrozenRuntimeState(self.model).restore()
        self.assertEqual(int(self.model.joint_router.trained_updates), 2)
        self.assertGreater(report["nominal_teacher_forward_flops"], 0)
        json.dumps(report, allow_nan=False)

    def test_observation_cost_reuses_the_existing_exit_projections(self):
        batch = self.windows[0][0]
        depths = batch.attention_mask.long() * 2
        widths = fixed_widths(self.config, depths)
        runtime = FrozenRuntimeState(self.model)
        head_rows = []
        hook = self.model.lm_head.register_forward_hook(
            lambda _module, args, _output: head_rows.append(args[0].shape[0])
        )
        try:
            plain, plain_report = repeated_plan_evaluation(
                self.model, batch, self.curriculum, seed=18, runtime_state=runtime,
                repeat_forwards=2, minimum_loss_delta=1e-5,
                force_depth=depths, force_routed_k=widths,
            )
            plain_rows = sum(head_rows)
            head_rows.clear()
            observed, observed_report = repeated_plan_evaluation(
                self.model, batch, self.curriculum, seed=18, runtime_state=runtime,
                repeat_forwards=2, minimum_loss_delta=1e-5,
                force_depth=depths, force_routed_k=widths,
                return_router_observations=True,
            )
        finally:
            hook.remove()
        self.assertEqual(sum(head_rows), plain_rows)
        self.assertEqual(plain.cost, observed.cost)
        expected_extra = 2 * float(observed.output.telemetry["joint_router_flops"]) / 3.0
        self.assertAlmostEqual(
            observed_report["nominal_forward_flops_all_calls"]
            - plain_report["nominal_forward_flops_all_calls"],
            expected_extra,
        )

    def test_teacher_forward_cost_adds_only_the_utility_predictor(self):
        expected = []

        def account(model, _args, kwargs, output):
            cost = plan_cost(model.config, output.chosen_depths, kwargs["force_routed_k"])
            expected.append(
                cost["nominal_forward_flops"]
                + float(output.telemetry["joint_router_flops"]) / 3.0
            )

        hook = self.model.register_forward_hook(account, with_kwargs=True)
        try:
            report = self.fit()
        finally:
            hook.remove()
        self.assertEqual(len(expected), report["teacher_forward_calls"])
        self.assertAlmostEqual(report["nominal_teacher_forward_flops"], sum(expected))

    def test_overlap_is_rejected_before_any_update(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_utility_in_memory(
                self.model, self.curriculum, self.windows, steps=2, seed=18,
                learning_rate=0.0003, max_depth=3,
                evaluation_window=self.windows[0][1],
            )
        self.assertEqual(int(self.model.joint_router.trained_updates), 0)

    def test_heldout_joint_comparison_replays_cost_matched_shuffle(self):
        self.fit()
        batch = tiny_batch(self.base, seed=901)
        runtime = FrozenRuntimeState(self.model)
        legacy = evaluate_in_memory(
            self.model, batch, self.curriculum, seed=18, runtime_state=runtime,
        )
        fixed_depths = batch.attention_mask.long() * 2
        fixed, fixed_noise = repeated_plan_evaluation(
            self.model, batch, replace(self.curriculum, continuation_mode="fixed_max", max_passes=2),
            seed=18, runtime_state=runtime, repeat_forwards=2, minimum_loss_delta=1e-5,
            force_depth=fixed_depths, force_routed_k=fixed_widths(self.config, fixed_depths),
        )
        report = joint_policy_probe(
            self.model, batch, self.curriculum, seed=18, runtime_state=runtime,
            legacy=legacy, repeat_forwards=2, max_passes=3, trained_max_depth=3,
            fixed_reference=fixed, fixed_noise=fixed_noise,
        )
        self.assertEqual(report["forward_calls"], 6)
        self.assertTrue(report["shuffled"]["same_realized_plan_cost"])
        self.assertEqual(report["cost"], report["shuffled"]["cost"])
        self.assertGreaterEqual(report["joint_unused_budget_flops"], 0)
        self.assertFalse(report["policy_quality_established"])
        self.assertTrue(report["uniform_depth2_k4_control_available"])
        self.assertIsNotNone(report["mean_global_loss_delta_from_uniform_depth2_k4"])
        if not report["beats_uniform_depth2_k4_above_numerical_noise"]:
            self.assertFalse(report["one_window_quality_checks_passed"])
        self.assertEqual(int(self.model.joint_router.trained_updates), 2)
        json.dumps(report, allow_nan=False)

    def test_head_artifact_roundtrip_rejects_checkpoint_and_config_transfer(self):
        report = self.fit()
        path = self.root / "utility.pt"
        binding = {
            "checkpoint_sha256": "a" * 64, "run_identity_sha256": "b" * 64,
            "base_model_config_sha256": json_sha256(self.base.to_dict()),
        }
        save_utility_artifact(
            self.model, path, report, **binding, source_revision="c" * 40,
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload["state_dict"]), set(self.model.joint_router.state_dict()))
        self.assertNotIn("input_ids", payload)
        restored = Metis16ForCausalLM(self.config).eval()
        metadata = load_utility_artifact(
            restored, path, **binding, evaluation_window=self.evaluation,
        )
        self.assertEqual(metadata["trained_updates"], 2)
        self.assertEqual(metadata["fit_max_depth"], 3)
        for name, value in self.model.joint_router.state_dict().items():
            torch.testing.assert_close(value, restored.joint_router.state_dict()[name], rtol=0, atol=0)
        for field in ("checkpoint_sha256", "base_model_config_sha256"):
            changed = dict(binding, **{field: "d" * 64})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_utility_artifact(restored, path, **changed, evaluation_window=self.evaluation)
        with self.assertRaises(FileExistsError):
            save_utility_artifact(
                self.model, path, report, **binding, source_revision="c" * 40,
            )


if __name__ == "__main__":
    unittest.main()
