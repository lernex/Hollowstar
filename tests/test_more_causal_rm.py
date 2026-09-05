"""Causal RM retains the legacy confidence feature, not a mislabeled hard gate."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from metis_training.compute_router import JointComputeCosts
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class CausalRMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self):
        return replace(
            Metis16Config.tiny_for_tests(), joint_compute_router=True,
            joint_router_hidden_dim=8, causal_compute_budget=True,
            terminal_action_critic=True, causal_memory_metadata="legacy_confidence",
        )

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                compute_allocation_mode="joint", continuation_mode="budgeted",
                routed_k_mode="budgeted", fixed_routed_k=2,
                memory_gate_scale=1, ngram_gate_scale=0,
                stochastic_routing=False, allow_untrained_joint_router=True,
            ),
            **changes,
        )

    def inputs(self):
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                             [11, 12, 13, 14, 15, 16, 17, 18]])

    def labels(self, inputs):
        labels = inputs.roll(-1, dims=1)
        labels[:, -1] = -100
        return labels

    def model(self):
        torch.manual_seed(701)
        model = Metis16ForCausalLM(self.config()).eval()
        with torch.no_grad():
            model.depth_memory.stream_gate_bias.zero_()
            model.joint_router.output.bias[:2].copy_(torch.tensor([1., 2.]))
            model.joint_router.output.bias[-1] = -6
        return model

    def test_fixed_rm_trajectory_preserves_legacy_outputs_and_backbone_gradients(self):
        candidate = self.model()
        legacy_config = replace(
            candidate.config, joint_compute_router=False, causal_compute_budget=False,
            terminal_action_critic=False, causal_memory_metadata="disabled",
        )
        legacy = Metis16ForCausalLM(legacy_config).eval()
        legacy.load_state_dict(
            {name: value for name, value in candidate.state_dict().items()
             if not name.startswith("joint_router.")}
        )
        inputs = self.inputs()
        labels = self.labels(inputs)
        curriculum = self.curriculum(
            compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", max_passes=2, pathway_mode="frozen",
        )
        left = legacy(inputs, labels, curriculum=curriculum)
        right = candidate(inputs, labels, curriculum=curriculum)
        torch.testing.assert_close(right.final_hidden_state, left.final_hidden_state, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(right.loss, left.loss, rtol=1e-6, atol=1e-6)
        left.loss.backward()
        right.loss.backward()
        for name in (
            "embedding.weight", "depth_memory.metadata_write.weight",
            "depth_memory.route_projection.weight", "layers.0.mixer.impl.in_proj.weight",
        ):
            torch.testing.assert_close(
                dict(candidate.named_parameters())[name].grad,
                dict(legacy.named_parameters())[name].grad, rtol=1e-4, atol=1e-5,
            )
        self.assertIsNone(candidate.continuation.output.weight.grad)
        # This mutation must fail the compatibility reference: the old column
        # is a continuous feature, not the new policy's binary continuation.
        with torch.no_grad(), patch.object(
            candidate, "_causal_memory_confidence",
            side_effect=lambda state, memory, difference, history, needed: (needed.float(), int(needed.sum())),
        ):
            corrupted = candidate(inputs, labels, curriculum=curriculum)
        self.assertGreater(float((corrupted.final_hidden_state - left.final_hidden_state).abs().max()), 1e-4)

    def test_metadata_cost_is_paid_without_increasing_the_core_reference_cap(self):
        model = self.model()
        costs = model.joint_router.costs
        core = JointComputeCosts.from_config(replace(model.config, causal_memory_metadata="disabled"))
        expected_fee = 2 * (
            sum(parameter.numel() for parameter in model.continuation.parameters())
            + sum(parameter.numel() for parameter in model.depth_memory.route_projection.parameters())
        )
        self.assertEqual(costs.metadata_transition_flops, expected_fee)
        self.assertEqual(costs.reference_per_token, core.reference_per_token)
        self.assertEqual(costs.base_pass_costs[0], core.base_pass_costs[0])
        self.assertEqual(costs.base_pass_costs[1:], tuple(cost + expected_fee for cost in core.base_pass_costs[1:]))
        inputs = self.inputs()
        with torch.no_grad():
            fixed = model(
                inputs, self.labels(inputs),
                curriculum=self.curriculum(
                    compute_allocation_mode="legacy", continuation_mode="fixed_max",
                    routed_k_mode="fixed", max_passes=2,
                ),
            )
        self.assertEqual(int(fixed.telemetry["causal_memory_metadata_tokens"]), inputs.numel())
        self.assertEqual(int(fixed.telemetry["causal_memory_metadata_forward_rows"]), inputs.numel())
        self.assertEqual(
            int(fixed.telemetry["joint_model_flops"]),
            inputs.numel() * (costs.reference_per_token + expected_fee),
        )
        self.assertEqual(int(fixed.telemetry["joint_budget_enforced"]), 0)
        with torch.no_grad():
            adaptive = model(inputs, self.labels(inputs), curriculum=self.curriculum())
        self.assertEqual(
            int(adaptive.telemetry["causal_memory_metadata_tokens"]),
            int((adaptive.chosen_depths - 1).sum()),
        )
        self.assertLessEqual(int(adaptive.telemetry["joint_model_flops"]), int(adaptive.telemetry["joint_budget_flops"]))

    def test_rm_prefix_and_batch_composition_are_invariant(self):
        model = self.model()
        inputs = self.inputs()
        changed = inputs.clone()
        changed[:, 4:] += 50
        with torch.no_grad():
            full = model(inputs, curriculum=self.curriculum(), return_logits=True)
            suffix = model(changed, curriculum=self.curriculum(), return_logits=True)
            solo = model(inputs[:1], curriculum=self.curriculum(), return_logits=True)
        self.assertGreater(int(full.chosen_depths.max()), 1)
        torch.testing.assert_close(full.chosen_depths[:, :4], suffix.chosen_depths[:, :4])
        torch.testing.assert_close(full.chosen_depths[0], solo.chosen_depths[0])
        torch.testing.assert_close(full.logits[:, :4], suffix.logits[:, :4], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(full.logits[0], solo.logits[0], rtol=1e-5, atol=1e-6)

    def test_metadata_only_runs_for_continuers_with_synchronized_dummy_support(self):
        model = self.model()
        state = torch.randn(1, 4, model.config.d_model, requires_grad=True)
        history = torch.randn(1, 4, model.config.route_feature_dim)
        needed = torch.tensor([[False, True, False, True]])
        confidence, rows = model._causal_memory_confidence(state, state, state, history, needed)
        self.assertEqual(rows, 2)
        self.assertFalse(confidence.requires_grad)
        self.assertEqual(int(torch.count_nonzero(confidence[~needed])), 0)
        empty = torch.zeros_like(needed)
        _, rows = model._causal_memory_confidence(state, state, state, history, empty)
        self.assertEqual(rows, 0)
        with (
            patch("metis_training.model._precision_requires_synchronized_schedule", return_value=True),
            patch("metis_training.model._group_world_size", return_value=2),
        ):
            confidence, rows = model._causal_memory_confidence(state, state, state, history, empty)
        self.assertEqual(rows, 1)
        self.assertEqual(int(torch.count_nonzero(confidence)), 0)

    def test_rm_terminal_learning_and_metadata_survive_layer_and_pass_replay(self):
        model = self.model().train()
        inputs = self.inputs()
        labels = self.labels(inputs)
        reference = None
        for replay in ("none", "layer", "pass"):
            with self.subTest(replay=replay):
                model.set_activation_recompute_policy(replay)
                model.zero_grad(set_to_none=True)
                output = model(inputs, labels, curriculum=self.curriculum(pathway_mode="frozen"))
                (output.loss + output.auxiliary_losses["joint_utility"]).backward()
                self.assertIsNone(model.continuation.output.weight.grad)
                self.assertGreater(float(model.depth_memory.metadata_write.weight.grad.abs().sum()), 0)
                current = (
                    output.loss.detach(), model.embedding.weight.grad.clone(),
                    model.depth_memory.metadata_write.weight.grad.clone(),
                    model.joint_router.output.weight.grad.clone(),
                )
                if reference is None:
                    reference = current
                else:
                    for actual, expected in zip(current, reference):
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_mode_is_explicit_and_disabled_gate_cannot_hide_metadata_charges(self):
        disabled = replace(self.config(), causal_memory_metadata="disabled")
        self.assertNotIn("causal_memory_metadata", disabled.to_dict())
        self.assertEqual(self.config().to_dict()["causal_memory_metadata"], "legacy_confidence")
        self.assertEqual(self.config().logical_parameter_audit(), disabled.logical_parameter_audit())
        with self.assertRaisesRegex(ValueError, "positive finite memory"):
            self.curriculum(memory_gate_scale=0).validate(self.config())
        with self.assertRaisesRegex(ValueError, "memory_gate_scale=0"):
            self.curriculum().validate(disabled)
        with self.assertRaisesRegex(ValueError, "requires causal"):
            replace(self.config(), causal_compute_budget=False)._validate_tiny()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_rm_causal_metadata_forward_backward_and_cap(self):
        model = self.model().cuda().train()
        inputs = self.inputs().cuda()
        output = model(inputs, self.labels(inputs), curriculum=self.curriculum())
        (output.loss + output.auxiliary_losses["joint_utility"]).backward()
        self.assertTrue(bool(torch.isfinite(model.embedding.weight.grad).all()))
        self.assertGreater(float(model.depth_memory.metadata_write.weight.grad.abs().sum()), 0)
        self.assertLessEqual(int(output.telemetry["joint_model_flops"]), int(output.telemetry["joint_budget_flops"]))
        self.assertEqual(
            int(output.telemetry["causal_memory_metadata_tokens"]),
            int((output.chosen_depths - 1).sum()),
        )


if __name__ == "__main__":
    unittest.main()
