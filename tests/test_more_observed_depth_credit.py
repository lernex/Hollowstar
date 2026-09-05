"""Observed-transition credit must not reward an uncomputed future loss of zero."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class ObservedDepthCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, enabled, **changes):
        return replace(
            Metis16Config.tiny_for_tests(),
            observed_depth_credit=enabled,
            **changes,
        )

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                continuation_mode="budgeted",
                routed_k_mode="fixed",
                fixed_routed_k=2,
                stochastic_routing=False,
                memory_gate_scale=0,
                ngram_gate_scale=0,
            ),
            **changes,
        )

    def witness(self, enabled):
        torch.manual_seed(71)
        model = Metis16ForCausalLM(self.config(enabled, max_passes=2)).eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = torch.tensor([[2, 3, 4, 5, 6, 7]])
        hard = torch.tensor([[False, True, False, True, False, True]])
        probability = torch.tensor([[.2, .8, .3, .9, .1, .7]], requires_grad=True)
        calls = []
        head_calls = []

        def gate(_module, _args, result):
            calls.append(result)
            return probability if len(calls) == 1 else result

        handle = model.continuation.register_forward_hook(gate)
        head_handle = model.lm_head.register_forward_hook(
            lambda module, args, result: head_calls.append(args[0].shape[0])
        )
        try:
            with patch.object(
                model, "_continuation_decision",
                side_effect=lambda probability, **kw: hard & kw["active_mask"],
            ):
                output = model(inputs, labels, curriculum=self.curriculum())
        finally:
            handle.remove()
            head_handle.remove()
        output.loss.backward()
        return model, output, probability, inputs, labels, hard, head_calls

    def test_full_model_preserves_loss_routes_backbone_gradients_and_head_work(self):
        legacy, left, _, _, _, _, left_calls = self.witness(False)
        candidate, right, _, _, _, _, right_calls = self.witness(True)
        torch.testing.assert_close(right.loss, left.loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(right.final_hidden_state, left.final_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(right.chosen_depths, left.chosen_depths, rtol=0, atol=0)
        self.assertEqual(left_calls, right_calls)
        self.assertEqual(len(right_calls), 2)
        self.assertEqual(int(right.telemetry["observed_depth_credit_enabled"]), 1)
        for name in (
            "embedding.weight",
            "layers.0.mixer.impl.in_proj.weight",
            "layers.0.moe.expert_router.weight",
        ):
            with self.subTest(parameter=name):
                expected = dict(legacy.named_parameters())[name].grad
                actual = dict(candidate.named_parameters())[name].grad
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_full_model_removes_exactly_the_halted_current_loss_incentive(self):
        legacy, _, old_probability, inputs, labels, hard, _ = self.witness(False)
        _, _, new_probability, _, _, _, _ = self.witness(True)
        with torch.no_grad():
            stopped = legacy(
                inputs, labels, curriculum=self.curriculum(),
                force_depth=1, return_logits=True,
            )
            stop_ce = F.cross_entropy(
                stopped.logits.reshape(-1, legacy.config.vocab_size),
                labels.flatten(), reduction="none",
            ).reshape_as(labels)
        missing_future = torch.where(~hard, stop_ce, 0)
        expected_correction = (missing_future - missing_future.mean()) / labels.numel()
        torch.testing.assert_close(
            new_probability.grad - old_probability.grad,
            expected_correction, rtol=1e-5, atol=2e-6,
        )
        halted_gradients = new_probability.grad.masked_select(~hard)
        torch.testing.assert_close(
            halted_gradients, halted_gradients[:1].expand_as(halted_gradients),
            rtol=0, atol=2e-6,
        )

    def test_previously_favored_harmful_equal_cost_swap_is_no_longer_favored(self):
        model, _, old_probability, inputs, labels, hard, _ = self.witness(False)
        _, _, new_probability, _, _, _, _ = self.witness(True)
        depths = hard.long() + 1
        swapped = depths.clone()
        swapped[0, 4], swapped[0, 5] = 2, 1
        self.assertEqual(int(swapped.sum()), int(depths.sum()))
        with torch.no_grad():
            before = model(inputs, labels, curriculum=self.curriculum(), force_depth=depths)
            after = model(inputs, labels, curriculum=self.curriculum(), force_depth=swapped)
        self.assertGreater(float(after.loss - before.loss), .01)
        self.assertLess(float(old_probability.grad[0, 4] - old_probability.grad[0, 5]), 0)
        self.assertGreater(float(new_probability.grad[0, 4] - new_probability.grad[0, 5]), 0)

    def test_fixed_and_forced_controls_bypass_the_new_credit_exactly(self):
        torch.manual_seed(19)
        legacy = Metis16ForCausalLM(self.config(False)).eval()
        candidate = Metis16ForCausalLM(self.config(True)).eval()
        candidate.load_state_dict(legacy.state_dict())
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = torch.tensor([[2, 3, 4, 5, 6, -100]])
        for forced in (None, 2):
            with self.subTest(force_depth=forced), torch.no_grad():
                curriculum = self.curriculum(
                    continuation_mode="fixed_max" if forced is None else "budgeted",
                    max_passes=2, memory_gate_scale=1, ngram_gate_scale=1,
                    pathway_mode="frozen",
                )
                left = legacy(
                    inputs, labels, curriculum=curriculum,
                    force_depth=forced, return_logits=True,
                )
                right = candidate(
                    inputs, labels, curriculum=curriculum,
                    force_depth=forced, return_logits=True,
                )
                torch.testing.assert_close(right.loss, left.loss, rtol=0, atol=0)
                torch.testing.assert_close(right.logits, left.logits, rtol=0, atol=0)
                self.assertEqual(int(right.telemetry["observed_depth_credit_enabled"]), 0)

    def test_halted_and_ignored_tokens_do_not_invent_an_observed_transition(self):
        torch.manual_seed(20)
        model = Metis16ForCausalLM(self.config(True, max_passes=2)).eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[2, -100, 4, -100]])
        probability = torch.full(inputs.shape, .2, requires_grad=True)
        handle = model.continuation.register_forward_hook(lambda module, args, result: probability)
        try:
            with patch.object(
                model, "_continuation_decision",
                side_effect=lambda probability, **kw: torch.zeros_like(kw["active_mask"]),
            ):
                output = model(inputs, labels, curriculum=self.curriculum(), return_logits=True)
        finally:
            handle.remove()
        gradient = torch.autograd.grad(output.loss, probability, allow_unused=True)[0]
        self.assertTrue(gradient is None or not bool(torch.count_nonzero(gradient)))
        expected = F.cross_entropy(
            output.logits.reshape(-1, model.config.vocab_size),
            labels.flatten(), ignore_index=-100,
        )
        torch.testing.assert_close(output.loss, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(output.chosen_depths.tolist(), [[1, 1, 1, 1]])

    def test_packed_three_pass_credit_and_gradients_survive_checkpoint_replay(self):
        torch.manual_seed(21)
        model = Metis16ForCausalLM(self.config(True)).train()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = torch.tensor([[2, 3, 4, 5, 6, -100]])
        depths = torch.tensor([[1, 2, 3, 1, 2, 3]])
        reference = None
        for replay in ("none", "layer", "pass"):
            with self.subTest(replay=replay):
                model.set_activation_recompute_policy(replay)
                model.zero_grad(set_to_none=True)
                probability = torch.tensor([[.2, .8, .3, .9, .1, .7]], requires_grad=True)
                calls = []

                def gate(_module, _args, result):
                    calls.append(result)
                    return probability if len(calls) == 1 else result

                handle = model.continuation.register_forward_hook(gate)
                try:
                    with patch.object(
                        model, "_continuation_decision",
                        side_effect=lambda probability, **kw: (
                            (depths >= kw["pass_index"] + 2) & kw["active_mask"]
                        ),
                    ):
                        output = model(inputs, labels, curriculum=self.curriculum(pathway_mode="frozen"))
                    output.loss.backward()
                finally:
                    handle.remove()
                current = (
                    output.loss.detach(),
                    model.embedding.weight.grad.detach().clone(),
                    probability.grad.detach().clone(),
                )
                torch.testing.assert_close(output.chosen_depths, depths)
                if reference is None:
                    reference = current
                else:
                    for actual, expected in zip(current, reference):
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_default_manifest_and_parameter_shapes_are_unchanged(self):
        default = self.config(False)
        enabled = self.config(True)
        self.assertNotIn("observed_depth_credit", default.to_dict())
        self.assertTrue(enabled.to_dict()["observed_depth_credit"])
        # Tiny fixtures use their dedicated validator, not production N-gram geometry.
        with patch.object(Metis16Config, "validate", Metis16Config._validate_tiny):
            restored = Metis16Config.from_mapping(enabled.to_dict())
        self.assertTrue(restored.observed_depth_credit)
        self.assertEqual(default.logical_parameter_audit(), enabled.logical_parameter_audit())
        left = Metis16ForCausalLM(default)
        right = Metis16ForCausalLM(enabled)
        self.assertEqual(
            {name: tuple(value.shape) for name, value in left.state_dict().items()},
            {name: tuple(value.shape) for name, value in right.state_dict().items()},
        )

    def test_flag_validation_rejects_nonboolean_and_production_activation(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.config(value)._validate_tiny()
        with self.assertRaisesRegex(ValueError, "research families"):
            replace(Metis16Config(), observed_depth_credit=True)._validate_depth_credit()


if __name__ == "__main__":
    unittest.main()
