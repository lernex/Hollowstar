from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from metis_training.compute_router import JointComputeRouter
from metis_training.model import (
    CurriculumState,
    Metis16ForCausalLM,
    _budgeted_binary_straight_through,
)
from metis_training.model_config import Metis16Config, load_family_config


def tiny_joint_config(**changes):
    return replace(
        Metis16Config.tiny_for_tests(),
        joint_compute_router=True,
        joint_router_hidden_dim=8,
        activation_recompute_policy="none",
        **changes,
    )


def batch(config, length=12):
    generator = torch.Generator().manual_seed(901)
    inputs = torch.randint(0, config.vocab_size, (2, length), generator=generator)
    labels = inputs.roll(-1, dims=1)
    labels[:, -1] = -100
    return inputs, labels


class JointCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_legacy_witness_and_observed_utility_have_opposite_credit(self):
        probability = torch.tensor([[0.25, 0.75]], requires_grad=True)
        hard = torch.tensor([[False, True]])
        mask = torch.ones_like(hard)
        gate, _ = _budgeted_binary_straight_through(hard, probability, mask)
        packed_loss = ((1 - gate) * 3).sum() + (gate[hard] * 2).sum()
        legacy_gradient = torch.autograd.grad(packed_loss, probability)[0]
        self.assertLess(float(legacy_gradient[0, 0]), 0)

        config = tiny_joint_config()
        router = JointComputeRouter(config)
        state = torch.randn(1, 2, config.d_model, requires_grad=True)
        history = torch.randn(1, 2, config.route_feature_dim, requires_grad=True)
        prediction = router(
            state, state, state, history,
            active_mask=mask, origin_pass=0, remaining_passes=2,
        )
        widths = torch.full((config.n_layers, 1, 2), 2, dtype=torch.long)
        loss, count = prediction.observed_loss(
            [widths], torch.tensor([[-1.0, 1.0]]), mask
        )
        gradient = torch.autograd.grad(loss, prediction.depth_utilities)[0]
        self.assertEqual(int(count), 2)
        self.assertGreater(float(gradient[0, 0, 1]), 0)
        self.assertLess(float(gradient[0, 1, 1]), 0)
        self.assertEqual(int(torch.count_nonzero(gradient[..., 2])), 0)

    def test_unobserved_continuation_has_no_invented_utility_target(self):
        config = tiny_joint_config()
        router = JointComputeRouter(config)
        state = torch.randn(1, 2, config.d_model)
        history = torch.randn(1, 2, config.route_feature_dim)
        mask = torch.ones(1, 2, dtype=torch.bool)
        prediction = router(
            state, state, state, history,
            active_mask=mask, origin_pass=0, remaining_passes=2,
        )
        widths = torch.full((config.n_layers, 1, 2), 2, dtype=torch.long)
        loss, count = prediction.observed_loss(
            [widths],
            torch.tensor([[1000.0, 1.0]]),
            torch.tensor([[False, True]]),
        )
        gradient = torch.autograd.grad(loss, prediction.depth_utilities)[0]
        self.assertEqual(int(count), 1)
        self.assertEqual(int(torch.count_nonzero(gradient[0, 0])), 0)
        self.assertLess(float(gradient[0, 1, 1]), 0)

    def test_default_parameters_and_forward_remain_identical(self):
        base = replace(Metis16Config.tiny_for_tests(), activation_recompute_policy="none")
        torch.manual_seed(37)
        legacy = Metis16ForCausalLM(base).eval()
        torch.manual_seed(37)
        candidate = Metis16ForCausalLM(tiny_joint_config()).eval()
        candidate_parameters = dict(candidate.named_parameters())
        for name, value in legacy.named_parameters():
            self.assertTrue(torch.equal(value, candidate_parameters[name]), name)
        inputs, labels = batch(base)
        curriculum = CurriculumState(
            continuation_mode="budgeted", routed_k_mode="budgeted",
            fixed_routed_k=2, stochastic_routing=False,
        )
        with torch.no_grad():
            left = legacy(inputs, labels, curriculum=curriculum)
            right = candidate(inputs, labels, curriculum=curriculum)
        torch.testing.assert_close(left.loss, right.loss, rtol=0, atol=0)
        torch.testing.assert_close(left.final_hidden_state, right.final_hidden_state, rtol=0, atol=0)
        self.assertEqual(candidate.config.logical_parameter_audit().joint_router,
                         sum(p.numel() for p in candidate.joint_router.parameters()))

    def test_production_families_cannot_silently_enable_joint_routing(self):
        for family in ("praxis", "logos"):
            config = load_family_config(family)
            self.assertFalse(config.joint_compute_router)
            with self.assertRaisesRegex(ValueError, "research families"):
                replace(config, joint_compute_router=True).validate()

    def test_cost_reference_excludes_new_controller_and_matches_fixed_work(self):
        config = tiny_joint_config()
        router = JointComputeRouter(config)
        mask = torch.ones(2, 12, dtype=torch.bool)
        widths = torch.full((config.n_layers, 2, 12), 2, dtype=torch.long)
        cost = router.costs.pass_cost(0, widths, mask) + router.costs.pass_cost(1, widths, mask)
        self.assertEqual(int(cost), int(mask.sum()) * router.costs.reference_per_token)
        self.assertGreater(router.costs.router_per_token, 0)

    def test_explicit_widths_match_fixed_k_and_survive_replay(self):
        config = tiny_joint_config()
        model = Metis16ForCausalLM(config).eval()
        inputs, labels = batch(config)
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        curriculum = CurriculumState(
            continuation_mode="fixed_max", routed_k_mode="fixed",
            fixed_routed_k=2, stochastic_routing=False,
        )
        with torch.no_grad():
            baseline = model(inputs, labels, curriculum=curriculum, force_depth=2)
            forced = model(
                inputs, labels, curriculum=curriculum,
                force_depth=2, force_routed_k=widths,
            )
        torch.testing.assert_close(baseline.loss, forced.loss, rtol=0, atol=0)
        depths = torch.tensor([[1, 2, 3] * 4, [3, 2, 1] * 4])
        widths[:, :, :, 1::2] = 3
        gradients = []
        losses = []
        for replay in ("none", "layer", "pass"):
            model.activation_recompute_policy = replay
            model.train()
            model.zero_grad(set_to_none=True)
            output = model(
                inputs, labels, curriculum=curriculum,
                force_depth=depths, force_routed_k=widths,
            )
            output.loss.backward()
            losses.append(output.loss.detach())
            gradients.append(model.embedding.weight.grad.detach().clone())
        for index in (1, 2):
            torch.testing.assert_close(losses[0], losses[index], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradients[0], gradients[index], rtol=1e-4, atol=1e-5)

    def test_outcome_regression_trains_router_not_backbone(self):
        config = tiny_joint_config()
        model = Metis16ForCausalLM(config).train()
        inputs, labels = batch(config)
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        widths[1] = 3
        output = model(
            inputs, labels,
            curriculum=CurriculumState(routed_k_mode="fixed", fixed_routed_k=2),
            force_depth=2,
            force_routed_k=widths,
            return_router_observations=True,
        )
        self.assertGreater(int(output.telemetry["joint_utility_observations"]), 0)
        output.auxiliary_losses["joint_utility"].backward()
        self.assertTrue(any(
            p.grad is not None and bool(torch.count_nonzero(p.grad))
            for p in model.joint_router.parameters()
        ))
        self.assertIsNone(model.embedding.weight.grad)
        self.assertIsNone(model.continuation.output.weight.grad)
        self.assertEqual(len(output.router_observations[0].width_history), 1)

    def test_stopped_teacher_does_not_train_an_unvisited_future(self):
        config = tiny_joint_config()
        model = Metis16ForCausalLM(config).train()
        inputs, labels = batch(config)
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        output = model(
            inputs, labels,
            curriculum=CurriculumState(routed_k_mode="fixed", fixed_routed_k=2),
            force_depth=1, force_routed_k=widths, return_router_observations=True,
        )
        self.assertEqual(int(output.telemetry["joint_utility_observations"]), 0)
        output.auxiliary_losses["joint_utility"].backward()
        self.assertTrue(all(
            p.grad is not None and not bool(torch.count_nonzero(p.grad))
            for p in model.joint_router.parameters()
        ))

    def test_joint_execution_accounts_for_controller_and_padding(self):
        config = tiny_joint_config(max_passes=5)
        model = Metis16ForCausalLM(config).eval()
        with torch.no_grad():
            model.joint_router.output.bias[:4].copy_(
                torch.tensor([1.0, 4.0, 9.0, 16.0])
            )
        inputs, labels = batch(config, length=18)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, -2:] = False
        labels[~mask] = -100
        curriculum = CurriculumState(
            compute_allocation_mode="joint", fixed_routed_k=2,
            stochastic_routing=False, allow_untrained_joint_router=True,
        )
        with torch.no_grad():
            output = model(inputs, labels, curriculum=curriculum, attention_mask=mask)
        self.assertEqual(int(output.chosen_depths[~mask].sum()), 0)
        self.assertEqual(int(output.chosen_depths.max()), 5)
        self.assertLessEqual(
            int(output.telemetry["joint_model_flops"]),
            int(output.telemetry["joint_budget_flops"]),
        )
        self.assertGreater(int(output.telemetry["joint_router_flops"]), 0)
        self.assertEqual(float(output.telemetry["ponder_exit_mass_max_error"]), 0)

    def test_untrained_joint_policy_cannot_masquerade_as_ready(self):
        config = tiny_joint_config()
        model = Metis16ForCausalLM(config).eval()
        inputs, labels = batch(config)
        with self.assertRaisesRegex(RuntimeError, "trained utility weights"):
            model(
                inputs, labels,
                curriculum=CurriculumState(
                    compute_allocation_mode="joint", fixed_routed_k=2
                ),
            )


if __name__ == "__main__":
    unittest.main()
