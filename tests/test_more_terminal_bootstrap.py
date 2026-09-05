"""Reference behavior can warm a loop without fabricating utility targets."""

from dataclasses import replace
import math
import unittest

import torch

from metis_training.compute_budget import allocate_causal_budget
from metis_training.compute_router import JointComputeCosts, JointComputeRouter
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class TerminalReferenceBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, steps=5, **changes):
        return replace(
            Metis16Config.tiny_for_tests(), joint_compute_router=True,
            joint_router_hidden_dim=8, causal_compute_budget=True,
            terminal_action_critic=True, terminal_reference_bootstrap_steps=steps,
            **changes,
        )

    def curriculum(self, step=0, **changes):
        return replace(
            CurriculumState(
                compute_allocation_mode="joint", continuation_mode="budgeted",
                routed_k_mode="budgeted", fixed_routed_k=2, memory_gate_scale=0,
                ngram_gate_scale=0, stochastic_routing=False,
                random_policy_seed=601, random_policy_step=step,
                allow_untrained_joint_router=True,
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

    def test_disabled_manifest_and_parameter_shapes_stay_unchanged(self):
        disabled, enabled = self.config(0), self.config(5)
        self.assertNotIn("terminal_reference_bootstrap_steps", disabled.to_dict())
        self.assertEqual(enabled.to_dict()["terminal_reference_bootstrap_steps"], 5)
        self.assertEqual(disabled.logical_parameter_audit(), enabled.logical_parameter_audit())
        self.assertEqual(JointComputeCosts.from_config(disabled), JointComputeCosts.from_config(enabled))

    def test_uniform_value_scale_prior_is_fixed_before_any_data(self):
        config = self.config()
        router = JointComputeRouter(config)
        expected = -math.log(config.vocab_size)
        self.assertAlmostEqual(float(router.output.bias[-1].detach()), expected, places=6)
        self.assertEqual(int(torch.count_nonzero(router.output.bias[:-1])), 0)
        self.assertEqual(int(torch.count_nonzero(router.output.weight)), 0)
        self.assertEqual(float(JointComputeRouter(self.config(0)).output.bias[-1].detach()), 0)

    def test_reference_scores_pay_for_the_critic_without_changing_prediction_targets(self):
        config = self.config()
        router = JointComputeRouter(config)
        state = torch.randn(1, 8, config.d_model)
        history = torch.randn(1, 8, config.route_feature_dim)
        mask = torch.ones(1, 8, dtype=torch.bool)
        prediction = router(
            state, state, state, history, active_mask=mask,
            origin_pass=0, remaining_passes=2,
        )
        old_depth, old_width = prediction.depth_utilities.clone(), prediction.width_utilities.clone()
        depth, width = router.allocation_utilities(
            prediction, exploration=0, generator=None,
            reference_bootstrap=True, reference_routed_k=2,
        )
        torch.testing.assert_close(prediction.depth_utilities, old_depth, rtol=0, atol=0)
        torch.testing.assert_close(prediction.width_utilities, old_width, rtol=0, atol=0)
        costs = router.costs
        credit = (
            costs.reference_per_token - costs.head_per_token - costs.router_per_token
            - costs.base_pass_costs[0] - 2 * sum(costs.expert_costs)
        )
        plan = allocate_causal_budget(
            depth, width, mask, base_pass_costs=costs.base_pass_costs[1:],
            expert_costs=costs.expert_costs, credit_per_token=credit,
            cost_scale=costs.reference_per_token,
        )
        self.assertTrue(bool(plan.depths.eq(1).all()))
        self.assertTrue(bool((plan.routed_k[0] >= 1).all()))
        self.assertTrue(bool((plan.routed_k[0] <= 2).all()))
        self.assertTrue(bool(plan.routed_k[0].eq(1).any()))
        self.assertLessEqual(int(plan.cost), int(plan.budget))

    def test_bootstrap_ends_at_the_declared_optimizer_step_and_is_never_eval_policy(self):
        torch.manual_seed(602)
        model = Metis16ForCausalLM(self.config()).train()
        inputs = self.inputs()
        with torch.no_grad():
            warm = model(inputs, self.labels(inputs), curriculum=self.curriculum(0))
            ended = model(inputs, self.labels(inputs), curriculum=self.curriculum(5))
            model.eval()
            evaluation = model(inputs, self.labels(inputs), curriculum=self.curriculum(0))
        self.assertTrue(bool(warm.chosen_depths.eq(2).all()))
        self.assertEqual(int(warm.telemetry["terminal_reference_bootstrap_active"]), 1)
        self.assertEqual(int(warm.telemetry["terminal_reference_bootstrap_step"]), 0)
        self.assertTrue(bool(ended.chosen_depths.eq(1).all()))
        self.assertEqual(int(ended.telemetry["terminal_reference_bootstrap_active"]), 0)
        self.assertTrue(bool(evaluation.chosen_depths.eq(1).all()))
        self.assertEqual(int(evaluation.telemetry["terminal_reference_bootstrap_active"]), 0)

    def test_bootstrap_critic_still_fits_only_actual_negative_terminal_ce(self):
        torch.manual_seed(603)
        model = Metis16ForCausalLM(self.config()).train()
        inputs = self.inputs()
        labels = self.labels(inputs)
        output = model(
            inputs, labels, curriculum=self.curriculum(),
            return_terminal_router_observations=True,
        )
        total = output.loss.new_zeros(())
        count = 0
        for record in output.terminal_router_observations:
            values = record.prediction.value_of_actions(record.depths, record.routed_k)
            total = total + (values + record.terminal_ce.detach()).square().masked_select(record.observed_mask).sum()
            count += int(record.observed_mask.sum())
        self.assertEqual(count, int(labels.ne(-100).sum()))
        torch.testing.assert_close(output.auxiliary_losses["joint_utility"], total / count)
        output.auxiliary_losses["joint_utility"].backward()
        self.assertIsNone(model.embedding.weight.grad)
        self.assertGreater(float(model.joint_router.output.weight.grad.abs().sum()), 0)

    def test_bootstrap_exploration_remains_prefix_and_batch_invariant(self):
        torch.manual_seed(604)
        model = Metis16ForCausalLM(self.config()).train()
        inputs = self.inputs()
        changed = inputs.clone()
        changed[:, 4:] += 50
        curriculum = self.curriculum(stochastic_routing=True)
        with torch.no_grad():
            full = model(inputs, curriculum=curriculum, return_logits=True)
            suffix = model(changed, curriculum=curriculum, return_logits=True)
            solo = model(inputs[:1], curriculum=curriculum, return_logits=True)
        torch.testing.assert_close(full.chosen_depths[:, :4], suffix.chosen_depths[:, :4])
        torch.testing.assert_close(full.chosen_depths[0], solo.chosen_depths[0])
        torch.testing.assert_close(full.logits[:, :4], suffix.logits[:, :4], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(full.logits[0], solo.logits[0], rtol=1e-5, atol=1e-6)

    def test_invalid_bootstrap_configuration_is_rejected(self):
        for steps in (-1, True, "5"):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                self.config(steps)._validate_tiny()
        with self.assertRaisesRegex(ValueError, "terminal-action"):
            replace(self.config(), terminal_action_critic=False)._validate_tiny()
        model = Metis16ForCausalLM(self.config()).train()
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            model(self.inputs(), curriculum=self.curriculum(-1))


if __name__ == "__main__":
    unittest.main()
