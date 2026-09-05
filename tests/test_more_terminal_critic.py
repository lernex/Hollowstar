"""Terminal Q learns actual exits without paying for every hypothetical LM exit."""

from dataclasses import replace
import unittest

import torch

from metis_training.compute_router import JointComputeCosts, JointComputeRouter
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class TerminalCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **changes):
        return replace(
            Metis16Config.tiny_for_tests(), joint_compute_router=True,
            joint_router_hidden_dim=8, causal_compute_budget=True,
            terminal_action_critic=True, **changes,
        )

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                compute_allocation_mode="joint", continuation_mode="budgeted",
                routed_k_mode="budgeted", fixed_routed_k=2,
                memory_gate_scale=0, ngram_gate_scale=0,
                allow_untrained_joint_router=True, stochastic_routing=False,
                random_policy_seed=412, random_policy_step=9,
            ),
            **changes,
        )

    def inputs(self):
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                             [20, 21, 22, 23, 24, 25, 26, 27]])

    def labels(self, inputs):
        labels = inputs.roll(-1, dims=1)
        labels[:, -1] = -100
        return labels

    def model(self):
        torch.manual_seed(201)
        model = Metis16ForCausalLM(self.config()).eval()
        with torch.no_grad():
            model.joint_router.output.weight.normal_(std=.1)
            model.joint_router.output.bias[:2].copy_(torch.tensor([1., 2.]))
            model.joint_router.output.bias[-1] = -6
        return model

    def prediction(self):
        config = self.config()
        router = JointComputeRouter(config)
        state = torch.randn(1, 2, config.d_model, requires_grad=True)
        history = torch.randn(1, 2, config.route_feature_dim)
        prediction = router(
            state, state, state, history, active_mask=torch.ones(1, 2, dtype=torch.bool),
            origin_pass=0, remaining_passes=2,
        )
        return config, router, state, prediction

    def test_halt_is_learned_and_unvisited_actions_have_no_terminal_target(self):
        config, _, state, prediction = self.prediction()
        depths = torch.tensor([[0, 1]])
        widths = torch.zeros(2, config.n_layers, 1, 2, dtype=torch.long)
        widths[0, :, 0, 1] = 2
        loss, count = prediction.terminal_loss(
            depths, widths, torch.tensor([[7., float("nan")]]), torch.tensor([[True, False]])
        )
        depth_gradient, width_gradient, state_gradient = torch.autograd.grad(
            loss, (prediction.depth_utilities, prediction.width_utilities, state),
            allow_unused=True,
        )
        self.assertEqual(int(count), 1)
        self.assertGreater(float(depth_gradient[0, 0, 0]), 0)
        self.assertEqual(int(torch.count_nonzero(depth_gradient[..., 1:])), 0)
        self.assertEqual(int(torch.count_nonzero(depth_gradient[0, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(width_gradient)), 0)
        self.assertIsNone(state_gradient)
        with self.assertRaisesRegex(ValueError, "not observed improvements"):
            prediction.observed_loss([widths[0]], torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))

    def test_terminal_target_is_negative_actual_ce_not_an_improvement(self):
        config, _, _, prediction = self.prediction()
        depths = torch.tensor([[0, 1]])
        widths = torch.zeros(2, config.n_layers, 1, 2, dtype=torch.long)
        widths[0, :, 0, 1] = 2
        loss, count = prediction.terminal_loss(
            depths, widths, torch.tensor([[7., 3.]]), torch.ones(1, 2, dtype=torch.bool)
        )
        self.assertEqual(int(count), 2)
        self.assertEqual(float(loss.detach()), 58)
        gradients = torch.autograd.grad(loss, prediction.depth_utilities)[0]
        self.assertEqual(gradients.tolist(), [[[14., 0., 0.], [0., 6., 0.]]])

    def test_one_terminal_head_preserves_fixed_trajectory_loss_and_backbone_gradients(self):
        torch.manual_seed(202)
        config = self.config()
        source = Metis16ForCausalLM(replace(config, terminal_action_critic=False)).eval()
        candidate = Metis16ForCausalLM(config).eval()
        body = {name: value for name, value in source.state_dict().items()
                if not name.startswith("joint_router.")}
        candidate.load_state_dict(body, strict=False)
        inputs = self.inputs()
        labels = self.labels(inputs)
        depths = torch.tensor([[1, 2, 3, 1, 2, 3, 2, 1], [3, 2, 1, 3, 2, 1, 2, 3]])
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        widths[1, :, :, ::2] = 1
        widths[2] = 3
        curriculum = self.curriculum(
            compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", pathway_mode="frozen",
        )
        counts = [[], []]
        handles = [
            model.lm_head.register_forward_hook(
                lambda module, args, result, index=index: counts[index].append(args[0].shape[0])
            )
            for index, model in enumerate((source, candidate))
        ]
        try:
            left = source(
                inputs, labels, curriculum=curriculum, force_depth=depths,
                force_routed_k=widths, return_router_observations=True,
            )
            right = candidate(
                inputs, labels, curriculum=curriculum, force_depth=depths,
                force_routed_k=widths, return_terminal_router_observations=True,
            )
            left.loss.backward()
            right.loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(right.final_hidden_state, left.final_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(right.loss, left.loss, rtol=1e-6, atol=1e-6)
        for name in ("embedding.weight", "layers.0.mixer.impl.in_proj.weight",
                     "layers.0.moe.expert_router.weight"):
            torch.testing.assert_close(
                dict(candidate.named_parameters())[name].grad,
                dict(source.named_parameters())[name].grad, rtol=1e-4, atol=1e-5,
            )
        self.assertEqual(sum(counts[1]), int(labels.ne(-100).sum()))
        self.assertEqual(sum(counts[0]), int(depths.masked_select(labels.ne(-100)).sum()))
        observed = sum(record.observed_mask.long() for record in right.terminal_router_observations)
        torch.testing.assert_close(observed, labels.ne(-100).long())
        self.assertFalse(right.router_observations)
        for record in right.terminal_router_observations:
            torch.testing.assert_close(
                record.depths[record.observed_mask],
                (depths - 1)[record.observed_mask],
            )
            for r in range(config.max_passes - 1):
                stopped = record.depths <= r
                self.assertEqual(int(record.routed_k[r].masked_select(stopped.unsqueeze(0)).sum()), 0)

    def test_exact_cost_is_body_plus_one_reserved_head_plus_one_critic(self):
        model = self.model()
        inputs = self.inputs()
        labels = self.labels(inputs)
        with torch.no_grad():
            output = model(
                inputs, labels, curriculum=self.curriculum(),
                return_terminal_router_observations=True,
            )
        costs = model.joint_router.costs
        original = JointComputeCosts.from_config(replace(model.config, terminal_action_critic=False))
        self.assertEqual(costs.reference_per_token, original.reference_per_token)
        self.assertEqual(
            costs.base_pass_costs,
            tuple(cost - costs.head_per_token for cost in original.base_pass_costs),
        )
        self.assertEqual(costs.router_per_token - original.router_per_token, 6 * (model.config.joint_router_hidden_dim + 1))
        per_token = torch.full_like(inputs, costs.head_per_token + costs.router_per_token)
        for r, active in enumerate(output.active_masks):
            per_token += active.long() * costs.base_pass_costs[r]
        per_token += round(model.config.target_mean_routed_k) * sum(costs.expert_costs)
        for record in output.terminal_router_observations:
            route_cost = sum(
                record.routed_k[:, layer].sum(dim=0) * cost
                for layer, cost in enumerate(costs.expert_costs)
            )
            per_token += torch.where(record.observed_mask, route_cost, 0)
        # Labels ignored at the last column still execute their admitted work.
        # Count it from the final actual trace, not from supervised observations.
        final = output.terminal_router_observations[-1]
        unsupervised_cost = sum(
            final.routed_k[:, layer].sum(dim=0) * cost
            for layer, cost in enumerate(costs.expert_costs)
        )
        per_token += torch.where(labels.eq(-100), unsupervised_cost, 0)
        self.assertEqual(int(per_token.sum()), int(output.telemetry["joint_model_flops"]))
        self.assertTrue(bool((per_token.cumsum(1) <= costs.reference_per_token * torch.arange(1, inputs.shape[1] + 1)).all()))
        self.assertEqual(int(output.telemetry["joint_utility_observations"]), int(labels.ne(-100).sum()))
        self.assertEqual(int(output.telemetry["terminal_lm_head_reserved_flops"]), inputs.numel() * costs.head_per_token)

    def test_lean_fixed_baseline_cost_and_logits_do_not_change(self):
        model = self.model()
        baseline = Metis16ForCausalLM(replace(model.config, terminal_action_critic=False)).eval()
        baseline.load_state_dict(
            {name: value for name, value in model.state_dict().items() if not name.startswith("joint_router.")},
            strict=False,
        )
        inputs = self.inputs()
        labels = self.labels(inputs)
        curriculum = self.curriculum(
            compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", max_passes=2, pathway_mode="frozen",
        )
        with torch.no_grad():
            left_logits = baseline(inputs, curriculum=curriculum, return_logits=True).logits
            right_logits = model(inputs, curriculum=curriculum, return_logits=True).logits
            left = baseline(inputs, labels, curriculum=curriculum)
            right = model(inputs, labels, curriculum=curriculum)
        torch.testing.assert_close(right_logits, left_logits, rtol=0, atol=0)
        torch.testing.assert_close(right.loss, left.loss, rtol=0, atol=0)
        self.assertEqual(int(right.telemetry["joint_model_flops"]), int(left.telemetry["joint_model_flops"]))
        self.assertEqual(int(right.telemetry["joint_router_flops"]), 0)

    def test_terminal_exploration_and_execution_are_causal_and_batch_independent(self):
        model = self.model().train()
        model.config = replace(model.config, terminal_critic_exploration=1.0)
        inputs = self.inputs()
        changed = inputs.clone()
        changed[:, 4:] += 50
        curriculum = self.curriculum(stochastic_routing=True)
        with torch.no_grad():
            full = model(inputs, curriculum=curriculum, return_logits=True)
            suffix = model(changed, curriculum=curriculum, return_logits=True)
            solo = model(inputs[:1], curriculum=curriculum, return_logits=True)
        torch.testing.assert_close(suffix.logits[:, :4], full.logits[:, :4], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(solo.logits[0], full.logits[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(suffix.chosen_depths[:, :4], full.chosen_depths[:, :4])
        torch.testing.assert_close(solo.chosen_depths[0], full.chosen_depths[0])

    def test_epsilon_exploration_can_override_pessimistic_unvisited_q_values(self):
        model = self.model().train()
        model.config = replace(model.config, terminal_critic_exploration=1.0)
        with torch.no_grad():
            model.joint_router.output.weight.zero_()
            model.joint_router.output.bias[:2].fill_(-1e6)
            model.joint_router.output.bias[-1] = -6
        inputs = self.inputs()
        with torch.no_grad():
            output = model(
                inputs, self.labels(inputs), curriculum=self.curriculum(stochastic_routing=True)
            )
        self.assertGreater(int(output.chosen_depths.max()), 1)
        self.assertLessEqual(int(output.telemetry["joint_model_flops"]), int(output.telemetry["joint_budget_flops"]))

    def test_terminal_gradients_and_actual_exit_heads_survive_checkpoint_replay(self):
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
                current = (
                    output.loss.detach(), model.embedding.weight.grad.clone(),
                    model.joint_router.output.weight.grad.clone(),
                )
                self.assertGreater(float(current[1].abs().sum()), 0)
                self.assertGreater(float(current[2].abs().sum()), 0)
                if reference is None:
                    reference = current
                else:
                    for actual, expected in zip(current, reference):
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_zero_remaining_horizon_still_learns_an_actual_halt_value(self):
        model = self.model().train()
        inputs = self.inputs()
        output = model(
            inputs, self.labels(inputs), curriculum=self.curriculum(max_passes=1),
            return_terminal_router_observations=True,
        )
        self.assertTrue(bool(output.chosen_depths.eq(1).all()))
        self.assertEqual(int(output.telemetry["joint_utility_observations"]), int(self.labels(inputs).ne(-100).sum()))
        output.auxiliary_losses["joint_utility"].backward()
        self.assertGreater(float(model.joint_router.output.bias.grad[-1].abs()), 0)
        self.assertEqual(int(torch.count_nonzero(model.joint_router.output.bias.grad[:-1])), 0)
        self.assertIsNone(model.embedding.weight.grad)

    def test_schema_objective_and_configuration_guards_are_explicit(self):
        config = self.config()
        disabled = replace(config, terminal_action_critic=False)
        self.assertNotIn("terminal_action_critic", disabled.to_dict())
        self.assertNotIn("terminal_critic_exploration", disabled.to_dict())
        self.assertTrue(config.to_dict()["terminal_action_critic"])
        self.assertEqual(
            config.logical_parameter_audit().joint_router - disabled.logical_parameter_audit().joint_router,
            config.joint_router_hidden_dim + 1,
        )
        with self.assertRaisesRegex(ValueError, "causal compute"):
            replace(config, causal_compute_budget=False)._validate_tiny()
        model = self.model()
        with self.assertRaisesRegex(ValueError, "reserved head"):
            model(
                self.inputs(), self.labels(self.inputs()), curriculum=self.curriculum(),
                return_logits=True,
            )
        with self.assertRaisesRegex(ValueError, "return_terminal_router_observations"):
            model(
                self.inputs(), self.labels(self.inputs()), curriculum=self.curriculum(),
                return_router_observations=True,
            )


if __name__ == "__main__":
    unittest.main()
