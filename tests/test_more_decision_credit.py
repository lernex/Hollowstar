from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from metis_ablation.decision_credit_probe import (
    assemble_panel_proposal,
    budget_summary,
    diagnose_decisions,
    effect_summary,
    interventions,
    terminal_outcomes,
    token_costs,
)
from metis_ablation.routing_credit_probe import (
    FrozenRuntimeState,
    evaluate_in_memory,
    plan_cost,
    plan_fingerprint,
)
from metis_training.compute_router import JointComputeCosts
from metis_training.model import CurriculumState, Metis16ForCausalLM
from tests.test_more_routing_probe import fixed_widths, tiny_batch, tiny_config


class DecisionCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, *, joint=True, rm=False):
        config = replace(
            tiny_config(), joint_compute_router=True, causal_compute_budget=True,
            terminal_action_critic=True, joint_router_hidden_dim=4,
            target_mean_routed_k=3,
            causal_memory_metadata="legacy_confidence" if rm else "disabled",
        )
        torch.manual_seed(7621)
        model = Metis16ForCausalLM(config).eval()
        with torch.no_grad():
            model.joint_router.output.bias[0] = 2
            future = config.max_passes - 1
            widths = model.joint_router.output.bias[
                future:future + future * config.n_layers * config.max_routed_k
            ].view(future, config.n_layers, config.max_routed_k)
            widths.fill_(-.2)
            widths[..., 2] = 0
        model.joint_router.mark_trained()
        batch = tiny_batch(config, length=12)
        curriculum = CurriculumState(
            compute_allocation_mode="joint" if joint else "legacy",
            continuation_mode="budgeted" if joint else "fixed_max",
            routed_k_mode="budgeted" if joint else "fixed", fixed_routed_k=3,
            max_passes=config.max_passes if joint else 2,
            memory_gate_scale=float(rm), ngram_gate_scale=0, stochastic_routing=False,
        )
        return model, batch, curriculum

    def test_terminal_capture_preserves_native_loss_plan_and_exactly_once_coverage(self):
        model, batch, curriculum = self.fixture()
        runtime = FrozenRuntimeState(model)
        plain = evaluate_in_memory(model, batch, curriculum, seed=91, runtime_state=runtime)
        captured = evaluate_in_memory(
            model, batch, curriculum, seed=91, runtime_state=runtime,
            return_terminal_router_observations=True,
        )
        torch.testing.assert_close(plain.output.loss, captured.output.loss, rtol=0, atol=0)
        torch.testing.assert_close(plain.output.chosen_depths, captured.output.chosen_depths, rtol=0, atol=0)
        self.assertEqual(plain.cost, captured.cost)
        self.assertEqual(
            int(plain.output.telemetry["lm_head_forward_rows"]),
            int(captured.output.telemetry["lm_head_forward_rows"]),
        )
        outcomes = terminal_outcomes(captured, batch)
        self.assertEqual(int(outcomes.losses[batch.labels.eq(-100)].count_nonzero()), 0)
        torch.testing.assert_close(
            outcomes.losses[batch.labels.ne(-100)].mean(), captured.output.loss,
        )
        bad = copy.copy(captured.output)
        observed = next(
            observation for observation in captured.output.terminal_router_observations
            if bool(observation.observed_mask.any())
        )
        bad.terminal_router_observations = (
            *captured.output.terminal_router_observations,
            observed,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            terminal_outcomes(replace(captured, output=bad), batch)
        observations = list(captured.output.terminal_router_observations)
        index = next(i for i, observation in enumerate(observations) if bool(observation.observed_mask.any()))
        depths = observations[index].depths.clone()
        depths[observations[index].observed_mask] = 0
        observations[index] = replace(observations[index], depths=depths)
        bad.terminal_router_observations = tuple(observations)
        with self.assertRaisesRegex(ValueError, "different executed action"):
            terminal_outcomes(replace(captured, output=bad), batch)

    def test_prefix_costs_match_the_shared_ledger_and_charge_the_critic_separately(self):
        model, batch, curriculum = self.fixture()
        result = evaluate_in_memory(model, batch, curriculum, seed=3)
        depths, widths = result.output.chosen_depths, result.widths
        costs = token_costs(model.config, depths, widths, critic=False)
        self.assertEqual(int(costs.sum()), plan_cost(model.config, depths, widths, terminal_only=True)["nominal_train_flops"])
        charged = token_costs(model.config, depths, widths, critic=True)
        expected = int(batch.attention_mask.sum()) * JointComputeCosts.from_config(model.config).router_per_token
        self.assertEqual(int((charged - costs).sum()), expected)
        certificate = budget_summary(model.config, depths, widths, critic=True, horizon=3)
        self.assertTrue(certificate["policy_feasible"])
        excessive_depth = batch.attention_mask.long() * model.config.max_passes
        excessive_width = fixed_widths(model.config, excessive_depth, model.config.max_routed_k)
        self.assertFalse(budget_summary(
            model.config, excessive_depth, excessive_width, critic=True, horizon=3,
        )["within_every_prefix_budget"])

    def test_proposals_preserve_first_pass_and_reallocations_really_fit_each_prefix(self):
        model, batch, curriculum = self.fixture()
        result = evaluate_in_memory(model, batch, curriculum, seed=3)
        depths, widths = result.output.chosen_depths, result.widths
        proposals, coverage = interventions(
            model.config, batch, depths, widths, trials=3, seed=22, horizon=3,
            min_context=1, critic=True,
        )
        self.assertGreater(coverage["shorten"]["generated"], 0)
        self.assertGreater(coverage["narrow_terminal"]["generated"], 0)
        self.assertEqual(len(proposals), len({plan_fingerprint(p.depths, p.widths) for p in proposals}))
        for proposal in proposals:
            with self.subTest(kind=proposal.kind, trial=proposal.trial):
                torch.testing.assert_close(proposal.widths[0], widths[0], rtol=0, atol=0)
                expected = proposal.depths.ne(depths) | proposal.widths.ne(widths).any(dim=(0, 1))
                torch.testing.assert_close(proposal.changed, expected, rtol=0, atol=0)
                self.assertFalse(bool((proposal.changed & batch.labels.eq(-100)).any()))
                self.assertTrue(budget_summary(
                    model.config, proposal.depths, proposal.widths, critic=True, horizon=3,
                )["within_every_prefix_budget"])
                if proposal.kind == "exchange":
                    torch.testing.assert_close(
                        proposal.widths.sum(dim=(2, 3)), widths.sum(dim=(2, 3)), rtol=0, atol=0,
                    )
                    for depth in range(model.config.max_passes):
                        self.assertEqual(int(proposal.depths.gt(depth).sum()), int(depths.gt(depth).sum()))
                if proposal.kind == "transfer_depth":
                    self.assertEqual(int(proposal.depths.sum()), int(depths.sum()))
                    self.assertTrue(budget_summary(
                        model.config, proposal.depths, proposal.widths, critic=True, horizon=3,
                    )["policy_feasible"])
        repeated, _ = interventions(
            model.config, batch, depths, widths, trials=3, seed=22, horizon=3,
            min_context=1, critic=True,
        )
        self.assertEqual(
            [plan_fingerprint(p.depths, p.widths) for p in proposals],
            [plan_fingerprint(p.depths, p.widths) for p in repeated],
        )

    def test_shortening_outside_a_floor_is_not_mislabeled_as_feasible_headroom(self):
        model, batch, curriculum = self.fixture()
        result = evaluate_in_memory(model, batch, curriculum, seed=3)
        proposals, _ = interventions(
            replace(model.config, causal_min_passes=2), batch,
            result.output.chosen_depths, result.widths,
            trials=1, seed=22, horizon=3, min_context=1, critic=True,
        )
        short = next(proposal for proposal in proposals if proposal.kind == "shorten")
        certificate = budget_summary(
            replace(model.config, causal_min_passes=2), short.depths, short.widths,
            critic=True, horizon=3,
        )
        self.assertTrue(certificate["within_every_prefix_budget"])
        self.assertFalse(certificate["within_declared_policy_support"])
        self.assertFalse(certificate["policy_feasible"])

    def test_effects_distinguish_own_reward_from_harm_to_other_future_tokens(self):
        model, batch, _ = self.fixture()
        before = torch.ones(3, *batch.labels.shape)
        after = before.clone()
        changed = torch.zeros_like(batch.attention_mask)
        changed[0, 3] = True
        after[:, 0, 3] -= .1
        after[:, 0, 4] += .4
        after[:, 0, 5] += .3
        effect = effect_summary(before, after, batch, changed, minimum_gain=1e-4)
        self.assertAlmostEqual(effect["changed"]["loss_sum_gain"], .1, places=6)
        self.assertAlmostEqual(effect["downstream"]["loss_sum_gain"], -.7, places=6)
        self.assertAlmostEqual(effect["total"]["loss_sum_gain"], -.6, places=6)
        self.assertTrue(effect["local_and_total_opposite_sign"])
        self.assertTrue(effect["causality_controls_ok"])
        self.assertEqual(effect["unaffected"]["max_abs_token_change"], 0)
        after[:, 1, 4] += .5
        contaminated = effect_summary(before, after, batch, changed, minimum_gain=1e-4)
        self.assertFalse(contaminated["causality_controls_ok"])
        self.assertAlmostEqual(contaminated["unaffected"]["loss_sum_gain"], -.5)

    def test_terminal_layer_width_has_no_downstream_path_but_halting_does(self):
        model, batch, curriculum = self.fixture(joint=False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".moe.local_experts." in name:
                    parameter.mul_(3)
        depth = batch.attention_mask.long() * 2
        width = fixed_widths(model.config, depth, 3)
        runtime = FrozenRuntimeState(model)

        def loss_vector(d, w):
            return terminal_outcomes(evaluate_in_memory(
                model, batch, curriculum, seed=91, runtime_state=runtime,
                force_depth=d, force_routed_k=w,
                return_terminal_router_observations=True,
            ), batch).losses

        baseline = loss_vector(depth, width)
        terminal_width = width.clone()
        terminal_width[1, -1, 0, 3] = 1
        narrow = loss_vector(depth, terminal_width)
        unchanged = batch.labels.ne(-100)
        unchanged[0, 3] = False
        torch.testing.assert_close(narrow[unchanged], baseline[unchanged], rtol=0, atol=2e-5)
        self.assertGreater(float((narrow[0, 3] - baseline[0, 3]).abs()), 1e-6)
        shallow, shallow_width = depth.clone(), width.clone()
        shallow[0, 3] = 1
        shallow_width[1:, :, 0, 3] = 0
        halted = loss_vector(shallow, shallow_width)
        torch.testing.assert_close(halted[0, :3], baseline[0, :3], rtol=0, atol=2e-5)
        torch.testing.assert_close(halted[1], baseline[1], rtol=0, atol=2e-5)
        self.assertGreater(float((halted[0, 4:-1] - baseline[0, 4:-1]).abs().max()), 1e-6)

    def test_full_diagnostic_replays_native_policy_without_mutating_weights_or_buffers(self):
        for joint, rm in ((True, False), (True, True), (False, False)):
            with self.subTest(joint=joint, rm=rm):
                model, batch, curriculum = self.fixture(joint=joint, rm=rm)
                before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                report = diagnose_decisions(
                    model, batch, curriculum, seed=7, repeat_forwards=2,
                    trials_per_kind=1, min_context=1,
                )
                self.assertEqual(report["status"], "diagnostic_complete")
                self.assertLessEqual(
                    abs(report["native"]["lm_loss"] - report["replay"]["mean_loss"]),
                    report["replay"]["tolerance"],
                )
                self.assertGreater(len(report["interventions"]), 0)
                self.assertGreater(report["nominal_forward_flops_all_calls"], 0)
                if joint:
                    self.assertIsNotNone(report["replay"]["terminal_value_mse"])
                else:
                    self.assertIsNone(report["replay"]["terminal_value_mse"])
                for name, expected in before.items():
                    torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_label_informed_panel_is_replayed_instead_of_claiming_additive_gains(self):
        model, batch, curriculum = self.fixture()
        report = diagnose_decisions(
            model, batch, curriculum, seed=7, repeat_forwards=2,
            trials_per_kind=1, min_context=1, panel_oracle=True,
        )
        panel = report["local_loss_panel_oracle"]
        self.assertGreater(len(panel["uniform_teacher_programs"]), 0)
        self.assertGreater(len(panel["replayed_proposals"]), 0)
        for proposal in panel["replayed_proposals"]:
            self.assertTrue(proposal["budget"]["policy_feasible"])
            self.assertAlmostEqual(
                proposal["replayed_loss_gain"],
                report["replay"]["mean_loss"] - proposal["replayed_loss"],
            )
            self.assertIn("teacher_local_surrogate_loss", proposal)
            self.assertGreater(proposal["diagnostic_train_flops"], 0)
            self.assertAlmostEqual(
                proposal["replayed_loss_gain"],
                proposal["effects"]["total"]["loss_sum_gain"] / int(batch.labels.ne(-100).sum()),
            )

    def test_panel_admission_keeps_first_pass_and_has_no_future_credit_borrowing(self):
        model, batch, _ = self.fixture()
        plans, losses = [], []
        for depth in (1, 2, 3):
            depths = batch.attention_mask.long() * depth
            widths = fixed_widths(model.config, depths, 1)
            widths[0] = batch.attention_mask * 3
            plans.append((depths, widths))
            losses.append(torch.full_like(batch.input_ids, 4.0 - depth, dtype=torch.float))
        depths, widths, predicted = assemble_panel_proposal(
            model.config, batch.attention_mask, plans, losses,
            critic=True, horizon=3, price=0,
        )
        self.assertTrue(budget_summary(
            model.config, depths, widths, critic=True, horizon=3,
        )["policy_feasible"])
        torch.testing.assert_close(widths[0], plans[0][1][0], rtol=0, atol=0)
        expected = torch.zeros_like(batch.labels, dtype=torch.float)
        for depth, loss in zip((1, 2, 3), losses):
            expected[depths.eq(depth)] = loss[depths.eq(depth)]
        self.assertEqual(predicted, float(expected.sum()))
        expensive = fixed_widths(model.config, plans[-1][0], model.config.max_routed_k)
        expensive[0] = batch.attention_mask * 3
        broken_plans = [*plans[:-1], (plans[-1][0], expensive)]

        def borrow_future(costs, scores, mask, credit):
            return scores.argmax(dim=-1), torch.zeros_like(mask, dtype=torch.int64)

        with patch(
            "metis_ablation.decision_credit_probe._causal_admission", side_effect=borrow_future,
        ), self.assertRaisesRegex(ArithmeticError, "prefix budget"):
            assemble_panel_proposal(
                model.config, batch.attention_mask, broken_plans, losses,
                critic=True, horizon=3, price=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_accelerator_native_replay_and_arbitrary_panel_menu(self):
        model, batch, curriculum = self.fixture()
        model.apply_parameter_storage_policy(device=torch.device("cuda", 0))
        batch = batch.to(torch.device("cuda", 0))
        report = diagnose_decisions(
            model, batch, curriculum, seed=7, repeat_forwards=2,
            trials_per_kind=1, min_context=1, panel_oracle=True,
        )
        self.assertEqual(report["status"], "diagnostic_complete")
        self.assertTrue(report["replay"]["budget"]["policy_feasible"])
        self.assertTrue(all(
            proposal["budget"]["policy_feasible"]
            for proposal in report["local_loss_panel_oracle"]["replayed_proposals"]
        ))


if __name__ == "__main__":
    unittest.main()
