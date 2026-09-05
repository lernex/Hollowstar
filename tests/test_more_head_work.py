"""Report actual LM-head rows and replay, not depth-based proxy row counts."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from metis_training.model import CurriculumState, Metis16ForCausalLM, _LMHeadWork
from metis_training.model_config import Metis16Config


class LMHeadWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def inputs(self):
        return torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, -100, 4, 5]])

    def fixed(self):
        return CurriculumState(
            continuation_mode="fixed_max", routed_k_mode="fixed",
            fixed_routed_k=2, max_passes=2, memory_gate_scale=0,
            ngram_gate_scale=0, stochastic_routing=False,
        )

    def model(self, **changes):
        torch.manual_seed(401)
        return Metis16ForCausalLM(replace(
            Metis16Config.tiny_for_tests(), lm_head_chunk_size=2, **changes
        ))

    def test_legacy_replay_counters_become_final_only_after_backward(self):
        model = self.model().train()
        inputs, labels = self.inputs()
        output = model(inputs, labels, curriculum=self.fixed())
        self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), 6)
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 0)
        output.loss.backward()
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 6)
        self.assertEqual(
            int(output.telemetry["lm_head_recompute_flops"]),
            6 * 2 * model.config.vocab_size * model.config.d_model,
        )
        model.zero_grad(set_to_none=True)
        second = model(inputs, labels, curriculum=self.fixed())
        self.assertEqual(int(second.telemetry["lm_head_recompute_rows"]), 0)
        second.loss.backward()
        self.assertEqual(int(second.telemetry["lm_head_recompute_rows"]), 6)
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 6)

    def test_lean_fixed_reference_only_replays_its_terminal_head(self):
        model = self.model(joint_compute_router=True, causal_compute_budget=True).train()
        inputs, labels = self.inputs()
        output = model(inputs, labels, curriculum=self.fixed())
        self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), 3)
        output.loss.backward()
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 3)
        self.assertEqual(int(output.telemetry["joint_budget_enforced"]), 0)

    def test_terminal_q_counts_one_evaluated_head_per_supervised_token(self):
        model = self.model(
            joint_compute_router=True, causal_compute_budget=True,
            terminal_action_critic=True, joint_router_hidden_dim=8,
        ).train()
        with torch.no_grad():
            model.joint_router.output.bias[:2].copy_(torch.tensor([1., 2.]))
            model.joint_router.output.bias[-1] = -5
        inputs, labels = self.inputs()
        output = model(
            inputs, labels,
            curriculum=replace(
                self.fixed(), compute_allocation_mode="joint",
                continuation_mode="budgeted", routed_k_mode="budgeted",
            ),
        )
        self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), 3)
        (output.loss + output.auxiliary_losses["joint_utility"]).backward()
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 3)
        self.assertEqual(int(output.telemetry["joint_utility_observations"]), 3)

    def test_no_grad_evaluation_has_no_checkpoint_replay(self):
        model = self.model().eval()
        inputs, labels = self.inputs()
        with torch.no_grad():
            output = model(inputs, labels, curriculum=self.fixed())
        self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), 6)
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 0)
        self.assertEqual(int(output.telemetry["lm_head_recompute_flops"]), 0)

    def test_optional_final_logits_count_as_an_additional_forward_only(self):
        model = self.model().train()
        inputs, labels = self.inputs()
        output = model(inputs, labels, curriculum=self.fixed(), return_logits=True)
        self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), 10)
        self.assertEqual(
            int(output.telemetry["lm_head_forward_flops"]),
            10 * 2 * model.config.vocab_size * model.config.d_model,
        )
        output.loss.backward()
        self.assertEqual(int(output.telemetry["lm_head_recompute_rows"]), 6)

    def test_synchronized_dummy_rows_are_counted_as_real_execution_work(self):
        model = self.model().train()
        hidden = torch.randn(1, 1, model.config.d_model, requires_grad=True)
        work = _LMHeadWork(
            torch.zeros((), dtype=torch.int64), torch.zeros((), dtype=torch.int64),
            torch.zeros((), dtype=torch.int64), 2 * model.config.vocab_size * model.config.d_model,
        )
        with (
            patch("metis_training.model._precision_requires_synchronized_schedule", return_value=True),
            patch("metis_training.model._group_world_size", return_value=2),
            patch("metis_training.model.dist.all_reduce", side_effect=lambda value, **kwargs: value.fill_(4)),
        ):
            weighted, _ = model._chunked_weighted_causal_loss_sum(
                hidden, torch.tensor([[1]]), torch.zeros(1, 1),
                compute_mask=torch.zeros(1, 1, dtype=torch.bool),
                return_token_losses=True, head_work=work,
            )
            self.assertEqual(int(work.forward_rows), 2)
            self.assertEqual(int(work.recompute_rows), 0)
            weighted.backward()
            self.assertEqual(int(work.recompute_rows), 2)
            self.assertEqual(int(work.recompute_flops), 2 * work.flops_per_row)


if __name__ == "__main__":
    unittest.main()
