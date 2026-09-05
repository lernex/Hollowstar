"""A conservative depth floor must retain context without relaxing the budget."""

from dataclasses import replace
import unittest

import torch

from metis_training.compute_router import JointComputeCosts
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class ContextFloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self, **changes):
        return replace(
            Metis16Config.tiny_for_tests(), joint_compute_router=True,
            joint_router_hidden_dim=8, causal_compute_budget=True,
            terminal_action_critic=True, causal_min_passes=2, **changes,
        )

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                compute_allocation_mode="joint", continuation_mode="budgeted",
                routed_k_mode="budgeted", fixed_routed_k=2, max_passes=3,
                memory_gate_scale=0, ngram_gate_scale=0, stochastic_routing=True,
                random_policy_seed=912, allow_untrained_joint_router=True,
            ), **changes,
        )

    def test_floor_keeps_second_pass_context_and_trains_both_core_and_rm(self):
        for rm in (False, True):
            with self.subTest(rm=rm):
                config = self.config(
                    causal_memory_metadata="legacy_confidence" if rm else "disabled",
                    terminal_critic_exploration=.5,
                    target_mean_routed_k=3,
                )
                torch.manual_seed(913)
                model = Metis16ForCausalLM(config).train()
                inputs = torch.arange(1, 65).view(2, 32)
                labels = inputs.roll(-1, 1)
                labels[:, -1] = -100
                mask = torch.ones_like(inputs, dtype=torch.bool)
                mask[0, -2:] = False
                labels[~mask] = -100
                output = model(
                    inputs, labels, attention_mask=mask,
                    curriculum=self.curriculum(memory_gate_scale=float(rm)),
                )
                torch.testing.assert_close(output.active_masks[1], mask)
                self.assertTrue(bool((output.chosen_depths[mask] >= 2).all()))
                self.assertLessEqual(
                    int(output.telemetry["joint_model_flops"]),
                    int(output.telemetry["joint_budget_flops"]),
                )
                self.assertEqual(int(output.telemetry["lm_head_forward_rows"]), int(labels.ne(-100).sum()))
                (output.loss + output.auxiliary_losses["joint_utility"]).backward()
                self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.joint_router.output.weight.grad.abs().sum()), 0)

    def test_floor_is_explicit_and_cannot_change_the_reference_compute(self):
        config = self.config()
        default = replace(config, causal_min_passes=1)
        self.assertNotIn("causal_min_passes", default.to_dict())
        self.assertEqual(config.to_dict()["causal_min_passes"], 2)
        self.assertEqual(JointComputeCosts.from_config(config), JointComputeCosts.from_config(default))
        for invalid in (0, True, 6):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(config, causal_min_passes=invalid)._validate_tiny()
        with self.assertRaisesRegex(ValueError, "requires causal"):
            replace(config, causal_compute_budget=False, terminal_action_critic=False)._validate_tiny()
        with self.assertRaisesRegex(ValueError, "below causal_min_passes"):
            self.curriculum(max_passes=1).validate(config)
        model = Metis16ForCausalLM(config).train()
        with self.assertRaisesRegex(ValueError, "below causal_min_passes"):
            model(torch.tensor([[1, 2]]), curriculum=self.curriculum(), max_passes=1)

    def test_larger_router_pays_for_capacity_and_keeps_body_initialization(self):
        config = self.config()
        torch.manual_seed(914)
        small = Metis16ForCausalLM(config)
        torch.manual_seed(914)
        large = Metis16ForCausalLM(replace(config, joint_router_hidden_dim=16))
        for name, parameter in small.named_parameters():
            if not name.startswith("joint_router."):
                torch.testing.assert_close(
                    parameter, dict(large.named_parameters())[name], rtol=0, atol=0,
                )
        self.assertEqual(small.joint_router.costs.reference_per_token, large.joint_router.costs.reference_per_token)
        self.assertGreater(large.joint_router.costs.router_per_token, small.joint_router.costs.router_per_token)


if __name__ == "__main__":
    unittest.main()
