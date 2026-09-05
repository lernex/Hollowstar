"""A causal compute policy must survive suffix and batch-composition changes."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from metis_training.compute_budget import allocate_causal_budget
from metis_training.compute_router import JointComputeCosts
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class CausalModelBudgetTests(unittest.TestCase):
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
            joint_router_hidden_dim=8, causal_compute_budget=True, **changes,
        )

    def model(self):
        torch.manual_seed(103)
        model = Metis16ForCausalLM(self.config()).eval()
        with torch.no_grad():
            model.joint_router.output.weight.normal_(std=.3)
            model.joint_router.output.bias[:2].copy_(torch.tensor([2., 4.]))
        return model

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                compute_allocation_mode="joint", continuation_mode="budgeted",
                routed_k_mode="budgeted", fixed_routed_k=2,
                memory_gate_scale=0, ngram_gate_scale=0,
                allow_untrained_joint_router=True, stochastic_routing=False,
                random_policy_seed=123, random_policy_step=7,
            ),
            **changes,
        )

    def inputs(self):
        return torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        ])

    def labels(self, inputs):
        labels = inputs.roll(-1, dims=1)
        labels[:, -1] = -100
        return labels

    def test_suffix_and_truncation_preserve_prefix_logits_and_decisions(self):
        model = self.model()
        inputs = self.inputs()
        changed = inputs.clone()
        changed[:, 5:] += 40
        with torch.no_grad():
            full = model(inputs, curriculum=self.curriculum(), return_logits=True)
            suffix = model(changed, curriculum=self.curriculum(), return_logits=True)
            prefix = model(inputs[:, :5], curriculum=self.curriculum(), return_logits=True)
        self.assertGreater(int(full.chosen_depths.max()), 1)
        for actual in (suffix, prefix):
            torch.testing.assert_close(actual.logits[:, :5], full.logits[:, :5], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual.chosen_depths[:, :5], full.chosen_depths[:, :5])

    def test_batch_permutation_duplicates_and_solo_inference_are_invariant(self):
        model = self.model()
        inputs = self.inputs()
        with torch.no_grad():
            full = model(inputs, curriculum=self.curriculum(), return_logits=True)
            solo = model(inputs[:1], curriculum=self.curriculum(), return_logits=True)
            changed = model(inputs[[1, 0, 0]], curriculum=self.curriculum(), return_logits=True)
        torch.testing.assert_close(solo.logits[0], full.logits[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(solo.chosen_depths[0], full.chosen_depths[0])
        for index in (1, 2):
            torch.testing.assert_close(changed.logits[index], full.logits[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(changed.chosen_depths[index], full.chosen_depths[0])

    def test_training_exploration_is_also_prefix_and_batch_invariant(self):
        model = self.model().train()
        inputs = self.inputs()
        changed = inputs.clone()
        changed[:, 5:] += 40
        curriculum = self.curriculum(stochastic_routing=True, joint_router_exploration=.2)
        with torch.no_grad():
            full = model(inputs, curriculum=curriculum, return_logits=True)
            suffix = model(changed, curriculum=curriculum, return_logits=True)
            solo = model(inputs[:1], curriculum=curriculum, return_logits=True)
        torch.testing.assert_close(suffix.logits[:, :5], full.logits[:, :5], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(solo.logits[0], full.logits[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(suffix.chosen_depths[:, :5], full.chosen_depths[:, :5])
        torch.testing.assert_close(solo.chosen_depths[0], full.chosen_depths[0])

    def test_commits_once_never_calls_global_solver_and_accounts_actual_prefix_cost(self):
        model = self.model()
        inputs = self.inputs()
        labels = self.labels(inputs)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[0, -2:] = False
        labels[~mask] = -100
        observed, predictions = [], []
        handles = [
            model.joint_router.register_forward_hook(
                lambda module, args, kwargs, result: predictions.append(kwargs["origin_pass"]),
                with_kwargs=True,
            )
        ]
        for index, layer in enumerate(model.layers):
            handles.append(layer.moe.register_forward_hook(
                lambda module, args, kwargs, result, index=index: observed.append(
                    (index, kwargs["pass_index"], result[1].mean_k.detach().long())
                ),
                with_kwargs=True,
            ))
        try:
            with (
                torch.no_grad(),
                patch("metis_training.compute_budget.allocate_causal_budget", wraps=allocate_causal_budget) as allocate,
                patch("metis_training.compute_budget.allocate_joint_budget", side_effect=AssertionError("noncausal solver used")),
            ):
                output = model(
                    inputs, labels, attention_mask=mask,
                    curriculum=self.curriculum(pathway_mode="frozen"),
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(predictions, [0])
        self.assertEqual(allocate.call_args.kwargs["price"], model.config.causal_compute_price)
        costs = model.joint_router.costs
        token_cost = mask.long() * costs.router_per_token
        for r, active in enumerate(output.active_masks):
            token_cost += active.long() * costs.base_pass_costs[r]
        for layer, r, widths in observed:
            active = output.active_masks[r]
            full_width = torch.zeros_like(inputs)
            full_width[active] = widths.flatten()
            token_cost += full_width * costs.expert_costs[layer]
        self.assertTrue(bool((token_cost.cumsum(1) <= mask.long().cumsum(1) * costs.reference_per_token).all()))
        self.assertEqual(int(token_cost.sum()), int(output.telemetry["joint_model_flops"]))
        self.assertLessEqual(int(token_cost.sum()), int(output.telemetry["joint_budget_flops"]))
        self.assertEqual(int(output.chosen_depths[~mask].sum()), 0)

    def test_lean_fixed_reference_emulates_baseline_without_dead_policy_or_head_work(self):
        config = self.config()
        old_config = replace(config, joint_compute_router=False, causal_compute_budget=False)
        torch.manual_seed(104)
        legacy = Metis16ForCausalLM(old_config).eval()
        candidate = Metis16ForCausalLM(config).eval()
        missing = candidate.load_state_dict(legacy.state_dict(), strict=False).missing_keys
        self.assertTrue(all(name.startswith("joint_router.") for name in missing))
        inputs = self.inputs()
        labels = self.labels(inputs)
        curriculum = self.curriculum(
            compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", max_passes=2, pathway_mode="frozen",
        )
        counts = Counter()
        modules = {"continuation": candidate.continuation, "critic": candidate.joint_router,
                   "lm_head": candidate.lm_head, "route_projection": candidate.depth_memory.route_projection}
        modules.update({f"k{i}": layer.moe.k_router for i, layer in enumerate(candidate.layers)})
        handles = [
            module.register_forward_hook(lambda module, args, result, name=name: counts.update([name]))
            for name, module in modules.items()
        ]
        try:
            left = legacy(inputs, labels, curriculum=curriculum)
            right = candidate(inputs, labels, curriculum=curriculum)
            left.loss.backward()
            right.loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(right.final_hidden_state, left.final_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(right.loss, left.loss, rtol=0, atol=0)
        torch.testing.assert_close(candidate.embedding.weight.grad, legacy.embedding.weight.grad, rtol=1e-4, atol=1e-5)
        self.assertEqual(counts["continuation"], 0)
        self.assertEqual(counts["critic"], 0)
        self.assertEqual(sum(counts[f"k{i}"] for i in range(config.n_layers)), 0)
        self.assertEqual(counts["lm_head"], 1)
        self.assertEqual(counts["route_projection"], config.n_layers * 2)
        costs = candidate.joint_router.costs
        self.assertEqual(int(right.telemetry["joint_model_flops"]), inputs.numel() * costs.reference_per_token)
        original = JointComputeCosts.from_config(old_config)
        self.assertEqual(
            costs.reference_per_token,
            original.reference_per_token - 2 * costs.removed_policy_per_pass - costs.head_per_token,
        )

    def test_outcomes_train_critic_and_lm_trains_backbone_with_replay_parity(self):
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
                self.assertGreater(int(output.telemetry["joint_utility_observations"]), 0)
                self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.joint_router.output.weight.grad.abs().sum()), 0)
                self.assertIsNone(model.continuation.output.weight.grad)
                current = (output.loss.detach(), model.embedding.weight.grad.clone(),
                           model.joint_router.output.weight.grad.clone())
                if reference is None:
                    reference = current
                else:
                    for actual, expected in zip(current, reference):
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_causal_mode_rejects_unrepaired_legacy_policy_and_unaudited_rm(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "fixed depth"):
            model(self.inputs(), curriculum=self.curriculum(compute_allocation_mode="legacy"))
        with self.assertRaisesRegex(ValueError, "memory_gate_scale=0"):
            model(self.inputs(), curriculum=self.curriculum(memory_gate_scale=1))
        with self.assertRaises(ValueError):
            replace(self.config(), joint_compute_router=False)._validate_tiny()
        for value in (-1, float("nan"), "current_batch"):
            with self.subTest(price=value), self.assertRaises(ValueError):
                self.config(causal_compute_price=value)._validate_tiny()

    def test_disabled_flags_preserve_manifest_and_enabled_price_is_explicit(self):
        legacy = Metis16Config.tiny_for_tests()
        self.assertNotIn("causal_compute_budget", legacy.to_dict())
        self.assertNotIn("causal_compute_price", legacy.to_dict())
        enabled = self.config(causal_compute_price=.1)
        self.assertTrue(enabled.to_dict()["causal_compute_budget"])
        self.assertEqual(enabled.to_dict()["causal_compute_price"], .1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_full_model_causal_prefix(self):
        model = self.model().cuda()
        inputs = self.inputs().cuda()
        changed = inputs.clone()
        changed[:, 5:] += 40
        with torch.no_grad():
            left = model(inputs, curriculum=self.curriculum(), return_logits=True)
            right = model(changed, curriculum=self.curriculum(), return_logits=True)
        torch.testing.assert_close(left.chosen_depths[:, :5], right.chosen_depths[:, :5])
        torch.testing.assert_close(left.logits[:, :5], right.logits[:, :5], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
