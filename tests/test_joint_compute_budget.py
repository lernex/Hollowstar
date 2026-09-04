"""Pin hard joint allocation to independent scalar accounting and tiny oracles."""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_training.compute_budget import JointBudgetPlan, allocate_joint_budget


def _scalar_account(plan, depth, width, mask, base, experts):
    cost = 0
    utility = 0.0
    batch, sequence, rounds, layers, _ = width.shape
    for b in range(batch):
        for s in range(sequence):
            d = int(plan.depths[b, s])
            if not bool(mask[b, s]):
                assert d == 0
            else:
                utility += float(depth[b, s, d])
            for r in range(rounds):
                if r < d:
                    cost += base[r]
                for layer in range(layers):
                    k = int(plan.routed_k[r, layer, b, s])
                    if r >= d or not bool(mask[b, s]):
                        assert k == 0
                    else:
                        assert 1 <= k <= width.shape[-1]
                        cost += experts[layer] * k
                        utility += float(width[b, s, r, layer, k - 1])
    return cost, utility


def _exhaustive_optimum(depth, width, mask, base, experts, budget):
    """Independent multiple-choice knapsack oracle, only for tiny test tensors."""
    token_options = []
    batch, sequence, rounds, layers, choices = width.shape
    for b in range(batch):
        for s in range(sequence):
            if not bool(mask[b, s]):
                token_options.append([(0, 0.0)])
                continue
            options = []
            for d in range(rounds + 1):
                for ks in itertools.product(range(1, choices + 1), repeat=d * layers):
                    cost = sum(base[:d])
                    utility = float(depth[b, s, d])
                    for index, k in enumerate(ks):
                        r, layer = divmod(index, layers)
                        cost += experts[layer] * k
                        utility += float(width[b, s, r, layer, k - 1])
                    options.append((cost, utility))
            token_options.append(options)
    return max(
        sum(option[1] for option in configuration)
        for configuration in itertools.product(*token_options)
        if sum(option[0] for option in configuration) <= budget
    )


class JointComputeBudgetTests(unittest.TestCase):
    def assert_accounting(self, plan, depth, width, mask, base, experts, budget):
        self.assertIsInstance(plan, JointBudgetPlan)
        self.assertEqual(plan.depths.shape, mask.shape)
        b, s, r, layer_count, _ = width.shape
        self.assertEqual(plan.routed_k.shape, (r, layer_count, b, s))
        self.assertEqual(plan.depths.dtype, torch.int64)
        self.assertEqual(plan.routed_k.dtype, torch.int64)
        self.assertEqual(plan.cost.dtype, torch.int64)
        self.assertEqual(plan.cost.ndim, 0)
        self.assertEqual(plan.utility.ndim, 0)
        self.assertEqual(plan.cost.device, depth.device)
        self.assertTrue(bool((plan.depths >= 0).all()))
        self.assertTrue(bool((plan.depths <= r).all()))
        expected_cost, expected_utility = _scalar_account(
            plan, depth, width, mask, base, experts
        )
        self.assertEqual(int(plan.cost), expected_cost)
        self.assertLessEqual(expected_cost, budget)
        self.assertAlmostEqual(float(plan.utility), expected_utility, places=9)
        self.assertEqual(int(plan.unused_budget), budget - expected_cost)
        self.assertGreaterEqual(float(plan.dual_upper_bound), float(plan.utility))
        self.assertAlmostEqual(
            float(plan.optimality_gap),
            float(plan.dual_upper_bound - plan.utility),
            places=9,
        )
        return expected_cost, expected_utility

    def test_one_cap_allows_depth_width_means_to_trade_from_legacy_reference(self):
        depth = torch.zeros(1, 1, 5)
        width = torch.zeros(1, 1, 4, 1, 9)
        width[0, 0, 0, 0] = 10 * torch.arange(9)
        mask = torch.ones(1, 1, dtype=torch.bool)
        base, experts = [1] * 4, [1]
        # Exactly the legacy remaining-depth=2, routed-k=4 cost, not two quotas.
        budget = 2 * (1 + 4)
        wide = allocate_joint_budget(
            depth, width, mask, base_pass_costs=base, expert_costs=experts, budget=budget
        )
        self.assertEqual(int(wide.depths), 1)
        self.assertEqual(int(wide.routed_k[0, 0]), 9)
        self.assertEqual(int(wide.cost), budget)
        depth[0, 0, 4] = 1000
        deep = allocate_joint_budget(
            depth, width, mask, base_pass_costs=base, expert_costs=experts, budget=budget
        )
        self.assertEqual(int(deep.depths), 4)
        self.assertLess(float(deep.routed_k.sum()) / float(deep.depths.sum()), 4)
        self.assertEqual(int(deep.cost), budget)
        self.assert_accounting(deep, depth, width, mask, base, experts, budget)

    def test_depth_preference_changes_width_at_fixed_actual_cost(self):
        width = torch.tensor([[[[[0.0, 4.0, 8.0]], [[0.0, 0.0, 0.0]]]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        arguments = dict(base_pass_costs=[1, 1], expert_costs=[1], budget=4)
        shallow_depth = torch.tensor([[[0.0, 0.0, 0.0]]])
        deep_depth = torch.tensor([[[0.0, 0.0, 20.0]]])
        shallow = allocate_joint_budget(shallow_depth, width, mask, **arguments)
        deep = allocate_joint_budget(deep_depth, width, mask, **arguments)
        self.assertEqual(shallow.depths.tolist(), [[1]])
        self.assertEqual(shallow.routed_k[:, 0, 0, 0].tolist(), [3, 0])
        self.assertEqual(deep.depths.tolist(), [[2]])
        self.assertEqual(deep.routed_k[:, 0, 0, 0].tolist(), [1, 1])
        self.assertEqual(int(shallow.cost), 4)
        self.assertEqual(int(deep.cost), 4)

    def test_random_caps_and_every_token_accounted_exactly_once(self):
        generator = torch.Generator().manual_seed(123)
        depth = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        width = torch.randn(2, 3, 2, 2, 3, generator=generator, dtype=torch.float64)
        mask = torch.tensor([[True, False, True], [True, True, False]])
        base, experts = [3, 7], [2, 5]
        for budget in (0, 1, 9, 10, 19, 41, 77, 1000):
            with self.subTest(budget=budget):
                plan = allocate_joint_budget(
                    depth, width, mask,
                    base_pass_costs=base, expert_costs=experts, budget=budget,
                )
                self.assert_accounting(plan, depth, width, mask, base, experts, budget)

    def test_independent_exhaustive_oracle_bounds_not_universal_optimality(self):
        rng = random.Random(51)
        mask = torch.tensor([[True, True]])
        base, experts = [1, 2], [2]
        for trial in range(12):
            depth = torch.tensor(
                [rng.randint(-3, 8) for _ in range(6)], dtype=torch.float64
            ).reshape(1, 2, 3)
            width = torch.tensor(
                [rng.randint(-4, 6) for _ in range(8)], dtype=torch.float64
            ).reshape(1, 2, 2, 1, 2)
            budget = rng.randrange(16)
            with self.subTest(trial=trial, budget=budget):
                optimum = _exhaustive_optimum(depth, width, mask, base, experts, budget)
                plan = allocate_joint_budget(
                    depth, width, mask,
                    base_pass_costs=base, expert_costs=experts, budget=budget,
                )
                self.assert_accounting(plan, depth, width, mask, base, experts, budget)
                self.assertLessEqual(float(plan.utility), optimum + 1e-10)
                self.assertGreaterEqual(float(plan.dual_upper_bound), optimum)

    def test_known_global_optimum_allocates_best_tokens_not_token_quotas(self):
        depth = torch.tensor([[[0.0, 9.0], [0.0, 3.0], [0.0, 7.0]]])
        width = torch.zeros(1, 3, 1, 1, 1)
        mask = torch.ones(1, 3, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[1], expert_costs=[1], budget=4
        )
        self.assertEqual(plan.depths.tolist(), [[1, 0, 1]])
        self.assertEqual(float(plan.utility), 16)
        self.assertEqual(
            float(plan.utility), _exhaustive_optimum(depth, width, mask, [1], [1], 4)
        )
        self.assertLess(float(plan.optimality_gap), 1e-4)

    def test_dual_bounds_nonuniform_layer_costs_and_depth_prefixes(self):
        depth = torch.tensor([[[2.0, -1.0, 5.0]]], dtype=torch.float64)
        width = torch.tensor(
            [[[[[1.0, 8.0], [-3.0, 2.0]], [[4.0, -2.0], [0.0, 7.0]]]]],
            dtype=torch.float64,
        )
        mask = torch.ones(1, 1, dtype=torch.bool)
        base, experts = [2, 1], [1, 3]
        for budget in (0, 5, 6, 9, 13, 17, 30):
            with self.subTest(budget=budget):
                plan = allocate_joint_budget(
                    depth, width, mask, base_pass_costs=base,
                    expert_costs=experts, budget=budget, iterations=1,
                )
                optimum = _exhaustive_optimum(depth, width, mask, base, experts, budget)
                self.assert_accounting(plan, depth, width, mask, base, experts, budget)
                self.assertGreaterEqual(float(plan.dual_upper_bound), optimum)
                self.assertLessEqual(float(plan.utility), optimum)

    def test_unsupported_positive_width_choice_is_available_to_repair(self):
        depth = torch.zeros(1, 1, 2)
        width = torch.tensor([[[[[0.0, 3.0, 10.0]]]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[1], expert_costs=[1], budget=3
        )
        self.assertEqual(int(plan.routed_k), 2)
        self.assertEqual(float(plan.utility), 3)
        self.assertEqual(int(plan.cost), 3)

    def test_single_layer_upgrade_uses_budget_uniform_policy_cannot(self):
        depth = torch.tensor([[[0.0, 100.0]]])
        width = torch.tensor([[[[[0.0, 1.0], [0.0, 1.0]]]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[1], expert_costs=[1, 1], budget=4
        )
        self.assertEqual(plan.routed_k[:, :, 0, 0].tolist(), [[2, 1]])
        self.assertEqual(int(plan.cost), 4)
        self.assertEqual(float(plan.utility), 101)

    def test_nonpositive_improvements_halt_and_preserve_active_halt_utility(self):
        depth = torch.tensor([[[2.0, -1.0, -5.0], [500.0, 999.0, 999.0]]])
        width = -torch.ones(1, 2, 2, 1, 3)
        mask = torch.tensor([[True, False]])
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[2, 3], expert_costs=[1], budget=1000
        )
        self.assertEqual(plan.depths.tolist(), [[0, 0]])
        self.assertEqual(int(plan.routed_k.sum()), 0)
        self.assertEqual(int(plan.cost), 0)
        self.assertEqual(float(plan.utility), 2)
        self.assertEqual(int(plan.unused_budget), 1000)

    def test_zero_utility_ties_do_not_create_work(self):
        depth = torch.zeros(1, 4, 3)
        width = torch.zeros(1, 4, 2, 2, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[1, 1], expert_costs=[1, 2], budget=999
        )
        self.assertEqual(int(plan.depths.sum()), 0)
        self.assertEqual(int(plan.cost), 0)

    def test_depth_and_width_ties_choose_cheapest_plan(self):
        depth = torch.tensor([[[0.0, 5.0, 5.0]]])
        width = torch.zeros(1, 1, 2, 2, 3)
        mask = torch.ones(1, 1, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[2, 1], expert_costs=[3, 1], budget=100
        )
        self.assertEqual(int(plan.depths), 1)
        self.assertEqual(plan.routed_k[:, :, 0, 0].tolist(), [[1, 1], [0, 0]])
        self.assertEqual(int(plan.cost), 6)

    def test_global_equal_density_tie_uses_first_active_token(self):
        depth = torch.tensor([[[0.0, 100.0], [0.0, 1.0], [0.0, 1.0]]])
        width = torch.zeros(1, 3, 1, 1, 1)
        mask = torch.tensor([[False, True, True]])
        plans = [
            allocate_joint_budget(
                depth, width, mask, base_pass_costs=[1], expert_costs=[1], budget=2
            )
            for _ in range(2)
        ]
        for plan in plans:
            self.assertEqual(plan.depths.tolist(), [[0, 1, 0]])
            self.assertEqual(int(plan.cost), 2)

    def test_budget_below_continuation_has_honest_nonzero_dual_gap(self):
        plan = allocate_joint_budget(
            torch.tensor([[[0.0, 1.0]]]), torch.zeros(1, 1, 1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            base_pass_costs=[1], expert_costs=[1], budget=1,
        )
        self.assertEqual(int(plan.cost), 0)
        self.assertEqual(float(plan.utility), 0)
        self.assertGreater(float(plan.optimality_gap), 0.49)
        self.assertLess(float(plan.optimality_gap), 0.51)

    def test_noncontiguous_inputs_and_tensor_costs(self):
        generator = torch.Generator().manual_seed(16)
        depth = torch.randn(2, 3, 6, generator=generator)[..., ::2]
        width = torch.randn(2, 3, 2, 2, 6, generator=generator)[..., ::2]
        mask = torch.tensor([[True, False], [True, True], [False, True]]).t()
        self.assertFalse(depth.is_contiguous())
        self.assertFalse(width.is_contiguous())
        self.assertFalse(mask.is_contiguous())
        arguments = dict(
            base_pass_costs=torch.tensor([2, 4]),
            expert_costs=torch.tensor([1, 3]),
            budget=torch.tensor(43, dtype=torch.int64),
        )
        plan = allocate_joint_budget(depth, width, mask, **arguments)
        contiguous = allocate_joint_budget(
            depth.contiguous(), width.contiguous(), mask.contiguous(), **arguments
        )
        self.assertTrue(torch.equal(plan.depths, contiguous.depths))
        self.assertTrue(torch.equal(plan.routed_k, contiguous.routed_k))
        self.assert_accounting(plan, depth, width, mask, [2, 4], [1, 3], 43)

    def test_large_integer_costs_do_not_round_budget_feasibility(self):
        unit = 2**53 + 17
        depth = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
        width = torch.zeros(1, 2, 1, 1, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        plan = allocate_joint_budget(
            depth, width, mask, base_pass_costs=[unit], expert_costs=[1], budget=unit + 1
        )
        self.assertEqual(int(plan.cost), unit + 1)
        self.assertEqual(plan.depths.tolist(), [[1, 0]])
        self.assert_accounting(plan, depth, width, mask, [unit], [1], unit + 1)

    def test_large_finite_utilities_are_normalized_before_float32_search(self):
        depth = torch.tensor([[[0.0, 1e100], [0.0, 2e100]]], dtype=torch.float64)
        width = torch.zeros(1, 2, 1, 1, 1, dtype=torch.float64)
        plan = allocate_joint_budget(
            depth, width, torch.ones(1, 2, dtype=torch.bool),
            base_pass_costs=[1], expert_costs=[1], budget=2,
        )
        self.assertEqual(plan.depths.tolist(), [[0, 1]])
        self.assertEqual(float(plan.utility), 2e100)
        self.assertTrue(bool(torch.isfinite(plan.dual_upper_bound)))
        self.assertGreaterEqual(float(plan.dual_upper_bound), 2e100)

    def test_outputs_are_detached_not_straight_through(self):
        depth = torch.tensor([[[0.0, 2.0]]], requires_grad=True)
        width = torch.zeros(1, 1, 1, 1, 2, requires_grad=True)
        plan = allocate_joint_budget(
            depth, width, torch.ones(1, 1, dtype=torch.bool),
            base_pass_costs=[1], expert_costs=[1], budget=2,
        )
        for value in (plan.depths, plan.routed_k, plan.cost, plan.utility,
                      plan.dual_upper_bound, plan.optimality_gap):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)

    def test_zero_active_empty_batch_and_no_remaining_rounds(self):
        for b, s, r in ((2, 3, 2), (0, 3, 2), (2, 0, 2), (2, 3, 0)):
            with self.subTest(shape=(b, s, r)):
                depth = torch.ones(b, s, r + 1)
                width = torch.zeros(b, s, r, 2, 3)
                mask = torch.ones(b, s, dtype=torch.bool) if r == 0 else torch.zeros(b, s, dtype=torch.bool)
                plan = allocate_joint_budget(
                    depth, width, mask, base_pass_costs=[1] * r,
                    expert_costs=[2, 3], budget=0,
                )
                self.assert_accounting(plan, depth, width, mask, [1] * r, [2, 3], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu_and_keeps_outputs_on_device(self):
        depth = torch.tensor([[[0.0, 9.0], [0.0, 3.0], [0.0, 7.0]]])
        width = torch.zeros(1, 3, 1, 1, 2)
        mask = torch.tensor([[True, True, True]])
        kwargs = dict(base_pass_costs=[1], expert_costs=[1], budget=4)
        cpu = allocate_joint_budget(depth, width, mask, **kwargs)
        gpu = allocate_joint_budget(depth.cuda(), width.cuda(), mask.cuda(), **kwargs)
        self.assertTrue(torch.equal(cpu.depths, gpu.depths.cpu()))
        self.assertTrue(torch.equal(cpu.routed_k, gpu.routed_k.cpu()))
        self.assertEqual(gpu.cost.device.type, "cuda")
        self.assertEqual(gpu.utility.device.type, "cuda")
        self.assertEqual(gpu.dual_upper_bound.device.type, "cuda")


class JointComputeBudgetValidationTests(unittest.TestCase):
    def setUp(self):
        self.depth = torch.zeros(1, 2, 3)
        self.width = torch.zeros(1, 2, 2, 1, 3)
        self.mask = torch.ones(1, 2, dtype=torch.bool)
        self.kwargs = dict(base_pass_costs=[1, 2], expert_costs=[1], budget=9)

    def test_invalid_shapes_and_mask_types(self):
        cases = (
            (self.depth[:, :, :2], self.width, self.mask),
            (self.depth, self.width.squeeze(3), self.mask),
            (self.depth, self.width, self.mask.float()),
            (self.depth, self.width, self.mask[:, :1]),
            (self.depth, self.width[..., :0], self.mask),
        )
        for inputs in cases:
            with self.subTest(shapes=[tuple(t.shape) for t in inputs]):
                with self.assertRaises(ValueError):
                    allocate_joint_budget(*inputs, **self.kwargs)

    def test_nonfinite_including_padding_is_explicit_error(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for which in ("depth", "width"):
                with self.subTest(value=value, which=which):
                    depth, width = self.depth.clone(), self.width.clone()
                    target = depth if which == "depth" else width
                    target.reshape(-1)[0] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        allocate_joint_budget(depth, width, torch.zeros_like(self.mask), **self.kwargs)

    def test_invalid_costs_budgets_and_iterations(self):
        cases = (
            ("base_pass_costs", [1], ValueError),
            ("base_pass_costs", [0, 1], ValueError),
            ("base_pass_costs", [-1, 1], ValueError),
            ("base_pass_costs", [1.0, 2.0], TypeError),
            ("base_pass_costs", torch.tensor([1.0, 2.0]), TypeError),
            ("expert_costs", [True], TypeError),
            ("expert_costs", [], ValueError),
            ("budget", -1, ValueError),
            ("budget", 3.0, TypeError),
            ("budget", True, TypeError),
            ("budget", torch.tensor([3]), TypeError),
            ("budget", torch.tensor(3, dtype=torch.int32), TypeError),
            ("budget", 2**63, ValueError),
            ("iterations", 0, ValueError),
            ("iterations", 2.5, ValueError),
            ("iterations", True, ValueError),
        )
        for key, value, error in cases:
            with self.subTest(key=key, value=value):
                kwargs = {**self.kwargs, key: value}
                with self.assertRaises(error):
                    allocate_joint_budget(self.depth, self.width, self.mask, **kwargs)

    def test_integer_utility_and_overflow_risks_rejected(self):
        with self.assertRaises(TypeError):
            allocate_joint_budget(self.depth.long(), self.width, self.mask, **self.kwargs)
        with self.assertRaises(OverflowError):
            allocate_joint_budget(
                self.depth, self.width, self.mask,
                **{**self.kwargs, "base_pass_costs": [2**62, 2**62]},
            )
        with self.assertRaises(OverflowError):
            allocate_joint_budget(
                self.depth, self.width, self.mask,
                **{**self.kwargs, "expert_costs": [2**63]},
            )
        with self.assertRaises(OverflowError):
            allocate_joint_budget(
                torch.full_like(self.depth, 1e308, dtype=torch.float64),
                self.width, self.mask, **self.kwargs,
            )


if __name__ == "__main__":
    unittest.main()
