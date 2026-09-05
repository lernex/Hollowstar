"""Causal admission spends only earned prefix credit, independently per row."""

import unittest
from unittest.mock import patch

import torch

import metis_training.compute_budget as budget_module
from metis_training.compute_budget import allocate_causal_budget
from tests.test_joint_compute_budget import _exhaustive_optimum, _scalar_account


class CausalComputeBudgetTests(unittest.TestCase):
    def allocate(self, depth, width, mask, **changes):
        kwargs = dict(
            base_pass_costs=[1] * width.shape[2],
            expert_costs=[1] * width.shape[3],
            credit_per_token=4,
        )
        return allocate_causal_budget(depth, width, mask, **(kwargs | changes))

    def fixture(self):
        generator = torch.Generator().manual_seed(80)
        return (
            torch.randn(2, 7, 3, generator=generator, dtype=torch.float64),
            torch.randn(2, 7, 2, 2, 3, generator=generator, dtype=torch.float64),
            torch.tensor([[True, True, True, False, True, True, True],
                          [True, False, True, True, True, True, False]]),
        )

    def test_every_prefix_cost_is_bounded_and_scalar_accounting_agrees(self):
        depth, width, mask = self.fixture()
        plan = self.allocate(depth, width, mask)
        cost, utility = _scalar_account(plan, depth, width, mask, [1, 1], [1, 1])
        self.assertEqual(int(plan.cost), cost)
        self.assertAlmostEqual(float(plan.utility), utility, places=10)
        earned = mask.long().cumsum(dim=1) * 4
        self.assertTrue(bool((plan.token_costs.cumsum(dim=1) <= earned).all()))
        torch.testing.assert_close(
            plan.prefix_slack, earned - plan.token_costs.cumsum(dim=1), rtol=0, atol=0
        )
        self.assertEqual(int(plan.unused_budget), int(plan.prefix_slack[:, -1].sum()))
        self.assertEqual(int(plan.budget), int(mask.sum()) * 4)

    def test_unused_actual_credit_carries_forward_but_never_backward(self):
        depth = torch.tensor([[[0., -1., 10.], [0., -1., 10.]]])
        width = torch.zeros(1, 2, 2, 1, 1)
        mask = torch.ones(1, 2, dtype=torch.bool)
        plan = self.allocate(depth, width, mask, credit_per_token=2)
        self.assertEqual(plan.depths.tolist(), [[0, 2]])
        self.assertEqual(plan.token_costs.tolist(), [[0, 4]])
        self.assertEqual(plan.prefix_slack.tolist(), [[2, 0]])
        self.assertEqual(int(plan.cost), 4)

    def test_minimum_depth_preserves_context_without_inventing_credit(self):
        depth, width, mask = self.fixture()
        depth[..., 0] = 1e6
        plan = self.allocate(depth, width, mask, minimum_depth=1)
        self.assertTrue(bool((plan.depths[mask] >= 1).all()))
        self.assertEqual(int(plan.depths[~mask].sum()), 0)
        self.assertTrue(bool((plan.token_costs.cumsum(1) <= mask.long().cumsum(1) * 4).all()))
        prefix = self.allocate(depth[:, :3], width[:, :3], mask[:, :3], minimum_depth=1)
        solo = self.allocate(depth[:1], width[:1], mask[:1], minimum_depth=1)
        torch.testing.assert_close(plan.depths[:, :3], prefix.depths)
        torch.testing.assert_close(plan.routed_k[..., :3], prefix.routed_k)
        torch.testing.assert_close(plan.depths[:1], solo.depths)
        torch.testing.assert_close(plan.routed_k[:, :, :1], solo.routed_k)
        cost, utility = _scalar_account(plan, depth, width, mask, [1, 1], [1, 1])
        self.assertEqual(int(plan.cost), cost)
        self.assertAlmostEqual(float(plan.utility), utility, places=10)
        with self.assertRaisesRegex(ValueError, "fund its minimum"):
            self.allocate(depth, width, mask, minimum_depth=2)
        for invalid in (-1, 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.allocate(depth, width, mask, minimum_depth=invalid)
        with self.assertRaises(TypeError):
            self.allocate(depth, width, mask, minimum_depth=True)

    def test_future_values_masks_and_sequence_length_do_not_change_prefix(self):
        depth, width, mask = self.fixture()
        prefix = 3
        expected = self.allocate(depth, width, mask, price=.2)
        changed_depth, changed_width, changed_mask = depth.clone(), width.clone(), mask.clone()
        changed_depth[:, prefix:] = 1e30
        changed_width[:, prefix:] = -1e30
        changed_mask[:, prefix:] = ~changed_mask[:, prefix:]
        changed = self.allocate(changed_depth, changed_width, changed_mask, price=.2)
        truncated = self.allocate(depth[:, :prefix], width[:, :prefix], mask[:, :prefix], price=.2)
        for actual in (changed, truncated):
            torch.testing.assert_close(actual.depths[:, :prefix], expected.depths[:, :prefix])
            torch.testing.assert_close(actual.routed_k[..., :prefix], expected.routed_k[..., :prefix])
            torch.testing.assert_close(actual.prefix_slack[:, :prefix], expected.prefix_slack[:, :prefix])

    def test_batch_permutation_composition_and_duplicate_rows_do_not_couple(self):
        depth, width, mask = self.fixture()
        full = self.allocate(depth, width, mask)
        solo = self.allocate(depth[:1], width[:1], mask[:1])
        order = torch.tensor([1, 0, 0])
        changed = self.allocate(depth[order], width[order], mask[order])
        torch.testing.assert_close(solo.depths[0], full.depths[0])
        torch.testing.assert_close(solo.routed_k[:, :, 0], full.routed_k[:, :, 0])
        for index in (1, 2):
            torch.testing.assert_close(changed.depths[index], full.depths[0])
            torch.testing.assert_close(changed.routed_k[:, :, index], full.routed_k[:, :, 0])

    def test_one_budget_allows_depth_width_trade_at_identical_cost(self):
        width = torch.tensor([[[[[0., 4., 8.]], [[0., 0., 0.]]]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        shallow = self.allocate(torch.tensor([[[0., 0., 0.]]]), width, mask)
        deep = self.allocate(torch.tensor([[[0., 0., 20.]]]), width, mask)
        self.assertEqual(shallow.depths.tolist(), [[1]])
        self.assertEqual(shallow.routed_k[:, 0, 0, 0].tolist(), [3, 0])
        self.assertEqual(deep.depths.tolist(), [[2]])
        self.assertEqual(deep.routed_k[:, 0, 0, 0].tolist(), [1, 1])
        self.assertEqual(int(shallow.cost), 4)
        self.assertEqual(int(deep.cost), 4)

    def test_ties_halt_and_masked_tokens_never_earn_or_spend_credit(self):
        depth = torch.zeros(1, 3, 2)
        width = torch.zeros(1, 3, 1, 1, 2)
        mask = torch.tensor([[True, False, True]])
        plan = self.allocate(depth, width, mask, credit_per_token=3)
        self.assertEqual(plan.depths.tolist(), [[0, 0, 0]])
        self.assertEqual(plan.prefix_slack.tolist(), [[3, 3, 6]])
        self.assertEqual(int(plan.cost), 0)

    def test_fixed_price_can_leave_honest_unused_credit(self):
        depth = torch.tensor([[[0., 1.]]])
        width = torch.zeros(1, 1, 1, 1, 1)
        mask = torch.ones(1, 1, dtype=torch.bool)
        free = self.allocate(depth, width, mask, price=0, cost_scale=2)
        priced = self.allocate(depth, width, mask, price=2, cost_scale=2)
        self.assertEqual(int(free.depths), 1)
        self.assertEqual(int(priced.depths), 0)
        self.assertEqual(int(priced.unused_budget), 4)

    def test_dual_certificate_bounds_even_the_relaxed_global_optimum(self):
        generator = torch.Generator().manual_seed(81)
        depth = torch.randn(1, 2, 3, generator=generator, dtype=torch.float64)
        width = torch.randn(1, 2, 2, 1, 2, generator=generator, dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        for credit in (0, 1, 3, 6):
            with self.subTest(credit=credit):
                plan = self.allocate(depth, width, mask, credit_per_token=credit, price=.3)
                optimum = _exhaustive_optimum(depth, width, mask, [1, 1], [1], credit * 2)
                self.assertGreaterEqual(float(plan.dual_upper_bound), optimum)
                self.assertLessEqual(float(plan.utility), optimum + 1e-12)

    def test_large_costs_are_exact_and_empty_shapes_are_supported(self):
        cost = 2**53 + 17
        plan = self.allocate(
            torch.tensor([[[0., 1.]]]), torch.zeros(1, 1, 1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            base_pass_costs=[cost], credit_per_token=cost + 1,
        )
        self.assertEqual(int(plan.cost), cost + 1)
        for b, s, r in ((0, 3, 2), (2, 0, 2), (2, 3, 0)):
            with self.subTest(shape=(b, s, r)):
                plan = self.allocate(
                    torch.zeros(b, s, r + 1), torch.zeros(b, s, r, 1, 2),
                    torch.ones(b, s, dtype=torch.bool),
                )
                self.assertEqual(plan.depths.shape, (b, s))
                self.assertEqual(int(plan.cost), 0)

    def test_invalid_price_credit_and_cost_overflow_are_explicit(self):
        depth, width, mask = self.fixture()
        for kwargs in (dict(price=-1), dict(price=float("nan")), dict(credit_per_token=-1),
                       dict(cost_scale=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.allocate(depth, width, mask, **kwargs)
        with self.assertRaises(TypeError):
            self.allocate(depth, width, mask, credit_per_token=1.5)
        with self.assertRaises(OverflowError):
            self.allocate(depth, width, mask, credit_per_token=2**62)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fused_cuda_admission_matches_cpu(self):
        depth, width, mask = self.fixture()
        for minimum_depth in (0, 1):
            with self.subTest(minimum_depth=minimum_depth):
                cpu = self.allocate(depth, width, mask, price=.2, minimum_depth=minimum_depth)
                gpu = self.allocate(
                    depth.cuda(), width.cuda(), mask.cuda(), price=.2, minimum_depth=minimum_depth,
                )
                for name in ("depths", "routed_k", "token_costs", "prefix_slack"):
                    torch.testing.assert_close(getattr(gpu, name).cpu(), getattr(cpu, name), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_full_length_cuda_admission_with_large_accumulated_credit(self):
        # Terminal exploration can halt long prefixes. Exercise the real
        # sequence/menu shape and an int64 balance far beyond int32, not only
        # the tiny fixture's frequently depleted balance.
        depth = torch.full((6, 4096, 5), -9.0, dtype=torch.float64)
        depth[..., 0] = -8
        depth[:, 2048:, 4] = 5
        width = torch.zeros(6, 4096, 4, 2, 8, dtype=torch.float64)
        mask = torch.ones(6, 4096, dtype=torch.bool)
        kwargs = dict(
            base_pass_costs=[1_000_000_003] * 4,
            expert_costs=[10_000_019, 20_000_033],
            credit_per_token=1_800_000_123, cost_scale=10_000_000_000,
        )
        cpu = allocate_causal_budget(depth, width, mask, **kwargs)
        self.assertGreater(int(cpu.prefix_slack.max()), 2**40)
        gpu = allocate_causal_budget(depth.cuda(), width.cuda(), mask.cuda(), **kwargs)
        for name in ("depths", "routed_k", "token_costs", "prefix_slack"):
            torch.testing.assert_close(getattr(gpu, name).cpu(), getattr(cpu, name), rtol=0, atol=0)
        self.assertEqual(int(gpu.cost), int(cpu.cost))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Multiple CUDA devices unavailable")
    def test_noncurrent_cuda_device_is_guarded_and_restored(self):
        depth, width, mask = self.fixture()
        original_device = torch.cuda.current_device()
        target = (original_device + 1) % torch.cuda.device_count()
        device = torch.device("cuda", target)
        expected = self.allocate(depth, width, mask)
        seen = []
        kernel = budget_module._causal_admission_kernel

        class LaunchSpy:
            def __getitem__(self, grid):
                seen.append(torch.cuda.current_device())
                launch = kernel[grid]

                def run(*args, **kwargs):
                    seen.append(torch.cuda.current_device())
                    return launch(*args, **kwargs)

                return run

        with patch.object(budget_module, "_causal_admission_kernel", LaunchSpy()):
            actual = self.allocate(depth.to(device), width.to(device), mask.to(device))
        self.assertEqual(seen, [target, target])
        self.assertEqual(torch.cuda.current_device(), original_device)
        torch.testing.assert_close(actual.depths.cpu(), expected.depths, rtol=0, atol=0)
        torch.testing.assert_close(actual.routed_k.cpu(), expected.routed_k, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
