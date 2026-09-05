"""Frozen expert identity is independent of explicit width and replay layout."""

from dataclasses import replace
import unittest

import torch

from metis_training.model import (
    CurriculumState,
    Metis16ForCausalLM,
    PathwayCache,
    _active_token_layout,
)
from metis_training.model_config import Metis16Config


class FrozenWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **changes):
        return replace(Metis16Config.tiny_for_tests(), **changes)

    def batch(self, config):
        generator = torch.Generator().manual_seed(510)
        inputs = torch.randint(0, config.vocab_size, (2, 6), generator=generator)
        labels = inputs.roll(-1, dims=1)
        labels[:, -1] = -100
        return inputs, labels

    def curriculum(self, **changes):
        return replace(
            CurriculumState(
                continuation_mode="fixed_max",
                routed_k_mode="fixed",
                fixed_routed_k=2,
                pathway_mode="frozen",
                stochastic_routing=False,
            ),
            **changes,
        )

    def test_legacy_cache_keeps_first_width_without_an_explicit_plan(self):
        torch.manual_seed(14)
        config = self.config()
        moe = Metis16ForCausalLM(config).eval().layers[0].moe
        hidden = torch.randn(1, 2, config.d_model)
        features = torch.randn(1, 2, config.route_feature_dim)
        mask = torch.ones(1, 2, dtype=torch.bool)
        cache = PathwayCache()
        with torch.no_grad():
            _, first = moe(
                hidden, route_features=features, active_mask=mask,
                curriculum=self.curriculum(fixed_routed_k=1),
                pass_index=0, pathway_cache=cache,
            )
            _, second = moe(
                hidden, route_features=features, active_mask=mask,
                curriculum=self.curriculum(fixed_routed_k=3),
                pass_index=1, pathway_cache=cache,
            )
        self.assertTrue(cache.freeze_widths)
        self.assertEqual(first.mean_k.tolist(), [[1, 1]])
        self.assertEqual(second.mean_k.tolist(), [[1, 1]])

    def test_explicit_width_is_not_silently_overwritten_by_cached_width(self):
        torch.manual_seed(14)
        config = self.config()
        moe = Metis16ForCausalLM(config).eval().layers[0].moe
        hidden = torch.randn(1, 2, config.d_model)
        features = torch.randn(1, 2, config.route_feature_dim)
        mask = torch.ones(1, 2, dtype=torch.bool)
        cache = PathwayCache()
        with torch.no_grad():
            moe(
                hidden, route_features=features, active_mask=mask,
                curriculum=self.curriculum(), pass_index=0, pathway_cache=cache,
                forced_routed_k=torch.ones_like(mask, dtype=torch.long),
            )
            _, second = moe(
                hidden, route_features=features, active_mask=mask,
                curriculum=self.curriculum(), pass_index=1, pathway_cache=cache,
                forced_routed_k=torch.full_like(mask, 3, dtype=torch.long),
            )
        self.assertEqual(second.mean_k.tolist(), [[3, 3]])
        self.assertEqual(int(second.assignments), 6)

    def test_replay_uses_its_own_layout_not_the_last_forward_layout(self):
        cache = PathwayCache(freeze_widths=False)
        indices = torch.arange(18).reshape(1, 6, 3)
        widths = torch.ones(1, 6, dtype=torch.long)
        cache.set_layout(None, pass_index=0)
        cache.store(0, indices, widths)
        middle = _active_token_layout(torch.tensor([[True, False, True, False, True, True]]))
        last = _active_token_layout(torch.tensor([[False, False, True, False, False, True]]))
        cache.set_layout(middle, pass_index=1)
        cache.set_layout(last, pass_index=2)
        for pass_index, layout in ((1, middle), (2, last)):
            selected, chosen = cache.lookup(0, pass_index=pass_index)
            torch.testing.assert_close(selected, layout.pack(indices), rtol=0, atol=0)
            torch.testing.assert_close(chosen, layout.pack(widths), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "no layout"):
            cache.lookup(0, pass_index=3)
        cache.clear()
        self.assertIsNone(cache.lookup(0, pass_index=1))

    def test_packed_explicit_plan_keeps_frozen_rankings_and_exact_widths(self):
        torch.manual_seed(15)
        config = self.config()
        model = Metis16ForCausalLM(config).eval()
        inputs, labels = self.batch(config)
        depths = torch.tensor([[1, 2, 3, 2, 3, 0], [3, 1, 2, 3, 2, 1]])
        mask = depths > 0
        labels[~mask] = -100
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        widths[1, :, :, ::2] = 1
        widths[1, :, :, 1::2] = 3
        widths[2] = 3
        observed = []
        handles = []
        for layer_index, layer in enumerate(model.layers):
            layer.moe.capture_selection = True

            def capture(module, args, kwargs, result, layer_index=layer_index):
                observed.append((
                    layer_index, kwargs["pass_index"],
                    result[1].mean_k.detach().clone(),
                    module._analysis_last_selection[2].detach().clone(),
                ))

            handles.append(layer.moe.register_forward_hook(capture, with_kwargs=True))
        try:
            with torch.no_grad():
                output = model(
                    inputs, labels, attention_mask=mask,
                    curriculum=self.curriculum(),
                    force_depth=depths, force_routed_k=widths,
                )
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(output.chosen_depths, depths, rtol=0, atol=0)
        self.assertFalse(model._pathway_cache.freeze_widths)
        self.assertEqual(len(observed), config.max_passes * config.n_layers)
        for layer_index, pass_index, executed_widths, experts in observed:
            active = depths > pass_index
            positions = torch.nonzero(active.flatten(), as_tuple=False).flatten()
            expected_widths = widths[pass_index, layer_index].flatten()[positions]
            torch.testing.assert_close(executed_widths.flatten(), expected_widths.float(), rtol=0, atol=0)
            frozen = model._pathway_cache._indices[layer_index].reshape(-1, config.max_routed_k)[positions]
            selected = torch.arange(config.max_routed_k)[None] < expected_widths[:, None]
            torch.testing.assert_close(
                experts.reshape(-1, config.max_routed_k),
                frozen.masked_fill(~selected, -1), rtol=0, atol=0,
            )

    def test_frozen_teacher_observations_preserve_fixed_baseline_forward(self):
        torch.manual_seed(16)
        config = self.config(joint_compute_router=True, joint_router_hidden_dim=8)
        model = Metis16ForCausalLM(config).eval()
        inputs, labels = self.batch(config)
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        with torch.no_grad():
            baseline = model(
                inputs, labels, curriculum=self.curriculum(),
                force_depth=2, return_logits=True,
            )
            teacher = model(
                inputs, labels, curriculum=self.curriculum(), force_depth=2,
                force_routed_k=widths, return_router_observations=True,
                return_logits=True,
            )
        torch.testing.assert_close(teacher.logits, baseline.logits, rtol=0, atol=0)
        torch.testing.assert_close(teacher.loss, baseline.loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(teacher.chosen_depths, baseline.chosen_depths)
        self.assertEqual(len(teacher.router_observations[0].width_history), 1)
        self.assertEqual(int(teacher.telemetry["joint_utility_observations"]), int(labels.ne(-100).sum()))
        self.assertEqual(int(teacher.telemetry["joint_budget_enforced"]), 0)
        # Collecting teacher targets is extra work, not a free baseline run.
        self.assertGreater(
            int(teacher.telemetry["joint_model_flops"]),
            int(teacher.telemetry["joint_budget_flops"]),
        )

    def test_joint_frozen_execution_matches_its_width_ledger_and_total_cap(self):
        torch.manual_seed(17)
        config = self.config(
            max_passes=2, joint_compute_router=True, joint_router_hidden_dim=8
        )
        model = Metis16ForCausalLM(config).eval()
        inputs, labels = self.batch(config)
        with torch.no_grad():
            model.joint_router.output.bias[0] = 1
        observed = []
        handles = [
            layer.moe.register_forward_hook(
                lambda module, args, kwargs, result, layer_index=i: observed.append(
                    (layer_index, kwargs["pass_index"], result[1].mean_k.detach().clone())
                ),
                with_kwargs=True,
            )
            for i, layer in enumerate(model.layers)
        ]
        try:
            with torch.no_grad():
                output = model(
                    inputs, labels,
                    curriculum=self.curriculum(
                        compute_allocation_mode="joint", allow_untrained_joint_router=True
                    ),
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(bool(output.chosen_depths.eq(2).all()))
        costs = model.joint_router.costs
        actual = inputs.numel() * sum(costs.base_pass_costs[:2])
        for layer, pass_index, executed in observed:
            expected = 2 if pass_index == 0 else 1
            self.assertTrue(bool(executed.eq(expected).all()))
            actual += int(executed.sum()) * costs.expert_costs[layer]
        actual += int(output.telemetry["joint_router_flops"])
        self.assertEqual(int(output.telemetry["joint_model_flops"]), actual)
        self.assertLessEqual(actual, int(output.telemetry["joint_budget_flops"]))
        self.assertEqual(
            int(output.telemetry["joint_unused_budget_flops"]),
            int(output.telemetry["joint_budget_flops"]) - actual,
        )

    def test_frozen_packed_gradients_match_without_layer_or_pass_replay(self):
        torch.manual_seed(18)
        config = self.config()
        model = Metis16ForCausalLM(config).train()
        inputs, labels = self.batch(config)
        depths = torch.tensor([[1, 2, 3, 2, 3, 1], [3, 1, 2, 3, 2, 1]])
        widths = torch.full((config.max_passes, config.n_layers, *inputs.shape), 2)
        widths[1, :, :, ::2] = 1
        widths[2] = 3
        for explicit in (False, True):
            reference = None
            for replay in ("none", "layer", "pass"):
                with self.subTest(explicit=explicit, replay=replay):
                    model.set_activation_recompute_policy(replay)
                    model.zero_grad(set_to_none=True)
                    output = model(
                        inputs, labels, curriculum=self.curriculum(), force_depth=depths,
                        force_routed_k=widths if explicit else None,
                    )
                    output.loss.backward()
                    current = (
                        output.loss.detach(),
                        model.embedding.weight.grad.detach().clone(),
                        model.layers[0].moe.expert_router.weight.grad.detach().clone(),
                    )
                    if reference is None:
                        reference = current
                    else:
                        for actual, expected in zip(current, reference):
                            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
