"""A parity probe must not turn trainable grouped experts into constant weights."""

from dataclasses import replace
import unittest

import torch

from metis_training.model import CurriculumState, Metis16ForCausalLM, _StackedGroupedLinear
from metis_training.model_config import Metis16Config


class GroupedCacheLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def layer(self):
        return _StackedGroupedLinear(
            4, 2, 2, weight_chunks=4, device="cpu", dtype=torch.bfloat16
        )

    def model(self):
        return Metis16ForCausalLM(replace(
            Metis16Config.tiny_for_tests(), expert_execution="grouped_gemm",
            expert_weight_chunks=4,
        )).apply_parameter_storage_policy("cpu").train()

    def inputs(self):
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                               [11, 12, 13, 14, 15, 16, 17, 18]])
        labels = inputs.roll(-1, dims=1)
        labels[:, -1] = -100
        return inputs, labels

    def curriculum(self):
        return CurriculumState(
            continuation_mode="fixed_max", routed_k_mode="fixed",
            fixed_routed_k=2, max_passes=2, stochastic_routing=False,
        )

    def test_no_grad_reference_cannot_suppress_full_model_expert_gradients(self):
        torch.manual_seed(502)
        clean, probed = self.model(), self.model()
        probed.load_state_dict(clean.state_dict())
        inputs, labels = self.inputs()
        with torch.no_grad():
            probed(inputs, labels, curriculum=self.curriculum())
        for layer in probed.layers:
            self.assertIsNone(layer.moe.local_experts.gate_up._materialized_weight)
            self.assertIsNone(layer.moe.local_experts.down._materialized_weight)
        left = clean(inputs, labels, curriculum=self.curriculum())
        right = probed(inputs, labels, curriculum=self.curriculum())
        torch.testing.assert_close(right.loss, left.loss, rtol=0, atol=0)
        left.loss.backward()
        right.loss.backward()
        expert_gradient = 0.0
        reference = dict(clean.named_parameters())
        for name, value in probed.named_parameters():
            if ".local_experts." not in name:
                continue
            self.assertIsNotNone(value.grad, name)
            torch.testing.assert_close(value.grad, reference[name].grad, rtol=0, atol=0)
            expert_gradient += float(value.grad.float().abs().sum())
        self.assertGreater(expert_gradient, 0)

    def test_optimizer_step_changes_experts_after_a_reference_probe(self):
        layer = self.layer()
        values = torch.ones(4, 2, dtype=torch.bfloat16)
        splits = torch.ones(4, dtype=torch.long)
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.fill_(.5)
        optimizer = torch.optim.SGD(layer.parameters(), lr=.25)
        with torch.no_grad():
            reference = layer(values, splits).clone()
        layer(values, splits).float().sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        optimizer.step()
        with torch.no_grad():
            updated = layer(values, splits)
        torch.testing.assert_close(updated.float(), reference.float() - .5, rtol=0, atol=0)

    def test_frozen_materializations_do_not_survive_unfreezing(self):
        layer = self.layer().requires_grad_(False)
        self.assertFalse(layer._forward_weight().requires_grad)
        self.assertIsNone(layer._materialized_weight)
        for parameter in layer.parameters():
            parameter.data.copy_(torch.full_like(parameter, 3))
        layer.requires_grad_(True)
        materialized = layer._forward_weight()
        self.assertTrue(materialized.requires_grad)
        self.assertTrue(bool(materialized.eq(3).all()))
        materialized.float().sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        self.assertIsNone(layer._materialized_weight)

    def test_load_reset_and_dtype_transforms_invalidate_cached_concatenation(self):
        layer = self.layer()
        original = layer._forward_weight()
        self.assertIs(layer._forward_weight(), original)
        state = {name: torch.full_like(value, 5) for name, value in layer.state_dict().items()}
        layer.load_state_dict(state)
        self.assertIsNone(layer._materialized_weight)
        self.assertTrue(bool(layer._forward_weight().eq(5).all()))
        layer.to(dtype=torch.float32)
        self.assertIsNone(layer._materialized_weight)
        self.assertEqual(layer._forward_weight().dtype, torch.float32)
        layer.reset_expert_parameters(0)
        self.assertIsNone(layer._materialized_weight)
        self.assertFalse(bool(layer._forward_weight()[0].eq(5).all()))

    def test_separate_model_forwards_do_not_share_a_backward_graph(self):
        torch.manual_seed(503)
        model = self.model()
        inputs, labels = self.inputs()
        first = model(inputs, labels, curriculum=self.curriculum())
        first_cache = model.layers[0].moe.local_experts.gate_up._materialized_weight
        second = model(inputs, labels, curriculum=self.curriculum())
        second_cache = model.layers[0].moe.local_experts.gate_up._materialized_weight
        self.assertIsNot(first_cache, second_cache)
        first.loss.backward()
        second.loss.backward()
        self.assertTrue(all(
            parameter.grad is not None
            for name, parameter in model.named_parameters() if ".local_experts." in name
        ))

    def test_model_forward_observes_unversioned_parameter_data_updates(self):
        torch.manual_seed(504)
        model = self.model()
        inputs, labels = self.inputs()
        model(inputs, labels, curriculum=self.curriculum())
        projection = model.layers[0].moe.local_experts.gate_up
        old = projection._materialized_weight
        versions = [parameter._version for parameter in projection.weight_chunks]
        for parameter in projection.weight_chunks:
            parameter.data.copy_(torch.full_like(parameter, .25))
        self.assertEqual(versions, [parameter._version for parameter in projection.weight_chunks])
        model(inputs, labels, curriculum=self.curriculum())
        self.assertIsNot(projection._materialized_weight, old)
        self.assertTrue(bool(projection._materialized_weight.eq(.25).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_grouped_probe_preserves_weight_gradients(self):
        layer = _StackedGroupedLinear(
            8, 64, 64, weight_chunks=4, device="cuda", dtype=torch.bfloat16
        )
        values = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        splits = torch.full((8,), 16, device="cuda", dtype=torch.long)
        with torch.no_grad():
            layer(values, splits)
        self.assertIsNone(layer._materialized_weight)
        layer(values, splits).float().square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        self.assertTrue(all(bool(torch.isfinite(parameter.grad).all()) for parameter in layer.parameters()))
        self.assertGreater(sum(float(parameter.grad.float().abs().sum()) for parameter in layer.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
