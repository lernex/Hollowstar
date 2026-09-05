from __future__ import annotations

import copy
import unittest

import torch

from metis_mamba.optim import MuonAdamWHybrid
from metis_training.optimizers import FP32MasterSparseAdam, OptimizerBundle


class OptimizerRestoreTests(unittest.TestCase):
    def sparse_gradient(self, parameter, scale=1.0):
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 2]]),
            torch.tensor([[0.17, 0.23], [0.31, -0.07]], dtype=parameter.dtype) * scale,
            parameter.shape,
            is_coalesced=True,
        )

    def test_sparse_adam_restores_original_fp32_values_not_rounded_upcasts(self):
        parameter = torch.nn.Parameter(torch.full((3, 2), 0.125, dtype=torch.bfloat16))
        source = FP32MasterSparseAdam([parameter], lr=0.001)
        self.sparse_gradient(parameter)
        source.step()
        saved = copy.deepcopy(source.state_dict())
        target_parameter = torch.nn.Parameter(parameter.detach().clone())
        target = FP32MasterSparseAdam([target_parameter], lr=0.001)
        target.load_state_dict(saved)
        for name in ("master_param", "exp_avg", "exp_avg_sq"):
            expected = saved["state"][0][name]
            actual = target.state[target_parameter][name]
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())
        self.sparse_gradient(parameter, 1.37)
        self.sparse_gradient(target_parameter, 1.37)
        source.step()
        target.step()
        torch.testing.assert_close(parameter, target_parameter, rtol=0, atol=0)
        for name in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(source.state[parameter][name], target.state[target_parameter][name], rtol=0, atol=0)

    def make_bundle(self, parameters):
        matrix, vector, sparse = parameters
        dense = MuonAdamWHybrid(
            [{"params": [matrix], "optimizer": "muon"},
             {"params": [vector], "optimizer": "adamw"}],
            lr=0.001, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
            muon_beta=0.95, muon_ns_steps=3, muon_nesterov=True,
            master_weights=True, muon_state_bits=8, adamw_impl="loop",
        )
        return OptimizerBundle(dense, FP32MasterSparseAdam([sparse], lr=0.001))

    def gradients(self, parameters, step):
        matrix, vector, sparse = parameters
        matrix.grad = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) * 0.013 + 0.017 * step
        ).to(matrix.dtype)
        vector.grad = torch.tensor([0.17, -0.23, 0.031, 0.127], dtype=vector.dtype) * step
        self.sparse_gradient(sparse, float(step))

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_bundle_preserves_muon_int8_scales_and_adam_master_moments(self):
        parameters = [
            torch.nn.Parameter(torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) / 32),
            torch.nn.Parameter(torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.full((3, 2), 0.125, dtype=torch.bfloat16)),
        ]
        source = self.make_bundle(parameters)
        self.gradients(parameters, 1)
        source.step()
        saved = copy.deepcopy(source.state_dict())
        restored_parameters = [torch.nn.Parameter(value.detach().clone()) for value in parameters]
        target = self.make_bundle(restored_parameters)
        target.load_state_dict(saved)
        self.assert_nested_equal(saved, target.state_dict())
        for step in (2, 3):
            self.gradients(parameters, step)
            self.gradients(restored_parameters, step)
            source.step()
            target.step()
            for first, second in zip(parameters, restored_parameters):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assert_nested_equal(source.state_dict(), target.state_dict())

    def test_parameter_group_validation_is_not_bypassed(self):
        parameter = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.bfloat16))
        optimizer = FP32MasterSparseAdam([parameter], lr=0.001)
        with self.assertRaises(ValueError):
            optimizer.load_state_dict({"state": {}, "param_groups": []})

    def test_previously_downcast_master_is_rejected_instead_of_inventing_precision(self):
        parameter = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.bfloat16))
        source = FP32MasterSparseAdam([parameter], lr=0.001)
        self.sparse_gradient(parameter)
        source.step()
        saved = copy.deepcopy(source.state_dict())
        saved["state"][0]["master_param"] = saved["state"][0]["master_param"].bfloat16()
        target = FP32MasterSparseAdam([torch.nn.Parameter(parameter.detach().clone())], lr=0.001)
        with self.assertRaisesRegex(RuntimeError, "discarded precision"):
            target.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
