from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from metis_training.contracts import (
    canonical_json_sha256,
    load_autotune_selection,
    load_family_manifest,
)
from metis_training.model_config import Metis16Config
from metis_training.precision import PrecisionPolicy, _dynamic_row_linear_type
from metis_training.precision_plan import (
    build_precision_role_inventory,
    build_precision_role_plan,
    exact_precision_role_specs,
    execution_role_dtype_map,
    measured_role_dtype_map,
    validate_precision_role_inventory,
    validate_precision_role_plan,
)


def _measurements(
    config: Metis16Config,
    *,
    slow_role: str | None = None,
    unsupported_role: str | None = None,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for spec in exact_precision_role_specs(config):
        fp8_ok = spec.role != unsupported_role
        result[spec.role] = {
            "bf16": {
                "ok": True,
                "finite_gradients": True,
                "median_seconds": 2.0,
            },
            "fp8": {
                "attempted": True,
                "ok": fp8_ok,
                "finite_gradients": fp8_ok,
                "median_seconds": 3.0 if spec.role == slow_role else 1.0,
                "loss_relative_error_vs_bf16": 0.01,
                "error": None if fp8_ok else "unsupported exact shape",
            },
        }
    return result


class PrecisionRolePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Metis16Config.tiny_for_tests()

    def test_inventory_is_deterministic_and_exact(self) -> None:
        first = build_precision_role_inventory(self.config)
        second = build_precision_role_inventory(self.config)
        self.assertEqual(first, second)
        self.assertEqual(validate_precision_role_inventory(first, config=self.config), first)
        roles = {row["role"] for row in first["roles"]}
        self.assertEqual(
            roles,
            {
                "mamba_in_projection",
                "mamba_out_projection",
                "attention_qkv_projection",
                "attention_out_projection",
                "attention_pass_lora_down",
                "attention_pass_lora_up",
                "expert_gate_up_projection",
                "expert_down_projection",
                "latent_down_projection",
                "latent_up_projection",
                "memory_state_write_projection",
                "memory_metadata_write_projection",
                "memory_query_projection",
                "memory_key_projection",
                "memory_value_projection",
                "memory_output_projection",
                "memory_route_projection",
                "ngram_projection",
                "mhc_controller",
                "lm_head",
            },
        )

    def test_dynamic_row_linear_pads_and_unpads_without_changing_gradients(self) -> None:
        linear_type = _dynamic_row_linear_type(torch.nn.Linear)
        for rows in (0, 1, 15, 17, 1_025):
            with self.subTest(rows=rows):
                padded = linear_type(16, 32, bias=True)
                reference = torch.nn.Linear(16, 32, bias=True)
                reference.load_state_dict(padded.state_dict())
                values = torch.randn(rows, 16, requires_grad=True)
                reference_values = values.detach().clone().requires_grad_(True)

                observed = padded(values)
                expected = reference(reference_values)
                torch.testing.assert_close(observed, expected)
                observed.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(values.grad, reference_values.grad)
                torch.testing.assert_close(padded.weight.grad, reference.weight.grad)
                torch.testing.assert_close(padded.bias.grad, reference.bias.grad)

    def test_dynamic_row_linear_accepts_every_rank_nn_linear_does(self) -> None:
        """The FP8 stand-in must accept the same input ranks as ``nn.Linear``.

        It rejected 1D outright, which ``nn.Linear`` accepts.  The mHC
        controller feeds it a pass embedding -- one vector per pass, not per
        token -- so every FP8 row died in the first forward while every BF16
        row ran, and no test noticed because the suite never builds the FP8
        path.
        """

        linear_type = _dynamic_row_linear_type(torch.nn.Linear)
        shapes = ((16,), (4, 16), (2, 3, 16), (0, 16))
        for shape in shapes:
            with self.subTest(shape=shape):
                padded = linear_type(16, 32, bias=True)
                reference = torch.nn.Linear(16, 32, bias=True)
                reference.load_state_dict(padded.state_dict())
                values = torch.randn(*shape, requires_grad=True)
                reference_values = values.detach().clone().requires_grad_(True)

                observed = padded(values)
                expected = reference(reference_values)
                self.assertEqual(observed.shape, expected.shape)
                torch.testing.assert_close(observed, expected)
                observed.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(values.grad, reference_values.grad)

    def test_delayed_scaling_recipe_supports_legacy_and_modern_te_signatures(
        self,
    ) -> None:
        policy = PrecisionPolicy(
            self.config.precision,
            requested_profile="bf16",
            device=torch.device("cpu"),
            production=False,
        )

        class Format:
            E4M3 = "e4m3"
            HYBRID = "hybrid"

        observed: list[dict[str, object]] = []

        class LegacyDelayedScaling:
            def __init__(
                self,
                *,
                margin,
                fp8_format,
                amax_history_len,
                amax_compute_algo,
                interval=1,
            ):
                observed.append(dict(locals()))

        class ModernDelayedScaling:
            def __init__(
                self,
                *,
                margin,
                fp8_format,
                amax_history_len,
                amax_compute_algo,
            ):
                observed.append(dict(locals()))

        for delayed in (LegacyDelayedScaling, ModernDelayedScaling):
            with self.subTest(delayed=delayed.__name__):
                policy._recipe_module = SimpleNamespace(
                    Format=Format,
                    DelayedScaling=delayed,
                )
                recipe = policy._build_fp8_recipe()
                self.assertIsInstance(recipe, delayed)
        self.assertEqual(len(observed), 2)

    def test_mixed_plan_keeps_small_slow_role_in_bf16(self) -> None:
        measurements = _measurements(
            self.config,
            slow_role="mhc_controller",
            unsupported_role="attention_pass_lora_down",
        )
        plan = build_precision_role_plan(
            self.config,
            measurements,
            maximum_relative_error=0.03,
        )
        role_map = measured_role_dtype_map(plan)
        self.assertEqual(role_map["mhc_controller"], "bf16")
        self.assertEqual(role_map["attention_pass_lora_down"], "bf16")
        self.assertEqual(role_map["expert_gate_up_projection"], "fp8")
        self.assertGreater(plan["classification"]["fp8_role_count"], 0)
        self.assertGreater(plan["classification"]["bf16_role_count"], 0)
        self.assertEqual(
            set(execution_role_dtype_map(plan, profile="bf16").values()),
            {"bf16"},
        )
        self.assertEqual(execution_role_dtype_map(plan, profile="fp8"), role_map)

    def test_missing_unknown_or_corrupt_role_plan_fails_closed(self) -> None:
        plan = build_precision_role_plan(
            self.config,
            _measurements(self.config),
            maximum_relative_error=0.03,
        )
        missing = copy.deepcopy(plan)
        missing["roles"].pop("lm_head")
        missing["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "corrupt|family-stale"):
            validate_precision_role_plan(missing, config=self.config)

        unknown = copy.deepcopy(plan)
        unknown["roles"]["made_up_projection"] = copy.deepcopy(
            unknown["roles"]["lm_head"]
        )
        from metis_training.precision_plan import _canonical_json_sha256

        unknown["plan_sha256"] = _canonical_json_sha256(
            {key: value for key, value in unknown.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "missing or contains unknown"):
            validate_precision_role_plan(unknown, config=self.config)

        corrupt = copy.deepcopy(plan)
        corrupt["roles"]["lm_head"]["selected_dtype"] = "bf16"
        corrupt["plan_sha256"] = _canonical_json_sha256(
            {key: value for key, value in corrupt.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "classification drifted"):
            validate_precision_role_plan(corrupt, config=self.config)

    def test_measurements_must_cover_every_exact_role(self) -> None:
        rows = _measurements(self.config)
        rows.pop("lm_head")
        with self.assertRaisesRegex(RuntimeError, "missing=.*lm_head"):
            build_precision_role_plan(
                self.config,
                rows,
                maximum_relative_error=0.03,
            )

    def test_bf16_oracle_must_be_finite_for_every_role(self) -> None:
        rows = _measurements(self.config)
        rows["mhc_controller"]["bf16"]["finite_gradients"] = False
        with self.assertRaisesRegex(RuntimeError, "safe BF16 oracle"):
            build_precision_role_plan(
                self.config,
                rows,
                maximum_relative_error=0.03,
            )

    def test_policy_dispatch_is_exact_and_bf16_oracle_retains_measured_map(
        self,
    ) -> None:
        plan = build_precision_role_plan(
            self.config,
            _measurements(
                self.config,
                slow_role="mhc_controller",
            ),
            maximum_relative_error=0.03,
        )
        role_map = measured_role_dtype_map(plan)
        policy = PrecisionPolicy(
            self.config.precision,
            requested_profile="bf16",
            device=torch.device("cpu"),
            production=False,
            measured_role_dtypes=role_map,
            precision_role_plan_sha256=plan["plan_sha256"],
        )
        self.assertEqual(policy.audit.measured_role_dtypes, role_map)
        self.assertEqual(
            set(policy.audit.execution_role_dtypes.values()),
            {"bf16"},
        )
        with self.assertRaisesRegex(RuntimeError, "absent"):
            policy.is_fp8_role("expert_projection")

        class MarkerLinear(torch.nn.Linear):
            pass

        policy.effective_profile = "fp8"
        policy._te = SimpleNamespace(Linear=MarkerLinear)
        fp8_role = next(
            role for role, dtype in role_map.items() if dtype == "fp8"
        )
        bf16_role = next(
            role for role, dtype in role_map.items() if dtype == "bf16"
        )
        self.assertIsInstance(
            policy.linear(4, 3, role=fp8_role),
            MarkerLinear,
        )
        self.assertIsInstance(
            policy.linear(4, 3, role=bf16_role),
            torch.nn.Linear,
        )
        self.assertNotIsInstance(
            policy.linear(4, 3, role=bf16_role),
            MarkerLinear,
        )

    def test_production_policy_rejects_missing_role_plan_and_hash(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sealed exact-role"):
            PrecisionPolicy(
                self.config.precision,
                requested_profile="bf16",
                device=torch.device("cpu"),
                production=True,
            )
        with self.assertRaisesRegex(RuntimeError, "hash"):
            PrecisionPolicy(
                self.config.precision,
                requested_profile="bf16",
                device=torch.device("cpu"),
                production=False,
                measured_role_dtypes={
                    spec.role: "bf16"
                    for spec in exact_precision_role_specs(self.config)
                },
                precision_role_plan_sha256="broken",
            )

    def test_autotune_profile_binds_full_plan_hash_and_map(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "metis16"
            / "praxis.yaml"
        )
        config = Metis16Config.from_yaml(manifest_path)
        plan = build_precision_role_plan(
            config,
            _measurements(config, slow_role="mhc_controller"),
            maximum_relative_error=0.03,
        )
        role_map = measured_role_dtype_map(plan)
        profile = {
            "schema": "metis.portage-autotune/v1",
            "family": "praxis",
            "environment_sha256": "a" * 64,
            "release_marker_sha256": "b" * 64,
            "precision_role_plan": plan,
            "precision_role_plan_sha256": plan["plan_sha256"],
            "precision_role_inventory_sha256": plan["inventory_sha256"],
            "measured_precision_role_map": role_map,
            "selected": {
                "micro_batch_size": 4,
                "grad_accum_steps": 1,
                "learning_rate": 0.00018,
                "precision_profile": "fp8",
                "compile_mode": "default",
                "dispatch_overlap": True,
                "ngram_table_mode": "replicated",
            },
        }
        profile["profile_sha256"] = canonical_json_sha256(profile)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "profile.json"
            path.write_text(
                json.dumps(profile, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            selection = load_autotune_selection(
                path,
                family_manifest=load_family_manifest(manifest_path),
                expected_environment_sha256="a" * 64,
            )
            self.assertEqual(
                selection.precision_role_plan_sha256,
                plan["plan_sha256"],
            )
            self.assertEqual(selection.measured_role_dtypes, role_map)

            corrupt = copy.deepcopy(profile)
            corrupt["measured_precision_role_map"]["mhc_controller"] = "fp8"
            corrupt["profile_sha256"] = canonical_json_sha256(
                {
                    key: value
                    for key, value in corrupt.items()
                    if key != "profile_sha256"
                }
            )
            path.write_text(
                json.dumps(corrupt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "binding is stale"):
                load_autotune_selection(
                    path,
                    family_manifest=load_family_manifest(manifest_path),
                )


if __name__ == "__main__":
    unittest.main()
