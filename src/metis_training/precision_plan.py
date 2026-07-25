from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_config import Metis16Config


ROLE_INVENTORY_SCHEMA = "metis.precision-role-inventory/v1"
ROLE_PLAN_SCHEMA = "metis.precision-role-plan/v1"
ROLE_DTYPES = frozenset({"fp8", "bf16"})


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class PrecisionRoleSpec:
    """One semantically stable linear surface with an exact weight shape."""

    role: str
    in_features: int
    out_features: int
    bias: bool
    occurrences: int
    heavy: bool

    @property
    def parameter_count_per_occurrence(self) -> int:
        return self.in_features * self.out_features + (
            self.out_features if self.bias else 0
        )

    @property
    def logical_parameter_count(self) -> int:
        return self.parameter_count_per_occurrence * self.occurrences

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameter_count_per_occurrence": self.parameter_count_per_occurrence,
            "logical_parameter_count": self.logical_parameter_count,
        }


def exact_precision_role_specs(config: Metis16Config) -> tuple[PrecisionRoleSpec, ...]:
    """Return the complete exact-shape FP8-eligible inventory for a family.

    Roles are intentionally semantic rather than shape-deduplicated. For
    example, attention output and latent projections can share a dimension in
    a small test model but retain separate entries so a measured plan cannot be
    silently reused after either surface changes.
    """

    d_inner = config.d_model * config.mamba_expand
    mamba_heads = d_inner // config.mamba_head_dim
    mamba_in_width = (
        2 * d_inner
        + 2 * config.mamba_ngroups * config.mamba_d_state
        + mamba_heads
    )
    qkv_width = (
        config.n_heads + 2 * config.n_kv_heads
    ) * config.head_dim
    controller_width = config.n_streams * config.n_streams + 2 * config.n_streams
    logical_experts_per_layer = config.n_routed_experts + config.n_shared_experts

    rows = (
        PrecisionRoleSpec(
            "mamba_in_projection",
            config.d_model,
            mamba_in_width,
            False,
            config.n_mamba_layers,
            True,
        ),
        PrecisionRoleSpec(
            "mamba_out_projection",
            d_inner,
            config.d_model,
            False,
            config.n_mamba_layers,
            True,
        ),
        PrecisionRoleSpec(
            "attention_qkv_projection",
            config.d_model,
            qkv_width,
            False,
            config.n_attention_layers,
            True,
        ),
        PrecisionRoleSpec(
            "attention_out_projection",
            config.d_model,
            config.d_model,
            False,
            config.n_attention_layers,
            True,
        ),
        PrecisionRoleSpec(
            "attention_pass_lora_down",
            config.d_model,
            config.attention_pass_lora_rank,
            False,
            config.n_attention_layers,
            False,
        ),
        PrecisionRoleSpec(
            "attention_pass_lora_up",
            config.attention_pass_lora_rank,
            qkv_width,
            False,
            config.n_attention_layers,
            False,
        ),
        PrecisionRoleSpec(
            "expert_gate_up_projection",
            config.latent_dim,
            2 * config.expert_intermediate_dim,
            False,
            config.n_layers * logical_experts_per_layer,
            True,
        ),
        PrecisionRoleSpec(
            "expert_down_projection",
            config.expert_intermediate_dim,
            config.latent_dim,
            False,
            config.n_layers * logical_experts_per_layer,
            True,
        ),
        PrecisionRoleSpec(
            "latent_down_projection",
            config.d_model,
            config.latent_dim,
            False,
            config.n_layers,
            True,
        ),
        PrecisionRoleSpec(
            "latent_up_projection",
            config.latent_dim,
            config.d_model,
            False,
            config.n_layers,
            True,
        ),
        PrecisionRoleSpec(
            "memory_state_write_projection",
            config.d_model,
            config.memory_dim,
            True,
            1,
            True,
        ),
        PrecisionRoleSpec(
            "memory_metadata_write_projection",
            config.route_feature_dim + 4,
            config.memory_dim,
            True,
            1,
            False,
        ),
        PrecisionRoleSpec(
            "memory_query_projection",
            config.d_model,
            config.memory_dim,
            True,
            1,
            True,
        ),
        PrecisionRoleSpec(
            "memory_key_projection",
            config.memory_dim,
            config.memory_dim,
            True,
            1,
            False,
        ),
        PrecisionRoleSpec(
            "memory_value_projection",
            config.memory_dim,
            config.memory_dim,
            True,
            1,
            False,
        ),
        PrecisionRoleSpec(
            "memory_output_projection",
            config.memory_dim,
            config.d_model,
            True,
            1,
            True,
        ),
        PrecisionRoleSpec(
            "memory_route_projection",
            3 * config.d_model + config.route_feature_dim,
            config.route_feature_dim,
            True,
            1,
            True,
        ),
        PrecisionRoleSpec(
            "ngram_projection",
            config.ngram_memory.concatenated_dim,
            config.d_model,
            True,
            1,
            True,
        ),
        PrecisionRoleSpec(
            "mhc_controller",
            config.mhc_pass_embedding_dim,
            controller_width,
            True,
            2 * config.n_layers,
            False,
        ),
        PrecisionRoleSpec(
            "lm_head",
            config.d_model,
            config.vocab_size,
            False,
            1,
            True,
        ),
    )
    if any(
        row.in_features <= 0
        or row.out_features <= 0
        or row.occurrences <= 0
        for row in rows
    ):
        raise RuntimeError("Precision role inventory contains a non-positive shape/count")
    names = [row.role for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Precision role inventory contains duplicate role names")
    return tuple(sorted(rows, key=lambda row: row.role))


def build_precision_role_inventory(config: Metis16Config) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ROLE_INVENTORY_SCHEMA,
        "family": config.family,
        "model_name": config.name,
        "roles": [row.to_dict() for row in exact_precision_role_specs(config)],
    }
    payload["inventory_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_precision_role_inventory(
    payload: Mapping[str, Any],
    *,
    config: Metis16Config,
) -> dict[str, Any]:
    cooked = dict(payload)
    unsigned = {
        key: value for key, value in cooked.items() if key != "inventory_sha256"
    }
    if (
        cooked.get("schema") != ROLE_INVENTORY_SCHEMA
        or cooked.get("family") != config.family
        or cooked.get("inventory_sha256") != _canonical_json_sha256(unsigned)
    ):
        raise RuntimeError("Precision role inventory is corrupt or family-stale")
    expected = build_precision_role_inventory(config)
    if cooked != expected:
        raise RuntimeError("Precision role inventory does not match the exact model geometry")
    return cooked


def representative_probe_rows(
    spec: PrecisionRoleSpec,
    *,
    maximum_rows: int = 2_048,
    maximum_activation_elements: int = 16_777_216,
    row_multiple: int = 64,
) -> int:
    """Choose a bounded, deterministic batch dimension for a role probe."""

    if maximum_rows <= 0 or maximum_activation_elements <= 0 or row_multiple <= 0:
        raise ValueError("Probe row bounds must be positive")
    element_width = max(1, spec.in_features + spec.out_features)
    rows = min(maximum_rows, maximum_activation_elements // element_width)
    rows = max(row_multiple, rows)
    rows = max(row_multiple, (rows // row_multiple) * row_multiple)
    return int(rows)


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    cooked = float(value)
    if not math.isfinite(cooked) or (positive and cooked <= 0.0):
        raise RuntimeError(f"{label} must be {'positive ' if positive else ''}finite")
    return cooked


def _validate_measurement(
    role: str,
    row: Mapping[str, Any],
    *,
    maximum_relative_error: float,
    minimum_fp8_speedup: float,
) -> tuple[str, str]:
    bf16 = row.get("bf16")
    fp8 = row.get("fp8")
    if not isinstance(bf16, Mapping) or not isinstance(fp8, Mapping):
        raise RuntimeError(f"Precision role {role} lacks BF16 or FP8 measurements")
    if bf16.get("ok") is not True or bf16.get("finite_gradients") is not True:
        raise RuntimeError(f"Precision role {role} has no safe BF16 oracle")
    bf16_seconds = _finite_number(
        bf16.get("median_seconds"),
        label=f"{role} BF16 median_seconds",
        positive=True,
    )
    if fp8.get("attempted") is not True:
        raise RuntimeError(f"Precision role {role} was not attempted in FP8")
    if fp8.get("ok") is not True:
        error = str(fp8.get("error", "FP8 execution unsupported"))
        return "bf16", f"fp8_unavailable: {error}"
    if fp8.get("finite_gradients") is not True:
        return "bf16", "fp8_nonfinite_gradient"
    fp8_seconds = _finite_number(
        fp8.get("median_seconds"),
        label=f"{role} FP8 median_seconds",
        positive=True,
    )
    relative_error = _finite_number(
        fp8.get("loss_relative_error_vs_bf16"),
        label=f"{role} FP8 loss_relative_error_vs_bf16",
    )
    if relative_error < 0.0 or relative_error > maximum_relative_error:
        return "bf16", "fp8_loss_parity_failed"
    speedup = bf16_seconds / fp8_seconds
    if speedup <= minimum_fp8_speedup:
        return "bf16", "fp8_not_throughput_positive"
    return "fp8", "fp8_safe_and_throughput_positive"


def build_precision_role_plan(
    config: Metis16Config,
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    maximum_relative_error: float,
    minimum_fp8_speedup: float = 1.0,
) -> dict[str, Any]:
    """Classify every exact role independently and seal the mixed plan."""

    if not 0.0 <= maximum_relative_error < 1.0:
        raise ValueError("maximum_relative_error must be in [0, 1)")
    if minimum_fp8_speedup < 1.0 or not math.isfinite(minimum_fp8_speedup):
        raise ValueError("minimum_fp8_speedup must be finite and at least 1.0")
    inventory = build_precision_role_inventory(config)
    specs = exact_precision_role_specs(config)
    expected = {row.role for row in specs}
    observed = {str(role) for role in measurements}
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise RuntimeError(
            f"Precision measurements do not cover the exact role inventory; "
            f"missing={missing}, unknown={unknown}"
        )
    role_rows: dict[str, Any] = {}
    for spec in specs:
        measurement = dict(measurements[spec.role])
        selected_dtype, reason = _validate_measurement(
            spec.role,
            measurement,
            maximum_relative_error=maximum_relative_error,
            minimum_fp8_speedup=minimum_fp8_speedup,
        )
        role_rows[spec.role] = {
            "selected_dtype": selected_dtype,
            "reason": reason,
            "shape": {
                "in_features": spec.in_features,
                "out_features": spec.out_features,
                "bias": spec.bias,
            },
            "occurrences": spec.occurrences,
            "heavy": spec.heavy,
            "probe_rows": representative_probe_rows(spec),
            "bf16": dict(measurement["bf16"]),
            "fp8": dict(measurement["fp8"]),
        }
    payload: dict[str, Any] = {
        "schema": ROLE_PLAN_SCHEMA,
        "family": config.family,
        "inventory_sha256": inventory["inventory_sha256"],
        "maximum_relative_error": float(maximum_relative_error),
        "minimum_fp8_speedup": float(minimum_fp8_speedup),
        "roles": role_rows,
        "classification": {
            "role_count": len(role_rows),
            "heavy_role_count": sum(row.heavy for row in specs),
            "fp8_role_count": sum(
                row["selected_dtype"] == "fp8" for row in role_rows.values()
            ),
            "bf16_role_count": sum(
                row["selected_dtype"] == "bf16" for row in role_rows.values()
            ),
            "all_roles_probed": True,
            "all_heavy_roles_safely_classified": True,
        },
    }
    payload["plan_sha256"] = _canonical_json_sha256(payload)
    return validate_precision_role_plan(payload, config=config)


def validate_precision_role_plan(
    payload: Mapping[str, Any],
    *,
    config: Metis16Config,
) -> dict[str, Any]:
    cooked = dict(payload)
    unsigned = {key: value for key, value in cooked.items() if key != "plan_sha256"}
    if (
        cooked.get("schema") != ROLE_PLAN_SCHEMA
        or cooked.get("family") != config.family
        or cooked.get("plan_sha256") != _canonical_json_sha256(unsigned)
    ):
        raise RuntimeError("Precision role plan is corrupt or family-stale")
    inventory = build_precision_role_inventory(config)
    if cooked.get("inventory_sha256") != inventory["inventory_sha256"]:
        raise RuntimeError("Precision role plan was measured for stale model geometry")
    maximum_error = _finite_number(
        cooked.get("maximum_relative_error"),
        label="maximum_relative_error",
    )
    minimum_speedup = _finite_number(
        cooked.get("minimum_fp8_speedup"),
        label="minimum_fp8_speedup",
        positive=True,
    )
    if not 0.0 <= maximum_error < 1.0 or minimum_speedup < 1.0:
        raise RuntimeError("Precision role plan thresholds are invalid")
    roles = cooked.get("roles")
    if not isinstance(roles, Mapping):
        raise RuntimeError("Precision role plan has no role map")
    specs = {row.role: row for row in exact_precision_role_specs(config)}
    if set(roles) != set(specs):
        raise RuntimeError("Precision role plan is missing or contains unknown roles")
    fp8_count = 0
    bf16_count = 0
    for role, spec in specs.items():
        row = roles.get(role)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Precision role plan row {role} is invalid")
        shape = row.get("shape")
        expected_shape = {
            "in_features": spec.in_features,
            "out_features": spec.out_features,
            "bias": spec.bias,
        }
        if (
            shape != expected_shape
            or int(row.get("occurrences", -1)) != spec.occurrences
            or row.get("heavy") is not spec.heavy
            or int(row.get("probe_rows", -1)) != representative_probe_rows(spec)
        ):
            raise RuntimeError(f"Precision role plan shape/count drifted for {role}")
        measurement = {"bf16": row.get("bf16"), "fp8": row.get("fp8")}
        expected_dtype, expected_reason = _validate_measurement(
            role,
            measurement,
            maximum_relative_error=maximum_error,
            minimum_fp8_speedup=minimum_speedup,
        )
        if (
            row.get("selected_dtype") not in ROLE_DTYPES
            or row.get("selected_dtype") != expected_dtype
            or row.get("reason") != expected_reason
        ):
            raise RuntimeError(f"Precision role plan classification drifted for {role}")
        fp8_count += expected_dtype == "fp8"
        bf16_count += expected_dtype == "bf16"
    classification = cooked.get("classification")
    expected_classification = {
        "role_count": len(specs),
        "heavy_role_count": sum(row.heavy for row in specs.values()),
        "fp8_role_count": fp8_count,
        "bf16_role_count": bf16_count,
        "all_roles_probed": True,
        "all_heavy_roles_safely_classified": True,
    }
    if classification != expected_classification:
        raise RuntimeError("Precision role plan classification summary is stale")
    return cooked


def load_precision_role_plan(
    path: str | Path,
    *,
    config: Metis16Config,
) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("Precision role plan must be a JSON object")
    return validate_precision_role_plan(payload, config=config)


def measured_role_dtype_map(plan: Mapping[str, Any]) -> dict[str, str]:
    roles = plan.get("roles")
    if not isinstance(roles, Mapping):
        raise RuntimeError("Precision role plan has no role map")
    result = {
        str(role): str(row.get("selected_dtype", ""))
        for role, row in roles.items()
        if isinstance(row, Mapping)
    }
    if len(result) != len(roles) or any(dtype not in ROLE_DTYPES for dtype in result.values()):
        raise RuntimeError("Precision role plan contains an invalid dtype map")
    return dict(sorted(result.items()))


def execution_role_dtype_map(
    plan: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, str]:
    measured = measured_role_dtype_map(plan)
    if profile == "bf16":
        return {role: "bf16" for role in measured}
    if profile == "fp8":
        return measured
    raise ValueError("precision profile must be fp8 or bf16")
