from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .model_config import Metis16Config


MI300A_DENSE_PEAK_FLOPS = {
    "fp8": 1.96116e15,
    "bf16": 9.8058e14,
}


@dataclass(frozen=True)
class StepMetrics:
    optimizer_step: int
    global_token_cursor: int
    phase: str
    loss: float
    learning_rate: float
    global_non_padding_tokens: int
    global_supervised_tokens: int
    step_time_s: float
    tokens_per_second: float
    estimated_train_flops: float
    estimated_mfu: float
    grad_norm: float
    update_to_weight_ratio: float | None
    peak_hbm_bytes: int
    precision_profile: str
    overflow_drop_tokens: int
    collective_errors: int
    telemetry: dict[str, Any]
    peak_hbm_allocated_bytes: int | None = None
    peak_hbm_reserved_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["peak_hbm_allocated_bytes"] = int(
            self.peak_hbm_allocated_bytes
            if self.peak_hbm_allocated_bytes is not None
            else self.peak_hbm_bytes
        )
        payload["peak_hbm_reserved_bytes"] = int(
            self.peak_hbm_reserved_bytes
            if self.peak_hbm_reserved_bytes is not None
            else self.peak_hbm_bytes
        )
        payload["mfu"] = self.estimated_mfu
        payload["all_to_all_bytes"] = int(
            self.telemetry.get("all_to_all_bytes", 0)
        )
        payload["all_to_all_seconds"] = float(
            self.telemetry.get("all_to_all_seconds", 0.0)
        )
        payload["expert_load_cv"] = float(
            self.telemetry.get("expert_load_cv", 0.0)
        )
        return payload


def estimate_train_flops(
    config: Metis16Config,
    *,
    tokens: int,
    observed_mean_passes: float | None = None,
    observed_mean_routed_k: float | None = None,
) -> float:
    """Return an auditable model-FLOP estimate, not hardware counters."""

    return sum(estimate_train_flop_terms(
        config,
        tokens=tokens,
        observed_mean_passes=observed_mean_passes,
        observed_mean_routed_k=observed_mean_routed_k,
    ).values())


def estimate_train_flop_terms(
    config: Metis16Config,
    *,
    tokens: int,
    observed_mean_passes: float | None = None,
    observed_mean_routed_k: float | None = None,
) -> dict[str, float]:
    """Break the model-FLOP estimate into independently auditable terms.

    The six-times-active-parameters convention accounts for forward and
    backward GEMMs.  ``logical_parameter_audit`` counts every parameter once,
    so three classes of executed work need explicit terms on top of it:

    * modules invoked more often than they are stored -- the shared depth
      memory runs once per layer per pass, and its query/output projections
      run once per mHC stream on top of that;
    * the tied ``lm_head``, whose weights the audit attributes to
      ``embedding`` and therefore excludes from ``active_per_pass``, and which
      executes exactly once per token because ACT gives each token one exit;
    * work with no parameters at all -- attention's quadratic score/value
      GEMMs and the Mamba-2 chunked-scan einsums.

    MoRE depth and routed-k observations scale the locked active-per-pass
    audit.  Every term is model FLOPs; see :func:`estimate_hardware_flops` for
    the additional forward replay that pass-level recompute actually executes.
    """

    audit = config.logical_parameter_audit()
    passes = float(observed_mean_passes or config.target_mean_passes)
    routed_k = float(observed_mean_routed_k or config.target_mean_routed_k)
    baseline_k = max(float(config.target_mean_routed_k), 1.0)
    if config.ffn_mode == "dense":
        # A dense feed-forward stack has no routed share to rescale: every
        # feed-forward weight is active on every token, so ``active_per_pass``
        # is already exact and the k-dependent terms vanish.
        active_per_pass = float(audit.active_per_pass_mean)
        routed_at_baseline = 0.0
        routed_active = 0.0
    else:
        # ``audit.routed_experts`` is the logical total for every routed expert,
        # whereas one pass activates only k / N of that total.  Scaling by the
        # logical share of the whole model would incorrectly scale embeddings,
        # mixers, memory, and controllers together with k.
        routed_at_baseline = (
            float(audit.routed_experts)
            * baseline_k
            / float(config.n_routed_experts)
        )
        non_routed_active = max(
            0.0,
            float(audit.active_per_pass_mean) - routed_at_baseline,
        )
        routed_active = (
            float(audit.routed_experts)
            * routed_k
            / float(config.n_routed_experts)
        )
        active_per_pass = non_routed_active + routed_active

    d = config.d_model
    layers = config.n_layers

    # RecurrentDepthMemory is one stored module driven ``n_layers`` times per
    # pass.  ``retrieve`` projects every mHC stream, ``routing_features``
    # projects the pooled state once, and key/value run over the live bank.
    stream_projections = 2 * (d * config.memory_dim + config.memory_dim) + d
    route_projection = (
        (3 * d + config.route_feature_dim) * config.route_feature_dim
        + config.route_feature_dim
    )
    bank_projections = 2 * (config.memory_dim * config.memory_dim + config.memory_dim)
    # The bank gains one anchor per attention layer per pass plus a
    # continuation anchor, so the mean live depth over a pass tracks the
    # observed depth rather than the ``max_passes`` ceiling.
    mean_live_bank_entries = 0.5 * (config.n_attention_layers + 1) * passes
    repeated_memory_parameters = max(
        0.0,
        layers * config.n_streams * stream_projections
        + layers * route_projection
        + layers * mean_live_bank_entries * bank_projections
        # the audit already charged one execution of each of the above
        - (stream_projections + route_projection + bank_projections),
    )

    # One ACT exit per token, so the vocabulary projection runs once per token
    # regardless of depth.  Tied or not, it is real executed work.
    head_parameters = config.vocab_size * d

    mamba_inner = d * config.mamba_expand
    mamba_heads = mamba_inner // config.mamba_head_dim
    chunk = config.mamba_chunk_size
    state = config.mamba_d_state
    head_dim = config.mamba_head_dim
    # Mamba-2 SSD per token: intra-chunk C@B^T and its application to X, plus
    # the inter-chunk state write and read.  No parameters are involved.
    scan_per_token = mamba_heads * (
        2 * chunk * state + 2 * chunk * head_dim + 4 * state * head_dim
    )

    return {
        "active_parameters": 6.0 * tokens * active_per_pass * passes,
        "repeated_depth_memory": 6.0 * tokens * repeated_memory_parameters * passes,
        "lm_head": 6.0 * tokens * head_parameters,
        # Causal attention halves the dense score/value product.  Later passes
        # attend only over the packed active subset, so this is an upper bound.
        "attention_scores": (
            6.0
            * tokens
            * config.sequence_length
            * d
            * config.n_attention_layers
            * passes
        ),
        "mamba_scan": 3.0 * tokens * scan_per_token * config.n_mamba_layers * passes,
    }


def estimate_hardware_flops(
    config: Metis16Config,
    *,
    tokens: int,
    observed_mean_passes: float | None = None,
    observed_mean_routed_k: float | None = None,
) -> float:
    """Return the FLOPs the accelerators actually issue.

    ``activation_recompute_policy: pass`` replays a whole physical pass before
    its backward, so the hardware performs two forwards per backward instead of
    one.  Model FLOPs are 2F + 4F; executed FLOPs are 2F + 2F + 4F.

    ``layer`` replays one block at a time instead of one pass at a time. Every
    block is still replayed exactly once, so the executed total is identical --
    the two policies differ in peak activation memory, not in arithmetic, which
    is what makes ``layer`` the cheap way through the context-extension memory
    wall.  Context parallelism likewise shards this total across ranks rather
    than adding to it, so no factor appears here for it either.

    ``moe`` replays only each block's feed-forward sublayer. Its extra forward
    is priced from the audited latent projections, shared/dense experts,
    observed routed experts, and expert routers rather than charging a second
    mixer and memory forward.
    """

    model_flops = estimate_train_flops(
        config,
        tokens=tokens,
        observed_mean_passes=observed_mean_passes,
        observed_mean_routed_k=observed_mean_routed_k,
    )
    if config.activation_recompute_policy in {"pass", "layer"}:
        return model_flops * 8.0 / 6.0
    if config.activation_recompute_policy == "moe":
        audit = config.logical_parameter_audit()
        passes = float(observed_mean_passes or config.target_mean_passes)
        routed_k = float(observed_mean_routed_k or config.target_mean_routed_k)
        routed_active = (
            0.0
            if config.ffn_mode == "dense"
            else float(audit.routed_experts)
            * routed_k
            / float(config.n_routed_experts)
        )
        moe_active = (
            float(audit.latent_projections)
            + float(audit.shared_experts)
            + float(audit.expert_routers)
            + routed_active
        )
        return model_flops + 2.0 * float(tokens) * passes * moe_active
    return model_flops


def estimated_mfu(
    *,
    estimated_flops: float,
    elapsed_seconds: float,
    world_size: int,
    precision_profile: str,
) -> float:
    peak = MI300A_DENSE_PEAK_FLOPS[precision_profile] * max(world_size, 1)
    return estimated_flops / max(elapsed_seconds * peak, 1.0)


def peak_memory_bytes(device: torch.device) -> int:
    return peak_memory_evidence(device)["peak_hbm_bytes"]


def peak_memory_evidence(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        allocated = 0
        reserved = 0
    return {
        "peak_hbm_bytes": max(allocated, reserved),
        "peak_hbm_allocated_bytes": allocated,
        "peak_hbm_reserved_bytes": reserved,
    }


class MetricsWriter:
    def __init__(
        self,
        output_root: str | Path,
        *,
        enabled: bool,
        rank: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = (
            Path(output_root).expanduser().resolve()
            / "telemetry"
            / f"rank-{rank:05d}.jsonl"
        )
        self._handle: Any | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, metrics: StepMetrics | Mapping[str, Any]) -> None:
        if not self.enabled or self._handle is None:
            return
        payload = metrics.to_dict() if isinstance(metrics, StepMetrics) else dict(metrics)
        payload["recorded_unix"] = time.time()
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def scalar_telemetry(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                output[str(key)] = float(item.detach().float().item())
            elif isinstance(item, (int, float, bool, str)):
                output[str(key)] = item
        return output
    return {}


def enforce_health_gates(
    *,
    loss: float,
    grad_norm_value: float,
    telemetry: Mapping[str, Any],
    maximum_grad_norm: float,
    abort_on_nonfinite: bool,
    abort_on_token_drop: bool,
    minimum_expert_entropy_ratio: float,
    maximum_expert_load_cv: float,
    maximum_halt_collapse_fraction: float,
    require_structural_telemetry: bool = False,
    maximum_sinkhorn_marginal_error: float = math.inf,
    maximum_ponder_exit_mass_error: float = math.inf,
    minimum_mhc_stream_diversity: float = 0.0,
) -> None:
    if abort_on_nonfinite and (
        not math.isfinite(loss) or not math.isfinite(grad_norm_value)
    ):
        raise FloatingPointError(
            f"Non-finite training state: loss={loss}, grad_norm={grad_norm_value}"
        )
    drop_tokens = int(
        telemetry.get(
            "overflow_drop_tokens",
            telemetry.get("dropped_tokens", telemetry.get("token_drop_count", 0)),
        )
    )
    if abort_on_token_drop and drop_tokens != 0:
        raise RuntimeError(f"Dropless MoE invariant violated: {drop_tokens} tokens dropped")
    structural_fields = {
        "moe_assignments",
        "moe_processed_assignments",
        "moe_dropped_assignments",
        "sinkhorn_max_marginal_error",
        "ponder_exit_mass_max_error",
        "mhc_stream_diversity",
    }
    if require_structural_telemetry:
        missing = sorted(structural_fields - set(telemetry))
        if missing:
            raise RuntimeError(
                "Model structural-health telemetry is incomplete: "
                + ", ".join(missing)
            )
    assignments = telemetry.get("moe_assignments")
    processed_assignments = telemetry.get("moe_processed_assignments")
    dropped_assignments = telemetry.get("moe_dropped_assignments")
    if (
        assignments is not None
        and processed_assignments is not None
        and dropped_assignments is not None
    ):
        expected_dropped = float(assignments) - float(processed_assignments)
        if (
            not math.isfinite(expected_dropped)
            or not math.isfinite(float(dropped_assignments))
            or abs(expected_dropped - float(dropped_assignments)) > 0.5
            or (abort_on_token_drop and expected_dropped != 0.0)
        ):
            raise RuntimeError(
                "Dropless MoE assignment accounting diverged: "
                f"assigned={assignments}, processed={processed_assignments}, "
                f"reported_dropped={dropped_assignments}"
            )
    entropy = telemetry.get("expert_entropy_ratio")
    if entropy is not None and float(entropy) < minimum_expert_entropy_ratio:
        raise RuntimeError(f"Expert routing entropy collapsed: {entropy}")
    load_cv = telemetry.get("expert_load_cv")
    if load_cv is not None and float(load_cv) > maximum_expert_load_cv:
        raise RuntimeError(f"Expert load coefficient of variation is unsafe: {load_cv}")
    halt_fraction = telemetry.get("halt_collapse_fraction")
    if halt_fraction is not None and float(halt_fraction) > maximum_halt_collapse_fraction:
        raise RuntimeError(f"Depth controller collapsed: {halt_fraction}")
    sinkhorn_error = telemetry.get("sinkhorn_max_marginal_error")
    if sinkhorn_error is not None and (
        not math.isfinite(float(sinkhorn_error))
        or float(sinkhorn_error) > maximum_sinkhorn_marginal_error
    ):
        raise RuntimeError(
            "mHC Sinkhorn manifold constraint diverged: "
            f"{sinkhorn_error} > {maximum_sinkhorn_marginal_error}"
        )
    ponder_error = telemetry.get("ponder_exit_mass_max_error")
    if ponder_error is not None and (
        not math.isfinite(float(ponder_error))
        or float(ponder_error) > maximum_ponder_exit_mass_error
    ):
        raise RuntimeError(
            "Ponder exit probability mass diverged: "
            f"{ponder_error} > {maximum_ponder_exit_mass_error}"
        )
    stream_diversity = telemetry.get("mhc_stream_diversity")
    if stream_diversity is not None and (
        not math.isfinite(float(stream_diversity))
        or float(stream_diversity) < minimum_mhc_stream_diversity
    ):
        raise RuntimeError(
            "mHC residual streams collapsed: "
            f"{stream_diversity} < {minimum_mhc_stream_diversity}"
        )
    if grad_norm_value > maximum_grad_norm:
        raise RuntimeError(
            f"Gradient norm {grad_norm_value:.6g} exceeds safety gate {maximum_grad_norm:.6g}"
        )
