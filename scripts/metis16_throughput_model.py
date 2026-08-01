#!/usr/bin/env python3
"""Predict Metis-1.6 pretraining throughput on Portage before burning an allocation.

The model is bottom-up: every FLOP is a counted GEMM taken from the locked
manifest and the module structure in ``metis_training.model``, and every byte is
a counted collective.  Nothing here is fitted to a measurement.  The only free
parameters are the achieved-efficiency assumptions in :class:`SiteProfile`, and
those are exactly the quantities the Portage bringup stages measure -- feed the
measurements back with ``--measured`` and the prediction stops being a guess.

Typical use::

    scripts/metis16_throughput_model.py                       # both families
    scripts/metis16_throughput_model.py --measured probe.json # after bringup
    scripts/metis16_throughput_model.py --sweep-ep            # price EP degree
    scripts/metis16_throughput_model.py --json report.json

``--measured`` accepts any subset of the :class:`SiteProfile` fields, so the
``compute_inventory``, ``node_collectives``, and ``multinode_collectives``
bringup stages can be transcribed straight in::

    {"nic_bytes_per_second": 2.5e10, "all_to_all_efficiency": 0.62,
     "fp8_gemm_efficiency": 0.41, "accelerators_per_nic": 1}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from metis_training.metrics import (  # noqa: E402
    MI300A_DENSE_PEAK_FLOPS,
    estimate_hardware_flops,
    estimate_train_flop_terms,
)
from metis_training.model_config import Metis16Config, load_family_config  # noqa: E402


TOTAL_TRAIN_TOKENS = 1_000_000_000_000
SECONDS_PER_DAY = 86_400.0


# ---------------------------------------------------------------------------
# Site profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteProfile:
    """Hardware facts and achieved-efficiency assumptions.

    Defaults describe an HPE Cray EX255a class node: four MI300A APUs each
    paired with one Slingshot-11 Cassini NIC.  ``accelerators_per_nic`` is the
    single most load-bearing unverified number in the whole model -- if Portage
    provisions one NIC per two APUs, every throughput figure halves.
    """

    peak_fp8_flops: float = MI300A_DENSE_PEAK_FLOPS["fp8"]
    peak_bf16_flops: float = MI300A_DENSE_PEAK_FLOPS["bf16"]
    hbm_bytes_per_second: float = 5.3e12
    nic_bytes_per_second: float = 25.0e9
    accelerators_per_node: int = 4
    accelerators_per_nic: int = 1

    fp8_gemm_efficiency: float = 0.45
    bf16_gemm_efficiency: float = 0.55
    attention_efficiency: float = 0.30
    mamba_scan_efficiency: float = 0.15
    hbm_efficiency: float = 0.70
    all_to_all_efficiency: float = 0.70
    all_gather_efficiency: float = 0.75
    expert_load_imbalance: float = 1.15

    # The dispatch -> expert -> combine chain is a serial dependency inside a
    # layer; only the shared expert overlaps it (``dispatch_overlap``).
    collective_overlap_fraction: float = 0.10
    availability: float = 0.92

    @property
    def injection_bytes_per_second(self) -> float:
        return self.nic_bytes_per_second / max(self.accelerators_per_nic, 1)

    def with_measurements(self, measured: dict[str, Any]) -> "SiteProfile":
        known = {f.name for f in self.__dataclass_fields__.values()}
        unknown = sorted(set(measured) - known)
        if unknown:
            raise SystemExit(
                "Unknown site-profile fields: " + ", ".join(unknown)
            )
        return replace(self, **measured)


PROFILES = {
    "central": SiteProfile(),
    "optimistic": SiteProfile(
        fp8_gemm_efficiency=0.55,
        bf16_gemm_efficiency=0.62,
        attention_efficiency=0.38,
        mamba_scan_efficiency=0.22,
        all_to_all_efficiency=0.85,
        all_gather_efficiency=0.85,
        expert_load_imbalance=1.05,
        availability=0.95,
    ),
    "pessimistic": SiteProfile(
        fp8_gemm_efficiency=0.35,
        bf16_gemm_efficiency=0.45,
        attention_efficiency=0.24,
        mamba_scan_efficiency=0.10,
        all_to_all_efficiency=0.55,
        all_gather_efficiency=0.65,
        expert_load_imbalance=1.30,
        availability=0.88,
    ),
}


# ---------------------------------------------------------------------------
# Topology under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """An expert-parallel placement for one family.

    ``expert_parallel_size`` shards the routed experts; the remaining
    ``world_size / expert_parallel_size`` ranks hold replicas and reconcile
    them with a gradient all-reduce once per optimizer step.  Dispatch traffic
    is charged per token, replica traffic per optimizer step, which is why the
    two scale so differently.
    """

    world_size: int
    expert_parallel_size: int

    @property
    def replicas(self) -> int:
        return self.world_size // self.expert_parallel_size

    def validate(self, config: Metis16Config) -> None:
        if self.world_size % self.expert_parallel_size:
            raise SystemExit(
                f"world_size {self.world_size} is not divisible by "
                f"expert_parallel_size {self.expert_parallel_size}"
            )
        if config.n_routed_experts % self.expert_parallel_size:
            raise SystemExit(
                f"{config.n_routed_experts} routed experts do not divide across "
                f"expert_parallel_size {self.expert_parallel_size}"
            )


def manifest_topology(config: Metis16Config) -> Topology:
    return Topology(config.world_size, config.expert_parallel_size)


# ---------------------------------------------------------------------------
# Depth occupancy
# ---------------------------------------------------------------------------


def pass_occupancy(config: Metis16Config, mean_passes: float | None = None) -> list[float]:
    """Fraction of tokens still active at each physical pass.

    A geometric continuation hazard is the maximum-entropy choice given only
    the mean depth the manifest targets, and the hazard is solved so the
    occupancies sum to that mean.
    """

    target = float(mean_passes or config.target_mean_passes)
    low, high = 0.0, 0.999
    for _ in range(200):
        hazard = 0.5 * (low + high)
        if sum(hazard ** index for index in range(config.max_passes)) < target:
            low = hazard
        else:
            high = hazard
    hazard = 0.5 * (low + high)
    return [hazard ** index for index in range(config.max_passes)]


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _non_gemm_bytes_per_token(config: Metis16Config, occupancy: list[float]) -> float:
    """HBM traffic from work no GEMM efficiency term already prices.

    Four mHC streams mean the residual tensor is four times a plain
    transformer's, and it is read and written several times per layer by the
    read/mix/write pair, the depth-memory gate, and the N-gram injection.  GEMM
    operand traffic is deliberately excluded -- the GEMM efficiency factors
    already account for it.
    """

    stream_bytes = config.n_streams * config.d_model * 2
    per_layer_pass = (
        10 * stream_bytes                      # two mHC connections, read/mix/write
        + 2 * stream_bytes                     # depth-memory gated fuse
        + 4 * config.target_mean_routed_k * config.latent_dim * 2  # gather/scatter
        + 3 * config.n_routed_experts * 4      # FP32 router logits and softmax
        + 4 * config.d_model * 2               # normalisations
    )
    total = sum(occupancy) * config.n_layers * per_layer_pass
    total += 4 * config.vocab_size * 2         # FP32 logits and cross entropy
    return total


def compute_seconds_per_token(
    config: Metis16Config,
    profile: SiteProfile,
) -> dict[str, float]:
    """Seconds of accelerator time per token, by kernel class."""

    terms = estimate_train_flop_terms(config, tokens=1)
    occupancy = pass_occupancy(config)
    recompute = 8.0 / 6.0 if config.activation_recompute_policy == "pass" else 1.0

    # ``estimate_train_flop_terms`` reports model FLOPs (2F forward + 4F
    # backward).  Split them back out by the unit that executes them.
    gemm = terms["active_parameters"] + terms["repeated_depth_memory"] + terms["lm_head"]
    attention = terms["attention_scores"]
    scan = terms["mamba_scan"]

    return {
        "fp8_gemm": recompute * gemm
        / (profile.peak_fp8_flops * profile.fp8_gemm_efficiency),
        "attention": recompute * attention
        / (profile.peak_bf16_flops * profile.attention_efficiency),
        "mamba_scan": recompute * scan
        / (profile.peak_bf16_flops * profile.mamba_scan_efficiency),
        "hbm_bound": recompute
        * _non_gemm_bytes_per_token(config, occupancy)
        / (profile.hbm_bytes_per_second * profile.hbm_efficiency),
    }


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------


def _wire_bytes_per_element(config: Metis16Config, direction: str) -> float:
    """Bytes per latent element on the expert wire, including per-row scales."""

    wire = config.precision.expert_collective_wire
    fp8 = wire == "fp8" or (wire == "fp8_dispatch" and direction == "dispatch")
    if not fp8:
        return 2.0
    return 1.0 + 4.0 / float(config.latent_dim)


def collective_bytes_per_token(
    config: Metis16Config,
    topology: Topology,
    profile: SiteProfile,
    *,
    micro_batch: int,
    grad_accum: int,
) -> dict[str, float]:
    """Per-rank injected bytes attributable to one locally owned token."""

    occupancy = pass_occupancy(config)
    layer_passes = config.n_layers * sum(occupancy)
    audit = config.logical_parameter_audit()

    # Ranks sharing a node exchange over Infinity Fabric, not the NIC.
    colocated = min(profile.accelerators_per_node, topology.expert_parallel_size)
    off_node = 1.0 - colocated / topology.expert_parallel_size

    per_layer_pass = config.target_mean_routed_k * config.latent_dim * (
        _wire_bytes_per_element(config, "dispatch")
        + _wire_bytes_per_element(config, "combine")
    )
    # Forward, the checkpointed replay of that forward, and the backward pair.
    replays = 3.0 if config.activation_recompute_policy == "pass" else 2.0
    dispatch = layer_passes * per_layer_pass * replays * off_node

    tokens_per_optimizer_step = (
        topology.world_size * micro_batch * config.sequence_length * grad_accum
    )
    tokens_per_rank_step = tokens_per_optimizer_step / topology.world_size

    def ring_all_reduce(parameters: float, ranks: int) -> float:
        if ranks <= 1:
            return 0.0
        return 2.0 * parameters * 2 * (ranks - 1) / ranks / tokens_per_rank_step

    # Non-expert, non-table parameters are replicated across the whole family.
    dense = audit.stored_total - audit.routed_experts - audit.ngram_tables
    dense_reduce = ring_all_reduce(dense, topology.world_size)
    # Routed experts are replicated only across expert-data replicas.
    expert_reduce = ring_all_reduce(
        audit.routed_experts / topology.expert_parallel_size,
        topology.replicas,
    )

    tables = config.ngram_memory.retrieved_rows_per_token
    value_bytes = config.ngram_memory.value_dim * 2
    if config.ngram_memory.table_mode == "replicated":
        # ``_sync_sparse_gradient`` all-gathers padded (indices, values) over
        # the whole family, so per-rank traffic grows with world size.
        ngram = tables * (value_bytes + 8) * (topology.world_size - 1)
        ngram += ring_all_reduce(0.0, topology.world_size)
    else:
        # Row-sharded lookup: ship row ids out, values back, once forward and
        # once backward.  Retrieval is cached across passes, so depth is not a
        # multiplier here.
        ngram = tables * (8 + value_bytes) * 2
        ngram += ring_all_reduce(
            audit.ngram_tables / topology.expert_parallel_size,
            topology.replicas,
        )

    return {
        "expert_dispatch": dispatch,
        "ngram_tables": ngram,
        "expert_replica_reduce": expert_reduce,
        "dense_reduce": dense_reduce,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    family: str
    topology: Topology
    micro_batch: int
    grad_accum: int
    model_flops_per_token: float
    hardware_flops_per_token: float
    compute_seconds: dict[str, float]
    collective_bytes: dict[str, float]
    collective_seconds: dict[str, float]
    step_seconds_per_token: float
    tokens_per_second_per_accelerator: float
    tokens_per_second: float
    mfu_fp8: float
    mfu_bf16: float
    hfu_fp8: float
    collective_share: float
    injection_bytes_per_second: float
    compute_bound_tokens_per_second: float
    global_token_batch: int = 0
    curriculum_discount: float = 1.0

    def days_for(self, tokens: int, availability: float) -> float:
        rate = self.tokens_per_second / self.curriculum_discount
        return tokens / rate / SECONDS_PER_DAY / max(availability, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["topology"] = {
            "world_size": self.topology.world_size,
            "expert_parallel_size": self.topology.expert_parallel_size,
            "replicas": self.topology.replicas,
        }
        return payload


def curriculum_discount(config: Metis16Config, runtime: dict[str, Any] | None) -> float:
    """Mean depth over the campaign relative to the steady-state target.

    The curriculum starts at depth one and ramps, so the run is cheaper than
    its steady state by this factor.
    """

    if not runtime:
        return 1.0
    schedule = runtime.get("schedule", {}).get("curriculum", {})
    warm = float(schedule.get("warm_start_fraction", 0.0))
    ramp = float(schedule.get("ramp_fraction", 0.0))
    initial = float(schedule.get("initial_depth", 1.0))
    target = float(schedule.get("target_mean_depth", config.target_mean_passes))
    if target <= 0:
        return 1.0
    mean = (
        warm * initial
        + ramp * 0.5 * (initial + target)
        + max(0.0, 1.0 - warm - ramp) * target
    )
    return mean / target


def predict(
    config: Metis16Config,
    profile: SiteProfile,
    topology: Topology,
    *,
    micro_batch: int,
    grad_accum: int,
    discount: float = 1.0,
) -> Prediction:
    topology.validate(config)
    compute = compute_seconds_per_token(config, profile)
    compute_total = sum(compute.values())

    collective_bytes = collective_bytes_per_token(
        config,
        topology,
        profile,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
    )
    injection = profile.injection_bytes_per_second
    collective_seconds = {
        "expert_dispatch": collective_bytes["expert_dispatch"]
        / (injection * profile.all_to_all_efficiency)
        * profile.expert_load_imbalance,
        "ngram_tables": collective_bytes["ngram_tables"]
        / (injection * profile.all_gather_efficiency),
        "expert_replica_reduce": collective_bytes["expert_replica_reduce"]
        / (injection * profile.all_gather_efficiency),
        "dense_reduce": collective_bytes["dense_reduce"]
        / (injection * profile.all_gather_efficiency),
    }
    collective_total = sum(collective_seconds.values())

    # The EP dispatch/expert/combine chain is the only pipelineable stage; the
    # N-gram lookup and the gradient reductions sit outside the layer loop.
    pipelined = collective_seconds["expert_dispatch"]
    serial_collectives = collective_total - pipelined
    chunks = max(int(config.moe_dispatch_chunks), 1)
    if chunks > 1:
        # Steady state is bounded by the busier resource; fill and drain leave
        # roughly one chunk of the other one exposed.
        step = (
            max(pipelined, compute_total)
            + min(pipelined, compute_total) / chunks
            + serial_collectives
        )
    else:
        overlap = min(
            compute_total * profile.collective_overlap_fraction,
            collective_total * 0.15,
        )
        step = compute_total + collective_total - overlap

    model_flops = sum(estimate_train_flop_terms(config, tokens=1).values())
    hardware_flops = estimate_hardware_flops(config, tokens=1)
    per_accelerator = 1.0 / step
    family_rate = per_accelerator * topology.world_size

    return Prediction(
        family=config.family,
        topology=topology,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        model_flops_per_token=model_flops,
        hardware_flops_per_token=hardware_flops,
        compute_seconds=compute,
        collective_bytes=collective_bytes,
        collective_seconds=collective_seconds,
        step_seconds_per_token=step,
        tokens_per_second_per_accelerator=per_accelerator,
        tokens_per_second=family_rate,
        mfu_fp8=model_flops * per_accelerator / profile.peak_fp8_flops,
        mfu_bf16=model_flops * per_accelerator / profile.peak_bf16_flops,
        hfu_fp8=hardware_flops * per_accelerator / profile.peak_fp8_flops,
        collective_share=collective_total / (compute_total + collective_total),
        injection_bytes_per_second=sum(collective_bytes.values()) * per_accelerator,
        compute_bound_tokens_per_second=1.0 / compute_total,
        global_token_batch=topology.world_size
        * micro_batch
        * config.sequence_length
        * grad_accum,
        curriculum_discount=discount,
    )


def batch_for(config: Metis16Config, topology: Topology) -> tuple[int, int]:
    """Largest manifest-legal (micro_batch, grad_accum) nearest the batch target."""

    bounds = config.autotune
    best: tuple[float, int, int] | None = None
    for micro in bounds.micro_batch_sizes:
        for accumulation in bounds.grad_accum_steps:
            tokens = (
                micro * accumulation * topology.world_size * config.sequence_length
            )
            if not (
                bounds.global_token_batch.minimum
                <= tokens
                <= bounds.global_token_batch.maximum
            ):
                continue
            distance = abs(tokens - bounds.global_token_batch.target)
            row = (distance, -micro, accumulation)
            if best is None or row < best:
                best = row
    if best is None:
        raise SystemExit(
            f"No manifest-legal batch fits world_size {topology.world_size}"
        )
    return -best[1], best[2]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _si(value: float) -> str:
    for unit, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= unit:
            return f"{value / unit:.3f}{suffix}"
    return f"{value:.2f}"


def print_prediction(prediction: Prediction, config: Metis16Config, profile: SiteProfile) -> None:
    topology = prediction.topology
    audit = config.logical_parameter_audit()
    print(f"\n### {config.name}")
    print(
        f"  {topology.world_size} APUs "
        f"({topology.world_size // profile.accelerators_per_node} nodes), "
        f"EP={topology.expert_parallel_size} x {topology.replicas} replicas, "
        f"{audit.routed_experts / topology.expert_parallel_size / 1e6:.0f}M expert "
        f"params/rank, {config.n_routed_experts // topology.expert_parallel_size} "
        f"expert(s)/rank"
    )
    print(
        f"  batch {prediction.micro_batch} x {prediction.grad_accum} accum "
        f"= {prediction.global_token_batch / 1e6:.2f}M tokens/step   "
        f"n-gram tables {config.ngram_memory.table_mode}   "
        f"expert wire {config.precision.expert_collective_wire}   "
        f"dispatch pipeline x{config.moe_dispatch_chunks}"
    )
    print(
        f"  model FLOPs/token {_si(prediction.model_flops_per_token)}   "
        f"executed {_si(prediction.hardware_flops_per_token)}"
    )
    print("  compute  " + "  ".join(
        f"{name} {seconds * 1e6:.2f}us"
        for name, seconds in prediction.compute_seconds.items()
    ))
    print("  comms    " + "  ".join(
        f"{name} {seconds * 1e6:.2f}us"
        for name, seconds in prediction.collective_seconds.items()
        if seconds * 1e6 >= 0.005
    ))
    print(
        f"  step {prediction.step_seconds_per_token * 1e6:.2f}us/token   "
        f"collectives are {100 * prediction.collective_share:.0f}% of the serial path   "
        f"{prediction.injection_bytes_per_second / 1e9:.1f} GB/s/APU injected "
        f"(NIC {profile.injection_bytes_per_second / 1e9:.0f})"
    )
    print(
        f"  {prediction.tokens_per_second_per_accelerator:,.0f} tok/s/APU   "
        f"{prediction.tokens_per_second:,.0f} tok/s family   "
        f"MFU {100 * prediction.mfu_fp8:.1f}% fp8 "
        f"({100 * prediction.mfu_bf16:.1f}% bf16)   "
        f"HFU {100 * prediction.hfu_fp8:.1f}% fp8"
    )
    print(
        f"  compute-bound ceiling "
        f"{prediction.compute_bound_tokens_per_second:,.0f} tok/s/APU -> "
        f"collectives cost "
        f"{100 * (1 - prediction.tokens_per_second_per_accelerator / prediction.compute_bound_tokens_per_second):.0f}%"
    )


def sweep_expert_parallel(
    config: Metis16Config,
    profile: SiteProfile,
    discount: float,
    tokens: int,
) -> list[Prediction]:
    """Price every legal expert-parallel degree at fixed world size.

    Replication does not reduce dispatch volume -- a token still ships to k
    experts wherever they live -- but it moves the exchange onto fewer, closer
    ranks with larger messages, and it converts nothing into per-token traffic
    because replica reconciliation amortises over a whole optimizer step.
    """

    world = config.world_size
    results: list[Prediction] = []
    degree = world
    while degree >= 1:
        if world % degree == 0 and config.n_routed_experts % degree == 0:
            topology = Topology(world, degree)
            micro, accumulation = batch_for(config, topology)
            local = profile
            if degree <= profile.accelerators_per_node:
                # A group that fits inside one node never touches the fabric.
                local = replace(profile, all_to_all_efficiency=1.0, expert_load_imbalance=1.05)
            results.append(
                predict(
                    config,
                    local,
                    topology,
                    micro_batch=micro,
                    grad_accum=accumulation,
                    discount=discount,
                )
            )
        degree //= 2
    return results


def print_sweep(
    config: Metis16Config,
    predictions: list[Prediction],
    profile: SiteProfile,
    tokens: int,
) -> None:
    print(f"\n  expert-parallel sweep for {config.name} at world_size {config.world_size}")
    print(
        "    EP  repl  experts/rank  GB/rank-state   EP kB/tok  tok/s/APU   "
        "MFU fp8   days"
    )
    audit = config.logical_parameter_audit()
    for prediction in predictions:
        topology = prediction.topology
        state = (
            audit.stored_total
            - audit.routed_experts
            + audit.routed_experts / topology.expert_parallel_size
        )
        if config.ngram_memory.table_mode == "row_sharded":
            state -= audit.ngram_tables * (1 - 1 / topology.expert_parallel_size)
        print(
            f"    {topology.expert_parallel_size:>3d}  {topology.replicas:>4d}  "
            f"{config.n_routed_experts // topology.expert_parallel_size:>12d}  "
            f"{state * 14 / 1e9:>13.0f}   "
            f"{prediction.collective_bytes['expert_dispatch'] / 1e3:>9.1f}   "
            f"{prediction.tokens_per_second_per_accelerator:>9,.0f}   "
            f"{100 * prediction.mfu_fp8:>6.1f}%   "
            f"{prediction.days_for(tokens, profile.availability):>4.1f}"
        )
    print(
        "    state column is bf16 weight + fp32 master + two fp32 AdaMuon "
        f"moments; the APU carries {128} GB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", action="append", choices=["praxis", "logos"])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="central")
    parser.add_argument("--measured", type=Path, help="JSON of measured SiteProfile fields")
    parser.add_argument("--tokens", type=int, default=TOTAL_TRAIN_TOKENS)
    parser.add_argument("--expert-parallel-size", type=int)
    parser.add_argument("--sweep-ep", action="store_true", help="price every legal EP degree")
    parser.add_argument("--sweep-chunks", action="store_true",
                        help="price the MoE dispatch pipeline depth")
    parser.add_argument("--all-profiles", action="store_true")
    parser.add_argument("--json", type=Path, help="write the full prediction as JSON")
    arguments = parser.parse_args(argv)

    families = arguments.family or ["praxis", "logos"]
    runtime = _load_runtime_contract()

    profiles = (
        {name: PROFILES[name] for name in ("optimistic", "central", "pessimistic")}
        if arguments.all_profiles
        else {arguments.profile: PROFILES[arguments.profile]}
    )
    if arguments.measured:
        measured = json.loads(arguments.measured.read_text())
        profiles = {
            f"{name}+measured": profile.with_measurements(measured)
            for name, profile in profiles.items()
        }

    report: dict[str, Any] = {"tokens": arguments.tokens, "profiles": {}}
    for name, profile in profiles.items():
        print("=" * 78)
        print(f"  {name.upper()}   "
              f"NIC {profile.injection_bytes_per_second / 1e9:.1f} GB/s/APU, "
              f"fp8 GEMM {profile.fp8_gemm_efficiency:.0%}, "
              f"a2a {profile.all_to_all_efficiency:.0%}, "
              f"availability {profile.availability:.0%}")
        print("=" * 78)
        entries = []
        total_injection = 0.0
        for family in families:
            config = load_family_config(family)
            discount = curriculum_discount(config, runtime)
            topology = (
                Topology(config.world_size, arguments.expert_parallel_size)
                if arguments.expert_parallel_size
                else manifest_topology(config)
            )
            micro, accumulation = batch_for(config, topology)
            prediction = predict(
                config,
                profile,
                topology,
                micro_batch=micro,
                grad_accum=accumulation,
                discount=discount,
            )
            print_prediction(prediction, config, profile)
            entries.append((config, prediction))
            total_injection += prediction.injection_bytes_per_second * topology.world_size
            if arguments.sweep_chunks:
                print(f"\n  dispatch-pipeline depth for {config.name}")
                print("      chunks  tok/s/APU   MFU fp8    days   vs serial")
                for depth in (1, 2, 3, 4, 6, 8):
                    variant = replace(config, moe_dispatch_chunks=depth)
                    row = predict(variant, profile, topology,
                                  micro_batch=micro, grad_accum=accumulation,
                                  discount=discount)
                    if depth == 1:
                        serial_rate = row.tokens_per_second_per_accelerator
                    print(f"      {depth:>6d}  {row.tokens_per_second_per_accelerator:>9,.0f}  "
                          f"{100*row.mfu_fp8:>7.1f}%  {row.days_for(arguments.tokens, profile.availability):>6.2f}"
                          f"  {row.tokens_per_second_per_accelerator/serial_rate:>8.2f}x")
            if arguments.sweep_ep:
                print_sweep(
                    config,
                    sweep_expert_parallel(config, profile, discount, arguments.tokens),
                    profile,
                    arguments.tokens,
                )

        print(f"\n  {arguments.tokens / 1e12:.0f}T-token campaign")
        for config, prediction in entries:
            days = prediction.days_for(arguments.tokens, profile.availability)
            steps = arguments.tokens / prediction.global_token_batch
            print(
                f"    {config.family:<7s} {days:6.1f} days at "
                f"{profile.availability:.0%} availability   "
                f"{steps:,.0f} optimizer steps   "
                f"{prediction.global_token_batch / prediction.tokens_per_second * prediction.curriculum_discount:.2f} s/step   "
                f"{arguments.tokens / prediction.tokens_per_second_per_accelerator / 3600 / 1e3:.0f}k APU-hours"
            )
        nodes = sum(
            prediction.topology.world_size for _config, prediction in entries
        ) / profile.accelerators_per_node
        print(
            f"    fabric: {total_injection / 1e12:.2f} TB/s aggregate over "
            f"{nodes:.0f} nodes = {total_injection / max(nodes, 1) / 1e9:.1f} GB/s/node "
            f"({100 * total_injection / max(nodes, 1) / (profile.accelerators_per_node * profile.injection_bytes_per_second):.0f}% "
            "of node injection capacity)"
        )
        report["profiles"][name] = {
            config.family: prediction.to_dict() for config, prediction in entries
        }

    if arguments.json:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {arguments.json}")
    return 0


def _load_runtime_contract() -> dict[str, Any] | None:
    path = REPOSITORY_ROOT / "configs" / "metis16" / "training-runtime.yaml"
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a mandatory package
        return None
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
