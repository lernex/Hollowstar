from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from metis_training.mhc_kernels import (
    run_mhc_fused_canary,
    validate_mhc_canary_report,
)
from metis_training.model_config import load_family_config
from metis_training.precision import benchmark_exact_precision_roles
from metis_training.precision_plan import measured_role_dtype_map

from .config import PortageConfig
from .distributed import DistributedContext
from .telemetry import percentile, snapshot_cxi
from .util import atomic_write_json, json_sha256, utc_now


_MANDATORY_FP8_ROLE_SHAPES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("praxis_shared_mixer_projection", (4096, 2048, 4096)),
    ("logos_shared_mixer_projection", (4096, 2560, 5120)),
    ("praxis_routed_expert_bottleneck", (8192, 1024, 512)),
    ("logos_routed_expert_bottleneck", (8192, 1024, 768)),
    ("praxis_vocabulary_projection", (4096, 2048, 65536)),
    ("logos_vocabulary_projection", (4096, 2560, 65536)),
)


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _timed(operation, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    _sync()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        _sync()
        samples.append(time.perf_counter() - started)
    return samples


def _quantiles(samples: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.fmean(samples),
        "p50_seconds": percentile(samples, 0.50),
        "p95_seconds": percentile(samples, 0.95),
        "p99_seconds": percentile(samples, 0.99),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def _bf16_gemm(m: int, k: int, n: int, *, warmup: int, iterations: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(16062026 + m + k + n)
    left = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    right = torch.randn((k, n), device="cuda", dtype=torch.bfloat16, generator=generator)

    def operation():
        torch.mm(left, right)

    samples = _timed(operation, warmup=warmup, iterations=iterations)
    timing = _quantiles(samples)
    flops = 2.0 * m * k * n
    timing.update(
        {
            "m": m,
            "k": k,
            "n": n,
            "dtype": "bf16",
            "tflops": flops / timing["mean_seconds"] / 1e12,
        }
    )
    return timing


def _fp8_dtype(torch) -> Any | None:
    for name in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            return dtype
    return None


def _scaled_mm(torch, left, right, scale_left, scale_right):
    operation = getattr(torch, "_scaled_mm", None)
    if operation is None:
        raise RuntimeError("torch._scaled_mm is unavailable")
    kwargs = {
        "scale_a": scale_left,
        "scale_b": scale_right,
        "out_dtype": torch.bfloat16,
    }
    try:
        return operation(left, right, **kwargs)
    except TypeError:
        # Older ROCm PyTorch builds accepted positional scales.
        return operation(left, right, scale_left, scale_right, out_dtype=torch.bfloat16)


def _fp8_gemm(
    m: int,
    k: int,
    n: int,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch

    dtype = _fp8_dtype(torch)
    if dtype is None:
        raise RuntimeError("PyTorch exposes no FP8 E4M3 dtype")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(16062026 + m + k + n)
    left_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    right_storage = torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    right_bf16 = right_storage.t()
    fp8_max = float(torch.finfo(dtype).max)
    left_scale = left_bf16.abs().amax().float().clamp_min(1e-12) / fp8_max
    right_scale = right_bf16.abs().amax().float().clamp_min(1e-12) / fp8_max
    left = (left_bf16 / left_scale).clamp(-fp8_max, fp8_max).to(dtype)
    # hipBLASLt expects the B operand in a kernel-compatible transposed layout.
    right = (
        (right_storage / right_scale)
        .clamp(-fp8_max, fp8_max)
        .to(dtype)
        .t()
    )
    reference = torch.mm(left_bf16.float(), right_bf16.float())

    def operation():
        _scaled_mm(torch, left, right, left_scale, right_scale)

    candidate = operation()
    if isinstance(candidate, tuple):
        candidate = candidate[0]
    relative_error = (
        (candidate.float() - reference).norm() / reference.norm().clamp_min(1e-12)
    ).item()
    samples = _timed(operation, warmup=warmup, iterations=iterations)
    timing = _quantiles(samples)
    flops = 2.0 * m * k * n
    timing.update(
        {
            "m": m,
            "k": k,
            "n": n,
            "dtype": str(dtype),
            "tflops": flops / timing["mean_seconds"] / 1e12,
            "relative_l2_error": relative_error,
        }
    )
    return timing


def _fp8_role_capabilities(
    measurements: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    maximum_error: float,
) -> dict[str, dict[str, Any]]:
    measured = {
        (
            int(row.get("m", -1)),
            int(row.get("k", -1)),
            int(row.get("n", -1)),
        ): row
        for row in measurements
    }
    failures = {
        tuple(int(item) for item in row.get("shape", ())): str(
            row.get("error", "unknown FP8 error")
        )
        for row in errors
        if isinstance(row.get("shape"), list) and len(row["shape"]) == 3
    }
    capabilities: dict[str, dict[str, Any]] = {}
    for role, shape in _MANDATORY_FP8_ROLE_SHAPES:
        row = measured.get(shape)
        relative_error = (
            float(row["relative_l2_error"])
            if row is not None
            and isinstance(row.get("relative_l2_error"), (int, float))
            and not isinstance(row.get("relative_l2_error"), bool)
            else float("inf")
        )
        finite = bool(
            row is not None
            and isinstance(row.get("tflops"), (int, float))
            and not isinstance(row.get("tflops"), bool)
            and math.isfinite(float(row["tflops"]))
            and math.isfinite(relative_error)
        )
        passed = finite and relative_error <= maximum_error
        capabilities[role] = {
            "shape": list(shape),
            "mandatory": True,
            "ok": passed,
            "relative_l2_error": (
                relative_error if math.isfinite(relative_error) else None
            ),
            "maximum_relative_l2_error": maximum_error,
            "tflops": (
                float(row["tflops"]) if finite and row is not None else None
            ),
            "error": (
                None
                if passed
                else failures.get(
                    shape,
                    "missing or numerically invalid exact-role FP8 measurement",
                )
            ),
        }
    return capabilities


def run_single_apu_probe(config: PortageConfig) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Single-APU probe requires PyTorch ROCm on a visible MI300A")
    torch.cuda.set_device(0)
    warmup = int(config.raw["bringup"]["warmup_iterations"])
    iterations = int(config.raw["bringup"]["timed_iterations"])
    maximum_error = float(config.raw["bringup"]["fp8_max_relative_error"])
    family_configs = tuple(
        load_family_config(family.manifest, family=family.name)
        for family in config.families
    )
    family_precision: dict[str, dict[str, Any]] = {}
    for family_config in family_configs:
        family_maximum_error = min(
            maximum_error,
            float(
                family_config.autotune.gates.max_fp8_loss_relative_error
            ),
        )
        family_precision[family_config.family] = benchmark_exact_precision_roles(
            family_config,
            device=torch.device("cuda", 0),
            warmup_iterations=warmup,
            timed_iterations=iterations,
            maximum_relative_error=family_maximum_error,
            minimum_fp8_speedup=float(
                config.raw["bringup"]["precision_probe_minimum_fp8_speedup"]
            ),
            maximum_probe_rows=int(
                config.raw["bringup"]["precision_probe_maximum_rows"]
            ),
            maximum_activation_elements=int(
                config.raw["bringup"][
                    "precision_probe_maximum_activation_elements"
                ]
            ),
        )
        torch.cuda.empty_cache()
    role_capabilities = {
        family: {
            role: {
                "selected_dtype": dtype,
                "ok": True,
                "heavy": bool(
                    result["precision_role_plan"]["roles"][role]["heavy"]
                ),
                "reason": result["precision_role_plan"]["roles"][role][
                    "reason"
                ],
            }
            for role, dtype in measured_role_dtype_map(
                result["precision_role_plan"]
            ).items()
        }
        for family, result in family_precision.items()
    }
    fp8_role_count = sum(
        row["selected_dtype"] == "fp8"
        for family_rows in role_capabilities.values()
        for row in family_rows.values()
    )
    bf16_role_count = sum(
        row["selected_dtype"] == "bf16"
        for family_rows in role_capabilities.values()
        for row in family_rows.values()
    )
    mhc_canary = run_mhc_fused_canary(
        family_configs,
        device=torch.device("cuda", 0),
        token_count=int(config.raw["bringup"]["mhc_canary_tokens"]),
        maximum_forward_relative_error=float(
            config.raw["bringup"]["mhc_max_forward_relative_error"]
        ),
        maximum_backward_relative_error=float(
            config.raw["bringup"]["mhc_max_backward_relative_error"]
        ),
        minimum_speedup=float(config.raw["bringup"]["mhc_minimum_speedup"]),
        warmup_iterations=warmup,
        timed_iterations=iterations,
    )
    validate_mhc_canary_report(
        mhc_canary,
        configs=family_configs,
        maximum_forward_relative_error=float(
            config.raw["bringup"]["mhc_max_forward_relative_error"]
        ),
        maximum_backward_relative_error=float(
            config.raw["bringup"]["mhc_max_backward_relative_error"]
        ),
        minimum_speedup=float(config.raw["bringup"]["mhc_minimum_speedup"]),
    )
    report: dict[str, Any] = {
        "schema": "metis.portage-probe/v1",
        "stage": "single_apu",
        "created_at": utc_now(),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "device_properties": {
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "gcn_arch_name": getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
        },
        "family_precision": family_precision,
        "precision_role_capabilities": role_capabilities,
        "precision_capability": {
            "mode": "sealed_per_exact_role",
            "family_count": len(family_precision),
            "role_count": fp8_role_count + bf16_role_count,
            "fp8_role_count": fp8_role_count,
            "bf16_role_count": bf16_role_count,
            "mixed_precision_authorized": True,
        },
        "mhc_fused_canary": mhc_canary,
        "cxi": snapshot_cxi(config.raw["telemetry"]["cxi_paths"]),
        "gates": {
            "all_exact_roles_probed": all(
                result.get("ok") is True
                and result.get("precision_role_plan", {})
                .get("classification", {})
                .get("all_roles_probed")
                is True
                for result in family_precision.values()
            ),
            "all_heavy_roles_safely_classified": all(
                result.get("precision_role_plan", {})
                .get("classification", {})
                .get("all_heavy_roles_safely_classified")
                is True
                for result in family_precision.values()
            ),
            "mhc_fused_forward_backward_parity": mhc_canary["ok"] is True,
        },
    }
    report["ok"] = all(report["gates"].values())
    report["report_sha256"] = json_sha256(report)
    return report


def _all_reduce_correctness(context: DistributedContext) -> None:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(context.rank + 1)], device="cuda", dtype=torch.float64)
    if context.initialized:
        dist.all_reduce(tensor)
    expected = context.world_size * (context.world_size + 1) / 2
    if not math.isclose(float(tensor.item()), expected, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(f"RCCL all-reduce correctness failed: {tensor.item()} != {expected}")


def _all_to_allv_correctness(context: DistributedContext) -> None:
    if not context.initialized:
        return
    import torch
    import torch.distributed as dist

    world = context.world_size

    def count(source: int, destination: int) -> int:
        return 1 + ((source * 17 + destination * 13) % 7)

    send_splits = [count(context.rank, destination) for destination in range(world)]
    receive_splits = [count(source, context.rank) for source in range(world)]
    send = torch.cat(
        [
            torch.full((size,), context.rank, dtype=torch.int32, device="cuda")
            for size in send_splits
        ]
    )
    receive = torch.empty(sum(receive_splits), dtype=torch.int32, device="cuda")
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=receive_splits,
        input_split_sizes=send_splits,
    )
    cursor = 0
    for source, size in enumerate(receive_splits):
        values = receive[cursor : cursor + size]
        if not torch.all(values == source):
            raise RuntimeError(
                f"RCCL all-to-all-v corruption: destination={context.rank}, source={source}"
            )
        cursor += size


def _collective_metrics(
    context: DistributedContext,
    *,
    message_bytes: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    elements = max(context.world_size, message_bytes // element_size)
    elements -= elements % context.world_size
    send = torch.full((elements,), float(context.rank), device="cuda", dtype=torch.bfloat16)
    receive = torch.empty_like(send)

    def all_to_all():
        dist.all_to_all_single(receive, send)

    def all_reduce():
        receive.copy_(send)
        dist.all_reduce(receive)

    a2a_samples = _timed(all_to_all, warmup=warmup, iterations=iterations)
    reduce_samples = _timed(all_reduce, warmup=warmup, iterations=iterations)
    a2a = _quantiles(a2a_samples)
    reduce = _quantiles(reduce_samples)
    actual_bytes = elements * element_size
    # Algorithm bandwidth is intentionally reported without pretending it is
    # physical link bandwidth.  The trainer also records per-layer wire bytes.
    a2a["algorithm_bandwidth_gbps"] = actual_bytes / a2a["mean_seconds"] / 1e9
    reduce["algorithm_bandwidth_gbps"] = actual_bytes / reduce["mean_seconds"] / 1e9
    return {
        "message_bytes": actual_bytes,
        "all_to_all": a2a,
        "all_reduce": reduce,
    }


def run_collective_probe(
    config: PortageConfig,
    context: DistributedContext,
    *,
    stage: str,
) -> dict[str, Any] | None:
    if not context.initialized:
        raise RuntimeError("Collective probes require more than one distributed rank")
    import torch.distributed as dist

    expected_world = {"node_collectives": 4, "multinode_collectives": 16}[stage]
    if context.world_size != expected_world:
        raise RuntimeError(
            f"{stage} requires exactly {expected_world} ranks, got {context.world_size}"
        )
    _all_reduce_correctness(context)
    _all_to_allv_correctness(context)
    dist.barrier()
    warmup = int(config.raw["bringup"]["warmup_iterations"])
    iterations = int(config.raw["bringup"]["timed_iterations"])
    local_metrics = [
        _collective_metrics(
            context,
            message_bytes=int(size),
            warmup=warmup,
            iterations=iterations,
        )
        for size in config.raw["bringup"]["collective_bytes"]
    ]
    gathered: list[list[dict[str, Any]] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_metrics)
    local_cxi = snapshot_cxi(config.raw["telemetry"]["cxi_paths"])
    gathered_cxi: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered_cxi, local_cxi)
    if not context.is_root:
        return None
    concrete = [item for item in gathered if item is not None]
    aggregate: list[dict[str, Any]] = []
    for position, message_bytes in enumerate(config.raw["bringup"]["collective_bytes"]):
        a2a_bandwidths = [
            float(item[position]["all_to_all"]["algorithm_bandwidth_gbps"])
            for item in concrete
        ]
        reduce_bandwidths = [
            float(item[position]["all_reduce"]["algorithm_bandwidth_gbps"])
            for item in concrete
        ]
        a2a_mean = statistics.fmean(a2a_bandwidths)
        aggregate.append(
            {
                "message_bytes": int(message_bytes),
                "all_to_all_bandwidth_gbps_mean": a2a_mean,
                "all_to_all_bandwidth_gbps_min": min(a2a_bandwidths),
                "all_to_all_bandwidth_cv": (
                    statistics.pstdev(a2a_bandwidths) / a2a_mean if a2a_mean else float("inf")
                ),
                "all_reduce_bandwidth_gbps_mean": statistics.fmean(reduce_bandwidths),
                "all_reduce_bandwidth_gbps_min": min(reduce_bandwidths),
            }
        )
    largest = aggregate[-1]
    gates = {
        "all_ranks_reported": len(concrete) == context.world_size,
        "alltoall_bandwidth": (
            stage != "multinode_collectives"
            or largest["all_to_all_bandwidth_gbps_min"]
            >= float(config.raw["bringup"]["minimum_multinode_bus_bandwidth_gbps"])
        ),
        "alltoall_rank_variation": largest["all_to_all_bandwidth_cv"]
        <= float(config.raw["bringup"]["maximum_collective_cv"]),
        "cxi-readable": (
            not bool(config.raw["site"]["require_cxi_counters"])
            or all(item and item.get("ok") for item in gathered_cxi)
        ),
    }
    report: dict[str, Any] = {
        "schema": "metis.portage-probe/v1",
        "stage": stage,
        "created_at": utc_now(),
        "world_size": context.world_size,
        "per_rank": concrete,
        "aggregate": aggregate,
        "cxi": gathered_cxi,
        "gates": gates,
        "overflow_drop_tokens": 0,
        "collective_errors": 0,
        "ok": all(gates.values()),
    }
    report["report_sha256"] = json_sha256(report)
    return report


def write_probe_report(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)
    if report.get("ok") is not True:
        failed = [key for key, value in report.get("gates", {}).items() if not value]
        raise RuntimeError(f"Portage {report.get('stage')} probe failed: {', '.join(failed)}")
