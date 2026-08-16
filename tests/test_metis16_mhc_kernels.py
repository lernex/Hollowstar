from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from metis_training.mhc_kernels import (
    MHC_CANARY_SCHEMA,
    _canary_geometry,
    _canonical_sha256,
    mhc_masked_write,
    mhc_read_mix,
    require_mhc_backend,
    run_mhc_fused_canary,
    triton_rocm_backend_status,
    validate_mhc_canary_report,
)
from metis_training.model_config import Metis16Config, load_family_config


def test_tiny_reference_path_preserves_forward_and_backward_contract() -> None:
    generator = torch.Generator().manual_seed(16062026)
    streams = torch.randn(2, 3, 4, 8, generator=generator, requires_grad=True)
    matrix = torch.randn(4, 4, generator=generator, requires_grad=True)
    read = torch.softmax(
        torch.randn(4, generator=generator), dim=-1
    ).requires_grad_(True)
    write = torch.softmax(
        torch.randn(4, generator=generator), dim=-1
    ).requires_grad_(True)
    update = torch.randn(2, 3, 8, generator=generator, requires_grad=True)
    active = torch.tensor([[True, False, True], [False, True, True]])

    source, mixed = mhc_read_mix(
        streams,
        matrix,
        read,
        backend="torch_reference",
        family="tiny",
    )
    output = mhc_masked_write(
        mixed,
        write,
        update,
        streams,
        active,
        backend="torch_reference",
        family="tiny",
    )
    expected_source = torch.einsum("i,...id->...d", read, streams)
    expected_mixed = torch.einsum("oi,...id->...od", matrix, streams)
    expected_update = expected_mixed + torch.einsum("i,...d->...id", write, update)
    expected_output = torch.where(
        active[..., None, None],
        expected_update,
        streams,
    )
    torch.testing.assert_close(source, expected_source)
    torch.testing.assert_close(output, expected_output)
    (source.square().mean() + output.square().mean()).backward()
    for value in (streams, matrix, read, write, update):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_production_config_rejects_reference_mhc_backend() -> None:
    for family in ("praxis", "logos"):
        config = load_family_config(family=family)
        assert config.mhc_backend == "fused_required"
        with pytest.raises(ValueError, match="fused Triton/ROCm"):
            replace(config, mhc_backend="torch_reference").validate()
    assert Metis16Config.tiny_for_tests().mhc_backend == "torch_reference"


def test_production_backend_fails_closed_without_rocm_triton() -> None:
    status = triton_rocm_backend_status(torch.device("cpu"))
    assert status["available"] is False
    with pytest.raises(RuntimeError, match="fused Triton/ROCm"):
        require_mhc_backend(
            backend="fused_required",
            family="praxis",
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="restricted to tiny"):
        require_mhc_backend(
            backend="torch_reference",
            family="logos",
            device=torch.device("cpu"),
        )
    # The MoRE ablation family may run the reference backend so the ladder can
    # be dry-run on CPU; its production launches still declare fused_required.
    require_mhc_backend(
        backend="torch_reference",
        family="ablation",
        device=torch.device("cpu"),
    )


def _sealed_synthetic_report(configs):
    rows = {}
    for config in configs:
        geometry = _canary_geometry(config)
        rows[config.family] = {
            **geometry,
            "geometry_sha256": _canonical_sha256(geometry),
            "tokens": 257,
            "dtype": "bfloat16",
            "forward_relative_l2": {
                "source": 0.001,
                "masked_write": 0.002,
            },
            "backward_relative_l2": {
                "streams": 0.003,
                "matrix": 0.004,
                "read_weights": 0.005,
                "write_weights": 0.006,
                "update": 0.007,
            },
            "performance": {
                "warmup_iterations": 4,
                "timed_iterations": 12,
                "scope": "read_mix_and_masked_write_forward_backward",
                "fused": {
                    "median_seconds": 0.001,
                    "p95_seconds": 0.0012,
                },
                "torch_reference": {
                    "median_seconds": 0.0011,
                    "p95_seconds": 0.0013,
                },
                "speedup_reference_over_fused": 1.1,
                "minimum_speedup": 1.0,
                "throughput_positive": True,
            },
            "finite": True,
            "passed": True,
        }
    report = {
        "schema": MHC_CANARY_SCHEMA,
        "created_at": "2026-07-24T00:00:00+00:00",
        "backend": {
            "available": True,
            "backend": "triton_rocm",
            "device": "cuda:0",
            "torch": "test",
            "rocm": "test",
            "triton": "test",
            "reason": None,
        },
        "thresholds": {
            "maximum_forward_relative_l2": 0.03,
            "maximum_backward_relative_l2": 0.08,
            "minimum_speedup_reference_over_fused": 1.0,
        },
        "families": rows,
        "ok": True,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def test_mhc_canary_report_is_family_complete_and_tamper_evident() -> None:
    configs = tuple(load_family_config(family=family) for family in ("praxis", "logos"))
    report = _sealed_synthetic_report(configs)
    validate_mhc_canary_report(report, configs=configs)

    report["families"]["logos"]["performance"]["speedup_reference_over_fused"] = 0.5
    with pytest.raises(RuntimeError, match="hash is invalid or stale"):
        validate_mhc_canary_report(report, configs=configs)

    report = _sealed_synthetic_report(configs)
    report["families"]["logos"]["performance"].update(
        {
            "speedup_reference_over_fused": 0.5,
            "throughput_positive": False,
        }
    )
    report["report_sha256"] = _canonical_sha256(report, omit=("report_sha256",))
    with pytest.raises(RuntimeError, match="parity gate failed"):
        validate_mhc_canary_report(report, configs=configs)

    report = _sealed_synthetic_report(configs)
    del report["families"]["logos"]
    report["report_sha256"] = _canonical_sha256(report, omit=("report_sha256",))
    with pytest.raises(RuntimeError, match="family coverage is incomplete"):
        validate_mhc_canary_report(report, configs=configs)


@pytest.mark.skipif(
    not triton_rocm_backend_status().get("available", False),
    reason="requires a visible ROCm device and Triton",
)
def test_actual_triton_rocm_kernel_matches_exact_family_widths() -> None:
    configs = tuple(load_family_config(family=family) for family in ("praxis", "logos"))
    report = run_mhc_fused_canary(
        configs,
        device=torch.device("cuda", 0),
        token_count=7,
    )
    validate_mhc_canary_report(report, configs=configs)


def test_mhc_stream_constant_mirrors_agree() -> None:
    """The kernels read a constexpr mirror of ``_N_STREAMS``; pin them together.

    Triton refuses to let a ``@jit`` kernel read a plain module global, so the
    stream count exists twice: an int for host-side shape checks and stride
    arithmetic, and a ``tl.constexpr`` the kernels index with. Two constants
    for one quantity is exactly the shape that drifts, and the failure would be
    silent -- the kernels would read the wrong stride and return plausible
    numbers rather than raising.
    """

    import metis_training.mhc_kernels as kernels

    if kernels.triton is None:
        pytest.skip("Triton is not installed in this runtime")

    mirror = kernels._N_STREAMS_TL
    assert int(getattr(mirror, "value", mirror)) == kernels._N_STREAMS
