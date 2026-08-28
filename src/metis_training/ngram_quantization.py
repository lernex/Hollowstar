"""Isolated low-precision codecs and benchmarks for N-gram lookup tables.

The N-gram memory is a sparse gather, not a GEMM.  Transformer Engine's FP8
and NVFP4 linear recipes therefore do not exercise the storage path that
matters here: packed rows must be gathered first and dequantized without ever
materializing a persistent dense BF16 copy.  This module provides that exact
surface while deliberately leaving the N-gram projection, gates, and model
backbone untouched.

Three formats are supported:

``bf16``
    The reference table, stored exactly as BF16.
``fp8_e4m3``
    E4M3 payloads with one BF16 scale per row block.  A block size of zero uses
    one scale for the complete table and matches the layout used by some
    exported inference checkpoints; the default block size of 64 gives each
    Metis row its own scale.
``nvfp4``
    Packed E2M1 values with one E4M3 scale per 16 consecutive values and one
    FP32 global scale per table.  A 64-wide Metis row is exactly four blocks,
    so no scale crosses a row boundary and no padding enters the model value.

Quantized snapshots are inference/evaluation objects.  Sparse training keeps
the BF16 parameter and FP32 optimizer state authoritative; a later QAT lane can
use :func:`fake_quantize_rows` on touched rows without changing that contract.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


SUPPORTED_NGRAM_QUANT_FORMATS = ("bf16", "fp8_e4m3", "nvfp4")
_E4M3_MAX = 448.0
_NVFP4_MAX = 6.0
_NVFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _require_e4m3() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("This PyTorch build does not expose float8_e4m3fn.")
    return dtype


def _e4m3_bits(values: Tensor) -> Tensor:
    """Return finite E4M3 bit patterns, clamping before the cast.

    PyTorch emits the E4M3 NaN code for sufficiently out-of-range inputs rather
    than saturating them.  Explicit clipping is therefore a correctness rule,
    not an optional numerical tweak.
    """

    dtype = _require_e4m3()
    finite = torch.nan_to_num(
        values.float(), nan=0.0, posinf=_E4M3_MAX, neginf=-_E4M3_MAX
    ).clamp(-_E4M3_MAX, _E4M3_MAX)
    return finite.to(dtype).view(torch.uint8)


def _e4m3_values(bits: Tensor) -> Tensor:
    dtype = _require_e4m3()
    return bits.contiguous().view(dtype).float()


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


@dataclass(frozen=True)
class NGramQuantizationSpec:
    """The storage recipe for one table snapshot."""

    format: str
    block_size: int | None = None
    rounding: str = "nearest"
    seed: int = 16_062_026

    def normalized(self, *, row_width: int) -> "NGramQuantizationSpec":
        name = self.format.strip().lower()
        if name not in SUPPORTED_NGRAM_QUANT_FORMATS:
            raise ValueError(
                f"Unsupported N-gram quantization format {self.format!r}; "
                f"choose from {SUPPORTED_NGRAM_QUANT_FORMATS}."
            )
        if self.rounding not in {"nearest", "stochastic"}:
            raise ValueError("rounding must be nearest or stochastic")
        if name != "nvfp4" and self.rounding != "nearest":
            raise ValueError("stochastic rounding is implemented only for NVFP4")
        if name == "bf16":
            block_size = row_width
        elif name == "fp8_e4m3":
            block_size = 64 if self.block_size is None else int(self.block_size)
            if block_size < 0:
                raise ValueError("FP8 block_size must be zero or positive")
            if block_size and row_width % block_size:
                raise ValueError(
                    f"FP8 block_size {block_size} does not divide row width {row_width}"
                )
        else:
            block_size = 16 if self.block_size is None else int(self.block_size)
            if block_size != 16:
                raise ValueError("NVFP4 uses one E4M3 scale per 16 values")
            if row_width % block_size:
                raise ValueError(
                    f"NVFP4 block size 16 does not divide row width {row_width}"
                )
        return NGramQuantizationSpec(
            format=name,
            block_size=block_size,
            rounding=self.rounding,
            seed=int(self.seed),
        )


@dataclass
class QuantizedNGramTable:
    """A packed table that dequantizes only requested rows."""

    spec: NGramQuantizationSpec
    logical_shape: tuple[int, int]
    payload: Tensor
    scales: Tensor | None = None
    global_scale: Tensor | None = None

    @property
    def parameter_count(self) -> int:
        return math.prod(self.logical_shape)

    @property
    def storage_bytes(self) -> int:
        return (
            _tensor_bytes(self.payload)
            + _tensor_bytes(self.scales)
            + _tensor_bytes(self.global_scale)
        )

    @property
    def bits_per_parameter(self) -> float:
        return 8.0 * self.storage_bytes / max(self.parameter_count, 1)

    def storage_report(self) -> dict[str, Any]:
        return {
            "format": self.spec.format,
            "logical_shape": list(self.logical_shape),
            "logical_parameters": self.parameter_count,
            "payload_bytes": _tensor_bytes(self.payload),
            "scale_bytes": _tensor_bytes(self.scales),
            "global_scale_bytes": _tensor_bytes(self.global_scale),
            "storage_bytes": self.storage_bytes,
            "bits_per_parameter": self.bits_per_parameter,
            "block_size": self.spec.block_size,
            "rounding": self.spec.rounding,
        }

    def to(self, device: torch.device | str) -> "QuantizedNGramTable":
        return QuantizedNGramTable(
            spec=self.spec,
            logical_shape=self.logical_shape,
            payload=self.payload.to(device),
            scales=self.scales.to(device) if self.scales is not None else None,
            global_scale=(
                self.global_scale.to(device) if self.global_scale is not None else None
            ),
        )

    def lookup(
        self,
        row_ids: Tensor,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """Gather and dequantize only ``row_ids`` from the packed table."""

        if row_ids.dtype != torch.long:
            row_ids = row_ids.long()
        flat = row_ids.reshape(-1)
        if flat.numel() and (
            int(flat.min().item()) < 0 or int(flat.max().item()) >= self.logical_shape[0]
        ):
            raise IndexError("Quantized N-gram row is outside the table")
        selected = self.payload.index_select(0, flat)
        width = self.logical_shape[1]
        if self.spec.format == "bf16":
            values = selected
        elif self.spec.format == "fp8_e4m3":
            quantized = _e4m3_values(selected)
            assert self.scales is not None
            if int(self.spec.block_size or 0) == 0:
                values = quantized * self.scales.float().reshape(1, 1)
            else:
                block_size = int(self.spec.block_size)
                blocks = width // block_size
                scale = self.scales.index_select(0, flat).float()
                values = (
                    quantized.view(-1, blocks, block_size)
                    * scale.view(-1, blocks, 1)
                ).reshape(-1, width)
        else:
            assert self.scales is not None and self.global_scale is not None
            low = selected & 0x0F
            high = (selected >> 4) & 0x0F
            codes = torch.empty(
                selected.shape[0], selected.shape[1] * 2,
                dtype=torch.uint8,
                device=selected.device,
            )
            codes[:, 0::2] = low
            codes[:, 1::2] = high
            codes = codes[:, :width]
            magnitudes = codes & 0x07
            levels = torch.tensor(
                _NVFP4_LEVELS, device=codes.device, dtype=torch.float32
            )
            values = levels[magnitudes.long()]
            values = torch.where((codes & 0x08).bool(), -values, values)
            blocks = width // 16
            block_scales = (
                _e4m3_values(self.scales.index_select(0, flat))
                * self.global_scale.float().reshape(1, 1)
            )
            values = (
                values.view(-1, blocks, 16) * block_scales.view(-1, blocks, 1)
            ).reshape(-1, width)
        return values.to(output_dtype).view(*row_ids.shape, width)

    def dequantize(self, *, output_dtype: torch.dtype = torch.float32) -> Tensor:
        rows = torch.arange(self.logical_shape[0], device=self.payload.device)
        return self.lookup(rows, output_dtype=output_dtype)


def _nvfp4_codes(
    normalized: Tensor,
    *,
    rounding: str,
    generator: torch.Generator | None,
) -> Tensor:
    absolute = normalized.abs().clamp(0.0, _NVFP4_MAX)
    levels = torch.tensor(_NVFP4_LEVELS, device=absolute.device, dtype=torch.float32)
    if rounding == "nearest":
        indices = (absolute.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    else:
        upper = torch.searchsorted(levels, absolute, right=False).clamp(1, len(levels) - 1)
        lower = upper - 1
        lower_value = levels[lower]
        upper_value = levels[upper]
        probability = (absolute - lower_value) / (upper_value - lower_value).clamp_min(1e-12)
        draw = torch.rand(
            absolute.shape,
            device=absolute.device,
            dtype=torch.float32,
            generator=generator,
        )
        indices = torch.where(draw < probability, upper, lower)
        indices = torch.where(absolute == 0, torch.zeros_like(indices), indices)
    sign = (normalized < 0).to(torch.uint8) << 3
    return indices.to(torch.uint8) | sign


@torch.no_grad()
def quantize_ngram_table(
    weight: Tensor,
    spec: NGramQuantizationSpec | str,
    *,
    chunk_rows: int = 65_536,
) -> QuantizedNGramTable:
    """Quantize a rank-two table without retaining full-sized FP32 temporaries."""

    if weight.ndim != 2:
        raise ValueError(f"N-gram table must be rank two, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError("N-gram table must contain floating-point values")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    rows, width = map(int, weight.shape)
    if rows <= 0 or width <= 0:
        raise ValueError("N-gram table dimensions must be positive")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("N-gram table contains non-finite values")
    requested = (
        NGramQuantizationSpec(spec) if isinstance(spec, str) else spec
    ).normalized(row_width=width)
    device = weight.device

    if requested.format == "bf16":
        return QuantizedNGramTable(
            spec=requested,
            logical_shape=(rows, width),
            payload=weight.detach().to(torch.bfloat16).clone(),
        )

    if requested.format == "fp8_e4m3":
        payload = torch.empty((rows, width), dtype=torch.uint8, device=device)
        block_size = int(requested.block_size or 0)
        if block_size == 0:
            amax = weight.detach().float().abs().amax()
            scale_value = (amax / _E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
            scales = scale_value.to(torch.bfloat16).reshape(1)
        else:
            scales = torch.empty(
                (rows, width // block_size), dtype=torch.bfloat16, device=device
            )
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            values = weight[start:end].detach().float()
            if block_size == 0:
                denominator = scales.float().reshape(1, 1)
                payload[start:end] = _e4m3_bits(values / denominator)
            else:
                shaped = values.view(end - start, width // block_size, block_size)
                scale = (shaped.abs().amax(dim=-1) / _E4M3_MAX).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                stored_scale = scale.to(torch.bfloat16)
                scales[start:end] = stored_scale
                normalized = shaped / stored_scale.float().unsqueeze(-1)
                payload[start:end] = _e4m3_bits(normalized).reshape(end - start, width)
        return QuantizedNGramTable(
            spec=requested,
            logical_shape=(rows, width),
            payload=payload,
            scales=scales,
        )

    # NVFP4's global scale is table-wide; block scales remain row-local.  The
    # global reduction is the only full-table operation and does not allocate a
    # full-table temporary.
    amax = weight.detach().float().abs().amax()
    global_value = amax / (_E4M3_MAX * _NVFP4_MAX)
    if not bool(torch.isfinite(global_value).item()):
        raise ValueError("N-gram table contains non-finite values")
    if float(global_value.item()) == 0.0:
        global_value = torch.ones((), dtype=torch.float32, device=device)
    global_scale = global_value.float().reshape(1)
    payload = torch.empty((rows, (width + 1) // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, width // 16), dtype=torch.uint8, device=device)
    generator = None
    if requested.rounding == "stochastic":
        generator = torch.Generator(device=device)
        generator.manual_seed(requested.seed)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        values = weight[start:end].detach().float()
        shaped = values.view(end - start, width // 16, 16)
        raw_scale = shaped.abs().amax(dim=-1) / (_NVFP4_MAX * global_scale)
        scale_bits = _e4m3_bits(raw_scale)
        scales[start:end] = scale_bits
        dequant_scale = _e4m3_values(scale_bits) * global_scale
        denominator = torch.where(
            dequant_scale > 0, dequant_scale, torch.ones_like(dequant_scale)
        )
        normalized = shaped / denominator.unsqueeze(-1)
        codes = _nvfp4_codes(
            normalized,
            rounding=requested.rounding,
            generator=generator,
        ).reshape(end - start, width)
        if width % 2:
            codes = F.pad(codes, (0, 1))
        payload[start:end] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return QuantizedNGramTable(
        spec=requested,
        logical_shape=(rows, width),
        payload=payload,
        scales=scales,
        global_scale=global_scale,
    )


def fake_quantize_rows(
    rows: Tensor,
    spec: NGramQuantizationSpec | str,
) -> Tensor:
    """Touched-row QAT oracle with a straight-through gradient.

    This intentionally does not claim packed storage: the BF16 sparse parameter
    and optimizer state remain authoritative.  It answers the narrower training
    question, "does injecting this table-format noise change optimization?"
    """

    if rows.ndim < 2:
        raise ValueError("fake_quantize_rows expects [..., row_width]")
    shape = rows.shape
    flattened = rows.reshape(-1, shape[-1])
    snapshot = quantize_ngram_table(flattened.detach(), spec)
    dequantized = snapshot.dequantize(output_dtype=rows.dtype).view(shape)
    return rows + (dequantized - rows).detach()


def error_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    ref = reference.float().reshape(-1, reference.shape[-1])
    got = candidate.float().reshape_as(ref)
    difference = got - ref
    ref_norm = ref.norm().clamp_min(1e-12)
    row_ref_norm = ref.norm(dim=-1)
    row_got_norm = got.norm(dim=-1)
    nonzero = row_ref_norm > 0
    cosine = (
        F.cosine_similarity(ref[nonzero], got[nonzero], dim=-1)
        if bool(nonzero.any())
        else ref.new_ones(1)
    )
    relative_norm_bias = torch.where(
        nonzero,
        (row_got_norm - row_ref_norm) / row_ref_norm.clamp_min(1e-12),
        torch.zeros_like(row_ref_norm),
    )
    return {
        "relative_l2_error": float(difference.norm().div(ref_norm).item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "minimum_cosine_similarity": float(cosine.min().item()),
        "mean_relative_norm_bias": float(relative_norm_bias.mean().item()),
        "maximum_absolute_error": float(difference.abs().amax().item()),
        "finite": bool(torch.isfinite(got).all().item()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _median_time(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup_iterations):
        operation()
    _synchronize(device)
    samples: list[float] = []
    for _ in range(timed_iterations):
        started = time.perf_counter()
        operation()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


@torch.no_grad()
def benchmark_table_collection(
    tables: Mapping[str, Tensor],
    *,
    formats: Sequence[NGramQuantizationSpec | str] = SUPPORTED_NGRAM_QUANT_FORMATS,
    projection_weight: Tensor | None = None,
    lookup_rows: int = 8_192,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
    seed: int = 16_062_026,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    """Benchmark the complete concatenated retrieval surface.

    Each synthetic token chooses one row from every table, matching Metis's
    sixteen independent gathers.  The keys are uniformly distributed because
    the production hash is designed to distribute real N-grams uniformly; a
    caller that wants an empirical cache/reuse trace can pass those row IDs to
    :meth:`QuantizedNGramTable.lookup` directly.
    """

    if not tables:
        raise ValueError("At least one N-gram table is required")
    ordered = sorted(tables.items())
    widths = {int(weight.shape[1]) for _, weight in ordered if weight.ndim == 2}
    rank_two_count = sum(weight.ndim == 2 for _, weight in ordered)
    if len(widths) != 1 or rank_two_count != len(ordered):
        raise ValueError("Every N-gram table must be rank two with the same row width")
    if lookup_rows <= 0 or timed_iterations <= 0 or warmup_iterations < 0:
        raise ValueError("Benchmark iteration counts must be positive")
    device = ordered[0][1].device
    if any(weight.device != device for _, weight in ordered):
        raise ValueError("Every N-gram table must be on the same device")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    row_ids = {
        name: torch.randint(
            0,
            int(weight.shape[0]),
            (lookup_rows,),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        for name, weight in ordered
    }
    reference = torch.cat(
        [weight.index_select(0, row_ids[name]).to(torch.bfloat16) for name, weight in ordered],
        dim=-1,
    )
    projected_reference = None
    if projection_weight is not None:
        if projection_weight.shape[1] != reference.shape[1]:
            raise ValueError(
                f"Projection expects {projection_weight.shape[1]} inputs but "
                f"concatenated lookup has {reference.shape[1]}"
            )
        projected_reference = F.linear(reference.float(), projection_weight.float())

    results: list[dict[str, Any]] = []
    bf16_storage = sum(weight.numel() * 2 for _, weight in ordered)
    total_parameters = sum(weight.numel() for _, weight in ordered)
    for requested in formats:
        spec = NGramQuantizationSpec(requested) if isinstance(requested, str) else requested
        started = time.perf_counter()
        snapshots = {
            name: quantize_ngram_table(weight, spec, chunk_rows=chunk_rows)
            for name, weight in ordered
        }
        _synchronize(device)
        quantize_seconds = time.perf_counter() - started

        def retrieve() -> Tensor:
            return torch.cat(
                [
                    snapshots[name].lookup(row_ids[name], output_dtype=torch.bfloat16)
                    for name, _ in ordered
                ],
                dim=-1,
            )

        candidate = retrieve()
        median, samples = _median_time(
            retrieve,
            device=device,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
        )
        storage_bytes = sum(snapshot.storage_bytes for snapshot in snapshots.values())
        row: dict[str, Any] = {
            "format": next(iter(snapshots.values())).spec.format,
            "spec": {
                "block_size": next(iter(snapshots.values())).spec.block_size,
                "rounding": next(iter(snapshots.values())).spec.rounding,
            },
            "quantize_seconds": quantize_seconds,
            "storage_bytes": storage_bytes,
            "bits_per_parameter": 8.0 * storage_bytes / total_parameters,
            "compression_vs_bf16": bf16_storage / storage_bytes,
            "lookup_median_seconds": median,
            "lookup_minimum_seconds": min(samples),
            "lookup_maximum_seconds": max(samples),
            "tokens_per_second": lookup_rows / median,
            # One token gathers one row from every independent table.  Dividing
            # aggregate storage by aggregate rows would undercount this by the
            # number of equal-sized tables.
            "retrieved_bytes_per_token": sum(
                snapshot.storage_bytes / snapshot.logical_shape[0]
                for snapshot in snapshots.values()
            ),
            "retrieval_error": error_metrics(reference, candidate),
            "tables": {
                name: snapshot.storage_report() for name, snapshot in snapshots.items()
            },
        }
        if projection_weight is not None and projected_reference is not None:
            projected = F.linear(candidate.float(), projection_weight.float())
            row["projected_error"] = error_metrics(projected_reference, projected)
        results.append(row)

    return {
        "schema": "metis.ngram-quantization-benchmark/v1",
        "device": str(device),
        "table_count": len(ordered),
        "row_width": next(iter(widths)),
        "lookup_rows": lookup_rows,
        "logical_parameters": total_parameters,
        "bf16_storage_bytes": bf16_storage,
        "formats": results,
    }


def _chunked_checkpoint_ngram_tensors(
    root: Path,
    *,
    key_contains: str,
    checkpoint_owner: str | None,
) -> tuple[dict[str, Tensor], Tensor | None]:
    """Read one rank-local table owner from a sealed Metis-1.6 checkpoint."""

    from .checkpointing import CHECKPOINT_LAYOUT, CHECKPOINT_SCHEMA
    from .contracts import canonical_json_sha256, sha256_file

    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("layout") != CHECKPOINT_LAYOUT
    ):
        raise ValueError(f"Unsupported Metis checkpoint manifest in {root}")
    unsigned = {key: value for key, value in manifest.items() if key != "checkpoint_sha256"}
    if manifest.get("checkpoint_sha256") != canonical_json_sha256(unsigned):
        raise ValueError(f"Checkpoint manifest self-hash is invalid: {manifest_path}")

    inventory = manifest.get("state_inventory")
    artifacts = manifest.get("artifacts")
    if not isinstance(inventory, list) or not isinstance(artifacts, list):
        raise ValueError("Checkpoint manifest has no state inventory or artifact list")
    table_inventory = [
        row
        for row in inventory
        if isinstance(row, Mapping)
        and row.get("kind") == "model_tensor"
        and key_contains in str(row.get("name", ""))
        and str(row.get("name", "")).endswith("embedding.weight")
    ]
    owners = sorted({str(row.get("owner")) for row in table_inventory})
    if not owners:
        raise KeyError(f"Checkpoint contains no N-gram table owners: {root}")
    owner = owners[0] if checkpoint_owner is None else checkpoint_owner
    if owner not in owners:
        raise KeyError(
            f"Unknown N-gram checkpoint owner {owner!r}; choose from {owners}"
        )

    selected_inventory = [row for row in table_inventory if row.get("owner") == owner]
    projection_inventory = next(
        (
            row
            for row in inventory
            if isinstance(row, Mapping)
            and row.get("kind") == "model_tensor"
            and str(row.get("name", "")).endswith("ngram_memory.projection.weight")
        ),
        None,
    )
    requested_rows = list(selected_inventory)
    if projection_inventory is not None:
        requested_rows.append(projection_inventory)

    targets: dict[tuple[str, str], Tensor] = {}
    for row in requested_rows:
        name = str(row["name"])
        row_owner = str(row["owner"])
        dtype_name = str(row.get("dtype", ""))
        if not dtype_name.startswith("torch."):
            raise ValueError(f"Invalid checkpoint dtype for {name}: {dtype_name}")
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported checkpoint dtype for {name}: {dtype_name}")
        shape = tuple(int(value) for value in row.get("shape", []))
        if math.prod(shape) != int(row.get("numel", -1)):
            raise ValueError(f"Checkpoint inventory shape is invalid for {name}")
        key = (row_owner, name)
        if key in targets:
            raise ValueError(f"Checkpoint inventory duplicates {row_owner}:{name}")
        targets[key] = torch.empty(shape, dtype=dtype, device="cpu")

    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {
        key: [] for key in targets
    }
    artifact_owners = {owner}
    if projection_inventory is not None:
        artifact_owners.add(str(projection_inventory["owner"]))
    for record in sorted(
        (
            row
            for row in artifacts
            if isinstance(row, Mapping)
            and row.get("kind") == "state_shard"
            and str(row.get("owner")) in artifact_owners
        ),
        key=lambda row: str(row.get("path", "")),
    ):
        relative = Path(str(record.get("path", "")))
        unresolved_artifact = root / relative
        artifact = unresolved_artifact.resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Checkpoint artifact escapes its root: {relative}") from exc
        if (
            relative.is_absolute()
            or unresolved_artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(artifact) != record.get("sha256")
        ):
            raise ValueError(f"Checkpoint artifact failed integrity validation: {relative}")
        payload = torch.load(artifact, map_location="cpu", weights_only=False, mmap=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != CHECKPOINT_LAYOUT
            or payload.get("owner") != record.get("owner")
            or not isinstance(payload.get("items"), list)
            or len(payload["items"]) != int(record.get("item_count", -1))
        ):
            raise ValueError(f"Checkpoint state shard is invalid: {relative}")
        for item in payload["items"]:
            if not isinstance(item, Mapping) or item.get("kind") != "model_tensor":
                continue
            key = (str(payload["owner"]), str(item.get("name", "")))
            target = targets.get(key)
            if target is None:
                continue
            chunk = item.get("tensor")
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if (
                not isinstance(chunk, Tensor)
                or chunk.dtype != target.dtype
                or start < 0
                or end < start
                or end > target.numel()
                or chunk.numel() != end - start
            ):
                raise ValueError(f"Checkpoint tensor chunk is invalid for {key[1]}")
            target.view(-1)[start:end].copy_(chunk.reshape(-1))
            intervals[key].append((start, end))

    for key, target in targets.items():
        expected_start = 0
        for start, end in sorted(intervals[key]):
            if start != expected_start:
                raise ValueError(
                    f"Checkpoint tensor chunks have a gap or overlap for {key[1]}"
                )
            expected_start = end
        if expected_start != target.numel():
            raise ValueError(f"Checkpoint tensor is incomplete for {key[1]}")

    tables = {
        name: targets[(owner, name)]
        for name in sorted(str(row["name"]) for row in selected_inventory)
    }
    projection = None
    if projection_inventory is not None:
        projection = targets[
            (str(projection_inventory["owner"]), str(projection_inventory["name"]))
        ]
    return tables, projection


def checkpoint_ngram_tensors(
    checkpoint: Path,
    *,
    key_contains: str = "ngram_memory.tables.",
    checkpoint_owner: str | None = None,
) -> tuple[dict[str, Tensor], Tensor | None]:
    """Load N-gram tables and the fusion projection from a trusted checkpoint.

    Production Metis tensor-chunk checkpoints hold one row-sharded table set
    per ``tables-ep-*`` owner.  By default this reads the lexicographically
    first owner as a representative rank-local benchmark; pass
    ``checkpoint_owner`` to select a different shard explicitly.
    """

    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        if (path / "MANIFEST.json").is_file():
            return _chunked_checkpoint_ngram_tensors(
                path,
                key_contains=key_contains,
                checkpoint_owner=checkpoint_owner,
            )
        path = path / "state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, Tensor):
        return {path.stem: payload}, None
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload {type(payload).__name__}")
    state: Mapping[str, Any] = payload
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    tables = {
        name: value
        for name, value in state.items()
        if isinstance(value, Tensor)
        and key_contains in name
        and name.endswith("embedding.weight")
    }
    if not tables:
        raise KeyError(
            f"No keys containing {key_contains!r} and ending in embedding.weight "
            f"were found in {path}"
        )
    projection = next(
        (
            value
            for name, value in state.items()
            if isinstance(value, Tensor)
            and name.endswith("ngram_memory.projection.weight")
        ),
        None,
    )
    return tables, projection


@torch.no_grad()
def compare_model_ngram_losses(
    model: Any,
    batches: Iterable[Any],
    *,
    formats: Sequence[NGramQuantizationSpec | str] = ("fp8_e4m3", "nvfp4"),
    forward: Callable[[Any, Any], Any] | None = None,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    """Compare model loss while changing only the table storage codec.

    ``model`` must expose ``ngram_memory.quantized_table_lookup``.  Batches are
    materialized once so every format sees byte-identical inputs in identical
    order.  The default forward accepts mapping batches and reads ``output.loss``;
    callers with a structured batch can supply a small adapter.
    """

    materialized = list(batches)
    if not materialized:
        raise ValueError("At least one batch is required")
    tables = getattr(getattr(model, "ngram_memory", None), "tables", None)
    if tables is None or not hasattr(tables, "values") or len(tables) == 0:
        raise TypeError("Model has no N-gram table collection")
    if any(table.embedding.weight.dtype != torch.bfloat16 for table in tables.values()):
        raise ValueError("Model-loss parity requires BF16 N-gram reference tables")
    if forward is None:
        def forward(candidate: Any, batch: Any) -> Any:
            if not isinstance(batch, Mapping):
                raise TypeError("Default model-loss forward requires mapping batches")
            return candidate(**batch)

    def losses() -> list[float]:
        values: list[float] = []
        for batch in materialized:
            output = forward(model, batch)
            loss = getattr(output, "loss", output)
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise TypeError("Model-loss forward must return a scalar tensor or output.loss")
            values.append(float(loss.detach().float().item()))
        return values

    baseline = losses()
    baseline_mean = statistics.fmean(baseline)
    rows: list[dict[str, Any]] = []
    for requested in formats:
        spec = NGramQuantizationSpec(requested) if isinstance(requested, str) else requested
        with model.ngram_memory.quantized_table_lookup(spec, chunk_rows=chunk_rows) as storage:
            candidate = losses()
        candidate_mean = statistics.fmean(candidate)
        paired_deltas = [got - reference for got, reference in zip(candidate, baseline)]
        delta_mean = statistics.fmean(paired_deltas)
        delta_standard_error = (
            statistics.stdev(paired_deltas) / math.sqrt(len(paired_deltas))
            if len(paired_deltas) > 1
            else 0.0
        )
        rows.append(
            {
                "format": storage["format"],
                "losses": candidate,
                "mean_loss": candidate_mean,
                "paired_loss_deltas": paired_deltas,
                "mean_loss_delta": delta_mean,
                "loss_delta_standard_error": delta_standard_error,
                "loss_delta_95pct_normal_interval": [
                    delta_mean - 1.96 * delta_standard_error,
                    delta_mean + 1.96 * delta_standard_error,
                ],
                "maximum_absolute_batch_loss_delta": max(
                    abs(delta) for delta in paired_deltas
                ),
                "perplexity_ratio_vs_bf16": math.exp(
                    max(min(delta_mean, 80.0), -80.0)
                ),
                "loss_relative_error_vs_bf16": abs(delta_mean)
                / max(abs(baseline_mean), 1e-12),
                "storage": storage,
            }
        )
    return {
        "schema": "metis.ngram-model-loss-parity/v1",
        "batches": len(materialized),
        "bf16_losses": baseline,
        "bf16_mean_loss": baseline_mean,
        "formats": rows,
    }


__all__ = [
    "NGramQuantizationSpec",
    "QuantizedNGramTable",
    "SUPPORTED_NGRAM_QUANT_FORMATS",
    "benchmark_table_collection",
    "checkpoint_ngram_tensors",
    "compare_model_ngram_losses",
    "error_metrics",
    "fake_quantize_rows",
    "quantize_ngram_table",
]
