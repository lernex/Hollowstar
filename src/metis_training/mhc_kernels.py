from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .model_config import Metis16Config

try:  # Triton is installed by the sealed Portage compute runtime.
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on the runtime image.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Exception | None = exc
else:
    _TRITON_IMPORT_ERROR = None


MHC_CANARY_SCHEMA = "metis.mhc-fused-canary/v1"
_N_STREAMS = 4
_BLOCK_D = 256


def _canonical_sha256(payload: Mapping[str, Any], *, omit: Sequence[str] = ()) -> str:
    cooked = {key: value for key, value in payload.items() if key not in set(omit)}
    encoded = json.dumps(
        cooked,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def triton_rocm_backend_status(
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    resolved = torch.device(device) if device is not None else None
    cuda_visible = bool(torch.cuda.is_available())
    rocm = getattr(torch.version, "hip", None)
    if resolved is None and cuda_visible:
        resolved = torch.device("cuda", torch.cuda.current_device())
    is_cuda = resolved is not None and resolved.type == "cuda"
    available = bool(triton is not None and cuda_visible and rocm and is_cuda)
    reason: str | None = None
    if triton is None:
        reason = (
            "Triton import failed"
            + (
                f": {type(_TRITON_IMPORT_ERROR).__name__}: {_TRITON_IMPORT_ERROR}"
                if _TRITON_IMPORT_ERROR is not None
                else ""
            )
        )
    elif not cuda_visible:
        reason = "PyTorch reports no visible CUDA/ROCm device"
    elif not rocm:
        reason = "PyTorch is not a ROCm build"
    elif not is_cuda:
        reason = f"mHC fused backend requires a CUDA/ROCm device, got {resolved}"
    return {
        "available": available,
        "backend": "triton_rocm" if available else "unavailable",
        "device": str(resolved) if resolved is not None else None,
        "torch": torch.__version__,
        "rocm": rocm,
        "triton": getattr(triton, "__version__", None),
        "reason": reason,
    }


def require_mhc_backend(
    *,
    backend: str,
    family: str,
    device: torch.device | str | None,
) -> None:
    if backend == "torch_reference":
        # ``ablation`` joins ``tiny`` here so the MoRE ladder can be smoke-tested
        # and dry-run on CPU.  Its production launches declare
        # ``fused_required`` in the manifest, exactly like Praxis and Logos, so
        # a real campaign run still cannot silently fall back to the reference.
        if family not in {"tiny", "ablation"}:
            raise RuntimeError(
                "The torch mHC reference backend is restricted to tiny and "
                f"ablation research runs; {family} requires the fused "
                "Triton/ROCm backend."
            )
        return
    if backend != "fused_required":
        raise RuntimeError(f"Unsupported mHC backend: {backend!r}")
    status = triton_rocm_backend_status(device)
    if not status["available"]:
        raise RuntimeError(
            "Production mHC requires the fused Triton/ROCm backend: "
            f"{status['reason']}"
        )


def mhc_read_mix_reference(
    streams: Tensor,
    matrix: Tensor,
    read_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Unfused correctness oracle, intentionally restricted by its caller."""

    if streams.shape[-2] != _N_STREAMS:
        raise ValueError("mHC read/mix requires exactly four streams.")
    if matrix.shape != (_N_STREAMS, _N_STREAMS):
        raise ValueError("mHC mix matrix must have shape [4, 4].")
    if read_weights.shape != (_N_STREAMS,):
        raise ValueError("mHC read weights must have shape [4].")
    mixed = torch.einsum(
        "oi,...id->...od",
        matrix.to(streams.dtype),
        streams,
    )
    source = torch.einsum(
        "i,...id->...d",
        read_weights.to(streams.dtype),
        streams,
    )
    return source, mixed


def mhc_masked_write_reference(
    mixed: Tensor,
    write_weights: Tensor,
    update: Tensor,
    original_streams: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Unfused masked-write oracle, intentionally restricted by its caller."""

    if mixed.shape != original_streams.shape:
        raise ValueError("mHC mixed and original streams must have identical shapes.")
    if mixed.shape[-2] != _N_STREAMS:
        raise ValueError("mHC masked write requires exactly four streams.")
    if update.shape != mixed.shape[:-2] + (mixed.shape[-1],):
        raise ValueError("mHC update shape does not match the stream prefix/width.")
    if active_mask.shape != mixed.shape[:-2]:
        raise ValueError("mHC active mask does not match the stream prefix.")
    if write_weights.shape != (_N_STREAMS,):
        raise ValueError("mHC write weights must have shape [4].")
    updated = mixed + torch.einsum(
        "i,...d->...id",
        write_weights.to(update.dtype),
        update,
    )
    return torch.where(active_mask[..., None, None], updated, original_streams)


if triton is not None:

    # Triton 3.x refuses to close over a plain module global inside @jit: a
    # kernel may only read globals that are tl.constexpr. _N_STREAMS stays an
    # int because the host-side shape checks and stride arithmetic below index
    # tuples with it; this mirrors it into the form the kernels can read.
    # test_mhc_stream_constant_mirrors_agree pins the two together.
    _N_STREAMS_TL = tl.constexpr(_N_STREAMS)

    @triton.jit
    def _mhc_read_mix_forward_kernel(
        streams_ptr,
        matrix_ptr,
        read_ptr,
        source_ptr,
        mixed_ptr,
        hidden_size,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < hidden_size
        base = token * _N_STREAMS_TL * hidden_size + offsets
        x0 = tl.load(streams_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            streams_ptr + base + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            streams_ptr + base + 2 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        x3 = tl.load(
            streams_ptr + base + 3 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)

        r0 = tl.load(read_ptr).to(tl.float32)
        r1 = tl.load(read_ptr + 1).to(tl.float32)
        r2 = tl.load(read_ptr + 2).to(tl.float32)
        r3 = tl.load(read_ptr + 3).to(tl.float32)
        source = r0 * x0 + r1 * x1 + r2 * x2 + r3 * x3
        tl.store(
            source_ptr + token * hidden_size + offsets,
            source,
            mask=mask,
        )

        m00 = tl.load(matrix_ptr).to(tl.float32)
        m01 = tl.load(matrix_ptr + 1).to(tl.float32)
        m02 = tl.load(matrix_ptr + 2).to(tl.float32)
        m03 = tl.load(matrix_ptr + 3).to(tl.float32)
        m10 = tl.load(matrix_ptr + 4).to(tl.float32)
        m11 = tl.load(matrix_ptr + 5).to(tl.float32)
        m12 = tl.load(matrix_ptr + 6).to(tl.float32)
        m13 = tl.load(matrix_ptr + 7).to(tl.float32)
        m20 = tl.load(matrix_ptr + 8).to(tl.float32)
        m21 = tl.load(matrix_ptr + 9).to(tl.float32)
        m22 = tl.load(matrix_ptr + 10).to(tl.float32)
        m23 = tl.load(matrix_ptr + 11).to(tl.float32)
        m30 = tl.load(matrix_ptr + 12).to(tl.float32)
        m31 = tl.load(matrix_ptr + 13).to(tl.float32)
        m32 = tl.load(matrix_ptr + 14).to(tl.float32)
        m33 = tl.load(matrix_ptr + 15).to(tl.float32)
        y0 = m00 * x0 + m01 * x1 + m02 * x2 + m03 * x3
        y1 = m10 * x0 + m11 * x1 + m12 * x2 + m13 * x3
        y2 = m20 * x0 + m21 * x1 + m22 * x2 + m23 * x3
        y3 = m30 * x0 + m31 * x1 + m32 * x2 + m33 * x3
        mixed_base = token * _N_STREAMS_TL * hidden_size + offsets
        tl.store(mixed_ptr + mixed_base, y0, mask=mask)
        tl.store(mixed_ptr + mixed_base + hidden_size, y1, mask=mask)
        tl.store(mixed_ptr + mixed_base + 2 * hidden_size, y2, mask=mask)
        tl.store(mixed_ptr + mixed_base + 3 * hidden_size, y3, mask=mask)


    @triton.jit
    def _mhc_read_mix_backward_kernel(
        streams_ptr,
        matrix_ptr,
        read_ptr,
        grad_source_ptr,
        grad_mixed_ptr,
        grad_streams_ptr,
        grad_matrix_ptr,
        grad_read_ptr,
        hidden_size,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < hidden_size
        stream_base = token * _N_STREAMS_TL * hidden_size + offsets
        source_base = token * hidden_size + offsets
        x0 = tl.load(streams_ptr + stream_base, mask=mask, other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            streams_ptr + stream_base + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            streams_ptr + stream_base + 2 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        x3 = tl.load(
            streams_ptr + stream_base + 3 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        gs = tl.load(grad_source_ptr + source_base, mask=mask, other=0.0).to(
            tl.float32
        )
        gm0 = tl.load(grad_mixed_ptr + stream_base, mask=mask, other=0.0).to(
            tl.float32
        )
        gm1 = tl.load(
            grad_mixed_ptr + stream_base + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        gm2 = tl.load(
            grad_mixed_ptr + stream_base + 2 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gm3 = tl.load(
            grad_mixed_ptr + stream_base + 3 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        r0 = tl.load(read_ptr).to(tl.float32)
        r1 = tl.load(read_ptr + 1).to(tl.float32)
        r2 = tl.load(read_ptr + 2).to(tl.float32)
        r3 = tl.load(read_ptr + 3).to(tl.float32)
        m00 = tl.load(matrix_ptr).to(tl.float32)
        m01 = tl.load(matrix_ptr + 1).to(tl.float32)
        m02 = tl.load(matrix_ptr + 2).to(tl.float32)
        m03 = tl.load(matrix_ptr + 3).to(tl.float32)
        m10 = tl.load(matrix_ptr + 4).to(tl.float32)
        m11 = tl.load(matrix_ptr + 5).to(tl.float32)
        m12 = tl.load(matrix_ptr + 6).to(tl.float32)
        m13 = tl.load(matrix_ptr + 7).to(tl.float32)
        m20 = tl.load(matrix_ptr + 8).to(tl.float32)
        m21 = tl.load(matrix_ptr + 9).to(tl.float32)
        m22 = tl.load(matrix_ptr + 10).to(tl.float32)
        m23 = tl.load(matrix_ptr + 11).to(tl.float32)
        m30 = tl.load(matrix_ptr + 12).to(tl.float32)
        m31 = tl.load(matrix_ptr + 13).to(tl.float32)
        m32 = tl.load(matrix_ptr + 14).to(tl.float32)
        m33 = tl.load(matrix_ptr + 15).to(tl.float32)

        gx0 = r0 * gs + m00 * gm0 + m10 * gm1 + m20 * gm2 + m30 * gm3
        gx1 = r1 * gs + m01 * gm0 + m11 * gm1 + m21 * gm2 + m31 * gm3
        gx2 = r2 * gs + m02 * gm0 + m12 * gm1 + m22 * gm2 + m32 * gm3
        gx3 = r3 * gs + m03 * gm0 + m13 * gm1 + m23 * gm2 + m33 * gm3
        tl.store(grad_streams_ptr + stream_base, gx0, mask=mask)
        tl.store(
            grad_streams_ptr + stream_base + hidden_size, gx1, mask=mask
        )
        tl.store(
            grad_streams_ptr + stream_base + 2 * hidden_size, gx2, mask=mask
        )
        tl.store(
            grad_streams_ptr + stream_base + 3 * hidden_size, gx3, mask=mask
        )

        tl.atomic_add(grad_read_ptr, tl.sum(gs * x0, axis=0))
        tl.atomic_add(grad_read_ptr + 1, tl.sum(gs * x1, axis=0))
        tl.atomic_add(grad_read_ptr + 2, tl.sum(gs * x2, axis=0))
        tl.atomic_add(grad_read_ptr + 3, tl.sum(gs * x3, axis=0))
        tl.atomic_add(grad_matrix_ptr, tl.sum(gm0 * x0, axis=0))
        tl.atomic_add(grad_matrix_ptr + 1, tl.sum(gm0 * x1, axis=0))
        tl.atomic_add(grad_matrix_ptr + 2, tl.sum(gm0 * x2, axis=0))
        tl.atomic_add(grad_matrix_ptr + 3, tl.sum(gm0 * x3, axis=0))
        tl.atomic_add(grad_matrix_ptr + 4, tl.sum(gm1 * x0, axis=0))
        tl.atomic_add(grad_matrix_ptr + 5, tl.sum(gm1 * x1, axis=0))
        tl.atomic_add(grad_matrix_ptr + 6, tl.sum(gm1 * x2, axis=0))
        tl.atomic_add(grad_matrix_ptr + 7, tl.sum(gm1 * x3, axis=0))
        tl.atomic_add(grad_matrix_ptr + 8, tl.sum(gm2 * x0, axis=0))
        tl.atomic_add(grad_matrix_ptr + 9, tl.sum(gm2 * x1, axis=0))
        tl.atomic_add(grad_matrix_ptr + 10, tl.sum(gm2 * x2, axis=0))
        tl.atomic_add(grad_matrix_ptr + 11, tl.sum(gm2 * x3, axis=0))
        tl.atomic_add(grad_matrix_ptr + 12, tl.sum(gm3 * x0, axis=0))
        tl.atomic_add(grad_matrix_ptr + 13, tl.sum(gm3 * x1, axis=0))
        tl.atomic_add(grad_matrix_ptr + 14, tl.sum(gm3 * x2, axis=0))
        tl.atomic_add(grad_matrix_ptr + 15, tl.sum(gm3 * x3, axis=0))


    @triton.jit
    def _mhc_masked_write_forward_kernel(
        mixed_ptr,
        write_ptr,
        update_ptr,
        original_ptr,
        active_ptr,
        output_ptr,
        hidden_size,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < hidden_size
        stream_base = token * _N_STREAMS_TL * hidden_size + offsets
        update_base = token * hidden_size + offsets
        active = tl.load(active_ptr + token)
        update = tl.load(update_ptr + update_base, mask=mask, other=0.0).to(
            tl.float32
        )
        w0 = tl.load(write_ptr).to(tl.float32)
        w1 = tl.load(write_ptr + 1).to(tl.float32)
        w2 = tl.load(write_ptr + 2).to(tl.float32)
        w3 = tl.load(write_ptr + 3).to(tl.float32)
        mixed0 = tl.load(mixed_ptr + stream_base, mask=mask, other=0.0).to(
            tl.float32
        )
        mixed1 = tl.load(
            mixed_ptr + stream_base + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        mixed2 = tl.load(
            mixed_ptr + stream_base + 2 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        mixed3 = tl.load(
            mixed_ptr + stream_base + 3 * hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        original0 = tl.load(
            original_ptr + stream_base, mask=mask, other=0.0
        ).to(tl.float32)
        original1 = tl.load(
            original_ptr + stream_base + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        original2 = tl.load(
            original_ptr + stream_base + 2 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        original3 = tl.load(
            original_ptr + stream_base + 3 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr + stream_base,
            tl.where(active, mixed0 + w0 * update, original0),
            mask=mask,
        )
        tl.store(
            output_ptr + stream_base + hidden_size,
            tl.where(active, mixed1 + w1 * update, original1),
            mask=mask,
        )
        tl.store(
            output_ptr + stream_base + 2 * hidden_size,
            tl.where(active, mixed2 + w2 * update, original2),
            mask=mask,
        )
        tl.store(
            output_ptr + stream_base + 3 * hidden_size,
            tl.where(active, mixed3 + w3 * update, original3),
            mask=mask,
        )


    @triton.jit
    def _mhc_masked_write_backward_kernel(
        write_ptr,
        update_ptr,
        active_ptr,
        grad_output_ptr,
        grad_mixed_ptr,
        grad_write_ptr,
        grad_update_ptr,
        grad_original_ptr,
        hidden_size,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < hidden_size
        stream_base = token * _N_STREAMS_TL * hidden_size + offsets
        update_base = token * hidden_size + offsets
        active = tl.load(active_ptr + token)
        active_f = active.to(tl.float32)
        inactive_f = 1.0 - active_f
        update = tl.load(update_ptr + update_base, mask=mask, other=0.0).to(
            tl.float32
        )
        g0 = tl.load(
            grad_output_ptr + stream_base, mask=mask, other=0.0
        ).to(tl.float32)
        g1 = tl.load(
            grad_output_ptr + stream_base + hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        g2 = tl.load(
            grad_output_ptr + stream_base + 2 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        g3 = tl.load(
            grad_output_ptr + stream_base + 3 * hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w0 = tl.load(write_ptr).to(tl.float32)
        w1 = tl.load(write_ptr + 1).to(tl.float32)
        w2 = tl.load(write_ptr + 2).to(tl.float32)
        w3 = tl.load(write_ptr + 3).to(tl.float32)
        tl.store(grad_mixed_ptr + stream_base, active_f * g0, mask=mask)
        tl.store(
            grad_mixed_ptr + stream_base + hidden_size,
            active_f * g1,
            mask=mask,
        )
        tl.store(
            grad_mixed_ptr + stream_base + 2 * hidden_size,
            active_f * g2,
            mask=mask,
        )
        tl.store(
            grad_mixed_ptr + stream_base + 3 * hidden_size,
            active_f * g3,
            mask=mask,
        )
        tl.store(grad_original_ptr + stream_base, inactive_f * g0, mask=mask)
        tl.store(
            grad_original_ptr + stream_base + hidden_size,
            inactive_f * g1,
            mask=mask,
        )
        tl.store(
            grad_original_ptr + stream_base + 2 * hidden_size,
            inactive_f * g2,
            mask=mask,
        )
        tl.store(
            grad_original_ptr + stream_base + 3 * hidden_size,
            inactive_f * g3,
            mask=mask,
        )
        grad_update = active_f * (w0 * g0 + w1 * g1 + w2 * g2 + w3 * g3)
        tl.store(grad_update_ptr + update_base, grad_update, mask=mask)
        tl.atomic_add(
            grad_write_ptr,
            tl.sum(active_f * g0 * update, axis=0),
        )
        tl.atomic_add(
            grad_write_ptr + 1,
            tl.sum(active_f * g1 * update, axis=0),
        )
        tl.atomic_add(
            grad_write_ptr + 2,
            tl.sum(active_f * g2 * update, axis=0),
        )
        tl.atomic_add(
            grad_write_ptr + 3,
            tl.sum(active_f * g3 * update, axis=0),
        )


class _MHCReadMixFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        streams: Tensor,
        matrix: Tensor,
        read_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if triton is None:  # pragma: no cover - guarded before invocation.
            raise RuntimeError("Triton is unavailable")
        streams_c = streams.contiguous()
        matrix_c = matrix.contiguous()
        read_c = read_weights.contiguous()
        hidden_size = int(streams_c.shape[-1])
        token_count = int(streams_c.numel() // (_N_STREAMS * hidden_size))
        source = torch.empty(
            (token_count, hidden_size),
            device=streams.device,
            dtype=streams.dtype,
        )
        mixed = torch.empty_like(streams_c).view(
            token_count, _N_STREAMS, hidden_size
        )
        grid = (token_count, triton.cdiv(hidden_size, _BLOCK_D))
        _mhc_read_mix_forward_kernel[grid](
            streams_c,
            matrix_c,
            read_c,
            source,
            mixed,
            hidden_size,
            BLOCK_D=_BLOCK_D,
        )
        ctx.save_for_backward(streams_c, matrix_c, read_c)
        ctx.original_shape = tuple(streams.shape)
        prefix = tuple(streams.shape[:-2])
        return source.view(*prefix, hidden_size), mixed.view_as(streams_c)

    @staticmethod
    def backward(
        ctx: Any,
        grad_source: Tensor | None,
        grad_mixed: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        streams, matrix, read_weights = ctx.saved_tensors
        hidden_size = int(streams.shape[-1])
        token_count = int(streams.numel() // (_N_STREAMS * hidden_size))
        if grad_source is None:
            grad_source = torch.zeros(
                (token_count, hidden_size),
                device=streams.device,
                dtype=streams.dtype,
            )
        if grad_mixed is None:
            grad_mixed = torch.zeros_like(streams)
        grad_streams = torch.empty_like(streams)
        grad_matrix_fp32 = torch.zeros(
            (_N_STREAMS, _N_STREAMS),
            device=streams.device,
            dtype=torch.float32,
        )
        grad_read_fp32 = torch.zeros(
            (_N_STREAMS,),
            device=streams.device,
            dtype=torch.float32,
        )
        grid = (token_count, triton.cdiv(hidden_size, _BLOCK_D))
        _mhc_read_mix_backward_kernel[grid](
            streams,
            matrix,
            read_weights,
            grad_source.contiguous(),
            grad_mixed.contiguous(),
            grad_streams,
            grad_matrix_fp32,
            grad_read_fp32,
            hidden_size,
            BLOCK_D=_BLOCK_D,
        )
        return (
            grad_streams.view(ctx.original_shape),
            grad_matrix_fp32.to(matrix.dtype),
            grad_read_fp32.to(read_weights.dtype),
        )


class _MHCMaskedWriteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        mixed: Tensor,
        write_weights: Tensor,
        update: Tensor,
        original_streams: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        if triton is None:  # pragma: no cover - guarded before invocation.
            raise RuntimeError("Triton is unavailable")
        mixed_c = mixed.contiguous()
        write_c = write_weights.contiguous()
        update_c = update.contiguous()
        original_c = original_streams.contiguous()
        active_c = active_mask.to(device=mixed.device, dtype=torch.bool).contiguous()
        hidden_size = int(mixed.shape[-1])
        token_count = int(mixed.numel() // (_N_STREAMS * hidden_size))
        output = torch.empty_like(mixed_c)
        grid = (token_count, triton.cdiv(hidden_size, _BLOCK_D))
        _mhc_masked_write_forward_kernel[grid](
            mixed_c,
            write_c,
            update_c,
            original_c,
            active_c,
            output,
            hidden_size,
            BLOCK_D=_BLOCK_D,
        )
        ctx.save_for_backward(write_c, update_c, active_c)
        ctx.original_shape = tuple(mixed.shape)
        return output.view_as(mixed)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        write_weights, update, active_mask = ctx.saved_tensors
        hidden_size = int(update.shape[-1])
        token_count = int(update.numel() // hidden_size)
        grad_output_c = grad_output.contiguous()
        grad_mixed = torch.empty_like(grad_output_c)
        grad_write_fp32 = torch.zeros(
            (_N_STREAMS,),
            device=update.device,
            dtype=torch.float32,
        )
        grad_update = torch.empty_like(update)
        grad_original = torch.empty_like(grad_output_c)
        grid = (token_count, triton.cdiv(hidden_size, _BLOCK_D))
        _mhc_masked_write_backward_kernel[grid](
            write_weights,
            update,
            active_mask,
            grad_output_c,
            grad_mixed,
            grad_write_fp32,
            grad_update,
            grad_original,
            hidden_size,
            BLOCK_D=_BLOCK_D,
        )
        return (
            grad_mixed.view(ctx.original_shape),
            grad_write_fp32.to(write_weights.dtype),
            grad_update.view_as(update),
            grad_original.view(ctx.original_shape),
            None,
        )


def _validate_streams_and_weights(streams: Tensor, weights: Tensor) -> None:
    if streams.shape[-2] != _N_STREAMS:
        raise ValueError("mHC fused kernels require exactly four streams.")
    if weights.shape != (_N_STREAMS,):
        raise ValueError("mHC read/write weights must have shape [4].")
    if not streams.is_contiguous():
        # Contiguity is repaired by the autograd function, but a non-strided
        # view with overlapping storage is not an accepted production input.
        if any(stride <= 0 for stride in streams.stride()):
            raise ValueError("mHC streams may not use overlapping/negative strides.")
    if weights.device != streams.device:
        raise ValueError("mHC operands must be colocated on one device.")


def mhc_read_mix(
    streams: Tensor,
    matrix: Tensor,
    read_weights: Tensor,
    *,
    backend: str,
    family: str,
) -> tuple[Tensor, Tensor]:
    _validate_streams_and_weights(streams, read_weights)
    if matrix.shape != (_N_STREAMS, _N_STREAMS):
        raise ValueError("mHC mix matrix must have shape [4, 4].")
    if matrix.device != streams.device:
        raise ValueError("mHC operands must be colocated on one device.")
    require_mhc_backend(backend=backend, family=family, device=streams.device)
    if backend == "torch_reference":
        return mhc_read_mix_reference(streams, matrix, read_weights)
    return _MHCReadMixFunction.apply(streams, matrix, read_weights)


def mhc_masked_write(
    mixed: Tensor,
    write_weights: Tensor,
    update: Tensor,
    original_streams: Tensor,
    active_mask: Tensor,
    *,
    backend: str,
    family: str,
) -> Tensor:
    _validate_streams_and_weights(mixed, write_weights)
    if mixed.shape != original_streams.shape:
        raise ValueError("mHC mixed and original streams must have identical shapes.")
    if update.shape != mixed.shape[:-2] + (mixed.shape[-1],):
        raise ValueError("mHC update shape does not match the stream prefix/width.")
    if active_mask.shape != mixed.shape[:-2]:
        raise ValueError("mHC active mask does not match the stream prefix.")
    if update.device != mixed.device or original_streams.device != mixed.device:
        raise ValueError("mHC write operands must be colocated on one device.")
    require_mhc_backend(backend=backend, family=family, device=mixed.device)
    if backend == "torch_reference":
        return mhc_masked_write_reference(
            mixed,
            write_weights,
            update,
            original_streams,
            active_mask,
        )
    return _MHCMaskedWriteFunction.apply(
        mixed,
        write_weights,
        update,
        original_streams,
        active_mask,
    )


def _relative_l2(candidate: Tensor, reference: Tensor) -> float:
    candidate_fp32 = candidate.detach().float()
    reference_fp32 = reference.detach().float()
    denominator = reference_fp32.norm().clamp_min(1.0e-12)
    return float(((candidate_fp32 - reference_fp32).norm() / denominator).item())


def _canary_geometry(config: "Metis16Config") -> dict[str, Any]:
    return {
        "family": config.family,
        "d_model": config.d_model,
        "n_streams": config.n_streams,
        "mhc_backend": config.mhc_backend,
        "mhc_sinkhorn_iterations": config.mhc_sinkhorn_iterations,
    }


def _run_one_canary(
    config: "Metis16Config",
    *,
    device: torch.device,
    token_count: int,
    maximum_forward_relative_error: float,
    maximum_backward_relative_error: float,
    minimum_speedup: float,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(16_062_026 + config.d_model)
    streams_seed = torch.randn(
        (token_count, _N_STREAMS, config.d_model),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.25
    update_seed = torch.randn(
        (token_count, config.d_model),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.25
    matrix_logits = torch.randn(
        (_N_STREAMS, _N_STREAMS),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    for _ in range(config.mhc_sinkhorn_iterations):
        matrix_logits = matrix_logits - torch.logsumexp(
            matrix_logits, dim=-1, keepdim=True
        )
        matrix_logits = matrix_logits - torch.logsumexp(
            matrix_logits, dim=-2, keepdim=True
        )
    matrix_seed = matrix_logits.exp()
    read_seed = torch.softmax(
        torch.randn(
            (_N_STREAMS,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    )
    write_seed = torch.softmax(
        torch.randn(
            (_N_STREAMS,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    )
    active_mask = (torch.arange(token_count, device=device) % 3) != 0
    source_probe = torch.randn(
        (token_count, config.d_model),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    output_probe = torch.randn(
        (token_count, _N_STREAMS, config.d_model),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    def execute(*, fused: bool) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        streams = streams_seed.detach().clone().requires_grad_(True)
        matrix = matrix_seed.detach().clone().requires_grad_(True)
        read = read_seed.detach().clone().requires_grad_(True)
        write = write_seed.detach().clone().requires_grad_(True)
        update = update_seed.detach().clone().requires_grad_(True)
        if fused:
            source, mixed = mhc_read_mix(
                streams,
                matrix,
                read,
                backend="fused_required",
                family=config.family,
            )
            output = mhc_masked_write(
                mixed,
                write,
                update,
                streams,
                active_mask,
                backend="fused_required",
                family=config.family,
            )
        else:
            source, mixed = mhc_read_mix_reference(streams, matrix, read)
            output = mhc_masked_write_reference(
                mixed,
                write,
                update,
                streams,
                active_mask,
            )
        objective = (
            (source.float() * source_probe).sum() / source.numel()
            + (output.float() * output_probe).sum() / output.numel()
        )
        gradients = torch.autograd.grad(
            objective,
            (streams, matrix, read, write, update),
        )
        return source.detach(), output.detach(), gradients

    reference_source, reference_output, reference_gradients = execute(fused=False)
    fused_source, fused_output, fused_gradients = execute(fused=True)
    torch.cuda.synchronize(device)
    forward_errors = {
        "source": _relative_l2(fused_source, reference_source),
        "masked_write": _relative_l2(fused_output, reference_output),
    }
    gradient_errors = {
        name: _relative_l2(candidate, reference)
        for name, candidate, reference in zip(
            ("streams", "matrix", "read_weights", "write_weights", "update"),
            fused_gradients,
            reference_gradients,
            strict=True,
        )
    }
    finite = all(
        math.isfinite(value)
        for value in (*forward_errors.values(), *gradient_errors.values())
    )
    for _ in range(warmup_iterations):
        execute(fused=True)
        execute(fused=False)
    torch.cuda.synchronize(device)
    timings: dict[str, list[float]] = {"fused": [], "reference": []}
    for index in range(timed_iterations):
        order = (True, False) if index % 2 == 0 else (False, True)
        for fused in order:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            execute(fused=fused)
            torch.cuda.synchronize(device)
            timings["fused" if fused else "reference"].append(
                time.perf_counter() - started
            )

    def timing_summary(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        p95_index = min(
            len(ordered) - 1,
            max(0, math.ceil(0.95 * len(ordered)) - 1),
        )
        return {
            "median_seconds": statistics.median(ordered),
            "p95_seconds": ordered[p95_index],
        }

    fused_timing = timing_summary(timings["fused"])
    reference_timing = timing_summary(timings["reference"])
    speedup = (
        reference_timing["median_seconds"]
        / max(fused_timing["median_seconds"], 1.0e-12)
    )
    performance = {
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "scope": "read_mix_and_masked_write_forward_backward",
        "fused": fused_timing,
        "torch_reference": reference_timing,
        "speedup_reference_over_fused": speedup,
        "minimum_speedup": minimum_speedup,
        "throughput_positive": bool(
            math.isfinite(speedup) and speedup >= minimum_speedup
        ),
    }
    passed = bool(
        finite
        and max(forward_errors.values()) <= maximum_forward_relative_error
        and max(gradient_errors.values()) <= maximum_backward_relative_error
        and performance["throughput_positive"]
    )
    geometry = _canary_geometry(config)
    return {
        **geometry,
        "geometry_sha256": _canonical_sha256(geometry),
        "tokens": token_count,
        "dtype": "bfloat16",
        "forward_relative_l2": forward_errors,
        "backward_relative_l2": gradient_errors,
        "performance": performance,
        "finite": finite,
        "passed": passed,
    }


def run_mhc_fused_canary(
    configs: Sequence["Metis16Config"],
    *,
    device: torch.device | str,
    token_count: int = 257,
    maximum_forward_relative_error: float = 0.03,
    maximum_backward_relative_error: float = 0.08,
    minimum_speedup: float = 1.0,
    warmup_iterations: int = 4,
    timed_iterations: int = 12,
) -> dict[str, Any]:
    resolved = torch.device(device)
    if token_count <= 0:
        raise ValueError("mHC canary token_count must be positive.")
    if maximum_forward_relative_error <= 0 or maximum_backward_relative_error <= 0:
        raise ValueError("mHC canary tolerances must be positive.")
    if minimum_speedup < 1.0:
        raise ValueError("mHC fused kernel must be throughput-positive (speedup >= 1).")
    if warmup_iterations <= 0 or timed_iterations <= 0:
        raise ValueError("mHC canary warmup/timed iteration counts must be positive.")
    status = triton_rocm_backend_status(resolved)
    if not status["available"]:
        raise RuntimeError(
            "Cannot run the fused mHC canary without Triton/ROCm: "
            f"{status['reason']}"
        )
    rows: dict[str, Any] = {}
    for config in configs:
        if config.family in rows:
            raise ValueError(f"Duplicate mHC canary family: {config.family}")
        require_mhc_backend(
            backend=config.mhc_backend,
            family=config.family,
            device=resolved,
        )
        rows[config.family] = _run_one_canary(
            config,
            device=resolved,
            token_count=token_count,
            maximum_forward_relative_error=maximum_forward_relative_error,
            maximum_backward_relative_error=maximum_backward_relative_error,
            minimum_speedup=minimum_speedup,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
        )
        torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "schema": MHC_CANARY_SCHEMA,
        "created_at": _utc_now(),
        "backend": status,
        "thresholds": {
            "maximum_forward_relative_l2": maximum_forward_relative_error,
            "maximum_backward_relative_l2": maximum_backward_relative_error,
            "minimum_speedup_reference_over_fused": minimum_speedup,
        },
        "families": rows,
        "ok": bool(rows) and all(bool(row["passed"]) for row in rows.values()),
    }
    report["report_sha256"] = _canonical_sha256(report)
    validate_mhc_canary_report(
        report,
        configs=configs,
        maximum_forward_relative_error=maximum_forward_relative_error,
        maximum_backward_relative_error=maximum_backward_relative_error,
        minimum_speedup=minimum_speedup,
    )
    return report


def validate_mhc_canary_report(
    report: Mapping[str, Any],
    *,
    configs: Sequence["Metis16Config"],
    maximum_forward_relative_error: float = 0.03,
    maximum_backward_relative_error: float = 0.08,
    minimum_speedup: float = 1.0,
) -> None:
    if report.get("schema") != MHC_CANARY_SCHEMA:
        raise RuntimeError("mHC canary report schema is invalid.")
    if report.get("report_sha256") != _canonical_sha256(
        report, omit=("report_sha256",)
    ):
        raise RuntimeError("mHC canary report hash is invalid or stale.")
    backend = report.get("backend")
    if (
        not isinstance(backend, Mapping)
        or backend.get("available") is not True
        or backend.get("backend") != "triton_rocm"
        or not backend.get("rocm")
    ):
        raise RuntimeError("mHC canary did not prove a Triton/ROCm execution path.")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise RuntimeError("mHC canary thresholds are missing.")
    maximum_forward = float(thresholds.get("maximum_forward_relative_l2", -1.0))
    maximum_backward = float(thresholds.get("maximum_backward_relative_l2", -1.0))
    reported_minimum_speedup = float(
        thresholds.get("minimum_speedup_reference_over_fused", -1.0)
    )
    if (
        maximum_forward != maximum_forward_relative_error
        or maximum_backward != maximum_backward_relative_error
        or reported_minimum_speedup != minimum_speedup
        or minimum_speedup < 1.0
    ):
        raise RuntimeError("mHC canary thresholds are invalid.")
    families = report.get("families")
    expected = {config.family: config for config in configs}
    if not isinstance(families, Mapping) or set(families) != set(expected):
        raise RuntimeError("mHC canary family coverage is incomplete.")
    for family, config in expected.items():
        row = families[family]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"mHC canary row for {family} is invalid.")
        geometry = _canary_geometry(config)
        if any(row.get(key) != value for key, value in geometry.items()):
            raise RuntimeError(f"mHC canary geometry changed for {family}.")
        if row.get("geometry_sha256") != _canonical_sha256(geometry):
            raise RuntimeError(f"mHC canary geometry hash is invalid for {family}.")
        if int(row.get("tokens", 0)) <= 0 or row.get("dtype") != "bfloat16":
            raise RuntimeError(f"mHC canary workload is invalid for {family}.")
        forward = row.get("forward_relative_l2")
        backward = row.get("backward_relative_l2")
        performance = row.get("performance")
        if not isinstance(forward, Mapping) or set(forward) != {
            "source",
            "masked_write",
        }:
            raise RuntimeError(f"mHC forward parity evidence is incomplete for {family}.")
        if not isinstance(backward, Mapping) or set(backward) != {
            "streams",
            "matrix",
            "read_weights",
            "write_weights",
            "update",
        }:
            raise RuntimeError(f"mHC backward parity evidence is incomplete for {family}.")
        if not isinstance(performance, Mapping):
            raise RuntimeError(f"mHC performance evidence is incomplete for {family}.")
        fused_timing = performance.get("fused")
        reference_timing = performance.get("torch_reference")
        if (
            performance.get("scope")
            != "read_mix_and_masked_write_forward_backward"
            or int(performance.get("warmup_iterations", 0)) <= 0
            or int(performance.get("timed_iterations", 0)) <= 0
            or not isinstance(fused_timing, Mapping)
            or not isinstance(reference_timing, Mapping)
        ):
            raise RuntimeError(f"mHC performance evidence is incomplete for {family}.")
        fused_median = float(fused_timing.get("median_seconds", -1.0))
        fused_p95 = float(fused_timing.get("p95_seconds", -1.0))
        reference_median = float(reference_timing.get("median_seconds", -1.0))
        reference_p95 = float(reference_timing.get("p95_seconds", -1.0))
        speedup = float(performance.get("speedup_reference_over_fused", -1.0))
        recomputed_speedup = reference_median / max(fused_median, 1.0e-12)
        performance_valid = bool(
            all(
                math.isfinite(value) and value > 0
                for value in (
                    fused_median,
                    fused_p95,
                    reference_median,
                    reference_p95,
                    speedup,
                )
            )
            and fused_p95 >= fused_median
            and reference_p95 >= reference_median
            and math.isclose(speedup, recomputed_speedup, rel_tol=1.0e-9)
            and float(performance.get("minimum_speedup", -1.0)) == minimum_speedup
            and speedup >= minimum_speedup
            and performance.get("throughput_positive") is True
        )
        forward_values = [float(value) for value in forward.values()]
        backward_values = [float(value) for value in backward.values()]
        if (
            row.get("finite") is not True
            or row.get("passed") is not True
            or not all(math.isfinite(value) for value in forward_values + backward_values)
            or max(forward_values) > maximum_forward
            or max(backward_values) > maximum_backward
            or not performance_valid
        ):
            raise RuntimeError(f"mHC fused parity gate failed for {family}.")
    if report.get("ok") is not True:
        raise RuntimeError("mHC fused canary is not approved.")
