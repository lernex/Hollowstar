from __future__ import annotations

import torch
from torch import Tensor

try:  # Triton is sealed into the Portage runtime.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - runtime dependent.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _stream_gate_forward_kernel(
        values,
        vectors,
        output,
        streams: tl.constexpr,
        width: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        stream = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros((BLOCK_D,), tl.float32)
        for start in range(0, width, BLOCK_D):
            columns = start + offsets
            mask = columns < width
            value = tl.load(
                values + (token * streams + stream) * width + columns,
                mask=mask,
                other=0.0,
            )
            vector = tl.load(
                vectors + stream * width + columns,
                mask=mask,
                other=0.0,
            )
            accumulator += value.to(tl.float32) * vector.to(tl.float32)
        tl.store(
            output + token * streams + stream,
            tl.sum(accumulator, axis=0),
        )


    @triton.jit
    def _stream_gate_input_grad_kernel(
        grad_output,
        vectors,
        grad_values,
        streams: tl.constexpr,
        width: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_stream = tl.program_id(0)
        block = tl.program_id(1)
        stream = token_stream % streams
        columns = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = columns < width
        gradient = tl.load(grad_output + token_stream)
        vector = tl.load(
            vectors + stream * width + columns,
            mask=mask,
            other=0.0,
        )
        tl.store(
            grad_values + token_stream * width + columns,
            gradient * vector,
            mask=mask,
        )


    @triton.jit
    def _stream_gate_vector_partial_kernel(
        grad_output,
        values,
        partial,
        total_tokens: tl.constexpr,
        streams: tl.constexpr,
        width: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        stream = tl.program_id(0)
        column_block = tl.program_id(1)
        token_block = tl.program_id(2)
        tokens = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        columns = column_block * BLOCK_D + tl.arange(0, BLOCK_D)
        token_mask = tokens < total_tokens
        column_mask = columns < width
        gradient = tl.load(
            grad_output + tokens * streams + stream,
            mask=token_mask,
            other=0.0,
        )
        values_block = tl.load(
            values
            + (tokens[:, None] * streams + stream) * width
            + columns[None, :],
            mask=token_mask[:, None] & column_mask[None, :],
            other=0.0,
        )
        reduced = tl.sum(
            gradient[:, None] * values_block.to(tl.float32),
            axis=0,
        )
        partial_offset = (
            (stream * tl.num_programs(2) + token_block) * width + columns
        )
        tl.store(partial + partial_offset, reduced, mask=column_mask)


    @triton.jit
    def _stream_gate_vector_reduce_kernel(
        partial,
        grad_vectors,
        token_blocks: tl.constexpr,
        width: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        stream = tl.program_id(0)
        column = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_T)
        accumulator = tl.zeros((BLOCK_T,), tl.float32)
        for start in range(0, token_blocks, BLOCK_T):
            blocks = start + offsets
            mask = blocks < token_blocks
            values = tl.load(
                partial + (stream * token_blocks + blocks) * width + column,
                mask=mask,
                other=0.0,
            )
            accumulator += values
        tl.store(
            grad_vectors + stream * width + column,
            tl.sum(accumulator, axis=0),
        )


class _FusedStreamGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: Tensor, vectors: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Fused stream gates require Triton.")
        shape = values.shape
        flat = values.reshape(-1, shape[-2], shape[-1]).contiguous()
        vectors = vectors.contiguous()
        output = torch.empty(
            flat.shape[0],
            flat.shape[1],
            device=values.device,
            dtype=torch.float32,
        )
        _stream_gate_forward_kernel[(flat.shape[0], flat.shape[1])](
            flat,
            vectors,
            output,
            streams=flat.shape[1],
            width=flat.shape[2],
            BLOCK_D=256,
        )
        ctx.save_for_backward(flat, vectors)
        ctx.input_shape = shape
        return output.view(*shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        values, vectors = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(
            values.shape[0],
            values.shape[1],
        )
        grad_values = torch.empty_like(values)
        _stream_gate_input_grad_kernel[
            (
                values.shape[0] * values.shape[1],
                triton.cdiv(values.shape[2], 256),
            )
        ](
            grad_output,
            vectors,
            grad_values,
            streams=values.shape[1],
            width=values.shape[2],
            BLOCK_D=256,
        )
        token_block = 128
        token_blocks = triton.cdiv(values.shape[0], token_block)
        partial = torch.empty(
            values.shape[1],
            token_blocks,
            values.shape[2],
            device=values.device,
            dtype=torch.float32,
        )
        _stream_gate_vector_partial_kernel[
            (
                values.shape[1],
                triton.cdiv(values.shape[2], 128),
                token_blocks,
            )
        ](
            grad_output,
            values,
            partial,
            total_tokens=values.shape[0],
            streams=values.shape[1],
            width=values.shape[2],
            TOKEN_BLOCK=token_block,
            BLOCK_D=128,
        )
        grad_vectors = torch.empty_like(vectors)
        _stream_gate_vector_reduce_kernel[
            (values.shape[1], values.shape[2])
        ](
            partial,
            grad_vectors,
            token_blocks=token_blocks,
            width=values.shape[2],
            BLOCK_T=256,
        )
        return grad_values.view(ctx.input_shape), grad_vectors


def stream_gate_logits(values: Tensor, vectors: Tensor) -> Tensor:
    if values.shape[-2:] != vectors.shape:
        raise ValueError("Stream values and gate vectors have incompatible shapes.")
    if values.is_cuda:
        if triton is None:
            raise RuntimeError("ROCm stream-gate fusion requires Triton.")
        return _FusedStreamGate.apply(values, vectors)
    return (values * vectors).sum(dim=-1)


__all__ = ["stream_gate_logits"]
