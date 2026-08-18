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


    @triton.jit
    def _memory_scores_forward_kernel(
        query,
        key,
        output,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        stream = tl.program_id(1)
        slot = tl.program_id(2)
        columns = tl.arange(0, BLOCK_H)
        mask = columns < width
        query_values = tl.load(
            query + (token * streams + stream) * width + columns,
            mask=mask,
            other=0.0,
        )
        key_values = tl.load(
            key + (token * slots + slot) * width + columns,
            mask=mask,
            other=0.0,
        )
        score = tl.sum(
            query_values.to(tl.float32) * key_values.to(tl.float32),
            axis=0,
        )
        tl.store(output + (token * streams + stream) * slots + slot, score)


    @triton.jit
    def _memory_scores_query_grad_kernel(
        grad_output,
        key,
        grad_query,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        stream = tl.program_id(1)
        block = tl.program_id(2)
        columns = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = columns < width
        accumulator = tl.zeros((BLOCK_H,), tl.float32)
        for slot in range(0, slots):
            gradient = tl.load(
                grad_output + (token * streams + stream) * slots + slot
            )
            key_values = tl.load(
                key + (token * slots + slot) * width + columns,
                mask=mask,
                other=0.0,
            )
            accumulator += gradient * key_values.to(tl.float32)
        tl.store(
            grad_query + (token * streams + stream) * width + columns,
            accumulator,
            mask=mask,
        )


    @triton.jit
    def _memory_scores_key_grad_kernel(
        grad_output,
        query,
        grad_key,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.program_id(1)
        block = tl.program_id(2)
        columns = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = columns < width
        accumulator = tl.zeros((BLOCK_H,), tl.float32)
        for stream in range(0, streams):
            gradient = tl.load(
                grad_output + (token * streams + stream) * slots + slot
            )
            query_values = tl.load(
                query + (token * streams + stream) * width + columns,
                mask=mask,
                other=0.0,
            )
            accumulator += gradient * query_values.to(tl.float32)
        tl.store(
            grad_key + (token * slots + slot) * width + columns,
            accumulator,
            mask=mask,
        )


    @triton.jit
    def _memory_combine_forward_kernel(
        weights,
        values,
        output,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        stream = tl.program_id(1)
        block = tl.program_id(2)
        columns = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = columns < width
        accumulator = tl.zeros((BLOCK_H,), tl.float32)
        for slot in range(0, slots):
            weight = tl.load(
                weights + (token * streams + stream) * slots + slot
            )
            value = tl.load(
                values + (token * slots + slot) * width + columns,
                mask=mask,
                other=0.0,
            )
            accumulator += weight.to(tl.float32) * value.to(tl.float32)
        tl.store(
            output + (token * streams + stream) * width + columns,
            accumulator,
            mask=mask,
        )


    @triton.jit
    def _memory_combine_weight_grad_kernel(
        grad_output,
        values,
        grad_weights,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        stream = tl.program_id(1)
        slot = tl.program_id(2)
        columns = tl.arange(0, BLOCK_H)
        mask = columns < width
        gradient = tl.load(
            grad_output + (token * streams + stream) * width + columns,
            mask=mask,
            other=0.0,
        )
        value = tl.load(
            values + (token * slots + slot) * width + columns,
            mask=mask,
            other=0.0,
        )
        result = tl.sum(
            gradient.to(tl.float32) * value.to(tl.float32),
            axis=0,
        )
        tl.store(
            grad_weights + (token * streams + stream) * slots + slot,
            result,
        )


    @triton.jit
    def _memory_combine_value_grad_kernel(
        grad_output,
        weights,
        grad_values,
        streams: tl.constexpr,
        slots: tl.constexpr,
        width: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.program_id(1)
        block = tl.program_id(2)
        columns = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = columns < width
        accumulator = tl.zeros((BLOCK_H,), tl.float32)
        for stream in range(0, streams):
            gradient = tl.load(
                grad_output + (token * streams + stream) * width + columns,
                mask=mask,
                other=0.0,
            )
            weight = tl.load(
                weights + (token * streams + stream) * slots + slot
            )
            accumulator += gradient.to(tl.float32) * weight.to(tl.float32)
        tl.store(
            grad_values + (token * slots + slot) * width + columns,
            accumulator,
            mask=mask,
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


class _FusedMemoryScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Fused memory scores require Triton.")
        query_shape = query.shape
        key_shape = key.shape
        flat_query = query.reshape(-1, query_shape[-2], query_shape[-1]).contiguous()
        flat_key = key.reshape(-1, key_shape[-2], key_shape[-1]).contiguous()
        if flat_query.shape[0] != flat_key.shape[0]:
            raise ValueError("Memory query and key token counts differ.")
        output = torch.empty(
            flat_query.shape[0],
            flat_query.shape[1],
            flat_key.shape[1],
            device=query.device,
            dtype=torch.float32,
        )
        _memory_scores_forward_kernel[
            (flat_query.shape[0], flat_query.shape[1], flat_key.shape[1])
        ](
            flat_query,
            flat_key,
            output,
            streams=flat_query.shape[1],
            slots=flat_key.shape[1],
            width=flat_query.shape[2],
            BLOCK_H=triton.next_power_of_2(flat_query.shape[2]),
        )
        ctx.save_for_backward(flat_query, flat_key)
        ctx.query_shape = query_shape
        ctx.key_shape = key_shape
        return output.view(*query_shape[:-2], query_shape[-2], key_shape[-2])

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        query, key = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(
            query.shape[0],
            query.shape[1],
            key.shape[1],
        )
        block_h = min(256, triton.next_power_of_2(query.shape[2]))
        grad_query = torch.empty_like(query)
        _memory_scores_query_grad_kernel[
            (
                query.shape[0],
                query.shape[1],
                triton.cdiv(query.shape[2], block_h),
            )
        ](
            grad_output,
            key,
            grad_query,
            streams=query.shape[1],
            slots=key.shape[1],
            width=query.shape[2],
            BLOCK_H=block_h,
        )
        grad_key = torch.empty_like(key)
        _memory_scores_key_grad_kernel[
            (
                key.shape[0],
                key.shape[1],
                triton.cdiv(key.shape[2], block_h),
            )
        ](
            grad_output,
            query,
            grad_key,
            streams=query.shape[1],
            slots=key.shape[1],
            width=key.shape[2],
            BLOCK_H=block_h,
        )
        return grad_query.view(ctx.query_shape), grad_key.view(ctx.key_shape)


class _FusedMemoryCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights: Tensor, values: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Fused memory combine requires Triton.")
        weight_shape = weights.shape
        value_shape = values.shape
        flat_weights = weights.reshape(
            -1,
            weight_shape[-2],
            weight_shape[-1],
        ).contiguous()
        flat_values = values.reshape(
            -1,
            value_shape[-2],
            value_shape[-1],
        ).contiguous()
        output = torch.empty(
            flat_weights.shape[0],
            flat_weights.shape[1],
            flat_values.shape[2],
            device=weights.device,
            dtype=values.dtype,
        )
        block_h = min(256, triton.next_power_of_2(flat_values.shape[2]))
        _memory_combine_forward_kernel[
            (
                flat_weights.shape[0],
                flat_weights.shape[1],
                triton.cdiv(flat_values.shape[2], block_h),
            )
        ](
            flat_weights,
            flat_values,
            output,
            streams=flat_weights.shape[1],
            slots=flat_weights.shape[2],
            width=flat_values.shape[2],
            BLOCK_H=block_h,
        )
        ctx.save_for_backward(flat_weights, flat_values)
        ctx.weight_shape = weight_shape
        ctx.value_shape = value_shape
        return output.view(
            *weight_shape[:-2],
            weight_shape[-2],
            value_shape[-1],
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        weights, values = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(
            weights.shape[0],
            weights.shape[1],
            values.shape[2],
        )
        grad_weights = torch.empty_like(weights)
        _memory_combine_weight_grad_kernel[
            (weights.shape[0], weights.shape[1], weights.shape[2])
        ](
            grad_output,
            values,
            grad_weights,
            streams=weights.shape[1],
            slots=weights.shape[2],
            width=values.shape[2],
            BLOCK_H=triton.next_power_of_2(values.shape[2]),
        )
        block_h = min(256, triton.next_power_of_2(values.shape[2]))
        grad_values = torch.empty_like(values)
        _memory_combine_value_grad_kernel[
            (
                values.shape[0],
                values.shape[1],
                triton.cdiv(values.shape[2], block_h),
            )
        ](
            grad_output,
            weights,
            grad_values,
            streams=weights.shape[1],
            slots=weights.shape[2],
            width=values.shape[2],
            BLOCK_H=block_h,
        )
        return (
            grad_weights.view(ctx.weight_shape),
            grad_values.view(ctx.value_shape),
        )


def stream_gate_logits(values: Tensor, vectors: Tensor) -> Tensor:
    if values.shape[-2:] != vectors.shape:
        raise ValueError("Stream values and gate vectors have incompatible shapes.")
    if values.is_cuda:
        if triton is None:
            raise RuntimeError("ROCm stream-gate fusion requires Triton.")
        return _FusedStreamGate.apply(values, vectors)
    return (values * vectors).sum(dim=-1)


def memory_attention_scores(query: Tensor, key: Tensor) -> Tensor:
    if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Memory query and key shapes are incompatible.")
    if query.is_cuda:
        if triton is None:
            raise RuntimeError("ROCm memory-score fusion requires Triton.")
        return _FusedMemoryScores.apply(query, key)
    return (query.unsqueeze(-2) * key.unsqueeze(-3)).sum(dim=-1)


def memory_attention_combine(weights: Tensor, values: Tensor) -> Tensor:
    if weights.shape[:-2] != values.shape[:-2] or weights.shape[-1] != values.shape[-2]:
        raise ValueError("Memory weights and values have incompatible shapes.")
    if weights.is_cuda:
        if triton is None:
            raise RuntimeError("ROCm memory-combine fusion requires Triton.")
        return _FusedMemoryCombine.apply(weights, values)
    return (weights.unsqueeze(-1) * values.unsqueeze(-3)).sum(dim=-2)


__all__ = [
    "memory_attention_combine",
    "memory_attention_scores",
    "stream_gate_logits",
]
