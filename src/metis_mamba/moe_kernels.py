from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


def triton_moe_kernels_available() -> bool:
    return triton is not None and tl is not None


if triton is not None and tl is not None:

    @triton.jit
    def _swiglu_forward_kernel(
        gate_up,
        out,
        total_rows: tl.constexpr,
        hidden_size: tl.constexpr,
        stride_gate_up_row: tl.constexpr,
        stride_out_row: tl.constexpr,
        block_m: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.program_id(1) * block_d + tl.arange(0, block_d)
        mask = (rows[:, None] < total_rows) & (cols[None, :] < hidden_size)
        gate = tl.load(
            gate_up + rows[:, None] * stride_gate_up_row + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up + rows[:, None] * stride_gate_up_row + hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        safe_gate = tl.minimum(tl.maximum(gate, -20.0), 20.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-safe_gate))
        silu = tl.where(gate < -20.0, 0.0, tl.where(gate > 20.0, gate, gate * sigmoid))
        values = silu * up
        tl.store(out + rows[:, None] * stride_out_row + cols[None, :], values, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        grad_out,
        gate_up,
        grad_gate_up,
        total_rows: tl.constexpr,
        hidden_size: tl.constexpr,
        stride_grad_out_row: tl.constexpr,
        stride_gate_up_row: tl.constexpr,
        stride_grad_gate_up_row: tl.constexpr,
        block_m: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.program_id(1) * block_d + tl.arange(0, block_d)
        mask = (rows[:, None] < total_rows) & (cols[None, :] < hidden_size)
        grad = tl.load(
            grad_out + rows[:, None] * stride_grad_out_row + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_up + rows[:, None] * stride_gate_up_row + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up + rows[:, None] * stride_gate_up_row + hidden_size + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        safe_gate = tl.minimum(tl.maximum(gate, -20.0), 20.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-safe_gate))
        silu = tl.where(gate < -20.0, 0.0, tl.where(gate > 20.0, gate, gate * sigmoid))
        dsilu_mid = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        dsilu = tl.where(gate < -20.0, 0.0, tl.where(gate > 20.0, 1.0, dsilu_mid))
        grad_gate = grad * up * dsilu
        grad_up = grad * silu
        tl.store(
            grad_gate_up + rows[:, None] * stride_grad_gate_up_row + cols[None, :],
            grad_gate,
            mask=mask,
        )
        tl.store(
            grad_gate_up + rows[:, None] * stride_grad_gate_up_row + hidden_size + cols[None, :],
            grad_up,
            mask=mask,
        )

    @triton.jit
    def _count_experts_kernel(
        expert_ids,
        counts,
        total_assignments: tl.constexpr,
        block_a: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_a + tl.arange(0, block_a)
        mask = offsets < total_assignments
        experts = tl.load(expert_ids + offsets, mask=mask, other=0).to(tl.int32)
        tl.atomic_add(counts + experts, tl.full((block_a,), 1, tl.int32), sem="relaxed", mask=mask)

    @triton.jit
    def _bucket_dispatch_kernel(
        x,
        expert_ids,
        weights_in,
        offsets,
        write_counts,
        x_perm,
        assignment_ids,
        reverse_positions,
        weights_perm,
        total_assignments: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        stride_x_row: tl.constexpr,
        block_a: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_a + tl.arange(0, block_a)
        cols = tl.arange(0, block_d)
        row_mask = rows < total_assignments

        experts = tl.load(expert_ids + rows, mask=row_mask, other=0).to(tl.int32)
        ranks = tl.atomic_add(
            write_counts + experts,
            tl.full((block_a,), 1, tl.int32),
            sem="relaxed",
            mask=row_mask,
        )
        out_rows = tl.load(offsets + experts, mask=row_mask, other=0).to(tl.int64) + ranks.to(tl.int64)
        source_rows = rows // top_k

        values = tl.load(
            x + source_rows[:, None] * stride_x_row + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < dim),
            other=0.0,
        )
        tl.store(
            x_perm + out_rows[:, None] * dim + cols[None, :],
            values,
            mask=row_mask[:, None] & (cols[None, :] < dim),
        )
        tl.store(assignment_ids + out_rows, rows.to(tl.int64), mask=row_mask)
        tl.store(reverse_positions + rows, out_rows.to(tl.int64), mask=row_mask)
        weights = tl.load(weights_in + rows, mask=row_mask, other=0.0)
        tl.store(weights_perm + out_rows, weights, mask=row_mask)

    @triton.jit
    def _bucket_dispatch_backward_kernel(
        grad_x_perm,
        grad_weights_perm,
        assignment_ids,
        grad_x,
        grad_weights,
        total_assignments: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        stride_grad_x_row: tl.constexpr,
        has_grad_x: tl.constexpr,
        has_grad_weights: tl.constexpr,
        block_a: tl.constexpr,
        block_d: tl.constexpr,
    ):
        perm_rows = tl.program_id(0) * block_a + tl.arange(0, block_a)
        cols = tl.arange(0, block_d)
        row_mask = perm_rows < total_assignments
        assign = tl.load(assignment_ids + perm_rows, mask=row_mask, other=0)
        row_mask = row_mask & (assign >= 0)
        source_rows = assign // top_k

        if has_grad_x:
            grad_values = tl.load(
                grad_x_perm + perm_rows[:, None] * dim + cols[None, :],
                mask=row_mask[:, None] & (cols[None, :] < dim),
                other=0.0,
            )
            tl.atomic_add(
                grad_x + source_rows[:, None] * stride_grad_x_row + cols[None, :],
                grad_values,
                sem="relaxed",
                mask=row_mask[:, None] & (cols[None, :] < dim),
            )

        if has_grad_weights:
            grad_w = tl.load(grad_weights_perm + perm_rows, mask=row_mask, other=0.0)
            tl.store(grad_weights + assign, grad_w, mask=row_mask)

    @triton.jit
    def _capacity_dispatch_kernel(
        x,
        expert_ids,
        weights_in,
        write_counts,
        overflow,
        x_perm,
        assignment_ids,
        reverse_positions,
        weights_perm,
        total_assignments: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        capacity: tl.constexpr,
        stride_x_row: tl.constexpr,
        block_a: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_a + tl.arange(0, block_a)
        cols = tl.arange(0, block_d)
        row_mask = rows < total_assignments

        experts = tl.load(expert_ids + rows, mask=row_mask, other=0).to(tl.int32)
        ranks = tl.atomic_add(
            write_counts + experts,
            tl.full((block_a,), 1, tl.int32),
            sem="relaxed",
            mask=row_mask,
        )
        overflow_rows = row_mask & (ranks >= capacity)
        valid = row_mask & (ranks < capacity)
        tl.store(overflow, 1, mask=tl.sum(tl.where(overflow_rows, 1, 0), axis=0) > 0)
        out_rows = experts.to(tl.int64) * capacity + ranks.to(tl.int64)
        source_rows = rows // top_k

        values = tl.load(
            x + source_rows[:, None] * stride_x_row + cols[None, :],
            mask=valid[:, None] & (cols[None, :] < dim),
            other=0.0,
        )
        tl.store(
            x_perm + out_rows[:, None] * dim + cols[None, :],
            values,
            mask=valid[:, None] & (cols[None, :] < dim),
        )
        tl.store(assignment_ids + out_rows, rows.to(tl.int64), mask=valid)
        tl.store(reverse_positions + rows, out_rows.to(tl.int64), mask=valid)
        weights = tl.load(weights_in + rows, mask=valid, other=0.0)
        tl.store(weights_perm + out_rows, weights, mask=valid)

    @triton.jit
    def _reverse_weighted_combine_kernel(
        y_perm,
        reverse_positions,
        weights_in,
        out,
        output_rows: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        block_m: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_d)
        row_mask = rows < output_rows
        acc = tl.zeros((block_m, block_d), tl.float32)
        for slot in range(0, top_k):
            assign = rows * top_k + slot
            perm_rows = tl.load(reverse_positions + assign, mask=row_mask, other=-1).to(tl.int64)
            valid = row_mask & (perm_rows >= 0)
            w = tl.load(weights_in + assign, mask=valid, other=0.0).to(tl.float32)
            values = tl.load(
                y_perm + perm_rows[:, None] * dim + cols[None, :],
                mask=valid[:, None] & (cols[None, :] < dim),
                other=0.0,
            ).to(tl.float32)
            acc += values * w[:, None]
        tl.store(
            out + rows[:, None] * dim + cols[None, :],
            acc,
            mask=row_mask[:, None] & (cols[None, :] < dim),
        )

    @triton.jit
    def _reverse_weighted_combine_backward_kernel(
        grad_out,
        y_perm,
        reverse_positions,
        weights_in,
        grad_y_perm,
        grad_weights,
        output_rows: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        has_grad_y: tl.constexpr,
        has_grad_weights: tl.constexpr,
        block_m: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_d)
        row_mask = rows < output_rows
        grad_values = tl.load(
            grad_out + rows[:, None] * dim + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < dim),
            other=0.0,
        )
        for slot in range(0, top_k):
            assign = rows * top_k + slot
            perm_rows = tl.load(reverse_positions + assign, mask=row_mask, other=-1).to(tl.int64)
            valid = row_mask & (perm_rows >= 0)
            if has_grad_y:
                w = tl.load(weights_in + assign, mask=valid, other=0.0)
                tl.store(
                    grad_y_perm + perm_rows[:, None] * dim + cols[None, :],
                    grad_values * w[:, None],
                    mask=valid[:, None] & (cols[None, :] < dim),
                )
            if has_grad_weights:
                y_values = tl.load(
                    y_perm + perm_rows[:, None] * dim + cols[None, :],
                    mask=valid[:, None] & (cols[None, :] < dim),
                    other=0.0,
                ).to(tl.float32)
                grad_w = tl.sum(grad_values.to(tl.float32) * y_values, axis=1)
                tl.store(grad_weights + assign, grad_w, mask=valid)

    @triton.jit
    def _weighted_unpermute_kernel(
        y_perm,
        assignment_ids,
        weights,
        out,
        total_assignments: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        block_a: tl.constexpr,
        block_d: tl.constexpr,
    ):
        perm_rows = tl.program_id(0) * block_a + tl.arange(0, block_a)
        cols = tl.arange(0, block_d)
        row_mask = perm_rows < total_assignments
        assign = tl.load(assignment_ids + perm_rows, mask=row_mask, other=0)
        row_mask = row_mask & (assign >= 0)
        out_rows = assign // top_k
        w = tl.load(weights + perm_rows, mask=row_mask, other=0.0).to(tl.float32)
        values = tl.load(
            y_perm + perm_rows[:, None] * dim + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < dim),
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            out + out_rows[:, None] * dim + cols[None, :],
            values * w[:, None],
            sem="relaxed",
            mask=row_mask[:, None] & (cols[None, :] < dim),
        )

    @triton.jit
    def _weighted_unpermute_backward_kernel(
        grad_out,
        y_perm,
        weights,
        assignment_ids,
        grad_y_perm,
        grad_weights,
        total_assignments: tl.constexpr,
        dim: tl.constexpr,
        top_k: tl.constexpr,
        has_grad_y: tl.constexpr,
        has_grad_weights: tl.constexpr,
        block_a: tl.constexpr,
        block_d: tl.constexpr,
    ):
        perm_rows = tl.program_id(0) * block_a + tl.arange(0, block_a)
        cols = tl.arange(0, block_d)
        row_mask = perm_rows < total_assignments
        assign = tl.load(assignment_ids + perm_rows, mask=row_mask, other=0)
        row_mask = row_mask & (assign >= 0)
        out_rows = assign // top_k
        grad_values = tl.load(
            grad_out + out_rows[:, None] * dim + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < dim),
            other=0.0,
        )

        if has_grad_y:
            w = tl.load(weights + perm_rows, mask=row_mask, other=0.0)
            tl.store(
                grad_y_perm + perm_rows[:, None] * dim + cols[None, :],
                grad_values * w[:, None],
                mask=row_mask[:, None] & (cols[None, :] < dim),
            )

        if has_grad_weights:
            y_values = tl.load(
                y_perm + perm_rows[:, None] * dim + cols[None, :],
                mask=row_mask[:, None] & (cols[None, :] < dim),
                other=0.0,
            ).to(tl.float32)
            grad_w = tl.sum(grad_values.to(tl.float32) * y_values, axis=1)
            tl.store(grad_weights + perm_rows, grad_w, mask=row_mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _swiglu_block_d(hidden_size: int) -> int:
    if hidden_size >= 4096:
        return 512
    if hidden_size >= 1024:
        return 256
    return min(256, _next_power_of_2(hidden_size))


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, hidden_size: int) -> torch.Tensor:
        if not triton_moe_kernels_available() or not gate_up.is_cuda:
            raise RuntimeError("Triton fused SwiGLU requires a CUDA tensor and Triton.")
        hidden_size = int(hidden_size)
        if gate_up.shape[-1] != 2 * hidden_size:
            raise ValueError(f"Expected last dimension {2 * hidden_size}, got {gate_up.shape[-1]}.")
        gate_up_2d = gate_up.reshape(-1, 2 * hidden_size).contiguous()
        out_2d = torch.empty((gate_up_2d.shape[0], hidden_size), device=gate_up.device, dtype=gate_up.dtype)
        block_m = 16
        block_d = _swiglu_block_d(hidden_size)
        grid = (triton.cdiv(gate_up_2d.shape[0], block_m), triton.cdiv(hidden_size, block_d))
        _swiglu_forward_kernel[grid](
            gate_up_2d,
            out_2d,
            int(gate_up_2d.shape[0]),
            hidden_size,
            int(gate_up_2d.stride(0)),
            int(out_2d.stride(0)),
            block_m,
            block_d,
            num_warps=8,
        )
        ctx.save_for_backward(gate_up_2d)
        ctx.hidden_size = hidden_size
        ctx.input_shape = tuple(gate_up.shape)
        return out_2d.reshape(*gate_up.shape[:-1], hidden_size)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (gate_up_2d,) = ctx.saved_tensors
        hidden_size = ctx.hidden_size
        grad_out_2d = grad_out.reshape(-1, hidden_size).contiguous()
        backward_mode = os.environ.get("METIS_TRITON_SWIGLU_BACKWARD", "torch").strip().lower()
        if backward_mode not in {"triton", "1", "true", "yes", "on"}:
            gate, up = gate_up_2d.split(hidden_size, dim=-1)
            grad = grad_out_2d.float()
            gate_f = gate.float()
            up_f = up.float()
            sigmoid = torch.sigmoid(gate_f)
            silu = gate_f * sigmoid
            dsilu = sigmoid * (1.0 + gate_f * (1.0 - sigmoid))
            grad_gate = grad * up_f * dsilu
            grad_up = grad * silu
            return torch.cat((grad_gate, grad_up), dim=-1).to(gate_up_2d.dtype).reshape(ctx.input_shape), None
        grad_gate_up = torch.empty_like(gate_up_2d)
        block_m = 16
        block_d = _swiglu_block_d(hidden_size)
        grid = (triton.cdiv(gate_up_2d.shape[0], block_m), triton.cdiv(hidden_size, block_d))
        _swiglu_backward_kernel[grid](
            grad_out_2d,
            gate_up_2d,
            grad_gate_up,
            int(gate_up_2d.shape[0]),
            hidden_size,
            int(grad_out_2d.stride(0)),
            int(gate_up_2d.stride(0)),
            int(grad_gate_up.stride(0)),
            block_m,
            block_d,
            num_warps=8,
        )
        return grad_gate_up.reshape(ctx.input_shape), None


class _BucketDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed_heads: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor, num_experts: int):
        if not triton_moe_kernels_available() or not routed_heads.is_cuda:
            raise RuntimeError("Triton MoE bucket dispatch requires a CUDA tensor and Triton.")
        if routed_heads.dim() != 2 or topk_indices.dim() != 2 or topk_weights.shape != topk_indices.shape:
            raise ValueError("Expected routed_heads [N, D], topk_indices/topk_weights [N, top_k].")
        if not routed_heads.is_contiguous():
            routed_heads = routed_heads.contiguous()
        if not topk_indices.is_contiguous():
            topk_indices = topk_indices.contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        num_rows, dim = routed_heads.shape
        top_k = topk_indices.shape[1]
        total_assignments = topk_indices.numel()
        flat_experts = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        counts = torch.zeros((num_experts,), device=routed_heads.device, dtype=torch.int32)
        block_count = 256
        grid_count = (triton.cdiv(total_assignments, block_count),)
        _count_experts_kernel[grid_count](flat_experts, counts, total_assignments, block_count)

        offsets = torch.empty((num_experts + 1,), device=routed_heads.device, dtype=torch.int32)
        offsets[0].zero_()
        offsets[1:].copy_(torch.cumsum(counts, dim=0))
        x_perm = torch.empty((total_assignments, dim), device=routed_heads.device, dtype=routed_heads.dtype)
        assignment_ids = torch.empty((total_assignments,), device=routed_heads.device, dtype=torch.int64)
        reverse_positions = torch.empty((total_assignments,), device=routed_heads.device, dtype=torch.int64)
        weights_perm = torch.empty((total_assignments,), device=routed_heads.device, dtype=topk_weights.dtype)
        write_counts = torch.zeros_like(counts)

        block_a = 16
        block_d = _next_power_of_2(dim)
        _bucket_dispatch_kernel[(triton.cdiv(total_assignments, block_a),)](
            routed_heads,
            flat_experts,
            flat_weights,
            offsets,
            write_counts,
            x_perm,
            assignment_ids,
            reverse_positions,
            weights_perm,
            total_assignments,
            dim,
            top_k,
            routed_heads.stride(0),
            block_a,
            block_d,
            num_warps=8,
        )

        ctx.save_for_backward(assignment_ids)
        ctx.routed_shape = tuple(routed_heads.shape)
        ctx.topk_shape = tuple(topk_indices.shape)
        ctx.top_k = int(top_k)
        ctx.dim = int(dim)
        return x_perm, assignment_ids, reverse_positions, weights_perm, counts

    @staticmethod
    def backward(ctx, grad_x_perm, grad_assignment_ids, grad_reverse_positions, grad_weights_perm, grad_counts):
        (assignment_ids,) = ctx.saved_tensors
        num_rows, dim = ctx.routed_shape
        total_assignments = assignment_ids.numel()
        top_k = ctx.top_k
        block_a = 16
        block_d = _next_power_of_2(dim)

        grad_x = None
        has_grad_x = grad_x_perm is not None
        if has_grad_x:
            grad_x = torch.zeros((num_rows, dim), device=grad_x_perm.device, dtype=grad_x_perm.dtype)

        grad_weights = None
        has_grad_weights = grad_weights_perm is not None
        if has_grad_weights:
            grad_weights = torch.empty((total_assignments,), device=grad_weights_perm.device, dtype=grad_weights_perm.dtype)

        if has_grad_x or has_grad_weights:
            _bucket_dispatch_backward_kernel[(triton.cdiv(total_assignments, block_a),)](
                grad_x_perm if has_grad_x else torch.empty((1, 1), device=assignment_ids.device, dtype=torch.float32),
                grad_weights_perm if has_grad_weights else torch.empty((1,), device=assignment_ids.device, dtype=torch.float32),
                assignment_ids,
                grad_x if has_grad_x else torch.empty((1, 1), device=assignment_ids.device, dtype=torch.float32),
                grad_weights if has_grad_weights else torch.empty((1,), device=assignment_ids.device, dtype=torch.float32),
                total_assignments,
                dim,
                top_k,
                dim,
                has_grad_x,
                has_grad_weights,
                block_a,
                block_d,
                num_warps=8,
            )

        if grad_weights is not None:
            grad_weights = grad_weights.reshape(ctx.topk_shape)
        return grad_x, None, grad_weights, None


class _CapacityDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        routed_heads: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        capacity: int,
    ):
        if not triton_moe_kernels_available() or not routed_heads.is_cuda:
            raise RuntimeError("Triton MoE capacity dispatch requires a CUDA tensor and Triton.")
        if not routed_heads.is_contiguous():
            routed_heads = routed_heads.contiguous()
        if not topk_indices.is_contiguous():
            topk_indices = topk_indices.contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        num_rows, dim = routed_heads.shape
        top_k = topk_indices.shape[1]
        total_assignments = topk_indices.numel()
        total_capacity = int(num_experts) * int(capacity)
        flat_experts = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        x_perm = torch.zeros((total_capacity, dim), device=routed_heads.device, dtype=routed_heads.dtype)
        assignment_ids = torch.full((total_capacity,), -1, device=routed_heads.device, dtype=torch.int64)
        reverse_positions = torch.full((total_assignments,), -1, device=routed_heads.device, dtype=torch.int64)
        weights_perm = torch.zeros((total_capacity,), device=routed_heads.device, dtype=topk_weights.dtype)
        write_counts = torch.zeros((int(num_experts),), device=routed_heads.device, dtype=torch.int32)
        overflow = torch.zeros((1,), device=routed_heads.device, dtype=torch.int32)

        block_a = 16
        block_d = _next_power_of_2(dim)
        _capacity_dispatch_kernel[(triton.cdiv(total_assignments, block_a),)](
            routed_heads,
            flat_experts,
            flat_weights,
            write_counts,
            overflow,
            x_perm,
            assignment_ids,
            reverse_positions,
            weights_perm,
            total_assignments,
            dim,
            top_k,
            int(capacity),
            routed_heads.stride(0),
            block_a,
            block_d,
            num_warps=8,
        )

        ctx.save_for_backward(assignment_ids)
        ctx.routed_shape = tuple(routed_heads.shape)
        ctx.topk_shape = tuple(topk_indices.shape)
        ctx.top_k = int(top_k)
        ctx.dim = int(dim)
        return x_perm, assignment_ids, reverse_positions, weights_perm, overflow

    @staticmethod
    def backward(ctx, grad_x_perm, grad_assignment_ids, grad_reverse_positions, grad_weights_perm, grad_overflow):
        (assignment_ids,) = ctx.saved_tensors
        num_rows, dim = ctx.routed_shape
        total_capacity = assignment_ids.numel()
        top_k = ctx.top_k
        block_a = 16
        block_d = _next_power_of_2(dim)

        grad_x = None
        has_grad_x = grad_x_perm is not None
        if has_grad_x:
            grad_x = torch.zeros((num_rows, dim), device=grad_x_perm.device, dtype=grad_x_perm.dtype)

        grad_weights = None
        has_grad_weights = grad_weights_perm is not None
        if has_grad_weights:
            grad_weights = torch.empty((ctx.topk_shape[0] * ctx.topk_shape[1],), device=grad_weights_perm.device, dtype=grad_weights_perm.dtype)

        if has_grad_x or has_grad_weights:
            _bucket_dispatch_backward_kernel[(triton.cdiv(total_capacity, block_a),)](
                grad_x_perm if has_grad_x else torch.empty((1, 1), device=assignment_ids.device, dtype=torch.float32),
                grad_weights_perm if has_grad_weights else torch.empty((1,), device=assignment_ids.device, dtype=torch.float32),
                assignment_ids,
                grad_x if has_grad_x else torch.empty((1, 1), device=assignment_ids.device, dtype=torch.float32),
                grad_weights if has_grad_weights else torch.empty((1,), device=assignment_ids.device, dtype=torch.float32),
                total_capacity,
                dim,
                top_k,
                dim,
                has_grad_x,
                has_grad_weights,
                block_a,
                block_d,
                num_warps=8,
            )

        if grad_weights is not None:
            grad_weights = grad_weights.reshape(ctx.topk_shape)
        return grad_x, None, grad_weights, None, None


class _WeightedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y_perm: torch.Tensor,
        assignment_ids: torch.Tensor,
        weights: torch.Tensor,
        output_rows: int,
        top_k: int,
    ) -> torch.Tensor:
        if not triton_moe_kernels_available() or not y_perm.is_cuda:
            raise RuntimeError("Triton weighted unpermute requires a CUDA tensor and Triton.")
        if not y_perm.is_contiguous():
            y_perm = y_perm.contiguous()
        if not assignment_ids.is_contiguous():
            assignment_ids = assignment_ids.contiguous()
        if not weights.is_contiguous():
            weights = weights.contiguous()
        total_assignments, dim = y_perm.shape
        out = torch.zeros((int(output_rows), dim), device=y_perm.device, dtype=y_perm.dtype)
        block_a = 16
        block_d = _next_power_of_2(dim)
        _weighted_unpermute_kernel[(triton.cdiv(total_assignments, block_a),)](
            y_perm,
            assignment_ids,
            weights,
            out,
            total_assignments,
            dim,
            int(top_k),
            block_a,
            block_d,
            num_warps=8,
        )
        ctx.save_for_backward(y_perm, assignment_ids, weights)
        ctx.output_rows = int(output_rows)
        ctx.top_k = int(top_k)
        ctx.dim = int(dim)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y_perm, assignment_ids, weights = ctx.saved_tensors
        total_assignments, dim = y_perm.shape
        block_a = 16
        block_d = _next_power_of_2(dim)
        grad_y = torch.zeros_like(y_perm)
        grad_weights = torch.zeros_like(weights)
        _weighted_unpermute_backward_kernel[(triton.cdiv(total_assignments, block_a),)](
            grad_out.contiguous(),
            y_perm,
            weights,
            assignment_ids,
            grad_y,
            grad_weights,
            total_assignments,
            dim,
            ctx.top_k,
            True,
            True,
            block_a,
            block_d,
            num_warps=8,
        )
        return grad_y, None, grad_weights, None, None


class _ReverseWeightedCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y_perm: torch.Tensor,
        reverse_positions: torch.Tensor,
        topk_weights: torch.Tensor,
        output_rows: int,
        top_k: int,
    ) -> torch.Tensor:
        if not triton_moe_kernels_available() or not y_perm.is_cuda:
            raise RuntimeError("Triton reverse weighted combine requires a CUDA tensor and Triton.")
        if not y_perm.is_contiguous():
            y_perm = y_perm.contiguous()
        if not reverse_positions.is_contiguous():
            reverse_positions = reverse_positions.contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()
        flat_weights = topk_weights.reshape(-1)
        total_rows, dim = y_perm.shape
        out = torch.empty((int(output_rows), dim), device=y_perm.device, dtype=y_perm.dtype)
        block_m = 16
        block_d = _next_power_of_2(dim)
        _reverse_weighted_combine_kernel[(triton.cdiv(int(output_rows), block_m),)](
            y_perm,
            reverse_positions,
            flat_weights,
            out,
            int(output_rows),
            dim,
            int(top_k),
            block_m,
            block_d,
            num_warps=8,
        )
        ctx.save_for_backward(y_perm, reverse_positions, flat_weights)
        ctx.topk_shape = tuple(topk_weights.shape)
        ctx.output_rows = int(output_rows)
        ctx.top_k = int(top_k)
        ctx.dim = int(dim)
        ctx.total_rows = int(total_rows)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y_perm, reverse_positions, flat_weights = ctx.saved_tensors
        block_m = 16
        block_d = _next_power_of_2(ctx.dim)
        grad_y = torch.zeros_like(y_perm)
        grad_weights = torch.zeros_like(flat_weights)
        _reverse_weighted_combine_backward_kernel[(triton.cdiv(ctx.output_rows, block_m),)](
            grad_out.contiguous(),
            y_perm,
            reverse_positions,
            flat_weights,
            grad_y,
            grad_weights,
            ctx.output_rows,
            ctx.dim,
            ctx.top_k,
            True,
            True,
            block_m,
            block_d,
            num_warps=8,
        )
        return grad_y, None, grad_weights.reshape(ctx.topk_shape), None, None


def bucket_dispatch(
    routed_heads: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    x_perm, assignment_ids, reverse_positions, weights_perm, counts = _BucketDispatch.apply(
        routed_heads,
        topk_indices,
        topk_weights,
        int(num_experts),
    )
    tokens_per_expert = [int(value) for value in counts.cpu().tolist()]
    return x_perm, assignment_ids, reverse_positions, weights_perm, tokens_per_expert


def bucket_dispatch_counts(
    routed_heads: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _BucketDispatch.apply(
        routed_heads,
        topk_indices,
        topk_weights,
        int(num_experts),
    )


def capacity_bucket_dispatch(
    routed_heads: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity: int,
    check_overflow: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], bool | None]:
    x_perm, assignment_ids, reverse_positions, weights_perm, overflow = _CapacityDispatch.apply(
        routed_heads,
        topk_indices,
        topk_weights,
        int(num_experts),
        int(capacity),
    )
    did_overflow = bool(overflow.item()) if check_overflow else None
    tokens_per_expert = [int(capacity)] * int(num_experts)
    return x_perm, assignment_ids, reverse_positions, weights_perm, tokens_per_expert, did_overflow


def weighted_unpermute(
    y_perm: torch.Tensor,
    assignment_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    output_rows: int,
    top_k: int,
) -> torch.Tensor:
    return _WeightedUnpermute.apply(y_perm, assignment_ids, weights, int(output_rows), int(top_k))


def unweighted_unpermute(
    y_perm: torch.Tensor,
    assignment_ids: torch.Tensor,
    *,
    output_rows: int,
    top_k: int,
) -> torch.Tensor:
    if y_perm.dim() != 2:
        raise ValueError("unweighted_unpermute expects y_perm with shape [assignments, hidden].")
    valid = assignment_ids >= 0
    if not bool(valid.any().item()):
        return y_perm.new_zeros((int(output_rows), y_perm.shape[-1]))
    valid_assignment_ids = assignment_ids.masked_select(valid)
    row_indices = torch.div(valid_assignment_ids, int(top_k), rounding_mode="floor")
    out = y_perm.new_zeros((int(output_rows), y_perm.shape[-1]))
    out.index_add_(0, row_indices, y_perm.index_select(0, torch.nonzero(valid, as_tuple=False).squeeze(-1)))
    return out


def reverse_weighted_combine(
    y_perm: torch.Tensor,
    reverse_positions: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output_rows: int,
    top_k: int,
) -> torch.Tensor:
    return _ReverseWeightedCombine.apply(y_perm, reverse_positions, topk_weights, int(output_rows), int(top_k))


def reverse_unweighted_combine(
    y_perm: torch.Tensor,
    reverse_positions: torch.Tensor,
    *,
    output_rows: int,
    top_k: int,
) -> torch.Tensor:
    if y_perm.dim() != 2:
        raise ValueError("reverse_unweighted_combine expects y_perm with shape [assignments, hidden].")
    output_rows = int(output_rows)
    top_k = int(top_k)
    positions = reverse_positions.reshape(output_rows, top_k)
    out = y_perm.new_zeros((output_rows, y_perm.shape[-1]))
    for slot in range(top_k):
        slot_positions = positions[:, slot]
        valid = slot_positions >= 0
        if bool(valid.any().item()):
            rows = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            out.index_add_(0, rows, y_perm.index_select(0, slot_positions.index_select(0, rows)))
    return out


def fused_swiglu(gate_up: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return _FusedSwiGLU.apply(gate_up, int(hidden_size))
