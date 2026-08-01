"""Context parallelism for the Metis-1.6 recurrent stack.

Context extension is the one stage where Metis-1.6 does not fit.  A physical
pass at 163,840 tokens keeps four mHC streams alive per layer, and the peak
activation footprint lands at roughly 137 GiB for Praxis and 287 GiB for Logos
against 128 GB of unified HBM3 per MI300A.  Both numbers are *per rank*, so
adding APUs does not help: data parallelism replicates the problem instead of
dividing it.

Two mechanisms divide it.  Layer-level activation recompute (see
``Metis16ForCausalLM._run_physical_pass``) keeps one block live at a time
instead of one whole pass.  Context parallelism, implemented here, shards the
sequence itself across a process group so each rank owns ``S / CP`` contiguous
tokens.  Together they bring Logos to roughly 63 GiB at ``CP=4``.

Design decisions worth stating once, because they constrain everything below:

**Shards are contiguous, not striped.**  Mamba-2's SSD recurrence carries state
strictly left to right, so a rank must own an interval to have a well-defined
incoming state.  Striped ("zigzag") sharding would balance causal attention
work perfectly but would destroy that property.  The cost of choosing Mamba's
side is that causal attention is imbalanced: rank ``r`` attends over roughly
``(r+1)/CP`` of the sequence, so the last rank does ``2·CP/(CP+1)`` times the
mean work — 1.6x at ``CP=4``.  Attention is a minority of the stack (3 of 20
layers on Logos), so this is a step-time penalty in the high teens, not a
factor.  A striped layout for the attention layers alone, with a permutation
all-to-all on either side, would recover it and is left as an optimisation.

**Gradients cross shard boundaries exactly.**  The obvious cheap implementation
detaches the incoming SSM state and truncates backpropagation at every shard
edge.  For a *context extension* run that is precisely the wrong corner to cut:
the run exists to teach long-range dependency, and truncating BPTT at 32k
boundaries teaches the opposite.  Every cross-rank tensor here therefore moves
through :class:`_AllGather`, whose backward is a reduce-scatter, so autograd
carries gradient back to the owning rank without any special-casing.

**Shard summaries are closed-form.**  Passing SSM state across ranks needs each
shard's cumulative decay and its zero-initial-state final state.  Both come out
of a single einsum over the shard rather than an extra scan, so the state
exchange costs one state-sized reduction per layer, not a second Mamba forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = [
    "ContextParallelAttentionLayout",
    "ContextParallelContext",
    "align_document_ids",
    "all_gather_differentiable",
    "all_gather_padded_differentiable",
    "build_context_parallel_attention_layout",
    "conv_left_halo",
    "gather_context_parallel_kv",
    "keep_graph_edge",
    "left_halo",
    "global_segment_stride",
    "mamba_incoming_state",
    "mamba_shard_summary",
    "packed_segment_keys",
    "reference_context_parallel_attention",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextParallelContext:
    """Placement of one rank inside a context-parallel group.

    ``local_length`` is the number of sequence positions this rank owns and is
    identical on every member of the group; uneven shards are rejected at
    construction so every collective below can assume equal shapes.
    """

    group: Any
    size: int
    rank: int
    local_length: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("Context-parallel size must be at least 1.")
        if not 0 <= self.rank < self.size:
            raise ValueError("Context-parallel rank is outside its group.")
        if self.local_length < 0:
            raise ValueError("Context-parallel local length must be non-negative.")
        if self.size > 1 and self.group is None:
            raise ValueError(
                "Context parallelism above size 1 requires an explicit process group."
            )

    @property
    def enabled(self) -> bool:
        return self.size > 1

    @property
    def global_length(self) -> int:
        return self.local_length * self.size

    @property
    def global_offset(self) -> int:
        return self.rank * self.local_length

    @classmethod
    def disabled(cls, local_length: int = 0) -> "ContextParallelContext":
        return cls(group=None, size=1, rank=0, local_length=local_length)

    def with_local_length(self, local_length: int) -> "ContextParallelContext":
        return ContextParallelContext(
            group=self.group,
            size=self.size,
            rank=self.rank,
            local_length=local_length,
        )



# ---------------------------------------------------------------------------
# Differentiable collectives
# ---------------------------------------------------------------------------


def _reduce_scatter(output: Tensor, source: Tensor, group: Any) -> None:
    """Sum-reduce ``source`` across the group and scatter chunks along dim 0.

    ``reduce_scatter_tensor`` is the right primitive and every ROCm/RCCL build
    has it.  Gloo, which the CPU correctness lane uses, has not always exposed
    it; the fallback is an all-reduce plus a narrow, which is numerically
    identical and only wasteful in bandwidth.
    """

    try:
        dist.reduce_scatter_tensor(output, source, op=dist.ReduceOp.SUM, group=group)
        return
    except (RuntimeError, AttributeError, NotImplementedError):
        pass
    reduced = source.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
    rank = dist.get_rank(group=group)
    chunk = output.shape[0]
    output.copy_(reduced.narrow(0, rank * chunk, chunk))


class _AllGather(torch.autograd.Function):
    """All-gather along dim 0 whose backward is the matching reduce-scatter.

    Every rank's output depends on every rank's input, so the gradient of one
    rank's input is the *sum* over ranks of the corresponding output-gradient
    slice.  That is a reduce-scatter, not a slice, and getting it wrong silently
    scales cross-shard gradients by ``1/CP``.
    """

    @staticmethod
    def forward(ctx, tensor: Tensor, group: Any, size: int) -> Tensor:  # type: ignore[override]
        ctx.group = group
        ctx.size = size
        ctx.input_shape = tuple(tensor.shape)
        source = tensor.contiguous()
        output = source.new_empty((source.shape[0] * size, *source.shape[1:]))
        dist.all_gather_into_tensor(output, source, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_empty(ctx.input_shape)
        _reduce_scatter(grad_input, grad_output, ctx.group)
        return grad_input, None, None


def all_gather_differentiable(tensor: Tensor, context: ContextParallelContext) -> Tensor:
    """Gather equal-shaped shards into ``[size * n, ...]`` in rank order.

    Returns a view of the local tensor with a leading singleton-group axis when
    context parallelism is off, so callers do not need a separate code path.
    """

    if not context.enabled:
        return tensor
    if tensor.is_floating_point() and tensor.requires_grad:
        return _AllGather.apply(tensor, context.group, context.size)
    source = tensor.contiguous()
    output = source.new_empty((source.shape[0] * context.size, *source.shape[1:]))
    dist.all_gather_into_tensor(output, source, group=context.group)
    return output


def all_gather_padded_differentiable(
    tensor: Tensor,
    context: ContextParallelContext,
    *,
    capacity: int,
) -> tuple[Tensor, Tensor]:
    """Gather variable-length dim-0 shards by padding each to ``capacity``.

    Continuation packing leaves a different number of active tokens on every
    rank, so the attention key gather cannot assume equal shapes.  Returns the
    padded gather ``[size * capacity, ...]`` together with the per-rank row
    counts, and leaves it to the caller to select the valid rows.
    """

    count = int(tensor.shape[0])
    if count > capacity:
        raise ValueError("Padded gather capacity is smaller than the local shard.")
    if not context.enabled:
        counts = torch.tensor([count], device=tensor.device, dtype=torch.long)
        return tensor, counts

    counts = torch.zeros(context.size, device=tensor.device, dtype=torch.long)
    counts[context.rank] = count
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=context.group)

    padded = tensor.new_zeros((capacity, *tensor.shape[1:]))
    if count:
        padded = torch.cat((tensor, padded[count:]), dim=0) if count < capacity else tensor
    gathered = all_gather_differentiable(padded, context)
    return gathered, counts


def keep_graph_edge(value: Tensor, *unused: Tensor) -> Tensor:
    """Attach zero-valued autograd edges for gathered tensors this rank ignores.

    A differentiable collective only issues its backward when its output is
    reachable from the loss.  That reachability is *rank dependent* here: rank 0
    has no predecessor, so it discards the halo it just gathered; a rank with no
    surviving tokens discards the keys it just gathered.  Those ranks would then
    skip a reduce-scatter that every other rank in the group still issues, and
    the job would hang on the mismatch rather than fail on it -- at step
    fourteen thousand, in the middle of the night, with no stack to read.

    Adding a zero-weighted edge costs one scalar and makes the backward
    collective sequence identical on every rank by construction.
    """

    for tensor in unused:
        if tensor.requires_grad and tensor.is_floating_point():
            # One element is enough to make the node reachable; the rest of the
            # gradient is zero-filled by the slice's own backward.
            value = value + tensor.flatten()[:1].sum() * 0
    return value




# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------


def align_document_ids(
    input_ids: Tensor,
    *,
    document_ids: Tensor | None,
    reset_mask: Tensor | None,
    context: ContextParallelContext,
) -> tuple[Tensor | None, Tensor | None]:
    """Derive globally-consistent document ids for one sequence shard.

    The single-rank derivation forces position 0 to begin a document and runs
    its boundary cumsum over whatever slice it was handed.  Both assumptions
    break the moment a sequence is sharded: rank 2's position 0 is usually the
    middle of a document rank 1 also holds, and rank 2's local ids restart at
    zero.  Left alone, a document that straddles a boundary acquires two
    identities and attention refuses to let its second half read its first --
    which is the specific dependency a context-extension run exists to teach.

    Both supported input forms are resolved exactly, without guessing:

    * **Explicit ``document_ids``** are already a shard of a global tensor, so
      they need no renumbering.  Only the reset at position 0 is unknown
      locally, and it is recovered by comparing against the predecessor's final
      id -- one gather of ``[batch]`` values.
    * **Explicit ``reset_mask``** is likewise a shard of a global tensor, so
      position 0 already carries the truth.  Global id is the running count of
      resets, so offsetting each rank's local cumsum by the reset count on all
      preceding ranks reproduces the global numbering; a straddling document
      lands on one id from both sides because the second shard contributes no
      reset of its own at position 0.

    With neither, there are no document boundaries and the shard is part of one
    sequence, which needs no alignment at all.
    """

    if not context.enabled:
        raise ValueError("align_document_ids requires an enabled context-parallel group.")
    if document_ids is not None:
        if document_ids.shape != input_ids.shape:
            raise ValueError("document_ids must have the same shape as input_ids.")
        if reset_mask is None:
            reset_mask = torch.ones_like(input_ids, dtype=torch.bool)
            reset_mask[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
            reset_mask[:, 0] = _starts_new_document(document_ids, context)
        return document_ids, reset_mask.to(torch.bool)
    if reset_mask is None:
        return None, None
    if reset_mask.shape != input_ids.shape:
        raise ValueError("reset_mask must have the same shape as input_ids.")

    local_reset = reset_mask.to(torch.bool).clone()
    if context.rank == 0:
        local_reset[:, 0] = True
    local_counts = local_reset.sum(dim=1, dtype=torch.long)
    gathered = all_gather_differentiable(
        local_counts.unsqueeze(0).contiguous(), context
    )
    offsets = (
        gathered[: context.rank].sum(dim=0)
        if context.rank
        else torch.zeros_like(local_counts)
    )
    aligned = local_reset.to(torch.long).cumsum(dim=1) - 1 + offsets.unsqueeze(1)
    return aligned, local_reset


def _starts_new_document(
    document_ids: Tensor, context: ContextParallelContext
) -> Tensor:
    """Whether each batch row's shard-local position 0 opens a new document."""

    if context.rank == 0:
        return torch.ones(
            document_ids.shape[0], device=document_ids.device, dtype=torch.bool
        )
    gathered = all_gather_differentiable(
        document_ids[:, -1].unsqueeze(0).contiguous(), context
    )
    return document_ids[:, 0] != gathered[context.rank - 1]


def global_segment_stride(
    document_ids: Tensor | None, context: ContextParallelContext
) -> int:
    """Stride wide enough to fold a batch row into a document id without collision.

    Read back to the host once per forward rather than per layer; the alternative
    is comparing ``(row, document)`` pairs, which no bucketize primitive accepts.
    """

    if document_ids is None:
        return 1
    largest = document_ids.max().to(torch.long).reshape(1)
    if context.enabled:
        dist.all_reduce(largest, op=dist.ReduceOp.MAX, group=context.group)
    return int(largest.item()) + 1


# ---------------------------------------------------------------------------
# Mamba-2 state passing
# ---------------------------------------------------------------------------


def mamba_shard_summary(
    x: Tensor,
    b_matrix: Tensor,
    delta: Tensor,
    a_log: Tensor,
    *,
    reset_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Closed-form ``(decay, state)`` summary of one shard's SSD recurrence.

    The SSD recurrence is ``S_t = a_t · S_{t-1} + Δ_t x_t B_tᵀ`` with a scalar
    per-head gate ``a_t = exp(Δ_t A_h)``.  Composing it over a shard of length
    ``T`` gives ``S_T = Λ · S_0 + Σ`` where

    ``Λ = exp(A_h · Σ_t Δ_t)``   and   ``Σ = Σ_t (Π_{s>t} a_s) Δ_t x_t B_tᵀ``.

    Both fall out of one cumulative sum of ``Δ`` and one einsum, so a rank can
    publish what its successors need without running a second scan.  ``A_h`` is
    negative and the exponents are differences of a monotone cumsum, so every
    exponential here is bounded above by 1 and the whole thing is stable in
    fp32 at 40k-token shards.

    Document resets truncate both quantities.  A reset anywhere in the shard
    means no incoming state survives to the end, so ``Λ`` is zero; and only the
    final segment contributes to ``Σ``.

    Args:
        x: ``[batch, seq, heads, head_dim]`` SSD input.
        b_matrix: ``[batch, seq, heads, state]`` input projection.
        delta: ``[batch, seq, heads]`` positive step sizes (post-softplus).
        a_log: ``[heads]`` log-negated state decay; ``A_h = -exp(a_log)``.
        reset_mask: ``[batch, seq]`` marking positions whose incoming state is
            zeroed, or ``None`` when the shard holds a single document.

    Returns:
        ``(decay, state)`` shaped ``[batch, heads]`` and
        ``[batch, heads, head_dim, state]``.
    """

    if delta.ndim != 3:
        raise ValueError("delta must have shape [batch, seq, heads].")
    # Accumulate in at least fp32: the cumulative sum runs the length of a whole
    # shard, and in bf16 it would lose the small steps that a long decay is made
    # of.  Promote rather than cast, so an fp64 correctness probe stays fp64.
    accumulator = torch.promote_types(delta.dtype, torch.float32)
    decay_rate = -torch.exp(a_log.to(accumulator))  # [heads], strictly negative
    delta = delta.to(accumulator)

    if reset_mask is not None:
        segment = reset_mask.to(torch.long).cumsum(dim=1)
        in_final_segment = segment == segment[:, -1:]
        delta = delta * in_final_segment.unsqueeze(-1).to(delta.dtype)
        no_reset = segment[:, -1] == 0  # [batch]
    else:
        in_final_segment = None
        no_reset = None

    cumulative = delta.cumsum(dim=1)  # [batch, seq, heads]
    total = cumulative[:, -1]  # [batch, heads]

    # exp(A · (total - cumulative)) is the decay applied from position t to the
    # shard end; the argument is <= 0 everywhere, so this cannot overflow.
    tail_decay = torch.exp(
        decay_rate.view(1, 1, -1) * (total.unsqueeze(1) - cumulative)
    )
    weight = tail_decay * delta
    if in_final_segment is not None:
        weight = weight * in_final_segment.unsqueeze(-1).to(weight.dtype)

    state = torch.einsum(
        "bth,bthp,bthn->bhpn",
        weight,
        x.to(accumulator),
        b_matrix.to(accumulator),
    )
    decay = torch.exp(decay_rate.view(1, -1) * total)
    if no_reset is not None:
        decay = decay * no_reset.unsqueeze(-1).to(decay.dtype)
    return decay, state


def mamba_incoming_state(
    decay: Tensor,
    state: Tensor,
    context: ContextParallelContext,
) -> Tensor:
    """Compose shard summaries into this rank's incoming SSM state.

    Runs the ``S_{r+1} = Λ_r S_r + Σ_r`` recurrence over the gathered shard
    summaries, which is a scan of length ``CP`` over a few megabytes rather than
    a serial dependency over ``CP`` full shards.  Every rank evaluates the same
    scan on the same gathered tensors, so the result is bit-identical across the
    group and no rank waits on its predecessor.

    The gather is differentiable, so the gradient that the local scan produces
    for this incoming state flows back to the ranks that actually own the
    parameters that produced it.  Detaching here instead would truncate
    backpropagation at every shard boundary — cheap, and exactly the dependency
    a context-extension run is trying to learn.
    """

    if not context.enabled:
        return torch.zeros_like(state)

    gathered_decay = all_gather_differentiable(
        decay.unsqueeze(0).contiguous(), context
    )
    gathered_state = all_gather_differentiable(
        state.unsqueeze(0).contiguous(), context
    )
    incoming = torch.zeros_like(state)
    for source in range(context.rank):
        incoming = (
            gathered_decay[source].unsqueeze(-1).unsqueeze(-1) * incoming
            + gathered_state[source]
        )
    if context.rank == 0:
        # Rank 0 has no predecessor and consumes neither gather.
        incoming = keep_graph_edge(incoming, gathered_decay, gathered_state)
    return incoming


def left_halo(
    values: Tensor,
    context: ContextParallelContext,
    *,
    width: int,
    fill: float | int | bool = 0,
) -> Tensor:
    """The predecessor shard's final ``width`` positions along dim 1.

    Every causal operator with a bounded left reach needs this: the Mamba-2
    short convolution looks back ``d_conv - 1`` positions, and the N-gram memory
    hashes ``order - 1`` tokens back.  Without the exchange those leading
    positions convolve or hash against zero padding on every shard but the
    first -- a corruption that is small, silent, permanent, and invisible in a
    loss curve.

    Rank 0 has no predecessor and receives ``fill``, which reproduces the
    unsharded behaviour at the true start of the sequence.

    Args:
        values: ``[batch, seq, ...]`` shard-local tensor.
        width: positions of left context required.
        fill: value handed to rank 0 in place of a predecessor.
    """

    halo = max(int(width), 0)
    tail_shape = (values.shape[0], halo, *values.shape[2:])
    if halo == 0:
        return values.new_zeros(tail_shape)
    if not context.enabled:
        return torch.full(tail_shape, fill, device=values.device, dtype=values.dtype)

    if values.shape[1] >= halo:
        tail = values[:, values.shape[1] - halo :]
    else:
        pad = torch.full(
            (values.shape[0], halo - values.shape[1], *values.shape[2:]),
            fill,
            device=values.device,
            dtype=values.dtype,
        )
        tail = torch.cat((pad, values), dim=1)
    gathered = all_gather_differentiable(tail.contiguous().unsqueeze(0), context)
    if context.rank == 0:
        return keep_graph_edge(torch.full_like(tail, fill), gathered)
    return gathered[context.rank - 1]


def conv_left_halo(
    values: Tensor,
    context: ContextParallelContext,
    *,
    width: int,
) -> Tensor:
    """Left context for a causal depthwise convolution of kernel ``width``."""

    return left_halo(values, context, width=max(int(width) - 1, 0), fill=0)


# ---------------------------------------------------------------------------
# Attention key/value gather
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextParallelAttentionLayout:
    """Variable-length attention layout for one context-parallel rank.

    ``key_indices`` selects, out of the group-wide gathered key/value buffer,
    exactly the keys the local queries are allowed to see, ordered so that each
    document's keys form one contiguous run.  ``cu_seqlens_k`` then delimits
    those runs and ``cu_seqlens_q`` delimits the matching local query runs.

    Per document the selected keys span from the document's first token up to
    and including the last local query in it.  That truncation is what makes
    bottom-right causal alignment correct for a middle shard: the local query
    block becomes the final ``len_q`` rows of the truncated attention matrix, so
    query ``i`` may see keys ``[0, len_k - len_q + i]``, which is precisely its
    own causal prefix.  Bottom-right is what FlashAttention does whenever
    ``seqlen_q != seqlen_k``; feeding it untruncated keys would instead let the
    first shard's queries read the future.
    """

    key_indices: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    query_count: int

    @property
    def empty(self) -> bool:
        return self.query_count == 0


def packed_segment_keys(
    document_ids: Tensor | None,
    flat_token_indices: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    stride: int,
) -> Tensor:
    """Composite ``(batch row, document)`` identity for each packed token.

    Document ids restart per batch row, so batch 0's document 3 and batch 1's
    document 3 compare equal — harmless while attention is built from run
    boundaries inside a single packed buffer, and a correctness bug the moment
    identity is compared across gathered shards.  Folding the batch row in with
    a stride wider than any document id keeps the key monotone along packed
    order, which every run-length computation downstream depends on.
    """

    device = flat_token_indices.device
    if flat_token_indices.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.long)
    batch_rows = torch.div(flat_token_indices, sequence_length, rounding_mode="floor")
    if document_ids is None:
        return batch_rows * stride
    flat_documents = document_ids.reshape(-1).index_select(0, flat_token_indices)
    return batch_rows * stride + flat_documents


def gather_context_parallel_kv(
    key: Tensor,
    value: Tensor,
    context: ContextParallelContext,
    *,
    capacity: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather packed keys and values from every rank in the group.

    Queries stay local so attention FLOPs shard with the sequence; only keys and
    values are replicated.  With grouped-query attention the key/value tensor is
    a small fraction of the layer's activation, so this trades a few hundred
    megabytes of intra-node traffic for the whole memory win.

    Returns ``(keys, values, counts)`` where the first two are padded to
    ``size * capacity`` rows in rank order and ``counts`` gives the valid rows
    per rank.
    """

    gathered_key, counts = all_gather_padded_differentiable(
        key, context, capacity=capacity
    )
    gathered_value, _ = all_gather_padded_differentiable(
        value, context, capacity=capacity
    )
    return gathered_key, gathered_value, counts


def build_context_parallel_attention_layout(
    *,
    local_segments: Tensor,
    gathered_segments: Tensor,
    counts: Tensor,
    context: ContextParallelContext,
    capacity: int,
) -> ContextParallelAttentionLayout:
    """Build this rank's truncated key layout, without a host synchronisation.

    The obvious implementation loops over documents, masks the gathered buffer
    once per document and reads two lengths back to the host.  At three
    attention layers times five passes that is thousands of stalls per step for
    work that is pure integer bookkeeping.  Everything here is therefore a
    single vectorised pass, and the only value read back is the run count.

    Correctness rests on one invariant: the gathered buffer is in global token
    order.  Packing preserves order within a rank, ranks own contiguous
    intervals, and the gather concatenates in rank order — so gathered position
    is monotone in global position, and "keys this query may see" reduces to a
    prefix test on that position.

    Args:
        local_segments: ``[n_local]`` composite segment key per packed local row.
        gathered_segments: ``[size * capacity]`` the same, gathered, with
            padding rows carrying a sentinel of ``-1``.
        counts: ``[size]`` valid packed rows per rank.
        capacity: padded rows per rank inside the gathered buffers.
    """

    device = local_segments.device
    local_count = int(local_segments.numel())
    if local_count == 0:
        zero = torch.zeros(1, device=device, dtype=torch.int32)
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return ContextParallelAttentionLayout(
            key_indices=empty,
            cu_seqlens_q=zero,
            cu_seqlens_k=zero.clone(),
            max_seqlen_q=0,
            max_seqlen_k=0,
            query_count=0,
        )

    total_rows = int(gathered_segments.numel())
    positions = torch.arange(total_rows, device=device)
    row_rank = torch.div(positions, max(capacity, 1), rounding_mode="floor")
    row_slot = positions - row_rank * max(capacity, 1)
    selectable = row_slot < counts.index_select(0, row_rank)

    # The local queries, expressed as positions inside the gathered buffer, so
    # query and key bounds live in one coordinate system.
    query_positions = (
        torch.arange(local_count, device=device) + context.rank * max(capacity, 1)
    )

    # Distinct documents that own at least one local query, in packed order.
    run_starts = torch.ones(local_count, device=device, dtype=torch.bool)
    run_starts[1:] = local_segments[1:] != local_segments[:-1]
    run_index = run_starts.to(torch.long).cumsum(dim=0) - 1
    run_count = int(run_index[-1].item()) + 1
    run_segments = local_segments.masked_select(run_starts)
    query_lengths = torch.bincount(run_index, minlength=run_count)

    # Causal bound per document: the gathered position of its last local query.
    # A key qualifies when it shares the document and sits at or before that
    # bound, which subsumes the "no rank above mine" and "no future token on my
    # own rank" conditions in a single comparison.
    last_query_position = torch.full(
        (run_count,), -1, device=device, dtype=torch.long
    )
    last_query_position.scatter_reduce_(
        0, run_index, query_positions, reduce="amax", include_self=True
    )

    # Map every gathered row to the run that may consume it, or to no run.
    run_of_row = torch.bucketize(gathered_segments, run_segments, right=False)
    run_of_row = run_of_row.clamp_(max=run_count - 1)
    matches = selectable & (
        run_segments.index_select(0, run_of_row) == gathered_segments
    )
    bounds = last_query_position.index_select(0, run_of_row)
    allowed = matches & (positions <= bounds)

    key_positions = torch.nonzero(allowed, as_tuple=False).flatten()
    key_runs = run_of_row.index_select(0, key_positions)
    key_lengths = torch.bincount(key_runs, minlength=run_count)

    return ContextParallelAttentionLayout(
        key_indices=key_positions,
        cu_seqlens_q=_cumulative(query_lengths),
        cu_seqlens_k=_cumulative(key_lengths),
        max_seqlen_q=int(query_lengths.max().item()),
        max_seqlen_k=int(key_lengths.max().item()) if key_positions.numel() else 0,
        query_count=local_count,
    )


def _cumulative(lengths: Tensor) -> Tensor:
    zero = torch.zeros(1, device=lengths.device, dtype=torch.int64)
    return torch.cat((zero, lengths.to(torch.int64).cumsum(dim=0))).to(torch.int32)


def reference_context_parallel_attention(
    query: Tensor,
    gathered_key: Tensor,
    gathered_value: Tensor,
    *,
    local_segments: Tensor,
    gathered_segments: Tensor,
    counts: Tensor,
    context: ContextParallelContext,
    capacity: int,
    scale: float,
) -> Tensor:
    """Dense masked attention over the gathered keys — the correctness oracle.

    Materialises the full ``[queries, gathered keys]`` score matrix, which is
    exactly what context parallelism exists to avoid, so this is bounded to the
    CPU lane and to tests.  Its value is that the mask is written directly from
    the definition — same document, and not in the future — with none of the
    truncation and bottom-right-alignment reasoning the fused path relies on.
    If the two ever disagree, this one is right.

    Args:
        query: ``[n_local, heads, head_dim]`` local packed queries.
        gathered_key: ``[size * capacity, kv_heads, head_dim]``.
        gathered_value: same shape as ``gathered_key``.
        scale: softmax scale, normally ``head_dim ** -0.5``.
    """

    device = query.device
    local_count = int(query.shape[0])
    if local_count == 0:
        return query
    total_rows = int(gathered_segments.numel())
    positions = torch.arange(total_rows, device=device)
    row_rank = torch.div(positions, max(capacity, 1), rounding_mode="floor")
    row_slot = positions - row_rank * max(capacity, 1)
    selectable = row_slot < counts.index_select(0, row_rank)

    query_positions = (
        torch.arange(local_count, device=device) + context.rank * max(capacity, 1)
    )
    same_document = local_segments.unsqueeze(1) == gathered_segments.unsqueeze(0)
    not_future = query_positions.unsqueeze(1) >= positions.unsqueeze(0)
    allowed = same_document & not_future & selectable.unsqueeze(0)

    heads = int(query.shape[1])
    kv_heads = int(gathered_key.shape[1])
    if heads % kv_heads:
        raise ValueError("Attention head count must be a multiple of the KV head count.")
    repeats = heads // kv_heads
    key = gathered_key.repeat_interleave(repeats, dim=1)
    value = gathered_value.repeat_interleave(repeats, dim=1)

    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # A query whose entire row is masked (a padded or fully isolated token)
    # produces NaN from softmax over -inf; it contributes nothing downstream, so
    # zero it rather than letting the NaN travel.
    weights = torch.nan_to_num(weights, nan=0.0)
    attended = torch.einsum("hqk,khd->qhd", weights, value.float())
    return attended.to(query.dtype)
