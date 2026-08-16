from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from inspect import signature
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .context_parallel import (
    ContextParallelContext,
    align_document_ids,
    all_gather_differentiable,
    build_context_parallel_attention_layout,
    conv_left_halo,
    gather_context_parallel_kv,
    global_segment_stride,
    keep_graph_edge,
    left_halo,
    mamba_incoming_state,
    mamba_shard_summary,
    packed_segment_keys,
    reference_context_parallel_attention,
)
from .mhc_kernels import mhc_masked_write, mhc_read_mix
from .model_config import Metis16Config, load_family_config


Tensor = torch.Tensor
PLACEMENT_REPLICATED = "replicated"
PLACEMENT_EXPERT_SHARDED = "expert_sharded"
PLACEMENT_SPARSE_TABLE = "sparse_table"
PLACEMENT_ROW_SHARDED_TABLE = "row_sharded_table"
ROUTE_AUXILIARY_LOSS_NAMES = (
    "expert_balance",
    "expert_router_z",
    "routed_k_budget",
)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _group_world_size(group: Any = None) -> int:
    # In this API None is a deliberate local-only sentinel. Callers must pass
    # dist.group.WORLD explicitly when WORLD ownership is intended.
    return dist.get_world_size(group=group) if _dist_ready() and group is not None else 1


def _group_rank(group: Any = None) -> int:
    return dist.get_rank(group=group) if _dist_ready() and group is not None else 0


@torch.no_grad()
def _group_any_active(active_mask: Tensor, *, group: Any, groups: Sequence[Any] = ()) -> bool:
    """Whether any rank that must stay in lockstep still has a live token.

    Continuation is per token, so one rank can exhaust its active set several
    passes before its peers.  If it left the loop there it would stop issuing
    collectives while the others kept going, and the job would deadlock on the
    next gather rather than fail.  Every group whose members share a pass
    schedule -- the data-parallel world and, when the sequence is sharded, the
    context-parallel group -- therefore votes here.
    """

    indicator = torch.tensor(
        int(bool(active_mask.any().item())),
        device=active_mask.device,
        dtype=torch.int32,
    )
    seen: set[int] = set()
    for candidate in (group, *groups):
        if candidate is None or _group_world_size(candidate) <= 1:
            continue
        # A group can appear twice (the CP group is a subset of the world);
        # reducing over it twice is harmless but pointless.
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        dist.all_reduce(indicator, op=dist.ReduceOp.MAX, group=candidate)
    return bool(indicator.item())


def _make_linear(
    precision_policy: Any,
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    role: str,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    """Build a projection through the trainer's FP8-aware policy when supplied."""

    if precision_policy is not None and callable(getattr(precision_policy, "linear", None)):
        factory = precision_policy.linear
        kwargs = {
            "bias": bias,
            "role": role,
            "device": device,
            "dtype": dtype,
        }
        try:
            module = factory(in_features, out_features, **kwargs)
        except TypeError:
            # Keep compatibility with policies that own device/dtype placement.
            kwargs.pop("device")
            kwargs.pop("dtype")
            module = factory(in_features, out_features, **kwargs)
        if not isinstance(module, nn.Module):
            raise TypeError("precision_policy.linear(...) must return torch.nn.Module.")
    else:
        module = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    setattr(module, "metis_precision_role", role)
    return module


def _execution_context(
    precision_policy: Any,
    *,
    module: nn.Module | None = None,
):
    if precision_policy is None:
        return nullcontext()
    context_factory = getattr(precision_policy, "execution_context", None)
    if not callable(context_factory):
        return nullcontext()
    if "module" in signature(context_factory).parameters:
        return context_factory(module=module)
    return context_factory()


def _activation_checkpoint_context_fn(precision_policy: Any) -> Callable[[], tuple[Any, Any]]:
    factory = getattr(
        precision_policy,
        "activation_checkpoint_context_fn",
        None,
    )
    if callable(factory):
        return factory

    def contexts() -> tuple[Any, Any]:
        return nullcontext(), nullcontext()

    return contexts


def _fp32_linear(module: nn.Module, values: Tensor) -> Tensor:
    device_type = values.device.type
    if device_type in {"cuda", "cpu"}:
        with torch.autocast(device_type=device_type, enabled=False):
            return module(values.float())
    return module(values.float())


@dataclass(frozen=True)
class MetisProcessGroups:
    """Distributed groups used by the model core.

    ``world`` is the full family job and fixes the collective FP8-autocast
    schedule across expert replicas. ``expert`` is one EP group.
    ``table_lookup`` owns the deterministic row
    partition for row-sharded N-gram tables. ``table_gradient`` contains
    replicas of the same table (or same local table shard) and is the group
    over which sparse gradients must be synchronized. ``expert_data`` joins
    replicas of the same EP rank so non-gradient routing state remains
    identical across Logos' two expert replicas.
    """

    world: Any = None
    expert: Any = None
    expert_data: Any = None
    table_lookup: Any = None
    table_gradient: Any = None
    # ``context`` holds the ranks that jointly own one sequence during context
    # extension.  It is orthogonal to ``expert``: an EP group shards parameters
    # across the same sequence, a CP group shards one sequence across the same
    # parameters.
    context: Any = None


@dataclass(frozen=True)
class ContextParallelPassState:
    """Everything the mixers need to cross a sequence-shard boundary.

    Rebuilt once per recurrent pass rather than per layer, because continuation
    packing is what changes between passes: the number of active tokens differs
    per rank, so the gather capacity and the per-token document identity both
    have to be re-derived after each halt.
    """

    context: ContextParallelContext
    capacity: int
    local_count: int
    counts: Tensor
    local_segments: Tensor
    gathered_segments: Tensor
    layout: Any
    continues_previous: bool


def _build_context_parallel_pass_state(
    context: ContextParallelContext,
    *,
    document_ids: Tensor | None,
    selector: Tensor,
    batch_size: int,
    sequence_length: int,
    segment_stride: int,
    local_count: int,
) -> ContextParallelPassState:
    """Resolve the group-wide attention layout for one recurrent pass.

    Everything here depends on the packing, not on the layer, so it is built
    once per pass and shared by all attention layers -- three builds saved per
    pass on Logos.  Continuation is what forces the rebuild in the first place:
    tokens halt at different depths on different ranks, so both the number of
    live tokens and their document identities change after every halt.

    ``local_count`` is zero when this rank has no surviving tokens.  It still
    joins every collective -- a rank that skipped the gather because it ran out
    of work would desynchronise the group and hang it on the next pass.
    """

    device = selector.device
    segments = packed_segment_keys(
        document_ids,
        selector,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=segment_stride,
    )
    local_segments = segments[:local_count]

    counts = torch.zeros(context.size, device=device, dtype=torch.long)
    counts[context.rank] = local_count
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=context.group)
    capacity = max(int(counts.max().item()), 1)

    # -1 can never collide with a real (batch row, document) key, so padded
    # rows drop out of every downstream comparison without a separate mask.
    padded = torch.full((capacity,), -1, device=device, dtype=torch.long)
    if local_count:
        padded[:local_count] = local_segments
    gathered_segments = all_gather_differentiable(padded, context)

    layout = build_context_parallel_attention_layout(
        local_segments=local_segments,
        gathered_segments=gathered_segments,
        counts=counts,
        context=context,
        capacity=capacity,
    )
    return ContextParallelPassState(
        context=context,
        capacity=capacity,
        local_count=local_count,
        counts=counts,
        local_segments=local_segments,
        gathered_segments=gathered_segments,
        layout=layout,
        continues_previous=_continues_previous_shard(
            local_segments,
            gathered_segments,
            counts,
            context=context,
            capacity=capacity,
        ),
    )


def _continues_previous_shard(
    local_segments: Tensor,
    gathered_segments: Tensor,
    counts: Tensor,
    *,
    context: ContextParallelContext,
    capacity: int,
) -> bool:
    """Whether this rank's first live token continues the preceding rank's last.

    The preceding rank is the nearest one below this rank that still has live
    tokens -- continuation can empty a shard entirely, and when it does the
    document simply spans the gap.
    """

    if context.rank == 0 or int(local_segments.numel()) == 0:
        return False
    host_counts = counts.tolist()
    for source in range(context.rank - 1, -1, -1):
        available = int(host_counts[source])
        if available == 0:
            continue
        last = gathered_segments[source * capacity + available - 1]
        return bool((last == local_segments[0]).item())
    return False


class CollectiveEventTimer:
    """Optional HIP/CUDA-event timing without per-collective synchronization."""

    def __init__(self) -> None:
        self.enabled = False
        self._event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.enqueue_seconds = 0.0

    def reset(self) -> None:
        self._event_pairs.clear()
        self.enqueue_seconds = 0.0

    def begin(self, reference: Tensor) -> tuple[float, torch.cuda.Event | None]:
        host_start = time.perf_counter()
        event: torch.cuda.Event | None = None
        if self.enabled and reference.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return host_start, event

    def end(
        self,
        token: tuple[float, torch.cuda.Event | None],
        reference: Tensor,
    ) -> None:
        host_start, start_event = token
        self.enqueue_seconds += time.perf_counter() - host_start
        if start_event is not None and reference.is_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._event_pairs.append((start_event, end_event))

    def finalize(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        measured_seconds = 0.0
        if self._event_pairs:
            # One synchronization per explicitly sampled forward, never one
            # synchronization per collective.
            self._event_pairs[-1][1].synchronize()
            measured_seconds = sum(
                start.elapsed_time(end) for start, end in self._event_pairs
            ) / 1_000.0
        elif not reference.is_cuda:
            # CPU collectives are synchronous; host elapsed time is real.
            measured_seconds = self.enqueue_seconds
        return (
            reference.new_tensor(measured_seconds, dtype=torch.float64),
            reference.new_tensor(self.enqueue_seconds, dtype=torch.float64),
        )


@dataclass(frozen=True)
class CurriculumState:
    continuation_mode: str = "adaptive"
    routed_k_mode: str = "adaptive"
    fixed_routed_k: int = 4
    max_passes: int | None = None
    memory_gate_scale: float = 1.0
    stochastic_routing: bool = True
    temperature: float = 1.0
    target_mean_depth: float | None = None
    target_mean_routed_k: float | None = None
    # ``per_pass`` re-decides the expert coalition at every pass, which is the
    # MoRE pathway axis.  ``frozen`` reuses pass 1's coalition for the whole
    # token, isolating that axis at identical executed FLOPs.
    pathway_mode: str = "per_pass"
    # Seed for the random-policy controls.  These deliberately spend the same
    # mean budget as the learned policies while carrying no information, which
    # is what makes them the sharpest test of whether the learned allocation
    # does anything at all.
    random_policy_seed: int = 0
    # Optimizer step, folded into the random controls' seed. The draws have to
    # be a pure function of (seed, step, layer, pass) rather than a running
    # generator: pass-level activation recompute replays the forward, and a
    # generator that advanced during the first forward would hand the backward
    # a different coalition than the one whose loss it is differentiating.
    random_policy_step: int = 0

    @classmethod
    def from_value(
        cls,
        value: "CurriculumState | Mapping[str, Any] | None",
    ) -> "CurriculumState":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    def validate(self, config: Metis16Config) -> None:
        if self.continuation_mode not in {
            "adaptive",
            "depth_one",
            "fixed_max",
            "random",
        }:
            raise ValueError(
                "continuation_mode must be adaptive, depth_one, fixed_max, or random."
            )
        if self.routed_k_mode not in {"adaptive", "fixed", "random"}:
            raise ValueError("routed_k_mode must be adaptive, fixed, or random.")
        if self.pathway_mode not in {"per_pass", "frozen"}:
            raise ValueError("pathway_mode must be per_pass or frozen.")
        if config.ffn_mode == "dense":
            if self.routed_k_mode != "adaptive" or self.pathway_mode != "per_pass":
                raise ValueError(
                    "A dense feed-forward stack has no width or pathway decision; "
                    "leave routed_k_mode and pathway_mode at their defaults."
                )
        elif not config.min_routed_k <= self.fixed_routed_k <= config.max_routed_k:
            raise ValueError("fixed_routed_k is outside the model's routed-k bounds.")
        if self.max_passes is not None and not 1 <= self.max_passes <= config.max_passes:
            raise ValueError("curriculum max_passes is outside [1, config.max_passes].")
        if self.memory_gate_scale < 0.0:
            raise ValueError("memory_gate_scale cannot be negative.")
        if self.temperature <= 0.0:
            raise ValueError("curriculum temperature must be positive.")
        if self.target_mean_depth is not None and not (
            1.0 <= self.target_mean_depth <= config.max_passes
        ):
            raise ValueError("target_mean_depth must lie in [1, config.max_passes].")
        if self.target_mean_routed_k is not None and not (
            config.min_routed_k
            <= self.target_mean_routed_k
            <= config.max_routed_k
        ):
            raise ValueError("target_mean_routed_k is outside the routed-k bounds.")


def max_entropy_categorical(values: Sequence[int], mean: float) -> tuple[float, ...]:
    """Least-informative distribution over ``values`` with the given mean.

    The random-policy ablation controls must spend the same compute budget as
    the learned policies while carrying no information about the token.  The
    exponential tilt ``p(v) is proportional to exp(lambda * v)`` is the maximum-entropy
    distribution subject to a mean constraint, so it is the honest way to say
    "same budget, no signal": any other choice would smuggle in a prior about
    which widths or depths matter.
    """

    support = [float(value) for value in values]
    if not support:
        raise ValueError("max_entropy_categorical needs a non-empty support.")
    low, high = min(support), max(support)
    if not low <= mean <= high:
        raise ValueError(f"mean {mean} lies outside the support [{low}, {high}].")
    if len(support) == 1:
        return (1.0,)
    if abs(mean - low) < 1e-12:
        return tuple(1.0 if value == low else 0.0 for value in support)
    if abs(mean - high) < 1e-12:
        return tuple(1.0 if value == high else 0.0 for value in support)

    def _mean_at(lam: float) -> float:
        peak = max(lam * value for value in support)
        weights = [math.exp(lam * value - peak) for value in support]
        total = sum(weights)
        return sum(w * v for w, v in zip(weights, support)) / total

    lower, upper = -50.0, 50.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if _mean_at(middle) < mean:
            lower = middle
        else:
            upper = middle
    lam = 0.5 * (lower + upper)
    peak = max(lam * value for value in support)
    weights = [math.exp(lam * value - peak) for value in support]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def geometric_continue_probability(max_passes: int, mean_depth: float) -> float:
    """Per-pass continuation probability giving ``mean_depth`` under a cap.

    A memoryless halt with continue-probability ``p`` and at most ``max_passes``
    passes has expected depth ``(1 - p**max_passes) / (1 - p)``.  This is the
    depth-axis analogue of :func:`max_entropy_categorical`: same mean budget,
    no dependence on the token.
    """

    if max_passes < 1:
        raise ValueError("max_passes must be at least one.")
    if not 1.0 <= mean_depth <= float(max_passes):
        raise ValueError("mean_depth must lie in [1, max_passes].")
    if max_passes == 1 or abs(mean_depth - 1.0) < 1e-12:
        return 0.0
    if abs(mean_depth - float(max_passes)) < 1e-12:
        return 1.0

    def _depth_at(p: float) -> float:
        if p >= 1.0 - 1e-12:
            return float(max_passes)
        return (1.0 - p ** max_passes) / (1.0 - p)

    lower, upper = 0.0, 1.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if _depth_at(middle) < mean_depth:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@dataclass
class Metis16CausalLMOutput:
    logits: Tensor | None
    loss: Tensor | None
    auxiliary_loss: Tensor
    auxiliary_losses: dict[str, Tensor]
    telemetry: dict[str, Tensor]
    chosen_depths: Tensor
    active_masks: Tensor
    final_hidden_state: Tensor


@dataclass
class RouteState:
    summary: Tensor
    mean_k: Tensor
    expected_k: Tensor
    entropy: Tensor
    confidence: Tensor
    token_difficulty: Tensor
    assignments: Tensor
    processed_assignments: Tensor
    expert_counts: Tensor
    expert_load_cv: Tensor
    all_to_all_bytes: Tensor
    all_to_all_seconds: Tensor
    auxiliary_losses: dict[str, Tensor] = field(default_factory=dict)


def fp8_wire_dtypes(device: torch.device) -> tuple[torch.dtype, torch.dtype]:
    """Return the (forward, backward) FP8 wire formats for this accelerator.

    gfx942 implements the OCP ``fnuz`` variants natively, so ROCm uses those and
    everything else falls back to the CUDA-style encodings.  Only casts happen
    on the wire -- no GEMM consumes these bytes directly -- so the choice only
    changes the representable range, which ``torch.finfo`` supplies.
    """

    if getattr(torch.version, "hip", None) and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz, torch.float8_e5m2fnuz
    return torch.float8_e4m3fn, torch.float8_e5m2


def _quantize_rows(values: Tensor, wire_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Scale each row to the FP8 range and return (byte payload, FP32 scales).

    Per-row scaling costs four bytes per row against a payload of ``latent_dim``
    bytes -- under half a percent -- and removes any dependence on cross-rank
    amax history, so the collective stays free of extra synchronisation.
    """

    limit = float(torch.finfo(wire_dtype).max)
    amax = values.detach().abs().amax(dim=-1).float()
    scale = (amax / limit).clamp_min(torch.finfo(torch.float32).tiny)
    payload = (
        values.float()
        .div(scale.unsqueeze(-1))
        .clamp(-limit, limit)
        .to(wire_dtype)
    )
    return payload.view(torch.uint8), scale


def _dequantize_rows(
    payload: Tensor,
    scale: Tensor,
    *,
    wire_dtype: torch.dtype,
    value_dtype: torch.dtype,
) -> Tensor:
    return (
        payload.view(wire_dtype).float().mul(scale.unsqueeze(-1)).to(value_dtype)
    )


def _fp8_all_to_all(
    values: Tensor,
    *,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group: Any,
    wire_dtype: torch.dtype,
) -> Tensor:
    """Exchange rows as one-byte payloads plus their per-row FP32 scales."""

    payload, scale = _quantize_rows(values.contiguous(), wire_dtype)
    received = payload.new_empty((sum(output_splits), *payload.shape[1:]))
    dist.all_to_all_single(
        received,
        payload,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    received_scale = scale.new_empty((sum(output_splits),))
    dist.all_to_all_single(
        received_scale,
        scale.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return _dequantize_rows(
        received,
        received_scale,
        wire_dtype=wire_dtype,
        value_dtype=values.dtype,
    )


def _bf16_all_to_all(
    values: Tensor,
    *,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group: Any,
) -> Tensor:
    output = values.new_empty((sum(output_splits), *values.shape[1:]))
    dist.all_to_all_single(
        output,
        values.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return output


class _VariableAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        group: Any,
        wire: str,
    ) -> Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.wire = wire
        if wire == "bfloat16":
            return _bf16_all_to_all(
                values,
                input_splits=input_splits,
                output_splits=output_splits,
                group=group,
            )
        forward_dtype, _ = fp8_wire_dtypes(values.device)
        return _fp8_all_to_all(
            values,
            input_splits=input_splits,
            output_splits=output_splits,
            group=group,
            wire_dtype=forward_dtype,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        if ctx.wire == "bfloat16":
            grad_input = _bf16_all_to_all(
                grad_output,
                input_splits=ctx.output_splits,
                output_splits=ctx.input_splits,
                group=ctx.group,
            )
        else:
            # Gradients span a wider dynamic range than activations, so the
            # return leg uses E5M2 as ``hybrid_e4m3_e5m2`` prescribes.
            _, backward_dtype = fp8_wire_dtypes(grad_output.device)
            grad_input = _fp8_all_to_all(
                grad_output,
                input_splits=ctx.output_splits,
                output_splits=ctx.input_splits,
                group=ctx.group,
                wire_dtype=backward_dtype,
            )
        return grad_input, None, None, None, None


class _CollectiveHandle:
    """Slot carrying one in-flight all-to-all between its start and its wait.

    ``start`` enqueues the transfer and hands back a correctly shaped carrier
    tensor whose contents are not yet valid.  ``complete`` synchronises and
    materialises the real payload, which for an FP8 wire also means applying
    the per-row scales that travelled alongside it.
    """

    __slots__ = ("_works", "_finish")

    def __init__(self) -> None:
        self._works: list[Any] = []
        self._finish: Callable[[], Tensor] | None = None

    def arm(self, works: list[Any], finish: Callable[[], Tensor]) -> None:
        self._works = works
        self._finish = finish

    def complete(self) -> Tensor:
        if self._finish is None:
            raise RuntimeError("Collective wait reached an unarmed handle")
        for work in self._works:
            if work is not None:
                work.wait()
        payload = self._finish()
        self._works = []
        self._finish = None
        return payload


def _start_all_to_all(
    values: Tensor,
    *,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group: Any,
    wire_dtype: torch.dtype | None,
    handle: _CollectiveHandle,
) -> Tensor:
    """Enqueue an asynchronous exchange and return its carrier tensor."""

    received_rows = sum(output_splits)
    if wire_dtype is None:
        output = values.new_empty((received_rows, *values.shape[1:]))
        work = dist.all_to_all_single(
            output,
            values.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
            async_op=True,
        )
        handle.arm([work], lambda: output)
        return output

    payload, scale = _quantize_rows(values.contiguous(), wire_dtype)
    received = payload.new_empty((received_rows, *payload.shape[1:]))
    received_scale = scale.new_empty((received_rows,))
    works = [
        dist.all_to_all_single(
            received,
            payload,
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
            async_op=True,
        ),
        dist.all_to_all_single(
            received_scale,
            scale.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
            async_op=True,
        ),
    ]
    value_dtype = values.dtype
    handle.arm(
        works,
        lambda: _dequantize_rows(
            received,
            received_scale,
            wire_dtype=wire_dtype,
            value_dtype=value_dtype,
        ),
    )
    # The carrier only has to carry shape and autograd lineage; the real values
    # arrive when the matching wait dequantises the received payload.
    return values.new_empty((received_rows, *values.shape[1:]))


class _PipelinedAllToAllStart(torch.autograd.Function):
    """Issue the forward exchange; complete the backward one.

    Paired with :class:`_PipelinedAllToAllWait`, which does the mirror image.
    Autograd runs every wait node before any start node, because the waits were
    created later and therefore hold higher sequence numbers, so the backward
    exchanges are all in flight before the first of them is consumed.  That is
    what makes the backward pipeline as well as the forward one.
    """

    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        group: Any,
        wire: str,
        forward_handle: _CollectiveHandle,
        backward_handle: _CollectiveHandle,
    ) -> Tensor:
        ctx.backward_handle = backward_handle
        forward_dtype = None if wire == "bfloat16" else fp8_wire_dtypes(values.device)[0]
        return _start_all_to_all(
            values,
            input_splits=input_splits,
            output_splits=output_splits,
            group=group,
            wire_dtype=forward_dtype,
            handle=forward_handle,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        del grad_output  # a carrier; the real gradient is in the handle
        return ctx.backward_handle.complete(), None, None, None, None, None, None


class _PipelinedAllToAllWait(torch.autograd.Function):
    """Complete the forward exchange; issue the backward one."""

    @staticmethod
    def forward(
        ctx: Any,
        carrier: Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        group: Any,
        wire: str,
        forward_handle: _CollectiveHandle,
        backward_handle: _CollectiveHandle,
    ) -> Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.wire = wire
        ctx.backward_handle = backward_handle
        del carrier
        return forward_handle.complete()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        # Gradients span a wider dynamic range than activations, so an FP8
        # return leg uses E5M2 as ``hybrid_e4m3_e5m2`` prescribes.
        backward_dtype = (
            None if ctx.wire == "bfloat16" else fp8_wire_dtypes(grad_output.device)[1]
        )
        _start_all_to_all(
            grad_output,
            input_splits=ctx.output_splits,
            output_splits=ctx.input_splits,
            group=ctx.group,
            wire_dtype=backward_dtype,
            handle=ctx.backward_handle,
        )
        return grad_output, None, None, None, None, None, None


def _pipelined_all_to_all_start(
    values: Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group: Any,
    wire: str,
    forward_handle: _CollectiveHandle,
    backward_handle: _CollectiveHandle,
) -> Tensor:
    return _PipelinedAllToAllStart.apply(
        values,
        tuple(int(value) for value in input_splits),
        tuple(int(value) for value in output_splits),
        group,
        wire,
        forward_handle,
        backward_handle,
    )


def _pipelined_all_to_all_wait(
    carrier: Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group: Any,
    wire: str,
    forward_handle: _CollectiveHandle,
    backward_handle: _CollectiveHandle,
) -> Tensor:
    return _PipelinedAllToAllWait.apply(
        carrier,
        tuple(int(value) for value in input_splits),
        tuple(int(value) for value in output_splits),
        group,
        wire,
        forward_handle,
        backward_handle,
    )


def _variable_all_to_all(
    values: Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group: Any,
    wire: str = "bfloat16",
) -> Tensor:
    if _group_world_size(group) == 1:
        return values
    return _VariableAllToAll.apply(
        values,
        tuple(int(value) for value in input_splits),
        tuple(int(value) for value in output_splits),
        group,
        wire,
    )


def _wire_element_bytes(wire: str, values: Tensor) -> float:
    """Bytes per element actually placed on the wire, including row scales."""

    if wire == "bfloat16":
        return float(values.element_size())
    return 1.0 + 4.0 / float(max(values.shape[-1], 1))


def expert_collective_wire_error(values: Tensor, *, gradient: bool = False) -> float:
    """Round-trip relative error the FP8 expert wire would introduce.

    ``allow_fp8_collectives_only_after_probe`` requires evidence before an FP8
    wire carries a production step; this is the local half of that evidence and
    needs no process group, so a single-APU canary can produce it.
    """

    forward_dtype, backward_dtype = fp8_wire_dtypes(values.device)
    wire_dtype = backward_dtype if gradient else forward_dtype
    payload, scale = _quantize_rows(values, wire_dtype)
    restored = _dequantize_rows(
        payload,
        scale,
        wire_dtype=wire_dtype,
        value_dtype=values.dtype,
    )
    reference = values.float()
    denominator = reference.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    return float((restored.float() - reference).abs().amax() / denominator)


@torch.no_grad()
def _exchange_counts(send_counts: Tensor, *, group: Any) -> Tensor:
    if _group_world_size(group) == 1:
        return send_counts.clone()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=group)
    return recv_counts


@torch.no_grad()
def _all_to_all_indices(
    values: Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    group: Any,
) -> Tensor:
    if _group_world_size(group) == 1:
        return values
    output = values.new_empty((sum(output_splits), *values.shape[1:]))
    dist.all_to_all_single(
        output,
        values.contiguous(),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = float(eps)
        self.metis_precision_role = "normalization"

    def forward(self, values: Tensor) -> Tensor:
        source_dtype = values.dtype
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(source_dtype)


def sinkhorn_doubly_stochastic(logits: Tensor, iterations: int) -> Tensor:
    """Project a square matrix to the doubly stochastic manifold in FP32."""

    log_matrix = logits.float()
    for _ in range(iterations):
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)
    return log_matrix.exp()


class MHCConnection(nn.Module):
    """Four-stream, pass-conditioned manifold Hyper-Connection."""

    def __init__(
        self,
        config: Metis16Config,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        n_streams = config.n_streams
        controller_width = n_streams * n_streams + 2 * n_streams
        self.n_streams = n_streams
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.mhc_backend = config.mhc_backend
        self.family = config.family
        self.norm = RMSNorm(config.d_model, device=device, dtype=dtype)
        self.read_logits = nn.Parameter(torch.zeros(n_streams, device=device, dtype=dtype))
        self.mix_logits = nn.Parameter(torch.eye(n_streams, device=device, dtype=dtype) * 2.0)
        self.write_logits = nn.Parameter(torch.zeros(n_streams, device=device, dtype=dtype))
        self.controller = _make_linear(
            precision_policy,
            config.mhc_pass_embedding_dim,
            controller_width,
            bias=True,
            role="mhc_controller",
            device=device,
            dtype=dtype,
        )
        self.last_sinkhorn_error: Tensor | None = None

    def read(
        self,
        streams: Tensor,
        pass_embedding: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        delta = self.controller(pass_embedding).float()
        n = self.n_streams
        matrix_delta, read_delta, write_delta = torch.split(delta, (n * n, n, n), dim=-1)
        matrix = sinkhorn_doubly_stochastic(
            self.mix_logits.float() + matrix_delta.view(n, n),
            self.sinkhorn_iterations,
        )
        read_weights = torch.softmax(self.read_logits.float() + read_delta, dim=-1)
        write_weights = torch.softmax(self.write_logits.float() + write_delta, dim=-1)
        source, mixed = mhc_read_mix(
            streams,
            matrix,
            read_weights,
            backend=self.mhc_backend,
            family=self.family,
        )
        self.last_sinkhorn_error = torch.maximum(
            (matrix.sum(dim=-1) - 1.0).abs().amax(),
            (matrix.sum(dim=-2) - 1.0).abs().amax(),
        )
        return self.norm(source), (mixed, write_weights, streams)

    def write(
        self,
        residual: tuple[Tensor, Tensor, Tensor],
        update: Tensor,
        *,
        active_mask: Tensor,
    ) -> Tensor:
        mixed, write_weights, original_streams = residual
        return mhc_masked_write(
            mixed,
            write_weights,
            update,
            original_streams,
            active_mask,
            backend=self.mhc_backend,
            family=self.family,
        )


def _derive_document_ids(
    input_ids: Tensor,
    *,
    document_ids: Tensor | None,
    reset_mask: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    if document_ids is not None:
        if document_ids.shape != input_ids.shape:
            raise ValueError("document_ids must have the same shape as input_ids.")
        if reset_mask is None:
            reset_mask = torch.ones_like(input_ids, dtype=torch.bool)
            reset_mask[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
        return document_ids, reset_mask.to(torch.bool)
    if reset_mask is None:
        return None, None
    if reset_mask.shape != input_ids.shape:
        raise ValueError("reset_mask must have the same shape as input_ids.")
    cooked_reset = reset_mask.to(torch.bool).clone()
    cooked_reset[:, 0] = True
    return cooked_reset.to(torch.long).cumsum(dim=1) - 1, cooked_reset


@dataclass(frozen=True)
class PackedDocumentLayout:
    flat_token_indices: Tensor
    cu_seqlens: Tensor
    max_seqlen: int


@dataclass(frozen=True)
class ActiveTokenLayout:
    """Exact local packing map for a monotonic continuation pass."""

    flat_token_indices: Tensor
    batch_size: int
    sequence_length: int

    @property
    def token_count(self) -> int:
        return int(self.flat_token_indices.numel())

    def pack(self, values: Tensor) -> Tensor:
        if values.shape[:2] != (self.batch_size, self.sequence_length):
            raise ValueError("Active-token pack input has the wrong leading shape.")
        tail = values.shape[2:]
        selected = values.reshape(
            self.batch_size * self.sequence_length,
            *tail,
        ).index_select(0, self.flat_token_indices)
        return selected.unsqueeze(0)

    def scatter(self, packed: Tensor, *, base: Tensor | None = None) -> Tensor:
        if packed.shape[0] != 1 or packed.shape[1] != self.token_count:
            raise ValueError("Active-token scatter input has the wrong packed shape.")
        tail = packed.shape[2:]
        if base is None:
            flat = packed.new_zeros(
                self.batch_size * self.sequence_length,
                *tail,
            )
        else:
            if base.shape != (self.batch_size, self.sequence_length, *tail):
                raise ValueError("Active-token scatter base has the wrong shape.")
            flat = base.reshape(self.batch_size * self.sequence_length, *tail)
        scattered = flat.index_copy(0, self.flat_token_indices, packed.squeeze(0))
        return scattered.view(self.batch_size, self.sequence_length, *tail)


def _active_token_layout(active_mask: Tensor) -> ActiveTokenLayout:
    if active_mask.ndim != 2:
        raise ValueError("active_mask must have shape [batch, sequence].")
    batch, sequence = active_mask.shape
    indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).flatten()
    return ActiveTokenLayout(indices, batch, sequence)


def _packed_document_metadata(
    layout: ActiveTokenLayout,
    document_ids: Tensor | None,
    *,
    continues_previous: bool = False,
) -> tuple[Tensor, Tensor]:
    """Document ids and reset flags for one pass's packed active tokens.

    ``continues_previous`` says whether the first packed token carries on a
    document that some other rank holds.  It is False for every unsharded run,
    where the packed buffer really does start a sequence.  Under context
    parallelism it is usually True, and getting it wrong is expensive but
    quiet: a spurious reset at position 0 zeroes the SSM state and the
    convolution history that the shard exchange just went to the trouble of
    fetching, so the model trains as if every shard boundary were a document
    boundary.
    """

    indices = layout.flat_token_indices
    batch_ids = torch.div(
        indices,
        layout.sequence_length,
        rounding_mode="floor",
    )
    if document_ids is None:
        source_documents = batch_ids
    else:
        source_documents = document_ids.reshape(-1).index_select(0, indices)
    boundaries = torch.ones(
        layout.token_count,
        device=indices.device,
        dtype=torch.bool,
    )
    boundaries[0] = not continues_previous
    if layout.token_count > 1:
        boundaries[1:] = (batch_ids[1:] != batch_ids[:-1]) | (
            source_documents[1:] != source_documents[:-1]
        )
    packed_document_ids = boundaries.to(torch.long).cumsum(dim=0) - 1
    return packed_document_ids.unsqueeze(0), boundaries.unsqueeze(0)


def _build_packed_document_layout(
    token_mask: Tensor,
    document_ids: Tensor | None,
) -> PackedDocumentLayout:
    batch, seq_len = token_mask.shape
    flat_valid = token_mask.reshape(-1)
    flat_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if flat_indices.numel() == 0:
        return PackedDocumentLayout(
            flat_token_indices=flat_indices,
            cu_seqlens=torch.zeros(1, device=token_mask.device, dtype=torch.int32),
            max_seqlen=0,
        )
    boundary = torch.zeros_like(token_mask, dtype=torch.bool)
    boundary[:, 0] = token_mask[:, 0]
    if seq_len > 1:
        if document_ids is None:
            boundary[:, 1:] = token_mask[:, 1:] & ~token_mask[:, :-1]
        else:
            boundary[:, 1:] = token_mask[:, 1:] & (
                ~token_mask[:, :-1]
                | (document_ids[:, 1:] != document_ids[:, :-1])
            )
    flat_boundary = boundary.reshape(-1)
    segment_ids = flat_boundary.cumsum(dim=0) - 1
    valid_segments = segment_ids.index_select(0, flat_indices)
    lengths = torch.bincount(valid_segments)
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, device=token_mask.device, dtype=torch.int64),
            lengths.cumsum(dim=0),
        )
    ).to(torch.int32)
    return PackedDocumentLayout(
        flat_token_indices=flat_indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(lengths.max().item()),
    )


class ReferenceMamba2(nn.Module):
    """Correctness lane for Mamba-2 with document-boundary state resets.

    Production manifests use ``fused_required`` and cannot silently enter this
    token loop. Tiny CPU tests opt in explicitly.
    """

    def __init__(
        self,
        config: Metis16Config,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = config.d_model * config.mamba_expand
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.head_dim = config.mamba_head_dim
        self.n_heads = self.d_inner // self.head_dim
        self.n_groups = config.mamba_ngroups
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        in_width = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = _make_linear(
            precision_policy,
            self.d_model,
            in_width,
            bias=False,
            role="mamba_in_projection",
            device=device,
            dtype=dtype,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(conv_dim, self.d_conv, device=device, dtype=dtype)
        )
        self.conv_bias = nn.Parameter(torch.zeros(conv_dim, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, device=device, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.n_heads, device=device, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.n_heads, device=device, dtype=torch.float32))
        self.gated_norm = RMSNorm(self.d_inner, device=device, dtype=dtype)
        self.out_proj = _make_linear(
            precision_policy,
            self.d_inner,
            self.d_model,
            bias=False,
            role="mamba_out_projection",
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.conv_weight, std=0.02)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        context_parallel: ContextParallelContext | None = None,
    ) -> Tensor:
        del document_ids
        batch_size, seq_len, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        z, xbc, dt = torch.split(
            projected,
            (
                self.d_inner,
                self.d_inner + 2 * self.n_groups * self.d_state,
                self.n_heads,
            ),
            dim=-1,
        )
        conv_dim = xbc.shape[-1]
        parallel = context_parallel if context_parallel is not None else None
        sharded = parallel is not None and parallel.enabled

        # The convolution and the scan are independent chains -- conv state
        # depends only on the projection, scan state only on the conv output --
        # so running them as two loops instead of one interleaved loop is an
        # exact refactor.  It buys the seam that context parallelism needs: the
        # shard summary has to see every post-convolution input before the first
        # scan step can begin.
        conv_state = hidden_states.new_zeros(batch_size, conv_dim, self.d_conv)
        if sharded and self.d_conv > 1:
            # Without the predecessor's tail every shard but the first would
            # convolve its opening tokens against zeros.
            halo = conv_left_halo(xbc, parallel, width=self.d_conv)
            conv_state = torch.cat(
                (conv_state[:, :, :1], halo.transpose(1, 2)),
                dim=-1,
            )
        conv_outputs: list[Tensor] = []
        for index in range(seq_len):
            if reset_mask is not None:
                reset = reset_mask[:, index].view(batch_size, 1, 1)
                conv_state = torch.where(reset, torch.zeros_like(conv_state), conv_state)
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = xbc[:, index]
            conv_out = (
                conv_state * self.conv_weight.unsqueeze(0)
            ).sum(dim=-1) + self.conv_bias
            conv_outputs.append(F.silu(conv_out))

        heads_per_group = self.n_heads // self.n_groups
        head_groups = torch.arange(self.n_heads, device=hidden_states.device) // heads_per_group
        decay = -torch.exp(self.A_log.float())
        convolved = torch.stack(conv_outputs, dim=1)
        x_all, b_all, c_all = torch.split(
            convolved,
            (
                self.d_inner,
                self.n_groups * self.d_state,
                self.n_groups * self.d_state,
            ),
            dim=-1,
        )
        x_all = x_all.view(batch_size, seq_len, self.n_heads, self.head_dim)
        b_all = b_all.view(batch_size, seq_len, self.n_groups, self.d_state).index_select(
            2, head_groups
        )
        c_all = c_all.view(batch_size, seq_len, self.n_groups, self.d_state).index_select(
            2, head_groups
        )
        delta_all = F.softplus(dt.float() + self.dt_bias)

        ssm_state = hidden_states.new_zeros(
            batch_size,
            self.n_heads,
            self.head_dim,
            self.d_state,
        )
        if sharded:
            shard_decay, shard_state = mamba_shard_summary(
                x_all,
                b_all,
                delta_all,
                self.A_log,
                reset_mask=reset_mask,
            )
            ssm_state = mamba_incoming_state(
                shard_decay,
                shard_state,
                parallel,
            ).to(ssm_state.dtype)

        outputs: list[Tensor] = []
        for index in range(seq_len):
            if reset_mask is not None:
                reset = reset_mask[:, index].view(batch_size, 1, 1)
                ssm_state = torch.where(
                    reset.unsqueeze(-1),
                    torch.zeros_like(ssm_state),
                    ssm_state,
                )
            x_t = x_all[:, index]
            b_t = b_all[:, index]
            c_t = c_all[:, index]
            delta = delta_all[:, index]
            transition = torch.exp(delta[:, :, None, None] * decay[None, :, None, None])
            update = (
                delta[:, :, None, None].to(x_t.dtype)
                * x_t[:, :, :, None]
                * b_t[:, :, None, :]
            )
            ssm_state = ssm_state * transition.to(ssm_state.dtype) + update
            y_t = torch.einsum("bhpn,bhn->bhp", ssm_state, c_t)
            y_t = y_t + self.D.to(y_t.dtype)[None, :, None] * x_t
            outputs.append(y_t.reshape(batch_size, self.d_inner))
        y = torch.stack(outputs, dim=1)
        y = self.gated_norm(y * F.silu(z))
        return self.out_proj(y)


def _load_fused_mamba2():
    try:
        from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on Portage
        raise RuntimeError(
            "Metis-1.6 production requires the ROCm-validated mamba_ssm Mamba2 "
            "kernel. Install the site-pinned build or use torch_reference only "
            "for a correctness canary."
        ) from exc
    return Mamba2


class FusedMamba2(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        layer_idx: int,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        implementation = _load_fused_mamba2()
        kwargs = {
            "d_model": config.d_model,
            "d_state": config.mamba_d_state,
            "d_conv": config.mamba_d_conv,
            "expand": config.mamba_expand,
            "headdim": config.mamba_head_dim,
            "ngroups": config.mamba_ngroups,
            "chunk_size": config.mamba_chunk_size,
            "layer_idx": layer_idx,
            "bias": False,
            "conv_bias": True,
            "device": device,
            "dtype": dtype,
        }
        accepted = set(signature(implementation).parameters)
        self.mixer = implementation(**{key: value for key, value in kwargs.items() if key in accepted})
        mamba_roles = {
            "in_proj": "mamba_in_projection",
            "out_proj": "mamba_out_projection",
        }
        fp8_mamba_names = [
            name
            for name, role in mamba_roles.items()
            if (
                precision_policy is not None
                and callable(getattr(precision_policy, "is_fp8_role", None))
                and bool(precision_policy.is_fp8_role(role))
            )
        ]
        if fp8_mamba_names:
            for name in fp8_mamba_names:
                original = getattr(self.mixer, name, None)
                if original is None or not hasattr(original, "weight"):
                    raise RuntimeError(f"Installed Mamba2 exposes no replaceable {name}.")
                replacement = _make_linear(
                    precision_policy,
                    int(original.in_features),
                    int(original.out_features),
                    bias=original.bias is not None,
                    role=mamba_roles[name],
                    device=device,
                    dtype=dtype,
                )
                if not hasattr(replacement, "weight"):
                    raise RuntimeError("FP8 Mamba projection must expose a weight parameter.")
                with torch.no_grad():
                    replacement.weight.copy_(original.weight)
                    if original.bias is not None:
                        replacement.bias.copy_(original.bias)
                setattr(self.mixer, name, replacement)
            # The combined Mamba path consumes out_proj.weight directly and
            # would bypass Transformer Engine quantization. Retain fused
            # convolution/SSD scan while forcing explicit FP8 in/out modules.
            if not hasattr(self.mixer, "use_mem_eff_path"):
                raise RuntimeError(
                    "Installed Mamba2 cannot expose its FP8 projection surface safely."
                )
            self.mixer.use_mem_eff_path = False
        self.accepted_forward = set(signature(self.mixer.forward).parameters)
        self.metis_precision_role = "mamba_mixed_surface"

    def forward(
        self,
        hidden_states: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        context_parallel: ContextParallelContext | None = None,
    ) -> Tensor:
        if context_parallel is not None and context_parallel.enabled:
            return self._context_parallel_forward(
                hidden_states,
                document_ids=document_ids,
                reset_mask=reset_mask,
                context_parallel=context_parallel,
            )
        kwargs: dict[str, Any] = {}
        if "seq_idx" in self.accepted_forward and document_ids is not None:
            # mamba_ssm's fused packed-sequence kernels use an int32 segment
            # index. Normalize explicitly instead of relying on a build-
            # dependent implicit cast inside the ROCm extension.
            kwargs["seq_idx"] = document_ids.to(dtype=torch.int32)
        elif reset_mask is not None and bool(reset_mask[:, 1:].any().item()):
            raise RuntimeError(
                "The installed fused Mamba2 kernel cannot reset state at packed "
                "document boundaries. Refusing to leak state across documents."
            )
        return self.mixer(hidden_states, **kwargs)

    def _context_parallel_forward(
        self,
        hidden_states: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        context_parallel: ContextParallelContext,
    ) -> Tensor:
        """Run the fused SSD scan with a state carried in from the left shard.

        ``Mamba2.forward`` owns its own zero initial state and offers no way to
        seed it, so context parallelism has to step one level down and drive
        ``mamba_chunk_scan_combined`` directly.  Everything around the scan --
        the projection split, the causal convolution, the gated norm -- is
        replayed here in the order the installed module uses.

        Replaying another package's internals is exactly the kind of code that
        rots silently across a version bump, so it is not trusted on faith:
        :meth:`assert_context_parallel_parity` runs both paths at ``CP=1`` and
        refuses the job if they disagree.  A layout change then costs a startup
        error instead of a week of quietly wrong long-context gradients.
        """

        scan = _load_fused_ssd_scan()
        mixer = self.mixer
        batch, seq_len, _ = hidden_states.shape
        d_inner = int(mixer.d_inner)
        n_groups = int(mixer.ngroups)
        d_state = int(mixer.d_state)
        n_heads = int(mixer.nheads)
        head_dim = int(mixer.headdim)
        d_conv = int(mixer.d_conv)

        projected = mixer.in_proj(hidden_states)
        extra = projected.shape[-1] - 2 * d_inner - 2 * n_groups * d_state - n_heads
        if extra < 0 or extra % 2:
            raise RuntimeError(
                "Installed Mamba2 in_proj width does not match its documented "
                "z/xBC/dt split; context parallelism cannot decompose it."
            )
        mlp_width = extra // 2
        _z0, _x0, z, xbc, dt = torch.split(
            projected,
            [mlp_width, mlp_width, d_inner, d_inner + 2 * n_groups * d_state, n_heads],
            dim=-1,
        )

        if d_conv > 1:
            halo = conv_left_halo(xbc, context_parallel, width=d_conv)
            extended = torch.cat((halo, xbc), dim=1)
        else:
            extended = xbc
        extended_len = extended.shape[1]
        convolved = mixer.conv1d(extended.transpose(1, 2))[..., :extended_len]
        xbc = mixer.act(convolved.transpose(1, 2)[:, extended_len - seq_len :])

        x, b_matrix, c_matrix = torch.split(
            xbc,
            [d_inner, n_groups * d_state, n_groups * d_state],
            dim=-1,
        )
        x = x.view(batch, seq_len, n_heads, head_dim)
        b_matrix = b_matrix.view(batch, seq_len, n_groups, d_state)
        c_matrix = c_matrix.view(batch, seq_len, n_groups, d_state)

        delta = F.softplus(dt.float() + mixer.dt_bias.float())
        heads_per_group = n_heads // n_groups
        shard_decay, shard_state = mamba_shard_summary(
            x,
            b_matrix.repeat_interleave(heads_per_group, dim=2),
            delta,
            mixer.A_log,
            reset_mask=reset_mask,
        )
        initial_states = mamba_incoming_state(
            shard_decay,
            shard_state,
            context_parallel,
        ).to(x.dtype)

        sequence_index = (
            document_ids.to(dtype=torch.int32) if document_ids is not None else None
        )
        attended = scan(
            x,
            dt,
            -torch.exp(mixer.A_log.float()),
            b_matrix,
            c_matrix,
            chunk_size=int(mixer.chunk_size),
            D=mixer.D,
            z=None,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            seq_idx=sequence_index,
            initial_states=initial_states,
            return_final_states=False,
        )
        if isinstance(attended, tuple):
            attended = attended[0]
        attended = attended.reshape(batch, seq_len, d_inner)
        return mixer.out_proj(mixer.norm(attended, z))

    @torch.no_grad()
    def assert_context_parallel_parity(
        self,
        *,
        batch: int = 1,
        seq_len: int = 64,
        tolerance: float = 2e-2,
    ) -> None:
        """Fail fast if the decomposed scan diverges from the module's own forward.

        Called once per job when context parallelism is enabled.  At ``CP=1``
        the decomposed path carries a zero initial state, so it must reproduce
        ``Mamba2.forward`` exactly up to bf16 accumulation order.
        """

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        probe = torch.randn(batch, seq_len, self.mixer.d_model, device=device, dtype=dtype)
        expected = self.forward(probe, document_ids=None, reset_mask=None)
        observed = self._context_parallel_forward(
            probe,
            document_ids=None,
            reset_mask=None,
            context_parallel=ContextParallelContext.disabled(seq_len).with_local_length(
                seq_len
            ),
        )
        difference = (expected.float() - observed.float()).abs().max().item()
        scale = expected.float().abs().max().clamp_min(1e-6).item()
        if difference > tolerance * scale:
            raise RuntimeError(
                "The decomposed context-parallel Mamba2 path disagrees with the "
                f"installed fused module (max |Δ| {difference:.3e} vs tolerance "
                f"{tolerance * scale:.3e}). The pinned mamba_ssm build changed "
                "its projection or normalization layout; update "
                "FusedMamba2._context_parallel_forward before training."
            )


def _load_fused_ssd_scan() -> Callable[..., Tensor]:
    try:
        from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore
            mamba_chunk_scan_combined,
        )
    except Exception as exc:  # pragma: no cover - exercised on Portage
        raise RuntimeError(
            "Context-parallel Mamba2 requires mamba_ssm's "
            "mamba_chunk_scan_combined, which accepts the initial_states "
            "argument that carries SSD state across a sequence shard boundary."
        ) from exc
    if "initial_states" not in signature(mamba_chunk_scan_combined).parameters:
        raise RuntimeError(
            "The installed mamba_chunk_scan_combined has no initial_states "
            "argument; context parallelism cannot seed the scan."
        )
    return mamba_chunk_scan_combined


class Mamba2Mixer(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        layer_idx: int,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if config.mamba_backend == "torch_reference":
            self.impl = ReferenceMamba2(
                config,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
        elif config.mamba_backend == "fused_required":
            self.impl = FusedMamba2(
                config,
                layer_idx=layer_idx,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
        else:
            try:
                self.impl = FusedMamba2(
                    config,
                    layer_idx=layer_idx,
                    precision_policy=precision_policy,
                    device=device,
                    dtype=dtype,
                )
            except RuntimeError:
                self.impl = ReferenceMamba2(
                    config,
                    precision_policy=precision_policy,
                    device=device,
                    dtype=dtype,
                )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        sequence_mask: Tensor,
        packed_layout: PackedDocumentLayout,
        pass_index: int,
        context_parallel: "ContextParallelPassState | None" = None,
    ) -> Tensor:
        del pass_index, sequence_mask, packed_layout
        return self.impl(
            hidden_states,
            document_ids=document_ids,
            reset_mask=reset_mask,
            context_parallel=(
                context_parallel.context if context_parallel is not None else None
            ),
        )


def _load_varlen_flash_attention() -> Callable[..., Tensor]:
    errors: list[str] = []
    for module_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
    ):
        try:
            module = __import__(module_name, fromlist=["flash_attn_varlen_func"])
            function = getattr(module, "flash_attn_varlen_func", None)
            if callable(function):
                return function
            errors.append(f"{module_name} has no flash_attn_varlen_func")
        except (ImportError, OSError, RuntimeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError(
        "Metis-1.6 production attention requires the ROCm FlashAttention "
        "variable-length kernel (CK or AITER/Triton backend). "
        + " | ".join(errors)
    )


class NoPEGQAAttention(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.backend = config.attention_backend
        self.varlen_attention: Callable[..., Tensor] | None = None
        if self.backend in {"varlen_fused_required", "auto"}:
            try:
                self.varlen_attention = _load_varlen_flash_attention()
            except RuntimeError:
                if self.backend == "varlen_fused_required":
                    raise
        qkv_width = (config.n_heads + 2 * config.n_kv_heads) * config.head_dim
        self.qkv = _make_linear(
            precision_policy,
            config.d_model,
            qkv_width,
            bias=False,
            role="attention_qkv_projection",
            device=device,
            dtype=dtype,
        )
        self.out = _make_linear(
            precision_policy,
            config.d_model,
            config.d_model,
            bias=False,
            role="attention_out_projection",
            device=device,
            dtype=dtype,
        )
        rank = config.attention_pass_lora_rank
        self.pass_lora_down = _make_linear(
            precision_policy,
            config.d_model,
            rank,
            bias=False,
            role="attention_pass_lora_down",
            device=device,
            dtype=dtype,
        )
        self.pass_lora_up = _make_linear(
            precision_policy,
            rank,
            qkv_width,
            bias=False,
            role="attention_pass_lora_up",
            device=device,
            dtype=dtype,
        )
        self.pass_gates = nn.Parameter(torch.zeros(config.max_passes, device=device, dtype=dtype))
        if hasattr(self.pass_lora_up, "weight"):
            nn.init.zeros_(self.pass_lora_up.weight)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        sequence_mask: Tensor,
        packed_layout: PackedDocumentLayout,
        pass_index: int,
        context_parallel: "ContextParallelPassState | None" = None,
    ) -> Tensor:
        del reset_mask
        batch, seq_len, _ = hidden_states.shape
        gate = torch.tanh(self.pass_gates[pass_index]).to(hidden_states.dtype)
        qkv = self.qkv(hidden_states) + gate * self.pass_lora_up(
            self.pass_lora_down(hidden_states)
        )
        q_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        query, key, value = torch.split(qkv, (q_width, kv_width, kv_width), dim=-1)
        query = query.view(batch, seq_len, self.n_heads, self.head_dim)
        key = key.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        if context_parallel is not None and context_parallel.context.enabled:
            return self._context_parallel_attention(
                query,
                key,
                value,
                packed_layout=packed_layout,
                state=context_parallel,
            )
        if self.varlen_attention is not None:
            if packed_layout.max_seqlen == 0:
                attended = torch.zeros_like(query)
            else:
                flat_indices = packed_layout.flat_token_indices
                packed_query = query.reshape(batch * seq_len, self.n_heads, self.head_dim).index_select(
                    0, flat_indices
                )
                packed_key = key.reshape(batch * seq_len, self.n_kv_heads, self.head_dim).index_select(
                    0, flat_indices
                )
                packed_value = value.reshape(batch * seq_len, self.n_kv_heads, self.head_dim).index_select(
                    0, flat_indices
                )
                packed_output = self.varlen_attention(
                    packed_query,
                    packed_key,
                    packed_value,
                    packed_layout.cu_seqlens,
                    packed_layout.cu_seqlens,
                    packed_layout.max_seqlen,
                    packed_layout.max_seqlen,
                    0.0,
                    self.head_dim ** -0.5,
                    True,
                )
                if isinstance(packed_output, tuple):
                    packed_output = packed_output[0]
                flat_output = torch.zeros_like(query.reshape(batch * seq_len, self.n_heads, self.head_dim))
                flat_output = flat_output.index_copy(0, flat_indices, packed_output)
                attended = flat_output.view(batch, seq_len, self.n_heads, self.head_dim)
            attended = attended.reshape(batch, seq_len, self.d_model)
            return self.out(attended)
        if seq_len > 8_192:
            raise RuntimeError(
                "Dense reference attention is bounded to 8,192 tokens; the "
                "context-extension lane requires the fused varlen backend."
            )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        repeats = self.n_heads // self.n_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        attention_mask: Tensor | None = None
        use_causal = document_ids is None and bool(sequence_mask.all().item())
        if not use_causal:
            causal = torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool).tril()
            same_document = (
                torch.ones(batch, seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
                if document_ids is None
                else document_ids[:, :, None] == document_ids[:, None, :]
            )
            valid_pairs = sequence_mask[:, :, None] & sequence_mask[:, None, :]
            attention_mask = (same_document & valid_pairs & causal).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=use_causal,
            scale=self.head_dim ** -0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out(attended)

    def _context_parallel_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        packed_layout: PackedDocumentLayout,
        state: "ContextParallelPassState",
    ) -> Tensor:
        """Attend local queries against keys gathered from the whole CP group.

        Only keys and values cross the wire.  Queries stay put, so attention
        FLOPs shard with the sequence exactly like every other layer -- at the
        price of a causal imbalance, since rank ``r`` attends over roughly
        ``(r+1)/CP`` of the sequence and the last rank therefore sets the pace.
        Grouped-query attention is what makes the trade worth taking: at four KV
        heads the gathered buffer is a fraction of the layer's activation, so a
        few hundred megabytes of intra-node traffic buys the entire memory
        reduction.
        """

        batch, seq_len, _, _ = query.shape
        flat_indices = packed_layout.flat_token_indices
        flat_query = query.reshape(batch * seq_len, self.n_heads, self.head_dim)
        packed_query = flat_query.index_select(0, flat_indices)
        # A rank whose tokens have all halted still joins the gather -- with a
        # zero-row contribution -- because leaving the collective would hang the
        # ranks whose tokens are still running.
        packed_key = key.reshape(
            batch * seq_len, self.n_kv_heads, self.head_dim
        ).index_select(0, flat_indices)[: state.local_count]
        packed_value = value.reshape(
            batch * seq_len, self.n_kv_heads, self.head_dim
        ).index_select(0, flat_indices)[: state.local_count]

        gathered_key, gathered_value, _counts = gather_context_parallel_kv(
            packed_key,
            packed_value,
            state.context,
            capacity=state.capacity,
        )

        layout = state.layout
        scale = self.head_dim ** -0.5
        if layout.empty:
            # Nothing left to attend from on this rank, but the gathered keys
            # still need a backward edge or this rank alone would skip the
            # reduce-scatter its peers are waiting on.
            packed_output = keep_graph_edge(
                torch.zeros_like(packed_query), gathered_key, gathered_value
            )
        elif self.varlen_attention is not None:
            # Keys are truncated per document at this rank's last local query,
            # which is what makes FlashAttention's bottom-right causal alignment
            # coincide with the true causal mask for a middle shard.
            selected_key = gathered_key.index_select(0, layout.key_indices)
            selected_value = gathered_value.index_select(0, layout.key_indices)
            packed_output = self.varlen_attention(
                packed_query,
                selected_key,
                selected_value,
                layout.cu_seqlens_q,
                layout.cu_seqlens_k,
                layout.max_seqlen_q,
                layout.max_seqlen_k,
                0.0,
                scale,
                True,
            )
            if isinstance(packed_output, tuple):
                packed_output = packed_output[0]
        else:
            if gathered_key.shape[0] > 8_192:
                raise RuntimeError(
                    "Dense reference attention is bounded to 8,192 gathered "
                    "tokens; the context-extension lane requires the fused "
                    "varlen backend."
                )
            packed_output = reference_context_parallel_attention(
                packed_query,
                gathered_key,
                gathered_value,
                local_segments=state.local_segments,
                gathered_segments=state.gathered_segments,
                counts=state.counts,
                context=state.context,
                capacity=state.capacity,
                scale=scale,
            )

        flat_output = torch.zeros_like(flat_query).index_copy(
            0, flat_indices, packed_output
        )
        attended = flat_output.view(batch, seq_len, self.d_model)
        return self.out(attended)


class SwiGLUExpert(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        intermediate_dim: int,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.gate_up = _make_linear(
            precision_policy,
            latent_dim,
            2 * intermediate_dim,
            bias=False,
            role="expert_gate_up_projection",
            device=device,
            dtype=dtype,
        )
        self.down = _make_linear(
            precision_policy,
            intermediate_dim,
            latent_dim,
            bias=False,
            role="expert_down_projection",
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

    @torch.no_grad()
    def reset_parameters_for_global_expert(self, seed: int) -> None:
        """Initialize a shard from its global expert identity.

        Corresponding Logos expert copies receive the same seed while
        different global experts never inherit identical rank-local weights.
        """

        devices: list[int] = []
        parameter = next(self.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            devices = [parameter.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if parameter is not None and parameter.is_cuda:
                torch.cuda.manual_seed(int(seed))
            for projection in (self.gate_up, self.down):
                reset = getattr(projection, "reset_parameters", None)
                if callable(reset):
                    reset()


class PathwayCache:
    """Pass-one expert identities, addressed in the full token layout.

    Later passes run over a packed subset of tokens, so the cache stores the
    unpacked ``[batch, sequence, max_k]`` selection and is gathered through the
    same active-token layout the streams use.  Only integer identity is cached;
    nothing here carries a gradient.
    """

    __slots__ = ("_indices", "_widths", "_layout")

    def __init__(self) -> None:
        self._indices: dict[int, Tensor] = {}
        self._widths: dict[int, Tensor] = {}
        self._layout: ActiveTokenLayout | None = None

    def set_layout(self, layout: "ActiveTokenLayout | None") -> None:
        self._layout = layout

    def store(self, layer_index: int, top_indices: Tensor, chosen_k: Tensor) -> None:
        # Write-once.  Pass-level activation recompute replays pass one during
        # the backward, by which point ``_layout`` belongs to a later pass; a
        # second write would scatter pass-one identities through the wrong map.
        if layer_index in self._indices:
            return
        indices = top_indices.detach()
        widths = chosen_k.detach()
        if self._layout is not None:
            indices = self._layout.scatter(indices)
            widths = self._layout.scatter(widths)
        self._indices[layer_index] = indices
        self._widths[layer_index] = widths

    def lookup(self, layer_index: int) -> tuple[Tensor, Tensor] | None:
        indices = self._indices.get(layer_index)
        widths = self._widths.get(layer_index)
        if indices is None or widths is None:
            return None
        if self._layout is not None:
            return self._layout.pack(indices), self._layout.pack(widths)
        return indices, widths

    def clear(self) -> None:
        self._indices.clear()
        self._widths.clear()
        self._layout = None


class DenseFFN(nn.Module):
    """Single SwiGLU sublayer standing in for the expert mixture.

    Rows 1, 2, and 7 of the ablation ladder need a feed-forward path that is
    architecturally identical to the mixture it replaces -- same latent
    bottleneck, same activation, same mHC read/write -- so that the only thing
    the comparison isolates is conditional routing.  It reports a zeroed
    ``RouteState`` so every downstream telemetry consumer keeps working without
    a branch.
    """

    def __init__(
        self,
        config: Metis16Config,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.latent_down = _make_linear(
            precision_policy,
            config.d_model,
            config.latent_dim,
            bias=False,
            role="latent_down_projection",
            device=device,
            dtype=dtype,
        )
        self.latent_up = _make_linear(
            precision_policy,
            config.latent_dim,
            config.d_model,
            bias=False,
            role="latent_up_projection",
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLUExpert(
            config.latent_dim,
            config.dense_ffn_intermediate_dim,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )
        self.collective_timer: CollectiveEventTimer | None = None
        self.dispatch_overlap = False

    def forward(
        self,
        hidden_states: Tensor,
        *,
        route_features: Tensor,
        active_mask: Tensor,
        curriculum: CurriculumState,
        pass_index: int = 0,
        pathway_cache: "PathwayCache | None" = None,
    ) -> tuple[Tensor, RouteState]:
        del route_features, curriculum, pass_index, pathway_cache
        latent = self.latent_down(hidden_states)
        activated = self.ffn(latent) * active_mask.unsqueeze(-1).to(latent.dtype)
        output = self.latent_up(activated)
        zero = output.float().sum() * 0.0
        batch, seq_len, _ = hidden_states.shape
        shape = (batch, seq_len)
        return output, RouteState(
            summary=output.new_zeros(batch, seq_len, self.config.route_feature_dim),
            mean_k=output.new_zeros(shape, dtype=torch.float32),
            expected_k=output.new_zeros(shape, dtype=torch.float32),
            entropy=output.new_zeros(shape, dtype=torch.float32),
            confidence=output.new_ones(shape, dtype=torch.float32),
            token_difficulty=output.new_zeros(shape, dtype=torch.float32),
            assignments=torch.zeros((), device=output.device, dtype=torch.long),
            processed_assignments=torch.zeros((), device=output.device, dtype=torch.long),
            expert_counts=torch.zeros(
                self.config.n_routed_experts, device=output.device, dtype=torch.long
            ),
            expert_load_cv=zero,
            all_to_all_bytes=torch.zeros((), device=output.device, dtype=torch.long),
            all_to_all_seconds=torch.zeros((), device=output.device, dtype=torch.float32),
            auxiliary_losses={name: zero for name in ROUTE_AUXILIARY_LOSS_NAMES},
        )


class AdaptiveDroplessMoE(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        layer_idx: int,
        process_group: Any,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.process_group = process_group
        self.collective_timer: CollectiveEventTimer | None = None
        self.dispatch_overlap = False
        self._shared_compute_stream: torch.cuda.Stream | None = None
        # Off by default and read only by the ablation routing analyzer, which
        # turns it on for a single forward.  Training never pays for it.
        self.capture_selection = False
        self._analysis_last_selection: tuple[Tensor, Tensor] | None = None
        # Dispatch payloads are about to be cast to E4M3 by the expert
        # ``gate_up`` GEMM anyway, and combine payloads by ``latent_up``, so an
        # FP8 wire mostly relocates a quantisation that already happens.  It is
        # still a numerical change, so it is bound to the effective FP8 profile:
        # the BF16 autotune candidate stays BF16 end to end and remains a true
        # parity reference for ``maximum_fp8_loss_relative_error``.  That is
        # what ``allow_fp8_collectives_only_after_probe`` asks for.
        wire = config.precision.expert_collective_wire
        if precision_policy is not None and not getattr(
            precision_policy, "fp8_enabled", False
        ):
            wire = "bfloat16"
        self.dispatch_wire = "fp8" if wire in {"fp8", "fp8_dispatch"} else "bfloat16"
        self.combine_wire = "fp8" if wire == "fp8" else "bfloat16"
        # One chunk reproduces the serial path exactly; more than one turns the
        # dispatch/expert/combine chain into a software pipeline so the fabric
        # and the matrix cores are busy at the same time.
        self.dispatch_chunks = int(config.moe_dispatch_chunks)
        self.world_size = _group_world_size(process_group)
        self.rank = _group_rank(process_group)
        if self.world_size not in {1, config.expert_parallel_size}:
            raise ValueError(
                f"Expert group has size {self.world_size}; expected 1 (canary) "
                f"or {config.expert_parallel_size} (production)."
            )
        if config.n_routed_experts % self.world_size:
            raise ValueError("Routed expert count must divide evenly across the live EP group.")
        self.local_expert_count = config.n_routed_experts // self.world_size
        self.first_local_expert = self.rank * self.local_expert_count
        self.latent_down = _make_linear(
            precision_policy,
            config.d_model,
            config.latent_dim,
            bias=False,
            role="latent_down_projection",
            device=device,
            dtype=dtype,
        )
        self.latent_up = _make_linear(
            precision_policy,
            config.latent_dim,
            config.d_model,
            bias=False,
            role="latent_up_projection",
            device=device,
            dtype=dtype,
        )
        route_input = config.latent_dim + config.route_feature_dim
        self.expert_router = nn.Linear(
            route_input,
            config.n_routed_experts,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.expert_router.metis_precision_role = "router_logits"
        self.k_router = nn.Linear(
            route_input,
            config.max_routed_k - config.min_routed_k + 1,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.k_router.metis_precision_role = "router_logits"
        self.expert_embeddings = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.route_feature_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.expert_embeddings, std=config.route_feature_dim ** -0.5)
        self.register_buffer(
            "selection_bias",
            torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32),
        )
        self.local_experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    config.latent_dim,
                    config.expert_intermediate_dim,
                    precision_policy=precision_policy,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.local_expert_count)
            ]
        )
        self.shared_expert = SwiGLUExpert(
            config.latent_dim,
            config.expert_intermediate_dim,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )
        seed_base = 16_062_026 + layer_idx * 1_000_003
        for local_index, expert in enumerate(self.local_experts):
            global_expert_id = self.first_local_expert + local_index
            expert.reset_parameters_for_global_expert(seed_base + global_expert_id * 97)
        self.shared_expert.reset_parameters_for_global_expert(seed_base + 90_000_001)

    def _random_k_weights(self, target: float, device: torch.device) -> Tensor:
        cached = getattr(self, "_random_k_weight_cache", None)
        if cached is not None and cached[0] == target and cached[1].device == device:
            return cached[1]
        weights = torch.tensor(
            max_entropy_categorical(
                range(self.config.min_routed_k, self.config.max_routed_k + 1),
                target,
            ),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self._random_k_weight_cache = (target, weights)
        return weights

    def _random_policy_generator(
        self,
        curriculum: CurriculumState,
        device: torch.device,
        pass_index: int,
    ) -> torch.Generator | None:
        """Per-layer, per-pass, per-step deterministic stream for the controls.

        Seeded fresh on every call rather than cached and advanced. A cached
        generator makes the draw depend on how many times it has been called,
        and pass-level activation recompute calls it a second time for the same
        pass: the backward would then differentiate a coalition the forward
        never chose. Seeding by layer and pass keeps the control from
        collapsing to one constant width per token, and by step from freezing
        into the same draw for the whole run.
        """

        if not curriculum.random_policy_seed:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(
            (
                int(curriculum.random_policy_seed)
                + 1_000_003 * (self.layer_idx + 1)
                + 10_000_019 * int(pass_index)
                + 100_003 * int(curriculum.random_policy_step)
            )
            % (2**63 - 1)
        )
        return generator

    def _choose_k(
        self,
        logits: Tensor,
        curriculum: CurriculumState,
        pass_index: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        choices = torch.arange(
            self.config.min_routed_k,
            self.config.max_routed_k + 1,
            device=logits.device,
            dtype=torch.float32,
        )
        probabilities = torch.softmax(logits.float() / curriculum.temperature, dim=-1)
        expected = torch.sum(probabilities * choices, dim=-1)
        if curriculum.routed_k_mode == "fixed":
            chosen = torch.full_like(expected, curriculum.fixed_routed_k, dtype=torch.long)
        elif curriculum.routed_k_mode == "random":
            # Same expected width as the learned policy, zero dependence on the
            # token.  ``expected`` is replaced by the constant target so the
            # k-budget loss stays well defined and the straight-through envelope
            # contributes no gradient -- there is no policy here to teach.
            target = (
                curriculum.target_mean_routed_k
                if curriculum.target_mean_routed_k is not None
                else self.config.target_mean_routed_k
            )
            weights = self._random_k_weights(float(target), logits.device)
            flat = torch.multinomial(
                weights.expand(logits[..., 0].numel(), -1),
                num_samples=1,
                replacement=True,
                generator=self._random_policy_generator(
                    curriculum, logits.device, pass_index
                ),
            ).squeeze(-1)
            chosen = flat.view(logits.shape[:-1]) + self.config.min_routed_k
            expected = torch.full_like(expected, float(target))
            probabilities = weights.expand_as(probabilities)
        elif self.training and curriculum.stochastic_routing:
            chosen_index = F.gumbel_softmax(
                logits.float(),
                tau=curriculum.temperature,
                hard=True,
                dim=-1,
            ).argmax(dim=-1)
            chosen = chosen_index + self.config.min_routed_k
        else:
            chosen = probabilities.argmax(dim=-1) + self.config.min_routed_k
        return chosen.to(torch.long), expected, probabilities

    def _execute_local_grouped(
        self,
        hidden_states: Tensor,
        local_indices: Tensor,
    ) -> Tensor:
        """Sorted, single-synchronization expert execution.

        The per-expert ``torch.nonzero`` in :meth:`_execute_local` forces a
        device-to-host synchronization for every expert, which is invisible when
        an expert-parallel rank owns a handful of experts and ruinous when a
        data-parallel rank replicates all of them: 96 experts x 10 layers x
        several passes is thousands of stalls per forward.  Sorting once and
        deriving segment boundaries from a single ``bincount`` reduces that to
        one synchronization per layer.  Numerics are unchanged -- the same rows
        reach the same experts -- only the scheduling differs.
        """

        expert_count = len(self.local_experts)
        output = torch.zeros_like(hidden_states)
        order = torch.argsort(local_indices, stable=True)
        sorted_hidden = hidden_states.index_select(0, order)
        counts = torch.bincount(local_indices, minlength=expert_count)
        # The one and only host synchronization in this path.
        segment_sizes = counts.tolist()
        start = 0
        pieces: list[Tensor] = []
        for local_index, expert in enumerate(self.local_experts):
            size = int(segment_sizes[local_index])
            if size == 0:
                # DelayedScaling reduces amax for every Transformer Engine
                # module at context exit, so every rank must still invoke every
                # local expert slot even when it received no tokens.  The zero
                # probe preserves graph and collective order without
                # fabricating an assignment.
                probe = expert(hidden_states[:1])
                output = output + probe.sum() * 0.0
                continue
            pieces.append((local_index, start, expert(sorted_hidden[start : start + size])))
            start += size
        if not pieces:
            return output
        combined = torch.cat([piece for _index, _start, piece in pieces], dim=0)
        # ``order`` maps sorted rows back to their original assignment slots.
        # Only rows belonging to a non-empty expert were computed, and they are
        # contiguous in sorted order, so the same iteration order rebuilds the
        # matching destination index.
        computed_positions = torch.cat(
            [
                order[start : start + int(segment_sizes[index])]
                for index, start, _piece in pieces
            ],
            dim=0,
        )
        return output.index_copy(0, computed_positions, combined.to(output.dtype))

    def _execute_local(self, hidden_states: Tensor, local_indices: Tensor) -> Tensor:
        if self.config.expert_execution == "grouped":
            return self._execute_local_grouped(hidden_states, local_indices)
        output = torch.zeros_like(hidden_states)
        for local_index, expert in enumerate(self.local_experts):
            positions = torch.nonzero(local_indices == local_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                # DelayedScaling reduces amax for every TE module at context
                # exit. Every EP rank must therefore invoke the same local
                # expert slots even when dropless routing sends a slot no
                # tokens on this rank. The zero probe preserves graph and
                # collective order without fabricating an assignment.
                probe = expert(hidden_states[:1])
                output = output + probe.sum() * 0.0
                continue
            expert_input = hidden_states.index_select(0, positions)
            output = output.index_copy(0, positions, expert(expert_input))
        return output

    def _execute_shared(self, flat_latent: Tensor, active_flat: Tensor) -> Tensor:
        active_positions = torch.nonzero(active_flat, as_tuple=False).flatten()
        output = torch.zeros_like(flat_latent)
        if active_positions.numel() == 0:
            # Keep the shared expert in the graph on zero-token ranks.
            probe = self.shared_expert(flat_latent[:1])
            return output + probe.sum() * 0.0
        active_hidden = flat_latent.index_select(0, active_positions)
        return output.index_copy(0, active_positions, self.shared_expert(active_hidden))

    def _select_experts(
        self,
        expert_logits: Tensor,
        chosen_k: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select with the balance bias, but combine with unbiased affinities."""

        selection_scores = expert_logits + self.selection_bias.view(1, 1, -1)
        top_indices = torch.topk(
            selection_scores,
            k=self.config.max_routed_k,
            dim=-1,
        ).indices
        top_logits = expert_logits.gather(-1, top_indices)
        slot_ids = torch.arange(
            self.config.max_routed_k,
            device=expert_logits.device,
        )
        selected = slot_ids.view(1, 1, -1) < chosen_k.unsqueeze(-1)
        selected = selected & active_mask.unsqueeze(-1)
        masked_logits = top_logits.masked_fill(~selected, float("-inf"))
        top_weights = torch.softmax(masked_logits, dim=-1)
        top_weights = torch.where(
            selected,
            top_weights,
            torch.zeros_like(top_weights),
        )
        return top_indices, top_weights, selected

    def _recombine_frozen_experts(
        self,
        expert_logits: Tensor,
        top_indices: Tensor,
        chosen_k: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Softmax over a pathway that was decided at pass one."""

        top_logits = expert_logits.gather(-1, top_indices)
        slot_ids = torch.arange(
            self.config.max_routed_k,
            device=expert_logits.device,
        )
        selected = slot_ids.view(1, 1, -1) < chosen_k.unsqueeze(-1)
        selected = selected & active_mask.unsqueeze(-1)
        masked_logits = top_logits.masked_fill(~selected, float("-inf"))
        top_weights = torch.softmax(masked_logits, dim=-1)
        top_weights = torch.where(selected, top_weights, torch.zeros_like(top_weights))
        return top_weights, selected

    def _chunk_permutation(
        self,
        destinations: Tensor,
        send_counts: Tensor,
        chunks: int,
    ) -> tuple[Tensor, Tensor]:
        """Split destination-sorted assignments into ``chunks`` balanced groups.

        Every rank uses the same chunk count, so every rank issues an identical
        collective sequence no matter how routing skewed its local load.  Within
        a chunk the rows stay grouped by destination, which is what keeps each
        chunk a well-formed all-to-all.  A chunk that receives no rows for some
        destination simply contributes a zero split.
        """

        total = int(destinations.numel())
        offsets = torch.cumsum(send_counts, dim=0) - send_counts
        per_destination_rank = (
            torch.arange(total, device=destinations.device)
            - offsets.index_select(0, destinations)
        )
        destination_size = send_counts.index_select(0, destinations).clamp_min(1)
        chunk_of_row = torch.div(
            per_destination_rank * chunks,
            destination_size,
            rounding_mode="floor",
        ).clamp_(0, chunks - 1)
        key = chunk_of_row * self.world_size + destinations
        permutation = torch.argsort(key, stable=True)
        counts = torch.bincount(
            key.index_select(0, permutation),
            minlength=chunks * self.world_size,
        ).view(chunks, self.world_size)
        return permutation, counts

    def _exchange_chunk_counts(self, counts: Tensor) -> Tensor:
        """Trade the whole [chunks, ranks] send plan in one small exchange."""

        if self.world_size == 1:
            return counts.clone()
        sent = counts.t().contiguous()
        received = torch.empty_like(sent)
        dist.all_to_all_single(received, sent, group=self.process_group)
        return received.t().contiguous()

    def _pipelined_dispatch(
        self,
        send_hidden: Tensor,
        send_local: Tensor,
        destinations: Tensor,
        send_counts_tensor: Tensor,
        chunks: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dispatch, run experts, and combine as a software pipeline.

        Chunk ``c``'s expert GEMMs execute while chunk ``c + 1``'s dispatch and
        chunk ``c - 1``'s combine are still on the wire.  The bytes, the peer
        set, and the fan-out are all identical to the serial path -- only the
        idle time between them disappears.
        """

        permutation, counts = self._chunk_permutation(
            destinations,
            send_counts_tensor,
            chunks,
        )
        received_counts = self._exchange_chunk_counts(counts)
        send_plan = counts.tolist()
        recv_plan = received_counts.tolist()

        send_hidden = send_hidden.index_select(0, permutation)
        send_local = send_local.index_select(0, permutation)
        send_bounds = [0]
        recv_bounds = [0]
        for chunk in range(chunks):
            send_bounds.append(send_bounds[-1] + sum(send_plan[chunk]))
            recv_bounds.append(recv_bounds[-1] + sum(recv_plan[chunk]))

        # Index routing carries no gradient and is three orders of magnitude
        # smaller than the payload, so it is exchanged up front rather than
        # occupying a pipeline stage.
        local_by_chunk = [
            _all_to_all_indices(
                send_local[send_bounds[chunk] : send_bounds[chunk + 1]],
                input_splits=send_plan[chunk],
                output_splits=recv_plan[chunk],
                group=self.process_group,
            )
            for chunk in range(chunks)
        ]

        dispatch_handles = [(_CollectiveHandle(), _CollectiveHandle()) for _ in range(chunks)]
        combine_handles = [(_CollectiveHandle(), _CollectiveHandle()) for _ in range(chunks)]

        def start_dispatch(chunk: int) -> Tensor:
            return _pipelined_all_to_all_start(
                send_hidden[send_bounds[chunk] : send_bounds[chunk + 1]],
                input_splits=send_plan[chunk],
                output_splits=recv_plan[chunk],
                group=self.process_group,
                wire=self.dispatch_wire,
                forward_handle=dispatch_handles[chunk][0],
                backward_handle=dispatch_handles[chunk][1],
            )

        carriers = [None] * chunks
        combined_carriers: list[Tensor] = []
        carriers[0] = start_dispatch(0)
        processed = 0
        for chunk in range(chunks):
            if chunk + 1 < chunks:
                carriers[chunk + 1] = start_dispatch(chunk + 1)
            received = _pipelined_all_to_all_wait(
                carriers[chunk],
                input_splits=send_plan[chunk],
                output_splits=recv_plan[chunk],
                group=self.process_group,
                wire=self.dispatch_wire,
                forward_handle=dispatch_handles[chunk][0],
                backward_handle=dispatch_handles[chunk][1],
            )
            output = self._execute_local(received, local_by_chunk[chunk])
            processed += int(output.shape[0])
            combined_carriers.append(
                _pipelined_all_to_all_start(
                    output,
                    input_splits=recv_plan[chunk],
                    output_splits=send_plan[chunk],
                    group=self.process_group,
                    wire=self.combine_wire,
                    forward_handle=combine_handles[chunk][0],
                    backward_handle=combine_handles[chunk][1],
                )
            )

        returned = torch.cat(
            [
                _pipelined_all_to_all_wait(
                    combined_carriers[chunk],
                    input_splits=recv_plan[chunk],
                    output_splits=send_plan[chunk],
                    group=self.process_group,
                    wire=self.combine_wire,
                    forward_handle=combine_handles[chunk][0],
                    backward_handle=combine_handles[chunk][1],
                )
                for chunk in range(chunks)
            ],
            dim=0,
        )
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
        return returned.index_select(0, inverse), send_hidden, send_local, torch.tensor(
            processed,
            device=send_hidden.device,
            dtype=torch.long,
        )

    def _dispatch(
        self,
        hidden_states: Tensor,
        expert_indices: Tensor,
        weights: Tensor,
        token_indices: Tensor,
        token_count: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        destinations = torch.div(
            expert_indices,
            self.local_expert_count,
            rounding_mode="floor",
        )
        local_indices = expert_indices.remainder(self.local_expert_count)
        order = torch.argsort(destinations, stable=True)
        destinations = destinations.index_select(0, order)
        send_hidden = hidden_states.index_select(0, order)
        send_local = local_indices.index_select(0, order)
        send_counts_tensor = torch.bincount(destinations, minlength=self.world_size).to(
            device=hidden_states.device,
            dtype=torch.int64,
        )
        chunks = self.dispatch_chunks if self.world_size > 1 else 1
        if chunks > 1:
            timing = (
                self.collective_timer.begin(send_hidden)
                if self.collective_timer is not None
                else (time.perf_counter(), None)
            )
            returned, send_hidden, send_local, processed = self._pipelined_dispatch(
                send_hidden,
                send_local,
                destinations,
                send_counts_tensor,
                chunks,
            )
            if self.collective_timer is not None:
                self.collective_timer.end(timing, returned)
                elapsed = 0.0
            else:
                elapsed = time.perf_counter() - timing[0]
            return self._combine_outputs(
                returned,
                order=order,
                send_hidden=send_hidden,
                send_local=send_local,
                weights=weights,
                token_indices=token_indices,
                token_count=token_count,
                hidden_states=hidden_states,
                processed=processed,
                elapsed=elapsed,
            )
        first_collectives = (
            self.collective_timer.begin(send_hidden)
            if self.collective_timer is not None and self.world_size > 1
            else (time.perf_counter(), None)
        )
        recv_counts_tensor = _exchange_counts(send_counts_tensor, group=self.process_group)
        send_counts = [int(value) for value in send_counts_tensor.tolist()]
        recv_counts = [int(value) for value in recv_counts_tensor.tolist()]
        recv_hidden = _variable_all_to_all(
            send_hidden,
            input_splits=send_counts,
            output_splits=recv_counts,
            group=self.process_group,
            wire=self.dispatch_wire,
        )
        recv_local = _all_to_all_indices(
            send_local,
            input_splits=send_counts,
            output_splits=recv_counts,
            group=self.process_group,
        )
        if self.collective_timer is not None and self.world_size > 1:
            self.collective_timer.end(first_collectives, recv_hidden)
            elapsed = 0.0
        else:
            elapsed = time.perf_counter() - first_collectives[0]
        recv_output = self._execute_local(recv_hidden, recv_local)
        return_collective = (
            self.collective_timer.begin(recv_output)
            if self.collective_timer is not None and self.world_size > 1
            else (time.perf_counter(), None)
        )
        returned = _variable_all_to_all(
            recv_output,
            input_splits=recv_counts,
            output_splits=send_counts,
            group=self.process_group,
            wire=self.combine_wire,
        )
        if self.collective_timer is not None and self.world_size > 1:
            self.collective_timer.end(return_collective, returned)
        else:
            elapsed += time.perf_counter() - return_collective[0]
        return self._combine_outputs(
            returned,
            order=order,
            send_hidden=send_hidden,
            send_local=send_local,
            weights=weights,
            token_indices=token_indices,
            token_count=token_count,
            hidden_states=hidden_states,
            processed=returned.new_tensor(recv_output.shape[0], dtype=torch.long),
            elapsed=elapsed,
        )

    def _combine_outputs(
        self,
        returned: Tensor,
        *,
        order: Tensor,
        send_hidden: Tensor,
        send_local: Tensor,
        weights: Tensor,
        token_indices: Tensor,
        token_count: int,
        hidden_states: Tensor,
        processed: Tensor,
        elapsed: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Undo the dispatch ordering and scatter the weighted expert results.

        Shared by the serial and pipelined dispatch paths so both produce
        identical values, telemetry, and gradients.
        """

        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        returned = returned.index_select(0, inverse_order)
        # Combine weights remain on the source rank. This preserves the exact
        # weighted result and its gradients while removing one EP all-to-all.
        returned = returned * weights.unsqueeze(-1).to(returned.dtype)
        combined = hidden_states.new_zeros(token_count, hidden_states.shape[-1])
        combined.index_add_(0, token_indices, returned)
        wire_bytes = (
            send_hidden.numel() * _wire_element_bytes(self.dispatch_wire, send_hidden)
            + send_local.numel() * send_local.element_size()
            + returned.numel() * _wire_element_bytes(self.combine_wire, returned)
        )
        if self.world_size == 1:
            wire_bytes = 0
        return (
            combined,
            processed.to(device=returned.device, dtype=torch.long),
            returned.new_tensor(wire_bytes, dtype=torch.long),
            returned.new_tensor(elapsed, dtype=torch.float64),
        )

    @torch.no_grad()
    def update_selection_bias(self, global_counts: Tensor, rate: float = 1.0e-3) -> None:
        if global_counts.numel() != self.config.n_routed_experts:
            raise ValueError("global_counts has the wrong expert dimension.")
        target = global_counts.float().mean()
        direction = torch.sign(target - global_counts.float())
        self.selection_bias.add_(rate * direction).clamp_(-5.0, 5.0)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        route_features: Tensor,
        active_mask: Tensor,
        curriculum: CurriculumState,
        pass_index: int = 0,
        pathway_cache: "PathwayCache | None" = None,
    ) -> tuple[Tensor, RouteState]:
        batch, seq_len, _ = hidden_states.shape
        latent = self.latent_down(hidden_states)
        route_input = torch.cat((latent, route_features.to(latent.dtype)), dim=-1)
        expert_logits = _fp32_linear(self.expert_router, route_input)
        expert_probabilities = torch.softmax(expert_logits, dim=-1)
        entropy = -torch.sum(
            expert_probabilities * expert_probabilities.clamp_min(1.0e-9).log(),
            dim=-1,
        )
        confidence = expert_probabilities.amax(dim=-1)
        # Expert uncertainty is a normalized token-difficulty prior. It is
        # detached here so K-budget gradients cannot reshape the expert router,
        # while a monotonic bias makes uncertain tokens prefer more experts.
        token_difficulty = (
            entropy / self.config.expert_entropy_normalizer
        ).clamp(0.0, 1.0)
        k_choices = torch.arange(
            self.config.min_routed_k,
            self.config.max_routed_k + 1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        centered_choices = k_choices - k_choices.mean()
        k_logits = _fp32_linear(self.k_router, route_input)
        k_logits = (
            k_logits
            + token_difficulty.detach().unsqueeze(-1) * centered_choices
        )
        chosen_k, expected_k, _k_probabilities = self._choose_k(
            k_logits, curriculum, pass_index
        )
        # ``pass_index`` is zero-based: the first pass through the shared stack
        # is 0, so pass one stores and every later pass reuses.
        frozen = (
            pathway_cache.lookup(self.layer_idx)
            if pathway_cache is not None and pass_index > 0
            else None
        )
        if frozen is None:
            top_indices, top_weights, selected = self._select_experts(
                expert_logits,
                chosen_k,
                active_mask,
            )
            if pathway_cache is not None and pass_index == 0:
                pathway_cache.store(self.layer_idx, top_indices, chosen_k)
        else:
            # Pathway-frozen control: reuse pass 1's expert identity and width,
            # but recompute the combination weights from this pass's logits.
            # Caching the weights themselves would thread a gradient path from
            # pass r back into pass 1's router across a checkpoint boundary; the
            # axis under test is *which* experts, not how they are blended.
            cached_indices, cached_k = frozen
            top_indices = cached_indices
            chosen_k = cached_k
            expected_k = expected_k.detach() * 0.0 + cached_k.to(expected_k.dtype)
            top_weights, selected = self._recombine_frozen_experts(
                expert_logits,
                top_indices,
                chosen_k,
                active_mask,
            )
        if self.capture_selection:
            self._analysis_last_selection = (
                top_indices[..., 0].detach(),
                chosen_k.detach(),
            )

        flat_latent = latent.reshape(batch * seq_len, self.config.latent_dim)
        flat_indices = top_indices.reshape(-1)
        flat_weights = top_weights.reshape(-1)
        flat_selected = selected.reshape(-1)
        assignment_positions = torch.nonzero(flat_selected, as_tuple=False).flatten()
        assignment_experts = flat_indices.index_select(0, assignment_positions)
        assignment_weights = flat_weights.index_select(0, assignment_positions)
        assignment_tokens = torch.div(
            assignment_positions,
            self.config.max_routed_k,
            rounding_mode="floor",
        )
        assignment_hidden = flat_latent.index_select(0, assignment_tokens)
        active_flat_mask = active_mask.reshape(-1)
        overlap_shared = (
            self.dispatch_overlap
            and self.world_size > 1
            and flat_latent.is_cuda
        )
        if overlap_shared:
            if self._shared_compute_stream is None:
                self._shared_compute_stream = torch.cuda.Stream(device=flat_latent.device)
            current_stream = torch.cuda.current_stream(flat_latent.device)
            self._shared_compute_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._shared_compute_stream):
                shared = self._execute_shared(flat_latent, active_flat_mask)
        routed, processed, all_to_all_bytes, all_to_all_seconds = self._dispatch(
            assignment_hidden,
            assignment_experts,
            assignment_weights,
            assignment_tokens,
            batch * seq_len,
        )
        if overlap_shared:
            assert self._shared_compute_stream is not None
            torch.cuda.current_stream(flat_latent.device).wait_stream(self._shared_compute_stream)
            shared.record_stream(torch.cuda.current_stream(flat_latent.device))
        else:
            shared = self._execute_shared(flat_latent, active_flat_mask)
        # Hard K controls the exact packed dispatch above. This straight-through
        # envelope is numerically one in the forward pass, but lets downstream
        # task loss teach the K router whether the routed update was useful.
        straight_through_k = (
            chosen_k.float() + expected_k - expected_k.detach()
        )
        routed_credit = (
            straight_through_k / chosen_k.detach().float().clamp_min(1.0)
        ).reshape(batch * seq_len, 1)
        routed = routed * routed_credit.to(routed.dtype)
        active_flat = active_flat_mask.reshape(-1, 1)
        combined = (routed + shared) * active_flat.to(shared.dtype)
        output = self.latent_up(combined.view(batch, seq_len, self.config.latent_dim))

        coalition = torch.sum(
            top_weights.unsqueeze(-1).to(self.expert_embeddings.dtype)
            * self.expert_embeddings[top_indices],
            dim=-2,
        )
        coalition = coalition * routed_credit.view(batch, seq_len, 1).to(
            coalition.dtype
        )
        expert_counts = torch.bincount(
            assignment_experts,
            minlength=self.config.n_routed_experts,
        )
        if self.world_size > 1:
            dist.all_reduce(expert_counts, group=self.process_group)
        load_fraction = expert_counts.float() / expert_counts.sum().clamp_min(1)
        expert_load_cv = expert_counts.float().std(unbiased=False) / expert_counts.float().mean().clamp_min(1.0)
        active_weights = active_mask.float()
        active_count = active_weights.sum()
        active_denominator = active_count.clamp_min(1.0)
        has_active = (active_count > 0).float()
        probability_fraction = (
            expert_probabilities
            * active_weights.unsqueeze(-1)
        ).sum(dim=(0, 1)) / active_denominator
        mean_expected_k = (
            expected_k * active_weights
        ).sum() / active_denominator
        balance_loss = self.config.n_routed_experts * torch.sum(
            load_fraction * probability_fraction
        )
        z_loss_per_token = torch.logsumexp(expert_logits, dim=-1).square()
        z_loss = (
            z_loss_per_token * active_weights
        ).sum() / active_denominator
        target_mean_k = (
            curriculum.target_mean_routed_k
            if curriculum.target_mean_routed_k is not None
            else self.config.target_mean_routed_k
        )
        k_budget = (mean_expected_k - target_mean_k).square() * has_active
        state = RouteState(
            summary=coalition,
            mean_k=chosen_k.float(),
            expected_k=expected_k,
            entropy=entropy,
            confidence=confidence,
            token_difficulty=token_difficulty,
            assignments=selected.sum(),
            processed_assignments=processed,
            expert_counts=expert_counts,
            expert_load_cv=expert_load_cv,
            all_to_all_bytes=all_to_all_bytes,
            all_to_all_seconds=all_to_all_seconds,
            auxiliary_losses={
                "expert_balance": balance_loss * self.config.expert_balance_coefficient,
                "expert_router_z": z_loss * self.config.router_z_loss_coefficient,
                "routed_k_budget": k_budget * self.config.k_budget_coefficient,
            },
        )
        return output, state


class DistributedHashEmbedding(nn.Module):
    """Sparse embedding with replicated or deterministic row-sharded storage."""

    def __init__(
        self,
        num_rows: int,
        embedding_dim: int,
        *,
        table_mode: str,
        process_group: Any,
        sparse: bool,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.num_rows = int(num_rows)
        self.embedding_dim = int(embedding_dim)
        self.table_mode = table_mode
        self.process_group = process_group
        self.world_size = _group_world_size(process_group)
        self.rank = _group_rank(process_group)
        if table_mode == "replicated":
            local_rows = self.num_rows
        elif table_mode == "row_sharded":
            local_rows = (
                ((self.num_rows - 1 - self.rank) // self.world_size) + 1
                if self.rank < self.num_rows
                else 0
            )
        else:
            raise ValueError(f"Unsupported N-gram table mode: {table_mode}")
        self.embedding = nn.Embedding(
            local_rows,
            embedding_dim,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )
        self.embedding.metis_precision_role = "ngram_table"

    def forward(self, row_ids: Tensor) -> Tensor:
        if row_ids.dtype != torch.long:
            row_ids = row_ids.to(torch.long)
        if row_ids.numel() and (
            int(row_ids.min().item()) < 0 or int(row_ids.max().item()) >= self.num_rows
        ):
            raise IndexError("Hashed N-gram row is outside the table.")
        if self.table_mode == "replicated" or self.world_size == 1:
            return self.embedding(row_ids)
        original_shape = row_ids.shape
        flattened = row_ids.reshape(-1)
        destinations = flattened.remainder(self.world_size)
        local_rows = torch.div(flattened, self.world_size, rounding_mode="floor")
        order = torch.argsort(destinations, stable=True)
        sorted_destinations = destinations.index_select(0, order)
        sorted_local_rows = local_rows.index_select(0, order)
        send_counts_tensor = torch.bincount(
            sorted_destinations,
            minlength=self.world_size,
        ).to(device=row_ids.device, dtype=torch.int64)
        recv_counts_tensor = _exchange_counts(send_counts_tensor, group=self.process_group)
        send_counts = [int(value) for value in send_counts_tensor.tolist()]
        recv_counts = [int(value) for value in recv_counts_tensor.tolist()]
        requested_rows = _all_to_all_indices(
            sorted_local_rows,
            input_splits=send_counts,
            output_splits=recv_counts,
            group=self.process_group,
        )
        local_values = self.embedding(requested_rows)
        returned = _variable_all_to_all(
            local_values,
            input_splits=recv_counts,
            output_splits=send_counts,
            group=self.process_group,
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return returned.index_select(0, inverse).view(*original_shape, self.embedding_dim)


def _sync_sparse_gradient(gradient: Tensor, *, group: Any) -> Tensor:
    """Coalesce and average sparse row gradients across table replicas."""

    if _group_world_size(group) == 1:
        return gradient.coalesce() if gradient.is_sparse else gradient
    if not gradient.is_sparse:
        dist.all_reduce(gradient, group=group)
        return gradient / _group_world_size(group)
    gradient = gradient.coalesce()
    indices = gradient.indices()
    values = gradient.values()
    local_nnz = torch.tensor([indices.shape[1]], device=values.device, dtype=torch.int64)
    gathered_nnz = [torch.empty_like(local_nnz) for _ in range(_group_world_size(group))]
    dist.all_gather(gathered_nnz, local_nnz, group=group)
    counts = [int(item.item()) for item in gathered_nnz]
    max_nnz = max(counts)
    padded_indices = torch.zeros(
        indices.shape[0],
        max_nnz,
        device=indices.device,
        dtype=indices.dtype,
    )
    padded_values = torch.zeros(
        max_nnz,
        *values.shape[1:],
        device=values.device,
        dtype=values.dtype,
    )
    if indices.shape[1]:
        padded_indices[:, : indices.shape[1]] = indices
        padded_values[: values.shape[0]] = values
    all_indices = [torch.empty_like(padded_indices) for _ in counts]
    all_values = [torch.empty_like(padded_values) for _ in counts]
    dist.all_gather(all_indices, padded_indices, group=group)
    dist.all_gather(all_values, padded_values, group=group)
    merged_indices = torch.cat(
        [value[:, :count] for value, count in zip(all_indices, counts)],
        dim=1,
    )
    merged_values = torch.cat(
        [value[:count] for value, count in zip(all_values, counts)],
        dim=0,
    ) / float(_group_world_size(group))
    return torch.sparse_coo_tensor(
        merged_indices,
        merged_values,
        size=gradient.shape,
        device=gradient.device,
        dtype=gradient.dtype,
    ).coalesce()


class NGramConditionalMemory(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        process_group: Any,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.spec = config.ngram_memory
        self.process_group = process_group
        if (
            self.spec.table_mode == "row_sharded"
            and config.world_size > 1
            and process_group is None
        ):
            raise RuntimeError(
                "row_sharded N-gram tables require an explicit table_lookup process group."
            )
        self.tables = nn.ModuleDict()
        for order in self.spec.orders:
            for head, rows in enumerate(self.spec.slots_by_order[order]):
                self.tables[f"o{order}_h{head}"] = DistributedHashEmbedding(
                    rows,
                    self.spec.value_dim,
                    table_mode=self.spec.table_mode,
                    process_group=process_group,
                    sparse=self.spec.sparse_gradients,
                    device=device,
                    dtype=dtype,
                )
        self.projection = _make_linear(
            precision_policy,
            self.spec.concatenated_dim,
            config.d_model,
            bias=True,
            role="ngram_projection",
            device=device,
            dtype=dtype,
        )
        injection_count = len(self.spec.injection_layers)
        self.gate_vectors = nn.Parameter(
            torch.zeros(
                injection_count,
                config.max_passes,
                config.n_streams,
                config.d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.gate_bias = nn.Parameter(
            torch.full(
                (injection_count, config.max_passes, config.n_streams),
                config.memory_gate_init,
                device=device,
                dtype=dtype,
            )
        )
        self._sync_enabled = False
        self._sync_handles: list[Any] = []

    def _valid_ngram_mask(
        self,
        order: int,
        input_ids: Tensor,
        *,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        attention_mask: Tensor,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        valid = attention_mask.clone()
        if order > 1:
            valid[:, : order - 1] = False
        if document_ids is not None:
            for offset in range(1, order):
                same = document_ids[:, offset:] == document_ids[:, :-offset]
                padded = torch.zeros(batch, seq_len, device=input_ids.device, dtype=torch.bool)
                padded[:, offset:] = same
                valid &= padded
        elif reset_mask is not None:
            for offset in range(order - 1):
                shifted = torch.zeros_like(reset_mask, dtype=torch.bool)
                shifted[:, offset:] = reset_mask[:, : seq_len - offset]
                valid &= ~shifted
        return valid

    def _keys(
        self,
        canonical_ids: Tensor,
        *,
        order: int,
        head: int,
        slots: int,
    ) -> Tensor:
        batch, seq_len = canonical_ids.shape
        key = torch.full(
            (batch, seq_len),
            int(self.spec.hash_seeds[head] % slots),
            device=canonical_ids.device,
            dtype=torch.long,
        )
        multiplier = 1_000_003 + 2 * head
        for offset in reversed(range(order)):
            shifted = torch.zeros_like(canonical_ids, dtype=torch.long)
            if offset == 0:
                shifted.copy_(canonical_ids)
            else:
                shifted[:, offset:] = canonical_ids[:, :-offset]
            key = torch.remainder(key * multiplier + shifted.long() + 1, slots)
        return key

    def retrieve(
        self,
        input_ids: Tensor,
        *,
        canonical_ids: Tensor | None,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        attention_mask: Tensor,
    ) -> Tensor:
        if canonical_ids is None:
            canonical_ids = input_ids
        if canonical_ids.shape != input_ids.shape:
            raise ValueError("canonical_ids must have the same shape as input_ids.")
        retrieved: list[Tensor] = []
        for order in self.spec.orders:
            valid = self._valid_ngram_mask(
                order,
                input_ids,
                document_ids=document_ids,
                reset_mask=reset_mask,
                attention_mask=attention_mask,
            )
            for head, slots in enumerate(self.spec.slots_by_order[order]):
                keys = self._keys(canonical_ids, order=order, head=head, slots=slots)
                values = self.tables[f"o{order}_h{head}"](keys)
                retrieved.append(values * valid.unsqueeze(-1).to(values.dtype))
        return self.projection(torch.cat(retrieved, dim=-1))

    def inject(
        self,
        streams: Tensor,
        cached_memory: Tensor,
        *,
        layer_index: int,
        pass_index: int,
        active_mask: Tensor,
        gate_scale: float,
    ) -> Tensor:
        try:
            injection_index = self.spec.injection_layers.index(layer_index)
        except ValueError:
            return streams
        vectors = self.gate_vectors[injection_index, pass_index]
        bias = self.gate_bias[injection_index, pass_index]
        logits = (
            torch.einsum("...sd,sd->...s", streams.float(), vectors.float())
            / math.sqrt(self.config.d_model)
        ) + bias.float()
        gates = torch.sigmoid(logits) * float(gate_scale)
        updated = streams + gates[..., None].to(streams.dtype) * cached_memory.unsqueeze(-2)
        return torch.where(active_mask[..., None, None], updated, streams)

    def enable_managed_sparse_gradient_sync(self, group: Any) -> None:
        if self._sync_enabled:
            return
        if _group_world_size(group) <= 1:
            self._sync_enabled = True
            return
        for table in self.tables.values():
            weight = table.embedding.weight
            self._sync_handles.append(
                weight.register_hook(
                    lambda gradient, sync_group=group: _sync_sparse_gradient(
                        gradient,
                        group=sync_group,
                    )
                )
            )
        self._sync_enabled = True

    @property
    def sparse_sync_enabled(self) -> bool:
        return self._sync_enabled

    def assert_sparse_sync_ready(self, gradient_group: Any, *, training: bool) -> None:
        if not training:
            return
        needs_sync = _group_world_size(gradient_group) > 1
        if needs_sync and not self._sync_enabled:
            raise RuntimeError(
                "Sparse N-gram gradients would diverge across table replicas. "
                "Call model.enable_managed_sparse_gradient_sync(group) before training."
            )


@dataclass
class RecurrentMemoryBank:
    entries: list[Tensor] = field(default_factory=list)
    valid_masks: list[Tensor] = field(default_factory=list)

    @property
    def slot_count(self) -> int:
        return len(self.entries)


class RecurrentDepthMemory(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.state_write = _make_linear(
            precision_policy,
            config.d_model,
            config.memory_dim,
            bias=True,
            role="memory_state_write_projection",
            device=device,
            dtype=dtype,
        )
        self.metadata_write = _make_linear(
            precision_policy,
            config.route_feature_dim + 4,
            config.memory_dim,
            bias=True,
            role="memory_metadata_write_projection",
            device=device,
            dtype=dtype,
        )
        self.pass_embeddings = nn.Parameter(
            torch.empty(config.max_passes, config.memory_dim, device=device, dtype=dtype)
        )
        self.anchor_embeddings = nn.Parameter(
            torch.empty(
                config.n_attention_layers + 1,
                config.memory_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.query = _make_linear(
            precision_policy,
            config.d_model,
            config.memory_dim,
            bias=True,
            role="memory_query_projection",
            device=device,
            dtype=dtype,
        )
        self.key = _make_linear(
            precision_policy,
            config.memory_dim,
            config.memory_dim,
            bias=True,
            role="memory_key_projection",
            device=device,
            dtype=dtype,
        )
        self.value = _make_linear(
            precision_policy,
            config.memory_dim,
            config.memory_dim,
            bias=True,
            role="memory_value_projection",
            device=device,
            dtype=dtype,
        )
        self.output = _make_linear(
            precision_policy,
            config.memory_dim,
            config.d_model,
            bias=True,
            role="memory_output_projection",
            device=device,
            dtype=dtype,
        )
        self.route_projection = _make_linear(
            precision_policy,
            3 * config.d_model + config.route_feature_dim,
            config.route_feature_dim,
            bias=True,
            role="memory_route_projection",
            device=device,
            dtype=dtype,
        )
        self.stream_gate = nn.Parameter(
            torch.zeros(config.n_streams, config.d_model, device=device, dtype=dtype)
        )
        self.stream_gate_bias = nn.Parameter(
            torch.full(
                (config.n_streams,),
                config.memory_gate_init,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.pass_embeddings, std=config.memory_dim ** -0.5)
        nn.init.normal_(self.anchor_embeddings, std=config.memory_dim ** -0.5)

    def routing_features(
        self,
        state: Tensor,
        retrieved_memory: Tensor,
        state_difference: Tensor,
        route_history: Tensor,
    ) -> Tensor:
        features = torch.cat(
            (state, retrieved_memory, state_difference, route_history),
            dim=-1,
        )
        return torch.tanh(self.route_projection(features))

    def write(
        self,
        bank: RecurrentMemoryBank,
        streams: Tensor,
        *,
        route_state: RouteState,
        pass_index: int,
        anchor_index: int,
        continuation_confidence: Tensor,
        active_mask: Tensor,
    ) -> None:
        if bank.slot_count >= self.config.memory_slots:
            raise RuntimeError("Recurrent depth memory exceeded its bounded manifest capacity.")
        state = streams.mean(dim=-2)
        metadata = torch.cat(
            (
                route_state.summary,
                route_state.mean_k.unsqueeze(-1) / float(self.config.max_routed_k),
                route_state.entropy.unsqueeze(-1)
                / self.config.expert_entropy_normalizer,
                route_state.confidence.unsqueeze(-1),
                continuation_confidence.unsqueeze(-1),
            ),
            dim=-1,
        )
        entry = (
            self.state_write(state)
            + self.metadata_write(metadata.to(state.dtype))
            + self.pass_embeddings[pass_index]
            + self.anchor_embeddings[anchor_index]
        )
        bank.entries.append(entry)
        bank.valid_masks.append(active_mask)

    def retrieve(
        self,
        bank: RecurrentMemoryBank,
        streams: Tensor,
        *,
        active_mask: Tensor,
        gate_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not bank.entries:
            zero = streams.new_zeros(*streams.shape[:-2], self.config.d_model)
            return streams, zero, streams.new_zeros(*streams.shape[:-2], 0)
        entries = torch.stack(bank.entries, dim=-2)
        valid = torch.stack(bank.valid_masks, dim=-1)
        query = self.query(streams)
        key = self.key(entries)
        value = self.value(entries)
        scores = torch.einsum("...sh,...mh->...sm", query.float(), key.float())
        scores = scores / math.sqrt(self.config.memory_dim)
        scores = scores.masked_fill(~valid.unsqueeze(-2), float("-inf"))
        no_valid = ~valid.any(dim=-1)
        scores = torch.where(
            no_valid[..., None, None],
            torch.zeros_like(scores),
            scores,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(
            valid.unsqueeze(-2),
            weights,
            torch.zeros_like(weights),
        )
        retrieved = torch.einsum(
            "...sm,...mh->...sh",
            weights.to(value.dtype),
            value,
        )
        projected = self.output(retrieved)
        gate_logits = (
            torch.einsum("...sd,sd->...s", streams.float(), self.stream_gate.float())
            / math.sqrt(self.config.d_model)
        ) + self.stream_gate_bias.float()
        gates = torch.sigmoid(gate_logits) * float(gate_scale)
        fused = streams + gates[..., None].to(streams.dtype) * projected
        fused = torch.where(active_mask[..., None, None], fused, streams)
        # The summary feeds the *routing* path -- continuation, adaptive k, and
        # expert identity all consume it -- so it must obey ``gate_scale`` too.
        # Scaling only ``fused`` would leave a model with the memory nominally
        # disabled still routing on retrieved memory, which is precisely the
        # difference the MoRE-Core / MoRE-RM ablation pair exists to measure.
        # At the production scale of 1.0 this multiplication is exact.
        summary = projected.mean(dim=-2)
        if gate_scale != 1.0:
            summary = summary * float(gate_scale)
        return fused, summary, weights


class ContinuationController(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.config = config
        input_dim = 3 * config.d_model + config.route_feature_dim
        self.hidden = nn.Linear(
            input_dim,
            config.route_feature_dim,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.output = nn.Linear(
            config.route_feature_dim,
            1,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.hidden.metis_precision_role = "router_logits"
        self.output.metis_precision_role = "router_logits"
        nn.init.constant_(self.output.bias, config.continuation_gate_init)

    def forward(
        self,
        state: Tensor,
        retrieved_memory: Tensor,
        state_difference: Tensor,
        route_features: Tensor,
    ) -> Tensor:
        features = torch.cat(
            (state, retrieved_memory, state_difference, route_features),
            dim=-1,
        )
        hidden = F.silu(_fp32_linear(self.hidden, features))
        logits = _fp32_linear(self.output, hidden).squeeze(-1)
        state_scale = state.float().square().mean(dim=-1).sqrt().clamp_min(1.0e-6)
        innovation = state_difference.float().square().mean(dim=-1).sqrt()
        # A bounded, detached innovation prior gives the controller a sensible
        # monotonic initialization: harder/changing tokens receive more depth.
        # Learned logits remain free to override it from task-loss credit.
        token_difficulty = torch.tanh(innovation / state_scale).detach()
        logits = logits + token_difficulty - 0.5
        return torch.sigmoid(logits)


class Metis16Block(nn.Module):
    def __init__(
        self,
        config: Metis16Config,
        *,
        layer_index: int,
        process_groups: MetisProcessGroups,
        precision_policy: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.is_attention = layer_index in config.attention_indices
        if self.is_attention:
            self.mixer = NoPEGQAAttention(
                config,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
        else:
            self.mixer = Mamba2Mixer(
                config,
                layer_idx=layer_index,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
        self.mixer_connection = MHCConnection(
            config,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )
        self.moe = (
            DenseFFN(
                config,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
            if config.ffn_mode == "dense"
            else AdaptiveDroplessMoE(
                config,
                layer_idx=layer_index,
                process_group=process_groups.expert,
                precision_policy=precision_policy,
                device=device,
                dtype=dtype,
            )
        )
        self.moe_connection = MHCConnection(
            config,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        streams: Tensor,
        *,
        route_features: Tensor,
        active_mask: Tensor,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        sequence_mask: Tensor,
        packed_layout: PackedDocumentLayout,
        pass_index: int,
        pass_embedding: Tensor,
        curriculum: CurriculumState,
        pathway_cache: "PathwayCache | None" = None,
        context_parallel: "ContextParallelPassState | None" = None,
    ) -> tuple[Tensor, RouteState]:
        mixer_input, mixer_residual = self.mixer_connection.read(streams, pass_embedding)
        # Only the mixers see the shard boundary. mHC, routing, the experts and
        # both memories are per token, so a sequence shard is just a smaller
        # batch of tokens to them and they need no CP awareness at all.
        mixer_output = self.mixer(
            mixer_input,
            document_ids=document_ids,
            reset_mask=reset_mask,
            sequence_mask=sequence_mask,
            packed_layout=packed_layout,
            pass_index=pass_index,
            context_parallel=context_parallel,
        )
        streams = self.mixer_connection.write(
            mixer_residual,
            mixer_output,
            active_mask=active_mask,
        )
        moe_input, moe_residual = self.moe_connection.read(streams, pass_embedding)
        moe_output, route_state = self.moe(
            moe_input,
            route_features=route_features,
            active_mask=active_mask,
            curriculum=curriculum,
            pass_index=pass_index,
            pathway_cache=pathway_cache,
        )
        streams = self.moe_connection.write(
            moe_residual,
            moe_output,
            active_mask=active_mask,
        )
        return streams, route_state


def _context_group_shape(group: Any, configured: int) -> tuple[int, int]:
    """Resolve the context-parallel group's size and this rank's place in it."""

    if configured <= 1:
        if group is not None:
            raise RuntimeError(
                "A context-parallel process group was supplied but "
                "context_parallel_size is 1; the manifest and the launcher "
                "disagree about whether the sequence is sharded."
            )
        return 1, 0
    if group is None or not _dist_ready():
        raise RuntimeError(
            "context_parallel_size above 1 requires an initialized process "
            "group; sequence shards cannot exchange mixer state without one."
        )
    size = dist.get_world_size(group=group)
    if size != configured:
        raise RuntimeError(
            f"Context-parallel group holds {size} ranks but the manifest "
            f"declares context_parallel_size={configured}."
        )
    return size, dist.get_rank(group=group)


_ROUTE_STATE_TENSOR_FIELDS = (
    "summary",
    "mean_k",
    "expected_k",
    "entropy",
    "confidence",
    "token_difficulty",
    "assignments",
    "processed_assignments",
    "expert_counts",
    "expert_load_cv",
    "all_to_all_bytes",
    "all_to_all_seconds",
)


def _route_state_to_flat(state: RouteState) -> tuple[Tensor, ...]:
    """Flatten a ``RouteState`` for transport across a checkpoint boundary.

    ``torch.utils.checkpoint`` only tracks tensors it can see in the output
    structure; a dataclass carrying a dict of losses would have its gradients
    silently dropped.  Every routed FFN -- sparse or dense -- emits the full
    auxiliary-loss key set, so the flattened layout is fixed and the round trip
    is total rather than best-effort.
    """

    missing = [
        name for name in ROUTE_AUXILIARY_LOSS_NAMES
        if name not in state.auxiliary_losses
    ]
    if missing:
        raise RuntimeError(
            f"RouteState is missing auxiliary losses {missing}; the checkpoint "
            "boundary requires the complete key set."
        )
    return (
        *(getattr(state, name) for name in _ROUTE_STATE_TENSOR_FIELDS),
        *(state.auxiliary_losses[name] for name in ROUTE_AUXILIARY_LOSS_NAMES),
    )


def _route_state_from_flat(values: Sequence[Tensor]) -> RouteState:
    expected = len(_ROUTE_STATE_TENSOR_FIELDS) + len(ROUTE_AUXILIARY_LOSS_NAMES)
    if len(values) != expected:
        raise RuntimeError(
            f"Flattened RouteState has {len(values)} entries; expected {expected}."
        )
    split = len(_ROUTE_STATE_TENSOR_FIELDS)
    fields = dict(zip(_ROUTE_STATE_TENSOR_FIELDS, values[:split], strict=True))
    losses = dict(zip(ROUTE_AUXILIARY_LOSS_NAMES, values[split:], strict=True))
    return RouteState(**fields, auxiliary_losses=losses)


def _add_loss(target: dict[str, Tensor], name: str, value: Tensor) -> None:
    target[name] = target[name] + value if name in target else value


class Metis16ForCausalLM(nn.Module):
    """Executable Metis-1.6 Praxis/Logos model core.

    The full physical stack is reused for every recurrent pass. Continuation
    masks are monotonic, expert routing is re-evaluated at every layer/pass,
    and all expert/table distributed paths use exact variable-size traffic
    without capacity drops.
    """

    def __init__(
        self,
        config: Metis16Config,
        *,
        process_groups: MetisProcessGroups | None = None,
        precision_policy: Any = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate() if config.family != "tiny" else config._validate_tiny()
        self.config = config
        self.process_groups = process_groups or MetisProcessGroups()
        self.precision_policy = precision_policy
        if (
            _dist_ready()
            and config.world_size > 1
            and self.process_groups.expert is None
        ):
            raise RuntimeError(
                "Distributed Praxis/Logos construction requires an explicit expert process group."
            )
        if dtype is None:
            dtype = torch.float32 if config.family == "tiny" else torch.bfloat16
        self.parameter_storage_dtype = dtype
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            device=device,
            dtype=dtype,
        )
        self.embedding.metis_precision_role = "embedding"
        self.stream_embeddings = nn.Parameter(
            torch.empty(config.n_streams, config.d_model, device=device, dtype=dtype)
        )
        self.pass_embeddings = nn.Parameter(
            torch.empty(
                config.max_passes,
                config.mhc_pass_embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.ngram_memory = NGramConditionalMemory(
            config,
            process_group=self.process_groups.table_lookup,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )
        self.depth_memory = RecurrentDepthMemory(
            config,
            precision_policy=precision_policy,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                Metis16Block(
                    config,
                    layer_index=layer_index,
                    process_groups=self.process_groups,
                    precision_policy=precision_policy,
                    device=device,
                    dtype=dtype,
                )
                for layer_index in range(config.n_layers)
            ]
        )
        self.collective_timer = CollectiveEventTimer()
        self.dispatch_overlap_enabled = False
        self.activation_recompute_policy = config.activation_recompute_policy
        self.context_parallel_size, self.context_parallel_rank = _context_group_shape(
            self.process_groups.context,
            config.context_parallel_size,
        )
        self._pathway_cache: PathwayCache | None = None
        self.analysis_telemetry_enabled = False
        for layer in self.layers:
            layer.moe.collective_timer = self.collective_timer
        self.continuation = ContinuationController(
            config,
            device=device,
            dtype=dtype,
        )
        self.final_norm = RMSNorm(config.d_model, device=device, dtype=dtype)
        self.lm_head = _make_linear(
            precision_policy,
            config.d_model,
            config.vocab_size,
            bias=False,
            role="lm_head",
            device=device,
            dtype=dtype,
        )
        if config.tie_embeddings:
            if not hasattr(self.lm_head, "weight"):
                raise TypeError("Tied embeddings require a linear implementation exposing .weight.")
            self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.stream_embeddings, std=config.d_model ** -0.5)
        nn.init.normal_(self.pass_embeddings, std=config.mhc_pass_embedding_dim ** -0.5)
        self._tag_sensitive_fp32_parameters()

    def _tag_sensitive_fp32_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if (
                name.endswith("A_log")
                or name.endswith("dt_bias")
                or name.endswith(".D")
                or ".expert_router." in name
                or ".k_router." in name
                or name.startswith("continuation.")
            ):
                setattr(parameter, "metis_storage_dtype", "float32")
            else:
                setattr(parameter, "metis_storage_dtype", "bfloat16")

    def apply_parameter_storage_policy(
        self,
        device: torch.device | str,
    ) -> "Metis16ForCausalLM":
        """Move parameters without downcasting intentionally FP32 Mamba state."""

        self.to(device=device)
        for parameter in self.parameters():
            storage = getattr(parameter, "metis_storage_dtype", "bfloat16")
            target_dtype = torch.float32 if storage == "float32" else torch.bfloat16
            parameter.data = parameter.data.to(device=device, dtype=target_dtype)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(device=device, dtype=target_dtype)
        return self

    def enable_managed_sparse_gradient_sync(self, group: Any | None = None) -> None:
        sync_group = group if group is not None else self.process_groups.table_gradient
        self.ngram_memory.enable_managed_sparse_gradient_sync(sync_group)

    def enable_collective_event_timing(self, enabled: bool = True) -> None:
        """Enable one-sync-per-forward HIP event timing for sampled telemetry."""

        self.collective_timer.enabled = bool(enabled)

    def _precision_call(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke one logical FP8 surface in one top-level TE region."""

        owner = (
            function
            if isinstance(function, nn.Module)
            else getattr(function, "__self__", None)
        )
        with _execution_context(
            self.precision_policy,
            module=owner if isinstance(owner, nn.Module) else None,
        ):
            return function(*args, **kwargs)

    def set_dispatch_overlap(self, enabled: bool) -> None:
        """Overlap shared-expert GEMMs with routed expert all-to-all on HIP."""

        for layer in self.layers:
            layer.moe.dispatch_overlap = bool(enabled)
        self.dispatch_overlap_enabled = bool(enabled)

    def set_activation_recompute_policy(self, policy: str) -> None:
        """Select pass-boundary recomputation without changing the manifest."""

        if policy not in {"none", "pass"}:
            raise ValueError("activation recompute policy must be none or pass.")
        self.activation_recompute_policy = policy

    @torch.no_grad()
    def update_expert_selection_biases(self, counts_by_layer: Tensor) -> None:
        """Advance the aux-loss-free routing controller once per optimizer batch.

        Each layer's counts have already been summed across its EP group by the
        dropless router. Logos has two independent EP replicas, so an additional
        expert-data reduction combines those replicas before every rank applies
        the same deterministic, non-gradient bias update.
        """

        if self.config.ffn_mode == "dense":
            # A dense ablation control has no routed experts and therefore no
            # load to balance. Returning is correct rather than lenient: there
            # is no selection bias to update.
            return
        expected = (self.config.n_layers, self.config.n_routed_experts)
        if tuple(counts_by_layer.shape) != expected:
            raise ValueError(
                "counts_by_layer must have shape "
                f"{expected}, observed {tuple(counts_by_layer.shape)}"
            )
        global_counts = counts_by_layer.detach().to(
            device=self.layers[0].moe.selection_bias.device,
            dtype=torch.long,
        )
        if _group_world_size(self.process_groups.expert_data) > 1:
            global_counts = global_counts.clone()
            dist.all_reduce(global_counts, group=self.process_groups.expert_data)
        rate = float(self.config.expert_balance_bias_update_rate)
        for layer_index, layer in enumerate(self.layers):
            layer.moe.update_selection_bias(global_counts[layer_index], rate=rate)

    @property
    def requires_sparse_gradient_sync(self) -> bool:
        return _group_world_size(self.process_groups.table_gradient) > 1

    def sparse_gradient_sync_enabled(self) -> bool:
        return self.ngram_memory.sparse_sync_enabled

    def parameter_placements(self) -> dict[str, str]:
        placements: dict[str, str] = {}
        for name, _parameter in self.named_parameters():
            if ".local_experts." in name:
                placements[name] = PLACEMENT_EXPERT_SHARDED
            elif name.startswith("ngram_memory.tables."):
                placements[name] = (
                    PLACEMENT_ROW_SHARDED_TABLE
                    if self.config.ngram_memory.table_mode == "row_sharded"
                    else PLACEMENT_SPARSE_TABLE
                )
            else:
                placements[name] = PLACEMENT_REPLICATED
        return placements

    def named_parameters_by_placement(
        self,
        placement: str,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        if placement not in {
            PLACEMENT_REPLICATED,
            PLACEMENT_EXPERT_SHARDED,
            PLACEMENT_SPARSE_TABLE,
            PLACEMENT_ROW_SHARDED_TABLE,
        }:
            raise ValueError(f"Unknown parameter placement: {placement}")
        placements = self.parameter_placements()
        for name, parameter in self.named_parameters():
            if placements[name] == placement:
                yield name, parameter

    def precision_roles(self) -> dict[str, str]:
        roles: dict[str, str] = {}
        for module_name, module in self.named_modules():
            role = getattr(module, "metis_precision_role", None)
            if role is None:
                continue
            for local_name, _parameter in module.named_parameters(recurse=False):
                name = f"{module_name}.{local_name}" if module_name else local_name
                roles[name] = str(role)
        for name, _parameter in self.named_parameters():
            if name not in roles:
                if "router" in name or name.startswith("continuation."):
                    roles[name] = "router_logits"
                elif name.startswith("embedding."):
                    roles[name] = "embedding"
                else:
                    roles[name] = "bfloat16_state"
        return roles

    def master_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def local_parameter_audit(self) -> dict[str, int]:
        placements = self.parameter_placements()
        counts = {
            PLACEMENT_REPLICATED: 0,
            PLACEMENT_EXPERT_SHARDED: 0,
            PLACEMENT_SPARSE_TABLE: 0,
            PLACEMENT_ROW_SHARDED_TABLE: 0,
        }
        unique: set[int] = set()
        for name, parameter in self.named_parameters():
            identity = id(parameter)
            if identity in unique:
                continue
            unique.add(identity)
            counts[placements[name]] += parameter.numel()
        counts["local_total"] = sum(counts.values())
        counts["logical_total"] = self.config.logical_parameter_audit().stored_total
        return counts

    def _zero_route_state(
        self,
        input_ids: Tensor,
        *,
        dtype: torch.dtype,
    ) -> RouteState:
        shape = (*input_ids.shape, self.config.route_feature_dim)
        zero_summary = torch.zeros(shape, device=input_ids.device, dtype=dtype)
        zero_scalar = torch.zeros_like(input_ids, dtype=dtype)
        return RouteState(
            summary=zero_summary,
            mean_k=zero_scalar,
            expected_k=zero_scalar,
            entropy=zero_scalar,
            confidence=zero_scalar,
            token_difficulty=zero_scalar,
            assignments=torch.zeros((), device=input_ids.device, dtype=torch.long),
            processed_assignments=torch.zeros((), device=input_ids.device, dtype=torch.long),
            expert_counts=torch.zeros(
                self.config.n_routed_experts,
                device=input_ids.device,
                dtype=torch.long,
            ),
            expert_load_cv=torch.zeros((), device=input_ids.device, dtype=torch.float32),
            all_to_all_bytes=torch.zeros((), device=input_ids.device, dtype=torch.long),
            all_to_all_seconds=torch.zeros((), device=input_ids.device, dtype=torch.float64),
        )

    def _random_depth_generator(
        self,
        curriculum: CurriculumState,
        device: torch.device,
        pass_index: int,
    ) -> torch.Generator | None:
        # Seeded per call for the same reason as _random_policy_generator: a
        # cached generator does not replay under activation recompute.
        if not curriculum.random_policy_seed:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(
            (
                int(curriculum.random_policy_seed)
                + 7_919
                + 10_000_019 * int(pass_index)
                + 100_003 * int(curriculum.random_policy_step)
            )
            % (2**63 - 1)
        )
        return generator

    def _continuation_decision(
        self,
        probability: Tensor,
        *,
        active_mask: Tensor,
        pass_index: int,
        curriculum: CurriculumState,
        force_depth: int | Tensor | None,
    ) -> Tensor:
        next_pass_number = pass_index + 2
        if force_depth is not None:
            if isinstance(force_depth, int):
                decision = torch.full_like(active_mask, force_depth >= next_pass_number)
            else:
                if force_depth.shape != active_mask.shape:
                    raise ValueError("force_depth tensor must have the same shape as input_ids.")
                decision = force_depth >= next_pass_number
        elif curriculum.continuation_mode == "depth_one":
            decision = torch.zeros_like(active_mask)
        elif curriculum.continuation_mode == "fixed_max":
            decision = torch.ones_like(active_mask)
        elif curriculum.continuation_mode == "random":
            # Memoryless halt tuned to the same mean depth as the learned
            # policy.  The continuation router still runs -- its parameters and
            # its budget loss stay in the model so the comparison is
            # parameter-matched -- but nothing it produces reaches this
            # decision.
            target = (
                curriculum.target_mean_depth
                if curriculum.target_mean_depth is not None
                else self.config.target_mean_passes
            )
            continue_probability = geometric_continue_probability(
                self.config.max_passes, float(target)
            )
            draws = torch.rand(
                probability.shape,
                device=probability.device,
                dtype=torch.float32,
                generator=self._random_depth_generator(
                    curriculum, probability.device, pass_index
                ),
            )
            decision = draws < continue_probability
        elif self.training and curriculum.stochastic_routing:
            decision = torch.rand_like(probability) < probability
        else:
            decision = probability >= 0.5
        return active_mask & decision

    def _retrieve_ngram_memory(
        self,
        input_ids: Tensor,
        *,
        canonical_ids: Tensor | None,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        attention_mask: Tensor,
        context_parallel: ContextParallelContext,
    ) -> Tensor:
        """Look up the N-gram memory, with the left context a shard is missing.

        An order-``n`` key hashes the ``n-1`` preceding tokens, so on every
        shard but the first those leading positions would hash against zero
        padding and then be masked out as invalid n-grams.  Only two tokens per
        shard, but they are wrong *by construction* and stay wrong for the whole
        run, so the token halo is worth its two-position gather.
        """

        if not context_parallel.enabled:
            return self._precision_call(
                self.ngram_memory.retrieve,
                input_ids,
                canonical_ids=canonical_ids,
                document_ids=document_ids,
                reset_mask=reset_mask,
                attention_mask=attention_mask,
            )

        width = max(self.config.ngram_memory.orders) - 1
        if width <= 0:
            return self._precision_call(
                self.ngram_memory.retrieve,
                input_ids,
                canonical_ids=canonical_ids,
                document_ids=document_ids,
                reset_mask=reset_mask,
                attention_mask=attention_mask,
            )

        def _extend(values: Tensor | None, fill: float | int | bool) -> Tensor | None:
            if values is None:
                return None
            return torch.cat(
                (left_halo(values, context_parallel, width=width, fill=fill), values),
                dim=1,
            )

        # Rank 0's halo is masked out, so its content is irrelevant; the fills
        # only have to keep the true first position looking like a start.
        extended_ids = _extend(input_ids, 0)
        retrieved = self._precision_call(
            self.ngram_memory.retrieve,
            extended_ids,
            canonical_ids=_extend(canonical_ids, 0),
            document_ids=_extend(document_ids, -1),
            reset_mask=_extend(reset_mask, True),
            attention_mask=_extend(attention_mask, False),
        )
        return retrieved[:, width:]

    def _run_layer_unit(
        self,
        streams: Tensor,
        route_history: Tensor,
        state_difference: Tensor,
        pass_embedding: Tensor,
        *memory_inputs: Tensor,
        layer_index: int,
        memory_entry_count: int,
        active_mask: Tensor,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        attention_mask: Tensor,
        packed_layout: PackedDocumentLayout,
        pass_index: int,
        curriculum: CurriculumState,
        context_parallel: "ContextParallelPassState | None",
    ) -> tuple[Tensor, ...]:
        """Run one block, plus the memory read that feeds it, as a pure function.

        This is the finer of the two activation-recompute boundaries.  A whole
        pass costs about 137 GiB of live activations for Praxis at 163,840
        tokens and 287 GiB for Logos, against 128 GB of unified HBM per APU --
        and because that is a per-rank figure, adding ranks does not help.
        Replaying a block at a time instead of a pass at a time keeps one
        block's activations live and brings Praxis under the limit outright.

        The FLOP cost is unchanged: the same forward is re-executed either way,
        which is already priced into the 8/6 replay factor the throughput model
        uses.  What changes is that recompute granularity is now decoupled from
        the recurrence, so context extension can pay for memory in blocks
        instead of in passes.

        Memory-bank entries arrive as explicit tensor arguments rather than
        through the bank object, so the checkpoint boundary keeps no closure
        over layer activations.
        """

        expected_memory_inputs = 2 * memory_entry_count
        if len(memory_inputs) != expected_memory_inputs:
            raise RuntimeError(
                "Layer-unit memory input count does not match its checkpoint boundary."
            )
        bank = RecurrentMemoryBank(
            entries=list(memory_inputs[:memory_entry_count]),
            valid_masks=list(memory_inputs[memory_entry_count:]),
        )
        layer = self.layers[layer_index]
        streams, memory_summary, _memory_weights = self._precision_call(
            self.depth_memory.retrieve,
            bank,
            streams,
            active_mask=active_mask,
            gate_scale=curriculum.memory_gate_scale,
        )
        state = streams.mean(dim=-2)
        route_features = self._precision_call(
            self.depth_memory.routing_features,
            state,
            memory_summary,
            state_difference,
            route_history,
        )
        streams, route_state = self._precision_call(
            layer,
            streams,
            route_features=route_features,
            active_mask=active_mask,
            document_ids=document_ids,
            reset_mask=reset_mask,
            sequence_mask=attention_mask,
            packed_layout=packed_layout,
            pass_index=pass_index,
            pass_embedding=pass_embedding,
            curriculum=curriculum,
            pathway_cache=self._pathway_cache,
            context_parallel=context_parallel,
        )
        return (streams, *_route_state_to_flat(route_state))

    def _run_physical_pass(
        self,
        streams: Tensor,
        route_history: Tensor,
        state_difference: Tensor,
        cached_ngram: Tensor,
        pass_embedding: Tensor,
        continuation_confidence: Tensor,
        active_mask: Tensor,
        *memory_inputs: Tensor,
        pass_index: int,
        memory_entry_count: int,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        attention_mask: Tensor,
        packed_layout: PackedDocumentLayout,
        curriculum: CurriculumState,
        context_parallel: "ContextParallelPassState | None" = None,
    ) -> tuple[Tensor, ...]:
        """Run one recurrent pass as a pure checkpointable tensor function.

        Existing memory entries and masks are explicit inputs so the checkpoint
        boundary does not retain layer activations through a closure. Attention
        anchors written inside this pass are returned as compact memory tensors;
        mutating the outer memory bank is deliberately left to the caller.
        """

        expected_memory_inputs = 2 * memory_entry_count
        if len(memory_inputs) != expected_memory_inputs:
            raise RuntimeError(
                "Physical-pass memory input count does not match its checkpoint boundary."
            )
        base_entries = list(memory_inputs[:memory_entry_count])
        base_masks = list(memory_inputs[memory_entry_count:])
        bank = RecurrentMemoryBank(entries=base_entries, valid_masks=base_masks)
        zero = streams.float().sum() * 0.0
        auxiliary_losses = {
            name: zero for name in ROUTE_AUXILIARY_LOSS_NAMES
        }
        assignments = torch.zeros((), device=streams.device, dtype=torch.long)
        processed_assignments = torch.zeros((), device=streams.device, dtype=torch.long)
        all_to_all_bytes = torch.zeros((), device=streams.device, dtype=torch.long)
        routed_k_sum = zero
        routed_k_count = torch.zeros((), device=streams.device, dtype=torch.long)
        expert_entropy_sum = zero
        expert_entropy_count = torch.zeros((), device=streams.device, dtype=torch.long)
        expert_load_cv_sum = zero
        expert_load_cv_count = torch.zeros((), device=streams.device, dtype=torch.long)
        expert_selection_counts = torch.zeros(
            (self.config.n_layers, self.config.n_routed_experts),
            device=streams.device,
            dtype=torch.long,
        )
        sinkhorn_error = zero
        last_route_state = self._zero_route_state(
            active_mask.to(torch.long),
            dtype=streams.dtype,
        )
        attention_anchor = 0

        recompute_layers = (
            self.training
            and torch.is_grad_enabled()
            and self.activation_recompute_policy == "layer"
        )

        for layer_index, layer in enumerate(self.layers):
            unit_arguments = (
                streams,
                route_history,
                state_difference,
                pass_embedding,
                *bank.entries,
                *bank.valid_masks,
            )
            unit_keywords = {
                "layer_index": layer_index,
                "memory_entry_count": bank.slot_count,
                "active_mask": active_mask,
                "document_ids": document_ids,
                "reset_mask": reset_mask,
                "attention_mask": attention_mask,
                "packed_layout": packed_layout,
                "pass_index": pass_index,
                "curriculum": curriculum,
                "context_parallel": context_parallel,
            }
            if recompute_layers:

                def checkpointed_layer(
                    *arguments: Tensor,
                    _keywords: dict[str, Any] = unit_keywords,
                ) -> tuple[Tensor, ...]:
                    return self._run_layer_unit(*arguments, **_keywords)

                # Same reasoning as the pass-level boundary: early stop would
                # let ranks replay different amounts of the block and desync the
                # expert-parallel collective sequence.
                with set_checkpoint_early_stop(False):
                    unit_outputs = checkpoint(
                        checkpointed_layer,
                        *unit_arguments,
                        use_reentrant=False,
                        preserve_rng_state=True,
                        context_fn=_activation_checkpoint_context_fn(
                            self.precision_policy
                        ),
                    )
            else:
                unit_outputs = self._run_layer_unit(*unit_arguments, **unit_keywords)
            streams = unit_outputs[0]
            route_state = _route_state_from_flat(unit_outputs[1:])
            route_history = torch.where(
                active_mask.unsqueeze(-1),
                0.5 * route_history + 0.5 * route_state.summary,
                route_history,
            )
            last_route_state = route_state
            valid_k = route_state.mean_k.masked_select(active_mask)
            if valid_k.numel():
                routed_k_sum = routed_k_sum + valid_k.float().sum()
                routed_k_count = routed_k_count + valid_k.new_tensor(
                    valid_k.numel(),
                    dtype=torch.long,
                )
            valid_entropy = route_state.entropy.masked_select(active_mask)
            if valid_entropy.numel():
                expert_entropy_sum = expert_entropy_sum + valid_entropy.float().sum()
                expert_entropy_count = expert_entropy_count + valid_entropy.new_tensor(
                    valid_entropy.numel(),
                    dtype=torch.long,
                )
            assignments = assignments + route_state.assignments
            processed_assignments = (
                processed_assignments + route_state.processed_assignments
            )
            all_to_all_bytes = all_to_all_bytes + route_state.all_to_all_bytes
            expert_load_cv_sum = expert_load_cv_sum + route_state.expert_load_cv.float()
            expert_load_cv_count = expert_load_cv_count + expert_load_cv_count.new_ones(())
            expert_selection_counts[layer_index].add_(route_state.expert_counts)
            for name, value in route_state.auxiliary_losses.items():
                if name not in auxiliary_losses:
                    raise RuntimeError(f"Unsupported routed auxiliary loss: {name}")
                auxiliary_losses[name] = auxiliary_losses[name] + value
            for connection in (layer.mixer_connection, layer.moe_connection):
                if connection.last_sinkhorn_error is not None:
                    sinkhorn_error = torch.maximum(
                        sinkhorn_error,
                        connection.last_sinkhorn_error.float(),
                    )
            if layer.is_attention:
                self._precision_call(
                    self.depth_memory.write,
                    bank,
                    streams,
                    route_state=route_state,
                    pass_index=pass_index,
                    anchor_index=attention_anchor,
                    continuation_confidence=continuation_confidence,
                    active_mask=active_mask,
                )
                attention_anchor += 1
            streams = self.ngram_memory.inject(
                streams,
                cached_ngram,
                layer_index=layer_index,
                pass_index=pass_index,
                active_mask=active_mask,
                gate_scale=curriculum.memory_gate_scale,
            )

        new_entries = tuple(bank.entries[memory_entry_count:])
        new_masks = tuple(bank.valid_masks[memory_entry_count:])
        if len(new_entries) != self.config.n_attention_layers:
            raise RuntimeError(
                "Physical pass did not write exactly one memory anchor per attention layer."
            )
        return (
            streams,
            route_history,
            last_route_state.summary,
            last_route_state.mean_k,
            last_route_state.expected_k,
            last_route_state.entropy,
            last_route_state.confidence,
            last_route_state.token_difficulty,
            assignments,
            processed_assignments,
            all_to_all_bytes,
            expert_selection_counts,
            routed_k_sum,
            routed_k_count,
            expert_entropy_sum,
            expert_entropy_count,
            expert_load_cv_sum,
            expert_load_cv_count,
            sinkhorn_error,
            *(auxiliary_losses[name] for name in ROUTE_AUXILIARY_LOSS_NAMES),
            *new_entries,
            *new_masks,
        )

    def _causal_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Cross entropy for data-pipeline next-token-aligned labels.

        ``labels[:, t]`` is already ``input_ids[:, t + 1]`` (including shard
        lookahead), so the model must not shift labels a second time.
        """

        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must have shape [batch, sequence].")
        valid = labels != -100
        if not bool(valid.any().item()):
            return logits.float().sum() * 0.0
        return F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def _chunked_causal_loss(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError("labels must have shape [batch, sequence].")
        valid_total = int((labels != -100).sum().item())
        loss_sum = hidden_states.float().sum() * 0.0
        chunk_size = self.config.lm_head_chunk_size
        for start in range(0, hidden_states.shape[1], chunk_size):
            end = min(hidden_states.shape[1], start + chunk_size)
            chunk_labels = labels[:, start:end]
            def chunk_loss(chunk_hidden: Tensor, fixed_labels: Tensor = chunk_labels) -> Tensor:
                with _execution_context(
                    self.precision_policy,
                    module=self.lm_head,
                ):
                    chunk_logits = self.lm_head(chunk_hidden)
                    if not bool((fixed_labels != -100).any().item()):
                        return chunk_logits.float().sum() * 0.0
                    return F.cross_entropy(
                        chunk_logits.float().reshape(-1, chunk_logits.shape[-1]),
                        fixed_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )

            if self.training and torch.is_grad_enabled():
                with set_checkpoint_early_stop(False):
                    chunk_value = checkpoint(
                        chunk_loss,
                        hidden_states[:, start:end],
                        use_reentrant=False,
                        context_fn=_activation_checkpoint_context_fn(
                            self.precision_policy
                        ),
                    )
            else:
                chunk_value = chunk_loss(hidden_states[:, start:end])
            loss_sum = loss_sum + chunk_value
        return loss_sum / float(max(valid_total, 1))

    def _chunked_weighted_causal_loss_sum(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        weights: Tensor,
        *,
        compute_mask: Tensor,
    ) -> Tensor:
        """Memory-bounded per-exit LM loss for hard-forward ACT credit.

        Only hard-active tokens run the vocabulary projection. ``weights`` are
        straight-through exit masses: exactly zero/one in the forward pass,
        differentiable with respect to cumulative continuation hazards.
        """

        if labels.shape != hidden_states.shape[:2]:
            raise ValueError("labels must have shape [batch, sequence].")
        if weights.shape != labels.shape or compute_mask.shape != labels.shape:
            raise ValueError("weighted loss masks must have the label shape.")
        selected = compute_mask & (labels != -100)
        flat_indices = torch.nonzero(selected.reshape(-1), as_tuple=False).flatten()
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_weights = weights.reshape(-1)
        loss_sum = hidden_states.float().sum() * 0.0
        chunk_size = self.config.lm_head_chunk_size
        global_selected = torch.tensor(
            int(flat_indices.numel()),
            device=hidden_states.device,
            dtype=torch.long,
        )
        if _group_world_size(self.process_groups.world) > 1:
            dist.all_reduce(
                global_selected,
                op=dist.ReduceOp.MAX,
                group=self.process_groups.world,
            )
        for start in range(0, int(global_selected.item()), chunk_size):
            positions = flat_indices[start : start + chunk_size]
            if positions.numel():
                chunk_hidden = flat_hidden.index_select(0, positions)
                chunk_labels = flat_labels.index_select(0, positions)
                chunk_weights = flat_weights.index_select(0, positions)
            else:
                # Every world rank must enter/exit the same delayed-scaling
                # collective schedule even when ACT leaves it no local exits.
                chunk_hidden = flat_hidden[:1] * 0.0
                chunk_labels = flat_labels.new_zeros((1,))
                chunk_weights = flat_weights.new_zeros((1,))

            def chunk_loss(
                values: Tensor,
                targets: Tensor,
                exit_weights: Tensor,
            ) -> Tensor:
                with _execution_context(
                    self.precision_policy,
                    module=self.lm_head,
                ):
                    chunk_logits = self.lm_head(values)
                    token_losses = F.cross_entropy(
                        chunk_logits.float(),
                        targets,
                        reduction="none",
                    )
                    return torch.sum(token_losses * exit_weights.float())

            if self.training and torch.is_grad_enabled():
                with set_checkpoint_early_stop(False):
                    chunk_value = checkpoint(
                        chunk_loss,
                        chunk_hidden,
                        chunk_labels,
                        chunk_weights,
                        use_reentrant=False,
                        context_fn=_activation_checkpoint_context_fn(
                            self.precision_policy
                        ),
                    )
            else:
                chunk_value = chunk_loss(
                    chunk_hidden,
                    chunk_labels,
                    chunk_weights,
                )
            loss_sum = loss_sum + chunk_value
        return loss_sum

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        curriculum: CurriculumState | Mapping[str, Any] | None = None,
        attention_mask: Tensor | None = None,
        document_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        canonical_ids: Tensor | None = None,
        max_passes: int | None = None,
        force_depth: int | Tensor | None = None,
        return_logits: bool | None = None,
    ) -> Metis16CausalLMOutput:
        return self._forward_impl(
            input_ids,
            labels,
            curriculum=curriculum,
            attention_mask=attention_mask,
            document_ids=document_ids,
            reset_mask=reset_mask,
            canonical_ids=canonical_ids,
            max_passes=max_passes,
            force_depth=force_depth,
            return_logits=return_logits,
        )

    def _forward_impl(
        self,
        input_ids: Tensor,
        labels: Tensor | None,
        *,
        curriculum: CurriculumState | Mapping[str, Any] | None,
        attention_mask: Tensor | None,
        document_ids: Tensor | None,
        reset_mask: Tensor | None,
        canonical_ids: Tensor | None,
        max_passes: int | None,
        force_depth: int | Tensor | None,
        return_logits: bool | None,
    ) -> Metis16CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        maximum_context = (
            self.config.context_extension_train_length
            if self.training
            else self.config.final_context_length
        )
        context_parallel = ContextParallelContext(
            group=self.process_groups.context,
            size=self.context_parallel_size,
            rank=self.context_parallel_rank,
            local_length=int(input_ids.shape[1]),
        )
        # Under context parallelism the limit applies to the sequence the group
        # jointly owns, not to the slice this rank happens to hold.
        global_length = input_ids.shape[1] * context_parallel.size
        if global_length > maximum_context:
            raise ValueError(
                f"Sequence length {global_length} exceeds the active "
                f"Metis-1.6 context limit {maximum_context}."
            )
        if context_parallel.enabled and input_ids.shape[0] != 1:
            # Continuation packing flattens [batch, sequence] into one row
            # ordered by batch and then position.  Concatenated across shards
            # that interleaves the batch rows, so rank r+1's row-0 tokens would
            # inherit rank r's row-1 SSM state instead of its own.  Context
            # extension runs one sequence per rank anyway -- 163,840 tokens is
            # already the whole micro-batch -- so this is a guard, not a
            # limitation anyone has to work around.
            raise ValueError(
                "Context parallelism requires micro-batch 1; observed "
                f"{input_ids.shape[0]}. Packing interleaves batch rows across "
                "shards, which would carry mixer state between them."
            )
        curriculum_state = CurriculumState.from_value(curriculum)
        curriculum_state.validate(self.config)
        self._pathway_cache = (
            PathwayCache() if curriculum_state.pathway_mode == "frozen" else None
        )
        effective_passes = max_passes or curriculum_state.max_passes or self.config.max_passes
        if not 1 <= effective_passes <= self.config.max_passes:
            raise ValueError("max_passes is outside [1, config.max_passes].")
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(torch.bool)
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")
            if bool(((labels != -100) & ~attention_mask).any().item()):
                raise ValueError(
                    "Labels outside attention_mask must use ignore_index -100."
                )
        document_ids, reset_mask = (
            align_document_ids(
                input_ids,
                document_ids=document_ids,
                reset_mask=reset_mask,
                context=context_parallel,
            )
            if context_parallel.enabled
            else _derive_document_ids(
                input_ids,
                document_ids=document_ids,
                reset_mask=reset_mask,
            )
        )
        packed_layout = _build_packed_document_layout(attention_mask, document_ids)
        segment_stride = (
            global_segment_stride(document_ids, context_parallel)
            if context_parallel.enabled
            else 1
        )
        if self.training and self.config.world_size > 1:
            replicated_requires_group = self.config.ngram_memory.table_mode == "replicated"
            sharded_replicas_require_group = (
                self.config.ngram_memory.table_mode == "row_sharded"
                and self.config.expert_replicas > 1
            )
            if (
                replicated_requires_group or sharded_replicas_require_group
            ) and self.process_groups.table_gradient is None:
                raise RuntimeError(
                    "The manifest requires an explicit table_gradient process "
                    "group; None is a local-only sentinel."
                )
        self.ngram_memory.assert_sparse_sync_ready(
            self.process_groups.table_gradient,
            training=self.training,
        )
        self.collective_timer.reset()
        embeddings = self.embedding(input_ids)
        streams = embeddings.unsqueeze(-2) + self.stream_embeddings.view(
            1,
            1,
            self.config.n_streams,
            self.config.d_model,
        )
        cached_ngram = self._retrieve_ngram_memory(
            input_ids,
            canonical_ids=canonical_ids,
            document_ids=document_ids,
            reset_mask=reset_mask,
            attention_mask=attention_mask,
            context_parallel=context_parallel,
        )
        bank = RecurrentMemoryBank()
        active_mask = attention_mask.clone()
        active_masks: list[Tensor] = []
        chosen_depths = torch.zeros_like(input_ids, dtype=torch.long)
        initial_state = streams.mean(dim=-2)
        previous_pass_state = initial_state
        route_history = embeddings.new_zeros(
            *input_ids.shape,
            self.config.route_feature_dim,
        )
        last_route_state = self._zero_route_state(input_ids, dtype=embeddings.dtype)
        continuation_confidence = embeddings.new_zeros(input_ids.shape)
        auxiliary_losses: dict[str, Tensor] = {}
        routed_k_sum = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        routed_k_count = torch.zeros((), device=input_ids.device, dtype=torch.long)
        expert_entropy_sum = torch.zeros(
            (), device=input_ids.device, dtype=torch.float32
        )
        expert_entropy_count = torch.zeros(
            (), device=input_ids.device, dtype=torch.long
        )
        assignments = torch.zeros((), device=input_ids.device, dtype=torch.long)
        processed_assignments = torch.zeros((), device=input_ids.device, dtype=torch.long)
        all_to_all_bytes = torch.zeros((), device=input_ids.device, dtype=torch.long)
        expert_load_cv_sum = torch.zeros(
            (), device=input_ids.device, dtype=torch.float32
        )
        expert_load_cv_count = torch.zeros(
            (), device=input_ids.device, dtype=torch.long
        )
        expert_selection_counts = torch.zeros(
            (self.config.n_layers, self.config.n_routed_experts),
            device=input_ids.device,
            dtype=torch.long,
        )
        sinkhorn_error = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        survival = attention_mask.float()
        expected_depth = survival.clone()
        last_memory_summary = torch.zeros_like(initial_state)
        activation_recompute_used = False
        pass_survival_gate = attention_mask.float()
        ponder_loss_sum = embeddings.float().sum() * 0.0
        exit_mass_sum = torch.zeros(
            (), device=input_ids.device, dtype=torch.float32
        )
        exit_mass_by_token = torch.zeros_like(attention_mask, dtype=torch.float32)
        executed_active_tokens = torch.zeros(
            (), device=input_ids.device, dtype=torch.long
        )
        packed_passes = torch.zeros((), device=input_ids.device, dtype=torch.long)

        for pass_index in range(effective_passes):
            active_masks.append(active_mask)
            local_has_active = bool(active_mask.any().item())
            if not _group_any_active(
                active_mask,
                group=(
                    self.process_groups.world
                    if self.process_groups.world is not None
                    else self.process_groups.expert
                ),
                groups=(self.process_groups.context,),
            ):
                active_masks.extend(
                    active_mask.clone()
                    for _ in range(pass_index + 1, effective_passes)
                )
                break
            chosen_depths = torch.where(
                active_mask,
                torch.full_like(chosen_depths, pass_index + 1),
                chosen_depths,
            )
            executed_active_tokens = executed_active_tokens + active_mask.sum()
            pass_embedding = self.pass_embeddings[pass_index]
            state_difference = streams.mean(dim=-2) - previous_pass_state
            pass_input_streams = streams
            pass_input_route_history = route_history
            memory_entry_count = bank.slot_count
            token_layout: ActiveTokenLayout | None = None
            pass_context_parallel: ContextParallelPassState | None = None
            run_document_ids = document_ids
            run_reset_mask = reset_mask
            run_attention_mask = attention_mask
            run_packed_layout = packed_layout
            run_streams = streams
            run_route_history = route_history
            run_state_difference = state_difference
            run_cached_ngram = cached_ngram
            run_continuation_confidence = continuation_confidence
            run_active_mask = active_mask
            run_memory_entries: tuple[Tensor, ...] = tuple(bank.entries)
            run_memory_masks: tuple[Tensor, ...] = tuple(bank.valid_masks)
            if int(active_mask.sum().item()) < input_ids.numel():
                token_layout = (
                    _active_token_layout(active_mask)
                    if local_has_active
                    else ActiveTokenLayout(
                        torch.zeros(
                            1,
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                        input_ids.shape[0],
                        input_ids.shape[1],
                    )
                )
                packed_passes = packed_passes + 1
                run_attention_mask = torch.ones(
                    (1, token_layout.token_count),
                    device=input_ids.device,
                    dtype=torch.bool,
                )
                # The packed buffer's first token continues a neighbouring
                # shard's document more often than not, and only the group
                # knows that, so the CP state has to be resolved before the
                # reset flags can be written.
                pass_context_parallel = (
                    _build_context_parallel_pass_state(
                        context_parallel,
                        document_ids=document_ids,
                        selector=token_layout.flat_token_indices,
                        batch_size=input_ids.shape[0],
                        sequence_length=input_ids.shape[1],
                        segment_stride=segment_stride,
                        local_count=(
                            token_layout.token_count if local_has_active else 0
                        ),
                    )
                    if context_parallel.enabled
                    else None
                )
                run_document_ids, run_reset_mask = _packed_document_metadata(
                    token_layout,
                    document_ids,
                    continues_previous=(
                        pass_context_parallel.continues_previous
                        if pass_context_parallel is not None
                        else False
                    ),
                )
                run_packed_layout = _build_packed_document_layout(
                    run_attention_mask,
                    run_document_ids,
                )
                run_streams = token_layout.pack(streams)
                run_route_history = token_layout.pack(route_history)
                run_state_difference = token_layout.pack(state_difference)
                run_cached_ngram = token_layout.pack(cached_ngram)
                run_continuation_confidence = token_layout.pack(
                    continuation_confidence
                )
                run_active_mask = (
                    run_attention_mask
                    if local_has_active
                    else torch.zeros_like(run_attention_mask)
                )
                run_memory_entries = tuple(
                    token_layout.pack(entry) for entry in bank.entries
                )
                run_memory_masks = tuple(
                    token_layout.pack(mask) for mask in bank.valid_masks
                )
            if context_parallel.enabled and pass_context_parallel is None:
                pass_context_parallel = _build_context_parallel_pass_state(
                    context_parallel,
                    document_ids=document_ids,
                    selector=run_packed_layout.flat_token_indices,
                    batch_size=input_ids.shape[0],
                    sequence_length=input_ids.shape[1],
                    segment_stride=segment_stride,
                    local_count=(
                        int(run_packed_layout.flat_token_indices.numel())
                        if local_has_active
                        else 0
                    ),
                )
            # The pathway cache stores pass-one identities in the unpacked token
            # layout, so it needs this pass's packing map to gather them back.
            if self._pathway_cache is not None:
                self._pathway_cache.set_layout(token_layout)
            pass_arguments = (
                run_streams,
                run_route_history,
                run_state_difference,
                run_cached_ngram,
                pass_embedding,
                run_continuation_confidence,
                run_active_mask,
                *run_memory_entries,
                *run_memory_masks,
            )
            recompute_this_pass = (
                self.training
                and torch.is_grad_enabled()
                and self.activation_recompute_policy == "pass"
            )
            if self.training and torch.is_grad_enabled() and (
                self.activation_recompute_policy in {"pass", "layer"}
            ):
                activation_recompute_used = True
            if recompute_this_pass:

                def checkpointed_pass(
                    *arguments: Tensor,
                    _pass_index: int = pass_index,
                    _memory_entry_count: int = memory_entry_count,
                    _document_ids: Tensor | None = run_document_ids,
                    _reset_mask: Tensor | None = run_reset_mask,
                    _attention_mask: Tensor = run_attention_mask,
                    _packed_layout: PackedDocumentLayout = run_packed_layout,
                    _context_parallel: ContextParallelPassState | None = pass_context_parallel,
                ) -> tuple[Tensor, ...]:
                    return self._run_physical_pass(
                        *arguments,
                        pass_index=_pass_index,
                        memory_entry_count=_memory_entry_count,
                        document_ids=_document_ids,
                        reset_mask=_reset_mask,
                        attention_mask=_attention_mask,
                        packed_layout=_packed_layout,
                        curriculum=curriculum_state,
                        context_parallel=_context_parallel,
                    )

                # Disabling non-reentrant early-stop makes every rank replay the
                # complete pass and therefore the same EP collective sequence.
                with set_checkpoint_early_stop(False):
                    pass_outputs = checkpoint(
                        checkpointed_pass,
                        *pass_arguments,
                        use_reentrant=False,
                        preserve_rng_state=True,
                        context_fn=_activation_checkpoint_context_fn(
                            self.precision_policy
                        ),
                    )
            else:
                pass_outputs = self._run_physical_pass(
                    *pass_arguments,
                    pass_index=pass_index,
                    memory_entry_count=memory_entry_count,
                    document_ids=run_document_ids,
                    reset_mask=run_reset_mask,
                    attention_mask=run_attention_mask,
                    packed_layout=run_packed_layout,
                    curriculum=curriculum_state,
                    context_parallel=pass_context_parallel,
                )
            (
                run_streams,
                run_route_history,
                run_last_route_summary,
                run_last_route_mean_k,
                run_last_route_expected_k,
                run_last_route_entropy,
                run_last_route_confidence,
                run_last_route_token_difficulty,
                pass_assignments,
                pass_processed_assignments,
                pass_all_to_all_bytes,
                pass_expert_selection_counts,
                pass_routed_k_sum,
                pass_routed_k_count,
                pass_expert_entropy_sum,
                pass_expert_entropy_count,
                pass_expert_load_cv_sum,
                pass_expert_load_cv_count,
                pass_sinkhorn_error,
                pass_expert_balance,
                pass_router_z,
                pass_routed_k_budget,
                *memory_outputs,
            ) = pass_outputs
            attention_memory_count = self.config.n_attention_layers
            if len(memory_outputs) != 2 * attention_memory_count:
                raise RuntimeError("Checkpointed pass returned an invalid memory layout.")
            run_new_entries = memory_outputs[:attention_memory_count]
            run_new_masks = memory_outputs[attention_memory_count:]
            if token_layout is not None:
                raw_streams = token_layout.scatter(
                    run_streams,
                    base=pass_input_streams,
                )
                raw_route_history = token_layout.scatter(
                    run_route_history,
                    base=pass_input_route_history,
                )
                last_route_summary = token_layout.scatter(run_last_route_summary)
                last_route_mean_k = token_layout.scatter(run_last_route_mean_k)
                last_route_expected_k = token_layout.scatter(
                    run_last_route_expected_k
                )
                last_route_entropy = token_layout.scatter(run_last_route_entropy)
                last_route_confidence = token_layout.scatter(
                    run_last_route_confidence
                )
                last_route_token_difficulty = token_layout.scatter(
                    run_last_route_token_difficulty
                )
                new_entries = tuple(
                    token_layout.scatter(entry) for entry in run_new_entries
                )
                new_masks = tuple(active_mask.clone() for _ in run_new_masks)
            else:
                raw_streams = run_streams
                raw_route_history = run_route_history
                last_route_summary = run_last_route_summary
                last_route_mean_k = run_last_route_mean_k
                last_route_expected_k = run_last_route_expected_k
                last_route_entropy = run_last_route_entropy
                last_route_confidence = run_last_route_confidence
                last_route_token_difficulty = run_last_route_token_difficulty
                new_entries = tuple(run_new_entries)
                new_masks = tuple(run_new_masks)
            stream_gate = pass_survival_gate[..., None, None].to(raw_streams.dtype)
            route_gate = pass_survival_gate[..., None].to(raw_route_history.dtype)
            streams = pass_input_streams + stream_gate * (
                raw_streams - pass_input_streams
            )
            route_history = pass_input_route_history + route_gate * (
                raw_route_history - pass_input_route_history
            )
            memory_gate = pass_survival_gate[..., None]
            bank.entries.extend(
                entry * memory_gate.to(entry.dtype) for entry in new_entries
            )
            bank.valid_masks.extend(new_masks)
            last_route_summary = last_route_summary * route_gate.to(
                last_route_summary.dtype
            )
            scalar_route_gate = pass_survival_gate.to(last_route_mean_k.dtype)
            last_route_mean_k = last_route_mean_k * scalar_route_gate
            last_route_expected_k = last_route_expected_k * scalar_route_gate
            last_route_entropy = last_route_entropy * scalar_route_gate
            last_route_confidence = last_route_confidence * scalar_route_gate
            last_route_token_difficulty = (
                last_route_token_difficulty * scalar_route_gate
            )
            last_route_state = RouteState(
                summary=last_route_summary,
                mean_k=last_route_mean_k,
                expected_k=last_route_expected_k,
                entropy=last_route_entropy,
                confidence=last_route_confidence,
                token_difficulty=last_route_token_difficulty,
                assignments=pass_assignments,
                processed_assignments=pass_processed_assignments,
                expert_counts=torch.zeros(
                    self.config.n_routed_experts,
                    device=input_ids.device,
                    dtype=torch.long,
                ),
                expert_load_cv=(
                    pass_expert_load_cv_sum
                    / pass_expert_load_cv_count.clamp_min(1).float()
                ),
                all_to_all_bytes=pass_all_to_all_bytes,
                all_to_all_seconds=torch.zeros(
                    (), device=input_ids.device, dtype=torch.float64
                ),
                auxiliary_losses={
                    "expert_balance": pass_expert_balance,
                    "expert_router_z": pass_router_z,
                    "routed_k_budget": pass_routed_k_budget,
                },
            )
            assignments = assignments + pass_assignments
            processed_assignments = (
                processed_assignments + pass_processed_assignments
            )
            all_to_all_bytes = all_to_all_bytes + pass_all_to_all_bytes
            expert_selection_counts = (
                expert_selection_counts + pass_expert_selection_counts
            )
            routed_k_sum = routed_k_sum + pass_routed_k_sum
            routed_k_count = routed_k_count + pass_routed_k_count
            expert_entropy_sum = expert_entropy_sum + pass_expert_entropy_sum
            expert_entropy_count = (
                expert_entropy_count + pass_expert_entropy_count
            )
            expert_load_cv_sum = expert_load_cv_sum + pass_expert_load_cv_sum
            expert_load_cv_count = (
                expert_load_cv_count + pass_expert_load_cv_count
            )
            sinkhorn_error = torch.maximum(sinkhorn_error, pass_sinkhorn_error)
            for name, value in last_route_state.auxiliary_losses.items():
                _add_loss(auxiliary_losses, name, value)

            before_continuation_streams = streams
            if token_layout is not None:
                continuation_bank = RecurrentMemoryBank(
                    entries=[
                        token_layout.pack(entry) for entry in bank.entries
                    ],
                    valid_masks=[
                        token_layout.pack(mask) for mask in bank.valid_masks
                    ],
                )
                continuation_streams = token_layout.pack(streams)
                continuation_active_mask = run_active_mask
                continuation_streams, packed_memory_summary, _memory_weights = (
                    self._precision_call(
                        self.depth_memory.retrieve,
                        continuation_bank,
                        continuation_streams,
                        active_mask=continuation_active_mask,
                        gate_scale=curriculum_state.memory_gate_scale,
                    )
                )
                packed_current_state = continuation_streams.mean(dim=-2)
                packed_state_difference = (
                    packed_current_state - token_layout.pack(previous_pass_state)
                )
                packed_route_features = self._precision_call(
                    self.depth_memory.routing_features,
                    packed_current_state,
                    packed_memory_summary,
                    packed_state_difference,
                    token_layout.pack(route_history),
                )
                packed_continuation_probability = self.continuation(
                    packed_current_state,
                    packed_memory_summary,
                    packed_state_difference,
                    packed_route_features,
                )
                raw_continuation_streams = token_layout.scatter(
                    continuation_streams,
                    base=before_continuation_streams,
                )
                streams = before_continuation_streams + stream_gate * (
                    raw_continuation_streams - before_continuation_streams
                )
                current_state = streams.mean(dim=-2)
                last_memory_summary = token_layout.scatter(packed_memory_summary)
                continuation_probability = token_layout.scatter(
                    packed_continuation_probability
                )
            else:
                raw_continuation_streams, last_memory_summary, _memory_weights = (
                    self._precision_call(
                        self.depth_memory.retrieve,
                        bank,
                        streams,
                        active_mask=active_mask,
                        gate_scale=curriculum_state.memory_gate_scale,
                    )
                )
                streams = before_continuation_streams + stream_gate * (
                    raw_continuation_streams - before_continuation_streams
                )
                current_state = streams.mean(dim=-2)
                state_difference = current_state - previous_pass_state
                continuation_route_features = self._precision_call(
                    self.depth_memory.routing_features,
                    current_state,
                    last_memory_summary,
                    state_difference,
                    route_history,
                )
                continuation_probability = self.continuation(
                    current_state,
                    last_memory_summary,
                    state_difference,
                    continuation_route_features,
                )
            continuation_confidence = torch.where(
                active_mask,
                continuation_probability.to(embeddings.dtype),
                continuation_confidence,
            )
            if token_layout is not None:
                continuation_bank = RecurrentMemoryBank(
                    entries=[
                        token_layout.pack(entry) for entry in bank.entries
                    ],
                    valid_masks=[
                        token_layout.pack(mask) for mask in bank.valid_masks
                    ],
                )
                packed_route_state = RouteState(
                    summary=token_layout.pack(last_route_state.summary),
                    mean_k=token_layout.pack(last_route_state.mean_k),
                    expected_k=token_layout.pack(last_route_state.expected_k),
                    entropy=token_layout.pack(last_route_state.entropy),
                    confidence=token_layout.pack(last_route_state.confidence),
                    token_difficulty=token_layout.pack(
                        last_route_state.token_difficulty
                    ),
                    assignments=last_route_state.assignments,
                    processed_assignments=last_route_state.processed_assignments,
                    expert_counts=last_route_state.expert_counts,
                    expert_load_cv=last_route_state.expert_load_cv,
                    all_to_all_bytes=last_route_state.all_to_all_bytes,
                    all_to_all_seconds=last_route_state.all_to_all_seconds,
                    auxiliary_losses=last_route_state.auxiliary_losses,
                )
                self._precision_call(
                    self.depth_memory.write,
                    continuation_bank,
                    token_layout.pack(streams),
                    route_state=packed_route_state,
                    pass_index=pass_index,
                    anchor_index=self.config.n_attention_layers,
                    continuation_confidence=token_layout.pack(
                        continuation_confidence
                    ),
                    active_mask=continuation_active_mask,
                )
                end_entry = token_layout.scatter(continuation_bank.entries[-1])
                bank.entries.append(
                    end_entry * memory_gate.to(end_entry.dtype)
                )
                bank.valid_masks.append(active_mask.clone())
            else:
                before_slots = bank.slot_count
                self._precision_call(
                    self.depth_memory.write,
                    bank,
                    streams,
                    route_state=last_route_state,
                    pass_index=pass_index,
                    anchor_index=self.config.n_attention_layers,
                    continuation_confidence=continuation_confidence,
                    active_mask=active_mask,
                )
                bank.entries[-1] = (
                    bank.entries[-1] * memory_gate.to(bank.entries[-1].dtype)
                )
                if bank.slot_count != before_slots + 1:
                    raise RuntimeError("Continuation memory write lost its slot.")
            if pass_index + 1 < effective_passes:
                next_active = self._continuation_decision(
                    continuation_probability,
                    active_mask=active_mask,
                    pass_index=pass_index,
                    curriculum=curriculum_state,
                    force_depth=force_depth,
                )
                soft_continue = continuation_probability * active_mask.float()
                local_continue_gate = (
                    next_active.float()
                    + soft_continue
                    - soft_continue.detach()
                )
                exit_gate = pass_survival_gate * (1.0 - local_continue_gate)
                next_survival_gate = pass_survival_gate * local_continue_gate
                survival = survival * continuation_probability * active_mask.float()
                expected_depth = expected_depth + survival
                valid_probability = continuation_probability.masked_select(active_mask)
                if valid_probability.numel():
                    entropy = -(
                        valid_probability * valid_probability.clamp_min(1.0e-8).log()
                        + (1.0 - valid_probability)
                        * (1.0 - valid_probability).clamp_min(1.0e-8).log()
                    ).mean()
                    _add_loss(
                        auxiliary_losses,
                        "continuation_entropy",
                        -entropy * self.config.continuation_entropy_coefficient,
                    )
            else:
                next_active = None
                exit_gate = pass_survival_gate
                next_survival_gate = pass_survival_gate
            exit_mass_sum = exit_mass_sum + exit_gate.detach().float().sum()
            exit_mass_by_token = (
                exit_mass_by_token + exit_gate.detach().float()
            )
            if labels is not None:
                exit_hidden = self.final_norm(streams.mean(dim=-2))
                ponder_loss_sum = ponder_loss_sum + self._chunked_weighted_causal_loss_sum(
                    exit_hidden,
                    labels,
                    exit_gate,
                    compute_mask=active_mask,
                )
            if next_active is not None:
                active_mask = next_active
                pass_survival_gate = next_survival_gate
            previous_pass_state = current_state

        valid_expected_depth = expected_depth.masked_select(attention_mask)
        mean_expected_depth = (
            valid_expected_depth.mean()
            if valid_expected_depth.numel()
            else expected_depth.sum() * 0.0
        )
        _add_loss(
            auxiliary_losses,
            "depth_budget",
            (
                mean_expected_depth
                - (
                    curriculum_state.target_mean_depth
                    if curriculum_state.target_mean_depth is not None
                    else self.config.target_mean_passes
                )
            ).square()
            * self.config.depth_budget_coefficient
            if valid_expected_depth.numel()
            else mean_expected_depth,
        )
        final_hidden = self.final_norm(streams.mean(dim=-2))
        if return_logits is None:
            return_logits = labels is None
        logits = (
            self._precision_call(self.lm_head, final_hidden)
            if return_logits
            else None
        )
        if labels is None:
            causal_loss = None
        else:
            valid_total = int((labels != -100).sum().item())
            causal_loss = ponder_loss_sum / float(max(valid_total, 1))
        graph_anchor = logits.float().sum() if logits is not None else final_hidden.float().sum()
        auxiliary_loss = (
            torch.stack([value.float() for value in auxiliary_losses.values()]).sum()
            if auxiliary_losses
            else graph_anchor * 0.0
        )
        valid_depths = chosen_depths.masked_select(attention_mask)
        mean_depth = (
            valid_depths.float().mean()
            if valid_depths.numel()
            else graph_anchor * 0.0
        )
        mean_k = (
            routed_k_sum / routed_k_count.float()
            if int(routed_k_count.item()) > 0
            else graph_anchor * 0.0
        )
        expert_entropy_ratio = (
            (
                expert_entropy_sum
                / expert_entropy_count.float()
                / self.config.expert_entropy_normalizer
            ).clamp(0.0, 1.0)
            if int(expert_entropy_count.item()) > 0
            else final_hidden.new_ones((), dtype=torch.float32)
        )
        adaptive_depth_enabled = (
            force_depth is None
            and curriculum_state.continuation_mode == "adaptive"
            and effective_passes > 1
        )
        if adaptive_depth_enabled and valid_depths.numel():
            depth_histogram = torch.bincount(
                valid_depths,
                minlength=effective_passes + 1,
            )[1 : effective_passes + 1]
            halt_collapse_fraction = (
                depth_histogram.max().float() / valid_depths.numel()
            )
        else:
            halt_collapse_fraction = final_hidden.new_zeros((), dtype=torch.float32)
        stacked_active = torch.stack(active_masks, dim=0)
        valid_tokens = attention_mask.sum().clamp_min(1)
        active_ratios = stacked_active.sum(dim=(1, 2)).float() / valid_tokens.float()
        stream_centered = streams.float() - streams.float().mean(dim=-2, keepdim=True)
        stream_diversity = stream_centered.square().mean()
        all_to_all_seconds, all_to_all_enqueue_seconds = self.collective_timer.finalize(
            final_hidden
        )
        expert_load_cv = (
            expert_load_cv_sum / expert_load_cv_count.float()
            if int(expert_load_cv_count.item()) > 0
            else graph_anchor * 0.0
        )
        telemetry = {
            "mean_depth": mean_depth,
            "mean_passes": mean_depth,
            "mean_expected_depth": mean_expected_depth,
            "mean_routed_k": mean_k,
            "active_token_ratios": active_ratios,
            "moe_assignments": assignments,
            "moe_processed_assignments": processed_assignments,
            "moe_dropped_assignments": assignments - processed_assignments,
            "overflow_drop_tokens": torch.zeros(
                (), device=input_ids.device, dtype=torch.long
            ),
            "expert_load_cv": expert_load_cv,
            # Kept as a tensor so scalar telemetry writers ignore it while the
            # trainer can advance each layer's synchronized selection bias.
            "expert_selection_counts": expert_selection_counts,
            "expert_entropy_ratio": expert_entropy_ratio,
            "halt_collapse_fraction": halt_collapse_fraction,
            "all_to_all_bytes": all_to_all_bytes,
            "all_to_all_seconds": all_to_all_seconds,
            "all_to_all_enqueue_seconds": all_to_all_enqueue_seconds,
            "dispatch_overlap_enabled": final_hidden.new_tensor(
                int(self.dispatch_overlap_enabled), dtype=torch.long
            ),
            "activation_recompute_enabled": final_hidden.new_tensor(
                int(activation_recompute_used), dtype=torch.long
            ),
            "packed_continuation_enabled": final_hidden.new_ones((), dtype=torch.long),
            "packed_continuation_passes": packed_passes,
            "executed_active_tokens": executed_active_tokens,
            "dense_envelope_tokens": final_hidden.new_tensor(
                int(attention_mask.sum().item()) * effective_passes,
                dtype=torch.long,
            ),
            "dense_pass_fallback_tokens": final_hidden.new_zeros((), dtype=torch.long),
            "ponder_exit_mass_error": (
                exit_mass_sum - attention_mask.sum().float()
            ).abs(),
            "ponder_exit_mass_max_error": (
                exit_mass_by_token - attention_mask.float()
            ).abs().amax(),
            "memory_slots_written": final_hidden.new_tensor(bank.slot_count, dtype=torch.long),
            "sinkhorn_max_marginal_error": sinkhorn_error,
            "mhc_stream_diversity": stream_diversity,
            "ngram_cached_vectors": final_hidden.new_tensor(input_ids.numel(), dtype=torch.long),
            "depth_memory_last_norm": last_memory_summary.float().norm(dim=-1).mean(),
        }
        return Metis16CausalLMOutput(
            logits=logits,
            loss=causal_loss,
            auxiliary_loss=auxiliary_loss,
            auxiliary_losses=auxiliary_losses,
            telemetry=telemetry,
            chosen_depths=chosen_depths,
            active_masks=stacked_active,
            final_hidden_state=final_hidden,
        )


__all__ = [
    "CurriculumState",
    "Metis16CausalLMOutput",
    "Metis16Config",
    "Metis16ForCausalLM",
    "MetisProcessGroups",
    "PLACEMENT_EXPERT_SHARDED",
    "PLACEMENT_REPLICATED",
    "PLACEMENT_ROW_SHARDED_TABLE",
    "PLACEMENT_SPARSE_TABLE",
    "expert_collective_wire_error",
    "fp8_wire_dtypes",
    "load_family_config",
    "sinkhorn_doubly_stochastic",
]
