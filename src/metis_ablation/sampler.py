"""Phase-proportional strided sampler over the immutable 1T release.

The ablation campaign trains on 50B tokens, but not the *first* 50B: a prefix
would over-weight phase A and under-weight the later curriculum, and the paper
claims the proxy uses the same mixture as the release.  Instead this walks the
whole 1T corpus with a uniform stride, which preserves the 700B/250B/50B phase
proportions exactly because a uniform stride preserves every proportion.

Two properties make the campaign's comparability claim true rather than
aspirational:

* Blocks are aligned to the global optimizer step, so a row running on 16 APUs
  and a row running on 56 APUs consume the *same contiguous token window* per
  step -- they merely partition it differently across ranks and micro-steps.
* The mapping depends only on the step index, never on world size, rank, or
  micro-batch, so it is reproducible from the manifest alone.
"""

from __future__ import annotations

import functools

from dataclasses import dataclass
from typing import Callable, Iterator

from metis_training.data import DeterministicReleaseStream, TrainingBatch
from metis_training.contracts import load_family_manifest  # noqa: F401  (re-export convenience)


PHASE_ORDER = ("phase_a", "phase_b", "phase_c")


@dataclass(frozen=True)
class PhasePlan:
    """How many release tokens a phase contributes and at what stride."""

    phase: str
    release_start: int
    release_tokens: int
    sampled_tokens: int
    blocks: int
    stride_blocks: int

    @property
    def sampled_fraction(self) -> float:
        return self.sampled_tokens / max(self.release_tokens, 1)


class AblationSampleStream:
    """Strided, block-aligned view of the release for a fixed token budget."""

    def __init__(
        self,
        stream: DeterministicReleaseStream,
        *,
        budget_tokens: int,
        block_tokens: int,
        phase_starts: dict[str, int],
        phase_tokens: dict[str, int],
    ) -> None:
        if budget_tokens <= 0 or block_tokens <= 0:
            raise ValueError("budget_tokens and block_tokens must be positive")
        total_release = sum(phase_tokens[phase] for phase in PHASE_ORDER)
        if budget_tokens > total_release:
            raise ValueError(
                f"Ablation budget {budget_tokens:,} exceeds the {total_release:,}-token release"
            )
        self.stream = stream
        self.budget_tokens = int(budget_tokens)
        self.block_tokens = int(block_tokens)
        self._phase_starts = dict(phase_starts)
        self._phase_tokens = dict(phase_tokens)

        plans: list[PhasePlan] = []
        for phase in PHASE_ORDER:
            release_tokens = int(phase_tokens[phase])
            share = release_tokens / total_release
            sampled = int(budget_tokens * share)
            blocks = sampled // self.block_tokens
            available_blocks = release_tokens // self.block_tokens
            if blocks <= 0:
                raise ValueError(
                    f"Phase {phase} contributes fewer than one {self.block_tokens:,}-token "
                    "block at this budget; raise the budget or shrink the global batch."
                )
            # Spread the sampled blocks evenly across the phase.  Integer
            # division here is deliberate: a stride that is rounded up would
            # walk off the end of the phase and a fractional stride would make
            # the mapping depend on floating-point rounding.
            stride = max(1, available_blocks // blocks)
            plans.append(
                PhasePlan(
                    phase=phase,
                    release_start=int(phase_starts[phase]),
                    release_tokens=release_tokens,
                    sampled_tokens=blocks * self.block_tokens,
                    blocks=blocks,
                    stride_blocks=stride,
                )
            )
        self.plans = tuple(plans)
        self.total_blocks = sum(plan.blocks for plan in self.plans)

    @property
    def sampled_tokens(self) -> int:
        return self.total_blocks * self.block_tokens

    def dropped_tokens(self) -> int:
        """Budget lost to block alignment. Logged, never silently absorbed."""

        return self.budget_tokens - self.sampled_tokens

    def release_cursor(self, step: int) -> int:
        """Map an optimizer-step index to its release token cursor."""

        if not 0 <= step < self.total_blocks:
            raise IndexError(f"step {step} is outside [0, {self.total_blocks})")
        remaining = step
        for plan in self.plans:
            if remaining < plan.blocks:
                offset = remaining * plan.stride_blocks * self.block_tokens
                cursor = plan.release_start + offset
                # A stride computed by floor division can only under-run the
                # phase, never over-run it, but assert rather than assume: a
                # block that crossed a phase boundary would read the wrong
                # curriculum and the loss curve would silently lie.
                if offset + self.block_tokens > plan.release_tokens:
                    raise RuntimeError(
                        f"Block {step} would cross the {plan.phase} boundary"
                    )
                return cursor
            remaining -= plan.blocks
        raise RuntimeError("Unreachable: step index exceeded the phase plan")

    def phase_for_step(self, step: int) -> str:
        remaining = step
        for plan in self.plans:
            if remaining < plan.blocks:
                return plan.phase
            remaining -= plan.blocks
        raise IndexError(f"step {step} is outside [0, {self.total_blocks})")

    def evaluation_cursor(self, step: int, *, gap_blocks: int, window_tokens: int) -> int:
        """Select a token-disjoint window in a declared training-stride gap."""

        if type(gap_blocks) is not int or gap_blocks < 1:
            raise ValueError("Evaluation gap_blocks must be a positive integer.")
        if type(window_tokens) is not int or window_tokens < 1:
            raise ValueError("Evaluation window_tokens must be a positive integer.")
        base = self.release_cursor(step)
        phase = self.phase_for_step(step)
        plan = next(candidate for candidate in self.plans if candidate.phase == phase)
        if gap_blocks >= plan.stride_blocks:
            raise ValueError("Evaluation gap reaches the next sampled training block.")
        # Training consumes a next-token target beyond its input block. Keep
        # that token out of evaluation, including when selecting the first gap.
        cursor = base + gap_blocks * self.block_tokens + 1
        boundary = min(
            base + plan.stride_blocks * self.block_tokens,
            plan.release_start + plan.release_tokens,
        )
        if cursor + window_tokens >= boundary:
            raise ValueError("Evaluation inputs or target lookahead cross a training/phase boundary.")
        return cursor

    def micro_batches(
        self,
        *,
        step: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        grad_accum: int,
    ) -> Iterator[TrainingBatch]:
        """Yield this rank's micro-batches for one optimizer step.

        Every rank walks the same block; ``DeterministicReleaseStream.batch``
        partitions it by rank, and successive accumulation micro-steps advance
        through the block.  The union over ranks and micro-steps is exactly the
        block, which is what makes the token set identical across rows with
        different world sizes.
        """

        span = world_size * micro_batch_size * self.stream.sequence_length
        if span * grad_accum != self.block_tokens:
            raise ValueError(
                f"world_size*micro_batch*sequence*grad_accum = {span * grad_accum:,} "
                f"does not tile the {self.block_tokens:,}-token block; every row "
                "must consume an identical global batch."
            )
        for read in self.micro_batch_reads(
            step=step,
            rank=rank,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            grad_accum=grad_accum,
        ):
            yield read()

    def micro_batch_reads(
        self,
        *,
        step: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        grad_accum: int,
    ) -> Iterator[Callable[[], TrainingBatch]]:
        """The same micro-batches, as reads not yet performed.

        Handing back the reads instead of their results lets a loader issue
        several at once. The release is not short of bandwidth -- a rank draws
        tens of kilobytes a second -- but each read is a separate round trip to
        Lustre, and one thread issuing them in series cannot stay ahead of the
        accelerators. The cursor arithmetic stays here, where the invariant that
        the union over ranks and micro-steps is exactly the block still lives.
        """

        span = world_size * micro_batch_size * self.stream.sequence_length
        if span * grad_accum != self.block_tokens:
            raise ValueError(
                f"world_size*micro_batch*sequence*grad_accum = {span * grad_accum:,} "
                f"does not tile the {self.block_tokens:,}-token block; every row "
                "must consume an identical global batch."
            )
        base = self.release_cursor(step)
        for accumulation in range(grad_accum):
            cursor = base + accumulation * span
            yield functools.partial(
                self.stream.batch,
                global_token_cursor=cursor,
                rank=rank,
                world_size=world_size,
                micro_batch_size=micro_batch_size,
            )

    def describe(self) -> dict[str, object]:
        return {
            "budget_tokens": self.budget_tokens,
            "sampled_tokens": self.sampled_tokens,
            "dropped_to_block_alignment": self.dropped_tokens(),
            "block_tokens": self.block_tokens,
            "total_blocks": self.total_blocks,
            "phases": [
                {
                    "phase": plan.phase,
                    "release_tokens": plan.release_tokens,
                    "sampled_tokens": plan.sampled_tokens,
                    "blocks": plan.blocks,
                    "stride_blocks": plan.stride_blocks,
                    "sampled_fraction": round(plan.sampled_fraction, 6),
                }
                for plan in self.plans
            ],
        }


def build_sample_stream(
    stream: DeterministicReleaseStream,
    *,
    budget_tokens: int,
    block_tokens: int,
) -> AblationSampleStream:
    from metis_training.data import PHASE_STARTS, PHASE_TOKENS

    return AblationSampleStream(
        stream,
        budget_tokens=budget_tokens,
        block_tokens=block_tokens,
        phase_starts=dict(PHASE_STARTS),
        phase_tokens=dict(PHASE_TOKENS),
    )


__all__ = [
    "AblationSampleStream",
    "PhasePlan",
    "build_sample_stream",
]
