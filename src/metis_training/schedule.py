from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import PHASE_ORDER, PHASE_STARTS, PHASE_TOKENS, TOTAL_TOKENS


@dataclass(frozen=True)
class ScheduleState:
    global_token_cursor: int
    phase: str
    phase_progress: float
    learning_rate: float
    learning_rate_ratio: float
    target_mean_depth: float
    target_mean_routed_k: float
    memory_gate_scale: float
    force_depth: int | None
    max_passes: int
    routed_k_min: int
    routed_k_max: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_curriculum(self) -> dict[str, Any]:
        return {
            "continuation_mode": (
                "depth_one" if self.force_depth == 1 else "adaptive"
            ),
            "routed_k_mode": "fixed" if self.force_depth is not None else "adaptive",
            "fixed_routed_k": int(round(self.target_mean_routed_k)),
            "target_mean_depth": self.target_mean_depth,
            "target_mean_routed_k": self.target_mean_routed_k,
            "memory_gate_scale": self.memory_gate_scale,
            "ngram_gate_scale": self.memory_gate_scale,
            "max_passes": self.max_passes,
            "stochastic_routing": True,
            "temperature": 1.0,
        }


def _cosine_between(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * weight


class TokenSchedule:
    """Exact token-axis learning-rate and MoRE compute curriculum."""

    def __init__(self, runtime_manifest: Mapping[str, Any], *, base_learning_rate: float) -> None:
        if runtime_manifest.get("schema") != "metis.training-runtime/v1":
            raise RuntimeError("Unexpected Metis training runtime schema")
        if base_learning_rate <= 0.0:
            raise ValueError("base_learning_rate must be positive")
        self.base_learning_rate = float(base_learning_rate)
        schedule = runtime_manifest.get("schedule")
        if not isinstance(schedule, Mapping):
            raise RuntimeError("Training runtime has no token schedule")
        if schedule.get("token_axis") != "globally_emitted_non_padding_tokens":
            raise RuntimeError("Schedule must be driven by globally emitted non-padding tokens")
        rows = schedule.get("phases")
        if not isinstance(rows, list) or len(rows) != 3:
            raise RuntimeError("Schedule must contain exactly three pretraining phases")
        self.phases = {str(row["id"]): dict(row) for row in rows}
        expected = [
            ("phase_a", 0, 700_000_000_000),
            ("phase_b", 700_000_000_000, 950_000_000_000),
            ("phase_c", 950_000_000_000, TOTAL_TOKENS),
        ]
        observed = [
            (
                phase,
                int(self.phases.get(phase, {}).get("start_token", -1)),
                int(self.phases.get(phase, {}).get("end_token_exclusive", -1)),
            )
            for phase in PHASE_ORDER
        ]
        if observed != expected:
            raise RuntimeError(f"Training runtime phase boundaries are stale: {observed}")
        curriculum = schedule.get("curriculum")
        if not isinstance(curriculum, Mapping):
            raise RuntimeError("Training runtime has no compute curriculum")
        self.curriculum = dict(curriculum)

    @classmethod
    def from_yaml(cls, path: str | Path, *, base_learning_rate: float) -> "TokenSchedule":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Expected a mapping in {path}")
        return cls(payload, base_learning_rate=base_learning_rate)

    @staticmethod
    def phase_at(cursor: int) -> str:
        if not 0 <= cursor <= TOTAL_TOKENS:
            raise ValueError("global token cursor is outside [0, 1T]")
        if cursor < PHASE_STARTS["phase_b"]:
            return "phase_a"
        if cursor < PHASE_STARTS["phase_c"]:
            return "phase_b"
        return "phase_c"

    def _lr_ratio(self, cursor: int, phase: str) -> float:
        row = self.phases[phase]
        start = int(row["start_token"])
        end = int(row["end_token_exclusive"])
        progress = min(1.0, max(0.0, (cursor - start) / max(1, end - start)))
        if phase == "phase_a":
            warmup = int(row["warmup_tokens"])
            if cursor < warmup:
                return max(1.0e-4, cursor / max(1, warmup))
            post_warmup = (cursor - warmup) / max(1, end - warmup)
            return _cosine_between(1.0, float(row["end_lr_ratio"]), post_warmup)
        return _cosine_between(
            float(row["start_lr_ratio"]),
            float(row["end_lr_ratio"]),
            progress,
        )

    def state(self, cursor: int) -> ScheduleState:
        phase = self.phase_at(cursor)
        effective_cursor = min(cursor, TOTAL_TOKENS - 1)
        phase_start = PHASE_STARTS[phase]
        phase_tokens = PHASE_TOKENS[phase]
        progress = min(1.0, max(0.0, (effective_cursor - phase_start) / phase_tokens))
        ratio = self._lr_ratio(cursor, phase)

        warm_end = int(float(self.curriculum["warm_start_fraction"]) * TOTAL_TOKENS)
        ramp_tokens = int(float(self.curriculum["ramp_fraction"]) * TOTAL_TOKENS)
        initial_depth = int(self.curriculum["initial_depth"])
        target_depth = float(self.curriculum["target_mean_depth"])
        if cursor < warm_end:
            compute_progress = 0.0
            force_depth: int | None = initial_depth
            max_passes = initial_depth
        else:
            compute_progress = min(1.0, max(0.0, (cursor - warm_end) / max(1, ramp_tokens)))
            force_depth = None
            max_passes = int(self.curriculum["max_passes"])
        depth = initial_depth + (target_depth - initial_depth) * compute_progress
        initial_k = float(self.curriculum["initial_routed_k"])
        target_k = float(self.curriculum["target_mean_routed_k"])
        routed_k = initial_k + (target_k - initial_k) * compute_progress
        memory_gate = float(self.curriculum["initial_memory_gate"]) + (
            1.0 - float(self.curriculum["initial_memory_gate"])
        ) * compute_progress
        return ScheduleState(
            global_token_cursor=cursor,
            phase=phase,
            phase_progress=progress,
            learning_rate=self.base_learning_rate * ratio,
            learning_rate_ratio=ratio,
            target_mean_depth=depth,
            target_mean_routed_k=routed_k,
            memory_gate_scale=memory_gate,
            force_depth=force_depth,
            max_passes=max_passes,
            routed_k_min=int(self.curriculum["routed_k_min"]),
            routed_k_max=int(self.curriculum["routed_k_max"]),
        )


def set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    """Set LR while preserving per-group scale selected at construction."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    for group in optimizer.param_groups:
        if "_metis_lr_scale" not in group:
            base = float(group.get("lr", learning_rate))
            group["_metis_lr_scale"] = base / max(learning_rate, 1.0e-30)
        group["lr"] = learning_rate * float(group["_metis_lr_scale"])
