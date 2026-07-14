#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Iterable


STEP_RE = re.compile(
    r"^step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+"
    r"lm\s+(?P<lm>[-+0-9.eE]+)\s+\|\s+moe_aux\s+(?P<moe_aux>[-+0-9.eE]+)\s+\|\s+"
    r"valid_assign\s+(?P<valid>[0-9,]+)\s+\|\s+tok/s\s+(?P<toks>[0-9,]+)\s+\|\s+"
    r"step_s\s+(?P<step_s>[-+0-9.eE]+)\s+\|\s+lr\s+(?P<lr>[-+0-9.eE]+)\s+\|\s+"
    r"tok_seen\s+(?P<tok_seen>[-+0-9.eE]+)B"
)
PROFILE_RE = re.compile(r"^profile_components\s+step=(?P<step>\d+)\s+(?P<body>.*)$")
QK_RE = re.compile(
    r"^qk_clip\s+step=(?P<step>\d+)\s+max_logit=(?P<max_logit>[-+0-9.eE]+)\s+"
    r"min_scale=(?P<min_scale>[-+0-9.eE]+)\s+scaled_heads=(?P<scaled_heads>[-+0-9.eE]+)"
)
EXPERT_RE = re.compile(
    r"^expert_hist\s+step=(?P<step>\d+)\s+layer=(?P<layer>\d+).*?"
    r"valid=(?P<valid>[0-9,]+)/(?P<assignments>[0-9,]+)\s+"
    r"dropped=(?P<dropped>[0-9,]+)\s+drop_frac=(?P<drop_frac>[-+0-9.eE]+)"
)
LAUNCH_RE = {
    "world_size": re.compile(r"^\s*world_size:\s+(?P<value>\d+)"),
    "layers_block": re.compile(r"^\s*config:\s+layers=(?P<layers>\d+)\s+d_model=\d+\s+block=(?P<block>\d+)"),
    "top_k": re.compile(r"^\s*moe:\s+experts=\d+\s+local=\d+\s+top_k=(?P<top_k>\d+)"),
    "local_batch": re.compile(r"^\s*local batch size:\s+(?P<value>\d+)"),
    "expected_valid": re.compile(r"^\s*expected valid assignments/logged step:\s+(?P<value>\d+)"),
}


@dataclass
class StepRow:
    step: int
    loss: float
    lm: float
    moe_aux: float
    valid_assign: int
    toks_per_s: int
    step_s: float
    lr: float
    tok_seen_b: float


@dataclass
class QkRow:
    step: int
    max_logit: float
    min_scale: float
    scaled_heads: float


@dataclass
class ExpertRow:
    step: int
    layer: int
    valid: int
    assignments: int
    dropped: int
    drop_frac: float


@dataclass
class LogSummary:
    path: Path
    steps: list[StepRow] = field(default_factory=list)
    qk_rows: list[QkRow] = field(default_factory=list)
    expert_rows: list[ExpertRow] = field(default_factory=list)
    profile_steps: set[int] = field(default_factory=set)
    launch: dict[str, int] = field(default_factory=dict)

    @property
    def expected_valid_assign(self) -> int | None:
        needed = {"world_size", "layers", "block", "top_k", "local_batch"}
        if not needed.issubset(self.launch):
            return self.launch.get("expected_valid")
        return (
            self.launch["world_size"]
            * self.launch["layers"]
            * self.launch["block"]
            * self.launch["top_k"]
            * self.launch["local_batch"]
        )


def _parse_kv_body(body: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for chunk in body.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def parse_log(path: Path) -> LogSummary:
    summary = LogSummary(path=path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := LAUNCH_RE["world_size"].match(line):
            summary.launch["world_size"] = int(match.group("value"))
        elif match := LAUNCH_RE["layers_block"].match(line):
            summary.launch["layers"] = int(match.group("layers"))
            summary.launch["block"] = int(match.group("block"))
        elif match := LAUNCH_RE["top_k"].match(line):
            summary.launch["top_k"] = int(match.group("top_k"))
        elif match := LAUNCH_RE["local_batch"].match(line):
            summary.launch["local_batch"] = int(match.group("value"))
        elif match := LAUNCH_RE["expected_valid"].match(line):
            summary.launch["expected_valid"] = int(match.group("value"))
        elif match := STEP_RE.match(line):
            summary.steps.append(
                StepRow(
                    step=int(match.group("step")),
                    loss=float(match.group("loss")),
                    lm=float(match.group("lm")),
                    moe_aux=float(match.group("moe_aux")),
                    valid_assign=int(match.group("valid").replace(",", "")),
                    toks_per_s=int(match.group("toks").replace(",", "")),
                    step_s=float(match.group("step_s")),
                    lr=float(match.group("lr")),
                    tok_seen_b=float(match.group("tok_seen")),
                )
            )
        elif match := PROFILE_RE.match(line):
            profile_values = _parse_kv_body(match.group("body"))
            if "p50_step_s" in profile_values and "p95_step_s" in profile_values:
                summary.profile_steps.add(int(match.group("step")))
        elif match := QK_RE.match(line):
            summary.qk_rows.append(
                QkRow(
                    step=int(match.group("step")),
                    max_logit=float(match.group("max_logit")),
                    min_scale=float(match.group("min_scale")),
                    scaled_heads=float(match.group("scaled_heads")),
                )
            )
        elif match := EXPERT_RE.match(line):
            summary.expert_rows.append(
                ExpertRow(
                    step=int(match.group("step")),
                    layer=int(match.group("layer")),
                    valid=int(match.group("valid").replace(",", "")),
                    assignments=int(match.group("assignments").replace(",", "")),
                    dropped=int(match.group("dropped").replace(",", "")),
                    drop_frac=float(match.group("drop_frac")),
                )
            )
    return summary


def percentile(values: Iterable[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_values[lower]
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def audit(summary: LogSummary, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if len(summary.steps) < args.min_logged_steps:
        failures.append(f"only {len(summary.steps)} logged step(s), expected at least {args.min_logged_steps}")
    if args.require_profile and summary.steps:
        missing = [row.step for row in summary.steps if row.step not in summary.profile_steps]
        if missing:
            failures.append(f"profile_components missing for logged steps: {missing[:8]}")
    if args.require_qk_clip and summary.steps:
        qk_steps = {row.step for row in summary.qk_rows}
        missing = [row.step for row in summary.steps if row.step not in qk_steps]
        if missing:
            failures.append(f"qk_clip metrics missing for logged steps: {missing[:8]}")
    if summary.steps:
        first = summary.steps[0]
        last = summary.steps[-1]
        if not all(finite(value) for row in summary.steps for value in (row.loss, row.lm, row.moe_aux, row.step_s)):
            failures.append("nonfinite loss/lm/moe_aux/step_s detected")
        if args.max_final_loss is not None and last.loss > args.max_final_loss:
            failures.append(f"final loss {last.loss:.4f} exceeds --max-final-loss {args.max_final_loss:.4f}")
        if args.require_loss_decrease and last.loss > first.loss * (1.0 - args.min_loss_drop_frac):
            failures.append(
                f"loss did not drop by {args.min_loss_drop_frac:.2%}: first={first.loss:.4f} last={last.loss:.4f}"
            )
        expected_valid = args.expected_valid_assign or summary.expected_valid_assign
        if expected_valid:
            worst_valid_frac = min(row.valid_assign / float(expected_valid) for row in summary.steps)
            if worst_valid_frac < args.min_valid_assign_frac:
                failures.append(
                    f"valid_assign fraction {worst_valid_frac:.4f} below minimum {args.min_valid_assign_frac:.4f} "
                    f"(expected {expected_valid})"
                )
        elif args.min_valid_assign_frac > 0:
            warnings.append("could not infer expected valid_assign; pass --expected-valid-assign for a strict gate")
        if args.min_toks_per_s is not None:
            post_warmup = [row for row in summary.steps if row.step > args.perf_warmup_steps]
            measured = post_warmup or summary.steps
            median_toks = percentile([row.toks_per_s for row in measured], 50.0)
            if median_toks < args.min_toks_per_s:
                failures.append(f"median tok/s {median_toks:.0f} below minimum {args.min_toks_per_s:.0f}")
    for row in summary.qk_rows:
        if not finite(row.max_logit) or not finite(row.min_scale) or not finite(row.scaled_heads):
            failures.append(f"nonfinite qk_clip metric at step {row.step}")
            continue
        if row.max_logit > args.max_qk_logit:
            failures.append(f"qk max_logit {row.max_logit:.3f} at step {row.step} exceeds {args.max_qk_logit:.3f}")
        if not 0.0 < row.min_scale <= 1.0:
            failures.append(f"qk min_scale {row.min_scale:.6f} at step {row.step} is outside (0, 1]")
    if summary.expert_rows:
        worst_drop = max(row.drop_frac for row in summary.expert_rows)
        if worst_drop > args.max_expert_drop_frac:
            failures.append(f"expert drop_frac {worst_drop:.4f} exceeds {args.max_expert_drop_frac:.4f}")
    elif args.require_expert_hist:
        failures.append("expert_hist metrics missing")
    return failures, warnings


def print_summary(summary: LogSummary) -> None:
    print(f"log={summary.path}")
    print(f"logged_steps={len(summary.steps)} profile_steps={len(summary.profile_steps)} qk_steps={len(summary.qk_rows)}")
    if summary.steps:
        losses = [row.loss for row in summary.steps]
        toks = [row.toks_per_s for row in summary.steps]
        step_s = [row.step_s for row in summary.steps]
        print(
            "training_summary "
            f"first_step={summary.steps[0].step} last_step={summary.steps[-1].step} "
            f"first_loss={losses[0]:.4f} final_loss={losses[-1]:.4f} "
            f"median_tok_s={percentile(toks, 50.0):.0f} p95_step_s={percentile(step_s, 95.0):.4f}"
        )
        expected = summary.expected_valid_assign
        if expected:
            valid_fracs = [row.valid_assign / float(expected) for row in summary.steps]
            print(f"valid_assign expected={expected} min_frac={min(valid_fracs):.4f}")
    if summary.qk_rows:
        print(
            "qk_summary "
            f"max_logit={max(row.max_logit for row in summary.qk_rows):.3f} "
            f"min_scale={min(row.min_scale for row in summary.qk_rows):.6f} "
            f"max_scaled_heads={max(row.scaled_heads for row in summary.qk_rows):.0f}"
        )
    if summary.expert_rows:
        print(
            "expert_summary "
            f"max_drop_frac={max(row.drop_frac for row in summary.expert_rows):.4f} "
            f"max_dropped={max(row.dropped for row in summary.expert_rows)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Metis-1.5 TPU logs for throughput-safe training health.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-logged-steps", type=int, default=1)
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument("--require-qk-clip", action="store_true")
    parser.add_argument("--require-expert-hist", action="store_true")
    parser.add_argument("--require-loss-decrease", action="store_true")
    parser.add_argument("--min-loss-drop-frac", type=float, default=0.0)
    parser.add_argument("--max-final-loss", type=float, default=None)
    parser.add_argument("--expected-valid-assign", type=int, default=None)
    parser.add_argument("--min-valid-assign-frac", type=float, default=0.99)
    parser.add_argument("--max-expert-drop-frac", type=float, default=0.01)
    parser.add_argument("--max-qk-logit", type=float, default=1000.0)
    parser.add_argument("--min-toks-per-s", type=float, default=None)
    parser.add_argument("--perf-warmup-steps", type=int, default=0)
    args = parser.parse_args()
    summary = parse_log(args.log)
    print_summary(summary)
    failures, warnings = audit(summary, args)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("metis15_tpu_log_audit_ok")


if __name__ == "__main__":
    main()
