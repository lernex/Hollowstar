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
    r"mor_aux\s+(?P<mor_aux>[-+0-9.eE]+)\s+\|\s+"
    r"(?:mean_depth\s+(?P<mean_depth>[-+0-9.eE]+)\s+\|\s+mor_target\s+(?P<mor_target>[-+0-9.eE]+)\s+\|\s+"
    r"mor_coef\s+(?P<mor_coef>[-+0-9.eE]+)\s+\|\s+)?"
    r"valid_assign\s+(?P<valid>[0-9,]+)\s+\|\s+"
    r"(?:total_assign\s+(?P<total>[0-9,]+)\s+\|\s+)?"
    r"drop\s+(?P<drop>[-+0-9.eE]+)\s+\|\s+accum\s+(?P<accum>\d+)\s+\|\s+"
    r"tok/s\s+(?P<toks>[0-9,]+)\s+\|\s+step_s\s+(?P<step_s>[-+0-9.eE]+)\s+\|\s+"
    r"qk_max\s+(?P<qk_max>[-+0-9.eE]+)\s+qk_scale\s+(?P<qk_scale>[-+0-9.eE]+)"
    r"(?:\s+qk_scaled_layers\s+(?P<qk_scaled_layers>\d+))?"
)
LAUNCH_RE = {
    "devices": re.compile(
        r"^\s+stage=(?P<stage>\S+)\s+devices=(?P<devices>\d+)\s+local_batch=(?P<local_batch>\d+)\s+block=(?P<block>\d+)"
    ),
    "model": re.compile(r"^\s+model layers=(?P<layers>\d+)\s+d_model=\d+\s+experts=(?P<experts>\d+)\s+top_k=(?P<top_k>\d+)"),
}


@dataclass
class StepRow:
    step: int
    loss: float
    lm: float
    moe_aux: float
    mor_aux: float
    mean_depth: float
    mor_target: float
    mor_coef: float
    valid_assign: int
    total_assign: int | None
    drop_frac: float
    accum: int
    toks_per_s: int
    step_s: float
    qk_max: float
    qk_scale: float
    qk_scaled_layers: int


@dataclass
class LogSummary:
    path: Path
    steps: list[StepRow] = field(default_factory=list)
    launch: dict[str, int | str] = field(default_factory=dict)

    @property
    def expected_valid_assign(self) -> int | None:
        needed = {"layers", "block", "top_k", "local_batch", "accum"}
        if not needed.issubset(self.launch):
            return None
        return (
            self.launch["layers"]
            * self.launch["block"]
            * self.launch["top_k"]
            * self.launch["local_batch"]
            * self.launch["accum"]
        )


def parse_log(path: Path) -> LogSummary:
    summary = LogSummary(path=path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := LAUNCH_RE["devices"].match(line):
            summary.launch["stage"] = match.group("stage")
            summary.launch["devices"] = int(match.group("devices"))
            summary.launch["local_batch"] = int(match.group("local_batch"))
            summary.launch["block"] = int(match.group("block"))
        elif match := LAUNCH_RE["model"].match(line):
            summary.launch["layers"] = int(match.group("layers"))
            summary.launch["experts"] = int(match.group("experts"))
            summary.launch["top_k"] = int(match.group("top_k"))
        elif match := STEP_RE.match(line):
            row = StepRow(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                lm=float(match.group("lm")),
                moe_aux=float(match.group("moe_aux")),
                mor_aux=float(match.group("mor_aux")),
                mean_depth=float(match.group("mean_depth") or 1.0),
                mor_target=float(match.group("mor_target") or 1.0),
                mor_coef=float(match.group("mor_coef") or 0.0),
                valid_assign=int(match.group("valid").replace(",", "")),
                total_assign=(
                    int(match.group("total").replace(",", ""))
                    if match.group("total") is not None
                    else None
                ),
                drop_frac=float(match.group("drop")),
                accum=int(match.group("accum")),
                toks_per_s=int(match.group("toks").replace(",", "")),
                step_s=float(match.group("step_s")),
                qk_max=float(match.group("qk_max")),
                qk_scale=float(match.group("qk_scale")),
                qk_scaled_layers=int(match.group("qk_scaled_layers") or 0),
            )
            summary.steps.append(row)
            summary.launch.setdefault("accum", row.accum)
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


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def audit(summary: LogSummary, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if len(summary.steps) < args.min_logged_steps:
        failures.append(f"only {len(summary.steps)} logged step(s), expected at least {args.min_logged_steps}")
    if args.require_tpu and summary.launch.get("devices") != 8:
        failures.append(f"expected 8 visible TPU devices; saw devices={summary.launch.get('devices')}")
    if not summary.steps:
        return failures, warnings
    first = summary.steps[0]
    last = summary.steps[-1]
    numeric_fields = (
        "loss",
        "lm",
        "moe_aux",
        "mor_aux",
        "mean_depth",
        "mor_target",
        "mor_coef",
        "drop_frac",
        "step_s",
        "qk_max",
        "qk_scale",
    )
    for row in summary.steps:
        if not all(_finite(getattr(row, field_name)) for field_name in numeric_fields):
            failures.append(f"nonfinite metric at step {row.step}")
        if row.accum != first.accum:
            failures.append(f"grad accumulation changed from {first.accum} to {row.accum} at step {row.step}")
        if not 0.0 < row.qk_scale <= 1.0:
            failures.append(f"qk_scale {row.qk_scale:.6f} at step {row.step} is outside (0, 1]")
        if row.qk_max > args.max_qk_logit:
            failures.append(f"qk_max {row.qk_max:.3f} at step {row.step} exceeds {args.max_qk_logit:.3f}")
        if row.drop_frac > args.max_expert_drop_frac:
            failures.append(f"expert drop {row.drop_frac:.4f} at step {row.step} exceeds {args.max_expert_drop_frac:.4f}")
    if args.require_loss_decrease and last.loss > first.loss * (1.0 - args.min_loss_drop_frac):
        failures.append(f"loss did not drop by {args.min_loss_drop_frac:.2%}: first={first.loss:.6f} last={last.loss:.6f}")
    if args.max_final_loss is not None and last.loss > args.max_final_loss:
        failures.append(f"final loss {last.loss:.6f} exceeds {args.max_final_loss:.6f}")
    if args.require_mor_active and max(row.mor_aux for row in summary.steps) <= args.mor_epsilon:
        failures.append(f"MoR aux stayed inactive; max mor_aux <= {args.mor_epsilon:.3e}")
    if args.require_mor_disabled and max(abs(row.mor_aux) for row in summary.steps) > args.mor_epsilon:
        failures.append(f"MoR aux was nonzero while disabled; max abs mor_aux > {args.mor_epsilon:.3e}")
    if args.require_mor_disabled and max(abs(row.mor_coef) for row in summary.steps) > args.mor_epsilon:
        failures.append(f"MoR coef was nonzero while disabled; max abs mor_coef > {args.mor_epsilon:.3e}")
    if args.require_mor_target_increase and last.mor_target <= first.mor_target + args.mor_epsilon:
        failures.append(
            f"MoR target did not increase: first={first.mor_target:.6f} last={last.mor_target:.6f}"
        )
    if args.require_mor_coef_increase and last.mor_coef <= first.mor_coef + args.mor_epsilon:
        failures.append(f"MoR coef did not increase: first={first.mor_coef:.6f} last={last.mor_coef:.6f}")
    if all(row.total_assign for row in summary.steps):
        valid_fracs = [row.valid_assign / float(row.total_assign) for row in summary.steps if row.total_assign]
        worst_valid_frac = min(valid_fracs)
        if worst_valid_frac < args.min_valid_assign_frac:
            failures.append(
                f"valid_assign fraction {worst_valid_frac:.4f} below minimum {args.min_valid_assign_frac:.4f} "
                f"(from logged total_assign)"
            )
    else:
        expected_valid = args.expected_valid_assign or summary.expected_valid_assign
        if expected_valid:
            worst_valid_frac = min(row.valid_assign / float(expected_valid) for row in summary.steps)
            if worst_valid_frac < args.min_valid_assign_frac:
                failures.append(
                    f"valid_assign fraction {worst_valid_frac:.4f} below minimum {args.min_valid_assign_frac:.4f} "
                    f"(expected {expected_valid})"
                )
        else:
            warnings.append("could not infer expected valid_assign; pass --expected-valid-assign for a strict gate")
    measured = [row for row in summary.steps if row.step > args.perf_warmup_steps] or summary.steps
    median_toks = percentile([row.toks_per_s for row in measured], 50.0)
    if args.min_toks_per_s is not None and median_toks < args.min_toks_per_s:
        failures.append(f"median tok/s {median_toks:.0f} below minimum {args.min_toks_per_s:.0f}")
    return failures, warnings


def print_summary(summary: LogSummary, *, perf_warmup_steps: int) -> None:
    print(f"log={summary.path}")
    print(
        "launch "
        f"devices={summary.launch.get('devices', 'unknown')} local_batch={summary.launch.get('local_batch', 'unknown')} "
        f"block={summary.launch.get('block', 'unknown')} layers={summary.launch.get('layers', 'unknown')} "
        f"top_k={summary.launch.get('top_k', 'unknown')} accum={summary.launch.get('accum', 'unknown')}"
    )
    print(f"logged_steps={len(summary.steps)}")
    if summary.steps:
        measured = [row for row in summary.steps if row.step > perf_warmup_steps] or summary.steps
        print(
            "training_summary "
            f"first_step={summary.steps[0].step} last_step={summary.steps[-1].step} "
            f"first_loss={summary.steps[0].loss:.6f} final_loss={summary.steps[-1].loss:.6f} "
            f"median_tok_s={percentile([row.toks_per_s for row in measured], 50.0):.0f} "
            f"p95_step_s={percentile([row.step_s for row in measured], 95.0):.4f}"
        )
        expected = summary.expected_valid_assign
        if all(row.total_assign for row in summary.steps):
            valid_fracs = [row.valid_assign / float(row.total_assign) for row in summary.steps if row.total_assign]
            total_values = sorted({row.total_assign for row in summary.steps if row.total_assign})
            print(f"valid_assign total_assign={total_values} min_frac={min(valid_fracs):.4f}")
        elif expected:
            valid_fracs = [row.valid_assign / float(expected) for row in summary.steps]
            print(f"valid_assign expected={expected} min_frac={min(valid_fracs):.4f}")
        print(
            "qk_summary "
            f"max_logit={max(row.qk_max for row in summary.steps):.3f} "
            f"min_scale={min(row.qk_scale for row in summary.steps):.6f} "
            f"max_scaled_layers={max(row.qk_scaled_layers for row in summary.steps)}"
        )
        print(
            "mor_summary "
            f"max_aux={max(row.mor_aux for row in summary.steps):.6f} "
            f"first_target={summary.steps[0].mor_target:.3f} final_target={summary.steps[-1].mor_target:.3f} "
            f"first_coef={summary.steps[0].mor_coef:.6f} final_coef={summary.steps[-1].mor_coef:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Metis-1.5 JAX TPU logs for training health.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-logged-steps", type=int, default=1)
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--require-loss-decrease", action="store_true")
    parser.add_argument("--min-loss-drop-frac", type=float, default=0.0)
    parser.add_argument("--max-final-loss", type=float, default=None)
    parser.add_argument("--require-mor-active", action="store_true")
    parser.add_argument("--require-mor-disabled", action="store_true")
    parser.add_argument("--require-mor-target-increase", action="store_true")
    parser.add_argument("--require-mor-coef-increase", action="store_true")
    parser.add_argument("--mor-epsilon", type=float, default=1e-9)
    parser.add_argument("--expected-valid-assign", type=int, default=None)
    parser.add_argument("--min-valid-assign-frac", type=float, default=0.99)
    parser.add_argument("--max-expert-drop-frac", type=float, default=0.01)
    parser.add_argument("--max-qk-logit", type=float, default=1000.0)
    parser.add_argument("--min-toks-per-s", type=float, default=None)
    parser.add_argument("--perf-warmup-steps", type=int, default=0)
    args = parser.parse_args()
    summary = parse_log(args.log)
    print_summary(summary, perf_warmup_steps=args.perf_warmup_steps)
    failures, warnings = audit(summary, args)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("metis15_jax_tpu_log_audit_ok")


if __name__ == "__main__":
    main()
