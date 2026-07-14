#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from types import SimpleNamespace
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_metis15_tpu_logs import audit, parse_log, percentile


RUN_ID_RE = re.compile(r"bs(?P<batch>\d+)_ga(?P<accum>\d+)_(?P<attention>.+)_bucket(?P<bucket>\d+)$")


def _parse_run_id(path: Path) -> dict[str, str]:
    match = RUN_ID_RE.match(path.parent.name)
    if not match:
        return {"run_id": path.parent.name}
    values = match.groupdict()
    values["run_id"] = path.parent.name
    return values


def _audit_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        min_logged_steps=args.min_logged_steps,
        require_profile=True,
        require_qk_clip=True,
        require_expert_hist=args.require_expert_hist,
        require_loss_decrease=args.require_loss_decrease,
        min_loss_drop_frac=args.min_loss_drop_frac,
        max_final_loss=args.max_final_loss,
        expected_valid_assign=None,
        min_valid_assign_frac=args.min_valid_assign_frac,
        max_expert_drop_frac=args.max_expert_drop_frac,
        max_qk_logit=args.max_qk_logit,
        min_toks_per_s=None,
        perf_warmup_steps=args.perf_warmup_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Metis TPU perf sweep logs by tok/s after safety gates.")
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--min-logged-steps", type=int, default=2)
    parser.add_argument("--perf-warmup-steps", type=int, default=3)
    parser.add_argument("--min-valid-assign-frac", type=float, default=0.99)
    parser.add_argument("--max-expert-drop-frac", type=float, default=0.01)
    parser.add_argument("--max-qk-logit", type=float, default=1000.0)
    parser.add_argument("--require-loss-decrease", action="store_true")
    parser.add_argument("--min-loss-drop-frac", type=float, default=0.0)
    parser.add_argument("--max-final-loss", type=float, default=None)
    parser.add_argument("--require-expert-hist", action="store_true")
    parser.add_argument("--write-best-env", type=Path, default=None)
    args = parser.parse_args()

    logs = sorted(path for path in args.log_root.glob("*/train.log") if path.is_file())
    if not logs:
        raise SystemExit(f"No sweep logs found under {args.log_root}")

    rows: list[dict[str, object]] = []
    audit_args = _audit_args(args)
    for log_path in logs:
        summary = parse_log(log_path)
        failures, warnings = audit(summary, audit_args)
        post_warmup = [row for row in summary.steps if row.step > args.perf_warmup_steps]
        measured = post_warmup or summary.steps
        toks = [row.toks_per_s for row in measured]
        step_s = [row.step_s for row in measured]
        qk_max = max((row.max_logit for row in summary.qk_rows), default=float("nan"))
        run = _parse_run_id(log_path)
        row = {
            **run,
            "path": log_path,
            "ok": not failures,
            "median_tok_s": percentile(toks, 50.0) if toks else 0.0,
            "p95_step_s": percentile(step_s, 95.0) if step_s else float("inf"),
            "first_loss": summary.steps[0].loss if summary.steps else float("nan"),
            "final_loss": summary.steps[-1].loss if summary.steps else float("nan"),
            "qk_max": qk_max,
            "failures": failures,
            "warnings": warnings,
        }
        rows.append(row)

    rows.sort(key=lambda row: (not bool(row["ok"]), -float(row["median_tok_s"]), float(row["p95_step_s"])))
    print("run_id,status,median_tok_s,p95_step_s,first_loss,final_loss,qk_max")
    for row in rows:
        status = "ok" if row["ok"] else "fail"
        print(
            f"{row['run_id']},{status},{float(row['median_tok_s']):.0f},"
            f"{float(row['p95_step_s']):.4f},{float(row['first_loss']):.4f},"
            f"{float(row['final_loss']):.4f},{float(row['qk_max']):.3f}"
        )
        for warning in row["warnings"]:
            print(f"  WARN {row['run_id']}: {warning}")
        for failure in row["failures"]:
            print(f"  FAIL {row['run_id']}: {failure}")

    winners = [row for row in rows if row["ok"]]
    if not winners:
        raise SystemExit("No sweep candidate passed the training-quality gates.")

    best = winners[0]
    print()
    print(
        "best_sweep_candidate "
        f"run_id={best['run_id']} median_tok_s={float(best['median_tok_s']):.0f} "
        f"p95_step_s={float(best['p95_step_s']):.4f}"
    )
    if args.write_best_env:
        parsed = _parse_run_id(Path(best["path"]))
        lines = [
            "# Source this only after the matching sweep log passed quality gates.",
            f"export METIS15_TPU_LOCAL_BATCH_SIZE={parsed.get('batch', '')}",
            f"export METIS15_TPU_GRAD_ACCUM_STEPS={parsed.get('accum', '')}",
            f"export METIS15_TPU_ATTENTION_KERNEL={parsed.get('attention', '')}",
            f"export METIS15_TPU_GRAD_SYNC_BUCKET_MB={parsed.get('bucket', '')}",
        ]
        args.write_best_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"best_env={args.write_best_env}")


if __name__ == "__main__":
    main()
