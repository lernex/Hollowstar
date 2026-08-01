"""Campaign planning: allocation arithmetic, cost model, and launcher emission.

``python -m metis_ablation.campaign plan`` prints the wave that
``docs/papers/more/ablation_campaign.md`` describes, computed from the same
audited FLOP model the trainer reports against rather than from a parallel
hand-derivation.  ``... slurm`` emits the sbatch files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from metis_training.metrics import MI300A_DENSE_PEAK_FLOPS, estimate_hardware_flops

from .specs import (
    ABLATION_LADDER,
    ALL_SPECS,
    SECOND_SEED,
    WAVES,
    AblationSpec,
    CAMPAIGN_APUS,
    GLOBAL_BATCH_TOKENS,
    dense_control_report,
    spec_by_name,
    validate_allocation,
    wave_for_row,
)


DEFAULT_BUDGET_TOKENS = 50_000_000_000
DEFAULT_MFU_BAND = (0.05, 0.10, 0.15)


def _wave_specs(wave: str) -> tuple[AblationSpec, ...]:
    if wave == "all":
        return ALL_SPECS
    try:
        return WAVES[wave]
    except KeyError:
        raise SystemExit(
            f"Unknown wave {wave!r}; choose 1 (ladder), 2 (scaling), 3 (seeds), or all."
        ) from None


def row_cost(spec: AblationSpec, *, budget_tokens: int) -> dict[str, Any]:
    config = spec.model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )
    # The plan uses the row's *target* policy rather than a measured one; the
    # trainer re-reports against observed depth and k every telemetry step, and
    # those are the numbers that belong in the paper.
    if spec.continuation_mode == "depth_one":
        depth = 1.0
    elif spec.curriculum_max_passes is not None:
        depth = float(spec.curriculum_max_passes)
    else:
        depth = float(config.target_mean_passes)
    width = (
        float(spec.fixed_routed_k)
        if spec.routed_k_mode == "fixed"
        else float(config.target_mean_routed_k)
    )
    per_token = estimate_hardware_flops(
        config, tokens=1, observed_mean_passes=depth, observed_mean_routed_k=width
    )
    audit = config.logical_parameter_audit()
    return {
        "index": spec.index,
        "row": spec.name,
        "title": spec.title,
        "isolates": spec.isolates,
        "apus": spec.apus,
        "nodes": spec.apus / 4,
        "micro_batch": spec.micro_batch,
        "grad_accum": spec.grad_accum,
        "planned_mean_depth": depth,
        "planned_mean_k": width,
        "stored_parameters": audit.stored_total,
        "active_parameters_per_pass": audit.active_per_pass_mean,
        "gflops_per_token": per_token / 1e9,
        "total_exaflops": per_token * budget_tokens / 1e18,
        "iso_flop": spec.iso_flop,
        "notes": spec.notes,
    }


def plan(
    *,
    wave: str = "1",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    mfu_band: Sequence[float] = DEFAULT_MFU_BAND,
    total_apus: int = CAMPAIGN_APUS,
) -> dict[str, Any]:
    specs = _wave_specs(wave)
    if wave == "all":
        # Waves run one after another, so the machine constraint applies per
        # wave.  Validating the union would reject a campaign that fits fine.
        for candidate in WAVES.values():
            validate_allocation(candidate)
        allocation = {
            "rows": len(specs),
            "allocated_apus": max(
                sum(spec.apus for spec in candidate) for candidate in WAVES.values()
            ),
            "spare_apus": CAMPAIGN_APUS
            - max(
                sum(spec.apus for spec in candidate) for candidate in WAVES.values()
            ),
            "global_batch_sequences": GLOBAL_BATCH_TOKENS // 4_096,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        }
    else:
        allocation = validate_allocation(specs)
    rows = [row_cost(spec, budget_tokens=budget_tokens) for spec in specs]
    peak = MI300A_DENSE_PEAK_FLOPS["fp8"]
    for row in rows:
        row["hours_at_mfu"] = {
            f"{mfu:.0%}": round(
                row["total_exaflops"] * 1e18 / (row["apus"] * peak * mfu) / 3600.0, 2
            )
            for mfu in mfu_band
        }
    campaign_exaflops = sum(row["total_exaflops"] for row in rows)
    steps = budget_tokens // GLOBAL_BATCH_TOKENS
    return {
        "wave": wave,
        "budget_tokens": budget_tokens,
        "optimizer_steps": steps,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "allocation": allocation,
        "campaign_exaflops": round(campaign_exaflops, 1),
        "wall_clock_hours": {
            f"{mfu:.0%}": round(
                max(row["hours_at_mfu"][f"{mfu:.0%}"] for row in rows), 2
            )
            for mfu in mfu_band
        },
        "aggregate_hours_if_serialized": {
            f"{mfu:.0%}": round(
                campaign_exaflops * 1e18 / (total_apus * peak * mfu) / 3600.0, 2
            )
            for mfu in mfu_band
        },
        "dense_controls": dense_control_report() if wave == "1" else {},
        "rows": rows,
    }


_SBATCH_TEMPLATE = """#!/bin/bash
# MoRE ablation row {index:02d}: {title}
# Isolates: {isolates}
# Generated by `python -m metis_ablation.campaign slurm`; edit the generator,
# not this file.
#SBATCH --job-name=more-{row}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --time={time_limit}
#SBATCH --output={output_root}/{row}/slurm-%j.out
#SBATCH --error={output_root}/{row}/slurm-%j.err

set -euo pipefail

# Every row consumes an identical global batch ({global_batch_tokens} tokens),
# so the rank count is part of the experiment. A mismatch is a hard error in
# the trainer rather than a silently different run.
export WORLD_SIZE={apus}
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT={master_port}
export OMP_NUM_THREADS=6
export PYTHONPATH="{repo_root}/src:${{PYTHONPATH:-}}"
# Deterministic RCCL ordering keeps the gradient all-reduce reproducible across
# restarts, which matters because rows are compared to each other, not to a
# tolerance.
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "{output_root}/{row}"

srun --kill-on-bad-exit=1 python -m metis_ablation.train \\
  --row {row} \\
  --output "{output_root}" \\
  --release-root "{release_root}" \\
  --budget-tokens {budget_tokens} \\
  --seed {seed} \\
  --checkpoint-every {checkpoint_every} \\
  --analysis-every {analysis_every} \\
  --telemetry-every {telemetry_every}
"""


def emit_slurm(
    destination: Path,
    *,
    wave: str = "1",
    repo_root: str,
    output_root: str,
    release_root: str,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    seed: int = 16_062_026,
    time_limit: str = "36:00:00",
    checkpoint_every: int = 5_000,
    analysis_every: int = 1_000,
    telemetry_every: int = 10,
    base_port: int = 29_500,
) -> list[Path]:
    specs = _wave_specs(wave)
    validate_allocation(specs)
    destination = destination / f"wave{wave}"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in specs:
        # Wave 3 is the paired-seed repeat: same data order, different
        # initialization. Its seed must differ or it is not a repeat at all.
        row_seed = SECOND_SEED if wave_for_row(spec.name) == "3" else seed
        body = _SBATCH_TEMPLATE.format(
            index=spec.index,
            row=spec.name,
            title=spec.title,
            isolates=spec.isolates,
            nodes=spec.apus // 4,
            apus=spec.apus,
            time_limit=time_limit,
            output_root=output_root,
            release_root=release_root,
            budget_tokens=budget_tokens,
            seed=row_seed,
            checkpoint_every=checkpoint_every,
            analysis_every=analysis_every,
            telemetry_every=telemetry_every,
            repo_root=repo_root,
            master_port=base_port + spec.index,
            global_batch_tokens=GLOBAL_BATCH_TOKENS,
        )
        path = destination / f"{spec.index:02d}-{spec.name}.sbatch"
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        written.append(path)

    launcher = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"# Launch wave {wave} ({len(written)} rows, "
        f"{sum(spec.apus for spec in specs)} APUs).",
        "",
    ]
    launcher += [f'sbatch "$(dirname "$0")/{path.name}"' for path in written]
    wave_path = destination / "launch-wave.sh"
    wave_path.write_text("\n".join(launcher) + "\n", encoding="utf-8")
    wave_path.chmod(0o755)
    written.append(wave_path)
    return written


# One learning rate across thirteen architectures of different active widths is
# not fair; tuning all thirteen is not worth the compute.  The compromise is a
# short sweep over the four *archetypes* -- dense, single-pass sparse, fixed
# loop, adaptive MoRE -- with every row inheriting its archetype's winner.  The
# compromise is reported in the paper rather than hidden, because a reviewer who
# sees it stated is satisfied and one who suspects it is not.
SWEEP_ARCHETYPES = ("dense-param-matched", "moe-k4", "loop-fixed", "more-core")
SWEEP_LEARNING_RATES = (1.2e-4, 1.8e-4, 2.6e-4)
SWEEP_BUDGET_TOKENS = 1_000_000_000


def emit_sweep(
    destination: Path,
    *,
    repo_root: str,
    output_root: str,
    release_root: str,
    budget_tokens: int = SWEEP_BUDGET_TOKENS,
    seed: int = 16_062_026,
    time_limit: str = "04:00:00",
    base_port: int = 29_700,
) -> list[Path]:
    """Emit the archetype learning-rate sweep as its own small wave."""

    destination = destination / "sweep"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    port = base_port
    for archetype in SWEEP_ARCHETYPES:
        spec = spec_by_name(archetype)
        for rate in SWEEP_LEARNING_RATES:
            tag = f"{archetype}-lr{rate:.0e}".replace("-0", "-").replace("+0", "")
            body = _SBATCH_TEMPLATE.format(
                index=spec.index,
                row=spec.name,
                title=f"{spec.title} @ lr={rate:g}",
                isolates=f"learning-rate sweep at {budget_tokens:,} tokens",
                nodes=spec.apus // 4,
                apus=spec.apus,
                time_limit=time_limit,
                output_root=f"{output_root}/sweep/{tag}",
                release_root=release_root,
                budget_tokens=budget_tokens,
                seed=seed,
                checkpoint_every=0,
                analysis_every=0,
                telemetry_every=10,
                repo_root=repo_root,
                master_port=port,
                global_batch_tokens=GLOBAL_BATCH_TOKENS,
            )
            body = body.replace(
                "  --telemetry-every 10",
                f"  --telemetry-every 10 \\\n  --learning-rate {rate:g} \\\n  --no-resume",
            )
            body = body.replace(f"--job-name=more-{spec.name}", f"--job-name=sweep-{tag}")
            path = destination / f"{tag}.sbatch"
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)
            written.append(path)
            port += 1

    launcher = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"# Archetype learning-rate sweep: {len(written)} short runs at "
        f"{budget_tokens:,} tokens.",
        "# Pick each archetype's winner by final loss, then launch wave 1 with",
        "# --learning-rate set per row from its archetype.",
        "",
    ]
    launcher += [f'sbatch "$(dirname "$0")/{path.name}"' for path in written]
    wave_path = destination / "launch-sweep.sh"
    wave_path.write_text("\n".join(launcher) + "\n", encoding="utf-8")
    wave_path.chmod(0o755)
    written.append(wave_path)
    return written


def _format_plan(payload: dict[str, Any]) -> str:
    lines = [
        f"MoRE ablation wave {payload['wave']} - "
        f"{payload['budget_tokens']:,} tokens per row",
        f"{payload['optimizer_steps']:,} optimizer steps of "
        f"{payload['global_batch_tokens']:,} tokens, identical for every row",
        "",
        f"{'#':>2}  {'row':<24} {'APU':>4} {'nd':>3} {'GF/tok':>7} "
        f"{'EFLOP':>7} {'h@5%':>6} {'h@10%':>6} {'h@15%':>6}",
    ]
    for row in payload["rows"]:
        lines.append(
            f"{row['index']:2d}  {row['row']:<24} {row['apus']:4d} "
            f"{int(row['nodes']):3d} {row['gflops_per_token']:7.2f} "
            f"{row['total_exaflops']:7.1f} "
            f"{row['hours_at_mfu']['5%']:6.1f} {row['hours_at_mfu']['10%']:6.1f} "
            f"{row['hours_at_mfu']['15%']:6.1f}"
        )
    allocation = payload["allocation"]
    lines += [
        "",
        f"{allocation['rows']} rows, {allocation['allocated_apus']} APUs allocated, "
        f"{allocation['spare_apus']} spare",
        f"campaign total {payload['campaign_exaflops']:,.1f} EFLOP",
        "wall clock (longest row, all rows concurrent): "
        + ", ".join(f"{k} MFU -> {v} h" for k, v in payload["wall_clock_hours"].items()),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoRE ablation campaign planner")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="Print the wave plan and cost model")
    plan_parser.add_argument("--wave", default="1", help="1, 2, 3, or all")
    plan_parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    plan_parser.add_argument("--json", action="store_true")

    slurm_parser = sub.add_parser("slurm", help="Emit sbatch files for the wave")
    slurm_parser.add_argument("--wave", default="1", help="1, 2, 3, or all")
    slurm_parser.add_argument("--destination", required=True)
    slurm_parser.add_argument("--output-root", required=True)
    slurm_parser.add_argument("--release-root", required=True)
    slurm_parser.add_argument(
        "--repo-root",
        default="${METIS_REPO:?set METIS_REPO}",
        help="Repo root on the target machine; may be a shell expression",
    )
    slurm_parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    slurm_parser.add_argument("--seed", type=int, default=16_062_026)
    slurm_parser.add_argument("--time-limit", default="36:00:00")

    sweep_parser = sub.add_parser(
        "sweep", help="Emit the archetype learning-rate sweep"
    )
    sweep_parser.add_argument("--destination", required=True)
    sweep_parser.add_argument("--output-root", required=True)
    sweep_parser.add_argument("--release-root", required=True)
    sweep_parser.add_argument(
        "--repo-root", default="${METIS_REPO:?set METIS_REPO}"
    )
    sweep_parser.add_argument(
        "--budget-tokens", type=int, default=SWEEP_BUDGET_TOKENS
    )
    sweep_parser.add_argument("--seed", type=int, default=16_062_026)

    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = plan(wave=args.wave, budget_tokens=args.budget_tokens)
        print(
            json.dumps(payload, indent=2, sort_keys=True)
            if args.json
            else _format_plan(payload)
        )
        return 0

    if args.command == "sweep":
        for path in emit_sweep(
            Path(args.destination),
            repo_root=args.repo_root,
            output_root=args.output_root,
            release_root=args.release_root,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
        ):
            print(path)
        return 0

    written = emit_slurm(
        Path(args.destination),
        wave=args.wave,
        # Kept as a raw string: the committed launchers reference shell
        # variables so they stay portable, and resolving would bake in
        # whichever machine generated them.
        repo_root=args.repo_root,
        output_root=args.output_root,
        release_root=args.release_root,
        budget_tokens=args.budget_tokens,
        seed=args.seed,
        time_limit=args.time_limit,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
