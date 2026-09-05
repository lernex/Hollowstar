"""Campaign planning: allocation arithmetic, cost model, and launcher emission.

``python -m metis_ablation.campaign plan`` prints the wave that
``docs/papers/more/ablation_campaign.md`` describes, computed from the same
audited FLOP model the trainer reports against rather than from a parallel
hand-derivation.  ``... slurm`` emits the sbatch files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from metis_training.metrics import (
    MI300A_DENSE_PEAK_FLOPS,
    estimate_hardware_flops,
    estimate_train_flops,
)

from .specs import (
    ABLATION_LADDER,
    ALL_SPECS,
    SECOND_SEED,
    WAVES,
    AblationSpec,
    CAMPAIGN_APUS,
    GLOBAL_BATCH_TOKENS,
    WAVE_1_BATCHES,
    WAVE_2_BATCHES,
    dense_control_report,
    spec_by_name,
    validate_allocation,
    wave_for_row,
)


DEFAULT_BUDGET_TOKENS = 50_000_000_000
DEFAULT_WAVE_BUDGET_TOKENS = {
    "1": 50_000_000_000,
    "2": 100_000_000_000,
    "3": 50_000_000_000,
}
DEFAULT_MFU_BAND = (0.05, 0.10, 0.15)
SWEEP_ARCHETYPES = ("dense-param-matched", "moe-k4", "loop-fixed", "more-core")
SWEEP_LEARNING_RATES = (1.2e-4, 1.8e-4, 2.6e-4)
SWEEP_BUDGET_TOKENS = 1_000_000_000
SWEEP_BATCHES = {
    "1a": ("dense-param-matched", "loop-fixed"),
    "1b": ("moe-k4", "more-core"),
}

_LEARNING_RATE_ARCHETYPE = {
    "dense-flop-matched": "dense-param-matched",
    "dense-param-matched": "dense-param-matched",
    "moe-k4": "moe-k4",
    "moe-k8": "moe-k4",
    "loop-fixed": "loop-fixed",
    "loop-pathway-frozen": "loop-fixed",
    "mor-dense-ffn": "more-core",
    "mor-fixed-k": "more-core",
    "fixed-depth-adaptive-k": "more-core",
    "more-core": "more-core",
    "more-rm": "more-core",
    "random-k": "more-core",
    "random-depth": "more-core",
}


def _wave_specs(wave: str) -> tuple[AblationSpec, ...]:
    if wave == "all":
        return ALL_SPECS
    try:
        return WAVES[wave]
    except KeyError:
        raise SystemExit(
            f"Unknown wave {wave!r}; choose 1 (ladder), 2 (scaling), 3 (seeds), or all."
        ) from None


def _wave_one_batches(
    specs: tuple[AblationSpec, ...],
) -> tuple[tuple[str, tuple[AblationSpec, ...]], ...]:
    by_name = {spec.name: spec for spec in specs}
    return tuple(
        (
            batch_name,
            tuple(by_name[spec.name] for spec in batch_specs),
        )
        for batch_name, batch_specs in WAVE_1_BATCHES.items()
    )


def _wave_two_batches(
    specs: tuple[AblationSpec, ...],
) -> tuple[tuple[str, tuple[AblationSpec, ...]], ...]:
    by_name = {spec.name: spec for spec in specs}
    return tuple(
        (
            batch_name,
            tuple(by_name[spec.name] for spec in batch_specs),
        )
        for batch_name, batch_specs in WAVE_2_BATCHES.items()
    )


def _execution_groups(
    wave: str,
    specs: tuple[AblationSpec, ...],
) -> tuple[tuple[str, tuple[AblationSpec, ...]], ...]:
    if wave == "1":
        return _wave_one_batches(specs)
    if wave == "2":
        return _wave_two_batches(specs)
    if wave == "all":
        return (
            *_wave_one_batches(WAVES["1"]),
            *_wave_two_batches(WAVES["2"]),
            ("3", WAVES["3"]),
        )
    return ((wave, specs),)


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
    model_per_token = estimate_train_flops(
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
        # At the row's own k and depth, not the config's defaults. The k=8
        # control fixes k outside the config, so its audited mean is four
        # experts a layer short of what it actually runs.
        "active_parameters_per_pass": config.active_parameters_per_pass(width),
        "active_parameters_per_token": config.active_parameters_per_pass(width) * depth,
        "model_gflops_per_token": model_per_token / 1e9,
        "gflops_per_token": per_token / 1e9,
        "total_exaflops": per_token * budget_tokens / 1e18,
        "iso_flop": spec.iso_flop,
        "measured_tokens_per_second": spec.measured_tokens_per_second,
        "hours_at_measured_rate": (
            budget_tokens / spec.measured_tokens_per_second / 3600.0
            if spec.measured_tokens_per_second is not None
            else None
        ),
        "notes": spec.notes,
    }


def plan(
    *,
    wave: str = "1",
    budget_tokens: int | None = None,
    mfu_band: Sequence[float] = DEFAULT_MFU_BAND,
    total_apus: int = CAMPAIGN_APUS,
) -> dict[str, Any]:
    specs = _wave_specs(wave)
    execution_groups = _execution_groups(wave, specs)
    group_reports = {
        name: validate_allocation(candidate, total_apus=total_apus)
        for name, candidate in execution_groups
    }
    max_allocated = max(
        report["allocated_apus"] for report in group_reports.values()
    )
    allocation = {
        "rows": len(specs),
        "allocated_apus": max_allocated,
        "lane_apus": sum(spec.apus for spec in specs),
        "spare_apus": total_apus - max_allocated,
        "global_batch_sequences": GLOBAL_BATCH_TOKENS // 4_096,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "execution_batches": {
            name: {
                "rows": [spec.name for spec in candidate],
                "allocated_apus": group_reports[name]["allocated_apus"],
                "spare_apus": group_reports[name]["spare_apus"],
            }
            for name, candidate in execution_groups
        },
    }
    row_budgets = {
        spec.name: (
            int(budget_tokens)
            if budget_tokens is not None
            else DEFAULT_WAVE_BUDGET_TOKENS[wave_for_row(spec.name)]
        )
        for spec in specs
    }
    rows = [
        row_cost(spec, budget_tokens=row_budgets[spec.name])
        for spec in specs
    ]
    peak = MI300A_DENSE_PEAK_FLOPS["fp8"]
    for row in rows:
        row["hours_at_mfu"] = {
            f"{mfu:.0%}": round(
                row["total_exaflops"] * 1e18 / (row["apus"] * peak * mfu) / 3600.0, 2
            )
            for mfu in mfu_band
        }
    campaign_exaflops = sum(row["total_exaflops"] for row in rows)
    distinct_budgets = sorted(set(row_budgets.values()))
    budget_payload: int | dict[str, int] = (
        distinct_budgets[0]
        if len(distinct_budgets) == 1
        else dict(DEFAULT_WAVE_BUDGET_TOKENS)
    )
    step_payload: int | dict[str, int] = (
        distinct_budgets[0] // GLOBAL_BATCH_TOKENS
        if len(distinct_budgets) == 1
        else {
            candidate: tokens // GLOBAL_BATCH_TOKENS
            for candidate, tokens in DEFAULT_WAVE_BUDGET_TOKENS.items()
        }
    )
    row_by_name = {row["row"]: row for row in rows}
    return {
        "wave": wave,
        "budget_tokens": budget_payload,
        "optimizer_steps": step_payload,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "allocation": allocation,
        "campaign_exaflops": round(campaign_exaflops, 1),
        "wall_clock_hours": {
            f"{mfu:.0%}": round(
                sum(
                    max(
                        row_by_name[spec.name]["hours_at_mfu"][f"{mfu:.0%}"]
                        for spec in candidate
                    )
                    for _name, candidate in execution_groups
                ),
                2,
            )
            for mfu in mfu_band
        },
        "measured_wall_clock_hours": round(
            sum(
                max(
                    row_by_name[spec.name]["hours_at_measured_rate"]
                    for spec in candidate
                    if row_by_name[spec.name]["hours_at_measured_rate"] is not None
                )
                for _name, candidate in execution_groups
                if all(
                    row_by_name[spec.name]["hours_at_measured_rate"] is not None
                    for spec in candidate
                )
            ),
            2,
        )
        if all(row["hours_at_measured_rate"] is not None for row in rows)
        else None,
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
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}

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

# The ROCm stack is not on the default path on Portage, and a login shell that
# happens to have the right venv activated is not a reproducible launch. The
# activation script is named explicitly so the job records which runtime it ran
# against; see docs/papers/more/ablation_campaign.md. Checked here so an unset
# variable fails the job immediately rather than once per task.
: "${{METIS_ABLATION_RUNTIME:?set METIS_ABLATION_RUNTIME to the runtime activation script}}"
export METIS_ABLATION_RUNTIME

mkdir -p "{output_root}/{row}"

# Resolved in the batch shell so that an output root written as a shell
# expression expands once, here, rather than inside the single-quoted step body.
export METIS_ABLATION_OUTPUT="{output_root}"
export METIS_ABLATION_RELEASE="{release_root}"
export METIS_ABLATION_TELEMETRY="$METIS_ABLATION_OUTPUT/{row}/telemetry/rank-00000.jsonl"

# Portage's parry partition defines no GPU gres, so the four MI300A APUs are
# addressed by local task id rather than requested from Slurm. RANK and
# LOCAL_RANK have to be resolved inside the step, because SLURM_PROCID only
# exists per task -- exporting them from the batch shell would launch every
# task as rank 0 against APU 0.
srun --kill-on-bad-exit=1 --network=disable_rdzv_get bash -c '
set -euo pipefail
export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
# Sourced per task rather than once in the batch shell. The runtime derives
# per-rank scratch paths from SLURM_PROCID, which only exists inside the step;
# sourced above, all four ranks on a node would share one Triton JIT cache and
# corrupt each other compiled kernels -- which surfaces as
# "LLVM ERROR: IO failure on output stream", not as anything about caches.
source "$METIS_ABLATION_RUNTIME"
exec python -m metis_ablation.train \\
  --row '"'"'{row}'"'"' \\
  --output "$METIS_ABLATION_OUTPUT" \\
  --release-root "$METIS_ABLATION_RELEASE" \\
  --budget-tokens {budget_tokens} \\
  --seed {seed} \\
{learning_rate_arg}{schedule_total_steps_arg}  --checkpoint-every {checkpoint_every} \\
  --analysis-every {analysis_every} \\
  --telemetry-every {telemetry_every}
' &
step_pid=$!

# A compute-node reboot can remove every rank on that node without making this
# Portage Slurm build finish the step. The surviving ranks then stay at 100%
# GPU inside collectives forever. Require rank-zero telemetry to advance; a
# healthy cold start takes minutes, while twenty minutes without one new byte
# is a lost-rank failure, not slow training.
stall_timeout="${{METIS_ABLATION_STALL_TIMEOUT_SECONDS:-1200}}"
stall_poll="${{METIS_ABLATION_STALL_POLL_SECONDS:-30}}"
(
  last_size=-1
  last_progress="$(date +%s)"
  while kill -0 "$step_pid" 2>/dev/null; do
    sleep "$stall_poll"
    current_size=0
    if [ -f "$METIS_ABLATION_TELEMETRY" ]; then
      current_size="$(wc -c < "$METIS_ABLATION_TELEMETRY")"
    fi
    now="$(date +%s)"
    if [ "$current_size" -ne "$last_size" ]; then
      last_size="$current_size"
      last_progress="$now"
    elif [ $((now - last_progress)) -ge "$stall_timeout" ]; then
      echo "metis: no rank-zero telemetry progress for ${{stall_timeout}}s; terminating stalled step" >&2
      kill -TERM "$step_pid" 2>/dev/null || true
      sleep 60
      kill -KILL "$step_pid" 2>/dev/null || true
      exit 124
    fi
  done
) &
watchdog_pid=$!

set +e
wait "$step_pid"
step_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
exit "$step_status"
"""


def emit_slurm(
    destination: Path,
    *,
    wave: str = "1",
    repo_root: str,
    output_root: str,
    release_root: str,
    budget_tokens: int | None = None,
    seed: int = 16_062_026,
    time_limit: str = "36:00:00",
    checkpoint_every: int = 500,
    analysis_every: int = 1_000,
    telemetry_every: int = 10,
    base_port: int = 29_500,
    cpus_per_task: int = 48,
    learning_rates: dict[str, float] | None = None,
) -> list[Path]:
    if wave == "all":
        raise ValueError("Emit waves separately so their launch order is explicit.")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative.")
    budget_tokens = (
        DEFAULT_WAVE_BUDGET_TOKENS[wave]
        if budget_tokens is None
        else int(budget_tokens)
    )
    specs = _wave_specs(wave)
    execution_groups = _execution_groups(wave, specs)
    for _name, candidate in execution_groups:
        validate_allocation(candidate)
    required_archetypes = {
        learning_rate_archetype(spec.name)
        for spec in specs
    }
    if learning_rates is not None:
        missing = required_archetypes - set(learning_rates)
        if missing:
            raise ValueError(
                "Learning-rate selection is missing archetypes: "
                + ", ".join(sorted(missing))
            )
        if any(
            not math.isfinite(learning_rates[name]) or learning_rates[name] <= 0
            for name in required_archetypes
        ):
            raise ValueError("Selected learning rates must be finite and positive.")
    destination = destination / f"wave{wave}"
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.sbatch"):
        stale.unlink()
    for stale in destination.glob("launch-wave*.sh"):
        stale.unlink()
    written: list[Path] = []
    for spec in specs:
        # Wave 3 is the paired-seed repeat: same data order, different
        # initialization. Its seed must differ or it is not a repeat at all.
        row_seed = SECOND_SEED if wave_for_row(spec.name) == "3" else seed
        archetype = learning_rate_archetype(spec.name)
        learning_rate_env = (
            "METIS_ABLATION_LR_" + archetype.upper().replace("-", "_")
        )
        selected_learning_rate = (
            learning_rates.get(archetype) if learning_rates is not None else None
        )
        learning_rate_arg = (
            f"  --learning-rate {selected_learning_rate:g} \\\n"
            if selected_learning_rate is not None
            else (
                f'  --learning-rate "${{{learning_rate_env}:?run the sweep and '
                f'set {learning_rate_env}, or regenerate with --learning-rates}}" \\\n'
            )
        )
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
            learning_rate_arg=learning_rate_arg,
            schedule_total_steps_arg="",
            checkpoint_every=checkpoint_every,
            analysis_every=analysis_every,
            telemetry_every=telemetry_every,
            repo_root=repo_root,
            master_port=base_port + spec.index,
            global_batch_tokens=GLOBAL_BATCH_TOKENS,
            cpus_per_task=cpus_per_task,
        )
        path = destination / f"{spec.index:02d}-{spec.name}.sbatch"
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        written.append(path)

    paths_by_row = {
        path.name.split("-", 1)[1].removesuffix(".sbatch"): path
        for path in written
    }
    for group_name, group_specs in execution_groups:
        launcher = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"# Launch execution batch {group_name} ({len(group_specs)} rows, "
            f"{sum(spec.apus for spec in group_specs)} APUs).",
            'exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[020,026,063-064]}"',
            "sbatch_args=()",
            'if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi',
            "",
        ]
        if learning_rates is None:
            launcher += [
                (
                    f': "${{METIS_ABLATION_LR_{archetype.upper().replace("-", "_")}'
                    f':?set the selected {archetype} learning rate}}"'
                )
                for archetype in sorted(
                    {learning_rate_archetype(spec.name) for spec in group_specs}
                )
            ]
            launcher.append("")
        for spec in group_specs:
            row_output = f"{output_root}/{spec.name}"
            launcher += [
                f'mkdir -p "{row_output}"',
                (
                    f'sbatch "${{sbatch_args[@]}}" --output="{row_output}/slurm-%j.out" '
                    f'--error="{row_output}/slurm-%j.err" '
                    f'"$(dirname "$0")/{paths_by_row[spec.name].name}"'
                ),
            ]
        launcher_name = (
            f"launch-wave-{group_name}.sh"
            if len(execution_groups) > 1
            else "launch-wave.sh"
        )
        wave_path = destination / launcher_name
        wave_path.write_text("\n".join(launcher) + "\n", encoding="utf-8")
        wave_path.chmod(0o755)
        written.append(wave_path)
    return written


def learning_rate_archetype(row: str) -> str:
    base = row.removesuffix("-seed2").removesuffix("-xs").removesuffix("-xl")
    try:
        return _LEARNING_RATE_ARCHETYPE[base]
    except KeyError:
        raise ValueError(f"No learning-rate archetype is defined for {row!r}.") from None


def load_learning_rates(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Learning-rate selection must be a JSON object.")
    missing = set(SWEEP_ARCHETYPES) - set(payload)
    extra = set(payload) - set(SWEEP_ARCHETYPES)
    if missing or extra:
        raise ValueError(
            "Learning-rate selection keys must be exactly "
            f"{SWEEP_ARCHETYPES}; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    rates = {name: float(payload[name]) for name in SWEEP_ARCHETYPES}
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError("Selected learning rates must be finite and positive.")
    return rates


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
    cpus_per_task: int = 48,
) -> list[Path]:
    """Emit the archetype learning-rate sweep as its own small wave."""

    destination = destination / "sweep"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paths_by_archetype: dict[str, list[Path]] = {
        archetype: [] for archetype in SWEEP_ARCHETYPES
    }
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
                learning_rate_arg=f"  --learning-rate {rate:g} \\\n",
                schedule_total_steps_arg=(
                    "  --schedule-total-steps "
                    f"{DEFAULT_BUDGET_TOKENS // GLOBAL_BATCH_TOKENS} \\\n"
                ),
                checkpoint_every=0,
                analysis_every=0,
                telemetry_every=10,
                repo_root=repo_root,
                master_port=port,
                global_batch_tokens=GLOBAL_BATCH_TOKENS,
                cpus_per_task=cpus_per_task,
            )
            body = body.replace(
                "  --telemetry-every 10",
                "  --telemetry-every 10 \\\n  --no-resume",
            )
            body = body.replace(f"--job-name=more-{spec.name}", f"--job-name=sweep-{tag}")
            path = destination / f"{tag}.sbatch"
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)
            written.append(path)
            paths_by_archetype[archetype].append(path)
            port += 1

    (destination / "launch-sweep.sh").unlink(missing_ok=True)
    for batch_name, archetypes in SWEEP_BATCHES.items():
        batch_paths = [
            path
            for archetype in archetypes
            for path in paths_by_archetype[archetype]
        ]
        batch_apus = sum(
            spec_by_name(archetype).apus * len(SWEEP_LEARNING_RATES)
            for archetype in archetypes
        )
        if batch_apus > CAMPAIGN_APUS:
            raise ValueError(
                f"Sweep batch {batch_name} requests {batch_apus} APUs, above "
                f"the {CAMPAIGN_APUS}-APU campaign allocation."
            )
        launcher = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"# Learning-rate sweep batch {batch_name}: {len(batch_paths)} runs, "
            f"{batch_apus} APUs, {budget_tokens:,} tokens per run.",
            'exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[020,026,063-064]}"',
            "sbatch_args=()",
            'if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi',
            "",
        ]
        for path in batch_paths:
            tag = path.stem
            row_output = f"{output_root}/sweep/{tag}"
            launcher += [
                f'mkdir -p "{row_output}"',
                (
                    f'sbatch "${{sbatch_args[@]}}" --output="{row_output}/slurm-%j.out" '
                    f'--error="{row_output}/slurm-%j.err" '
                    f'"$(dirname "$0")/{path.name}"'
                ),
            ]
        wave_path = destination / f"launch-sweep-{batch_name}.sh"
        wave_path.write_text("\n".join(launcher) + "\n", encoding="utf-8")
        wave_path.chmod(0o755)
        written.append(wave_path)
    return written


def _format_plan(payload: dict[str, Any]) -> str:
    if isinstance(payload["budget_tokens"], dict):
        budgets = ", ".join(
            f"wave {candidate}: {tokens:,}"
            for candidate, tokens in payload["budget_tokens"].items()
        )
        steps = ", ".join(
            f"wave {candidate}: {count:,}"
            for candidate, count in payload["optimizer_steps"].items()
        )
        heading = [
            f"MoRE ablation wave {payload['wave']} - {budgets} tokens per row",
            f"optimizer steps ({steps}) of "
            f"{payload['global_batch_tokens']:,} tokens",
        ]
    else:
        heading = [
            f"MoRE ablation wave {payload['wave']} - "
            f"{payload['budget_tokens']:,} tokens per row",
            f"{payload['optimizer_steps']:,} optimizer steps of "
            f"{payload['global_batch_tokens']:,} tokens, identical for every row",
        ]
    lines = [
        *heading,
        "",
        f"{'#':>2}  {'row':<24} {'APU':>4} {'nd':>3} {'stored':>9} "
        f"{'act/tok':>9} {'model':>7} {'exec':>7} {'EFLOP':>7} "
        f"{'h@5%':>6} {'h@10%':>6} {'h@15%':>6}  iso",
    ]
    for row in payload["rows"]:
        lines.append(
            f"{row['index']:2d}  {row['row']:<24} {row['apus']:4d} "
            f"{int(row['nodes']):3d} {row['stored_parameters'] / 1e6:8.1f}M "
            f"{row['active_parameters_per_token'] / 1e6:8.1f}M "
            f"{row['model_gflops_per_token']:7.2f} "
            f"{row['gflops_per_token']:7.2f} "
            f"{row['total_exaflops']:7.1f} "
            f"{row['hours_at_mfu']['5%']:6.1f} {row['hours_at_mfu']['10%']:6.1f} "
            f"{row['hours_at_mfu']['15%']:6.1f}  "
            f"{'yes' if row['iso_flop'] else 'NO '}"
        )
    allocation = payload["allocation"]
    lines += [
        "",
        f"{allocation['rows']} rows, {allocation['lane_apus']} lane APUs total, "
        f"{allocation['allocated_apus']} APUs maximum concurrent, "
        f"{allocation['spare_apus']} spare",
        f"campaign total {payload['campaign_exaflops']:,.1f} EFLOP",
        "model = training FLOPs excluding checkpoint replay; exec = actual "
        "hardware FLOPs including replay.",
        "iso = does this row match the reference model FLOPs per token. Rows "
        "marked NO are deliberately off-budget and must be reported against "
        "FLOPs, not against steps or tokens.",
        "wall clock (sequential execution batches): "
        + ", ".join(f"{k} MFU -> {v} h" for k, v in payload["wall_clock_hours"].items()),
    ]
    if payload["measured_wall_clock_hours"] is not None:
        lines.append(
            "measured Wave-1 schedule: "
            f"{payload['measured_wall_clock_hours']:.1f} h across its execution batches"
        )
    for name, batch in allocation["execution_batches"].items():
        lines.append(
            f"batch {name}: {batch['allocated_apus']} APUs -> "
            + ", ".join(batch["rows"])
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoRE ablation campaign planner")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="Print the wave plan and cost model")
    plan_parser.add_argument("--wave", default="1", help="1, 2, 3, or all")
    plan_parser.add_argument("--budget-tokens", type=int, default=None)
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
    slurm_parser.add_argument("--budget-tokens", type=int, default=None)
    slurm_parser.add_argument("--seed", type=int, default=16_062_026)
    slurm_parser.add_argument("--time-limit", default="36:00:00")
    slurm_parser.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="Optimizer steps between recovery checkpoints; zero disables periodic checkpoints",
    )
    slurm_parser.add_argument(
        "--learning-rates",
        default=None,
        help=(
            "JSON file mapping the four sweep archetypes to their selected "
            "learning rates"
        ),
    )

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
        checkpoint_every=args.checkpoint_every,
        learning_rates=(
            load_learning_rates(Path(args.learning_rates))
            if args.learning_rates
            else None
        ),
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
