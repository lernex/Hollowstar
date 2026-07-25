from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .config import FamilyTopology, PortageConfig
from .util import atomic_write_json, file_sha256, json_sha256, read_json, utc_now
from metis_training.precision_plan import measured_role_dtype_map


@dataclass(frozen=True)
class Candidate:
    micro_batch_size: int
    grad_accum_steps: int
    precision_profile: str
    compile_mode: str
    dispatch_overlap: bool
    ngram_table_mode: str = "replicated"
    learning_rate: float | None = None


@dataclass(frozen=True)
class TuningBounds:
    micro_batch_sizes: tuple[int, ...]
    grad_accum_steps: tuple[int, ...]
    precision_profiles: tuple[str, ...]
    compile_modes: tuple[str, ...]
    dispatch_overlap: tuple[bool, ...]
    ngram_table_modes: tuple[str, ...]
    learning_rates: tuple[float, ...]
    preferred_learning_rate: float
    global_token_batch_min: int
    global_token_batch_max: int
    global_token_batch_target: int
    maximum_hbm_fraction: float
    maximum_fp8_loss_relative_error: float
    maximum_ngram_layout_loss_relative_error: float
    maximum_update_to_weight_ratio: float
    maximum_grad_norm: float


def _positive_ints(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a non-empty list")
    rows = tuple(int(item) for item in value)
    if any(item <= 0 for item in rows) or len(set(rows)) != len(rows):
        raise RuntimeError(f"{label} must contain unique positive integers")
    return rows


def _positive_floats(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} must be a non-empty list")
    rows = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0 for item in rows) or len(set(rows)) != len(rows):
        raise RuntimeError(f"{label} must contain unique positive finite values")
    return rows


def load_tuning_bounds(
    manifest_path: str | Path,
    *,
    default_maximum_hbm_fraction: float,
) -> TuningBounds:
    payload = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    autotune = payload.get("autotune", {})
    raw = autotune.get("bounds")
    if raw is None:
        raw = payload.get("training", {}).get("autotune", {}).get("bounds")
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Manifest {manifest_path} is missing manifest-bounded autotune.bounds"
        )
    batch = raw.get("global_token_batch")
    gates = autotune.get("gates", raw.get("gates", {}))
    if not isinstance(batch, dict) or not isinstance(gates, dict):
        raise RuntimeError("autotune.bounds requires global_token_batch and gates mappings")
    precision = tuple(str(item) for item in raw.get("precision_profiles", []))
    compile_modes = tuple(
        "eager" if str(item) == "none" else str(item)
        for item in raw.get("compile_modes", [])
    )
    overlaps_list: list[bool] = []
    for item in raw.get("dispatch_overlap", []):
        if isinstance(item, bool):
            overlaps_list.append(item)
        elif str(item).lower() in {"on", "off"}:
            overlaps_list.append(str(item).lower() == "on")
        else:
            raise RuntimeError("dispatch_overlap candidates must be on/off")
    overlaps = tuple(overlaps_list)
    ngram_table_modes = tuple(
        str(item) for item in raw.get("ngram_table_modes", [])
    )
    learning_rates = _positive_floats(raw.get("learning_rates"), label="learning_rates")
    preferred = float(raw.get("preferred_learning_rate", learning_rates[0]))
    if preferred not in learning_rates:
        raise RuntimeError("preferred_learning_rate must be one of the bounded candidates")
    if (
        not precision
        or any(item not in {"fp8", "bf16"} for item in precision)
        or not compile_modes
        or any(item not in {"max-autotune", "reduce-overhead", "default", "eager"} for item in compile_modes)
        or not overlaps
        or not ngram_table_modes
        or any(
            item not in {"replicated", "row_sharded"}
            for item in ngram_table_modes
        )
    ):
        raise RuntimeError("Manifest contains unsupported precision/compile/overlap candidates")
    bounds = TuningBounds(
        micro_batch_sizes=_positive_ints(raw.get("micro_batch_sizes"), label="micro_batch_sizes"),
        grad_accum_steps=_positive_ints(raw.get("grad_accum_steps"), label="grad_accum_steps"),
        precision_profiles=precision,
        compile_modes=compile_modes,
        dispatch_overlap=overlaps,
        ngram_table_modes=ngram_table_modes,
        learning_rates=learning_rates,
        preferred_learning_rate=preferred,
        global_token_batch_min=int(batch["min"]),
        global_token_batch_max=int(batch["max"]),
        global_token_batch_target=int(batch["target"]),
        maximum_hbm_fraction=float(
            gates.get("max_hbm_fraction", default_maximum_hbm_fraction)
        ),
        maximum_fp8_loss_relative_error=float(
            gates["max_fp8_loss_relative_error"]
        ),
        maximum_ngram_layout_loss_relative_error=float(
            gates["max_ngram_layout_loss_relative_error"]
        ),
        maximum_update_to_weight_ratio=float(gates["max_update_to_weight_ratio"]),
        maximum_grad_norm=float(gates["max_grad_norm"]),
    )
    if not (
        0 < bounds.maximum_hbm_fraction < 1
        and 0 <= bounds.maximum_fp8_loss_relative_error < 1
        and 0 <= bounds.maximum_ngram_layout_loss_relative_error < 1
        and 0 < bounds.maximum_update_to_weight_ratio < 1
        and bounds.maximum_grad_norm > 0
        and 0 < bounds.global_token_batch_min
        <= bounds.global_token_batch_target
        <= bounds.global_token_batch_max
    ):
        raise RuntimeError("Manifest autotune gate/batch bounds are invalid")
    return bounds


def enumerate_performance_candidates(
    bounds: TuningBounds,
    *,
    world_size: int,
    sequence_length: int,
    maximum_trials: int,
) -> list[Candidate]:
    if maximum_trials <= 0:
        raise RuntimeError("maximum_trials must be positive")
    rows: list[tuple[int, Candidate]] = []
    for (
        precision,
        compile_mode,
        overlap,
        table_mode,
        micro_batch,
        accumulation,
    ) in itertools.product(
        bounds.precision_profiles,
        bounds.compile_modes,
        bounds.dispatch_overlap,
        bounds.ngram_table_modes,
        bounds.micro_batch_sizes,
        bounds.grad_accum_steps,
    ):
        global_tokens = micro_batch * accumulation * world_size * sequence_length
        if not bounds.global_token_batch_min <= global_tokens <= bounds.global_token_batch_max:
            continue
        candidate = Candidate(
            micro_batch_size=micro_batch,
            grad_accum_steps=accumulation,
            precision_profile=precision,
            compile_mode=compile_mode,
            dispatch_overlap=overlap,
            ngram_table_mode=table_mode,
        )
        distance = abs(global_tokens - bounds.global_token_batch_target)
        rows.append((distance, candidate))
    if not rows:
        raise RuntimeError("No manifest-bounded batch candidate fits the global token batch range")

    compile_order = {
        value: index for index, value in enumerate(bounds.compile_modes)
    }
    overlap_order = {
        value: index for index, value in enumerate(bounds.dispatch_overlap)
    }
    table_order = {
        value: index for index, value in enumerate(bounds.ngram_table_modes)
    }
    micro_order = {
        value: index for index, value in enumerate(bounds.micro_batch_sizes)
    }

    def balanced(pool: list[tuple[int, Candidate]], budget: int) -> list[Candidate]:
        """Build a marginal-coverage-first, throughput-oriented schedule."""

        remaining = list(pool)
        selected: list[Candidate] = []
        counts: dict[tuple[str, Any], int] = {}
        while remaining and len(selected) < budget:
            def score(row: tuple[int, Candidate]) -> tuple[Any, ...]:
                distance, candidate = row
                axes = (
                    ("compile", candidate.compile_mode),
                    ("overlap", candidate.dispatch_overlap),
                    ("ngram", candidate.ngram_table_mode),
                    ("micro_batch", candidate.micro_batch_size),
                )
                unseen = sum(counts.get(axis, 0) == 0 for axis in axes)
                balance = sum(1.0 / (1 + counts.get(axis, 0)) for axis in axes)
                return (
                    unseen,
                    balance,
                    -distance,
                    candidate.micro_batch_size,
                    -candidate.grad_accum_steps,
                    -compile_order[candidate.compile_mode],
                    -overlap_order[candidate.dispatch_overlap],
                    -table_order[candidate.ngram_table_mode],
                    -micro_order[candidate.micro_batch_size],
                )

            chosen_row = max(remaining, key=score)
            remaining.remove(chosen_row)
            chosen = chosen_row[1]
            selected.append(chosen)
            for axis in (
                ("compile", chosen.compile_mode),
                ("overlap", chosen.dispatch_overlap),
                ("ngram", chosen.ngram_table_mode),
                ("micro_batch", chosen.micro_batch_size),
            ):
                counts[axis] = counts.get(axis, 0) + 1
        return selected

    by_precision = {
        precision: [row for row in rows if row[1].precision_profile == precision]
        for precision in bounds.precision_profiles
    }
    fp8 = by_precision.get("fp8", [])
    bf16 = by_precision.get("bf16", [])
    if fp8 and bf16:
        # Reserve a quarter of the bounded performance budget for the BF16
        # numerical oracle/fallback. A row-sharded BF16 lane may require its
        # matching replicated reference, hence the minimum of two.
        bf16_budget = min(max(2, maximum_trials // 4), len(bf16))
        fp8_budget = maximum_trials - bf16_budget
        if fp8_budget <= 0:
            raise RuntimeError(
                "maximum_trials cannot cover FP8 exploration and a BF16 oracle"
            )
        return [
            *balanced(fp8, fp8_budget),
            *balanced(bf16, bf16_budget),
        ]
    precision = "fp8" if fp8 else "bf16"
    return balanced(by_precision[precision], maximum_trials)


def _required_performance_coverage(
    bounds: TuningBounds,
    *,
    world_size: int,
    sequence_length: int,
) -> dict[str, set[Any]]:
    feasible_micro_batches = {
        micro_batch
        for micro_batch in bounds.micro_batch_sizes
        if any(
            bounds.global_token_batch_min
            <= micro_batch
            * accumulation
            * world_size
            * sequence_length
            <= bounds.global_token_batch_max
            for accumulation in bounds.grad_accum_steps
        )
    }
    if not feasible_micro_batches:
        raise RuntimeError("No micro-batch size is feasible within the token-batch bounds")
    return {
        "compile_mode": set(bounds.compile_modes),
        "dispatch_overlap": set(bounds.dispatch_overlap),
        "ngram_table_mode": set(bounds.ngram_table_modes),
        "micro_batch_size": feasible_micro_batches,
    }


def _missing_performance_coverage(
    required: Mapping[str, set[Any]],
    measurements: Iterable[Candidate],
    *,
    precision_profile: str,
) -> dict[str, list[Any]]:
    observed = {
        "compile_mode": set(),
        "dispatch_overlap": set(),
        "ngram_table_mode": set(),
        "micro_batch_size": set(),
    }
    for candidate in measurements:
        if candidate.precision_profile != precision_profile:
            continue
        observed["compile_mode"].add(candidate.compile_mode)
        observed["dispatch_overlap"].add(candidate.dispatch_overlap)
        observed["ngram_table_mode"].add(candidate.ngram_table_mode)
        observed["micro_batch_size"].add(candidate.micro_batch_size)
    return {
        axis: sorted(values - observed[axis], key=str)
        for axis, values in required.items()
        if values - observed[axis]
    }


def _number(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Trainer probe omitted numeric {key}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Trainer probe returned non-finite {key}")
    return number


def validate_performance_report(
    report: dict[str, Any],
    *,
    candidate: Candidate,
    bounds: TuningBounds,
    hbm_bytes: int,
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if report.get("ok") is not True:
        reasons.append("trainer reported ok=false")
    if report.get("finite_loss") is not True:
        reasons.append("loss is not finite")
    for key in (
        "step_time_s",
        "non_padding_tokens",
        "estimated_train_flops",
        "peak_hbm_bytes",
        "overflow_drop_tokens",
        "collective_errors",
    ):
        try:
            _number(report, key)
        except RuntimeError as exc:
            reasons.append(str(exc))
    if reasons:
        return False, reasons, 0.0
    if _number(report, "overflow_drop_tokens") != 0:
        reasons.append("MoE overflow dropped tokens")
    if _number(report, "collective_errors") != 0:
        reasons.append("collective errors occurred")
    if _number(report, "peak_hbm_bytes") > hbm_bytes * bounds.maximum_hbm_fraction:
        reasons.append("HBM headroom gate failed")
    if candidate.precision_profile == "fp8":
        try:
            error = _number(report, "loss_relative_error_vs_bf16")
            if error > bounds.maximum_fp8_loss_relative_error:
                reasons.append("FP8/BF16 loss parity gate failed")
        except RuntimeError as exc:
            reasons.append(str(exc))
    if report.get("ngram_table_mode") != candidate.ngram_table_mode:
        reasons.append("trainer did not execute the selected N-gram table mode")
    if candidate.ngram_table_mode == "row_sharded":
        try:
            error = _number(report, "ngram_layout_loss_relative_error")
            if error > bounds.maximum_ngram_layout_loss_relative_error:
                reasons.append(
                    "row-sharded/replicated N-gram loss parity gate failed"
                )
        except RuntimeError as exc:
            reasons.append(str(exc))
    step_time = _number(report, "step_time_s")
    throughput = _number(report, "non_padding_tokens") / step_time if step_time > 0 else 0.0
    if throughput <= 0 or not math.isfinite(throughput):
        reasons.append("measured throughput is not positive")
    return not reasons, reasons, throughput


def validate_optimizer_report(
    report: dict[str, Any],
    *,
    bounds: TuningBounds,
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if report.get("ok") is not True or report.get("finite_loss") is not True:
        reasons.append("optimizer canary did not complete with finite loss")
    numbers: dict[str, float] = {}
    for key in (
        "initial_loss",
        "final_loss",
        "max_grad_norm",
        "nonfinite_steps",
        "update_to_weight_ratio",
    ):
        try:
            numbers[key] = _number(report, key)
        except RuntimeError as exc:
            reasons.append(str(exc))
    if reasons:
        return False, reasons, float("inf")
    if numbers["nonfinite_steps"] != 0:
        reasons.append("optimizer canary had non-finite steps")
    if numbers["max_grad_norm"] > bounds.maximum_grad_norm:
        reasons.append("gradient norm gate failed")
    if numbers["update_to_weight_ratio"] > bounds.maximum_update_to_weight_ratio:
        reasons.append("update/weight ratio gate failed")
    if numbers["final_loss"] > numbers["initial_loss"] * 1.10:
        reasons.append("short canary loss diverged")
    return not reasons, reasons, numbers["final_loss"]


TrialRunner = Callable[[Candidate, Path, bool], dict[str, Any]]


def tune_family(
    *,
    config: PortageConfig,
    family: FamilyTopology,
    inventory_fingerprint: str,
    release_marker: dict[str, Any],
    hbm_bytes: int,
    output_directory: str | Path,
    run_trial: TrialRunner,
    excluded_candidates: Iterable[dict[str, Any]] = (),
    available_precision_profiles: Iterable[str] | None = None,
    precision_role_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(precision_role_plan, Mapping):
        raise RuntimeError(
            f"{family.name} tuning requires its sealed exact-role precision plan"
        )
    role_plan = dict(precision_role_plan)
    if (
        role_plan.get("schema") != "metis.precision-role-plan/v1"
        or role_plan.get("family") != family.name
        or role_plan.get("plan_sha256")
        != json_sha256(role_plan, omit=("plan_sha256",))
    ):
        raise RuntimeError(
            f"{family.name} exact-role precision plan is corrupt or family-stale"
        )
    measured_role_map = measured_role_dtype_map(role_plan)
    fp8_role_available = "fp8" in set(measured_role_map.values())
    bounds = load_tuning_bounds(
        family.manifest,
        default_maximum_hbm_fraction=float(
            config.raw["autotune"]["default_maximum_hbm_fraction"]
        ),
    )
    sequence_length = int(
        yaml.safe_load(config.training_contract.read_text(encoding="utf-8"))["sequence_length"]
    )
    maximum_performance_trials = int(
        config.raw["autotune"]["maximum_trials_per_family"]
    )
    candidates = enumerate_performance_candidates(
        bounds,
        world_size=family.world_size,
        sequence_length=sequence_length,
        maximum_trials=maximum_performance_trials,
    )
    if available_precision_profiles is not None:
        available = {str(item) for item in available_precision_profiles}
        if not available or not available <= {"fp8", "bf16"}:
            raise RuntimeError("Runtime reported invalid available precision profiles")
        if "fp8" in available and "bf16" not in available:
            raise RuntimeError("FP8 exploration requires the BF16 numerical oracle")
        if "fp8" in available and not fp8_role_available:
            available.remove("fp8")
        candidates = [
            candidate
            for candidate in candidates
            if candidate.precision_profile in available
        ]
    excluded = {
        json_sha256(candidate)
        for candidate in excluded_candidates
        if isinstance(candidate, dict)
    }
    candidates = [
        candidate
        for candidate in candidates
        if json_sha256(asdict(candidate)) not in excluded
    ]
    if not candidates:
        raise RuntimeError(f"Every manifest-bounded candidate was rejected for {family.name}")
    required_coverage = _required_performance_coverage(
        bounds,
        world_size=family.world_size,
        sequence_length=sequence_length,
    )
    scheduled_precisions = {
        candidate.precision_profile for candidate in candidates
    }
    primary_precision = "fp8" if "fp8" in scheduled_precisions else "bf16"
    scheduled_missing = _missing_performance_coverage(
        required_coverage,
        candidates,
        precision_profile=primary_precision,
    )
    if scheduled_missing:
        raise RuntimeError(
            f"maximum_trials cannot cover every {primary_precision} performance "
            f"axis for {family.name}: {scheduled_missing}"
        )
    trial_rows: list[dict[str, Any]] = []
    accepted: list[tuple[float, Candidate, dict[str, Any]]] = []
    safe_measurements: list[Candidate] = []
    performance_trials_used = 0
    replicated_references: dict[
        tuple[Any, ...],
        tuple[bool, list[str], float, dict[str, Any], Path],
    ] = {}

    def evaluate_performance(
        candidate: Candidate,
        report_path: Path,
        *,
        kind: str,
    ) -> tuple[bool, list[str], float, dict[str, Any]]:
        nonlocal performance_trials_used

        def execute(
            measured_candidate: Candidate,
            measured_path: Path,
        ) -> dict[str, Any]:
            nonlocal performance_trials_used
            if performance_trials_used >= maximum_performance_trials:
                raise RuntimeError(
                    f"Performance probe budget exhausted for {family.name}"
                )
            performance_trials_used += 1
            return run_trial(measured_candidate, measured_path, False)

        report = execute(candidate, report_path)
        if candidate.ngram_table_mode == "row_sharded":
            replicated = Candidate(
                **{
                    **asdict(candidate),
                    "ngram_table_mode": "replicated",
                }
            )
            key = (
                replicated.micro_batch_size,
                replicated.grad_accum_steps,
                replicated.precision_profile,
                replicated.compile_mode,
                replicated.dispatch_overlap,
                replicated.learning_rate,
            )
            reference = replicated_references.get(key)
            if reference is None:
                reference_path = report_path.with_name(
                    report_path.stem + "-ngram-replicated.json"
                )
                reference_report = execute(
                    replicated,
                    reference_path,
                )
                reference_passed, reference_reasons, reference_throughput = (
                    validate_performance_report(
                        reference_report,
                        candidate=replicated,
                        bounds=bounds,
                        hbm_bytes=hbm_bytes,
                    )
                )
                if (
                    reference_report.get("precision_role_plan_sha256")
                    != role_plan["plan_sha256"]
                ):
                    reference_passed = False
                    reference_reasons = [
                        *reference_reasons,
                        "replicated reference changed the precision role plan",
                    ]
                reference = (
                    reference_passed,
                    reference_reasons,
                    reference_throughput,
                    reference_report,
                    reference_path,
                )
                replicated_references[key] = reference
                trial_rows.append(
                    {
                        "kind": "ngram_replicated_reference",
                        "candidate": asdict(replicated),
                        "report": str(reference_path),
                        "report_sha256": file_sha256(reference_path),
                        "passed": reference_passed,
                        "reasons": reference_reasons,
                        "tokens_per_second": reference_throughput,
                    }
                )
                if reference_passed:
                    accepted.append(
                        (
                            reference_throughput,
                            replicated,
                            reference_report,
                        )
                    )
                    safe_measurements.append(replicated)
            reference_passed, _, _, reference_report, reference_path = reference
            try:
                row_loss = _number(report, "final_loss")
                reference_loss = _number(reference_report, "final_loss")
                relative_error = abs(row_loss - reference_loss) / max(
                    abs(reference_loss),
                    1.0e-12,
                )
            except RuntimeError:
                relative_error = 1.0e30
            report = {
                **report,
                "ngram_layout_reference_ok": reference_passed,
                "ngram_layout_loss_relative_error": relative_error,
                "ngram_replicated_reference_report": str(reference_path),
                "ngram_replicated_reference_report_sha256": file_sha256(
                    reference_path
                ),
            }
            atomic_write_json(report_path, report)
        passed, reasons, throughput = validate_performance_report(
            report,
            candidate=candidate,
            bounds=bounds,
            hbm_bytes=hbm_bytes,
        )
        if report.get("precision_role_plan_sha256") != role_plan["plan_sha256"]:
            passed = False
            reasons = [
                *reasons,
                "trainer did not execute the sealed precision role plan",
            ]
        if (
            candidate.ngram_table_mode == "row_sharded"
            and report.get("ngram_layout_reference_ok") is not True
        ):
            passed = False
            reasons = [*reasons, "replicated N-gram reference lane failed"]
        trial_rows.append(
            {
                "kind": kind,
                "candidate": asdict(candidate),
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "passed": passed,
                "reasons": reasons,
                "tokens_per_second": throughput,
            }
        )
        if passed:
            accepted.append((throughput, candidate, report))
            safe_measurements.append(candidate)
        return passed, reasons, throughput, report

    def reference_key(candidate: Candidate) -> tuple[Any, ...]:
        return (
            candidate.micro_batch_size,
            candidate.grad_accum_steps,
            candidate.precision_profile,
            candidate.compile_mode,
            candidate.dispatch_overlap,
            candidate.learning_rate,
        )

    def estimated_trial_cost(candidate: Candidate) -> int:
        if candidate.ngram_table_mode != "row_sharded":
            return 1
        replicated = Candidate(
            **{
                **asdict(candidate),
                "ngram_table_mode": "replicated",
            }
        )
        return 1 if reference_key(replicated) in replicated_references else 2

    performance_index = 0

    def measure_until_covered(
        schedule: list[Candidate],
        *,
        precision_profile: str,
        trial_limit: int,
    ) -> dict[str, list[Any]]:
        nonlocal performance_index
        missing = _missing_performance_coverage(
            required_coverage,
            safe_measurements,
            precision_profile=precision_profile,
        )
        for candidate in schedule:
            if not missing:
                break
            cost = estimated_trial_cost(candidate)
            if performance_trials_used + cost > trial_limit:
                continue
            report_path = (
                output
                / "trials"
                / f"performance-{performance_index:03d}.json"
            )
            performance_index += 1
            evaluate_performance(
                candidate,
                report_path,
                kind="performance",
            )
            missing = _missing_performance_coverage(
                required_coverage,
                safe_measurements,
                precision_profile=precision_profile,
            )
        return missing

    fp8_candidates = [
        candidate for candidate in candidates
        if candidate.precision_profile == "fp8"
    ]
    bf16_candidates = [
        candidate for candidate in candidates
        if candidate.precision_profile == "bf16"
    ]
    if fp8_candidates and not bf16_candidates:
        raise RuntimeError("FP8 performance tuning requires bounded BF16 candidates")
    if fp8_candidates:
        oracle_reserve = 2 if bf16_candidates else 0
        missing_fp8 = measure_until_covered(
            fp8_candidates,
            precision_profile="fp8",
            trial_limit=maximum_performance_trials - oracle_reserve,
        )
        fp8_accepted_now = [
            item for item in accepted if item[1].precision_profile == "fp8"
        ]
        if missing_fp8 and fp8_accepted_now:
            raise RuntimeError(
                f"Safe FP8 coverage is incomplete for {family.name}: "
                f"{missing_fp8}"
            )
        if missing_fp8:
            missing_bf16 = measure_until_covered(
                bf16_candidates,
                precision_profile="bf16",
                trial_limit=maximum_performance_trials,
            )
            if missing_bf16:
                raise RuntimeError(
                    f"FP8 was unavailable and safe BF16 fallback coverage is "
                    f"incomplete for {family.name}: {missing_bf16}"
                )
    else:
        missing_bf16 = measure_until_covered(
            bf16_candidates,
            precision_profile="bf16",
            trial_limit=maximum_performance_trials,
        )
        if missing_bf16:
            raise RuntimeError(
                f"Safe BF16 performance coverage is incomplete for "
                f"{family.name}: {missing_bf16}"
            )
    if not accepted:
        raise RuntimeError(f"No safe measured performance candidate passed for {family.name}")
    fp8_accepted = [item for item in accepted if item[1].precision_profile == "fp8"]
    if fp8_accepted:
        best_fp8 = max(fp8_accepted, key=lambda item: item[0])
        reference_candidate = Candidate(
            micro_batch_size=best_fp8[1].micro_batch_size,
            grad_accum_steps=best_fp8[1].grad_accum_steps,
            precision_profile="bf16",
            compile_mode=best_fp8[1].compile_mode,
            dispatch_overlap=best_fp8[1].dispatch_overlap,
            ngram_table_mode=best_fp8[1].ngram_table_mode,
        )
        matching = [
            item
            for item in accepted
            if item[1] == reference_candidate
        ]
        if matching:
            bf16_reference = matching[0]
        else:
            report_path = output / "trials" / "bf16-reference.json"
            passed, reasons, reference_throughput, report = evaluate_performance(
                reference_candidate,
                report_path,
                kind="bf16_reference",
            )
            bf16_reference = (
                (reference_throughput, reference_candidate, report)
                if passed
                else None
            )
        if bf16_reference is None:
            # A valid BF16 oracle is a production requirement even when the
            # measured FP8 path itself passed.
            raise RuntimeError(f"BF16 reference lane failed for {family.name}")
        # FP8 is the production numerical contract, not merely a throughput
        # suggestion.  A passing BF16 lane is retained as the numerical oracle
        # and as the explicit fallback only when no FP8 candidate survives the
        # correctness/runtime gates.  Do not silently prefer BF16 just because
        # a short canary happens to time faster.
        pool = [best_fp8]
    else:
        pool = accepted
    throughput, performance_candidate, performance_report = max(pool, key=lambda item: item[0])
    selected_coverage_missing = _missing_performance_coverage(
        required_coverage,
        safe_measurements,
        precision_profile=performance_candidate.precision_profile,
    )
    if selected_coverage_missing:
        raise RuntimeError(
            f"Selected precision lacks complete safe performance coverage: "
            f"{selected_coverage_missing}"
        )
    optimizer_rows: list[
        tuple[float, float, Candidate, dict[str, Any]]
    ] = []
    for index, learning_rate in enumerate(bounds.learning_rates):
        candidate = Candidate(
            **{
                **asdict(performance_candidate),
                "learning_rate": learning_rate,
            }
        )
        report_path = output / "trials" / f"optimizer-{index:03d}.json"
        report = run_trial(candidate, report_path, True)
        passed, reasons, final_loss = validate_optimizer_report(report, bounds=bounds)
        if report.get("precision_role_plan_sha256") != role_plan["plan_sha256"]:
            passed = False
            reasons = [
                *reasons,
                "optimizer canary did not retain the sealed precision role plan",
            ]
        trial_rows.append(
            {
                "kind": "optimizer",
                "candidate": asdict(candidate),
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "passed": passed,
                "reasons": reasons,
                "final_loss": final_loss,
            }
        )
        if passed:
            distance = abs(
                math.log(learning_rate) - math.log(bounds.preferred_learning_rate)
            )
            optimizer_rows.append((distance, final_loss, candidate, report))
    if not optimizer_rows:
        raise RuntimeError(f"No manifest-bounded optimizer canary passed for {family.name}")
    # Prefer the manifest's science-backed LR when stable.  Only then use the
    # short-window loss as a tie breaker; a tiny canary is not an LR study.
    _, _, selected, optimizer_report = min(
        optimizer_rows,
        key=lambda row: (row[0], row[1]),
    )
    global_token_batch = (
        selected.micro_batch_size
        * selected.grad_accum_steps
        * family.world_size
        * sequence_length
    )
    profile: dict[str, Any] = {
        "schema": config.raw["autotune"]["cache_schema"],
        "family": family.name,
        "created_at": utc_now(),
        "inventory_fingerprint": inventory_fingerprint,
        "release_marker_sha256": release_marker["marker_sha256"],
        "manifest_path": str(family.manifest),
        "manifest_sha256": file_sha256(family.manifest),
        "training_contract_sha256": file_sha256(config.training_contract),
        "world_size": family.world_size,
        "expert_parallel_size": family.expert_parallel_size,
        "expert_replicas": family.expert_replicas,
        "precision_role_plan": role_plan,
        "precision_role_plan_sha256": role_plan["plan_sha256"],
        "precision_role_inventory_sha256": role_plan["inventory_sha256"],
        "measured_precision_role_map": measured_role_map,
        "selected": asdict(selected),
        "global_token_batch": global_token_batch,
        "measured_tokens_per_second": throughput,
        "measured_estimated_train_flops": _number(
            performance_report, "estimated_train_flops"
        ),
        "peak_hbm_bytes": int(_number(performance_report, "peak_hbm_bytes")),
        "performance_coverage": {
            "precision_profile": performance_candidate.precision_profile,
            "required": {
                axis: sorted(values, key=str)
                for axis, values in required_coverage.items()
            },
            "missing": selected_coverage_missing,
            "performance_trials_used": performance_trials_used,
            "maximum_performance_trials": maximum_performance_trials,
        },
        "optimizer_canary": {
            key: optimizer_report[key]
            for key in (
                "initial_loss",
                "final_loss",
                "max_grad_norm",
                "nonfinite_steps",
                "update_to_weight_ratio",
            )
        },
        "trials": trial_rows,
    }
    profile["profile_sha256"] = json_sha256(profile)
    atomic_write_json(output / "profile.json", profile)
    return profile


def validate_profile(
    profile_path: str | Path,
    *,
    family: FamilyTopology,
    inventory_fingerprint: str,
    release_marker_sha256: str,
    maximum_age_days: int = 30,
) -> dict[str, Any]:
    profile = read_json(profile_path)
    role_plan = profile.get("precision_role_plan")
    role_plan_valid = bool(
        isinstance(role_plan, dict)
        and role_plan.get("schema") == "metis.precision-role-plan/v1"
        and role_plan.get("family") == family.name
        and role_plan.get("plan_sha256")
        == json_sha256(role_plan, omit=("plan_sha256",))
        and profile.get("precision_role_plan_sha256")
        == role_plan.get("plan_sha256")
        and profile.get("precision_role_inventory_sha256")
        == role_plan.get("inventory_sha256")
        and profile.get("measured_precision_role_map")
        == measured_role_dtype_map(role_plan)
    )
    try:
        created_at = datetime.fromisoformat(str(profile.get("created_at", "")))
        if created_at.tzinfo is None:
            raise ValueError("timezone missing")
        fresh = datetime.now(timezone.utc) - created_at <= timedelta(
            days=maximum_age_days
        )
    except ValueError:
        fresh = False
    if (
        profile.get("schema") != "metis.portage-autotune/v1"
        or profile.get("profile_sha256")
        != json_sha256(profile, omit=("profile_sha256",))
        or profile.get("family") != family.name
        or profile.get("inventory_fingerprint") != inventory_fingerprint
        or profile.get("release_marker_sha256") != release_marker_sha256
        or profile.get("manifest_sha256") != file_sha256(family.manifest)
        or int(profile.get("world_size", -1)) != family.world_size
        or int(profile.get("expert_parallel_size", -1))
        != family.expert_parallel_size
        or int(profile.get("expert_replicas", -1)) != family.expert_replicas
        or not role_plan_valid
        or not fresh
    ):
        raise RuntimeError(f"Autotune profile is stale or invalid: {profile_path}")
    return profile


def inventory_fingerprint(
    *,
    compute_inventory: dict[str, Any],
    git_commit: str,
    config: PortageConfig,
) -> str:
    facts = compute_inventory.get("facts", {})
    torch = facts.get("torch", {})
    return json_sha256(
        {
            "git_commit": git_commit,
            "config_sha256": file_sha256(config.path),
            "gpu_arch": facts.get("gpu_arch"),
            "torch": torch.get("torch"),
            "rocm": torch.get("hip"),
            "device_name": torch.get("device_name"),
            "total_memory": torch.get("total_memory"),
            "loaded_modules": facts.get("loaded_modules"),
            "rccl": facts.get("rccl"),
            "runtime_compute_sha256": facts.get("runtime_compute_sha256"),
            "available_precision_profiles": facts.get(
                "available_precision_profiles"
            ),
        }
    )
