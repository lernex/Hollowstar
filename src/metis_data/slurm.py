from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import repository_root
from .manifest import PHASES, load_manifest
from .selection import schedule_shard_count
from .state import StateStore, utc_now
from .build_inputs import prepare_build_inputs


@dataclass(frozen=True)
class SubmittedJob:
    stage: str
    job_id: str
    array: str | None
    dependency: str | None
    task_offset: int
    command: tuple[str, ...]
    tasks_per_job: int = 1


def stage_tasks_per_job(profile: dict[str, Any], stage: str) -> int:
    """Number of global task indices one allocation runs concurrently.

    On a partition that hands out whole nodes, one Slurm task per unit of work
    would leave almost every core of a 96-core node idle, because each stage
    task is a single-threaded process. Grouping also keeps the array well under
    the site ``MaxArraySize`` and cuts scheduler and Lustre metadata pressure.
    """

    scheduler = profile.get("scheduler", {})
    stage_config = scheduler.get(stage, {})
    value = stage_config.get("tasks_per_job", scheduler.get("default_tasks_per_job", 1))
    tasks_per_job = int(value or 1)
    if tasks_per_job < 1:
        raise ValueError(f"scheduler.{stage}.tasks_per_job must be at least 1")
    return tasks_per_job


def _auto_scheduler_value(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return None if not normalized or normalized.lower() == "auto" else normalized


def _indices_expression(
    indices: Iterable[int],
    maximum_concurrent: int | None = None,
    maximum_array_size: int | None = None,
) -> str:
    if isinstance(indices, range) and indices.step == 1:
        if not indices:
            raise ValueError("Cannot submit an empty Slurm array")
        count = len(indices)
        first = indices.start
        last = indices.stop - 1
        if maximum_array_size and (count > maximum_array_size or last >= maximum_array_size):
            raise ValueError(
                f"Slurm array has {count:,} entries through index {last:,}, exceeding configured "
                f"maximum_array_size={maximum_array_size:,}"
            )
        expression = str(first) if first == last else f"{first}-{last}"
        if maximum_concurrent:
            expression += f"%{maximum_concurrent}"
        return expression
    values = sorted(set(int(index) for index in indices))
    if not values:
        raise ValueError("Cannot submit an empty Slurm array")
    if maximum_array_size and (len(values) > maximum_array_size or values[-1] >= maximum_array_size):
        raise ValueError(
            f"Slurm array has {len(values):,} entries through index {values[-1]:,}, exceeding configured "
            f"maximum_array_size={maximum_array_size:,}"
        )
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    expression = ",".join(ranges)
    if maximum_concurrent:
        expression += f"%{maximum_concurrent}"
    return expression


def submit_stage(
    *,
    stage: str,
    profile_path: Path,
    profile: dict[str, Any],
    indices: Iterable[int] | None = None,
    dependency: str | None = None,
    task_offset: int = 0,
    tasks_per_job: int = 1,
    task_limit: int = 0,
    task_stride: int = 0,
    dry_run: bool = False,
) -> SubmittedJob:
    scheduler = profile["scheduler"]
    stage_config = scheduler.get(stage, scheduler.get("normalize", {}))
    lustre = Path(profile["storage"]["lustre_root"])
    logs = lustre / profile["storage"]["directories"]["logs"] / stage
    logs.mkdir(parents=True, exist_ok=True)
    script = repository_root() / "slurm" / "metis16" / "stage.sbatch"
    command = ["sbatch", "--parsable", f"--job-name=metis16-{stage}"]
    # Every stage is idempotent: each task writes a completion marker and a
    # rerun skips finished work. So letting Slurm reschedule a task it killed
    # for reasons of its own -- a node draining under the allocation, a node
    # failure -- costs a repeat of at most one task. Without this, one evicted
    # task fails the array, afterok holds all 51 downstream jobs, and the
    # operator has to notice and resubmit by hand.
    command.append("--requeue")
    command.append(f"--output={logs}/%A_%a.out")
    command.append(f"--error={logs}/%A_%a.err")
    if indices is not None:
        # With grouping enabled, max_concurrent throttles allocations (nodes),
        # not individual work units.
        maximum = int(stage_config.get("max_concurrent", stage_config.get("workers", 0)) or 0)
        maximum_array_size = int(scheduler.get("max_array_size", 1000))
        command.append(
            f"--array={_indices_expression(indices, maximum or None, maximum_array_size)}"
        )
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    account = _auto_scheduler_value(scheduler.get("account"))
    partition = _auto_scheduler_value(scheduler.get("partition"))
    qos = _auto_scheduler_value(scheduler.get("qos"))
    reservation = _auto_scheduler_value(scheduler.get("reservation"))
    if account:
        command.append(f"--account={account}")
    if partition:
        command.append(f"--partition={partition}")
    if qos:
        command.append(f"--qos={qos}")
    if reservation:
        command.append(f"--reservation={reservation}")
    if stage_config.get("time"):
        command.append(f"--time={stage_config['time']}")
    if scheduler.get("exclusive_nodes"):
        # A whole-node partition cannot pack several allocations onto one node,
        # so the allocation asks for the entire node and the stage runner fans
        # out across its cores itself.
        command.append("--nodes=1")
        command.append("--exclusive")
        command.append("--mem=0")
    else:
        if stage_config.get("cpus_per_task"):
            command.append(f"--cpus-per-task={int(stage_config['cpus_per_task'])}")
        if stage_config.get("memory_gb"):
            command.append(f"--mem={int(stage_config['memory_gb'])}G")
    for option in scheduler.get("extra_sbatch_options", []):
        command.append(str(option))
    command.extend(
        [
            # METIS_ROOT is not a convenience: Slurm executes a staged copy of
            # the batch script out of /var/spool, so the script cannot find the
            # checkout -- and therefore neither the runtime nor src -- on its own.
            #
            # METIS_PYTHON pins the stages to this exact interpreter. The runtime
            # is not always inside the checkout (metisctl falls back to
            # ~/.cache/metis/runtime-<host>), and a wrapper that re-derives it
            # from a default path can disagree with the operator's about which
            # hash-locked environment is in use. Whatever ran the submission runs
            # the stages.
            f"--export=ALL,METIS_ROOT={repository_root()}"
            f",METIS_PYTHON={sys.executable}"
            f",METIS_PROFILE={profile_path},METIS_STAGE={stage}"
            f",METIS_TASK_OFFSET={int(task_offset)}"
            f",METIS_TASKS_PER_JOB={int(tasks_per_job)}"
            f",METIS_TASK_LIMIT={int(task_limit)}"
            f",METIS_TASK_STRIDE={int(task_stride)}",
            str(script),
        ]
    )
    array = next((arg.split("=", 1)[1] for arg in command if arg.startswith("--array=")), None)
    if dry_run:
        return SubmittedJob(
            stage, f"dry-{stage}", array, dependency, int(task_offset), tuple(command), int(tasks_per_job)
        )
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch is unavailable; use --dry-run outside the compute cluster")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    return SubmittedJob(
        stage, job_id, array, dependency, int(task_offset), tuple(command), int(tasks_per_job)
    )


def _contiguous_chunks(
    indices: Iterable[int], maximum_array_size: int, tasks_per_job: int = 1
) -> list[range]:
    """Map arbitrary global task indices onto Slurm-safe local arrays.

    Slurm sites commonly cap both the number of array entries and the largest
    array task ID.  Every returned range is contiguous, and one array entry
    covers ``tasks_per_job`` consecutive global indices, so a run is capped at
    ``maximum_array_size * tasks_per_job`` indices. The range start is exported
    as ``METIS_TASK_OFFSET`` and the range end as ``METIS_TASK_LIMIT``, which is
    what keeps a trailing partial group from reaching into the next chunk.
    """

    if maximum_array_size <= 0:
        raise ValueError("scheduler.max_array_size must be a positive integer")
    if tasks_per_job <= 0:
        raise ValueError("tasks_per_job must be a positive integer")
    span = maximum_array_size * tasks_per_job
    values = sorted(set(int(index) for index in indices))
    if not values:
        return []
    chunks: list[range] = []
    run_start = run_previous = values[0]
    for value in values[1:] + [values[-1] + 2]:
        if value == run_previous + 1 and value - run_start < span:
            run_previous = value
            continue
        chunks.append(range(run_start, run_previous + 1))
        run_start = run_previous = value
    return chunks


def _submit_array_chunks(
    *,
    stage: str,
    global_indices: Iterable[int],
    profile_path: Path,
    profile: dict[str, Any],
    dependency: str | None,
    dry_run: bool,
) -> list[SubmittedJob]:
    maximum_array_size = int(profile["scheduler"].get("max_array_size", 1000))
    tasks_per_job = stage_tasks_per_job(profile, stage)
    # Shard size correlates with shard index, so a contiguous block can hand one
    # array entry an entire run of oversized shards. Striding deals them
    # round-robin instead. Both layouts cover the chunk exactly once.
    stride_enabled = bool(profile["scheduler"].get("stride_task_assignment", True))
    chunks = _contiguous_chunks(global_indices, maximum_array_size, tasks_per_job)
    jobs: list[SubmittedJob] = []
    for chunk_number, chunk in enumerate(chunks):
        # Local array IDs always start at zero. The batch wrapper adds the
        # immutable global offset and the group stride before dispatching the
        # Python stage.
        entries = -(-len(chunk) // tasks_per_job)
        job = submit_stage(
            stage=stage,
            profile_path=profile_path,
            profile=profile,
            indices=range(entries),
            dependency=dependency,
            task_offset=chunk.start,
            tasks_per_job=tasks_per_job,
            task_limit=chunk.stop,
            task_stride=entries if stride_enabled else 0,
            dry_run=dry_run,
        )
        if dry_run and len(chunks) > 1:
            job = SubmittedJob(
                stage=job.stage,
                job_id=f"{job.job_id}-chunk-{chunk_number:04d}",
                array=job.array,
                dependency=job.dependency,
                task_offset=job.task_offset,
                command=job.command,
                tasks_per_job=job.tasks_per_job,
            )
        jobs.append(job)
    return jobs


BUILD_GRAPH = (
    ("normalize", "normalize_tasks"),
    ("cleanup_raw", None),
    ("exact_signature", "normalize_tasks"),
    ("exact_find", "exact_find_tasks"),
    ("exact_filter", "normalize_tasks"),
    ("cleanup_exact", None),
    ("span_prefilter_signature", "normalize_tasks"),
    ("span_prefilter_find", "span_buckets"),
    ("span_signature", "normalize_tasks"),
    ("span_find", "span_buckets"),
    ("span_filter", "normalize_tasks"),
    ("cleanup_span", None),
    ("minhash_signature", "normalize_tasks"),
    ("minhash_buckets", "minhash_buckets"),
    ("minhash_components", None),
    ("minhash_priority_candidates", "normalize_tasks"),
    ("minhash_priority_resolve", "minhash_priority_buckets"),
    ("minhash_priority_finalize", "normalize_tasks"),
    ("minhash_priority_verify", None),
    ("minhash_filter", "normalize_tasks"),
    ("cleanup_minhash", None),
    ("code_signature", "normalize_tasks"),
    ("code_find", "code_buckets"),
    ("code_filter", "normalize_tasks"),
    ("cleanup_code", None),
    ("decontam_index", None),
    ("decontam_filter", "normalize_tasks"),
    ("cleanup_decontam", None),
    ("final_hash_signature", "normalize_tasks"),
    ("final_hash_find", "final_hash_buckets"),
    ("final_hash_filter", "normalize_tasks"),
    ("cleanup_final_hash", None),
    # The tokenizer sample is a stratified read of the whole eligible corpus.
    # Scan counts availability per shard, plan apportions the exact per-source
    # byte targets across shards, and the write pass emits only its own slice.
    ("tokenizer_sample_scan", "normalize_tasks"),
    ("tokenizer_sample_plan", None),
    ("tokenizer_sample", "normalize_tasks"),
    ("tokenizer_train", None),
    ("cleanup_tokenizer_sample", None),
    ("token_count", "normalize_tasks"),
    ("context_select", None),
    ("context_prepare", None),
    ("context_pack", "context_pack_tasks"),
    ("context_verify", None),
    ("select", None),
    ("cleanup_selection_inputs", None),
    ("pack", "pack_tasks"),
    # Per-shard re-hash and index audit of the 1.82TiB release, parallel by
    # shard; `verify` then aggregates the receipts and re-checks anything
    # missing or produced under a different selection policy.
    ("verify_shard", "pack_tasks"),
    ("verify", None),
    ("cleanup_pack_inputs", None),
    ("release", None),
    ("cleanup_release_workspace", None),
)


def submit_graph(
    *,
    profile_path: Path,
    profile: dict[str, Any],
    state: StateStore,
    include_download: bool,
    include_build: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_lock = state.read("sources.lock.json")
    if source_lock is None:
        raise RuntimeError("sources.lock.json is missing; resolve sources before submission")
    jobs: list[SubmittedJob] = []
    dependency: str | None = None
    download_count = len(source_lock["download_tasks"])
    if include_download:
        incomplete = [index for index, task in enumerate(source_lock["download_tasks"]) if not state.is_complete("download", task["task_id"])]
        if incomplete:
            stage_jobs = _submit_array_chunks(
                stage="download",
                profile_path=profile_path,
                profile=profile,
                global_indices=incomplete,
                dependency=dependency,
                dry_run=dry_run,
            )
            jobs.extend(stage_jobs)
            dependency = ":".join(job.job_id for job in stage_jobs)
    if include_build:
        build_inputs = prepare_build_inputs(profile, state)
        if profile.get("gates", {}).get("require_deep_handoff_verification", False):
            from .handoff_verification import handoff_verification_plan

            verification_plan = handoff_verification_plan(profile, state)
            if not verification_plan["complete"]:
                missing_indices = verification_plan["missing_indices"]
                if missing_indices:
                    stage_jobs = _submit_array_chunks(
                        stage="handoff_signature",
                        global_indices=missing_indices,
                        profile_path=profile_path,
                        profile=profile,
                        dependency=dependency,
                        dry_run=dry_run,
                    )
                    jobs.extend(stage_jobs)
                    dependency = ":".join(job.job_id for job in stage_jobs)
                reducer = submit_stage(
                    stage="handoff_verify",
                    profile_path=profile_path,
                    profile=profile,
                    dependency=dependency,
                    dry_run=dry_run,
                )
                jobs.append(reducer)
                dependency = reducer.job_id
        normalize_tasks = int(build_inputs["input_count"])
        exact_find_tasks = int(profile["scheduler"].get("exact_dedup", {}).get("find_tasks", 256))
        span_buckets = int(profile["scheduler"].get("repeated_span", {}).get("finder_tasks", 64))
        minhash_buckets = int(profile["scheduler"].get("minhash", {}).get("num_buckets", 20))
        minhash_priority_buckets = int(
            profile["scheduler"].get("minhash_priority", {}).get("bucket_count", 256)
        )
        code_buckets = int(profile["scheduler"].get("code_structural", {}).get("finder_tasks", 64))
        final_hash_buckets = int(profile["scheduler"].get("final_hash", {}).get("finder_tasks", 64))
        manifest_path = Path(profile["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = repository_root() / manifest_path
        schedule = load_manifest(manifest_path)["schedule"]
        shard_tokens = int(profile["storage"].get("final_shard_tokens", 1_000_000_000))
        counts = {
            "normalize_tasks": normalize_tasks,
            "exact_find_tasks": exact_find_tasks,
            "span_buckets": span_buckets,
            "minhash_buckets": minhash_buckets,
            "minhash_priority_buckets": minhash_priority_buckets,
            "code_buckets": code_buckets,
            "final_hash_buckets": final_hash_buckets,
            "context_pack_tasks": 96,
            "pack_tasks": schedule_shard_count(schedule, shard_tokens),
        }
        for stage, count_key in BUILD_GRAPH:
            if count_key:
                incomplete = [
                    index
                    for index in range(counts[count_key])
                    if not state.is_complete(stage, f"task-{index:06d}")
                ]
                if not incomplete:
                    continue
                stage_jobs = _submit_array_chunks(
                    stage=stage,
                    global_indices=incomplete,
                    profile_path=profile_path,
                    profile=profile,
                    dependency=dependency,
                    dry_run=dry_run,
                )
                jobs.extend(stage_jobs)
                dependency = ":".join(job.job_id for job in stage_jobs)
            else:
                if state.is_complete(stage, "task-000000"):
                    continue
                job = submit_stage(
                    stage=stage,
                    profile_path=profile_path,
                    profile=profile,
                    dependency=dependency,
                    dry_run=dry_run,
                )
                jobs.append(job)
                dependency = job.job_id
    payload = {
        "submitted_at": utc_now(),
        "dry_run": dry_run,
        "jobs": [job.__dict__ for job in jobs],
    }
    state.write("submissions", f"{payload['submitted_at'].replace(':', '-')}.json", payload=payload)
    return payload
