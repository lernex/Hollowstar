from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import repository_root
from .manifest import load_manifest
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
    dry_run: bool = False,
) -> SubmittedJob:
    scheduler = profile["scheduler"]
    stage_config = scheduler.get(stage, scheduler.get("normalize", {}))
    lustre = Path(profile["storage"]["lustre_root"])
    logs = lustre / profile["storage"]["directories"]["logs"] / stage
    logs.mkdir(parents=True, exist_ok=True)
    script = repository_root() / "slurm" / "metis16" / "stage.sbatch"
    command = ["sbatch", "--parsable", f"--job-name=metis16-{stage}"]
    command.append(f"--output={logs}/%A_%a.out")
    command.append(f"--error={logs}/%A_%a.err")
    if indices is not None:
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
    if account:
        command.append(f"--account={account}")
    if partition:
        command.append(f"--partition={partition}")
    if qos:
        command.append(f"--qos={qos}")
    if stage_config.get("time"):
        command.append(f"--time={stage_config['time']}")
    if stage_config.get("cpus_per_task"):
        command.append(f"--cpus-per-task={int(stage_config['cpus_per_task'])}")
    if stage_config.get("memory_gb"):
        command.append(f"--mem={int(stage_config['memory_gb'])}G")
    command.extend(
        [
            f"--export=ALL,METIS_PROFILE={profile_path},METIS_STAGE={stage},METIS_TASK_OFFSET={int(task_offset)}",
            str(script),
        ]
    )
    array = next((arg.split("=", 1)[1] for arg in command if arg.startswith("--array=")), None)
    if dry_run:
        return SubmittedJob(stage, f"dry-{stage}", array, dependency, int(task_offset), tuple(command))
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch is unavailable; use --dry-run outside Portage")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    return SubmittedJob(stage, job_id, array, dependency, int(task_offset), tuple(command))


def _contiguous_chunks(indices: Iterable[int], maximum_array_size: int) -> list[range]:
    """Map arbitrary global task indices onto Slurm-safe local arrays.

    Slurm sites commonly cap both the number of array entries and the largest
    array task ID.  Every returned range is contiguous and no larger than the
    configured cap, so it can be submitted as ``0..N-1`` with the range start
    exported as ``METIS_TASK_OFFSET``.
    """

    if maximum_array_size <= 0:
        raise ValueError("scheduler.max_array_size must be a positive integer")
    values = sorted(set(int(index) for index in indices))
    if not values:
        return []
    chunks: list[range] = []
    run_start = run_previous = values[0]
    for value in values[1:] + [values[-1] + 2]:
        if value == run_previous + 1 and value - run_start < maximum_array_size:
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
    chunks = _contiguous_chunks(global_indices, maximum_array_size)
    jobs: list[SubmittedJob] = []
    for chunk_number, chunk in enumerate(chunks):
        # Local array IDs always start at zero. The batch wrapper adds the
        # immutable global offset before dispatching the Python stage.
        job = submit_stage(
            stage=stage,
            profile_path=profile_path,
            profile=profile,
            indices=range(len(chunk)),
            dependency=dependency,
            task_offset=chunk.start,
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
    ("tokenizer_sample", None),
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
        target = int(load_manifest(manifest_path)["schedule"]["target_tokens"])
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
            "pack_tasks": (target + shard_tokens - 1) // shard_tokens,
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
