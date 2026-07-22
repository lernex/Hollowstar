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


@dataclass(frozen=True)
class SubmittedJob:
    stage: str
    job_id: str
    array: str | None
    dependency: str | None
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
            f"--export=ALL,METIS_PROFILE={profile_path},METIS_STAGE={stage}",
            str(script),
        ]
    )
    array = next((arg.split("=", 1)[1] for arg in command if arg.startswith("--array=")), None)
    if dry_run:
        return SubmittedJob(stage, f"dry-{stage}", array, dependency, tuple(command))
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch is unavailable; use --dry-run outside Portage")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    return SubmittedJob(stage, job_id, array, dependency, tuple(command))


BUILD_GRAPH = (
    ("normalize", "download_tasks"),
    ("exact_signature", "normalize_tasks"),
    ("exact_find", "exact_find_tasks"),
    ("exact_filter", "normalize_tasks"),
    ("minhash_signature", "normalize_tasks"),
    ("minhash_buckets", "minhash_buckets"),
    ("minhash_cluster", None),
    ("minhash_filter", "normalize_tasks"),
    ("decontam_index", None),
    ("decontam_filter", "normalize_tasks"),
    ("tokenizer_sample", None),
    ("tokenizer_train", None),
    ("token_count", "normalize_tasks"),
    ("select", None),
    ("pack", "pack_tasks"),
    ("verify", None),
    ("release", None),
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
            job = submit_stage(
                stage="download",
                profile_path=profile_path,
                profile=profile,
                indices=incomplete,
                dry_run=dry_run,
            )
            jobs.append(job)
            dependency = job.job_id
    if include_build:
        normalize_tasks = max(1, download_count)
        exact_find_tasks = int(profile["scheduler"].get("exact_dedup", {}).get("find_tasks", 256))
        minhash_buckets = int(profile["scheduler"].get("minhash", {}).get("num_buckets", 20))
        manifest_path = Path(profile["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = repository_root() / manifest_path
        target = int(load_manifest(manifest_path)["schedule"]["target_tokens"])
        shard_tokens = int(profile["storage"].get("final_shard_tokens", 1_000_000_000))
        counts = {
            "download_tasks": download_count,
            "normalize_tasks": normalize_tasks,
            "exact_find_tasks": exact_find_tasks,
            "minhash_buckets": minhash_buckets,
            "pack_tasks": (target + shard_tokens - 1) // shard_tokens,
        }
        for stage, count_key in BUILD_GRAPH:
            indices = range(counts[count_key]) if count_key else None
            job = submit_stage(
                stage=stage,
                profile_path=profile_path,
                profile=profile,
                indices=indices,
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
