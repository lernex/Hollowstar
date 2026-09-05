from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import atomic_json, digest_json, read_receipt, utc_now, write_receipt


def idle_nodes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["sinfo", "-N", "-h", "--partition=parry", "-t", "idle", "-o", "%N|%c|%m"],
        text=True,
    )
    nodes = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        name, cpus, memory = line.split("|")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise RuntimeError("Slurm returned an invalid node name")
        nodes[name] = {"name": name, "cpus": int(cpus), "memory_mb": int(memory)}
    return [nodes[name] for name in sorted(nodes)]


def _submit_workers(
    root: Path,
    checkout: Path,
    python: Path,
    *,
    maximum_nodes: int = 4,
    workers_per_node: int = 32,
    stage: str = "prep",
    defer_compaction: bool = False,
) -> dict[str, Any]:
    if maximum_nodes < 1 or workers_per_node < 1:
        raise ValueError("Positive worker/node limits required")
    if type(defer_compaction) is not bool:
        raise ValueError("defer_compaction must be a boolean")
    if not python.is_file():
        raise FileNotFoundError("The pinned data interpreter is unavailable")
    scripts = {"prep": ("prepare.sbatch", "12:00:00"), "tokenizer": ("tokenizer.sbatch", "2-00:00:00")}
    if stage not in scripts:
        raise ValueError("Unsupported Metis-1.7 worker stage")
    script_name, time_limit = scripts[stage]
    script = checkout / "slurm" / "metis17" / script_name
    if not script.is_file():
        raise FileNotFoundError("The committed 1.7 Slurm wrapper is unavailable")
    registry_path = root / "state" / f"{stage}-jobs.json"
    previous = read_receipt(registry_path) if registry_path.exists() else {"jobs": []}
    live_ids: set[str] = set()
    if previous["jobs"]:
        output = subprocess.check_output(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%A"],
            text=True,
        )
        live_ids = set(output.split())
    active = [j for j in previous["jobs"] if str(j["job_id"]) in live_ids]
    needed = max(0, maximum_nodes - len(active))
    free = idle_nodes()[:needed] if needed else []
    if not free and not active:
        return {"active_jobs": [], "registered_jobs": previous["jobs"], "status": "waiting_for_idle_nodes"}
    logs = root / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    recent = [
        job for job in previous["jobs"]
        if job.get("code_commit") == commit and job.get("python") == str(python)
        and datetime.fromisoformat(job["submitted_at"]) >= cutoff
    ]
    if free and len(recent) >= 3 * maximum_nodes:
        raise RuntimeError("Repeated short-lived prep launches; inspect Slurm logs before submitting more")
    for value in (root, checkout, python):
        if any(char in str(value) for char in ",\n\r"):
            raise ValueError("Slurm environment paths cannot contain commas or newlines")
    for node in free:
        memory_mb = min(480_000, node["memory_mb"] - 16_000)
        if memory_mb < 32_000:
            continue
        # Idle snapshots bound admission, not placement: a pinned node can
        # become busy while another suitable node stays idle.
        command = [
            "sbatch", "--parsable", f"--job-name=metis17-{stage}", "--partition=parry",
            "--nodes=1", "--ntasks=1", "--exclusive",
            f"--cpus-per-task={node['cpus']}", f"--mem={memory_mb}M",
            f"--time={time_limit}", "--nice=10000", "--requeue",
            f"--output={logs}/%j.out", f"--error={logs}/%j.err",
            f"--chdir={checkout}",
            f"--export=METIS17_ROOT={root},METIS17_CODE={checkout},METIS17_PYTHON={python},"
            f"METIS17_WORKERS={workers_per_node},METIS17_DEFER_COMPACTION={int(defer_compaction)}",
            str(script),
        ]
        output = subprocess.check_output(command, text=True).strip()
        job_id = output.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unrecognized sbatch response: {output}")
        job = {
            "job_id": job_id,
            "node": "scheduler-selected",
            "code_commit": commit,
            "python": str(python),
            "submitted_at": utc_now(),
            "workers": workers_per_node,
            "defer_compaction": defer_compaction,
        }
        previous["jobs"].append(job)
        active.append(job)
        write_receipt(registry_path, previous)
    return {"status": "supervising", "active_jobs": active, "registered_jobs": previous["jobs"]}


def submit_prep_workers(
    root: Path,
    checkout: Path,
    python: Path,
    *,
    maximum_nodes: int = 4,
    workers_per_node: int = 32,
    defer_compaction: bool = False,
) -> dict[str, Any]:
    from .worker import claim

    lock = claim(root / "locks" / "prep-submission.flock")
    if lock is None:
        raise RuntimeError("Another preparation submission is already in progress")
    try:
        return _submit_workers(
            root, checkout, python,
            maximum_nodes=maximum_nodes, workers_per_node=workers_per_node,
            defer_compaction=defer_compaction,
        )
    finally:
        lock.close()


def submit_tokenizer_worker(
    root: Path, checkout: Path, python: Path, *, workers: int = 64,
) -> dict[str, Any]:
    from .worker import claim

    lock = claim(root / "locks" / "tokenizer-submission.flock")
    if lock is None:
        raise RuntimeError("Another tokenizer submission is already in progress")
    try:
        return _submit_workers(
            root, checkout, python, maximum_nodes=1, workers_per_node=workers, stage="tokenizer",
        )
    finally:
        lock.close()


def supervise_prep(
    root: Path, checkout: Path, python: Path, *,
    maximum_nodes: int = 4, workers_per_node: int = 32, tokenizer: bool = False,
    defer_compaction: bool = False,
) -> None:
    from .cli import _stop_event, safe_error
    from .worker import EventTail, claim, failure_blocks, observe_failure, worker_configuration

    if maximum_nodes < 1 or workers_per_node < 1:
        raise ValueError("Positive node/worker counts required")
    lock = claim(root / "locks" / "prep-supervisor.flock")
    if lock is None:
        raise RuntimeError("A preparation supervisor already owns this release")
    status_path = root / "status" / "prep-supervisor.json"
    try:
        config, _ = worker_configuration(root)
        generation = config["generation"]
        index_generation = config["index_generation"]
        stop = _stop_event()
        tail = EventTail()
        acquired: set[str] = set()
        completed: set[str] = set()
        failures: dict[str, dict[str, Any]] = {}
        while not stop.is_set() and not (root / "STOP").exists():
            for stream, seen in (("raw", acquired), ("prepared", completed)):
                for path in sorted((root / "events" / stream).glob("*.jsonl")):
                    for event in tail.read(path):
                        if stream == "raw" or event.get("index_generation") == index_generation:
                            seen.add(event["object_id"])
            for path in sorted((root / "events" / "prep-errors").glob("*.jsonl")):
                for event in tail.read(path):
                    if event.get("index_generation") == index_generation:
                        observe_failure(failures, event)
            limits_sha256 = digest_json(read_receipt(root / "limits.json"))
            failed = {object_id for object_id, value in failures.items() if failure_blocks(value, limits_sha256)}
            pending = acquired - completed - failed
            tokenizer_worker = submit_tokenizer_worker(root, checkout, python) if tokenizer else None
            if pending:
                result = submit_prep_workers(
                    root, checkout, python, maximum_nodes=min(maximum_nodes, len(pending)),
                    workers_per_node=workers_per_node,
                    defer_compaction=defer_compaction,
                )
            else:
                result = {"status": "waiting_for_new_raw_objects"}
            if tokenizer:
                result["tokenizer_worker"] = tokenizer_worker
            atomic_json(status_path, {
                "schema": "metis17.prep-supervisor/v1",
                "host": socket.gethostname(), "pid": os.getpid(), "generation": generation,
                "index_generation": index_generation,
                "updated_at": utc_now(), "acquired_objects": len(acquired),
                "completed_objects": len(completed), "failed_objects": len(failed),
                "pending_objects": len(pending), "maximum_nodes": maximum_nodes, **result,
            })
            stop.wait(60)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        atomic_json(status_path, {
            "schema": "metis17.prep-supervisor/v1", "status": "failed",
            "host": socket.gethostname(), "pid": os.getpid(),
            "updated_at": utc_now(), **safe_error(exc),
        })
        raise
    finally:
        lock.close()
