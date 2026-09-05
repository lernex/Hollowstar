"""Operational rank identity and bounded, read-only failure snapshots.

Run this file directly with the system Python from a watchdog. Using ``-m``
would import the training package and unnecessarily initialize its dependencies.
None of this metadata participates in scientific campaign or resume identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Mapping
import uuid


STARTUP_SCHEMA = "more.rank-startup/v1"
SNAPSHOT_SCHEMA = "more.failure-snapshot/v1"
_JOB_ID = re.compile(r"[0-9]+(?:[_+][0-9]+)?")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RANK_FILE = re.compile(r"rank-([0-9]+)\.jsonl")
_FIELD = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=")
_JOB_FIELDS = {
    "JobId", "JobName", "JobState", "Reason", "ExitCode", "StartTime",
    "EndTime", "RunTime", "NodeList", "BatchHost", "NumNodes", "NumCPUs",
    "NumTasks", "Restarts", "Requeue",
}
_NODE_FIELDS = {
    "NodeName", "NodeHostName", "State", "Reason", "BootTime",
    "SlurmdStartTime", "LastBusyTime", "CPULoad", "FreeMem", "RealMemory",
    "AllocMem", "CPUAlloc",
}


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _job_tag(job_id: str | None) -> str:
    if job_id is None:
        return "local"
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("Expected a numeric Slurm job ID, optionally with an array/heterogeneous suffix")
    return job_id


def boot_identity(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    result: dict[str, Any] = {"boot_id": None, "boot_time_unix": None, "errors": []}
    try:
        raw = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
        result["boot_id"] = str(uuid.UUID(raw))
    except (OSError, ValueError) as error:
        result["errors"].append(f"boot_id: {type(error).__name__}")
    try:
        with (proc_root / "stat").open() as handle:
            text = handle.read(256 * 1024)
        for line in text.splitlines():
            if line.startswith("btime "):
                result["boot_time_unix"] = int(line.split()[1])
                break
        if result["boot_time_unix"] is None:
            result["errors"].append("boot_time: missing btime")
    except (OSError, ValueError, IndexError) as error:
        result["errors"].append(f"boot_time: {type(error).__name__}")
    return result


def record_rank_startup(
    run_root: Path, *, rank: int, world_size: int, local_rank: int,
    device: str, environment: Mapping[str, str] | None = None,
    proc_root: Path = Path("/proc"),
) -> Path:
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid runtime rank identity")
    env = os.environ if environment is None else environment
    job_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
    tag = _job_tag(job_id)
    started_ns = time.time_ns()
    payload = {
        "schema": STARTUP_SCHEMA,
        "job_id": job_id, "slurm_step_id": env.get("SLURM_STEP_ID"),
        "rank": rank, "world_size": world_size, "local_rank": local_rank,
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "device": device, "boot_identity": boot_identity(proc_root),
        "started_unix_ns": started_ns, "started_utc": _utc(started_ns / 1e9),
        "stage": "after_runtime_initialization_and_row_lease_before_model",
        "scientific_identity_member": False,
    }
    directory = Path(run_root) / "operational" / "startups"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"job-{tag}-rank-{rank:05d}-{uuid.uuid4().hex}.json"
    staging = target.with_suffix(".partial")
    with staging.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # A hard link publishes the complete record without replacing an older one.
    os.link(staging, target)
    staging.unlink()
    return target


def last_rank_progress(path: Path, *, tail_bytes: int = 65536) -> dict[str, Any]:
    if not 1024 <= tail_bytes <= 1024 * 1024:
        raise ValueError("tail_bytes must lie in [1024, 1048576]")
    result: dict[str, Any] = {"path": str(path), "status": "unavailable"}
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            offset = max(0, stat.st_size - tail_bytes)
            handle.seek(offset)
            data = handle.read(tail_bytes)
        result.update({"bytes": stat.st_size, "mtime_utc": _utc(stat.st_mtime)})
        lines = data.splitlines(keepends=True)
        if offset and lines:
            lines = lines[1:]
        for line in reversed(lines):
            if not line.endswith(b"\n"):
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict) or type(record.get("step")) is not int:
                continue
            result.update({"status": "ok", "step": record["step"]})
            for field in ("recorded_unix", "cumulative_tokens", "step_time_s"):
                value = record.get(field)
                if type(value) in (int, float) and math.isfinite(value):
                    result[field] = value
            if "recorded_unix" in result:
                try:
                    result["recorded_utc"] = _utc(result["recorded_unix"])
                except (OverflowError, OSError, ValueError):
                    result["timestamp_error"] = "recorded_unix is outside the supported date range"
            return result
        result["status"] = "no_complete_record_in_bounded_tail"
    except OSError as error:
        result["error"] = type(error).__name__
    return result


def parse_slurm_records(text: str, allowed_fields: set[str]) -> list[dict[str, str]]:
    records = []
    for line in text.splitlines():
        matches = list(_FIELD.finditer(line))
        record = {}
        for index, match in enumerate(matches):
            key = match.group(1)
            if key not in allowed_fields:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            record[key] = line[match.end():end].strip()
        if record:
            records.append(record)
    return records


def scheduler_snapshot(
    job_id: str | None, hostnames: list[str], *, timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    _job_tag(job_id)
    if not 0 < timeout_seconds <= 30 or not math.isfinite(timeout_seconds):
        raise ValueError("Scheduler timeout must lie in (0, 30] seconds")
    result: dict[str, Any] = {"timezone": "UTC", "job": [], "nodes": [], "errors": []}

    def query(arguments: list[str], allowed: set[str]) -> list[dict[str, str]]:
        try:
            command = subprocess.run(
                ["scontrol", *arguments], capture_output=True, text=True,
                timeout=timeout_seconds, check=False, env={**os.environ, "TZ": "UTC"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            result["errors"].append({"query": arguments[:2], "error": type(error).__name__})
            return []
        if command.returncode:
            result["errors"].append({
                "query": arguments[:2], "returncode": command.returncode,
                "stderr": command.stderr[:512],
            })
            return []
        return parse_slurm_records(command.stdout, allowed)

    if job_id is not None:
        result["job"] = query(["show", "job", job_id, "-o"], _JOB_FIELDS)
    names = sorted({name.split(".")[0] for name in hostnames if _HOST.fullmatch(name)})
    if names:
        node_list = ",".join(names)
    else:
        node_list = next((record.get("NodeList", "") for record in result["job"]), "")
        if not re.fullmatch(r"[A-Za-z0-9_,.\[\]-]+", node_list):
            node_list = ""
    if node_list:
        result["nodes"] = query(["show", "node", node_list, "-o"], _NODE_FIELDS)
    return result


def _scheduler_boot_time(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def failure_snapshot(
    run_root: Path, *, job_id: str | None, include_scheduler: bool = True,
    scheduler_timeout: float = 5.0, tail_bytes: int = 65536,
) -> dict[str, Any]:
    tag = _job_tag(job_id)
    root = Path(run_root).expanduser().resolve()
    captured = time.time()
    errors = []
    startups: dict[int, dict[str, Any]] = {}
    directory = root / "operational" / "startups"
    try:
        paths = sorted(directory.glob(f"job-{tag}-rank-*.json"))
        for path in paths:
            try:
                with path.open() as handle:
                    record = json.loads(handle.read(16384))
                if (
                    record.get("schema") != STARTUP_SCHEMA or record.get("job_id") != job_id
                    or type(record.get("rank")) is not int
                    or type(record.get("started_unix_ns")) is not int
                    or type(record.get("world_size")) is not int
                    or not 0 <= record["rank"] < record["world_size"] <= 65536
                    or not isinstance(record.get("hostname"), str)
                    or not _HOST.fullmatch(record["hostname"])
                    or not isinstance(record.get("boot_identity"), dict)
                ):
                    raise ValueError("Invalid startup record")
                rank = record["rank"]
                if rank not in startups or record["started_unix_ns"] > startups[rank]["started_unix_ns"]:
                    startups[rank] = record
            except (OSError, ValueError, AttributeError) as error:
                errors.append({"path": str(path), "error": type(error).__name__})
    except OSError as error:
        errors.append({"path": str(directory), "error": type(error).__name__})
    progress = {}
    try:
        for path in sorted((root / "telemetry").glob("rank-*.jsonl")):
            matched = _RANK_FILE.fullmatch(path.name)
            if matched:
                progress[int(matched.group(1))] = last_rank_progress(path, tail_bytes=tail_bytes)
    except OSError as error:
        errors.append({"path": str(root / "telemetry"), "error": type(error).__name__})
    expected = {record.get("world_size") for record in startups.values()}
    expected = {value for value in expected if type(value) is int and 0 < value <= 65536}
    expected_world = max(expected) if expected else None
    ranks = set(startups) | set(progress)
    if expected_world is not None:
        ranks.update(range(expected_world))
    hosts = [record["hostname"] for record in startups.values() if isinstance(record.get("hostname"), str)]
    scheduler = scheduler_snapshot(job_id, hosts, timeout_seconds=scheduler_timeout) if include_scheduler else {
        "timezone": "UTC", "job": [], "nodes": [], "errors": [], "disabled": True,
    }
    nodes = {node["NodeName"]: node for node in scheduler["nodes"] if "NodeName" in node}
    summaries = []
    for rank in sorted(ranks):
        startup = startups.get(rank)
        item: dict[str, Any] = {
            "rank": rank, "startup": startup,
            "last_progress": progress.get(rank, {"status": "missing"}),
            "node_reboot_since_startup": None,
        }
        if startup is not None:
            hostname = startup.get("hostname", "").split(".")[0]
            node = nodes.get(hostname)
            current_boot = _scheduler_boot_time(node.get("BootTime")) if node else None
            original_boot = startup.get("boot_identity", {}).get("boot_time_unix")
            if current_boot is not None and type(original_boot) in (int, float):
                delta = current_boot - original_boot
                item["node_boot_time_difference_seconds"] = delta
                if delta >= -5:
                    item["node_reboot_since_startup"] = delta > 5
            recorded = item["last_progress"].get("recorded_unix")
            if recorded is not None:
                item["progress_predates_selected_startup"] = recorded < startup["started_unix_ns"] / 1e9
        summaries.append(item)
    step_counts: dict[str, int] = {}
    for item in summaries:
        last = item["last_progress"]
        if last["status"] == "ok" and not item.get("progress_predates_selected_startup", False):
            step = str(last["step"])
            step_counts[step] = step_counts.get(step, 0) + 1
    return {
        "schema": SNAPSHOT_SCHEMA, "captured_utc": _utc(captured), "job_id": job_id,
        "run_root": str(root), "scientific_identity_member": False,
        "expected_world_size": expected_world, "startup_world_sizes": sorted(expected),
        "ranks": summaries, "scheduler": scheduler, "errors": errors,
        "summary": {
            "last_step_counts_excluding_predating_progress": step_counts,
            "ranks_without_startup": [item["rank"] for item in summaries if item["startup"] is None],
            "ranks_without_complete_progress": [
                item["rank"] for item in summaries if item["last_progress"]["status"] != "ok"
            ],
            "rebooted_hosts": sorted({
                item["startup"]["hostname"] for item in summaries if item["node_reboot_since_startup"] is True
            }),
        },
        "limitations": (
            "Missing/stale telemetry does not establish the failing code stage. Scheduler boot-time "
            "changes establish a reboot, not its OS/hardware/administrative trigger. No node SSH, "
            "process signalling, GPU probing, environment dump, or output-file writes are performed."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--scheduler-timeout", type=float, default=5.0)
    parser.add_argument("--tail-bytes", type=int, default=65536)
    parser.add_argument("--no-scheduler", action="store_true")
    args = parser.parse_args(argv)
    snapshot = failure_snapshot(
        args.run_root, job_id=args.job_id, include_scheduler=not args.no_scheduler,
        scheduler_timeout=args.scheduler_timeout, tail_bytes=args.tail_bytes,
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
