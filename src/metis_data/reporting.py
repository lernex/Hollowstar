from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .manifest import candidate_plan
from .state import StateStore


def status(profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    lock = state.read("sources.lock.json", default={})
    tasks = lock.get("download_tasks", [])
    complete = sum(state.is_complete("download", task["task_id"]) for task in tasks)
    materialized_files = 0
    unresolved_remote_plans = 0
    for task in tasks:
        completion = state.read("completed", "download", f"{task['task_id']}.json", default={})
        for file_record in completion.get("files", []):
            if file_record.get("kind") == "remote_source_plan":
                unresolved_remote_plans += 1
            else:
                materialized_files += 1
    slurm_jobs: list[str] = []
    if shutil.which("squeue"):
        result = subprocess.run(
            ["squeue", "--noheader", "--name", "metis16-*", "--format", "%i|%T|%j|%M"],
            capture_output=True,
            text=True,
            check=False,
        )
        slurm_jobs = [line for line in result.stdout.splitlines() if line.strip()]
    stage_completion: dict[str, int] = {}
    completed_root = state.path("completed")
    if completed_root.exists():
        for stage_root in sorted(path for path in completed_root.iterdir() if path.is_dir()):
            stage_completion[stage_root.name] = len(list(stage_root.glob("*.json")))
    release_root = Path(profile["storage"]["lustre_root"]) / profile["storage"]["directories"]["release"]
    return {
        "profile": profile["name"],
        "lustre_root": profile["storage"]["lustre_root"],
        "source_lock": bool(lock),
        "download": {
            "complete": complete,
            "total": len(tasks),
            "pending": len(tasks) - complete,
            "materialized_files": materialized_files,
            "unresolved_remote_plans": unresolved_remote_plans,
            "build_ready": complete == len(tasks) and unresolved_remote_plans == 0,
        },
        "stage_completion_markers": stage_completion,
        "release_ready": (release_root / "RELEASE.json").exists(),
        "release_path": str(release_root),
        "slurm_jobs": slurm_jobs,
    }


def report(profile: dict[str, Any], manifest: dict[str, Any], state: StateStore) -> dict[str, Any]:
    root = Path(profile["storage"]["lustre_root"])
    usage = shutil.disk_usage(root)
    completion_files = list(state.path("completed").glob("**/*.json")) if state.path("completed").exists() else []
    downloaded_bytes = 0
    for path in state.path("completed", "download").glob("*.json") if state.path("completed", "download").exists() else []:
        downloaded_bytes += int(json.loads(path.read_text(encoding="utf-8")).get("downloaded_bytes", 0))
    return {
        "release": manifest["release"],
        "status": status(profile, state),
        "candidate_plan": candidate_plan(manifest),
        "downloaded_bytes": downloaded_bytes,
        "free_bytes": usage.free,
        "completion_markers": len(completion_files),
        "release_path": str(root / profile["storage"]["directories"]["release"]),
    }
