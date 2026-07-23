from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import socket
import subprocess
import sys
import shutil
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import load_profile, repository_root
from .download import run_download_task
from .handoff import write_acquisition_handoff
from .holdouts import prepare_holdouts
from .manifest import validate_manifest
from .state import StateStore, utc_now


DEFAULT_DRIVER_LANES = {
    "common_crawl_ranges": "common_crawl",
    "github_repositories": "github",
    "github_discussions": "github",
    "repository_index": "github",
    "canonical_web": "canonical",
    "canonical_git": "canonical",
    "canonical_http": "canonical",
    "derived_after_download": "derived",
}
DEFAULT_LANE_LIMITS = {"github": 1, "common_crawl": 2, "canonical": 2, "derived": 1}


def _state(profile: dict[str, Any]) -> StateStore:
    root = Path(profile["storage"]["lustre_root"])
    return StateStore(root / profile["storage"]["directories"]["state"])


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _task_drivers(task: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(item.get("driver"))
                for item in task.get("items", [])
                if item.get("kind") == "builder" and item.get("driver")
            }
        )
    )


def _task_wave(task: dict[str, Any]) -> int:
    """Order payloads, primary builders, dependent discussions, derivations."""

    drivers = set(_task_drivers(task))
    if not drivers:
        return 0
    if "derived_after_download" in drivers:
        return 3
    # Discussion records are retained only for repositories whose pinned
    # source archive supplied a reviewed permissive license file. Build that
    # repository-policy index across every month first.
    if "github_discussions" in drivers:
        return 2
    return 1


def _pending_task_waves(
    lock: dict[str, Any], state: StateStore
) -> list[tuple[int, list[int]]]:
    waves: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for index, task in enumerate(lock.get("download_tasks", [])):
        if not state.is_complete("download", task["task_id"]):
            waves[_task_wave(task)].append(index)
    return [(wave, waves[wave]) for wave in sorted(waves) if waves[wave]]


def _lane_configuration(
    profile: dict[str, Any]
) -> tuple[dict[str, str], dict[str, threading.BoundedSemaphore]]:
    acquisition = profile.get("acquisition", {})
    driver_lanes = {**DEFAULT_DRIVER_LANES, **acquisition.get("driver_lanes", {})}
    limits = {**DEFAULT_LANE_LIMITS, **acquisition.get("lane_max_workers", {})}
    semaphores = {
        str(lane): threading.BoundedSemaphore(max(1, int(limit)))
        for lane, limit in limits.items()
    }
    return {str(driver): str(lane) for driver, lane in driver_lanes.items()}, semaphores


def _run_task_in_lanes(
    profile: dict[str, Any],
    lock: dict[str, Any],
    task_index: int,
    driver_lanes: dict[str, str],
    semaphores: dict[str, threading.BoundedSemaphore],
) -> dict[str, Any]:
    lanes = sorted(
        {
            driver_lanes[driver]
            for driver in _task_drivers(lock["download_tasks"][task_index])
            if driver in driver_lanes
        }
    )
    with ExitStack() as stack:
        for lane in lanes:
            semaphore = semaphores.get(lane)
            if semaphore is not None:
                stack.enter_context(semaphore)
        return run_download_task(profile, task_index)


class _AcquisitionProgress:
    """Small, atomic heartbeat visible through `metisctl status`."""

    def __init__(
        self,
        state: StateStore,
        *,
        profile_path: Path,
        started_at: str,
        task_indices: list[int],
        lock: dict[str, Any],
    ) -> None:
        self.state = state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.payload: dict[str, Any] = {
            "schema": "metis.local-download-progress/v1",
            "status": "running",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "profile": str(profile_path),
            "started_at": started_at,
            "updated_at": utc_now(),
            "current_wave": None,
            "total_tasks_this_run": len(task_indices),
            "completed_tasks_this_run": 0,
            "failed_tasks_this_run": 0,
            "active": [],
            "pending_task_ids": [str(lock["download_tasks"][index]["task_id"]) for index in task_indices],
        }

    def _write_locked(self) -> None:
        self.payload["updated_at"] = utc_now()
        self.state.write("local-download", "progress.json", payload=dict(self.payload))

    def write(self, **changes: Any) -> None:
        with self._lock:
            self.payload.update(changes)
            self._write_locked()

    def task_started(self, task_index: int, task_id: str, wave: int) -> None:
        with self._lock:
            active = list(self.payload["active"])
            active.append({"task_index": task_index, "task_id": task_id, "wave": wave, "started_at": utc_now()})
            pending = list(self.payload["pending_task_ids"])
            if task_id not in pending:
                pending.append(task_id)
            self.payload.update(
                {
                    "current_wave": wave,
                    "active": sorted(active, key=lambda row: row["task_index"]),
                    "pending_task_ids": pending,
                }
            )
            self._write_locked()

    def task_finished(self, task_index: int, task_id: str, *, failed: bool) -> None:
        with self._lock:
            self.payload["active"] = [
                row for row in self.payload["active"] if int(row["task_index"]) != task_index
            ]
            self.payload["pending_task_ids"] = [
                value for value in self.payload["pending_task_ids"] if value != task_id
            ]
            field = "failed_tasks_this_run" if failed else "completed_tasks_this_run"
            self.payload[field] = int(self.payload[field]) + 1
            self._write_locked()

    def _heartbeat(self) -> None:
        while not self._stop.wait(60):
            with self._lock:
                self._write_locked()

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="metis-acquisition-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self, status: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.write(status=status, finished_at=utc_now(), active=[])


def _run_tracked_task(
    profile: dict[str, Any],
    lock: dict[str, Any],
    task_index: int,
    wave: int,
    driver_lanes: dict[str, str],
    semaphores: dict[str, threading.BoundedSemaphore],
    progress: _AcquisitionProgress,
) -> dict[str, Any]:
    task_id = str(lock["download_tasks"][task_index]["task_id"])
    progress.task_started(task_index, task_id, wave)
    failed = True
    try:
        result = _run_task_in_lanes(profile, lock, task_index, driver_lanes, semaphores)
        failed = False
        return result
    finally:
        progress.task_finished(task_index, task_id, failed=failed)


@contextmanager
def _supervisor_lock(state: StateStore) -> Iterator[Path]:
    """Hold a crash-safe singleton lock for the whole acquisition run."""

    lock_path = state.path("local-download", "supervisor.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another acquisition supervisor already owns the singleton lock") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "started_at": utc_now()}))
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def local_download_status(profile: dict[str, Any], state: StateStore | None = None) -> dict[str, Any]:
    state = state or _state(profile)
    payload = state.read("local-download", "supervisor.json", default={})
    progress = state.read("local-download", "progress.json", default={})
    progress_path = state.path("local-download", "progress.json")
    if progress and progress_path.is_file():
        heartbeat_age = max(0, int(time.time() - progress_path.stat().st_mtime))
        progress = {
            **progress,
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_stale": progress.get("status") == "running" and heartbeat_age > 180,
        }
    if not payload:
        return {"status": "not_started", "running": False, "progress": progress or None}
    active_status = payload.get("status") in {"launching", "running"}
    same_host = payload.get("hostname") in {None, "", socket.gethostname()}
    running = active_status and same_host and _pid_is_alive(int(payload.get("pid", 0)))
    if active_status and not same_host:
        return {
            **payload,
            "status": "running_on_other_host",
            "running": True,
            "locally_verifiable": False,
            "progress": progress or None,
        }
    if active_status and not running:
        return {**payload, "status": "interrupted", "running": False, "progress": progress or None}
    return {**payload, "running": running, "progress": progress or None}


def run_local_download_supervisor(profile_name: str) -> dict[str, Any]:
    profile_path, profile = load_profile(profile_name)
    state = _state(profile)
    with _supervisor_lock(state):
        return _run_local_download_supervisor_locked(profile_path, profile, state)


def _run_local_download_supervisor_locked(
    profile_path: Path, profile: dict[str, Any], state: StateStore
) -> dict[str, Any]:
    lock = state.read("sources.lock.json")
    if lock is None:
        raise RuntimeError("sources.lock.json is missing; run `metisctl resolve` first")
    # A marker is not enough to skip a multi-day task. Reconcile its immutable
    # task fingerprint and materialized paths/sizes before planning the resume.
    for index, task in enumerate(lock.get("download_tasks", [])):
        if state.is_complete("download", task["task_id"]):
            run_download_task(profile, index)
    task_waves = _pending_task_waves(lock, state)
    task_indices = [index for _, indices in task_waves for index in indices]
    workers = max(1, int(profile.get("acquisition", {}).get("max_workers", 8)))
    maximum_task_attempts = max(
        1, int(profile.get("acquisition", {}).get("maximum_task_attempts", 4))
    )
    retry_initial_seconds = max(
        0, int(profile.get("acquisition", {}).get("retry_initial_seconds", 15))
    )
    retry_maximum_seconds = max(
        retry_initial_seconds,
        int(profile.get("acquisition", {}).get("retry_maximum_seconds", 60)),
    )
    pending_bytes = sum(int(lock["download_tasks"][index].get("planned_bytes", 0)) for index in task_indices)
    safety_bytes = int(float(profile["storage"].get("safety_free_tb", 0)) * 1_000_000_000_000)
    free_bytes = shutil.disk_usage(Path(profile["storage"]["lustre_root"])).free
    if free_bytes - pending_bytes < safety_bytes:
        raise RuntimeError(
            f"Acquisition preflight requires {pending_bytes:,} planned bytes plus a {safety_bytes:,}-byte "
            f"safety reserve, but only {free_bytes:,} bytes are free"
        )
    started = utc_now()
    state.write(
        "local-download",
        "supervisor.json",
        payload={
            "schema": "metis.local-download-supervisor/v1",
            "status": "running",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "profile": str(profile_path),
            "started_at": started,
            "pending_at_start": len(task_indices),
            "task_waves": {str(wave): indices for wave, indices in task_waves},
            "max_workers": workers,
            "maximum_task_attempts": maximum_task_attempts,
            "planned_bytes": pending_bytes,
            "free_bytes_at_start": free_bytes,
        },
    )
    progress = _AcquisitionProgress(
        state,
        profile_path=profile_path,
        started_at=started,
        task_indices=task_indices,
        lock=lock,
    )
    progress.start()
    failures: list[dict[str, Any]] = []
    retry_history: list[dict[str, Any]] = []
    completed = 0
    try:
        driver_lanes, semaphores = _lane_configuration(profile)
        for wave, wave_indices in task_waves:
            remaining = list(wave_indices)
            last_errors: dict[int, dict[str, Any]] = {}
            for attempt in range(1, maximum_task_attempts + 1):
                failed_this_attempt: list[int] = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(workers, len(remaining)),
                    thread_name_prefix=f"metis-download-wave-{wave}-attempt-{attempt}",
                ) as pool:
                    future_to_index = {
                        pool.submit(
                            _run_tracked_task,
                            profile,
                            lock,
                            task_index,
                            wave,
                            driver_lanes,
                            semaphores,
                            progress,
                        ): task_index
                        for task_index in remaining
                    }
                    for future in concurrent.futures.as_completed(future_to_index):
                        task_index = future_to_index[future]
                        try:
                            future.result()
                            completed += 1
                            last_errors.pop(task_index, None)
                        except Exception as exc:
                            failed_this_attempt.append(task_index)
                            error = {
                                "wave": wave,
                                "task_index": task_index,
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            last_errors[task_index] = error
                            retry_history.append(error)
                if not failed_this_attempt:
                    remaining = []
                    break
                remaining = sorted(failed_this_attempt)
                if attempt < maximum_task_attempts:
                    delay = min(
                        retry_maximum_seconds,
                        retry_initial_seconds * (2 ** (attempt - 1)),
                    )
                    progress.write(
                        retrying_task_ids=[
                            str(lock["download_tasks"][index]["task_id"])
                            for index in remaining
                        ],
                        retry_attempt=attempt + 1,
                        retry_delay_seconds=delay,
                    )
                    if delay:
                        time.sleep(delay)
            failures.extend(last_errors[index] for index in sorted(remaining))
            if failures:
                # A later builder wave may consume the outputs of every task in
                # the previous wave. Never continue across a failed boundary.
                break
        holdouts: dict[str, Any] | None = None
        handoff: dict[str, Any] | None = None
        if not failures:
            holdouts = prepare_holdouts(profile, state)
            manifest_path = Path(profile.get("manifest", "manifests/metis-1.6.yaml"))
            if not manifest_path.is_absolute():
                manifest_path = repository_root() / manifest_path
            manifest = validate_manifest(manifest_path).require_valid()
            handoff = write_acquisition_handoff(profile, manifest, state)
        result = {
            "schema": "metis.local-download-supervisor/v1",
            "status": "failed" if failures else "complete",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "profile": str(profile_path),
            "started_at": started,
            "finished_at": utc_now(),
            "tasks_completed_this_run": completed,
            "task_failures": failures,
            "retry_history": retry_history,
            "holdouts": holdouts,
            "acquisition_handoff": handoff,
        }
        state.write("local-download", "supervisor.json", payload=result)
        state.write("local-download", "runs", f"{started.replace(':', '-')}.json", payload=result)
        progress.stop(result["status"])
        return result
    except BaseException as exc:
        state.write(
            "local-download",
            "supervisor.json",
            payload={
                "schema": "metis.local-download-supervisor/v1",
                "status": "failed",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "profile": str(profile_path),
                "started_at": started,
                "finished_at": utc_now(),
                "tasks_completed_this_run": completed,
                "task_failures": failures,
                "supervisor_error": f"{type(exc).__name__}: {exc}",
            },
        )
        progress.stop("failed")
        raise


def launch_local_download(profile_path: Path, profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    current = local_download_status(profile, state)
    if current.get("running"):
        return {"launched": False, "reason": "already_running", "supervisor": current}
    root = Path(profile["storage"]["lustre_root"])
    logs = root / profile["storage"]["directories"]["logs"] / "download"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"local-download-{utc_now().replace(':', '-')}.log"
    env = os.environ.copy()
    source_root = str(repository_root() / "src")
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "metis_data.local_download", "--profile", str(profile_path)],
            cwd=repository_root(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "schema": "metis.local-download-launch/v1",
        "launched": True,
        "pid": process.pid,
        "log": str(log_path),
        "profile": str(profile_path),
        "launched_at": utc_now(),
        "command": "detached local acquisition on Lustre server",
    }
    state.write("local-download", "last-launch.json", payload=payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal detached Metis acquisition supervisor")
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_local_download_supervisor(args.profile)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete" else 1
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
