from __future__ import annotations

import json
import os
import tempfile
import time
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class StateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def read(self, *parts: str, default: Any = None) -> Any:
        path = self.path(*parts)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, *parts: str, payload: dict[str, Any]) -> Path:
        path = self.path(*parts)
        atomic_json(path, payload)
        return path

    def complete(self, stage: str, task_id: str, payload: dict[str, Any]) -> Path:
        return self.write("completed", stage, f"{task_id}.json", payload={**payload, "completed_at": utc_now()})

    def is_complete(self, stage: str, task_id: str) -> bool:
        return self.path("completed", stage, f"{task_id}.json").exists()

    @contextmanager
    def task_lock(self, stage: str, task_id: str) -> Iterator[Path]:
        lock = self.path("locks", stage, f"{task_id}.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(f"Task already has an active lock: {stage}/{task_id}") from exc
        try:
            atomic_json(
                lock / "OWNER.json",
                {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": utc_now()},
            )
            yield lock
        finally:
            (lock / "OWNER.json").unlink(missing_ok=True)
            lock.rmdir()

    def clear_stale_locks(self, older_than_seconds: int) -> list[str]:
        removed: list[str] = []
        locks_root = self.path("locks")
        if not locks_root.exists():
            return removed
        cutoff = time.time() - older_than_seconds
        for lock in sorted(path for path in locks_root.glob("*/*.lock") if path.is_dir()):
            if lock.stat().st_mtime > cutoff:
                continue
            owner = lock / "OWNER.json"
            owner.unlink(missing_ok=True)
            try:
                lock.rmdir()
            except OSError:
                continue
            removed.append(str(lock.relative_to(locks_root)))
        return removed
