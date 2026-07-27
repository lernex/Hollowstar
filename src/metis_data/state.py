from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


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


class ScratchBackedDatabase:
    """A SQLite database that may be worked on node-local disk.

    SQLite is explicitly not designed for network filesystems: POSIX advisory
    locking is unreliable there and every small write pays a round trip.  An
    acquisition builder whose hot tables are keyed by a hash inserts at random
    positions in a large B-tree, so once the tree outgrows the page cache it
    spends nearly all of its wall-clock waiting on single-page reads from
    Lustre.

    When ``scratch_root`` is configured the working copy lives on local disk
    and is republished to the durable root at checkpoints.  Losing the node
    between checkpoints rewinds the database to the last published state, so
    callers must keep their durable side effects recoverable from it: publish
    output only after the ledger that records it has been checkpointed, or
    reconcile the two on restart.

    The publication counter lives in the caller's own settings table, so a
    checkpoint is atomic with the data it describes and a resumed run can tell
    which of the two copies is newer without trusting clocks across two
    filesystems.
    """

    def __init__(
        self,
        durable_path: Path,
        *,
        connect: Callable[[Path, bool], sqlite3.Connection],
        scratch_root: str | None = None,
        checkpoint_seconds: float = 300.0,
        identity: str = "",
        settings_table: str = "metadata",
        sequence_key: str = "state_sequence",
    ) -> None:
        self.durable_path = durable_path
        self.checkpoint_seconds = checkpoint_seconds
        self.settings_table = settings_table
        self.sequence_key = sequence_key
        self.working_path = durable_path
        if scratch_root:
            token = _digest(f"{identity}\0{durable_path}")[:32]
            self.working_path = (
                Path(scratch_root).expanduser() / f"metis-{token}" / durable_path.name
            )
            self._seed_working_copy()
        self.local = self.working_path != durable_path
        self.connection = connect(self.working_path, self.local)
        self._last_checkpoint = time.monotonic()

    def _seed_working_copy(self) -> None:
        self.working_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.durable_path.is_file():
            return
        if self.working_path.is_file() and self._sequence(self.working_path) >= self._sequence(
            self.durable_path
        ):
            # A resumed run on the same node already holds the newer copy.
            return
        temporary = self.working_path.with_name(f".{self.working_path.name}.seed")
        temporary.unlink(missing_ok=True)
        shutil.copyfile(self.durable_path, temporary)
        os.replace(temporary, self.working_path)

    def _sequence(self, path: Path) -> int:
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
        except sqlite3.Error:
            return -1
        try:
            row = connection.execute(
                f"SELECT value FROM {self.settings_table} WHERE key = ?", (self.sequence_key,)
            ).fetchone()
            return int(row[0]) if row else 0
        except (sqlite3.Error, TypeError, ValueError):
            return -1
        finally:
            connection.close()

    def checkpoint(self, *, force: bool = False) -> None:
        """Publish the working database to the durable root."""

        if not self.local:
            return
        if not force and time.monotonic() - self._last_checkpoint < self.checkpoint_seconds:
            return
        row = self.connection.execute(
            f"SELECT value FROM {self.settings_table} WHERE key = ?", (self.sequence_key,)
        ).fetchone()
        self.connection.execute(
            f"INSERT INTO {self.settings_table}(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (self.sequence_key, str(int(row[0]) + 1 if row else 1)),
        )
        self.connection.commit()
        self.durable_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.durable_path.with_name(
            f".{self.durable_path.name}.{os.getpid()}.publish"
        )
        temporary.unlink(missing_ok=True)
        published = sqlite3.connect(temporary, timeout=600)
        try:
            self.connection.backup(published)
            published.commit()
        finally:
            published.close()
        os.replace(temporary, self.durable_path)
        self._last_checkpoint = time.monotonic()

    def close(self) -> None:
        try:
            self.checkpoint(force=True)
        finally:
            self.connection.close()


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            owner_path = lock / "OWNER.json"
            owner: dict[str, Any] = {}
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            owner_pid = int(owner.get("pid", 0) or 0)
            same_host = owner.get("hostname") == socket.gethostname()
            alive = False
            if same_host and owner_pid > 0:
                try:
                    os.kill(owner_pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except (PermissionError, OSError):
                    # Failure other than ESRCH does not prove the process is
                    # gone; preserve the lock.
                    alive = True
            if same_host and owner_pid > 0 and not alive:
                owner_path.unlink(missing_ok=True)
                try:
                    lock.rmdir()
                    lock.mkdir()
                except OSError as reclaim_error:
                    raise RuntimeError(
                        f"Could not reclaim dead task lock: {stage}/{task_id}"
                    ) from reclaim_error
            else:
                detail = (
                    f"pid={owner_pid} host={owner.get('hostname')}"
                    if owner
                    else "owner metadata unavailable"
                )
                raise RuntimeError(
                    f"Task already has an active or unverified lock: {stage}/{task_id} ({detail})"
                ) from exc
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
            try:
                owner_payload = json.loads(owner.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Age alone cannot prove abandonment. Preserve an unverifiable
                # lock for manual inspection instead of risking two writers.
                continue
            if owner_payload.get("hostname") != socket.gethostname():
                continue
            owner_pid = int(owner_payload.get("pid", 0) or 0)
            if owner_pid <= 0:
                continue
            try:
                os.kill(owner_pid, 0)
                # A legitimate acquisition/materialization task can run for
                # days. Never reclaim a lock held by a live process.
                continue
            except ProcessLookupError:
                pass
            except (PermissionError, OSError):
                continue
            owner.unlink(missing_ok=True)
            try:
                lock.rmdir()
            except OSError:
                continue
            removed.append(str(lock.relative_to(locks_root)))
        return removed
