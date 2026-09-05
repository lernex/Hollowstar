"""Meter bulk dedup outputs without claiming or scanning the whole index per job.

Batch leaves, each compaction generation, and each immutable proof blob have
separate WorkingBudget namespaces. Small shared control receipts/zero-byte
locks use WorkingBudget's explicitly reserved policy/metadata allowance.
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any, Iterator, Mapping

from .common import digest_json, read_receipt, write_receipt
from .dedup_locks import metadata_lock


def namespace_name(kind: str, directory: Path) -> str:
    return f"dedup:{kind}:{digest_json(str(directory.resolve()))}"


@contextlib.contextmanager
def storage_namespace(budget: Any, kind: str, directory: Path) -> Iterator[Any]:
    if budget is None:
        yield None
        return
    name = namespace_name(kind, directory)
    try:
        with budget.quota(name, directory) as quota:
            yield quota
    except BaseException as exc:
        try:
            with budget.quota(name, directory):
                pass
        except Exception as recovery:
            exc.add_note(f"Quota recovery remains pending for {name}: {type(recovery).__name__}")
        raise


def quota_receipt(quota: Any, path: Path, payload: Mapping[str, Any]) -> None:
    if quota is None:
        write_receipt(path, payload)
    else:
        quota.write_receipt(path, payload)


def quota_unlink(quota: Any, path: Path) -> None:
    if not path.exists():
        return
    if quota is None:
        path.unlink()
    else:
        quota.unlink(path)


def quota_rmtree(quota: Any, directory: Path) -> None:
    if not directory.exists():
        return
    if quota is None:
        shutil.rmtree(directory)
        return
    for path in directory.rglob("*"):
        if path.is_file() or path.is_symlink():
            quota.unlink(path)
    for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.rmdir()
    if directory != quota.directory:
        directory.rmdir()


def storage_descriptor(root: Path, quota: Any, kind: str) -> dict[str, str] | None:
    if quota is None:
        return None
    return {
        "namespace": quota.namespace, "directory": str(quota.directory.relative_to(root)),
        "kind": kind,
    }


def bind_working_budget(root: Path, budget: Any) -> Any:
    """One-time adoption; retries inspect only their own already-bound namespace."""
    root.mkdir(parents=True, exist_ok=True)
    if budget is None:
        for parent in (root, *root.parents):
            run = parent / "RUN.json"
            if run.is_file() and (parent / "limits.json").is_file():
                if read_receipt(run).get("schema") == "metis17.run/v1":
                    from .storage import WorkingBudget
                    budget = WorkingBudget(parent)
                    break
    if budget is not None and (not hasattr(budget, "root") or not callable(getattr(budget, "quota", None))):
        raise TypeError("working_budget must be a WorkingBudget, not an entered WorkingQuota")
    marker = root / "WORKING_BUDGET.json"
    with metadata_lock(root / "locks" / "working-budget"):
        previous = read_receipt(marker) if marker.exists() else None
        if budget is None:
            if previous is not None:
                raise ValueError("This dedup root requires its WorkingBudget; unmetered writes are forbidden")
            return None
        expected = {
            "schema": "metis17.dedup-working-budget/v1", "root": str(Path(budget.root).resolve()),
            "namespaces": "batch,compaction-generation,receipt-blob/v1",
        }
        if previous is not None:
            if {key: previous.get(key) for key in expected} != expected:
                raise ValueError("Dedup working-budget binding changed")
            if previous.get("state") == "ready":
                return budget
        lock_directories = [root / "locks", *(root / "scopes").glob("*/locks")]
        if any(path.is_dir() for locks in lock_directories for path in locks.iterdir()):
            raise RuntimeError("Quiesce legacy mkdir-lock workers before attaching a working budget")
        write_receipt(marker, {**expected, "state": "adopting"})
        for directory in sorted((root / "batches").glob("*")):
            if directory.is_dir():
                with storage_namespace(budget, "exact-batch", directory):
                    pass
        for directory in sorted((root / "scopes").glob("*/batches/*")):
            if directory.is_dir():
                with storage_namespace(budget, "signature-batch", directory):
                    pass
        for name, kind in (("compactions", "legacy-compactions"), ("receipts", "legacy-receipts")):
            directory = root / name
            if directory.is_dir():
                with storage_namespace(budget, kind, directory):
                    pass
        for pattern, kind in (("compaction-runs/*/*", "compaction"), ("receipt-blobs/*/*", "receipt-blob")):
            for directory in sorted(root.glob(pattern)):
                if directory.is_dir():
                    with storage_namespace(budget, kind, directory):
                        pass
        write_receipt(marker, {**expected, "state": "ready"})
    return budget
