from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

import torch


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")

    try:
        torch.save(payload, tmp_path)
        _fsync_file(tmp_path)
        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError(f"Checkpoint write produced an invalid archive: {tmp_path}")
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise RuntimeError(f"Checkpoint write produced an empty file: {path}")
        return size_bytes
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def publish_directory(tmp_dir: Path, out_dir: Path) -> None:
    parent_dir = out_dir.parent
    backup_dir = out_dir.with_name(f".{out_dir.name}.bak-{os.getpid()}-{uuid.uuid4().hex}")
    had_existing = out_dir.exists()
    published = False

    try:
        if had_existing:
            out_dir.replace(backup_dir)
            _fsync_dir(parent_dir)
        tmp_dir.replace(out_dir)
        _fsync_dir(parent_dir)
        published = True
    except Exception:
        if had_existing and backup_dir.exists() and not out_dir.exists():
            backup_dir.replace(out_dir)
            _fsync_dir(parent_dir)
        raise
    finally:
        if published and backup_dir.exists():
            shutil.rmtree(backup_dir)
