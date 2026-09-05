from __future__ import annotations

import errno
import fcntl
import os
import re
import stat
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator


@lru_cache(maxsize=1)
def _mounts() -> tuple[tuple[Path, str, frozenset[str]], ...]:
    path = Path("/proc/self/mountinfo")
    if not path.exists():
        return ()
    mounts = []
    for line in path.read_text().splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 3:
            continue
        name = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), left[4])
        options = frozenset(left[5].split(",") + right[2].split(","))
        mounts.append((Path(name), right[0], options))
    return tuple(sorted(mounts, key=lambda item: len(item[0].parts), reverse=True))


def require_distributed_locks(path: Path) -> None:
    for mount, filesystem, options in _mounts():
        if path.is_relative_to(mount):
            if filesystem == "lustre" and options.intersection({"localflock", "noflock"}):
                raise RuntimeError("Shared 1.7 stages require distributed Lustre flock, not localflock/noflock")
            if filesystem.startswith("nfs") and options.intersection({"nolock", "local_lock=all", "local_lock=flock"}):
                raise RuntimeError("Shared 1.7 stages require distributed NFS file locks")
            return


@contextmanager
def metadata_lock(
    path: Path, *, timeout: float | None = 3600, create: bool = True,
) -> Iterator[int]:
    """Kernel/DLM ownership survives host changes and is released on worker death.

    Persistent lock inodes must not be unlinked while another process can open
    them. Legacy mkdir locks require a quiescent migration, never a TTL guess.
    Indefinite admission waits queue in the kernel to avoid DLM polling storms.
    """
    if timeout is not None and timeout < 0:
        raise ValueError("Lock timeout cannot be negative")
    path = Path(path).absolute()
    require_distributed_locks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        raise RuntimeError(
            f"Legacy directory lock requires quiescent migration before flock use: {path}"
        )
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    descriptor = os.open(path, flags, 0o600)
    deadline = time.monotonic() + timeout if timeout is not None else None
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Dedup locks require regular files")
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | (fcntl.LOCK_NB if deadline is not None else 0))
                acquired = True
                break
            except BlockingIOError:
                if deadline is None:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Dedup lock is held by a live owner: {path}") from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            except OSError as exc:
                if exc.errno in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                    raise RuntimeError("The dedup filesystem does not support distributed file locks") from exc
                raise
        yield descriptor
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
