"""Shared, restart-safe byte reservations for derived release data.

Use ``with WorkingBudget(root).quota(name, directory) as quota`` and write data
through ``quota.open(path, "wb")``. Names and directories are permanently paired;
directories cannot overlap. A namespace has one live owner (POSIX flock).
Include a generation or root-relative directory digest in names when retaining
multiple generations of the same object or stage.
``quota.reserve(total_bytes)`` optionally pre-reserves an operation's entire
namespace footprint and fixes its hard byte ceiling for that context.
Renames within a namespace cost nothing; use ``replace``/``unlink`` for files
removed while other streams are open, or ``reconcile`` after closing streams.
``write_bytes`` and ``write_receipt`` meter atomic file replacements, including
their staging copies; immutability checks remain the publishing stage's policy.

Only namespace entry, successful exit, and explicit reconciliation inspect its
directory. Reservations coordinate in coarse increments, not per document.
An interrupted owner retains its reservation until a new owner measures the
surviving files. Small, unmetered receipts use the fixed metadata allowance.
"""

from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import stat
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

from .acquisition import CapacityPending
from .common import canonical_json, digest_json, read_receipt


ALLOCATION_BYTES = 64 * 1024 * 1024
_GLOBAL_SCHEMA = "metis17.working-budget/v1"
_NAMESPACE_SCHEMA = "metis17.working-namespace/v1"
_PENDING_SCHEMA = "metis17.working-budget-operation/v1"
_CLAIM_SCHEMA = "metis17.working-directory-claim/v1"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _serialized(method: Any) -> Any:
    @wraps(method)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        # FileIO releases the GIL; another stream must not spend the same local
        # credit between reservation and accounting for the completed write.
        with self._mutex:
            return method(self, *args, **kwargs)
    return call


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir(path: Path) -> None:
    if path.is_dir():
        if path.is_symlink():
            raise ValueError("Storage accounting directories cannot be symlinks")
        return
    _mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir() or path.is_symlink():
            raise
    _sync_directory(path.parent)


def _write_sealed(path: Path, value: Mapping[str, Any]) -> None:
    # A fixed sibling is safe under flock and does not accumulate after crashes.
    pending = path.with_name(path.name + ".next")
    payload = {**value, "receipt_sha256": digest_json(value)}
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical_json(payload) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    _sync_directory(path.parent)


@contextmanager
def _flock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _nonnegative(value: Mapping[str, Any], name: str) -> int:
    result = value.get(name)
    if type(result) is not int or result < 0:
        raise ValueError(f"Invalid working-storage counter: {name}")
    return result


def _validate_counters(value: Mapping[str, Any], schema: str) -> None:
    if value.get("schema") != schema:
        raise ValueError("Unsupported working-storage ledger schema")
    if _nonnegative(value, "committed_bytes") > _nonnegative(value, "reserved_bytes"):
        raise ValueError("Working-storage ledger has unreserved data")
    if schema == _GLOBAL_SCHEMA:
        _nonnegative(value, "sequence")
        _nonnegative(value, "namespaces")


class WorkingBudget:
    def __init__(self, root: Path, *, allocation_bytes: int = ALLOCATION_BYTES) -> None:
        self.root = Path(root).expanduser().resolve()
        if type(allocation_bytes) is not int or allocation_bytes < 1:
            raise ValueError("A positive storage allocation increment is required")
        self.allocation_bytes = allocation_bytes
        self._limits()
        self.directory = self.root / "state" / "working-budget"
        if self.directory.resolve() != self.directory:
            raise ValueError("Storage accounting state cannot be redirected")
        _mkdir(self.directory)
        self.path = self.directory / "total.json"
        self.pending_path = self.directory / "pending.json"
        self.lock_path = self.directory / "global.lock"
        with self._locked():
            pass

    def _limits(self) -> dict[str, Any]:
        limits = read_receipt(self.root / "limits.json")
        confirmation = limits.get("capacity_confirmation")
        if confirmation not in {"pending", "administrator-confirmed", "unlimited"}:
            raise ValueError("Unsupported capacity confirmation")
        for name in ("max_raw_bytes", "max_working_bytes"):
            if _nonnegative(limits, name) == 0:
                raise ValueError(f"A positive capacity bound is required: {name}")
        limits.setdefault("policy_and_metadata_reserve_bytes", 20_000_000_000)
        limits.setdefault("filesystem_free_floor_bytes", 100_000_000_000)
        _nonnegative(limits, "policy_and_metadata_reserve_bytes")
        _nonnegative(limits, "filesystem_free_floor_bytes")
        if confirmation == "pending" and (
            limits["max_raw_bytes"] > 400_000_000_000
            or limits["max_working_bytes"] > 2_000_000_000_000
        ):
            raise CapacityPending("Full working storage needs explicit capacity confirmation")
        if limits["max_raw_bytes"] > 200_000_000_000_000:
            raise ValueError("Raw allocation exceeds the 200 TB envelope")
        limits["derived_limit_bytes"] = (
            limits["max_working_bytes"] - limits["max_raw_bytes"]
            - limits["policy_and_metadata_reserve_bytes"]
        )
        if limits["derived_limit_bytes"] < 0:
            raise CapacityPending("Raw and metadata reservations exhaust working storage")
        return limits

    def _namespace_path(self, namespace: str) -> Path:
        key = digest_json(namespace)
        return self.directory / "namespaces" / key[:2] / f"{key}.json"

    def _read_global(self) -> dict[str, Any]:
        if not self.path.exists():
            if (
                self.pending_path.exists()
                or (self.directory / "namespaces").exists()
                or (self.directory / "claims").exists()
            ):
                raise ValueError("Working-storage total is missing from an existing ledger")
            _write_sealed(self.path, {
                "schema": _GLOBAL_SCHEMA, "sequence": 0, "namespaces": 0,
                "reserved_bytes": 0, "committed_bytes": 0,
            })
        value = read_receipt(self.path)
        _validate_counters(value, _GLOBAL_SCHEMA)
        return value

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        with _flock(self.lock_path):
            state = self._read_global()
            if self.pending_path.exists():
                state = self._recover(state)
            yield state

    def _recover(self, state: dict[str, Any]) -> dict[str, Any]:
        operation = read_receipt(self.pending_path)
        if operation.get("schema") != _PENDING_SCHEMA:
            raise ValueError("Unsupported working-storage recovery journal")
        before, after = operation["global_before"], operation["global_after"]
        old, new = operation["namespace_before"], operation["namespace_after"]
        _validate_counters(before, _GLOBAL_SCHEMA)
        _validate_counters(after, _GLOBAL_SCHEMA)
        _validate_counters(new, _NAMESPACE_SCHEMA)
        if old is not None:
            _validate_counters(old, _NAMESPACE_SCHEMA)
            if (old["namespace"], old["directory"]) != (new["namespace"], new["directory"]):
                raise ValueError("Working-storage recovery changes namespace ownership")
        if (
            after["sequence"] != before["sequence"] + 1
            or after["namespaces"] != before["namespaces"] + (old is None)
            or any(after[key] != before[key] + new[key] - (old[key] if old else 0)
                   for key in ("reserved_bytes", "committed_bytes"))
            or state not in (before, after)
        ):
            raise ValueError("Working-storage recovery journal conflicts with the total")
        path = self._namespace_path(new["namespace"])
        current = read_receipt(path) if path.exists() else None
        if current not in (old, new):
            raise ValueError("Working-storage recovery journal conflicts with the namespace")
        _mkdir(path.parent)
        if current != new:
            _write_sealed(path, new)
        if state != after:
            _write_sealed(self.path, after)
        self.pending_path.unlink()
        _sync_directory(self.directory)
        return after

    def _commit(
        self, state: dict[str, Any], old: dict[str, Any] | None, new: dict[str, Any],
    ) -> None:
        _validate_counters(new, _NAMESPACE_SCHEMA)
        after = {
            **state, "sequence": state["sequence"] + 1,
            "namespaces": state["namespaces"] + (old is None),
            **{key: state[key] + new[key] - (old[key] if old else 0)
               for key in ("reserved_bytes", "committed_bytes")},
        }
        _validate_counters(after, _GLOBAL_SCHEMA)
        _mkdir(self._namespace_path(new["namespace"]).parent)
        _write_sealed(self.pending_path, {
            "schema": _PENDING_SCHEMA, "global_before": state, "global_after": after,
            "namespace_before": old, "namespace_after": new,
        })
        self._recover(state)

    def _claim(self, namespace: str, directory: Path) -> None:
        # A trie detects ancestor/descendant claims without enumerating releases
        # or the potentially millions of independent namespace receipts.
        cursor = self.directory / "claims"
        _mkdir(cursor)
        for component in directory.relative_to(self.root).parts:
            if (cursor / "owner.json").exists():
                raise ValueError("Working-storage namespace directories cannot overlap")
            cursor /= digest_json(component)
            _mkdir(cursor)
        claim = {
            "schema": _CLAIM_SCHEMA, "namespace": namespace,
            "directory": str(directory.relative_to(self.root)),
        }
        owner = cursor / "owner.json"
        if owner.exists():
            if read_receipt(owner) != claim:
                raise ValueError("Working-storage directory already belongs to another namespace")
            return
        with os.scandir(cursor) as entries:
            if any(entry.is_dir(follow_symlinks=False) for entry in entries):
                raise ValueError("Working-storage namespace directories cannot overlap")
        _write_sealed(owner, claim)

    def _physical_allowance(self, limits: Mapping[str, Any], outstanding: int) -> int:
        raw_path = self.root / "state" / "intake-budget.json"
        raw_committed = _nonnegative(read_receipt(raw_path), "raw_bytes") if raw_path.exists() else 0
        # Raw has its own intake ledger. Reserving its remaining entire cap here
        # protects the floor even while downloaders consume their reservations.
        raw_remaining = max(0, limits["max_raw_bytes"] - raw_committed)
        return (
            shutil.disk_usage(self.root).free - limits["filesystem_free_floor_bytes"]
            - raw_remaining - limits["policy_and_metadata_reserve_bytes"] - outstanding
        )

    def quota(self, namespace: str, directory: Path) -> WorkingQuota:
        return WorkingQuota(self, namespace, directory)

    def snapshot(self) -> dict[str, Any]:
        """Derived-only counters; committed bytes lag open writers.

        The raw cap and fixed metadata reserve are already deducted from
        ``derived_limit_bytes`` inside ``max_working_bytes``.
        """
        with self._locked() as state:
            limits = self._limits()
            return {**limits, **state,
                    "outstanding_bytes": state["reserved_bytes"] - state["committed_bytes"]}


class WorkingQuota:
    def __init__(self, budget: WorkingBudget, namespace: str, directory: Path) -> None:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError("A bounded, nonempty storage namespace is required")
        self.budget, self.namespace = budget, namespace
        self.directory = Path(directory).expanduser().absolute()
        if (
            self.directory == budget.root or not self.directory.is_relative_to(budget.root)
            or self.directory.resolve() != self.directory
            or ".." in self.directory.parts
        ):
            raise ValueError("Working-storage namespace must be confined below the release root")
        relative = self.directory.relative_to(budget.root)
        if relative.parts[0] in {"raw", "state", "locks", "ready", "transfers"}:
            raise ValueError("A derived namespace cannot own raw or accounting infrastructure")
        if len(relative.parts) > 32:
            raise ValueError("Working-storage namespace is too deeply nested")
        self._active = False
        self._failed = False
        self._accounting_ok = True
        self._byte_limit: int | None = None
        self._used = 0
        self._files: dict[str, tuple[int, int, int]] = {}
        self._writers: dict[str, QuotaWriter] = {}
        self._state: dict[str, Any] = {}
        self._resources = ExitStack()
        self._directory_fd = -1
        self._mutex = RLock()

    @property
    def reserved_bytes(self) -> int:
        return self._state["reserved_bytes"]

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def byte_limit(self) -> int | None:
        return self._byte_limit

    def _check(self) -> None:
        if not self._active or not self._accounting_ok:
            raise ValueError("Storage quota is closed or requires recovery by a new owner")

    def __enter__(self) -> WorkingQuota:
        if self._active or self._state:
            raise ValueError("A storage quota context cannot be entered twice")
        path = self.budget._namespace_path(self.namespace)
        with ExitStack() as resources:
            _mkdir(path.parent)
            resources.enter_context(_flock(path.with_suffix(".lock"), blocking=False))
            with self.budget._locked() as state:
                old = read_receipt(path) if path.exists() else None
                relative = str(self.directory.relative_to(self.budget.root))
                if old is not None:
                    _validate_counters(old, _NAMESPACE_SCHEMA)
                    if (old["namespace"], old["directory"]) != (self.namespace, relative):
                        raise ValueError("Working-storage namespace cannot change its directory")
                self.budget._claim(self.namespace, self.directory)
                _mkdir(self.directory)
                self._state = old or {
                    "schema": _NAMESPACE_SCHEMA, "namespace": self.namespace, "directory": relative,
                    "reserved_bytes": 0, "committed_bytes": 0,
                }
                if old is None:
                    self.budget._commit(state, None, self._state)
            descriptor = os.open(self.budget.root, _DIRECTORY_FLAGS)
            resources.callback(os.close, descriptor)
            for component in self.directory.relative_to(self.budget.root).parts:
                descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                resources.callback(os.close, descriptor)
            self._directory_fd = descriptor
            self._active = True
            try:
                self.reconcile()
            finally:
                self._active = False
            self._resources = resources.pop_all()
        self._active = True
        return self

    @_serialized
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            with ExitStack() as closing:
                for writer in tuple(self._writers.values()):
                    closing.callback(writer.close)
            if exc_type is None and not self._failed and self._accounting_ok:
                self.reconcile()
        finally:
            self._active = False
            self._resources.close()

    def _inventory(self) -> dict[str, tuple[int, int, int]]:
        files: dict[str, tuple[int, int, int]] = {}
        device = self.budget.root.stat().st_dev

        def visit(descriptor: int, prefix: str) -> None:
            if os.fstat(descriptor).st_dev != device:
                raise ValueError("A storage namespace cannot cross filesystems")
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    name = prefix + entry.name
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                        try:
                            visit(child, name + "/")
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_dev == device:
                        files[name] = (info.st_dev, info.st_ino, info.st_size)
                    else:
                        raise ValueError("Storage namespace contains a link or nonregular file")
            # A deletion must survive a host failure before its reservation is
            # returned, including abandoned generation cleanup by the caller.
            os.fsync(descriptor)

        visit(self._directory_fd, "")
        return files

    def _update(self, state: dict[str, Any], reserved: int) -> None:
        new = {**self._state, "reserved_bytes": reserved, "committed_bytes": self._used}
        self._accounting_ok = False
        if new != self._state:
            self.budget._commit(state, self._state, new)
        self._state = new
        self._accounting_ok = True

    @_serialized
    def reconcile(self) -> None:
        """Measure this namespace only; all its quota streams must be closed."""
        self._check()
        if self._writers:
            raise ValueError("Close quota streams before reconciling their files")
        files = self._inventory()
        actual = sum(value[2] for value in files.values())
        with self.budget._locked() as state:
            self._used = actual
            self._update(state, actual)
            self._files = files
            # Existing data is charged even when it already exceeds the cap.
            # Refusing to record it would let a different namespace overspend.
            total = self.budget._read_global()["reserved_bytes"]
            if total > self.budget._limits()["derived_limit_bytes"]:
                raise CapacityPending("Existing derived data exhausts working storage")
            if self._byte_limit is not None and actual > self._byte_limit:
                raise CapacityPending("Existing namespace data exceeds the operation's byte ceiling")

    @_serialized
    def reserve(self, total_bytes: int) -> int:
        """Pre-reserve a hard total footprint, including already-existing files.

        Call before opening streams. The ceiling is fixed for this context;
        deletion can reclaim space for reuse beneath it. Raw and fixed metadata
        allowances are separately protected inside the release's working cap.
        """
        self._check()
        if type(total_bytes) is not int or total_bytes < 0:
            raise ValueError("A nonnegative namespace byte ceiling is required")
        if self._writers:
            raise ValueError("Reserve the operation bound before opening quota streams")
        if self._byte_limit is not None and self._byte_limit != total_bytes:
            raise ValueError("A namespace byte ceiling cannot change within its context")
        if self._used > total_bytes:
            raise CapacityPending("Existing namespace data exceeds the requested byte ceiling")
        previous_limit = self._byte_limit
        self._byte_limit = total_bytes
        try:
            self._reserve(total_bytes - self._used)
        except CapacityPending:
            self._byte_limit = previous_limit
            raise
        if self.reserved_bytes > total_bytes:
            with self.budget._locked() as state:
                self._update(state, total_bytes)
        return self.reserved_bytes

    def _reserve(self, growth: int) -> None:
        self._check()
        if self._byte_limit is not None and self._used + growth > self._byte_limit:
            raise CapacityPending("The operation's namespace byte ceiling is exhausted")
        if self._used + growth <= self.reserved_bytes:
            return
        with self.budget._locked() as state:
            limits = self.budget._limits()
            required = self._used + growth - self.reserved_bytes
            outstanding = (
                state["reserved_bytes"] - state["committed_bytes"]
                - (self._used - self._state["committed_bytes"])
            )
            available = min(
                limits["derived_limit_bytes"] - state["reserved_bytes"],
                self.budget._physical_allowance(limits, outstanding),
            )
            if self._byte_limit is not None:
                available = min(available, self._byte_limit - self.reserved_bytes)
            if required > available:
                raise CapacityPending("Derived working-storage allowance or free-space floor exhausted")
            rounded = ((required + self.budget.allocation_bytes - 1)
                       // self.budget.allocation_bytes * self.budget.allocation_bytes)
            self._update(state, self.reserved_bytes + min(rounded, available))

    def _removed(self, size: int) -> None:
        if size:
            with self.budget._locked() as state:
                self._update(state, self.reserved_bytes - size)

    def _key(self, path: Path) -> str:
        path = Path(path)
        if not path.is_absolute():
            path = self.directory / path
        if path == self.directory or ".." in path.parts or not path.is_relative_to(self.directory):
            raise ValueError("File path escapes its working-storage namespace")
        return str(path.relative_to(self.directory))

    @contextmanager
    def _parent(self, key: str) -> Iterator[tuple[int, str]]:
        with ExitStack() as resources:
            descriptor = self._directory_fd
            for part in Path(key).parts[:-1]:
                descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                resources.callback(os.close, descriptor)
            if os.fstat(descriptor).st_dev != self.budget.root.stat().st_dev:
                raise ValueError("A storage namespace cannot cross filesystems")
            yield descriptor, Path(key).name

    def _file_info(self, key: str, info: os.stat_result) -> tuple[int, int, int]:
        if (
            not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_dev != os.fstat(self._directory_fd).st_dev
        ):
            raise ValueError("Quota output must be a singly linked, regular namespace file")
        value = (info.st_dev, info.st_ino, info.st_size)
        if key in self._files and self._files[key] != value:
            raise ValueError("Namespace file changed outside its quota writer")
        if key not in self._files and info.st_size:
            raise ValueError("Reconcile existing namespace files before writing them")
        return value

    @_serialized
    def open(self, path: Path, mode: str = "wb") -> QuotaWriter:
        self._check()
        modes = {"wb", "xb", "ab", "w+b", "x+b", "a+b", "r+b"}
        if mode not in modes:
            raise ValueError("Quota output requires a supported binary write mode")
        key = self._key(path)
        if key in self._writers:
            raise ValueError("A namespace file already has an open quota writer")
        flags = os.O_RDWR if "+" in mode else os.O_WRONLY
        flags |= os.O_NOFOLLOW
        if mode[0] in "wxa":
            flags |= os.O_CREAT
        if mode[0] == "x":
            flags |= os.O_EXCL
        if mode[0] == "a":
            flags |= os.O_APPEND
        with self._parent(key) as (parent, name):
            descriptor = os.open(name, flags, 0o600, dir_fd=parent)
        result = None
        try:
            self._files[key] = self._file_info(key, os.fstat(descriptor))
            result = QuotaWriter(self, key, descriptor, mode)
            self._writers[key] = result
            if mode[0] == "w":
                result.truncate(0)
            if mode[0] == "a":
                result.seek(0, os.SEEK_END)
            return result
        finally:
            if result is None:
                os.close(descriptor)

    @_serialized
    def unlink(self, path: Path) -> None:
        self._check()
        key = self._key(path)
        if key in self._writers:
            raise ValueError("Close a quota writer before removing its file")
        with self._parent(key) as (parent, name):
            value = self._file_info(key, os.stat(name, dir_fd=parent, follow_symlinks=False))
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)
        self._files.pop(key, None)
        self._used -= value[2]
        self._removed(value[2])

    @_serialized
    def replace(self, source: Path, destination: Path) -> None:
        self._check()
        source_key, destination_key = self._key(source), self._key(destination)
        if source_key == destination_key:
            return
        if source_key in self._writers or destination_key in self._writers:
            raise ValueError("Close quota writers before moving their files")
        with self._parent(source_key) as (source_parent, source_name), \
                self._parent(destination_key) as (destination_parent, destination_name):
            source_info = self._file_info(
                source_key, os.stat(source_name, dir_fd=source_parent, follow_symlinks=False))
            try:
                target = os.stat(destination_name, dir_fd=destination_parent, follow_symlinks=False)
            except FileNotFoundError:
                removed = 0
            else:
                removed = self._file_info(destination_key, target)[2]
            os.replace(source_name, destination_name, src_dir_fd=source_parent, dst_dir_fd=destination_parent)
            os.fsync(source_parent)
            os.fsync(destination_parent)
        self._files.pop(source_key, None)
        self._files[destination_key] = source_info
        self._used -= removed
        self._removed(removed)

    @_serialized
    def write_bytes(self, path: Path, data: bytes) -> None:
        """Atomic, metered replacement; a failure preserves the old destination."""
        self._check()
        key = self._key(path)
        if key in self._writers:
            raise ValueError("Close a quota writer before replacing its file")
        target = self.directory / key
        # One reusable sibling bounds interrupted retries without scanning or
        # discarding data belonging to another receipt.
        staging = target.with_name(f".{digest_json(key)}.receipt-partial")
        with memoryview(data).cast("B") as view, self.open(staging, "wb") as stream:
            written = 0
            while written < len(view):
                count = stream.write(view[written:])
                if not count:
                    self._failed = True
                    raise OSError("An atomic quota output made no write progress")
                written += count
        self.replace(staging, target)

    def write_receipt(self, path: Path, value: Mapping[str, Any]) -> None:
        """Write common.write_receipt-compatible bytes through this quota."""
        if "receipt_sha256" in value:
            raise ValueError("Pass receipt contents without an existing seal")
        payload = {**value, "receipt_sha256": digest_json(value)}
        self.write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


class QuotaWriter(io.RawIOBase):
    """Unbuffered binary IO: reserve logical file growth before the OS write."""

    def __init__(self, quota: WorkingQuota, key: str, descriptor: int, mode: str) -> None:
        super().__init__()
        self.quota, self.key = quota, key
        self._mutex = quota._mutex
        self._file = io.FileIO(descriptor, mode=mode, closefd=True)
        self._size = quota._files[key][2]
        self._append = mode[0] == "a"
        self._failure: CapacityPending | OSError | None = None

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return self._file.readable()

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._file.fileno()

    @_serialized
    def tell(self) -> int:
        return self._file.tell()

    @_serialized
    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._file.seek(offset, whence)

    @_serialized
    def readinto(self, buffer: Any) -> int | None:
        return self._file.readinto(buffer)

    def _resize(self, size: int) -> None:
        self.quota._used += size - self._size
        self._size = size
        device, inode, _ = self.quota._files[self.key]
        self.quota._files[self.key] = (device, inode, size)

    @_serialized
    def write(self, data: Any) -> int:
        self._checkClosed()
        if self._failure is not None:
            raise self._failure
        with memoryview(data).cast("B") as view:
            size = view.nbytes
            position = self._size if self._append else self.tell()
            try:
                self.quota._reserve(max(0, position + size - self._size) if size else 0)
                written = 0
                while written < size:
                    self.quota._accounting_ok = False
                    count = self._file.write(view[written:])
                    if not count:
                        raise OSError("A regular quota output made no write progress")
                    written += count
                    self._resize(max(self._size, position + written))
                    self.quota._accounting_ok = True
            except (CapacityPending, OSError) as exc:
                self._failure = exc
                self.quota._failed = True
                if isinstance(exc, OSError):
                    self.quota._accounting_ok = False
                raise
            return written

    @_serialized
    def truncate(self, size: int | None = None) -> int:
        self._checkClosed()
        self.quota._check()
        if self._failure is not None:
            raise self._failure
        size = self.tell() if size is None else size
        if type(size) is not int or size < 0:
            raise ValueError("A nonnegative truncation size is required")
        old = self._size
        try:
            self.quota._reserve(max(0, size - old))
            self.quota._accounting_ok = False
            result = self._file.truncate(size)
            if size < old:
                os.fsync(self.fileno())
        except (CapacityPending, OSError) as exc:
            self._failure = exc
            self.quota._failed = True
            if isinstance(exc, OSError):
                self.quota._accounting_ok = False
            raise
        self._resize(size)
        if size < old:
            self.quota._removed(old - size)
        self.quota._accounting_ok = True
        return result

    @_serialized
    def flush(self) -> None:
        self._checkClosed()
        self._file.flush()

    @_serialized
    def close(self) -> None:
        if not self.closed:
            try:
                super().close()
                os.fsync(self._file.fileno())
            except OSError:
                self.quota._failed = True
                raise
            finally:
                self._file.close()
                self.quota._writers.pop(self.key, None)
