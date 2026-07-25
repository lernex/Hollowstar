from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_SECRET_NAME = re.compile(r"(token|secret|password|credential|private.?key)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def json_sha256(value: Any, *, omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key not in omitted}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any, *, mode: int = 0o640) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_beneath(path: str | Path, root: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its permitted root: {resolved}") from exc
    if resolved == resolved_root:
        raise RuntimeError(f"{label} may not be the root itself: {resolved}")
    return resolved


def safe_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if environment is None else environment)
    return {
        key: value
        for key, value in source.items()
        if not _SECRET_NAME.search(key)
    }


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": self.elapsed_seconds,
        }


class CommandRunner:
    """Small injectable subprocess boundary used by discovery and tests."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 60.0,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = False,
    ) -> CommandResult:
        import time

        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(argv),
                cwd=None if cwd is None else str(cwd),
                env=None if env is None else dict(env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            stdout = completed.stdout[-MAX_CAPTURE_BYTES:]
            stderr = completed.stderr[-MAX_CAPTURE_BYTES:]
            result = CommandResult(
                argv=tuple(str(item) for item in argv),
                returncode=int(completed.returncode),
                stdout=stdout,
                stderr=stderr,
                elapsed_seconds=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            result = CommandResult(
                argv=tuple(str(item) for item in argv),
                returncode=124,
                stdout=stdout[-MAX_CAPTURE_BYTES:],
                stderr=(stderr + f"\nTimed out after {timeout:.1f}s")[-MAX_CAPTURE_BYTES:],
                elapsed_seconds=time.monotonic() - started,
            )
        if check and not result.ok:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(result.argv)}\n{detail}"
            )
        return result


def parse_slurm_time(value: str) -> int | None:
    """Return seconds for Slurm's D-HH:MM:SS forms, or None for unlimited."""

    raw = value.strip()
    if raw.upper() in {"UNLIMITED", "INFINITE"}:
        return None
    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in raw.split(":")]
    if len(parts) == 1:
        return days * 86400 + parts[0] * 60
    if len(parts) == 2:
        return days * 86400 + parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported Slurm time: {value!r}")


def split_key_values(text: str) -> dict[str, str]:
    """Parse the flat Key=Value form emitted by ``scontrol -o``."""

    values: dict[str, str] = {}
    for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/.-]*)=", text):
        start = match.end()
        next_match = re.search(r"\s[A-Za-z][A-Za-z0-9_/.-]*=", text[start:])
        end = len(text) if next_match is None else start + next_match.start()
        values[match.group(1)] = text[start:end].strip()
    return values
