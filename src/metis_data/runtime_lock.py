from __future__ import annotations

import hashlib
import platform
import re
import sys
from pathlib import Path
from typing import Any


RUNTIME_CONTRACT_SCHEMA = "metis.python-runtime/v1"
RUNTIME_INPUT_NAME = "requirements-metis16-data.txt"
RUNTIME_LOCK_NAME = "requirements-metis16-data.lock"
PYTHON_REQUIRES = ">=3.11,<3.13"
SUPPORTED_PYTHON_ABIS = ("cp311", "cp312")
_INPUT_DIGEST_PATTERN = re.compile(r"^# metis-input-sha256: ([0-9a-f]{64})$", re.MULTILINE)
_PYTHON_REQUIRES_PATTERN = re.compile(r"^# metis-python-requires: (.+)$", re.MULTILINE)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_interpreter() -> None:
    if sys.implementation.name != "cpython":
        raise RuntimeError("The Metis-1.6 data runtime requires CPython")
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if abi not in SUPPORTED_PYTHON_ABIS:
        raise RuntimeError(
            f"The Metis-1.6 data runtime requires CPython 3.11 or 3.12; "
            f"the active interpreter is {platform.python_version()}"
        )


def runtime_contract(
    root: str | Path | None = None,
    *,
    validate_interpreter: bool = True,
) -> dict[str, Any]:
    """Return and validate the immutable dependency/runtime contract.

    The generated lock embeds the digest of its human-edited input file. This
    catches both failure modes before acquisition: changing the input without
    regenerating the lock, and changing the lock after a source release was
    frozen.
    """

    if validate_interpreter:
        _validate_interpreter()
    repository = Path(root).expanduser().resolve() if root is not None else _repository_root()
    input_path = repository / RUNTIME_INPUT_NAME
    lock_path = repository / RUNTIME_LOCK_NAME
    if not input_path.is_file():
        raise RuntimeError(f"Runtime input file is missing: {input_path}")
    if not lock_path.is_file():
        raise RuntimeError(f"Hash-locked runtime file is missing: {lock_path}")
    lock_text = lock_path.read_text(encoding="utf-8")
    input_match = _INPUT_DIGEST_PATTERN.search(lock_text)
    if input_match is None:
        raise RuntimeError(f"{RUNTIME_LOCK_NAME} does not declare metis-input-sha256")
    actual_input_sha256 = _sha256_file(input_path)
    if input_match.group(1) != actual_input_sha256:
        raise RuntimeError(
            f"{RUNTIME_INPUT_NAME} changed without regenerating {RUNTIME_LOCK_NAME}"
        )
    python_match = _PYTHON_REQUIRES_PATTERN.search(lock_text)
    if python_match is None or python_match.group(1).strip() != PYTHON_REQUIRES:
        raise RuntimeError(
            f"{RUNTIME_LOCK_NAME} must declare metis-python-requires: {PYTHON_REQUIRES}"
        )
    return {
        "schema": RUNTIME_CONTRACT_SCHEMA,
        "input_file": RUNTIME_INPUT_NAME,
        "input_sha256": actual_input_sha256,
        "lock_file": RUNTIME_LOCK_NAME,
        "lock_sha256": _sha256_file(lock_path),
        "lock_size": lock_path.stat().st_size,
        "python_requires": PYTHON_REQUIRES,
        "supported_python_abis": list(SUPPORTED_PYTHON_ABIS),
        "binary_policy": "only-binary",
        "hash_policy": "require-hashes",
    }


def runtime_identity() -> dict[str, str]:
    _validate_interpreter()
    return {
        "implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }
