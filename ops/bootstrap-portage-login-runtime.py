#!/usr/bin/env python3
"""Bootstrap the small login-node launcher runtime without network access.

This file intentionally uses only the Python standard library.  The full
ROCm/PyTorch/kernel runtime remains the responsibility of
``metis_portage.runtime`` after the YAML configuration can be loaded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "metis.portage-runtime-bundle/v1"
RECEIPT_SCHEMA = "metis.portage-login-bootstrap/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODULE = re.compile(r"^[A-Za-z0-9_.+/@:-]+$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_hash(value: Mapping[str, Any], *, omit: Sequence[str] = ()) -> str:
    skipped = set(omit)
    return hashlib.sha256(
        _canonical_bytes(
            {key: item for key, item in value.items() if key not in skipped}
        )
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(
        int(item) for item in re.findall(r"\d+", value.split("+", 1)[0])
    )


def _supported_python(version: Sequence[int]) -> bool:
    return (3, 11) <= tuple(version[:2]) < (3, 13)


def _probe_command(
    python: str,
    *,
    setup: Sequence[str] = (),
) -> dict[str, Any] | None:
    source = (
        "import importlib.metadata,json,sys;"
        "print(json.dumps({'python':[sys.version_info.major,sys.version_info.minor,"
        "sys.version_info.micro],'abi':f'cp{sys.version_info.major}{sys.version_info.minor}',"
        "'pyyaml':importlib.metadata.version(\"PyYAML\")}))"
    )
    if setup:
        command = "; ".join(
            (
                "set -euo pipefail",
                *setup,
                shlex.join((python, "-c", source)),
            )
        )
        argv = ("bash", "-lc", command)
    else:
        argv = (python, "-c", source)
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        row = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    version = row.get("python")
    pyyaml = str(row.get("pyyaml", ""))
    if (
        not isinstance(version, list)
        or not _supported_python([int(item) for item in version])
        or not (6, 0) <= _version_tuple(pyyaml) < (7,)
    ):
        return None
    return row


def _safe_bundle_artifact(bundle_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise RuntimeError("Runtime bundle paths must be non-empty and relative")
    root = bundle_path.parent.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Runtime bundle artifact escapes its directory") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Runtime bundle artifact is missing or unsafe: {path}")
    return path


def validate_login_bundle(
    bundle_path: Path,
    *,
    python_abi: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated bundle and its exact PyYAML wheel record."""

    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid runtime bundle JSON: {bundle_path}") from exc
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("bundle_sha256")
        != _json_hash(bundle, omit=("bundle_sha256",))
        or bundle.get("python_abi") != python_abi
    ):
        raise RuntimeError(
            "Login bootstrap bundle must be self-hashed and match the Python ABI"
        )
    matches: list[dict[str, Any]] = []
    wheels = bundle.get("wheels")
    if not isinstance(wheels, list):
        raise RuntimeError("Runtime bundle wheels must be a list")
    for raw in wheels:
        if not isinstance(raw, dict):
            raise RuntimeError("Runtime bundle wheel record must be an object")
        distribution = re.sub(r"[-_.]+", "", str(raw.get("distribution", ""))).lower()
        if distribution != "pyyaml":
            continue
        path = _safe_bundle_artifact(bundle_path, raw.get("path"))
        expected = str(raw.get("sha256", "")).lower()
        version = str(raw.get("version", ""))
        if (
            not _SHA256.fullmatch(expected)
            or _file_hash(path) != expected
            or not (6, 0) <= _version_tuple(version) < (7,)
        ):
            raise RuntimeError("Pinned PyYAML wheel failed its hash/version policy")
        matches.append(
            {
                **raw,
                "resolved_path": str(path),
                "sha256": expected,
                "version": version,
            }
        )
    if len(matches) != 1:
        raise RuntimeError(
            "Runtime bundle must contain exactly one hash-pinned PyYAML wheel"
        )
    return bundle, matches[0]


def _bundle_candidates(root: Path, lustre_root: Path) -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("METIS_PORTAGE_RUNTIME_BUNDLE", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser().resolve())
    names = ("runtime-bundle.json", "metis-portage-runtime-bundle.json")
    for directory in (
        lustre_root / "runtime" / "portage",
        root / "runtime" / "portage",
    ):
        paths.extend(directory / name for name in names)
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file() and not resolved.is_symlink() and resolved not in result:
            result.append(resolved)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_login_bundle(
    *,
    base_python: str,
    bundle_path: Path,
    lustre_root: Path,
) -> tuple[Path, Path]:
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    bundle, wheel = validate_login_bundle(bundle_path, python_abi=abi)
    bundle_sha = str(bundle["bundle_sha256"])
    wheel_sha = str(wheel["sha256"])
    cache_root = (
        lustre_root
        / "training"
        / "portage"
        / "bootstrap-login"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".bootstrap.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        target = cache_root / f"{bundle_sha[:16]}-{wheel_sha[:16]}-{abi}"
        receipt_path = target / "BOOTSTRAP.json"
        python_path = target / "bin" / "python"
        if python_path.is_file() and receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
            if (
                isinstance(receipt, dict)
                and receipt.get("schema") == RECEIPT_SCHEMA
                and receipt.get("receipt_sha256")
                == _json_hash(receipt, omit=("receipt_sha256",))
                and receipt.get("bundle_sha256") == bundle_sha
                and receipt.get("pyyaml_wheel_sha256") == wheel_sha
                and _probe_command(str(python_path)) is not None
            ):
                return python_path, receipt_path
        temporary = Path(
            tempfile.mkdtemp(prefix=".bootstrap-", dir=cache_root)
        )
        try:
            commands = (
                (
                    base_python,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(temporary),
                ),
                (
                    str(temporary / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    str(wheel["resolved_path"]),
                ),
            )
            for argv in commands:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Pinned login-runtime bootstrap failed: "
                        + completed.stdout[-4000:]
                    )
            audit = _probe_command(str(temporary / "bin" / "python"))
            if audit is None or audit.get("pyyaml") != wheel["version"]:
                raise RuntimeError(
                    "Pinned login runtime did not load the exact bundled PyYAML"
                )
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "created_at_unix": int(time.time()),
                "bundle_path": str(bundle_path),
                "bundle_file_sha256": _file_hash(bundle_path),
                "bundle_sha256": bundle_sha,
                "pyyaml_wheel_path": str(wheel["resolved_path"]),
                "pyyaml_wheel_sha256": wheel_sha,
                "pyyaml_version": wheel["version"],
                "python_abi": abi,
                "python_version": audit["python"],
                "network_resolution": False,
            }
            receipt["receipt_sha256"] = _json_hash(receipt)
            _atomic_json(temporary / "BOOTSTRAP.json", receipt)
            if target.exists():
                quarantine = cache_root / (
                    f".invalid-{target.name}-{int(time.time())}-{os.getpid()}"
                )
                os.replace(target, quarantine)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return target / "bin" / "python", target / "BOOTSTRAP.json"


def _available_modules() -> list[str]:
    completed = subprocess.run(
        (
            "bash",
            "-lc",
            "source /etc/profile >/dev/null 2>&1 || true; "
            "type module >/dev/null 2>&1 && module -t avail 2>&1 || true",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    candidates: list[str] = []
    for line in completed.stdout.splitlines():
        value = line.strip().split()[0].rstrip(":") if line.strip() else ""
        if (
            value
            and _MODULE.fullmatch(value)
            and re.search(r"(python|pytorch|torch)", value, re.IGNORECASE)
            and value not in candidates
        ):
            candidates.append(value)
    return candidates[:80]


def _module_setup(module: str) -> tuple[str, ...]:
    return (
        "source /etc/profile >/dev/null 2>&1 || true",
        "type module >/dev/null 2>&1",
        "module purge >/dev/null 2>&1",
        f"module load {shlex.quote(module)}",
    )


def _launcher_argv(
    *,
    python: str,
    config: Path,
    mode: str,
) -> list[str]:
    return [
        python,
        "-m",
        "metis_portage.launcher",
        "--config",
        str(config),
        mode,
    ]


def _exec_python(
    *,
    python: str,
    root: Path,
    lustre_root: Path,
    config: Path,
    mode: str,
    receipt: Path | None = None,
) -> None:
    environment = dict(os.environ)
    environment["METIS_LUSTRE_ROOT"] = str(lustre_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    if receipt is not None:
        environment["METIS_LOGIN_BOOTSTRAP_RECEIPT"] = str(receipt)
    os.execve(
        python,
        _launcher_argv(python=python, config=config, mode=mode),
        environment,
    )


def _exec_module(
    *,
    module: str,
    root: Path,
    lustre_root: Path,
    config: Path,
    mode: str,
) -> None:
    command = "; ".join(
        (
            "set -euo pipefail",
            *_module_setup(module),
            f"export METIS_LUSTRE_ROOT={shlex.quote(str(lustre_root))}",
            "export PYTHONPATH="
            + shlex.quote(str(root / "src"))
            + "${PYTHONPATH:+:$PYTHONPATH}",
            "exec "
            + shlex.join(
                _launcher_argv(
                    python="python3",
                    config=config,
                    mode=mode,
                )
            ),
        )
    )
    os.execvp("bash", ("bash", "-lc", command))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--lustre-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("launch", "status"), default="launch")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    lustre_root = args.lustre_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not (root / "src" / "metis_portage").is_dir() or not config.is_file():
        raise RuntimeError("Repository root or Portage config is missing")

    direct = [sys.executable, str(root / ".metis-runtime" / "bin" / "python")]
    for raw in direct:
        candidate = Path(raw).expanduser()
        if candidate.is_file() and _probe_command(str(candidate)) is not None:
            _exec_python(
                python=str(candidate.resolve()),
                root=root,
                lustre_root=lustre_root,
                config=config,
                mode=args.mode,
            )

    for module in _available_modules():
        if _probe_command("python3", setup=_module_setup(module)) is not None:
            _exec_module(
                module=module,
                root=root,
                lustre_root=lustre_root,
                config=config,
                mode=args.mode,
            )

    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not _supported_python(sys.version_info):
        raise RuntimeError(
            "No policy-compatible Python 3.11/3.12 was found for login bootstrap"
        )
    failures: list[str] = []
    for bundle_path in _bundle_candidates(root, lustre_root):
        try:
            validate_login_bundle(bundle_path, python_abi=abi)
            python, receipt = _install_login_bundle(
                base_python=sys.executable,
                bundle_path=bundle_path,
                lustre_root=lustre_root,
            )
            _exec_python(
                python=str(python),
                root=root,
                lustre_root=lustre_root,
                config=config,
                mode=args.mode,
                receipt=receipt,
            )
        except Exception as exc:
            failures.append(f"{bundle_path}: {type(exc).__name__}: {exc}")
    detail = "; ".join(failures) if failures else "no runtime bundle was discovered"
    raise RuntimeError(
        "No offline, policy-compatible login runtime could load PyYAML. "
        "The bootstrap searched the active Python, live site modules, and "
        f"hash-pinned local runtime bundles; {detail}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
