from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .config import PortageConfig
from .util import (
    CommandRunner,
    atomic_write_json,
    file_sha256,
    json_sha256,
    read_json,
    utc_now,
)


LOGIN_RUNTIME_SCHEMA = "metis.portage-runtime/v2"
COMPUTE_RUNTIME_SCHEMA = "metis.portage-compute-runtime/v1"
RUNTIME_POLICY_SCHEMA = "metis.portage-runtime-policy/v1"
RUNTIME_BUNDLE_SCHEMA = "metis.portage-runtime-bundle/v1"
_MODULE_NAME = re.compile(r"^[A-Za-z0-9_.+/@:-]+$")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {str(key): _expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        resolved = environment.get(match.group(1), "")
        if resolved:
            return resolved
        fallback = match.group(2)
        if fallback is not None:
            return fallback
        raise RuntimeError(f"Required runtime-policy environment {match.group(1)} is unset")

    return _ENV.sub(replace, value)


def load_runtime_policy(config: PortageConfig) -> dict[str, Any]:
    payload = yaml.safe_load(config.runtime_policy.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Portage runtime policy must be a mapping")
    payload = _expand(payload, os.environ)
    if payload.get("schema") != RUNTIME_POLICY_SCHEMA:
        raise RuntimeError(f"Unexpected runtime policy schema: {payload.get('schema')!r}")
    if payload.get("discovery", {}).get("allow_network") is not False:
        raise RuntimeError("Portage runtime policy must forbid network package resolution")
    if payload.get("discovery", {}).get("allow_unhashed_artifacts") is not False:
        raise RuntimeError("Portage runtime policy must forbid unhashed runtime artifacts")
    mandatory = payload.get("packages", {}).get("mandatory")
    accelerators = payload.get("packages", {}).get("accelerators")
    if not isinstance(mandatory, list) or not mandatory:
        raise RuntimeError("Runtime policy has no mandatory Python packages")
    if not isinstance(accelerators, list):
        raise RuntimeError("Runtime policy accelerators must be a list")
    compute_smoke = payload.get("compute_smoke")
    if not isinstance(compute_smoke, dict):
        raise RuntimeError("Runtime policy compute_smoke must be a mapping")
    for field in (
        "document_isolation_max_abs_error",
        "document_isolation_max_gradient_error",
        "document_isolation_min_control_delta",
    ):
        value = compute_smoke.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RuntimeError(
                f"Runtime policy compute_smoke.{field} must be finite and positive"
            )
    return payload


def _numeric_version(value: str) -> tuple[int, ...]:
    numbers = [int(item) for item in re.findall(r"\d+", value.split("+", 1)[0])]
    return tuple(numbers or [0])


def _version_in_range(
    value: str | None,
    *,
    minimum: str | None,
    maximum_exclusive: str | None,
) -> bool:
    if not value:
        return False
    observed = _numeric_version(value)

    def padded(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        width = max(len(left), len(right))
        return left + (0,) * (width - len(left)), right + (0,) * (width - len(right))

    if minimum:
        left, right = padded(observed, _numeric_version(minimum))
        if left < right:
            return False
    if maximum_exclusive:
        left, right = padded(observed, _numeric_version(maximum_exclusive))
        if left >= right:
            return False
    return True


def _package_specs(policy: dict[str, Any]) -> list[dict[str, Any]]:
    packages = policy["packages"]
    return [
        *packages["mandatory"],
        packages["torch"],
        *packages["accelerators"],
    ]


def _manifest_requirements(config: PortageConfig, policy: dict[str, Any]) -> set[str]:
    required = {
        str(item["distribution"])
        for item in policy["packages"]["mandatory"]
    }
    required.add(str(policy["packages"]["torch"]["distribution"]))
    manifests = [
        yaml.safe_load(family.manifest.read_text(encoding="utf-8"))
        for family in config.families
    ]
    for spec in policy["packages"]["accelerators"]:
        condition = str(spec.get("required_when_manifest", ""))
        if spec.get("required") is True:
            required.add(str(spec["distribution"]))
        elif "=" in condition:
            field, expected = condition.split("=", 1)
            if field and any(str(manifest.get(field)) == expected for manifest in manifests):
                required.add(str(spec["distribution"]))
    return required


def _runtime_probe_source(policy: dict[str, Any]) -> str:
    rows = [
        {
            "distribution": str(item["distribution"]),
            "module": str(item["import"]),
        }
        for item in _package_specs(policy)
    ]
    return f"""
# metis_runtime_probe
import importlib
import importlib.metadata
import json
import platform
import sys
import sysconfig

specs = {json.dumps(rows, sort_keys=True)}
packages = {{}}
for spec in specs:
    name = spec["distribution"]
    try:
        module = importlib.import_module(spec["module"])
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        packages[name] = {{
            "ok": True,
            "version": None if version is None else str(version),
            "module": spec["module"],
            "module_file": str(getattr(module, "__file__", "") or ""),
        }}
    except BaseException as exc:
        packages[name] = {{
            "ok": False,
            "version": None,
            "module": spec["module"],
            "error": type(exc).__name__ + ": " + str(exc),
        }}

torch_row = {{}}
try:
    import torch
    torch_row = {{
        "ok": True,
        "version": str(torch.__version__),
        "hip": None if torch.version.hip is None else str(torch.version.hip),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "git_version": str(getattr(torch.version, "git_version", "") or ""),
        "cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }}
except BaseException as exc:
    torch_row = {{"ok": False, "error": type(exc).__name__ + ": " + str(exc)}}

print(json.dumps({{
    "marker": "metis_runtime_probe",
    "python": {{
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "abi": "cp" + str(sys.version_info.major) + str(sys.version_info.minor),
        "soabi": sysconfig.get_config_var("SOABI"),
        "executable": sys.executable,
    }},
    "platform": {{
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
    }},
    "packages": packages,
    "torch": torch_row,
}}, sort_keys=True))
"""


def _parse_probe(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("marker") == "metis_runtime_probe":
            return row
    return None


def _probe_errors(
    audit: dict[str, Any] | None,
    *,
    policy: dict[str, Any],
    required_distributions: set[str],
    require_all: bool,
) -> list[str]:
    if audit is None:
        return ["runtime probe did not emit its marker"]
    errors: list[str] = []
    python = audit.get("python", {})
    python_policy = policy["python"]
    if python.get("implementation") != python_policy["implementation"]:
        errors.append("Python implementation does not match policy")
    if not _version_in_range(
        str(python.get("version", "")),
        minimum=str(python_policy["minimum"]),
        maximum_exclusive=str(python_policy["maximum_exclusive"]),
    ):
        errors.append(f"Python {python.get('version')!r} is outside policy")
    torch_row = audit.get("torch", {})
    torch_policy = policy["packages"]["torch"]
    if (
        torch_row.get("ok") is not True
        or (
            torch_policy.get("require_rocm") is True
            and (not torch_row.get("hip") or torch_row.get("cuda"))
        )
        or not _version_in_range(
            str(torch_row.get("version", "")),
            minimum=str(torch_policy.get("minimum", "")),
            maximum_exclusive=str(torch_policy.get("maximum_exclusive", "")),
        )
    ):
        errors.append("PyTorch is not a policy-compatible ROCm build")
    packages = audit.get("packages", {})
    for spec in _package_specs(policy):
        name = str(spec["distribution"])
        if name == str(torch_policy["distribution"]):
            continue
        if not require_all and name not in required_distributions:
            continue
        row = packages.get(name, {})
        if (
            row.get("ok") is not True
            or not _version_in_range(
                row.get("version"),
                minimum=str(spec.get("minimum", "")) or None,
                maximum_exclusive=str(spec.get("maximum_exclusive", "")) or None,
            )
        ):
            errors.append(f"{name} is missing, unloadable, or outside policy")
    return errors


def _module_inventory(text: str) -> list[str]:
    rows: list[str] = []
    pattern = re.compile(
        r"(rocm|torch|python|transformer.?engine|mamba|aiter|causal.?conv|craype)",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        value = line.strip().split()[0] if line.strip() else ""
        value = value.rstrip(":")
        if (
            value
            and _MODULE_NAME.fullmatch(value)
            and not value.lower().startswith(("where:", "use:", "module"))
            and pattern.search(value)
        ):
            rows.append(value)
    return sorted(set(rows))


def _module_candidates(modules: Sequence[str], maximum: int) -> list[list[str]]:
    pytorch = [item for item in modules if re.search(r"(?:^|[/_-])(?:py)?torch", item, re.I)]
    rocm = [item for item in modules if "rocm" in item.lower()]
    auxiliaries = [
        item
        for item in modules
        if re.search(r"(transformer.?engine|mamba|aiter|causal.?conv)", item, re.I)
    ]
    rows: list[list[str]] = []
    bases = [[name] for name in reversed(pytorch)]
    bases.extend(
        [rocm_name, torch_name]
        for rocm_name in reversed(rocm)
        for torch_name in reversed(pytorch)
    )
    for base in bases:
        rows.append(base)
        if auxiliaries:
            rows.append([*base, *auxiliaries])
            rows.extend([*base, auxiliary] for auxiliary in auxiliaries)
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row)
        if key not in seen and all(_MODULE_NAME.fullmatch(item) for item in row):
            seen.add(key)
            unique.append(row)
    return unique[:maximum]


def _bash_setup(modules: Sequence[str]) -> list[str]:
    if not modules:
        return []
    return [
        "type module >/dev/null 2>&1 || source /etc/profile",
        "module purge >/dev/null 2>&1",
        "module load " + " ".join(shlex.quote(item) for item in modules),
    ]


def _run_python(
    runner: CommandRunner,
    *,
    python: str,
    python_args: Sequence[str],
    setup: Sequence[str] = (),
    prefix: Sequence[str] = (),
    timeout: float = 180,
) -> Any:
    if setup:
        command = "; ".join(
            [
                "set -e",
                *setup,
                shlex.join([python, *python_args]),
            ]
        )
        return runner.run(["bash", "-lc", command], timeout=timeout)
    return runner.run([*prefix, python, *python_args], timeout=timeout)


def _probe_candidate(
    runner: CommandRunner,
    *,
    policy: dict[str, Any],
    python: str = "python3",
    setup: Sequence[str] = (),
    prefix: Sequence[str] = (),
) -> tuple[Any, dict[str, Any] | None]:
    result = _run_python(
        runner,
        python=python,
        python_args=("-c", _runtime_probe_source(policy)),
        setup=setup,
        prefix=prefix,
        timeout=240,
    )
    return result, _parse_probe(result.stdout) if result.ok else None


def _search_roots(config: PortageConfig, policy: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for raw in policy["discovery"].get("search_roots", []):
        value = str(raw).strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config.repository / candidate
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _sidecar_sha256(path: Path, suffix: str) -> str | None:
    sidecar = Path(str(path) + suffix)
    if not sidecar.is_file():
        return None
    token = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    return token if _SHA256.fullmatch(token) else None


def _container_candidates(
    config: PortageConfig,
    policy: dict[str, Any],
) -> list[dict[str, str]]:
    suffix = str(policy["discovery"]["container_sidecar_suffix"])
    paths: list[Path] = []
    explicit = os.environ.get("METIS_PORTAGE_CONTAINER", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser().resolve())
    for root in _search_roots(config, policy):
        if root.is_dir():
            paths.extend(sorted(root.glob("*.sif")))
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not path.is_file() or path.is_symlink():
            continue
        seen.add(path)
        expected = _sidecar_sha256(path, suffix)
        if expected and file_sha256(path) == expected:
            rows.append({"path": str(path), "sha256": expected})
    return rows


def _bundle_candidates(
    config: PortageConfig,
    policy: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("METIS_PORTAGE_RUNTIME_BUNDLE", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser().resolve())
    names = [str(item) for item in policy["discovery"]["bundle_filenames"]]
    for root in _search_roots(config, policy):
        for name in names:
            candidate = root / name
            if candidate.is_file():
                paths.append(candidate.resolve())
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _artifact_path(bundle_path: Path, raw: str) -> Path:
    if not raw or Path(raw).is_absolute():
        raise RuntimeError("Runtime bundle artifact paths must be safe relative paths")
    root = bundle_path.parent.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Runtime bundle artifact escapes bundle directory") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Runtime bundle artifact is missing or unsafe: {path}")
    return path


def _validate_bundle(
    bundle_path: Path,
    *,
    base_audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = read_json(bundle_path)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != RUNTIME_BUNDLE_SCHEMA
        or bundle.get("bundle_sha256")
        != json_sha256(bundle, omit=("bundle_sha256",))
    ):
        raise RuntimeError(f"Runtime bundle is not self-hashed {RUNTIME_BUNDLE_SCHEMA}")
    python = base_audit["python"]
    torch_row = base_audit["torch"]
    for field, observed in (
        ("python_abi", python.get("abi")),
        ("torch_version", torch_row.get("version")),
        ("rocm_version", torch_row.get("hip")),
    ):
        if not bundle.get(field) or str(bundle[field]) != str(observed):
            raise RuntimeError(
                f"Runtime bundle {field}={bundle.get(field)!r} does not bind "
                f"the selected site runtime {observed!r}"
            )
    verified_wheels: list[dict[str, Any]] = []
    verified_sources: list[dict[str, Any]] = []
    for field, destination in (
        ("wheels", verified_wheels),
        ("sources", verified_sources),
    ):
        rows = bundle.get(field, [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Runtime bundle {field} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"Runtime bundle {field} contains a non-object")
            path = _artifact_path(bundle_path, str(row.get("path", "")))
            expected = str(row.get("sha256", "")).lower()
            if not _SHA256.fullmatch(expected) or file_sha256(path) != expected:
                raise RuntimeError(f"Runtime bundle hash mismatch: {path}")
            destination.append({**row, "resolved_path": str(path)})
    if not verified_wheels and not verified_sources:
        raise RuntimeError("Runtime bundle contains no pinned wheels or sources")
    return bundle, verified_wheels, verified_sources


def _run_shell(
    runner: CommandRunner,
    *,
    setup: Sequence[str],
    argv: Sequence[str],
    timeout: float,
) -> Any:
    command = "; ".join(["set -e", *setup, shlex.join(list(argv))])
    return runner.run(["bash", "-lc", command], timeout=timeout)


def _prepare_bundle_runtime(
    config: PortageConfig,
    *,
    bundle_path: Path,
    base: dict[str, Any],
    output: Path,
    policy: dict[str, Any],
    runner: CommandRunner,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempt: dict[str, Any] = {
        "kind": "pinned_bundle",
        "bundle": str(bundle_path),
        "base_modules": list(base["modules"]),
        "ok": False,
    }
    try:
        bundle, wheels, sources = _validate_bundle(
            bundle_path,
            base_audit=base["audit"],
        )
        setup = _bash_setup(base["modules"])
        cache_key = json_sha256(
            {
                "bundle_sha256": bundle["bundle_sha256"],
                "python": base["audit"]["python"],
                "torch": base["audit"]["torch"],
                "modules": base["modules"],
            }
        )
        cache_root = config.state_root / "runtime-cache" / cache_key
        built_root = cache_root / "wheels"
        built_root.mkdir(parents=True, exist_ok=True)
        built: list[dict[str, Any]] = []
        if sources and not policy["discovery"].get("allow_source_builds"):
            raise RuntimeError("Runtime policy forbids source builds")
        for index, source in enumerate(sources):
            source_path = Path(source["resolved_path"])
            identity = json_sha256(
                {
                    "source_sha256": source["sha256"],
                    "distribution": source.get("distribution"),
                    "version": source.get("version"),
                    "base": cache_key,
                }
            )
            receipt_path = cache_root / f"source-{index:03d}-{identity}.json"
            cached: dict[str, Any] | None = None
            if receipt_path.is_file():
                row = read_json(receipt_path)
                wheel_path = Path(str(row.get("wheel_path", "")))
                if (
                    row.get("schema") == "metis.portage-built-wheel/v1"
                    and row.get("receipt_sha256")
                    == json_sha256(row, omit=("receipt_sha256",))
                    and row.get("source_sha256") == source["sha256"]
                    and wheel_path.is_file()
                    and file_sha256(wheel_path) == row.get("wheel_sha256")
                ):
                    cached = row
            if cached is None:
                before = set(built_root.glob("*.whl"))
                result = _run_shell(
                    runner,
                    setup=setup,
                    argv=(
                        "python3",
                        "-m",
                        "pip",
                        "wheel",
                        "--no-index",
                        "--no-deps",
                        "--no-build-isolation",
                        "--disable-pip-version-check",
                        "--wheel-dir",
                        str(built_root),
                        str(source_path),
                    ),
                    timeout=3600,
                )
                if not result.ok:
                    raise RuntimeError(
                        "Pinned source build failed: "
                        + (result.stderr.strip() or result.stdout.strip())
                    )
                after = set(built_root.glob("*.whl"))
                created = sorted(after - before, key=lambda path: path.stat().st_mtime)
                if not created:
                    expected_name = str(source.get("wheel_filename", ""))
                    created = [built_root / expected_name] if expected_name else []
                if len(created) != 1 or not created[0].is_file():
                    raise RuntimeError("Pinned source build did not produce exactly one wheel")
                wheel_path = created[0].resolve()
                cached = {
                    "schema": "metis.portage-built-wheel/v1",
                    "created_at": utc_now(),
                    "source_path": str(source_path),
                    "source_sha256": source["sha256"],
                    "distribution": source.get("distribution"),
                    "version": source.get("version"),
                    "wheel_path": str(wheel_path),
                    "wheel_sha256": file_sha256(wheel_path),
                    "base_runtime_key": cache_key,
                }
                cached["receipt_sha256"] = json_sha256(cached)
                atomic_write_json(receipt_path, cached)
            built.append(cached)
        venv = output / f"venv-{str(bundle['bundle_sha256'])[:16]}"
        result = _run_shell(
            runner,
            setup=setup,
            argv=("python3", "-m", "venv", "--system-site-packages", str(venv)),
            timeout=600,
        )
        if not result.ok:
            raise RuntimeError(
                "Unable to create pinned runtime venv: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        install_paths = [
            *(str(item["resolved_path"]) for item in wheels),
            *(str(item["wheel_path"]) for item in built),
        ]
        result = _run_shell(
            runner,
            setup=setup,
            argv=(
                str(venv / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--disable-pip-version-check",
                "--no-cache-dir",
                *install_paths,
            ),
            timeout=1800,
        )
        if not result.ok:
            raise RuntimeError(
                "Pinned runtime wheel installation failed: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        probe_result, audit = _probe_candidate(
            runner,
            policy=policy,
            python=str(venv / "bin" / "python"),
            setup=setup,
        )
        errors = _probe_errors(
            audit,
            policy=policy,
            required_distributions=_manifest_requirements(config, policy),
            require_all=False,
        )
        attempt.update(
            {
                "returncode": probe_result.returncode,
                "errors": errors,
                "bundle_sha256": bundle["bundle_sha256"],
                "wheels": [
                    {
                        "path": item["resolved_path"],
                        "sha256": item["sha256"],
                    }
                    for item in wheels
                ],
                "built_wheels": built,
                "ok": not errors,
            }
        )
        if errors or audit is None:
            return None, attempt
        return (
            {
                "kind": "pinned_bundle",
                "modules": list(base["modules"]),
                "setup": setup,
                "python": str(venv / "bin" / "python"),
                "venv": str(venv),
                "bundle": str(bundle_path),
                "bundle_sha256": bundle["bundle_sha256"],
                "artifacts": {
                    "wheels": attempt["wheels"],
                    "built_wheels": built,
                },
                "audit": audit,
            },
            attempt,
        )
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        return None, attempt


def _write_runtime_setup(
    config: PortageConfig,
    *,
    output: Path,
    selected: dict[str, Any],
) -> Path:
    setup_path = output / "runtime.sh"
    report_path = output / "runtime.json"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.extend(selected.get("setup", []))
    if selected["kind"] == "container":
        shim = output / "bin"
        shim.mkdir(parents=True, exist_ok=True)
        python_shim = shim / "python3"
        image = selected["container"]["path"]
        python_shim.write_text(
            "\n".join(
                (
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "exec apptainer exec --rocm "
                    + shlex.quote(str(image))
                    + ' python3 "$@"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        python_shim.chmod(0o750)
        lines.append(f"export PATH={shlex.quote(str(shim))}:$PATH")
    elif selected.get("venv"):
        lines.append(
            f"export PATH={shlex.quote(str(Path(selected['venv']) / 'bin'))}:$PATH"
        )
    lines.extend(
        (
            f"export PYTHONPATH={shlex.quote(str(config.repository / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
            f"export METIS_RUNTIME_LOCK={shlex.quote(str(report_path))}",
            "export METIS_PYTHON=python3",
            "export TORCH_BLAS_PREFER_HIPBLASLT=1",
            "export AMD_COMGR_CACHE=1",
        )
    )
    setup_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    setup_path.chmod(0o750)
    return setup_path


def resolve_runtime(
    config: PortageConfig,
    *,
    output_directory: str | Path,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Discover and seal a policy-compatible ROCm training runtime.

    Resolution order is active environment, live site modules, a SHA-256-pinned
    Apptainer image, then a local self-hashed wheel/source bundle layered over a
    measured site ROCm/PyTorch base.  No path uses an index URL, VCS branch, or
    unpinned download.
    """

    runner = runner or CommandRunner()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = load_runtime_policy(config)
    required = _manifest_requirements(config, policy)
    attempts: list[dict[str, Any]] = []
    base_candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    active_result, active_audit = _probe_candidate(runner, policy=policy)
    active_errors = _probe_errors(
        active_audit,
        policy=policy,
        required_distributions=required,
        require_all=False,
    )
    attempts.append(
        {
            "kind": "active",
            "modules": [],
            "returncode": active_result.returncode,
            "errors": active_errors,
            "stderr": active_result.stderr[-65536:],
            "ok": not active_errors,
        }
    )
    if active_audit is not None and not _probe_errors(
        active_audit,
        policy=policy,
        required_distributions=set(),
        require_all=False,
    ):
        base_candidates.append(
            {
                "kind": "active",
                "modules": [],
                "setup": [],
                "python": "python3",
                "audit": active_audit,
            }
        )
    if not active_errors and active_audit is not None:
        selected = base_candidates[-1]

    if selected is None:
        inventory_result = runner.run(
            [
                "bash",
                "-lc",
                (
                    "type module >/dev/null 2>&1 || "
                    "source /etc/profile >/dev/null 2>&1 || true; "
                    "module -t avail 2>&1"
                ),
            ],
            timeout=180,
        )
        modules = _module_inventory(
            inventory_result.stdout + "\n" + inventory_result.stderr
        )
        maximum = int(policy["discovery"]["maximum_module_candidates"])
        for module_names in _module_candidates(modules, maximum):
            setup = _bash_setup(module_names)
            result, audit = _probe_candidate(
                runner,
                policy=policy,
                setup=setup,
            )
            errors = _probe_errors(
                audit,
                policy=policy,
                required_distributions=required,
                require_all=False,
            )
            attempts.append(
                {
                    "kind": "modules",
                    "modules": module_names,
                    "returncode": result.returncode,
                    "errors": errors,
                    "stderr": result.stderr[-65536:],
                    "ok": not errors,
                }
            )
            if audit is not None and not _probe_errors(
                audit,
                policy=policy,
                required_distributions=set(),
                require_all=False,
            ):
                base = {
                    "kind": "modules",
                    "modules": module_names,
                    "setup": setup,
                    "python": "python3",
                    "audit": audit,
                }
                base_candidates.append(base)
                if not errors:
                    selected = base
                    break

    if selected is None and shutil.which("apptainer"):
        for container in _container_candidates(config, policy):
            prefix = ("apptainer", "exec", "--rocm", container["path"])
            result, audit = _probe_candidate(
                runner,
                policy=policy,
                prefix=prefix,
            )
            errors = _probe_errors(
                audit,
                policy=policy,
                required_distributions=required,
                require_all=False,
            )
            attempts.append(
                {
                    "kind": "container",
                    "container": container,
                    "returncode": result.returncode,
                    "errors": errors,
                    "stderr": result.stderr[-65536:],
                    "ok": not errors,
                }
            )
            if not errors and audit is not None:
                selected = {
                    "kind": "container",
                    "modules": [],
                    "setup": [],
                    "python": "python3",
                    "container": container,
                    "audit": audit,
                }
                break

    if selected is None and base_candidates:
        for bundle_path in _bundle_candidates(config, policy):
            for base in base_candidates:
                candidate, attempt = _prepare_bundle_runtime(
                    config,
                    bundle_path=bundle_path,
                    base=base,
                    output=output,
                    policy=policy,
                    runner=runner,
                )
                attempts.append(attempt)
                if candidate is not None:
                    selected = candidate
                    break
            if selected is not None:
                break

    if selected is None:
        failure = {
            "schema": "metis.portage-runtime-attempts/v2",
            "created_at": utc_now(),
            "policy_path": str(config.runtime_policy),
            "policy_sha256": file_sha256(config.runtime_policy),
            "required_distributions": sorted(required),
            "attempts": attempts,
            "ok": False,
        }
        failure["attempts_sha256"] = json_sha256(failure)
        atomic_write_json(output / "runtime-attempts.json", failure)
        raise RuntimeError(
            "No site module, pinned container, or local hash-locked bundle "
            "provided the mandatory ROCm training runtime. See "
            f"{output / 'runtime-attempts.json'}"
        )

    setup_path = _write_runtime_setup(config, output=output, selected=selected)
    report: dict[str, Any] = {
        "schema": LOGIN_RUNTIME_SCHEMA,
        "created_at": utc_now(),
        "kind": selected["kind"],
        "modules": selected.get("modules", []),
        "container": selected.get("container"),
        "venv": selected.get("venv"),
        "bundle": selected.get("bundle"),
        "bundle_sha256": selected.get("bundle_sha256"),
        "artifacts": selected.get("artifacts", {}),
        "policy_path": str(config.runtime_policy),
        "policy_sha256": file_sha256(config.runtime_policy),
        "required_distributions": sorted(required),
        "audit": selected["audit"],
        "setup_path": str(setup_path),
        "setup_sha256": file_sha256(setup_path),
        "attempts": attempts,
        "ok": True,
    }
    report["runtime_sha256"] = json_sha256(report)
    atomic_write_json(output / "runtime.json", report)
    return report


def validate_login_runtime(path: str | Path) -> dict[str, Any]:
    report = read_json(path)
    setup = Path(str(report.get("setup_path", "")))
    if (
        report.get("schema") != LOGIN_RUNTIME_SCHEMA
        or report.get("ok") is not True
        or report.get("runtime_sha256")
        != json_sha256(report, omit=("runtime_sha256",))
        or not setup.is_file()
        or file_sha256(setup) != report.get("setup_sha256")
    ):
        raise RuntimeError("Login runtime lock is invalid or stale")
    return report


def _installed_packages(policy: dict[str, Any]) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for spec in _package_specs(policy):
        distribution = str(spec["distribution"])
        module_name = str(spec["import"])
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            packages[distribution] = {
                "ok": True,
                "version": None if version is None else str(version),
                "module": module_name,
                "module_file": str(getattr(module, "__file__", "") or ""),
            }
        except BaseException as exc:
            packages[distribution] = {
                "ok": False,
                "version": None,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return packages


def _hipblaslt_probe(config: PortageConfig) -> dict[str, Any]:
    import torch

    library_name = ctypes.util.find_library("hipblaslt")
    load_error: str | None = None
    loaded = False
    if library_name:
        try:
            ctypes.CDLL(library_name)
            loaded = True
        except OSError as exc:
            load_error = str(exc)
    candidates: list[str] = []
    for root in filter(
        None,
        (
            os.environ.get("ROCM_PATH"),
            os.environ.get("ROCM_HOME"),
            "/opt/rocm",
        ),
    ):
        path = Path(root)
        for pattern in ("lib/libhipblaslt.so*", "lib64/libhipblaslt.so*"):
            candidates.extend(str(item.resolve()) for item in path.glob(pattern))
    if not loaded:
        for candidate in candidates:
            try:
                ctypes.CDLL(candidate)
                library_name = candidate
                loaded = True
                load_error = None
                break
            except OSError as exc:
                load_error = str(exc)
    preferred_before: str | None = None
    preferred_after: str | None = None
    preference_error: str | None = None
    try:
        preferred_before = str(torch.backends.cuda.preferred_blas_library())
        torch.backends.cuda.preferred_blas_library("hipblaslt")
        preferred_after = str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:
        preference_error = f"{type(exc).__name__}: {exc}"
    rows: dict[str, Any] = {}
    gemm_ok = True
    for family in config.families:
        manifest = yaml.safe_load(family.manifest.read_text(encoding="utf-8"))
        width = int(manifest["d_model"])
        torch.manual_seed(16062026)
        left = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(width, width, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = left @ right
        loss = output.float().square().mean()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item() and torch.isfinite(loss).item())
        gemm_ok = gemm_ok and finite
        rows[family.name] = {
            "shape": [128, width, width],
            "seconds": time.perf_counter() - started,
            "finite": finite,
            "output_dtype": str(output.dtype),
        }
        del left, right, output, loss
    return {
        "ok": loaded and gemm_ok,
        "library": library_name,
        "library_candidates": sorted(set(candidates)),
        "load_error": load_error,
        "preferred_before": preferred_before,
        "preferred_after": preferred_after,
        "preference_error": preference_error,
        "gemms": rows,
    }


def _ck_probe() -> dict[str, Any]:
    roots = [
        Path(item)
        for item in filter(
            None,
            (
                os.environ.get("ROCM_PATH"),
                os.environ.get("ROCM_HOME"),
                "/opt/rocm",
            ),
        )
    ]
    candidates: list[str] = []
    for root in roots:
        for relative in (
            "include/ck",
            "include/composable_kernel",
            "lib/libck*.so*",
            "lib64/libck*.so*",
        ):
            candidates.extend(str(path.resolve()) for path in root.glob(relative))
    try:
        module = importlib.import_module("ck")
        python_module = str(getattr(module, "__file__", "") or "")
    except Exception:
        python_module = ""
    return {
        "ok": bool(candidates or python_module),
        "paths": sorted(set(candidates)),
        "python_module": python_module or None,
        "note": "CK may be embedded in hipBLASLt or AITER when no standalone path exists",
    }


def _fp8_probe(config: PortageConfig) -> dict[str, Any]:
    import torch

    from metis_training.model_config import load_family_config
    from metis_training.precision import build_precision_policy

    rows: dict[str, Any] = {}
    ok = True
    for family in config.families:
        try:
            model_config = load_family_config(family.manifest)
            policy = build_precision_policy(
                model_config.precision,
                profile="fp8",
                device=torch.device("cuda", 0),
                production=False,
                permit_fallback=False,
            )
            row = policy.validate_execution()
            row["audit"] = policy.audit.to_dict()
            rows[family.name] = row
        except BaseException as exc:
            ok = False
            rows[family.name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {"ok": ok, "families": rows}


def _fused_mamba_probe(config: PortageConfig, policy: dict[str, Any]) -> dict[str, Any]:
    import torch

    from metis_training.model import FusedMamba2
    from metis_training.model_config import load_family_config
    from metis_training.precision import build_precision_policy

    length = int(policy["compute_smoke"]["sequence_length"])
    reset_index = int(policy["compute_smoke"]["reset_index"])
    isolation_tolerance = float(
        policy["compute_smoke"]["document_isolation_max_abs_error"]
    )
    gradient_tolerance = float(
        policy["compute_smoke"]["document_isolation_max_gradient_error"]
    )
    minimum_control_delta = float(
        policy["compute_smoke"]["document_isolation_min_control_delta"]
    )
    if not 0 < reset_index < length:
        raise RuntimeError("runtime compute_smoke.reset_index must be inside sequence")
    rows: dict[str, Any] = {}
    ok = True
    for family in config.families:
        try:
            model_config = load_family_config(family.manifest)
            precision_policy = build_precision_policy(
                model_config.precision,
                profile="bf16",
                device=torch.device("cuda", 0),
                production=True,
                permit_fallback=False,
            )
            torch.manual_seed(16062026)
            mixer = FusedMamba2(
                model_config,
                layer_idx=0,
                precision_policy=precision_policy,
                device=torch.device("cuda", 0),
                dtype=torch.bfloat16,
            )
            hidden = torch.randn(
                1,
                length,
                model_config.d_model,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            document_ids = torch.zeros(1, length, device="cuda", dtype=torch.long)
            document_ids[:, reset_index:] = 1
            reset_mask = torch.zeros(1, length, device="cuda", dtype=torch.bool)
            reset_mask[:, 0] = True
            reset_mask[:, reset_index] = True
            started = time.perf_counter()
            output = mixer(
                hidden,
                document_ids=document_ids,
                reset_mask=reset_mask,
            )
            changed_hidden = hidden.detach().clone()
            changed_hidden[:, :reset_index].add_(3.0)
            changed_hidden.requires_grad_(True)
            isolated_output = mixer(
                changed_hidden,
                document_ids=document_ids,
                reset_mask=reset_mask,
            )
            prefix_control_delta = float(
                (
                    output[:, :reset_index].float()
                    - isolated_output[:, :reset_index].float()
                )
                .abs()
                .max()
                .item()
            )
            suffix_max_abs_error = float(
                (
                    output[:, reset_index:].float()
                    - isolated_output[:, reset_index:].float()
                )
                .abs()
                .max()
                .item()
            )
            reference_suffix_gradient = torch.autograd.grad(
                output[:, reset_index:].float().square().mean(),
                hidden,
                retain_graph=True,
            )[0]
            changed_suffix_gradient = torch.autograd.grad(
                isolated_output[:, reset_index:].float().square().mean(),
                changed_hidden,
            )[0]
            prefix_gradient_leak = max(
                float(
                    reference_suffix_gradient[:, :reset_index]
                    .float()
                    .abs()
                    .max()
                    .item()
                ),
                float(
                    changed_suffix_gradient[:, :reset_index]
                    .float()
                    .abs()
                    .max()
                    .item()
                ),
            )
            suffix_gradient_error = float(
                (
                    reference_suffix_gradient[:, reset_index:].float()
                    - changed_suffix_gradient[:, reset_index:].float()
                )
                .abs()
                .max()
                .item()
            )
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            finite = bool(
                torch.isfinite(output).all().item()
                and torch.isfinite(loss).item()
                and hidden.grad is not None
                and torch.isfinite(hidden.grad).all().item()
            )
            accepted_forward = sorted(mixer.accepted_forward)
            reset_capable = "seq_idx" in mixer.accepted_forward
            document_isolation = bool(
                prefix_control_delta >= minimum_control_delta
                and suffix_max_abs_error <= isolation_tolerance
                and prefix_gradient_leak <= gradient_tolerance
                and suffix_gradient_error <= gradient_tolerance
            )
            passed = finite and reset_capable and document_isolation
            ok = ok and passed
            rows[family.name] = {
                "ok": passed,
                "finite": finite,
                "reset_capable": reset_capable,
                "document_isolation": document_isolation,
                "prefix_control_max_abs_delta": prefix_control_delta,
                "suffix_max_abs_error": suffix_max_abs_error,
                "prefix_gradient_leak": prefix_gradient_leak,
                "suffix_gradient_max_abs_error": suffix_gradient_error,
                "seq_idx_dtype": "torch.int32",
                "accepted_forward_arguments": accepted_forward,
                "shape": [1, length, model_config.d_model],
                "seconds": time.perf_counter() - started,
                "output_dtype": str(output.dtype),
            }
            del (
                mixer,
                hidden,
                changed_hidden,
                output,
                isolated_output,
                loss,
                document_ids,
                reset_mask,
                reference_suffix_gradient,
                changed_suffix_gradient,
            )
            torch.cuda.empty_cache()
        except BaseException as exc:
            ok = False
            rows[family.name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            torch.cuda.empty_cache()
    return {"ok": ok, "families": rows}


def _varlen_attention_probe(
    config: PortageConfig,
    policy: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from metis_training.model import _load_varlen_flash_attention

    kernel = _load_varlen_flash_attention()
    length = int(policy["compute_smoke"]["sequence_length"])
    reset_index = int(policy["compute_smoke"]["reset_index"])
    isolation_tolerance = float(
        policy["compute_smoke"]["document_isolation_max_abs_error"]
    )
    gradient_tolerance = float(
        policy["compute_smoke"]["document_isolation_max_gradient_error"]
    )
    minimum_control_delta = float(
        policy["compute_smoke"]["document_isolation_min_control_delta"]
    )
    cu_seqlens = torch.tensor(
        [0, reset_index, length],
        device="cuda",
        dtype=torch.int32,
    )
    rows: dict[str, Any] = {}
    ok = True
    for family in config.families:
        manifest = yaml.safe_load(family.manifest.read_text(encoding="utf-8"))
        q_heads = int(manifest["n_heads"])
        kv_heads = int(manifest["n_kv_heads"])
        head_dim = int(manifest["head_dim"])
        try:
            torch.manual_seed(16062026)
            query = torch.randn(
                length,
                q_heads,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            key = torch.randn(
                length,
                kv_heads,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            value = torch.randn(
                length,
                kv_heads,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            started = time.perf_counter()
            output = kernel(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                reset_index,
                reset_index,
                0.0,
                head_dim**-0.5,
                True,
            )
            if isinstance(output, tuple):
                output = output[0]
            changed_query = query.detach().clone()
            changed_key = key.detach().clone()
            changed_value = value.detach().clone()
            changed_query[:reset_index] += 5
            changed_key[:reset_index] -= 7
            changed_value[:reset_index] *= -3
            changed_query.requires_grad_(True)
            changed_key.requires_grad_(True)
            changed_value.requires_grad_(True)
            isolated = kernel(
                changed_query,
                changed_key,
                changed_value,
                cu_seqlens,
                cu_seqlens,
                reset_index,
                reset_index,
                0.0,
                head_dim**-0.5,
                True,
            )
            if isinstance(isolated, tuple):
                isolated = isolated[0]
            first_document_control_delta = float(
                (
                    output[:reset_index].float()
                    - isolated[:reset_index].float()
                )
                .abs()
                .max()
                .item()
            )
            second_document_error = float(
                (
                    output[reset_index:].float()
                    - isolated[reset_index:].float()
                )
                .abs()
                .max()
                .item()
            )
            reference_suffix_gradients = torch.autograd.grad(
                output[reset_index:].float().square().mean(),
                (query, key, value),
                retain_graph=True,
            )
            changed_suffix_gradients = torch.autograd.grad(
                isolated[reset_index:].float().square().mean(),
                (changed_query, changed_key, changed_value),
            )
            first_document_gradient_leak = max(
                float(gradient[:reset_index].float().abs().max().item())
                for gradient in (
                    *reference_suffix_gradients,
                    *changed_suffix_gradients,
                )
            )
            second_document_gradient_error = max(
                float(
                    (
                        reference_gradient[reset_index:].float()
                        - changed_gradient[reset_index:].float()
                    )
                    .abs()
                    .max()
                    .item()
                )
                for reference_gradient, changed_gradient in zip(
                    reference_suffix_gradients,
                    changed_suffix_gradients,
                    strict=True,
                )
            )
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            finite = bool(
                output.shape == query.shape
                and torch.isfinite(output).all().item()
                and query.grad is not None
                and key.grad is not None
                and value.grad is not None
                and torch.isfinite(query.grad).all().item()
                and torch.isfinite(key.grad).all().item()
                and torch.isfinite(value.grad).all().item()
            )
            document_isolation = bool(
                first_document_control_delta >= minimum_control_delta
                and second_document_error <= isolation_tolerance
                and first_document_gradient_leak <= gradient_tolerance
                and second_document_gradient_error <= gradient_tolerance
            )
            passed = finite and document_isolation
            ok = ok and passed
            rows[family.name] = {
                "ok": passed,
                "finite_forward_backward": finite,
                "document_isolation": document_isolation,
                "first_document_control_max_abs_delta": (
                    first_document_control_delta
                ),
                "second_document_max_abs_error": second_document_error,
                "first_document_gradient_leak": first_document_gradient_leak,
                "second_document_gradient_max_abs_error": (
                    second_document_gradient_error
                ),
                "causal": True,
                "gqa": q_heads != kv_heads,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "packed_sequence_lengths": [reset_index, length - reset_index],
                "seconds": time.perf_counter() - started,
                "output_dtype": str(output.dtype),
            }
            del (
                query,
                key,
                value,
                output,
                isolated,
                changed_query,
                changed_key,
                changed_value,
                reference_suffix_gradients,
                changed_suffix_gradients,
                loss,
            )
            torch.cuda.empty_cache()
        except BaseException as exc:
            ok = False
            rows[family.name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            torch.cuda.empty_cache()
    return {"ok": ok, "families": rows}


def audit_compute_runtime(
    config: PortageConfig,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Exercise the accepted runtime on a real allocated MI300A.

    FP8 is a measured optional capability because both manifests permit an
    explicit BF16 fallback.  Fused Mamba2 and hipBLASLt are hard gates: both
    production manifests declare ``fused_required`` and may never enter the
    reference token loop.
    """

    import torch

    lock_path = os.environ.get("METIS_RUNTIME_LOCK", "").strip()
    if not lock_path:
        raise RuntimeError("METIS_RUNTIME_LOCK is absent from the sealed runtime")
    login = validate_login_runtime(lock_path)
    policy = load_runtime_policy(config)
    packages = _installed_packages(policy)
    current = {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(item) for item in sys.version_info[:3]),
            "abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "soabi": sysconfig.get_config_var("SOABI"),
            "executable": sys.executable,
        },
        "packages": packages,
        "torch": {
            "version": str(torch.__version__),
            "hip": None if torch.version.hip is None else str(torch.version.hip),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
        },
    }
    identity_errors: list[str] = []
    login_audit = login["audit"]
    for field in ("version", "hip", "cuda", "cxx11_abi"):
        if current["torch"].get(field) != login_audit["torch"].get(field):
            identity_errors.append(f"torch.{field} changed between login and compute")
    if current["python"]["abi"] != login_audit["python"].get("abi"):
        identity_errors.append("Python ABI changed between login and compute")
    for distribution in login["required_distributions"]:
        if distribution == policy["packages"]["torch"]["distribution"]:
            continue
        before = login_audit.get("packages", {}).get(distribution, {})
        after = packages.get(distribution, {})
        if before.get("version") != after.get("version") or after.get("ok") is not True:
            identity_errors.append(
                f"{distribution} version/import changed between login and compute"
            )
    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    gpu_arch = (
        str(
            getattr(device, "gcnArchName", "")
            or getattr(device, "gcn_arch_name", "")
        )
        if device is not None
        else ""
    )
    hipblaslt: dict[str, Any]
    ck: dict[str, Any]
    fp8: dict[str, Any]
    mamba: dict[str, Any]
    attention: dict[str, Any]
    try:
        hipblaslt = _hipblaslt_probe(config)
    except BaseException as exc:
        hipblaslt = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        ck = _ck_probe()
    except BaseException as exc:
        ck = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        fp8 = _fp8_probe(config)
    except BaseException as exc:
        fp8 = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        mamba = _fused_mamba_probe(config, policy)
    except BaseException as exc:
        mamba = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        attention = _varlen_attention_probe(config, policy)
    except BaseException as exc:
        attention = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    aiter = packages.get("aiter", {"ok": False, "version": None})
    standalone_ck_required = bool(policy["hardware"]["require_standalone_ck"])
    gates = [
        {
            "name": "sealed-login-runtime",
            "ok": not identity_errors,
            "detail": identity_errors or ["login and compute ABIs match"],
        },
        {
            "name": "rocm-gfx942-bf16",
            "ok": bool(
                torch.cuda.is_available()
                and torch.version.hip
                and not torch.version.cuda
                and str(policy["hardware"]["gpu_arch"]).lower() in gpu_arch.lower()
                and torch.cuda.is_bf16_supported()
            ),
            "detail": {
                "gpu_arch": gpu_arch,
                "hip": torch.version.hip,
                "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            },
        },
        {
            "name": "hipblaslt",
            "ok": (
                hipblaslt.get("ok") is True
                if policy["hardware"]["require_hipblaslt"]
                else True
            ),
            "detail": hipblaslt,
        },
        {
            "name": "fused-mamba2",
            "ok": mamba.get("ok") is True,
            "detail": mamba,
        },
        {
            "name": "packed-varlen-attention",
            "ok": attention.get("ok") is True,
            "detail": attention,
        },
        {
            "name": "standalone-ck",
            "ok": ck.get("ok") is True if standalone_ck_required else True,
            "detail": ck,
        },
    ]
    report: dict[str, Any] = {
        "schema": COMPUTE_RUNTIME_SCHEMA,
        "created_at": utc_now(),
        "login_runtime_path": str(Path(lock_path).resolve()),
        "login_runtime_sha256": login["runtime_sha256"],
        "policy_path": str(config.runtime_policy),
        "policy_sha256": file_sha256(config.runtime_policy),
        "identity": current,
        "gpu": {
            "name": None if device is None else str(device.name),
            "arch": gpu_arch,
            "total_memory": None if device is None else int(device.total_memory),
        },
        "capabilities": {
            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "fp8": fp8,
            "fused_mamba2": mamba,
            "packed_varlen_attention": attention,
            "hipblaslt": hipblaslt,
            "composable_kernel": ck,
            "aiter": aiter,
        },
        "available_precision_profiles": (
            ["fp8", "bf16"] if fp8.get("ok") is True else ["bf16"]
        ),
        "gates": gates,
        "ok": all(row["ok"] for row in gates),
    }
    report["runtime_compute_sha256"] = json_sha256(report)
    atomic_write_json(output_path, report)
    if report["ok"] is not True:
        failed = [row["name"] for row in gates if not row["ok"]]
        raise RuntimeError(
            "Compute runtime failed closed before model probes: " + ", ".join(failed)
        )
    return report


def validate_compute_runtime(path: str | Path, *, config: PortageConfig) -> dict[str, Any]:
    report = read_json(path)
    if (
        report.get("schema") != COMPUTE_RUNTIME_SCHEMA
        or report.get("ok") is not True
        or report.get("runtime_compute_sha256")
        != json_sha256(report, omit=("runtime_compute_sha256",))
        or report.get("policy_sha256") != file_sha256(config.runtime_policy)
        or not report.get("capabilities", {}).get("fused_mamba2", {}).get("ok")
        or not report.get("capabilities", {}).get(
            "packed_varlen_attention", {}
        ).get("ok")
        or not report.get("capabilities", {}).get("hipblaslt", {}).get("ok")
    ):
        raise RuntimeError("Compute runtime report is invalid, stale, or missing hard kernels")
    return report
