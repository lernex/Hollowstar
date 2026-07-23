from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return payload


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value
    match = ENV_PATTERN.match(value)
    if not match:
        return value
    name, default = match.groups()
    return os.environ.get(name, default or "")


def resolve_profile_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate.resolve()
    configured = repository_root() / "configs" / "metis16" / f"{profile}.yaml"
    if configured.exists():
        return configured.resolve()
    raise FileNotFoundError(f"Unknown profile {profile!r}; expected {configured}")


def infer_lustre_root(profile: dict[str, Any]) -> Path:
    storage = profile.get("storage", {})
    configured = str(storage.get("lustre_root", "auto"))
    if configured and configured.lower() != "auto":
        root = Path(configured).expanduser().resolve()
        return validate_storage_root(profile, root)
    if storage.get("require_explicit_root"):
        raise RuntimeError(
            "This production profile requires an explicit Lustre data directory. "
            "Set METIS_LUSTRE_ROOT or pass --lustre-root to the operator launcher; "
            "the shared Lustre filesystem root is not a valid data directory."
        )
    for name in ("METIS_LUSTRE_ROOT", "SCRATCH", "PROJECT", "WORK"):
        value = os.environ.get(name)
        if value:
            base = Path(value).expanduser()
            suffix = "metis-1.6" if name != "METIS_LUSTRE_ROOT" else ""
            return validate_storage_root(profile, (base / suffix).resolve())
    return validate_storage_root(profile, (repository_root() / ".metis-portage").resolve())


def validate_storage_root(profile: dict[str, Any], root: Path) -> Path:
    """Reject shared filesystem roots and unapproved production locations.

    A typo in the production root must not silently populate the repository or
    the top of a shared Lustre mount with many terabytes of data.
    """

    storage = profile.get("storage", {})
    resolved = root.expanduser().resolve()
    forbidden = {
        Path(value).expanduser().resolve()
        for value in storage.get("forbidden_roots", ["/", "/lus", "/lus/lustre1"])
    }
    if resolved in forbidden:
        raise RuntimeError(f"Refusing unsafe shared storage root: {resolved}")
    prefixes = [Path(value).expanduser().resolve() for value in storage.get("allowed_root_prefixes", [])]
    if prefixes and not any(_is_relative_to(resolved, prefix) and resolved != prefix for prefix in prefixes):
        expected = ", ".join(str(path) for path in prefixes)
        raise RuntimeError(f"Storage root {resolved} is outside the allowed project prefixes: {expected}")
    if storage.get("must_exist") and not resolved.is_dir():
        raise RuntimeError(f"Configured storage root does not exist on this host: {resolved}")
    return resolved


def load_profile(profile: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_profile_path(profile)
    payload = expand_environment(load_yaml(path))
    payload.setdefault("name", path.stem)
    payload.setdefault("storage", {})["lustre_root"] = str(infer_lustre_root(payload))
    payload["_path"] = str(path)
    return path, payload
