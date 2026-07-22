from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


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
    configured = str(profile.get("storage", {}).get("lustre_root", "auto"))
    if configured and configured.lower() != "auto":
        return Path(configured).expanduser().resolve()
    for name in ("METIS_LUSTRE_ROOT", "SCRATCH", "PROJECT", "WORK"):
        value = os.environ.get(name)
        if value:
            base = Path(value).expanduser()
            suffix = "metis-1.6" if name != "METIS_LUSTRE_ROOT" else ""
            return (base / suffix).resolve()
    return (repository_root() / ".metis-portage").resolve()


def load_profile(profile: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_profile_path(profile)
    payload = expand_environment(load_yaml(path))
    payload.setdefault("name", path.stem)
    payload.setdefault("storage", {})["lustre_root"] = str(infer_lustre_root(payload))
    payload["_path"] = str(path)
    return path, payload

