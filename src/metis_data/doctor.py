from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from huggingface_hub import HfApi, get_token

from .config import repository_root
from .manifest import validate_manifest
from .config import load_yaml


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _command_check(name: str, required: bool = True) -> Check:
    path = shutil.which(name)
    if path:
        return Check(name, "PASS", path)
    return Check(name, "FAIL" if required else "WARN", "not found")


def _python_runtime_check() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    return Check(
        "python-runtime",
        "PASS" if supported else "FAIL",
        f"{version}; required >=3.11,<3.13",
    )


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["df", "-T", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return lines[-1].split()[1] if len(lines) >= 2 else "unknown"
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def _hf_checks(manifest: dict[str, Any], tiny_probe: bool, *, auth_required: bool) -> list[Check]:
    checks: list[Check] = []
    token = get_token() or os.environ.get("HF_TOKEN")
    gated = sorted(
        {
            access["repo_id"]
            for source in manifest["sources"]
            for access in [source.get("access", {})]
            if access.get("type") == "huggingface" and access.get("gated") in {"manual", "auto", True}
        }
        | {
            component["repo_id"]
            for source in manifest["sources"]
            for component in source.get("access", {}).get("components", [])
            if component.get("gated") in {"manual", "auto", True}
        }
    )
    checks.append(
        Check(
            "huggingface-token",
            "PASS" if token else ("FAIL" if auth_required else "WARN"),
            "available" if token else "HF_TOKEN or `hf auth login` required for gated production sources",
        )
    )
    if tiny_probe:
        return checks
    api = HfApi(token=token)
    for repo_id in gated:
        try:
            info = api.dataset_info(repo_id, timeout=30)
            checks.append(Check(f"hf:{repo_id}", "PASS", f"revision={info.sha}"))
        except Exception as exc:  # live service/access failure
            checks.append(Check(f"hf:{repo_id}", "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"))
    return checks


def _network_checks(tiny_probe: bool) -> list[Check]:
    if tiny_probe:
        return []
    checks: list[Check] = []
    for name, url in (
        ("common-crawl-egress", "https://index.commoncrawl.org/collinfo.json"),
        ("github-egress", "https://api.github.com/rate_limit"),
    ):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            checks.append(Check(name, "PASS", f"HTTP {response.status_code}"))
        except Exception as exc:
            checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    checks.append(
        Check(
            "github-token",
            "PASS" if github_token else "FAIL",
            "available" if github_token else "GITHUB_TOKEN or GH_TOKEN required for production repository acquisition",
        )
    )
    return checks


def _holdout_access_checks(tiny_probe: bool) -> list[Check]:
    if tiny_probe:
        return []
    registry = load_yaml(repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml")
    api = HfApi(token=get_token() or os.environ.get("HF_TOKEN"))
    checks: list[Check] = []
    for entry in registry["benchmarks"]:
        try:
            info = api.dataset_info(entry["repo_id"], revision=entry["revision"], timeout=30)
            checks.append(Check(f"holdout:{entry['id']}", "PASS", f"revision={info.sha}"))
        except Exception as exc:
            checks.append(Check(f"holdout:{entry['id']}", "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"))
    return checks


def run_doctor(profile: dict[str, Any], *, tiny_probe: bool = False) -> dict[str, Any]:
    checks: list[Check] = []
    validation = validate_manifest(profile.get("manifest"))
    checks.append(
        Check(
            "manifest",
            "PASS" if validation.ok else "FAIL",
            f"{len(validation.manifest.get('sources', []))} sources; {len(validation.errors)} errors",
        )
    )
    license_review_complete = bool(profile.get("gates", {}).get("license_review_complete", False))
    checks.append(
        Check(
            "license-review",
            "PASS" if license_review_complete else ("WARN" if tiny_probe else "FAIL"),
            "attested complete" if license_review_complete else "pending source terms and per-record policy review",
        )
    )

    storage = profile["storage"]
    root = Path(storage["lustre_root"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=".metis-write-probe-", dir=root, delete=True) as handle:
            handle.write(b"metis")
            handle.flush()
            os.fsync(handle.fileno())
        checks.append(Check("lustre-write", "PASS", str(root)))
    except OSError as exc:
        checks.append(Check("lustre-write", "FAIL", str(exc)))

    usage = shutil.disk_usage(root)
    free_tb = usage.free / 1_000_000_000_000
    minimum = float(storage.get("minimum_free_tb", 0))
    recommended = float(storage.get("recommended_free_tb", minimum))
    status = "PASS" if free_tb >= minimum else "FAIL"
    detail = f"{free_tb:.2f} TB free; minimum {minimum:.2f} TB; recommended {recommended:.2f} TB"
    if status == "PASS" and free_tb < recommended:
        status = "WARN"
    checks.append(Check("free-space", status, detail))

    fs_type = _filesystem_type(root)
    expected_lustre = profile.get("name") == "portage"
    checks.append(Check("filesystem", "PASS" if fs_type == "lustre" or not expected_lustre else "WARN", fs_type))

    scheduler_mode = profile.get("scheduler", {}).get("mode", "slurm")
    checks.extend(
        [
            _command_check("git"),
            _python_runtime_check(),
            _command_check("sbatch", required=scheduler_mode == "slurm"),
            _command_check("squeue", required=scheduler_mode == "slurm"),
            _command_check("apptainer", required=bool(profile.get("runtime", {}).get("require_apptainer_for_compute"))),
        ]
    )
    checks.extend(
        _hf_checks(
            validation.manifest,
            tiny_probe=tiny_probe,
            auth_required=bool(profile.get("gates", {}).get("require_hf_auth_for_gated_sources", True)),
        )
    )
    checks.extend(_network_checks(tiny_probe=tiny_probe))
    checks.extend(_holdout_access_checks(tiny_probe=tiny_probe))

    dynamic_drivers = sorted(
        {
            source.get("acquisition", {}).get("driver")
            for source in validation.manifest.get("sources", [])
            if source.get("acquisition", {}).get("driver")
            in {
                "common_crawl_ranges",
                "github_repositories",
                "github_discussions",
                "canonical_web",
                "canonical_git",
                "canonical_http",
                "repository_index",
                "derived_after_download",
            }
        }
    )
    materializers_enabled = bool(profile.get("runtime", {}).get("dynamic_materializers_enabled", False))
    checks.append(
        Check(
            "dynamic-source-materializers",
            "PASS" if materializers_enabled else ("WARN" if tiny_probe else "FAIL"),
            "enabled" if materializers_enabled else f"not connected for: {', '.join(dynamic_drivers)}",
        )
    )

    contamination_path = repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
    checks.append(Check("contamination-registry", "PASS" if contamination_path.exists() else "FAIL", str(contamination_path)))
    failed = [check for check in checks if check.status == "FAIL"]
    return {
        "ok": not failed,
        "profile": profile.get("name"),
        "lustre_root": str(root),
        "checks": [asdict(check) for check in checks],
        "manifest_errors": list(validation.errors),
        "manifest_warnings": list(validation.warnings),
    }
