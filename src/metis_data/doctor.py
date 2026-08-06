from __future__ import annotations

import os
import fnmatch
import gzip
import getpass
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


def _python_module_check(module: str) -> Check:
    try:
        __import__(module)
        return Check(f"python-module:{module}", "PASS", "importable")
    except Exception as exc:
        return Check(f"python-module:{module}", "FAIL", f"{type(exc).__name__}: {str(exc)[:120]}")


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


_QUOTA_UNKNOWN_ACKNOWLEDGEMENTS = {"administrator-confirmed", "unlimited"}


def _ambiguous_lustre_quota_check(
    profile: dict[str, Any],
    detail: str,
) -> Check:
    """Gate production when Lustre reports an ambiguous zero hard quota."""

    storage = profile.get("storage", {})
    acknowledgement = str(storage.get("quota_unknown_acknowledgement") or "").strip().lower()
    if acknowledgement in _QUOTA_UNKNOWN_ACKNOWLEDGEMENTS:
        return Check(
            "lustre-quota",
            "PASS",
            f"{detail}; explicit capacity acknowledgement={acknowledgement}",
        )
    if acknowledgement:
        return Check(
            "lustre-quota",
            "FAIL",
            f"unsupported quota acknowledgement {acknowledgement!r}; use "
            "'administrator-confirmed' or 'unlimited'",
        )
    if storage.get("require_explicit_quota_acknowledgement"):
        return Check(
            "lustre-quota",
            "FAIL",
            f"{detail}; 0/0 can mean default or unlimited. An administrator must confirm "
            "sufficient capacity, then pass --quota-acknowledgement "
            "administrator-confirmed (or unlimited when that is the confirmed policy).",
        )
    return Check(
        "lustre-quota",
        "WARN",
        f"{detail}; 0/0 can mean default or unlimited and needs administrator confirmation",
    )


def _lustre_quota_check(profile: dict[str, Any], root: Path) -> Check:
    if not shutil.which("lfs"):
        return Check("lustre-quota", "FAIL", "lfs command not found")
    subjects = [("user", "-u", getpass.getuser())]
    group = str(profile.get("storage", {}).get("quota_group") or "").strip()
    if group:
        subjects.append(("group", "-g", group))
    parsed: list[str] = []
    remaining_values: list[int] = []
    remaining_inode_values: list[int] = []
    for label, flag, subject in subjects:
        try:
            result = subprocess.run(
                ["lfs", "quota", flag, subject, str(root)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return Check("lustre-quota", "WARN", f"{label} {subject}: {type(exc).__name__}; ask the Lustre administrator")
        quota_row: list[str] | None = None
        for line in reversed(result.stdout.splitlines()):
            fields = line.split()
            if len(fields) >= 8 and fields[0].startswith("/"):
                quota_row = fields
                break
        if quota_row is None:
            parsed.append(f"{label}={subject}:unparsed")
            continue
        try:
            used_kib = int(quota_row[1].rstrip("*"))
            hard_kib = int(quota_row[3].rstrip("*"))
            used_inodes = int(quota_row[5].rstrip("*"))
            hard_inodes = int(quota_row[7].rstrip("*"))
        except ValueError:
            parsed.append(f"{label}={subject}:unparsed")
            continue
        if hard_kib > 0:
            remaining_values.append(max(0, hard_kib - used_kib) * 1024)
        if hard_inodes > 0:
            remaining_inode_values.append(max(0, hard_inodes - used_inodes))
        parsed.append(
            f"{label}={subject}:bytes_hard={hard_kib * 1024 if hard_kib else 0},"
            f"inodes={used_inodes}/{hard_inodes or 0}"
        )
    minimum_bytes = int(float(profile.get("storage", {}).get("minimum_free_tb", 0)) * 1_000_000_000_000)
    minimum_inodes = int(profile.get("storage", {}).get("minimum_free_inodes", 0))
    if remaining_values and min(remaining_values) < minimum_bytes:
        return Check(
            "lustre-quota",
            "FAIL",
            f"remaining hard-quota capacity {min(remaining_values):,} bytes is below {minimum_bytes:,}; " + "; ".join(parsed),
        )
    if remaining_inode_values and min(remaining_inode_values) < minimum_inodes:
        return Check(
            "lustre-quota",
            "FAIL",
            f"remaining hard-quota inodes {min(remaining_inode_values):,} is below {minimum_inodes:,}; "
            + "; ".join(parsed),
        )
    if remaining_values:
        return Check("lustre-quota", "PASS", "; ".join(parsed))
    return _ambiguous_lustre_quota_check(
        profile,
        "no nonzero hard limit was reported; " + "; ".join(parsed),
    )


def _hf_checks(manifest: dict[str, Any], tiny_probe: bool, *, auth_required: bool) -> list[Check]:
    checks: list[Check] = []
    token = get_token() or os.environ.get("HF_TOKEN")
    repositories: dict[tuple[str, str], bool] = {}
    for source in manifest["sources"]:
        access = source.get("access", {})
        if access.get("type") == "huggingface":
            repositories[(str(access["repo_id"]), str(access["revision"]))] = access.get("gated") in {
                "manual", "auto", True
            }
        for component in access.get("components", []):
            repositories[(str(component["repo_id"]), str(component["revision"]))] = component.get("gated") in {
                "manual", "auto", True
            }
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
    for repo_id, revision in sorted(repositories):
        try:
            info = api.dataset_info(repo_id, revision=revision, timeout=30)
            if info.sha != revision:
                raise RuntimeError(f"resolved {info.sha}, expected {revision}")
            checks.append(Check(f"hf:{repo_id}", "PASS", f"revision={revision}"))
        except Exception as exc:  # live service/access failure
            checks.append(Check(f"hf:{repo_id}", "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"))
    return checks


def _network_checks(
    tiny_probe: bool,
    *,
    require_github_auth: bool,
    required_crawls: set[str],
) -> list[Check]:
    if tiny_probe:
        return []
    checks: list[Check] = []
    collection_payload: list[dict[str, Any]] | None = None
    for name, url in (
        ("common-crawl-egress", "https://index.commoncrawl.org/collinfo.json"),
        ("gharchive-egress", "https://data.gharchive.org/2025-01-01-0.json.gz"),
        ("github-codeload-egress", "https://codeload.github.com/octocat/Hello-World/tar.gz/refs/heads/master"),
    ):
        try:
            stream = name in {"gharchive-egress", "github-codeload-egress"}
            response = requests.get(url, timeout=30, stream=stream)
            response.raise_for_status()
            if name == "common-crawl-egress":
                value = response.json()
                if not isinstance(value, list):
                    raise RuntimeError("collinfo.json was not a collection list")
                collection_payload = value
            elif stream:
                prefix = next(response.iter_content(chunk_size=2), b"")
                if prefix != b"\x1f\x8b":
                    raise RuntimeError(f"{name} did not return a gzip archive")
                response.close()
            checks.append(Check(name, "PASS", f"HTTP {response.status_code}"))
        except Exception as exc:
            checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))
    if collection_payload is not None:
        available = {str(entry.get("id")) for entry in collection_payload}
        missing = sorted(required_crawls - available)
        checks.append(
            Check(
                "common-crawl-collections",
                "FAIL" if missing else "PASS",
                f"available={sorted(required_crawls)}" if not missing else f"missing={missing}",
            )
        )
        if required_crawls and not missing:
            crawl = sorted(required_crawls)[-1]
            try:
                listing_url = f"https://data.commoncrawl.org/crawl-data/{crawl}/cc-index-table.paths.gz"
                listing_response = requests.get(listing_url, timeout=30)
                listing_response.raise_for_status()
                paths = [
                    line
                    for line in gzip.decompress(listing_response.content).decode("utf-8").splitlines()
                    if f"crawl={crawl}/subset=warc/" in line
                ]
                if not paths:
                    raise RuntimeError("WARC URL-index listing contained no partitions")
                range_response = requests.get(
                    "https://data.commoncrawl.org/" + paths[0],
                    headers={"Range": "bytes=0-1023", "Accept-Encoding": "identity"},
                    timeout=30,
                )
                content_range = range_response.headers.get("Content-Range", "")
                if range_response.status_code != 206 or not content_range.lower().startswith("bytes 0-1023/"):
                    raise RuntimeError(
                        f"expected HTTP 206 bytes 0-1023, got {range_response.status_code} {content_range!r}"
                    )
                checks.append(Check("common-crawl-range", "PASS", f"{crawl}; {content_range}"))
            except Exception as exc:
                checks.append(Check("common-crawl-range", "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    checks.append(
        Check(
            "github-token",
            "PASS" if github_token else ("FAIL" if require_github_auth else "WARN"),
            (
                "available"
                if github_token
                else (
                    "GITHUB_TOKEN or GH_TOKEN is required by this profile"
                    if require_github_auth
                    else "not set; production acquisition uses public GH Archive and codeload endpoints"
                )
            ),
        )
    )
    return checks


def _operator_role_checks(profile: dict[str, Any], role: str) -> list[Check]:
    operator = profile.get("operator", {})
    roles = {str(value) for value in operator.get("roles", [])}
    checks: list[Check] = []
    requested = {"acquisition", "compute"} if role == "all" else {role}
    unsupported = sorted(requested - roles) if roles else []
    checks.append(
        Check(
            "profile-role",
            "FAIL" if unsupported else "PASS",
            f"allowed={sorted(roles) or ['legacy-any']}; requested={sorted(requested)}"
            + (f"; unsupported={unsupported}" if unsupported else ""),
        )
    )
    patterns = [str(value) for value in operator.get("expected_host_patterns", [])]
    hostname = socket.gethostname()
    if patterns:
        matched = any(fnmatch.fnmatch(hostname, pattern) for pattern in patterns)
        checks.append(
            Check(
                "operator-host",
                "PASS" if matched else "FAIL",
                f"hostname={hostname}; expected one of {patterns}",
            )
        )
    return checks


def _scheduler_site_checks(profile: dict[str, Any], *, required: bool, tiny_probe: bool) -> list[Check]:
    if not required or profile.get("scheduler", {}).get("mode") != "slurm":
        return []
    scheduler = profile["scheduler"]
    checks = [
        Check(
            "scheduler-site-values",
            "PASS" if scheduler.get("site_values_confirmed") else ("WARN" if tiny_probe else "FAIL"),
            "confirmed" if scheduler.get("site_values_confirmed") else "Rhea scheduler facts are still unconfirmed",
        )
    ]
    required_fields = scheduler.get("required_fields", ["account", "partition", "max_array_size"])
    for field in required_fields:
        value = str(scheduler.get(field, "")).strip()
        checks.append(
            Check(
                f"scheduler-{field}",
                ("WARN" if tiny_probe else "FAIL") if not value or value.lower() == "auto" else "PASS",
                value or "unset",
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
            if info.sha != entry["revision"]:
                raise RuntimeError(f"resolved {info.sha}, expected {entry['revision']}")
            if entry.get("files"):
                available_files = {str(item.rfilename) for item in (info.siblings or [])}
                missing_files = sorted(set(map(str, entry["files"])) - available_files)
                if missing_files:
                    raise RuntimeError(f"pinned holdout files are missing: {missing_files}")
                detail = f"revision={info.sha}; files={len(entry['files'])}"
            else:
                if entry.get("jobs"):
                    jobs = [
                        (str(job.get("config") or "default"), str(job["split"]))
                        for job in entry["jobs"]
                    ]
                else:
                    configurations = entry.get("configs") or [entry.get("config")]
                    splits = entry.get("splits") or [entry.get("split")]
                    jobs = [
                        (str(configuration or "default"), str(split))
                        for configuration in configurations
                        for split in splits
                        if split
                    ]
                token = get_token() or os.environ.get("HF_TOKEN")
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                viewer = requests.get(
                    "https://datasets-server.huggingface.co/splits"
                    f"?dataset={quote(str(entry['repo_id']), safe='')}"
                    f"&revision={quote(str(entry['revision']), safe='')}",
                    headers=headers,
                    timeout=60,
                )
                viewer.raise_for_status()
                available_jobs = {
                    (str(row["config"]), str(row["split"]))
                    for row in viewer.json().get("splits", [])
                }
                missing_jobs = [f"{config}:{split}" for config, split in jobs if (config, split) not in available_jobs]
                if missing_jobs:
                    raise RuntimeError(f"pinned holdout jobs are missing: {missing_jobs}")
                detail = f"revision={info.sha}; jobs={len(jobs)}"
            checks.append(Check(f"holdout:{entry['id']}", "PASS", detail))
        except Exception as exc:
            checks.append(Check(f"holdout:{entry['id']}", "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"))
    return checks


# Stages that bucket records by hash and write each bucket through an LRU pool
# of open handles. Where the pool size comes from differs by stage, and getting
# that wrong makes the check itself lie: an earlier version assumed all four
# read maximum_open_files and reported three failures against stages whose
# pools were already correct.
#
# Only repeated_span takes its pool from the profile.
_PROFILE_POOL_WRITERS: tuple[tuple[str, str, str], ...] = (
    ("repeated_span", "finder_tasks", "maximum_open_files"),
)

# These derive it in code as max(32, bucket_count) -- see final_dedup.
# write_final_signatures and code_dedup.write_code_signatures -- so the pool can
# never be smaller than the bucket count and the invariant holds structurally.
# A maximum_open_files key in one of these blocks is read by nothing, which is
# its own trap, so say so rather than let it look effective.
_CODE_POOL_WRITERS: tuple[tuple[str, str], ...] = (
    ("exact_dedup", "find_tasks"),
    ("code_structural", "finder_tasks"),
    ("final_hash", "finder_tasks"),
)


def _bucket_writer_pool_checks(profile: dict[str, Any]) -> list[Check]:
    """A handle pool smaller than its bucket count thrashes, and only that.

    The bucket for a record is a hash, so under an LRU of K handles over N
    buckets the steady-state miss rate is 1 - K/N, and every miss is an open and
    a close. Measured on Portage's Lustre at K=32, N=64: 49.8% of writes missed,
    1.18 ms per open-and-close, 1,706 records/second against 479,779 with the
    pool sized to the buckets. The stage still finished, and still produced
    correct output, which is why this went unnoticed for 11.5 hours.

    Nothing else in the profile says these two numbers are related, so nothing
    else would catch them drifting apart.
    """

    scheduler = profile.get("scheduler", {})
    checks: list[Check] = []
    for stage, bucket_key, pool_key in _PROFILE_POOL_WRITERS:
        block = scheduler.get(stage)
        if not isinstance(block, dict) or bucket_key not in block:
            continue
        buckets = int(block[bucket_key])
        pool = int(block.get(pool_key, 32))
        name = f"writer-pool:{stage}"
        if pool >= buckets:
            checks.append(Check(name, "PASS", f"{pool} handles >= {buckets} buckets"))
        else:
            checks.append(
                Check(
                    name,
                    "FAIL",
                    f"{pool} handles for {buckets} buckets: about "
                    f"{100 * (1 - pool / buckets):.0f}% of writes will reopen a file. "
                    f"Set scheduler.{stage}.{pool_key} to at least {buckets}.",
                )
            )
    for stage, bucket_key in _CODE_POOL_WRITERS:
        block = scheduler.get(stage)
        if not isinstance(block, dict) or bucket_key not in block:
            continue
        buckets = int(block[bucket_key])
        name = f"writer-pool:{stage}"
        if "maximum_open_files" in block:
            checks.append(
                Check(
                    name,
                    "WARN",
                    f"scheduler.{stage}.maximum_open_files is ignored: this stage sizes "
                    f"its pool in code as max(32, {bucket_key}) = {max(32, buckets)}. "
                    "Remove the key so it cannot read as effective.",
                )
            )
        else:
            checks.append(
                Check(name, "PASS", f"pool derived in code: max(32, {buckets}) >= {buckets} buckets")
            )
    return checks


def run_doctor(
    profile: dict[str, Any], *, tiny_probe: bool = False, role: str = "all"
) -> dict[str, Any]:
    checks: list[Check] = []
    compute_checks = role in {"all", "compute"}
    acquisition_checks = role in {"all", "acquisition"}
    checks.extend(_operator_role_checks(profile, role))
    if compute_checks:
        checks.extend(_bucket_writer_pool_checks(profile))
    validation = validate_manifest(profile.get("manifest"))
    checks.append(
        Check(
            "manifest",
            "PASS" if validation.ok else "FAIL",
            f"{len(validation.manifest.get('sources', []))} sources; {len(validation.errors)} errors",
        )
    )
    license_review_complete = bool(profile.get("gates", {}).get("license_review_complete", False))
    # Warn, never fail. The review is of the per-record license ledger, and the
    # ledger is produced *by* the build -- so failing the compute preflight on
    # it blocked the only thing that could generate the evidence it asks for.
    # The gate itself is not weakened: _verify fail-closes on exactly this key
    # immediately before release (stage_runner._verify), which is where the
    # ledger exists and where an unreviewed corpus would actually cause harm.
    license_status = "PASS" if license_review_complete else "WARN"
    checks.append(
        Check(
            "license-review",
            license_status,
            (
                "attested complete"
                if license_review_complete
                else "pending final source/per-record review; candidate acquisition may proceed, but Rhea verification and release remain blocked"
            ),
        )
    )

    storage = profile["storage"]
    root = Path(storage["lustre_root"])
    if not root.exists() and storage.get("must_exist"):
        checks.append(Check("storage-root", "FAIL", f"does not exist: {root}"))
    else:
        root.mkdir(parents=True, exist_ok=True)
        checks.append(Check("storage-root", "PASS", str(root)))
    try:
        with tempfile.NamedTemporaryFile(prefix=".metis-write-probe-", dir=root, delete=True) as handle:
            handle.write(b"metis")
            handle.flush()
            os.fsync(handle.fileno())
        checks.append(Check("lustre-write", "PASS", str(root)))
    except OSError as exc:
        checks.append(Check("lustre-write", "FAIL", str(exc)))

    usage = shutil.disk_usage(root if root.exists() else root.parent)
    free_tb = usage.free / 1_000_000_000_000
    minimum = float(storage.get("minimum_free_tb", 0))
    recommended = float(storage.get("recommended_free_tb", minimum))
    status = "PASS" if free_tb >= minimum else "FAIL"
    detail = f"{free_tb:.2f} TB free; minimum {minimum:.2f} TB; recommended {recommended:.2f} TB"
    if status == "PASS" and free_tb < recommended:
        status = "WARN"
    checks.append(Check("free-space", status, detail))

    fs_type = _filesystem_type(root)
    expected_fs = str(storage.get("expected_filesystem", "")).strip()
    filesystem_status = "PASS" if not expected_fs or fs_type == expected_fs else "FAIL"
    checks.append(Check("filesystem", filesystem_status, f"actual={fs_type}; expected={expected_fs or 'any'}"))
    if expected_fs == "lustre" and root.exists():
        checks.append(_lustre_quota_check(profile, root))

    scheduler_mode = profile.get("scheduler", {}).get("mode", "slurm")
    checks.extend(
        [
            _command_check("git"),
            _python_runtime_check(),
            _command_check("sbatch", required=compute_checks and scheduler_mode == "slurm"),
            _command_check("squeue", required=compute_checks and scheduler_mode == "slurm"),
            _command_check(
                "apptainer",
                required=compute_checks and bool(profile.get("runtime", {}).get("require_apptainer_for_compute")),
            ),
        ]
    )
    if acquisition_checks and profile.get("acquisition", {}).get("mode") == "screen_foreground":
        checks.append(_command_check("screen"))
    if acquisition_checks:
        checks.extend((_python_module_check("hf_xet"), _python_module_check("brotli")))
    checks.extend(_scheduler_site_checks(profile, required=compute_checks, tiny_probe=tiny_probe))
    if acquisition_checks:
        checks.extend(
            _hf_checks(
                validation.manifest,
                tiny_probe=tiny_probe,
                auth_required=bool(profile.get("gates", {}).get("require_hf_auth_for_gated_sources", True)),
            )
        )
        checks.extend(
            _network_checks(
                tiny_probe=tiny_probe,
                require_github_auth=bool(profile.get("gates", {}).get("require_github_auth", False)),
                required_crawls={
                    str(crawl)
                    for source in validation.manifest.get("sources", [])
                    for crawl in source.get("access", {}).get("crawls", [])
                },
            )
        )
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
    if acquisition_checks:
        materializers_enabled = bool(profile.get("runtime", {}).get("dynamic_materializers_enabled", False))
        from .materializers import SUPPORTED_MATERIALIZER_DRIVERS

        registered_drivers = set(SUPPORTED_MATERIALIZER_DRIVERS) | {
            "common_crawl_ranges",
            "github_repositories",
            "github_discussions",
        }
        missing_drivers = sorted(set(dynamic_drivers) - registered_drivers)
        materializers_ready = materializers_enabled and not missing_drivers
        checks.append(
            Check(
                "dynamic-source-materializers",
                "PASS" if materializers_ready else ("WARN" if tiny_probe else "FAIL"),
                (
                    f"registered: {', '.join(dynamic_drivers)}"
                    if materializers_ready
                    else (
                        f"no implementation for: {', '.join(missing_drivers)}"
                        if missing_drivers
                        else f"profile disabled materializers for: {', '.join(dynamic_drivers)}"
                    )
                ),
            )
        )

    if compute_checks and profile.get("gates", {}).get("require_acquisition_handoff"):
        state_root = root / storage["directories"]["state"]
        handoff = state_root / "ACQUISITION_READY.json"
        checks.append(
            Check(
                "acquisition-handoff",
                "PASS" if handoff.is_file() else "FAIL",
                str(handoff),
            )
        )

    contamination_path = repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
    checks.append(Check("contamination-registry", "PASS" if contamination_path.exists() else "FAIL", str(contamination_path)))
    failed = [check for check in checks if check.status == "FAIL"]
    return {
        "ok": not failed,
        "profile": profile.get("name"),
        "role": role,
        "lustre_root": str(root),
        "checks": [asdict(check) for check in checks],
        "manifest_errors": list(validation.errors),
        "manifest_warnings": list(validation.warnings),
    }
