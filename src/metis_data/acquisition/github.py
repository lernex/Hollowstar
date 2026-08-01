from __future__ import annotations

import calendar
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tarfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote

import requests

from .io import (
    JsonlShardWriter,
    RetrySession,
    complete_materialization,
    ensure_validated_download,
    existing_materialization,
    reset_incomplete_materialization,
)
from ..repository_license import (
    DEFAULT_REPOSITORY_LICENSE_ALLOWLIST,
    classify_repository_archive,
)


DEFAULT_LICENSE_ALLOWLIST = DEFAULT_REPOSITORY_LICENSE_ALLOWLIST

TEXT_EXTENSIONS = {
    ".asm", ".bash", ".c", ".cc", ".cfg", ".clj", ".cmake", ".cpp", ".cs", ".css", ".cu",
    ".cuh", ".dart", ".ex", ".exs", ".f", ".f90", ".fs", ".fsx", ".go", ".graphql", ".h",
    ".hpp", ".hs", ".html", ".java", ".jl", ".js", ".json", ".jsx", ".kt", ".kts", ".lean",
    ".lua", ".md", ".mjs", ".mm", ".php", ".pl", ".proto", ".ps1", ".py", ".r", ".rb",
    ".rs", ".scala", ".scss", ".sh", ".sol", ".sql", ".swift", ".tex", ".tf", ".toml",
    ".ts", ".tsx", ".vue", ".xml", ".yaml", ".yml", ".zig",
}

NOISY_PARTS = {
    ".git", ".idea", ".next", ".tox", ".venv", "bower_components", "build", "coverage", "dist",
    "node_modules", "target", "vendor", "vendors", "venv", "__pycache__",
}

NOISY_FILENAMES = {
    "cargo.lock", "composer.lock", "gemfile.lock", "package-lock.json", "pnpm-lock.yaml", "poetry.lock",
    "yarn.lock",
}

GENERATED_MARKERS = re.compile(
    r"(?:do not edit|auto[- ]?generated|generated (?:file|code)|code generated .* do not edit)", re.IGNORECASE
)

DISCUSSION_EVENTS = {
    "IssueCommentEvent", "IssuesEvent", "PullRequestEvent", "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent", "ReleaseEvent",
}

GHARCHIVE_VALIDATOR = "metis-gharchive-jsonl-gzip-v1"
GITHUB_CODELOAD_VALIDATOR = "metis-github-codeload-tar-gzip-v1"
GITHUB_REPOSITORY_IDENTITY_POLICY = "metis-github-repository-identity-v2"
GITHUB_API_VERSION = "2022-11-28"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def month_windows(start: str, end: str) -> list[tuple[date, date]]:
    first = _parse_date(start)
    last = _parse_date(end)
    cursor = first.replace(day=1)
    windows: list[tuple[date, date]] = []
    while cursor <= last:
        month_last = cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1])
        windows.append((max(first, cursor), min(last, month_last)))
        cursor = (month_last + timedelta(days=1)).replace(day=1)
    return windows


def _archive_hours(start: date, end: date) -> Iterator[datetime]:
    cursor = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    while cursor < stop:
        yield cursor
        cursor += timedelta(hours=1)


def _archive_file(client: RetrySession, cache: Path, hour: datetime) -> Path:
    relative = Path(f"{hour:%Y}") / f"{hour:%m}" / f"{hour:%Y-%m-%d-%-H}.json.gz"
    destination = cache / relative
    url = f"https://data.gharchive.org/{hour:%Y-%m-%d-%-H}.json.gz"
    return ensure_validated_download(
        client,
        url,
        destination,
        validator_id=GHARCHIVE_VALIDATOR,
        validator=_validate_gharchive_gzip,
    )


def _validate_gharchive_gzip(path: Path) -> None:
    decompressed = 0
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            decompressed += len(chunk)
    if decompressed <= 0:
        raise RuntimeError(f"GH Archive gzip is empty: {path}")


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def _public_client(profile: dict[str, Any]) -> RetrySession:
    return RetrySession(
        retries=int(profile.get("runtime", {}).get("download_retries", 8)),
        timeout=int(profile.get("runtime", {}).get("request_timeout_seconds", 900)),
    )


def _archive_license(bundle: tarfile.TarFile) -> tuple[str | None, str | None]:
    return classify_repository_archive(bundle)


def _safe_code_member(member: tarfile.TarInfo) -> bool:
    if not member.isfile() or member.size <= 0 or member.size > 2_000_000:
        return False
    path = PurePosixPath(member.name)
    relative = PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else path
    lowered = {part.lower() for part in relative.parts}
    if lowered & NOISY_PARTS or relative.name.lower() in NOISY_FILENAMES:
        return False
    return relative.suffix.lower() in TEXT_EXTENSIONS or relative.name.lower() in {
        "dockerfile", "makefile", "readme", "license",
    }


def _repository_archive(
    client: RetrySession,
    repo_name: str,
    sha: str,
    output_cache: Path,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_name):
        raise ValueError(f"Unsafe GitHub repository name: {repo_name!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"Invalid GitHub commit for {repo_name}: {sha!r}")
    archive = output_cache / repo_name.replace("/", "--") / f"{sha}.tar.gz"
    owner, repository = repo_name.split("/", 1)
    url = (
        f"https://codeload.github.com/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}/tar.gz/{sha}"
    )
    return ensure_validated_download(
        client,
        url,
        archive,
        validator_id=GITHUB_CODELOAD_VALIDATOR,
        validator=_validate_repository_tar,
    )


def _permanent_codeload_miss(error: requests.HTTPError) -> bool:
    response = error.response
    return response is not None and response.status_code in {404, 410, 451}


def _validate_repository_tar(path: Path) -> None:
    members = 0
    regular_files = 0
    with tarfile.open(path, "r:gz") as bundle:
        for member in bundle:
            members += 1
            if not member.isfile():
                continue
            regular_files += 1
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read tar member {member.name!r} from {path}")
            while extracted.read(8 * 1024 * 1024):
                pass
    if members <= 0 or regular_files <= 0:
        raise RuntimeError(f"GitHub codeload archive has no regular files: {path}")


def _repository_rows(
    archive: Path,
    *,
    repo_name: str,
    sha: str,
    commit_date: str,
    license_id: str,
    license_file: str,
    identity: dict[str, Any],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not _safe_code_member(member):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            if b"\0" in raw:
                counters["binary"] += 1
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if len(text) < 80 or GENERATED_MARKERS.search(text[:4096]):
                counters["short_or_generated"] += 1
                continue
            relative = "/".join(PurePosixPath(member.name).parts[1:])
            content_hash = hashlib.sha256(raw).hexdigest()
            yield {
                "id": f"github:{repo_name}:{sha}:{relative}",
                "text": text,
                "repo": repo_name,
                "repo_path": relative,
                "commit_id": sha,
                "commit_date": commit_date,
                "repository_active": True,
                "repository_is_fork": bool(identity["is_fork"]),
                "repository_is_mirror": bool(identity["is_mirror"]),
                "repository_identity_classification": str(identity["classification"]),
                "repository_identity_source": str(identity["source"]),
                "repository_identity_sha256": str(identity["evidence_sha256"]),
                "repository_identity_observed_at": str(identity["observed_at"]),
                "repository_parent": identity.get("parent"),
                "repository_mirror_url": identity.get("mirror_url"),
                "canonical_url": f"https://github.com/{repo_name}/blob/{sha}/{relative}",
                "license": license_id,
                "license_file": license_file,
                "license_basis": "repository-license-file",
                "content_sha256": content_hash,
                "language_probability": 1.0,
                "generated_file_probability": 0.0,
                "source_document_id": f"github:{repo_name}:{sha}",
            }


def _discussion_fragments(event: dict[str, Any]) -> Iterator[tuple[str, str, str | None]]:
    payload = event.get("payload") or {}
    event_type = str(event.get("type") or "")
    objects: list[tuple[str, Any]] = []
    for key in ("issue", "pull_request", "comment", "review", "release"):
        value = payload.get(key)
        if isinstance(value, dict):
            objects.append((key, value))
    for kind, value in objects:
        title = str(value.get("title") or "").strip()
        body = str(value.get("body") or "").strip()
        if not body or len(body) < 120:
            continue
        text = f"{title}\n\n{body}".strip() if title else body
        html_url = value.get("html_url")
        yield kind, text, str(html_url) if html_url else None


def _repository_name(value: Any) -> str | None:
    candidate = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate):
        return None
    return candidate


def _nested_repository_name(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    direct = _repository_name(value.get("full_name") or value.get("nameWithOwner"))
    if direct:
        return direct
    owner = value.get("owner")
    owner_name = (
        str(owner.get("login") or "").strip()
        if isinstance(owner, dict)
        else str(owner or "").strip()
    )
    name = str(value.get("name") or "").strip()
    return _repository_name(f"{owner_name}/{name}") if owner_name and name else None


def _identity_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repository_identity_evidence(
    repository: Any,
    *,
    source: str,
    observed_at: str,
    expected_repo: str | None = None,
) -> dict[str, Any] | None:
    """Normalize explicit GitHub repository identity metadata.

    GH Archive's top-level ``repo`` object normally contains only an id, name,
    and URL.  Absence of fork/mirror fields is therefore *not* negative
    evidence.  Only full repository objects carrying a boolean ``fork`` and an
    explicit ``mirror_url`` field are accepted here.
    """

    if not isinstance(repository, dict):
        return None
    repo_name = _nested_repository_name(repository)
    if repo_name is None:
        return None
    if expected_repo and repo_name.casefold() != expected_repo.casefold():
        return None
    fork_value = repository.get("fork")
    if not isinstance(fork_value, bool) or "mirror_url" not in repository:
        return None
    raw_mirror = repository.get("mirror_url")
    if raw_mirror is not None and not isinstance(raw_mirror, str):
        return None
    mirror_url = raw_mirror.strip() if isinstance(raw_mirror, str) else None
    parent = _nested_repository_name(repository.get("parent"))
    upstream = _nested_repository_name(repository.get("source"))
    is_fork = bool(fork_value)
    is_mirror = bool(mirror_url)
    if is_fork and is_mirror:
        classification = "fork_and_mirror"
    elif is_fork:
        classification = "fork"
    elif is_mirror:
        classification = "mirror"
    else:
        classification = "canonical"
    normalized = {
        "repo": repo_name,
        "classification": classification,
        "is_fork": is_fork,
        "is_mirror": is_mirror,
        "mirror_url": mirror_url,
        "parent": parent or upstream,
        "source": source,
        "observed_at": observed_at,
    }
    normalized["evidence_sha256"] = _identity_digest(normalized)
    return normalized


def _event_repository_identities(event: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only explicit repository objects embedded in a GH Archive event."""

    event_type = str(event.get("type") or "")
    event_id = str(event.get("id") or "")
    observed_at = str(event.get("created_at") or "")
    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    candidates: list[tuple[str, Any]] = [
        (f"gharchive:{event_type}:repo:{event_id}", event.get("repo")),
        (f"gharchive:{event_type}:payload.repository:{event_id}", payload.get("repository")),
    ]
    if event_type == "ForkEvent":
        candidates.append(
            (f"gharchive:{event_type}:payload.forkee:{event_id}", payload.get("forkee"))
        )
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict):
        for side in ("base", "head"):
            reference = pull_request.get(side)
            repository = reference.get("repo") if isinstance(reference, dict) else None
            candidates.append(
                (
                    f"gharchive:{event_type}:payload.pull_request.{side}.repo:{event_id}",
                    repository,
                )
            )
    emitted: set[tuple[str, str]] = set()
    for source, repository in candidates:
        evidence = _repository_identity_evidence(
            repository,
            source=source,
            observed_at=observed_at,
        )
        if evidence is None:
            continue
        key = (str(evidence["repo"]).casefold(), str(evidence["evidence_sha256"]))
        if key in emitted:
            continue
        emitted.add(key)
        yield evidence


def _merge_repository_identity(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    if existing is None:
        return incoming
    existing_rejected = bool(existing["is_fork"] or existing["is_mirror"])
    incoming_rejected = bool(incoming["is_fork"] or incoming["is_mirror"])
    if existing_rejected and not incoming_rejected:
        return existing
    if incoming_rejected and not existing_rejected:
        return incoming
    if existing["classification"] == incoming["classification"]:
        return min(
            (existing, incoming),
            key=lambda value: (
                str(value["observed_at"]),
                str(value["source"]),
                str(value["evidence_sha256"]),
            ),
        )
    combined = {
        "repo": existing["repo"],
        "classification": "fork_and_mirror",
        "is_fork": bool(existing["is_fork"] or incoming["is_fork"]),
        "is_mirror": bool(existing["is_mirror"] or incoming["is_mirror"]),
        "mirror_url": existing.get("mirror_url") or incoming.get("mirror_url"),
        "parent": existing.get("parent") or incoming.get("parent"),
        "source": "multiple-explicit-github-observations",
        "observed_at": min(
            str(existing.get("observed_at") or ""),
            str(incoming.get("observed_at") or ""),
        ),
    }
    combined["evidence_sha256"] = _identity_digest(
        {
            **combined,
            "inputs": sorted(
                [str(existing["evidence_sha256"]), str(incoming["evidence_sha256"])]
            ),
        }
    )
    return combined


def _metadata_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS repository_identity ("
        "repo TEXT PRIMARY KEY COLLATE NOCASE, classification TEXT NOT NULL, "
        "is_fork INTEGER NOT NULL, is_mirror INTEGER NOT NULL, mirror_url TEXT, "
        "parent TEXT, source TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, "
        "observed_at TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def _identity_from_row(row: sqlite3.Row) -> dict[str, Any]:
    identity = {
        "repo": str(row["repo"]),
        "classification": str(row["classification"]),
        "is_fork": bool(row["is_fork"]),
        "is_mirror": bool(row["is_mirror"]),
        "mirror_url": str(row["mirror_url"]) if row["mirror_url"] else None,
        "parent": str(row["parent"]) if row["parent"] else None,
        "source": str(row["source"]),
        "evidence_sha256": str(row["evidence_sha256"]),
        "observed_at": str(row["observed_at"]),
    }
    evidence_sha256 = identity.pop("evidence_sha256")
    expected = _identity_digest(identity)
    identity["evidence_sha256"] = evidence_sha256
    if evidence_sha256 != expected:
        raise RuntimeError(
            "Cached GitHub repository identity evidence failed its SHA-256 "
            f"integrity check for {identity['repo']}"
        )
    return identity


def _cache_repository_identity(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT INTO repository_identity("
        "repo, classification, is_fork, is_mirror, mirror_url, parent, source, "
        "evidence_sha256, observed_at) VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(repo) DO NOTHING",
        (
            identity["repo"],
            identity["classification"],
            int(bool(identity["is_fork"])),
            int(bool(identity["is_mirror"])),
            identity.get("mirror_url"),
            identity.get("parent"),
            identity["source"],
            identity["evidence_sha256"],
            identity["observed_at"],
        ),
    )
    connection.commit()


def _resolve_repository_identity(
    client: RetrySession,
    connection: sqlite3.Connection,
    repo_name: str,
) -> dict[str, Any]:
    cached = connection.execute(
        "SELECT * FROM repository_identity WHERE repo = ?", (repo_name,)
    ).fetchone()
    if cached is not None:
        return _identity_from_row(cached)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError(
            "FreshGitHub cannot prove that repository "
            f"{repo_name!r} is canonical. Set GITHUB_TOKEN or GH_TOKEN to a "
            "read-only GitHub token; unknown fork/mirror status is never accepted."
        )
    owner, repository = repo_name.split("/", 1)
    url = (
        f"https://api.github.com/repos/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}"
    )
    response = client.request(
        "GET",
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    if response.status_code in {404, 410, 451}:
        observed_at = str(response.headers.get("Date") or "")
        response.close()
        unavailable = {
            "repo": repo_name,
            "classification": "unavailable",
            "is_fork": False,
            "is_mirror": False,
            "mirror_url": None,
            "parent": None,
            "source": f"github-rest-v3:{response.status_code}",
            "observed_at": observed_at,
        }
        unavailable["evidence_sha256"] = _identity_digest(unavailable)
        _cache_repository_identity(connection, unavailable)
        return unavailable
    try:
        response.raise_for_status()
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"GitHub repository metadata was not valid JSON for {repo_name}"
            ) from exc
    finally:
        response.close()
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub repository metadata was not an object for {repo_name}")
    observed_at = str(response.headers.get("Date") or payload.get("updated_at") or "")
    identity = _repository_identity_evidence(
        payload,
        source=f"github-rest-v3:{GITHUB_API_VERSION}",
        observed_at=observed_at,
        expected_repo=repo_name,
    )
    if identity is None:
        raise RuntimeError(
            f"GitHub repository metadata omitted explicit fork/mirror fields for {repo_name}"
        )
    _cache_repository_identity(connection, identity)
    return identity


def _identity_rejection_reason(identity: dict[str, Any]) -> str | None:
    if identity["classification"] == "unavailable":
        return "repository_identity_unavailable"
    if identity["is_fork"] and identity["is_mirror"]:
        return "repository_fork_and_mirror"
    if identity["is_fork"]:
        return "repository_fork"
    if identity["is_mirror"]:
        return "repository_mirror"
    if identity["classification"] != "canonical":
        return "repository_identity_unresolved"
    return None


def _activity_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS activity ("
        "repo TEXT PRIMARY KEY, events INTEGER NOT NULL, last_event TEXT NOT NULL, head_sha TEXT, "
        "identity_classification TEXT, identity_source TEXT, identity_sha256 TEXT, "
        "identity_observed_at TEXT, identity_parent TEXT, identity_mirror_url TEXT)"
    )
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(activity)")}
    if "head_sha" not in columns:
        connection.execute("ALTER TABLE activity ADD COLUMN head_sha TEXT")
    for name in (
        "identity_classification",
        "identity_source",
        "identity_sha256",
        "identity_observed_at",
        "identity_parent",
        "identity_mirror_url",
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE activity ADD COLUMN {name} TEXT")
    connection.commit()
    return connection


def _policy_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS repositories ("
        "repo TEXT PRIMARY KEY, license TEXT NOT NULL, license_file TEXT NOT NULL, "
        "commit_sha TEXT NOT NULL, last_event TEXT NOT NULL, "
        "repository_is_fork INTEGER NOT NULL DEFAULT 0, "
        "repository_is_mirror INTEGER NOT NULL DEFAULT 0, "
        "identity_source TEXT, identity_sha256 TEXT, identity_observed_at TEXT, "
        "identity_parent TEXT, identity_mirror_url TEXT)"
    )
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(repositories)")}
    additions = {
        "repository_is_fork": "INTEGER NOT NULL DEFAULT 0",
        "repository_is_mirror": "INTEGER NOT NULL DEFAULT 0",
        "identity_source": "TEXT",
        "identity_sha256": "TEXT",
        "identity_observed_at": "TEXT",
        "identity_parent": "TEXT",
        "identity_mirror_url": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE repositories ADD COLUMN {name} {declaration}")
    connection.commit()
    return connection


def materialize_github(
    item: dict[str, Any], *, profile: dict[str, Any], root: Path
) -> dict[str, Any]:
    driver = str(item["driver"])
    if driver not in {"github_repositories", "github_discussions"}:
        raise ValueError(f"Unsupported GitHub driver: {driver}")
    access = item["access"]
    partition = item.get("partition") or {
        "start": access["cutoff_start"],
        "end": access["cutoff_end"],
        "id": f"{access['cutoff_start']}_{access['cutoff_end']}",
    }
    source_id = str(item.get("source_id") or "")
    partition_id = str(partition.get("id") or "")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id)
        or source_id in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", partition_id)
        or partition_id in {".", ".."}
    ):
        raise ValueError("GitHub materialization contains an unsafe source or partition identifier")
    assigned_root = root.expanduser().resolve()
    output = (assigned_root / "raw" / source_id / partition_id).resolve()
    try:
        output.relative_to(assigned_root / "raw")
    except ValueError as exc:
        raise ValueError("GitHub materialization output escapes the assigned root") from exc
    existing = existing_materialization(output)
    if existing:
        receipt = json.loads(Path(existing["receipt"]).read_text(encoding="utf-8"))
        if (
            receipt.get("repository_identity_policy")
            != GITHUB_REPOSITORY_IDENTITY_POLICY
        ):
            raise RuntimeError(
                "The completed FreshGitHub partition predates strict fork/mirror "
                f"proof ({GITHUB_REPOSITORY_IDENTITY_POLICY}). Remove and rebuild "
                f"the partition explicitly: {output}"
            )
        return existing
    # A monthly materializer has multiple durability domains: SQLite state,
    # compressed shard writers, and the final receipt.  Only the receipt commits
    # the month.  Rebuild any uncommitted month from checksum-validated caches so
    # a crash can never leave dedup state ahead of durable output.
    reset_incomplete_materialization(output)
    archive_client = _public_client(profile)
    cache = root / profile["storage"]["directories"].get("cache", "cache")
    gharchive = cache / "gharchive"
    repo_archives = cache / "github-repositories"
    policy = _policy_database(cache / "github-policy" / "repositories.sqlite3")
    metadata = _metadata_database(
        cache / "github-policy" / "repository-identity.sqlite3"
    )
    counters: Counter[str] = Counter()
    target_tokens = int(item.get("candidate_tokens", 0))
    # Per-run ceiling on repository archive fetches. The candidate list is
    # every repository with activity in the window -- millions per month -- and
    # the loop below only stops early when the candidate target is met. On a
    # host where a codeload archive costs about a gigabyte and roughly one in
    # twelve survives the root-license gate, that target is unreachable and the
    # walk does not terminate, which blocks acquisition from ever completing.
    # Bounding the fetch converts an open-ended walk into a fixed contribution
    # and lets the ordinary shortfall path route the remainder to donors.
    # -1 leaves the walk unbounded.
    maximum_archive_fetches = int(
        profile.get("acquisition", {}).get(
            "github_repository_maximum_archive_fetches", -1
        )
    )
    writer = JsonlShardWriter(output)

    if driver == "github_discussions":
        seen = sqlite3.connect(output / "seen.sqlite3")
        seen.execute("PRAGMA journal_mode=WAL")
        seen.execute("CREATE TABLE IF NOT EXISTS documents (digest TEXT PRIMARY KEY)")
        license_cache: dict[str, tuple[Any, ...] | None] = {}
        try:
            for hour in _archive_hours(_parse_date(partition["start"]), _parse_date(partition["end"])):
                path = _archive_file(archive_client, gharchive, hour)
                counters["archives"] += 1
                for event in _iter_events(path):
                    if event.get("type") not in DISCUSSION_EVENTS:
                        continue
                    repo_name = str((event.get("repo") or {}).get("name") or "")
                    if not repo_name:
                        continue
                    if repo_name not in license_cache:
                        row = policy.execute(
                            "SELECT license, license_file, repository_is_fork, "
                            "repository_is_mirror, identity_source, identity_sha256, "
                            "identity_observed_at, identity_parent, identity_mirror_url "
                            "FROM repositories WHERE repo = ?",
                            (repo_name,),
                        ).fetchone()
                        license_cache[repo_name] = (
                            tuple(row) if row else None
                        )
                    license_record = license_cache[repo_name]
                    if not license_record:
                        counters["repository_not_in_reviewed_code_slice"] += 1
                        continue
                    (
                        license_id,
                        license_file,
                        repository_is_fork,
                        repository_is_mirror,
                        identity_source,
                        identity_sha256,
                        identity_observed_at,
                        identity_parent,
                        identity_mirror_url,
                    ) = license_record
                    if (
                        bool(repository_is_fork)
                        or bool(repository_is_mirror)
                        or not identity_source
                        or not identity_sha256
                    ):
                        counters["repository_identity_not_canonical"] += 1
                        continue
                    identity_row = metadata.execute(
                        "SELECT * FROM repository_identity WHERE repo = ?",
                        (repo_name,),
                    ).fetchone()
                    if identity_row is None:
                        counters["repository_identity_evidence_missing"] += 1
                        continue
                    cached_identity = _identity_from_row(identity_row)
                    if (
                        _identity_rejection_reason(cached_identity)
                        or str(cached_identity["evidence_sha256"])
                        != str(identity_sha256)
                        or str(cached_identity["source"]) != str(identity_source)
                    ):
                        counters["repository_identity_evidence_mismatch"] += 1
                        continue
                    for kind, text, url in _discussion_fragments(event):
                        created_at = str(event.get("created_at") or "")
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        inserted = seen.execute(
                            "INSERT OR IGNORE INTO documents(digest) VALUES (?)", (digest,)
                        ).rowcount
                        if not inserted:
                            counters["exact_duplicate"] += 1
                            continue
                        writer.write(
                            {
                                "id": f"github-discussion:{digest}",
                                "text": text,
                                "repo": repo_name,
                                "discussion_kind": kind,
                                "publication_date": created_at,
                                "capture_date": created_at,
                                "canonical_url": url or f"https://github.com/{repo_name}",
                                "license": license_id,
                                "license_file": license_file,
                                "license_basis": "associated-repository-license-file; final-review-required",
                                "repository_is_fork": False,
                                "repository_is_mirror": False,
                                "repository_identity_classification": "canonical",
                                "repository_identity_source": str(identity_source),
                                "repository_identity_sha256": str(identity_sha256),
                                "repository_identity_observed_at": str(
                                    identity_observed_at or ""
                                ),
                                "repository_parent": (
                                    str(identity_parent) if identity_parent else None
                                ),
                                "repository_mirror_url": (
                                    str(identity_mirror_url)
                                    if identity_mirror_url
                                    else None
                                ),
                                "technical_context": True,
                                "language_probability": 1.0,
                                "content_sha256": digest,
                            }
                        )
                        counters["accepted"] += 1
                    seen.commit()
                    if target_tokens and writer.text_characters // 4 >= target_tokens:
                        break
                if target_tokens and writer.text_characters // 4 >= target_tokens:
                    break
        finally:
            seen.close()
    else:
        database = output / "activity.sqlite3"
        connection = _activity_database(database)
        try:
            for hour in _archive_hours(_parse_date(partition["start"]), _parse_date(partition["end"])):
                path = _archive_file(archive_client, gharchive, hour)
                counters["archives"] += 1
                pending: dict[str, tuple[int, str, str | None]] = {}
                pending_identities: dict[str, dict[str, Any]] = {}
                for event in _iter_events(path):
                    for identity in _event_repository_identities(event):
                        identity_repo = str(identity["repo"])
                        pending_identities[identity_repo] = _merge_repository_identity(
                            pending_identities.get(identity_repo),
                            identity,
                        )
                    if event.get("type") not in {"PushEvent", "CreateEvent", "ReleaseEvent", "PullRequestEvent"}:
                        continue
                    repo_name = str((event.get("repo") or {}).get("name") or "")
                    created = str(event.get("created_at") or "")
                    if repo_name:
                        count, latest, head = pending.get(repo_name, (0, "", None))
                        candidate_head = str((event.get("payload") or {}).get("head") or "")
                        if not re.fullmatch(r"[0-9a-f]{40}", candidate_head):
                            candidate_head = ""
                        if created >= latest:
                            latest = created
                            head = candidate_head or head
                        pending[repo_name] = (count + 1, latest, head)
                connection.executemany(
                    "INSERT INTO activity(repo, events, last_event, head_sha) VALUES(?,?,?,?) "
                    "ON CONFLICT(repo) DO UPDATE SET events=events+excluded.events, "
                    "last_event=MAX(last_event, excluded.last_event), "
                    "head_sha=COALESCE(excluded.head_sha, activity.head_sha)",
                    ((repo, count, latest, head) for repo, (count, latest, head) in pending.items()),
                )
                for identity_repo, identity in sorted(
                    pending_identities.items(), key=lambda value: value[0].casefold()
                ):
                    previous = connection.execute(
                        "SELECT identity_classification, identity_source, "
                        "identity_sha256, identity_observed_at, identity_parent, "
                        "identity_mirror_url FROM activity WHERE repo = ?",
                        (identity_repo,),
                    ).fetchone()
                    if previous and previous[0]:
                        previous_classification = str(previous[0])
                        identity = _merge_repository_identity(
                            {
                                "repo": identity_repo,
                                "classification": previous_classification,
                                "is_fork": previous_classification
                                in {"fork", "fork_and_mirror"},
                                "is_mirror": previous_classification
                                in {"mirror", "fork_and_mirror"},
                                "source": str(previous[1]),
                                "evidence_sha256": str(previous[2]),
                                "observed_at": str(previous[3]),
                                "parent": previous[4],
                                "mirror_url": previous[5],
                            },
                            identity,
                        )
                    connection.execute(
                        "INSERT INTO activity("
                        "repo, events, last_event, head_sha, identity_classification, "
                        "identity_source, identity_sha256, identity_observed_at, "
                        "identity_parent, identity_mirror_url"
                        ") VALUES(?,0,?,NULL,?,?,?,?,?,?) "
                        "ON CONFLICT(repo) DO UPDATE SET "
                        "identity_classification=excluded.identity_classification, "
                        "identity_source=excluded.identity_source, "
                        "identity_sha256=excluded.identity_sha256, "
                        "identity_observed_at=excluded.identity_observed_at, "
                        "identity_parent=excluded.identity_parent, "
                        "identity_mirror_url=excluded.identity_mirror_url",
                        (
                            identity_repo,
                            str(identity["observed_at"]),
                            str(identity["classification"]),
                            str(identity["source"]),
                            str(identity["evidence_sha256"]),
                            str(identity["observed_at"]),
                            identity.get("parent"),
                            identity.get("mirror_url"),
                        ),
                    )
                connection.commit()
            cursor = connection.execute(
                "SELECT repo, events, last_event, head_sha, identity_classification, "
                "identity_source, identity_sha256, identity_observed_at, "
                "identity_parent, identity_mirror_url "
                "FROM activity WHERE head_sha IS NOT NULL "
                "ORDER BY events DESC, repo ASC"
            )
            for (
                repo_name,
                events,
                last_event,
                head_sha,
                archive_identity_classification,
                archive_identity_source,
                archive_identity_sha256,
                archive_identity_observed_at,
                archive_identity_parent,
                archive_identity_mirror_url,
            ) in cursor:
                if archive_identity_classification in {
                    "fork",
                    "mirror",
                    "fork_and_mirror",
                }:
                    counters[f"repository_{archive_identity_classification}"] += 1
                    continue
                identity = _resolve_repository_identity(
                    archive_client,
                    metadata,
                    str(repo_name),
                )
                rejection_reason = _identity_rejection_reason(identity)
                if rejection_reason:
                    counters[rejection_reason] += 1
                    continue
                if (
                    maximum_archive_fetches >= 0
                    and counters["archive_fetches"] >= maximum_archive_fetches
                ):
                    counters["stopped_at_archive_fetch_budget"] += 1
                    break
                counters["archive_fetches"] += 1
                try:
                    archive = _repository_archive(
                        archive_client, str(repo_name), str(head_sha), repo_archives
                    )
                    with tarfile.open(archive, "r:gz") as bundle:
                        license_id, license_file = _archive_license(bundle)
                except ValueError:
                    counters["unsafe_repository_identity"] += 1
                    continue
                except requests.HTTPError as exc:
                    if _permanent_codeload_miss(exc):
                        counters["repository_permanently_unavailable"] += 1
                        continue
                    raise
                if license_id not in DEFAULT_LICENSE_ALLOWLIST or not license_file:
                    counters["repository_policy"] += 1
                    continue
                if last_event and not (str(access["cutoff_start"]) <= str(last_event)[:10] <= str(access["cutoff_end"])):
                    counters["outside_cutoff"] += 1
                    continue
                accepted_before = writer.records
                for row in _repository_rows(
                    archive,
                    repo_name=str(repo_name),
                    sha=str(head_sha),
                    commit_date=str(last_event),
                    license_id=str(license_id),
                    license_file=str(license_file),
                    identity=identity,
                    counters=counters,
                ):
                    row["repository_activity_events"] = int(events)
                    writer.write(row)
                    counters["accepted_files"] += 1
                if writer.records == accepted_before:
                    counters["empty_repository_after_filters"] += 1
                    continue
                policy.execute(
                    "INSERT INTO repositories("
                    "repo, license, license_file, commit_sha, last_event, "
                    "repository_is_fork, repository_is_mirror, identity_source, "
                    "identity_sha256, identity_observed_at, identity_parent, "
                    "identity_mirror_url"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(repo) DO UPDATE SET license=excluded.license, license_file=excluded.license_file, "
                    "commit_sha=excluded.commit_sha, last_event=excluded.last_event, "
                    "repository_is_fork=excluded.repository_is_fork, "
                    "repository_is_mirror=excluded.repository_is_mirror, "
                    "identity_source=excluded.identity_source, "
                    "identity_sha256=excluded.identity_sha256, "
                    "identity_observed_at=excluded.identity_observed_at, "
                    "identity_parent=excluded.identity_parent, "
                    "identity_mirror_url=excluded.identity_mirror_url",
                    (
                        str(repo_name),
                        str(license_id),
                        str(license_file),
                        str(head_sha),
                        str(last_event),
                        0,
                        0,
                        str(identity["source"]),
                        str(identity["evidence_sha256"]),
                        str(identity["observed_at"]),
                        identity.get("parent"),
                        identity.get("mirror_url"),
                    ),
                )
                policy.commit()
                counters["accepted_repositories"] += 1
                if target_tokens and writer.text_characters // 4 >= target_tokens:
                    break
        finally:
            connection.close()

    policy.close()
    metadata.close()

    return complete_materialization(
        output,
        source_id=source_id,
        driver=driver,
        writer=writer,
        receipt={
            "partition": partition,
            "candidate_token_target": target_tokens,
            "cutoff_start": access["cutoff_start"],
            "cutoff_end": access["cutoff_end"],
            "incomplete_recovery": "rebuild_from_validated_cache",
            "gharchive_validator": GHARCHIVE_VALIDATOR,
            "codeload_validator": (
                GITHUB_CODELOAD_VALIDATOR
                if driver == "github_repositories"
                else None
            ),
            "license_allowlist": sorted(DEFAULT_LICENSE_ALLOWLIST),
            "repository_identity_policy": GITHUB_REPOSITORY_IDENTITY_POLICY,
            "repository_identity_acceptance": (
                "explicit canonical GitHub REST metadata cached before codeload"
            ),
            "repository_identity_unknowns_accepted": 0,
            "counters": dict(counters),
        },
    )
