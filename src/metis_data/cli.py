from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import load_profile, repository_root
from .doctor import run_doctor
from .download import run_download_task
from .manifest import candidate_plan, dump_json, validate_manifest
from .holdouts import prepare_holdouts
from .handoff import verify_acquisition_handoff, write_acquisition_handoff
from .reporting import report, status
from .slurm import submit_graph
from .source_lock import _repository_commit, resolve_sources
from .state import StateStore, utc_now
from .training_contract import validate_training_release
from .local_download import launch_local_download
from .local_download import run_local_download_supervisor


def _context(profile_name: str) -> tuple[Path, dict[str, Any], dict[str, Any], StateStore]:
    profile_path, profile = load_profile(profile_name)
    manifest_path = profile.get("manifest", "manifests/metis-1.6.yaml")
    candidate = Path(manifest_path)
    if not candidate.is_absolute():
        candidate = repository_root() / candidate
    validation = validate_manifest(candidate)
    manifest = validation.require_valid()
    root = Path(profile["storage"]["lustre_root"])
    state = StateStore(root / profile["storage"]["directories"]["state"])
    return profile_path, profile, manifest, state


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _profile_roles(profile: dict[str, Any]) -> set[str]:
    return {str(role) for role in profile.get("operator", {}).get("roles", [])}


def _require_profile_role(profile: dict[str, Any], role: str) -> None:
    roles = _profile_roles(profile)
    if roles and role not in roles:
        raise RuntimeError(
            f"Profile {profile.get('name')} is restricted to {sorted(roles)} and cannot run the {role} role"
        )


def _require_preflight(profile: dict[str, Any], role: str) -> None:
    _require_profile_role(profile, role)
    result = run_doctor(profile, tiny_probe=False, role=role)
    failed = [check["name"] for check in result["checks"] if check["status"] == "FAIL"]
    if failed:
        raise RuntimeError(
            f"{role.title()} preflight failed: "
            + ", ".join(failed)
            + f". Run `metisctl doctor --profile {profile.get('name')} --role {role}` for details."
        )


def _require_acquisition_preflight(profile: dict[str, Any]) -> None:
    _require_preflight(profile, "acquisition")


def cmd_init(args: argparse.Namespace) -> int:
    profile_path, profile, manifest, state = _context(args.profile)
    root = Path(profile["storage"]["lustre_root"])
    for relative in profile["storage"]["directories"].values():
        (root / relative).mkdir(parents=True, exist_ok=True)
    state.write(
        "OPERATOR.json",
        payload={
            "schema": "metis.operator-state/v1",
            "initialized_at": utc_now(),
            "profile": profile["name"],
            "profile_path": str(profile_path),
            "manifest_path": manifest["_path"],
            "release": manifest["release"],
            "repository_commit": os.environ.get("METIS_REPOSITORY_COMMIT", "working-tree"),
        },
    )
    _print({"ok": True, "lustre_root": str(root), "state": str(state.root)})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_manifest(args.manifest)
    _print(
        {
            "ok": result.ok,
            "release": result.manifest.get("release"),
            "sources": len(result.manifest.get("sources", [])),
            "errors": result.errors,
            "warnings": result.warnings,
            "candidate_plan": candidate_plan(result.manifest) if result.ok else None,
        }
    )
    return 0 if result.ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    _, _, manifest, _ = _context(args.profile)
    _print(candidate_plan(manifest))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    _, profile, _, _ = _context(args.profile)
    payload = run_doctor(profile, tiny_probe=args.tiny_probe, role=args.role)
    for check in payload["checks"]:
        print(f"{check['status']:<4}  {check['name']:<34} {check['detail']}")
    return 0 if payload["ok"] else 1


def cmd_resolve(args: argparse.Namespace) -> int:
    _, profile, manifest, state = _context(args.profile)
    lock = resolve_sources(manifest, profile, state)
    _print(
        {
            "ok": True,
            "sources": len(lock["sources"]),
            "download_tasks": len(lock["download_tasks"]),
            "lock": str(state.path("sources.lock.json")),
        }
    )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    profile_path, profile, manifest, state = _context(args.profile)
    acquisition_mode = profile.get("acquisition", {}).get("mode", "slurm")
    split_mode = acquisition_mode in {"local_detached", "screen_foreground", "external_complete"}
    if args.target == "pipeline" and split_mode:
        raise RuntimeError(
            "This site uses split execution: run `metisctl submit download` on the Lustre server, "
            "wait for the acquisition handoff, then run `metisctl submit build` from Rhea."
        )
    if args.target in {"download", "pipeline"} and acquisition_mode in {"local_detached", "screen_foreground"} and not args.dry_run:
        _require_acquisition_preflight(profile)
    # Resolving is also the immutable-lock verification boundary on resume.
    # It performs no Hub work when a valid lock already exists.
    resolve_sources(manifest, profile, state)
    include_download = args.target in {"download", "pipeline"}
    include_build = args.target in {"build", "pipeline"}
    if include_download:
        _require_profile_role(profile, "acquisition")
    if include_build:
        _require_profile_role(profile, "compute")
    if include_build and not args.dry_run and not profile.get("runtime", {}).get("dynamic_materializers_enabled", False):
        raise RuntimeError(
            "Production build is gated: dynamic acquisition materializers are disabled in this profile."
        )
    if include_build and not args.dry_run:
        _require_preflight(profile, "compute")
        if profile.get("gates", {}).get("require_acquisition_handoff"):
            verify_acquisition_handoff(
                profile,
                manifest,
                state,
                # The Rhea graph hashes the frozen payload in a restartable
                # Slurm array. Submission performs only the fast structural
                # handoff checks so it returns promptly.
                verify_artifact_hashes=False,
            )
        download_status = status(profile, state)["download"]
        if not download_status["build_ready"]:
            raise RuntimeError(
                "Acquisition is not build-ready: downloads, dynamic materialization, and evaluation holdouts "
                "must all be complete on the Lustre server."
            )
        holdouts = Path(profile["storage"]["lustre_root"]) / profile["storage"]["directories"]["contamination"] / "holdouts.jsonl"
        if not holdouts.exists():
            raise RuntimeError(
                "Evaluation holdouts are missing. Complete `metisctl submit download` on the Lustre server first."
            )
    if include_download and not include_build and acquisition_mode == "screen_foreground":
        raise RuntimeError(
            "This profile runs acquisition in GNU Screen. Use `./ops/start-acquisition.sh --lustre-root PATH`; "
            "it invokes the foreground supervisor without double-detaching."
        )
    if include_download and not include_build and acquisition_mode == "local_detached":
        if args.dry_run:
            _print(
                {
                    "dry_run": True,
                    "mode": "local_detached",
                    "host_role": "lustre_server",
                    "max_workers": int(profile.get("acquisition", {}).get("max_workers", 8)),
                    "pending_tasks": len(state.read("sources.lock.json")["download_tasks"]),
                }
            )
        else:
            _print(launch_local_download(profile_path, profile, state))
        return 0
    payload = submit_graph(
        profile_path=profile_path,
        profile=profile,
        state=state,
        include_download=include_download,
        include_build=include_build,
        dry_run=args.dry_run,
    )
    _print(payload)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _, profile, _, state = _context(args.profile)
    _print(status(profile, state))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    _, profile, manifest, state = _context(args.profile)
    _print(report(profile, manifest, state))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    profile_path, profile, manifest, state = _context(args.profile)
    acquisition_mode = profile.get("acquisition", {}).get("mode", "slurm")
    split_mode = acquisition_mode in {"local_detached", "screen_foreground", "external_complete"}
    if args.target == "pipeline" and split_mode:
        raise RuntimeError("Resume download and build separately on their respective hosts")
    if args.target in {"download", "pipeline"} and acquisition_mode in {"local_detached", "screen_foreground"} and not args.dry_run:
        _require_acquisition_preflight(profile)
    # Resolving is also the immutable-lock verification boundary on resume.
    # It performs no Hub work when a valid lock already exists.
    resolve_sources(manifest, profile, state)
    if (
        args.target in {"build", "pipeline"}
        and not args.dry_run
        and not profile.get("runtime", {}).get("dynamic_materializers_enabled", False)
    ):
        raise RuntimeError("Cannot resume the build until dynamic source materializers are connected")
    if args.target == "download" and acquisition_mode == "screen_foreground":
        raise RuntimeError(
            "Rerun `./ops/start-acquisition.sh --lustre-root PATH`; completed tasks will be skipped safely"
        )
    if args.target == "download" and acquisition_mode == "local_detached":
        if args.dry_run:
            _print(
                {
                    "dry_run": True,
                    "mode": "local_detached",
                    "host_role": "lustre_server",
                    "max_workers": int(profile.get("acquisition", {}).get("max_workers", 8)),
                }
            )
        else:
            _print(launch_local_download(profile_path, profile, state))
        return 0
    if args.target == "build" and not args.dry_run:
        _require_preflight(profile, "compute")
        if profile.get("gates", {}).get("require_acquisition_handoff"):
            verify_acquisition_handoff(
                profile,
                manifest,
                state,
                verify_artifact_hashes=False,
            )
        if not status(profile, state)["download"]["build_ready"]:
            raise RuntimeError("Acquisition is not build-ready; resume it on the Lustre server first")
        holdouts = Path(profile["storage"]["lustre_root"]) / profile["storage"]["directories"]["contamination"] / "holdouts.jsonl"
        if not holdouts.exists():
            raise RuntimeError("Evaluation holdouts are missing; resume acquisition on the Lustre server first")
    payload = submit_graph(
        profile_path=profile_path,
        profile=profile,
        state=state,
        include_download=args.target in {"download", "pipeline"},
        include_build=args.target in {"build", "pipeline"},
        dry_run=args.dry_run,
    )
    _print(payload)
    return 0


def cmd_download_task(args: argparse.Namespace) -> int:
    _, profile, manifest, state = _context(args.profile)
    resolve_sources(manifest, profile, state)
    _print(run_download_task(profile, args.task_index))
    return 0


def cmd_prepare_holdouts(args: argparse.Namespace) -> int:
    _, profile, _, state = _context(args.profile)
    _print(prepare_holdouts(profile, state))
    return 0


def cmd_run_acquisition(args: argparse.Namespace) -> int:
    """Run the acquisition supervisor in the current Screen session."""

    _, profile, manifest, state = _context(args.profile)
    _require_acquisition_preflight(profile)
    # Always validate an existing immutable source lock before trusting its
    # task identities. This performs no Hub work when the lock is valid.
    resolve_sources(manifest, profile, state)
    result = run_local_download_supervisor(args.profile)
    _print(result)
    if result.get("status") != "complete":
        return 1
    if not state.path("ACQUISITION_READY.json").is_file():
        raise RuntimeError(
            "Acquisition supervisor reported completion without ACQUISITION_READY.json"
        )
    if not isinstance(result.get("acquisition_handoff"), dict):
        raise RuntimeError(
            "Acquisition supervisor reported completion without a verified handoff payload"
        )
    return 0


def cmd_verify_handoff(args: argparse.Namespace) -> int:
    _, profile, manifest, state = _context(args.profile)
    _print(
        verify_acquisition_handoff(
            profile,
            manifest,
            state,
            verify_artifact_hashes=args.deep,
        )
    )
    return 0


def cmd_rehandoff(args: argparse.Namespace) -> int:
    """Rebind a still-true acquisition handoff to a re-resolved source lock.

    The handoff binds the source lock's digest, and the source lock binds the
    repository commit. So any code change after acquisition invalidates the
    handoff even though every acquired byte is untouched. The only alternative
    to this command is deleting a self-hashed attestation by hand, which is
    exactly the habit that makes forged provenance easy. Re-attesting runs the
    identical `write_acquisition_handoff` validation the original ran, and
    refuses outright if the acquired data itself moved.
    """

    _, profile, manifest, state = _context(args.profile)
    _require_profile_role(profile, "acquisition")
    live = state.path("ACQUISITION_READY.json")
    previous = state.read("ACQUISITION_READY.json")
    if not previous:
        raise RuntimeError(
            "There is no acquisition handoff to re-attest; finish acquisition first"
        )
    stamp = utc_now().replace(":", "-")
    archive_dir = state.path("handoff-archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"ACQUISITION_READY.{stamp}.json"
    # Copy rather than move: every superseded artifact is kept byte-for-byte for
    # audit, and the copies double as the rollback sources below.
    shutil.copy2(live, archive)
    restorable: list[tuple[Path, Path]] = [(archive, live)]

    def restore() -> None:
        for source, destination in restorable:
            shutil.copy2(source, destination)

    # Rebind the lock to this checkout before re-attesting, so the new handoff
    # records the lock the build will actually verify against. `resolve_sources`
    # only ever validates a lock that already exists, so a stale one has to be
    # archived out of the way for it to resolve a replacement. That is safe
    # rather than convenient: `_validate_outputs` demands a completion marker
    # whose task_sha256 matches every task in whatever lock comes back, so a
    # re-resolve that moved any task identity fails instead of rebinding.
    lock_live = state.path("sources.lock.json")
    existing_lock = state.read("sources.lock.json") or {}
    lock_archive: Path | None = None
    if existing_lock.get("repository_commit") not in (None, _repository_commit()[0]):
        lock_archive = archive_dir / f"sources.lock.{stamp}.json"
        shutil.copy2(lock_live, lock_archive)
        restorable.append((lock_archive, lock_live))
        lock_live.unlink()
    try:
        resolve_sources(manifest, profile, state)
    except BaseException:
        restore()
        raise

    live.unlink()
    try:
        current = write_acquisition_handoff(profile, manifest, state)
    except BaseException:
        # Never leave acquisition without an attestation on disk.
        restore()
        raise
    # Re-attestation rebinds provenance. It must never become a way to move a
    # different set of bytes into a build, so anything describing the acquired
    # data itself has to be identical.
    sealed = {
        "release": (previous.get("release"), current.get("release")),
        "manifest_sha256": (previous.get("manifest_sha256"), current.get("manifest_sha256")),
        "completion_markers_sha256": (
            previous.get("completion_markers_sha256"),
            current.get("completion_markers_sha256"),
        ),
        "artifact_count": (previous.get("artifact_count"), current.get("artifact_count")),
        "artifact_bytes": (previous.get("artifact_bytes"), current.get("artifact_bytes")),
        "holdouts.sha256": (
            previous.get("holdouts", {}).get("sha256"),
            current.get("holdouts", {}).get("sha256"),
        ),
        "holdouts.report_sha256": (
            previous.get("holdouts", {}).get("report_sha256"),
            current.get("holdouts", {}).get("report_sha256"),
        ),
    }
    violations = {
        field: {"before": before, "after": after}
        for field, (before, after) in sealed.items()
        if before != after
    }
    if violations:
        restore()
        raise RuntimeError(
            "Refusing to re-attest: the acquired data changed, not just its provenance "
            "binding. Re-attestation only rebinds a stale source lock. "
            + json.dumps(violations, sort_keys=True)
        )
    _print(
        {
            "ok": True,
            "archived_previous_handoff": str(archive),
            "archived_previous_source_lock": str(lock_archive) if lock_archive else None,
            "rebound": {
                field: {
                    "before": previous.get(field),
                    "after": current.get(field),
                }
                for field in ("source_lock_sha256", "handoff_sha256")
            },
            "repository_commit": {
                "before": previous.get("repository", {}).get("commit"),
                "after": current.get("repository", {}).get("commit"),
            },
            "unchanged": {field: current.get(field) for field in ("release", "artifact_count", "artifact_bytes")},
        }
    )
    return 0


def cmd_training_contract(args: argparse.Namespace) -> int:
    contract = Path(args.contract)
    if not contract.is_absolute():
        contract = repository_root() / contract
    _print(validate_training_release(args.release, contract))
    return 0


def cmd_unlock_stale(args: argparse.Namespace) -> int:
    _, _, _, state = _context(args.profile)
    removed = state.clear_stale_locks(int(float(args.older_than_hours) * 3600))
    _print({"ok": True, "removed": removed, "count": len(removed)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metisctl", description="Metis-1.6 login2/Rhea data factory"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create the content-addressed Lustre directory layout")
    init.add_argument("--profile", default="login2")
    init.set_defaults(func=cmd_init)

    validate = subparsers.add_parser("validate", help="Validate all 1T phase/source/freshness contracts")
    validate.add_argument("--manifest", default=None)
    validate.set_defaults(func=cmd_validate)

    plan = subparsers.add_parser("plan", help="Print final exposures and candidate acquisition headroom")
    plan.add_argument("--profile", default="login2")
    plan.set_defaults(func=cmd_plan)

    doctor = subparsers.add_parser("doctor", help="Check Lustre, Slurm, auth, tools, and release gates")
    doctor.add_argument("--profile", default="login2")
    doctor.add_argument("--tiny-probe", action="store_true")
    doctor.add_argument("--role", choices=("acquisition", "compute", "all"), default="acquisition")
    doctor.set_defaults(func=cmd_doctor)

    resolve = subparsers.add_parser("resolve", help="Resolve and immutably lock upstream source files")
    resolve.add_argument("--profile", default="login2")
    resolve.set_defaults(func=cmd_resolve)

    holdouts = subparsers.add_parser("prepare-holdouts", help="Build the evaluation-only contamination index input")
    holdouts.add_argument("--profile", default="login2")
    holdouts.set_defaults(func=cmd_prepare_holdouts)

    run_acquisition = subparsers.add_parser("run-acquisition", help=argparse.SUPPRESS)
    run_acquisition.add_argument("--profile", default="login2")
    run_acquisition.set_defaults(func=cmd_run_acquisition)

    handoff = subparsers.add_parser("verify-handoff", help="Verify the immutable login2-to-Rhea acquisition handoff")
    handoff.add_argument("--profile", default="rhea")
    handoff.add_argument("--deep", action="store_true", help="Rehash every acquired artifact")
    handoff.set_defaults(func=cmd_verify_handoff)

    rehandoff = subparsers.add_parser(
        "rehandoff",
        help="Rebind an unchanged acquisition to a re-resolved source lock after a code change",
    )
    rehandoff.add_argument("--profile", default="login2")
    rehandoff.set_defaults(func=cmd_rehandoff)

    submit = subparsers.add_parser("submit", help="Launch local acquisition or submit the Slurm build graph")
    submit.add_argument("target", choices=("download", "build", "pipeline"))
    submit.add_argument("--profile", required=True)
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    resume = subparsers.add_parser("resume", help="Resubmit the restart-safe graph after a failure")
    resume.add_argument("--target", choices=("download", "build", "pipeline"), default="download")
    resume.add_argument("--profile", required=True)
    resume.add_argument("--dry-run", action="store_true")
    resume.set_defaults(func=cmd_resume)

    status_parser = subparsers.add_parser("status", help="Show completion counts and live Slurm jobs")
    status_parser.add_argument("--profile", default="login2")
    status_parser.set_defaults(func=cmd_status)

    report_parser = subparsers.add_parser("report", help="Emit the human/machine-readable build report")
    report_parser.add_argument("--profile", default="login2")
    report_parser.set_defaults(func=cmd_report)

    download_task = subparsers.add_parser("download-task", help=argparse.SUPPRESS)
    download_task.add_argument("--profile", required=True)
    download_task.add_argument("--task-index", type=int, required=True)
    download_task.set_defaults(func=cmd_download_task)

    training_contract = subparsers.add_parser(
        "training-contract", help="Verify that an immutable data release is safe for the Metis-1.6 trainer"
    )
    training_contract.add_argument("--release", required=True)
    training_contract.add_argument("--contract", default="configs/metis16/pretraining.yaml")
    training_contract.set_defaults(func=cmd_training_contract)

    unlock = subparsers.add_parser("unlock-stale", help="Remove only abandoned task locks older than a safety window")
    unlock.add_argument("--profile", default="login2")
    unlock.add_argument("--older-than-hours", type=float, default=24.0)
    unlock.set_defaults(func=cmd_unlock_stale)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Keep the one-line FAIL for log greps, but never only that. A
        # production tool that hides its stack trace turns a five-minute fix
        # into a day of bisecting by symptom, which is exactly what happened
        # to the holdout builder. Set METIS_NO_TRACEBACK=1 to suppress.
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("METIS_NO_TRACEBACK", "") not in {"1", "true", "yes"}:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
