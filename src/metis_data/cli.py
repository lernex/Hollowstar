from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import load_profile, repository_root
from .doctor import run_doctor
from .download import run_download_task
from .manifest import candidate_plan, dump_json, validate_manifest
from .holdouts import prepare_holdouts
from .reporting import report, status
from .slurm import submit_graph
from .source_lock import resolve_sources
from .state import StateStore, utc_now
from .training_contract import validate_training_release


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
    payload = run_doctor(profile, tiny_probe=args.tiny_probe)
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
    if state.read("sources.lock.json") is None:
        resolve_sources(manifest, profile, state)
    include_download = args.target in {"download", "pipeline"}
    include_build = args.target in {"build", "pipeline"}
    if include_build and not args.dry_run and not profile.get("runtime", {}).get("dynamic_materializers_enabled", False):
        raise RuntimeError(
            "Production build is gated: dynamic Common Crawl, GitHub/repository, canonical-source, "
            "and derived-data materializers are not connected in this Portage profile."
        )
    if include_build and not args.dry_run:
        prepare_holdouts(profile, state)
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
    if state.read("sources.lock.json") is None:
        resolve_sources(manifest, profile, state)
    if (
        args.target in {"build", "pipeline"}
        and not args.dry_run
        and not profile.get("runtime", {}).get("dynamic_materializers_enabled", False)
    ):
        raise RuntimeError("Cannot resume the build until dynamic source materializers are connected")
    if args.target in {"build", "pipeline"} and not args.dry_run:
        prepare_holdouts(profile, state)
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
    _, profile, _, _ = _context(args.profile)
    _print(run_download_task(profile, args.task_index))
    return 0


def cmd_prepare_holdouts(args: argparse.Namespace) -> int:
    _, profile, _, state = _context(args.profile)
    _print(prepare_holdouts(profile, state))
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
    parser = argparse.ArgumentParser(prog="metisctl", description="Metis-1.6 Portage data factory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create the content-addressed Lustre directory layout")
    init.add_argument("--profile", default="portage")
    init.set_defaults(func=cmd_init)

    validate = subparsers.add_parser("validate", help="Validate all 1T phase/source/freshness contracts")
    validate.add_argument("--manifest", default=None)
    validate.set_defaults(func=cmd_validate)

    plan = subparsers.add_parser("plan", help="Print final exposures and candidate acquisition headroom")
    plan.add_argument("--profile", default="portage")
    plan.set_defaults(func=cmd_plan)

    doctor = subparsers.add_parser("doctor", help="Check Lustre, Slurm, auth, tools, and release gates")
    doctor.add_argument("--profile", default="portage")
    doctor.add_argument("--tiny-probe", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    resolve = subparsers.add_parser("resolve", help="Resolve and immutably lock upstream source files")
    resolve.add_argument("--profile", default="portage")
    resolve.set_defaults(func=cmd_resolve)

    holdouts = subparsers.add_parser("prepare-holdouts", help="Build the evaluation-only contamination index input")
    holdouts.add_argument("--profile", default="portage")
    holdouts.set_defaults(func=cmd_prepare_holdouts)

    submit = subparsers.add_parser("submit", help="Submit a restartable Slurm dependency graph")
    submit.add_argument("target", choices=("download", "build", "pipeline"))
    submit.add_argument("--profile", default="portage")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    resume = subparsers.add_parser("resume", help="Resubmit the restart-safe graph after a failure")
    resume.add_argument("--target", choices=("download", "build", "pipeline"), default="pipeline")
    resume.add_argument("--profile", default="portage")
    resume.add_argument("--dry-run", action="store_true")
    resume.set_defaults(func=cmd_resume)

    status_parser = subparsers.add_parser("status", help="Show completion counts and live Slurm jobs")
    status_parser.add_argument("--profile", default="portage")
    status_parser.set_defaults(func=cmd_status)

    report_parser = subparsers.add_parser("report", help="Emit the human/machine-readable build report")
    report_parser.add_argument("--profile", default="portage")
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
    unlock.add_argument("--profile", default="portage")
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
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
