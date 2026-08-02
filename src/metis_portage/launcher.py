from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .config import PortageConfig, load_portage_config
from .discovery import collect_login_inventory, require_inventory, write_inventory
from .posttraining_release import (
    inspect_posttraining_release_index,
    require_posttraining_release_index,
)
from .release import validate_release_fast
from .runtime import resolve_runtime
from .util import (
    CommandRunner,
    atomic_write_json,
    file_sha256,
    json_sha256,
    read_json,
    utc_now,
)


@contextmanager
def _launch_lock(state_root: Path) -> Iterator[None]:
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / ".launch.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _campaign_id(
    config: PortageConfig,
    git_commit: str,
    release: dict[str, Any],
    posttraining_release: dict[str, Any],
) -> str:
    return json_sha256(
        {
            "git_commit": git_commit,
            "config_sha256": file_sha256(config.path),
            "release_json_sha256": release["release_json_sha256"],
            "training_contract_sha256": release["training_contract_sha256"],
            "posttraining_release_index_sha256": posttraining_release[
                "index_file_sha256"
            ],
            "posttraining_contract_sha256": posttraining_release[
                "posttraining_contract_sha256"
            ],
        }
    )[:20]


def _job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+(?:_\d+)?", value):
        raise RuntimeError(f"sbatch returned an invalid parsable job id: {stdout!r}")
    return value


def _runtime_command(runtime_path: Path, command: Sequence[str]) -> list[str]:
    return [
        "bash",
        "-c",
        'source "$1"; shift; exec "$@"',
        "metis-portage-runtime",
        str(runtime_path),
        *command,
    ]


def _trainer_prelaunch_audit(
    config: PortageConfig,
    *,
    campaign_root: Path,
    runtime_path: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    marker_placeholder = campaign_root / "gates" / "release_verification.json"
    environment = dict(os.environ)
    environment["METIS_RELEASE_VERIFICATION_MARKER"] = str(marker_placeholder)
    environment["METIS_PORTAGE_PREFLIGHT_ONLY"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(config.repository / "src"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    for family in config.families:
        if not family.manifest.is_file():
            raise RuntimeError(f"Missing executable {family.name} manifest: {family.manifest}")
        output = campaign_root / "audit" / f"prelaunch-{family.name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *config.trainer_argv,
            "--manifest",
            str(family.manifest),
            "--data-release",
            str(config.release_root),
            "--output",
            str(campaign_root / "runs" / family.name),
            "--resume",
            "auto",
            "--family",
            family.name,
            "--stage",
            "pretrain",
            "--audit-config",
            "--json-output",
            str(output),
        ]
        result = runner.run(
            _runtime_command(runtime_path, command),
            timeout=600,
            cwd=config.repository,
            env=environment,
        )
        if not result.ok or not output.is_file():
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Trainer prelaunch audit is unavailable for {family.name}: {detail}"
            )
        report = read_json(output)
        release_status = report.get("release", {})
        if (
            report.get("ok") is not True
            or release_status.get("status") != "pending_distributed_gate"
            or release_status.get("training_permitted") is not False
        ):
            raise RuntimeError(f"Trainer prelaunch audit rejected {family.name}")
        reports[family.name] = report
    return reports


class SlurmSubmitter:
    def __init__(
        self,
        config: PortageConfig,
        campaign_root: Path,
        runtime_path: Path,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.campaign_root = campaign_root
        self.runtime_path = runtime_path
        self.runner = runner or CommandRunner()
        self.submitted: list[str] = []

    def _site_options(self) -> list[str]:
        options: list[str] = []
        for field, option in (
            ("account", "--account"),
            ("qos", "--qos"),
            ("reservation", "--reservation"),
        ):
            value = str(self.config.raw["site"].get(field, "")).strip()
            if value:
                options.extend((option, value))
        return options

    def _common(
        self,
        *,
        name: str,
        nodes: int,
        tasks: int,
        gpus_per_node: int,
        time_limit: str,
        dependency: str | None,
    ) -> list[str]:
        if tasks % nodes:
            raise RuntimeError(f"{name} tasks must divide evenly across nodes")
        tasks_per_node = tasks // nodes
        cpus_per_task = max(
            1,
            int(self.config.raw["site"]["cpu_cores_per_node"]) // tasks_per_node,
        )
        argv = [
            "sbatch",
            "--parsable",
            "--job-name",
            name,
            "--partition",
            self.config.partition,
            "--nodes",
            str(nodes),
            "--ntasks",
            str(tasks),
            "--ntasks-per-node",
            str(tasks_per_node),
            "--cpus-per-task",
            str(cpus_per_task),
            "--gpus-per-node",
            str(gpus_per_node),
            "--time",
            time_limit,
            "--exclusive",
            "--hint=nomultithread",
            "--mem=0",
            "--requeue",
            "--kill-on-invalid-dep=yes",
            "--chdir",
            str(self.config.repository),
            "--output",
            str(self.campaign_root / "logs" / "slurm-%x-%j.out"),
            "--error",
            str(self.campaign_root / "logs" / "slurm-%x-%j.out"),
            # Slurm runs a staged copy of the batch script from /var/spool, so
            # the script cannot derive the checkout from its own path.
            f"--export=ALL,METIS_ROOT={self.config.repository}",
            *self._site_options(),
        ]
        if dependency is not None:
            argv.extend(("--dependency", f"afterok:{dependency}"))
        return argv

    def submit_stage(self, row: dict[str, Any], dependency: str | None) -> str:
        stage = str(row["id"])
        script = self.config.repository / "slurm" / "metis16" / "portage-stage.sbatch"
        argv = [
            *self._common(
                name=f"metis16-{stage}",
                nodes=int(row["nodes"]),
                tasks=int(row["tasks"]),
                gpus_per_node=int(row["gpus_per_node"]),
                time_limit=str(row["time"]),
                dependency=dependency,
            ),
            str(script),
            str(self.config.path),
            str(self.campaign_root),
            stage,
            str(self.runtime_path),
            str(self.config.raw["bringup"]["retries_per_probe"]),
        ]
        result = self.runner.run(argv, timeout=120, cwd=self.config.repository)
        if not result.ok:
            raise RuntimeError(
                f"Failed to submit {stage}: {result.stderr.strip() or result.stdout.strip()}"
            )
        job_id = _job_id(result.stdout)
        self.submitted.append(job_id)
        return job_id

    def submit_family(self, dependency: str) -> str:
        script = self.config.repository / "slurm" / "metis16" / "portage-family.sbatch"
        signal_seconds = int(self.config.raw["site"]["checkpoint_signal_seconds"])
        argv = [
            *self._common(
                name="metis16-praxis-logos",
                nodes=128,
                tasks=512,
                gpus_per_node=4,
                time_limit=str(self.config.raw["site"]["production_segment_time"]),
                dependency=dependency,
            ),
            f"--signal=B:USR1@{signal_seconds}",
            str(script),
            str(self.config.path),
            str(self.campaign_root),
            str(self.runtime_path),
        ]
        result = self.runner.run(argv, timeout=120, cwd=self.config.repository)
        if not result.ok:
            raise RuntimeError(
                "Failed to submit simultaneous Praxis/Logos allocation: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        job_id = _job_id(result.stdout)
        self.submitted.append(job_id)
        return job_id

    def cancel_submitted(self) -> None:
        if not self.submitted:
            return
        self.runner.run(["scancel", *self.submitted], timeout=120)


def launch(
    config: PortageConfig,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    runner = runner or CommandRunner()
    with _launch_lock(config.state_root):
        login_inventory = collect_login_inventory(config, runner=runner)
        require_inventory(login_inventory)
        release = validate_release_fast(config.release_root, config.training_contract)
        posttraining_release = inspect_posttraining_release_index(config)
        preflight_directory = config.state_root / "preflight"
        preflight_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            preflight_directory / "posttraining-release.json",
            posttraining_release,
        )
        require_posttraining_release_index(posttraining_release)
        git_commit = str(login_inventory["facts"]["git_commit"])
        campaign_id = _campaign_id(
            config,
            git_commit,
            release,
            posttraining_release,
        )
        campaign_root = config.state_root / "campaigns" / campaign_id
        campaign_path = campaign_root / "campaign.json"
        if campaign_path.is_file():
            campaign = read_json(campaign_path)
            if (campaign_root / "COMPLETE.json").is_file():
                return {**campaign, "status": "complete", "idempotent": True}
            job_ids = list(campaign.get("jobs", {}).values())
            if job_ids:
                live = runner.run(
                    ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A|%T|%R"],
                    timeout=60,
                )
                if live.ok and live.stdout.strip():
                    return {
                        **campaign,
                        "status": "active",
                        "scheduler": live.stdout.strip().splitlines(),
                        "idempotent": True,
                    }
            raise RuntimeError(
                f"Campaign {campaign_id} exists but is not active or complete; "
                f"inspect {campaign_root / 'failure.json'} before resubmitting"
            )
        for directory in (
            campaign_root / "logs",
            campaign_root / "gates",
            campaign_root / "audit",
            campaign_root / "autotune",
            campaign_root / "runs",
            campaign_root / "telemetry",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        write_inventory(campaign_root / "login-inventory.json", login_inventory)
        atomic_write_json(campaign_root / "release-fast-check.json", release)
        atomic_write_json(
            campaign_root / "posttraining-release-preflight.json",
            posttraining_release,
        )
        runtime = resolve_runtime(
            config,
            output_directory=campaign_root / "runtime",
            runner=runner,
        )
        audits = _trainer_prelaunch_audit(
            config,
            campaign_root=campaign_root,
            runtime_path=Path(runtime["setup_path"]),
            runner=runner,
        )
        campaign: dict[str, Any] = {
            "schema": "metis.portage-campaign/v1",
            "campaign_id": campaign_id,
            "created_at": utc_now(),
            "campaign_root": str(campaign_root),
            "config_path": str(config.path),
            "config_sha256": file_sha256(config.path),
            "git_commit": git_commit,
            "release": release,
            "posttraining_release": {
                "index_path": posttraining_release["index_path"],
                "index_file_sha256": posttraining_release[
                    "index_file_sha256"
                ],
                "preflight_sha256": posttraining_release[
                    "preflight_sha256"
                ],
            },
            "runtime": runtime,
            "prelaunch_audits": {
                name: {
                    "ok": report.get("ok"),
                    "report_sha256": json_sha256(report),
                }
                for name, report in audits.items()
            },
            "jobs": {},
            "status": "submitting",
        }
        campaign["campaign_sha256"] = json_sha256(campaign)
        atomic_write_json(campaign_path, campaign)
        submitter = SlurmSubmitter(
            config,
            campaign_root,
            Path(runtime["setup_path"]),
            runner=runner,
        )
        try:
            dependency: str | None = None
            for stage in config.raw["bringup"]["stages"]:
                job_id = submitter.submit_stage(stage, dependency)
                campaign["jobs"][stage["id"]] = job_id
                dependency = job_id
                campaign["campaign_sha256"] = json_sha256(
                    campaign, omit=("campaign_sha256",)
                )
                atomic_write_json(campaign_path, campaign)
            assert dependency is not None
            family_job = submitter.submit_family(dependency)
            campaign["jobs"]["family"] = family_job
            campaign["status"] = "queued"
            campaign["submitted_at"] = utc_now()
            campaign["campaign_sha256"] = json_sha256(
                campaign, omit=("campaign_sha256",)
            )
            atomic_write_json(campaign_path, campaign)
            return campaign
        except Exception:
            submitter.cancel_submitted()
            campaign["status"] = "submission_failed"
            campaign["failed_at"] = utc_now()
            campaign["campaign_sha256"] = json_sha256(
                campaign, omit=("campaign_sha256",)
            )
            atomic_write_json(campaign_path, campaign)
            raise


def status(config: PortageConfig, *, runner: CommandRunner | None = None) -> dict[str, Any]:
    runner = runner or CommandRunner()
    campaigns_root = config.state_root / "campaigns"
    campaigns = sorted(
        (path for path in campaigns_root.glob("*") if (path / "campaign.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not campaigns:
        return {"schema": "metis.portage-status/v1", "status": "not-launched"}
    root = campaigns[0]
    campaign = read_json(root / "campaign.json")
    jobs = list(campaign.get("jobs", {}).values())
    squeue = runner.run(
        ["squeue", "-h", "-j", ",".join(jobs), "-o", "%A|%j|%T|%M|%D|%R"],
        timeout=60,
    ) if jobs else None
    sacct = runner.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(jobs),
            "--format=JobID,JobName,State,ExitCode,Elapsed,AllocNodes,ReqTRES",
        ],
        timeout=90,
    ) if jobs else None
    state = "complete" if (root / "COMPLETE.json").is_file() else campaign.get("status")
    if (root / "failure.json").is_file():
        state = "failed"
    return {
        "schema": "metis.portage-status/v1",
        "campaign_id": campaign["campaign_id"],
        "campaign_root": str(root),
        "status": state,
        "jobs": campaign.get("jobs", {}),
        "queue": [] if squeue is None else squeue.stdout.strip().splitlines(),
        "accounting": [] if sacct is None else sacct.stdout.strip().splitlines(),
        "gates": sorted(path.name for path in (root / "gates").glob("*.complete.json")),
        "complete": read_json(root / "COMPLETE.json") if (root / "COMPLETE.json").is_file() else None,
        "failure": read_json(root / "failure.json") if (root / "failure.json").is_file() else None,
    }


def validate_requeue(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    path = root / "requeue.json"
    marker = read_json(path)
    if (
        marker.get("schema") != "metis.portage-requeue/v1"
        or marker.get("marker_sha256")
        != json_sha256(marker, omit=("marker_sha256",))
        or marker.get("resume_safe") is not True
    ):
        raise RuntimeError("Family job did not emit a resume-safe requeue marker")
    campaign = read_json(root / "campaign.json")
    config = load_portage_config(campaign["config_path"])
    try:
        root.relative_to(config.state_root)
    except ValueError as exc:
        raise RuntimeError("Requeue campaign root escapes configured state") from exc
    from .family import (
        validate_checkpoint_for_requeue,
        validate_posttraining_state_for_requeue,
        validate_posttraining_batch_migration_for_requeue,
    )

    observed: dict[str, Any] = {}
    for family in config.families:
        output = (
            root
            / config.raw["training"]["output_subdirectory"]
            / family.name
        )
        expected = marker.get("checkpoints", {}).get(family.name)
        if not isinstance(expected, dict):
            raise RuntimeError(
                f"Requeue marker has no checkpoint state for {family.name}"
            )
        state = validate_checkpoint_for_requeue(
            output,
            family=family,
            require_checkpoint=expected.get("status") == "durable_checkpoint",
        )
        for field in (
            "status",
            "checkpoint",
            "checkpoint_sha256",
            "autotune_profile_sha256",
            "global_token_cursor",
            "artifact_count",
            "artifacts_total_bytes",
        ):
            if state.get(field) != expected.get(field):
                raise RuntimeError(
                    f"{family.name} checkpoint state changed before requeue: {field}"
                )
        observed[family.name] = state
        expected_posttraining = marker.get("posttraining", {}).get(
            family.name
        )
        if not isinstance(expected_posttraining, Mapping):
            raise RuntimeError(
                f"Requeue marker has no post-training state for {family.name}"
            )
        posttraining = validate_posttraining_state_for_requeue(
            output,
            family=family,
        )
        if posttraining != expected_posttraining:
            raise RuntimeError(
                f"{family.name} post-training state changed before requeue"
            )
        migration_summary = marker.get("profile_migrations", {}).get(
            family.name
        )
        if migration_summary is not None:
            migration_path = (
                root
                / "autotune"
                / family.name
                / "profile-migration.json"
            )
            migration = read_json(migration_path)
            profile_path = Path(str(migration.get("new_profile_path", "")))
            profile = read_json(profile_path)
            if (
                migration.get("schema")
                != "metis.autotune-profile-migration/v1"
                or migration.get("receipt_sha256")
                != json_sha256(migration, omit=("receipt_sha256",))
                or migration.get("checkpoint_sha256")
                != state.get("checkpoint_sha256")
                or migration.get("old_profile_sha256")
                != state.get("autotune_profile_sha256")
                or profile.get("profile_sha256")
                != migration.get("new_profile_sha256")
                or profile.get("profile_sha256")
                != json_sha256(profile, omit=("profile_sha256",))
                or migration_summary.get("receipt_sha256")
                != migration.get("receipt_sha256")
            ):
                raise RuntimeError(
                    f"{family.name} profile migration receipt is invalid or stale"
                )
        stage_migration_summary = marker.get(
            "posttraining_batch_migrations", {}
        ).get(family.name)
        if stage_migration_summary is not None:
            if not isinstance(stage_migration_summary, Mapping):
                raise RuntimeError(
                    f"{family.name} stage batch migration summary is invalid"
                )
            validate_posttraining_batch_migration_for_requeue(
                stage_migration_summary,
                campaign_root=root,
                output_root=output,
                family=family,
            )
    marker["validated_checkpoints"] = observed
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Portage bring-up and Praxis/Logos launcher."
    )
    parser.add_argument("--config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("launch")
    subparsers.add_parser("status")
    requeue = subparsers.add_parser("validate-requeue")
    requeue.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    if args.command == "validate-requeue":
        validate_requeue(args.campaign_root)
        return 0
    config = load_portage_config(args.config)
    result = launch(config) if args.command == "launch" else status(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
