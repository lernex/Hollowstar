from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_portage_config
from .discovery import collect_compute_inventory, require_inventory, write_inventory
from .distributed import destroy_distributed, initialize_distributed
from .probes import run_collective_probe, run_single_apu_probe, write_probe_report
from .posttraining_release import verify_posttraining_release_distributed
from .release import verify_release_distributed
from .runtime import audit_compute_runtime
from .util import atomic_write_json, file_sha256, json_sha256, read_json, utc_now


STAGES = (
    "compute_inventory",
    "single_apu",
    "node_collectives",
    "multinode_collectives",
    "release_verification",
)


def _stage_paths(campaign_root: Path, stage: str) -> tuple[Path, Path]:
    report = campaign_root / "gates" / f"{stage}.json"
    complete = campaign_root / "gates" / f"{stage}.complete.json"
    return report, complete


def _require_prerequisite(campaign_root: Path, stage: str) -> None:
    index = STAGES.index(stage)
    if index == 0:
        return
    previous = STAGES[index - 1]
    report, complete = _stage_paths(campaign_root, previous)
    if not report.is_file() or not complete.is_file():
        raise RuntimeError(f"{stage} requires completed {previous}")
    marker = read_json(complete)
    if (
        marker.get("schema") != "metis.portage-stage-complete/v1"
        or marker.get("stage") != previous
        or marker.get("report_sha256") != file_sha256(report)
        or marker.get("marker_sha256")
        != json_sha256(marker, omit=("marker_sha256",))
    ):
        raise RuntimeError(f"Prerequisite marker for {previous} is invalid or stale")


def _complete(campaign_root: Path, stage: str, report_path: Path) -> None:
    marker = {
        "schema": "metis.portage-stage-complete/v1",
        "stage": stage,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "completed_at": utc_now(),
    }
    marker["marker_sha256"] = json_sha256(marker)
    _, complete = _stage_paths(campaign_root, stage)
    atomic_write_json(complete, marker)


def run_stage(*, config_path: str, campaign_root: str, stage: str) -> None:
    if stage not in STAGES:
        raise RuntimeError(f"Unknown Portage bring-up stage: {stage}")
    config = load_portage_config(config_path)
    root = Path(campaign_root).expanduser().resolve()
    try:
        root.relative_to(config.state_root)
    except ValueError as exc:
        raise RuntimeError("Campaign root escapes configured Portage state root") from exc
    _require_prerequisite(root, stage)
    report_path, _complete_path = _stage_paths(root, stage)
    if stage == "compute_inventory":
        report = collect_compute_inventory(config)
        write_inventory(report_path, report)
        require_inventory(report)
        runtime_path = root / "gates" / "runtime_compute.json"
        runtime = audit_compute_runtime(config, output_path=runtime_path)
        report["facts"]["runtime_compute_path"] = str(runtime_path)
        report["facts"]["runtime_compute_sha256"] = runtime[
            "runtime_compute_sha256"
        ]
        report["facts"]["available_precision_profiles"] = runtime[
            "available_precision_profiles"
        ]
        report["gates"].append(
            {
                "name": "runtime-kernels",
                "ok": runtime["ok"] is True,
                "detail": {
                    "runtime_compute_sha256": runtime[
                        "runtime_compute_sha256"
                    ],
                    "available_precision_profiles": runtime[
                        "available_precision_profiles"
                    ],
                },
            }
        )
        report["ok"] = all(bool(row.get("ok")) for row in report["gates"])
        write_inventory(report_path, report)
        require_inventory(report)
        _complete(root, stage, report_path)
        return
    if stage == "single_apu":
        report = run_single_apu_probe(config)
        write_probe_report(report_path, report)
        _complete(root, stage, report_path)
        return
    require_gpu = stage != "release_verification"
    context = initialize_distributed(require_gpu=require_gpu)
    try:
        if stage in {"node_collectives", "multinode_collectives"}:
            report = run_collective_probe(config, context, stage=stage)
            if context.is_root:
                assert report is not None
                write_probe_report(report_path, report)
                _complete(root, stage, report_path)
            return
        marker = verify_release_distributed(
            release_root=config.release_root,
            training_contract_path=config.training_contract,
            output_path=report_path,
            receipt_directory=root / "gates" / "release-receipts",
            context=context,
        )
        posttraining_marker = verify_posttraining_release_distributed(
            preflight_path=root / "posttraining-release-preflight.json",
            output_path=root
            / "gates"
            / "posttraining_release_verification.json",
            receipt_directory=root
            / "gates"
            / "posttraining-release-receipts",
            context=context,
        )
        if context.is_root:
            if marker is None or posttraining_marker is None:
                raise RuntimeError("Root release-verification rank did not create a marker")
            marker["posttraining"] = posttraining_marker
            marker["marker_sha256"] = json_sha256(
                marker, omit=("marker_sha256",)
            )
            atomic_write_json(report_path, marker)
            _complete(root, stage, report_path)
    finally:
        destroy_distributed(context)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one fail-closed Portage bring-up stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args()
    run_stage(
        config_path=args.config,
        campaign_root=args.campaign_root,
        stage=args.stage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
