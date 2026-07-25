from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operate the production Metis-1.6 in-process post-training path "
            "and its sealed Rhea-to-Portage release."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate the immutable post-training contract"
    )
    validate.add_argument("--config", required=True)

    plan = subparsers.add_parser(
        "plan", help="print the production in-process stage order"
    )
    plan.add_argument("--config", required=True)

    release_build = subparsers.add_parser(
        "release-build",
        help="seal and byte-verify the Rhea post-training release",
    )
    release_build.add_argument("--portage-config", required=True)
    release_build.add_argument("--spec", required=True)
    release_build.add_argument("--workers", type=int, default=8)
    release_build.add_argument("--json-output")

    release_status = subparsers.add_parser(
        "release-status",
        help="verify the sealed post-training release without allocating nodes",
    )
    release_status.add_argument("--portage-config", required=True)

    status = subparsers.add_parser(
        "status", help="show the autonomous Portage campaign state"
    )
    status.add_argument("--portage-config", required=True)

    run = subparsers.add_parser(
        "run",
        help=(
            "launch/resume the production Portage base-to-post-training "
            "campaign; post-training runs inside each family trainer"
        ),
    )
    run.add_argument("--portage-config", required=True)

    legacy = subparsers.add_parser(
        "legacy-run",
        help=(
            "invoke the deprecated external-backend orchestrator explicitly; "
            "not used by the autonomous Portage path"
        ),
    )
    legacy.add_argument("legacy_args", nargs=argparse.REMAINDER)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"validate", "plan"}:
            from .posttraining import (
                EXPECTED_STAGE_IDS,
                PipelineContractError,
                _stages_by_id,
                load_pipeline,
            )

            pipeline = load_pipeline(args.config)
            if args.command == "validate":
                _print_json(
                    {
                        "ok": True,
                        "schema": pipeline["schema"],
                        "backend": "in_process",
                        "stages": list(_stages_by_id(pipeline)),
                    }
                )
                return 0
            stages = _stages_by_id(pipeline)
            if tuple(stages) != EXPECTED_STAGE_IDS:
                raise PipelineContractError(
                    "loaded post-training stages differ from the locked order"
                )
            for index, stage in enumerate(stages.values(), start=1):
                print(
                    f"{index:02d} {stage['id']} <- {stage['input_stage']} "
                    "[in-process]"
                )
            return 0
        if args.command == "legacy-run":
            from .posttraining import main as legacy_main

            return legacy_main(args.legacy_args)

        from metis_portage.config import load_portage_config

        config = load_portage_config(args.portage_config)
        if args.command == "release-build":
            from metis_portage.posttraining_builder import (
                build_posttraining_release,
            )

            result = build_posttraining_release(
                config=config,
                spec_path=args.spec,
                workers=args.workers,
            )
            encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if args.json_output:
                output = Path(args.json_output).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                temporary.write_text(encoded, encoding="utf-8")
                temporary.replace(output)
            else:
                print(encoded, end="")
            return 0
        if args.command == "release-status":
            from metis_portage.posttraining_release import (
                inspect_posttraining_release_index,
            )

            report = inspect_posttraining_release_index(config)
            _print_json(report)
            return 0 if report.get("ok") is True else 2
        if args.command in {"status", "run"}:
            from metis_portage.launcher import launch, status

            result = launch(config) if args.command == "run" else status(config)
            _print_json(result)
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"metis-posttrain: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
