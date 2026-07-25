from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set rank-safe ROCm tuning state, then exec the Metis trainer."
    )
    parser.add_argument("--tunable-directory", required=True)
    parser.add_argument("--tuning", choices=("0", "1"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a trainer command is required after --")
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    directory = Path(args.tunable_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = args.tuning
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(
        directory / f"rank-{rank:05d}-device-%d.csv"
    )
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("AMD_COMGR_CACHE", "1")
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
