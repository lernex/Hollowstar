from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Any

from .config import load_portage_config
from .telemetry import sample_cxi_counters
from .util import utc_now


def _rocm_sample() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showpower",
                "--showtemp",
                "--showclocks",
                "--json",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-8192:],
        }
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "rocm-smi returned non-JSON output"}
    return {"ok": True, "data": data}


def run_monitor(*, config_path: str, output_directory: str) -> None:
    config = load_portage_config(config_path)
    stop = Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    hostname = socket.gethostname()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{hostname}-rank-{rank:05d}.jsonl"
    interval = max(5, int(config.raw["telemetry"]["interval_seconds"]))
    samples = 0
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        while not stop.is_set():
            row = {
                "schema": "metis.portage-node-sample/v1",
                "created_at": utc_now(),
                "monotonic_seconds": time.monotonic(),
                "hostname": socket.getfqdn(),
                "rank": rank,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "rocm": _rocm_sample(),
                "cxi": sample_cxi_counters(
                    config.raw["telemetry"]["cxi_paths"],
                    config.raw["telemetry"]["cxi_counter_names"],
                ),
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            samples += 1
            if samples % 20 == 0:
                os.fsync(handle.fileno())
            stop.wait(interval)
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample ROCm and CXI counters once per node.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_monitor(config_path=args.config, output_directory=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
