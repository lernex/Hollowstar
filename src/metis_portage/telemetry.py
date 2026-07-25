from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_write_json, utc_now


def _read_small_text(path: Path, *, maximum_bytes: int = 65536) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return None


def snapshot_cxi(roots: Iterable[str | Path] = ("/run/cxi",)) -> dict[str, Any]:
    counters: dict[str, str] = {}
    errors: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            errors.append(f"missing:{root}")
            continue
        for path in root.rglob("*"):
            text = _read_small_text(path)
            if text is None:
                continue
            relative = f"{root}:{path.relative_to(root)}"
            # Retain only scalar-ish counter/status files.  This keeps one
            # snapshot bounded even on nodes exposing many CXI debug entries.
            if len(text.splitlines()) <= 8 and len(text) <= 4096:
                counters[relative] = text
    return {
        "schema": "metis.cxi-snapshot/v1",
        "hostname": socket.getfqdn(),
        "created_at": utc_now(),
        "counters": counters,
        "errors": errors,
        "ok": bool(counters),
    }


def sample_cxi_counters(
    roots: Iterable[str | Path],
    counter_names: Iterable[str],
) -> dict[str, Any]:
    names = {str(name).lower() for name in counter_names}
    values: dict[str, str] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name.lower() not in names:
                continue
            text = _read_small_text(path, maximum_bytes=4096)
            if text is not None:
                values[f"{root}:{path.relative_to(root)}"] = text
    return values


def snapshot_rocm() -> dict[str, Any]:
    import subprocess

    argv = [
        "rocm-smi",
        "--showuse",
        "--showmemuse",
        "--showpower",
        "--showtemp",
        "--showclocks",
        "--json",
    ]
    completed = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    parsed: Any = None
    if completed.returncode == 0:
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = completed.stdout[-262144:]
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "data": parsed,
        "stderr": completed.stderr[-65536:],
    }


def write_node_snapshot(
    output_directory: str | Path,
    *,
    label: str,
    cxi_roots: Iterable[str | Path] = ("/run/cxi",),
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    row = {
        "schema": "metis.portage-node-telemetry/v1",
        "label": label,
        "rank": rank,
        "hostname": socket.getfqdn(),
        "created_at": utc_now(),
        "cxi": snapshot_cxi(cxi_roots),
        "rocm": snapshot_rocm(),
    }
    path = output / label / f"{socket.gethostname()}-rank-{rank:05d}.json"
    atomic_write_json(path, row)
    return path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mfu(
    *,
    estimated_train_flops: float,
    elapsed_seconds: float,
    world_size: int,
    dense_peak_flops_per_apu: float,
) -> float:
    denominator = elapsed_seconds * world_size * dense_peak_flops_per_apu
    if denominator <= 0:
        raise ValueError("MFU denominator must be positive")
    value = estimated_train_flops / denominator
    if not 0 <= value <= 1.25:
        raise ValueError(f"Implausible MFU {value:.6f}; FLOP accounting is inconsistent")
    return value
