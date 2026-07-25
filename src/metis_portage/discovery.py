from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import PortageConfig
from .util import (
    CommandResult,
    CommandRunner,
    atomic_write_json,
    parse_slurm_time,
    split_key_values,
    utc_now,
)


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return default if match is None else match.group(1).strip()


def _gpu_count_from_partition(text: str) -> int:
    values = split_key_values(text)
    total = values.get("TRES", "") + "," + values.get("TRESBillingWeights", "")
    matches = [int(value) for value in re.findall(r"(?:gres/)?gpu(?:[:/][^=,]+)?=(\d+)", total)]
    return max(matches, default=0)


def _gpu_count_per_node(text: str) -> int:
    counts: list[int] = []
    for line in text.splitlines():
        values = split_key_values(line)
        combined = ",".join(
            (
                values.get("CfgTRES", ""),
                values.get("Gres", ""),
                values.get("AllocTRES", ""),
            )
        )
        matches = [
            int(value)
            for value in re.findall(
                r"(?:gres/)?gpu(?:[:/][^=,:()]+)?(?::|=)(\d+)",
                combined,
            )
        ]
        if matches:
            counts.append(max(matches))
    return min(counts, default=0)


def _partition_nodes(text: str) -> int:
    values = split_key_values(text)
    for key in ("TotalNodes", "MaxNodes"):
        raw = values.get(key, "")
        if raw.isdigit():
            return int(raw)
    return 0


def _max_time(text: str) -> str:
    return split_key_values(text).get("MaxTime", "")


def _cluster_name(text: str) -> str:
    value = _first_match(r"^\s*ClusterName\s*=\s*(\S+)", text)
    if value:
        return value
    return split_key_values(text).get("ClusterName", "")


def _command(
    runner: CommandRunner,
    argv: list[str],
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    return runner.run(argv, timeout=timeout).as_dict()


def _result(record: dict[str, Any], key: str) -> CommandResult:
    row = record["commands"][key]
    return CommandResult(
        argv=tuple(row["argv"]),
        returncode=int(row["returncode"]),
        stdout=str(row["stdout"]),
        stderr=str(row["stderr"]),
        elapsed_seconds=float(row["elapsed_seconds"]),
    )


def collect_login_inventory(
    config: PortageConfig,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    runner = runner or CommandRunner()
    site_options: list[str] = []
    for field, option in (
        ("account", "--account"),
        ("qos", "--qos"),
        ("reservation", "--reservation"),
    ):
        value = str(config.raw["site"].get(field, "")).strip()
        if value:
            site_options.extend((option, value))
    submission_probe = [
        "sbatch",
        "--test-only",
        "--partition",
        config.partition,
        "--nodes",
        str(config.raw["site"]["nodes"]),
        "--ntasks",
        str(sum(family.world_size for family in config.families)),
        "--ntasks-per-node",
        str(config.accelerators_per_node),
        "--cpus-per-task",
        str(
            int(config.raw["site"]["cpu_cores_per_node"])
            // config.accelerators_per_node
        ),
        "--gpus-per-node",
        str(config.accelerators_per_node),
        "--time",
        str(config.raw["site"]["production_segment_time"]),
        "--exclusive",
        "--hint=nomultithread",
        "--mem=0",
        *site_options,
        "--wrap",
        "true",
    ]
    commands = {
        "scontrol_config": _command(runner, ["scontrol", "show", "config"], timeout=90),
        "partition": _command(
            runner,
            ["scontrol", "show", "partition", config.partition, "-o"],
            timeout=90,
        ),
        "nodes": _command(
            runner,
            ["scontrol", "show", "nodes", "-o"],
            timeout=120,
        ),
        "sinfo": _command(
            runner,
            ["sinfo", "-p", config.partition, "-h", "-o", "%P|%D|%G|%l|%a|%F"],
            timeout=90,
        ),
        "findmnt": _command(
            runner,
            ["findmnt", "-T", str(config.release_root), "-n", "-o", "FSTYPE,SOURCE,TARGET"],
        ),
        "lfs": _command(runner, ["lfs", "getstripe", "-d", str(config.release_root)]),
        "git": _command(runner, ["git", "-C", str(config.repository), "rev-parse", "HEAD"]),
        "git_status": _command(
            runner,
            [
                "git",
                "-C",
                str(config.repository),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
        ),
        "sbatch_help": _command(runner, ["sbatch", "--help"], timeout=30),
        "srun_help": _command(runner, ["srun", "--help"], timeout=30),
        "sacct_help": _command(runner, ["sacct", "--help"], timeout=30),
        "submission_test": _command(
            runner,
            submission_probe,
            timeout=90,
        ),
        "python": _command(
            runner,
            [
                "python3",
                "-c",
                (
                    "import json,platform,sys;"
                    "print(json.dumps({'executable':sys.executable,'version':sys.version,"
                    "'platform':platform.platform()}))"
                ),
            ],
        ),
        "torch": _command(
            runner,
            [
                "python3",
                "-c",
                (
                    "import json,torch;"
                    "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
                    "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))"
                ),
            ],
        ),
        "modules": _command(
            runner,
            [
                "bash",
                "-lc",
                "type module >/dev/null 2>&1 && module -t list 2>&1 || true",
            ],
        ),
    }
    partition = _result({"commands": commands}, "partition")
    nodes = _result({"commands": commands}, "nodes")
    slurm_config = _result({"commands": commands}, "scontrol_config")
    findmnt = _result({"commands": commands}, "findmnt")
    cluster = _cluster_name(slurm_config.stdout)
    partition_nodes = _partition_nodes(partition.stdout)
    gpu_per_node = _gpu_count_per_node(nodes.stdout)
    if gpu_per_node == 0:
        gpu_total = _gpu_count_from_partition(partition.stdout)
        if partition_nodes > 0 and gpu_total:
            gpu_per_node = gpu_total // partition_nodes
    facts = {
        "hostname": socket.getfqdn(),
        "platform": platform.platform(),
        "cluster_name": cluster,
        "partition": config.partition,
        "partition_nodes": partition_nodes,
        "gpu_count_per_node": gpu_per_node,
        "partition_max_time": _max_time(partition.stdout),
        "release_filesystem": findmnt.stdout.strip(),
        "git_commit": _result({"commands": commands}, "git").stdout.strip(),
        "slurm_account": str(config.raw["site"].get("account", "")).strip()
        or "<site-default>",
        "slurm_qos": str(config.raw["site"].get("qos", "")).strip()
        or "<site-default>",
        "slurm_reservation": str(
            config.raw["site"].get("reservation", "")
        ).strip()
        or "<none>",
    }
    required_commands = ("scontrol_config", "partition", "nodes", "sinfo", "sbatch_help", "srun_help")
    gates = [
        Gate(
            "slurm-commands",
            all(_result({"commands": commands}, key).ok for key in required_commands),
            "required Slurm client commands responded",
        ),
        Gate(
            "cluster-identity",
            bool(re.search(config.raw["site"]["expected_cluster_regex"], cluster)),
            f"observed ClusterName={cluster!r}",
        ),
        Gate(
            "partition-capacity",
            partition.ok and partition_nodes >= int(config.raw["site"]["nodes"]),
            f"{config.partition} reports {partition_nodes} nodes",
        ),
        Gate(
            "gpu-tres",
            gpu_per_node == config.accelerators_per_node,
            f"observed {gpu_per_node} GPU/APU TRES per node",
        ),
        Gate(
            "partition-wall-time",
            bool(facts["partition_max_time"])
            and parse_slurm_time(facts["partition_max_time"])
            == int(config.raw["site"]["maximum_partition_wall_seconds"]),
            f"observed MaxTime={facts['partition_max_time']!r}",
        ),
        Gate(
            "lustre-release",
            findmnt.ok and "lustre" in findmnt.stdout.lower(),
            findmnt.stdout.strip() or findmnt.stderr.strip(),
        ),
        Gate(
            "release-present",
            config.release_root.is_dir() and (config.release_root / "RELEASE.json").is_file(),
            str(config.release_root),
        ),
        Gate(
            "repository-commit",
            _result({"commands": commands}, "git").ok,
            facts["git_commit"],
        ),
        Gate(
            "repository-clean",
            (
                not config.raw["training"]["require_clean_repository"]
                or (
                    _result({"commands": commands}, "git_status").ok
                    and not _result({"commands": commands}, "git_status").stdout.strip()
                )
            ),
            "tracked worktree changes are absent",
        ),
        Gate(
            "python",
            _result({"commands": commands}, "python").ok,
            _result({"commands": commands}, "python").stdout.strip(),
        ),
        Gate(
            "slurm-launch-features",
            all(
                feature
                in (
                    _result({"commands": commands}, "sbatch_help").stdout
                    + _result({"commands": commands}, "srun_help").stdout
                )
                for feature in (
                    "--gpus-per-node",
                    "--signal",
                    "--relative",
                    "--exact",
                    "--test-only",
                )
            ),
            "Slurm exposes GPU, checkpoint signal, and disjoint-step options",
        ),
        Gate(
            "slurm-submission-access",
            _result({"commands": commands}, "submission_test").ok,
            (
                _result({"commands": commands}, "submission_test").stdout.strip()
                or _result(
                    {"commands": commands}, "submission_test"
                ).stderr.strip()
                or "exact production-envelope sbatch --test-only completed"
            ),
        ),
    ]
    return {
        "schema": "metis.portage-inventory/v1",
        "scope": "login",
        "created_at": utc_now(),
        "facts": facts,
        "commands": commands,
        "gates": [gate.as_dict() for gate in gates],
        "ok": all(gate.ok for gate in gates),
    }


def collect_compute_inventory(
    config: PortageConfig,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    runner = runner or CommandRunner()
    python_probe = """
import json, os, torch
row = {
  "torch": torch.__version__,
  "hip": torch.version.hip,
  "cuda": torch.version.cuda,
  "distributed_available": torch.distributed.is_available(),
  "nccl_available": torch.distributed.is_nccl_available(),
  "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
  p = torch.cuda.get_device_properties(0)
  row.update({
    "device_name": p.name,
    "total_memory": p.total_memory,
    "gcn_arch_name": getattr(p, "gcnArchName", ""),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "float8_dtypes": [name for name in ("float8_e4m3fnuz","float8_e5m2fnuz","float8_e4m3fn","float8_e5m2") if hasattr(torch, name)],
  })
print(json.dumps(row, sort_keys=True))
"""
    cxi_probe = """
import json
from pathlib import Path
p = Path('/run/cxi')
readable = []
if p.exists():
    for candidate in p.rglob('*'):
        try:
            if candidate.is_file() and candidate.stat().st_size < 1048576:
                with candidate.open('rb') as handle:
                    handle.read(1)
                readable.append(str(candidate))
        except (OSError, PermissionError):
            pass
print(json.dumps({'exists': p.exists(), 'readable_files': len(readable), 'sample': readable[:20]}))
"""
    commands = {
        "rocminfo": _command(runner, ["rocminfo"], timeout=120),
        "rocm_smi": _command(
            runner,
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuniqueid", "--json"],
            timeout=120,
        ),
        "hipconfig": _command(runner, ["hipconfig", "--full"], timeout=90),
        "fi_info": _command(runner, ["fi_info", "-p", "cxi"], timeout=90),
        "torch": _command(runner, ["python3", "-c", python_probe], timeout=120),
        "rccl": _command(
            runner,
            [
                "python3",
                "-c",
                (
                    "import json,torch;"
                    "print(json.dumps({'nccl_version':"
                    "(torch.cuda.nccl.version() if torch.cuda.is_available() else None)}))"
                ),
            ],
            timeout=60,
        ),
        "mounts": _command(runner, ["findmnt", "-t", "lustre", "-n", "-o", "SOURCE,TARGET"]),
        "numa": _command(runner, ["numactl", "--hardware"]),
        "lspci": _command(runner, ["lspci", "-nn"]),
        "cxi": _command(runner, ["python3", "-c", cxi_probe]),
    }
    torch_result = _result({"commands": commands}, "torch")
    try:
        torch_facts = json.loads(torch_result.stdout) if torch_result.ok else {}
    except json.JSONDecodeError:
        torch_facts = {}
    rocminfo = _result({"commands": commands}, "rocminfo")
    cxi_result = _result({"commands": commands}, "cxi")
    try:
        cxi_facts = json.loads(cxi_result.stdout) if cxi_result.ok else {}
    except json.JSONDecodeError:
        cxi_facts = {}
    rccl_result = _result({"commands": commands}, "rccl")
    try:
        rccl_facts = json.loads(rccl_result.stdout) if rccl_result.ok else {}
    except json.JSONDecodeError:
        rccl_facts = {}
    observed_arch = str(torch_facts.get("gcn_arch_name", ""))
    if not observed_arch:
        observed_arch = _first_match(r"Name:\s*(gfx[0-9a-z]+)", rocminfo.stdout)
    total_memory = int(torch_facts.get("total_memory", 0) or 0)
    gates = [
        Gate("rocminfo", rocminfo.ok, "rocminfo completed"),
        Gate(
            "mi300a-architecture",
            str(config.raw["site"]["expected_gpu_arch"]).lower() in observed_arch.lower()
            or (
                str(config.raw["site"]["expected_gpu_arch"]).lower()
                in rocminfo.stdout.lower()
            ),
            f"observed architecture={observed_arch!r}",
        ),
        Gate(
            "hbm-capacity",
            total_memory >= int(config.raw["site"]["minimum_hbm_bytes_per_apu"]),
            f"torch reports {total_memory} bytes",
        ),
        Gate(
            "pytorch-rocm",
            torch_result.ok
            and bool(torch_facts.get("hip"))
            and not bool(torch_facts.get("cuda"))
            and int(torch_facts.get("device_count", 0)) >= 1,
            json.dumps(torch_facts, sort_keys=True),
        ),
        Gate(
            "rccl",
            torch_result.ok
            and bool(torch_facts.get("distributed_available"))
            and bool(torch_facts.get("nccl_available")),
            "PyTorch distributed NCCL API is backed by RCCL on ROCm",
        ),
        Gate(
            "bf16",
            bool(torch_facts.get("bf16_supported")),
            "PyTorch reports BF16 support",
        ),
        Gate(
            "cxi-counters",
            (
                not bool(config.raw["site"]["require_cxi_counters"])
                or int(cxi_facts.get("readable_files", 0)) > 0
            ),
            json.dumps(cxi_facts, sort_keys=True),
        ),
        Gate(
            "lustre-mounted",
            _result({"commands": commands}, "mounts").ok
            and bool(_result({"commands": commands}, "mounts").stdout.strip()),
            _result({"commands": commands}, "mounts").stdout.strip(),
        ),
    ]
    return {
        "schema": "metis.portage-inventory/v1",
        "scope": "compute",
        "created_at": utc_now(),
        "facts": {
            "hostname": socket.getfqdn(),
            "torch": torch_facts,
            "gpu_arch": observed_arch,
            "cxi": cxi_facts,
            "rccl": rccl_facts,
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "python": shutil.which("python3"),
        },
        "commands": commands,
        "gates": [gate.as_dict() for gate in gates],
        "ok": all(gate.ok for gate in gates),
    }


def require_inventory(record: dict[str, Any]) -> None:
    failed = [gate for gate in record.get("gates", []) if not gate.get("ok")]
    if record.get("ok") is not True or failed:
        details = "; ".join(f"{row.get('name')}: {row.get('detail')}" for row in failed)
        raise RuntimeError(f"Portage inventory failed closed: {details or 'unknown gate'}")


def write_inventory(path: str | Path, record: dict[str, Any]) -> None:
    atomic_write_json(path, record)
