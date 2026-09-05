"""Data-parallel trainer for the MoRE ablation ladder.

Deliberately separate from ``metis_training.train``: the production trainer is
bound to the 1T release contract, its phase boundaries, its autotune lineage,
and its signal/requeue protocol, and the Praxis/Logos runs must not be
destabilized by research plumbing.  Everything that affects comparability is
shared -- the model, the release stream, the optimizer, the precision policy,
and the FLOP accounting all come from ``metis_training``.

Parallelism is pure data parallelism with every routed expert replicated on
every rank.  The proxy's training state is roughly 21GB against 128GB of
coherent HBM, so expert parallelism would buy nothing and cost an all-to-all --
and, worse for a science campaign, it would make row-to-row wall-clock
differences partly an artifact of routing skew.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import queue
import threading
import math
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist

from metis_training.data import ReleaseInventory, DeterministicReleaseStream
from metis_training.distributed import (
    OverlappedGradientReducer,
    ParallelTopology,
    Runtime,
    all_reduce_sum,
    broadcast_initial_parameters,
    destroy_runtime,
    initialize_runtime,
    normalize_summed_gradients,
    synchronize_gradients,
)
from metis_training.metrics import (
    MetricsWriter,
    enforce_health_gates,
    estimate_hardware_flops,
    estimated_mfu,
    peak_memory_evidence,
)
from metis_training.model import Metis16ForCausalLM, MetisProcessGroups
from metis_training.optimizers import (
    WorldShardedOptimizerBundle,
    build_training_optimizers,
    clip_grad_norm_,
)
from metis_training.precision import build_precision_policy
from metis_training.schedule import set_optimizer_learning_rate

from .analysis import RoutingAnalyzer
from .runtime_diagnostics import record_rank_startup
from .sampler import AblationSampleStream, build_sample_stream
from .specs import (
    ABLATION_LADDER,
    AblationSpec,
    GLOBAL_BATCH_SEQUENCES,
    GLOBAL_BATCH_TOKENS,
    spec_by_name,
    wave_for_row,
)


DEFAULT_BUDGET_TOKENS = 50_000_000_000
BUDGET_GRADIENT_GROUP_MAX_NORM = 1.0


# --------------------------------------------------------------------------
# topology


def build_replicated_expert_topology(runtime: Runtime, *, family: str) -> ParallelTopology:
    """Data-parallel topology in which every rank owns every routed expert.

    ``metis_training.distributed.build_parallel_topology`` shards experts across
    the world, which is right for Praxis and Logos and wrong here.  This builds
    the mirror image: expert parallel size one, replica count equal to the world,
    so routed-expert gradients all-reduce over the whole job exactly like dense
    weights and nothing is dispatched over the fabric.
    """

    if not runtime.distributed:
        return ParallelTopology(
            family=family,
            world_size=1,
            rank=0,
            local_rank=0,
            expert_parallel_size=1,
            expert_replica_count=1,
            expert_group=None,
            expert_group_ranks=(0,),
            expert_data_group=None,
            expert_data_group_ranks=(0,),
            dense_data_group=None,
        )
    # A single-rank expert group: the model requires an explicit group whenever
    # the world is larger than one, and a group of size one is exactly the
    # statement "this rank owns all of the experts".
    expert_group = dist.new_group(ranks=[runtime.rank], backend="nccl")
    return ParallelTopology(
        family=family,
        world_size=runtime.world_size,
        rank=runtime.rank,
        local_rank=runtime.local_rank,
        expert_parallel_size=1,
        expert_replica_count=runtime.world_size,
        expert_group=expert_group,
        expert_group_ranks=(runtime.rank,),
        expert_data_group=dist.group.WORLD,
        expert_data_group_ranks=tuple(range(runtime.world_size)),
        dense_data_group=dist.group.WORLD,
    )


# --------------------------------------------------------------------------
# schedule


@dataclass(frozen=True)
class AblationSchedule:
    """Warmup then cosine decay over the campaign's own step count.

    ``metis_training.schedule.TokenSchedule`` is keyed to the 1T phase
    boundaries, which the strided sampler deliberately does not walk in order.
    The shape is identical across all thirteen rows so the schedule cannot
    explain a difference between them.
    """

    total_steps: int
    base_learning_rate: float
    warmup_fraction: float = 0.01
    final_fraction: float = 0.10

    def learning_rate(self, step: int) -> float:
        warmup_steps = max(1, int(self.total_steps * self.warmup_fraction))
        if step < warmup_steps:
            return self.base_learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.base_learning_rate * self.final_fraction
        return floor + (self.base_learning_rate - floor) * cosine


# --------------------------------------------------------------------------
# run state


@dataclass
class RunPaths:
    root: Path

    @property
    def telemetry(self) -> Path:
        return self.root / "telemetry"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    def prepare(self) -> None:
        for path in (self.telemetry, self.checkpoints, self.analysis):
            path.mkdir(parents=True, exist_ok=True)


def _existing_run_artifacts(paths: RunPaths) -> list[Path]:
    artifacts = [
        path
        for path in (paths.root / "run.json", paths.root / "summary.json")
        if path.exists()
    ]
    for directory, pattern in (
        (paths.telemetry, "*.jsonl"),
        (paths.checkpoints, "step-*/state.pt"),
        (paths.analysis, "*.json"),
    ):
        if directory.exists():
            artifacts.extend(directory.glob(pattern))
    return sorted(artifacts)


def _source_revision(*, require_clean: bool) -> str:
    repository = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        if require_clean:
            raise RuntimeError(
                "A real ablation run requires a Git-bound source revision"
            ) from exc
        return "unavailable"
    if dirty and require_clean:
        raise RuntimeError(
            "A real ablation run requires a clean tracked Git worktree"
        )
    return revision + ("+dirty" if dirty else "")


def _run_identity(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def _validate_campaign_identity(
    output_root: Path,
    payload: dict[str, Any],
    runtime: Runtime,
) -> str:
    _, identity_sha256 = _run_identity(payload)
    error = None
    if runtime.rank == 0:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            wave = str(payload["wave"])
            path = output_root / f"CAMPAIGN_IDENTITY-wave{wave}.json"
            document = {
                "schema": "more.ablation-campaign-identity/v1",
                "identity": payload,
                "identity_sha256": identity_sha256,
            }
            encoded = (
                json.dumps(document, indent=2, sort_keys=True, default=str)
                + "\n"
            ).encode("utf-8")
            try:
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError:
                existing = None
                for _ in range(50):
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                        break
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(0.1)
                if existing is None:
                    raise RuntimeError(
                        f"Campaign identity is incomplete: {path}"
                    )
                if (
                    existing.get("identity_sha256") != identity_sha256
                    or existing.get("identity") != payload
                ):
                    raise RuntimeError(
                        f"Campaign identity changed across rows: {path}"
                    )
            else:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
        except (OSError, RuntimeError) as exc:
            error = str(exc)
    if runtime.distributed:
        status = [error]
        dist.broadcast_object_list(status, src=0, group=dist.group.WORLD)
        error = status[0]
    if error is not None:
        raise RuntimeError(error)
    return identity_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint(
    runtime: Runtime,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    script_raw = os.environ.get("METIS_ABLATION_RUNTIME")
    script = Path(script_raw).expanduser().resolve() if script_raw else None
    if require_complete and (script is None or not script.is_file()):
        raise RuntimeError("A real run requires a readable METIS_ABLATION_RUNTIME")
    plugin_raw = os.environ.get("NCCL_NET_PLUGIN")
    plugin = Path(plugin_raw).expanduser().resolve() if plugin_raw else None
    if (
        require_complete
        and runtime.world_size > 4
        and (plugin is None or not plugin.is_file())
    ):
        raise RuntimeError("A multi-node real run requires a readable RCCL plugin")

    packages: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "transformer-engine",
        "transformer-engine-torch",
        "triton",
        "aiter",
        "flash-attn",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    interpreter = Path(os.path.realpath(os.sys.executable))
    return {
        "runtime_script": (
            {
                "path": str(script),
                "sha256": _sha256_file(script),
            }
            if script is not None and script.is_file()
            else None
        ),
        "interpreter": {
            "path": str(interpreter),
            "sha256": (
                _sha256_file(interpreter) if interpreter.is_file() else None
            ),
        },
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "packages": packages,
        "rccl_plugin": (
            {
                "path": str(plugin),
                "sha256": _sha256_file(plugin),
            }
            if plugin is not None and plugin.is_file()
            else None
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "ROCM_PATH",
                "PYTORCH_ROCM_ARCH",
                "TORCH_BLAS_PREFER_HIPBLASLT",
                "NVTE_USE_CK_GROUPED_GEMM",
                "NCCL_ALGO",
                "NCCL_NET",
                "FI_PROVIDER",
            )
        },
    }


def _acquire_row_lease(paths: RunPaths, runtime: Runtime) -> Any | None:
    handle = None
    error = None
    if runtime.rank == 0:
        try:
            paths.root.mkdir(parents=True, exist_ok=True)
            handle = (paths.root / ".active-run.lock").open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "job_id": os.environ.get("SLURM_JOB_ID"),
                        "host": os.uname().nodename,
                        "acquired_unix": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        except (OSError, BlockingIOError) as exc:
            error = f"Row output directory is already leased: {paths.root}: {exc}"
    if runtime.distributed:
        status = [error]
        dist.broadcast_object_list(status, src=0, group=dist.group.WORLD)
        error = status[0]
    if error is not None:
        if handle is not None:
            handle.close()
        raise RuntimeError(error)
    return handle


def _release_row_lease(handle: Any | None) -> None:
    if handle is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _truncate_telemetry(path: Path, *, start_step: int) -> None:
    if not path.is_file():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", -1)) < start_step:
            retained.append(json.dumps(record, sort_keys=True))
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(
        "\n".join(retained) + ("\n" if retained else ""),
        encoding="utf-8",
    )
    staging.replace(path)


def _freeze_inactive_policy_parameters(
    model: Metis16ForCausalLM,
    curriculum: Any,
) -> tuple[str, ...]:
    frozen: list[str] = []
    joint = getattr(curriculum, "compute_allocation_mode", "legacy") == "joint"
    if joint or curriculum.continuation_mode in {"random", "fixed_max", "depth_one"}:
        for name, parameter in model.continuation.named_parameters(
            prefix="continuation"
        ):
            parameter.requires_grad_(False)
            frozen.append(name)
    if joint or curriculum.routed_k_mode in {"random", "fixed"}:
        for layer_index, layer in enumerate(model.layers):
            router = getattr(layer.moe, "k_router", None)
            if router is None:
                continue
            for name, parameter in router.named_parameters(
                prefix=f"layers.{layer_index}.moe.k_router"
            ):
                parameter.requires_grad_(False)
                frozen.append(name)
    return tuple(sorted(frozen))


def _clip_exact_budget_gradient_groups(
    model: Metis16ForCausalLM,
    curriculum: Any,
    *,
    max_norm: float = BUDGET_GRADIENT_GROUP_MAX_NORM,
) -> dict[str, float]:
    """Clip the model and exact-budget policy heads independently."""

    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    groups: dict[str, list[torch.nn.Parameter]] = {
        "model": [],
        "depth_policy": [],
        "width_policy": [],
    }
    joint = getattr(curriculum, "compute_allocation_mode", "legacy") == "joint"
    if joint:
        groups["joint_policy"] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if joint and name.startswith("joint_router."):
            group = "joint_policy"
        elif (
            not joint
            and curriculum.routed_k_mode == "budgeted"
            and ".k_router." in name
        ):
            group = "width_policy"
        elif (
            not joint
            and curriculum.continuation_mode == "budgeted"
            and (
                name.startswith("continuation.")
                or name.startswith("depth_memory.route_projection.")
            )
        ):
            group = "depth_policy"
        else:
            group = "model"
        groups[group].append(parameter)

    raw_norms: dict[str, float] = {}
    for name, parameters in groups.items():
        contributions = []
        for parameter in parameters:
            gradient = (
                parameter.grad.coalesce().values()
                if parameter.grad.is_sparse
                else parameter.grad
            )
            contributions.append(
                torch.linalg.vector_norm(
                    gradient,
                    ord=2,
                    dtype=torch.float32,
                )
            )
        if not contributions:
            raw_norms[name] = 0.0
            continue
        norm = torch.linalg.vector_norm(
            torch.stack(contributions),
            ord=2,
            dtype=torch.float32,
        )
        raw_norm = float(norm.detach().item())
        if not math.isfinite(raw_norm):
            raise FloatingPointError(
                f"Non-finite {name} gradient norm: {raw_norm}"
            )
        raw_norms[name] = raw_norm
        coefficient = max_norm / (raw_norm + 1.0e-6)
        if coefficient >= 1.0:
            continue
        for parameter in parameters:
            if parameter.grad.is_sparse:
                parameter.grad._values().mul_(coefficient)
            else:
                parameter.grad.mul_(coefficient)
    return raw_norms


def _assert_storage_policy(model: Any) -> None:
    """Fail loudly if a trainable parameter escaped the storage policy.

    FP32 masters are optimizer-owned; model parameters are BF16 apart from the
    roles the model explicitly tags for FP32.  A silent BF16 router would not
    crash anything, it would just quietly make the routing decisions noisier,
    which is the hardest class of bug to catch from a loss curve.
    """

    offenders: list[tuple[str, str, str]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tagged = getattr(parameter, "metis_storage_dtype", None)
        if tagged == "float32":
            # The direction that actually matters.  A router left in BF16 is
            # not an error anywhere else in the stack -- it simply makes the
            # discrete decisions noisier -- so nothing but this check would
            # ever notice.
            if parameter.dtype != torch.float32:
                offenders.append((name, "float32", str(parameter.dtype)))
        elif parameter.dtype != torch.bfloat16:
            offenders.append((name, "bfloat16", str(parameter.dtype)))
    if offenders:
        raise RuntimeError(
            "Parameter storage policy was not applied (name, expected, actual): "
            f"{offenders[:8]}"
        )


def _save_checkpoint(
    paths: RunPaths,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    spec: AblationSpec,
    rank: int,
    device: torch.device,
    total_steps: int,
    learning_rate: float,
    run_identity_sha256: str,
    keep_last: int = 2,
) -> Path | None:
    """Write a resumable checkpoint and prune older ones.

    Rank zero owns the write because parameters are replicated; RNG state is
    included so a resumed run draws the same stochastic routing and Gumbel
    samples it would have drawn without the interruption.
    """

    target = paths.checkpoints / f"step-{step:07d}"
    sharded = bool(getattr(optimizer, "is_world_sharded", False))
    if rank == 0:
        target.mkdir(parents=True, exist_ok=True)
    if sharded and dist.is_initialized():
        dist.barrier()
    if sharded:
        shard_name = f"optimizer-rank-{rank:05d}.pt"
        shard_payload = {
            "schema": "more.ablation-optimizer-shard/v1",
            "rank": rank,
            "world_size": int(optimizer.shard_world_size),
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            shard_payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        shard_staging = target / f"{shard_name}.partial"
        torch.save(shard_payload, shard_staging)
        shard_staging.replace(target / shard_name)
        if dist.is_initialized():
            dist.barrier()
    elif rank != 0:
        return None

    if rank != 0:
        if sharded and dist.is_initialized():
            dist.barrier()
        return None

    payload = {
        "schema": (
            "more.ablation-checkpoint/v3"
            if sharded
            else "more.ablation-checkpoint/v2"
        ),
        "model": model.state_dict(),
        "run_identity_sha256": run_identity_sha256,
        "step_semantics": "next_unexecuted",
        "step": step,
        "spec": asdict(spec),
        # The schedule shape is part of the run's identity: resuming a cosine
        # decay against a different horizon silently trains a different model.
        "total_steps": int(total_steps),
        "base_learning_rate": float(learning_rate),
    }
    if sharded:
        shards = []
        for shard_rank in range(int(optimizer.shard_world_size)):
            shard_name = f"optimizer-rank-{shard_rank:05d}.pt"
            shard_path = target / shard_name
            digest = hashlib.sha256()
            with shard_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            shards.append(
                {
                    "rank": shard_rank,
                    "path": shard_name,
                    "bytes": shard_path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        payload["optimizer_shards"] = shards
    else:
        payload["optimizer"] = optimizer.state_dict()
        payload["cpu_rng_state"] = torch.get_rng_state()
        if device.type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    # Write beside the target and rename, so a job killed mid-write leaves the
    # previous checkpoint intact rather than a truncated one.
    staging = target / "state.pt.partial"
    torch.save(payload, staging)
    staging.replace(target / "state.pt")

    existing = sorted(
        path for path in paths.checkpoints.glob("step-*") if (path / "state.pt").exists()
    )
    for stale in existing[:-keep_last]:
        for item in stale.iterdir():
            item.unlink()
        stale.rmdir()
    if sharded and dist.is_initialized():
        dist.barrier()
    return target


def _latest_checkpoint(paths: RunPaths) -> Path | None:
    candidates = sorted(
        path for path in paths.checkpoints.glob("step-*") if (path / "state.pt").exists()
    )
    return candidates[-1] if candidates else None


def _restore_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    device: torch.device,
    total_steps: int,
    learning_rate: float,
    expected_run_identity_sha256: str | None = None,
) -> int:
    payload = torch.load(path / "state.pt", map_location=device, weights_only=False)
    if expected_run_identity_sha256 is not None:
        stored_identity = payload.get("run_identity_sha256")
        if stored_identity is None:
            raise RuntimeError(
                f"Refusing to resume {path.name}: legacy checkpoint has no "
                "run-identity binding. Start in a clean output directory."
            )
        if str(stored_identity) != expected_run_identity_sha256:
            raise RuntimeError(
                f"Refusing to resume {path.name}: run identity changed"
            )
    stored_steps = int(payload.get("total_steps", total_steps))
    stored_lr = float(payload.get("base_learning_rate", learning_rate))
    if stored_steps != int(total_steps) or abs(stored_lr - float(learning_rate)) > 1e-12:
        raise RuntimeError(
            f"Refusing to resume {path.name}: it was written for "
            f"{stored_steps} steps at lr {stored_lr:g}, but this launch asks for "
            f"{int(total_steps)} steps at lr {learning_rate:g}. Resuming across a "
            "schedule change would silently train a different model. Pass "
            "--no-resume to start over, or restore the original arguments."
        )
    model_state = payload["model"]
    fp8_scaling = getattr(
        getattr(getattr(model, "config", None), "precision", None),
        "fp8_scaling",
        "delayed",
    )
    if fp8_scaling == "current":
        # Current scaling derives scale factors from the tensor being
        # quantized and has no delayed history to resume. Loading a delayed
        # scaling module's TE _extra_state is accepted by state_dict, but the
        # stale metadata inventory then changes on the first recompute and
        # makes checkpoint replay fail. Weights and ordinary buffers remain
        # strict; only recipe-owned transient metadata is deliberately reset.
        model_state = {
            name: value
            for name, value in model_state.items()
            if "_extra_state" not in name
        }
        incompatible = model.load_state_dict(model_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if "_extra_state" not in name
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Current-scaling checkpoint restore changed non-FP8 model state: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
    else:
        model.load_state_dict(model_state)
    if payload.get("schema") == "more.ablation-checkpoint/v3":
        if not bool(getattr(optimizer, "is_world_sharded", False)):
            raise RuntimeError("Checkpoint requires a world-sharded optimizer")
        shards = payload.get("optimizer_shards")
        if not isinstance(shards, list) or len(shards) != optimizer.shard_world_size:
            raise RuntimeError("Checkpoint optimizer shard manifest is incomplete")
        entry = shards[optimizer.shard_rank]
        if int(entry.get("rank", -1)) != optimizer.shard_rank:
            raise RuntimeError("Checkpoint optimizer shard rank order changed")
        shard_path = path / str(entry["path"])
        digest = hashlib.sha256()
        with shard_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if shard_path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError("Checkpoint optimizer shard size changed")
        if digest.hexdigest() != str(entry["sha256"]):
            raise RuntimeError("Checkpoint optimizer shard hash changed")
        shard = torch.load(shard_path, map_location=device, weights_only=False)
        if int(shard.get("rank", -1)) != optimizer.shard_rank:
            raise RuntimeError("Optimizer shard payload belongs to another rank")
        if int(shard.get("world_size", -1)) != optimizer.shard_world_size:
            raise RuntimeError("Optimizer shard world size changed")
        optimizer.load_state_dict(shard["optimizer"])
        torch.set_rng_state(shard["cpu_rng_state"].to(torch.uint8).cpu())
        if device.type == "cuda" and "cuda_rng_state" in shard:
            torch.cuda.set_rng_state(
                shard["cuda_rng_state"].to(torch.uint8).cpu(),
                device,
            )
    else:
        optimizer.load_state_dict(payload["optimizer"])
        if "cpu_rng_state" in payload:
            torch.set_rng_state(payload["cpu_rng_state"].to(torch.uint8).cpu())
        if device.type == "cuda" and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state(
                payload["cuda_rng_state"].to(torch.uint8).cpu(),
                device,
            )
    stored_step = int(payload["step"])
    if payload.get("step_semantics") == "next_unexecuted":
        return stored_step
    # Legacy checkpoints recorded the just-completed zero-based loop index.
    # Their terminal checkpoint already used total_steps, so only intermediate
    # values need advancing.
    return stored_step if stored_step >= total_steps else stored_step + 1


# --------------------------------------------------------------------------
# training


def train_row(
    spec: AblationSpec,
    *,
    output_root: Path,
    release_root: Path | None,
    budget_tokens: int,
    learning_rate: float | None,
    seed: int,
    checkpoint_every: int,
    analysis_every: int,
    telemetry_every: int,
    max_steps: int | None,
    schedule_total_steps: int | None,
    device_override: str | None,
    synthetic: bool,
    resume: bool = True,
    final_checkpoint: bool = True,
    compute_allocation_mode: str = "legacy",
    joint_router_exploration: float = 0.05,
    joint_utility_coefficient: float = 1.0,
    joint_max_passes: int | None = None,
) -> dict[str, Any]:
    runtime = initialize_runtime(device=device_override)
    lease = None
    try:
        lease = _acquire_row_lease(
            RunPaths(Path(output_root).expanduser().resolve() / spec.name),
            runtime,
        )
        return _train_row_inner(
            spec,
            runtime=runtime,
            output_root=output_root,
            release_root=release_root,
            budget_tokens=budget_tokens,
            learning_rate=learning_rate,
            seed=seed,
            checkpoint_every=checkpoint_every,
            analysis_every=analysis_every,
            telemetry_every=telemetry_every,
            max_steps=max_steps,
            schedule_total_steps=schedule_total_steps,
            synthetic=synthetic,
            resume=resume,
            final_checkpoint=final_checkpoint,
            compute_allocation_mode=compute_allocation_mode,
            joint_router_exploration=joint_router_exploration,
            joint_utility_coefficient=joint_utility_coefficient,
            joint_max_passes=joint_max_passes,
        )
    finally:
        _release_row_lease(lease)
        destroy_runtime()


def _train_row_inner(
    spec: AblationSpec,
    *,
    runtime: Runtime,
    output_root: Path,
    release_root: Path | None,
    budget_tokens: int,
    learning_rate: float | None,
    seed: int,
    checkpoint_every: int,
    analysis_every: int,
    telemetry_every: int,
    max_steps: int | None,
    schedule_total_steps: int | None,
    synthetic: bool,
    resume: bool = True,
    final_checkpoint: bool = True,
    compute_allocation_mode: str = "legacy",
    joint_router_exploration: float = 0.05,
    joint_utility_coefficient: float = 1.0,
    joint_max_passes: int | None = None,
) -> dict[str, Any]:
    if runtime.distributed and runtime.world_size != spec.apus:
        raise RuntimeError(
            f"{spec.name} is specified for {spec.apus} ranks but was launched on "
            f"{runtime.world_size}. The global batch is fixed across rows, so the "
            "rank count is part of the experiment, not a scheduling detail."
        )

    paths = RunPaths(Path(output_root).expanduser().resolve() / spec.name)
    existing_artifacts = _existing_run_artifacts(paths)
    if not resume and existing_artifacts:
        raise RuntimeError(
            "--no-resume requires a clean row output directory; found "
            f"{existing_artifacts[0]}"
        )
    record_rank_startup(
        paths.root, rank=runtime.rank, world_size=runtime.world_size,
        local_rank=runtime.local_rank, device=str(runtime.device),
    )

    # Matched rows must begin from identical weights. Row-specific randomness
    # belongs to the routing controls below, not to model initialization, or a
    # Core/RM or fixed/frozen loss gap is confounded by a different draw.
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    curriculum = spec.curriculum(random_policy_seed=seed + spec.index)

    on_accelerator = runtime.device.type == "cuda"
    config = spec.model_config(
        mhc_backend="fused_required" if on_accelerator else "torch_reference",
        mamba_backend="fused_required" if on_accelerator else "torch_reference",
        attention_backend=(
            "varlen_fused_required" if on_accelerator else "torch_reference"
        ),
    )
    if compute_allocation_mode not in {"legacy", "joint"}:
        raise ValueError("compute_allocation_mode must be legacy or joint")
    if compute_allocation_mode == "joint":
        if spec.continuation_mode != "budgeted" or spec.routed_k_mode != "budgeted":
            raise ValueError("Joint routing must not silently replace a fixed or random control")
        config = replace(config, joint_compute_router=True, budgeted_depth_values=())
        config._validate_tiny() if config.family == "tiny" else config.validate()
        curriculum = replace(
            curriculum,
            compute_allocation_mode="joint",
            joint_router_exploration=joint_router_exploration,
            joint_utility_coefficient=joint_utility_coefficient,
            max_passes=joint_max_passes,
        )
        curriculum.validate(config)
    elif joint_max_passes is not None:
        raise ValueError("joint_max_passes requires joint allocation")
    topology = build_replicated_expert_topology(runtime, family=config.family)
    policy = build_precision_policy(
        config.precision,
        profile="fp8" if on_accelerator else "bf16",
        device=runtime.device,
        # ``production`` gates the release-contract assertions, not numerical
        # rigour; the FP8 parity check below is the thing that matters here.
        production=False,
        permit_fallback=not on_accelerator,
    )
    model = Metis16ForCausalLM(
        config,
        process_groups=MetisProcessGroups(
            world=topology.dense_data_group,
            expert=topology.expert_group,
            expert_data=topology.expert_data_group,
            table_lookup=topology.expert_group,
            table_gradient=topology.dense_data_group,
        ),
        precision_policy=policy,
    )
    # Not ``model.to(device)``: the model tags routers, continuation heads, and
    # the numerically sensitive Mamba parameters for FP32 *storage*, and only
    # this call applies those tags.  Skipping it silently leaves every router in
    # BF16, which degrades exactly the discrete decisions this campaign is
    # measuring.
    model.apply_parameter_storage_policy(device=runtime.device)
    frozen_policy_parameters = _freeze_inactive_policy_parameters(
        model,
        curriculum,
    )
    _assert_storage_policy(model)
    if topology.distributed and config.ngram_memory.table_mode == "replicated":
        # N-gram tables produce sparse gradients, which the dense reducer
        # refuses to touch.  The model owns their coalesced row synchronization
        # because it knows the hashing and partition semantics; without this
        # call ``synchronize_gradients`` raises on the first multi-rank step.
        model.enable_managed_sparse_gradient_sync(topology.dense_data_group)
    broadcast_initial_parameters(model, topology)

    base_lr = (
        learning_rate
        if learning_rate is not None
        else float(config.autotune.preferred_learning_rate)
    )
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    optimizer, optimizer_summary = build_training_optimizers(
        model,
        learning_rate=base_lr,
        beta1=0.9,
        beta2=0.95,
        eps=1.0e-8,
        weight_decay=0.1,
        sparse_learning_rate_scale=float(config.ngram_memory.learning_rate_scale),
        muon_beta=0.95,
        muon_ns_steps=spec.muon_ns_steps,
        muon_nesterov=True,
        include_routed_experts=True,
        muon_state_bits=spec.muon_state_bits,
    )
    optimizer_manifest = optimizer_summary.to_dict()
    optimizer_manifest["frozen_policy_parameters"] = list(
        frozen_policy_parameters
    )
    exact_budget_groups = {
        "depth_policy": curriculum.continuation_mode == "budgeted" and compute_allocation_mode != "joint",
        "width_policy": curriculum.routed_k_mode == "budgeted" and compute_allocation_mode != "joint",
    }
    if compute_allocation_mode == "joint":
        exact_budget_groups["joint_policy"] = True
    optimizer_manifest["gradient_clipping"] = (
        {
            "mode": "independent_exact_budget_groups",
            "max_norm": BUDGET_GRADIENT_GROUP_MAX_NORM,
            "groups": exact_budget_groups,
        }
        if any(exact_budget_groups.values())
        else {"mode": "global", "max_norm": 1.0}
    )
    if spec.optimizer_sharding == "world":
        if not runtime.distributed or topology.expert_parallel_size != 1:
            raise RuntimeError(
                "World optimizer sharding requires replicated experts on a "
                "multi-rank run."
            )
        optimizer = WorldShardedOptimizerBundle(
            optimizer,
            model,
            process_group=topology.dense_data_group,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
        optimizer_manifest["sharding"] = {
            "mode": "world",
            "world_size": runtime.world_size,
            "owner_loads": list(optimizer.owner_loads),
        }
    else:
        optimizer_manifest["sharding"] = {"mode": "none"}

    if runtime.rank == 0:
        paths.prepare()
    if runtime.distributed:
        dist.barrier()

    inventory: ReleaseInventory | None = None
    sample_stream: AblationSampleStream | None = None
    if not synthetic:
        if release_root is None:
            raise ValueError("A real run requires --release-root")
        inventory = ReleaseInventory.from_release_root(Path(release_root))
        stream = DeterministicReleaseStream(
            inventory,
            sequence_length=config.sequence_length,
            # The same shard permutation as the production runs, so the proxy
            # samples the release the release was built to be sampled as.
            shard_order_seed=16_062_026,
        )
        sample_stream = build_sample_stream(
            stream,
            budget_tokens=budget_tokens,
            block_tokens=GLOBAL_BATCH_TOKENS,
        )
        total_steps = sample_stream.total_blocks
    else:
        total_steps = max(1, budget_tokens // GLOBAL_BATCH_TOKENS)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    resolved_schedule_steps = (
        total_steps if schedule_total_steps is None else int(schedule_total_steps)
    )
    if resolved_schedule_steps < total_steps:
        raise ValueError(
            "schedule_total_steps cannot be shorter than the executed run"
        )
    schedule = AblationSchedule(
        total_steps=resolved_schedule_steps,
        base_learning_rate=base_lr,
    )
    analyzer = RoutingAnalyzer(config, max_passes=config.max_passes)
    source_revision: str | None = (
        _source_revision(require_clean=not synthetic)
        if runtime.rank == 0
        else None
    )
    runtime_fingerprint: dict[str, Any] | None = (
        _runtime_fingerprint(runtime, require_complete=not synthetic)
        if runtime.rank == 0
        else None
    )
    if runtime.distributed:
        source_payload = [source_revision, runtime_fingerprint]
        dist.broadcast_object_list(
            source_payload,
            src=0,
            group=topology.dense_data_group,
        )
        source_revision, runtime_fingerprint = source_payload
    if not isinstance(source_revision, str) or not source_revision:
        raise RuntimeError("Ablation source revision could not be established")
    if not isinstance(runtime_fingerprint, dict):
        raise RuntimeError("Ablation runtime fingerprint could not be established")
    sampler_manifest = (
        sample_stream.describe() if sample_stream else {"synthetic": True}
    )
    release_manifest = (
        {
            "release_sha256": inventory.release_sha256,
            "shard_manifest_sha256": inventory.shard_manifest_sha256,
        }
        if inventory is not None
        else {"synthetic": True}
    )
    campaign_identity_sha256 = _validate_campaign_identity(
        Path(output_root).expanduser().resolve(),
        {
            "schema": "more.ablation-campaign-core/v1",
            "wave": wave_for_row(spec.name),
            "source_revision": source_revision,
            "runtime": runtime_fingerprint,
            "release": release_manifest,
            "seed": int(seed),
            "total_steps": total_steps,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "sampler": sampler_manifest,
        },
        runtime,
    )
    identity_payload, identity_sha256 = _run_identity(
        {
            "schema": "more.ablation-run-identity/v1",
            "row": spec.name,
            "row_index": spec.index,
            "model": config.to_dict(),
            "curriculum": asdict(curriculum),
            "optimizer": optimizer_manifest,
            "schedule": asdict(schedule),
            "seed": int(seed),
            "world_size": runtime.world_size,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "precision_profile": policy.requested_profile,
            "sampler": sampler_manifest,
            "release": release_manifest,
            "source_revision": source_revision,
            "runtime": runtime_fingerprint,
            "campaign_identity_sha256": campaign_identity_sha256,
        }
    )

    # Resume before the reducer is built: loading a state dict in place keeps
    # parameter identity, but restoring after registering post-accumulate hooks
    # would be fragile for no benefit.
    start_step = 0
    resumed_from: str | None = None
    if resume:
        checkpoint = _latest_checkpoint(paths)
        if checkpoint is None and existing_artifacts:
            raise RuntimeError(
                "Existing run artifacts have no resumable checkpoint; use a "
                "new output directory rather than appending a fresh run"
            )
        if checkpoint is not None:
            start_step = _restore_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                device=runtime.device,
                total_steps=total_steps,
                learning_rate=base_lr,
                expected_run_identity_sha256=identity_sha256,
            )
            resumed_from = str(checkpoint)
        if topology.distributed:
            # Every rank must agree on the resume point or the data stream and
            # the collective sequence diverge immediately.
            agreed = all_reduce_sum(
                torch.tensor([float(start_step)], device=runtime.device), topology
            )
            expected = float(start_step) * runtime.world_size
            if abs(float(agreed[0].item()) - expected) > 0.5:
                raise RuntimeError(
                    "Ranks disagree about the resume step; the shared checkpoint "
                    "directory is inconsistent."
                )

    _truncate_telemetry(
        paths.telemetry / f"rank-{runtime.rank:05d}.jsonl",
        start_step=start_step,
    )
    if runtime.rank == 0 and paths.analysis.exists():
        for path in paths.analysis.glob("routing-step-*.json"):
            try:
                analyzed_step = int(path.stem.removeprefix("routing-step-"))
            except ValueError:
                continue
            if analyzed_step >= start_step:
                path.unlink()
    if runtime.distributed:
        dist.barrier()

    gradient_reducer = (
        OverlappedGradientReducer(model, topology) if topology.distributed else None
    )
    metrics = MetricsWriter(paths.root, enabled=True, rank=runtime.rank)

    audit = config.logical_parameter_audit()
    if runtime.rank == 0:
        manifest = {
            "schema": "more.ablation-run/v1",
            "spec": asdict(spec),
            "model": config.to_dict(),
            "optimizer": optimizer_manifest,
            "parameters": {
                "stored_total": audit.stored_total,
                "active_per_pass_mean": audit.active_per_pass_mean,
            },
            "schedule": asdict(schedule),
            "curriculum": asdict(curriculum),
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "total_steps": total_steps,
            "start_step": start_step,
            "resumed_from": resumed_from,
            "world_size": runtime.world_size,
            "precision_profile": policy.requested_profile,
            "run_identity": identity_payload,
            "run_identity_sha256": identity_sha256,
            "campaign_identity_sha256": campaign_identity_sha256,
            "final_checkpoint": bool(final_checkpoint),
            "sampler": sampler_manifest,
        }
        (paths.root / "run.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    summary: dict[str, Any] = {
        "row": spec.name,
        "resumed_from": resumed_from,
        "start_step": start_step,
        "steps": start_step,
        "tokens": 0,
        "final_loss": None,
    }
    total_tokens = 0
    # FP8 parity is measured once, on the first batch of the run.
    fp8_parity_error: float | None = None
    reference_loss: float | None = None
    started = time.perf_counter()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    release_batches = (
        _PrefetchedBatches(
            (
                batch
                for pending in range(start_step, total_steps)
                for batch in sample_stream.micro_batches(
                    step=pending,
                    rank=runtime.rank,
                    world_size=runtime.world_size,
                    micro_batch_size=spec.micro_batch,
                    grad_accum=spec.grad_accum,
                )
            ),
            depth=2 * spec.grad_accum,
        )
        if sample_stream is not None
        else None
    )

    with metrics:
        for step in range(start_step, total_steps):
            optimizer.zero_grad(set_to_none=True)
            set_optimizer_learning_rate(optimizer, schedule.learning_rate(step))
            collect_analysis = (
                analysis_every > 0 and step % analysis_every == 0 and step > 0
            )
            step_started = time.perf_counter()

            loss_numerator = 0.0
            supervised_total = 0
            step_telemetry: dict[str, Any] = {}
            joint_step_flops = 0.0
            joint_step_router_flops = 0.0
            joint_step_active_tokens = 0.0
            joint_step_observations = 0.0
            expert_selection_counts: torch.Tensor | None = None
            depth_histogram = torch.zeros(
                config.max_passes + 1, device=runtime.device, dtype=torch.long
            )

            batches = (
                itertools.islice(release_batches, spec.grad_accum)
                if release_batches is not None
                else _synthetic_batches(
                    config=config,
                    spec=spec,
                    step=step,
                    rank=runtime.rank,
                    device=runtime.device,
                )
            )

            for accumulation_index, cpu_batch in enumerate(batches):
                if gradient_reducer is not None and accumulation_index == spec.grad_accum - 1:
                    gradient_reducer.arm()
                batch = (
                    cpu_batch
                    if cpu_batch.input_ids.device == runtime.device
                    else cpu_batch.to(runtime.device)
                )
                analysis_context = (
                    analyzer.capture(model)
                    if collect_analysis and accumulation_index == 0
                    else nullcontext()
                )
                # The random controls draw from (seed, step, layer, pass), so
                # the step has to reach the model; without it every step of a
                # random row would draw the identical coalition.
                step_curriculum = replace(curriculum, random_policy_step=step)
                forward_kwargs = dict(
                    curriculum=step_curriculum,
                    attention_mask=batch.attention_mask,
                    document_ids=batch.document_ids,
                    reset_mask=batch.reset_mask,
                    canonical_ids=batch.canonical_ids,
                    return_logits=False,
                )
                if (
                    fp8_parity_error is None
                    and policy.fp8_enabled
                    and step == start_step
                    and accumulation_index == 0
                ):
                    # One BF16 reference forward on the first batch, with RNG
                    # restored so the FP8 forward that follows is the one the
                    # run would have taken anyway. This is the campaign's
                    # equivalent of the production parity gate: FP8 buys
                    # throughput, and a row whose numerics drifted would be
                    # compared against twelve rows whose numerics did not.
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = (
                        torch.cuda.get_rng_state(runtime.device)
                        if runtime.device.type == "cuda"
                        else None
                    )
                    with torch.no_grad(), policy.bf16_reference_context():
                        reference = model(batch.input_ids, batch.labels, **forward_kwargs)
                    reference_loss = float(reference.loss.detach().float().item())
                    if not math.isfinite(reference_loss):
                        raise FloatingPointError(
                            "BF16 reference produced a non-finite first-batch loss"
                        )
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, runtime.device)
                    del reference
                with analysis_context:
                    output = model(batch.input_ids, batch.labels, **forward_kwargs)
                if reference_loss is not None and fp8_parity_error is None:
                    actual = float(output.loss.detach().float().item())
                    if not math.isfinite(actual):
                        raise FloatingPointError(
                            "FP8 execution produced a non-finite first-batch loss"
                        )
                    fp8_parity_error = abs(actual - reference_loss) / max(
                        abs(reference_loss), 1e-9
                    )
                    limit = float(config.autotune.gates.max_fp8_loss_relative_error)
                    if fp8_parity_error > limit:
                        raise RuntimeError(
                            f"FP8 loss diverged from the BF16 reference by "
                            f"{fp8_parity_error:.4f} (gate {limit}); rerun this row "
                            "with a BF16 profile rather than comparing it against "
                            "rows that stayed in parity."
                        )
                loss = output.loss + output.auxiliary_loss
                supervised = int(batch.supervised_tokens)
                if supervised > 0:
                    (loss * float(supervised)).backward()
                else:
                    (loss * 0.0).backward()

                loss_numerator += float(output.loss.detach().float().item()) * supervised
                supervised_total += supervised
                total_tokens += int(batch.non_padding_tokens)

                telemetry = output.telemetry
                if compute_allocation_mode == "joint":
                    joint_step_flops += float(telemetry["joint_model_flops"].detach().item())
                    joint_step_router_flops += float(telemetry["joint_router_flops"].detach().item())
                    joint_step_active_tokens += float(telemetry["executed_active_tokens"].detach().item())
                    joint_step_observations += float(telemetry["joint_utility_observations"].detach().item())
                counts = telemetry.get("expert_selection_counts")
                if isinstance(counts, torch.Tensor):
                    detached = counts.detach()
                    expert_selection_counts = (
                        detached.clone()
                        if expert_selection_counts is None
                        else expert_selection_counts + detached
                    )
                for key, value in telemetry.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        step_telemetry[key] = float(value.detach().float().item())
                    elif isinstance(value, (int, float)):
                        step_telemetry[key] = float(value)
                depths = output.chosen_depths.masked_select(batch.attention_mask)
                if depths.numel():
                    depth_histogram += torch.bincount(
                        depths, minlength=config.max_passes + 1
                    )
                if collect_analysis and accumulation_index == 0:
                    analyzer.observe(output, batch.attention_mask)

            # Losses were summed with token weights; reduce the weights too so
            # every rank divides by the same global denominator.
            reduced = all_reduce_sum(
                torch.tensor(
                    [loss_numerator, float(supervised_total)],
                    device=runtime.device,
                    dtype=torch.float64,
                ),
                topology,
            )
            global_loss_numerator = float(reduced[0].item())
            global_supervised = int(reduced[1].item())
            if global_supervised <= 0:
                raise RuntimeError("An optimizer step contained no supervised tokens")

            synchronize_gradients(model, topology, reducer=gradient_reducer)
            normalize_summed_gradients(
                model, topology, global_supervised_tokens=global_supervised
            )
            if (
                compute_allocation_mode == "joint"
                or curriculum.continuation_mode == "budgeted"
                or curriculum.routed_k_mode == "budgeted"
            ):
                gradient_group_norms = _clip_exact_budget_gradient_groups(
                    model,
                    curriculum,
                )
                grad_norm = gradient_group_norms["model"]
                step_telemetry.update(
                    {
                        f"{name}_grad_norm": value
                        for name, value in gradient_group_norms.items()
                    }
                )
            else:
                grad_norm = float(
                    clip_grad_norm_(model, 1.0, topology=topology).detach().item()
                )
            step_loss = global_loss_numerator / global_supervised

            # Freshly initialized models spike the pre-clip gradient norm for
            # the first handful of steps.  Gradients are clipped either way; the
            # gate exists to catch divergence later in the run, and aborting a
            # 20-hour row on a step-zero transient would be the gate causing the
            # failure it is meant to detect.
            warmup_steps = max(
                100,
                int(schedule.total_steps * schedule.warmup_fraction),
            )
            enforce_health_gates(
                loss=step_loss,
                grad_norm_value=grad_norm,
                telemetry=step_telemetry,
                maximum_grad_norm=(
                    math.inf
                    if step < warmup_steps
                    else float(config.autotune.gates.max_grad_norm)
                ),
                abort_on_nonfinite=True,
                abort_on_token_drop=True,
                # The ablation ladder deliberately contains rows whose routing
                # is degenerate by construction -- fixed k, random k, depth one
                # -- so the entropy and halt-collapse gates that protect a
                # production run would abort the experiment they exist to
                # measure.  Non-finite loss and token drops still abort.
                minimum_expert_entropy_ratio=0.0,
                maximum_expert_load_cv=math.inf,
                maximum_halt_collapse_fraction=math.inf,
                require_structural_telemetry=False,
            )

            optimizer.step()
            if compute_allocation_mode == "joint" and curriculum.joint_utility_coefficient > 0.0:
                global_utility_observations = all_reduce_sum(
                    torch.tensor([joint_step_observations], device=runtime.device),
                    topology,
                )
                if float(global_utility_observations[0].item()) > 0:
                    model.joint_router.mark_trained()
            updater = getattr(model, "update_expert_selection_biases", None)
            if callable(updater) and expert_selection_counts is not None:
                updater(expert_selection_counts)

            elapsed = time.perf_counter() - step_started
            if step % max(1, telemetry_every) == 0 or step == total_steps - 1:
                global_depth = all_reduce_sum(depth_histogram, topology)
                flops = estimate_hardware_flops(
                    config,
                    tokens=GLOBAL_BATCH_TOKENS,
                    observed_mean_passes=step_telemetry.get("mean_depth"),
                    observed_mean_routed_k=step_telemetry.get("mean_routed_k"),
                )
                if compute_allocation_mode == "joint":
                    joint_totals = all_reduce_sum(
                        torch.tensor(
                            [joint_step_flops, joint_step_router_flops, joint_step_active_tokens],
                            device=runtime.device,
                            dtype=torch.float64,
                        ),
                        topology,
                    )
                    model_flops, router_flops, active_tokens = [
                        float(value.item()) for value in joint_totals
                    ]
                    replay_flops = (
                        (model_flops - router_flops) / 3.0
                        if config.activation_recompute_policy in {"pass", "layer"}
                        else 2.0 * config.vocab_size * config.d_model * active_tokens
                    )
                    flops = model_flops + replay_flops
                    step_telemetry["global_joint_model_flops"] = model_flops
                    step_telemetry["global_joint_router_flops"] = router_flops
                memory = peak_memory_evidence(runtime.device)
                metrics.write(
                    {
                        "row": spec.name,
                        "row_index": spec.index,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": step_loss,
                        "learning_rate": schedule.learning_rate(step),
                        "grad_norm": grad_norm,
                        "global_supervised_tokens": global_supervised,
                        "cumulative_tokens": (step + 1) * GLOBAL_BATCH_TOKENS,
                        "step_time_s": elapsed,
                        "tokens_per_second": GLOBAL_BATCH_TOKENS / max(elapsed, 1e-9),
                        "estimated_hardware_flops": flops,
                        "estimated_mfu": estimated_mfu(
                            estimated_flops=flops,
                            elapsed_seconds=elapsed,
                            world_size=runtime.world_size,
                            precision_profile=policy.requested_profile,
                        ),
                        "depth_histogram": global_depth.tolist(),
                        "fp8_parity_relative_error": fp8_parity_error,
                        "telemetry": step_telemetry,
                        **memory,
                    }
                )
            if collect_analysis and runtime.rank == 0:
                analyzer.flush(paths.analysis / f"routing-step-{step:07d}.json", step=step)
            completed_steps = step + 1
            if (
                checkpoint_every > 0
                and completed_steps % checkpoint_every == 0
            ):
                _save_checkpoint(
                    paths,
                    model=model,
                    optimizer=optimizer,
                    step=completed_steps,
                    spec=spec,
                    rank=runtime.rank,
                    device=runtime.device,
                    total_steps=total_steps,
                    learning_rate=base_lr,
                    run_identity_sha256=identity_sha256,
                )
            summary["steps"] = step + 1
            summary["final_loss"] = step_loss

    if final_checkpoint:
        _save_checkpoint(
            paths,
            model=model,
            optimizer=optimizer,
            step=total_steps,
            spec=spec,
            rank=runtime.rank,
            device=runtime.device,
            total_steps=total_steps,
            learning_rate=base_lr,
            run_identity_sha256=identity_sha256,
        )
    summary["fp8_parity_relative_error"] = fp8_parity_error
    summary["tokens"] = summary["steps"] * GLOBAL_BATCH_TOKENS
    summary["wall_clock_s"] = time.perf_counter() - started
    if runtime.rank == 0:
        (paths.root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    return summary


class _PrefetchedBatches:
    """Read the next micro-batch while the current one is still computing.

    The sampler pulls a micro-batch out of the release when the training loop
    asks for it, so every accumulation step began with a blocking Lustre read
    and a host-to-device copy that nothing overlapped. Measured on the same row
    at the same rank count: 43 s per optimizer step against the release and
    21 s against synthetic batches. Half the campaign's wall clock was the
    loader, and none of it was work.

    One background thread runs one micro-batch ahead and pins its tensors, so
    the copy the training loop still issues is from pinned memory and can
    overlap. The thread only reads; it never touches the device, which keeps
    every CUDA call on the loop's own thread and in the loop's own order.
    """

    _DONE = object()

    def __init__(self, batches: Any, *, depth: int = 2) -> None:
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, depth))
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._pump, args=(batches,), daemon=True
        )
        self._thread.start()

    def _pump(self, batches: Any) -> None:
        try:
            for batch in batches:
                self._queue.put(batch.pin_memory())
        except BaseException as error:  # surfaced on the consuming thread
            self._error = error
        finally:
            self._queue.put(self._DONE)

    def __iter__(self) -> "Iterator[Any]":
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is self._DONE:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return item


def _synthetic_batches(*, config: Any, spec: AblationSpec, step: int, rank: int, device):
    """Deterministic fake batches for smoke tests and CPU dry runs."""

    from metis_training.data import TrainingBatch

    generator = torch.Generator(device="cpu").manual_seed(step * 1_000_003 + rank)
    for _ in range(spec.grad_accum):
        shape = (spec.micro_batch, config.sequence_length)
        input_ids = torch.randint(
            0, config.vocab_size, shape, generator=generator, dtype=torch.long
        )
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[:, -1] = -100
        attention_mask = torch.ones(shape, dtype=torch.bool)
        reset_mask = torch.zeros(shape, dtype=torch.bool)
        reset_mask[:, 0] = True
        yield TrainingBatch(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            attention_mask=attention_mask.to(device),
            reset_mask=reset_mask.to(device),
            document_ids=torch.zeros(shape, dtype=torch.int32, device=device),
            canonical_ids=input_ids.to(device),
            phase="phase_a",
            global_token_cursor=step * GLOBAL_BATCH_TOKENS,
            next_global_token_cursor=(step + 1) * GLOBAL_BATCH_TOKENS,
            supervised_tokens=int(labels.ne(-100).sum().item()),
            non_padding_tokens=int(attention_mask.sum().item()),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one MoRE ablation row")
    parser.add_argument("--row", required=True, help="Ablation row name, e.g. more-core")
    parser.add_argument("--compute-allocation-mode", choices=("legacy", "joint"), default="legacy")
    parser.add_argument("--joint-router-exploration", type=float, default=0.05)
    parser.add_argument("--joint-utility-coefficient", type=float, default=1.0)
    parser.add_argument("--joint-max-passes", type=int, default=None)
    parser.add_argument(
        "--diagnostic-apus", type=int, default=None,
        help="Explicit short-canary DP size; preserve the row micro-batch and global token batch",
    )
    parser.add_argument("--output", required=True, help="Campaign output root")
    parser.add_argument("--release-root", default=None, help="1T release inventory root")
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=16_062_026)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--analysis-every", type=int, default=1_000)
    parser.add_argument("--telemetry-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        default=None,
        help=(
            "Use a longer reference schedule while executing a short canary; "
            "the LR sweep uses the full 50B horizon"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start from step zero",
    )
    parser.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="Skip the terminal checkpoint for disposable throughput canaries",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Random tokens instead of the release; smoke tests only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = spec_by_name(args.row)
    if args.diagnostic_apus is not None:
        if args.max_steps is None or args.max_steps < 1:
            raise ValueError("A diagnostic allocation requires an explicit positive max_steps")
        denominator = args.diagnostic_apus * spec.micro_batch
        if args.diagnostic_apus < 2 or denominator <= 0 or GLOBAL_BATCH_SEQUENCES % denominator:
            raise ValueError("Diagnostic APUs must tile the unchanged global batch with this micro-batch")
        spec = replace(
            spec,
            apus=args.diagnostic_apus,
            grad_accum=GLOBAL_BATCH_SEQUENCES // denominator,
            measured_tokens_per_second=None,
            notes=spec.notes + " Explicit diagnostic DP geometry; original lane throughput does not apply.",
        )
    summary = train_row(
        spec,
        output_root=Path(args.output),
        release_root=Path(args.release_root) if args.release_root else None,
        budget_tokens=args.budget_tokens,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        analysis_every=args.analysis_every,
        telemetry_every=args.telemetry_every,
        max_steps=args.max_steps,
        schedule_total_steps=args.schedule_total_steps,
        device_override=args.device,
        synthetic=args.synthetic,
        resume=not args.no_resume,
        final_checkpoint=not args.no_final_checkpoint,
        compute_allocation_mode=args.compute_allocation_mode,
        joint_router_exploration=args.joint_router_exploration,
        joint_utility_coefficient=args.joint_utility_coefficient,
        joint_max_passes=args.joint_max_passes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
