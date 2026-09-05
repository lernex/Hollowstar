"""Sealed, aggregate-only diagnostics of frozen MoRE routing on release data.

This is a diagnostic warm start, never a distributed optimizer resume. Public
helpers keep token-level tensors in memory for callers fitting utility heads;
the CLI writes only the explicitly constructed aggregate report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import subprocess
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import Tensor

from metis_training.data import (
    DeterministicReleaseStream,
    ReleaseInventory,
    TrainingBatch,
)
from metis_training.metrics import estimate_train_flops
from metis_training.model import AdaptiveDroplessMoE, CurriculumState, DenseFFN, Metis16ForCausalLM
from metis_training.model_config import Metis16Config
from metis_training.precision import build_precision_policy

from .sampler import AblationSampleStream, build_sample_stream


SHARD_ORDER_SEED = 16_062_026
UTILITY_SCHEMA = "more.routing-utility/v1"


class CapabilityError(RuntimeError):
    """The requested probe needs an API or artifact that is not available."""


def _select_probe_device(device: torch.device) -> torch.device:
    """Keep implicit TE/Triton allocations on the explicit probe device."""
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def identify_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being hashed: {path}")
    return {
        "path": str(path),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def assert_file_unchanged(identity: Mapping[str, Any]) -> None:
    stat = Path(identity["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
        raise RuntimeError(f"Input changed during the probe: {identity['path']}")


def source_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit tracked changes before measuring a routing probe")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src")
    if any(path.endswith(".py") for path in untracked.splitlines()):
        raise RuntimeError("Commit untracked source modules before measuring a routing probe")
    relative = str(Path(__file__).resolve().relative_to(repository))
    git("ls-files", "--error-unmatch", relative)
    return {
        "revision": revision,
        "tracked_worktree_clean": True,
        "probe_sha256": sha256_file(Path(__file__).resolve()),
    }


def fresh_output_directory(path: Path, *, inputs: tuple[Path, ...] = ()) -> Path:
    raw = path.expanduser().absolute()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"Probe output must be a fresh directory: {raw}")
    resolved = raw.resolve()
    for item in inputs:
        item = item.expanduser().resolve()
        protected = item if item.is_dir() else item.parent
        if resolved == protected or protected in resolved.parents:
            raise ValueError("Probe output must not be inside an input directory")
    # mkdir(exist_ok=False) is also the race-safe ownership claim.
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def infer_run_manifest(checkpoint: Path) -> Path:
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if directory.parent.name != "checkpoints" or not directory.name.startswith("step-"):
        raise ValueError("Cannot infer run.json; supply --run-manifest explicitly")
    return directory.parent.parent / "run.json"


def validate_checkpoint(payload: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") not in {
        "more.ablation-checkpoint/v2", "more.ablation-checkpoint/v3",
    }:
        raise ValueError("Unsupported or missing ablation checkpoint schema")
    required = ("model", "spec", "step", "step_semantics", "run_identity_sha256")
    if any(name not in payload for name in required):
        raise ValueError("Checkpoint is missing required model/identity fields")
    if payload["step_semantics"] != "next_unexecuted":
        raise ValueError("Checkpoint step must mean next_unexecuted")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if not isinstance(payload["model"], Mapping) or not payload["model"]:
        raise ValueError("Checkpoint has no model state dictionary")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Run manifest has no sealed run_identity")
    expected = json_sha256(identity)
    if manifest.get("run_identity_sha256") != expected:
        raise ValueError("Run manifest identity checksum does not match")
    if payload["run_identity_sha256"] != expected:
        raise ValueError("Checkpoint and run manifest identities differ")
    if manifest.get("schema") != "more.ablation-run/v1":
        raise ValueError("Unsupported run manifest schema")
    for name in ("model", "curriculum", "sampler", "precision_profile"):
        if manifest.get(name) != identity.get(name) or name not in identity:
            raise ValueError(f"Run manifest {name} differs from its sealed identity")
    if payload["spec"] != manifest.get("spec"):
        raise ValueError("Checkpoint and run manifest specs differ")
    return identity


def unpack_active(values: Tensor, active_mask: Tensor) -> Tensor:
    """Scatter the model's [1, active_tokens, ...] layout into [B, S, ...]."""
    if active_mask.ndim != 2 or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be boolean [batch, sequence]")
    if values.ndim < 2:
        raise ValueError("Packed values must have at least two dimensions")
    if values.shape[:2] == active_mask.shape:
        return values.masked_fill(
            ~active_mask.reshape(*active_mask.shape, *([1] * (values.ndim - 2))), 0,
        )
    indices = active_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if values.shape[:2] != (1, indices.numel()):
        raise ValueError("Packed values do not match the pass's active token count")
    flat = values.new_zeros((active_mask.numel(), *values.shape[2:]))
    return flat.index_copy(0, indices, values.squeeze(0)).reshape(
        *active_mask.shape, *values.shape[2:],
    )


class RoutingCapture:
    """Forward-only hook tape; never interpret packed rows as original tokens."""

    def __init__(self, model: Metis16ForCausalLM, *, continuation: bool = False):
        self.model = model
        self.capture_continuation = continuation
        self.widths: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self.probabilities: list[Tensor] = []
        self._handles: list[Any] = []

    def __enter__(self) -> "RoutingCapture":
        for layer_index, layer in enumerate(self.model.layers):
            if not isinstance(layer.moe, (AdaptiveDroplessMoE, DenseFFN)):
                self.__exit__()
                raise CapabilityError("Model assessment requires an instrumented MoE or dense FFN")

            def hook(_module, _args, kwargs, output, index=layer_index):
                key = (int(kwargs["pass_index"]), index)
                if key in self.widths:
                    raise RuntimeError("Duplicate layer/pass hook: activation replay is not disabled")
                self.widths[key] = (
                    output[1].mean_k.detach().clone(),
                    kwargs["active_mask"].detach().clone(),
                )

            self._handles.append(layer.moe.register_forward_hook(hook, with_kwargs=True))
        if self.capture_continuation:
            self._handles.append(self.model.continuation.register_forward_hook(
                lambda _module, _args, output: self.probabilities.append(output),
            ))
        return self

    def __exit__(self, *_args):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def full_widths(self, active_masks: Tensor) -> Tensor:
        config = self.model.config
        widths = torch.zeros(
            (config.max_passes, config.n_layers, *active_masks.shape[1:]),
            device=active_masks.device, dtype=torch.long,
        )
        expected = {
            (p, layer) for p, mask in enumerate(active_masks)
            if bool(mask.any()) for layer in range(config.n_layers)
        }
        if set(self.widths) != expected:
            raise RuntimeError("MoE hook coverage has missing or duplicate layer/pass work")
        for (p, layer), (values, local_mask) in self.widths.items():
            mask = active_masks[p]
            if values.shape != local_mask.shape:
                raise ValueError("RouteState.mean_k and hook active_mask shapes differ")
            if not torch.equal(unpack_active(local_mask, mask), mask):
                raise ValueError("MoE local active mask does not match the original pass")
            unpacked = unpack_active(values, mask)
            valid = unpacked[mask]
            if not bool(torch.isfinite(valid).all()) or not torch.equal(valid, valid.round()):
                raise ValueError("Actual chosen k must be finite integers")
            invalid = (
                valid.ne(0) if config.ffn_mode == "dense"
                else (valid < config.min_routed_k) | (valid > config.max_routed_k)
            )
            if bool(invalid.any()):
                raise ValueError("Actual chosen k is outside model bounds")
            widths[p, layer] = unpacked.long()
        return widths


@dataclass(frozen=True)
class TokenPair:
    shallow: int
    deep: int
    shallow_depth: int
    deep_depth: int

    @property
    def boundary(self) -> int:
        return self.shallow_depth - 1


def select_depth_pairs(
    depths: Tensor, attention_mask: Tensor, *, pairs: int, seed: int,
    document_ids: Tensor | None = None,
) -> list[TokenPair]:
    """Disjoint unequal-depth pairs, confined to one sequence/document."""
    if pairs < 0 or depths.shape != attention_mask.shape or depths.ndim != 2:
        raise ValueError("Invalid pair count or depth/mask shape")
    if document_ids is not None and document_ids.shape != depths.shape:
        raise ValueError("document_ids must match depths")
    flat_depths = depths.detach().cpu().reshape(-1)
    valid = attention_mask.detach().cpu().reshape(-1).nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    order = valid[torch.randperm(valid.numel(), generator=generator)].tolist()
    docs = document_ids.detach().cpu().reshape(-1) if document_ids is not None else None
    length = depths.shape[1]
    used: set[int] = set()
    selected: list[TokenPair] = []
    for first in order:
        if first in used or len(selected) >= pairs:
            continue
        for second in order:
            if second == first or second in used or first // length != second // length:
                continue
            if docs is not None and int(docs[first]) != int(docs[second]):
                continue
            a, b = int(flat_depths[first]), int(flat_depths[second])
            if a == b:
                continue
            shallow, deep = (first, second) if a < b else (second, first)
            selected.append(TokenPair(shallow, deep, min(a, b), max(a, b)))
            used.update((first, second))
            break
    return selected


def swap_plan(
    depths: Tensor, widths: Tensor, pair: TokenPair,
) -> tuple[Tensor, Tensor]:
    """Swap complete histories, not average k, to conserve every work unit."""
    if widths.ndim != 4 or widths.shape[2:] != depths.shape:
        raise ValueError("Widths must have shape [passes, layers, batch, sequence]")
    swapped_depths, swapped_widths = depths.clone(), widths.clone()
    indices = torch.tensor([pair.shallow, pair.deep], device=depths.device)
    swapped_depths.reshape(-1)[indices] = depths.reshape(-1)[indices.flip(0)]
    swapped_widths.flatten(2)[:, :, indices] = widths.flatten(2)[:, :, indices.flip(0)]
    return swapped_depths, swapped_widths


def plan_cost(
    config: Metis16Config, depths: Tensor, widths: Tensor, *, terminal_only: bool = False,
) -> dict[str, Any]:
    """Audit a complete trajectory, excluding the separately reported critic.

    Causal fixed controls use one terminal head; outcome policies pay at each
    active pass. Identical depth/width plans therefore need an explicit head regime.
    Terminal-action critics instead prepay one head outside all pass costs.
    """
    if widths.shape != (config.max_passes, config.n_layers, *depths.shape):
        raise ValueError("Widths do not match the model's full plan geometry")
    if depths.dtype not in (torch.int32, torch.int64) or widths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cost plans require integer depths and widths")
    if bool(((depths < 0) | (depths > config.max_passes)).any()):
        raise ValueError("Depth plan is outside the architectural cap")
    expert_cost = (
        0 if config.ffn_mode == "dense"
        else 18 * config.latent_dim * config.expert_intermediate_dim
    )
    causal = bool(getattr(config, "causal_compute_budget", False))
    costs = None
    if causal:
        from metis_training.compute_router import JointComputeCosts

        costs = JointComputeCosts.from_config(config)
        reference = None
    else:
        if terminal_only:
            raise ValueError("Terminal-only cost accounting requires a causal-budget model")
        reference = replace(config, joint_compute_router=False) if hasattr(config, "joint_compute_router") else config
    previous = 0
    modeled = 0
    active_counts, assignments, histograms = [], [], []
    for p in range(config.max_passes):
        active = depths > p
        valid = widths[p, :, active]
        invalid = (
            valid.ne(0) if config.ffn_mode == "dense"
            else (valid < config.min_routed_k) | (valid > config.max_routed_k)
        )
        if valid.numel() and bool(invalid.any()):
            raise ValueError("Active plan widths are outside routed-k bounds")
        if bool((widths[p, :, ~active] != 0).any()):
            raise ValueError("Inactive plan widths must be zero")
        if costs is None:
            prefix = round(estimate_train_flops(
                reference, tokens=1, observed_mean_passes=float(p + 1),
                observed_mean_routed_k=1.0,
            ) - (p + 1) * config.n_layers * expert_cost)
            base = prefix - previous
            previous = prefix
        else:
            base = costs.base_pass_costs[p]
        count = int(active.sum())
        layer_assignments = [int(widths[p, layer].sum()) for layer in range(config.n_layers)]
        modeled += count * base + (
            sum(layer_assignments) * expert_cost if costs is None else
            sum(value * cost for value, cost in zip(layer_assignments, costs.expert_costs, strict=True))
        )
        active_counts.append(count)
        assignments.append(layer_assignments)
        histograms.append([histogram(widths[p, layer][active]) for layer in range(config.n_layers)])
    tokens = int((depths > 0).sum())
    token_passes = int(depths.sum())
    prepaid_terminal = costs is not None and bool(getattr(costs, "terminal_head_only", False))
    if prepaid_terminal:
        terminal_only = True
        modeled += tokens * costs.head_per_token
    elif costs is not None and terminal_only:
        modeled -= (token_passes - tokens) * costs.head_per_token
    result = {
        "nominal_train_flops": modeled,
        "nominal_forward_flops": modeled / 3.0,
        "active_tokens_by_pass": active_counts,
        "expert_assignments_by_pass_layer": assignments,
        "chosen_k_by_pass_layer": histograms,
        "tokens": tokens,
        "token_passes": token_passes,
        "expert_assignments": sum(map(sum, assignments)),
    }
    if costs is not None:
        result.update({
            "accounting_basis": "causal_shared_cost_ledger",
            "terminal_only_head": terminal_only,
            "modeled_lm_head_tokens": tokens if terminal_only else token_passes,
            "head_cost_mode": (
                "prepaid_terminal" if prepaid_terminal else
                "terminal_deduction" if terminal_only else "per_active_pass"
            ),
            "removed_legacy_policy_train_flops": token_passes * costs.removed_policy_per_pass,
            "critic_included": False,
        })
    return result


def _terminal_only_cost(
    config: Metis16Config, curriculum: CurriculumState, return_router_observations: bool,
) -> bool:
    return bool(
        getattr(config, "causal_compute_budget", False)
        and (
            getattr(config, "terminal_action_critic", False)
            or (curriculum.compute_allocation_mode != "joint" and not return_router_observations)
        )
    )


def histogram(values: Tensor) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values.detach().cpu().reshape(-1).tolist()).items())}


def credit_alignment(predicted: float, observed: float, *, tolerance: float = 1e-8) -> str:
    if not math.isfinite(predicted) or not math.isfinite(observed):
        raise ValueError("Credit comparison requires finite loss deltas")
    if abs(predicted) <= tolerance:
        return "predicted_tie"
    if abs(observed) <= tolerance:
        return "observed_tie"
    return "aligned" if (predicted > 0) == (observed > 0) else "opposed"


def repeat_noise(losses: list[float], *, minimum_delta: float = 1e-5) -> dict[str, Any]:
    if len(losses) < 2 or any(not math.isfinite(value) for value in losses):
        raise ValueError("Noise estimation requires at least two finite repeated losses")
    if minimum_delta < 0 or not math.isfinite(minimum_delta):
        raise ValueError("The minimum decisive loss delta must be finite and nonnegative")
    spread = max(losses) - min(losses)
    scale = max(max(map(abs, losses)), 1.0)
    threshold = max(minimum_delta, 3.0 * spread, 8.0 * torch.finfo(torch.float32).eps * scale)
    return {
        "repeat_losses": losses,
        "repeat_count": len(losses),
        "repeat_loss_range": spread,
        "decisive_absolute_loss_delta_threshold": threshold,
        "threshold_rule": "max(minimum_delta, 3*repeat_range, 8*float32_epsilon*loss_scale)",
        "limitation": "A small same-plan repeat sample measures replay noise, not a confidence interval or BF16 accuracy.",
    }


def plan_repeat_statistics(
    records: list[Mapping[str, Any]], *, minimum_loss_delta: float,
) -> dict[str, Any]:
    losses = [float(record["lm_loss"]) for record in records]
    noise = repeat_noise(losses, minimum_delta=minimum_loss_delta)
    mean = sum(losses) / len(losses)
    distinct = len({record["plan_sha256"] for record in records})
    return {
        **noise, "distinct_complete_plans": distinct,
        "same_complete_plan_every_repeat": distinct == 1,
        "mean_lm_loss": mean,
        "sample_standard_deviation": math.sqrt(sum((value - mean) ** 2 for value in losses) / (len(losses) - 1)),
        "min_lm_loss": min(losses), "max_lm_loss": max(losses),
    }


def aggregate_pairs(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["alignment"] for record in records)
    compared = counts["aligned"] + counts["opposed"]
    return {
        "pairs": len(records),
        "alignment_counts": dict(sorted(counts.items())),
        "non_tied_pairs": compared,
        "credit_alignment_fraction": counts["aligned"] / compared if compared else None,
        "mean_global_loss_delta": (
            sum(float(record["global_loss_delta"]) for record in records) / len(records)
            if records else None
        ),
    }


class FrozenRuntimeState:
    """Replay mutable buffers/FP8 extra state so paired forwards share a start."""

    def __init__(self, model: Metis16ForCausalLM, *, exclude_prefixes: tuple[str, ...] = ()):
        self.model = model
        _select_probe_device(next(model.parameters()).device)
        self.exclude_prefixes = exclude_prefixes
        self.buffers = {
            name: value.detach().clone() for name, value in model.named_buffers()
            if not name.startswith(exclude_prefixes)
        }
        self.extra = {}
        for name, module in model.named_modules():
            if (name + ".").startswith(exclude_prefixes):
                continue
            if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
                self.extra[name] = copy.deepcopy(module.get_extra_state())

    @torch.no_grad()
    def restore(self) -> None:
        _select_probe_device(next(self.model.parameters()).device)
        for name, value in self.model.named_buffers():
            if name.startswith(self.exclude_prefixes):
                continue
            if name not in self.buffers or value.shape != self.buffers[name].shape:
                raise RuntimeError("Mutable buffer inventory changed during frozen evaluation")
            value.copy_(self.buffers[name])
        modules = dict(self.model.named_modules())
        for name, value in self.extra.items():
            modules[name].set_extra_state(copy.deepcopy(value))


@contextmanager
def frozen_model(
    model: Metis16ForCausalLM, *, continuation_grad: bool = False, utility_grad: bool = False,
) -> Iterator[None]:
    if continuation_grad and utility_grad:
        raise ValueError("Continuation diagnostics and utility fitting require separate gradient scopes")
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    replay = model.activation_recompute_policy
    try:
        model.eval()
        model.activation_recompute_policy = "none"
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        if continuation_grad:
            for parameter in model.continuation.parameters():
                parameter.requires_grad_(True)
        if utility_grad:
            router = getattr(model, "joint_router", None)
            if router is None:
                raise CapabilityError("Utility fitting requires an enabled joint_router")
            for parameter in router.parameters():
                parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        model.activation_recompute_policy = replay


def probe_precision_context(model: Metis16ForCausalLM):
    if getattr(model, "_routing_probe_bf16_reference", False):
        factory = getattr(model.precision_policy, "bf16_reference_context", None)
        if not callable(factory):
            raise CapabilityError("The original checkpoint backend cannot provide a BF16 reference context")
        return factory()
    return nullcontext()


@dataclass
class ProbeForward:
    output: Any
    widths: Tensor
    elapsed_seconds: float
    continuation_gradients: Tensor | None
    continuation_observed: Tensor | None
    cost: dict[str, Any]


def evaluate_in_memory(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    seed: int, runtime_state: FrozenRuntimeState | None = None,
    continuation_grad: bool = False, force_depth: Tensor | None = None,
    force_routed_k: Tensor | None = None, return_router_observations: bool = False,
) -> ProbeForward:
    """Evaluate without updating weights; callers own returned in-memory data."""
    if force_routed_k is not None or return_router_observations:
        signature = inspect.signature(model.forward).parameters
        for required, requested in (
            ("force_routed_k", force_routed_k is not None),
            ("return_router_observations", return_router_observations),
        ):
            if requested and required not in signature:
                raise CapabilityError(f"Model.forward does not implement {required}")
    device = _select_probe_device(batch.input_ids.device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if runtime_state is not None:
        runtime_state.restore()
    extras = {}
    if force_routed_k is not None:
        extras["force_routed_k"] = force_routed_k
    if return_router_observations:
        extras["return_router_observations"] = True
    gradients = observed = None
    with frozen_model(model, continuation_grad=continuation_grad), torch.random.fork_rng(devices=cuda_devices), probe_precision_context(model):
        torch.manual_seed(seed)
        synchronize()
        started = time.perf_counter()
        with torch.set_grad_enabled(continuation_grad), RoutingCapture(model, continuation=continuation_grad) as capture:
            output = model(
                batch.input_ids, batch.labels, canonical_ids=batch.canonical_ids,
                attention_mask=batch.attention_mask, document_ids=batch.document_ids,
                reset_mask=batch.reset_mask, curriculum=curriculum,
                force_depth=force_depth, return_logits=False, **extras,
            )
        widths = capture.full_widths(output.active_masks)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise RuntimeError("Probe produced no finite supervised LM loss")
        if continuation_grad:
            differentiable = [p for p in capture.probabilities if p.requires_grad]
            if not differentiable or not output.loss.requires_grad:
                raise CapabilityError("No differentiable continuation-to-LM-loss path")
            derivatives = torch.autograd.grad(output.loss, differentiable, allow_unused=True)
            by_id = dict(zip(map(id, differentiable), derivatives))
            gradients = torch.zeros_like(output.active_masks, dtype=torch.float32)
            observed = torch.zeros_like(output.active_masks)
            if len(capture.probabilities) > output.active_masks.shape[0]:
                raise RuntimeError("Continuation hooks exceed executed pass count")
            for p, probability in enumerate(capture.probabilities):
                derivative = by_id.get(id(probability))
                if derivative is None:
                    continue
                gradients[p] = unpack_active(derivative.detach(), output.active_masks[p])
                observed[p] = output.active_masks[p]
            if not bool(torch.isfinite(gradients).all()):
                raise RuntimeError("Continuation credit contains nonfinite gradients")
        synchronize()
        elapsed = time.perf_counter() - started
    reconstructed = output.active_masks.sum(dim=0)
    if not torch.equal(reconstructed, output.chosen_depths):
        raise RuntimeError("Chosen depth does not match the actual active-pass history")
    cost = plan_cost(
        model.config, output.chosen_depths, widths,
        terminal_only=_terminal_only_cost(model.config, curriculum, return_router_observations),
    )
    if getattr(model.config, "causal_compute_budget", False):
        if "joint_model_flops" not in output.telemetry or "joint_router_flops" not in output.telemetry:
            raise CapabilityError("Causal evaluation requires the model's actual compute ledger")
        if cost["nominal_train_flops"] + int(output.telemetry["joint_router_flops"]) != int(output.telemetry["joint_model_flops"]):
            raise RuntimeError("Reconstructed causal depth/width/head cost differs from the model ledger")
    return ProbeForward(output, widths, elapsed, gradients, observed, cost)


def forward_summary(result: ProbeForward, batch: TrainingBatch) -> dict[str, Any]:
    """Snapshot scalar counters after evaluation and any requested backward."""
    depths = result.output.chosen_depths[batch.attention_mask]
    routed = result.widths[result.widths > 0]
    total_flops = int(result.output.telemetry.get("joint_model_flops", result.cost["nominal_train_flops"]))
    summary = {
        "lm_loss": float(result.output.loss.detach()),
        "supervised_tokens": int((batch.labels != -100).sum()),
        "non_padding_tokens": int(batch.attention_mask.sum()),
        "chosen_depth_histogram": histogram(depths),
        "chosen_k_histogram_over_active_layer_pass_tokens": histogram(routed),
        "mean_chosen_depth": float(depths.float().mean()),
        "mean_chosen_k_over_active_layer_pass_tokens": float(routed.float().mean()),
        "elapsed_seconds_including_hooks_and_gradient": result.elapsed_seconds,
        "cost": result.cost,
        "modeled_total_train_flops": total_flops,
        "nominal_probe_forward_flops": total_flops / 3.0,
    }
    head_fields = (
        "lm_head_forward_rows", "lm_head_forward_flops",
        "lm_head_recompute_rows", "lm_head_recompute_flops",
    )
    head_work = {
        name: int(result.output.telemetry[name])
        for name in head_fields if name in result.output.telemetry
    }
    summary.update(head_work)
    if head_work:
        summary["lm_head_work_basis"] = "model_reported_logical_projection_work_not_backend_padding"
    counter = (
        "lm_head_forward_rows" if "lm_head_forward_rows" in head_work else
        "executed_lm_head_tokens" if "executed_lm_head_tokens" in result.output.telemetry else None
    )
    summary.update({
        "lm_head_tokens": (
            int(result.output.telemetry[counter]) if counter is not None else
            result.cost.get("modeled_lm_head_tokens", result.cost["token_passes"])
        ),
        "lm_head_tokens_basis": "model_execution_counter" if counter is not None else "nominal_nonpadding_fallback",
        "lm_head_tokens_counter": counter,
    })
    return summary


def repeated_plan_evaluation(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    seed: int, runtime_state: FrozenRuntimeState, repeat_forwards: int,
    minimum_loss_delta: float, force_depth: Tensor | None = None,
    force_routed_k: Tensor | None = None, return_router_observations: bool = False,
) -> tuple[ProbeForward, dict[str, Any]]:
    """Distinguish natural plan variability from noise of an enforced plan."""
    if repeat_forwards < 2:
        raise ValueError("Repeated evaluation requires at least two forwards")
    if (force_depth is None) != (force_routed_k is None):
        raise ValueError("Fixed-plan replay requires both depths and complete width histories")
    first = None
    records = []
    elapsed = nominal_forward = 0.0
    expected_cost = (
        plan_cost(
            model.config, force_depth, force_routed_k,
            terminal_only=_terminal_only_cost(model.config, curriculum, return_router_observations),
        )
        if force_depth is not None else None
    )
    for _ in range(repeat_forwards):
        current = evaluate_in_memory(
            model, batch, curriculum, seed=seed, runtime_state=runtime_state,
            force_depth=force_depth, force_routed_k=force_routed_k,
            return_router_observations=return_router_observations,
        )
        if first is None:
            first = current
        if expected_cost is not None and (
            current.cost != expected_cost
            or not torch.equal(current.output.chosen_depths, force_depth)
            or not torch.equal(current.widths, force_routed_k)
        ):
            raise RuntimeError("Fixed-plan replay changed the requested depths, widths, or realized cost")
        loss = float(current.output.loss.detach())
        elapsed += current.elapsed_seconds
        digest = hashlib.sha256()
        for tensor in (current.output.chosen_depths, current.widths):
            values = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([str(values.dtype), list(values.shape)]).encode())
            digest.update(values.numpy().tobytes())
        records.append({
            "plan_sha256": digest.hexdigest(),
            "lm_loss": loss, "cost": current.cost,
            "modeled_total_train_flops": int(current.output.telemetry.get(
                "joint_model_flops", current.cost["nominal_train_flops"],
            )),
            "chosen_depth_histogram": histogram(current.output.chosen_depths[batch.attention_mask]),
        })
        forward = current.cost["nominal_forward_flops"]
        if "joint_router_flops" in current.output.telemetry:
            forward += float(current.output.telemetry["joint_router_flops"]) / 3.0
        nominal_forward += forward
    assert first is not None
    return first, {
        **plan_repeat_statistics(records, minimum_loss_delta=minimum_loss_delta),
        "repeat_kind": "fixed_depth_and_width_plan" if force_depth is not None else "natural_policy",
        "repeats": records, "forward_calls": repeat_forwards,
        "total_evaluation_seconds": elapsed,
        "nominal_forward_flops_all_calls": nominal_forward,
        "interpretation": (
            "Natural repeats include depth/width allocation variability, not pure numerical noise."
            if force_depth is None else
            "Depths and all layer/pass widths are forced and verified. Residual variation includes "
            "expert-identity ties and kernel numerics; expert identities themselves are not frozen."
        ),
    }


def depth_credit_probe(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    pairs: int, seed: int, runtime_state: FrozenRuntimeState | None = None,
    repeat_forwards: int = 3, minimum_loss_delta: float = 1e-5,
) -> dict[str, Any]:
    if repeat_forwards < 2:
        raise ValueError("Depth-credit noise estimation requires at least two repeats")
    if not model.config.min_routed_k <= 4 <= model.config.max_routed_k:
        raise CapabilityError("Depth-credit isolation requires legal fixed k=4")
    fixed = replace(curriculum, routed_k_mode="fixed", fixed_routed_k=4, stochastic_routing=False)
    reference = evaluate_in_memory(
        model, batch, fixed, seed=seed, runtime_state=runtime_state, continuation_grad=True,
    )
    depths = reference.output.chosen_depths.detach()
    candidates = select_depth_pairs(
        depths, batch.attention_mask, pairs=pairs, seed=seed, document_ids=batch.document_ids,
    )
    replay = evaluate_in_memory(
        model, batch, fixed, seed=seed, runtime_state=runtime_state, force_depth=depths,
    )
    if reference.cost != replay.cost:
        raise RuntimeError("Forced original depths did not preserve the fixed-k cost")
    replay_loss = float(replay.output.loss.detach())
    original_loss = float(reference.output.loss.detach())
    repeats = [replay_loss]
    elapsed = reference.elapsed_seconds + replay.elapsed_seconds
    forward_flops = reference.cost["nominal_forward_flops"] + replay.cost["nominal_forward_flops"]
    for _ in range(repeat_forwards - 1):
        repeated = evaluate_in_memory(
            model, batch, fixed, seed=seed, runtime_state=runtime_state, force_depth=depths,
        )
        if repeated.cost != replay.cost or not torch.equal(repeated.widths, replay.widths):
            raise RuntimeError("Identical fixed-depth replay changed its executed width plan")
        repeats.append(float(repeated.output.loss.detach()))
        elapsed += repeated.elapsed_seconds
        forward_flops += repeated.cost["nominal_forward_flops"]
    noise = repeat_noise(repeats, minimum_delta=minimum_loss_delta)
    threshold = noise["decisive_absolute_loss_delta_threshold"]
    replay_loss = sum(repeats) / len(repeats)
    reference_parity = abs(original_loss - replay_loss) <= threshold
    gradients = reference.continuation_gradients
    observed = reference.continuation_observed
    assert gradients is not None and observed is not None
    records = []
    skipped = 0
    for pair in candidates:
        indices = [pair.shallow, pair.deep]
        if not bool(observed[pair.boundary].reshape(-1)[indices].all()):
            skipped += 1
            continue
        selected = gradients[pair.boundary].reshape(-1)[indices]
        predicted = float(selected[0] - selected[1])
        swapped, swapped_widths = swap_plan(depths, reference.widths, pair)
        expected_cost = plan_cost(model.config, swapped, swapped_widths)
        if expected_cost != reference.cost:
            raise RuntimeError("Depth exchange failed the exact per-pass/layer budget invariant")
        result = evaluate_in_memory(
            model, batch, fixed, seed=seed, runtime_state=runtime_state, force_depth=swapped,
        )
        if result.cost != expected_cost or not torch.equal(result.output.chosen_depths, swapped):
            raise RuntimeError("Forced depth exchange did not execute the requested plan")
        delta = float(result.output.loss.detach()) - replay_loss
        elapsed += result.elapsed_seconds
        forward_flops += result.cost["nominal_forward_flops"]
        records.append({
            "shallow_depth": pair.shallow_depth, "deep_depth": pair.deep_depth,
            "continuation_pass_one_based": pair.boundary + 1,
            "predicted_first_boundary_loss_delta": predicted,
            "predicted_score_change_toward_swap": -predicted,
            "global_loss_delta": delta,
            "alignment": (
                "reference_replay_mismatch" if not reference_parity
                else "below_numerical_noise" if abs(delta) <= threshold
                else credit_alignment(predicted, delta)
            ),
            "gradient_interpretation_valid": reference_parity,
            "above_numerical_noise_threshold": abs(delta) > threshold,
            "nominal_train_flops_delta": result.cost["nominal_train_flops"] - replay.cost["nominal_train_flops"],
            "same_depth_multiset": True,
            "same_pass_layer_expert_assignments": True,
        })
    summary = forward_summary(reference, batch)
    return {
        "fixed_routed_k": 4, "reference": summary,
        "forced_original_lm_loss": replay_loss,
        "forced_original_loss_delta": replay_loss - original_loss,
        "gradient_interpretation_valid": reference_parity,
        "reference_parity_failure": (
            None if reference_parity else
            "Natural gradient-forward loss differs from forced reference beyond fixed-plan noise; "
            "gradient alignment is invalid and excluded, not silently accepted."
        ),
        "deterministic_repeat_noise": noise,
        "requested_pairs": pairs, "available_disjoint_pairs": len(candidates),
        "unobserved_credit_pairs_skipped": skipped,
        "aggregate": aggregate_pairs(records), "paired_swaps": records,
        "total_evaluation_seconds": elapsed,
        "forward_calls": 1 + repeat_forwards + len(records), "autograd_calls": 1,
        "nominal_forward_flops_all_calls": forward_flops,
        "gradient_work_nominal_full_backward_upper_bound": 2 * reference.cost["nominal_forward_flops"],
        "interpretation": (
            "First-divergence continuation-probability credit versus global-batch LM loss "
            "under a complete depth exchange. For depth gaps >1 this is not a full "
            "trajectory derivative. Ties are excluded from alignment; pairs share one "
            "batch and are not independent convergence evidence."
        ),
    }


def identify_batch(batch: TrainingBatch) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name in ("input_ids", "canonical_ids", "labels", "attention_mask", "document_ids", "reset_mask"):
        tensor = getattr(batch, name).detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return {
        "tensor_window_sha256": digest.hexdigest(), "phase": batch.phase,
        "global_token_cursor": batch.global_token_cursor,
        "next_global_token_cursor": batch.next_global_token_cursor,
        "sequences": batch.input_ids.shape[0], "sequence_length": batch.input_ids.shape[1],
        "non_padding_tokens": batch.non_padding_tokens, "supervised_tokens": batch.supervised_tokens,
    }


def held_out_batch(
    inventory: ReleaseInventory, identity: Mapping[str, Any], *,
    checkpoint_step: int, step: int, sequences: int, sequence_length: int,
    gap_blocks: int = 0,
) -> tuple[TrainingBatch, dict[str, Any]]:
    if type(gap_blocks) is not int or gap_blocks < 0:
        raise ValueError("Probe gap_blocks must be a nonnegative integer")
    if step < checkpoint_step and gap_blocks == 0:
        raise ValueError("Probe sampler step precedes the checkpoint's unexecuted boundary")
    if sequences <= 0 or sequence_length <= 0:
        raise ValueError("Sequences and sequence length must be positive")
    expected_release = identity.get("release", {})
    for key in ("release_sha256", "shard_manifest_sha256"):
        if not getattr(inventory, key) or getattr(inventory, key) != expected_release.get(key):
            raise ValueError(f"Release identity mismatch: {key}")
    sampler_spec = identity["sampler"]
    block = int(sampler_spec["block_tokens"])
    if sequences * sequence_length > block:
        raise ValueError("Probe window crosses its frozen sampler block")
    stream = DeterministicReleaseStream(
        inventory, sequence_length=sequence_length, shard_order_seed=SHARD_ORDER_SEED,
        expected_vocabulary_size=int(identity["model"]["vocab_size"]),
    )
    sampler: AblationSampleStream = build_sample_stream(
        stream, budget_tokens=int(sampler_spec["budget_tokens"]), block_tokens=block,
    )
    if sampler.describe() != sampler_spec:
        raise ValueError("Current sampler arithmetic differs from the frozen run manifest")
    cursor = (
        sampler.evaluation_cursor(
            step, gap_blocks=gap_blocks, window_tokens=sequences * sequence_length
        )
        if gap_blocks
        else sampler.release_cursor(step)
    )
    last_consumed_end = None
    if checkpoint_step:
        last_consumed_end = sampler.release_cursor(checkpoint_step - 1) + block
        if gap_blocks == 0 and cursor <= last_consumed_end:
            raise ValueError("Probe overlaps a trained block or its target lookahead; choose a later --step")
    batch = stream.batch(
        global_token_cursor=cursor, rank=0, world_size=1, micro_batch_size=sequences,
    )
    if batch.supervised_tokens <= 0:
        raise RuntimeError("Held-out window has no supervised tokens")
    return batch, {
        **identify_batch(batch), "sampler_step": step, "checkpoint_next_unexecuted_step": checkpoint_step,
        "block_tokens": block, "sampled_fraction_of_block": sequences * sequence_length / block,
        "gap_blocks": gap_blocks,
        "disjoint_from_entire_declared_training_sampler": bool(gap_blocks),
        "last_consumed_block_end_including_target_lookahead": last_consumed_end,
        "shard_order_seed": SHARD_ORDER_SEED, "sampler_sha256": json_sha256(sampler_spec),
        "held_out_scope": (
            "Token-disjoint from every block and target in the declared training sampler; "
            "not a deduplicated document holdout. Bound to this sampler identity and budget."
            if gap_blocks else
            "Not consumed by this checkpoint's deterministic sampler; not a deduplicated "
            "document holdout. A small future-block prefix, not representative full-release validation."
        ),
    }


def load_utility_artifact(
    model: Metis16ForCausalLM, path: Path, *, checkpoint_sha256: str,
    run_identity_sha256: str, base_model_config_sha256: str,
    evaluation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    router = getattr(model, "joint_router", None)
    if router is None:
        raise CapabilityError("The model has no enabled joint utility router")
    identity = identify_file(path)
    payload = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != UTILITY_SCHEMA:
        raise CapabilityError(f"Utility artifact must use schema {UTILITY_SCHEMA}")
    for key, value in (
        ("source_checkpoint_sha256", checkpoint_sha256),
        ("run_identity_sha256", run_identity_sha256),
        ("base_model_config_sha256", base_model_config_sha256),
    ):
        if payload.get(key) != value:
            raise ValueError(f"Utility artifact {key} mismatch")
    provenance = validate_utility_provenance(
        payload, model.config, evaluation_window=evaluation_window,
    )
    expected_precision = {
        "bf16_reference_context": bool(getattr(model, "_routing_probe_bf16_reference", False)),
        "backend_profile": getattr(model.precision_policy, "requested_profile", None),
    }
    if "teacher_precision" in payload and payload["teacher_precision"] != expected_precision:
        raise ValueError("Utility artifact teacher/evaluation precision regimes differ")
    weights = payload.get("state_dict")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Utility artifact has no router-only state_dict")
    router.load_state_dict(weights, strict=True)
    updates = getattr(router, "trained_updates", None)
    if updates is None or int(updates.item()) <= 0:
        raise CapabilityError("Utility artifact does not identify trained utility weights")
    assert_file_unchanged(identity)
    return {
        **identity, "schema": UTILITY_SCHEMA, "trained_updates": int(updates.item()),
        "base_model_config_sha256": base_model_config_sha256, **provenance,
    }


def validate_utility_provenance(
    payload: Mapping[str, Any], config: Metis16Config, *,
    evaluation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def digest(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    output: dict[str, Any] = {}
    for key in ("joint_router_hidden_dim", "max_passes", "n_layers", "max_routed_k"):
        if type(payload.get(key)) is not int or payload[key] != getattr(config, key, None):
            raise ValueError(f"Utility artifact geometry mismatch: {key}")
        output[key] = payload[key]
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or len(revision) not in (40, 64) or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Utility artifact must identify its committed training source_revision")
    output["source_revision"] = revision
    for key, minimum in (
        ("training_seed", 0), ("teacher_token_count", 1),
        ("teacher_forward_calls", 1), ("teacher_backward_calls", 0),
    ):
        value = payload.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Utility artifact requires valid {key}")
        output[key] = value
    elapsed = payload.get("teacher_elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Utility artifact requires finite teacher_elapsed_seconds")
    output["teacher_elapsed_seconds"] = elapsed
    windows = payload.get("training_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Utility artifact requires identified training_windows")
    sanitized = []
    for window in windows:
        if not isinstance(window, Mapping) or not digest(window.get("tensor_window_sha256")):
            raise ValueError("Utility training window requires tensor_window_sha256")
        start, end = window.get("global_token_cursor"), window.get("next_global_token_cursor")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("Utility training window requires a valid token interval")
        if evaluation_window is not None:
            if (
                window["tensor_window_sha256"] == evaluation_window["tensor_window_sha256"]
                or (
                    start <= evaluation_window["next_global_token_cursor"]
                    and evaluation_window["global_token_cursor"] <= end
                )
            ):
                raise ValueError("Utility training and evaluation windows overlap, including target lookahead")
        sanitized.append({
            "tensor_window_sha256": window["tensor_window_sha256"],
            "global_token_cursor": start, "next_global_token_cursor": end,
        })
    output["training_windows"] = sanitized
    if "fit_max_depth" in payload:
        if type(payload["fit_max_depth"]) is not int or not 2 <= payload["fit_max_depth"] <= config.max_passes:
            raise ValueError("Utility artifact fit_max_depth is outside its architecture")
        output["fit_max_depth"] = payload["fit_max_depth"]
    for name in (
        "fit_steps", "fit_learning_rate", "visited_depth_histogram", "visited_width_histogram",
        "nominal_teacher_forward_flops", "nominal_teacher_utility_backward_flops",
        "teacher_precision", "teacher_curriculum", "head_initialization_seed",
        "warmup_forward_calls", "warmup_token_count", "warmup_elapsed_seconds",
        "nominal_warmup_forward_flops", "warmup_window",
    ):
        if name in payload:
            output[name] = payload[name]
    output["target_interpretation"] = (
        "Observed per-exit CE improvement, not an unbiased global counterfactual utility. "
        "Unknown continuation outcomes must remain unlabeled."
    )
    return output


def load_frozen_model(
    config: Metis16Config, weights: Mapping[str, Any], *,
    device: torch.device, precision: str, enable_joint: bool,
    checkpoint_precision: str | None = None,
    initialization_seed: int = 16062026,
) -> tuple[Metis16ForCausalLM, dict[str, Any]]:
    if config.expert_parallel_size != 1 or config.context_parallel_size != 1:
        raise CapabilityError("Probe requires replicated experts and unsharded sequences")
    if enable_joint and "joint_compute_router" not in config.__dataclass_fields__:
        raise CapabilityError("Config does not support joint_compute_router")
    changes: dict[str, Any] = {}
    if enable_joint:
        changes["joint_compute_router"] = True
    # Evaluation does not execute activation replay; no optimizer or distributed
    # group is constructed, and the stored architecture remains unchanged.
    effective = replace(config, **changes)
    checkpoint_precision = checkpoint_precision or precision
    if precision != checkpoint_precision and precision != "bf16":
        raise CapabilityError("Changing checkpoint backend layout is not supported; use its precision or BF16 reference")
    device = _select_probe_device(device)
    policy = build_precision_policy(
        effective.precision, profile=checkpoint_precision, device=device,
        production=False, permit_fallback=False,
    )
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        model = Metis16ForCausalLM(effective, precision_policy=policy)
    model.apply_parameter_storage_policy(device=device)
    target_keys = set(model.state_dict())
    new_joint = enable_joint and not getattr(config, "joint_compute_router", False)
    if new_joint and any(key.startswith("joint_router.") for key in weights):
        raise ValueError("Checkpoint utility weights are not declared by the original model config")
    loaded = model.load_state_dict(weights, strict=not enable_joint)
    allowed_missing = {
        key for key in target_keys if new_joint and key.startswith("joint_router.")
    }
    if set(loaded.missing_keys) - allowed_missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint weight mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}"
        )
    for name, parameter in model.named_parameters():
        expected = torch.float32 if getattr(parameter, "metis_storage_dtype", None) == "float32" else torch.bfloat16
        if parameter.dtype != expected:
            raise RuntimeError(f"Parameter storage policy mismatch: {name}")
    if precision == "bf16" and not callable(getattr(policy, "bf16_reference_context", None)):
        raise CapabilityError("Checkpoint precision policy lacks its BF16 reference context")
    model._routing_probe_bf16_reference = precision == "bf16"
    model._routing_probe_initialization_seed = initialization_seed
    return model.eval(), {
        "backend_precision_audit": policy.audit.to_dict(),
        "effective_profile": precision,
        "constructed_backend_profile": checkpoint_precision,
        "bf16_reference_context": precision == "bf16",
        "discarded_checkpoint_state_keys": [],
        "new_joint_state_keys": sorted(set(loaded.missing_keys)),
        "initialization_seed": initialization_seed,
        "model_config": effective.to_dict(),
        "optimizer_loaded": False, "optimizer_shards_loaded": 0,
        "original_world_size": config.world_size, "probe_world_size": 1,
        "activation_recompute_executed": False,
    }


def joint_policy_probe(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    seed: int, runtime_state: FrozenRuntimeState, legacy: ProbeForward,
    repeat_forwards: int = 3, minimum_loss_delta: float = 1e-5,
    max_passes: int | None = None, trained_max_depth: int | None = None,
    legacy_noise: Mapping[str, Any] | None = None,
    fixed_reference: ProbeForward | None = None,
    fixed_noise: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if "compute_allocation_mode" not in curriculum.__dataclass_fields__:
        raise CapabilityError("CurriculumState does not support joint allocation")
    joint = replace(
        curriculum, compute_allocation_mode="joint", stochastic_routing=False,
        joint_router_exploration=0.0, allow_untrained_joint_router=False,
        max_passes=max_passes,
    )
    if repeat_forwards < 2:
        raise ValueError("Joint comparisons require at least two numerical repeats")
    elapsed = nominal_forward = 0.0
    forward_calls = 0

    def repeated_policy(state, *, depths=None, widths=None):
        nonlocal elapsed, nominal_forward, forward_calls
        result, metadata = repeated_plan_evaluation(
            model, batch, state, seed=seed, runtime_state=runtime_state,
            repeat_forwards=repeat_forwards, minimum_loss_delta=minimum_loss_delta,
            force_depth=depths, force_routed_k=widths,
            return_router_observations=depths is not None,
        )
        elapsed += metadata["total_evaluation_seconds"]
        nominal_forward += metadata["nominal_forward_flops_all_calls"]
        forward_calls += metadata["forward_calls"]
        return result, metadata

    result, noise = repeated_policy(joint)
    telemetry = result.output.telemetry
    required = ("joint_model_flops", "joint_router_flops", "joint_unused_budget_flops", "joint_budget_flops")
    if any(key not in telemetry for key in required):
        raise CapabilityError("Joint model must expose joint_model_flops and joint_router_flops")
    delta = result.cost["nominal_train_flops"] - legacy.cost["nominal_train_flops"]
    depths = result.output.chosen_depths.detach()
    shuffled_depths, shuffled_widths = shuffle_plan(
        depths, result.widths, batch.attention_mask, seed=seed + 1,
        document_ids=batch.document_ids, supervised_mask=batch.labels.ne(-100),
    )
    if plan_cost(model.config, shuffled_depths, shuffled_widths) != result.cost:
        raise RuntimeError("Whole-trajectory shuffle failed exact per-pass/layer cost preservation")
    replay_curriculum = replace(
        joint, compute_allocation_mode="legacy", routed_k_mode="fixed", fixed_routed_k=4,
    )
    replay, replay_noise = repeated_policy(replay_curriculum, depths=depths, widths=result.widths)
    shuffled, shuffled_noise = repeated_policy(
        replay_curriculum, depths=shuffled_depths, widths=shuffled_widths,
    )
    for execution, requested_depths, requested_widths in (
        (replay, depths, result.widths), (shuffled, shuffled_depths, shuffled_widths),
    ):
        if (
            execution.cost != result.cost
            or not torch.equal(execution.output.chosen_depths, requested_depths)
            or not torch.equal(execution.widths, requested_widths)
            or int(execution.output.telemetry["joint_model_flops"]) != int(telemetry["joint_model_flops"])
        ):
            raise RuntimeError("Explicit joint-plan replay changed the requested realized cost or trajectory")
    threshold = max(
        item["decisive_absolute_loss_delta_threshold"]
        for item in (noise, replay_noise, shuffled_noise, legacy_noise or noise, fixed_noise or noise)
    )
    joint_loss = float(result.output.loss.detach())
    replay_loss = replay_noise["mean_lm_loss"]
    shuffled_loss = shuffled_noise["mean_lm_loss"]
    replay_matches = abs(joint_loss - replay_loss) <= replay_noise["decisive_absolute_loss_delta_threshold"]
    changed = not (torch.equal(depths, shuffled_depths) and torch.equal(result.widths, shuffled_widths))
    evaluation_cap = max_passes or model.config.max_passes
    trained_horizon = trained_max_depth is not None and evaluation_cap <= trained_max_depth
    legacy_minimum = legacy_noise["min_lm_loss"] if legacy_noise and "min_lm_loss" in legacy_noise else float(legacy.output.loss.detach())
    beats_legacy = noise["max_lm_loss"] < legacy_minimum - threshold
    beats_shuffle = changed and replay_matches and noise["max_lm_loss"] < shuffled_noise["min_lm_loss"] - threshold
    legacy_minimum_cost = (
        min(record["modeled_total_train_flops"] for record in legacy_noise["repeats"])
        if legacy_noise and "repeats" in legacy_noise else legacy.cost["nominal_train_flops"]
    )
    no_more_compute = max(record["modeled_total_train_flops"] for record in noise["repeats"]) <= legacy_minimum_cost
    beats_fixed = fixed_cost_ok = None
    fixed_delta = None
    if fixed_reference is not None:
        fixed_minimum = fixed_noise["min_lm_loss"] if fixed_noise else float(fixed_reference.output.loss.detach())
        fixed_mean = fixed_noise["mean_lm_loss"] if fixed_noise else fixed_minimum
        beats_fixed = noise["max_lm_loss"] < fixed_minimum - threshold
        fixed_delta = noise["mean_lm_loss"] - fixed_mean
        fixed_cost_ok = (
            max(record["modeled_total_train_flops"] for record in noise["repeats"])
            <= fixed_reference.cost["nominal_train_flops"]
        )
    return {
        **forward_summary(result, batch),
        "global_loss_delta_from_legacy": float(result.output.loss.detach() - legacy.output.loss.detach()),
        "mean_natural_policy_lm_loss": noise["mean_lm_loss"],
        "nominal_backbone_train_flops_delta_from_legacy": delta,
        "exact_equal_backbone_budget": delta == 0,
        "joint_model_flops": float(telemetry["joint_model_flops"]),
        "joint_router_flops": float(telemetry["joint_router_flops"]),
        "joint_budget_flops": int(telemetry["joint_budget_flops"]),
        "joint_unused_budget_flops": int(telemetry["joint_unused_budget_flops"]),
        "max_passes": evaluation_cap,
        "trained_max_depth": trained_max_depth,
        "within_recorded_training_horizon": trained_horizon,
        "natural_policy_variability": noise,
        "deterministic_repeat_noise": replay_noise,
        "comparison_absolute_loss_delta_threshold": threshold,
        "comparison_threshold_includes_natural_policy_variability": True,
        "forced_original": {
            **forward_summary(replay, batch), "deterministic_repeat_noise": replay_noise,
            "mean_lm_loss": replay_loss,
            "global_loss_delta_from_joint": replay_loss - joint_loss,
            "matches_joint_within_numerical_noise": replay_matches,
        },
        "shuffled": {
            **forward_summary(shuffled, batch), "deterministic_repeat_noise": shuffled_noise,
            "mean_lm_loss": shuffled_loss,
            "global_loss_delta_from_joint": shuffled_loss - joint_loss,
            "global_loss_delta_from_forced_original": shuffled_loss - replay_loss,
            "different_plan": changed, "same_realized_plan_cost": True,
            "shuffle_seed": seed + 1, "shuffle_strata": "sequence/document/supervision status",
        },
        "beats_legacy_above_numerical_noise": beats_legacy,
        "beats_shuffled_above_numerical_noise": beats_shuffle,
        "uniform_depth2_k4_control_available": fixed_reference is not None,
        "mean_global_loss_delta_from_uniform_depth2_k4": fixed_delta,
        "beats_uniform_depth2_k4_above_numerical_noise": beats_fixed,
        "no_more_modeled_compute_than_uniform_depth2_k4": fixed_cost_ok,
        "no_more_modeled_compute_than_legacy": no_more_compute,
        "one_window_quality_checks_passed": bool(
            beats_legacy and beats_shuffle and beats_fixed and fixed_cost_ok
            and trained_horizon and no_more_compute
        ),
        "policy_quality_established": False,
        "forward_calls": forward_calls, "total_evaluation_seconds": elapsed,
        "nominal_forward_flops_all_calls": nominal_forward,
        "interpretation": (
            "One held-out window is not proof of policy quality or convergence. "
            "Joint-versus-shuffle evidence additionally requires forced-original replay parity; "
            "a parity failure can expose continuation-confidence differences outside the explicit plan. "
            "All comparisons preserve the recorded precision. Extra observation LM-head work is "
            "included in probe execution estimates, not silently added to the model's budget ledger."
        ),
    }


def assert_disjoint_windows(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if (
        left["tensor_window_sha256"] == right["tensor_window_sha256"]
        or (
            left["global_token_cursor"] <= right["next_global_token_cursor"]
            and right["global_token_cursor"] <= left["next_global_token_cursor"]
        )
    ):
        raise ValueError("Training/evaluation windows overlap, including target lookahead")


def teacher_plan(
    config: Metis16Config, attention_mask: Tensor, *, max_depth: int, generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if not 2 <= max_depth <= config.max_passes:
        raise ValueError("Teacher max_depth must lie in [2, model.max_passes]")
    if config.min_routed_k != 1 or not config.max_routed_k >= 4:
        raise CapabilityError("Teacher exploration requires widths 1..K and first-pass k=4")
    shape = attention_mask.shape
    depths = torch.randint(2, max_depth + 1, shape, generator=generator)
    widths = torch.randint(
        1, config.max_routed_k + 1,
        (config.max_passes, config.n_layers, *shape), generator=generator,
    )
    widths[0].fill_(4)
    depths = depths.to(attention_mask.device) * attention_mask
    widths = widths.to(attention_mask.device)
    active = torch.arange(config.max_passes, device=depths.device)[:, None, None] < depths
    widths *= active[:, None]
    return depths, widths


def fit_utility_in_memory(
    model: Metis16ForCausalLM, curriculum: CurriculumState,
    batches: Iterable[tuple[TrainingBatch, Mapping[str, Any]]], *,
    steps: int, seed: int, learning_rate: float, max_depth: int,
    evaluation_window: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit only observed-outcome utility, returning aggregate provenance."""
    if steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Utility fitting requires positive steps and learning rate")
    router = getattr(model, "joint_router", None)
    if router is None or not callable(getattr(router, "mark_trained", None)):
        raise CapabilityError("Utility fitting requires joint_router.mark_trained")
    if any(name not in inspect.signature(model.forward).parameters for name in (
        "force_routed_k", "return_router_observations",
    )):
        raise CapabilityError("Utility teacher mode requires explicit widths and router observations")
    if not 2 <= max_depth <= model.config.max_passes:
        raise ValueError("Fit depth exceeds the model's stored architecture")
    teacher = replace(
        curriculum, compute_allocation_mode="legacy", routed_k_mode="fixed",
        fixed_routed_k=4, stochastic_routing=False, max_passes=max_depth,
        joint_utility_coefficient=1.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = list(router.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    # Never replay the head's trained_updates buffer from its pre-fit snapshot.
    body_runtime = FrozenRuntimeState(model, exclude_prefixes=("joint_router.",))
    body = {name: parameter for name, parameter in model.named_parameters() if not name.startswith("joint_router.")}
    if any(parameter.grad is not None for parameter in body.values()):
        raise ValueError("Frozen body must not carry stale gradients into utility fitting")
    versions = {name: parameter._version for name, parameter in body.items()}
    updates_before = int(router.trained_updates.item())
    if updates_before != 0:
        raise ValueError("Bounded fitting requires a fresh utility head, not silent artifact continuation")
    windows: list[dict[str, Any]] = []
    records = []
    token_count = observed_count = 0
    depth_hist: Counter = Counter()
    width_hist: Counter = Counter()
    nominal_forward = nominal_router_backward = 0.0
    started = time.perf_counter()
    iterator = iter(batches)
    with frozen_model(model, utility_grad=True):
        for index in range(steps):
            try:
                batch, supplied_identity = next(iterator)
            except StopIteration as exc:
                raise ValueError("Fewer distinct teacher batches than requested fit steps") from exc
            actual_identity = identify_batch(batch)
            for name, value in actual_identity.items():
                if supplied_identity.get(name) != value:
                    raise ValueError(f"Teacher window identity mismatch: {name}")
            assert_disjoint_windows(actual_identity, evaluation_window)
            for previous in windows:
                assert_disjoint_windows(actual_identity, previous)
            windows.append(actual_identity)
            device = _select_probe_device(next(model.parameters()).device)
            batch = batch.to(device)
            depths, widths = teacher_plan(
                model.config, batch.attention_mask, max_depth=max_depth, generator=generator,
            )
            expected_cost = plan_cost(model.config, depths, widths)
            body_runtime.restore()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            call_started = time.perf_counter()
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.enable_grad(), probe_precision_context(model), RoutingCapture(model) as capture:
                torch.manual_seed(seed + index)
                output = model(
                    batch.input_ids, batch.labels, canonical_ids=batch.canonical_ids,
                    attention_mask=batch.attention_mask, document_ids=batch.document_ids,
                    reset_mask=batch.reset_mask, curriculum=teacher,
                    force_depth=depths, force_routed_k=widths,
                    return_router_observations=True, return_logits=False,
                )
                executed = capture.full_widths(output.active_masks)
                if not torch.equal(output.chosen_depths, depths) or not torch.equal(executed, widths):
                    raise RuntimeError("Utility teacher did not execute its explicit trajectory")
                loss = output.auxiliary_losses.get("joint_utility")
                count = int(output.telemetry.get("joint_utility_observations", 0))
                if count <= 0 or not output.router_observations:
                    raise RuntimeError("No actually visited utility observations; refusing an optimizer update")
                if loss is None or not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError("Utility teacher produced no finite differentiable observed-outcome loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True,
                )
                if not bool(gradient_norm > 0):
                    raise RuntimeError("Utility update has no nonzero observed-outcome gradient")
                if any(parameter.grad is not None for parameter in body.values()):
                    raise RuntimeError("Utility loss leaked gradients into the frozen backbone")
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    raise RuntimeError("Utility optimizer produced nonfinite weights")
                router.mark_trained()
            if any(parameter._version != versions[name] for name, parameter in body.items()):
                raise RuntimeError("Utility fitting changed a frozen body parameter")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - call_started
            router_forward = float(output.telemetry["joint_router_flops"]) / 3.0
            # Observation targets reuse the per-exit head work already priced
            # in the trajectory; only the utility predictor is additional.
            forward = expected_cost["nominal_forward_flops"] + router_forward
            nominal_forward += forward
            nominal_router_backward += 2.0 * router_forward
            token_count += batch.non_padding_tokens
            observed_count += count
            depth_hist.update(histogram(depths[batch.attention_mask]))
            width_hist.update(histogram(widths[widths > 0]))
            records.append({
                "update": index + 1, "utility_mse": float(loss.detach()),
                "observed_targets": count, "gradient_norm_before_clipping": float(gradient_norm),
                "elapsed_seconds": elapsed, "nominal_forward_flops": forward,
                "nominal_utility_backward_flops": 2.0 * router_forward,
            })
            del output, loss, capture
        optimizer.zero_grad(set_to_none=True)
        body_runtime.restore()
    updates = int(router.trained_updates.item()) - updates_before
    if updates != steps:
        raise RuntimeError("Utility trained_updates does not match successful optimizer steps")
    return {
        "training_windows": windows, "training_seed": seed,
        "head_initialization_seed": getattr(model, "_routing_probe_initialization_seed", None),
        "teacher_curriculum": asdict(teacher),
        "teacher_precision": {
            "bf16_reference_context": bool(getattr(model, "_routing_probe_bf16_reference", False)),
            "backend_profile": getattr(model.precision_policy, "requested_profile", None),
        },
        "teacher_token_count": token_count, "teacher_forward_calls": steps,
        "teacher_backward_calls": steps, "teacher_elapsed_seconds": time.perf_counter() - started,
        "fit_steps": steps, "fit_learning_rate": learning_rate, "fit_max_depth": max_depth,
        "utility_optimizer": "AdamW", "utility_weight_decay": 0.0,
        "utility_gradient_clip_norm": 1.0, "observed_target_count": observed_count,
        "visited_depth_histogram": dict(sorted(depth_hist.items())),
        "visited_width_histogram": dict(sorted(width_hist.items())),
        "nominal_teacher_forward_flops": nominal_forward,
        "nominal_teacher_utility_backward_flops": nominal_router_backward,
        "teacher_updates": records,
        "target_interpretation": "Observed visited per-exit CE improvements, not unbiased global counterfactual utility",
        "unknown_continuation_targets": "unlabeled",
    }


def save_utility_artifact(
    model: Metis16ForCausalLM, path: Path, fitting: Mapping[str, Any], *,
    checkpoint_sha256: str, run_identity_sha256: str,
    base_model_config_sha256: str, source_revision: str,
) -> dict[str, Any]:
    router = getattr(model, "joint_router", None)
    if router is None or int(router.trained_updates.item()) <= 0:
        raise CapabilityError("Cannot save an untrained utility artifact")
    allowed_fitting = {
        "training_windows", "training_seed", "teacher_token_count", "teacher_forward_calls",
        "teacher_backward_calls", "teacher_elapsed_seconds", "fit_steps", "fit_learning_rate",
        "fit_max_depth", "utility_optimizer", "utility_weight_decay", "utility_gradient_clip_norm",
        "observed_target_count", "visited_depth_histogram", "visited_width_histogram",
        "nominal_teacher_forward_flops", "nominal_teacher_utility_backward_flops",
        "teacher_updates", "target_interpretation", "unknown_continuation_targets",
        "head_initialization_seed", "teacher_curriculum", "teacher_precision",
        "warmup_forward_calls", "warmup_token_count", "warmup_elapsed_seconds",
        "nominal_warmup_forward_flops", "warmup_window",
    }
    if set(fitting) - allowed_fitting:
        raise ValueError("Utility artifact accepts only whitelisted aggregate fitting metadata")
    payload = {
        "schema": UTILITY_SCHEMA,
        "state_dict": {name: value.detach().cpu().clone() for name, value in router.state_dict().items()},
        "source_checkpoint_sha256": checkpoint_sha256,
        "run_identity_sha256": run_identity_sha256,
        "base_model_config_sha256": base_model_config_sha256,
        "source_revision": source_revision, **dict(fitting),
    }
    for name in ("joint_router_hidden_dim", "max_passes", "n_layers", "max_routed_k"):
        payload[name] = getattr(model.config, name)
    validate_utility_provenance(payload, model.config)
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return {**identify_file(path), "schema": UTILITY_SCHEMA, "trained_updates": int(router.trained_updates.item())}


def shuffle_plan(
    depths: Tensor, widths: Tensor, attention_mask: Tensor, *, seed: int,
    document_ids: Tensor | None = None, supervised_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Permute entire trajectories within sequence/document, preserving work."""
    if depths.shape != attention_mask.shape or widths.shape[2:] != depths.shape:
        raise ValueError("Shuffle depth/width/mask shapes do not agree")
    if document_ids is not None and document_ids.shape != depths.shape:
        raise ValueError("Shuffle document ids have the wrong shape")
    if supervised_mask is not None and supervised_mask.shape != depths.shape:
        raise ValueError("Shuffle supervision mask has the wrong shape")
    shuffled_depths, shuffled_widths = depths.clone(), widths.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch_index in range(depths.shape[0]):
        mask = attention_mask[batch_index]
        docs = document_ids[batch_index] if document_ids is not None else torch.zeros_like(depths[batch_index])
        if supervised_mask is not None:
            docs = 2 * docs.long() + supervised_mask[batch_index].long()
        for document in docs[mask].unique():
            indices = torch.nonzero(mask & docs.eq(document), as_tuple=False).flatten()
            order = torch.randperm(indices.numel(), generator=generator).to(indices.device)
            source = indices[order]
            shuffled_depths[batch_index, indices] = depths[batch_index, source]
            shuffled_widths[:, :, batch_index, indices] = widths[:, :, batch_index, source]
    return shuffled_depths, shuffled_widths


def compare_fixed_policies(
    model: Metis16ForCausalLM,
    batch: TrainingBatch,
    curriculum: CurriculumState,
    policies: Sequence[tuple[int, int]],
    *,
    seed: int,
    runtime_state: FrozenRuntimeState,
    repeat_forwards: int,
    minimum_loss_delta: float,
) -> list[dict[str, Any]]:
    """Separate learned execution from backbone quality without changing weights."""
    if model.config.ffn_mode != "moe":
        raise ValueError("Fixed depth/width interventions require an MoE checkpoint.")
    if len(set(policies)) != len(policies):
        raise ValueError("Fixed policy interventions must be distinct.")
    records = []
    for depth, width in policies:
        if not 1 <= depth <= model.config.max_passes:
            raise ValueError("Intervention depth is outside the checkpoint architecture.")
        if not model.config.min_routed_k <= width <= model.config.max_routed_k:
            raise ValueError("Intervention width is outside the checkpoint architecture.")
        depths = batch.attention_mask.long() * depth
        active = torch.arange(
            model.config.max_passes, device=depths.device
        )[:, None, None] < depths
        widths = (
            active[:, None].expand(-1, model.config.n_layers, -1, -1).long() * width
        )
        fixed = replace(
            curriculum, compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", fixed_routed_k=width, max_passes=depth,
            stochastic_routing=False,
        )
        result, statistics = repeated_plan_evaluation(
            model, batch, fixed, seed=seed, runtime_state=runtime_state,
            repeat_forwards=repeat_forwards, minimum_loss_delta=minimum_loss_delta,
            force_depth=depths, force_routed_k=widths,
        )
        records.append({
            "depth": depth,
            "routed_k": width,
            **forward_summary(result, batch),
            "mean_lm_loss": statistics["mean_lm_loss"],
            "repeat_statistics": statistics,
            "interpretation": (
                "Same frozen checkpoint under an explicit fixed policy; not its trained "
                "policy, not necessarily iso-compute, and not a deployment qualification."
            ),
        })
    runtime_state.restore()
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=int, default=7000)
    parser.add_argument(
        "--quality-only", action="store_true",
        help="Assess the checkpoint's recorded policy, including fixed/dense controls, without credit interventions",
    )
    parser.add_argument(
        "--fixed-policy", type=int, nargs=2, action="append", default=[],
        metavar=("DEPTH", "WIDTH"),
        help="Additional frozen-weight diagnostic alongside --quality-only; repeat for each fixed policy",
    )
    parser.add_argument(
        "--evaluation-gap-blocks", type=int, default=0,
        help="Use an unsampled stride-gap block, disjoint from the entire declared training run",
    )
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--repeat-forwards", type=int, default=3)
    parser.add_argument("--warmup-forwards", type=int, default=1)
    parser.add_argument("--minimum-loss-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=16062026)
    parser.add_argument("--policy", choices=("legacy", "joint"), default="legacy")
    parser.add_argument("--precision", choices=("checkpoint", "bf16", "fp8"), default="checkpoint")
    parser.add_argument("--utility-checkpoint", type=Path)
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-start-step", type=int, default=7000)
    parser.add_argument("--fit-learning-rate", type=float, default=0.0003)
    parser.add_argument("--fit-max-depth", type=int, default=5)
    parser.add_argument("--joint-caps", type=int, nargs="+", help="Joint evaluation depth caps, e.g. 3 5")
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.fixed_policy and not args.quality_only:
        raise ValueError("--fixed-policy requires --quality-only; it never replaces the primary metric.")
    if args.pairs < 0:
        raise ValueError("--pairs cannot be negative")
    if args.repeat_forwards < 2:
        raise ValueError("--repeat-forwards must be at least two")
    if args.warmup_forwards < 0:
        raise ValueError("--warmup-forwards cannot be negative")
    if not math.isfinite(args.minimum_loss_delta) or args.minimum_loss_delta < 0:
        raise ValueError("--minimum-loss-delta must be finite and nonnegative")
    if args.fit_steps < 0:
        raise ValueError("--fit-steps cannot be negative")
    if args.fit_steps and args.utility_checkpoint is not None:
        raise ValueError("Fitting starts a fresh head; do not combine it with --utility-checkpoint")
    joint_requested = args.policy == "joint" or args.fit_steps > 0
    if args.quality_only and (
        joint_requested or args.utility_checkpoint is not None or args.joint_caps is not None
    ):
        raise ValueError("Quality-only assessment uses the checkpoint's own trained policy and weights")
    if joint_requested and not args.fit_steps and args.utility_checkpoint is None:
        raise CapabilityError("Joint evaluation requires fitted utility weights or --fit-steps")
    if not joint_requested and (args.utility_checkpoint is not None or args.joint_caps is not None):
        raise CapabilityError("Utility artifacts and joint caps require joint evaluation")
    if (
        args.fit_steps and args.evaluation_gap_blocks == 0
        and args.fit_start_step <= args.step < args.fit_start_step + args.fit_steps
    ):
        raise ValueError("Evaluation sampler step overlaps the utility fitting steps")
    source = source_identity()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint = checkpoint / "state.pt" if checkpoint.is_dir() else checkpoint
    manifest_path = (args.run_manifest or infer_run_manifest(checkpoint)).expanduser().resolve(strict=True)
    output = fresh_output_directory(
        args.output, inputs=(checkpoint.parent.parent, args.release_root, manifest_path),
    )
    checkpoint_identity = identify_file(checkpoint)
    manifest_identity = identify_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # mmap limits peak host memory and never reads any optimizer-rank file.
    # weights_only does not execute arbitrary checkpoint pickle globals.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    identity = validate_checkpoint(payload, manifest)
    checkpoint_step = int(payload["step"])
    config = Metis16Config.from_mapping(identity["model"])
    if args.fit_steps and not 2 <= args.fit_max_depth <= config.max_passes:
        raise ValueError("--fit-max-depth must lie within the checkpoint architecture")
    caps = args.joint_caps or [config.max_passes]
    if joint_requested and (
        len(caps) != len(set(caps)) or any(not 2 <= cap <= config.max_passes for cap in caps)
    ):
        raise ValueError("Joint caps must be distinct integers within [2, model.max_passes]")
    if args.sequence_length > config.final_context_length:
        raise ValueError("Requested probe sequence exceeds the model context limit")
    curriculum = CurriculumState.from_value(identity["curriculum"])
    curriculum = replace(curriculum, stochastic_routing=False, random_policy_step=args.step)
    if hasattr(curriculum, "compute_allocation_mode") and not args.quality_only:
        curriculum = replace(curriculum, compute_allocation_mode="legacy")
    if not args.quality_only and curriculum.continuation_mode not in {"adaptive", "budgeted"}:
        raise CapabilityError("Depth-credit probe requires a learned continuation policy")
    inventory = ReleaseInventory.from_release_root(args.release_root)
    release_descriptor = identify_file(inventory.root / "RELEASE.json")
    batch, sampling = held_out_batch(
        inventory, identity, checkpoint_step=checkpoint_step, step=args.step,
        sequences=args.sequences, sequence_length=args.sequence_length,
        gap_blocks=args.evaluation_gap_blocks,
    )
    precision = identity["precision_profile"] if args.precision == "checkpoint" else args.precision
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise CapabilityError("Probe precision policy supports explicit cpu or cuda devices")
    device = _select_probe_device(device)
    model, numerical = load_frozen_model(
        config, payload["model"], device=device, precision=precision,
        enable_joint=(
            bool(getattr(config, "joint_compute_router", False))
            if args.quality_only else joint_requested
        ),
        checkpoint_precision=identity["precision_profile"],
        initialization_seed=args.seed,
    )
    del payload
    utility = None
    fitting = None
    warmups = []
    warmup_window = None
    if args.warmup_forwards:
        if args.fit_steps:
            warmup_batch, warmup_window = held_out_batch(
                inventory, identity, checkpoint_step=checkpoint_step, step=args.fit_start_step,
                sequences=args.sequences, sequence_length=args.sequence_length,
            )
            assert_disjoint_windows(warmup_window, sampling)
        else:
            warmup_batch, warmup_window = batch, sampling
        warmup_batch = warmup_batch.to(device)
        for _ in range(args.warmup_forwards):
            warmup_result = evaluate_in_memory(
                model, warmup_batch, curriculum, seed=args.seed, runtime_state=None,
            )
            warmups.append(forward_summary(warmup_result, warmup_batch))
        del warmup_batch, warmup_result
    warmup_cost = sum(item["nominal_probe_forward_flops"] for item in warmups)
    warmup_elapsed = sum(item["elapsed_seconds_including_hooks_and_gradient"] for item in warmups)
    if args.fit_steps:
        def fitting_batches():
            for fit_step in range(args.fit_start_step, args.fit_start_step + args.fit_steps):
                yield held_out_batch(
                    inventory, identity, checkpoint_step=checkpoint_step, step=fit_step,
                    sequences=args.sequences, sequence_length=args.sequence_length,
                )

        fitting = fit_utility_in_memory(
            model, curriculum, fitting_batches(), steps=args.fit_steps, seed=args.seed,
            learning_rate=args.fit_learning_rate, max_depth=args.fit_max_depth,
            evaluation_window=sampling,
        )
        fitting.update({
            "warmup_forward_calls": len(warmups),
            "warmup_token_count": sum(item["non_padding_tokens"] for item in warmups),
            "warmup_elapsed_seconds": warmup_elapsed,
            "nominal_warmup_forward_flops": warmup_cost,
            "warmup_window": warmup_window,
        })
        for item in (checkpoint_identity, manifest_identity, release_descriptor):
            assert_file_unchanged(item)
        if source_identity() != source:
            raise RuntimeError("Source changed during utility fitting; refusing to seal the artifact")
        saved = save_utility_artifact(
            model, output / "utility.pt", fitting,
            checkpoint_sha256=checkpoint_identity["sha256"],
            run_identity_sha256=manifest["run_identity_sha256"],
            base_model_config_sha256=json_sha256(identity["model"]),
            source_revision=source["revision"],
        )
        utility = load_utility_artifact(
            model, Path(saved["path"]), checkpoint_sha256=checkpoint_identity["sha256"],
            run_identity_sha256=manifest["run_identity_sha256"],
            base_model_config_sha256=json_sha256(identity["model"]), evaluation_window=sampling,
        )
    if args.utility_checkpoint is not None:
        utility = load_utility_artifact(
            model, args.utility_checkpoint, checkpoint_sha256=checkpoint_identity["sha256"],
            run_identity_sha256=manifest["run_identity_sha256"],
            base_model_config_sha256=json_sha256(identity["model"]),
            evaluation_window=sampling,
        )
    batch = batch.to(device)
    runtime = FrozenRuntimeState(model)
    legacy, legacy_noise = repeated_plan_evaluation(
        model, batch, curriculum, seed=args.seed, runtime_state=runtime,
        repeat_forwards=args.repeat_forwards, minimum_loss_delta=args.minimum_loss_delta,
    )
    if args.quality_only:
        interventions = compare_fixed_policies(
            model, batch, curriculum, [tuple(policy) for policy in args.fixed_policy],
            seed=args.seed, runtime_state=runtime, repeat_forwards=args.repeat_forwards,
            minimum_loss_delta=args.minimum_loss_delta,
        ) if args.fixed_policy else []
        runtime.restore()
        for item in (checkpoint_identity, manifest_identity, release_descriptor):
            assert_file_unchanged(item)
        if source_identity() != source:
            raise RuntimeError("Source identity changed during checkpoint assessment")
        report = {
            "schema": "more.checkpoint-quality/v1",
            "status": "complete",
            "source": source,
            "checkpoint": checkpoint_identity,
            "run_manifest": manifest_identity,
            "run_identity_sha256": manifest["run_identity_sha256"],
            "base_model_config_sha256": json_sha256(identity["model"]),
            "training_source_revision": identity.get("source_revision"),
            "training_curriculum": identity["curriculum"],
            "release": {"descriptor": release_descriptor, **identity["release"]},
            "sampling": sampling,
            "seed": args.seed,
            "policy": "recorded_checkpoint_policy",
            "numerical_policy": numerical,
            "precision_changed_from_checkpoint": precision != identity["precision_profile"],
            "evaluation": {
                **forward_summary(legacy, batch),
                "mean_lm_loss": legacy_noise["mean_lm_loss"],
                "repeat_statistics": legacy_noise,
                "modeled_total_train_flops": int(legacy.output.telemetry.get(
                    "joint_model_flops", legacy.cost["nominal_train_flops"]
                )),
            },
            "warmup_forward_calls": len(warmups),
            "warmup_token_count": sum(item["non_padding_tokens"] for item in warmups),
            "warmup_elapsed_seconds": warmup_elapsed,
            "forward_calls": len(warmups) + args.repeat_forwards * (1 + len(interventions)),
            "nominal_forward_flops_all_calls": (
                warmup_cost + legacy_noise["nominal_forward_flops_all_calls"]
                + sum(
                    item["repeat_statistics"]["nominal_forward_flops_all_calls"]
                    for item in interventions
                )
            ),
            "total_wall_seconds_including_loading": time.perf_counter() - started,
            "limitations": [
                "Checkpoint-policy likelihood, not a counterfactual credit assessment.",
                "Token-disjoint evaluation is not deduplicated document holdout.",
                "A single window does not establish superiority or full-training convergence.",
            ],
        }
        if interventions:
            report["fixed_policy_interventions"] = interventions
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report
    legacy_replay, legacy_replay_noise = repeated_plan_evaluation(
        model, batch, curriculum, seed=args.seed, runtime_state=runtime,
        repeat_forwards=args.repeat_forwards, minimum_loss_delta=args.minimum_loss_delta,
        force_depth=legacy.output.chosen_depths.detach(), force_routed_k=legacy.widths,
    )
    legacy_replay_gap = legacy_replay_noise["mean_lm_loss"] - float(legacy.output.loss.detach())
    legacy_replay_parity = abs(legacy_replay_gap) <= legacy_replay_noise["decisive_absolute_loss_delta_threshold"]
    uniform_depths = batch.attention_mask.long() * 2
    uniform_widths = torch.zeros(
        (config.max_passes, config.n_layers, *batch.input_ids.shape),
        dtype=torch.long, device=device,
    )
    uniform_widths[:2] = batch.attention_mask.long()[None, None] * 4
    uniform, uniform_noise = repeated_plan_evaluation(
        model, batch,
        replace(curriculum, continuation_mode="fixed_max", routed_k_mode="fixed", fixed_routed_k=4, max_passes=2),
        seed=args.seed, runtime_state=runtime, repeat_forwards=args.repeat_forwards,
        minimum_loss_delta=args.minimum_loss_delta,
        force_depth=uniform_depths, force_routed_k=uniform_widths,
    )
    depth = depth_credit_probe(
        model, batch, curriculum, pairs=args.pairs, seed=args.seed, runtime_state=runtime,
        repeat_forwards=args.repeat_forwards, minimum_loss_delta=args.minimum_loss_delta,
    )
    joint_evaluations = [
        joint_policy_probe(
            model, batch, curriculum, seed=args.seed, runtime_state=runtime, legacy=legacy,
            repeat_forwards=args.repeat_forwards, minimum_loss_delta=args.minimum_loss_delta,
            max_passes=cap, trained_max_depth=utility.get("fit_max_depth"),
            legacy_noise=legacy_noise,
            fixed_reference=uniform, fixed_noise=uniform_noise,
        ) for cap in caps
    ] if joint_requested else []
    joint = joint_evaluations[0] if joint_evaluations else None
    runtime.restore()
    for item in (checkpoint_identity, manifest_identity, release_descriptor):
        assert_file_unchanged(item)
    if utility is not None:
        assert_file_unchanged(utility)
    if source_identity() != source:
        raise RuntimeError("Source identity changed while the probe was executing")
    report = {
        "schema": "more.routing-credit-probe/v1", "status": "complete",
        "source": source, "checkpoint": checkpoint_identity,
        "run_manifest": manifest_identity, "run_identity_sha256": manifest["run_identity_sha256"],
        "base_model_config_sha256": json_sha256(identity["model"]),
        "training_source_revision": identity.get("source_revision"),
        "release": {"descriptor": release_descriptor, **identity["release"]},
        "utility_checkpoint": utility,
        "utility_fitting": fitting,
        "warmup": {
            "forward_calls": len(warmups), "window": warmup_window,
            "forwards": warmups, "elapsed_seconds": warmup_elapsed,
            "nominal_forward_flops": warmup_cost,
            "before_runtime_snapshots": True,
            "interpretation": (
                "Explicitly counted ordinary legacy warm-up before runtime snapshots. Fitting uses "
                "a training window, never held-out inputs, for this warm-up. Materialized-weight and "
                "kernel caches may still vary by shape; natural-policy variability remains reported."
            ),
        },
        "sampling": sampling, "seed": args.seed, "policy": "joint" if joint_requested else "legacy",
        "numerical_policy": numerical,
        "precision_changed_from_checkpoint": precision != identity["precision_profile"],
        "legacy": {
            **forward_summary(legacy, batch),
            "mean_natural_policy_lm_loss": legacy_noise["mean_lm_loss"],
            "natural_policy_variability": legacy_noise,
            "deterministic_repeat_noise": legacy_replay_noise,
            "forced_reference": {
                **forward_summary(legacy_replay, batch),
                "mean_lm_loss": legacy_replay_noise["mean_lm_loss"],
                "mean_loss_delta_from_captured_natural": legacy_replay_gap,
                "matches_captured_natural_within_fixed_plan_noise": legacy_replay_parity,
            },
            "total_repeat_evaluation_seconds": (
                legacy_noise["total_evaluation_seconds"] + legacy_replay_noise["total_evaluation_seconds"]
            ),
            "forward_calls": 2 * args.repeat_forwards,
        },
        "depth_credit": depth, "joint": joint, "joint_evaluations": joint_evaluations,
        "uniform_depth2_k4": {
            **forward_summary(uniform, batch),
            "mean_lm_loss": uniform_noise["mean_lm_loss"],
            "deterministic_repeat_noise": uniform_noise,
            "forward_calls": uniform_noise["forward_calls"],
            "interpretation": "Same frozen backbone and precision, uniform depth 2 and k=4; a candidate must beat this control, not merely approximate it.",
        },
        "interpretation_checks": {
            "legacy_captured_plan_reference_parity": legacy_replay_parity,
            "depth_gradient_reference_parity": depth["gradient_interpretation_valid"],
            "joint_captured_plan_reference_parity_by_cap": {
                str(item["max_passes"]): item["forced_original"]["matches_joint_within_numerical_noise"]
                for item in joint_evaluations
            },
            "failed_reference_checks_invalidate_related_gradient_or_policy_alignment_claims": True,
        },
        "total_wall_seconds_including_loading": time.perf_counter() - started,
        "forward_calls": (
            len(warmups) + 2 * args.repeat_forwards + depth["forward_calls"] + uniform_noise["forward_calls"]
            + sum(item["forward_calls"] for item in joint_evaluations)
            + (fitting["teacher_forward_calls"] if fitting else 0)
        ),
        "autograd_calls": depth["autograd_calls"] + (fitting["teacher_backward_calls"] if fitting else 0),
        "nominal_forward_flops_all_calls": (
            warmup_cost + legacy_noise["nominal_forward_flops_all_calls"] + legacy_replay_noise["nominal_forward_flops_all_calls"]
            + uniform_noise["nominal_forward_flops_all_calls"]
            + depth["nominal_forward_flops_all_calls"]
            + sum(item["nominal_forward_flops_all_calls"] for item in joint_evaluations)
            + (fitting["nominal_teacher_forward_flops"] if fitting else 0)
        ),
        "limitations": [
            "No backbone optimizer resume or backbone updates. Only optional utility-head weights and "
            "aggregate provenance are persisted, never training samples, tokens, text, or hidden states.",
            "Nominal costs use the existing train-FLOP ledger divided by three for forward estimates; "
            "they are not hardware measurements and exclude padding, hashing, hooks, and state replay.",
            "The original configured sequence length is retained for campaign cost accounting; "
            "actual probe attention context is the shorter reported sampling length.",
            "Depth-pair swaps preserve per-pass token counts and per-layer expert assignments within one "
            "document, but global loss is necessary because downstream causal context can change.",
            "Continuation autograd freezes all other parameters; nominal full-backward cost is an upper "
            "bound, not measured issued FLOPs.",
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return report


def main(argv: list[str] | None = None) -> int:
    report = run_probe(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
