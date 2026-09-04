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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor

from metis_training.data import (
    DeterministicReleaseStream,
    ReleaseInventory,
    TrainingBatch,
)
from metis_training.metrics import estimate_train_flops
from metis_training.model import AdaptiveDroplessMoE, CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config
from metis_training.precision import build_precision_policy

from .sampler import AblationSampleStream, build_sample_stream


SHARD_ORDER_SEED = 16_062_026
UTILITY_SCHEMA = "more.routing-utility/v1"


class CapabilityError(RuntimeError):
    """The requested probe needs an API or artifact that is not available."""


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
            if not isinstance(layer.moe, AdaptiveDroplessMoE):
                self.__exit__()
                raise CapabilityError("Routing probes require AdaptiveDroplessMoE layers")

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
            if bool(((valid < config.min_routed_k) | (valid > config.max_routed_k)).any()):
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


def plan_cost(config: Metis16Config, depths: Tensor, widths: Tensor) -> dict[str, Any]:
    """Integer ledger using audited train-FLOP prefix increments, not mean k."""
    if widths.shape != (config.max_passes, config.n_layers, *depths.shape):
        raise ValueError("Widths do not match the model's full plan geometry")
    if depths.dtype not in (torch.int32, torch.int64) or widths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cost plans require integer depths and widths")
    if bool(((depths < 0) | (depths > config.max_passes)).any()):
        raise ValueError("Depth plan is outside the architectural cap")
    expert_cost = 18 * config.latent_dim * config.expert_intermediate_dim
    reference = replace(config, joint_compute_router=False) if hasattr(config, "joint_compute_router") else config
    previous = 0
    modeled = 0
    active_counts, assignments, histograms = [], [], []
    for p in range(config.max_passes):
        active = depths > p
        valid = widths[p, :, active]
        if valid.numel() and bool(((valid < config.min_routed_k) | (valid > config.max_routed_k)).any()):
            raise ValueError("Active plan widths are outside routed-k bounds")
        if bool((widths[p, :, ~active] != 0).any()):
            raise ValueError("Inactive plan widths must be zero")
        prefix = round(estimate_train_flops(
            reference, tokens=1, observed_mean_passes=float(p + 1),
            observed_mean_routed_k=1.0,
        ) - (p + 1) * config.n_layers * expert_cost)
        base = prefix - previous
        previous = prefix
        count = int(active.sum())
        layer_assignments = [int(widths[p, layer].sum()) for layer in range(config.n_layers)]
        modeled += count * base + sum(layer_assignments) * expert_cost
        active_counts.append(count)
        assignments.append(layer_assignments)
        histograms.append([histogram(widths[p, layer][active]) for layer in range(config.n_layers)])
    return {
        "nominal_train_flops": modeled,
        "nominal_forward_flops": modeled / 3.0,
        "active_tokens_by_pass": active_counts,
        "expert_assignments_by_pass_layer": assignments,
        "chosen_k_by_pass_layer": histograms,
        "tokens": int((depths > 0).sum()),
        "token_passes": int(depths.sum()),
        "expert_assignments": sum(map(sum, assignments)),
    }


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

    def __init__(self, model: Metis16ForCausalLM):
        self.model = model
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        self.extra = {}
        for name, module in model.named_modules():
            if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
                self.extra[name] = copy.deepcopy(module.get_extra_state())

    @torch.no_grad()
    def restore(self) -> None:
        for name, value in self.model.named_buffers():
            if name not in self.buffers or value.shape != self.buffers[name].shape:
                raise RuntimeError("Mutable buffer inventory changed during frozen evaluation")
            value.copy_(self.buffers[name])
        modules = dict(self.model.named_modules())
        for name, value in self.extra.items():
            modules[name].set_extra_state(copy.deepcopy(value))


@contextmanager
def frozen_model(model: Metis16ForCausalLM, *, continuation_grad: bool = False) -> Iterator[None]:
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
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        model.activation_recompute_policy = replay


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
    device = batch.input_ids.device
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
    precision_context = nullcontext()
    if getattr(model, "_routing_probe_bf16_reference", False):
        factory = getattr(model.precision_policy, "bf16_reference_context", None)
        if not callable(factory):
            raise CapabilityError("The original checkpoint backend cannot provide a BF16 reference context")
        precision_context = factory()
    with frozen_model(model, continuation_grad=continuation_grad), torch.random.fork_rng(devices=cuda_devices), precision_context:
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
    cost = plan_cost(model.config, output.chosen_depths, widths)
    return ProbeForward(output, widths, elapsed, gradients, observed, cost)


def forward_summary(result: ProbeForward, batch: TrainingBatch) -> dict[str, Any]:
    depths = result.output.chosen_depths[batch.attention_mask]
    routed = result.widths[result.widths > 0]
    return {
        "lm_loss": float(result.output.loss.detach()),
        "supervised_tokens": int((batch.labels != -100).sum()),
        "non_padding_tokens": int(batch.attention_mask.sum()),
        "chosen_depth_histogram": histogram(depths),
        "chosen_k_histogram_over_active_layer_pass_tokens": histogram(routed),
        "mean_chosen_depth": float(depths.float().mean()),
        "mean_chosen_k_over_active_layer_pass_tokens": float(routed.float().mean()),
        "elapsed_seconds_including_hooks_and_gradient": result.elapsed_seconds,
        "cost": result.cost,
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
    if abs(original_loss - replay_loss) > threshold:
        raise RuntimeError("Forced original plan changes LM loss beyond repeat noise; comparison is confounded")
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
                "below_numerical_noise" if abs(delta) <= threshold
                else credit_alignment(predicted, delta)
            ),
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
) -> tuple[TrainingBatch, dict[str, Any]]:
    if step < checkpoint_step:
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
    cursor = sampler.release_cursor(step)
    last_consumed_end = None
    if checkpoint_step:
        last_consumed_end = sampler.release_cursor(checkpoint_step - 1) + block
        if cursor <= last_consumed_end:
            raise ValueError("Probe overlaps a trained block or its target lookahead; choose a later --step")
    batch = stream.batch(
        global_token_cursor=cursor, rank=0, world_size=1, micro_batch_size=sequences,
    )
    if batch.supervised_tokens <= 0:
        raise RuntimeError("Held-out window has no supervised tokens")
    return batch, {
        **identify_batch(batch), "sampler_step": step, "checkpoint_next_unexecuted_step": checkpoint_step,
        "block_tokens": block, "sampled_fraction_of_block": sequences * sequence_length / block,
        "last_consumed_block_end_including_target_lookahead": last_consumed_end,
        "shard_order_seed": SHARD_ORDER_SEED, "sampler_sha256": json_sha256(sampler_spec),
        "held_out_scope": (
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
    output["target_interpretation"] = (
        "Observed per-exit CE improvement, not an unbiased global counterfactual utility. "
        "Unknown continuation outcomes must remain unlabeled."
    )
    return output


def load_frozen_model(
    config: Metis16Config, weights: Mapping[str, Any], *,
    device: torch.device, precision: str, enable_joint: bool,
    checkpoint_precision: str | None = None,
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
    policy = build_precision_policy(
        effective.precision, profile=checkpoint_precision, device=device,
        production=False, permit_fallback=False,
    )
    model = Metis16ForCausalLM(effective, precision_policy=policy)
    model.apply_parameter_storage_policy(device=device)
    target_keys = set(model.state_dict())
    loaded = model.load_state_dict(weights, strict=not enable_joint)
    allowed_missing = {
        key for key in target_keys if enable_joint and key.startswith("joint_router.")
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
    return model.eval(), {
        "backend_precision_audit": policy.audit.to_dict(),
        "effective_profile": precision,
        "constructed_backend_profile": checkpoint_precision,
        "bf16_reference_context": precision == "bf16",
        "discarded_checkpoint_state_keys": [],
        "new_joint_state_keys": sorted(set(loaded.missing_keys)),
        "model_config": effective.to_dict(),
        "optimizer_loaded": False, "optimizer_shards_loaded": 0,
        "original_world_size": config.world_size, "probe_world_size": 1,
        "activation_recompute_executed": False,
    }


def joint_policy_probe(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    seed: int, runtime_state: FrozenRuntimeState, legacy: ProbeForward,
) -> dict[str, Any]:
    if "compute_allocation_mode" not in curriculum.__dataclass_fields__:
        raise CapabilityError("CurriculumState does not support joint allocation")
    joint = replace(
        curriculum, compute_allocation_mode="joint", stochastic_routing=False,
        joint_router_exploration=0.0, allow_untrained_joint_router=False,
    )
    result = evaluate_in_memory(model, batch, joint, seed=seed, runtime_state=runtime_state)
    telemetry = result.output.telemetry
    required = ("joint_model_flops", "joint_router_flops")
    if any(key not in telemetry for key in required):
        raise CapabilityError("Joint model must expose joint_model_flops and joint_router_flops")
    delta = result.cost["nominal_train_flops"] - legacy.cost["nominal_train_flops"]
    return {
        **forward_summary(result, batch),
        "global_loss_delta_from_legacy": float(result.output.loss.detach() - legacy.output.loss.detach()),
        "nominal_backbone_train_flops_delta_from_legacy": delta,
        "exact_equal_backbone_budget": delta == 0,
        "joint_model_flops": float(telemetry["joint_model_flops"]),
        "joint_router_flops": float(telemetry["joint_router_flops"]),
        "policy_quality_established": False,
        "missing_quality_control": "Cost-matched shuffled complete depth/width trajectories",
        "interpretation": (
            "Frozen-weight, same-token policy comparison. Cost differences are reported, "
            "not normalized away; this is not a convergence or equal-wall-time claim."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=int, default=7000)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--repeat-forwards", type=int, default=3)
    parser.add_argument("--minimum-loss-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=16062026)
    parser.add_argument("--policy", choices=("legacy", "joint"), default="legacy")
    parser.add_argument("--precision", choices=("checkpoint", "bf16", "fp8"), default="checkpoint")
    parser.add_argument("--utility-checkpoint", type=Path)
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.pairs < 0:
        raise ValueError("--pairs cannot be negative")
    if args.repeat_forwards < 2:
        raise ValueError("--repeat-forwards must be at least two")
    if not math.isfinite(args.minimum_loss_delta) or args.minimum_loss_delta < 0:
        raise ValueError("--minimum-loss-delta must be finite and nonnegative")
    if (args.policy == "joint") != (args.utility_checkpoint is not None):
        raise CapabilityError("--policy joint requires --utility-checkpoint, and legacy does not use one")
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
    if args.sequence_length > config.final_context_length:
        raise ValueError("Requested probe sequence exceeds the model context limit")
    curriculum = CurriculumState.from_value(identity["curriculum"])
    curriculum = replace(curriculum, stochastic_routing=False, random_policy_step=args.step)
    if hasattr(curriculum, "compute_allocation_mode"):
        curriculum = replace(curriculum, compute_allocation_mode="legacy")
    if curriculum.continuation_mode not in {"adaptive", "budgeted"}:
        raise CapabilityError("Depth-credit probe requires a learned continuation policy")
    inventory = ReleaseInventory.from_release_root(args.release_root)
    release_descriptor = identify_file(inventory.root / "RELEASE.json")
    batch, sampling = held_out_batch(
        inventory, identity, checkpoint_step=checkpoint_step, step=args.step,
        sequences=args.sequences, sequence_length=args.sequence_length,
    )
    precision = identity["precision_profile"] if args.precision == "checkpoint" else args.precision
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise CapabilityError("Probe precision policy supports explicit cpu or cuda devices")
    model, numerical = load_frozen_model(
        config, payload["model"], device=device, precision=precision,
        enable_joint=args.policy == "joint",
        checkpoint_precision=identity["precision_profile"],
    )
    del payload
    utility = None
    if args.utility_checkpoint is not None:
        utility = load_utility_artifact(
            model, args.utility_checkpoint, checkpoint_sha256=checkpoint_identity["sha256"],
            run_identity_sha256=manifest["run_identity_sha256"],
            base_model_config_sha256=json_sha256(identity["model"]),
            evaluation_window=sampling,
        )
    batch = batch.to(device)
    runtime = FrozenRuntimeState(model)
    legacy = evaluate_in_memory(model, batch, curriculum, seed=args.seed, runtime_state=runtime)
    legacy_losses = [float(legacy.output.loss.detach())]
    legacy_elapsed = legacy.elapsed_seconds
    for _ in range(args.repeat_forwards - 1):
        repeated = evaluate_in_memory(model, batch, curriculum, seed=args.seed, runtime_state=runtime)
        if (
            repeated.cost != legacy.cost
            or not torch.equal(repeated.widths, legacy.widths)
            or not torch.equal(repeated.output.chosen_depths, legacy.output.chosen_depths)
        ):
            raise RuntimeError("Repeated legacy evaluation changed its actual routing plan")
        legacy_losses.append(float(repeated.output.loss.detach()))
        legacy_elapsed += repeated.elapsed_seconds
    legacy_noise = repeat_noise(legacy_losses, minimum_delta=args.minimum_loss_delta)
    depth = depth_credit_probe(
        model, batch, curriculum, pairs=args.pairs, seed=args.seed, runtime_state=runtime,
        repeat_forwards=args.repeat_forwards, minimum_loss_delta=args.minimum_loss_delta,
    )
    joint = joint_policy_probe(
        model, batch, curriculum, seed=args.seed, runtime_state=runtime, legacy=legacy,
    ) if args.policy == "joint" else None
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
        "sampling": sampling, "seed": args.seed, "policy": args.policy,
        "numerical_policy": numerical,
        "precision_changed_from_checkpoint": precision != identity["precision_profile"],
        "legacy": {
            **forward_summary(legacy, batch),
            "deterministic_repeat_noise": legacy_noise,
            "total_repeat_evaluation_seconds": legacy_elapsed,
            "forward_calls": args.repeat_forwards,
        },
        "depth_credit": depth, "joint": joint,
        "total_wall_seconds_including_loading": time.perf_counter() - started,
        "forward_calls": args.repeat_forwards + depth["forward_calls"] + int(joint is not None),
        "autograd_calls": depth["autograd_calls"],
        "nominal_forward_flops_all_calls": (
            args.repeat_forwards * legacy.cost["nominal_forward_flops"] + depth["nominal_forward_flops_all_calls"]
            + (
                joint["cost"]["nominal_forward_flops"] + joint["joint_router_flops"] / 3.0
                if joint is not None else 0
            )
        ),
        "limitations": [
            "No optimizer resume, updates, training samples, tokens, text, or hidden states are persisted.",
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
