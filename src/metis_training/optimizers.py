from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from metis_mamba.optim import OptimizerBuildSummary, build_muon_adamw_optimizer

from .distributed import ParallelTopology


@dataclass(frozen=True)
class TrainingOptimizerSummary:
    dense: dict[str, Any]
    sparse_tensor_count: int
    sparse_parameter_count: int
    fp32_master_weights: bool
    fp32_optimizer_states: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FP32MasterSparseAdam(Optimizer):
    """Sparse Adam with FP32 master rows and FP32 moments.

    State tensors have the local parameter shape. In production the N-gram
    tables should be row-sharded, so this cost is divided across the EP group.
    Only touched rows participate in each optimizer update.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1.0e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("Sparse Adam learning rate must be positive")
        super().__init__(
            [{"params": list(params), "lr": lr, "_metis_lr_scale": 1.0}],
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0},
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not gradient.is_sparse:
                    raise RuntimeError(
                        "N-gram table produced a dense gradient; refusing to allocate "
                        "an implicit full-table update"
                    )
                gradient = gradient.coalesce()
                indices = gradient.indices()
                if indices.ndim != 2 or indices.shape[0] != 1:
                    raise RuntimeError("Sparse table gradients must be row-sparse")
                rows = indices[0]
                values = gradient.values().float()
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["master_param"] = parameter.detach().float().clone()
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] = int(state["step"]) + 1
                step = int(state["step"])
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                row_avg = exp_avg.index_select(0, rows)
                row_avg_sq = exp_avg_sq.index_select(0, rows)
                row_avg.mul_(beta1).add_(values, alpha=1.0 - beta1)
                row_avg_sq.mul_(beta2).addcmul_(values, values, value=1.0 - beta2)
                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias2) / bias1
                master_rows = master.index_select(0, rows)
                master_rows.addcdiv_(
                    row_avg,
                    row_avg_sq.sqrt().add_(eps),
                    value=-step_size,
                )
                exp_avg.index_copy_(0, rows, row_avg)
                exp_avg_sq.index_copy_(0, rows, row_avg_sq)
                master.index_copy_(0, rows, master_rows)
                parameter.index_copy_(0, rows, master_rows.to(dtype=parameter.dtype))
        return loss


class OptimizerBundle:
    def __init__(
        self,
        dense: Optimizer,
        sparse: FP32MasterSparseAdam | None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups = list(self.dense.param_groups)
        if self.sparse is not None:
            groups.extend(self.sparse.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.dense.zero_grad(set_to_none=set_to_none)
        if self.sparse is not None:
            self.sparse.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.dense.step()
        if self.sparse is not None:
            self.sparse.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "dense": self.dense.state_dict(),
            "sparse": self.sparse.state_dict() if self.sparse is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.dense.load_state_dict(state["dense"])
        sparse_state = state.get("sparse")
        if sparse_state is not None:
            if self.sparse is None:
                raise RuntimeError("Checkpoint has sparse optimizer state but model does not")
            self.sparse.load_state_dict(sparse_state)
        elif self.sparse is not None:
            raise RuntimeError("Checkpoint is missing sparse optimizer state")


def _parameter_placements(model: nn.Module) -> dict[str, str]:
    provider = getattr(model, "parameter_placements", None)
    if not callable(provider):
        return {name: "replicated" for name, _ in model.named_parameters()}
    placements = {str(name): str(value) for name, value in provider().items()}
    names = {name for name, _ in model.named_parameters()}
    if names != set(placements):
        missing = sorted(names - set(placements))
        extra = sorted(set(placements) - names)
        raise RuntimeError(
            f"Parameter placement inventory mismatch; missing={missing[:4]} extra={extra[:4]}"
        )
    valid = {
        "replicated",
        "expert_sharded",
        "sparse_table",
        "row_sharded_table",
    }
    unknown = sorted(set(placements.values()) - valid)
    if unknown:
        raise RuntimeError(f"Unknown parameter placement tags: {unknown}")
    return placements


def build_training_optimizers(
    model: nn.Module,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    sparse_learning_rate_scale: float,
    muon_beta: float,
    muon_ns_steps: int,
    muon_nesterov: bool,
    include_routed_experts: bool,
) -> tuple[OptimizerBundle, TrainingOptimizerSummary]:
    placements = _parameter_placements(model)
    named = list(model.named_parameters())
    sparse_parameters = [
        parameter
        for name, parameter in named
        if placements[name] in {"sparse_table", "row_sharded_table"} and parameter.requires_grad
    ]
    sparse_ids = {id(parameter) for parameter in sparse_parameters}
    dense_parameters = [
        parameter for _name, parameter in named if parameter.requires_grad and id(parameter) not in sparse_ids
    ]
    if not dense_parameters:
        raise RuntimeError("Model exposes no trainable dense parameters")

    # Reuse the repository's audited AdaMuon implementation while keeping
    # row-sparse embeddings out of its dense-gradient groups.
    previous: dict[int, bool] = {}
    for parameter in sparse_parameters:
        previous[id(parameter)] = parameter.requires_grad
        parameter.requires_grad_(False)
    try:
        dense_optimizer, dense_summary = build_muon_adamw_optimizer(
            model,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            muon_beta=muon_beta,
            muon_ns_steps=muon_ns_steps,
            muon_nesterov=muon_nesterov,
            include_routed_experts=include_routed_experts,
            adamw_impl="loop",
            master_weights=True,
        )
    finally:
        for parameter in sparse_parameters:
            parameter.requires_grad_(previous[id(parameter)])

    dense_ids = {
        id(parameter)
        for group in dense_optimizer.param_groups
        for parameter in group["params"]
    }
    expected_dense_ids = {id(parameter) for parameter in dense_parameters}
    if dense_ids != expected_dense_ids:
        raise RuntimeError("Dense optimizer omitted or duplicated trainable parameters")
    for group in dense_optimizer.param_groups:
        group["_metis_lr_scale"] = float(group["lr"]) / learning_rate

    sparse_optimizer = (
        FP32MasterSparseAdam(
            sparse_parameters,
            lr=learning_rate * sparse_learning_rate_scale,
            betas=(beta1, beta2),
            eps=eps,
        )
        if sparse_parameters
        else None
    )
    if sparse_optimizer is not None:
        for group in sparse_optimizer.param_groups:
            group["_metis_lr_scale"] = sparse_learning_rate_scale
    summary = TrainingOptimizerSummary(
        dense=dense_summary.to_dict()
        if isinstance(dense_summary, OptimizerBuildSummary)
        else {},
        sparse_tensor_count=len(sparse_parameters),
        sparse_parameter_count=sum(parameter.numel() for parameter in sparse_parameters),
        fp32_master_weights=True,
        fp32_optimizer_states=True,
    )
    return OptimizerBundle(dense_optimizer, sparse_optimizer), summary


def grad_norm(
    model: nn.Module,
    *,
    norm_type: float = 2.0,
    topology: ParallelTopology | None = None,
) -> torch.Tensor:
    if topology is not None and norm_type != 2.0:
        raise ValueError("Distributed Metis gradient norms support only L2")
    if topology is not None:
        return _placement_aware_global_l2_grad_norm(model, topology)

    contributions: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
        contributions.append(torch.linalg.vector_norm(gradient.float(), ord=norm_type))
    if not contributions:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, dtype=torch.float32)
    stacked = torch.stack(contributions)
    return torch.linalg.vector_norm(stacked, ord=norm_type)


def _placement_aware_global_l2_grad_norm(
    model: nn.Module,
    topology: ParallelTopology,
) -> torch.Tensor:
    """Return the logical model's L2 gradient norm on every rank.

    Replicated tensors occur on every world rank, while expert and row-sharded
    tensors occur once per expert replica. Weighting each local squared norm by
    the corresponding replica count before a world reduction counts every
    logical parameter exactly once. The resulting scalar, and therefore the
    clipping coefficient derived from it, is identical on every rank.
    """

    if topology.world_size <= 0 or topology.expert_replica_count <= 0:
        raise RuntimeError("Gradient norm topology has an invalid replica count")
    if topology.world_size != (
        topology.expert_parallel_size * topology.expert_replica_count
    ):
        raise RuntimeError("Gradient norm topology does not cover the world")

    placements = _parameter_placements(model)
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise RuntimeError("Cannot compute a gradient norm for a parameterless model")
    local_squared = torch.zeros(
        (),
        device=first_parameter.device,
        dtype=torch.float64,
    )
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = (
            parameter.grad.coalesce().values()
            if parameter.grad.is_sparse
            else parameter.grad
        )
        # Compute each tensor norm in FP32 without materializing an FP32 copy of
        # the full gradient, then retain FP64 precision in the scalar sum.
        tensor_norm = torch.linalg.vector_norm(
            gradient,
            ord=2,
            dtype=torch.float32,
        )
        replica_count = (
            topology.world_size
            if placements[name] in {"replicated", "sparse_table"}
            else topology.expert_replica_count
        )
        local_squared.add_(
            tensor_norm.to(dtype=torch.float64).square(),
            alpha=1.0 / float(replica_count),
        )

    if topology.distributed:
        dist.all_reduce(
            local_squared,
            op=dist.ReduceOp.SUM,
            group=topology.dense_data_group,
        )
    return local_squared.clamp_min_(0.0).sqrt_().to(dtype=torch.float32)


def clip_grad_norm_(
    model: nn.Module,
    max_norm: float,
    *,
    topology: ParallelTopology | None = None,
) -> torch.Tensor:
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    total = grad_norm(model, topology=topology)
    coefficient = float(max_norm) / (float(total.detach().item()) + 1.0e-6)
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                parameter.grad._values().mul_(coefficient)
            else:
                parameter.grad.mul_(coefficient)
    return total


def sampled_update_to_weight_ratio(
    before: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> float:
    numerator_sq = 0.0
    denominator_sq = 0.0
    for name, parameter in model.named_parameters():
        snapshot = before.get(name)
        if snapshot is None:
            continue
        current = parameter.detach().float()
        delta = current - snapshot.to(device=current.device)
        numerator_sq += float(delta.square().sum().item())
        denominator_sq += float(snapshot.float().square().sum().item())
    return math.sqrt(numerator_sq) / max(math.sqrt(denominator_sq), 1.0e-30)


def sample_parameters(
    model: nn.Module,
    *,
    maximum_elements: int = 4_000_000,
) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    remaining = maximum_elements
    for name, parameter in model.named_parameters():
        if remaining <= 0 or not parameter.requires_grad:
            break
        if parameter.numel() <= remaining and not parameter.is_meta:
            snapshots[name] = parameter.detach().float().clone()
            remaining -= parameter.numel()
    return snapshots
