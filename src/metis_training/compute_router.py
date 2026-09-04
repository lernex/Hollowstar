from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model_config import Metis16Config


def utility_router_parameter_count(config: Metis16Config) -> int:
    inputs = 3 * config.d_model + config.route_feature_dim + config.max_passes
    hidden = config.joint_router_hidden_dim
    future = config.max_passes - 1
    outputs = future * (1 + config.n_layers * config.max_routed_k)
    return inputs * hidden + hidden + hidden * outputs + outputs


@dataclass(frozen=True)
class JointComputeCosts:
    base_pass_costs: tuple[int, ...]
    expert_costs: tuple[int, ...]
    reference_per_token: int
    router_per_token: int

    @classmethod
    def from_config(cls, config: Metis16Config) -> JointComputeCosts:
        from .metrics import estimate_train_flops

        reference = replace(config, joint_compute_router=False)
        expert = 6 * 3 * config.latent_dim * config.expert_intermediate_dim
        previous = 0
        increments = []
        for depth in range(1, config.max_passes + 1):
            prefix = round(
                estimate_train_flops(
                    reference,
                    tokens=1,
                    observed_mean_passes=float(depth),
                    observed_mean_routed_k=1.0,
                )
                - depth * config.n_layers * expert
            )
            increments.append(prefix - previous)
            previous = prefix
        reference_cost = round(
            estimate_train_flops(
                reference,
                tokens=1,
                observed_mean_passes=config.target_mean_passes,
                observed_mean_routed_k=config.target_mean_routed_k,
            )
        )
        if min(increments) <= 0 or reference_cost <= 0:
            raise ValueError("Joint routing requires positive audited compute costs.")
        return cls(
            tuple(increments),
            (expert,) * config.n_layers,
            reference_cost,
            6 * utility_router_parameter_count(config),
        )

    def pass_cost(self, pass_index: int, widths: Tensor, active_mask: Tensor) -> Tensor:
        if not 0 <= pass_index < len(self.base_pass_costs):
            raise ValueError("Pass index is outside the compute ledger.")
        if widths.ndim != 3 or widths.shape[1:] != active_mask.shape:
            raise ValueError("Pass widths must have shape [layers, batch, sequence].")
        if widths.shape[0] != len(self.expert_costs):
            raise ValueError("Pass width layer count differs from the compute ledger.")
        if widths.dtype not in {torch.int32, torch.int64} or active_mask.dtype != torch.bool:
            raise ValueError("Compute accounting requires integer widths and a boolean mask.")
        experts = torch.tensor(
            self.expert_costs, device=widths.device, dtype=torch.int64
        )
        return (
            active_mask.sum(dtype=torch.int64) * self.base_pass_costs[pass_index]
            + (
                widths.to(torch.int64)
                * active_mask.unsqueeze(0)
                * experts[:, None, None]
            ).sum()
        )


@dataclass(frozen=True)
class JointUtilityPrediction:
    depth_utilities: Tensor
    width_utilities: Tensor
    active_mask: Tensor
    origin_pass: int

    def value_of(self, width_history: Sequence[Tensor]) -> Tensor:
        depth = len(width_history)
        if not 1 <= depth < self.depth_utilities.shape[-1]:
            raise ValueError("Observed trajectory is outside the prediction horizon.")
        value = self.depth_utilities[..., depth]
        for offset, widths in enumerate(width_history):
            if widths.shape != (
                self.width_utilities.shape[-2], *self.active_mask.shape
            ):
                raise ValueError("Observed widths do not match the utility prediction.")
            if widths.dtype not in {torch.int32, torch.int64}:
                raise ValueError("Observed widths must be integer decisions.")
            if bool(((widths < 0) | (widths > self.width_utilities.shape[-1])).any().item()):
                raise ValueError("Observed width is outside the prediction support.")
            for layer in range(widths.shape[0]):
                indices = widths[layer].long().clamp_min(1) - 1
                value = value + self.width_utilities[..., offset, layer, :].gather(
                    -1, indices.unsqueeze(-1)
                ).squeeze(-1)
        return value

    def observed_loss(
        self,
        width_history: Sequence[Tensor],
        improvement: Tensor,
        observed_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if improvement.shape != self.active_mask.shape:
            raise ValueError("Utility targets must have the original token shape.")
        if observed_mask.shape != self.active_mask.shape:
            raise ValueError("Utility observation mask has the wrong shape.")
        if observed_mask.dtype != torch.bool:
            raise ValueError("Utility observation mask must be boolean.")
        if bool((observed_mask & ~self.active_mask).any().item()):
            raise ValueError("A utility target cannot precede its active context.")
        value = self.value_of(width_history)
        count = observed_mask.sum()
        if not bool(count.item()):
            return value.sum() * 0.0, count
        if not bool(torch.isfinite(improvement.masked_select(observed_mask)).all().item()):
            raise ValueError("Observed utility targets must be finite.")
        # Allocation needs expected loss reduction, so fit its conditional
        # mean; clipping this head separately protects the backbone.
        return F.mse_loss(
            value.masked_select(observed_mask),
            improvement.detach().float().masked_select(observed_mask),
            reduction="sum",
        ), count


@dataclass
class JointRouterObservation:
    prediction: JointUtilityPrediction
    stop_losses: Tensor
    width_history: list[Tensor]


class JointComputeRouter(nn.Module):
    """Predict the value of feasible trajectories, not confidence in a quota.

    Targets are observed changes in next-token loss. Unvisited continuations
    have no target: stopping must not manufacture a zero-cost future reward.
    The hard allocator is separate from this outcome regression.
    """

    def __init__(
        self,
        config: Metis16Config,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        inputs = 3 * config.d_model + config.route_feature_dim + config.max_passes
        future = config.max_passes - 1
        outputs = future * (1 + config.n_layers * config.max_routed_k)
        self.hidden = nn.Linear(
            inputs, config.joint_router_hidden_dim, device=device, dtype=torch.float32
        )
        self.output = nn.Linear(
            config.joint_router_hidden_dim, outputs, device=device, dtype=torch.float32
        )
        self.hidden.metis_precision_role = "router_logits"
        self.output.metis_precision_role = "router_logits"
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("trained_updates", torch.zeros((), dtype=torch.int64, device=device))
        self.costs = JointComputeCosts.from_config(config)

    @torch.no_grad()
    def mark_trained(self) -> None:
        self.trained_updates.add_(1)

    def forward(
        self,
        state: Tensor,
        memory: Tensor,
        difference: Tensor,
        route_history: Tensor,
        *,
        active_mask: Tensor,
        origin_pass: int,
        remaining_passes: int,
    ) -> JointUtilityPrediction:
        if not 0 <= origin_pass < self.config.max_passes - 1:
            raise ValueError("Utility context is outside the recurrent horizon.")
        if not 1 <= remaining_passes <= self.config.max_passes - origin_pass - 1:
            raise ValueError("Invalid remaining utility horizon.")
        if state.shape[:-1] != active_mask.shape:
            raise ValueError("Utility context must have the original token shape.")
        positions = torch.nonzero(active_mask.reshape(-1), as_tuple=False).flatten()
        pieces = [
            value.detach().reshape(-1, value.shape[-1]).index_select(0, positions).float()
            for value in (state, memory, difference, route_history)
        ]
        pass_ids = torch.full(
            (positions.numel(),), origin_pass, device=state.device, dtype=torch.long
        )
        pieces.append(F.one_hot(pass_ids, self.config.max_passes).float())
        features = torch.cat(pieces, dim=-1)
        features = F.layer_norm(features, (features.shape[-1],))
        with torch.autocast(device_type=features.device.type, enabled=False):
            raw = self.output(F.silu(self.hidden(features.float())))
        future = self.config.max_passes - 1
        depth = raw[:, :future]
        width = raw[:, future:].reshape(
            -1, future, self.config.n_layers, self.config.max_routed_k
        )
        # A fixed-k reference identifies the depth term separately from the
        # additive width adjustments; otherwise those intercepts can drift.
        reference_k = round(self.config.target_mean_routed_k) - 1
        width = width - width[..., reference_k : reference_k + 1]
        flat_depth = raw.new_zeros(active_mask.numel(), remaining_passes)
        flat_width = raw.new_zeros(
            active_mask.numel(),
            remaining_passes,
            self.config.n_layers,
            self.config.max_routed_k,
        )
        flat_depth = flat_depth.index_copy(0, positions, depth[:, :remaining_passes])
        flat_width = flat_width.index_copy(0, positions, width[:, :remaining_passes])
        depth_values = flat_depth.reshape(*active_mask.shape, remaining_passes)
        depth_values = torch.cat(
            (depth_values.new_zeros(*active_mask.shape, 1), depth_values), dim=-1
        )
        return JointUtilityPrediction(
            depth_values,
            flat_width.reshape(
                *active_mask.shape,
                remaining_passes,
                self.config.n_layers,
                self.config.max_routed_k,
            ),
            active_mask,
            origin_pass,
        )

    @staticmethod
    def allocation_utilities(
        prediction: JointUtilityPrediction,
        *,
        exploration: float,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("Utility exploration must lie in [0, 1] loss units.")
        depth = prediction.depth_utilities.detach()
        width = prediction.width_utilities.detach()
        if exploration:
            depth_noise = torch.rand(
                depth.shape, device=depth.device, dtype=depth.dtype, generator=generator
            )
            width_noise = torch.rand(
                width.shape, device=width.device, dtype=width.dtype, generator=generator
            )
            depth = depth + (depth_noise * 2.0 - 1.0) * exploration
            depth[..., 0] = 0.0
            width = width + (width_noise * 2.0 - 1.0) * exploration
        return depth, width
