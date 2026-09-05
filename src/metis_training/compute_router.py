from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, log, prod
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
    outputs = future * (1 + config.n_layers * config.max_routed_k) + int(config.terminal_action_critic)
    return inputs * hidden + hidden + hidden * outputs + outputs


@dataclass(frozen=True)
class JointComputeCosts:
    base_pass_costs: tuple[int, ...]
    expert_costs: tuple[int, ...]
    reference_per_token: int
    router_per_token: int
    removed_policy_per_pass: int = 0
    head_per_token: int = 0
    terminal_head_only: bool = False
    metadata_transition_flops: int = 0

    @classmethod
    def from_config(cls, config: Metis16Config) -> JointComputeCosts:
        from .metrics import estimate_train_flops

        reference = replace(
            config, joint_compute_router=False, causal_compute_budget=False,
            terminal_action_critic=False, terminal_reference_bootstrap_steps=0,
            causal_memory_metadata="disabled", causal_min_passes=1,
        )
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
        removed_policy = 0
        metadata_transition = 0
        head = 6 * config.vocab_size * config.d_model
        if config.causal_compute_budget:
            continuation = (
                (3 * config.d_model + config.route_feature_dim) * config.route_feature_dim
                + 2 * config.route_feature_dim + 1
            )
            width_heads = config.n_layers * (
                config.latent_dim + config.route_feature_dim + 1
            ) * (config.max_routed_k - config.min_routed_k + 1)
            removed_policy = 6 * (continuation + width_heads)
            increments = [cost - removed_policy for cost in increments]
            # A lean fixed two-pass control needs no policy predictions and
            # projects to vocabulary only at its terminal exit. Outcome
            # training still pays for every head it actually requests.
            reference_cost -= 2 * removed_policy + head
            if config.causal_memory_metadata == "legacy_confidence":
                route_projection = (
                    (3 * config.d_model + config.route_feature_dim) * config.route_feature_dim
                    + config.route_feature_dim
                )
                # This is a detached compatibility feature, not a policy
                # trained through the memory path: price forward work only.
                metadata_transition = 2 * (continuation + route_projection)
                increments = [
                    cost + (metadata_transition if index > 0 else 0)
                    for index, cost in enumerate(increments)
                ]
        if config.terminal_action_critic:
            # This objective observes only terminal CE. Charge its one head
            # separately, before admitting any remaining trajectory.
            increments = [cost - head for cost in increments]
        if min(increments) <= 0 or reference_cost <= 0:
            raise ValueError("Joint routing requires positive audited compute costs.")
        return cls(
            tuple(increments),
            (expert,) * config.n_layers,
            reference_cost,
            6 * utility_router_parameter_count(config),
            removed_policy,
            head,
            config.terminal_action_critic,
            metadata_transition,
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
    terminal_values: bool = False

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
        if self.terminal_values:
            raise ValueError("Terminal Q values require terminal CE targets, not observed improvements.")
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

    def value_of_actions(self, depths: Tensor, routed_k: Tensor) -> Tensor:
        rounds, layers, choices = self.width_utilities.shape[-3:]
        if depths.shape != self.active_mask.shape or depths.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Action depths must be integer [batch, sequence].")
        if routed_k.shape != (rounds, layers, *self.active_mask.shape):
            raise ValueError("Action widths must be [rounds, layers, batch, sequence].")
        if routed_k.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Action widths must be integers.")
        if bool(((depths < 0) | (depths > rounds) | (~self.active_mask & depths.ne(0))).any()):
            raise ValueError("Action depth is outside the active prediction horizon.")
        live = torch.arange(rounds, device=depths.device)[:, None, None, None] < depths[None, None]
        if bool(torch.where(live, (routed_k < 1) | (routed_k > choices), routed_k.ne(0)).any()):
            raise ValueError("Action widths must match actually executed depth prefixes.")
        value = self.depth_utilities.gather(-1, depths.long().unsqueeze(-1)).squeeze(-1)
        for r in range(rounds):
            for layer in range(layers):
                indices = routed_k[r, layer].long().clamp_min(1) - 1
                width_value = self.width_utilities[..., r, layer, :].gather(
                    -1, indices.unsqueeze(-1)
                ).squeeze(-1)
                value = value + torch.where(depths > r, width_value, 0.0)
        return value

    def terminal_loss(
        self,
        depths: Tensor,
        routed_k: Tensor,
        terminal_ce: Tensor,
        observed_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.terminal_values:
            raise ValueError("Terminal CE targets require a terminal-action critic.")
        if terminal_ce.shape != self.active_mask.shape:
            raise ValueError("Terminal CE must have the original token shape.")
        if observed_mask.shape != self.active_mask.shape or observed_mask.dtype != torch.bool:
            raise ValueError("Terminal observations must be bool [batch, sequence].")
        if bool((observed_mask & ~self.active_mask).any()):
            raise ValueError("Terminal observations cannot precede their prediction context.")
        targets = terminal_ce.detach().float().masked_select(observed_mask)
        if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
            raise ValueError("Observed terminal CE must be finite and nonnegative.")
        value = self.value_of_actions(depths, routed_k)
        count = observed_mask.sum()
        if not bool(count):
            return value.sum() * 0.0, count
        return F.mse_loss(value.masked_select(observed_mask), -targets, reduction="sum"), count


@dataclass
class JointRouterObservation:
    prediction: JointUtilityPrediction
    stop_losses: Tensor
    width_history: list[Tensor]


@dataclass
class TerminalRouterObservation:
    prediction: JointUtilityPrediction
    depths: Tensor
    routed_k: Tensor
    terminal_ce: Tensor
    observed_mask: Tensor


class JointComputeRouter(nn.Module):
    """Predict the value of feasible trajectories, not confidence in a quota.

    The default target is observed loss improvement. The separately identified
    terminal-action candidate fits negative terminal CE, with a shared halt
    value and action advantages. Neither objective labels unvisited actions.
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
        outputs = future * (1 + config.n_layers * config.max_routed_k) + int(config.terminal_action_critic)
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
        if config.terminal_reference_bootstrap_steps:
            # A fixed pre-data scale prior avoids treating zero loss as the
            # initial value of every action. It cannot reveal future labels.
            nn.init.constant_(self.output.bias[-1:], -log(config.vocab_size))
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
        minimum_horizon = 0 if self.config.terminal_action_critic else 1
        if not minimum_horizon <= remaining_passes <= self.config.max_passes - origin_pass - 1:
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
        width = raw[:, future: future + future * self.config.n_layers * self.config.max_routed_k].reshape(
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
        if self.config.terminal_action_critic:
            state_value = raw.new_zeros(active_mask.numel()).index_copy(0, positions, raw[:, -1])
            depth_values = depth_values + state_value.reshape(*active_mask.shape, 1)
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
            self.config.terminal_action_critic,
        )

    @staticmethod
    def allocation_utilities(
        prediction: JointUtilityPrediction,
        *,
        exploration: float,
        generator: torch.Generator | None,
        causal_keys: Tensor | None = None,
        causal_seed: int = 0,
        exploration_price_margin: float = 1.0,
        reference_bootstrap: bool = False,
        reference_routed_k: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("Utility exploration must lie in [0, 1] loss units.")
        depth = prediction.depth_utilities.detach()
        width = prediction.width_utilities.detach()
        if reference_bootstrap:
            if not prediction.terminal_values:
                raise ValueError("Reference bootstrap requires terminal Q values.")
            if (
                isinstance(reference_routed_k, bool)
                or not isinstance(reference_routed_k, int)
                or not 1 <= reference_routed_k <= width.shape[-1]
            ):
                raise ValueError("Reference bootstrap requires a supported integer routed width.")
            advantage = 2.0 * exploration_price_margin
            if not isfinite(advantage) or not 0 < advantage <= torch.finfo(depth.dtype).max:
                raise ValueError("Reference bootstrap price margin must be finite and positive.")
            if depth.shape[-1] > 1:
                # These are temporary behavior scores, never regression
                # targets. Admission may reduce width or halt to pay the
                # controller; it does not enforce two independent marginals.
                depth = depth[..., :1].expand_as(depth).clone()
                depth[..., 1] += advantage
                choices = torch.arange(1, width.shape[-1] + 1, device=width.device)
                penalty = advantage / (
                    2 * width.shape[-2] * max(1, reference_routed_k - 1)
                )
                width = (
                    -penalty * (choices - reference_routed_k).abs()
                ).to(width.dtype).view(*([1] * (width.ndim - 1)), -1).expand_as(width)
        if exploration:
            if prediction.terminal_values and causal_keys is None:
                raise ValueError("Terminal-action exploration requires causal token keys.")
            if causal_keys is None:
                depth_noise = torch.rand(
                    depth.shape, device=depth.device, dtype=depth.dtype, generator=generator
                )
                width_noise = torch.rand(
                    width.shape, device=width.device, dtype=width.dtype, generator=generator
                )
            else:
                if causal_keys.shape != prediction.active_mask.shape or causal_keys.dtype != torch.int64:
                    raise ValueError("Causal exploration keys must be int64 [batch, sequence].")

                def noise(values: Tensor, stream: int) -> Tensor:
                    # Stateless common random numbers depend on a token's own
                    # prefix position, never its row index or the batch shape.
                    prime = 2_147_483_647
                    coordinates = torch.arange(
                        prod(values.shape[2:]), device=values.device, dtype=torch.int64
                    ).reshape(*([1] * 2), *values.shape[2:])
                    keys = causal_keys.reshape(*causal_keys.shape, *([1] * (values.ndim - 2)))
                    mixed = (keys.remainder(prime) + coordinates * 1_000_003 + causal_seed % prime + stream) % prime
                    mixed = (mixed * mixed + 48_271 * mixed + 12_820_163) % prime
                    mixed = (mixed * mixed + 69_621 * mixed + 9_173) % prime
                    return (mixed.double() / prime).to(values.dtype)

                depth_noise = noise(depth, 17)
                width_noise = noise(width, 104729)
            if prediction.terminal_values:
                if not torch.isfinite(depth.new_tensor(exploration_price_margin)) or exploration_price_margin <= 0:
                    raise ValueError("Exploration price margin must be finite and positive.")
                if depth.shape[-1] == 1:
                    return depth, width
                explore = (depth_noise[..., 0] < exploration) | (exploration == 1.0)
                selected_depth = (depth_noise[..., -1] * depth.shape[-1]).long().clamp_max(depth.shape[-1] - 1)
                selected_width = (width_noise[..., 0] * width.shape[-1]).long().clamp_max(width.shape[-1] - 1)
                depth_choice = torch.arange(depth.shape[-1], device=depth.device)
                width_choice = torch.arange(width.shape[-1], device=width.device)
                random_depth = depth[..., :1] + (
                    depth_choice == selected_depth[..., None]
                ).to(depth.dtype) * exploration_price_margin
                random_width = -(
                    width_choice != selected_width[..., None]
                ).to(width.dtype) * exploration_price_margin
                return (
                    torch.where(explore[..., None], random_depth, depth),
                    torch.where(explore[..., None, None, None], random_width, width),
                )
            halt_value = depth[..., 0]
            depth = depth + (depth_noise * 2.0 - 1.0) * exploration
            depth[..., 0] = halt_value
            width = width + (width_noise * 2.0 - 1.0) * exploration
        return depth, width
