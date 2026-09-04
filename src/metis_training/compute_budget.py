"""Hard joint depth/width allocation under one integer modeled-compute cap.

The price search and discrete repair are a heuristic, not an exact knapsack
solver. The dual certificate bounds the supplied learned-utility objective,
not language-model quality. There is deliberately no gradient through choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class JointBudgetPlan:
    depths: Tensor
    routed_k: Tensor
    cost: Tensor
    utility: Tensor
    budget: int | Tensor
    dual_upper_bound: Tensor | None = None
    optimality_gap: Tensor | None = None

    @property
    def unused_budget(self) -> Tensor:
        return torch.as_tensor(self.budget, dtype=torch.int64, device=self.cost.device) - self.cost


@dataclass
class _Choice:
    depths: Tensor
    widths: Tensor
    costs: Tensor


def _integers(value: Sequence[int] | Tensor, length: int, name: str) -> list[int]:
    if isinstance(value, Tensor):
        if value.ndim != 1 or value.numel() != length:
            raise ValueError(f"{name} must have shape [{length}]")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"{name} must contain integers")
        values = value.detach().cpu().tolist()
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer sequence or tensor") from exc
        if len(values) != length:
            raise ValueError(f"{name} must have length {length}")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        raise TypeError(f"{name} must contain integers")
    if any(x <= 0 for x in values):
        raise ValueError(f"{name} must contain positive costs")
    return [int(x) for x in values]


def _up(value: Tensor) -> Tensor:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _down(value: Tensor) -> Tensor:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _sum_upper(values: Tensor) -> Tensor:
    """Directed pairwise addition avoids a reduction-order-dependent certificate."""
    values = values.reshape(-1)
    if not values.numel():
        return values.new_zeros(())
    while values.numel() > 1:
        pairs = values.numel() // 2
        reduced = _up(values[: 2 * pairs : 2] + values[1 : 2 * pairs : 2])
        values = torch.cat((reduced, values[-1:])) if values.numel() % 2 else reduced
    return values[0]


def _dual_bound(
    depth: Tensor,
    width: Tensor,
    active: Tensor,
    base: Tensor,
    experts: Tensor,
    budget: int,
    price: float,
) -> Tensor:
    """Upper-round every operation, including integer-to-float cost conversion."""
    _, rounds, layers, choices = width.shape
    k = torch.arange(1, choices + 1, device=width.device, dtype=torch.int64)
    width_penalty = _down(_down((experts[:, None] * k).double()) * price)
    base_penalty = _down(_down(base.double()) * price)
    best_width = _up(width - width_penalty[None, None]).amax(dim=-1)
    prefix = depth.new_zeros(depth.shape[0])
    best = depth[:, 0]
    for r in range(rounds):
        for layer in range(layers):
            prefix = _up(prefix + best_width[:, r, layer])
        prefix = _up(prefix - base_penalty[r])
        best = torch.maximum(best, _up(depth[:, r + 1] + prefix))
    total = _sum_upper(torch.where(active, best, 0.0))
    budget_float = depth.new_tensor(budget)
    return _up(total + _up(_up(budget_float) * price))


def _repair(
    under: _Choice,
    over: _Choice,
    free: _Choice,
    depth: Tensor,
    width: Tensor,
    active: Tensor,
    base: Tensor,
    experts: Tensor,
    budget: int,
) -> _Choice:
    """Rank one upgrade per token; apply an exactly affordable global prefix.

    Include unsupported depth/width choices that a Lagrange search can skip.
    Uniform-width policies and individual layer changes are a small candidate
    set, not enumeration of every possible width configuration.
    """
    n, rounds, layers, choices = width.shape
    device = width.device
    rows = torch.arange(n, device=device)
    pass_index = torch.arange(rounds, device=device)
    uniforms = torch.arange(1, choices + 1, device=device).view(1, choices, 1, 1)
    policies = torch.cat(
        (
            torch.stack((under.widths, over.widths, free.widths), dim=1),
            uniforms.expand(n, choices, rounds, layers),
        ),
        dim=1,
    )
    policy_values = torch.gather(
        width[:, None].expand(-1, policies.shape[1], -1, -1, -1),
        -1,
        (policies - 1).unsqueeze(-1),
    ).squeeze(-1)
    policy_costs = (policies * experts).sum(dim=-1) + base
    zeros = torch.zeros((n, policies.shape[1], 1), dtype=torch.int64, device=device)
    costs = torch.cat((zeros, policy_costs.cumsum(dim=-1)), dim=-1)
    values = torch.cat(
        (zeros.double(), policy_values.sum(dim=-1).cumsum(dim=-1)), dim=-1
    ) + depth[:, None]
    costs = (costs * active[:, None, None]).flatten(1)
    values = torch.where(active[:, None, None], values, 0.0).flatten(1)
    full_count = costs.shape[1]

    original_cost = under.costs
    original_value = values[rows, under.depths]
    original_width_value = width.gather(-1, (under.widths - 1).unsqueeze(-1)).squeeze(-1)
    live = (pass_index[None, :, None] < under.depths[:, None, None]) & active[:, None, None]
    alternatives = torch.arange(1, choices + 1, device=device)
    delta_cost = (alternatives - under.widths[..., None]) * experts[None, None, :, None]
    delta_value = width - original_width_value[..., None]
    coordinate_costs = original_cost[:, None, None, None] + torch.where(
        live[..., None], delta_cost, 0
    )
    coordinate_values = original_value[:, None, None, None] + torch.where(
        live[..., None], delta_value, 0.0
    )
    costs = torch.cat((costs, coordinate_costs.flatten(1)), dim=1)
    values = torch.cat((values, coordinate_values.flatten(1)), dim=1)

    # First retain improvements that do not consume more budget. Ties prefer
    # lower exact cost, rather than gratuitous work at a zero price.
    eligible = (costs <= original_cost[:, None]) & (values >= original_value[:, None])
    best_value = values.masked_fill(~eligible, -math.inf).amax(dim=1, keepdim=True)
    cheapest = costs.masked_fill(~eligible | (values != best_value), torch.iinfo(torch.int64).max)
    selected = cheapest.argmin(dim=1)
    current_cost = costs[rows, selected]
    current_value = values[rows, selected]
    remaining = budget - current_cost.sum()

    gains = values - current_value[:, None]
    increments = costs - current_cost[:, None]
    eligible = (gains > 0) & (increments > 0) & (increments <= remaining)
    density = (gains / increments.clamp_min(1).double()).masked_fill(~eligible, -math.inf)
    best_density = density.amax(dim=1, keepdim=True)
    best_gain = gains.masked_fill(
        ~eligible | (density != best_density), -math.inf
    ).amax(dim=1, keepdim=True)
    cheapest = increments.masked_fill(
        ~eligible | (density != best_density) | (gains != best_gain),
        torch.iinfo(torch.int64).max,
    )
    upgrade = cheapest.argmin(dim=1)
    can_upgrade = eligible[rows, upgrade]
    upgrade_cost = torch.where(can_upgrade, increments[rows, upgrade], 0)
    upgrade_density = density[rows, upgrade]
    # Stable sorts resolve density ties by cost, then flattened token position.
    order = torch.argsort(upgrade_cost, stable=True)
    order = order[torch.argsort(upgrade_density[order], descending=True, stable=True)]
    affordable = upgrade_cost[order].cumsum(dim=0) <= remaining
    take = torch.zeros(n, dtype=torch.bool, device=device)
    take[order] = affordable & can_upgrade[order]
    selected = torch.where(take, upgrade, selected)

    full = selected < full_count
    policy = (selected // (rounds + 1)).clamp_max(policies.shape[1] - 1)
    selected_widths = policies[rows, policy]
    coordinate = (selected - full_count).clamp_min(0)
    coordinate_widths = under.widths.reshape(n, rounds * layers).clone()
    coordinate_widths.scatter_(1, (coordinate // choices)[:, None], (coordinate % choices + 1)[:, None])
    selected_widths = torch.where(
        full[:, None, None], selected_widths, coordinate_widths.reshape(n, rounds, layers)
    )
    selected_depths = torch.where(full, selected % (rounds + 1), under.depths)
    selected_depths = torch.where(active, selected_depths, 0)
    return _Choice(selected_depths, selected_widths, costs[rows, selected])


@torch.no_grad()
def allocate_joint_budget(
    depth_utilities: Tensor,
    width_utilities: Tensor,
    active_mask: Tensor,
    *,
    base_pass_costs: Sequence[int] | Tensor,
    expert_costs: Sequence[int] | Tensor,
    budget: int | Tensor,
    iterations: int = 24,
) -> JointBudgetPlan:
    """Allocate remaining passes and routed experts against a single total cap.

    Shapes are depth ``[B,S,R+1]``, width ``[B,S,R,L,K]``, mask ``[B,S]``.
    Width utility is for choosing k=1..K, not a marginal increment. Depth zero
    contributes its supplied utility for an active token; padding contributes
    nothing. Outputs and certificates are detached. Utility/certificates use
    float64; hard costs use int64. CPU and CUDA tensors are supported.

    Cost inputs must be positive integers; budget is a nonnegative Python int
    or scalar int64 tensor. Overflow-risk instances are explicitly rejected.
    Repair leaves unused discrete budget; it never inserts nonpositive-value
    work to hit a quota. No separate depth or width marginal is constrained.
    """
    if not all(isinstance(t, Tensor) for t in (depth_utilities, width_utilities, active_mask)):
        raise TypeError("utilities and active_mask must be tensors")
    if depth_utilities.ndim != 3 or width_utilities.ndim != 5:
        raise ValueError("expected depth [B,S,R+1] and width [B,S,R,L,K]")
    batch, sequence, rounds, layers, choices = width_utilities.shape
    if depth_utilities.shape != (batch, sequence, rounds + 1):
        raise ValueError("depth and width shapes disagree")
    if active_mask.shape != (batch, sequence) or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be bool [B,S]")
    if layers < 1 or choices < 1:
        raise ValueError("at least one layer and one routed expert choice are required")
    device = depth_utilities.device
    if width_utilities.device != device or active_mask.device != device:
        raise ValueError("utilities and active_mask must be on the same device")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("joint allocation requires CPU or CUDA float64 diagnostics")
    if not depth_utilities.is_floating_point() or not width_utilities.is_floating_point():
        raise TypeError("utilities must be real floating-point tensors")
    if not bool(torch.isfinite(depth_utilities).all()) or not bool(torch.isfinite(width_utilities).all()):
        raise ValueError("utilities must be finite, including masked positions")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(budget, Tensor):
        if budget.ndim != 0 or budget.dtype != torch.int64:
            raise TypeError("budget tensor must be scalar int64")
        budget_value = int(budget.detach().cpu())
    elif isinstance(budget, Integral) and not isinstance(budget, bool):
        budget_value = int(budget)
    else:
        raise TypeError("budget must be an integer or scalar int64 tensor")
    integer_max = torch.iinfo(torch.int64).max
    if not 0 <= budget_value <= integer_max:
        raise ValueError("budget must be in the nonnegative int64 range")
    base_list = _integers(base_pass_costs, rounds, "base_pass_costs")
    expert_list = _integers(expert_costs, layers, "expert_costs")
    max_token_cost = sum(base_list) + rounds * choices * sum(expert_list)
    n = batch * sequence
    # Intermediate cumsums include inactive rows, so bound the full shape.
    if any(c > integer_max for c in base_list + expert_list) or max_token_cost * max(n, 1) > integer_max:
        raise OverflowError("integer cost accounting could overflow int64")
    base = torch.tensor(base_list, dtype=torch.int64, device=device)
    experts = torch.tensor(expert_list, dtype=torch.int64, device=device)
    depth = depth_utilities.detach().double().reshape(n, rounds + 1)
    width = width_utilities.detach().double().reshape(n, rounds, layers, choices)
    active = active_mask.detach().reshape(n)
    depth_scale = depth.abs().amax() if depth.numel() else depth.new_zeros(())
    width_scale = width.abs().amax() if width.numel() else depth.new_zeros(())
    scale = max(float(torch.maximum(depth_scale, width_scale)), torch.finfo(torch.float64).tiny)
    if scale > torch.finfo(torch.float64).max / (8 * max(n, 1) * (1 + rounds * layers)):
        raise OverflowError("utility accounting could overflow float64")

    if n == 0 or rounds == 0 or not bool(active.any()):
        depths = torch.zeros((batch, sequence), dtype=torch.int64, device=device)
        widths = torch.zeros((rounds, layers, batch, sequence), dtype=torch.int64, device=device)
        utility = torch.where(active, depth[:, 0], 0.0).sum()
        upper = _sum_upper(torch.where(active, depth[:, 0], 0.0)) if bool(active.any()) else utility
        return JointBudgetPlan(depths, widths, base.new_zeros(()), utility, budget_value, upper, (upper - utility).clamp_min(0))

    normalized_depth = (depth / scale).float()
    normalized_width = (width / scale).float()
    cost_scale = max_token_cost
    k = torch.arange(1, choices + 1, dtype=torch.int64, device=device)
    normalized_expert_cost = ((experts[:, None] * k).double() / cost_scale).float()
    normalized_base = (base.double() / cost_scale).float()
    positions = torch.arange(rounds, device=device)

    def choose(price: float) -> _Choice:
        scores = normalized_width - price * normalized_expert_cost[None, None]
        best_width, indices = scores.max(dim=-1)
        widths = indices + 1
        prefixes = (best_width.sum(dim=-1) - price * normalized_base).cumsum(dim=-1)
        scores_depth = normalized_depth + torch.cat((depth.new_zeros((n, 1)).float(), prefixes), dim=-1)
        depths = scores_depth.argmax(dim=-1)
        depths = torch.where(active, depths, 0)
        live = positions[None] < depths[:, None]
        costs = (((widths * experts).sum(dim=-1) + base) * live).sum(dim=-1)
        return _Choice(depths, widths, costs)

    free = choose(0.0)
    best = free
    low_price = high_price = 0.0
    if int(free.costs.sum()) > budget_value:
        low_price, high_price = 0.0, 1.0
        over = free
        under = choose(high_price)
        # The finite, normalized utilities make this bracket terminate.
        while int(under.costs.sum()) > budget_value:
            low_price, over = high_price, under
            high_price *= 2
            if not math.isfinite(high_price) or high_price > torch.finfo(torch.float32).max / 16:
                raise ArithmeticError("could not bracket a finite allocation price")
            under = choose(high_price)
        for _ in range(int(iterations)):
            middle = (low_price + high_price) / 2
            candidate = choose(middle)
            if int(candidate.costs.sum()) <= budget_value:
                high_price, under = middle, candidate
            else:
                low_price, over = middle, candidate
        best = _repair(under, over, free, depth, width, active, base, experts, budget_value)

    live = (positions[None, :, None] < best.depths[:, None, None]) & active[:, None, None]
    routed = torch.where(live, best.widths, 0)
    cost = ((routed * experts).sum(dim=-1) + base * live.squeeze(-1)).sum()
    chosen_width_utility = width.gather(-1, (best.widths - 1).unsqueeze(-1)).squeeze(-1)
    token_utility = depth.gather(-1, best.depths[:, None]).squeeze(-1)
    token_utility = token_utility + torch.where(live, chosen_width_utility, 0.0).sum(dim=(-1, -2))
    utility = torch.where(active, token_utility, 0.0).sum()
    if int(cost) > budget_value or int(cost) < 0:
        raise ArithmeticError("joint allocator violated its integer budget")
    if not bool(torch.isfinite(utility)):
        raise ArithmeticError("joint utility accounting is not finite")

    upper = _dual_bound(depth, width, active, base, experts, budget_value, 0.0)
    for normalized_price in sorted({low_price, high_price} - {0.0}):
        price = (normalized_price / cost_scale) * scale
        if math.isfinite(price):
            upper = torch.minimum(
                upper, _dual_bound(depth, width, active, base, experts, budget_value, price)
            )
    upper = torch.maximum(upper, utility)
    return JointBudgetPlan(
        depths=best.depths.reshape(batch, sequence),
        routed_k=routed.permute(1, 2, 0).reshape(rounds, layers, batch, sequence),
        cost=cost,
        utility=utility,
        budget=budget_value,
        dual_upper_bound=upper,
        optimality_gap=(upper - utility).clamp_min(0),
    )
