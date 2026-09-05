"""Frozen native-policy decision effects, including uncredited downstream loss.

Token losses stay in memory. Reports contain aggregate effects and plan hashes;
optional plan artifacts contain execution decisions, never corpus token IDs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from metis_training.compute_budget import _causal_admission
from metis_training.compute_router import JointComputeCosts, JointUtilityPrediction
from metis_training.data import ReleaseInventory, TrainingBatch
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config

from .routing_credit_probe import (
    CapabilityError,
    FrozenRuntimeState,
    ProbeForward,
    TokenPair,
    assert_file_unchanged,
    evaluate_in_memory,
    forward_summary,
    fresh_output_directory,
    held_out_batch,
    identify_file,
    infer_run_manifest,
    json_sha256,
    load_frozen_model,
    plan_cost,
    plan_fingerprint,
    repeated_plan_evaluation,
    sha256_file,
    source_identity,
    swap_plan,
    validate_checkpoint,
)


@dataclass(frozen=True)
class TerminalOutcomes:
    losses: Tensor
    values: Tensor
    prediction: JointUtilityPrediction


def terminal_outcomes(result: ProbeForward, batch: TrainingBatch) -> TerminalOutcomes:
    observations = result.output.terminal_router_observations
    if not observations:
        raise CapabilityError("Decision effects require captured terminal observations.")
    expected = batch.labels.ne(-100)
    counts = torch.zeros_like(batch.labels, dtype=torch.int64)
    losses = torch.zeros_like(batch.labels, dtype=torch.float32)
    values = torch.zeros_like(losses)
    prediction = observations[0].prediction
    for observation in observations:
        if observation.prediction is not prediction or prediction.origin_pass != 0:
            raise ValueError("Expected one shared causal prediction context after pass one.")
        mask = observation.observed_mask
        if mask.shape != expected.shape or mask.dtype != torch.bool:
            raise ValueError("Terminal observation mask has incompatible geometry.")
        if bool((mask & ~expected).any()):
            raise ValueError("An unsupervised token received a terminal observation.")
        horizon = prediction.width_utilities.shape[2]
        if not torch.equal(
            observation.depths[mask], (result.output.chosen_depths[mask] - 1),
        ) or not torch.equal(
            observation.routed_k[:, :, mask], result.widths[1:horizon + 1, :, mask],
        ):
            raise ValueError("Terminal observations describe a different executed action.")
        counts += mask
        losses += torch.where(mask, observation.terminal_ce, 0.0)
        predicted = prediction.value_of_actions(observation.depths, observation.routed_k)
        values += torch.where(mask, predicted, 0.0)
    if not torch.equal(counts, expected.long()):
        raise ValueError("Every supervised token must have exactly one terminal observation.")
    if not bool(torch.isfinite(losses[expected]).all() & torch.isfinite(values[expected]).all()):
        raise ValueError("Terminal losses and predictions must be finite.")
    if not torch.allclose(
        losses[expected].double().mean(), result.output.loss.detach().double(),
        rtol=1e-6, atol=1e-5,
    ):
        raise ValueError("Captured token losses do not reproduce the model's LM loss.")
    return TerminalOutcomes(losses.detach(), values.detach(), prediction)


def token_costs(
    config: Metis16Config, depths: Tensor, widths: Tensor, *, critic: bool,
) -> Tensor:
    if not config.causal_compute_budget or not config.terminal_action_critic:
        raise ValueError("Token costs require the causal, prepaid-terminal-head ledger.")
    plan_cost(config, depths, widths, terminal_only=True)
    ledger = JointComputeCosts.from_config(config)
    active = depths.gt(0)
    bases = torch.tensor(ledger.base_pass_costs, device=depths.device, dtype=torch.int64)
    experts = torch.tensor(ledger.expert_costs, device=depths.device, dtype=torch.int64)
    passes = torch.arange(config.max_passes, device=depths.device)[:, None, None]
    cost = ((passes < depths) * bases[:, None, None]).sum(dim=0)
    cost += (widths * experts[None, :, None, None]).sum(dim=(0, 1))
    cost += active * (ledger.head_per_token + (ledger.router_per_token if critic else 0))
    return cost


def budget_summary(
    config: Metis16Config, depths: Tensor, widths: Tensor, *, critic: bool,
    horizon: int,
) -> dict[str, Any]:
    costs = token_costs(config, depths, widths, critic=critic)
    active = depths.gt(0)
    reference = JointComputeCosts.from_config(config).reference_per_token
    balance = active.long().cumsum(dim=1) * reference - costs.cumsum(dim=1)
    floor_ok = not bool((depths[active] < config.causal_min_passes).any())
    horizon_ok = not bool((depths[active] > horizon).any())
    first_ok = bool(widths[0, :, active].eq(round(config.target_mean_routed_k)).all())
    prefix_ok = not bool(balance.lt(0).any())
    return {
        "execution_train_flops": int(costs.sum()),
        "reference_train_flops": int(active.sum()) * reference,
        "critic_included_in_policy_cost": critic,
        "within_every_prefix_budget": prefix_ok,
        "minimum_prefix_slack": int(balance.min()) if balance.numel() else 0,
        "within_declared_policy_support": floor_ok and horizon_ok and first_ok,
        "policy_feasible": prefix_ok and floor_ok and horizon_ok and first_ok,
    }


@dataclass(frozen=True)
class Intervention:
    kind: str
    trial: int
    depths: Tensor
    widths: Tensor
    changed: Tensor


def interventions(
    config: Metis16Config, batch: TrainingBatch, depths: Tensor, widths: Tensor, *,
    trials: int, seed: int, horizon: int, min_context: int, critic: bool,
) -> tuple[list[Intervention], dict[str, Any]]:
    if type(trials) is not int or trials < 1 or type(min_context) is not int or min_context < 0:
        raise ValueError("Require positive trials and nonnegative minimum context.")
    d, w = depths.detach().cpu(), widths.detach().cpu()
    mask = batch.attention_mask.detach().cpu()
    supervised = batch.labels.detach().cpu().ne(-100)
    docs = batch.document_ids.detach().cpu()
    ledger = JointComputeCosts.from_config(config)
    base_cost = token_costs(config, d, w, critic=critic)
    slack = mask.long().cumsum(dim=1) * ledger.reference_per_token - base_cost.cumsum(dim=1)
    base_feasible = budget_summary(config, d, w, critic=critic, horizon=horizon)["policy_feasible"]
    eligible: list[list[list[int]]] = []
    for row in range(d.shape[0]):
        groups = []
        for document in docs[row, mask[row]].unique().tolist():
            positions = (mask[row] & docs[row].eq(document)).nonzero().flatten().tolist()
            candidates = [
                position for index, position in enumerate(positions)
                if index >= min_context and len(positions) - index - 1 >= min_context
                and bool(supervised[row, position])
            ]
            if candidates:
                groups.append(candidates)
        eligible.append(groups)
    kinds = ("shorten", "narrow_early", "narrow_terminal", "exchange", "transfer_depth")
    records, coverage, seen = [], {}, set()
    for kind_index, kind in enumerate(kinds):
        generated = changed_tokens = 0
        for trial in range(trials):
            generator = torch.Generator().manual_seed(seed + 100003 * kind_index + 997 * trial)
            changed_depths, changed_widths = d.clone(), w.clone()
            for row, groups in enumerate(eligible):
                order = torch.randperm(len(groups), generator=generator).tolist()
                found = False
                for group_index in order:
                    group = groups[group_index]
                    positions = [
                        group[index] for index in torch.randperm(len(group), generator=generator).tolist()
                    ]
                    if kind in {"exchange", "transfer_depth"}:
                        if not base_feasible:
                            continue
                        for a in positions[:32]:
                            for b in positions[:32]:
                                if a >= b:
                                    continue
                                if kind == "exchange":
                                    if int(d[row, a]) == int(d[row, b]) and torch.equal(w[:, :, row, a], w[:, :, row, b]):
                                        continue
                                    extra_before_b = int(base_cost[row, b] - base_cost[row, a])
                                    if extra_before_b > int(slack[row, a:b].min()):
                                        continue
                                    pair = TokenPair(
                                        row * d.shape[1] + a, row * d.shape[1] + b,
                                        int(d[row, a]), int(d[row, b]),
                                    )
                                    proposal_d, proposal_w = swap_plan(changed_depths, changed_widths, pair)
                                else:
                                    donor, recipient = int(d[row, a]), int(d[row, b])
                                    if donor <= config.causal_min_passes or recipient >= horizon:
                                        continue
                                    donated = w[donor - 1, :, row, a].clone()
                                    saved = ledger.base_pass_costs[donor - 1] + sum(
                                        int(width) * price for width, price in zip(donated, ledger.expert_costs, strict=True)
                                    )
                                    allowance = saved + int(slack[row, b:].min())
                                    required = ledger.base_pass_costs[recipient] + sum(
                                        int(width) * price for width, price in zip(donated, ledger.expert_costs, strict=True)
                                    )
                                    # Later passes can have larger nonexpert charges.
                                    # Narrow the donated payload rather than borrowing.
                                    for layer in range(config.n_layers - 1, -1, -1):
                                        while required > allowance and int(donated[layer]) > config.min_routed_k:
                                            donated[layer] -= 1
                                            required -= ledger.expert_costs[layer]
                                    if required > allowance:
                                        continue
                                    proposal_d, proposal_w = changed_depths.clone(), changed_widths.clone()
                                    proposal_d[row, a] -= 1
                                    proposal_d[row, b] += 1
                                    proposal_w[recipient, :, row, b] = donated
                                    proposal_w[donor - 1, :, row, a] = 0
                                changed_depths, changed_widths = proposal_d, proposal_w
                                found = True
                                break
                            if found:
                                break
                    else:
                        for position in positions:
                            depth = int(d[row, position])
                            if depth < 2:
                                continue
                            if kind == "shorten":
                                # Outside-floor deletions remain informative interventions,
                                # but are excluded from feasible adaptive-headroom claims.
                                changed_depths[row, position] -= 1
                                changed_widths[depth - 1, :, row, position] = 0
                            else:
                                p, layer = (
                                    (1, 0) if kind == "narrow_early"
                                    else (depth - 1, config.n_layers - 1)
                                )
                                if int(w[p, layer, row, position]) <= config.min_routed_k:
                                    continue
                                changed_widths[p, layer, row, position] = config.min_routed_k
                            found = True
                            break
                    if found:
                        break
            changed = changed_depths.ne(d) | changed_widths.ne(w).any(dim=(0, 1))
            if not bool(changed.any()):
                continue
            fingerprint = plan_fingerprint(changed_depths, changed_widths)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            generated += 1
            changed_tokens += int(changed.sum())
            records.append(Intervention(
                kind, trial, changed_depths.to(depths.device), changed_widths.to(widths.device),
                changed.to(depths.device),
            ))
        coverage[kind] = {
            "requested": trials, "generated": generated, "changed_positions": changed_tokens,
            "status": "generated" if generated else "no_eligible_distinct_interventions",
        }
    return records, coverage


def effect_summary(
    before: Tensor, after: Tensor, batch: TrainingBatch, changed: Tensor, *,
    minimum_gain: float,
) -> dict[str, Any]:
    if before.ndim != 3 or before.shape != after.shape or before.shape[1:] != batch.labels.shape:
        raise ValueError("Repeated loss tensors must be [repeats, batch, sequence].")
    if changed.shape != batch.labels.shape or changed.dtype != torch.bool or not bool(changed.any()):
        raise ValueError("Require a nonempty boolean changed-position mask.")
    if not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError("Minimum effect resolution must be finite and positive.")
    valid = batch.labels.ne(-100)
    if bool((changed & ~batch.attention_mask).any()) or not bool(torch.isfinite(before[:, valid]).all() & torch.isfinite(after[:, valid]).all()):
        raise ValueError("Changed positions must be active and all observed losses finite.")
    downstream = torch.zeros_like(valid)
    columns = torch.arange(valid.shape[1], device=valid.device)
    for row in range(valid.shape[0]):
        for position in changed[row].nonzero().flatten().tolist():
            downstream[row] |= (
                (columns > position) & batch.document_ids[row].eq(batch.document_ids[row, position])
            )
    downstream &= valid & ~changed
    untouched = valid & ~changed & ~downstream
    delta = before.double().mean(dim=0) - after.double().mean(dim=0)
    token_range = (
        before.double().amax(dim=0) - before.double().amin(dim=0)
        + after.double().amax(dim=0) - after.double().amin(dim=0)
    )
    control_max = float(delta[untouched].abs().max()) if bool(untouched.any()) else None
    control_range = float(token_range[untouched].max()) if bool(untouched.any()) else None
    token_resolution = max(
        minimum_gain, 3 * (control_max or 0), 3 * (control_range or 0),
    )
    result: dict[str, Any] = {
        "gain_sign": "Positive means lower CE under the intervention.",
        "intervention_positions": int(changed.sum()),
        "unaffected_max_abs_token_change": control_max,
        "unaffected_max_repeat_range": control_range,
        "causality_controls_ok": (
            control_max <= max(minimum_gain, 3 * control_range)
            if control_max is not None and control_range is not None else None
        ),
        "token_effect_resolution": token_resolution,
        "resolution_interpretation": "Conservative empirical noise screen, not a statistical confidence interval.",
    }
    for name, region in (
        ("changed", changed & valid), ("downstream", downstream), ("unaffected", untouched), ("total", valid),
    ):
        count = int(region.sum())
        original_sums = before[:, region].double().sum(dim=1)
        changed_sums = after[:, region].double().sum(dim=1)
        repeat_range = float(
            original_sums.max() - original_sums.min() + changed_sums.max() - changed_sums.min()
        )
        gain = float(delta[region].sum())
        result[name] = {
            "tokens": count, "loss_sum_gain": gain,
            "mean_loss_gain": gain / count if count else None,
            "loss_sum_repeat_range": repeat_range,
            "sum_gain_above_repeat_noise": count > 0 and abs(gain) > max(minimum_gain, 3 * repeat_range),
            "tokens_above_noise_screen": int((delta[region].abs() > token_resolution).sum()),
            "max_abs_token_change": float(delta[region].abs().max()) if count else None,
        }
    result["local_and_total_opposite_sign"] = (
        result["changed"]["sum_gain_above_repeat_noise"]
        and result["total"]["sum_gain_above_repeat_noise"]
        and result["changed"]["loss_sum_gain"] * result["total"]["loss_sum_gain"] < 0
    )
    return result


def assemble_panel_proposal(
    config: Metis16Config, attention_mask: Tensor, plans: list[tuple[Tensor, Tensor]],
    losses: list[Tensor], *, critic: bool, horizon: int, price: float,
) -> tuple[Tensor, Tensor, float]:
    """Use cross-context teacher losses only to propose a plan requiring replay."""
    if len(plans) != len(losses) or not plans:
        raise ValueError("Every panel plan requires an observed local-loss contribution.")
    if not math.isfinite(price) or price < 0:
        raise ValueError("Panel price must be finite and nonnegative.")
    if any(
        loss.shape != attention_mask.shape
        or not bool(torch.isfinite(loss).all()) or bool(loss.lt(0).any())
        for loss in losses
    ):
        raise ValueError("Panel losses must be finite, nonnegative and match the token geometry.")
    if attention_mask.dtype != torch.bool or not bool(attention_mask.any()):
        raise ValueError("Panel admission requires a nonempty boolean attention mask.")
    if any(
        depths.shape != attention_mask.shape or not torch.equal(depths.gt(0), attention_mask)
        for depths, _ in plans
    ):
        raise ValueError("Panel plans must cover exactly the original active positions.")
    costs = torch.stack([
        token_costs(config, depths, widths, critic=critic) for depths, widths in plans
    ], dim=-1)
    contributions = torch.stack(losses, dim=-1).double()
    reference = JointComputeCosts.from_config(config).reference_per_token
    scores = -contributions - price * costs.double() / reference
    maximum = max(int(costs.max()), reference) * attention_mask.numel()
    if maximum > torch.iinfo(torch.int64).max:
        raise OverflowError("Panel prefix accounting could overflow.")
    selected, _ = _causal_admission(costs, scores, attention_mask, reference)
    depths = torch.zeros_like(plans[0][0])
    widths = torch.zeros_like(plans[0][1])
    for index, (candidate_depth, candidate_width) in enumerate(plans):
        take = selected.eq(index) & attention_mask
        depths[take] = candidate_depth[take]
        widths[:, :, take] = candidate_width[:, :, take]
    if not budget_summary(config, depths, widths, critic=critic, horizon=horizon)["policy_feasible"]:
        raise ArithmeticError("Panel proposal violated its declared prefix budget or policy support.")
    predicted = contributions.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return depths, widths, float(predicted.sum())


def diagnose_decisions(
    model: Metis16ForCausalLM, batch: TrainingBatch, curriculum: CurriculumState, *,
    seed: int, repeat_forwards: int = 3, trials_per_kind: int = 4,
    min_context: int = 32, minimum_gain: float = 1e-3,
    plan_output: Path | None = None, panel_oracle: bool = False,
) -> dict[str, Any]:
    if (
        not model.config.causal_compute_budget or not model.config.terminal_action_critic
        or model.config.ffn_mode != "moe"
    ):
        raise CapabilityError("Decision diagnosis requires a causal terminal-action checkpoint.")
    if repeat_forwards < 2:
        raise ValueError("Decision diagnosis requires repeated forwards.")
    runtime = FrozenRuntimeState(model)
    policy_is_joint = curriculum.compute_allocation_mode == "joint"
    total_forward_flops = 0.0
    forward_calls = 0
    try:
        native, native_stats = repeated_plan_evaluation(
            model, batch, curriculum, seed=seed, runtime_state=runtime,
            repeat_forwards=repeat_forwards, minimum_loss_delta=1e-5,
            return_terminal_router_observations=policy_is_joint,
        )
        total_forward_flops += native_stats["nominal_forward_flops_all_calls"]
        forward_calls += repeat_forwards
        base_d, base_w = native.output.chosen_depths, native.widths
        horizon = curriculum.max_passes or model.config.max_passes
        explicit = replace(
            curriculum, compute_allocation_mode="legacy", continuation_mode="fixed_max",
            routed_k_mode="fixed", fixed_routed_k=round(model.config.target_mean_routed_k),
            stochastic_routing=False, max_passes=horizon,
        )

        def replay(depths: Tensor, widths: Tensor):
            nonlocal total_forward_flops, forward_calls
            losses, first, summaries = [], None, []
            for _ in range(repeat_forwards):
                result = evaluate_in_memory(
                    model, batch, explicit, seed=seed, runtime_state=runtime,
                    force_depth=depths, force_routed_k=widths,
                    return_terminal_router_observations=True,
                )
                if not torch.equal(result.output.chosen_depths, depths) or not torch.equal(result.widths, widths):
                    raise RuntimeError("Explicit replay changed the requested execution plan.")
                outcomes = terminal_outcomes(result, batch)
                if first is None:
                    first = outcomes
                losses.append(outcomes.losses)
                summary = forward_summary(result, batch)
                summaries.append(summary)
                total_forward_flops += summary["nominal_probe_forward_flops"]
                forward_calls += 1
            return torch.stack(losses), first, summaries

        base_losses, base_outcomes, base_summaries = replay(base_d, base_w)
        valid = batch.labels.ne(-100)
        replay_mean = float(base_losses[:, valid].double().mean())
        replay_range = float(
            base_losses[:, valid].double().mean(dim=1).max()
            - base_losses[:, valid].double().mean(dim=1).min()
        )
        fingerprint = plan_fingerprint(base_d, base_w)
        same_plan_losses = [
            record["lm_loss"] for record in native_stats["repeats"]
            if record["plan_sha256"] == fingerprint
        ]
        replay_tolerance = max(
            1e-4, 3 * (max(same_plan_losses) - min(same_plan_losses)), 3 * replay_range,
        )
        native_anchor = float(native.output.loss)
        if abs(replay_mean - native_anchor) > replay_tolerance:
            raise RuntimeError(
                f"Native replay is not faithful: native={native_anchor}, "
                f"replay={replay_mean}, tolerance={replay_tolerance}."
            )
        assert base_outcomes is not None
        prediction = base_outcomes.prediction
        available = prediction.width_utilities.shape[2]

        def predicted_values(depths: Tensor, widths: Tensor) -> Tensor:
            return prediction.value_of_actions(
                (depths - 1).clamp_min(0), widths[1:available + 1],
            ).detach()

        base_values = predicted_values(base_d, base_w)
        torch.testing.assert_close(
            base_values[valid], base_outcomes.values[valid], rtol=1e-6, atol=1e-5,
        )
        proposals, coverage = interventions(
            model.config, batch, base_d, base_w, trials=trials_per_kind,
            seed=seed, horizon=horizon, min_context=min_context, critic=policy_is_joint,
        )
        records = []
        if plan_output is not None:
            plan_output.mkdir(exist_ok=False)
            torch.save({"depths": base_d.cpu(), "widths": base_w.cpu()}, plan_output / "native.pt")
        for proposal in proposals:
            losses, _, summaries = replay(proposal.depths, proposal.widths)
            effect = effect_summary(
                base_losses, losses, batch, proposal.changed, minimum_gain=minimum_gain,
            )
            predicted_delta = predicted_values(proposal.depths, proposal.widths) - base_values
            expected_gain = float(predicted_delta[proposal.changed].double().sum())
            budget = budget_summary(
                model.config, proposal.depths, proposal.widths,
                critic=policy_is_joint, horizon=horizon,
            )
            record = {
                "kind": proposal.kind, "trial": proposal.trial,
                "plan_sha256": plan_fingerprint(proposal.depths, proposal.widths),
                "changed_positions": int(proposal.changed.sum()),
                "predicted_local_loss_sum_gain": expected_gain if policy_is_joint else None,
                "effects": effect, "budget": budget,
                "model_loss": sum(item["lm_loss"] for item in summaries) / repeat_forwards,
            }
            per_sequence = []
            for row in range(batch.input_ids.shape[0]):
                sites = proposal.changed[row:row + 1]
                if bool(sites.any()):
                    row_batch = replace(
                        batch, input_ids=batch.input_ids[row:row + 1],
                        canonical_ids=batch.canonical_ids[row:row + 1],
                        labels=batch.labels[row:row + 1],
                        attention_mask=batch.attention_mask[row:row + 1],
                        document_ids=batch.document_ids[row:row + 1],
                        reset_mask=batch.reset_mask[row:row + 1],
                        non_padding_tokens=int(batch.attention_mask[row].sum()),
                        supervised_tokens=int(valid[row].sum()),
                    )
                    row_effect = effect_summary(
                        base_losses[:, row:row + 1], losses[:, row:row + 1],
                        row_batch, sites, minimum_gain=minimum_gain,
                    )
                    per_sequence.append({
                        "sequence": row,
                        "changed_positions": sites[0].nonzero().flatten().tolist(),
                        "predicted_local_loss_sum_gain": float(
                            predicted_delta[row][sites[0]].double().sum()
                        ) if policy_is_joint else None,
                        "effects": row_effect,
                    })
            record["per_sequence_effects"] = per_sequence
            if plan_output is not None:
                artifact = plan_output / f"{proposal.kind}-{proposal.trial}.pt"
                torch.save(
                    {"depths": proposal.depths.cpu(), "widths": proposal.widths.cpu()},
                    artifact,
                )
                record["plan_artifact"] = identify_file(artifact)
            records.append(record)
        panel_report = None
        if panel_oracle:
            panel_plans = [(base_d, base_w)]
            panel_losses = [base_losses.double().mean(dim=0)]
            panel_rows = []
            widths_to_try = sorted({
                model.config.min_routed_k,
                max(model.config.min_routed_k, min(2, model.config.max_routed_k)),
                round(model.config.target_mean_routed_k), model.config.max_routed_k,
            })
            seen_panel = {plan_fingerprint(base_d, base_w)}
            for depth in range(model.config.causal_min_passes, horizon + 1):
                for width in widths_to_try:
                    depths = batch.attention_mask.long() * depth
                    routed = torch.zeros_like(base_w)
                    routed[0] = batch.attention_mask * round(model.config.target_mean_routed_k)
                    if depth > 1:
                        routed[1:depth] = batch.attention_mask * width
                    fingerprint = plan_fingerprint(depths, routed)
                    if fingerprint in seen_panel:
                        continue
                    seen_panel.add(fingerprint)
                    observed, _, summaries = replay(depths, routed)
                    panel_plans.append((depths, routed))
                    panel_losses.append(observed.double().mean(dim=0))
                    panel_rows.append({
                        "depth": depth, "future_k": width,
                        "loss": float(observed[:, valid].double().mean()),
                        "diagnostic_train_flops": summaries[0]["modeled_total_train_flops"],
                        "budget": budget_summary(
                            model.config, depths, routed, critic=policy_is_joint, horizon=horizon,
                        ),
                    })
            proposed = []
            seen_proposals = {plan_fingerprint(base_d, base_w)}
            for price in (0.0, 0.1, 1.0):
                depths, routed, predicted_sum = assemble_panel_proposal(
                    model.config, batch.attention_mask, panel_plans, panel_losses,
                    critic=policy_is_joint, horizon=horizon, price=price,
                )
                fingerprint = plan_fingerprint(depths, routed)
                if fingerprint in seen_proposals:
                    continue
                seen_proposals.add(fingerprint)
                observed, _, summaries = replay(depths, routed)
                changed = depths.ne(base_d) | routed.ne(base_w).any(dim=(0, 1))
                effect = effect_summary(
                    base_losses, observed, batch, changed, minimum_gain=minimum_gain,
                )
                proposal_loss = float(observed[:, valid].double().mean())
                proposal = {
                    "price": price, "plan_sha256": fingerprint,
                    "changed_positions": int(changed.sum()), "effects": effect,
                    "teacher_local_surrogate_loss": predicted_sum / int(valid.sum()),
                    "replayed_loss": proposal_loss,
                    "replayed_loss_gain": replay_mean - proposal_loss,
                    "predicted_local_loss_sum_gain": float(
                        (predicted_values(depths, routed) - base_values)[changed & valid].double().sum()
                    ) if policy_is_joint else None,
                    "budget": budget_summary(
                        model.config, depths, routed, critic=policy_is_joint, horizon=horizon,
                    ),
                    "diagnostic_train_flops": summaries[0]["modeled_total_train_flops"],
                }
                if plan_output is not None:
                    artifact = plan_output / f"panel-price-{price:g}.pt"
                    torch.save({"depths": depths.cpu(), "widths": routed.cpu()}, artifact)
                    proposal["plan_artifact"] = identify_file(artifact)
                if proposal_loss < replay_mean and effect["causality_controls_ok"] is not False:
                    confirm_base, _, _ = replay(base_d, base_w)
                    confirm_proposal, _, _ = replay(depths, routed)
                    proposal["independent_repeat_confirmation"] = effect_summary(
                        confirm_base, confirm_proposal, batch, changed, minimum_gain=minimum_gain,
                    )
                proposed.append(proposal)
            panel_report = {
                "uniform_teacher_programs": panel_rows,
                "replayed_proposals": proposed,
                "interpretation": (
                    "These label-informed proposal scores were measured under different uniform "
                    "contexts, not isolated causal interventions. Only the assembled plan's actual "
                    "replay loss is its result. Unsupervised positions contribute no local LM loss; "
                    "this is not an invented CE target or a deployable prefix-only policy."
                ),
            }
        feasible = [
            record for record in records
            if record["budget"]["policy_feasible"] and record["effects"]["causality_controls_ok"]
        ]
        best = min(feasible, key=lambda record: record["model_loss"]) if feasible else None
        confirmation = None
        if best is not None and best["model_loss"] < replay_mean:
            selected = next(
                proposal for proposal in proposals
                if proposal.kind == best["kind"] and proposal.trial == best["trial"]
            )
            confirmation_base, _, _ = replay(base_d, base_w)
            confirmation_candidate, _, _ = replay(selected.depths, selected.widths)
            confirmation = effect_summary(
                confirmation_base, confirmation_candidate, batch, selected.changed,
                minimum_gain=minimum_gain,
            )
        fit_error = (base_values + base_losses.double().mean(dim=0))[valid]
        return {
            "status": "diagnostic_complete",
            "native": {
                **forward_summary(native, batch),
                "mean_lm_loss": native_stats["mean_lm_loss"], "repeat_statistics": native_stats,
            },
            "replay": {
                "mean_loss": replay_mean, "tolerance": replay_tolerance,
                "native_anchor_loss": native_anchor,
                "plan_sha256": plan_fingerprint(base_d, base_w),
                "budget": budget_summary(
                    model.config, base_d, base_w, critic=policy_is_joint, horizon=horizon,
                ),
                "observation_forward_cost": base_summaries[0]["modeled_total_train_flops"],
                "terminal_value_mse": float(fit_error.square().mean()) if policy_is_joint else None,
            },
            "coverage": coverage, "interventions": records,
            "local_loss_panel_oracle": panel_report,
            "best_sampled_feasible_alternative": ({
                "kind": best["kind"], "trial": best["trial"], "model_loss": best["model_loss"],
                "loss_gain": replay_mean - best["model_loss"],
                "plan_sha256": best["plan_sha256"],
                "independent_repeat_confirmation": confirmation,
            } if best else None),
            "forward_calls": forward_calls,
            "nominal_forward_flops_all_calls": total_forward_flops,
            "limitations": [
                "No weights or optimizer states were fitted.",
                "Predicted values describe the changed positions' own terminal losses, not suffix returns.",
                "Changed plans are replayed jointly; per-token effects are not assumed additive.",
                "The best sampled alternative uses evaluation labels, is not a deployable policy or global optimum, and does not prove prefix-only learnability.",
                "Failure to find a better sampled plan does not prove that no useful adaptive computation exists.",
                "Budget and declared-policy violations are explicitly reported and excluded from feasible-headroom summaries.",
                "Controls without a learned joint policy have no meaningful router-value comparison; captured critic work is diagnostic overhead.",
                "Numerical screens are empirical, not confidence intervals. Expert identities are rerouted, not forcibly held fixed.",
            ],
        }
    finally:
        runtime.restore()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("checkpoint", "bf16"), default="bf16")
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--evaluation-gap-blocks", type=int, default=1)
    parser.add_argument("--sequences", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--repeat-forwards", type=int, default=3)
    parser.add_argument("--trials-per-kind", type=int, default=4)
    parser.add_argument("--min-context", type=int, default=32)
    parser.add_argument("--minimum-gain", type=float, default=1e-3)
    parser.add_argument("--panel-oracle", action="store_true")
    parser.add_argument("--seed", type=int, default=16062026)
    args = parser.parse_args(argv)
    source = source_identity()
    source["decision_probe_sha256"] = sha256_file(Path(__file__))
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint = checkpoint / "state.pt" if checkpoint.is_dir() else checkpoint
    manifest_path = args.run_manifest or infer_run_manifest(checkpoint)
    output = fresh_output_directory(
        args.output, inputs=(checkpoint.parent.parent, manifest_path, args.release_root),
    )
    checkpoint_identity, manifest_identity = identify_file(checkpoint), identify_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    identity = validate_checkpoint(payload, manifest)
    config = Metis16Config.from_mapping(identity["model"])
    if args.sequence_length != config.sequence_length:
        raise ValueError("Keep the checkpoint's declared sequence length and compute geometry.")
    curriculum = replace(
        CurriculumState.from_value(identity["curriculum"]), stochastic_routing=False,
        random_policy_step=args.step,
    )
    inventory = ReleaseInventory.from_release_root(args.release_root)
    release_identity = identify_file(inventory.root / "RELEASE.json")
    batch, sampling = held_out_batch(
        inventory, identity, checkpoint_step=payload["step"], step=args.step,
        sequences=args.sequences, sequence_length=args.sequence_length,
        gap_blocks=args.evaluation_gap_blocks,
    )
    precision = identity["precision_profile"] if args.precision == "checkpoint" else args.precision
    device = torch.device(args.device)
    model, numerical = load_frozen_model(
        config, payload["model"], device=device, precision=precision,
        enable_joint=config.joint_compute_router,
        checkpoint_precision=identity["precision_profile"], initialization_seed=args.seed,
    )
    checkpoint_step = payload["step"]
    del payload
    batch = batch.to(device)
    warmup = evaluate_in_memory(model, batch, curriculum, seed=args.seed)
    warmup_summary = forward_summary(warmup, batch)
    del warmup
    report = diagnose_decisions(
        model, batch, curriculum, seed=args.seed, repeat_forwards=args.repeat_forwards,
        trials_per_kind=args.trials_per_kind, min_context=args.min_context,
        minimum_gain=args.minimum_gain, plan_output=output / "plans",
        panel_oracle=args.panel_oracle,
    )
    report["forward_calls"] += 1
    report["nominal_forward_flops_all_calls"] += warmup_summary["nominal_probe_forward_flops"]
    report.update({
        "schema": "more.decision-credit/v1", "source": source,
        "checkpoint": checkpoint_identity, "run_manifest": manifest_identity,
        "run_identity_sha256": manifest["run_identity_sha256"],
        "training_source_revision": identity["source_revision"],
        "checkpoint_next_step": checkpoint_step, "sampling": sampling,
        "base_model_config_sha256": json_sha256(identity["model"]),
        "release": release_identity, "numerical_policy": numerical,
        "arguments": vars(args), "warmup": warmup_summary,
    })
    for item in (checkpoint_identity, manifest_identity, release_identity):
        assert_file_unchanged(item)
    if {**source_identity(), "decision_probe_sha256": sha256_file(Path(__file__))} != source:
        raise RuntimeError("Diagnostic source changed while measurements were running.")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "output": str(output), "native_loss": report["native"]["mean_lm_loss"],
        "interventions": len(report["interventions"]),
        "best_sampled_feasible_alternative": report["best_sampled_feasible_alternative"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
