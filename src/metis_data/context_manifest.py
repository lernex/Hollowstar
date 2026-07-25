from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .config import load_yaml


CONTEXT_PLAN_SCHEMA = "metis.context-extension-plan/v1"
CONTEXT_TOKEN_BUDGET = 18_000_000_000
CONTEXT_GATES = (6_000_000_000, 12_000_000_000, 18_000_000_000)
CONTEXT_TRAIN_LENGTH = 163_840
CONTEXT_DEPLOY_LENGTH = 131_072
CONTEXT_SEQUENCE_MIX = {
    "natural_long": 0.70,
    "dependency_constructed": 0.20,
    "short_replay": 0.10,
}


def load_context_plan(
    path: str | Path,
    *,
    base_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = load_yaml(path)
    validate_context_plan(plan, base_manifest=base_manifest)
    plan["_path"] = str(Path(path).expanduser().resolve())
    return plan


def _finite_fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _source_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = plan.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("context plan must contain source rows")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"context source {index} must be a mapping")
        source_id = str(raw.get("id") or "")
        if not source_id or source_id in result:
            raise ValueError("context source ids must be unique and non-empty")
        result[source_id] = raw
    return result


def validate_context_plan(
    plan: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if plan.get("schema") != CONTEXT_PLAN_SCHEMA:
        raise ValueError(f"context plan schema must be {CONTEXT_PLAN_SCHEMA}")
    if int(plan.get("token_budget", -1)) != CONTEXT_TOKEN_BUDGET:
        raise ValueError("context token budget must be exactly 18B")
    if int(plan.get("unique_active_tokens", -1)) != CONTEXT_TOKEN_BUDGET:
        raise ValueError("context plan must expose exactly 18B unique active tokens")
    if int(plan.get("train_context", -1)) != CONTEXT_TRAIN_LENGTH:
        raise ValueError("context training length must be exactly 163,840")
    if int(plan.get("deploy_context", -1)) != CONTEXT_DEPLOY_LENGTH:
        raise ValueError("context deployment length must be exactly 131,072")
    if tuple(int(value) for value in plan.get("checkpoint_gates", ())) != CONTEXT_GATES:
        raise ValueError("context checkpoint gates must be exactly 6B/12B/18B")
    packing_multiple = int(plan.get("packing_multiple_records", 0))
    if packing_multiple <= 0 or packing_multiple % 384:
        raise ValueError("context packing multiple must be a positive multiple of 384")

    raw_mix = plan.get("sequence_mix")
    if not isinstance(raw_mix, Mapping) or set(raw_mix) != set(CONTEXT_SEQUENCE_MIX):
        raise ValueError("context sequence mix has an unexpected lane")
    mix = {
        str(name): _finite_fraction(value, f"sequence_mix.{name}")
        for name, value in raw_mix.items()
    }
    if not math.isclose(sum(mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("context sequence mix must sum to one")
    if any(
        not math.isclose(mix[name], expected, rel_tol=0.0, abs_tol=1e-9)
        for name, expected in CONTEXT_SEQUENCE_MIX.items()
    ):
        raise ValueError("context sequence mix must remain 70/20/10")

    sources = _source_map(plan)
    total = 0
    domains: dict[str, set[str]] = defaultdict(set)
    for source_id, source in sources.items():
        domain = str(source.get("domain") or "")
        tokens = int(source.get("tokens", 0))
        multiplier = _finite_fraction(
            source.get("retrieval_multiplier"),
            f"{source_id}.retrieval_multiplier",
        )
        if not domain or tokens <= 0 or multiplier < 1:
            raise ValueError(
                f"{source_id} requires a domain, positive tokens, and retrieval multiplier >= 1"
            )
        total += tokens
        domains[domain].add(source_id)
    if total != CONTEXT_TOKEN_BUDGET:
        raise ValueError(f"context source quotas sum to {total:,}, not 18B")

    if base_manifest is not None:
        base_sources = {
            str(source["id"]): source
            for source in base_manifest.get("sources", [])
            if isinstance(source, Mapping) and source.get("id")
        }
        missing = sorted(set(sources) - set(base_sources))
        if missing:
            raise ValueError(f"context plan references unknown base sources: {missing}")
        generated = sorted(
            source_id
            for source_id in sources
            if base_sources[source_id].get("provenance", {}).get("generated")
        )
        if generated:
            raise ValueError(
                "context organic retrieval quotas may not use generated base sources: "
                f"{generated}"
            )

    fallback = plan.get("fallbacks")
    if not isinstance(fallback, Mapping):
        raise ValueError("context plan must declare fail-closed fallbacks")
    if (
        fallback.get("exhaustion_policy") != "fail_closed"
        or fallback.get("preserve_domain") is not True
        or fallback.get("generated_donor_forbidden") is not True
    ):
        raise ValueError("context fallbacks must be fail-closed and domain preserving")
    seen: set[str] = set()
    groups = fallback.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("context fallback groups are missing")
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("context fallback group must be a mapping")
        group_id = str(raw_group.get("id") or "")
        members = [str(value) for value in raw_group.get("members", [])]
        donors = [str(value) for value in raw_group.get("donor_order", [])]
        if group_id not in domains or set(members) != domains[group_id]:
            raise ValueError(
                f"context fallback group {group_id!r} must contain exactly its domain sources"
            )
        if set(donors) != set(members) or len(donors) != len(set(donors)):
            raise ValueError(
                f"context fallback group {group_id!r} needs one ordered copy of every member"
            )
        if seen.intersection(members):
            raise ValueError("context sources may appear in only one fallback group")
        seen.update(members)
    if seen != set(sources):
        raise ValueError("context fallback groups do not cover every source")

    selection = plan.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("context selection contract is missing")
    if (
        int(selection.get("minimum_long_document_tokens", 0)) < 4_096
        or int(selection.get("preferred_long_document_tokens", 0))
        < int(selection.get("minimum_long_document_tokens", 0))
        or int(selection.get("short_replay_document_tokens", 0)) != 4_096
        or selection.get("fail_on_source_exhaustion") is not True
        or selection.get("exact_gate_domain_balance") is not True
    ):
        raise ValueError("context length-selection thresholds are invalid")
    long_filter = selection.get("long_range_filter")
    if (
        not isinstance(long_filter, Mapping)
        or long_filter.get("implementation") != "metis.long-range-information/v1"
        or long_filter.get("require_structural_prefilter") is not True
        or long_filter.get("model_calibration_at_context_gates") is not True
        or long_filter.get("model_rescore_full_corpus") is not False
    ):
        raise ValueError(
            "context plan must use structural selection plus model-calibrated gates"
        )
    if (
        int(selection.get("gate_evaluation_records", 0)) != 384
        or int(selection.get("gate_evaluation_context", 0))
        != CONTEXT_DEPLOY_LENGTH
        or int(selection.get("gate_evaluation_tail_tokens", 0)) != 4_096
        or not 2
        <= int(selection.get("gate_evaluation_probe_prefix_tokens", 0))
        <= 32
    ):
        raise ValueError("context gate-evaluation geometry is invalid")
    construction = selection.get("constructed_sequence")
    if (
        not isinstance(construction, Mapping)
        or construction.get("implementation")
        != "metis.negative-document-extension/v1"
        or construction.get("preserve_source_text") is not True
        or construction.get("generated_text") is not False
    ):
        raise ValueError("context dependency construction contract is invalid")

    gate_policy = plan.get("gate_policy")
    if (
        not isinstance(gate_policy, Mapping)
        or gate_policy.get("schema") != "metis.context-gate-policy/v1"
        or tuple(int(value) for value in gate_policy.get("checkpoints", ()))
        != CONTEXT_GATES
        or gate_policy.get("retain_all_gate_checkpoints") is not True
        or gate_policy.get("promote_best_passing_gate") is not True
    ):
        raise ValueError("context autonomous gate policy is invalid")
    return plan


def context_retrieval_reserve_tokens(plan: Mapping[str, Any]) -> dict[str, int]:
    validate_context_plan(plan)
    return {
        str(source["id"]): int(
            Decimal(int(source["tokens"]))
            * (Decimal(str(source["retrieval_multiplier"])) - Decimal(1))
        )
        for source in plan["sources"]
    }


def context_candidate_targets(plan: Mapping[str, Any]) -> dict[str, int]:
    validate_context_plan(plan)
    return {
        str(source["id"]): int(
            Decimal(int(source["tokens"]))
            * Decimal(str(source["retrieval_multiplier"]))
        )
        for source in plan["sources"]
    }


def context_quota_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_context_plan(plan)
    gates = list(CONTEXT_GATES)
    previous = 0
    rows: list[dict[str, Any]] = []
    for gate_index, gate in enumerate(gates):
        tranche = gate - previous
        previous = gate
        remaining = tranche
        weighted = [
            (
                str(source["id"]),
                int(source["tokens"]),
                str(source["domain"]),
            )
            for source in plan["sources"]
        ]
        denominator = sum(tokens for _source_id, tokens, _domain in weighted)
        apportioned = [
            (source_id, tranche * tokens // denominator, domain)
            for source_id, tokens, domain in weighted
        ]
        remainders = sorted(
            (
                (tranche * tokens % denominator, source_id)
                for source_id, tokens, _domain in weighted
            ),
            key=lambda item: (-item[0], item[1]),
        )
        additions = {
            source_id for _remainder, source_id in remainders[: tranche - sum(v for _, v, _ in apportioned)]
        }
        for source_id, tokens, domain in apportioned:
            value = tokens + int(source_id in additions)
            rows.append(
                {
                    "gate_index": gate_index,
                    "gate_target_tokens": gate,
                    "source_id": source_id,
                    "domain": domain,
                    "tokens": value,
                }
            )
            remaining -= value
        if remaining:
            raise AssertionError("context gate apportionment is not exact")
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[str(row["source_id"])] += int(row["tokens"])
    expected = {str(source["id"]): int(source["tokens"]) for source in plan["sources"]}
    if totals != expected:
        # Source quotas need not split perfectly under an independent Hamilton
        # allocation per gate. Rebalance deterministically without changing any
        # 6B gate total.
        by_gate = {
            gate: [row for row in rows if int(row["gate_index"]) == gate]
            for gate in range(len(gates))
        }
        deficits = {source: expected[source] - totals.get(source, 0) for source in expected}
        while any(value for value in deficits.values()):
            receiver = min(
                (source for source, value in deficits.items() if value > 0),
                key=lambda source: (-deficits[source], source),
            )
            donor = min(
                (source for source, value in deficits.items() if value < 0),
                key=lambda source: (deficits[source], source),
            )
            moved = False
            for gate in range(len(gates)):
                receiver_row = next(row for row in by_gate[gate] if row["source_id"] == receiver)
                donor_row = next(row for row in by_gate[gate] if row["source_id"] == donor)
                if int(donor_row["tokens"]) <= 0:
                    continue
                amount = min(deficits[receiver], -deficits[donor], int(donor_row["tokens"]))
                receiver_row["tokens"] = int(receiver_row["tokens"]) + amount
                donor_row["tokens"] = int(donor_row["tokens"]) - amount
                deficits[receiver] -= amount
                deficits[donor] += amount
                moved = True
                break
            if not moved:
                raise AssertionError("context gate/source quota rebalance failed")
    return rows
