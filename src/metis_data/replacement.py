from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


PHASES = ("phase_a", "phase_b", "phase_c")


class ReplacementError(RuntimeError):
    """Raised when an immutable source mixture cannot be filled safely."""


def _source_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(source["id"]): source for source in manifest.get("sources", [])}


def _freshness_bucket(source: Mapping[str, Any]) -> str | None:
    provenance = source.get("provenance", {})
    if not provenance.get("fresh"):
        return None
    return str(provenance.get("freshness_bucket") or "")


def _compatible(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    group: Mapping[str, Any],
) -> bool:
    if defaults.get("preserve_category", True) and donor.get("category") != target.get("category"):
        return False
    if defaults.get("preserve_freshness_bucket", True):
        if _freshness_bucket(donor) != _freshness_bucket(target):
            return False
    target_provenance = target.get("provenance", {})
    donor_provenance = donor.get("provenance", {})
    if (
        defaults.get("no_generated_increase", True)
        and not target_provenance.get("generated")
        and donor_provenance.get("generated")
    ):
        return False
    if (
        defaults.get("no_transformed_increase", True)
        and not group.get("allow_transformed_increase", False)
        and not target_provenance.get("transformed")
        and donor_provenance.get("transformed")
    ):
        return False
    if int(target.get("phase_tokens", {}).get("phase_c", 0)) and donor_provenance.get(
        "generated"
    ):
        return False
    return True


def replacement_chains(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    policy = manifest.get("replacement_policy")
    if not isinstance(policy, Mapping):
        sources = _source_map(manifest)
        return (
            {source_id: [] for source_id in sources},
            {source_id: "legacy_fail_closed" for source_id in sources},
        )
    defaults = policy.get("defaults", {})
    sources = _source_map(manifest)
    chains: dict[str, list[str]] = {}
    groups_by_source: dict[str, str] = {}
    for group in policy.get("groups", []):
        group_id = str(group.get("id") or "")
        members = [str(value) for value in group.get("members", [])]
        donors = [str(value) for value in group.get("donor_order", [])]
        for source_id in members:
            if source_id in groups_by_source:
                raise ReplacementError(
                    f"Replacement source {source_id} belongs to multiple groups"
                )
            groups_by_source[source_id] = group_id
            target = sources.get(source_id)
            if target is None:
                continue
            chains[source_id] = [
                donor_id
                for donor_id in donors
                if donor_id != source_id
                and donor_id in sources
                and _compatible(
                    target,
                    sources[donor_id],
                    defaults=defaults,
                    group=group,
                )
            ]
    return chains, groups_by_source


def validate_replacement_policy(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = manifest.get("replacement_policy")
    if not isinstance(policy, Mapping):
        return ["replacement_policy must be loaded"]
    if policy.get("schema") != "metis.replacement-policy/v1":
        errors.append("replacement policy schema must be metis.replacement-policy/v1")
    sources = _source_map(manifest)
    defaults = policy.get("defaults", {})
    if defaults.get("preserve_category") is not True:
        errors.append("replacement policy must preserve category")
    if defaults.get("preserve_freshness_bucket") is not True:
        errors.append("replacement policy must preserve freshness bucket")
    if defaults.get("no_generated_increase") is not True:
        errors.append("replacement policy must forbid generated-data increases")
    if defaults.get("exhaustion_policy") != "fail_closed":
        errors.append("replacement policy exhaustion_policy must be fail_closed")

    seen: dict[str, str] = {}
    groups = policy.get("groups", [])
    if not isinstance(groups, list) or not groups:
        errors.append("replacement policy must contain groups")
        return errors
    for group in groups:
        group_id = str(group.get("id") or "")
        members = [str(value) for value in group.get("members", [])]
        donors = [str(value) for value in group.get("donor_order", [])]
        if not group_id:
            errors.append("replacement group is missing id")
        if not members:
            errors.append(f"replacement group {group_id or '<missing>'} has no members")
        if not donors:
            errors.append(f"replacement group {group_id or '<missing>'} has no donors")
        for source_id in members:
            if source_id not in sources:
                errors.append(f"replacement group {group_id} has unknown member {source_id}")
            if source_id in seen:
                errors.append(
                    f"replacement source {source_id} appears in {seen[source_id]} and {group_id}"
                )
            seen[source_id] = group_id
        for donor_id in donors:
            if donor_id not in sources:
                errors.append(f"replacement group {group_id} has unknown donor {donor_id}")
        if group.get("self_reserve_required"):
            for source_id in members:
                source = sources.get(source_id, {})
                reserve_crawls = source.get("access", {}).get("reserve_crawls", [])
                if not reserve_crawls:
                    errors.append(
                        f"replacement source {source_id} requires a declared cold reserve"
                    )

    missing = sorted(set(sources) - set(seen))
    unexpected = sorted(set(seen) - set(sources))
    if missing:
        errors.append(f"replacement policy omits sources: {missing}")
    if unexpected:
        errors.append(f"replacement policy references unknown sources: {unexpected}")
    try:
        chains, groups_by_source = replacement_chains(manifest)
    except ReplacementError as exc:
        errors.append(str(exc))
        return errors
    group_payload = {str(group.get("id")): group for group in groups}
    for source_id in sorted(sources):
        if chains.get(source_id):
            continue
        group = group_payload.get(groups_by_source.get(source_id, ""), {})
        if not group.get("self_reserve_required"):
            errors.append(
                f"replacement source {source_id} has neither a compatible donor nor a cold reserve"
            )
    return errors


def _hamilton(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    positive = {key: int(value) for key, value in weights.items() if int(value) > 0}
    denominator = sum(positive.values())
    if total <= 0:
        return {key: 0 for key in weights}
    if denominator <= 0:
        raise ReplacementError("Cannot apportion positive replacement tokens over zero weights")
    floors: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for key, weight in positive.items():
        numerator = total * weight
        floors[key] = numerator // denominator
        remainders.append((numerator % denominator, key))
    missing = total - sum(floors.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:missing]:
        floors[key] += 1
    return {key: floors.get(key, 0) for key in weights}


def allocate_replacements(
    manifest: Mapping[str, Any],
    *,
    requirements: Mapping[str, Mapping[str, int]],
    available_tokens: Mapping[str, int],
    strict: bool = True,
) -> dict[str, Any]:
    """Allocate exact source deficits from ordered, compatible donor surplus.

    Requirements are immutable quota-source counts. Available tokens are
    measured for the actual source after the relevant pipeline stage. The
    original source retains as much of every phase as possible; shortages are
    proportionally distributed across its phases, then filled from donor
    surplus in the policy's declared order.
    """

    sources = _source_map(manifest)
    chains, groups_by_source = replacement_chains(manifest)
    missing_requirements = sorted(set(sources) - set(requirements))
    if missing_requirements:
        raise ReplacementError(
            f"Replacement requirements omit manifest sources: {missing_requirements}"
        )

    own: dict[str, dict[str, int]] = {}
    deficits: dict[str, dict[str, int]] = {}
    surplus: dict[str, int] = {}
    for source_id in sorted(sources):
        source_requirements = {
            phase: max(0, int(requirements[source_id].get(phase, 0)))
            for phase in PHASES
        }
        required_total = sum(source_requirements.values())
        available = max(0, int(available_tokens.get(source_id, 0)))
        own_total = min(required_total, available)
        own[source_id] = (
            dict(source_requirements)
            if own_total == required_total
            else _hamilton(own_total, source_requirements)
        )
        deficits[source_id] = {
            phase: source_requirements[phase] - own[source_id][phase]
            for phase in PHASES
        }
        surplus[source_id] = max(0, available - own_total)

    policy = manifest.get(
        "replacement_policy",
        {
            "version": "legacy-fail-closed",
            "defaults": {"phase_resolution_order": ["phase_b", "phase_a", "phase_c"]},
        },
    )
    phase_order = [
        str(phase)
        for phase in policy.get("defaults", {}).get(
            "phase_resolution_order", ["phase_b", "phase_a", "phase_c"]
        )
    ]
    if sorted(phase_order) != sorted(PHASES):
        raise ReplacementError(f"Invalid replacement phase order: {phase_order}")

    def source_resolution_key(source_id: str) -> tuple[int, int, int, str]:
        source = sources[source_id]
        provenance = source.get("provenance", {})
        return (
            -int(bool(provenance.get("fresh"))),
            -int(bool(int(source.get("phase_tokens", {}).get("phase_c", 0)))),
            -int(source.get("processing", {}).get("priority", 0)),
            source_id,
        )

    transfers: list[dict[str, Any]] = []
    for target_id in sorted(sources, key=source_resolution_key):
        for phase in phase_order:
            needed = deficits[target_id][phase]
            if needed <= 0:
                continue
            for donor_id in chains.get(target_id, []):
                take = min(needed, surplus.get(donor_id, 0))
                if take <= 0:
                    continue
                transfers.append(
                    {
                        "target_source_id": target_id,
                        "actual_source_id": donor_id,
                        "phase": phase,
                        "tokens": take,
                        "replacement_group": groups_by_source[target_id],
                    }
                )
                surplus[donor_id] -= take
                deficits[target_id][phase] -= take
                needed -= take
                if needed == 0:
                    break

    unresolved = {
        source_id: {
            phase: tokens
            for phase, tokens in phases.items()
            if int(tokens) > 0
        }
        for source_id, phases in deficits.items()
        if any(int(tokens) > 0 for tokens in phases.values())
    }
    if unresolved and strict:
        detail = "; ".join(
            f"{source_id}="
            + ",".join(f"{phase}:{tokens:,}" for phase, tokens in phases.items())
            for source_id, phases in sorted(unresolved.items())
        )
        raise ReplacementError(
            "Replacement reserves cannot satisfy immutable source quotas: " + detail
        )

    assignments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_id in sorted(sources):
        for phase in PHASES:
            tokens = own[source_id][phase]
            if tokens:
                assignments[source_id].append(
                    {
                        "target_source_id": source_id,
                        "actual_source_id": source_id,
                        "phase": phase,
                        "tokens": tokens,
                        "replacement": False,
                    }
                )
    for transfer in transfers:
        assignments[transfer["actual_source_id"]].append(
            {
                **transfer,
                "replacement": True,
            }
        )

    return {
        "schema": "metis.replacement-allocation/v1",
        "policy_version": policy.get("version"),
        "own_allocations": own,
        "transfers": transfers,
        "assignments_by_actual_source": dict(assignments),
        "available_tokens": {
            source_id: max(0, int(available_tokens.get(source_id, 0)))
            for source_id in sorted(sources)
        },
        "unused_reserve_tokens": dict(sorted(surplus.items())),
        "unresolved": unresolved,
        "replacement_tokens": sum(int(row["tokens"]) for row in transfers),
    }


def replacement_resilience_report(
    manifest: Mapping[str, Any],
    *,
    requirements: Mapping[str, Mapping[str, int]],
    candidate_tokens: Mapping[str, int],
) -> dict[str, Any]:
    """Simulate complete loss of each source's normal candidate allocation."""

    sources = _source_map(manifest)
    _, groups_by_source = replacement_chains(manifest)
    groups = {
        str(group.get("id")): group
        for group in manifest.get("replacement_policy", {}).get("groups", [])
    }
    rows: dict[str, dict[str, Any]] = {}
    for source_id in sorted(sources):
        simulated = {
            candidate_id: max(0, int(tokens))
            for candidate_id, tokens in candidate_tokens.items()
        }
        simulated[source_id] = 0
        allocation = allocate_replacements(
            manifest,
            requirements=requirements,
            available_tokens=simulated,
            strict=False,
        )
        unresolved = sum(
            int(tokens)
            for phases in allocation["unresolved"].values()
            for tokens in phases.values()
        )
        group = groups.get(groups_by_source.get(source_id, ""), {})
        cold_reserve = bool(group.get("self_reserve_required"))
        rows[source_id] = {
            "required_unique_tokens": sum(
                int(tokens) for tokens in requirements[source_id].values()
            ),
            "simulated_unresolved_tokens_after_donors": unresolved,
            "donor_pool_covers_complete_source_loss": unresolved == 0,
            "cold_reserve_widening_available": cold_reserve,
            "automatic_shortfall_path_available": unresolved == 0 or cold_reserve,
            "exhaustion_policy": "fail_closed",
        }
    return {
        "schema": "metis.replacement-resilience/v1",
        "sources": rows,
        "all_sources_have_automatic_shortfall_path": all(
            row["automatic_shortfall_path_available"] for row in rows.values()
        ),
        "complete_loss_covered_by_other_donors": sum(
            int(row["donor_pool_covers_complete_source_loss"])
            for row in rows.values()
        ),
        "cold_reserve_only_sources": sorted(
            source_id
            for source_id, row in rows.items()
            if not row["donor_pool_covers_complete_source_loss"]
            and row["cold_reserve_widening_available"]
        ),
    }
