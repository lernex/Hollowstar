from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml


PIPELINE_SCHEMA = "metis.posttraining-pipeline/v1"
SEALED_ARTIFACT_SCHEMA = "metis.sealed-artifact/v1"
CHECKPOINT_RECEIPT_SCHEMA = "metis.checkpoint-receipt/v1"
STAGE_OUTPUT_SCHEMA = "metis.stage-output/v1"
STATE_SCHEMA = "metis.posttraining-state/v1"
RUNTIME_SCHEMA = "metis.posttraining-runtime/v1"

# Continued pretraining produces the base model; everything after it aligns
# that base model.  The executed order is the concatenation, unchanged, but the
# boundary is now explicit rather than an operator convention.
BASE_MODEL_STAGE_IDS = ("context_extension",)
ALIGNMENT_STAGE_IDS = (
    "cold_start_sft",
    "overall_sft",
    "deepseek_dpd_pilot",
    "deepseek_dpd",
    "specialist_reasoning",
    "specialist_code",
    "specialist_knowledge",
    "specialist_writing",
    "specialist_agentic",
    "opd_consolidation",
    "pairwise_reward_model",
    "preference_alignment",
    "evaluation",
    "publish_gate",
)
EXPECTED_STAGE_IDS = BASE_MODEL_STAGE_IDS + ALIGNMENT_STAGE_IDS
DPD_STAGE_IDS = ("deepseek_dpd_pilot", "deepseek_dpd")
SPECIALIST_STAGE_IDS = (
    "specialist_reasoning",
    "specialist_code",
    "specialist_knowledge",
    "specialist_writing",
    "specialist_agentic",
)
EXPECTED_FAMILIES = ("praxis", "logos")
REQUIRED_SFT_AUDITS = (
    "identity_scrub",
    "safety_calibration",
    "abstention",
    "deduplication",
    "contamination",
    "answer_mode_balance",
)


class PipelineContractError(RuntimeError):
    """Raised when an immutable training or artifact contract is violated."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_hash(value: Any, *, omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in omitted}
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineContractError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PipelineContractError(f"{label} must be a list")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineContractError(f"{label} must be finite")
    return result


def _resolve_relative(root: Path, raw: str, label: str) -> Path:
    if not raw or Path(raw).is_absolute():
        raise PipelineContractError(f"{label} must be a non-empty relative path")
    candidate = root / raw
    if candidate.is_symlink():
        raise PipelineContractError(f"{label} may not be a symlink: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PipelineContractError(f"{label} escapes its sealed root") from exc
    return resolved


def _repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "METIS_1.6_PLAN.md"
        ).is_file():
            return candidate
    raise PipelineContractError(f"could not locate repository root above {path}")


def _validate_mix(mix: Mapping[str, Any], expected: Mapping[str, float], label: str) -> None:
    observed = {str(key): _require_number(value, f"{label}.{key}") for key, value in mix.items()}
    if set(observed) != set(expected):
        raise PipelineContractError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(observed)}"
        )
    if not math.isclose(sum(observed.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PipelineContractError(f"{label} fractions must sum to 1")
    for key, target in expected.items():
        if not math.isclose(observed[key], target, rel_tol=0.0, abs_tol=1e-9):
            raise PipelineContractError(
                f"{label}.{key} must be {target}, got {observed[key]}"
            )


def _stages_by_id(pipeline: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stages = _require_list(pipeline.get("stages"), "stages")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_stage in enumerate(stages):
        stage = _require_mapping(raw_stage, f"stages[{index}]")
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            raise PipelineContractError(f"stages[{index}].id must be a non-empty string")
        if stage_id in result:
            raise PipelineContractError(f"Duplicate stage id: {stage_id}")
        result[stage_id] = stage
    return result


def _validate_live_canary_policy(
    autotune: Mapping[str, Any],
    *,
    stage_id: str,
    evaluator_implementation: str,
) -> None:
    policy = _require_mapping(
        autotune.get("live_canary"), f"{stage_id}.autotune.live_canary"
    )
    integer_fields = (
        "training_optimizer_steps",
        "minimum_evaluation_records",
        "maximum_working_set_candidates",
        "working_set_warmup_trials",
        "working_set_measurement_trials",
    )
    if (
        policy.get("schema")
        != "metis.posttraining-live-canary-policy/v1"
        or policy.get("evaluator_implementation")
        != evaluator_implementation
        or policy.get("restore_parent_between_trials") is not True
        or policy.get("restore_rng_between_trials") is not True
        or policy.get("require_working_set_autotune") is not True
        or any(
            isinstance(policy.get(field), bool)
            or not isinstance(policy.get(field), int)
            or int(policy[field]) <= 0
            for field in integer_fields
        )
        or int(policy["training_optimizer_steps"]) > 8
        or int(policy["maximum_working_set_candidates"]) > 12
        or not 1 <= int(policy["working_set_warmup_trials"]) <= 3
        or not 2 <= int(policy["working_set_measurement_trials"]) <= 7
    ):
        raise PipelineContractError(
            f"{stage_id} must require bounded live family-rank canaries"
        )
    tolerance = policy.get("maximum_reproduction_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 <= float(tolerance) <= 0.01
    ):
        raise PipelineContractError(
            f"{stage_id} live canary reproduction tolerance is invalid"
        )


def validate_pipeline(pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the immutable Metis-1.6 context-extension/post-training contract."""

    if pipeline.get("schema") != PIPELINE_SCHEMA:
        raise PipelineContractError(f"schema must be {PIPELINE_SCHEMA}")
    if pipeline.get("model") != "metis-1.6":
        raise PipelineContractError("model must be metis-1.6")
    if tuple(pipeline.get("families", ())) != EXPECTED_FAMILIES:
        raise PipelineContractError("families must be ordered as [praxis, logos]")
    if pipeline.get("fail_closed") is not True:
        raise PipelineContractError("pipeline must fail closed")

    tokenizer = _require_mapping(pipeline.get("tokenizer"), "tokenizer")
    if (
        int(tokenizer.get("vocabulary_size", -1)) != 65_536
        or tokenizer.get("manifest_env") != "METIS_TOKENIZER_MANIFEST"
    ):
        raise PipelineContractError("tokenizer must require the sealed 65,536-vocabulary manifest")

    boundary = pipeline.get("base_model_boundary")
    if boundary != BASE_MODEL_STAGE_IDS[-1]:
        raise PipelineContractError(
            "base_model_boundary must name the last base-model stage "
            f"({BASE_MODEL_STAGE_IDS[-1]}); the base model is not the 1T checkpoint"
        )
    if not str(pipeline.get("continued_pretraining_contract", "")).endswith(
        "pretraining.yaml"
    ):
        raise PipelineContractError(
            "continued_pretraining_contract must point at the pretraining "
            "contract that owns the continued-pretraining declaration"
        )

    context = _require_mapping(pipeline.get("context_extension"), "context_extension")
    if (
        int(context.get("base_context", -1)) != 4_096
        or int(context.get("deploy_context", -1)) != 131_072
        or int(context.get("train_context", -1)) != 163_840
        or context.get("schedule") != "single_jump"
    ):
        raise PipelineContractError(
            "context extension must be one 4,096 -> 163,840 training jump for 131,072 deployment"
        )
    token_budget = int(context.get("token_budget", -1))
    if token_budget != 18_000_000_000:
        raise PipelineContractError(
            "context-extension exposure must remain exactly 18B active tokens"
        )
    if tuple(int(value) for value in context.get("checkpoint_gates", ())) != (
        6_000_000_000,
        12_000_000_000,
        18_000_000_000,
    ):
        raise PipelineContractError(
            "context-extension checkpoint gates must be exactly 6B/12B/18B"
        )
    _validate_mix(
        _require_mapping(context.get("sequence_mix"), "context_extension.sequence_mix"),
        {"long": 0.90, "base": 0.10},
        "context_extension.sequence_mix",
    )
    _validate_mix(
        _require_mapping(context.get("data_mix"), "context_extension.data_mix"),
        {"pretrain_style": 0.80, "synthetic_long_context": 0.20},
        "context_extension.data_mix",
    )
    if context.get("positional_encoding") != "nope":
        raise PipelineContractError("direct context extension is allowed only with NoPE")

    stages = _stages_by_id(pipeline)
    if tuple(stages) != EXPECTED_STAGE_IDS:
        raise PipelineContractError(
            "stage order must be " + " -> ".join(EXPECTED_STAGE_IDS)
        )
    expected_inputs = {
        "context_extension": "base_pretraining",
        "cold_start_sft": "context_extension",
        "overall_sft": "cold_start_sft",
        "deepseek_dpd_pilot": "overall_sft",
        "deepseek_dpd": "deepseek_dpd_pilot",
        **{stage_id: "deepseek_dpd" for stage_id in SPECIALIST_STAGE_IDS},
        "opd_consolidation": "deepseek_dpd",
        "pairwise_reward_model": "opd_consolidation",
        "preference_alignment": "pairwise_reward_model",
        "evaluation": "preference_alignment",
        "publish_gate": "evaluation",
    }
    requirements_by_stage: dict[
        str, dict[str, Mapping[str, Any]]
    ] = {}
    for stage_id in EXPECTED_STAGE_IDS:
        stage = stages[stage_id]
        expected_input = expected_inputs[stage_id]
        if stage.get("input_stage") != expected_input:
            raise PipelineContractError(
                f"{stage_id}.input_stage must be {expected_input!r}, got {stage.get('input_stage')!r}"
            )
        if stage.get("enabled") is not True:
            raise PipelineContractError(f"{stage_id} may not be silently disabled")
        if stage.get("output_kind") not in {"checkpoint", "reward_model", "evaluation", "publish"}:
            raise PipelineContractError(f"{stage_id}.output_kind is invalid")
        requirements = _require_list(stage.get("requirements"), f"{stage_id}.requirements")
        named_requirements: dict[str, Mapping[str, Any]] = {}
        for requirement_index, raw_requirement in enumerate(requirements):
            requirement = _require_mapping(
                raw_requirement,
                f"{stage_id}.requirements[{requirement_index}]",
            )
            name = requirement.get("name")
            if (
                not isinstance(name, str)
                or not name
                or name in named_requirements
            ):
                raise PipelineContractError(
                    f"{stage_id} requirement names must be unique and non-empty"
                )
            named_requirements[name] = requirement
            if not isinstance(requirement.get("env"), str) or not requirement.get("env"):
                raise PipelineContractError(
                    f"{stage_id}.requirements[{requirement_index}].env is required"
                )
            if not isinstance(requirement.get("schema"), str) or not requirement.get("schema"):
                raise PipelineContractError(
                    f"{stage_id}.requirements[{requirement_index}].schema is required"
                )
        requirements_by_stage[stage_id] = named_requirements

    context_stage = stages["context_extension"]
    if (
        int(context_stage.get("token_budget", -1)) != token_budget
        or int(context_stage.get("sequence_length", -1))
        != int(context["train_context"])
        or int(context_stage.get("deploy_context", -1))
        != int(context["deploy_context"])
    ):
        raise PipelineContractError(
            "context-extension stage must inherit the exact 18B/163,840/131,072 "
            "top-level contract"
        )
    if tuple(
        int(value) for value in context_stage.get("checkpoint_gates", ())
    ) != tuple(int(value) for value in context["checkpoint_gates"]):
        raise PipelineContractError(
            "context stage checkpoint gates differ from the top-level contract"
        )
    context_gate_policy = _require_mapping(
        context_stage.get("gate_policy"), "context_extension.gate_policy"
    )
    if (
        context_gate_policy.get(
            "checkpoint_at_first_optimizer_boundary_after_target"
        )
        is not True
        or int(
            context_gate_policy.get("maximum_gate_overshoot_tokens", 0)
        )
        != 65_000_000
        or context_gate_policy.get("retain_all_gate_checkpoints") is not True
        or context_gate_policy.get("promote_best_passing_gate") is not True
    ):
        raise PipelineContractError(
            "context stage must autonomously retain and promote 6B/12B/18B gates"
        )

    cold_start = stages["cold_start_sft"]
    if int(cold_start.get("sequence_length", -1)) != 65_536:
        raise PipelineContractError("cold_start_sft maximum length must be 65,536")
    _validate_mix(
        _require_mapping(cold_start.get("length_mix"), "cold_start_sft.length_mix"),
        {"8192_to_32768": 0.92, "32769_to_65536": 0.08},
        "cold_start_sft.length_mix",
    )
    _validate_mix(
        _require_mapping(cold_start.get("domain_mix"), "cold_start_sft.domain_mix"),
        {"math": 0.50, "science": 0.30, "code": 0.20},
        "cold_start_sft.domain_mix",
    )
    overall = stages["overall_sft"]
    if int(overall.get("sequence_length", -1)) != 131_072:
        raise PipelineContractError("overall_sft maximum length must be 131,072")
    _validate_mix(
        _require_mapping(overall.get("length_mix"), "overall_sft.length_mix"),
        {
            "8192_to_32768": 0.65,
            "32769_to_65536": 0.25,
            "65537_to_131072": 0.10,
        },
        "overall_sft.length_mix",
    )
    _validate_mix(
        _require_mapping(overall.get("domain_mix"), "overall_sft.domain_mix"),
        {
            "reasoning": 0.40,
            "general_qa_writing": 0.30,
            "agent_tool_use": 0.20,
            "code": 0.10,
        },
        "overall_sft.domain_mix",
    )
    for stage_id in ("cold_start_sft", "overall_sft"):
        stage = stages[stage_id]
        modes = _require_mapping(stage.get("answer_modes"), f"{stage_id}.answer_modes")
        if set(modes) != {"direct", "think"}:
            raise PipelineContractError(f"{stage_id} must include direct and think answer modes")
        if min(_require_number(value, f"{stage_id}.answer_modes") for value in modes.values()) <= 0:
            raise PipelineContractError(f"{stage_id} answer modes must both have positive mass")
        if tuple(stage.get("required_audits", ())) != REQUIRED_SFT_AUDITS:
            raise PipelineContractError(
                f"{stage_id} must require every Metis-1.5 remediation audit"
            )

    dpd = _require_mapping(
        stages["deepseek_dpd"].get("objective"),
        "deepseek_dpd.objective",
    )
    if (
        dpd.get("name") != "dual_preference_distillation"
        or dpd.get("teacher_interface")
        != "cross_tokenizer_sequence_preferences"
        or dpd.get("positive_token_distillation") is not False
        or dpd.get("negative_token_distillation") is not False
        or dpd.get("sequence_preference") != "dpo_margin"
        or float(dpd.get("token_distillation_weight", -1)) != 0.0
    ):
        raise PipelineContractError(
            "DeepSeek bootstrap must use cross-tokenizer sequence DPD; "
            "fabricated token-logit alignment is forbidden"
        )
    pilot = stages["deepseek_dpd_pilot"]
    if pilot.get("promotion_gate") is None:
        raise PipelineContractError("DPD pilot requires an explicit promotion gate")
    pilot_autotune = _require_mapping(
        pilot.get("autotune"), "deepseek_dpd_pilot.autotune"
    )
    if (
        int(pilot_autotune.get("maximum_candidates", 0)) < 1
        or int(pilot_autotune.get("maximum_candidates", 0)) > 12
        or pilot_autotune.get("fail_on_nonfinite") is not True
        or not pilot_autotune.get("beta_candidates")
        or not pilot_autotune.get("token_distillation_weight_candidates")
        or not pilot_autotune.get("sequence_preference_weight_candidates")
    ):
        raise PipelineContractError("DPD pilot must use a bounded, fail-closed hyperparameter search")
    _validate_live_canary_policy(
        pilot_autotune,
        stage_id="deepseek_dpd_pilot",
        evaluator_implementation="metis.dpd-preference-replay/v1",
    )
    if dpd.get("hyperparameters_from") != "deepseek_dpd_pilot":
        raise PipelineContractError("full DPD must consume the promoted pilot hyperparameters")

    for stage_id, domain in (
        ("specialist_reasoning", "stem"),
        ("specialist_code", "code"),
        ("specialist_knowledge", "knowledge"),
        ("specialist_writing", "writing"),
        ("specialist_agentic", "agentic"),
    ):
        stage = stages[stage_id]
        if stage.get("domain") != domain:
            raise PipelineContractError(f"{stage_id}.domain must be {domain}")
        if (
            stage.get("branch_from") != "deepseek_dpd"
            or stage.get("preserves_unified_policy") is not True
            or stage.get("optimizer_state") != "reset"
        ):
            raise PipelineContractError(
                f"{stage_id} must be an independent reset branch from deepseek_dpd"
            )
        objective = _require_mapping(stage.get("objective"), f"{stage_id}.objective")
        if (
            objective.get("algorithm") != "gspo"
            or objective.get("ratio_unit") != "sequence"
            or objective.get("length_normalized_ratio") is not True
            or objective.get("kl_penalty") != 0
            or objective.get("mask_truncated") is not True
        ):
            raise PipelineContractError(f"{stage_id} must use sequence GSPO without KL")
        dynamic_length = objective.get("dynamic_thinking_length") is True
        schedule = objective.get("length_schedule")
        if dynamic_length:
            schedule = _require_mapping(
                schedule, f"{stage_id}.objective.length_schedule"
            )
            correctness_fraction = _require_number(
                schedule.get("correctness_only_fraction"),
                f"{stage_id}.correctness_only_fraction",
            )
            adaptive_fraction = _require_number(
                schedule.get("adaptive_budget_fraction"),
                f"{stage_id}.adaptive_budget_fraction",
            )
            if (
                not math.isclose(
                    correctness_fraction + adaptive_fraction,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not 0.5 <= correctness_fraction < 1.0
                or not 0 < adaptive_fraction <= 0.5
            ):
                raise PipelineContractError(
                    f"{stage_id} must bootstrap correctness before adaptive "
                    "thinking-length shaping"
                )
        elif schedule is not None:
            raise PipelineContractError(
                f"{stage_id} may not declare a disabled length schedule"
            )
        autotune = _require_mapping(stage.get("autotune"), f"{stage_id}.autotune")
        clip_pairs = _require_list(autotune.get("clip_pairs"), f"{stage_id}.clip_pairs")
        normalized_clip_pairs: list[list[float]] = []
        for index, raw_pair in enumerate(clip_pairs):
            pair = _require_list(raw_pair, f"{stage_id}.clip_pairs[{index}]")
            if len(pair) != 2:
                raise PipelineContractError(
                    f"{stage_id}.clip_pairs[{index}] must contain [low, high]"
                )
            normalized_clip_pairs.append(
                [
                    _require_number(pair[0], f"{stage_id}.clip_pairs[{index}].low"),
                    _require_number(pair[1], f"{stage_id}.clip_pairs[{index}].high"),
                ]
            )
        if (
            [float(objective["clip_low"]), float(objective["clip_high"])]
            not in normalized_clip_pairs
            or float(stage["reward"]["length_coefficient"])
            not in [float(item) for item in autotune.get("length_coefficients", [])]
            or not autotune.get("gate")
        ):
            raise PipelineContractError(
                f"{stage_id} must bound clip/length selection around its declared default"
            )
        _validate_live_canary_policy(
            autotune,
            stage_id=stage_id,
            evaluator_implementation="metis.rlvr-offline-policy-replay/v1",
        )
        on_policy = _require_mapping(stage.get("on_policy_filter"), f"{stage_id}.on_policy_filter")
        if (
            int(on_policy.get("samples_per_prompt", -1)) != 16
            or float(on_policy.get("strict_min_pass_rate", -1)) != 0.10
            or float(on_policy.get("strict_max_pass_rate", -1)) != 0.90
        ):
            raise PipelineContractError(f"{stage_id} must use strict 10%-90% avg@16 filtering")

    length_reward = _require_mapping(
        pipeline.get("dynamic_thinking_length"),
        "dynamic_thinking_length",
    )
    if (
        length_reward.get("enabled") is not True
        or length_reward.get("difficulty_signal") != "avg_at_16_pass_rate"
        or length_reward.get("formula")
        != "two_sided_proximity_to_difficulty_budget"
        or length_reward.get("wrong_response_shaping") != 0
        or not 0 < float(length_reward.get("lambda", 0)) < 0.5
        or not 0 < float(length_reward.get("deadband_fraction", 0)) < 0.5
    ):
        raise PipelineContractError(
            "dynamic thinking must be correctness-dominant and reuse avg@16"
        )

    opd = stages["opd_consolidation"]
    opd_objective = _require_mapping(
        opd.get("objective"), "opd_consolidation.objective"
    )
    if (
        tuple(opd.get("teacher_stages", ())) != SPECIALIST_STAGE_IDS
        or opd_objective.get("algorithm") != "on_policy_distillation"
        or opd_objective.get("divergence") != "reverse_kl"
        or opd_objective.get("trajectory_source")
        != "current_unified_student"
        or int(opd_objective.get("top_k_per_model", 0)) != 32
        or opd_objective.get("union_student_and_teacher_top_k") is not True
        or opd_objective.get("one_epoch_single_use") is not True
    ):
        raise PipelineContractError(
            "OPD must consolidate every same-tokenizer Metis specialist with "
            "single-use student trajectories and reverse KL over the top-k union"
        )
    opd_requirements = requirements_by_stage["opd_consolidation"]
    opd_rollouts = opd_requirements.get("specialist_opd_rollouts")
    opd_adapter = opd_requirements.get("opd_generation_adapter")
    if (
        not isinstance(opd_rollouts, Mapping)
        or opd_rollouts.get("schema") != "metis.opd-data/v1"
        or opd_rollouts.get("checkpoint_bound") is not True
        or opd_rollouts.get("family_bound") is not True
        or not isinstance(opd_adapter, Mapping)
        or opd_adapter.get("schema")
        != "metis.opd-generation-capabilities/v1"
        or opd_adapter.get("checkpoint_bound") is True
        or _require_mapping(
            opd_adapter.get("required_metadata"),
            "opd_generation_adapter.required_metadata",
        ).get("generation_adapter_present")
        is not True
    ):
        raise PipelineContractError(
            "OPD requires checkpoint-bound rollouts and a separately sealed "
            "same-tokenizer generation adapter"
        )

    pairwise = stages["pairwise_reward_model"]
    if (
        pairwise.get("loss") != "bradley_terry_with_swap_consistency"
        or pairwise.get("preserves_policy_checkpoint") is not True
    ):
        raise PipelineContractError("pairwise reward model must include swap-consistency")
    alignment = stages["preference_alignment"]
    if (
        alignment.get("reward_source") != "pairwise_reward_model"
        or alignment.get("reward_scoring")
        != "frozen_parent_checkpoint_offline"
        or alignment.get("reward_score_contract")
        != "metis.frozen-reward-scores/v1"
    ):
        raise PipelineContractError(
            "final preference alignment must consume hash-bound offline scores "
            "from the dedicated frozen pairwise RM"
        )

    evaluation = stages["evaluation"]
    gate = _require_mapping(evaluation.get("gate"), "evaluation.gate")
    if gate.get("fail_on_missing_metric") is not True or not gate.get("metrics"):
        raise PipelineContractError("evaluation must fail closed on a non-empty metric gate")
    evaluation_requirements = requirements_by_stage["evaluation"]
    evaluation_results = evaluation_requirements.get("evaluation_results")
    evaluation_suite = evaluation_requirements.get("evaluation_suite")
    if (
        not isinstance(evaluation_results, Mapping)
        or evaluation_results.get("schema")
        != "metis.evaluation-results/v1"
        or evaluation_results.get("checkpoint_bound") is not True
        or evaluation_results.get("family_bound") is not True
        or evaluation_results.get("generated_from_stage")
        != "preference_alignment"
        or not isinstance(evaluation_suite, Mapping)
        or evaluation_suite.get("schema")
        != "metis.evaluation-suite/v1"
        or evaluation_suite.get("checkpoint_bound") is True
        or _require_mapping(
            evaluation_suite.get("required_metadata"),
            "evaluation_suite.required_metadata",
        ).get("generation_adapter_present")
        is not True
    ):
        raise PipelineContractError(
            "evaluation requires a static sealed suite and results generated "
            "against the exact preference-aligned checkpoint"
        )
    publish = stages["publish_gate"]
    if (
        publish.get("requires_evaluation_pass") is not True
        or publish.get("external_upload") is not False
    ):
        raise PipelineContractError(
            "publish_gate may seal a release candidate but may not upload externally"
        )
    return pipeline


# Fields the pipeline's context_extension block mirrors from the authoritative
# continued_pretraining declaration in the pretraining contract.
_CONTINUED_PRETRAINING_MIRRORED = (
    "base_context",
    "train_context",
    "deploy_context",
    "schedule",
    "positional_encoding",
    "token_budget",
    "checkpoint_gates",
    "sequence_mix",
    "data_mix",
)


def cross_check_continued_pretraining(
    pipeline: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Fail closed when the two declarations of the base model disagree.

    The pretraining contract owns continued pretraining because its output is
    the base model.  The pipeline still carries a copy so the post-training
    validator can run standalone, so the copy has to be provably identical.
    """

    declared = _require_mapping(
        contract.get("continued_pretraining"), "continued_pretraining"
    )
    mirrored = _require_mapping(pipeline.get("context_extension"), "context_extension")
    if declared.get("stage_id") != BASE_MODEL_STAGE_IDS[-1]:
        raise PipelineContractError(
            "continued_pretraining.stage_id must name the base-model stage"
        )
    for field in _CONTINUED_PRETRAINING_MIRRORED:
        if field not in declared:
            raise PipelineContractError(
                f"pretraining contract is missing continued_pretraining.{field}"
            )
        if declared[field] != mirrored.get(field):
            raise PipelineContractError(
                f"continued_pretraining.{field} differs between the pretraining "
                f"contract ({declared[field]!r}) and the pipeline "
                f"({mirrored.get(field)!r})"
            )
    if declared.get("in_pretraining_release") is not False:
        raise PipelineContractError(
            "continued_pretraining must declare its corpus separate from the "
            "1T pretraining release"
        )
    boundary = int(declared.get("maximum_gate_overshoot_tokens", 0)) or int(
        _require_mapping(declared.get("gate_policy"), "continued_pretraining.gate_policy")
        .get("maximum_gate_overshoot_tokens", 0)
    )
    train_context = int(declared["train_context"])
    maximum_world = int(declared.get("maximum_world_size", 0))
    # One sequence per rank is the floor, so world_size x train_context is the
    # smallest possible global batch and the overshoot allowance caps the world.
    if maximum_world <= 0 or maximum_world * train_context > boundary:
        raise PipelineContractError(
            "continued_pretraining.maximum_world_size must keep one optimizer "
            f"step ({maximum_world} x {train_context} tokens) inside the "
            f"{boundary}-token gate overshoot allowance"
        )


def load_pipeline(path: str | Path) -> dict[str, Any]:
    pipeline_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    pipeline = _require_mapping(payload, "pipeline")
    validate_pipeline(pipeline)
    reference = pipeline.get("continued_pretraining_contract")
    contract_path = (pipeline_path.parent.parent.parent / str(reference)).resolve()
    if not contract_path.is_file():
        contract_path = (pipeline_path.parent / Path(str(reference)).name).resolve()
    if not contract_path.is_file():
        raise PipelineContractError(
            f"continued_pretraining_contract does not resolve to a file: {reference}"
        )
    cross_check_continued_pretraining(
        pipeline,
        _require_mapping(
            yaml.safe_load(contract_path.read_text(encoding="utf-8")),
            "pretraining contract",
        ),
    )
    return dict(payload)


def strict_on_policy_filter(
    pass_rates: torch.Tensor,
    *,
    minimum: float = 0.10,
    maximum: float = 0.90,
) -> torch.Tensor:
    """Nanbeige's strict 10%-90% on-policy prompt filter."""

    if not 0 <= minimum < maximum <= 1:
        raise ValueError("pass-rate bounds must satisfy 0 <= minimum < maximum <= 1")
    if not torch.is_floating_point(pass_rates):
        pass_rates = pass_rates.float()
    if not torch.isfinite(pass_rates).all():
        raise ValueError("pass_rates must be finite")
    return (pass_rates > minimum) & (pass_rates < maximum)


def avg_at_k(correct: torch.Tensor, *, k: int = 16) -> torch.Tensor:
    if correct.ndim < 1 or correct.shape[-1] != k:
        raise ValueError(f"correct must have exactly {k} samples on its last axis")
    if correct.dtype == torch.bool:
        correct = correct.float()
    if not torch.isfinite(correct).all() or torch.any((correct < 0) | (correct > 1)):
        raise ValueError("correct must contain finite values in [0, 1]")
    return correct.mean(dim=-1)


def masked_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """FP32 next-token loss for CPT/SFT after the caller performs the shift."""

    if logits.ndim < 2 or labels.shape != logits.shape[:-1] or loss_mask.shape != labels.shape:
        raise ValueError("labels and loss_mask must match logits without the vocabulary axis")
    valid = loss_mask.to(dtype=torch.bool) & labels.ne(ignore_index)
    if not torch.any(valid):
        raise ValueError("causal loss mask contains no supervised tokens")
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_labels = labels.long().reshape(-1)
    token_losses = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        ignore_index=ignore_index,
    ).reshape_as(labels)
    return token_losses.masked_select(valid).mean()


def _masked_token_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim < 2 or mask.shape != student_logits.shape[:-1]:
        raise ValueError("mask must match logits without the vocabulary dimension")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("token distillation mask contains no valid tokens")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    token_loss = -(teacher_probs * student_log_probs).sum(dim=-1) * (temperature**2)
    return token_loss.masked_select(valid).mean()


def _sequence_log_prob(
    token_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if token_log_probs.shape != mask.shape:
        raise ValueError("token_log_probs and mask must have identical shapes")
    return (token_log_probs.float() * mask.to(token_log_probs.dtype)).sum(dim=-1)


def dual_preference_distillation_loss(
    *,
    positive_student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_student_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    negative_mask: torch.Tensor,
    policy_positive_token_log_probs: torch.Tensor,
    policy_negative_token_log_probs: torch.Tensor,
    reference_positive_token_log_probs: torch.Tensor,
    reference_negative_token_log_probs: torch.Tensor,
    beta: float = 0.1,
    token_distillation_weight: float = 1.0,
    sequence_preference_weight: float = 1.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Nanbeige DPD: token KD on both samples plus a sequence DPO margin."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if token_distillation_weight < 0 or sequence_preference_weight < 0:
        raise ValueError("loss weights must be non-negative")
    positive_kd = _masked_token_kd(
        positive_student_logits,
        positive_teacher_logits,
        positive_mask,
        temperature=temperature,
    )
    negative_kd = _masked_token_kd(
        negative_student_logits,
        negative_teacher_logits,
        negative_mask,
        temperature=temperature,
    )
    policy_margin = _sequence_log_prob(
        policy_positive_token_log_probs, positive_mask
    ) - _sequence_log_prob(policy_negative_token_log_probs, negative_mask)
    reference_margin = _sequence_log_prob(
        reference_positive_token_log_probs, positive_mask
    ) - _sequence_log_prob(reference_negative_token_log_probs, negative_mask)
    dpo_logits = beta * (policy_margin - reference_margin.detach())
    dpo = -F.logsigmoid(dpo_logits).mean()
    kd = 0.5 * (positive_kd + negative_kd)
    total = token_distillation_weight * kd + sequence_preference_weight * dpo
    return {
        "loss": total,
        "token_distillation": kd.detach(),
        "positive_token_distillation": positive_kd.detach(),
        "negative_token_distillation": negative_kd.detach(),
        "sequence_preference": dpo.detach(),
        "preference_margin": policy_margin.detach().mean(),
    }


def gspo_loss(
    *,
    current_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    truncated: torch.Tensor | None = None,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
    advantage_epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Group Sequence Policy Optimization with DAPO-style truncation masking.

    Shapes are ``[prompt_batch, group, response_tokens]`` for log probabilities
    and masks, and ``[prompt_batch, group]`` for rewards/truncation flags.
    """

    if (
        current_token_log_probs.shape != old_token_log_probs.shape
        or current_token_log_probs.shape != response_mask.shape
        or current_token_log_probs.ndim != 3
    ):
        raise ValueError("GSPO log-probability tensors and mask must share [B, G, T]")
    if rewards.shape != current_token_log_probs.shape[:2]:
        raise ValueError("rewards must have shape [B, G]")
    if clip_low <= 0 or clip_high <= 0 or clip_low >= 1:
        raise ValueError("GSPO clipping ranges must be positive and clip_low < 1")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    valid_tokens = response_mask.to(dtype=torch.bool)
    lengths = valid_tokens.sum(dim=-1)
    valid_sequences = lengths > 0
    if truncated is not None:
        if truncated.shape != rewards.shape:
            raise ValueError("truncated must have shape [B, G]")
        valid_sequences = valid_sequences & ~truncated.to(dtype=torch.bool)

    reward_mean = rewards.float().mean(dim=1, keepdim=True)
    reward_std = rewards.float().std(dim=1, keepdim=True, unbiased=False)
    informative_groups = reward_std.squeeze(1) > advantage_epsilon
    valid_sequences = valid_sequences & informative_groups[:, None]
    if not torch.any(valid_sequences):
        raise ValueError("GSPO batch has no informative, non-truncated sequences")

    advantages = (rewards.float() - reward_mean) / reward_std.clamp_min(advantage_epsilon)
    token_delta = (
        current_token_log_probs.float() - old_token_log_probs.float().detach()
    ) * valid_tokens.to(current_token_log_probs.dtype)
    normalized_log_ratio = token_delta.sum(dim=-1) / lengths.clamp_min(1).float()
    ratio = normalized_log_ratio.clamp(min=-20.0, max=20.0).exp()
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    objective = surrogate.masked_select(valid_sequences).mean()
    clipped = ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)) & valid_sequences
    return {
        "loss": -objective,
        "objective": objective.detach(),
        "mean_reward": rewards.float().masked_select(valid_sequences).mean().detach(),
        "mean_advantage": advantages.masked_select(valid_sequences).mean().detach(),
        "mean_sequence_ratio": ratio.masked_select(valid_sequences).mean().detach(),
        "clipped_fraction": clipped.float().sum().div(valid_sequences.float().sum()).detach(),
        "valid_sequences": valid_sequences.sum().detach(),
    }


def gspo_token_loss(
    *,
    current_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    truncated: torch.Tensor | None = None,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
) -> dict[str, torch.Tensor]:
    """GSPO-token for turn-level/agentic credit with one ratio per response."""

    if (
        current_token_log_probs.shape != old_token_log_probs.shape
        or current_token_log_probs.shape != token_advantages.shape
        or current_token_log_probs.shape != response_mask.shape
        or current_token_log_probs.ndim != 3
    ):
        raise ValueError("GSPO-token inputs must share [B, G, T]")
    if clip_low <= 0 or clip_high <= 0 or clip_low >= 1:
        raise ValueError("GSPO clipping ranges must be positive and clip_low < 1")
    if not torch.isfinite(token_advantages).all():
        raise ValueError("token advantages must be finite")
    valid_tokens = response_mask.to(dtype=torch.bool)
    lengths = valid_tokens.sum(dim=-1)
    valid_sequences = lengths > 0
    if truncated is not None:
        if truncated.shape != lengths.shape:
            raise ValueError("truncated must have shape [B, G]")
        valid_sequences = valid_sequences & ~truncated.to(dtype=torch.bool)
    if not torch.any(valid_sequences):
        raise ValueError("GSPO-token batch has no non-truncated responses")

    token_delta = (
        current_token_log_probs.float() - old_token_log_probs.float().detach()
    ) * valid_tokens.to(current_token_log_probs.dtype)
    normalized_log_ratio = token_delta.sum(dim=-1) / lengths.clamp_min(1).float()
    ratio = normalized_log_ratio.clamp(min=-20.0, max=20.0).exp()
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    advantages = token_advantages.float()
    surrogate = torch.minimum(
        ratio.unsqueeze(-1) * advantages,
        clipped_ratio.unsqueeze(-1) * advantages,
    )
    per_sequence = (
        surrogate * valid_tokens.to(surrogate.dtype)
    ).sum(dim=-1) / lengths.clamp_min(1).float()
    objective = per_sequence.masked_select(valid_sequences).mean()
    clipped = ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)) & valid_sequences
    return {
        "loss": -objective,
        "objective": objective.detach(),
        "mean_sequence_ratio": ratio.masked_select(valid_sequences).mean().detach(),
        "clipped_fraction": clipped.float().sum().div(valid_sequences.float().sum()).detach(),
        "valid_sequences": valid_sequences.sum().detach(),
        "valid_tokens": (
            valid_tokens & valid_sequences.unsqueeze(-1)
        ).sum().detach(),
    }


def gated_code_efficiency_reward(
    pass_fraction: torch.Tensor,
    efficiency_reward: torch.Tensor,
    *,
    format_reward: torch.Tensor | float = 0.0,
    correctness_weight: float = 1.0,
    efficiency_weight: float = 1.0,
) -> torch.Tensor:
    """Nanbeige4.1-style efficiency reward, active only for fully correct code."""

    if pass_fraction.shape != efficiency_reward.shape:
        raise ValueError("pass_fraction and efficiency_reward must share shape")
    if (
        not torch.isfinite(pass_fraction).all()
        or not torch.isfinite(efficiency_reward).all()
        or torch.any((pass_fraction < 0) | (pass_fraction > 1))
    ):
        raise ValueError("invalid code reward inputs")
    formatting = torch.as_tensor(
        format_reward,
        device=pass_fraction.device,
        dtype=torch.float32,
    )
    return (
        formatting
        + correctness_weight * pass_fraction.float()
        + efficiency_weight
        * efficiency_reward.float()
        * torch.isclose(pass_fraction.float(), torch.ones_like(pass_fraction.float()))
    )


def difficulty_adaptive_length_budget(
    *,
    pass_rate: torch.Tensor,
    mean_correct_length: torch.Tensor,
    maximum_length: int | float,
) -> torch.Tensor:
    """DAST token-length budget: p*L_correct + (1-p)*L_max."""

    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    if pass_rate.shape != mean_correct_length.shape:
        raise ValueError("pass_rate and mean_correct_length must have identical shapes")
    if (
        not torch.isfinite(pass_rate).all()
        or not torch.isfinite(mean_correct_length).all()
        or torch.any((pass_rate < 0) | (pass_rate > 1))
        or torch.any((mean_correct_length <= 0) | (mean_correct_length > maximum_length))
    ):
        raise ValueError("invalid pass rates or correct-response lengths")
    return (
        pass_rate.float() * mean_correct_length.float()
        + (1.0 - pass_rate.float()) * float(maximum_length)
    )


def difficulty_adaptive_length_reward(
    *,
    correctness: torch.Tensor,
    response_lengths: torch.Tensor,
    pass_rate: torch.Tensor,
    mean_correct_length: torch.Tensor,
    maximum_length: int | float,
    coefficient: float = 0.05,
    deadband_fraction: float = 0.10,
) -> torch.Tensor:
    """Two-sided, correctness-dominant thinking-budget reward.

    DAST's difficulty budget is retained, but correct responses are rewarded for
    staying near that budget rather than merely being shorter than it. This
    prevents the old one-sided term from rewarding a correct but suspiciously
    short response on a hard prompt. Wrong responses receive exactly zero length
    shaping, so brevity can never compensate for incorrectness.
    """

    if not 0 <= coefficient < 1:
        raise ValueError("coefficient must be in [0, 1)")
    if not 0 <= deadband_fraction < 1:
        raise ValueError("deadband_fraction must be in [0, 1)")
    target_shape = correctness.shape
    if response_lengths.shape != target_shape:
        raise ValueError("correctness and response_lengths must share shape")
    if pass_rate.shape != target_shape or mean_correct_length.shape != target_shape:
        try:
            pass_rate = torch.broadcast_to(pass_rate, target_shape)
            mean_correct_length = torch.broadcast_to(mean_correct_length, target_shape)
        except RuntimeError as exc:
            raise ValueError("difficulty inputs are not broadcastable to response shape") from exc
    if (
        not torch.isfinite(correctness).all()
        or torch.any((correctness < 0) | (correctness > 1))
        or torch.any((response_lengths <= 0) | (response_lengths > maximum_length))
    ):
        raise ValueError("invalid correctness or response lengths")
    budget = difficulty_adaptive_length_budget(
        pass_rate=pass_rate,
        mean_correct_length=mean_correct_length,
        maximum_length=maximum_length,
    )
    relative_deviation = (
        (response_lengths.float() - budget).abs() / budget.clamp_min(1.0)
    )
    outside_deadband = (
        (relative_deviation - deadband_fraction)
        / max(1.0 - deadband_fraction, 1.0e-6)
    ).clamp(min=0.0, max=2.0)
    # +1 inside the target band, falling continuously to -1 for severe
    # underthinking or overthinking.
    proximity = 1.0 - outside_deadband
    shaping = (
        coefficient
        * proximity
        * (correctness > 0).to(torch.float32)
    )
    return correctness.float() + shaping


def thinking_length_diagnostics(
    *,
    correctness: torch.Tensor,
    response_lengths: torch.Tensor,
    pass_rate: torch.Tensor,
    mean_correct_length: torch.Tensor,
    maximum_length: int | float,
    deadband_fraction: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Report under/overthinking only over correct responses."""

    if not 0 <= deadband_fraction < 1:
        raise ValueError("deadband_fraction must be in [0, 1)")
    target_shape = correctness.shape
    try:
        pass_rate = torch.broadcast_to(pass_rate, target_shape)
        mean_correct_length = torch.broadcast_to(
            mean_correct_length, target_shape
        )
    except RuntimeError as exc:
        raise ValueError(
            "difficulty inputs are not broadcastable to response shape"
        ) from exc
    budget = difficulty_adaptive_length_budget(
        pass_rate=pass_rate,
        mean_correct_length=mean_correct_length,
        maximum_length=maximum_length,
    )
    correct = correctness > 0
    raw_count = correct.float().sum()
    count = raw_count.clamp_min(1.0)
    lower = budget * (1.0 - deadband_fraction)
    upper = budget * (1.0 + deadband_fraction)
    under_count = (
        correct & (response_lengths.float() < lower)
    ).float().sum()
    over_count = (
        correct & (response_lengths.float() > upper)
    ).float().sum()
    on_budget_count = (
        correct
        & (response_lengths.float() >= lower)
        & (response_lengths.float() <= upper)
    ).float().sum()
    target_sum = budget.masked_select(correct).sum()
    return {
        "thinking_target_tokens": budget[correct].mean()
        if bool(correct.any())
        else budget.mean() * 0.0,
        "underthinking_rate": under_count / count,
        "overthinking_rate": over_count / count,
        "on_budget_rate": on_budget_count / count,
        "correct_response_count": raw_count,
        "thinking_target_token_sum": target_sum,
        "underthinking_count": under_count,
        "overthinking_count": over_count,
        "on_budget_count": on_budget_count,
    }


def bradley_terry_pairwise_loss(
    preferred_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    *,
    swapped_preferred_scores: torch.Tensor | None = None,
    swapped_rejected_scores: torch.Tensor | None = None,
    swap_consistency_weight: float = 0.1,
    margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Position-balanced pairwise loss with an explicit swapped presentation.

    ``swapped_preferred_scores`` is the score of the *same preferred response*
    after moving it to the opposite pair slot (and likewise for rejected).
    The semantic preference margin should therefore be unchanged by the swap.
    """

    if preferred_scores.shape != rejected_scores.shape:
        raise ValueError("preferred and rejected scores must share shape")
    if swap_consistency_weight < 0 or margin < 0:
        raise ValueError("weights and margin must be non-negative")
    difference = preferred_scores.float() - rejected_scores.float()
    primary_ranking = -F.logsigmoid(difference - margin).mean()
    ranking = primary_ranking
    consistency = torch.zeros((), device=ranking.device, dtype=ranking.dtype)
    swapped_difference: torch.Tensor | None = None
    if (swapped_preferred_scores is None) != (swapped_rejected_scores is None):
        raise ValueError("both swapped score tensors must be supplied together")
    if swapped_preferred_scores is not None and swapped_rejected_scores is not None:
        if (
            swapped_preferred_scores.shape != preferred_scores.shape
            or swapped_rejected_scores.shape != rejected_scores.shape
        ):
            raise ValueError("swapped score tensors must match original score shapes")
        swapped_difference = (
            swapped_preferred_scores.float() - swapped_rejected_scores.float()
        )
        swapped_ranking = -F.logsigmoid(swapped_difference - margin).mean()
        ranking = 0.5 * (primary_ranking + swapped_ranking)
        consistency = F.mse_loss(swapped_difference, difference.detach())
    total = ranking + swap_consistency_weight * consistency
    accuracy_difference = (
        0.5 * (difference + swapped_difference)
        if swapped_difference is not None
        else difference
    )
    return {
        "loss": total,
        "ranking": ranking.detach(),
        "swap_consistency": consistency.detach(),
        "accuracy": (accuracy_difference > 0).float().mean().detach(),
        "margin": accuracy_difference.mean().detach(),
    }


def evaluate_metric_gate(
    metrics: Mapping[str, float],
    gate: Mapping[str, Any],
    *,
    baselines: Mapping[str, Mapping[str, float]] | None = None,
    suite_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate the sealed publish gate without silently ignoring metrics."""

    baseline_values = baselines or {}
    threshold_values = suite_thresholds or {}
    missing: list[str] = []
    failures: list[dict[str, Any]] = []
    checked: dict[str, float] = {}
    for raw_rule in _require_list(gate.get("metrics"), "evaluation.gate.metrics"):
        rule = _require_mapping(raw_rule, "evaluation metric rule")
        name = str(rule.get("name", ""))
        if not name or name not in metrics:
            missing.append(name or "<unnamed>")
            continue
        value = _require_number(metrics[name], f"metric {name}")
        checked[name] = value
        comparison = str(rule.get("comparison", ""))
        passed = False
        expected: Any = None
        if comparison == "minimum":
            expected = _require_number(rule.get("value"), f"{name}.minimum")
            passed = value >= expected
        elif comparison == "maximum":
            expected = _require_number(rule.get("value"), f"{name}.maximum")
            passed = value <= expected
        elif comparison == "suite_threshold":
            bounds = _require_mapping(threshold_values.get(name), f"{name}.suite_threshold")
            minimum = bounds.get("minimum")
            maximum = bounds.get("maximum")
            if minimum is None and maximum is None:
                raise PipelineContractError(f"{name} suite threshold has no bound")
            passed = True
            if minimum is not None:
                passed = passed and value >= _require_number(minimum, f"{name}.minimum")
            if maximum is not None:
                passed = passed and value <= _require_number(maximum, f"{name}.maximum")
            expected = dict(bounds)
        elif comparison.startswith("no_regression_vs_"):
            baseline_stage = comparison.removeprefix("no_regression_vs_")
            baseline = _require_mapping(
                baseline_values.get(baseline_stage),
                f"baseline {baseline_stage}",
            )
            expected = _require_number(baseline.get(name), f"{baseline_stage}.{name}")
            tolerance = _require_number(
                rule.get("relative_tolerance", 0.0),
                f"{name}.relative_tolerance",
            )
            direction = str(rule.get("higher_is_better", True)).lower() != "false"
            if direction:
                passed = value >= expected * (1.0 - tolerance)
            else:
                passed = value <= expected * (1.0 + tolerance)
        elif comparison.startswith("improve_vs_"):
            baseline_stage = comparison.removeprefix("improve_vs_")
            baseline = _require_mapping(
                baseline_values.get(baseline_stage),
                f"baseline {baseline_stage}",
            )
            expected = _require_number(baseline.get(name), f"{baseline_stage}.{name}")
            minimum_delta = _require_number(
                rule.get("minimum_delta", 0.0),
                f"{name}.minimum_delta",
            )
            passed = value >= expected + minimum_delta
        else:
            raise PipelineContractError(f"unsupported metric comparison for {name}: {comparison}")
        if not passed:
            failures.append(
                {
                    "name": name,
                    "observed": value,
                    "comparison": comparison,
                    "expected": expected,
                }
            )
    if missing and gate.get("fail_on_missing_metric") is not True:
        raise PipelineContractError("evaluation gate attempted to ignore missing metrics")
    return {
        "passed": not missing and not failures,
        "checked": checked,
        "missing_metrics": missing,
        "failed_metrics": failures,
    }


def _validate_sealed_artifact(
    manifest_path: Path,
    *,
    expected_schema: str,
    tokenizer_sha256: str | None,
    verify_payload_hashes: bool,
) -> tuple[dict[str, Any], str]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PipelineContractError(f"sealed artifact manifest is missing or unsafe: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("envelope_schema") != SEALED_ARTIFACT_SCHEMA:
        raise PipelineContractError(f"{manifest_path} is not a sealed Metis artifact")
    if payload.get("schema") != expected_schema:
        raise PipelineContractError(
            f"{manifest_path} schema {payload.get('schema')!r} != {expected_schema!r}"
        )
    if payload.get("complete") is not True:
        raise PipelineContractError(f"{manifest_path} is not marked complete")
    observed_hash = _json_hash(payload, omit={"manifest_sha256"})
    if payload.get("manifest_sha256") != observed_hash:
        raise PipelineContractError(f"{manifest_path} failed its self-hash")
    if tokenizer_sha256 is not None and payload.get("tokenizer_sha256") != tokenizer_sha256:
        raise PipelineContractError(f"{manifest_path} tokenizer lineage does not match")

    root = manifest_path.parent
    files = _require_list(payload.get("files"), f"{manifest_path}.files")
    if not files:
        raise PipelineContractError(f"{manifest_path} seals no payload files")
    for index, raw_file in enumerate(files):
        file_record = _require_mapping(raw_file, f"{manifest_path}.files[{index}]")
        path = _resolve_relative(root, str(file_record.get("path", "")), "sealed payload")
        if not path.is_file():
            raise PipelineContractError(f"sealed payload is missing: {path}")
        if int(file_record.get("bytes", -1)) != path.stat().st_size:
            raise PipelineContractError(f"sealed payload size changed: {path}")
        expected_hash = file_record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise PipelineContractError(f"sealed payload hash is invalid: {path}")
        if verify_payload_hashes and _file_hash(path) != expected_hash:
            raise PipelineContractError(f"sealed payload hash changed: {path}")
    return payload, observed_hash


def _validate_checkpoint_receipt(
    receipt_path: Path,
    *,
    family: str,
    expected_stage: str,
    expected_parent_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    verify_payload_hashes: bool,
) -> tuple[dict[str, Any], str]:
    receipt_path = receipt_path.expanduser().resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise PipelineContractError(f"checkpoint receipt is missing or unsafe: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != CHECKPOINT_RECEIPT_SCHEMA:
        raise PipelineContractError(f"invalid checkpoint receipt schema: {receipt_path}")
    receipt_hash = _json_hash(receipt, omit={"receipt_sha256"})
    if receipt.get("receipt_sha256") != receipt_hash:
        raise PipelineContractError(f"checkpoint receipt failed self-hash: {receipt_path}")
    if receipt.get("family") != family or receipt.get("stage") != expected_stage:
        raise PipelineContractError(f"checkpoint receipt has wrong family/stage: {receipt_path}")
    if (
        expected_parent_sha256 is not None
        and receipt.get("parent_checkpoint_sha256") != expected_parent_sha256
    ):
        raise PipelineContractError(f"checkpoint parent lineage mismatch: {receipt_path}")
    if (
        expected_config_sha256 is not None
        and receipt.get("config_sha256") != expected_config_sha256
    ):
        raise PipelineContractError(f"checkpoint config lineage mismatch: {receipt_path}")
    checkpoint_manifest_raw = receipt.get("checkpoint_manifest")
    if not isinstance(checkpoint_manifest_raw, str):
        raise PipelineContractError(f"checkpoint receipt lacks checkpoint_manifest: {receipt_path}")
    checkpoint_manifest = Path(checkpoint_manifest_raw).expanduser().resolve()
    if not checkpoint_manifest.is_file() or checkpoint_manifest.is_symlink():
        raise PipelineContractError(
            f"checkpoint manifest is missing or unsafe: {checkpoint_manifest}"
        )
    checkpoint_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    if checkpoint_payload.get("schema") == "metis.distributed-checkpoint/v1":
        checkpoint_hash = _json_hash(checkpoint_payload, omit={"checkpoint_sha256"})
        if checkpoint_payload.get("checkpoint_sha256") != checkpoint_hash:
            raise PipelineContractError(
                f"distributed checkpoint failed its self-hash: {checkpoint_manifest}"
            )
        if checkpoint_payload.get("family") != family:
            raise PipelineContractError(
                f"distributed checkpoint has wrong family: {checkpoint_manifest}"
            )
        extra_state = _require_mapping(
            checkpoint_payload.get("extra_state", {}),
            f"{checkpoint_manifest}.extra_state",
        )
        if expected_stage != "base_pretraining" and (
            extra_state.get("posttraining_stage") != expected_stage
            or extra_state.get("parent_checkpoint_sha256") != expected_parent_sha256
            or extra_state.get("stage_config_sha256") != expected_config_sha256
        ):
            raise PipelineContractError(
                f"distributed checkpoint lacks post-training lineage: {checkpoint_manifest}"
            )
        records = _require_list(
            checkpoint_payload.get("artifacts"),
            f"{checkpoint_manifest}.artifacts",
        )
        seen: set[str] = set()
        for index, raw_record in enumerate(records):
            record = _require_mapping(
                raw_record,
                f"{checkpoint_manifest}.artifacts[{index}]",
            )
            raw_path = str(record.get("path", ""))
            if raw_path in seen:
                raise PipelineContractError(
                    f"distributed checkpoint duplicates artifact {raw_path}"
                )
            seen.add(raw_path)
            artifact = _resolve_relative(
                checkpoint_manifest.parent,
                raw_path,
                "distributed checkpoint artifact",
            )
            if not artifact.is_file():
                raise PipelineContractError(
                    f"distributed checkpoint artifact is missing: {artifact}"
                )
            if artifact.stat().st_size != int(record.get("bytes", -1)):
                raise PipelineContractError(
                    f"distributed checkpoint artifact size changed: {artifact}"
                )
            expected_hash = record.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise PipelineContractError(
                    f"distributed checkpoint artifact hash is invalid: {artifact}"
                )
            if verify_payload_hashes and _file_hash(artifact) != expected_hash:
                raise PipelineContractError(
                    f"distributed checkpoint artifact hash changed: {artifact}"
                )
    else:
        _, checkpoint_hash = _validate_sealed_artifact(
            checkpoint_manifest,
            expected_schema="metis.model-checkpoint/v1",
            tokenizer_sha256=None,
            verify_payload_hashes=verify_payload_hashes,
        )
    if receipt.get("checkpoint_sha256") != checkpoint_hash:
        raise PipelineContractError(f"checkpoint receipt hash does not match manifest: {receipt_path}")
    return receipt, receipt_hash


def _validate_stage_output_metadata(
    stage: Mapping[str, Any],
    output: Mapping[str, Any],
) -> None:
    stage_id = str(stage["id"])
    if stage_id != "deepseek_dpd_pilot" and "autotune" not in stage:
        return
    metadata = _require_mapping(
        output.get("metadata"),
        f"{stage_id} output metadata",
    )
    if (
        stage_id == "deepseek_dpd_pilot"
        and metadata.get("promotion_gate_passed") is not True
    ):
        raise PipelineContractError("DPD pilot did not pass its promotion gate")
    if (
        stage_id in SPECIALIST_STAGE_IDS
        and metadata.get("autotune_gate_passed") is not True
    ):
        raise PipelineContractError(f"{stage_id} did not pass its bounded autotune gate")
    profile_hash = metadata.get("selected_profile_sha256")
    if not isinstance(profile_hash, str) or len(profile_hash) != 64:
        raise PipelineContractError(f"{stage_id} did not seal its selected profile")


@dataclasses.dataclass(frozen=True)
class RequirementEvidence:
    name: str
    environment_variable: str
    path: str
    schema: str
    manifest_sha256: str


@dataclasses.dataclass(frozen=True)
class CompletedStage:
    stage_id: str
    output_receipt: str
    output_receipt_sha256: str
    checkpoint_receipt: str | None
    checkpoint_sha256: str | None
    requirements: tuple[RequirementEvidence, ...]
    completed_at_unix: int


class PostTrainingOrchestrator:
    """Fail-closed, resume-safe driver around a site/model-specific train backend."""

    def __init__(
        self,
        pipeline_path: str | Path,
        *,
        family: str,
        state_dir: str | Path,
        initial_checkpoint_receipt: str | Path,
        backend_command: str | Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        verify_payload_hashes: bool = True,
    ) -> None:
        if family not in EXPECTED_FAMILIES:
            raise PipelineContractError(f"unknown Metis family: {family}")
        self.pipeline_path = Path(pipeline_path).expanduser().resolve()
        self.repository_root = _repository_root(self.pipeline_path)
        self.pipeline = load_pipeline(self.pipeline_path)
        self.pipeline_sha256 = _file_hash(self.pipeline_path)
        self.family = family
        self.state_dir = Path(state_dir).expanduser().resolve() / family
        self.initial_checkpoint_receipt = (
            Path(initial_checkpoint_receipt).expanduser().resolve()
        )
        self.environment = dict(os.environ)
        if environment is not None:
            self.environment.update({str(key): str(value) for key, value in environment.items()})
        configured_backend = backend_command or self.environment.get("METIS_POSTTRAIN_BACKEND")
        if isinstance(configured_backend, str):
            self.backend_command = tuple(shlex.split(configured_backend))
        elif configured_backend is None:
            self.backend_command = ()
        else:
            self.backend_command = tuple(str(item) for item in configured_backend)
        self.verify_payload_hashes = verify_payload_hashes
        self.state_path = self.state_dir / "STATE.json"
        self.lock_path = self.state_dir / ".orchestrator.lock"

    @contextlib.contextmanager
    def _lock(self) -> Any:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _initial_state(self, checkpoint_sha256: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "pipeline_path": str(self.pipeline_path),
            "pipeline_sha256": self.pipeline_sha256,
            "family": self.family,
            "initial_checkpoint_receipt": str(self.initial_checkpoint_receipt),
            "initial_checkpoint_sha256": checkpoint_sha256,
            "completed": [],
            "state_sha256": "",
        }
        payload["state_sha256"] = _json_hash(payload, omit={"state_sha256"})
        return payload

    def _read_state(self, checkpoint_sha256: str) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state(checkpoint_sha256)
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != STATE_SCHEMA:
            raise PipelineContractError("post-training state schema changed")
        if payload.get("state_sha256") != _json_hash(payload, omit={"state_sha256"}):
            raise PipelineContractError("post-training state failed self-hash")
        if (
            payload.get("pipeline_sha256") != self.pipeline_sha256
            or payload.get("family") != self.family
            or payload.get("initial_checkpoint_sha256") != checkpoint_sha256
        ):
            raise PipelineContractError(
                "resume refused because pipeline, family, or base checkpoint changed"
            )
        return payload

    def _write_state(self, state: dict[str, Any]) -> None:
        state["state_sha256"] = _json_hash(state, omit={"state_sha256"})
        _atomic_json(self.state_path, state)

    def _tokenizer_evidence(self) -> tuple[dict[str, Any], str]:
        raw = self.environment.get("METIS_TOKENIZER_MANIFEST")
        if not raw:
            raise PipelineContractError("METIS_TOKENIZER_MANIFEST is required")
        return _validate_sealed_artifact(
            Path(raw),
            expected_schema="metis.tokenizer/v1",
            tokenizer_sha256=None,
            verify_payload_hashes=self.verify_payload_hashes,
        )

    def _requirements(
        self,
        stage: Mapping[str, Any],
        *,
        tokenizer_sha256: str,
        parent_stage: str,
        parent_checkpoint_sha256: str,
    ) -> tuple[RequirementEvidence, ...]:
        evidence: list[RequirementEvidence] = []
        for raw_requirement in _require_list(
            stage.get("requirements"),
            f"{stage['id']}.requirements",
        ):
            requirement = _require_mapping(raw_requirement, "stage requirement")
            base_environment_variable = str(requirement["env"])
            family_environment_variable = (
                f"{base_environment_variable}_{self.family.upper()}"
            )
            if requirement.get("family_bound") is True:
                environment_variable = family_environment_variable
            elif family_environment_variable in self.environment:
                environment_variable = family_environment_variable
            else:
                environment_variable = base_environment_variable
            raw_path = self.environment.get(environment_variable)
            if not raw_path:
                raise PipelineContractError(
                    f"{stage['id']} requires {environment_variable}; refusing to synthesize missing data"
                )
            payload, manifest_hash = _validate_sealed_artifact(
                Path(raw_path),
                expected_schema=str(requirement["schema"]),
                tokenizer_sha256=(
                    tokenizer_sha256 if requirement.get("tokenizer_bound", True) else None
                ),
                verify_payload_hashes=self.verify_payload_hashes,
            )
            metadata = _require_mapping(payload.get("metadata"), f"{raw_path}.metadata")
            minimum_records = requirement.get("minimum_records")
            if minimum_records is not None and int(metadata.get("records", -1)) < int(
                minimum_records
            ):
                raise PipelineContractError(
                    f"{environment_variable} has too few records for {stage['id']}"
                )
            minimum_tokens = requirement.get("minimum_tokens")
            if minimum_tokens is not None and int(metadata.get("tokens", -1)) < int(
                minimum_tokens
            ):
                raise PipelineContractError(
                    f"{environment_variable} has too few tokens for {stage['id']}"
                )
            maximum_tokens = requirement.get("maximum_tokens")
            if maximum_tokens is not None and int(metadata.get("tokens", -1)) > int(
                maximum_tokens
            ):
                raise PipelineContractError(
                    f"{environment_variable} exceeds the locked budget for {stage['id']}"
                )
            required_parent = requirement.get("generated_from_stage")
            if required_parent is not None and metadata.get("generated_from_stage") != parent_stage:
                raise PipelineContractError(
                    f"{environment_variable} must be generated fresh from {parent_stage}"
                )
            if requirement.get("family_bound") is True and metadata.get("family") != self.family:
                raise PipelineContractError(
                    f"{environment_variable} must be sealed specifically for {self.family}"
                )
            if (
                requirement.get("checkpoint_bound") is True
                and metadata.get("generated_from_checkpoint_sha256")
                != parent_checkpoint_sha256
            ):
                raise PipelineContractError(
                    f"{environment_variable} is not bound to the current {parent_stage} checkpoint"
                )
            for field_name, expected in _require_mapping(
                requirement.get("required_metadata", {}),
                "required_metadata",
            ).items():
                if metadata.get(field_name) != expected:
                    raise PipelineContractError(
                        f"{environment_variable} metadata {field_name!r} "
                        f"must be {expected!r}, got {metadata.get(field_name)!r}"
                    )
            evidence.append(
                RequirementEvidence(
                    name=str(requirement.get("name", environment_variable)),
                    environment_variable=environment_variable,
                    path=str(Path(raw_path).expanduser().resolve()),
                    schema=str(requirement["schema"]),
                    manifest_sha256=manifest_hash,
                )
            )
        return tuple(evidence)

    def _completed_by_id(self, state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        completed = _require_list(state.get("completed"), "state.completed")
        result: dict[str, Mapping[str, Any]] = {}
        for raw_record in completed:
            record = _require_mapping(raw_record, "completed stage")
            stage_id = str(record.get("stage_id", ""))
            if stage_id in result:
                raise PipelineContractError(f"state duplicates completed stage {stage_id}")
            result[stage_id] = record
        return result

    def _verify_completed_stage(
        self,
        record: Mapping[str, Any],
        *,
        stage: Mapping[str, Any],
        expected_stage: str,
        expected_parent_sha256: str,
        expected_config_sha256: str,
    ) -> str:
        if record.get("stage_id") != expected_stage:
            raise PipelineContractError("completed stage order is corrupt")
        receipt_path = Path(str(record.get("output_receipt", ""))).expanduser().resolve()
        if _file_hash(receipt_path) != record.get("output_receipt_sha256"):
            raise PipelineContractError(f"completed output receipt changed: {receipt_path}")
        output = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            output.get("schema") != STAGE_OUTPUT_SCHEMA
            or output.get("receipt_sha256")
            != _json_hash(output, omit={"receipt_sha256"})
            or output.get("family") != self.family
            or output.get("stage") != expected_stage
            or output.get("parent_checkpoint_sha256") != expected_parent_sha256
            or output.get("config_sha256") != expected_config_sha256
            or output.get("success") is not True
        ):
            raise PipelineContractError(f"completed output receipt lineage is invalid: {receipt_path}")
        _validate_stage_output_metadata(stage, output)
        checkpoint_receipt = output.get("checkpoint_receipt")
        if checkpoint_receipt:
            checkpoint, _ = _validate_checkpoint_receipt(
                Path(checkpoint_receipt),
                family=self.family,
                expected_stage=expected_stage,
                expected_parent_sha256=expected_parent_sha256,
                expected_config_sha256=expected_config_sha256,
                verify_payload_hashes=False,
            )
            if stage.get("preserves_policy_checkpoint") is True:
                return expected_parent_sha256
            return str(checkpoint["checkpoint_sha256"])
        return expected_parent_sha256

    def _runtime_spec(
        self,
        *,
        stage: Mapping[str, Any],
        parent_receipt: Path,
        parent_checkpoint_sha256: str,
        tokenizer_path: str,
        tokenizer_sha256: str,
        requirements: Sequence[RequirementEvidence],
        completed_stage_outputs: Mapping[str, str],
        config_sha256: str,
        output_receipt: Path,
    ) -> dict[str, Any]:
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "pipeline": str(self.pipeline_path),
            "pipeline_sha256": self.pipeline_sha256,
            "family": self.family,
            "stage": dict(stage),
            "stage_config_sha256": config_sha256,
            "parent_checkpoint_receipt": str(parent_receipt),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "tokenizer_manifest": tokenizer_path,
            "tokenizer_sha256": tokenizer_sha256,
            "requirements": [dataclasses.asdict(item) for item in requirements],
            "completed_stage_outputs": dict(completed_stage_outputs),
            "output_receipt": str(output_receipt),
            "runtime_sha256": "",
        }
        runtime["runtime_sha256"] = _json_hash(runtime, omit={"runtime_sha256"})
        return runtime

    def run(
        self,
        *,
        until: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if until is not None and until not in EXPECTED_STAGE_IDS:
            raise PipelineContractError(f"unknown stop stage: {until}")
        if not dry_run and not self.backend_command:
            raise PipelineContractError(
                "METIS_POSTTRAIN_BACKEND (or backend_command) is required; no trainer is guessed"
            )
        with self._lock():
            tokenizer, tokenizer_sha256 = self._tokenizer_evidence()
            initial, _ = _validate_checkpoint_receipt(
                self.initial_checkpoint_receipt,
                family=self.family,
                expected_stage="base_pretraining",
                verify_payload_hashes=self.verify_payload_hashes,
            )
            parent_checkpoint_sha256 = str(initial["checkpoint_sha256"])
            parent_receipt = self.initial_checkpoint_receipt
            state = self._read_state(parent_checkpoint_sha256)
            completed = self._completed_by_id(state)
            stages = _stages_by_id(self.pipeline)

            for stage_id in EXPECTED_STAGE_IDS:
                stage = stages[stage_id]
                config_sha256 = _json_hash(stage)
                requirements = self._requirements(
                    stage,
                    tokenizer_sha256=tokenizer_sha256,
                    parent_stage=str(stage["input_stage"]),
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                )
                if stage_id in completed:
                    parent_checkpoint_sha256 = self._verify_completed_stage(
                        completed[stage_id],
                        stage=stage,
                        expected_stage=stage_id,
                        expected_parent_sha256=parent_checkpoint_sha256,
                        expected_config_sha256=config_sha256,
                    )
                    output = json.loads(
                        Path(str(completed[stage_id]["output_receipt"])).read_text(
                            encoding="utf-8"
                        )
                    )
                    if (
                        output.get("checkpoint_receipt")
                        and stage.get("preserves_policy_checkpoint") is not True
                    ):
                        parent_receipt = Path(str(output["checkpoint_receipt"])).resolve()
                    if stage_id == until:
                        break
                    continue

                stage_dir = self.state_dir / "stages" / stage_id
                stage_dir.mkdir(parents=True, exist_ok=True)
                runtime_path = stage_dir / "RUNTIME.json"
                output_receipt = stage_dir / "OUTPUT.json"
                runtime = self._runtime_spec(
                    stage=stage,
                    parent_receipt=parent_receipt,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    tokenizer_path=str(
                        Path(self.environment["METIS_TOKENIZER_MANIFEST"]).resolve()
                    ),
                    tokenizer_sha256=tokenizer_sha256,
                    requirements=requirements,
                    completed_stage_outputs={
                        completed_stage_id: str(record["output_receipt"])
                        for completed_stage_id, record in completed.items()
                    },
                    config_sha256=config_sha256,
                    output_receipt=output_receipt,
                )
                _atomic_json(runtime_path, runtime)
                if dry_run:
                    if stage_id == until:
                        break
                    continue

                log_path = stage_dir / "backend.log"
                command = [*self.backend_command, "--runtime-spec", str(runtime_path)]
                started = int(time.time())
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.run(
                        command,
                        cwd=self.repository_root,
                        env=self.environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode != 0:
                    raise PipelineContractError(
                        f"{stage_id} backend exited {process.returncode}; see {log_path}"
                    )
                if not output_receipt.is_file() or output_receipt.is_symlink():
                    raise PipelineContractError(
                        f"{stage_id} backend returned without an output receipt"
                    )
                output = json.loads(output_receipt.read_text(encoding="utf-8"))
                if (
                    output.get("schema") != STAGE_OUTPUT_SCHEMA
                    or output.get("receipt_sha256")
                    != _json_hash(output, omit={"receipt_sha256"})
                    or output.get("success") is not True
                    or output.get("stage") != stage_id
                    or output.get("family") != self.family
                    or output.get("parent_checkpoint_sha256") != parent_checkpoint_sha256
                    or output.get("config_sha256") != config_sha256
                ):
                    raise PipelineContractError(f"{stage_id} returned an invalid output receipt")
                _validate_stage_output_metadata(stage, output)
                output_receipt_sha256 = _file_hash(output_receipt)
                checkpoint_receipt_raw = output.get("checkpoint_receipt")
                checkpoint_sha256: str | None = None
                if stage["output_kind"] in {"checkpoint", "reward_model"}:
                    if not checkpoint_receipt_raw:
                        raise PipelineContractError(
                            f"{stage_id} must return a checkpoint receipt"
                        )
                    checkpoint, _ = _validate_checkpoint_receipt(
                        Path(str(checkpoint_receipt_raw)),
                        family=self.family,
                        expected_stage=stage_id,
                        expected_parent_sha256=parent_checkpoint_sha256,
                        expected_config_sha256=config_sha256,
                        verify_payload_hashes=self.verify_payload_hashes,
                    )
                    checkpoint_sha256 = str(checkpoint["checkpoint_sha256"])
                    if stage.get("preserves_policy_checkpoint") is not True:
                        parent_checkpoint_sha256 = checkpoint_sha256
                        parent_receipt = Path(str(checkpoint_receipt_raw)).resolve()
                else:
                    artifact_manifest_raw = output.get("artifact_manifest")
                    if not artifact_manifest_raw:
                        raise PipelineContractError(
                            f"{stage_id} must return an artifact manifest"
                        )
                    expected_schema = {
                        "evaluation": "metis.evaluation-results/v1",
                        "publish": "metis.publish-candidate/v1",
                    }[str(stage["output_kind"])]
                    artifact, _ = _validate_sealed_artifact(
                        Path(str(artifact_manifest_raw)),
                        expected_schema=expected_schema,
                        tokenizer_sha256=tokenizer_sha256,
                        verify_payload_hashes=self.verify_payload_hashes,
                    )
                    artifact_metadata = _require_mapping(
                        artifact.get("metadata"),
                        f"{stage_id} artifact metadata",
                    )
                    if artifact_metadata.get("checkpoint_sha256") != parent_checkpoint_sha256:
                        raise PipelineContractError(
                            f"{stage_id} artifact is not bound to the current policy checkpoint"
                        )
                    if stage["output_kind"] == "evaluation" and (
                        artifact_metadata.get("gate_passed") is not True
                        or artifact_metadata.get("missing_metrics") not in ([], ())
                        or artifact_metadata.get("failed_metrics") not in ([], ())
                    ):
                        raise PipelineContractError(
                            "evaluation returned without a complete passing metric gate"
                        )
                    if stage["output_kind"] == "publish" and (
                        artifact_metadata.get("evaluation_gate_passed") is not True
                        or artifact_metadata.get("external_upload_performed") is not False
                    ):
                        raise PipelineContractError(
                            "publish gate may seal only a locally validated, non-uploaded candidate"
                        )

                record = CompletedStage(
                    stage_id=stage_id,
                    output_receipt=str(output_receipt),
                    output_receipt_sha256=output_receipt_sha256,
                    checkpoint_receipt=(
                        str(Path(str(checkpoint_receipt_raw)).resolve())
                        if checkpoint_receipt_raw
                        else None
                    ),
                    checkpoint_sha256=checkpoint_sha256,
                    requirements=tuple(requirements),
                    completed_at_unix=max(started, int(time.time())),
                )
                state["completed"].append(dataclasses.asdict(record))
                self._write_state(state)
                completed[stage_id] = state["completed"][-1]
                if stage_id == until:
                    break
            return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run the fail-closed Metis-1.6 post-training pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate the immutable YAML contract")
    validate.add_argument("--config", required=True)
    plan = subparsers.add_parser("plan", help="print the ordered stage plan")
    plan.add_argument("--config", required=True)
    run = subparsers.add_parser("run", help="run or resume one model-family pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--family", choices=EXPECTED_FAMILIES, required=True)
    run.add_argument("--state-dir", required=True)
    run.add_argument("--initial-checkpoint-receipt", required=True)
    run.add_argument("--backend")
    run.add_argument("--until", choices=EXPECTED_STAGE_IDS)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--trust-sealed-payload-hashes",
        action="store_true",
        help="verify manifests and sizes but skip re-hashing payload bytes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            pipeline = load_pipeline(args.config)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "schema": pipeline["schema"],
                        "stages": list(_stages_by_id(pipeline)),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "plan":
            pipeline = load_pipeline(args.config)
            for index, stage in enumerate(_stages_by_id(pipeline).values(), start=1):
                print(f"{index:02d} {stage['id']} <- {stage['input_stage']}")
            return 0
        if args.command == "run":
            orchestrator = PostTrainingOrchestrator(
                args.config,
                family=args.family,
                state_dir=args.state_dir,
                initial_checkpoint_receipt=args.initial_checkpoint_receipt,
                backend_command=args.backend,
                verify_payload_hashes=not args.trust_sealed_payload_hashes,
            )
            state = orchestrator.run(until=args.until, dry_run=args.dry_run)
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
    except (PipelineContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"metis-posttrain: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
