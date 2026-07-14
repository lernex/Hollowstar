from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def bucket_sum(config: dict[str, Any], key: str) -> int:
    return sum(int(bucket.get(key, 0)) for bucket in config.get("buckets", []))


def source_sum(bucket: dict[str, Any], key: str) -> int:
    return sum(int(source.get(key, 0)) for source in bucket.get("sources", []))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def iter_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [source for bucket in config.get("buckets", []) for source in bucket.get("sources", [])]


def source_blob(source: dict[str, Any]) -> str:
    return " ".join(
        str(source.get(key, ""))
        for key in ("name", "dataset_name", "dataset_config", "split", "format")
    ).lower()


def require_source_contains(
    config_path: Path,
    config: dict[str, Any],
    needles: list[str],
    errors: list[str],
    *,
    label: str,
) -> None:
    blobs = [source_blob(source) for source in iter_sources(config)]
    for needle in needles:
        normalized = needle.lower()
        require(
            any(normalized in blob for blob in blobs),
            f"{config_path}: missing required {label} source containing {needle!r}",
            errors,
        )


def require_bucket_targets(
    config_path: Path,
    config: dict[str, Any],
    key: str,
    expected: dict[str, int],
    errors: list[str],
) -> None:
    actual = {str(bucket.get("name")): int(bucket.get(key, 0)) for bucket in config.get("buckets", [])}
    for bucket_name, expected_value in expected.items():
        require(
            actual.get(bucket_name) == expected_value,
            f"{config_path}: bucket {bucket_name} has {actual.get(bucket_name):,} {key}, expected {expected_value:,}",
            errors,
        )


def source_total_matching(config: dict[str, Any], needle: str, key: str) -> int:
    normalized = needle.lower()
    return sum(int(source.get(key, 0)) for source in iter_sources(config) if normalized in source_blob(source))


def validate_bucketed_totals(
    *,
    config_path: Path,
    config: dict[str, Any],
    total_key: str,
    source_key: str,
    expected_total: int,
    errors: list[str],
    aggregate_key: str = "dataset_name",
) -> dict[str, Any]:
    actual_total = int(config.get(total_key, 0))
    require(
        actual_total == expected_total,
        f"{config_path}: {total_key}={actual_total:,} does not match manifest target {expected_total:,}",
        errors,
    )
    summed_buckets = bucket_sum(config, source_key)
    require(
        summed_buckets == expected_total,
        f"{config_path}: bucket {source_key} sum {summed_buckets:,} does not match expected {expected_total:,}",
        errors,
    )
    dataset_totals: dict[str, int] = defaultdict(int)
    bucket_summaries = []
    for bucket in config.get("buckets", []):
        bucket_name = str(bucket.get("name", "<unnamed>"))
        bucket_target = int(bucket.get(source_key, 0))
        summed_sources = source_sum(bucket, source_key)
        require(
            summed_sources == bucket_target,
            f"{config_path}: bucket {bucket_name} source sum {summed_sources:,} does not match bucket target {bucket_target:,}",
            errors,
        )
        fallback = bucket.get("fallback_policy", {})
        require(
            bool(fallback.get("same_bucket_only", False)),
            f"{config_path}: bucket {bucket_name} is missing fallback_policy.same_bucket_only=true",
            errors,
        )
        for source in bucket.get("sources", []):
            dataset_totals[str(source.get(aggregate_key, source.get("name", "<unknown>")))] += int(source.get(source_key, 0))
        bucket_summaries.append({"name": bucket_name, source_key: bucket_target})

    max_single_share = float(config.get("caps", {}).get("max_single_dataset_share", config.get("caps", {}).get("max_single_source_share", 1.0)))
    if max_single_share < 1.0 and expected_total > 0:
        for dataset_name, target in dataset_totals.items():
            share = target / expected_total
            require(
                share <= max_single_share + 1e-9,
                f"{config_path}: {dataset_name} has share {share:.3f}, above cap {max_single_share:.3f}",
                errors,
            )
    return {
        "path": str(config_path),
        "target": actual_total,
        "bucket_count": len(config.get("buckets", [])),
        "largest_dataset_share": max((tokens / expected_total for tokens in dataset_totals.values()), default=0.0),
        "buckets": bucket_summaries,
    }


def validate_example_config(
    *,
    config_path: Path,
    config: dict[str, Any],
    expected_total: int,
    errors: list[str],
) -> dict[str, Any]:
    return validate_bucketed_totals(
        config_path=config_path,
        config=config,
        total_key="target_examples",
        source_key="target_examples",
        expected_total=expected_total,
        errors=errors,
        aggregate_key="dataset_name",
    )


def validate_preference_config(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_preference: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    reward_target = int(manifest_preference["reward_model_examples"])
    dpo_target = int(manifest_preference["target_pairs"])
    reward_bootstrap = int(manifest_preference.get("bootstrap_reward_model_examples", reward_target))
    dpo_bootstrap = int(manifest_preference.get("bootstrap_target_pairs", dpo_target))

    require(
        int(config.get("target_reward_pairs", 0)) == reward_target,
        f"{config_path}: target_reward_pairs does not match manifest reward_model_examples",
        errors,
    )
    require(
        int(config.get("target_dpo_pairs", 0)) == dpo_target,
        f"{config_path}: target_dpo_pairs does not match manifest target_pairs",
        errors,
    )
    require(
        int(config.get("bootstrap_target_pairs", 0)) == reward_bootstrap,
        f"{config_path}: bootstrap_target_pairs does not match manifest bootstrap_reward_model_examples",
        errors,
    )

    summed_bootstrap = bucket_sum(config, "target_pairs")
    require(
        summed_bootstrap == reward_bootstrap,
        f"{config_path}: bootstrap bucket target_pairs sum {summed_bootstrap:,} does not match {reward_bootstrap:,}",
        errors,
    )
    for bucket in config.get("buckets", []):
        bucket_name = str(bucket.get("name", "<unnamed>"))
        bucket_target = int(bucket.get("target_pairs", 0))
        summed_sources = source_sum(bucket, "target_pairs")
        require(
            summed_sources == bucket_target,
            f"{config_path}: bucket {bucket_name} source sum {summed_sources:,} does not match {bucket_target:,}",
            errors,
        )
        require(
            bool(bucket.get("fallback_policy", {}).get("same_bucket_only", False)),
            f"{config_path}: bucket {bucket_name} is missing fallback_policy.same_bucket_only=true",
            errors,
        )

    on_policy = config.get("on_policy_generation", {})
    reward_on_policy = int(on_policy.get("reward_model_pairs", 0))
    dpo_on_policy = int(on_policy.get("dpo_pairs", 0))
    require(
        reward_bootstrap + reward_on_policy == reward_target,
        f"{config_path}: reward bootstrap {reward_bootstrap:,} + on-policy {reward_on_policy:,} != {reward_target:,}",
        errors,
    )
    require(
        dpo_bootstrap + dpo_on_policy == dpo_target,
        f"{config_path}: DPO bootstrap {dpo_bootstrap:,} + on-policy {dpo_on_policy:,} != {dpo_target:,}",
        errors,
    )
    nectar_target = sum(
        int(source.get("target_pairs", 0))
        for bucket in config.get("buckets", [])
        for source in bucket.get("sources", [])
        if "nectar" in str(source.get("name", "")).lower()
    )
    nectar_share = nectar_target / reward_target if reward_target else 0.0
    require(
        nectar_share <= float(config.get("caps", {}).get("nectar_max_share", 0.15)) + 1e-9,
        f"{config_path}: Nectar share {nectar_share:.3f} exceeds cap",
        errors,
    )
    return {
        "path": str(config_path),
        "reward_target": reward_target,
        "reward_bootstrap": reward_bootstrap,
        "reward_on_policy": reward_on_policy,
        "dpo_target": dpo_target,
        "dpo_bootstrap": dpo_bootstrap,
        "dpo_on_policy": dpo_on_policy,
        "bootstrap_bucket_count": len(config.get("buckets", [])),
        "nectar_share": nectar_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Metis-1.5 data-plan manifest and bucketed mixtures.")
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = repo_path(args.manifest, root=root)
    manifest = load_json(manifest_path)
    data_manifests = manifest["data_manifests"]
    errors: list[str] = []
    summaries: dict[str, Any] = {}

    pretrain_path = repo_path(data_manifests["pretrain_best_research"], root=root)
    pretrain_config = load_json(pretrain_path)
    require_bucket_targets(
        pretrain_path,
        pretrain_config,
        "target_tokens",
        {
            "high_quality_web_dclm": 12_000_000_000,
            "high_quality_web_fineweb": 9_000_000_000,
            "nemotron_quality_web": 5_000_000_000,
            "reference_encyclopedic": 3_000_000_000,
            "academic_stem_science": 5_000_000_000,
            "math_proof_symbolic": 6_000_000_000,
            "open_textbooks_educational_reference": 3_000_000_000,
            "long_form_books": 2_000_000_000,
            "knowledge_qa_explanations": 2_000_000_000,
            "synthetic_educational_prose": 2_000_000_000,
            "reserve_balancing_pool": 1_000_000_000,
        },
        errors,
    )
    require_source_contains(
        pretrain_path,
        pretrain_config,
        [
            "mlfoundations/dclm-baseline",
            "dclm-edu",
            "fineweb-edu",
            "fineweb-hq",
            "essential-web",
            "txt360",
            "zyda-2",
            "ultra-fineweb",
            "finewiki",
            "wikimedia",
            "pes2o",
            "proof-pile-2",
            "pubmed",
            "finepdfs",
            "nemotron-cc-math",
            "finemath",
            "open-web-math",
            "megamath",
            "openstax",
            "common_corpus",
            "pg19",
            "project_gutenberg",
            "stackexchange",
            "smollm-corpus",
        ],
        errors,
        label="base pretrain",
    )
    summaries["pretrain"] = validate_bucketed_totals(
        config_path=pretrain_path,
        config=pretrain_config,
        total_key="target_train_tokens",
        source_key="target_tokens",
        expected_total=int(manifest["pretrain"]["target_train_tokens"]),
        errors=errors,
    )

    continued_path = repo_path(data_manifests["continued_pretrain"], root=root)
    continued_config = load_json(continued_path)
    require_bucket_targets(
        continued_path,
        continued_config,
        "target_tokens",
        {
            "high_quality_general_replay": 1_200_000_000,
            "academic_stem": 2_000_000_000,
            "math_proof_documents": 2_000_000_000,
            "verifiable_math_problem_solution_prose": 1_200_000_000,
            "reference_wiki_stackexchange": 1_200_000_000,
            "finepdfs_technical_ocr": 800_000_000,
            "science_instruction_literature_text": 600_000_000,
            "long_form_book_replay": 500_000_000,
            "hard_general_decontaminated": 500_000_000,
        },
        errors,
    )
    require_source_contains(
        continued_path,
        continued_config,
        [
            "dclm",
            "fineweb",
            "pes2o",
            "proof-pile-2",
            "openstax",
            "pubmed",
            "finepdfs",
            "nemotron-cc-math",
            "finemath",
            "open-web-math",
            "megamath",
            "openr1-math",
            "numinamath",
            "openmathinstruct",
            "finewiki",
            "stackexchange",
            "sciriff",
            "sciinstruct",
            "pg19",
        ],
        errors,
        label="continued pretrain",
    )
    summaries["continued_pretrain"] = validate_bucketed_totals(
        config_path=continued_path,
        config=continued_config,
        total_key="target_train_tokens",
        source_key="target_tokens",
        expected_total=int(manifest["continued_pretrain"]["target_train_tokens"]),
        errors=errors,
    )

    chat_path = repo_path(data_manifests["chat_sft"], root=root)
    chat_config = load_json(chat_path)
    require_source_contains(
        chat_path,
        chat_config,
        [
            "tulu-3-sft",
            "smoltalk2",
            "wildchat",
            "sciriff",
            "sciinstruct",
            "openstax",
            "openhermes",
            "metis15_identity_manual",
        ],
        errors,
        label="chat SFT",
    )
    require(
        source_total_matching(chat_config, "openhermes", "target_examples") <= 80_000,
        f"{chat_path}: OpenHermes exceeds 80K cap",
        errors,
    )
    summaries["chat_sft"] = validate_example_config(
        config_path=chat_path,
        config=chat_config,
        expected_total=int(manifest["chat_sft"]["target_examples"]),
        errors=errors,
    )

    reasoning_path = repo_path(data_manifests["reasoning_sft"], root=root)
    reasoning_config = load_json(reasoning_path)
    require_source_contains(
        reasoning_path,
        reasoning_config,
        [
            "openthoughts3",
            "openr1-math",
            "numinamath-1.5",
            "numinamath-cot",
            "openmathinstruct",
            "sciriff",
            "templategsm",
        ],
        errors,
        label="reasoning SFT",
    )
    require(
        source_total_matching(reasoning_config, "templategsm", "target_examples") <= 25_000,
        f"{reasoning_path}: TemplateGSM exceeds 25K cap",
        errors,
    )
    summaries["reasoning_sft"] = validate_example_config(
        config_path=reasoning_path,
        config=reasoning_config,
        expected_total=int(manifest["reasoning_sft"]["target_examples"]),
        errors=errors,
    )

    preference_path = repo_path(data_manifests["preference"], root=root)
    preference_config = load_json(preference_path)
    require_source_contains(
        preference_path,
        preference_config,
        [
            "helpsteer3",
            "skywork-reward-preference-80k-v0.2",
            "ultrafeedback-binarized-preferences-cleaned",
            "smoltalk2",
            "olmo",
            "nectar",
            "openr1-math",
        ],
        errors,
        label="preference",
    )
    summaries["preference"] = validate_preference_config(
        config_path=preference_path,
        config=preference_config,
        manifest_preference=manifest["preference_optimization"],
        errors=errors,
    )

    release_repos = manifest.get("release", {}).get("repos", {})
    require("base" in release_repos, "Manifest release.repos must include base", errors)
    require("think" in release_repos, "Manifest release.repos must include think", errors)
    require("chat" not in release_repos, "Metis-1.5 should not define a chat release target", errors)

    output = {
        "manifest": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "summaries": summaries,
    }
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
