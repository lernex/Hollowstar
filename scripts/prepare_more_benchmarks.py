"""Freeze likelihood benchmark requests before allocating accelerator time."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def validate_suite(suite: Mapping[str, Any]) -> None:
    if suite["schema"] != "more.benchmark-suite/v1":
        raise ValueError("Unsupported benchmark suite schema")
    if suite["apply_chat_template"] is not False:
        raise ValueError("This suite evaluates base models without a chat template")
    if (
        not isinstance(suite["max_length"], int)
        or isinstance(suite["max_length"], bool)
        or suite["max_length"] < 1
    ):
        raise ValueError("The model input limit must be a positive integer")
    benchmarks = suite["benchmarks"]
    names = [benchmark["task"] for benchmark in benchmarks]
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Benchmark names must be nonempty and unique")
    workers = suite["workers"]
    if not workers or any(not worker for worker in workers):
        raise ValueError("Every configured benchmark worker must have work")
    assigned = [name for worker in workers for name in worker]
    if Counter(assigned) != Counter(names):
        raise ValueError("Workers must cover every benchmark exactly once")
    revisions = suite["dataset_revisions"]
    if set(revisions) != {benchmark["dataset"] for benchmark in benchmarks}:
        raise ValueError("Dataset pins must cover exactly the selected datasets")
    if any(re.fullmatch(r"[0-9a-f]{40}", revision) is None
           for revision in revisions.values()):
        raise ValueError("Every dataset revision must be an immutable commit SHA")
    for benchmark in benchmarks:
        shots = benchmark["num_fewshot"]
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 0:
            raise ValueError("Few-shot counts must be nonnegative integers")


def registered_leaves(index: Mapping[str, Any], name: str) -> list[str]:
    def walk(current: str, ancestors: frozenset[str]) -> list[str]:
        if current in ancestors:
            raise ValueError(f"Cycle in benchmark task hierarchy: {current}")
        entry = index[current]
        if entry.kind.name == "TAG":
            members = sorted(entry.tags)
        elif entry.kind.name == "GROUP":
            members = []
            for member in entry.cfg["task"]:
                if isinstance(member, str):
                    members.append(member)
                elif isinstance(member, dict):
                    members.append(member.get("group", member.get("task")))
                else:
                    raise ValueError(f"Unsupported member of task group {current}")
        elif entry.kind.name == "TASK":
            return [current]
        else:
            raise ValueError(f"Unsupported task kind for {current}")
        return [
            leaf
            for member in members
            for leaf in walk(member, ancestors | {current})
        ]

    leaves = walk(name, frozenset())
    if not leaves or len(leaves) != len(set(leaves)):
        raise ValueError(f"{name} must cover its leaf tasks exactly once")
    return sorted(leaves)


def request_input_length(tokenizer: Any, context: str, continuation: str) -> int:
    if not continuation:
        raise ValueError("A likelihood request must have a continuation")
    context_ids = tokenizer.encode(context.rstrip(), add_special_tokens=False).ids
    joined_ids = tokenizer.encode(
        context + continuation, add_special_tokens=False
    ).ids
    if not joined_ids or (
        context_ids and joined_ids[:len(context_ids)] != context_ids
    ):
        raise ValueError("Likelihood request has a non-prefix tokenization boundary")
    if len(joined_ids) <= len(context_ids):
        raise ValueError("The continuation must contain at least one token")
    # The final target is predicted rather than fed; an empty context needs EOS.
    return len(joined_ids) - 1 if context_ids else len(joined_ids)


def prepare(args: argparse.Namespace) -> None:
    suite = json.loads(args.suite.read_text())
    validate_suite(suite)
    if version("lm-eval") != suite["harness_version"]:
        raise ValueError("The installed harness does not match the pinned protocol")
    source_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("Prepare benchmarks from a clean, committed source checkout")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh preparation directory: {args.output}")
    args.output.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HOME": str(args.cache),
        "HF_HUB_CACHE": str(args.cache / "hub"),
        "HF_DATASETS_CACHE": str(args.cache / "datasets"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })

    from lm_eval.tasks import TaskManager
    from tokenizers import Tokenizer
    import yaml

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    manager = TaskManager(include_path=str(ROOT / "configs" / "more_eval_tasks"))
    index = manager.task_index
    specs = {benchmark["task"]: benchmark for benchmark in suite["benchmarks"]}
    leaf_owner: dict[str, str] = {}
    members: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        leaves = registered_leaves(index, name)
        if len(leaves) != spec.get("expected_leaf_tasks", 1):
            raise ValueError(f"Unexpected number of leaf tasks for {name}")
        if name == "mmlu":
            subjects = {index[leaf].cfg["dataset_name"] for leaf in leaves}
            if len(subjects) != len(leaves) or subjects & {"all", "auxiliary_train"}:
                raise ValueError("MMLU must cover each distinct subject exactly once")
        defaults = [index[leaf].cfg.get("dataset_kwargs") or {} for leaf in leaves]
        if any(kwargs != defaults[0] for kwargs in defaults):
            raise ValueError(f"{name} needs per-leaf dataset overrides")
        for leaf in leaves:
            if leaf in leaf_owner:
                raise ValueError(f"Leaf task belongs to multiple benchmarks: {leaf}")
            if index[leaf].cfg["dataset_path"] != spec["dataset"]:
                raise ValueError(f"Unexpected dataset for {leaf}")
            leaf_owner[leaf] = name
        key = "group" if index[name].kind.name == "GROUP" else "task"
        members[name] = {
            key: name,
            "num_fewshot": spec["num_fewshot"],
            "dataset_kwargs": {
                **defaults[0],
                "revision": suite["dataset_revisions"][spec["dataset"]],
            },
        }

    manifest: dict[str, Any] = {
        "schema": "more.benchmark-preparation/v1",
        "source_revision": source_revision,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "harness_version": version("lm-eval"),
        "max_length": suite["max_length"],
        "fewshot_seed": suite["fewshot_seed"],
        "cache_directory": str(args.cache),
        "dataset_revisions": suite["dataset_revisions"],
        "workers": [],
        "tasks": {},
    }
    total_overflow = 0
    pro_expected_ids: set[int] | None = None
    pro_observed_ids: Counter[int] = Counter()
    for worker_id, names in enumerate(suite["workers"]):
        group = f"more_dense_benchmarks_worker_{worker_id}"
        config_path = args.output / f"worker-{worker_id}.yaml"
        config_path.write_text(yaml.safe_dump(
            {"group": group, "task": [members[name] for name in names]},
            sort_keys=False,
        ))
        # Loading a YAML path preserves upstream groups and their aggregation rules.
        loaded = manager.load(str(config_path))
        expected = {leaf for leaf, owner in leaf_owner.items() if owner in names}
        if set(loaded["tasks"]) != expected:
            raise ValueError(f"Worker {worker_id} did not load exactly its assigned tasks")
        manifest["workers"].append({
            "worker": worker_id,
            "group": group,
            "task_config": str(config_path),
            "task_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "benchmarks": names,
            "leaf_tasks": sorted(expected),
        })
        for name, task in sorted(loaded["tasks"].items()):
            owner = leaf_owner[name]
            spec = specs[owner]
            revision = suite["dataset_revisions"][spec["dataset"]]
            if task.config.dataset_kwargs["revision"] != revision:
                raise ValueError(f"Dataset pin was not applied to {name}")
            if task.config.num_fewshot != spec["num_fewshot"]:
                raise ValueError(f"Few-shot override was not applied to {name}")
            count = len(task.eval_docs)
            if count == 0:
                raise ValueError(f"Empty evaluation split for {name}")
            if owner == "more_mmlu_pro_mc":
                raw_ids = task.dataset["test"]["question_id"]
                expected_ids = set(raw_ids)
                if len(expected_ids) != len(raw_ids):
                    raise ValueError("MMLU-Pro source question IDs are not unique")
                if pro_expected_ids is None:
                    pro_expected_ids = expected_ids
                elif expected_ids != pro_expected_ids:
                    raise ValueError("MMLU-Pro categories loaded different source data")
                pro_observed_ids.update(task.eval_docs["question_id"])
                subject = name.removeprefix("more_mmlu_pro_mc_").replace("_", " ")
                if set(task.eval_docs["category"]) != {subject}:
                    raise ValueError(f"Unexpected MMLU-Pro category for {name}")
                demonstrations = task.fewshot_docs()
                if (
                    len(demonstrations) != 5
                    or set(demonstrations["category"]) != {subject}
                ):
                    raise ValueError(f"{name} must have five category-matched examples")
            else:
                split = task.config.test_split or task.config.validation_split
                if count != len(task.dataset[split]):
                    raise ValueError(f"{name} filtered out evaluation documents")
            task.set_fewshot_seed(suite["fewshot_seed"])
            task.build_all_requests(
                limit=None, rank=0, world_size=1, cache_requests=False,
                apply_chat_template=False,
            )
            seen: set[tuple[int, int]] = set()
            document_ids: set[int] = set()
            digest = hashlib.sha256()
            max_input = 0
            overflow = 0
            for request in task.instances:
                if request.request_type != "loglikelihood" or request.repeats != 1:
                    raise ValueError(f"{name} is not a single-pass likelihood task")
                key = (request.doc_id, request.idx)
                if key in seen:
                    raise ValueError(f"Duplicate request index in {name}: {key}")
                seen.add(key)
                document_ids.add(request.doc_id)
                context, continuation = request.arguments
                length = request_input_length(tokenizer, context, continuation)
                max_input = max(max_input, length)
                overflow += length > suite["max_length"]
                digest.update(json.dumps(
                    [name, request.doc_id, request.idx, context, continuation],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8") + b"\n")
            if document_ids != set(range(count)):
                raise ValueError(f"{name} did not cover every evaluation document")
            total_overflow += overflow
            manifest["tasks"][name] = {
                "benchmark": owner,
                "dataset": task.DATASET_PATH,
                "dataset_config": task.DATASET_NAME,
                "dataset_revision": revision,
                "num_fewshot": task.config.num_fewshot,
                "evaluation_examples": count,
                "evaluation_fingerprint": task.eval_docs._fingerprint,
                "split_fingerprints": {
                    split: dataset._fingerprint
                    for split, dataset in task.dataset.items()
                },
                "requests": len(seen),
                "request_sha256": digest.hexdigest(),
                "max_model_input_tokens": max_input,
                "overflow_requests": overflow,
            }
            print(
                f"{name}: {count} examples, {len(seen)} requests, "
                f"max input {max_input}, overflows {overflow}", flush=True,
            )
    if pro_expected_ids is None or pro_observed_ids != Counter(pro_expected_ids):
        raise ValueError("MMLU-Pro categories must cover every source question exactly once")
    manifest["benchmarks"] = {
        name: {
            "leaf_tasks": sorted(
                leaf for leaf, record in manifest["tasks"].items()
                if record["benchmark"] == name
            ),
            "evaluation_examples": sum(
                record["evaluation_examples"] for record in manifest["tasks"].values()
                if record["benchmark"] == name
            ),
        }
        for name in specs
    }
    manifest["overflow_requests"] = total_overflow
    manifest["status"] = "blocked_context_overflow" if total_overflow else "ready"
    output = args.output / "preparation.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    if total_overflow:
        raise ValueError(
            f"{total_overflow} requests exceed the context limit; no examples "
            "were dropped or truncated. Choose and document a revised protocol."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs/more_eval_suite.json")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
