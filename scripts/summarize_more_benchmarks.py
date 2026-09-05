"""Publish aggregate dense scores only after all prepared workers finish."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_RUN_IDENTITY = "736f74c8a6fe978d535cea0ce9f2905acaee43f957646abcada76c54512797e6"


def assemble_results(
    suite: Mapping[str, Any],
    preparation: Mapping[str, Any],
    submission: Mapping[str, Any],
    workers: Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    expected_workers = {worker["worker"] for worker in preparation["workers"]}
    if set(workers) != expected_workers or preparation["status"] != "ready":
        raise ValueError("All prepared workers must be present")
    if len(expected_workers) != len(suite["workers"]):
        raise ValueError("Worker geometry differs from the frozen suite")
    expected_by_worker = {
        worker["worker"]: set(worker["leaf_tasks"]) for worker in preparation["workers"]
    }
    covered: set[str] = set()
    checkpoint_sha = None
    benchmark_owner = {}
    for worker in preparation["workers"]:
        for name in worker["benchmarks"]:
            if name in benchmark_owner:
                raise ValueError("A benchmark was assigned to multiple workers")
            benchmark_owner[name] = worker["worker"]
    specs = {spec["task"]: spec for spec in suite["benchmarks"]}
    if set(benchmark_owner) != set(specs):
        raise ValueError("Worker assignment does not cover the requested benchmarks")
    for worker_id, (metadata, raw) in workers.items():
        if (
            metadata["schema"] != "more.native-lm-eval/v1"
            or metadata["status"] != "completed"
            or metadata["diagnostic"] is not False
            or metadata["full_score"] is not True
            or metadata["prepared_request_identity_checked"] is not True
            or metadata["optimizer_loaded"] is not False
            or metadata["optimizer_shards_opened"] != 0
            or metadata["run_identity_sha256"] != EXPECTED_RUN_IDENTITY
            or metadata["preparation_worker"] != worker_id
            or metadata["source"]["model_source_matches_checkpoint"] is not True
            or metadata["source"]["verified_inference_revision"] != submission["native_source_revision"]
            or metadata["execution_precision"] != "bf16"
            or metadata["forward_max_passes"] != 1
            or metadata["forward_force_depth"] != 1
            or metadata["effective_curriculum"]["continuation_mode"] != "depth_one"
            or metadata["effective_curriculum"]["memory_gate_scale"] != 0.0
            or metadata["effective_curriculum"]["ngram_gate_scale"] != 1.0
            or metadata["tokenizer_release"]["canonical_semantics_recomputed"] is not True
            or metadata["tokenizer_release"]["tokenizer"]["sha256"] != preparation["tokenizer_sha256"]
            or metadata["packages"]["lm-eval"] != suite["harness_version"]
        ):
            raise ValueError(f"Worker {worker_id} is not a completed, source-matched full score")
        if "samples" in raw:
            raise ValueError("Benchmark samples must not be published")
        current_sha = metadata["checkpoint"]["sha256"]
        if checkpoint_sha is None:
            checkpoint_sha = current_sha
        elif current_sha != checkpoint_sha:
            raise ValueError("Workers evaluated different checkpoints")
        expected_tasks = expected_by_worker[worker_id]
        if set(raw["configs"]) != expected_tasks or covered & expected_tasks:
            raise ValueError("Worker task coverage differs from preparation")
        covered.update(expected_tasks)
        for name in expected_tasks:
            record = preparation["tasks"][name]
            expected_count = record["evaluation_examples"]
            if (
                raw["n-samples"][name] != {"original": expected_count, "effective": expected_count}
                or raw["n-shot"][name] != record["num_fewshot"]
                or raw["configs"][name]["dataset_kwargs"]["revision"] != record["dataset_revision"]
            ):
                raise ValueError(f"Task coverage or protocol differs: {name}")
        expected_requests = sum(preparation["tasks"][name]["requests"] for name in expected_tasks)
        stats = metadata["adapter"]["stats"]
        if (
            stats["loglikelihood_requests"] != expected_requests
            or stats["completed_likelihood_pairs"] != expected_requests
            or stats["context_overflows"] != 0
            or stats["boundary_errors"] != 0
            or stats["generation_requests"] != 0
        ):
            raise ValueError(f"Worker {worker_id} did not score exactly its prepared requests")
    if covered != set(preparation["tasks"]):
        raise ValueError("The completed workers do not cover the prepared task census")

    rows = []
    for spec in suite["benchmarks"]:
        name = spec["task"]
        _metadata, raw = workers[benchmark_owner[name]]
        census = preparation["benchmarks"][name]
        if len(census["leaf_tasks"]) > 1:
            metric_row = raw["groups"][name]
        else:
            metric_row = raw["results"][name]
        metrics = {
            key: value for key, value in metric_row.items()
            if key.endswith(",none") and "_stderr," not in key
        }
        for metric, value in metrics.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite benchmark score: {name} {metric}")
            if metric.startswith("acc") and not 0 <= value <= 1:
                raise ValueError(f"Accuracy outside [0, 1]: {name}")
            if metric == "perplexity,none" and value < 1:
                raise ValueError(f"Perplexity below one: {name}")
        if spec["primary_metric"] not in metrics or "acc,none" not in metrics:
            raise ValueError(f"Missing requested benchmark metric: {name}")
        rows.append({
            **spec,
            "evaluation_examples": census["evaluation_examples"],
            "metrics": metrics,
            "standard_errors": {
                key: value for key, value in metric_row.items() if "_stderr," in key
            },
            "score_percent": 100 * metrics[spec["primary_metric"]],
        })
    first = workers[min(workers)][0]
    if any(
        metadata[key] != first[key]
        for metadata, _raw in workers.values()
        for key in ("run_identity_sha256", "completed_steps", "training_tokens", "parameter_count")
    ):
        raise ValueError("Worker training identities differ")
    return {
        "schema": "more.benchmark-results/v1",
        "status": "completed",
        "row": "dense-flop-matched",
        "job_id": submission["job_id"],
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_revision": submission["exports"]["METIS_EVAL_REVISION"],
        "native_source_revision": submission["native_source_revision"],
        "checkpoint_sha256": checkpoint_sha,
        "run_identity_sha256": first["run_identity_sha256"],
        "completed_steps": first["completed_steps"],
        "training_tokens": first["training_tokens"],
        "stored_parameters": first["parameter_count"],
        "harness_version": suite["harness_version"],
        "tokenizer_sha256": preparation["tokenizer_sha256"],
        "suite_sha256": preparation["suite_sha256"],
        "context_length": preparation["max_length"],
        "execution_precision": "bf16",
        "chat_template": False,
        "fewshot_seed": preparation["fewshot_seed"],
        "evaluation_examples": sum(row["evaluation_examples"] for row in rows),
        "benchmarks": rows,
    }


def render_table(result: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Benchmark & Shots & Examples & Acc.\ (\%) & Acc.\ norm.\ (\%) \\",
        r"\midrule",
    ]
    for row in result["benchmarks"]:
        accuracy = 100 * row["metrics"]["acc,none"]
        normalized = row["metrics"].get("acc_norm,none")
        normalized_text = "--" if normalized is None else f"{100 * normalized:.2f}"
        label = row["label"]
        if any(character in label for character in "\\{}&%$#_^~\n"):
            raise ValueError("Benchmark label requires explicit LaTeX escaping")
        lines.append(
            f"{label} & {row['num_fewshot']} & {row['evaluation_examples']:,} & "
            f"{accuracy:.2f} & {normalized_text} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Completed dense FLOP-matched reference after 49.995B training tokens.",
        r"Full evaluation splits; raw and harness choice-length-normalized accuracies",
        r"are shown where available. MMLU-Pro uses direct MC likelihood, not CoT.}",
        r"\label{tab:dense-flop-benchmarks}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    preparation = json.loads((args.report / "preparation.json").read_text())
    submission = json.loads((args.report / "submission.json").read_text())
    if hashlib.sha256(args.suite.read_bytes()).hexdigest() != preparation["suite_sha256"]:
        raise ValueError("Suite definition changed after preparation")
    workers = {}
    for worker in preparation["workers"]:
        directory = args.report / "workers" / f"worker-{worker['worker']}"
        metadata = json.loads((directory / "metadata.json").read_text())
        raw_path = directory / "harness-results.json"
        with raw_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            digest != metadata["harness_results"]["sha256"]
            or raw_path.stat().st_size != metadata["harness_results"]["bytes"]
        ):
            raise ValueError("Archived harness result differs from its completion record")
        workers[worker["worker"]] = (metadata, json.loads(raw_path.read_text()))
    result = assemble_results(suite, preparation, submission, workers)
    (args.report / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.table.write_text(render_table(result))


if __name__ == "__main__":
    main()
