"""Evaluate a sealed, completed dense control using native lm-eval likelihoods.

Run ``python -m metis_ablation.evaluate --help`` in a separate evaluation
environment that can import the checkpoint's original native model source.
Only caller-selected likelihood tasks are run; generation tasks are rejected.
Task few-shot settings are left intact unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from tokenizers import Tokenizer

from metis_data.ngram_canonical import validate_canonical_id_sidecar
from metis_training import model as model_module
from metis_training.model import Metis16ForCausalLM
from metis_training.model_config import Metis16Config
from metis_training.precision import build_precision_policy

from .lm_eval_adapter import NativeDenseLM, as_harness_lm, fixed_dense_curriculum


def json_sha256(value: Any, *, ensure_ascii: bool = True) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii,
    ).encode("utf-8")).hexdigest()


def identify_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being hashed: {path}")
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest()}


def assert_unchanged(files: list[Mapping[str, Any]]) -> None:
    for identity in files:
        stat = Path(identity["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise RuntimeError(f"Evaluation input changed: {identity['path']}")


def load_checkpoint(source: Any, *, mmap: bool = True) -> Mapping[str, Any]:
    # TE extra state can be a BytesIO container. Permit that inert container,
    # not arbitrary checkpoint globals or an unrestricted pickle fallback.
    permitted = [] if io.BytesIO in torch.serialization.get_safe_globals() else [io.BytesIO]
    with torch.serialization.safe_globals(permitted):
        return torch.load(source, map_location="cpu", weights_only=True, mmap=mmap)


def validate_completed_checkpoint(
    payload: Mapping[str, Any], run: Mapping[str, Any], *, expected_run_identity: str,
) -> Mapping[str, Any]:
    if run.get("schema") != "more.ablation-run/v1":
        raise ValueError("Unsupported run manifest schema")
    identity = run.get("run_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema") != "more.ablation-run-identity/v1"
        or not re.fullmatch(r"[0-9a-f]{64}", expected_run_identity)
        or json_sha256(identity) != expected_run_identity
        or run.get("run_identity_sha256") != expected_run_identity
        or payload.get("run_identity_sha256") != expected_run_identity
    ):
        raise ValueError("Checkpoint/run identity does not match the externally pinned identity")
    if payload.get("schema") != "more.ablation-checkpoint/v3" or "optimizer" in payload:
        raise ValueError("Evaluation requires a model-only v3 state.pt; optimizer files are never loaded")
    if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
        raise ValueError("Checkpoint has no model state")
    if payload.get("step_semantics") != "next_unexecuted":
        raise ValueError("Checkpoint does not use completed-step semantics")
    if run.get("final_checkpoint") is not True or identity.get("row") != "dense-flop-matched":
        raise ValueError("Expected the final dense-flop-matched control")
    for key in ("model", "curriculum", "sampler", "schedule", "global_batch_tokens", "precision_profile"):
        if key not in identity or run.get(key) != identity[key]:
            raise ValueError(f"Run manifest {key} differs from its sealed identity")
    if payload.get("spec") != run.get("spec") or run["spec"].get("name") != identity["row"]:
        raise ValueError("Checkpoint and run specs differ")
    step = payload.get("step")
    counts = (step, payload.get("total_steps"), run.get("total_steps"),
              identity["schedule"].get("total_steps"), identity["sampler"].get("total_blocks"))
    if any(type(count) is not int or count < 1 or count != step for count in counts):
        raise ValueError("Checkpoint is not the completed schedule's final step")
    batch_tokens = identity["global_batch_tokens"]
    if (
        type(batch_tokens) is not int or batch_tokens < 1
        or identity["sampler"].get("sampled_tokens") != step * batch_tokens
        or payload.get("base_learning_rate") != identity["schedule"].get("base_learning_rate")
    ):
        raise ValueError("Completed checkpoint token or learning-rate accounting differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("source_revision", ""))):
        raise ValueError("Run identity has no exact source commit")
    return identity


def source_identity(expected_revision: str, *, checkpoint_revision: str | None = None) -> dict[str, Any]:
    repository = Path(model_module.__file__).resolve().parents[2]
    protected = (
        "src/metis_training", "src/metis_data/__init__.py",
        "src/metis_data/ngram_canonical.py", "src/metis_data/state.py",
    )

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "--no-pager", "-C", str(repository), *arguments],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    if git("rev-parse", f"{expected_revision}^{{commit}}") != expected_revision:
        raise RuntimeError("Checkpoint source commit is unavailable")
    if git("diff", "--name-only", expected_revision, "--", *protected):
        raise RuntimeError("Imported native model/tokenizer source differs from the checkpoint commit")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *protected)
    if any(path.endswith(".py") for path in untracked.splitlines()):
        raise RuntimeError("Untracked native model source prevents an exact source comparison")
    paths = git("ls-tree", "-r", "--name-only", expected_revision, "--", *protected).splitlines()
    files = [identify_file(repository / path) for path in paths if path.endswith(".py")]
    if not files:
        raise RuntimeError("No sealed native model source found")
    for name, module in list(sys.modules.items()):
        if name == "metis_training" or name.startswith("metis_training."):
            module_path = getattr(module, "__file__", None)
            if module_path and not Path(module_path).resolve().is_relative_to(repository / "src/metis_training"):
                raise RuntimeError("Native model modules are imported from mixed source trees")
    return {
        "repository": str(repository), "checkpoint_revision": checkpoint_revision or expected_revision,
        "verified_inference_revision": expected_revision,
        "evaluation_checkout_revision": git("rev-parse", "HEAD"),
        "model_source_matches_checkpoint": not bool(git(
            "diff", "--name-only", checkpoint_revision or expected_revision, "--", *protected,
        )),
        "files": files,
        "adapter_files": [identify_file(Path(__file__)), identify_file(Path(__file__).with_name("lm_eval_adapter.py"))],
    }


def load_release_tokenizer(
    root: Path, *, expected_release: Mapping[str, str], vocab_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    root = root.expanduser().resolve(strict=True)
    release_file = identify_file(root / "RELEASE.json")
    release = json.loads(Path(release_file["path"]).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in release.items() if key != "release_sha256"}
    if (
        release.get("schema") != "metis.data-release/v2"
        or release.get("verification", {}).get("ok") is not True
        or release.get("release_sha256") != json_sha256(unsigned, ensure_ascii=False)
        or any(release.get(key) != value for key, value in expected_release.items())
    ):
        raise ValueError("Tokenizer release is not the verified, run-bound release")

    def artifact(key: str) -> Path:
        raw = release.get("artifacts", {}).get(key)
        if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
            raise ValueError(f"Unsafe release artifact: {key}")
        path = root / raw
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe release artifact: {key}")
        return path

    tokenizer_path = artifact("tokenizer")
    manifest_path = artifact("ngram_canonical_map")
    binary_path = artifact("ngram_canonical_ids")
    records = {
        "tokenizer": identify_file(tokenizer_path),
        "ngram_canonical_map_manifest": identify_file(manifest_path),
        "ngram_canonical_ids": identify_file(binary_path),
    }
    for key, record in records.items():
        if record["sha256"] != release.get(f"{key}_sha256"):
            raise ValueError(f"Released {key} checksum differs")
    descriptor, lookup = validate_canonical_id_sidecar(
        manifest_path=manifest_path, binary_path=binary_path, tokenizer_path=tokenizer_path,
        expected_vocabulary_size=vocab_size,
        expected_manifest_sha256=release.get("ngram_canonical_map_self_sha256"),
        expected_binary_sha256=release.get("ngram_canonical_ids_sha256"),
        recompute_from_tokenizer=True,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    if len(vocabulary) != vocab_size or set(vocabulary.values()) != set(range(vocab_size)):
        raise ValueError("Released tokenizer does not exactly cover the model vocabulary")
    return tokenizer, lookup.astype("int64"), {
        "release": release_file, "release_sha256": release["release_sha256"],
        **records, "canonical_map_self_sha256": descriptor["manifest_sha256"],
        "canonical_semantics_recomputed": True,
    }


def special_token_id(tokenizer: Any, candidates: tuple[str, ...], *, required: bool) -> int | None:
    ids = {tokenizer.token_to_id(token) for token in candidates} - {None}
    if len(ids) > 1:
        raise ValueError("Tokenizer contains ambiguous special-token candidates")
    if not ids:
        if required:
            raise ValueError("Tokenizer is missing a recognized EOS token")
        return None
    return ids.pop()


def load_model(
    config: Metis16Config, weights: Mapping[str, Any], *,
    device: torch.device, checkpoint_precision: str,
):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The full dense checkpoint requires an explicit evaluation GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    # Construct the checkpoint's TE module layout so all weights and buffers
    # load strictly. Execute those same modules through the existing BF16
    # reference context; never discard FP8 extra-state keys or retile weights.
    policy = build_precision_policy(
        config.precision, profile=checkpoint_precision, device=device,
        production=False, permit_fallback=False,
    )
    model = Metis16ForCausalLM(config, precision_policy=policy, device=device, dtype=torch.bfloat16)
    model.apply_parameter_storage_policy(device=device)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.eval()
    for name, parameter in model.named_parameters():
        expected = torch.float32 if getattr(parameter, "metis_storage_dtype", None) == "float32" else torch.bfloat16
        if parameter.dtype != expected:
            raise RuntimeError(f"Checkpoint parameter storage policy differs: {name}")
    return model, policy


def validate_coverage(results: Mapping[str, Any], *, diagnostic: bool) -> dict[str, Any]:
    counts = results.get("n-samples")
    tasks = results.get("configs")
    if not isinstance(counts, Mapping) or not isinstance(tasks, Mapping) or not tasks:
        raise RuntimeError("Harness results do not contain verifiable per-task sample counts")
    for task in tasks:
        count = counts.get(task, {})
        original, effective = count.get("original"), count.get("effective")
        if (
            type(original) is not int or type(effective) is not int
            or not 0 < effective <= original
            or (not diagnostic and original != effective)
        ):
            raise RuntimeError(f"Harness did not evaluate the complete declared task: {task}")
    return {task: counts[task] for task in tasks}


def require_likelihood_tasks(loaded: Mapping[str, Any]) -> None:
    if not loaded.get("tasks"):
        raise ValueError("No likelihood tasks were selected")
    for name, task in loaded["tasks"].items():
        if task.get_config("output_type") not in {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}:
            raise ValueError(f"{name} is not likelihood-based; select the explicit non-CoT MC task")


def validate_prepared_requests(requests, expected: Mapping[str, Any]) -> None:
    digests: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for request in requests:
        name = request.task_name
        if name not in expected or request.request_type != "loglikelihood":
            raise ValueError("Scoring requests differ from the prepared likelihood suite")
        context, continuation = request.arguments
        digests.setdefault(name, hashlib.sha256()).update(json.dumps(
            [name, request.doc_id, request.idx, context, continuation],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8") + b"\n")
        counts[name] = counts.get(name, 0) + 1
    if set(digests) != set(expected):
        raise ValueError("Scoring requests do not cover exactly the prepared tasks")
    for name, record in expected.items():
        if counts[name] != record["requests"] or digests[name].hexdigest() != record["request_sha256"]:
            raise ValueError(f"Scoring requests changed after preparation: {name}")


def likelihood_task_manager(include_paths: list[Path], expected: Mapping[str, Any] | None = None):
    from lm_eval.tasks import TaskManager

    class LikelihoodTaskManager(TaskManager):
        def load(self, tasks):
            loaded = super().load(tasks)
            require_likelihood_tasks(loaded)
            if expected is not None:
                if set(loaded["tasks"]) != set(expected):
                    raise ValueError("Selected tasks do not match the prepared worker")
                for name, task in loaded["tasks"].items():
                    record = expected[name]
                    if (
                        task.DATASET_PATH != record["dataset"]
                        or task.DATASET_NAME != record["dataset_config"]
                        or task.config.dataset_kwargs["revision"] != record["dataset_revision"]
                        or task.config.num_fewshot != record["num_fewshot"]
                        or len(task.eval_docs) != record["evaluation_examples"]
                        or task.eval_docs._fingerprint != record["evaluation_fingerprint"]
                    ):
                        raise ValueError(f"Task data or protocol changed after preparation: {name}")
            return loaded

    return LikelihoodTaskManager(include_path=[str(path) for path in include_paths])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Final step-*/state.pt; never optimizer shards")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--expected-run-identity", required=True, help="Externally pinned run-identity SHA256")
    parser.add_argument("--inference-source-revision", help="Explicitly approved full source commit; default: checkpoint source")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--expected-harness-version", default="0.4.13", help="Exact lm-eval overlay version (default: 0.4.13)")
    parser.add_argument("--tasks", nargs="+", required=True, help="Harness task names, including custom likelihood tasks")
    parser.add_argument("--include-path", type=Path, action="append", default=[], help="Custom harness task directory; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory, outside checkpoint/run/release inputs")
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--preparation-worker", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-context-reuse", action="store_true", help="Disable exact one-token context reuse for parity checks")
    parser.add_argument("--num-fewshot", type=int, help="Explicit protocol override; otherwise preserve each task's defaults")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fewshot-seed", type=int, default=1234)
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--bos-token-id", type=int, help="Explicitly prepend this token; default: no BOS, matching packing")
    parser.add_argument("--diagnostic-limit", type=int, help="Diagnostic only: examples per task, never a full score")
    parser.add_argument("--diagnostic-max-context", type=int, help="Diagnostic native context override; no truncation is enabled")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("batch_size", "diagnostic_limit", "diagnostic_max_context"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if (args.num_fewshot is not None and args.num_fewshot < 0) or args.bootstrap_iters < 0:
        raise ValueError("Few-shot and bootstrap counts cannot be negative")
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("LOCAL_RANK", "0")) != 0
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("This CLI is single-process; do not launch under distributed training")
    import lm_eval
    from lm_eval.utils import handle_non_serializable
    from packaging.version import Version

    harness_version = importlib.metadata.version("lm-eval")
    if Version(harness_version) < Version("0.4.13"):
        raise ValueError("Native evaluation requires lm-eval >= 0.4.13")
    if harness_version != args.expected_harness_version:
        raise ValueError(f"Expected lm-eval {args.expected_harness_version}, found {harness_version}")
    checkpoint = identify_file(args.checkpoint)
    run_file = identify_file(args.run_manifest)
    run = json.loads(Path(run_file["path"]).read_text(encoding="utf-8"))
    payload = load_checkpoint(checkpoint["path"])
    identity = validate_completed_checkpoint(payload, run, expected_run_identity=args.expected_run_identity)
    sources = source_identity(
        args.inference_source_revision or identity["source_revision"],
        checkpoint_revision=identity["source_revision"],
    )
    config = Metis16Config.from_mapping(identity["model"])
    curriculum = fixed_dense_curriculum(config, identity["curriculum"])
    trained_context = config.sequence_length
    if config.final_context_length != trained_context:
        raise ValueError("Saved evaluation/training context lengths differ; protocol approval is required")
    if args.diagnostic_max_context is not None:
        config = replace(
            config, final_context_length=args.diagnostic_max_context,
            context_extension_train_length=max(config.context_extension_train_length, args.diagnostic_max_context),
        )
        config.validate()
    tokenizer, lookup, release = load_release_tokenizer(
        args.release_root, expected_release=identity["release"], vocab_size=config.vocab_size,
    )
    eos = special_token_id(tokenizer, ("<|endoftext|>", "<eos>", "</s>"), required=True)
    pad = special_token_id(tokenizer, ("<|padding|>", "<pad>", "<|pad|>"), required=False)
    include_paths = [path.expanduser().resolve(strict=True) for path in args.include_path]
    custom_task_files = []
    for path in include_paths:
        if not path.is_dir():
            raise ValueError("--include-path must name a custom task directory")
        custom_task_files.extend(
            identify_file(file) for file in sorted(path.rglob("*"))
            if file.is_file() and (
                file.suffix in {".yaml", ".yml", ".py", ".json"}
                or file.name.endswith("_yaml")
            )
        )
    preparation = None
    expected_tasks = None
    if (args.preparation_manifest is None) != (args.preparation_worker is None):
        raise ValueError("Provide both preparation manifest and worker index")
    if args.preparation_manifest is not None:
        preparation_file = identify_file(args.preparation_manifest)
        preparation = json.loads(Path(preparation_file["path"]).read_text())
        if (
            preparation["schema"] != "more.benchmark-preparation/v1"
            or preparation["status"] != "ready"
            or preparation["overflow_requests"] != 0
            or preparation["tokenizer_sha256"] != release["tokenizer"]["sha256"]
            or preparation["harness_version"] != harness_version
            or preparation["max_length"] != trained_context
            or preparation["fewshot_seed"] != args.fewshot_seed
            or args.num_fewshot is not None
        ):
            raise ValueError("Prepared protocol does not match evaluation")
        workers = [worker for worker in preparation["workers"]
                   if worker["worker"] == args.preparation_worker]
        if len(workers) != 1 or args.tasks != [workers[0]["group"]]:
            raise ValueError("Select exactly the prepared worker group")
        expected_tasks = {
            name: preparation["tasks"][name] for name in workers[0]["leaf_tasks"]
        }
        custom_task_files.append(preparation_file)
    task_manager = likelihood_task_manager(include_paths, expected_tasks)
    output = args.output_dir.expanduser().resolve()
    for protected in (Path(checkpoint["path"]).parent, Path(run_file["path"]).parent, args.release_root.resolve()):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Evaluation output must be outside checkpoint/run/release input directories")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [checkpoint, run_file, *sources["files"], *sources["adapter_files"], *custom_task_files]
    inputs.extend(value for value in release.values() if isinstance(value, dict) and "mtime_ns" in value)
    diagnostic = any(value is not None for value in (
        args.diagnostic_limit, args.diagnostic_max_context,
    ))
    packages = {}
    for name in ("lm-eval", "torch", "tokenizers", "transformers", "datasets",
                 "mamba-ssm", "causal-conv1d", "flash-attn", "transformer-engine", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata: dict[str, Any] = {
        "schema": "more.native-lm-eval/v1", "status": "running",
        "diagnostic": diagnostic, "full_score": False, "checkpoint": checkpoint,
        "run_manifest": run_file, "run_identity_sha256": args.expected_run_identity,
        "completed_steps": payload["step"],
        "training_tokens": identity["sampler"]["sampled_tokens"],
        "training_parameter_audit": run.get("parameters"),
        "source": sources, "tokenizer_release": release, "packages": packages,
        "python": sys.version, "platform": platform.platform(),
        "harness_api_files": [identify_file(Path(lm_eval.__file__)),
                              identify_file(Path(lm_eval.__file__).parent / "evaluator.py"),
                              identify_file(Path(lm_eval.__file__).parent / "api/model.py")],
        "options": vars(args), "saved_curriculum": identity["curriculum"],
        "effective_curriculum": asdict(curriculum),
        "forward_max_passes": 1, "forward_force_depth": 1,
        "trained_context": trained_context, "effective_model_config": config.to_dict(),
        "execution_precision": "bf16", "constructed_precision": identity["precision_profile"],
        "optimizer_loaded": False, "optimizer_shards_opened": 0,
        "sample_logging": False, "request_cache": False,
        "evaluation_kind": "likelihood_only", "custom_task_files": custom_task_files,
        "preparation_worker": args.preparation_worker,
        "prepared_request_identity_checked": False,
    }

    def write_metadata() -> None:
        temporary = output / "metadata.json.tmp"
        temporary.write_text(
            json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8",
        )
        temporary.replace(output / "metadata.json")

    last_progress = 0.0

    def progress(stats: Mapping[str, int | float]) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress < 30:
            return
        last_progress = now
        record = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **stats}
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output / "progress.json")
        print(f"Scored {stats['completed_likelihood_pairs']} likelihood requests; "
              f"{stats['forward_calls']} native forwards", flush=True)

    def check_requests(requests) -> None:
        if expected_tasks is None:
            raise RuntimeError("Prepared request validation has no task census")
        validate_prepared_requests(requests, expected_tasks)
        metadata["prepared_request_identity_checked"] = True

    write_metadata()
    native = None
    started = time.perf_counter()
    try:
        model, policy = load_model(
            config, payload["model"], device=torch.device(args.device),
            checkpoint_precision=identity["precision_profile"],
        )
        del payload
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != run.get("parameters", {}).get("stored_total"):
            raise ValueError("Loaded model parameter count differs from the completed training run")
        native = NativeDenseLM(
            model, tokenizer, curriculum=curriculum, canonical_id_lookup=lookup,
            eos_token_id=eos, pad_token_id=pad, bos_token_id=args.bos_token_id,
            batch_size=args.batch_size, device=args.device,
            precision_context=policy.bf16_reference_context,
            reuse_single_token_contexts=not args.no_context_reuse,
            request_validator=check_requests if expected_tasks is not None and not diagnostic else None,
            progress_callback=progress,
        )
        metadata["parameter_count"] = actual_parameters
        metadata["precision_backend_audit"] = policy.audit.to_dict()
        metadata["device_name"] = torch.cuda.get_device_name(torch.device(args.device))
        print("Likelihood-only evaluation; exact one-token context reuse is "
              f"{'disabled' if args.no_context_reuse else 'enabled'}. Context overflow fails without truncation.",
              file=sys.stderr)
        tasks = [name for entry in args.tasks for name in entry.split(",") if name]
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("Select nonempty, unique official task names")
        results = lm_eval.simple_evaluate(
            model=as_harness_lm(native), tasks=tasks, batch_size=args.batch_size, device=args.device,
            num_fewshot=args.num_fewshot, limit=args.diagnostic_limit,
            bootstrap_iters=args.bootstrap_iters, log_samples=False, write_out=False,
            apply_chat_template=False, fewshot_as_multiturn=False,
            cache_requests=False, use_cache=None,
            random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
            fewshot_random_seed=args.fewshot_seed,
            task_manager=task_manager,
        )
        if not isinstance(results, Mapping) or "samples" in results:
            raise RuntimeError("Expected aggregate-only harness results")
        if expected_tasks is not None and not diagnostic and not metadata["prepared_request_identity_checked"]:
            raise RuntimeError("Full evaluation did not validate its prepared requests")
        metadata["sample_counts"] = validate_coverage(results, diagnostic=diagnostic)
        assert_unchanged(inputs)
        raw = output / "harness-results.json"
        raw.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                  default=handle_non_serializable) + "\n", encoding="utf-8")
        metadata["harness_results"] = identify_file(raw)
        metadata["status"] = "completed"
        metadata["full_score"] = not diagnostic
    finally:
        if metadata["status"] != "completed":
            metadata["status"] = "failed"
        metadata["elapsed_seconds"] = time.perf_counter() - started
        if native is not None:
            metadata["adapter"] = native.metadata()
        write_metadata()
    print(f"Completed {'diagnostic' if diagnostic else 'full'} evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
