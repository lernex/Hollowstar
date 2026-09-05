import importlib.util
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "configs" / "more_eval_tasks" / "mmlu_pro_mc"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "more_mmlu_pro_mc_utils", TASK_ROOT / "utils.py"
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("Cannot load the MMLU-Pro prompt helpers")
utils = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(utils)
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_more_benchmarks", ROOT / "scripts" / "prepare_more_benchmarks.py"
)
if PREPARE_SPEC is None or PREPARE_SPEC.loader is None:
    raise RuntimeError("Cannot load the benchmark preparation helpers")
prepare = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(prepare)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_more_benchmarks", ROOT / "scripts" / "summarize_more_benchmarks.py"
)
if SUMMARY_SPEC is None or SUMMARY_SPEC.loader is None:
    raise RuntimeError("Cannot load the benchmark summary helpers")
summary = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(summary)


class MoreEvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.doc = {
            "question": "Which label names the second option?",
            "options": [" first ", " second "],
            "answer": "B",
            "answer_index": 1,
            "cot_content": "This rationale must never appear in the prompt.",
        }

    def test_mc_prompt_has_options_but_no_rationale_or_answer(self):
        self.assertEqual(
            utils.doc_to_text(self.doc),
            "Question:\nWhich label names the second option?\n"
            "Options:\nA. first\nB. second\nAnswer:",
        )
        self.assertEqual(utils.doc_to_choice(self.doc), ["A", "B"])
        self.assertEqual(utils.doc_to_target(self.doc), 1)
        self.assertNotIn(self.doc["cot_content"], utils.doc_to_text(self.doc))

    def test_ten_options_retain_the_last_choice(self):
        doc = dict(self.doc, options=[str(index) for index in range(10)],
                   answer="J", answer_index=9)
        self.assertEqual(utils.doc_to_choice(doc)[-1], "J")
        self.assertEqual(utils.doc_to_target(doc), 9)
        self.assertIn("\nJ. 9\n", utils.doc_to_text(doc))

    def test_invalid_options_and_answer_mapping_fail_closed(self):
        for options in ([], ["only"], [""] * 2, ["x"] * 11):
            with self.subTest(options=options), self.assertRaises(ValueError):
                utils.doc_to_text(dict(self.doc, options=options))
        for update in (
            {"answer": "A"}, {"answer_index": 2}, {"answer_index": -1},
            {"answer_index": True},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                utils.doc_to_target(dict(self.doc, **update))

    def test_each_official_subject_has_exactly_one_leaf_and_group_entry(self):
        subjects = {
            "biology", "business", "chemistry", "computer_science",
            "economics", "engineering", "health", "history", "law", "math",
            "other", "philosophy", "physics", "psychology",
        }
        leaves = {path.stem for path in TASK_ROOT.glob("*.yaml")
                  if not path.name.startswith("_")}
        self.assertEqual(leaves, subjects)
        group = (TASK_ROOT / "_more_mmlu_pro_mc.yaml").read_text()
        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertEqual(
                    group.count(f"  - more_mmlu_pro_mc_{subject}\n"), 1
                )
                leaf = (TASK_ROOT / f"{subject}.yaml").read_text()
                self.assertIn(f"task: more_mmlu_pro_mc_{subject}\n", leaf)
                self.assertIn(f"process_docs: !function utils.process_{subject}\n", leaf)
                self.assertEqual(
                    getattr(utils, f"process_{subject}").keywords["subject"],
                    subject.replace("_", " "),
                )

    def test_worker_assignment_covers_the_requested_ten_benchmarks_once(self):
        suite = json.loads((ROOT / "configs/more_eval_suite.json").read_text())
        prepare.validate_suite(suite)
        self.assertEqual(
            {item["task"] for item in suite["benchmarks"]},
            {"mmlu", "more_mmlu_pro_mc", "arc_easy", "arc_challenge",
             "hellaswag", "winogrande", "piqa", "boolq", "openbookqa",
             "lambada_openai"},
        )
        for mutation in ("drop", "duplicate"):
            changed = copy.deepcopy(suite)
            if mutation == "drop":
                changed["workers"][0].pop()
            else:
                changed["workers"][1].append("mmlu")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare.validate_suite(changed)

    def test_nested_group_coverage_rejects_duplicates_and_cycles(self):
        index = {
            "root": SimpleNamespace(
                kind=SimpleNamespace(name="GROUP"), cfg={"task": ["tag", "b"]}
            ),
            "tag": SimpleNamespace(kind=SimpleNamespace(name="TAG"), tags={"a"}),
            "a": SimpleNamespace(kind=SimpleNamespace(name="TASK")),
            "b": SimpleNamespace(kind=SimpleNamespace(name="TASK")),
        }
        self.assertEqual(prepare.registered_leaves(index, "root"), ["a", "b"])
        index["tag"].tags.add("b")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            prepare.registered_leaves(index, "root")
        index["root"].cfg["task"] = ["root"]
        with self.assertRaisesRegex(ValueError, "Cycle"):
            prepare.registered_leaves(index, "root")

    def test_input_length_counts_prediction_and_empty_context_correctly(self):
        class CharacterTokenizer:
            def encode(self, text, *, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("Unexpected special-token insertion")
                return SimpleNamespace(ids=list(text))

        tokenizer = CharacterTokenizer()
        self.assertEqual(prepare.request_input_length(tokenizer, "abc", " d"), 4)
        self.assertEqual(prepare.request_input_length(tokenizer, "abc ", "d"), 4)
        self.assertEqual(prepare.request_input_length(tokenizer, "", "ab"), 2)
        with self.assertRaisesRegex(ValueError, "continuation"):
            prepare.request_input_length(tokenizer, "abc", "")


class DenseScoreSummaryTests(unittest.TestCase):
    def fixture(self):
        suite = {
            "workers": [["mmlu"], ["arc_easy"]],
            "harness_version": "0.4.13",
            "benchmarks": [
                {"task": "mmlu", "label": "MMLU", "num_fewshot": 5,
                 "primary_metric": "acc,none"},
                {"task": "arc_easy", "label": "ARC-Easy", "num_fewshot": 0,
                 "primary_metric": "acc_norm,none"},
            ],
        }
        tasks = {
            "a": {"evaluation_examples": 1, "requests": 4, "num_fewshot": 5,
                  "dataset_revision": "a" * 40},
            "b": {"evaluation_examples": 3, "requests": 12, "num_fewshot": 5,
                  "dataset_revision": "a" * 40},
            "arc_easy": {"evaluation_examples": 2, "requests": 8, "num_fewshot": 0,
                         "dataset_revision": "b" * 40},
        }
        preparation = {
            "status": "ready", "tokenizer_sha256": "c" * 64,
            "suite_sha256": "d" * 64, "max_length": 4096, "fewshot_seed": 1234,
            "workers": [
                {"worker": 0, "benchmarks": ["mmlu"], "leaf_tasks": ["a", "b"]},
                {"worker": 1, "benchmarks": ["arc_easy"], "leaf_tasks": ["arc_easy"]},
            ],
            "tasks": tasks,
            "benchmarks": {
                "mmlu": {"leaf_tasks": ["a", "b"], "evaluation_examples": 4},
                "arc_easy": {"leaf_tasks": ["arc_easy"], "evaluation_examples": 2},
            },
        }
        submission = {
            "job_id": 123, "native_source_revision": "e" * 40,
            "exports": {"METIS_EVAL_REVISION": "f" * 40},
        }
        workers = {}
        for worker in preparation["workers"]:
            names = worker["leaf_tasks"]
            count = sum(tasks[name]["requests"] for name in names)
            metadata = {
                "schema": "more.native-lm-eval/v1", "status": "completed",
                "diagnostic": False, "full_score": True,
                "prepared_request_identity_checked": True,
                "optimizer_loaded": False, "optimizer_shards_opened": 0,
                "run_identity_sha256": summary.EXPECTED_RUN_IDENTITY,
                "preparation_worker": worker["worker"],
                "source": {"model_source_matches_checkpoint": True,
                           "verified_inference_revision": "e" * 40},
                "execution_precision": "bf16", "forward_max_passes": 1,
                "forward_force_depth": 1,
                "effective_curriculum": {
                    "continuation_mode": "depth_one", "memory_gate_scale": 0.0,
                    "ngram_gate_scale": 1.0,
                },
                "tokenizer_release": {
                    "canonical_semantics_recomputed": True,
                    "tokenizer": {"sha256": "c" * 64},
                },
                "packages": {"lm-eval": "0.4.13"},
                "checkpoint": {"sha256": "1" * 64},
                "adapter": {"stats": {
                    "loglikelihood_requests": count, "completed_likelihood_pairs": count,
                    "context_overflows": 0, "boundary_errors": 0, "generation_requests": 0,
                }},
                "completed_steps": 25429, "training_tokens": 49995448320,
                "parameter_count": 1437743410,
            }
            raw = {
                "configs": {
                    name: {"dataset_kwargs": {"revision": tasks[name]["dataset_revision"]}}
                    for name in names
                },
                "n-samples": {
                    name: {"original": tasks[name]["evaluation_examples"],
                           "effective": tasks[name]["evaluation_examples"]}
                    for name in names
                },
                "n-shot": {name: tasks[name]["num_fewshot"] for name in names},
                "results": (
                    {"a": {"acc,none": 0.0}, "b": {"acc,none": 1.0}}
                    if worker["worker"] == 0
                    else {"arc_easy": {"acc,none": 0.5, "acc_norm,none": 1.0}}
                ),
                "groups": {"mmlu": {"acc,none": 0.75}},
            }
            workers[worker["worker"]] = (metadata, raw)
        return suite, preparation, submission, workers

    def test_uses_harness_weighted_group_score_and_renders_both_accuracy_metrics(self):
        result = summary.assemble_results(*self.fixture())
        self.assertEqual(result["evaluation_examples"], 6)
        self.assertEqual(result["benchmarks"][0]["score_percent"], 75.0)
        self.assertEqual(result["benchmarks"][1]["score_percent"], 100.0)
        table = summary.render_table(result)
        self.assertIn("MMLU & 5 & 4 & 75.00 & --", table)
        self.assertIn("ARC-Easy & 0 & 2 & 50.00 & 100.00", table)

    def test_diagnostics_missing_workers_and_changed_protocol_never_become_full_scores(self):
        for mutation in ("diagnostic", "worker", "coverage", "shots", "revision",
                         "checkpoint", "requests", "samples", "identity"):
            suite, preparation, submission, workers = self.fixture()
            metadata, raw = workers[0]
            if mutation == "diagnostic":
                metadata["diagnostic"] = True
            elif mutation == "worker":
                del workers[1]
            elif mutation == "coverage":
                raw["n-samples"]["b"]["effective"] = 2
            elif mutation == "shots":
                raw["n-shot"]["b"] = 0
            elif mutation == "revision":
                raw["configs"]["b"]["dataset_kwargs"]["revision"] = "0" * 40
            elif mutation == "checkpoint":
                workers[1][0]["checkpoint"]["sha256"] = "2" * 64
            elif mutation == "requests":
                metadata["prepared_request_identity_checked"] = False
            elif mutation == "samples":
                raw["samples"] = {}
            elif mutation == "identity":
                metadata["run_identity_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summary.assemble_results(suite, preparation, submission, workers)
