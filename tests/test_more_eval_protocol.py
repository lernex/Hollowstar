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
