"""The training package's schedule must match the data manifest's.

`metis_training.data` restates the phase schedule so training can check a
release independently of the pipeline that produced it. That independence is
worth keeping, but it means the schedule exists in two places, and the 1.6
refit proved they drift: the manifest was rebuilt from measured supply while
`PHASE_TOKENS` still described the aspirational 1T plan, which would have made
training refuse its own verified release.

Parsed with `ast` rather than imported, because `metis_training.data` imports
torch and this invariant is about two literals agreeing.
"""

import ast
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WANTED = {"PHASE_STARTS", "PHASE_TOKENS", "TOTAL_TOKENS"}


def _training_constants() -> dict[str, object]:
    tree = ast.parse((_ROOT / "src" / "metis_training" / "data.py").read_text())
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in _WANTED:
            found[target.id] = ast.literal_eval(node.value)
    return found


class TrainingScheduleMatchesManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.constants = _training_constants()
        self.assertEqual(set(self.constants), _WANTED, "training constants moved or were renamed")
        self.schedule = yaml.safe_load(
            (_ROOT / "manifests" / "metis-1.6.yaml").read_text()
        )["schedule"]

    def test_phase_tokens_match_the_manifest(self) -> None:
        expected = {
            phase: int(row["unique_tokens"]) + int(row["replay_tokens"])
            for phase, row in self.schedule["phases"].items()
        }
        self.assertEqual(self.constants["PHASE_TOKENS"], expected)

    def test_total_matches_the_manifest(self) -> None:
        self.assertEqual(
            self.constants["TOTAL_TOKENS"], int(self.schedule["target_tokens"])
        )

    def test_phase_starts_are_the_running_total(self) -> None:
        cursor = 0
        for phase in ("phase_a", "phase_b", "phase_c"):
            self.assertEqual(
                self.constants["PHASE_STARTS"][phase],
                cursor,
                f"{phase} does not start where the previous phase ends",
            )
            cursor += self.constants["PHASE_TOKENS"][phase]
        self.assertEqual(cursor, self.constants["TOTAL_TOKENS"])

    def test_unique_and_replay_split_matches(self) -> None:
        self.assertEqual(
            int(self.schedule["unique_target_tokens"])
            + int(self.schedule["replay_target_tokens"]),
            int(self.schedule["target_tokens"]),
        )


if __name__ == "__main__":
    unittest.main()
