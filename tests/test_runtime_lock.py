from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from metis_data.runtime_lock import (
    PYTHON_REQUIRES,
    RUNTIME_INPUT_NAME,
    RUNTIME_LOCK_NAME,
    runtime_contract,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeLockTests(unittest.TestCase):
    def test_runtime_contract_binds_input_and_transitive_lock(self) -> None:
        contract = runtime_contract(ROOT)
        input_path = ROOT / RUNTIME_INPUT_NAME
        lock_path = ROOT / RUNTIME_LOCK_NAME
        self.assertEqual(
            contract["input_sha256"],
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            contract["lock_sha256"],
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(contract["python_requires"], PYTHON_REQUIRES)
        self.assertEqual(contract["supported_python_abis"], ["cp311", "cp312"])
        self.assertEqual(contract["hash_policy"], "require-hashes")
        self.assertEqual(contract["binary_policy"], "only-binary")

    def test_changed_input_is_rejected_until_lock_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(ROOT / RUNTIME_INPUT_NAME, root / RUNTIME_INPUT_NAME)
            shutil.copy2(ROOT / RUNTIME_LOCK_NAME, root / RUNTIME_LOCK_NAME)
            with (root / RUNTIME_INPUT_NAME).open("a", encoding="utf-8") as handle:
                handle.write("\n# deliberate drift\n")
            with self.assertRaisesRegex(RuntimeError, "changed without regenerating"):
                runtime_contract(root)

    def test_every_locked_distribution_is_exact_and_hashed(self) -> None:
        lock = (ROOT / RUNTIME_LOCK_NAME).read_text(encoding="utf-8")
        requirement_starts = list(
            re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock)
        )
        self.assertGreater(len(requirement_starts), 50)
        for index, match in enumerate(requirement_starts):
            end = (
                requirement_starts[index + 1].start()
                if index + 1 < len(requirement_starts)
                else len(lock)
            )
            block = lock[match.start():end]
            self.assertRegex(block, r"--hash=sha256:[0-9a-f]{64}")

    def test_bootstrap_installs_only_the_hash_locked_binary_runtime(self) -> None:
        script = (ROOT / "ops" / "bootstrap.sh").read_text(encoding="utf-8")
        self.assertIn("--require-hashes", script)
        self.assertIn("--only-binary=:all:", script)
        self.assertIn('requirements-metis16-data.lock', script)
        self.assertNotIn("pip install --upgrade pip", script)
        self.assertNotIn('--editable "$ROOT"', script)


if __name__ == "__main__":
    unittest.main()
