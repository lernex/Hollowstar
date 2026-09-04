"""Keep every build stage on an explicit code-fingerprint path."""

import unittest

from metis_data.slurm import BUILD_GRAPH
from metis_data.stage_code import (
    COMMON_MODULES,
    STAGE_MODULES,
    _all_module_names,
    stage_code_sha256,
)


class StageCodeMapTests(unittest.TestCase):
    def test_every_build_stage_is_mapped(self) -> None:
        built = {str(stage) for stage, _ in BUILD_GRAPH}
        unmapped = sorted(built - set(STAGE_MODULES))
        self.assertEqual(
            unmapped,
            [],
            f"build stages missing from STAGE_MODULES: {unmapped}",
        )

    def test_context_stages_bind_narrowly(self) -> None:
        for stage in (
            "context_select",
            "context_prepare",
            "context_pack",
            "context_verify",
        ):
            self.assertIn(stage, STAGE_MODULES)
            self.assertIn("context_extension.py", STAGE_MODULES[stage])
            self.assertIn("final_dedup.py", STAGE_MODULES[stage])
            self.assertIn("dedup.py", STAGE_MODULES[stage])

    def test_mapped_stages_do_not_take_the_fallback_inventory(self) -> None:
        all_modules = set(_all_module_names())
        for stage, names in STAGE_MODULES.items():
            bound = set(COMMON_MODULES) | set(names)
            self.assertTrue(
                bound < all_modules,
                f"{stage} explicitly binds the whole package",
            )

    def test_hashes_are_stable_and_stage_specific(self) -> None:
        self.assertEqual(
            stage_code_sha256("context_pack"),
            stage_code_sha256("context_pack"),
        )
        self.assertNotEqual(
            stage_code_sha256("context_pack"),
            stage_code_sha256("normalize"),
        )


if __name__ == "__main__":
    unittest.main()
