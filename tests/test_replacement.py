from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from metis_data.manifest import candidate_plan, load_manifest, validate_manifest
from metis_data.handoff import _validate_materialized_token_targets
from metis_data.replacement import (
    ReplacementError,
    allocate_replacements,
    replacement_chains,
    validate_replacement_policy,
)
from metis_data.selection import build_selection
from metis_data.state import StateStore


def _source(
    source_id: str,
    *,
    category: str = "web",
    generated: bool = False,
    transformed: bool = False,
    fresh: bool = False,
    freshness_bucket: str | None = None,
    phase_a: int = 0,
    phase_b: int = 0,
    phase_c: int = 0,
    priority: int = 80,
) -> dict:
    provenance = {
        "generated": generated,
        "transformed": transformed,
        "fresh": fresh,
    }
    if freshness_bucket is not None:
        provenance["freshness_bucket"] = freshness_bucket
    return {
        "id": source_id,
        "category": category,
        "phase_tokens": {
            "phase_a": phase_a,
            "phase_b": phase_b,
            "phase_c": phase_c,
        },
        "provenance": provenance,
        "processing": {"priority": priority},
    }


def _policy(members: list[str], donors: list[str]) -> dict:
    return {
        "schema": "metis.replacement-policy/v1",
        "version": "test-r1",
        "defaults": {
            "preserve_category": True,
            "preserve_freshness_bucket": True,
            "no_generated_increase": True,
            "no_transformed_increase": True,
            "phase_resolution_order": ["phase_b", "phase_a", "phase_c"],
            "exhaustion_policy": "fail_closed",
        },
        "groups": [
            {
                "id": "test",
                "members": members,
                "donor_order": donors,
            }
        ],
    }


class ReplacementPolicyTests(unittest.TestCase):
    def test_production_policy_covers_every_source(self) -> None:
        manifest = load_manifest()
        self.assertEqual(validate_replacement_policy(manifest), [])
        chains, groups = replacement_chains(manifest)
        source_ids = {source["id"] for source in manifest["sources"]}
        self.assertEqual(set(chains), source_ids)
        self.assertEqual(set(groups), source_ids)
        self.assertEqual(validate_manifest().errors, ())
        resilience = candidate_plan(manifest)["replacement_resilience"]
        self.assertTrue(resilience["all_sources_have_automatic_shortfall_path"])
        self.assertEqual(resilience["complete_loss_covered_by_other_donors"], 53)
        self.assertEqual(
            resilience["cold_reserve_only_sources"],
            [
                "metis_freshdocs_2025_26",
                "metis_freshscience_2025_26",
                "metis_freshweb_2026",
            ],
        )

    def test_allocator_uses_ordered_donor_surplus_and_preserves_quotas(self) -> None:
        sources = [
            _source("target", phase_a=6, phase_b=4, priority=100),
            _source("preferred", phase_a=5, priority=90),
            _source("last", phase_a=5, priority=80),
        ]
        manifest = {
            "sources": sources,
            "replacement_policy": _policy(
                ["target", "preferred", "last"],
                ["preferred", "last", "target"],
            ),
        }
        requirements = {
            "target": {"phase_a": 6, "phase_b": 4, "phase_c": 0},
            "preferred": {"phase_a": 5, "phase_b": 0, "phase_c": 0},
            "last": {"phase_a": 5, "phase_b": 0, "phase_c": 0},
        }
        allocation = allocate_replacements(
            manifest,
            requirements=requirements,
            available_tokens={"target": 4, "preferred": 11, "last": 8},
        )
        self.assertEqual(allocation["replacement_tokens"], 6)
        self.assertEqual(
            [
                (
                    row["actual_source_id"],
                    row["target_source_id"],
                    row["phase"],
                    row["tokens"],
                )
                for row in allocation["transfers"]
            ],
            [
                ("preferred", "target", "phase_b", 2),
                ("preferred", "target", "phase_a", 4),
            ],
        )
        self.assertEqual(allocation["unresolved"], {})

    def test_generated_data_cannot_replace_an_organic_quota(self) -> None:
        sources = [
            _source("organic", phase_a=5),
            _source("generated", generated=True, phase_a=5),
        ]
        manifest = {
            "sources": sources,
            "replacement_policy": _policy(
                ["organic", "generated"], ["generated", "organic"]
            ),
        }
        chains, _ = replacement_chains(manifest)
        self.assertEqual(chains["organic"], [])
        with self.assertRaisesRegex(ReplacementError, "organic=phase_a:5"):
            allocate_replacements(
                manifest,
                requirements={
                    "organic": {"phase_a": 5, "phase_b": 0, "phase_c": 0},
                    "generated": {"phase_a": 5, "phase_b": 0, "phase_c": 0},
                },
                available_tokens={"organic": 0, "generated": 10},
            )

    def test_selection_records_actual_and_quota_sources(self) -> None:
        sources = [
            _source("target", phase_a=6, priority=100),
            _source("donor", phase_a=4, priority=90),
        ]
        manifest = {
            "sources": sources,
            "replacement_policy": _policy(
                ["target", "donor"], ["donor", "target"]
            ),
            "selection": {
                "seed": 7,
                "replay": {"maximum_document_exposures": 4},
            },
            "schedule": {
                "phases": {
                    "phase_a": {
                        "target_tokens": 10,
                        "replay_tokens": 0,
                    },
                    "phase_b": {
                        "target_tokens": 0,
                        "replay_tokens": 0,
                    },
                    "phase_c": {
                        "target_tokens": 0,
                        "replay_tokens": 0,
                    },
                }
            },
        }
        records = [
            {
                "source_id": "target",
                "doc_id": "target-1",
                "text": "target",
                "token_count": 2,
            },
            {
                "source_id": "donor",
                "doc_id": "donor-1",
                "text": "donor",
                "token_count": 8,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_selection(
                records,
                manifest=manifest,
                eligible_tokens={"target": 2, "donor": 8},
                output_root=root,
                shard_tokens=10,
            )
            self.assertEqual(result["replacement_tokens"], 4)
            self.assertEqual(
                result["unique_written"],
                {
                    "target": {"phase_a": 6, "phase_b": 0, "phase_c": 0},
                    "donor": {"phase_a": 4, "phase_b": 0, "phase_c": 0},
                },
            )
            schedule = Path(result["shards"][0]["path"])
            with schedule.open("rb") as raw:
                with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                    rows = [
                        json.loads(line)
                        for line in stream.read().decode("utf-8").splitlines()
                        if line
                    ]
            replacements = [row for row in rows if row["replacement"]]
            self.assertEqual(sum(row["token_count"] for row in replacements), 4)
            self.assertTrue(
                all(row["source_id"] == "donor" for row in replacements)
            )
            self.assertTrue(
                all(row["quota_source_id"] == "target" for row in replacements)
            )

    def test_acquisition_handoff_accepts_a_policy_covered_source_shortfall(
        self,
    ) -> None:
        sources = [
            _source("target", phase_a=6, priority=100),
            _source("donor", phase_a=4, priority=90),
        ]
        manifest = {
            "sources": sources,
            "replacement_policy": _policy(
                ["target", "donor"], ["donor", "target"]
            ),
            "schedule": {
                "phases": {
                    "phase_a": {"replay_tokens": 0},
                    "phase_b": {"replay_tokens": 0},
                    "phase_c": {"replay_tokens": 0},
                }
            },
        }
        lock = {
            "sources": [
                {"id": "target", "driver": "hf_snapshot", "candidate_tokens": 8},
                {"id": "donor", "driver": "hf_snapshot", "candidate_tokens": 8},
            ],
            "download_tasks": [
                {"task_id": "download-000000", "items": []},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            state.complete(
                "download",
                "download-000000",
                {
                    "files": [
                        {
                            "source_id": "target",
                            "candidate_token_estimate": 2,
                            "candidate_estimator": "test",
                        },
                        {
                            "source_id": "donor",
                            "candidate_token_estimate": 8,
                            "candidate_estimator": "test",
                        },
                    ]
                },
            )
            report = _validate_materialized_token_targets(
                lock, state, manifest
            )
            self.assertTrue(report["sources"]["target"]["target_met"])
            self.assertEqual(
                report["sources"]["target"]["replacement_tokens_received"], 4
            )
            self.assertEqual(
                report["replacement_allocation"]["replacement_tokens"], 4
            )


if __name__ == "__main__":
    unittest.main()
