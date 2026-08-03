"""Run the real normalization gate over a sample of every source.

Quality profiles are assertions about data that nobody had checked against the
data. Sixteen of fifty sources normalized to zero accepted records on the first
production build -- not from bad content, but because profiles demanded evidence
their publishers never emit. Each one was found by letting a multi-hour Slurm
stage fail, reading a traceback, and fixing one source.

This module collapses that loop. It samples rows, runs the identical
extract -> evidence -> quality path `normalize` runs, and reports what every
source would yield. It takes about a minute for the whole corpus, and a source
that yields zero here yields zero in the build.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .normalization_evidence import derive_normalization_evidence, extract_training_text
from .quality import evaluate_quality, load_quality_profiles
from .state import StateStore

# Statuses whose licence must be proven per record. Mirrors the stage's own
# check, which runs before the quality profile and is reported separately there.
PER_RECORD_LICENSE = {"per_record_required", "inherited", "requires_review"}


# Readers take every input file a source has and draw from them in turn until
# the sample is full. Sampling only the first file assumes a file holds many
# records, which is true of a 800k-row parquet shard and false of a corpus that
# ships one document per file: `openstax` is 76 textbooks in 76 `.txt` files, so
# the sweep read one book, called it the whole source, and reported `0/1`. Each
# row is paired with the file record it came from, because that record is what
# the evidence step reads licence and partition facts out of.
def _fixture_rows(
    entries: list[tuple[dict[str, Any], Path]], limit: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    emitted = 0
    for file_record, path in entries:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if emitted >= limit:
                    return
                line = line.rstrip("\r")
                if line.strip():
                    emitted += 1
                    yield file_record, json.loads(line)


def _live_rows(
    entries: list[tuple[dict[str, Any], Path]], limit: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    # Imported lazily: the stage runner pulls in the whole build stack, which a
    # fixture-only sweep on a laptop has no reason to require.
    from .stage_runner import _iter_rows

    emitted = 0
    for file_record, path in entries:
        for row in _iter_rows(path):
            if emitted >= limit:
                return
            emitted += 1
            yield file_record, row


def evaluate_source_sample(
    source: dict[str, Any],
    rows: Iterator[tuple[dict[str, Any], dict[str, Any]]],
    *,
    profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return accept/reject counts for one source, by the stage's own rules."""

    profiles = profiles or load_quality_profiles()
    profile_name = source["processing"]["quality_profile"]
    per_record_license = source["license"]["status"] in PER_RECORD_LICENSE
    reasons: Counter[str] = Counter()
    accepted = sampled = 0
    for file_record, row in rows:
        sampled += 1
        text = extract_training_text(row)
        if not text:
            reasons["no_text"] += 1
            continue
        metadata = derive_normalization_evidence(row, source, file_record, text)
        if per_record_license and not metadata.get("license"):
            reasons["missing_license"] += 1
            continue
        decision = evaluate_quality(
            text,
            profile_name=profile_name,
            metadata=metadata,
            profiles=profiles,
            fail_closed=True,
        )
        if decision.keep:
            accepted += 1
        else:
            reasons[decision.reason] += 1
    return {
        "source_id": source["id"],
        "category": source["category"],
        "quality_profile": profile_name,
        "sampled": sampled,
        "accepted": accepted,
        "accept_rate": (accepted / sampled) if sampled else 0.0,
        "rejections": dict(reasons.most_common()),
    }


def run_profile_preflight(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    state: StateStore | None = None,
    *,
    rows: int = 60,
    fixture: Path | None = None,
) -> dict[str, Any]:
    """Sample every source and report what the normalization gate would keep.

    `fixture` reads pre-extracted JSONL samples, so the sweep runs anywhere.
    Without it the sources are read from the frozen build inputs in place.
    """

    sources = {str(item["id"]): item for item in manifest["sources"]}
    profiles = load_quality_profiles()
    targets: list[tuple[dict[str, Any], list[tuple[dict[str, Any], Path]], bool]] = []
    if fixture is not None:
        index = json.loads((fixture / "FIXTURE.json").read_text(encoding="utf-8"))
        for source_id, entry in sorted(index.items()):
            source = sources.get(source_id)
            sample = fixture / f"{source_id}.jsonl"
            if source is None or not sample.is_file():
                continue
            targets.append((source, [(entry.get("file_record") or {}, sample)], True))
    else:
        if state is None:
            raise ValueError("a StateStore is required when no fixture is given")
        inputs = state.read("build.inputs.json")
        if not inputs:
            raise RuntimeError("build.inputs.json is missing; run the build graph first")
        by_source: dict[str, list[tuple[dict[str, Any], Path]]] = {}
        for record in inputs["inputs"]:
            source_id = str(record["source_id"])
            if source_id not in sources:
                continue
            by_source.setdefault(source_id, []).append(
                (record, Path(record["local_path"]))
            )
        for source_id, entries in by_source.items():
            targets.append((sources[source_id], entries, False))

    reports: list[dict[str, Any]] = []
    for source, entries, from_fixture in sorted(targets, key=lambda item: item[0]["id"]):
        reader = _fixture_rows if from_fixture else _live_rows
        try:
            report = evaluate_source_sample(
                source, reader(entries, rows), profiles=profiles
            )
        except Exception as exc:  # noqa: BLE001 - reported per source, never fatal
            report = {
                "source_id": source["id"],
                "category": source["category"],
                "quality_profile": source["processing"]["quality_profile"],
                "sampled": 0,
                "accepted": 0,
                "accept_rate": 0.0,
                "rejections": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        reports.append(report)

    dead = [item["source_id"] for item in reports if item["sampled"] and not item["accepted"]]
    starved = [
        item["source_id"]
        for item in reports
        if item["accepted"] and item["accept_rate"] < 0.10
    ]
    # A source that could not be read yields nothing either, and it reaches here
    # with sampled == 0, outside the zero-yield test. Counting it as passing
    # would make the sweep quietest about the sources it understands least.
    unreadable = [item["source_id"] for item in reports if item.get("error") or not item["sampled"]]
    return {
        "schema": "metis.profile-preflight/v1",
        "release": manifest.get("release"),
        "rows_per_source": rows,
        "source_count": len(reports),
        # A source yielding nothing fails its whole normalization task, and
        # `afterok` then strands every downstream job in the graph.
        "zero_yield_sources": dead,
        "below_ten_percent_sources": starved,
        "unreadable_sources": unreadable,
        "ok": not dead and not unreadable,
        "sources": reports,
    }


def format_preflight(payload: dict[str, Any]) -> str:
    lines = [f"{'keep/n':>9}  {'rate':>5}  {'source':36s} top rejections"]
    for item in payload["sources"]:
        top = ", ".join(
            f"{reason} x{count}" for reason, count in list(item["rejections"].items())[:3]
        )
        flag = "!!" if item["sampled"] and not item["accepted"] else "  "
        lines.append(
            f"{flag}{item['accepted']:>3}/{item['sampled']:<4} "
            f"{item['accept_rate'] * 100:5.1f}%  {item['source_id']:36s} "
            f"{item.get('error') or top}"
        )
    lines.append("")
    lines.append(f"zero-yield: {len(payload['zero_yield_sources'])}  "
                 f"below-10%: {len(payload['below_ten_percent_sources'])}  "
                 f"unreadable: {len(payload.get('unreadable_sources', []))}  "
                 f"of {payload['source_count']} sources")
    return "\n".join(lines)
