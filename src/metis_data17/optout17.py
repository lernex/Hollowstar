"""Strict 1.7 interpretation of the unchanged public Common Crawl registry."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from metis_data.freshweb import (
    DEFAULT_OPT_OUT_CSV_URL,
    FreshWebOptions,
    OptOutPolicy,
    OptOutUrlRule,
    _HttpClient,
    _atomic_bytes,
    _normalise_host,
    _normalise_path,
    _normalise_query,
    _opt_out_token,
)

from .common import digest_json, sha256_file, utc_now, write_receipt


PARSER_VERSION = "metis17-cc-optout-v1"
PARSER_SHA256 = sha256_file(Path(__file__))
AUDITED_SNAPSHOT_SHA256 = "f845aa6896869ee540f48dc168f311d71c2948e22826698abcdb6ff515c184f5"
AUDIT_DATE = "2026-09-04"
HEADER = ["Publisher/Requester", "Date of notice", "List of domains/URLs"]


class OptOut17Error(ValueError):
    def __init__(self, reason: str, *, row: int = 0, line: int = 0) -> None:
        self.reason = reason
        super().__init__(f"Common Crawl opt-out policy row={row} line={line}: {reason}")


@dataclass(frozen=True)
class OptOutAnnotation:
    row: int
    line: int
    publisher: str
    notice_date: str
    kind: str
    annotation: str
    previous_entry: str
    next_entry: str


@dataclass(frozen=True)
class OptOutPolicy17(OptOutPolicy):
    parsed_entries: int = 0
    non_rule_entries: int = 0
    annotations: tuple[OptOutAnnotation, ...] = ()

    @cached_property
    def rules_sha256(self) -> str:
        return digest_json({
            "domains": sorted(self.domains),
            "url_paths": sorted(self.url_paths),
            "url_rules": [asdict(rule) for rule in self.url_rules],
        })

    def audit(self) -> dict[str, Any]:
        return {
            "parser_version": PARSER_VERSION,
            "parser_sha256": PARSER_SHA256,
            "source_sha256": self.snapshot_sha256,
            "effective_rules_sha256": self.rules_sha256,
            "input_entries": self.input_entries,
            "parsed_rule_entries": self.parsed_entries,
            "non_rule_entries": self.non_rule_entries,
            "unparsed_entries": self.unparsed_entries,
            "domains": len(self.domains),
            "url_rules": len(self.url_rules),
            "annotations": [asdict(annotation) for annotation in self.annotations],
            "audit_reference": {
                "snapshot_sha256": AUDITED_SNAPSHOT_SHA256,
                "observed_date": AUDIT_DATE,
                "source": DEFAULT_OPT_OUT_CSV_URL,
            },
        }


def _token17(value: str) -> tuple[str, str | OptOutUrlRule] | None:
    if "@" not in value:
        return _opt_out_token(value)
    value = value.strip().lstrip("-*•").strip().strip("<>()[]{}\"'").rstrip(".,;")
    value = value.replace(r"\.", ".").replace(r"\?", "?")
    candidate = value if value.lower().startswith(("http://", "https://")) else f"https://{value}"
    try:
        parsed = urlsplit(candidate)
        if parsed.username is not None or parsed.password is not None:
            return None
    except ValueError:
        return None
    host = _normalise_host((parsed.hostname or "").removeprefix("*."))
    if not host:
        return None
    raw_path, raw_query = parsed.path or "/", parsed.query
    path = _normalise_path(raw_path.split("*", 1)[0].rstrip("$") or "/")
    query = _normalise_query(raw_query.split("*", 1)[0].rstrip("$")) if raw_query else None
    if path == "/" and query is None:
        return "domain", host
    return "url", OptOutUrlRule(
        host=host, path=path, query=query,
        path_prefix="*" in raw_path, query_prefix="*" in raw_query,
    )


def _annotated_line(
    publisher: str, notice: str, lines: list[str], index: int,
) -> tuple[str, str | None, str]:
    value = lines[index]
    previous = next((item for item in reversed(lines[:index]) if item), "")
    following = next((item for item in lines[index + 1:] if item), "")
    # September 2026's public CSV has a requester heading merged into WBD's
    # domain cell. Requiring its observed neighbours preserves every actual
    # Washington Post rule and does not license skipping arbitrary DCN text.
    if (
        value == "DCN - Washington Post"
        and publisher == "DCN - Warner Brothers Discovery" and notice == "2026-06-04"
        and previous == "boingtv.it" and following == "washingtonpost.com"
        and lines[0] == "wbd.com"
    ):
        return "", "embedded_requester_heading", value
    if (
        value == "thrillist.com Ziff Davis"
        and publisher == "News Media Alliance" and notice == "2026-04-29"
        and previous == "thedodo.com" and following == "aberdeen.com"
    ):
        return "thrillist.com", "inline_requester_heading", "Ziff Davis"
    if publisher == "Le Monde" and notice == "2024-03-16":
        if value == "lemonde.fr to start (2024-03-16)" and following == "Additional (2024-04-12):":
            return "lemonde.fr", "inline_notice_date", "to start (2024-03-16)"
        if (
            value == "Additional (2024-04-12):"
            and previous == "lemonde.fr to start (2024-03-16)"
            and following == "https://www.courrierinternational.com/"
        ):
            return "", "additional_notice_heading", value
    if value.lower() in {"n/a", "na", "none"}:
        return "", "explicit_no_rules_placeholder", value
    return value, None, ""


@lru_cache(maxsize=4)
def parse_opt_out_registry17(payload: bytes) -> OptOutPolicy17:
    """Require every nonempty entry/token to be a rule or an audited annotation."""
    try:
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig")), strict=True))
    except (UnicodeError, csv.Error):
        raise OptOut17Error("invalid_csv_encoding_or_structure") from None
    if not rows or rows[0] != HEADER:
        raise OptOut17Error("registry_schema_changed")
    domains: set[str] = set()
    paths: set[tuple[str, str]] = set()
    rules: set[OptOutUrlRule] = set()
    annotations: list[OptOutAnnotation] = []
    inputs = parsed_count = non_rules = 0
    last_updated = None
    for row_number, row in enumerate(rows[1:], start=2):
        if not any(value.strip() for value in row):
            continue
        if row[0].strip().upper().startswith("LAST UPDATED:") and not any(row[1:]):
            last_updated = row[0].split(":", 1)[1].strip()
            continue
        if len(row) != 3:
            raise OptOut17Error("registry_row_schema_changed", row=row_number)
        publisher, notice = row[0].strip(), row[1].strip()
        lines = [value.strip() for value in row[2].splitlines()]
        for index, line in enumerate(lines):
            if not line:
                continue
            inputs += 1
            value, kind, annotation = _annotated_line(publisher, notice, lines, index)
            if kind is not None:
                annotations.append(OptOutAnnotation(
                    row_number, index + 1, publisher, notice, kind, annotation,
                    next((item for item in reversed(lines[:index]) if item), ""),
                    next((item for item in lines[index + 1:] if item), ""),
                ))
            if not value:
                non_rules += 1
                continue
            value = re.sub(r"^[-*•]\s+", "", value)
            tokens = [token for token in re.split(r"[,;\s]+", value) if token]
            if not tokens:
                raise OptOut17Error("empty_rule_entry", row=row_number, line=index + 1)
            for token in tokens:
                entry = _token17(token)
                if entry is None:
                    raise OptOut17Error("unrecognized_registry_token", row=row_number, line=index + 1)
                entry_kind, rule = entry
                if entry_kind == "domain":
                    assert isinstance(rule, str)
                    domains.add(rule)
                else:
                    assert isinstance(rule, OptOutUrlRule)
                    rules.add(rule)
                    if rule.query is None and not rule.path_prefix:
                        paths.add((rule.host, rule.path))
            parsed_count += 1
    if not domains and not rules:
        raise OptOut17Error("empty_registry")
    if inputs != parsed_count + non_rules:
        raise OptOut17Error("entry_coverage_mismatch")
    return OptOutPolicy17(
        domains=frozenset(domains), url_paths=frozenset(paths),
        snapshot_sha256=hashlib.sha256(payload).hexdigest(), last_updated=last_updated,
        url_rules=tuple(sorted(rules, key=lambda rule: (
            rule.host, rule.path, rule.query or "", rule.path_prefix, rule.query_prefix,
        ))),
        input_entries=inputs, unparsed_entries=0, parsed_entries=parsed_count,
        non_rule_entries=non_rules, annotations=tuple(annotations),
    )


def load_opt_out_snapshot17(path: Path, expected_sha256: str | None = None) -> OptOutPolicy17:
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise OptOut17Error("raw_snapshot_sha256_mismatch")
    return parse_opt_out_registry17(payload)


def snapshot_common_crawl_opt_out17(
    root: str | Path, *, url: str = DEFAULT_OPT_OUT_CSV_URL,
    options: FreshWebOptions | None = None, session: Any | None = None,
) -> dict[str, Any]:
    options = options or FreshWebOptions()
    options.validate()
    payload = _HttpClient(options, session).bytes(url)
    policy = parse_opt_out_registry17(payload)
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / f"common-crawl-opt-out-{policy.snapshot_sha256}.csv"
    if snapshot.exists():
        if sha256_file(snapshot) != policy.snapshot_sha256:
            raise OptOut17Error("existing_raw_snapshot_sha256_mismatch")
    else:
        _atomic_bytes(snapshot, payload)
        snapshot.chmod(0o444)
    audit = policy.audit()
    report = {
        "schema": "metis17.common-crawl-opt-out/v1",
        "source": url, "retrieved_at": utc_now(), "path": str(snapshot),
        "sha256": policy.snapshot_sha256, "last_updated": policy.last_updated,
        **audit,
    }
    report_path = destination / f"common-crawl-opt-out17-{digest_json(audit)}.json"
    if not report_path.exists():
        write_receipt(report_path, report)
        report_path.chmod(0o444)
    return report
