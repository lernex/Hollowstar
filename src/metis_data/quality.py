from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml, repository_root


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
# A phone number is punctuated or announced; three space-separated digit groups
# are not. The original pattern accepted a bare space between every group, so
# "100 200 3000" in a results table, "105 220 3340 cm-1" in a spectrum, and
# "001 002 0003" in a data column all registered as personal data. That is why
# scientific PDFs and legal texts -- the two corpora densest in tabular numbers
# -- lost the majority of their records to `personal_data`. Matching real
# numbers more precisely raises recall on documents that actually carry contact
# details and stops discarding measurements.
PHONE_PUNCTUATED = re.compile(
    r"(?:\+1[-. ]?)?(?:\(\d{3}\)[-. ]?|\d{3}[-.])\d{3}[-.]\d{4}\b"
)
# Space separation is only phone-shaped when something says so nearby.
PHONE_ANNOUNCED = re.compile(
    r"(?:phone|telephone|tel|fax|mobile|cell|call|contact)\b[^\d\n]{0,24}"
    r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b",
    re.IGNORECASE,
)
# A role mailbox belongs to a desk, not a person. Publishers print them on
# purpose -- OpenStax puts `support@openstax.org` in the colophon of every
# book -- so matching them discarded a 1.6M-character textbook over the one
# address its publisher wants readers to use. The gate is aimed at a private
# individual's contact details, and a role address is the one kind of address
# that is definitionally not that.
ROLE_MAILBOXES = (
    "abuse", "admin", "billing", "contact", "customerservice", "editor",
    "editorial", "enquiries", "feedback", "help", "hello", "info", "inquiries",
    "legal", "mail", "media", "office", "orders", "permissions", "postmaster",
    "press", "privacy", "sales", "security", "service", "support", "team",
    "webmaster", "noreply", "no-reply", "donotreply", "do-not-reply",
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# RFC 2606 and RFC 6761 reserve these names so that documentation can print an
# address that cannot resolve to anyone. Technical PDFs use them exactly as
# intended -- `firstname.lastname@example.org` and `email@example.com` are both
# real matches in FinePDFs-Edu -- and treating a reserved placeholder as a
# person's contact details discards the document that was being careful.
RESERVED_EMAIL_DOMAIN_RE = re.compile(
    r"^(?:.+\.)?(?:example\.(?:com|net|org)|example|test|invalid|localhost)$",
    re.IGNORECASE,
)
# Matching the exemption inside the pattern does not work: a negative lookahead
# only suppresses the match that starts where it is anchored, and the engine
# then restarts one character later, so `no-reply@x.org` was skipped at `no`
# and matched again at `reply`. The local part has to be parsed and compared.
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PERSONAL_DATA_PATTERNS = (
    PHONE_PUNCTUATED,
    PHONE_ANNOUNCED,
    SSN_RE,
)


def personal_contacts(text: str) -> set[tuple[str, str]]:
    """Distinct personal contact identifiers, role and placeholder ones removed.

    Distinct rather than counted: a technical document repeats the one address
    it wants you to write to, and repetition of a single contact is not more
    personal data than one mention of it.
    """

    found: set[tuple[str, str]] = set()
    for match in EMAIL_RE.finditer(text):
        local, _, domain = match.group(0).partition("@")
        if local.lower() in ROLE_MAILBOXES:
            continue
        if RESERVED_EMAIL_DOMAIN_RE.match(domain):
            continue
        found.add(("email", match.group(0).lower()))
    for pattern in (PHONE_PUNCTUATED, PHONE_ANNOUNCED):
        for match in pattern.finditer(text):
            digits = "".join(c for c in match.group(0) if c.isdigit())[-10:]
            found.add(("phone", digits))
    for match in SSN_RE.finditer(text):
        found.add(("ssn", match.group(0)))
    return found


def contains_personal_email(text: str) -> bool:
    for match in EMAIL_RE.finditer(text):
        local, _, domain = match.group(0).partition("@")
        if local.lower() in ROLE_MAILBOXES:
            continue
        if RESERVED_EMAIL_DOMAIN_RE.match(domain):
            continue
        return True
    return False
MODEL_BOILERPLATE = re.compile(
    r"\b(?:as an ai language model|i do not have personal opinions|here (?:is|are) (?:a|the) (?:five|seven|ten)-section)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    reason: str
    features: dict[str, float | int | bool]


def load_quality_profiles(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repository_root() / "configs" / "metis16" / "quality-profiles.yaml")


def text_features(text: str) -> dict[str, float | int | bool]:
    characters = len(text)
    words = WORD_RE.findall(text)
    word_count = len(words)
    alpha = sum(character.isalpha() for character in text)
    symbols = sum(not character.isalnum() and not character.isspace() for character in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_lines = len(set(lines))
    return {
        "characters": characters,
        "words": word_count,
        "alpha_fraction": alpha / max(1, characters),
        "symbol_fraction": symbols / max(1, characters),
        "repeated_line_fraction": 1.0 - unique_lines / max(1, len(lines)),
        "url_density_per_100_words": 100.0 * len(URL_RE.findall(text)) / max(1, word_count),
        "contains_secret": any(pattern.search(text) is not None for pattern in SECRET_PATTERNS),
        "contains_personal_data": (
            any(pattern.search(text) is not None for pattern in PERSONAL_DATA_PATTERNS)
            or contains_personal_email(text)
        ),
        "distinct_personal_contacts": len(personal_contacts(text)),
        "probable_model_boilerplate": MODEL_BOILERPLATE.search(text) is not None,
    }


def _merged_profile(profiles: dict[str, Any], name: str) -> dict[str, Any]:
    profile = dict(profiles.get("defaults", {}))
    try:
        profile.update(profiles["profiles"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown quality profile {name!r}") from exc
    return profile


def evaluate_quality(
    text: str,
    *,
    profile_name: str,
    metadata: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    fail_closed: bool = True,
) -> QualityDecision:
    profiles = profiles or load_quality_profiles()
    profile = _merged_profile(profiles, profile_name)
    metadata = metadata or {}
    features = text_features(text)

    numeric_rules = (
        ("minimum_characters", "characters", lambda actual, threshold: actual >= threshold),
        ("maximum_characters", "characters", lambda actual, threshold: actual <= threshold),
        ("minimum_alpha_fraction", "alpha_fraction", lambda actual, threshold: actual >= threshold),
        ("maximum_symbol_fraction", "symbol_fraction", lambda actual, threshold: actual <= threshold),
        ("maximum_repeated_line_fraction", "repeated_line_fraction", lambda actual, threshold: actual <= threshold),
        ("maximum_url_density_per_100_words", "url_density_per_100_words", lambda actual, threshold: actual <= threshold),
    )
    for rule, feature, predicate in numeric_rules:
        if rule in profile and not predicate(float(features[feature]), float(profile[rule])):
            return QualityDecision(False, rule, features)
    if profile.get("reject_secrets") and features["contains_secret"]:
        return QualityDecision(False, "secret", features)
    # A published technical document names the person to contact about it. That
    # is one contact block, and rejecting the whole document over it discarded
    # 18 of 60 sampled FinePDFs-Edu records. A staff directory is a different
    # object, and it is distinguishable by how many distinct people it lists:
    # over 300 sampled records the count is 0 for 258, 1 or 2 for 31, and only
    # a thin tail runs higher. `personal_data_maximum_contacts` lets a profile
    # say where the line sits; without it the gate is unchanged and any contact
    # still rejects, so no profile that has not opted in is affected.
    if profile.get("reject_personal_data") and features["contains_personal_data"]:
        allowance = profile.get("personal_data_maximum_contacts")
        if allowance is None or int(features["distinct_personal_contacts"]) > int(allowance):
            return QualityDecision(False, "personal_data", features)
    if profile.get("reject_probable_model_generated") and features["probable_model_boilerplate"]:
        return QualityDecision(False, "probable_model_generated", features)

    metadata_thresholds = {
        "quality_score_minimum": "quality_score",
        "educational_score_minimum": "educational_score",
        "math_score_minimum": "math_score",
        "ocr_confidence_minimum": "ocr_confidence",
        "language_probability_minimum": "language_probability",
        "code_text_interleave_minimum": "code_text_interleave",
        "minimum_components": "component_count",
        "answer_score_minimum": "answer_score",
    }
    for rule, field in metadata_thresholds.items():
        if rule not in profile:
            continue
        value = metadata.get(field)
        if value is None:
            if fail_closed:
                return QualityDecision(False, f"missing_{field}", features)
            continue
        if float(value) < float(profile[rule]):
            return QualityDecision(False, rule, features)

    metadata_maximums = {
        "maximum_generated_file_probability": "generated_file_probability",
        "repeated_header_footer_maximum": "repeated_header_footer_fraction",
        "bibliography_body_ratio_maximum": "bibliography_body_ratio",
    }
    for rule, field in metadata_maximums.items():
        if rule not in profile:
            continue
        value = metadata.get(field)
        if value is None:
            if fail_closed:
                return QualityDecision(False, f"missing_{field}", features)
            continue
        if float(value) > float(profile[rule]):
            return QualityDecision(False, rule, features)

    required_metadata = {
        "require_license": "license",
        "require_capture_date": "capture_date",
        "require_publication_date": "publication_date",
        "require_open_access": "open_access",
        "require_canonical_source": "canonical_url",
        "require_version_metadata": "version",
        "require_source_document": "source_document_id",
        "require_genealogy": "genealogy",
        "require_recent_commit": "commit_date",
        "require_active_repository": "repository_active",
        "require_execution_pass": "execution_passed",
        "require_execution_or_static_verification": "verification_passed",
        "require_programmatic_or_source_verification": "verification_passed",
        "require_technical_context": "technical_context",
        "require_tests_or_examples": "tests_or_examples",
        "require_statement_and_argument": "statement_and_argument",
        "require_title_or_abstract": "title_or_abstract",
        "require_question_and_answer": "question_and_answer",
        "require_primary_source": "primary_source",
        "require_jurisdiction": "jurisdiction",
    }
    for rule, field in required_metadata.items():
        if profile.get(rule) and not metadata.get(field):
            return QualityDecision(False, f"missing_{field}", features)

    evidence_rules = {
        "equation_integrity": "equation_integrity_passed",
        "parser_or_compiler_check": "parser_or_compiler_passed",
        "reading_order_check": "reading_order_passed",
        "structural_completeness": "structurally_complete",
        "verification": "verification_passed",
        "stylistic_diversity_check": "stylistic_diversity_passed",
        "chapter_integrity": "chapter_integrity_passed",
        "translation_quality_check": "translation_quality_passed",
        "allowed_language_registry": "allowed_language",
    }
    if profile.get("secret_scan") == "required" and features["contains_secret"]:
        return QualityDecision(False, "secret", features)
    if profile.get("pii_scan") == "required" and features["contains_personal_data"]:
        return QualityDecision(False, "personal_data", features)
    for rule, field in evidence_rules.items():
        if profile.get(rule) == "required" and not metadata.get(field):
            return QualityDecision(False, f"missing_{field}", features)

    return QualityDecision(True, "accepted", features)


def priority_score(source_priority: int, metadata: dict[str, Any]) -> int:
    score = int(source_priority) * 500
    score += int(float(metadata.get("quality_score", 0.0)) * 250)
    score += 200 if metadata.get("canonical_url") else 0
    score += 150 if metadata.get("human_original", True) else 0
    score += 100 if metadata.get("structurally_complete") else 0
    score += 50 if metadata.get("current_version") else 0
    return max(1, min(65535, score))
