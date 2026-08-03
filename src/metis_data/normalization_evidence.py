"""Auditable normalization-time evidence derivation.

The data sources used by Metis do not share a schema.  This module maps
upstream fields to the small canonical evidence vocabulary consumed by
``quality.evaluate_quality`` and computes a few document-local structural
checks.  Every derived value is accompanied by a compact
``normalization_evidence`` entry explaining its origin.

The important boundary is that a source label is not a verification result.
For example, a synthetic corpus name may say "reasoning", but that does not
prove that an individual answer was checked.  Consequently this module maps
generator genealogy when it is present, while leaving source-grounding and
verification absent unless the row contains evidence for them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .freshweb import OptOutPolicy, parse_opt_out_registry
from .state import StateStore


_PAYLOAD_FIELDS = {
    "text",
    "content",
    "code",
    "body",
    "document",
    "wikitext",
    "seed_data",
    "prompt",
    "problem",
    "formal_statement",
    "proof",
    "solution",
    "answer",
    "messages",
}
_BAD_LICENSE_VALUES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "no_license",
    "no license",
    "unknown",
}
_ENGLISH_LABELS = {
    "en",
    "eng",
    "english",
    "en-latn",
    "en_latn",
    "eng-latn",
    "eng_latn",
}
# Profiles whose content is machine syntax rather than natural language. An
# English-confidence gate is a category error against them: a Lean proof scores
# low because it is not prose, not because it is bad. The category == "code"
# branch below already encodes this, but formal mathematics is filed under
# `math`, so Lean, Coq, Isabelle, and Metamath fell through and every record was
# rejected on `language_probability_minimum`.
# Evidence that can be true of a pinned corpus as a whole and simply not
# repeated on every row: mathlib and the AFP are compiler-checked by their own
# CI, Hansard's canonical origin is the parliamentary publication it was taken
# from, a synthetic corpus names one generator for every record it contains.
#
# The boundary is deliberate and narrow. These are facts about the *source*.
# Anything measured from an individual document -- its length, its structure,
# its own licence, the specific document a synthetic row was grounded in --
# is not attestable, because a corpus-level answer to a per-record question is
# not evidence, it is a rubber stamp. `source_document_id` is excluded for
# exactly that reason: one value shared by every row identifies nothing.
_FORMAL_LANGUAGE_EXTENSIONS = frozenset(
    {
        "agda", "lagda",
        "lean", "hlean",
        "v", "coq",
        "thy", "isabelle",
        "mm", "metamath",
        "rkt", "idr", "idris",
    }
)
_ATTESTABLE_FIELDS = frozenset(
    {
        "genealogy",
        "verification_passed",
        "parser_or_compiler_passed",
        "canonical_url",
        "primary_source",
        "jurisdiction",
        "version",
        "open_access",
    }
)
_NON_PROSE_QUALITY_PROFILES = frozenset(
    {
        "formal_proof_v1",
        "repository_code_v1",
        "fresh_repository_code_v1",
        "executable_code_v1",
        "verified_synthetic_code_v1",
        "synthetic_code_v1",
    }
)
# Profiles whose text is judged on measured hygiene rather than on an upstream
# score their corpora do not publish.
_COMPUTED_QUALITY_PROFILES = frozenset(
    {
        "web_hq_v1",
        "web_general_v1",
        "web_diversity_v1",
        "fresh_web_2026_v1",
        "code_interleaved_v1",
        "math_score3_v1",
        "grounded_synthetic_v1",
        "synthetic_qa_v1",
        "synthetic_reasoning_v1",
        "legal_synthetic_v1",
    }
)
_ENGLISH_FUNCTION_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "may",
    "more",
    "not",
    "of",
    "on",
    "one",
    "or",
    "other",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "two",
    "use",
    "used",
    "was",
    "we",
    "were",
    "which",
    "will",
    "with",
    "would",
}
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+|(?:chapter|section|lesson|unit|exercise|example|"
    r"introduction|discussion|conclusion|references?)\b)"
)
_CHAPTER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?(?:chapter|book|part)\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five)\b"
)
_OPEN_LICENSE_POSITIVE = re.compile(
    r"(?:creativecommons\.org/licenses/(?:by|by-sa)/|"
    r"creative commons\s*[-–—:]?\s*attribution(?:\s+share[- ]alike)?|"
    r"\bcc\s*[- ]?by(?:\s*[- ]?sa)?\b|\bcc0\b|"
    r"creativecommons\.org/publicdomain/(?:zero|mark)/|"
    r"\bpublic domain\b|\bopen parliament licen[cs]e\b|\bopen government licen[cs]e\b)",
    re.IGNORECASE,
)
_OPEN_LICENSE_NEGATIVE = re.compile(
    r"(?:by[-_ /]?(?:nc|nd)|non[- ]?commercial|no[- ]?derivatives?|all rights reserved)",
    re.IGNORECASE,
)


def _json_like(value: Any) -> Any:
    """Decode JSON/Python-literal metadata strings without executing code."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if len(stripped) > 2_000_000 or not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            decoded = ast.literal_eval(stripped)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return value
        return decoded if isinstance(decoded, (dict, list, tuple)) else value


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return None
    if hasattr(value, "as_py"):
        try:
            value = value.as_py()
        except (AttributeError, TypeError, ValueError):
            return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    decoded = _json_like(value)
    if decoded is not value:
        return _safe_value(decoded, depth=depth + 1)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= 250:
                break
            safe = _safe_value(nested, depth=depth + 1)
            if safe is not None:
                output[str(key)] = safe
        return output
    if isinstance(value, (list, tuple)):
        return [
            safe
            for nested in value[:250]
            if (safe := _safe_value(nested, depth=depth + 1)) is not None
        ]
    return None


def _row_metadata(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return bounded output metadata and a complete searchable metadata view."""

    searchable: dict[str, Any] = {}
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key in _PAYLOAD_FIELDS:
            continue
        safe = _safe_value(value)
        if safe is None:
            continue
        searchable[str(key)] = safe
        if isinstance(safe, str) and len(safe) > 16_384:
            output[str(key) + "_sha256"] = hashlib.sha256(safe.encode("utf-8")).hexdigest()
        else:
            output[str(key)] = safe

    nested = _json_like(row.get("metadata"))
    nested_safe = _safe_value(nested)
    if isinstance(nested_safe, dict):
        searchable["metadata"] = nested_safe
        output["upstream_metadata"] = nested_safe
        for key, value in nested_safe.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                output.setdefault(str(key), value)
            elif isinstance(value, list) and len(value) <= 100:
                output.setdefault(str(key), value)
    output.pop("metadata", None)
    return output, searchable


def _walk(mapping: Any, path: Sequence[str]) -> Any:
    value = mapping
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _find(searchable: Mapping[str, Any], *paths: str) -> tuple[Any, str | None]:
    for path in paths:
        parts = tuple(part for part in path.split(".") if part)
        value = _walk(searchable, parts)
        if value is not None and value != "" and value != []:
            return value, path
        nested = searchable.get("metadata")
        if isinstance(nested, Mapping):
            value = _walk(nested, parts)
            if value is not None and value != "" and value != []:
                return value, f"metadata.{path}"
    return None, None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_sequence(value: Any) -> list[Any]:
    value = _json_like(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] if value is not None and value != "" else []


def _evidence_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_evidence_value(item) for item in value[:12]]
    if isinstance(value, Mapping):
        return {str(key): _evidence_value(item) for key, item in list(value.items())[:12]}
    return str(value)[:500]


def _set_evidence(
    metadata: dict[str, Any],
    field: str,
    value: Any,
    *,
    method: str,
    source_field: str | None = None,
    overwrite: bool = False,
) -> None:
    if value is None or value == "" or value == []:
        return
    if not overwrite and metadata.get(field) not in (None, "", []):
        return
    metadata[field] = value
    item: dict[str, Any] = {
        "field": field,
        "method": method,
        "value": _evidence_value(value),
    }
    if source_field:
        item["source_field"] = source_field
    metadata.setdefault("normalization_evidence", []).append(item)


def _normalise_license(value: Any) -> str | None:
    values: list[str] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            item = item.get("spdx_id") or item.get("name") or item.get("license")
        text = str(item or "").strip()
        if text.lower() in _BAD_LICENSE_VALUES:
            continue
        if text and text not in values:
            values.append(text)
    return ",".join(values) if values else None


def _text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()]


def _computed_english_probability(text: str) -> tuple[float, dict[str, Any]] | None:
    letters = [character for character in text if character.isalpha()]
    words = [match.group(0).lower() for match in _WORD_RE.finditer(text[:250_000])]
    # A short row is uncertain, not unmeasurable. The old 100-letter/30-word
    # floor returned None for a grounded QA answer, and None fails closed, so
    # `nemotron_specialized_fact_seeking` lost 44 of 60 rows to
    # `missing_language_probability` -- rejected for being short rather than
    # for being non-English. Its rows run 18-55 words.
    if len(letters) < 40 or len(words) < 12:
        return None
    latin_letters = sum(
        "LATIN" in unicodedata.name(character, "") for character in letters
    )
    latin_fraction = latin_letters / len(letters)
    hits = [word for word in words if word in _ENGLISH_FUNCTION_WORDS]
    distinct_hits = len(set(hits))
    hit_fraction = len(hits) / len(words)
    script_score = min(1.0, latin_fraction / 0.985)
    # The distinct-function-word target scales with how many words there are to
    # draw from: a 20-word answer cannot contain 12 distinct function words, so
    # a fixed target measured length rather than language. At 30+ words the
    # target is still 12, so nothing longer changes.
    vocabulary_target = min(12.0, max(3.0, len(words) / 2.5))
    vocabulary_score = min(1.0, distinct_hits / vocabulary_target)
    frequency_score = min(1.0, hit_fraction / 0.11)
    probability = 0.50 * script_score + 0.30 * vocabulary_score + 0.20 * frequency_score
    details = {
        "latin_letter_fraction": round(latin_fraction, 6),
        "english_function_word_fraction": round(hit_fraction, 6),
        "distinct_english_function_words": distinct_hits,
        "sampled_words": len(words),
    }
    return round(max(0.0, min(1.0, probability)), 6), details


def _computed_document_quality(text: str) -> tuple[float, dict[str, Any]]:
    lines = _text_lines(text)
    characters = len(text)
    alpha_fraction = sum(character.isalpha() for character in text) / max(1, characters)
    replacement_fraction = (
        text.count("\ufffd") + sum(unicodedata.category(character) == "Cc" and character not in "\n\r\t" for character in text)
    ) / max(1, characters)
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    urls = len(re.findall(r"https?://\S+", text, flags=re.IGNORECASE))
    repeated = 1.0 - len(set(lines)) / max(1, len(lines))
    components = {
        "length": min(1.0, math.sqrt(characters / 2_000.0)),
        "alphabetic": max(0.0, min(1.0, (alpha_fraction - 0.30) / 0.35)),
        "line_uniqueness": max(0.0, 1.0 - repeated),
        "url_cleanliness": max(0.0, 1.0 - (100.0 * urls / max(1, len(words))) / 4.0),
        "decoding": max(0.0, 1.0 - replacement_fraction * 250.0),
    }
    score = (
        0.20 * components["length"]
        + 0.25 * components["alphabetic"]
        + 0.20 * components["line_uniqueness"]
        + 0.15 * components["url_cleanliness"]
        + 0.20 * components["decoding"]
    )
    return round(max(0.0, min(1.0, score)), 6), {
        **{key: round(value, 6) for key, value in components.items()},
        "alpha_fraction": round(alpha_fraction, 6),
        "replacement_or_control_fraction": round(replacement_fraction, 8),
    }


def _equation_integrity(text: str) -> tuple[bool, dict[str, Any]]:
    unescaped_dollars = len(re.findall(r"(?<!\\)\$", text))
    inline_open = text.count(r"\(")
    inline_close = text.count(r"\)")
    display_open = text.count(r"\[")
    display_close = text.count(r"\]")
    environments_open = re.findall(r"\\begin\{([^{}]+)\}", text)
    environments_close = re.findall(r"\\end\{([^{}]+)\}", text)
    latex_signal_count = (
        unescaped_dollars // 2
        + inline_open
        + display_open
        + len(environments_open)
        + len(re.findall(r"\\(?:frac|sum|int|sqrt|begin)\b", text))
    )
    # Most web mathematics is not written in LaTeX. FineMath and MegaMath carry
    # forum answers and worked solutions in plain text, so counting only markup
    # made "has equations" mean "has markup" and the gate failed the majority of
    # two corpora selected upstream for being mathematical. A relation with
    # numbers or variables on both sides is an equation whether or not anyone
    # typeset it.
    plain_relations = len(
        re.findall(
            r"(?<![=<>!])[=≠≤≥≈≡<>](?![=])",
            re.sub(r"https?://\S+", " ", text),
        )
    )
    plain_operators = len(re.findall(r"[∫∑∏√±×÷∞≈≤≥∈∉⊆∪∩→∂∇]", text))
    plain_expressions = len(
        re.findall(
            r"(?:\d+|\b[a-zA-Z]\b)\s*[-+*/^]\s*(?:\d+|\b[a-zA-Z]\b)",
            text,
        )
    )
    plain_signal_count = plain_operators + (
        plain_expressions if plain_relations >= 2 else 0
    )
    math_signal_count = latex_signal_count + plain_signal_count
    passed = (
        math_signal_count > 0
        # Balance is a property of markup. A plain-text document has no
        # delimiters to leave unclosed, so these four checks are vacuously true
        # for it and remain a real check on anything that does use LaTeX.
        and unescaped_dollars % 2 == 0
        and inline_open == inline_close
        and display_open == display_close
        and Counter(environments_open) == Counter(environments_close)
    )
    return passed, {
        "math_signal_count": math_signal_count,
        "latex_signal_count": latex_signal_count,
        "plain_signal_count": plain_signal_count,
        "plain_relation_count": plain_relations,
        "unescaped_dollar_count": unescaped_dollars,
        "inline_pairs": [inline_open, inline_close],
        "display_pairs": [display_open, display_close],
        "environment_pairs_match": Counter(environments_open) == Counter(environments_close),
    }


def _code_text_interleave(text: str) -> tuple[float, dict[str, int]]:
    code_pattern = re.compile(
        r"^\s*(?:```|~~~|#include\b|(?:async\s+)?def\b|class\b|function\b|"
        r"import\b|from\s+\S+\s+import\b|(?:const|let|var)\b|SELECT\b|"
        r"(?:if|for|while)\s*\(|[\w.\[\]]+\s*=\s*\S+|[{};]\s*$)",
        re.IGNORECASE,
    )
    prose_pattern = re.compile(r"[A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){4,}")
    code_lines = 0
    prose_lines = 0
    fenced = False
    for line in _text_lines(text):
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            code_lines += 1
        elif fenced or code_pattern.search(line):
            code_lines += 1
        elif prose_pattern.search(line):
            prose_lines += 1
    total = code_lines + prose_lines
    score = min(code_lines, prose_lines) / total if total else 0.0
    return round(score, 6), {"code_lines": code_lines, "prose_lines": prose_lines}


def _generated_file_probability(text: str, path: str | None) -> tuple[float, dict[str, Any]]:
    """Estimate whether a source file was emitted by a tool rather than written.

    This used to return None without a path, and the caller turned that into a
    missing signal, which `maximum_generated_file_probability` then rejected. A
    ceiling meant to drop minified and machine-emitted files instead dropped
    every record of any corpus that names its path field something unexpected.
    Absence of a path is not evidence of generation, and the text alone carries
    most of the signal, so classification no longer depends on one.
    """

    candidate = f"{path or ''}\n{text[:8_192]}".lower()
    markers = [
        marker
        for marker in (
            "generated file",
            "auto-generated",
            "autogenerated",
            "do not edit",
            ".min.js",
            ".min.css",
            "generated by",
            "@generated",
            "this file was automatically generated",
            "code generated by",
        )
        if marker in candidate
    ]
    details: dict[str, Any] = {"path": path, "markers": markers}
    if markers:
        return 1.0, details
    # Minified and machine-emitted output is structurally distinctive: it packs
    # content into very few, very long lines. Human-written source of any
    # language wraps. This is the fallback when nothing declares itself.
    lines = text.splitlines() or [text]
    longest = max((len(line) for line in lines), default=0)
    long_lines = sum(1 for line in lines if len(line) > 500)
    long_fraction = long_lines / max(1, len(lines))
    details.update(
        {"longest_line": longest, "long_line_fraction": round(long_fraction, 4)}
    )
    if longest >= 10_000 or long_fraction >= 0.5:
        return 0.90, details
    if longest >= 3_000 and long_fraction >= 0.10:
        return 0.50, details
    return 0.05, details


def _parse_offsets(value: Any, text_length: int) -> list[int]:
    offsets = []
    for item in _as_sequence(value):
        try:
            offset = int(item)
        except (TypeError, ValueError):
            return []
        if offset <= 0 or offset > text_length or (offsets and offset <= offsets[-1]):
            return []
        offsets.append(offset)
    return offsets


def _pdf_evidence(text: str, searchable: Mapping[str, Any]) -> dict[str, Any]:
    characters = max(1, len(text))
    replacement_or_control = text.count("\ufffd") + sum(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )
    suspicious = len(re.findall(r"(?:[|]{3,}|[_]{12,}|\b\w{30,}\b)", text))
    extraction_confidence = max(
        0.0,
        1.0 - 200.0 * replacement_or_control / characters - 2.0 * suspicious / max(1, len(_text_lines(text))),
    )
    page_ends, _ = _find(searchable, "page_ends")
    offsets = _parse_offsets(page_ends, len(text))
    page_boundaries = offsets if offsets else [len(text)]
    pages: list[str] = []
    start = 0
    for end in page_boundaries:
        pages.append(text[start:end])
        start = end
    if start < len(text):
        pages.append(text[start:])
    edge_lines: list[str] = []
    for page in pages:
        lines = _text_lines(page)
        if lines:
            edge_lines.extend(
                re.sub(r"\d+", "#", line.casefold())[:160]
                for line in (lines[0], lines[-1])
                if len(line) <= 160
            )
    repeated_edges = sum(count - 1 for count in Counter(edge_lines).values() if count > 1)
    repeated_fraction = repeated_edges / max(1, len(edge_lines))
    all_lines = _text_lines(text)
    line_lengths = [len(line) for line in all_lines]
    single_token_fraction = (
        sum(len(line.split()) <= 1 for line in all_lines) / max(1, len(all_lines))
    )
    reading_order = bool(
        extraction_confidence >= 0.90
        and repeated_fraction <= 0.08
        and single_token_fraction <= 0.35
        and (not line_lengths or median(line_lengths) >= 20)
        and (not page_ends or bool(offsets))
    )
    return {
        "ocr_confidence": round(extraction_confidence, 6),
        "repeated_header_footer_fraction": round(repeated_fraction, 6),
        "reading_order_passed": reading_order,
        "details": {
            "pages": len(pages),
            "replacement_or_control_characters": replacement_or_control,
            "suspicious_runs": suspicious,
            "single_token_line_fraction": round(single_token_fraction, 6),
            "median_nonempty_line_length": median(line_lengths) if line_lengths else 0,
        },
    }


def _bibliography_ratio(text: str) -> float:
    matches = list(re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography)\s*$", text))
    for match in reversed(matches):
        if match.start() >= len(text) * 0.35:
            return round((len(text) - match.start()) / max(1, len(text)), 6)
    citation_lines = sum(
        bool(re.search(r"(?:\bdoi:|\bet al\.|\(\d{4}\)|\[\d+\])", line, re.IGNORECASE))
        for line in _text_lines(text)
    )
    return round(min(0.34, citation_lines / max(1, len(_text_lines(text)))), 6)


def _title_or_abstract(row: Mapping[str, Any], searchable: Mapping[str, Any], text: str) -> bool:
    value, _ = _find(
        searchable,
        "title",
        "document_title",
        "paper_title",
        "abstract",
        "metadata.title",
        "metadata.abstract",
    )
    if isinstance(value, str) and len(value.strip()) >= 3:
        return True
    if re.search(r"(?im)^\s*(?:#{1,6}\s*)?abstract\s*$", text[:20_000]) or re.search(
        r"(?m)^\s*#{1,2}\s+\S.{3,200}$", text[:5_000]
    ):
        return True
    # Plain-text paper corpora carry the title as the opening line with no
    # markup and no separate field -- peS2o's s2ag records are exactly a title
    # followed by its abstract. Requiring a markdown heading rejected all of
    # them for lacking a structure the corpus never uses.
    for line in text.split("\n", 8)[:8]:
        heading = line.strip()
        if not heading:
            continue
        return 10 <= len(heading) <= 300 and len(text) > 2 * len(heading)
    return False


def _structurally_complete_textbook(
    source_id: str,
    text: str,
    searchable: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    headings = len(_HEADING_RE.findall(text))
    title, _ = _find(searchable, "title", "metadata.title")
    if source_id == "openstax":
        passed = len(text) >= 50_000 and headings >= 4
        method = "whole_snapshot_textbook_structure_v1"
    else:
        # LibreTexts records are complete lessons rather than whole books.
        passed = len(text) >= 800 and bool(title) and headings >= 2
        method = "licensed_lesson_structure_v1"
    return passed, {"method": method, "headings": headings, "has_title": bool(title)}


def _chapter_integrity(text: str) -> tuple[bool, dict[str, Any]]:
    chapters = len(_CHAPTER_RE.findall(text))
    gutenberg_begin = bool(re.search(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE))
    gutenberg_end = bool(re.search(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE))
    # What this gate is for is catching a book that stops in the middle. Chapter
    # headings are one way to see that and not the only way: the pre-1929 and
    # Library of Congress scans include verse, reference works, statutes, and
    # reports that are complete without a single "Chapter". For those, the
    # observable evidence of completeness is book-length text that ends on a
    # finished sentence rather than mid-word.
    ending = text.rstrip()[-1:] if text.strip() else ""
    finished_ending = ending in {".", "!", "?", '"', "'", "”", "’", "*", ")"}
    unchaptered_complete = len(text) >= 20_000 and finished_ending
    passed = len(text) >= 10_000 and (
        chapters >= 2 or (gutenberg_begin and gutenberg_end) or unchaptered_complete
    )
    return passed, {
        "chapter_heading_count": chapters,
        "gutenberg_start_marker": gutenberg_begin,
        "gutenberg_end_marker": gutenberg_end,
        "unchaptered_complete_work": unchaptered_complete,
        "final_character": ending,
    }


def _question_and_answer(text: str) -> tuple[bool, dict[str, Any]]:
    """Does this record actually pose a question and answer it?

    Stands in for ``answer_score_minimum``, which asked the Common Pile
    StackExchange release for a vote count it does not ship. A score was only
    ever a proxy for "someone answered this usefully"; the two halves being
    present, and the answer having substance, is the part that can be observed
    in the record itself.
    """

    head = text[:4_000]
    question = bool(
        re.search(r"\?(?:\s|$)", head)
        or re.search(r"^\s*(?:#+\s*)?(?:question|q)\b\s*[:.\-]", head, re.IGNORECASE | re.MULTILINE)
        or re.search(
            r"\b(?:how do i|how can i|why does|what is the|is it possible)\b",
            head,
            re.IGNORECASE,
        )
    )
    answer_marker = re.search(
        r"^\s*(?:#+\s*)?(?:answer|a|accepted answer|best answer)\b\s*[:.\-]",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    # Dolma-formatted Stack Exchange concatenates the post and its replies
    # without labelling them, so the fallback is positional: substantive prose
    # that continues well past the question mark.
    first_question = text.find("?")
    trailing = len(text) - first_question if first_question >= 0 else 0
    passed = question and (bool(answer_marker) or trailing >= 400)
    return passed, {
        "question_signal": question,
        "answer_marker": bool(answer_marker),
        "characters_after_first_question": trailing,
    }


def _engineering_structure(text: str) -> tuple[bool, dict[str, Any]]:
    markers = {
        name: bool(re.search(pattern, text, re.IGNORECASE))
        for name, pattern in {
            "patent_office": r"\b(?:united states )?patent office\b",
            "application": r"\bapplication (?:filed|number|no\.)\b",
            "description": r"\b(?:detailed )?description\b|\bthis invention\b",
            "claims": r"\b(?:what is claimed|we claim|i claim)\b",
            "figures": r"\bfig(?:ure|\.)\s*\d+\b",
        }.items()
    }
    return len(text) >= 1_200 and sum(markers.values()) >= 2, markers


def _open_access_from_license(license_expression: str | None) -> bool:
    if not license_expression or _OPEN_LICENSE_NEGATIVE.search(license_expression):
        return False
    return bool(_OPEN_LICENSE_POSITIVE.search(license_expression))


def _freshweb_nested_evidence(
    metadata: dict[str, Any], searchable: Mapping[str, Any]
) -> None:
    evaluation, evaluation_path = _find(searchable, "license_evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("decision"):
        matches = evaluation.get("matches")
        candidate = None
        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, Mapping):
                    continue
                evidence = match.get("evidence")
                if isinstance(evidence, Mapping):
                    candidate = evidence.get("resolved_url") or evidence.get("value")
                    if candidate:
                        break
        license_value = _normalise_license(candidate)
        if license_value:
            _set_evidence(
                metadata,
                "license",
                license_value,
                method="freshweb_explicit_open_license_evaluation_v1",
                source_field=evaluation_path,
            )
    english, english_path = _find(searchable, "english_evidence")
    if isinstance(english, Mapping) and english.get("decision"):
        score = _numeric(
            english.get("probability")
            or english.get("score")
            or english.get("confidence")
        )
        _set_evidence(
            metadata,
            "language_probability",
            score if score is not None else 1.0,
            method="freshweb_materializer_english_evidence_v1",
            source_field=english_path,
        )
    versions, version_path = _find(searchable, "version_evidence")
    if isinstance(versions, list):
        for item in versions:
            if isinstance(item, Mapping):
                candidate = item.get("value") or item.get("resolved_url")
                if candidate:
                    _set_evidence(
                        metadata,
                        "version",
                        str(candidate),
                        method="freshweb_explicit_version_evidence_v1",
                        source_field=version_path,
                    )
                    break


def load_frozen_common_crawl_opt_out(
    root: Path,
    state: StateStore,
) -> OptOutPolicy:
    """Load and verify the handoff-frozen opt-out CSV without network access."""

    handoff = state.read("ACQUISITION_READY.json")
    if not isinstance(handoff, dict):
        raise RuntimeError(
            "ACQUISITION_READY.json is missing; Common Crawl normalization "
            "cannot reapply the final publisher opt-out snapshot"
        )
    policy_record = handoff.get("common_crawl_opt_out")
    if not isinstance(policy_record, dict):
        raise RuntimeError(
            "Common Crawl normalization requires common_crawl_opt_out in the acquisition handoff"
        )
    if policy_record.get("normalization_reapplication_required") is not True:
        raise RuntimeError(
            "The acquisition handoff does not require final Common Crawl opt-out reapplication"
        )
    artifacts = policy_record.get("artifacts")
    snapshot_record = artifacts.get("snapshot") if isinstance(artifacts, Mapping) else None
    if not isinstance(snapshot_record, Mapping):
        raise RuntimeError("The acquisition handoff has no frozen Common Crawl opt-out snapshot")
    relative = Path(str(snapshot_record.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("The Common Crawl opt-out snapshot path is not a safe Lustre-relative path")
    resolved_root = root.expanduser().resolve()
    snapshot = (resolved_root / relative).resolve()
    try:
        snapshot.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("The Common Crawl opt-out snapshot escapes the Lustre root") from exc
    if not snapshot.is_file():
        raise RuntimeError(f"The frozen Common Crawl opt-out snapshot is missing: {snapshot}")
    payload = snapshot.read_bytes()
    expected_size = snapshot_record.get("size")
    if expected_size is not None and len(payload) != int(expected_size):
        raise RuntimeError("The frozen Common Crawl opt-out snapshot size changed after handoff")
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest = str(snapshot_record.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or digest != expected_digest:
        raise RuntimeError("The frozen Common Crawl opt-out snapshot failed its SHA-256 check")
    policy = parse_opt_out_registry(payload)
    if policy.snapshot_sha256 != expected_digest:
        raise RuntimeError("The parsed Common Crawl opt-out policy does not match the handoff checksum")
    return policy


def final_common_crawl_opt_out_reason(
    row: Mapping[str, Any],
    policy: OptOutPolicy,
) -> tuple[str | None, str | None]:
    """Return a final-handoff rejection reason and the matched URL, if any."""

    _, searchable = _row_metadata(row)
    candidates: list[str] = []
    for path in (
        "url",
        "original_url",
        "canonical_url",
        "declared_canonical_url",
        "source_url",
    ):
        value, _ = _find(searchable, path)
        if isinstance(value, str) and value.strip() and value not in candidates:
            candidates.append(value.strip())
    for candidate in candidates:
        reason = policy.reason(candidate)
        if reason in {"common_crawl_opt_out_domain", "common_crawl_opt_out_url"}:
            return f"final_{reason}", candidate
    return None, None


def extract_training_text(row: Mapping[str, Any]) -> str:
    """Extract a training document, including proof records with split fields."""

    for key in ("text", "content", "code", "body", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    problem = row.get("problem")
    statement = row.get("formal_statement")
    proof = row.get("proof") or row.get("solution") or row.get("answer")
    sections: list[str] = []
    if isinstance(problem, str) and problem.strip():
        sections.append("Problem:\n" + problem.strip())
    if isinstance(statement, str) and statement.strip():
        sections.append("Formal statement:\n" + statement.strip())
    if isinstance(proof, str) and proof.strip():
        sections.append("Proof:\n" + proof.strip())
    return "\n\n".join(sections)


def validated_attestations(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return a source's manifest attestations, or raise if they are malformed.

    Split out of the evidence pass so ``metisctl validate`` rejects a bad
    attestation on a login node instead of a normalize array discovering it
    hours into a build. The first version of this read the block from
    ``provenance`` while both manifests wrote it at the source top level, so
    two attestations sat in the repository doing nothing at all -- silence is
    the one failure mode a fail-closed pipeline cannot catch for you.
    """

    attestations = source.get("attestations") or {}
    if not attestations:
        return {}
    source_id = str(source.get("id", "<missing>"))
    if not isinstance(attestations, Mapping):
        raise ValueError(f"{source_id}: attestations must be a mapping of field to value")
    if not str(source.get("attestation_basis") or "").strip():
        raise ValueError(
            f"{source_id}: attestations requires an attestation_basis recording why "
            "the claim is true of this corpus"
        )
    unknown = sorted(set(attestations) - _ATTESTABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"{source_id}: {unknown} cannot be attested at manifest level; "
            f"attestable fields are {sorted(_ATTESTABLE_FIELDS)}"
        )
    return dict(attestations)


def derive_normalization_evidence(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    file_record: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    """Return canonical metadata plus an audit trail for every derived field."""

    metadata, searchable = _row_metadata(row)
    source_id = str(source["id"])
    profile_name = str(source["processing"]["quality_profile"])
    source_file = str(file_record.get("repo_path") or "")

    # Proof-Pile-2 files AlgebraicStack sources under a prose proof profile, so
    # a row can be a formal development while its profile expects English. This
    # is settled once here because it governs both the language gate and the
    # proof-structure check below.
    extension_value, _ = _find(searchable, "ext", "meta.ext", "extension", "language", "lang")
    formal_language = str(extension_value or "").strip().lower().lstrip(".")
    is_formal_language_row = (
        profile_name in {"proof_v1", "formal_proof_v1"}
        and formal_language in _FORMAL_LANGUAGE_EXTENSIONS
    )

    # Direct canonical aliases. Nested Common Pile metadata and JSON-encoded
    # metadata are searched before source-level fallbacks.
    direct_aliases: dict[str, tuple[str, ...]] = {
        "canonical_url": (
            "canonical_url",
            "url",
            "source_url",
            "item_url",
            "book_url",
            "ia_url",
            "hathi_url",
            "text_file_url",
        ),
        "capture_date": ("capture_date", "fetch_time", "crawl_date", "timestamp", "date"),
        "publication_date": (
            "publication_date",
            "published",
            "published_at",
            "posted_date",
            "created",
            "date",
        ),
        "version": ("version", "documentation_version", "release", "tag"),
        "commit_date": ("commit_date", "committed_at", "pushed_at"),
        "repository_active": ("repository_active",),
        "technical_context": ("technical_context",),
        "execution_passed": ("execution_passed",),
        "parser_or_compiler_passed": ("parser_or_compiler_passed",),
        "verification_passed": (
            "verification_passed",
            "programmatic_verification_passed",
            "source_verification_passed",
            "static_analysis_passed",
            "verified",
        ),
        "source_document_id": (
            "source_document_id",
            "source_doc_id",
            "grounding_document_id",
            # Nemotron-CC document-grounded QA names the Common Crawl record it
            # was generated from. That is a real per-record grounding
            # identifier, and every row carries one.
            "warc_record_id",
            "metadata.warc_record_id",
            "seed_document_id",
            "source_id",
        ),
    }
    for canonical, aliases in direct_aliases.items():
        value, path = _find(searchable, *aliases)
        if canonical.endswith("_passed") or canonical in {
            "repository_active",
            "technical_context",
        }:
            if value is not True:
                continue
        _set_evidence(
            metadata,
            canonical,
            value,
            method="upstream_field_v1",
            source_field=path,
        )

    license_value, license_path = _find(
        searchable,
        "license",
        "license_name",
        "repo_license",
        "spdx_license",
        "detected_licenses",
        "all_licenses",
        "licenses",
        "file_info.detected_licenses",
        # Stack-derived corpora keep provenance in a "meta" blob rather than
        # the "metadata" key _find already descends into, and name the field
        # after the repository statistic it was collected with. Proof-Pile-2
        # carries a real per-record license there -- e.g. ["MIT"] -- so reading
        # it is evidence, not a relaxation.
        "max_stars_repo_licenses",
        "meta.max_stars_repo_licenses",
        "meta.detected_licenses",
        "meta.license",
    )
    license_expression = _normalise_license(license_value)
    if license_expression:
        _set_evidence(
            metadata,
            "license",
            license_expression,
            method="upstream_per_record_license_v1",
            source_field=license_path,
        )

    if source_id == "openstax" and metadata.get("license") is None:
        # The pinned OpenStax files retain their own attribution page.  Accept
        # only a license statement present in the individual book text; the
        # dataset-card label alone is not treated as per-book evidence.
        license_match = re.search(
            r"(?:creative commons\s+attribution\s+4\.0(?:\s+international)?"
            r"(?:\s+license)?|\bcc\s+by\s+4\.0\b)",
            text[:50_000],
            re.IGNORECASE,
        )
        if license_match:
            _set_evidence(
                metadata,
                "license",
                "CC-BY-4.0",
                method="openstax_in_book_license_statement_v1",
                source_field="text:first_50000_characters",
            )

    _freshweb_nested_evidence(metadata, searchable)

    license_status = str(source["license"]["status"])
    if license_status in {"reviewed", "requires_acceptance", "public_domain_or_reviewed"}:
        _set_evidence(
            metadata,
            "license",
            str(source["license"]["expression"]),
            method="pinned_source_manifest_license_v1",
            source_field="source.license",
        )

    quality_value, quality_path = _find(searchable, "quality_score", "quality_rating")
    quality_score = _numeric(quality_value)
    if quality_score is not None:
        _set_evidence(
            metadata,
            "quality_score",
            quality_score,
            method="upstream_quality_score_v1",
            source_field=quality_path,
        )

    partition = f"{source_file} {' '.join(str(item) for item in source.get('access', {}).get('allow_patterns', []))}".lower()
    if not metadata.get("quality_score"):
        if "medium-high-quality" in partition:
            _set_evidence(
                metadata,
                "quality_score",
                0.80,
                method="pinned_upstream_quality_partition_v1",
                source_field="access.allow_patterns",
            )
        elif "high-quality" in partition or "/4plus/" in f"/{source_file.lower()}":
            _set_evidence(
                metadata,
                "quality_score",
                0.90,
                method="pinned_upstream_quality_partition_v1",
                source_field="access.allow_patterns",
            )

    if profile_name in _COMPUTED_QUALITY_PROFILES and metadata.get("quality_score") is None:
        computed_quality, details = _computed_document_quality(text)
        _set_evidence(
            metadata,
            "quality_score",
            computed_quality,
            method="computed_document_quality_v1",
            source_field="text",
        )
        metadata["computed_document_quality"] = details

    if profile_name == "web_edu_v1":
        value, path = _find(searchable, "educational_score", "education_score", "edu_score", "int_score", "score")
        score = _numeric(value)
        if score is None and source_id == "fineweb_edu":
            # The manifest pins the released FineWeb-Edu score >=3 partition,
            # but row-level scores are still preferred whenever present.
            score, path = 3.0, "access.repo_id"
        _set_evidence(
            metadata,
            "educational_score",
            score,
            method="upstream_educational_score_or_partition_v1",
            source_field=path,
        )

    if profile_name in {"math_4plus_v1", "math_score3_v1"}:
        value, path = _find(searchable, "math_score", "int_score", "score")
        score = _numeric(value)
        if source_id == "nemotron_cc_math_4plus":
            score, path = max(score or 0.0, 4.0), "access.allow_patterns"
        elif source_id == "nemotron_cc_math_unique_3":
            score, path = max(score or 0.0, 3.0), "access.allow_patterns"
        equation_passed, equation_details = _equation_integrity(text)
        if score is None and source_id in {
            "openwebmath_unique",
            # MegaMath ships no score column either. Both corpora were selected
            # upstream for being mathematical; what the profile still needs to
            # know is that a given row really is mathematics rather than a page
            # that mentions it.
            "megamath_unique",
        }:
            # Split by signal kind rather than by count. Typeset mathematics in
            # a corpus curated for mathematics is decisive on its own, which is
            # how openwebmath has always qualified. Plain-text signal is not:
            # an invoice has an equals sign, so that path has to clear a density
            # bar before it stands in for a score.
            plain = equation_details["plain_signal_count"]
            density = plain / max(1.0, len(text) / 1_000.0)
            if equation_passed and (
                equation_details["latex_signal_count"] >= 1
                or (plain >= 8 and density >= 2.0)
            ):
                score, path = 3.0, "computed_equation_integrity_v1"
        _set_evidence(
            metadata,
            "math_score",
            score,
            method="upstream_or_structural_math_score_v1",
            source_field=path,
        )
        if equation_passed:
            _set_evidence(
                metadata,
                "equation_integrity_passed",
                True,
                method="computed_balanced_math_markup_v1",
                source_field="text",
            )
        metadata["equation_integrity"] = equation_details

    # Language evidence: prefer a document detector, then an explicit upstream
    # label, then a conservative text-local English calculation.
    full_lid, full_lid_path = _find(searchable, "full_doc_lid")
    full_lid_score, full_score_path = _find(searchable, "full_doc_lid_score")
    language, language_path = _find(
        searchable,
        "language",
        "in_language",
        "language_code",
        "content_languages",
        "lang",
    )
    language_score, language_score_path = _find(
        searchable,
        "language_probability",
        "language_score",
        "language_confidence",
        "lang_score",
        "page_average_lid_score",
    )
    label = str(full_lid or language or "").strip().lower().replace("-", "_")
    score = _numeric(full_lid_score if full_lid else language_score)
    score_path = full_score_path if full_lid else language_score_path
    english_label = label in {item.replace("-", "_") for item in _ENGLISH_LABELS}
    if profile_name == "multilingual_native_v1":
        if label and label not in {"unknown", "und", "none"} and score is not None:
            _set_evidence(
                metadata,
                "language_probability",
                score,
                method="upstream_language_detector_v1",
                source_field=score_path,
            )
            _set_evidence(
                metadata,
                "allowed_language",
                label,
                method="pinned_multilingual_partition_language_v1",
                source_field=full_lid_path if full_lid else language_path,
            )
    elif (
        source.get("category") == "code"
        or profile_name in _NON_PROSE_QUALITY_PROFILES
        or is_formal_language_row
    ):
        if source.get("category") == "code":
            gate_source = "source.category"
        elif is_formal_language_row:
            gate_source = "row.ext"
        else:
            gate_source = "processing.quality_profile"
        _set_evidence(
            metadata,
            "language_probability",
            1.0,
            method="natural_language_gate_not_applicable_to_code_v1",
            source_field=gate_source,
        )
    elif label:
        if english_label:
            _set_evidence(
                metadata,
                "language_probability",
                score if score is not None else 1.0,
                method="upstream_english_language_evidence_v1",
                source_field=score_path or full_lid_path or language_path,
            )
        elif full_lid and score is not None:
            # A high-confidence non-English full-document result is affirmative
            # negative evidence.  This prevents an English partition label from
            # masking a mislabeled PDF.
            _set_evidence(
                metadata,
                "language_probability",
                max(0.0, 1.0 - score),
                method="upstream_non_english_full_document_evidence_v1",
                source_field=score_path,
            )

    translated_partition = "high-quality-translated-to-english" in partition
    if metadata.get("language_probability") is None and translated_partition:
        _set_evidence(
            metadata,
            "language_probability",
            1.0,
            method="pinned_translated_to_english_partition_v1",
            source_field="access.allow_patterns",
        )
    if profile_name == "translated_english_v1" and translated_partition:
        _set_evidence(
            metadata,
            "translation_quality_passed",
            True,
            method="pinned_high_quality_translation_partition_v1",
            source_field="access.allow_patterns",
        )
    if metadata.get("language_probability") is None:
        computed_language = _computed_english_probability(text)
        if computed_language is not None:
            probability, details = computed_language
            _set_evidence(
                metadata,
                "language_probability",
                probability,
                method="computed_english_text_evidence_v1",
                source_field="text",
            )
            metadata["computed_language_evidence"] = details

    if profile_name == "code_interleaved_v1":
        value, path = _find(searchable, "code_text_interleave")
        interleave = _numeric(value)
        details: dict[str, Any] | None = None
        if interleave is None:
            interleave, details = _code_text_interleave(text)
            path = "text"
        _set_evidence(
            metadata,
            "code_text_interleave",
            interleave,
            method="upstream_or_computed_code_text_interleave_v1",
            source_field=path,
        )
        if details:
            metadata["code_text_interleave_evidence"] = details

    if profile_name in {"repository_code_v1", "fresh_repository_code_v1"}:
        value, path = _find(searchable, "generated_file_probability", "is_generated")
        generated_probability = _numeric(value)
        if isinstance(value, bool):
            generated_probability = 1.0 if value else 0.0
        if generated_probability is None:
            path_value, path_path = _find(
                searchable,
                "path",
                "repo_path",
                "file_info.path",
                # Stack-derived corpora name the file after the repository
                # statistic it was collected with. StarCoderData ships only
                # these, so the classifier saw no path and every record was
                # rejected for an unmeasured signal.
                "max_stars_repo_path",
                "max_issues_repo_path",
                "max_forks_repo_path",
                "meta.max_stars_repo_path",
                "file_path",
                "filename",
            )
            generated_probability, details = _generated_file_probability(
                text, str(path_value) if path_value else None
            )
            metadata["generated_file_evidence"] = details
            path = path_path or "text"
        _set_evidence(
            metadata,
            "generated_file_probability",
            generated_probability,
            method="upstream_or_computed_generated_file_classifier_v1",
            source_field=path,
        )

    if profile_name in {"proof_v1", "formal_proof_v1"}:
        statement, _ = _find(searchable, "formal_statement", "problem", "statement")
        statement_signal = bool(statement) or bool(
            re.search(r"\b(?:theorem|lemma|proposition|claim|prove|problem)\b", text, re.IGNORECASE)
        )
        argument_signal = bool(
            re.search(r"\b(?:proof|solution|therefore|hence|thus|by\s+(?:induction|contradiction|sorry))\b", text, re.IGNORECASE)
            or re.search(r"(?m):=\s*by\b", text)
        )
        # Proof-Pile-2's AlgebraicStack files are Agda, Coq, Isabelle, and Lean
        # sources under proof_v1. They are statements with arguments -- that is
        # what a formal development is -- but the English prose regexes above
        # cannot see one, so the whole component normalized to zero. Read the
        # declaration syntax of the language the row says it is written in.
        if not (statement_signal and argument_signal):
            if is_formal_language_row:
                declaration = re.search(
                    r"(?m)^\s*(?:private\s+|protected\s+|public\s+|noncomputable\s+|"
                    r"@\[[^\]]*\]\s*)*"
                    r"(?:theorem|lemma|corollary|proposition|example|instance|"
                    r"definition|def|abbrev|record|structure|data|postulate|"
                    r"inductive|Theorem|Lemma|Definition|Fixpoint|Inductive|"
                    r"Proposition|Corollary|axiom|abstract_theorem|\$[pa]\b)\b",
                    text,
                )
                # Agda declares without a keyword: a top-level `name : type` is
                # the statement and the defining equations below it are the
                # argument. Requiring a keyword would have kept rejecting the
                # component this branch exists for.
                signature = re.search(r"(?m)^[^\s:=#/-][^\n:]{0,80}\s:\s\S", text)
                body = re.search(r"(?m)^\S[^\n]*=\s*\S", text) or re.search(
                    r"(?mi)^\s*(?:proof|begin|by|:=\s*by)\b", text
                )
                if (declaration or signature) and body:
                    statement_signal = argument_signal = True
                    metadata["formal_language_proof_structure"] = {
                        "extension": formal_language,
                        "declaration": (declaration or signature).group(0).strip()[:80],
                    }
        if statement_signal and argument_signal:
            _set_evidence(
                metadata,
                "statement_and_argument",
                True,
                method="computed_proof_structure_v1",
                source_field="text",
            )

    if profile_name == "pdf_technical_v1":
        pdf = _pdf_evidence(text, searchable)
        for field in (
            "ocr_confidence",
            "repeated_header_footer_fraction",
            "reading_order_passed",
        ):
            _set_evidence(
                metadata,
                field,
                pdf[field],
                method="computed_pdf_text_structure_v1",
                source_field="text",
            )
        metadata["pdf_text_structure_evidence"] = pdf["details"]

    if profile_name in {"scientific_paper_v1", "biomedical_paper_v1"}:
        if _title_or_abstract(row, searchable, text):
            _set_evidence(
                metadata,
                "title_or_abstract",
                True,
                method="upstream_or_computed_paper_structure_v1",
                source_field="title_or_abstract_or_text",
            )
        _set_evidence(
            metadata,
            "bibliography_body_ratio",
            _bibliography_ratio(text),
            method="computed_bibliography_body_ratio_v1",
            source_field="text",
        )

    if profile_name in {"biomedical_paper_v1", "fresh_science_v1"}:
        if _open_access_from_license(str(metadata.get("license") or "")):
            _set_evidence(
                metadata,
                "open_access",
                True,
                method="explicit_reusable_open_license_v1",
                source_field="license",
            )

    if profile_name == "textbook_v1":
        complete, details = _structurally_complete_textbook(source_id, text, searchable)
        if complete:
            _set_evidence(
                metadata,
                "structurally_complete",
                True,
                method=details["method"],
                source_field="text",
            )
        metadata["textbook_structure_evidence"] = details

    if profile_name == "longform_book_v1":
        complete, details = _chapter_integrity(text)
        if (
            not complete
            and source_id == "public_domain_books_gutenberg"
            and len(text) >= 10_000
            and "project gutenberg" in text[:10_000].casefold()
            and (
                bool(
                    re.search(
                        r"(?:\*{3}\s*)?END OF (?:THE|THIS) PROJECT GUTENBERG",
                        text[-20_000:],
                        re.IGNORECASE,
                    )
                )
                or text.rstrip().endswith("***")
            )
        ):
            complete = True
            details["nonchapter_complete_work"] = "gutenberg_header_and_terminal_marker"
        if complete:
            _set_evidence(
                metadata,
                "chapter_integrity_passed",
                True,
                method="computed_longform_chapter_integrity_v1",
                source_field="text",
            )
        metadata["chapter_integrity_evidence"] = details

    if profile_name == "engineering_report_v1":
        complete, details = _engineering_structure(text)
        if complete:
            _set_evidence(
                metadata,
                "structurally_complete",
                True,
                method="computed_patent_report_structure_v1",
                source_field="text",
            )
        metadata["engineering_structure_evidence"] = details

    # Government records are immutable dated editions. A record date/year is
    # valid version evidence for those two source-specific official corpora,
    # but is not generalized to ordinary software documentation.
    if profile_name == "official_docs_v1" and metadata.get("version") is None:
        if source_id in {"metis_govreference_uk_hansard", "metis_govreference_regulations"}:
            value, path = _find(searchable, "year", "posted_date", "date", "created")
            _set_evidence(
                metadata,
                "version",
                value,
                method="government_record_edition_date_v1",
                source_field=path,
            )

    if profile_name == "legal_primary_v1":
        if source_id == "open_law_usgpo":
            _set_evidence(
                metadata,
                "primary_source",
                True,
                method="pinned_us_gpo_primary_record_source_v1",
                source_field="source.id",
            )
            _set_evidence(
                metadata,
                "jurisdiction",
                "US-federal",
                method="pinned_us_gpo_jurisdiction_v1",
                source_field="source.id",
            )
        elif source_id == "open_law_caselaw" and re.search(
            r"\b(?:united states court|supreme court|court of appeals|district court)\b",
            text[:8_000],
            re.IGNORECASE,
        ):
            _set_evidence(
                metadata,
                "primary_source",
                True,
                method="caselaw_court_record_structure_v1",
                source_field="text",
            )
            _set_evidence(
                metadata,
                "jurisdiction",
                "US",
                method="caselaw_court_record_structure_v1",
                source_field="text",
            )

    models, models_path = _find(searchable, "models_used", "generator_models", "genealogy")
    if models:
        generators = [
            value.strip()
            for value in re.split(r"[,;]", str(models))
            if value.strip()
        ]
        _set_evidence(
            metadata,
            "genealogy",
            {"generator_models": generators},
            method="upstream_generator_genealogy_v1",
            source_field=models_path,
        )

    seed_data = row.get("seed_data")
    if (
        profile_name == "textbook_synthetic_v1"
        and isinstance(seed_data, str)
        and seed_data.strip()
    ):
        _set_evidence(
            metadata,
            "source_document_id",
            "sha256:" + hashlib.sha256(seed_data.strip().encode("utf-8")).hexdigest(),
            method="hashed_upstream_seed_document_v1",
            source_field="seed_data",
        )
        format_value, format_path = _find(searchable, "format")
        audience, audience_path = _find(searchable, "audience")
        if format_value and audience:
            _set_evidence(
                metadata,
                "stylistic_diversity_passed",
                True,
                method="upstream_format_and_audience_dimensions_v1",
                source_field=f"{format_path},{audience_path}",
            )

    # An explicit per-row answer score is still read where one exists. The fact
    # that Stack Exchange rows are sorted by votes is not itself a score and is
    # deliberately not converted into ``answer_score``.
    answer_score, answer_score_path = _find(searchable, "answer_score", "accepted_answer_score")
    _set_evidence(
        metadata,
        "answer_score",
        _numeric(answer_score),
        method="upstream_answer_score_v1",
        source_field=answer_score_path,
    )

    if profile_name == "explanatory_qa_v1":
        answered, qa_details = _question_and_answer(text)
        if answered:
            _set_evidence(
                metadata,
                "question_and_answer",
                True,
                method="computed_question_and_answer_structure_v1",
                source_field="text",
            )
        metadata["question_and_answer_evidence"] = qa_details

    # Corpus-level attestations, applied last and only where the row itself
    # supplied nothing. Five sources normalized to zero because a profile asked
    # every record to restate something true of the whole pinned collection.
    #
    # This is not a way to pass a gate. Each value is written by hand in the
    # manifest beside an `attestation_basis` stating why it holds -- the field
    # is mandatory below -- it can never overwrite evidence found in the
    # document, and the audit trail records the manifest as its source so a
    # reviewer can tell an attested field from an observed one.
    for field, value in validated_attestations(source).items():
        if metadata.get(field) in (None, "", [], {}):
            _set_evidence(
                metadata,
                field,
                value,
                method="pinned_source_manifest_attestation_v1",
                source_field=f"attestations.{field}",
            )

    return metadata
