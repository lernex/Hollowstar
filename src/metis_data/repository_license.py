from __future__ import annotations

import re
import tarfile
from pathlib import PurePosixPath


DEFAULT_REPOSITORY_LICENSE_ALLOWLIST = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "MIT",
        "MPL-2.0",
        "Unlicense",
    }
)

ROOT_LICENSE_FILENAME = re.compile(
    r"^(?:licen[cs]e|copying|copyright)(?:[._-].*)?$",
    re.IGNORECASE,
)


def classify_repository_license_text(text: str) -> str | None:
    """Conservatively recognize a small, reviewed set of license texts.

    Dataset metadata, package metadata, and README claims are intentionally not
    inputs to this classifier.  It only recognizes the actual text of a
    repository-root license file in the archive pinned by the source index.
    """

    normalized = re.sub(r"\s+", " ", text).lower()
    if "apache license" in normalized and "version 2.0" in normalized:
        return "Apache-2.0"
    if (
        "permission is hereby granted, free of charge" in normalized
        and 'the software is provided "as is"' in normalized
    ):
        return "MIT"
    if "mozilla public license version 2.0" in normalized:
        return "MPL-2.0"
    if "this is free and unencumbered software released into the public domain" in normalized:
        return "Unlicense"
    if "cc0 1.0 universal" in normalized and "public domain" in normalized:
        return "CC0-1.0"
    if (
        "permission to use, copy, modify, and/or distribute this software" in normalized
        and 'the software is provided "as is"' in normalized
    ):
        return "ISC"
    if "redistribution and use in source and binary forms" in normalized:
        return "BSD-3-Clause" if "neither the name" in normalized else "BSD-2-Clause"
    return None


def _archive_relative_path(member_name: str) -> PurePosixPath | None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        return None
    # GitHub codeload archives always add one synthetic top-level directory.
    return PurePosixPath(*path.parts[1:])


def classify_repository_archive(
    bundle: tarfile.TarFile,
    *,
    allowlist: frozenset[str] = DEFAULT_REPOSITORY_LICENSE_ALLOWLIST,
) -> tuple[str | None, str | None]:
    """Classify an archive from repository-root license files only.

    Strong-copyleft markers in any root license-like file reject the archive,
    even if a second permissive-looking file is present.  This is deliberately
    conservative; final release still requires the independent license ledger.
    """

    candidates: list[tuple[str, str]] = []
    strong_copyleft = (
        "gnu affero general public license",
        "gnu general public license",
        "gnu lesser general public license",
        "server side public license",
        "business source license",
    )
    for member in bundle:
        if not member.isfile() or member.size <= 0 or member.size > 1_000_000:
            continue
        relative = _archive_relative_path(member.name)
        if relative is None or len(relative.parts) != 1 or not ROOT_LICENSE_FILENAME.match(relative.name):
            continue
        extracted = bundle.extractfile(member)
        if extracted is None:
            continue
        text = extracted.read().decode("utf-8", errors="replace")
        normalized = re.sub(r"\s+", " ", text).lower()
        if any(marker in normalized for marker in strong_copyleft):
            return None, None
        license_id = classify_repository_license_text(text)
        if license_id in allowlist:
            candidates.append((license_id, relative.as_posix()))
    if not candidates:
        return None, None
    return sorted(candidates, key=lambda value: (value[1].lower(), value[0]))[0]
