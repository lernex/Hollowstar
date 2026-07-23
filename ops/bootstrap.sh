#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${METIS_PROFILE_NAME:-login2}"
ROLE="acquisition"
LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --role) ROLE="$2"; shift 2 ;;
    --lustre-root) LUSTRE_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$LUSTRE_ROOT" ]]; then
  export METIS_LUSTRE_ROOT="$LUSTRE_ROOT"
fi

PYTHON="${METIS_BOOTSTRAP_PYTHON:-python3}"
RUNTIME="${METIS_RUNTIME_DIR:-$ROOT/.metis-runtime}"
RUNTIME_INPUT="$ROOT/requirements-metis16-data.txt"
RUNTIME_LOCK="$ROOT/requirements-metis16-data.lock"
RUNTIME_MARKER="$RUNTIME/.metis-runtime-lock.json"

command -v git >/dev/null || { echo "FAIL git is not available" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "FAIL $PYTHON is not available" >&2; exit 1; }

if ! "$PYTHON" - <<'PY'
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)
PY
then
  version="$($PYTHON -c 'import platform; print(platform.python_version())')"
  echo "FAIL Python 3.11 or 3.12 is required; $PYTHON is $version." >&2
  echo "Load the site Python module first, or set METIS_BOOTSTRAP_PYTHON to an approved interpreter." >&2
  exit 1
fi

[[ -f "$RUNTIME_INPUT" ]] || {
  echo "FAIL runtime input is missing: $RUNTIME_INPUT" >&2
  exit 1
}
[[ -f "$RUNTIME_LOCK" ]] || {
  echo "FAIL hash-locked runtime is missing: $RUNTIME_LOCK" >&2
  exit 1
}

read -r RUNTIME_INPUT_SHA256 RUNTIME_LOCK_SHA256 < <(
  PYTHONPATH="$ROOT/src" "$PYTHON" - <<'PY'
from metis_data.runtime_lock import runtime_contract

contract = runtime_contract()
print(contract["input_sha256"], contract["lock_sha256"])
PY
)

runtime_matches_lock() {
  [[ -x "$RUNTIME/bin/python" && -f "$RUNTIME_MARKER" ]] || return 1
  "$RUNTIME/bin/python" - "$RUNTIME_LOCK" "$RUNTIME_MARKER" \
    "$RUNTIME_INPUT_SHA256" "$RUNTIME_LOCK_SHA256" <<'PY'
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
input_sha256 = sys.argv[3]
lock_sha256 = sys.argv[4]

def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

expected = {}
for line in lock_path.read_text(encoding="utf-8").splitlines():
    match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
    if match:
        expected[canonical(match.group(1))] = match.group(2)
installed = {
    canonical(distribution.metadata["Name"]): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
expected_marker = {
    "schema": "metis.python-runtime-install/v1",
    "input_sha256": input_sha256,
    "lock_sha256": lock_sha256,
    "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "python_version": platform.python_version(),
}
if (
    sys.implementation.name != "cpython"
    or sys.version_info[:2] not in {(3, 11), (3, 12)}
    or installed != expected
    or json.loads(marker_path.read_text(encoding="utf-8")) != expected_marker
):
    raise SystemExit(1)
PY
  "$RUNTIME/bin/python" -m pip check >/dev/null
}

if ! runtime_matches_lock >/dev/null 2>&1; then
  echo "Installing the immutable Metis-1.6 Python runtime..."
  "$PYTHON" -m venv --clear "$RUNTIME"
  "$RUNTIME/bin/python" -m pip install \
    --disable-pip-version-check \
    --only-binary=:all: \
    --require-hashes \
    --requirement "$RUNTIME_LOCK"
  "$RUNTIME/bin/python" -m pip check
  "$RUNTIME/bin/python" - "$RUNTIME_LOCK" "$RUNTIME_MARKER" \
    "$RUNTIME_INPUT_SHA256" "$RUNTIME_LOCK_SHA256" <<'PY'
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
input_sha256 = sys.argv[3]
lock_sha256 = sys.argv[4]

def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

expected = {}
for line in lock_path.read_text(encoding="utf-8").splitlines():
    match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
    if match:
        expected[canonical(match.group(1))] = match.group(2)
installed = {
    canonical(distribution.metadata["Name"]): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
if installed != expected:
    missing = sorted(set(expected) - set(installed))
    extra = sorted(set(installed) - set(expected))
    wrong = sorted(
        name for name in expected.keys() & installed.keys()
        if expected[name] != installed[name]
    )
    raise SystemExit(
        f"Runtime package set does not match lock: missing={missing}, "
        f"extra={extra}, wrong_versions={wrong}"
    )
if sys.implementation.name != "cpython" or sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(f"Unsupported runtime interpreter: {platform.python_version()}")
marker = {
    "schema": "metis.python-runtime-install/v1",
    "input_sha256": input_sha256,
    "lock_sha256": lock_sha256,
    "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "python_version": platform.python_version(),
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
PY
  runtime_matches_lock || {
    echo "FAIL installed Python environment does not match the immutable runtime lock." >&2
    exit 1
  }
fi

export METIS_RUNTIME_DIR="$RUNTIME"
"$ROOT/metisctl" init --profile "$PROFILE"
"$ROOT/metisctl" doctor --profile "$PROFILE" --role "$ROLE" --tiny-probe

echo
echo "Bootstrap complete. Run the full production preflight next:"
echo "  ./metisctl doctor --profile $PROFILE --role $ROLE"
if [[ "$ROLE" == "acquisition" ]]; then
  echo "For login2, the operator launcher runs acquisition inside GNU Screen."
fi
