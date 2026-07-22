#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="portage"
LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --lustre-root) LUSTRE_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$LUSTRE_ROOT" ]]; then
  export METIS_LUSTRE_ROOT="$LUSTRE_ROOT"
fi

PYTHON="${METIS_BOOTSTRAP_PYTHON:-python3}"
RUNTIME="${METIS_RUNTIME_DIR:-$ROOT/.metis-runtime}"

command -v git >/dev/null || { echo "FAIL git is not available" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "FAIL $PYTHON is not available" >&2; exit 1; }

if ! "$PYTHON" - <<'PY'
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)
PY
then
  version="$($PYTHON -c 'import platform; print(platform.python_version())')"
  echo "FAIL Python 3.11 or 3.12 is required; $PYTHON is $version." >&2
  echo "Load the Portage Python module first, or set METIS_BOOTSTRAP_PYTHON to an approved interpreter." >&2
  exit 1
fi

if [[ ! -x "$RUNTIME/bin/python" ]]; then
  "$PYTHON" -m venv "$RUNTIME"
fi

"$RUNTIME/bin/python" -m pip install --upgrade pip
"$RUNTIME/bin/python" -m pip install --requirement "$ROOT/requirements-metis16-data.txt"
"$RUNTIME/bin/python" -m pip install --no-deps --editable "$ROOT"

export METIS_RUNTIME_DIR="$RUNTIME"
"$ROOT/metisctl" init --profile "$PROFILE"
"$ROOT/metisctl" doctor --profile "$PROFILE" --tiny-probe

echo
echo "Bootstrap complete. Run the full production preflight next:"
echo "  ./metisctl doctor --profile $PROFILE"
echo "Only after every required check passes, submit independent acquisition with:"
echo "  ./metisctl submit download --profile $PROFILE"
