#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-/lus/lustre1/vollmerc/metis-1.6}"
CONFIG="${METIS_PORTAGE_CONFIG:-$ROOT/configs/metis16/portage-training.yaml}"
MODE="launch"

usage() {
  cat <<'EOF'
Usage: ./ops/start-portage-training.sh [OPTIONS]

Autonomously validate, tune, and launch simultaneous Praxis and Logos training.

Options:
  --lustre-root PATH  User-owned Portage Lustre root
  --config PATH       Portage campaign configuration
  --status            Inspect the campaign without submitting work
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lustre-root)
      [[ $# -ge 2 ]] || {
        echo "FAIL: --lustre-root requires a path." >&2
        exit 2
      }
      LUSTRE_ROOT="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || {
        echo "FAIL: --config requires a path." >&2
        exit 2
      }
      CONFIG="$2"
      shift 2
      ;;
    --status) MODE="status"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${LUSTRE_ROOT%/}" in
  /lus/lustre1/vollmerc/*) ;;
  *)
    echo "FAIL: Portage training requires a user-owned child of /lus/lustre1/vollmerc." >&2
    exit 2
    ;;
esac

for command in python3 git sbatch srun scontrol squeue; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "FAIL: required Portage command is unavailable: $command" >&2
    exit 1
  }
done

PYTHON=python3
if ! "$PYTHON" -c 'import sys, yaml; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1; then
  export METIS_LUSTRE_ROOT="${LUSTRE_ROOT%/}"
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT"
  exec "$PYTHON" "$ROOT/ops/bootstrap-portage-login-runtime.py" \
    --root "$ROOT" \
    --lustre-root "${LUSTRE_ROOT%/}" \
    --config "$CONFIG" \
    --mode "$MODE"
fi

export METIS_LUSTRE_ROOT="${LUSTRE_ROOT%/}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

exec "$PYTHON" -m metis_portage.launcher --config "$CONFIG" "$MODE"
