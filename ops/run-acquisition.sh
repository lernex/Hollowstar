#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${METIS_PROFILE_NAME:-login2}"
LUSTRE_ROOT="${METIS_LUSTRE_ROOT:?METIS_LUSTRE_ROOT is required}"
LOG="${METIS_ACQUISITION_LOG:-$LUSTRE_ROOT/logs/metis-1.6-data-r1/acquisition/screen.log}"

export HF_HOME="$LUSTRE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$LUSTRE_ROOT/cache/tmp"
export PIP_CACHE_DIR="$LUSTRE_ROOT/cache/pip"

mkdir -p "$(dirname "$LOG")" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$PIP_CACHE_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Metis-1.6 login2 acquisition"
"$ROOT/ops/bootstrap.sh" \
  --profile "$PROFILE" \
  --role acquisition \
  --lustre-root "$LUSTRE_ROOT"
"$ROOT/metisctl" run-acquisition --profile "$PROFILE"
HANDOFF="$LUSTRE_ROOT/state/metis-1.6-data-r1/ACQUISITION_READY.json"
if [[ ! -f "$HANDOFF" ]]; then
  echo "FAIL: acquisition returned without the immutable handoff: $HANDOFF" >&2
  exit 1
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Acquisition and immutable handoff complete"
