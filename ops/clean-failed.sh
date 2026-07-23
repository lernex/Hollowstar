#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${METIS_PROFILE_NAME:-login2}"
OLDER_THAN_HOURS="${METIS_STALE_LOCK_HOURS:-24}"

exec "$ROOT/metisctl" unlock-stale \
  --profile "$PROFILE" \
  --older-than-hours "$OLDER_THAN_HOURS"
