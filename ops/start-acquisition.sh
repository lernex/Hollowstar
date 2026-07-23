#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="login2"
SESSION="metis16-acquisition"
LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-}"
QUOTA_ACKNOWLEDGEMENT="${METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --lustre-root) LUSTRE_ROOT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --quota-acknowledgement) QUOTA_ACKNOWLEDGEMENT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$LUSTRE_ROOT" ]]; then
  echo "FAIL: pass the confirmed user-owned directory with --lustre-root." >&2
  echo "Do not use /lus/lustre1 itself." >&2
  exit 2
fi
if [[ ! "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "FAIL: --session may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ "$LUSTRE_ROOT" != /* ]]; then
  echo "FAIL: --lustre-root must be an absolute path." >&2
  exit 2
fi
case "$QUOTA_ACKNOWLEDGEMENT" in
  ""|administrator-confirmed|unlimited) ;;
  *)
    echo "FAIL: --quota-acknowledgement must be administrator-confirmed or unlimited." >&2
    exit 2
    ;;
esac
case "${LUSTRE_ROOT%/}" in
  /|/lus|/lus/lustre1|/lus/lustre1/vollmerc)
    echo "FAIL: refusing unsafe shared filesystem root ${LUSTRE_ROOT%/}." >&2
    exit 2
    ;;
  /lus/lustre1/vollmerc/*) ;;
  *)
    echo "FAIL: login2 acquisition must use a child of /lus/lustre1/vollmerc." >&2
    exit 2
    ;;
esac

command -v screen >/dev/null 2>&1 || {
  echo "FAIL: GNU Screen is required on login2." >&2
  exit 1
}
command -v git >/dev/null 2>&1 || { echo "FAIL: git is unavailable." >&2; exit 1; }

mkdir -p "$LUSTRE_ROOT"
[[ -d "$LUSTRE_ROOT" && -w "$LUSTRE_ROOT" ]] || {
  echo "FAIL: Lustre directory is not writable: $LUSTRE_ROOT" >&2
  exit 1
}
LUSTRE_ROOT="$(cd "$LUSTRE_ROOT" && pwd -P)"
case "$LUSTRE_ROOT" in
  /lus/lustre1/vollmerc/*) ;;
  *)
    echo "FAIL: resolved Lustre directory escaped /lus/lustre1/vollmerc: $LUSTRE_ROOT" >&2
    exit 2
    ;;
esac

REPOSITORY_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
SESSION_DESCRIPTOR_DIR="$HOME/.cache/metis/sessions"
SESSION_DESCRIPTOR="$SESSION_DESCRIPTOR_DIR/$SESSION"
mkdir -p "$SESSION_DESCRIPTOR_DIR"
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}[[:space:]]"; then
  existing_root="$(sed -n '1p' "$SESSION_DESCRIPTOR" 2>/dev/null || true)"
  existing_profile="$(sed -n '2p' "$SESSION_DESCRIPTOR" 2>/dev/null || true)"
  existing_commit="$(sed -n '3p' "$SESSION_DESCRIPTOR" 2>/dev/null || true)"
  if [[ "$existing_root" != "$LUSTRE_ROOT" || "$existing_profile" != "$PROFILE" || "$existing_commit" != "$REPOSITORY_COMMIT" ]]; then
    echo "FAIL: Screen session $SESSION belongs to another root, profile, or repository commit." >&2
    echo "Attach to it and inspect before choosing a different --session name." >&2
    exit 1
  fi
  echo "Acquisition session already exists: $SESSION"
  echo "Attach with: screen -r $SESSION"
  exit 0
fi

# Hugging Face can read its normal 0600 credential file. If none is present,
# capture a token without echoing or placing it in argv, shell history, or logs.
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -s "$HF_TOKEN_FILE" ]]; then
    "$ROOT/ops/validate-hf-token-file.sh" "$HF_TOKEN_FILE"
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  elif [[ -s "$HOME/.huggingface/token" ]]; then
    "$ROOT/ops/validate-hf-token-file.sh" "$HOME/.huggingface/token"
    HF_TOKEN="$(<"$HOME/.huggingface/token")"
  else
    read -rsp "Hugging Face read token: " HF_TOKEN
    echo
  fi
  [[ -n "$HF_TOKEN" ]] || { echo "FAIL: a Hugging Face token is required." >&2; exit 1; }
  export HF_TOKEN
fi

# GitHub Archive's compact event rows do not reliably carry fork/mirror
# identity. Resolve that metadata with a read-only token, inherited by Screen
# but never written to argv, logs, receipts, or repository files.
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  if [[ -n "${GH_TOKEN:-}" ]]; then
    GITHUB_TOKEN="$GH_TOKEN"
  elif command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
    GITHUB_TOKEN="$(gh auth token)"
  else
    read -rsp "GitHub read-only token (repository Metadata: read): " GITHUB_TOKEN
    echo
  fi
  [[ -n "$GITHUB_TOKEN" ]] || { echo "FAIL: a read-only GitHub token is required." >&2; exit 1; }
  export GITHUB_TOKEN
fi

export METIS_LUSTRE_ROOT="$LUSTRE_ROOT"
export METIS_PROFILE_NAME="$PROFILE"
if [[ -n "$QUOTA_ACKNOWLEDGEMENT" ]]; then
  export METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT="$QUOTA_ACKNOWLEDGEMENT"
fi
export METIS_RUNTIME_DIR="${METIS_RUNTIME_DIR:-$HOME/.cache/metis/runtime-login2}"
export METIS_SCREEN_SESSION="$SESSION"
if [[ -z "${METIS_BOOTSTRAP_PYTHON:-}" && -x /usr/bin/python3.11 ]]; then
  export METIS_BOOTSTRAP_PYTHON=/usr/bin/python3.11
fi

LOG_DIR="$LUSTRE_ROOT/logs/metis-1.6-data-r1/acquisition"
mkdir -p "$LOG_DIR"
export METIS_ACQUISITION_LOG="$LOG_DIR/screen.log"

descriptor_temporary="$SESSION_DESCRIPTOR.$$"
printf '%s\n%s\n%s\n' "$LUSTRE_ROOT" "$PROFILE" "$REPOSITORY_COMMIT" > "$descriptor_temporary"
mv "$descriptor_temporary" "$SESSION_DESCRIPTOR"

# Screen inherits the already-exported credential environment. Tokens never
# appear in this command line or in the generated state/log files.
ACQUISITION_COMMAND=("$ROOT/ops/run-acquisition.sh")
if command -v nice >/dev/null 2>&1; then
  ACQUISITION_COMMAND=(nice -n 10 "${ACQUISITION_COMMAND[@]}")
fi
screen -DmS "$SESSION" "${ACQUISITION_COMMAND[@]}"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}[[:space:]]"; then
  rm -f "$SESSION_DESCRIPTOR"
  echo "FAIL: the Screen session exited during startup. Last log lines:" >&2
  tail -n 30 "$METIS_ACQUISITION_LOG" >&2 2>/dev/null || true
  exit 1
fi

echo "Metis-1.6 acquisition started in GNU Screen on $(hostname)."
echo "Attach:  screen -r $SESSION"
echo "Status:  METIS_LUSTRE_ROOT='$LUSTRE_ROOT' ./metisctl status --profile $PROFILE"
echo "Log:     tail -f '$METIS_ACQUISITION_LOG'"
