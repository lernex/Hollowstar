#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "FAIL: validate-hf-token-file.sh expects exactly one token-file path." >&2
  exit 2
fi

TOKEN_FILE="$1"
if [[ -L "$TOKEN_FILE" ]]; then
  echo "FAIL: Hugging Face token file must not be a symbolic link: $TOKEN_FILE" >&2
  exit 1
fi
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "FAIL: Hugging Face token path is not a regular file: $TOKEN_FILE" >&2
  exit 1
fi

metadata=""
if metadata="$(stat -c '%u %a' -- "$TOKEN_FILE" 2>/dev/null)"; then
  :
elif metadata="$(stat -f '%u %Lp' "$TOKEN_FILE" 2>/dev/null)"; then
  :
else
  echo "FAIL: unable to inspect Hugging Face token file ownership and permissions." >&2
  exit 1
fi

read -r owner_uid mode <<<"$metadata"
current_uid="$(id -u)"
if [[ "$owner_uid" != "$current_uid" ]]; then
  echo "FAIL: Hugging Face token file must be owned by the current user." >&2
  exit 1
fi
if [[ ! "$mode" =~ ^[0-7]{3,4}$ ]]; then
  echo "FAIL: unable to parse Hugging Face token file permissions." >&2
  exit 1
fi
mode_value=$((8#$mode))
if (( mode_value & 07177 )); then
  echo "FAIL: Hugging Face token file permissions must be 0600 or stricter (no execute, special, group, or world bits)." >&2
  exit 1
fi

echo "PASS: Hugging Face token file owner and permissions are safe."
