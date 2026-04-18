#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

log_file="${POD_LOG_FILE:-training.log}"
mkdir -p "$(dirname "$log_file")"
touch "$log_file"

stdout_targets=("$log_file")
stderr_targets=("$log_file")
if [[ -w /proc/1/fd/1 ]]; then
  stdout_targets+=("/proc/1/fd/1")
fi
if [[ -w /proc/1/fd/2 ]]; then
  stderr_targets+=("/proc/1/fd/2")
fi

export PYTHONUNBUFFERED=1

exec > >(stdbuf -oL -eL tee -a "${stdout_targets[@]}")
exec 2> >(stdbuf -oL -eL tee -a "${stderr_targets[@]}" >&2)

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

echo "[$(timestamp)] Starting: $*"
"$@"
status=$?
echo "[$(timestamp)] Finished with status $status"
exit "$status"
