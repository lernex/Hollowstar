#!/usr/bin/env bash
set -euo pipefail
umask 027

CONFIG="${1:?config path is required}"
CAMPAIGN_ROOT="${2:?campaign root is required}"
RUNTIME_SETUP="${3:?runtime setup is required}"

source "$RUNTIME_SETUP"
# BASH_SOURCE resolves into /var/spool/slurmd here: Slurm runs a staged copy of
# this script, not the one in the checkout. The launcher exports METIS_ROOT.
cd "${METIS_ROOT:?METIS_ROOT is required: submit through the Portage launcher}"
export PYTHONUNBUFFERED=1

checkpoint_requested=0
supervisor_pid=""

request_checkpoint() {
  checkpoint_requested=1
  if [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    kill -USR1 "$supervisor_pid"
  fi
}

trap request_checkpoint USR1 TERM

python3 -m metis_portage.family \
  --config "$CONFIG" \
  --campaign-root "$CAMPAIGN_ROOT" &
supervisor_pid=$!

set +e
wait "$supervisor_pid"
status=$?
if (( checkpoint_requested == 1 )) && kill -0 "$supervisor_pid" 2>/dev/null; then
  wait "$supervisor_pid"
  status=$?
fi
set -e

if (( status == 75 )); then
  python3 -m metis_portage.launcher validate-requeue \
    --campaign-root "$CAMPAIGN_ROOT"
  echo "Durable family checkpoints verified; requeueing Slurm job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

exit "$status"
