#!/bin/bash
set -euo pipefail
# Launch execution batch 1b (2 rows, 160 APUs).
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak026}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k/slurm-%j.err" "$(dirname "$0")/12-random-k.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth/slurm-%j.err" "$(dirname "$0")/13-random-depth.sbatch"
