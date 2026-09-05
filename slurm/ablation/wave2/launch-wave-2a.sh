#!/bin/bash
set -euo pipefail
# Launch execution batch 2a (4 rows, 220 APUs).
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[007,012,020,026,056,063-064]}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

: "${METIS_ABLATION_LR_DENSE_PARAM_MATCHED:?set the selected dense-param-matched learning rate}"
: "${METIS_ABLATION_LR_MOE_K4:?set the selected moe-k4 learning rate}"
: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xs/slurm-%j.err" "$(dirname "$0")/20-dense-param-matched-xs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xs/slurm-%j.err" "$(dirname "$0")/21-moe-k4-xs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xs/slurm-%j.err" "$(dirname "$0")/22-more-core-xs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xs/slurm-%j.err" "$(dirname "$0")/23-more-rm-xs.sbatch"
