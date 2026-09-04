#!/bin/bash
set -euo pipefail
# Launch execution batch 2 (8 rows, 308 APUs).
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[020,026]}"
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
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xxs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xxs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-xxs/slurm-%j.err" "$(dirname "$0")/24-dense-param-matched-xxs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xxs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xxs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4-xxs/slurm-%j.err" "$(dirname "$0")/25-moe-k4-xxs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xxs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xxs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-xxs/slurm-%j.err" "$(dirname "$0")/26-more-core-xxs.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xxs"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xxs/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-xxs/slurm-%j.err" "$(dirname "$0")/27-more-rm-xxs.sbatch"
