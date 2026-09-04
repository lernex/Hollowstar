#!/bin/bash
set -euo pipefail
# Launch execution batch 3 (5 rows, 320 APUs).
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[020,026]}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

: "${METIS_ABLATION_LR_DENSE_PARAM_MATCHED:?set the selected dense-param-matched learning rate}"
: "${METIS_ABLATION_LR_LOOP_FIXED:?set the selected loop-fixed learning rate}"
: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-seed2"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-seed2/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched-seed2/slurm-%j.err" "$(dirname "$0")/40-dense-param-matched-seed2.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-seed2"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-seed2/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core-seed2/slurm-%j.err" "$(dirname "$0")/41-more-core-seed2.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-seed2"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-seed2/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm-seed2/slurm-%j.err" "$(dirname "$0")/42-more-rm-seed2.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed-seed2"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed-seed2/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed-seed2/slurm-%j.err" "$(dirname "$0")/43-loop-fixed-seed2.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen-seed2"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen-seed2/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen-seed2/slurm-%j.err" "$(dirname "$0")/44-loop-pathway-frozen-seed2.sbatch"
