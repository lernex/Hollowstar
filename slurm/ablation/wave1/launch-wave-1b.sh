#!/bin/bash
set -euo pipefail
# Launch execution batch 1b (6 rows, 360 APUs).
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[007,020,026,063-064]}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

: "${METIS_ABLATION_LR_DENSE_PARAM_MATCHED:?set the selected dense-param-matched learning rate}"
: "${METIS_ABLATION_LR_LOOP_FIXED:?set the selected loop-fixed learning rate}"
: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn/slurm-%j.err" "$(dirname "$0")/07-mor-dense-ffn.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k/slurm-%j.err" "$(dirname "$0")/08-mor-fixed-k.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k/slurm-%j.err" "$(dirname "$0")/09-fixed-depth-adaptive-k.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen/slurm-%j.err" "$(dirname "$0")/06-loop-pathway-frozen.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed/slurm-%j.err" "$(dirname "$0")/05-loop-fixed.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched/slurm-%j.err" "$(dirname "$0")/02-dense-param-matched.sbatch"
