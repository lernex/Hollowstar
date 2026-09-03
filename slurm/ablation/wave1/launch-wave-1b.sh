#!/bin/bash
set -euo pipefail
# Launch execution batch 1b (6 rows, 360 APUs).

: "${METIS_ABLATION_LR_DENSE_PARAM_MATCHED:?set the selected dense-param-matched learning rate}"
: "${METIS_ABLATION_LR_LOOP_FIXED:?set the selected loop-fixed learning rate}"
: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-param-matched/slurm-%j.err" "$(dirname "$0")/02-dense-param-matched.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-fixed/slurm-%j.err" "$(dirname "$0")/05-loop-fixed.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/loop-pathway-frozen/slurm-%j.err" "$(dirname "$0")/06-loop-pathway-frozen.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-dense-ffn/slurm-%j.err" "$(dirname "$0")/07-mor-dense-ffn.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/mor-fixed-k/slurm-%j.err" "$(dirname "$0")/08-mor-fixed-k.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/fixed-depth-adaptive-k/slurm-%j.err" "$(dirname "$0")/09-fixed-depth-adaptive-k.sbatch"
