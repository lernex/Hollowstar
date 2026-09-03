#!/bin/bash
set -euo pipefail
# Launch execution batch 1a (7 rows, 384 APUs).

: "${METIS_ABLATION_LR_DENSE_PARAM_MATCHED:?set the selected dense-param-matched learning rate}"
: "${METIS_ABLATION_LR_MOE_K4:?set the selected moe-k4 learning rate}"
: "${METIS_ABLATION_LR_MORE_CORE:?set the selected more-core learning rate}"

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-flop-matched"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-flop-matched/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/dense-flop-matched/slurm-%j.err" "$(dirname "$0")/01-dense-flop-matched.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k4/slurm-%j.err" "$(dirname "$0")/03-moe-k4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k8"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k8/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/moe-k8/slurm-%j.err" "$(dirname "$0")/04-moe-k8.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-core/slurm-%j.err" "$(dirname "$0")/10-more-core.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/more-rm/slurm-%j.err" "$(dirname "$0")/11-more-rm.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-k/slurm-%j.err" "$(dirname "$0")/12-random-k.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/random-depth/slurm-%j.err" "$(dirname "$0")/13-random-depth.sbatch"
