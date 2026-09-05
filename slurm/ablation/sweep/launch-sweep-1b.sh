#!/bin/bash
set -euo pipefail
# Learning-rate sweep batch 1b: 6 runs, 300 APUs, 1,000,000,000 tokens per run.
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[007,012,020,026,056,063-064]}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr1e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr1e-4/slurm-%j.err" "$(dirname "$0")/moe-k4-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr2e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr2e-4/slurm-%j.err" "$(dirname "$0")/moe-k4-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr3e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/moe-k4-lr3e-4/slurm-%j.err" "$(dirname "$0")/moe-k4-lr3e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr1e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr1e-4/slurm-%j.err" "$(dirname "$0")/more-core-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr2e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr2e-4/slurm-%j.err" "$(dirname "$0")/more-core-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr3e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/more-core-lr3e-4/slurm-%j.err" "$(dirname "$0")/more-core-lr3e-4.sbatch"
