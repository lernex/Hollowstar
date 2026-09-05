#!/bin/bash
set -euo pipefail
# Learning-rate sweep batch 1a: 6 runs, 360 APUs, 1,000,000,000 tokens per run.
exclude_nodes="${METIS_ABLATION_EXCLUDE_NODES:-parrypeak[007,012,020,026,056,063-064]}"
sbatch_args=()
if [ -n "$exclude_nodes" ]; then sbatch_args+=(--exclude="$exclude_nodes"); fi

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr3e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4"
sbatch "${sbatch_args[@]}" --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr3e-4.sbatch"
