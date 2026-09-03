#!/bin/bash
set -euo pipefail
# Learning-rate sweep batch 1a: 6 runs, 360 APUs, 1,000,000,000 tokens per run.

mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr1e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr2e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/dense-param-matched-lr3e-4/slurm-%j.err" "$(dirname "$0")/dense-param-matched-lr3e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr1e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr1e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr2e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr2e-4.sbatch"
mkdir -p "${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4"
sbatch --output="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4/slurm-%j.out" --error="${METIS_SCRATCH:?set METIS_SCRATCH}/more-ablations/sweep/loop-fixed-lr3e-4/slurm-%j.err" "$(dirname "$0")/loop-fixed-lr3e-4.sbatch"
