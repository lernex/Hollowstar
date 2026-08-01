#!/bin/bash
set -euo pipefail
# Archetype learning-rate sweep: 12 short runs at 1,000,000,000 tokens.
# Pick each archetype's winner by final loss, then launch wave 1 with
# --learning-rate set per row from its archetype.

sbatch "$(dirname "$0")/dense-param-matched-lr1e-4.sbatch"
sbatch "$(dirname "$0")/dense-param-matched-lr2e-4.sbatch"
sbatch "$(dirname "$0")/dense-param-matched-lr3e-4.sbatch"
sbatch "$(dirname "$0")/moe-k4-lr1e-4.sbatch"
sbatch "$(dirname "$0")/moe-k4-lr2e-4.sbatch"
sbatch "$(dirname "$0")/moe-k4-lr3e-4.sbatch"
sbatch "$(dirname "$0")/loop-fixed-lr1e-4.sbatch"
sbatch "$(dirname "$0")/loop-fixed-lr2e-4.sbatch"
sbatch "$(dirname "$0")/loop-fixed-lr3e-4.sbatch"
sbatch "$(dirname "$0")/more-core-lr1e-4.sbatch"
sbatch "$(dirname "$0")/more-core-lr2e-4.sbatch"
sbatch "$(dirname "$0")/more-core-lr3e-4.sbatch"
