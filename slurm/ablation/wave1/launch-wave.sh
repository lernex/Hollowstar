#!/bin/bash
set -euo pipefail
# Launch wave 1 (13 rows, 372 APUs).

sbatch "$(dirname "$0")/01-dense-flop-matched.sbatch"
sbatch "$(dirname "$0")/02-dense-param-matched.sbatch"
sbatch "$(dirname "$0")/03-moe-k4.sbatch"
sbatch "$(dirname "$0")/04-moe-k8.sbatch"
sbatch "$(dirname "$0")/05-loop-fixed.sbatch"
sbatch "$(dirname "$0")/06-loop-pathway-frozen.sbatch"
sbatch "$(dirname "$0")/07-mor-dense-ffn.sbatch"
sbatch "$(dirname "$0")/08-mor-fixed-k.sbatch"
sbatch "$(dirname "$0")/09-fixed-depth-adaptive-k.sbatch"
sbatch "$(dirname "$0")/10-more-core.sbatch"
sbatch "$(dirname "$0")/11-more-rm.sbatch"
sbatch "$(dirname "$0")/12-random-k.sbatch"
sbatch "$(dirname "$0")/13-random-depth.sbatch"
