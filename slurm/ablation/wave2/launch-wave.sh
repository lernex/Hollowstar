#!/bin/bash
set -euo pipefail
# Launch wave 2 (8 rows, 308 APUs).

sbatch "$(dirname "$0")/20-dense-param-matched-xs.sbatch"
sbatch "$(dirname "$0")/21-moe-k4-xs.sbatch"
sbatch "$(dirname "$0")/22-more-core-xs.sbatch"
sbatch "$(dirname "$0")/23-more-rm-xs.sbatch"
sbatch "$(dirname "$0")/24-dense-param-matched-xxs.sbatch"
sbatch "$(dirname "$0")/25-moe-k4-xxs.sbatch"
sbatch "$(dirname "$0")/26-more-core-xxs.sbatch"
sbatch "$(dirname "$0")/27-more-rm-xxs.sbatch"
