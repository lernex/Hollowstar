#!/bin/bash
set -euo pipefail
# Launch wave 3 (3 rows, 336 APUs).

sbatch "$(dirname "$0")/40-dense-param-matched-seed2.sbatch"
sbatch "$(dirname "$0")/41-more-core-seed2.sbatch"
sbatch "$(dirname "$0")/42-more-rm-seed2.sbatch"
