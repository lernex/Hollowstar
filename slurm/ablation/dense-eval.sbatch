#!/bin/bash
# Each APU owns whole benchmarks; the harness retains its native aggregation.
#SBATCH --job-name=more-eval-dense-flop
#SBATCH --partition=parry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --exclude=parrypeak[007,012,020,026,056,063-064]

set -euo pipefail
ulimit -c 0

: "${METIS_EVAL_SOURCE:?set the committed evaluator checkout}"
: "${METIS_EVAL_REVISION:?set its full commit SHA}"
: "${METIS_NATIVE_SOURCE:?set the original training-source checkout}"
: "${METIS_EVAL_RUNTIME:?set the isolated evaluator virtualenv}"
: "${METIS_EVAL_ROOT:?set the benchmark run root}"
: "${METIS_EVAL_PREPARATION:?set the completed preparation directory}"
: "${METIS_EVAL_TASK_SOURCE:?set the immutable custom-task source checkout}"
: "${METIS_EVAL_CHECKPOINT:?set the completed dense state.pt}"
: "${METIS_EVAL_RUN_MANIFEST:?set the completed dense run.json}"
: "${METIS_EVAL_RELEASE:?set the sealed training release}"
: "${METIS_EVAL_OUTPUT:?set a fresh output directory}"

if [ "$(git -C "$METIS_EVAL_SOURCE" rev-parse HEAD)" != "$METIS_EVAL_REVISION" ] ||
   [ "$(git -C "$METIS_NATIVE_SOURCE" rev-parse HEAD)" != ba9809284cf787d3e69df96620048dd96345d66f ] ||
   [ "${SLURM_NTASKS}" -ne 4 ]; then
  echo "Refusing an unpinned source or incorrect evaluation geometry" >&2
  exit 2
fi
git -C "$METIS_EVAL_SOURCE" diff --exit-code
git -C "$METIS_EVAL_SOURCE" diff --cached --exit-code
git -C "$METIS_NATIVE_SOURCE" diff --exit-code
git -C "$METIS_NATIVE_SOURCE" diff --cached --exit-code

srun --kill-on-bad-exit=1 --network=disable_rdzv_get \
  --output="$METIS_EVAL_ROOT/logs/eval-%j-worker-%t.out" \
  --error="$METIS_EVAL_ROOT/logs/eval-%j-worker-%t.err" bash -c '
    set -euo pipefail
    export LOCAL_RANK="$SLURM_LOCALID"
    source "$METIS_EVAL_SOURCE/ops/more-ablation-runtime.sh"
    source "$METIS_EVAL_RUNTIME/bin/activate"
    export OMP_NUM_THREADS=6
    export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT
    export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
    export HF_HOME="$METIS_EVAL_ROOT/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONPATH="$METIS_NATIVE_SOURCE/src"
    common=(
      "$METIS_EVAL_SOURCE/scripts/run_more_dense_benchmark.py"
      --native-source "$METIS_NATIVE_SOURCE"
      --checkpoint "$METIS_EVAL_CHECKPOINT"
      --run-manifest "$METIS_EVAL_RUN_MANIFEST"
      --expected-run-identity 736f74c8a6fe978d535cea0ce9f2905acaee43f957646abcada76c54512797e6
      --release-root "$METIS_EVAL_RELEASE"
      --expected-harness-version 0.4.13
      --include-path "$METIS_EVAL_TASK_SOURCE/configs/more_eval_tasks"
      --include-path "$METIS_EVAL_PREPARATION"
      --preparation-manifest "$METIS_EVAL_PREPARATION/preparation.json"
      --preparation-worker "$SLURM_PROCID"
      --tasks "more_dense_benchmarks_worker_$SLURM_PROCID"
      --batch-size 8 --device cuda:0 --fewshot-seed 1234
    )
    python "${common[@]}" --diagnostic-limit 2 --bootstrap-iters 0 \
      --output-dir "$METIS_EVAL_OUTPUT/diagnostic-worker-$SLURM_PROCID"
    exec python "${common[@]}" \
      --output-dir "$METIS_EVAL_OUTPUT/worker-$SLURM_PROCID"
  '
