#!/usr/bin/env bash
set -u

LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-}"
ROLE="acquisition"
PARTITION="${METIS_SLURM_PARTITION:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lustre-root) LUSTRE_ROOT="$2"; shift 2 ;;
    --role) ROLE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$LUSTRE_ROOT" ]]; then
  echo "Usage: ./ops/collect-site-info.sh --lustre-root /assigned/lustre/path [--role acquisition|compute|all] [--partition NAME]" >&2
  exit 2
fi

case "$ROLE" in
  acquisition|compute|all) ;;
  *) echo "--role must be acquisition, compute, or all" >&2; exit 2 ;;
esac

echo "METIS SITE INFORMATION (read-only; secrets are never printed)"
echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname -f 2>/dev/null || hostname)"
echo "user=$(id -un) uid=$(id -u)"
echo "kernel=$(uname -srm)"
echo "lustre_root=$LUSTRE_ROOT"
echo "open_file_limit=$(ulimit -n)"
echo

echo "FILESYSTEM"
df -Th "$LUSTRE_ROOT" 2>&1 || true
df -ih "$LUSTRE_ROOT" 2>&1 || true
if command -v lfs >/dev/null 2>&1; then
  lfs getstripe -d "$LUSTRE_ROOT" 2>&1 || true
  lfs quota -u "$(id -u)" "$LUSTRE_ROOT" 2>&1 || true
else
  echo "lfs=not_found"
fi
echo

echo "TOOLS"
for tool in python3 git curl hf apptainer sbatch squeue sinfo tmux screen aria2c; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "$tool=$(command -v "$tool")"
  else
    echo "$tool=not_found"
  fi
done
python3 --version 2>&1 || true
git --version 2>&1 || true
echo "HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo unset)"
echo "GITHUB_TOKEN=$([[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]] && echo set || echo unset)"
echo

echo "OUTBOUND HTTPS"
if command -v curl >/dev/null 2>&1; then
  for url in \
    https://huggingface.co \
    https://cdn-lfs.hf.co \
    https://cas-bridge.xethub.hf.co \
    https://api.github.com/rate_limit \
    https://index.commoncrawl.org/collinfo.json \
    https://data.commoncrawl.org; do
    code=$(curl --location --range 0-0 --silent --show-error --max-time 20 --output /dev/null --write-out '%{http_code}' "$url" 2>&1) || true
    echo "$url $code"
  done
else
  echo "curl unavailable"
fi
echo

echo "SLURM (expected only in the later compute environment)"
if command -v sinfo >/dev/null 2>&1; then
  sinfo --noheader --format='%P|%a|%l|%c|%m|%G' 2>&1 | head -n 40 || true
else
  echo "sinfo=not_found"
fi
if command -v scontrol >/dev/null 2>&1; then
  scontrol show config 2>/dev/null | grep -E 'MaxArraySize|MaxJobCount|MaxStepCount|SlurmctldHost' || true
fi
echo

if [[ "$ROLE" == "compute" || "$ROLE" == "all" ]]; then
  echo "==================== CPU BUILD ENVIRONMENT ===================="
  echo "The CPU data build runs thousands of short array tasks that each want a"
  echo "few cores and tens of GB, plus a handful of single large-memory reducers."
  echo "Everything below is read-only discovery; nothing is submitted."
  echo

  echo "PARTITIONS (name|avail|timelimit|cpus_per_node|mem_MB|gres|oversubscribe|nodes)"
  if command -v sinfo >/dev/null 2>&1; then
    sinfo --noheader --format='%P|%a|%l|%c|%m|%G|%h|%D' 2>&1 | head -n 60 || true
  else
    echo "sinfo=not_found"
  fi
  echo

  echo "NODE SHAPES (one representative line per distinct CPU/memory/feature shape)"
  if command -v sinfo >/dev/null 2>&1; then
    sinfo --noheader --format='%c cores | %m MB | sockets_cores_threads=%z | features=%f | gres=%G | partition=%P' 2>&1 \
      | sort -u | head -n 40 || true
  fi
  echo

  # Whether a node can host several array tasks at once decides the entire array
  # shape. Exclusive whole-node allocation would make a 400-way array request
  # 400 nodes instead of the intended ~16-32.
  echo "NODE SHARING POLICY"
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show config 2>/dev/null \
      | grep -E 'SelectType|SelectTypeParameters|DefMemPerCPU|MaxMemPerCPU|MaxMemPerNode|EnforcePartLimits|TaskPlugin|SchedulerType|PriorityType' || true
  fi
  if [[ -n "$PARTITION" ]] && command -v scontrol >/dev/null 2>&1; then
    echo "--- partition $PARTITION ---"
    scontrol show partition "$PARTITION" -o 2>&1 || true
  else
    echo "(pass --partition NAME to dump the exact partition record)"
  fi
  echo

  echo "ARRAY AND JOB LIMITS"
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show config 2>/dev/null \
      | grep -E 'MaxArraySize|MaxJobCount|MaxSubmitJobs|MaxStepCount|MinJobAge|KillWait|AccountingStorageType' || true
  fi
  echo

  echo "ACCOUNT, QOS, AND PER-USER CAPS"
  if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr --noheader --parsable2 show associations user="$(id -un)" \
      format=Cluster,Account,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit,MaxWall 2>&1 | head -n 20 || true
    echo "--- qos ---"
    sacctmgr --noheader --parsable2 show qos \
      format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU,GrpTRES 2>&1 | head -n 20 || true
  else
    echo "sacctmgr=not_found"
  fi
  echo

  # MinHash bucketing, repeated-span finding, and external sort spill heavily.
  # Node-local NVMe keeps that traffic off Lustre; if there is none, the build
  # must be told to spill to Lustre and will run slower.
  echo "NODE-LOCAL SCRATCH"
  echo "SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
  echo "TMPDIR=${TMPDIR:-unset}"
  for candidate in /tmp /local /localscratch /scratch/local /var/tmp; do
    if [[ -d "$candidate" ]]; then
      echo "--- $candidate ---"
      df -Th "$candidate" 2>&1 | tail -n 1 || true
    fi
  done
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show config 2>/dev/null | grep -E 'TmpFS|JobContainerType' || true
  fi
  echo

  echo "COMPUTE-NODE PYTHON AND CONTAINERS"
  if command -v module >/dev/null 2>&1 || [[ -n "${MODULEPATH:-}" ]]; then
    (module avail 2>&1 || true) | grep -iE 'python|cray-python|apptainer|singularity|gcc|zstd' | head -n 30 || true
  else
    echo "module=not_found"
  fi
  for interpreter in python3 python3.11 python3.12; do
    if command -v "$interpreter" >/dev/null 2>&1; then
      echo "$interpreter=$("$interpreter" --version 2>&1)"
    else
      echo "$interpreter=not_found"
    fi
  done
  echo

  echo "LUSTRE STRIPING AND INODES FOR THE BUILD ROOT"
  if command -v lfs >/dev/null 2>&1; then
    lfs df -h "$LUSTRE_ROOT" 2>&1 | tail -n 20 || true
    lfs getstripe -d "$LUSTRE_ROOT" 2>&1 || true
  fi
  df -ih "$LUSTRE_ROOT" 2>&1 || true
  echo

  echo "REMAINING QUESTIONS THAT ARE POLICY, NOT DISCOVERY"
  cat <<'QUESTIONS'
1. Which partition should CPU-only data preparation use? If the GPU partition is
   the only option, are jobs charged in GPU-hours even with --gres=none?
2. Is there an allocation budget (node-hours or core-hours) for this work, and
   roughly what is left?
3. Can compute nodes reach the same Lustre path as the login host, at the same
   absolute path? If not, what path do they see?
4. Do compute nodes have outbound HTTPS? (Not required after acquisition, but it
   determines which diagnostics can run there.)
5. Is there a preferred way to request one ~500GB-memory job for the
   contamination index, tokenizer training, and final selection reducers?
QUESTIONS
  echo
fi

if [[ "$ROLE" == "acquisition" || "$ROLE" == "all" ]]; then
  echo "Please also answer: Are detached long-running processes allowed on this Lustre server, and is there a site-preferred transfer service or scheduler?"
fi
