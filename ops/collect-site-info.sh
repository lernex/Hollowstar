#!/usr/bin/env bash
set -u

LUSTRE_ROOT="${METIS_LUSTRE_ROOT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lustre-root) LUSTRE_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$LUSTRE_ROOT" ]]; then
  echo "Usage: ./ops/collect-site-info.sh --lustre-root /assigned/lustre/path" >&2
  exit 2
fi

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
echo "Please also answer: Are detached long-running processes allowed on this Lustre server, and is there a site-preferred transfer service or scheduler?"
