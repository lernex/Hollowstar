#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/metis15-trainium-bootstrap.log | logger -t metis15-bootstrap -s 2>/dev/console) 2>&1

export AWS_REGION="${AWS_REGION:-us-east-1}"
CODE_S3_URI="${METIS15_CODE_S3_URI:-s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/code/metis-trainium-runtime.tar.gz}"
ENV_S3_URI="${METIS15_ENV_S3_URI:-s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/config/trainium.env}"
WORKER_PATCH_S3_URI="${METIS15_WORKER_PATCH_S3_URI:-s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/patches/metis15_neuron_spot_worker.sh}"
ROOT_DIR="/mnt/trn1/src/metis"
LOG_DIR="/mnt/trn1/logs/metis15"
VENV="/mnt/trn1/venvs/aws_neuron_venv_pytorch"

dnf install -y mdadm xfsprogs git jq awscli >/dev/null

mkdir -p /mnt/trn1
mapfile -t instance_disks < <(
  find /dev/disk/by-id -maxdepth 1 -type l -name 'nvme-Amazon_EC2_NVMe_Instance_Storage*' -print \
    | sort \
    | while read -r disk; do readlink -f "$disk"; done \
    | sort -u
)
if (( ${#instance_disks[@]} > 0 )); then
  if ! grep -qs ' /mnt/trn1 ' /proc/mounts; then
    if (( ${#instance_disks[@]} == 1 )); then
      target="$(readlink -f "${instance_disks[0]}")"
      mkfs.xfs -f "$target"
      mount "$target" /mnt/trn1
    else
      if [[ -e /dev/md0 ]]; then
        mdadm --stop /dev/md0 || true
      fi
      wipefs -a "${instance_disks[@]}" || true
      mdadm --create /dev/md0 --level=0 --raid-devices="${#instance_disks[@]}" "${instance_disks[@]}" --force
      mkfs.xfs -f /dev/md0
      mount /dev/md0 /mnt/trn1
    fi
  fi
fi

chown -R ec2-user:ec2-user /mnt/trn1
mkdir -p /mnt/trn1/src /mnt/trn1/data /mnt/trn1/checkpoints /mnt/trn1/neuron_cc_cache /mnt/trn1/venvs "$LOG_DIR"
chown -R ec2-user:ec2-user /mnt/trn1

if [[ -d /opt/aws_neuronx_venv_pytorch_2_8 ]]; then
  ln -sfn /opt/aws_neuronx_venv_pytorch_2_8 "$VENV"
elif [[ -d /opt/aws_neuronx_venv_pytorch ]]; then
  ln -sfn /opt/aws_neuronx_venv_pytorch "$VENV"
else
  dnf install -y python3.11 python3.11-pip python3.11-devel gcc-c++ >/dev/null
  python3.11 -m venv /mnt/trn1/venvs/aws_neuron_venv_pytorch_2_8
  ln -sfn /mnt/trn1/venvs/aws_neuron_venv_pytorch_2_8 "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip wheel setuptools >/dev/null
  "$VENV/bin/python" -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com >/dev/null
  "$VENV/bin/python" -m pip install \
    'neuronx-cc==2.*' \
    'torch-neuronx==2.8.*' \
    'torch-xla==2.8.*' \
    'neuronx-distributed' \
    boto3 numpy >/dev/null
fi

tmp_code="/mnt/trn1/src/metis-trainium-runtime.tar.gz"
aws s3 cp "$CODE_S3_URI" "$tmp_code"
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR"
tar --warning=no-unknown-keyword -xzf "$tmp_code" -C "$ROOT_DIR" --strip-components=1
aws s3 cp "$WORKER_PATCH_S3_URI" "$ROOT_DIR/scripts/metis15_neuron_spot_worker.sh" --only-show-errors || true
chown -R ec2-user:ec2-user "$ROOT_DIR"
chmod +x "$ROOT_DIR/scripts/metis15_neuron_pretrain.sh" "$ROOT_DIR/scripts/metis15_neuron_spot_worker.sh"

cat >/etc/metis15-trainium.env <<'ENV'
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
METIS15_ROOT_DIR=/mnt/trn1/src/metis
METIS15_S3_ROOT=s3://lernex-metis-artifacts-151025633969-us-east-1/metis15
METIS15_S3_PRETRAIN_URI=s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/pretrain-shards/base
METIS15_S3_CHECKPOINTS_URI=s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/checkpoints/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master
METIS15_S3_NEURON_CACHE_URI=s3://lernex-metis-artifacts-151025633969-us-east-1/metis15/neuron-cc-cache/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master
METIS15_S3_MAX_CONCURRENCY=32
METIS15_S3_MULTIPART_MB=16
METIS15_DATA_DIR=/mnt/trn1/data/metis15_base
METIS15_OUT_DIR=/mnt/trn1/checkpoints/metis15_base_neuron_groupstatic_cf2_hidden_lr1p5e4_sched_master
METIS15_LOG_DIR=/mnt/trn1/logs/metis15
METIS15_LOCK_TABLE=metis15-training-locks
METIS15_RUN_ID=metis15-base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master
METIS15_NEURON_VENV=/mnt/trn1/venvs/aws_neuron_venv_pytorch
PATH=/mnt/trn1/venvs/aws_neuron_venv_pytorch/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/mnt/trn1/venvs/aws_neuron_venv_pytorch/lib64/python3.12/site-packages/libneuronxla:/mnt/trn1/venvs/aws_neuron_venv_pytorch/lib/python3.12/site-packages/libneuronxla
PJRT_DEVICE=NEURON
NEURON_RT_NUM_CORES=32
NEURON_CC_CACHE=/mnt/trn1/neuron_cc_cache
NEURON_CC_FLAGS=--cache_dir=/mnt/trn1/neuron_cc_cache --auto-cast=none
METIS15_NEURON_LOCAL_BATCH_SIZE=1
METIS15_NEURON_GRAD_ACCUM_STEPS=16
METIS15_NEURON_LR=1.5e-4
METIS15_NEURON_WARMUP_STEPS=200
METIS15_NEURON_CONSTANT_LR=0
METIS15_NEURON_OVERRIDE_LR_ON_RESUME=1
METIS15_NEURON_MARK_STEP_EACH_MICROBATCH=1
METIS15_NEURON_LOCAL_LOG_METRICS=0
METIS15_NEURON_TIE_EMBEDDINGS=0
METIS15_NEURON_DISPATCH_PACK_IMPL=group_static
METIS15_NEURON_BALANCED_STATIC_LAYOUT=indexed
METIS15_NEURON_BALANCED_STATIC_ROUTER_WEIGHTS=uniform
METIS15_NEURON_BALANCED_STATIC_ROUTER_INPUT=hidden
METIS15_NEURON_ROUTER_OVERRIDE=learned
METIS15_NEURON_EXPERT_CAPACITY_FACTOR=2.0
METIS15_NEURON_ATTENTION_MODE=real
METIS15_NEURON_ATTENTION_KERNEL=nki_flash_1k
METIS15_NEURON_NKI_FLASH_LSE_DTYPE=bfloat16
METIS15_NEURON_MOE_MODE=real
METIS15_NEURON_GRAD_SYNC_MODE=all_reduce_staged
METIS15_NEURON_GRAD_SYNC_BUCKET_MB=32
METIS15_NEURON_GRAD_CLIP=0
METIS15_NEURON_EXPERT_ACTIVATION_SAFETY=clamp
METIS15_NEURON_PREINIT_OPTIMIZER_STATE=1
METIS15_NEURON_OPTIMIZER_MASTER_WEIGHTS=1
METIS15_NEURON_MUON_SCALE_MODE=match_rms_adamw
METIS15_NEURON_CE_IMPL=cross_entropy
METIS15_NEURON_CE_LOGITS_DTYPE=bfloat16
METIS15_NEURON_PERF_WARMUP_STEPS=20
METIS15_NEURON_PROFILE_COMPONENTS=1
METIS15_NEURON_DEBUG_GRAD_NORM_INTERVAL=1000
METIS15_NEURON_DEBUG_PARAM_DELTA_INTERVAL=1000
METIS15_NEURON_CHECKPOINT_INTERVAL=5000
METIS15_CHECKPOINT_SYNC_SECONDS=60
ENV

aws s3 cp "$ENV_S3_URI" /tmp/metis15-trainium.env.override --only-show-errors \
  && cat /tmp/metis15-trainium.env.override >>/etc/metis15-trainium.env \
  || true

cat >/etc/systemd/system/metis15-trainium.service <<'UNIT'
[Unit]
Description=Metis-1.5 Trainium spot worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/mnt/trn1/src/metis
EnvironmentFile=/etc/metis15-trainium.env
ExecStart=/mnt/trn1/src/metis/scripts/metis15_neuron_spot_worker.sh
Restart=on-failure
RestartSec=30
KillSignal=SIGTERM
KillMode=process
SuccessExitStatus=143
TimeoutStopSec=900
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now metis15-trainium.service
