#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"

BUCKET="${METIS15_BUCKET:-lernex-metis-artifacts-151025633969-us-east-1}"
CODE_KEY="${METIS15_CODE_KEY:-metis15/code/metis-trainium-runtime.tar.gz}"
OLD_SFR="${METIS15_OLD_SPOT_FLEET_ID:-sfr-1df08b96-ca76-4868-9e96-a6b724e506a2}"
WORK_DIR="${METIS15_CLOUDSHELL_WORK_DIR:-$HOME/metis15-trainium-redeploy}"
CODE_BUNDLE="$WORK_DIR/metis-trainium-runtime.tar.gz"

mkdir -p "$WORK_DIR"
aws s3 cp "s3://$BUCKET/$CODE_KEY" "$CODE_BUNDLE" --only-show-errors
tar --warning=no-unknown-keyword -xzf "$CODE_BUNDLE" -C "$WORK_DIR" \
  ./infra/aws/deploy_metis15_trainium_spot_fleet.sh \
  ./infra/aws/metis15_trainium_spot_user_data.sh \
  ./infra/aws/metis15_trainium_env_override.env

cd "$WORK_DIR"
METIS15_CODE_BUNDLE="$CODE_BUNDLE" \
METIS15_OLD_SPOT_FLEET_ID="$OLD_SFR" \
bash ./infra/aws/deploy_metis15_trainium_spot_fleet.sh
