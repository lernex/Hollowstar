#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${METIS15_BUCKET:-lernex-metis-artifacts-151025633969-us-east-1}"
CODE_KEY="${METIS15_CODE_KEY:-metis15/code/metis-trainium-runtime.tar.gz}"
CODE_S3="s3://$BUCKET/$CODE_KEY"
RUN_ID="${METIS15_RUN_ID:-metis15-base-neuron}"
LOCK_TABLE="${METIS15_LOCK_TABLE:-metis15-training-locks}"
LT_NAME="${METIS15_LAUNCH_TEMPLATE_NAME:-metis15-trainium-spot-worker}"
ROLE_NAME="${METIS15_INSTANCE_ROLE_NAME:-MetisTrainiumSpotInstanceRole}"
PROFILE_NAME="${METIS15_INSTANCE_PROFILE_NAME:-MetisTrainiumSpotInstanceProfile}"
SG_NAME="${METIS15_SECURITY_GROUP_NAME:-metis15-trainium-ssh}"
AMI_ID="${METIS15_AMI_ID:-ami-0fad8fde16baeeccf}"
KEY_NAME="${METIS15_KEY_NAME:-Codex}"
SSH_CIDR="${METIS15_SSH_CIDR:-38.246.0.232/32}"
VPC_ID="${METIS15_VPC_ID:-vpc-091ebf2fe4e9d1546}"
SUBNET_1C="${METIS15_SUBNET_1C:-subnet-0a7102a248d3631c4}"
SUBNET_1D="${METIS15_SUBNET_1D:-subnet-0b4eb3ba01520bf0e}"
SUBNET_1F="${METIS15_SUBNET_1F:-subnet-0547a0465cf1a6d3b}"
OLD_SFR="${METIS15_OLD_SPOT_FLEET_ID:-sfr-489ba74d-15a3-4964-a9c6-46c216bee857}"
CODE_BUNDLE="${METIS15_CODE_BUNDLE:-$HOME/metis-trainium-runtime.tar.gz}"
ENV_KEY="${METIS15_ENV_KEY:-metis15/config/trainium.env}"
ENV_S3="s3://$BUCKET/$ENV_KEY"
ENV_OVERRIDE_FILE="${METIS15_ENV_OVERRIDE_FILE:-infra/aws/metis15_trainium_env_override.env}"

start_time_utc() {
  python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
}

base64_one_line() {
  if base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$1"
  else
    base64 <"$1" | tr -d '\n'
  fi
}

if [[ ! -f "$CODE_BUNDLE" ]]; then
  echo "Code bundle not found: $CODE_BUNDLE" >&2
  echo "Build it first, or set METIS15_CODE_BUNDLE." >&2
  exit 1
fi

echo "== upload code bundle =="
aws s3 cp "$CODE_BUNDLE" "$CODE_S3" --only-show-errors
aws s3 ls "$CODE_S3"

echo "== upload mutable trainium env overlay =="
aws s3 cp "$ENV_OVERRIDE_FILE" "$ENV_S3" --only-show-errors
aws s3 ls "$ENV_S3"

echo "== dynamodb lock table =="
if ! aws dynamodb describe-table --table-name "$LOCK_TABLE" >/dev/null 2>&1; then
  aws dynamodb create-table \
    --table-name "$LOCK_TABLE" \
    --attribute-definitions AttributeName=run_id,AttributeType=S \
    --key-schema AttributeName=run_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST >/dev/null
  aws dynamodb wait table-exists --table-name "$LOCK_TABLE"
fi
aws dynamodb update-time-to-live \
  --table-name "$LOCK_TABLE" \
  --time-to-live-specification Enabled=true,AttributeName=expires_at >/dev/null || true

echo "== IAM instance role/profile =="
cat >/tmp/metis-ec2-trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file:///tmp/metis-ec2-trust.json >/dev/null
cat >/tmp/metis-instance-policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": [
        "arn:aws:s3:::$BUCKET",
        "arn:aws:s3:::$BUCKET/metis15/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:DescribeTable"],
      "Resource": "arn:aws:dynamodb:us-east-1:$ACCOUNT_ID:table/$LOCK_TABLE"
    }
  ]
}
JSON
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name MetisTrainiumSpotWorkerPolicy \
  --policy-document file:///tmp/metis-instance-policy.json
aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1 \
  || aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
aws iam add-role-to-instance-profile \
  --instance-profile-name "$PROFILE_NAME" \
  --role-name "$ROLE_NAME" >/dev/null 2>&1 || true

echo "== security group =="
SG_ID="$(aws ec2 describe-security-groups \
  --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values="$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)"
if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  SG_ID="$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Metis Trainium SSH" \
    --vpc-id "$VPC_ID" \
    --query GroupId \
    --output text)"
fi
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$SSH_CIDR,Description=Codex SSH}]" >/dev/null 2>&1 || true

echo "== S3 gateway endpoint =="
S3_SERVICE="com.amazonaws.$AWS_REGION.s3"
ROUTE_TABLE_IDS=()
while IFS= read -r route_table_id; do
  [[ -n "$route_table_id" ]] && ROUTE_TABLE_IDS+=("$route_table_id")
done < <(
  aws ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'RouteTables[].RouteTableId' \
    --output text | tr '\t' '\n' | sort -u
)
if (( ${#ROUTE_TABLE_IDS[@]} > 0 )); then
  S3_ENDPOINT_ID="$(aws ec2 describe-vpc-endpoints \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=service-name,Values=$S3_SERVICE" "Name=vpc-endpoint-type,Values=Gateway" \
    --query 'VpcEndpoints[0].VpcEndpointId' \
    --output text 2>/dev/null || true)"
  if [[ -z "$S3_ENDPOINT_ID" || "$S3_ENDPOINT_ID" == "None" ]]; then
    S3_ENDPOINT_ID="$(aws ec2 create-vpc-endpoint \
      --vpc-id "$VPC_ID" \
      --service-name "$S3_SERVICE" \
      --vpc-endpoint-type Gateway \
      --route-table-ids "${ROUTE_TABLE_IDS[@]}" \
      --query 'VpcEndpoint.VpcEndpointId' \
      --output text)"
    echo "s3_gateway_endpoint_created=$S3_ENDPOINT_ID route_tables=${ROUTE_TABLE_IDS[*]}"
  else
    aws ec2 modify-vpc-endpoint \
      --vpc-endpoint-id "$S3_ENDPOINT_ID" \
      --add-route-table-ids "${ROUTE_TABLE_IDS[@]}" >/dev/null || true
    echo "s3_gateway_endpoint=$S3_ENDPOINT_ID route_tables=${ROUTE_TABLE_IDS[*]}"
  fi
else
  echo "s3_gateway_endpoint_skipped=no_route_tables_found"
fi

echo "== launch template =="
rm -rf /tmp/metis-lt
mkdir -p /tmp/metis-lt
tar -xzf "$CODE_BUNDLE" -C /tmp/metis-lt './infra/aws/metis15_trainium_spot_user_data.sh'
USER_DATA_B64="$(base64_one_line /tmp/metis-lt/infra/aws/metis15_trainium_spot_user_data.sh)"
cat >/tmp/metis-lt-data.json <<JSON
{
  "ImageId": "$AMI_ID",
  "KeyName": "$KEY_NAME",
  "IamInstanceProfile": {"Name": "$PROFILE_NAME"},
  "SecurityGroupIds": ["$SG_ID"],
  "UserData": "$USER_DATA_B64",
  "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
  "BlockDeviceMappings": [
    {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": true}}
  ],
  "TagSpecifications": [
    {"ResourceType": "instance", "Tags": [
      {"Key": "Name", "Value": "metis15-trainium-spot-worker"},
      {"Key": "MetisRun", "Value": "$RUN_ID"},
      {"Key": "ManagedBy", "Value": "codex"}
    ]},
    {"ResourceType": "volume", "Tags": [
      {"Key": "Name", "Value": "metis15-trainium-root"},
      {"Key": "MetisRun", "Value": "$RUN_ID"},
      {"Key": "ManagedBy", "Value": "codex"}
    ]}
  ]
}
JSON
if aws ec2 describe-launch-templates --launch-template-names "$LT_NAME" >/dev/null 2>&1; then
  LT_VER="$(aws ec2 create-launch-template-version \
    --launch-template-name "$LT_NAME" \
    --source-version '$Default' \
    --launch-template-data file:///tmp/metis-lt-data.json \
    --query 'LaunchTemplateVersion.VersionNumber' \
    --output text)"
  aws ec2 modify-launch-template --launch-template-name "$LT_NAME" --default-version "$LT_VER" >/dev/null
else
  aws ec2 create-launch-template --launch-template-name "$LT_NAME" --launch-template-data file:///tmp/metis-lt-data.json >/dev/null
  LT_VER=1
fi
echo "launch_template=$LT_NAME version=$LT_VER sg=$SG_ID"

echo "== request maintained spot fleet =="
VALID_UNTIL="$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)"
cat >/tmp/metis-spot-fleet.json <<JSON
{
  "IamFleetRole": "arn:aws:iam::$ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role",
  "AllocationStrategy": "priceCapacityOptimized",
  "TargetCapacity": 1,
  "Type": "maintain",
  "ValidUntil": "$VALID_UNTIL",
  "TerminateInstancesWithExpiration": true,
  "InstanceInterruptionBehavior": "terminate",
  "SpotMaintenanceStrategies": {
    "CapacityRebalance": {
      "ReplacementStrategy": "launch-before-terminate",
      "TerminationDelay": 120
    }
  },
  "LaunchTemplateConfigs": [
    {
      "LaunchTemplateSpecification": {
        "LaunchTemplateName": "$LT_NAME",
        "Version": "$LT_VER"
      },
      "Overrides": [
        {"InstanceType": "trn1n.32xlarge", "SubnetId": "$SUBNET_1D"},
        {"InstanceType": "trn1.32xlarge", "SubnetId": "$SUBNET_1D"},
        {"InstanceType": "trn1.32xlarge", "SubnetId": "$SUBNET_1C"},
        {"InstanceType": "trn1.32xlarge", "SubnetId": "$SUBNET_1F"}
      ]
    }
  ]
}
JSON
FLEET_ID="$(aws ec2 request-spot-fleet \
  --spot-fleet-request-config file:///tmp/metis-spot-fleet.json \
  --query 'SpotFleetRequestId' \
  --output text)"
echo "new_fleet_id=$FLEET_ID"
echo "== wait for new fleet capacity =="
NEW_ACTIVE_COUNT=0
for _ in $(seq 1 45); do
  NEW_ACTIVE_COUNT="$(aws ec2 describe-spot-fleet-instances \
    --spot-fleet-request-id "$FLEET_ID" \
    --query 'length(ActiveInstances)' \
    --output text 2>/dev/null || echo 0)"
  if [[ "$NEW_ACTIVE_COUNT" != "None" && "$NEW_ACTIVE_COUNT" -gt 0 ]]; then
    break
  fi
  sleep 20
done
echo "new_fleet_active_instances=$NEW_ACTIVE_COUNT"
aws ec2 describe-spot-fleet-requests \
  --spot-fleet-request-ids "$FLEET_ID" \
  --query 'SpotFleetRequestConfigs[0].{Id:SpotFleetRequestId,State:SpotFleetRequestState,Activity:ActivityStatus,Target:SpotFleetRequestConfig.TargetCapacity,Type:SpotFleetRequestConfig.Type,Alloc:SpotFleetRequestConfig.AllocationStrategy,Rebalance:SpotFleetRequestConfig.SpotMaintenanceStrategies}' \
  --output json
aws ec2 describe-spot-fleet-request-history \
  --spot-fleet-request-id "$FLEET_ID" \
  --start-time "$(start_time_utc)" \
  --query 'HistoryRecords[].{Time:Timestamp,Type:EventType,Info:EventInformation.EventDescription}' \
  --output table || true
aws ec2 describe-instances \
  --filters Name=tag:MetisRun,Values="$RUN_ID" Name=instance-state-name,Values=pending,running \
  --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,AZ:Placement.AvailabilityZone,PublicIp:PublicIpAddress,Launch:LaunchTime}' \
  --output table || true

echo "== cancel old fleet if active =="
OLD_STATE="$(aws ec2 describe-spot-fleet-requests \
  --spot-fleet-request-ids "$OLD_SFR" \
  --query 'SpotFleetRequestConfigs[0].SpotFleetRequestState' \
  --output text 2>/dev/null || true)"
if [[ "$NEW_ACTIVE_COUNT" -gt 0 && -n "$OLD_STATE" && "$OLD_STATE" != "cancelled" && "$OLD_STATE" != "cancelled_terminating" && "$OLD_STATE" != "cancelled_running" ]]; then
  aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids "$OLD_SFR" --terminate-instances >/dev/null || true
  echo "old_fleet_cancelled=$OLD_SFR previous_state=$OLD_STATE"
elif [[ "$NEW_ACTIVE_COUNT" -le 0 && -n "$OLD_STATE" && "$OLD_STATE" != "cancelled" && "$OLD_STATE" != "cancelled_terminating" && "$OLD_STATE" != "cancelled_running" ]]; then
  echo "old_fleet_kept=$OLD_SFR previous_state=$OLD_STATE reason=no_new_active_capacity"
else
  echo "old_fleet_state=${OLD_STATE:-missing}"
fi
