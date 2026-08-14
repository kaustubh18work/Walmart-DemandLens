#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
CLUSTER="${ECS_CLUSTER:-walmart-demandlens}"
OLD_SERVICE="${OLD_ECS_SERVICE:-walmart-demandlens-service}"
OLD_TASK_FAMILY="${OLD_TASK_FAMILY:-walmart-demandlens}"
OLD_SECURITY_GROUP="${OLD_SECURITY_GROUP:-sg-0eda88c095a8e7715}"
OLD_LOG_GROUP="${OLD_LOG_GROUP:-/ecs/walmart-demandlens}"

printf '\n== Current services ==\n'
aws ecs list-services --cluster "$CLUSTER" --region "$REGION"

printf '\n== Deleting the old customer-managed ECS service: %s ==\n' "$OLD_SERVICE"
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$OLD_SERVICE" \
  --desired-count 0 \
  --region "$REGION" >/dev/null

aws ecs wait services-stable \
  --cluster "$CLUSTER" \
  --services "$OLD_SERVICE" \
  --region "$REGION"

aws ecs delete-service \
  --cluster "$CLUSTER" \
  --service "$OLD_SERVICE" \
  --force \
  --region "$REGION"

printf '\n== Deregistering old task-definition revisions ==\n'
REVISIONS=$(aws ecs list-task-definitions \
  --family-prefix "$OLD_TASK_FAMILY" \
  --status ACTIVE \
  --region "$REGION" \
  --query 'taskDefinitionArns[]' \
  --output text)

for ARN in $REVISIONS; do
  echo "Deregistering $ARN"
  aws ecs deregister-task-definition \
    --task-definition "$ARN" \
    --region "$REGION" >/dev/null
done

printf '\n== Checking whether the old security group is still attached ==\n'
ENIS=$(aws ec2 describe-network-interfaces \
  --region "$REGION" \
  --filters "Name=group-id,Values=$OLD_SECURITY_GROUP" \
  --query 'NetworkInterfaces[].NetworkInterfaceId' \
  --output text)

if [ -n "$ENIS" ]; then
  echo "Security group $OLD_SECURITY_GROUP is still attached to: $ENIS"
  echo "Leave the security group in place until those ENIs disappear."
else
  echo "No ENIs use $OLD_SECURITY_GROUP. Deleting it."
  aws ec2 delete-security-group --group-id "$OLD_SECURITY_GROUP" --region "$REGION"
fi

printf '\n== Log group review ==\n'
echo "The log group $OLD_LOG_GROUP is intentionally NOT deleted automatically."
echo "Delete it only after confirming the Express service uses a different log group."

printf '\n== Final services ==\n'
aws ecs list-services --cluster "$CLUSTER" --region "$REGION"
