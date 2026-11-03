#!/usr/bin/env bash
# Docker image -> ECR -> App Runner.  Prereq: aws cli configured, docker running.
# Usage: REGION=ap-south-1 bash deploy/aws_deploy.sh
set -euo pipefail
REGION=${REGION:-ap-south-1}
REPO=${REPO:-claims-copilot}
SERVICE=${SERVICE:-claims-copilot}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest"

# 1) ECR repo + push
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker build --platform linux/amd64 -t "$REPO" .
docker tag "$REPO:latest" "$IMAGE"
docker push "$IMAGE"

# 2) OpenAI key -> Secrets Manager (one-time)
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id claims-copilot/openai --region "$REGION" --query ARN --output text 2>/dev/null || true)
if [ -z "$SECRET_ARN" ]; then
  read -rsp "OPENAI_API_KEY: " KEY; echo
  SECRET_ARN=$(aws secretsmanager create-secret --name claims-copilot/openai --secret-string "$KEY" --region "$REGION" --query ARN --output text)
fi

# 3) IAM roles for App Runner (ECR pull + secret read) — create once in the Console:
#    - AppRunnerECRAccessRole  (policy: AWSAppRunnerServicePolicyForECRAccess)
#    - AppRunnerInstanceRole   (inline: secretsmanager:GetSecretValue on $SECRET_ARN)
ACCESS_ROLE="arn:aws:iam::$ACCOUNT:role/AppRunnerECRAccessRole"
INSTANCE_ROLE="arn:aws:iam::$ACCOUNT:role/AppRunnerInstanceRole"

# 4) App Runner service create / update
ARN=$(aws apprunner list-services --region "$REGION" --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn" --output text)
SRC=$(cat <<JSON
{"ImageRepository":{"ImageIdentifier":"$IMAGE","ImageRepositoryType":"ECR",
  "ImageConfiguration":{"Port":"8000","RuntimeEnvironmentSecrets":{"OPENAI_API_KEY":"$SECRET_ARN"}}},
 "AuthenticationConfiguration":{"AccessRoleArn":"$ACCESS_ROLE"},"AutoDeploymentsEnabled":true}
JSON
)
if [ -z "$ARN" ]; then
  aws apprunner create-service --region "$REGION" --service-name "$SERVICE" --source-configuration "$SRC" \
    --instance-configuration "Cpu=1024,Memory=2048,InstanceRoleArn=$INSTANCE_ROLE" \
    --health-check-configuration "Protocol=HTTP,Path=/health"
else
  aws apprunner start-deployment --service-arn "$ARN" --region "$REGION"
fi
echo "Done. URL: aws apprunner list-services --region $REGION"
