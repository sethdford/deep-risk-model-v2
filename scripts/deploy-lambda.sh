#!/bin/bash
set -e

# Configuration
IMAGE_NAME="lambda-rust-function"
ECR_REPOSITORY_NAME="deep-risk-lambda"
LAMBDA_FUNCTION_NAME="DeepRiskModel"
AWS_REGION="us-east-1"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"
IMAGE_URI="${ECR_REPOSITORY}:latest"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME} -f Dockerfile.lambda .

# Check if ECR repository exists, create if not
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} > /dev/null 2>&1 || \
  aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}

echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "Tagging and pushing Docker image to ECR..."
docker tag ${IMAGE_NAME}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}

# Check if Lambda function exists
FUNCTION_EXISTS=$(aws lambda list-functions --region ${AWS_REGION} --query "Functions[?FunctionName==\`${LAMBDA_FUNCTION_NAME}\`].FunctionName" --output text)

if [ -n "$FUNCTION_EXISTS" ]; then
  echo "Updating existing Lambda function..."
  aws lambda update-function-code \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --image-uri ${IMAGE_URI} \
    --region ${AWS_REGION}
else
  echo "Creating new Lambda function..."
  aws lambda create-function \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --package-type Image \
    --code ImageUri=${IMAGE_URI} \
    --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-basic-execution \
    --architectures arm64 \
    --timeout 30 \
    --memory-size 1024 \
    --region ${AWS_REGION}
fi

echo "Deployment completed. Lambda function: ${LAMBDA_FUNCTION_NAME}, Image URI: ${IMAGE_URI}" 