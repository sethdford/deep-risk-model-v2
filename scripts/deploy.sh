#!/bin/bash
set -e

# Configuration
FUNCTION_NAME="deep-risk-model"
REGION="${AWS_REGION:-us-east-1}"  # Default to us-east-1 if not set
IMAGE_NAME="lambda-rust-function"
ECR_REPOSITORY="$FUNCTION_NAME"
MEMORY_SIZE=2048
TIMEOUT=30

echo "Building Docker image..."
docker build -t $IMAGE_NAME -f Dockerfile.lambda .

# Check if the ECR repository exists, create it if not
if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION &> /dev/null; then
    echo "Creating ECR repository $ECR_REPOSITORY..."
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $REGION
fi

# Get the ECR repository URI
ECR_REPO_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION --query 'repositories[0].repositoryUri' --output text)

# Authenticate Docker to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO_URI

# Tag and push the image
echo "Tagging and pushing image to ECR..."
docker tag $IMAGE_NAME:latest $ECR_REPO_URI:latest
docker push $ECR_REPO_URI:latest

# Check if Lambda function exists
FUNCTION_EXISTS=$(aws lambda list-functions --region $REGION --query "Functions[?FunctionName=='$FUNCTION_NAME'].FunctionName" --output text)

if [ -z "$FUNCTION_EXISTS" ]; then
    # Create Lambda function if it doesn't exist
    echo "Creating Lambda function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$ECR_REPO_URI:latest \
        --role $(aws iam get-role --role-name lambda-execution-role --query 'Role.Arn' --output text) \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION
else
    # Update Lambda function if it exists
    echo "Updating Lambda function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $ECR_REPO_URI:latest \
        --region $REGION
fi

echo "Deployment completed successfully!" 