#!/bin/bash

# Set required AWS Lambda environment variables
export AWS_LAMBDA_FUNCTION_NAME="test_function"
export AWS_LAMBDA_FUNCTION_VERSION="$LATEST"
export AWS_LAMBDA_FUNCTION_MEMORY_SIZE=1024
export AWS_LAMBDA_LOG_GROUP_NAME="/aws/lambda/test_function"
export AWS_LAMBDA_LOG_STREAM_NAME="2023/01/01/[$LATEST]abcdef123456"
export AWS_LAMBDA_RUNTIME_API="localhost:8080"
export _HANDLER="lambda.handler"

# Run the Lambda function with our test payload
cargo run --bin lambda < lambda_test_payload.json 