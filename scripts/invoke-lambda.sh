#!/bin/bash
set -e

# Configuration
LAMBDA_FUNCTION_NAME="DeepRiskModel"
EVENT_FILE="events/test-event.json"
RESPONSE_FILE="events/lambda_response.json"
AWS_REGION="us-east-1"

echo "Invoking Lambda function: ${LAMBDA_FUNCTION_NAME}"
aws lambda invoke \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --payload file://${EVENT_FILE} \
  --cli-binary-format raw-in-base64-out \
  --region ${AWS_REGION} \
  ${RESPONSE_FILE}

if [ $? -eq 0 ]; then
  echo "Lambda function invoked successfully."
  echo "Response saved to ${RESPONSE_FILE}"
  echo "Response content:"
  cat ${RESPONSE_FILE} | python3 -m json.tool
else
  echo "Error invoking Lambda function."
  exit 1
fi 