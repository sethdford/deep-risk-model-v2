#!/bin/bash
set -e

# Configuration
IMAGE_NAME="lambda-rust-function"
PORT=9090
EVENT_FILE="events/test-event.json"

echo "Starting Lambda container on port $PORT"
docker run -d --name lambda-test -p ${PORT}:8080 ${IMAGE_NAME}

# Give the container a moment to start
sleep 2

echo "Invoking Lambda function with direct JSON body..."
OUTPUT=$(curl -s -X POST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d @${EVENT_FILE})

echo "Response:"
echo $OUTPUT | python3 -m json.tool

echo "Stopping container..."
docker stop lambda-test
docker rm -f lambda-test > /dev/null 2>&1 || true

echo "Direct test completed." 