#!/bin/bash
set -e

# Configuration
IMAGE_NAME="lambda-rust-function"
PORT=9090
EVENT_FILE="events/event.json"

# Check if the event file exists, if not create a sample one
if [ ! -f "$EVENT_FILE" ]; then
  echo "Creating sample event file at $EVENT_FILE"
  mkdir -p events
  cat > "$EVENT_FILE" << 'EOF'
{
  "features": [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1, 20.1, 21.1, 22.1, 23.1, 24.1, 25.1, 26.1, 27.1, 28.1, 29.1, 30.1, 31.1, 32.1]
  ],
  "returns": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
  ]
}
EOF
fi

echo "Starting Lambda container locally on port ${PORT}..."
docker run -d --rm -p ${PORT}:8080 ${IMAGE_NAME}

# Give the container a moment to start
sleep 2

echo "Invoking Lambda function..."
curl -s -X POST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" -d @${EVENT_FILE} | python3 -m json.tool

echo "Stopping container..."
docker stop $(docker ps -q --filter ancestor=${IMAGE_NAME}) > /dev/null 2>&1 || true

echo "Local test completed!" 