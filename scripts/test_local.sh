#!/bin/bash
set -e

echo "Cleaning previous build artifacts..."
make clean

echo "Building Lambda function..."
ARTIFACTS_DIR=./target/lambda make build

echo "Invoking Lambda function locally..."
cargo lambda invoke --data-file events/test_event.json

echo "Local test completed successfully!"

# The API will be available at http://127.0.0.1:3000/risk-factors
# You can test it with:
# python scripts/test_api.py --api-url http://127.0.0.1:3000/risk-factors 