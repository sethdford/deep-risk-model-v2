#!/bin/bash
set -e

echo "Cleaning previous build artifacts..."
make clean

echo "Building the SAM application..."
sam build

echo "Starting the local API Gateway and Lambda..."
sam local start-api

# The API will be available at http://127.0.0.1:3000/risk-factors
# You can test it with:
# python scripts/test_api.py --api-url http://127.0.0.1:3000/risk-factors 