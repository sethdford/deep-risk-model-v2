#!/bin/bash
set -e

echo "Cleaning previous build artifacts..."
make clean

echo "Building the SAM application..."
sam build

echo "Invoking the Lambda function locally..."
sam local invoke DeepRiskModelFunction --event scripts/test_event.json

# This will invoke the Lambda function directly with the test event
# and display the response in the terminal 