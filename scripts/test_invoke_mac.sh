#!/bin/bash
set -e

echo "Building the Rust application with Accelerate support for macOS..."
cargo build --release --features accelerate --no-default-features --bin bootstrap

echo "Creating SAM build directory..."
mkdir -p .aws-sam/build/DeepRiskModelFunction/
cp target/release/bootstrap .aws-sam/build/DeepRiskModelFunction/

echo "Building the SAM application..."
sam build

echo "Invoking the Lambda function locally..."
sam local invoke DeepRiskModelFunction --event scripts/test_event.json

# This will invoke the Lambda function directly with the test event
# and display the response in the terminal 