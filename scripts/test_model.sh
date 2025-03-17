#!/bin/bash
set -e

echo "Building the test model binary..."
# Enable both accelerate and blas-enabled features
cargo build --release --features "accelerate blas-enabled" --no-default-features --bin run_model_with_test_data

echo "Running the test model..."
./target/release/run_model_with_test_data 