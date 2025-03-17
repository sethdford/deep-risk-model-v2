#!/bin/bash
set -e

echo "Building the test model binary..."
# Enable both accelerate and blas-enabled features
cargo build --release --features "accelerate blas-enabled" --no-default-features --bin test_model

echo "Running the test model..."
./target/release/test_model 