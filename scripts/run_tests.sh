#!/bin/bash
set -e

echo "=== Building with OpenBLAS ==="
cargo build --features openblas

echo "=== Running tests with OpenBLAS ==="
cargo test --features openblas

echo "=== Running examples with OpenBLAS ==="
for example in $(cargo metadata --no-deps --format-version=1 | jq -r '.packages[0].targets[] | select(.kind | contains(["example"])) | .name'); do
    echo "Running example: $example"
    cargo run --example $example --features openblas
done

echo "=== Building with no-blas ==="
cargo build --no-default-features --features no-blas

echo "=== Running basic tests with no-blas ==="
# Only run tests that don't require BLAS operations
cargo test --no-default-features --features no-blas -- --skip factor_analysis::tests::test_factor_selection --skip gpu_model::tests::test_gpu_factor_metrics --skip model::tests::test_factor_generation --skip model::tests::test_factor_metrics --skip model::tests::test_covariance_estimation --skip gpu_model::tests::test_gpu_factor_generation --skip gpu_model::tests::test_gpu_vs_cpu_performance

echo "=== All tests and examples completed ===" 