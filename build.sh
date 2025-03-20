#!/bin/bash

# Install Rust if not installed
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Install cargo-lambda if not installed
if ! command -v cargo-lambda &> /dev/null; then
    curl -LO https://github.com/cargo-lambda/cargo-lambda/releases/latest/download/cargo-lambda-aarch64-apple-darwin.tar.gz
    tar -xvf cargo-lambda-aarch64-apple-darwin.tar.gz
    mv cargo-lambda ~/.cargo/bin/
fi

# Build the project
cargo lambda build --release --arm64 --output-format binary 