#!/bin/bash

# Install Rust if not installed
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Install cargo-lambda if not installed
if ! command -v cargo-lambda &> /dev/null; then
    echo "Installing cargo-lambda..."
    cargo install cargo-lambda
fi

# Add the target
rustup target add aarch64-unknown-linux-gnu

# Build the project
cargo lambda build --release --arm64 --output-format binary 