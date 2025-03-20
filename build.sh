#!/bin/bash

# Install Rust if not installed
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Install cargo-lambda if not installed
if ! command -v cargo-lambda &> /dev/null; then
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    if [ "$OS" = "darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            LAMBDA_BINARY="cargo-lambda-aarch64-apple-darwin.tar.gz"
        else
            LAMBDA_BINARY="cargo-lambda-x86_64-apple-darwin.tar.gz"
        fi
    elif [ "$OS" = "linux" ]; then
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            LAMBDA_BINARY="cargo-lambda-aarch64-unknown-linux-gnu.tar.gz"
        else
            LAMBDA_BINARY="cargo-lambda-x86_64-unknown-linux-gnu.tar.gz"
        fi
    else
        echo "Unsupported operating system: $OS"
        exit 1
    fi
    
    echo "Downloading cargo-lambda for $OS $ARCH..."
    curl -LO "https://github.com/cargo-lambda/cargo-lambda/releases/latest/download/$LAMBDA_BINARY"
    tar -xvf "$LAMBDA_BINARY"
    mkdir -p ~/.cargo/bin
    mv cargo-lambda ~/.cargo/bin/
    rm "$LAMBDA_BINARY"
fi

# Add the target
rustup target add aarch64-unknown-linux-gnu

# Build the project
cargo lambda build --release --arm64 --output-format binary 