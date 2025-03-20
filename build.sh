#!/bin/bash

# Function to install Rust and required tools
install_rust() {
    if ! command -v rustc &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Install cargo-lambda
    echo "Installing cargo-lambda..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew tap cargo-lambda/cargo-lambda
        brew install cargo-lambda
    else
        curl -sSf https://get.cargo-lambda.dev/latest/cargo-lambda-installer.sh | bash
    fi

    # Add required target
    echo "Adding required Rust target..."
    rustup target add aarch64-unknown-linux-gnu
}

# Function to install dependencies
install_dependencies() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS dependencies
        echo "Installing macOS dependencies..."
        brew install openssl@3 openblas
    else
        # Linux dependencies
        echo "Installing Linux dependencies..."
        sudo apt-get update
        sudo apt-get install -y build-essential pkg-config libssl-dev libopenblas-dev
    fi
}

# Main build script
echo "Starting build process..."

# Install Rust and dependencies
install_rust
install_dependencies

# Check for OpenBLAS support
if [[ "$OSTYPE" == "darwin"* ]]; then
    if brew list openblas &>/dev/null; then
        echo "Building with OpenBLAS support"
        export OPENBLAS_DIR=$(brew --prefix openblas)
        export OPENBLAS_LIB_DIR=$(brew --prefix openblas)/lib
    else
        echo "Building without OpenBLAS support"
    fi
else
    if dpkg -l | grep -q libopenblas-dev; then
        echo "Building with OpenBLAS support"
        export OPENBLAS_DIR=/usr
        export OPENBLAS_LIB_DIR=/usr/lib
    else
        echo "Building without OpenBLAS support"
    fi
fi

# Build the project using cargo-lambda
echo "Building project..."
cargo lambda build --release --arm64 --output-format binary

# Copy the binary to the artifacts directory
echo "Copying binary to artifacts directory..."
mkdir -p artifacts
cp target/lambda/deep-risk-model/bootstrap artifacts/
chmod +x artifacts/bootstrap 