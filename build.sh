#!/bin/bash

# Function to install Rust and required tools
install_rust() {
    if ! command -v rustc &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
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
        sudo apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross
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

# Build the project
echo "Building project..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Use Docker for cross-compilation on macOS
    echo "Using Docker for cross-compilation..."
    docker run --rm -v $(pwd):/volume -w /volume \
        -e OPENBLAS_DIR=$OPENBLAS_DIR \
        -e OPENBLAS_LIB_DIR=$OPENBLAS_LIB_DIR \
        messense/rust-musl-cross:aarch64-musl \
        cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu
else
    # Direct cross-compilation on Linux
    export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    
    cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu
fi

# Copy the binary to the artifacts directory
echo "Copying binary to artifacts directory..."
mkdir -p artifacts
cp target/aarch64-unknown-linux-gnu/release/bootstrap artifacts/
chmod +x artifacts/bootstrap 