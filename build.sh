#!/bin/bash

# Function to install Rust and required targets
install_rust() {
    if ! command -v rustc &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Force reinstall the target
    echo "Adding required Rust targets..."
    rustup target remove aarch64-unknown-linux-gnu || true
    rustup target add aarch64-unknown-linux-gnu
    
    # Set up environment variables for cross-compilation
    export RUSTFLAGS="-C target-feature=+crt-static"
    export OPENSSL_DIR=$(brew --prefix openssl@3)
    export OPENSSL_LIB_DIR=$(brew --prefix openssl@3)/lib
}

# Function to install dependencies
install_dependencies() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS dependencies
        echo "Installing macOS dependencies..."
        brew install openssl@3 openblas
        brew tap filosottile/musl-cross
        brew install musl-cross
        brew tap messense/macos-cross-toolchains
        brew install aarch64-unknown-linux-gnu
    else
        # Linux dependencies
        echo "Installing Linux dependencies..."
        sudo apt-get update
        sudo apt-get install -y build-essential pkg-config libssl-dev libopenblas-dev
        sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    fi
}

# Main build script
echo "Starting build process..."

# Install Rust and dependencies
install_rust
install_dependencies

# Set up build flags
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific flags
    export CC_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-unknown-linux-gnu-gcc
else
    # Linux specific flags
    export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
fi

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
cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

# Copy the binary to the artifacts directory
echo "Copying binary to artifacts directory..."
mkdir -p artifacts
cp target/aarch64-unknown-linux-gnu/release/bootstrap artifacts/
chmod +x artifacts/bootstrap 