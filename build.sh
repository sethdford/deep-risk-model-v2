#!/bin/bash

# Function to install Rust and required tools
install_rust() {
    if ! command -v rustc &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Install cross
    echo "Installing cross..."
    cargo install cross

    # Add required target
    echo "Adding required Rust target..."
    rustup target add aarch64-unknown-linux-gnu
}

# Function to install dependencies
install_dependencies() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS dependencies
        echo "Installing macOS dependencies..."
        brew install openssl@3 openblas docker
        # Start Docker if not running
        if ! docker info > /dev/null 2>&1; then
            echo "Starting Docker..."
            open -a Docker
            # Wait for Docker to start
            while ! docker info > /dev/null 2>&1; do
                echo "Waiting for Docker to start..."
                sleep 1
            done
        fi
    else
        # Linux dependencies
        echo "Installing Linux dependencies..."
        sudo apt-get update
        sudo apt-get install -y build-essential pkg-config libssl-dev libopenblas-dev docker.io
        # Start Docker if not running
        if ! systemctl is-active --quiet docker; then
            echo "Starting Docker..."
            sudo systemctl start docker
        fi
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

# Build the project using cross
echo "Building project..."
cross build --release --target aarch64-unknown-linux-gnu --bin bootstrap

# Copy the binary to the artifacts directory
echo "Copying binary to artifacts directory..."
mkdir -p artifacts
cp target/aarch64-unknown-linux-gnu/release/bootstrap artifacts/
chmod +x artifacts/bootstrap 