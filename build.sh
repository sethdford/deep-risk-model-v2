#!/bin/bash

# Function to install Rust
install_rust() {
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
    rustup target add aarch64-unknown-linux-gnu
}

# Function to install dependencies
install_dependencies() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS dependencies
        brew install openssl openblas
        brew install FiloSottile/musl-cross/musl-cross
        brew install SergioBenitez/osxct/x86_64-unknown-linux-gnu
        brew install SergioBenitez/osxct/aarch64-unknown-linux-gnu
    else
        # Linux dependencies
        if command -v apt-get &> /dev/null; then
            apt-get update
            apt-get install -y build-essential pkg-config libssl-dev libopenblas-dev
            apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
        elif command -v yum &> /dev/null; then
            yum clean all
            yum update -y
            yum install -y gcc cmake make openssl-devel openblas-devel
            yum install -y gcc-aarch64-linux-gnu gcc-c++-aarch64-linux-gnu
        fi
    fi
}

# Install Rust and dependencies
install_rust
install_dependencies

# Set up build flags
if [[ -f /usr/lib64/libopenblas.a ]]; then
    echo "Building with OpenBLAS support"
    export RUSTFLAGS="-C target-feature=+crt-static -L /usr/lib64 -l static=openblas"
else
    echo "Building without OpenBLAS support"
    export RUSTFLAGS="-C target-feature=+crt-static"
fi

# Set up cross-compilation environment
if [[ "$OSTYPE" == "darwin"* ]]; then
    export CC_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-unknown-linux-gnu-gcc
else
    export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
fi

# Build the project
cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

# Copy the binary to the artifacts directory
mkdir -p ${ARTIFACTS_DIR:-target}
cp target/aarch64-unknown-linux-gnu/release/bootstrap ${ARTIFACTS_DIR:-target}/bootstrap
chmod +x ${ARTIFACTS_DIR:-target}/bootstrap 