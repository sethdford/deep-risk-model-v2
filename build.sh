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
    # Create a temporary Dockerfile for the build
    cat > Dockerfile.build << 'EOF'
# Stage 1: Build the binary
FROM public.ecr.aws/lambda/provided:al2-arm64 as builder

# Install build dependencies
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y openssl-devel pkg-config gcc

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add target
RUN rustup target add aarch64-unknown-linux-gnu

# Create a new empty project
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the project
RUN cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

# Stage 2: Create the runtime image
FROM public.ecr.aws/lambda/provided:al2-arm64

# Copy the binary
COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set permissions
RUN chmod 755 ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the CMD
CMD ["bootstrap"]
EOF

    # Build using Docker
    echo "Building with Docker..."
    docker build -t lambda-rust-builder -f Dockerfile.build .
    
    # Extract the binary
    echo "Extracting binary..."
    mkdir -p artifacts
    docker create --name temp lambda-rust-builder
    docker cp temp:/var/runtime/bootstrap artifacts/
    docker rm temp
    docker rmi lambda-rust-builder
    
    # Clean up
    rm Dockerfile.build
else
    # Direct cross-compilation on Linux
    export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
    export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    
    cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu
    
    # Copy the binary to the artifacts directory
    echo "Copying binary to artifacts directory..."
    mkdir -p artifacts
    cp target/aarch64-unknown-linux-gnu/release/bootstrap artifacts/
fi

# Ensure the binary is executable
chmod +x artifacts/bootstrap 