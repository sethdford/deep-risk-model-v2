#!/bin/bash
set -e

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

# Create a temporary Dockerfile
cat > Dockerfile.build << 'EOF'
FROM amazonlinux:2 as builder

# Install build dependencies
RUN yum update -y && \
    yum install -y gcc openssl-devel pkg-config make git && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc-c++ && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source $HOME/.cargo/env && \
    rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation tools
RUN amazon-linux-extras enable epel && \
    yum clean metadata && \
    yum -y install epel-release && \
    yum install -y gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu

# Set up cross-compilation environment
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++ \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar \
    RUSTFLAGS="-C target-feature=+crt-static"

# Copy the project files
WORKDIR /usr/src/app
COPY . .

# Build the project
RUN source $HOME/.cargo/env && \
    cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

FROM public.ecr.aws/lambda/provided:al2 as runtime

# Copy the binary from builder
COPY --from=builder /usr/src/app/target/aarch64-unknown-linux-gnu/release/bootstrap /var/runtime/bootstrap

# Set permissions
RUN chmod +x /var/runtime/bootstrap

# Create artifacts directory and copy binary
RUN mkdir -p /artifacts && \
    cp /var/runtime/bootstrap /artifacts/

EOF

# Build the Docker image
docker build -t lambda-rust-builder -f Dockerfile.build .

# Extract the binary
docker create --name temp lambda-rust-builder
docker cp temp:/artifacts/bootstrap artifacts/
docker rm temp

# Clean up
docker rmi lambda-rust-builder

# Set executable permissions
chmod +x artifacts/bootstrap 