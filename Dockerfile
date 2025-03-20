FROM public.ecr.aws/lambda/provided:al2-arm64 as builder

# Install Rust and other dependencies
RUN yum clean all && \
    yum update -y && \
    yum install -y gcc cmake make curl tar gzip

# Try to install OpenSSL development libraries
RUN yum install -y openssl-devel || echo "Warning: Failed to install openssl-devel"

# Try to install OpenBLAS - we will gracefully handle if not available
RUN yum install -y openblas-devel || echo "Warning: OpenBLAS development packages not available"

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add the target
RUN rustup target add aarch64-unknown-linux-gnu

# Create a new empty project
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY events ./events

# Create necessary directories 
RUN mkdir -p examples tests

# Ensure we have all required source files
COPY examples ./examples
COPY tests ./tests

# Check if OpenBLAS is installed and set RUSTFLAGS accordingly
RUN if [ -f /usr/lib64/libopenblas.a ]; then \
      echo "Building with OpenBLAS support"; \
      export RUSTFLAGS="-C target-feature=+crt-static -L /usr/lib64 -l static=openblas"; \
    else \
      echo "Building without OpenBLAS support"; \
      export RUSTFLAGS="-C target-feature=+crt-static"; \
    fi && \
    cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

# Verify the binary exists and has the right permissions
RUN ls -la /build/target/aarch64-unknown-linux-gnu/release/bootstrap && \
    chmod +x /build/target/aarch64-unknown-linux-gnu/release/bootstrap

# Copy the bootstrap binary to the Lambda runtime directory
FROM public.ecr.aws/lambda/provided:al2-arm64

# Try to install OpenBLAS runtime, but don't fail if not available
RUN yum clean all && \
    yum update -y && \
    yum install -y openblas || echo "Warning: OpenBLAS runtime not available"

# Copy the bootstrap binary
COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the correct permissions for the bootstrap file
RUN chmod 755 ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the CMD to the handler
CMD ["bootstrap"] 