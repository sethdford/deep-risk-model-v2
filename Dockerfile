FROM public.ecr.aws/lambda/provided:al2-arm64 as builder

# Install Rust and other dependencies
RUN yum install -y gcc openssl-devel cmake make curl tar gzip

# Install OpenBLAS - ensure we get the development headers and libraries
RUN yum install -y openblas-devel

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

# Build the Lambda function with OpenBLAS support
ENV RUSTFLAGS="-C target-feature=+crt-static -L /usr/lib64 -l static=openblas"
RUN cargo build --release --bin bootstrap --target aarch64-unknown-linux-gnu

# Verify the binary exists and has the right permissions
RUN ls -la /build/target/aarch64-unknown-linux-gnu/release/bootstrap && \
    chmod +x /build/target/aarch64-unknown-linux-gnu/release/bootstrap

# Copy the bootstrap binary to the Lambda runtime directory
FROM public.ecr.aws/lambda/provided:al2-arm64

# Install OpenBLAS runtime
RUN yum install -y openblas

# Copy the bootstrap binary
COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the correct permissions for the bootstrap file
RUN chmod 755 ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the CMD to the handler
CMD ["bootstrap"] 