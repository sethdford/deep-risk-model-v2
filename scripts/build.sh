#!/bin/bash

echo "Starting build process..."

# Build the Docker image
docker build -t deep-risk-model -f Dockerfile.build .

# Create a temporary container to extract the binary
docker create --name temp-container deep-risk-model

# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Copy the binary from the container
docker cp temp-container:/var/runtime/bootstrap artifacts/

# Clean up the temporary container
docker rm temp-container

echo "Build complete. Binary can be found in artifacts/bootstrap" 