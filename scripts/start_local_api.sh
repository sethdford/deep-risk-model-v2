#!/bin/bash
set -e

echo "Cleaning previous build artifacts..."
make clean

echo "Building the SAM application..."
sam build

echo "Starting the local API..."
export DOCKER_HOST=unix:///Users/sethford/.colima/default/docker.sock
sam local start-api

# This will start the API locally at http://127.0.0.1:3000/risk-factors 