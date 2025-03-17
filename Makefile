.PHONY: build test clean deploy local-invoke

# Build the project
build:
	cargo build --release

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Build with SAM
sam-build:
	sam build --use-container

# Deploy with SAM
sam-deploy: sam-build
	sam deploy --stack-name deep-risk-model --no-confirm-changeset --no-fail-on-empty-changeset

# Invoke the Lambda function locally
local-invoke: build
	cargo run --bin lambda_local < lambda_test_payload.json

# Generate test payload
generate-payload:
	cargo run --bin test_lambda_local

# Start local API
sam-local-api: sam-build
	sam local start-api

# Invoke the Lambda function locally with SAM
sam-local-invoke: sam-build
	sam local invoke DeepRiskModelFunction -e events/test_event_lambda.json 