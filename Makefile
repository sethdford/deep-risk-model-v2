.PHONY: build test clean deploy local-invoke build-DeepRiskModelFunction

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
	cargo run --bin run_model_with_test_data < events/test_event_direct.json

# Generate test payload
generate-payload:
	cargo run --bin generate_lambda_payload

# Start local API
sam-local-api: sam-build
	sam local start-api

# Invoke the Lambda function locally with SAM
sam-local-invoke: sam-build
	sam local invoke DeepRiskModelFunction -e events/test_event_lambda.json

# Build target for SAM
build-DeepRiskModelFunction:
	@echo "Building DeepRiskModelFunction using Docker..."
	mkdir -p $(ARTIFACTS_DIR)
	docker build --platform linux/arm64 -t deep-risk-model:latest .
	docker create --name extract deep-risk-model:latest
	docker cp extract:/var/runtime/bootstrap $(ARTIFACTS_DIR)/bootstrap
	docker rm extract
	chmod +x $(ARTIFACTS_DIR)/bootstrap
	@echo "Build completed successfully!" 