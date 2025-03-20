.PHONY: build test clean deploy local-invoke build-DeepRiskModelFunction

# Build the project
build:
	cargo build --release --bin bootstrap

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean
	rm -rf ./target
	rm -rf ./.aws-sam

# Build with SAM
sam-build:
	sam build --use-container

# Deploy with SAM
sam-deploy: sam-build
	sam deploy --stack-name deep-risk-model --no-confirm-changeset --no-fail-on-empty-changeset

# Invoke the Lambda function locally
local-invoke: build
	cargo run --release --bin bootstrap < events/event.json

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
	mkdir -p $(ARTIFACTS_DIR)
	cp -r . $(ARTIFACTS_DIR)
	cd $(ARTIFACTS_DIR) && \
	chmod +x build.sh && \
	./build.sh 