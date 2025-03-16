.PHONY: build-DeepRiskModelFunction build clean

build-DeepRiskModelFunction:
	@echo "Building DeepRiskModelFunction..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Building for macOS with Accelerate..."; \
		cargo build --release --features accelerate --no-default-features --bin bootstrap; \
	else \
		echo "Building for Linux with OpenBLAS..."; \
		cargo build --release --features openblas --no-default-features --bin bootstrap; \
	fi
	@mkdir -p $(ARTIFACTS_DIR)
	@cp target/release/bootstrap $(ARTIFACTS_DIR)
	@echo "Build completed successfully!"

build: build-DeepRiskModelFunction

clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf .aws-sam
	@echo "Clean completed successfully!" 