# Deep Risk Model - SAM Deployment

This project contains the AWS Serverless Application Model (SAM) configuration for deploying the Deep Risk Model as a Lambda function with API Gateway.

## Prerequisites

- [AWS CLI](https://aws.amazon.com/cli/)
- [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- [Rust](https://www.rust-lang.org/tools/install)
- OpenBLAS development libraries (for Linux)

## Local Development

### Building the Application

```bash
# Build with OpenBLAS support (Linux)
cargo build --release --features openblas --no-default-features

# Build with Accelerate support (macOS)
cargo build --release --features accelerate --no-default-features
```

### Testing the Application

```bash
# Run tests with OpenBLAS support (Linux)
cargo test --features openblas --no-default-features

# Run tests with Accelerate support (macOS)
cargo test --features accelerate --no-default-features
```

### Local Testing with SAM

You can test the API locally before deploying to AWS using the SAM CLI. There are two ways to test:

#### 1. Testing with API Gateway (start-api)

This method starts a local API Gateway that forwards requests to your Lambda function:

##### On Linux:

```bash
# Run the local testing script
./scripts/test_local.sh
```

##### On macOS:

```bash
# Run the local testing script for macOS
./scripts/test_local_mac.sh
```

This will:
1. Build the Rust application with the appropriate features
2. Set up the SAM build directory
3. Start a local API Gateway and Lambda environment

Once running, the API will be available at http://127.0.0.1:3000/risk-factors. You can test it with:

```bash
# Install Python dependencies
pip install -r scripts/requirements.txt

# Test the local API
python scripts/test_api.py --api-url http://127.0.0.1:3000/risk-factors
```

To stop the local API, press Ctrl+C in the terminal where it's running.

#### 2. Testing Lambda Directly (invoke)

This method invokes the Lambda function directly with a sample event:

##### On Linux:

```bash
# Run the Lambda invoke script
./scripts/test_invoke.sh
```

##### On macOS:

```bash
# Run the Lambda invoke script for macOS
./scripts/test_invoke_mac.sh
```

This will:
1. Build the Rust application with the appropriate features
2. Set up the SAM build directory
3. Invoke the Lambda function with a sample event from `scripts/test_event.json`
4. Display the response in the terminal

You can modify the sample event in `scripts/test_event.json` to test different inputs.

## Deployment

### Manual Deployment

1. Build the Lambda binary:

```bash
cargo build --release --features openblas --no-default-features --bin bootstrap
mkdir -p .aws-sam/build/DeepRiskModelFunction/
cp target/release/bootstrap .aws-sam/build/DeepRiskModelFunction/
```

2. Validate the SAM template:

```bash
sam validate
```

3. Build the SAM application:

```bash
sam build
```

4. Deploy the SAM application:

```bash
sam deploy --guided
```

### GitHub Actions Deployment

This project includes a GitHub Actions workflow for automated deployment. To use it:

1. Set up the following secrets in your GitHub repository:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key ID
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
   - `AWS_REGION`: The AWS region to deploy to

2. Go to the "Actions" tab in your GitHub repository and select the "Deploy SAM Application" workflow.

3. Click "Run workflow" and select the environment to deploy to (dev, staging, or prod).

## API Usage

Once deployed, the API will be available at the URL provided in the CloudFormation outputs. You can use it as follows:

### Using cURL

```bash
curl -X POST https://your-api-gateway-url/Prod/risk-factors \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "returns": [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]
  }'
```

The API will return a JSON response with the risk factors and covariance matrix:

```json
{
  "factors": [...],
  "covariance": [...]
}
```

### Using the Test Script

We've included a Python script to test the API with generated sample data:

1. Install the required Python packages:

```bash
pip install requests numpy matplotlib
```

2. Run the test script:

```bash
python scripts/test_api.py --api-url https://your-api-gateway-url/Prod/risk-factors
```

This will:
- Generate sample market data
- Call the API with the sample data
- Display a summary of the results
- Create visualizations of the risk factors and returns

Additional options:
```bash
# Specify the number of time samples and assets
python scripts/test_api.py --api-url https://your-api-gateway-url/Prod/risk-factors --samples 200 --assets 10

# Disable plotting
python scripts/test_api.py --api-url https://your-api-gateway-url/Prod/risk-factors --no-plot
```

## Cleanup

To delete the deployed application:

```bash
sam delete --stack-name deep-risk-model-dev
``` 