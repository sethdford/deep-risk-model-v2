# Deep Risk Model Deployment Guide

This guide provides instructions for deploying the Deep Risk Model Lambda function to AWS using the Serverless Application Model (SAM).

## Prerequisites

- [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html) installed
- [Docker](https://www.docker.com/get-started) installed and running
- [Rust](https://www.rust-lang.org/tools/install) installed
- [cargo-lambda](https://github.com/cargo-lambda/cargo-lambda) installed (optional, for local testing)

## Project Structure

The Deep Risk Model project is structured as follows:

- `src/`: Contains the Rust source code
  - `bin/`: Contains binary executables
    - `bootstrap.rs`: The Lambda function entry point
    - `run_model_with_test_data.rs`: A utility for testing the model locally
  - `lib.rs`: The library code
  - Other modules: Implementation of the risk model
- `events/`: Contains test event files
  - `test_event_lambda.json`: Test event for the Lambda function
  - `test_event_direct.json`: Test event for direct model testing
- `template.yaml`: SAM template defining the Lambda function and API Gateway
- `Makefile`: Contains build targets for SAM

## Local Testing

Before deploying to AWS, you can test the model locally:

1. Test the model directly:
   ```bash
   cargo run --bin run_model_with_test_data < events/test_event_direct.json
   ```

2. Test the Lambda function locally with SAM (requires Docker running):
   ```bash
   sam build
   sam local invoke -e events/test_event_lambda.json
   ```

3. Start a local API endpoint:
   ```bash
   sam local start-api
   ```
   Then you can send requests to `http://localhost:3000/risk-model`

## Deployment to AWS

Follow these steps to deploy the Deep Risk Model Lambda function to AWS:

1. Build the Lambda function:
   ```bash
   sam build
   ```

2. Deploy to AWS:
   ```bash
   sam deploy --guided
   ```
   This will start an interactive deployment process where you can specify:
   - Stack name (e.g., `deep-risk-model`)
   - AWS Region
   - Confirmation of IAM role creation
   - Allow SAM CLI to create IAM roles

3. After deployment, SAM will output the API Gateway endpoint URL that you can use to invoke the Lambda function.

## Lambda Function Configuration

The Lambda function is configured in the `template.yaml` file:

- Runtime: `provided.al2` (custom runtime for Rust)
- Architecture: `arm64` (for better performance and lower cost)
- Memory: 1024 MB (can be adjusted based on performance needs)
- Timeout: 30 seconds
- API Gateway: Configured to accept POST requests at `/risk-model`

## Input and Output Format

### Input

The Lambda function expects a JSON payload with the following structure:

```json
{
  "features": [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  ],
  "returns": [
    [0.01, 0.02, 0.03],
    [0.04, 0.05, 0.06]
  ]
}
```

Where:
- `features`: A 2D array of feature values (each row represents a time step, each column a feature)
- `returns`: A 2D array of return values (each row represents a time step, each column an asset)

### Output

The Lambda function returns a JSON response with the following structure:

```json
{
  "factors": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
  "covariance": [[1.0, 0.5], [0.5, 1.0]]
}
```

Where:
- `factors`: A 2D array representing the risk factors
- `covariance`: A 2D array representing the covariance matrix

## Monitoring and Troubleshooting

- CloudWatch Logs: All Lambda function logs are sent to CloudWatch Logs
- CloudWatch Metrics: Lambda metrics are available in CloudWatch
- X-Ray: You can enable X-Ray tracing for the Lambda function

## Cleanup

To remove the deployed resources:

```bash
sam delete
```

This will delete the CloudFormation stack and all associated resources.

## Advanced Configuration

### Scaling

The Lambda function will automatically scale based on the number of incoming requests. You can configure the following:

- Provisioned Concurrency: For predictable performance
- Reserved Concurrency: To limit the number of concurrent executions

### Custom Domain

To use a custom domain for the API Gateway:

1. Create a custom domain in API Gateway
2. Create a DNS record pointing to the API Gateway endpoint
3. Map the custom domain to the API Gateway stage

## Security Considerations

- IAM Roles: The Lambda function uses the least privilege principle
- API Gateway: Consider adding authentication (API keys, IAM, Cognito, etc.)
- VPC: Consider deploying the Lambda function in a VPC for additional security

## Performance Optimization

- Memory: Increase the Lambda function memory for better CPU performance
- Cold Starts: Use Provisioned Concurrency to eliminate cold starts
- Payload Size: Minimize the size of input and output payloads 