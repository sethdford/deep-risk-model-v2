# Lambda Function Usage Guide

This guide explains how to use the Deep Risk Model Lambda function.

## API Endpoint

After deployment, the Lambda function will be accessible via an API Gateway endpoint. The URL will be displayed in the CloudFormation stack outputs:

```
https://{api-id}.execute-api.{region}.amazonaws.com/Prod/risk-model/
```

## Request Format

The Lambda function expects a POST request with a JSON body in the following format:

```json
{
  "features": [
    [feature1_1, feature1_2, ..., feature1_n, feature1_n+1, ..., feature1_2n],
    [feature2_1, feature2_2, ..., feature2_n, feature2_n+1, ..., feature2_2n],
    ...
  ],
  "returns": [
    [return1_1, return1_2, ..., return1_n],
    [return2_1, return2_2, ..., return2_n],
    ...
  ]
}
```

Where:
- `features` is a 2D array of feature values (each row has 2n values)
- `returns` is a 2D array of return values (each row has n values)
- n is the number of assets (default is 16)

## Response Format

The Lambda function will return a JSON response in the following format:

```json
{
  "factors": [[factor_values]],
  "covariance": [[covariance_matrix]]
}
```

Where:
- `factors` is a 2D array of risk factor values
- `covariance` is a 2D array representing the covariance matrix

## Example Usage

### Using curl

```bash
curl -X POST \
  https://{api-id}.execute-api.{region}.amazonaws.com/Prod/risk-model/ \
  -H 'Content-Type: application/json' \
  -d @lambda_test_payload.json
```

### Using Python

```python
import requests
import json

url = "https://{api-id}.execute-api.{region}.amazonaws.com/Prod/risk-model/"

with open("lambda_test_payload.json", "r") as f:
    payload = json.load(f)

response = requests.post(url, json=payload)
result = response.json()

print("Risk factors shape:", len(result["factors"]), "x", len(result["factors"][0]))
print("Covariance matrix shape:", len(result["covariance"]), "x", len(result["covariance"][0]))
```

## Error Handling

The Lambda function will return appropriate HTTP status codes and error messages:

- 200: Success
- 400: Bad request (invalid input format)
- 500: Internal server error

Error responses will include a JSON body with an `error` field describing the issue.

## Performance Considerations

- The Lambda function has a timeout of 30 seconds
- For large datasets, consider batching the requests
- The function is configured with 1024 MB of memory

## Monitoring and Logging

- Lambda function logs are available in CloudWatch Logs
- API Gateway access logs can be enabled for monitoring requests 