# CI/CD Setup Guide

This guide explains how to set up the CI/CD pipeline for deploying the Deep Risk Model Lambda function to AWS.

## Prerequisites

1. GitHub repository with the Deep Risk Model code
2. AWS account with appropriate permissions
3. IAM user with programmatic access for GitHub Actions

## Setting Up AWS Credentials

1. Create an IAM user with programmatic access:
   - Go to the AWS Management Console
   - Navigate to IAM
   - Create a new user with programmatic access
   - Attach the following policies:
     - `AWSLambdaFullAccess`
     - `AmazonAPIGatewayAdministrator`
     - `AWSCloudFormationFullAccess`
     - `IAMFullAccess` (or a more restricted policy if preferred)

2. Note the Access Key ID and Secret Access Key

## Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Add the following secrets:
   - `AWS_ACCESS_KEY_ID`: Your IAM user's access key ID
   - `AWS_SECRET_ACCESS_KEY`: Your IAM user's secret access key
   - `AWS_REGION`: The AWS region to deploy to (e.g., `us-east-1`)

## GitHub Actions Workflow

The workflow is defined in `.github/workflows/deploy.yml` and will:

1. Set up Rust with cross-compilation support for ARM64
2. Install the SAM CLI
3. Configure AWS credentials
4. Build the Lambda function with SAM
5. Deploy the Lambda function to AWS

## Local Testing Before Deployment

Before pushing changes to trigger the CI/CD pipeline, you can test locally:

1. Start Docker
2. Build with SAM:
   ```bash
   make sam-build
   ```
3. Test locally:
   ```bash
   make sam-local-invoke
   ```

## Monitoring Deployments

1. Go to your GitHub repository
2. Navigate to Actions
3. Click on the latest workflow run to see the deployment status

## Troubleshooting

### Common Issues

1. **Build Failures**: Check that your Rust code compiles correctly
2. **Deployment Failures**: Check the CloudFormation stack in the AWS Console
3. **Permission Issues**: Verify that the IAM user has the necessary permissions

### Logs

- GitHub Actions logs: Available in the Actions tab of your repository
- Lambda function logs: Available in AWS CloudWatch Logs

## Cleanup

To delete the deployed resources:

```bash
aws cloudformation delete-stack --stack-name deep-risk-model
``` 