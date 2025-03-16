#!/usr/bin/env python3
import argparse
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def generate_sample_data(n_samples=100, n_assets=5):
    """Generate sample market data for testing."""
    # Generate random returns with some correlation structure
    np.random.seed(42)
    cov_matrix = np.random.rand(n_assets, n_assets)
    cov_matrix = np.dot(cov_matrix, cov_matrix.transpose())
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets), 
        cov=cov_matrix / 10, 
        size=n_samples
    )
    
    # Generate some features (e.g., technical indicators)
    features = np.zeros((n_samples, n_assets))
    for i in range(n_assets):
        # Simple moving average
        features[:, i] = np.convolve(returns[:, i], np.ones(10)/10, mode='same')
    
    # Add some noise to features
    features += np.random.normal(0, 0.01, size=features.shape)
    
    return returns.tolist(), features.tolist()

def call_risk_model_api(api_url, returns, features):
    """Call the risk model API with the provided data."""
    payload = {
        "returns": returns,
        "features": features
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def plot_results(risk_factors, returns):
    """Plot the risk factors and returns."""
    factors = np.array(risk_factors["factors"])
    covariance = np.array(risk_factors["covariance"])
    returns_array = np.array(returns)
    
    # Plot risk factors
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(factors)
    plt.title("Risk Factors")
    plt.xlabel("Time")
    plt.ylabel("Factor Value")
    
    # Plot covariance matrix heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(covariance, cmap='viridis')
    plt.colorbar()
    plt.title("Covariance Matrix")
    
    # Plot returns
    plt.subplot(2, 2, 3)
    plt.plot(returns_array)
    plt.title("Asset Returns")
    plt.xlabel("Time")
    plt.ylabel("Return")
    
    # Plot cumulative returns
    plt.subplot(2, 2, 4)
    plt.plot(np.cumprod(1 + returns_array, axis=0) - 1)
    plt.title("Cumulative Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"risk_analysis_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test the Deep Risk Model API")
    parser.add_argument("--api-url", required=True, help="URL of the API endpoint")
    parser.add_argument("--samples", type=int, default=100, help="Number of time samples")
    parser.add_argument("--assets", type=int, default=5, help="Number of assets")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    
    args = parser.parse_args()
    
    print(f"Generating sample data with {args.samples} samples and {args.assets} assets...")
    returns, features = generate_sample_data(args.samples, args.assets)
    
    print(f"Calling API at {args.api_url}...")
    risk_factors = call_risk_model_api(args.api_url, returns, features)
    
    if risk_factors:
        print("API call successful!")
        print("\nRisk Factors Summary:")
        factors = np.array(risk_factors["factors"])
        covariance = np.array(risk_factors["covariance"])
        
        print(f"Number of factors: {factors.shape[1]}")
        print(f"Factor time series length: {factors.shape[0]}")
        print(f"Covariance matrix shape: {covariance.shape}")
        
        # Print eigenvalues of covariance matrix
        eigenvalues = np.linalg.eigvals(covariance)
        print(f"Covariance eigenvalues: {eigenvalues}")
        
        if not args.no_plot:
            plot_results(risk_factors, returns)
    else:
        print("API call failed.")

if __name__ == "__main__":
    main() 