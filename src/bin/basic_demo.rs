use deep_risk_model::{
    error::ModelError,
    model::DeepRiskModel,
    types::{MarketData, RiskModel},
};
use ndarray::{Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), ModelError> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Print build configuration
    #[cfg(feature = "no-blas")]
    println!("Running in no-blas mode (pure Rust implementation)");
    
    #[cfg(not(feature = "no-blas"))]
    println!("Running with BLAS support");

    println!("Starting basic demo...");
    
    // Generate synthetic market data
    let n_assets = 10;
    let n_samples = 1000;
    
    // Generate random returns data
    let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 0.01).unwrap());
    
    // Generate features data (twice the number of columns as assets)
    let features = Array2::random((n_samples, n_assets * 2), Normal::new(0.0, 1.0).unwrap());
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    // Create model parameters
    let n_factors = 5;
    let max_seq_len = 20;
    let d_model = n_assets * 2; // Must match the feature dimension
    let n_heads = 4;
    let d_ff = 128;
    let n_layers = 2;
    
    // Create a deep risk model
    let mut model = DeepRiskModel::new(
        n_assets,
        n_factors,
        max_seq_len,
        d_model,
        n_heads,
        d_ff,
        n_layers,
    )?;
    
    // Use very lenient thresholds for the demo
    // In a real application, these would typically be higher
    model.set_factor_selection_thresholds(0.01, 10.0, 0.01)?;
    
    // Generate risk factors
    println!("Generating {} risk factors...", n_factors);
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    
    // Print the number of factor samples
    println!("Generated {} factor samples (from {} returns samples)", 
             risk_factors.factors().shape()[0], market_data.returns().shape()[0]);
    
    // Estimate covariance matrix
    println!("Estimating {}x{} covariance matrix...", n_assets, n_assets);
    let covariance = model.estimate_covariance(&market_data).await?;
    
    // Verify the covariance matrix is symmetric
    let is_symmetric = is_matrix_symmetric(&covariance);
    println!("Covariance matrix is symmetric: {}", is_symmetric);
    
    println!("Demo completed successfully!");
    Ok(())
}

// Helper function to check if a matrix is symmetric
fn is_matrix_symmetric(matrix: &Array2<f32>) -> bool {
    let (n, m) = matrix.dim();
    if n != m {
        return false;
    }
    
    for i in 0..n {
        for j in (i+1)..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-5 {
                return false;
            }
        }
    }
    
    true
} 