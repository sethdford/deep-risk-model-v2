use deep_risk_model::prelude::{
    MarketData,
    RiskModel,
    DeepRiskModel,
};
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Normal};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Print build configuration
    #[cfg(feature = "no-blas")]
    println!("Running in no-blas mode (pure Rust implementation)");
    
    #[cfg(not(feature = "no-blas"))]
    println!("Running with BLAS support");

    // Initialize model with appropriate size based on BLAS availability
    #[cfg(not(feature = "no-blas"))]
    let mut model = DeepRiskModel::new(100, 10)?;
    
    #[cfg(feature = "no-blas")]
    let mut model = DeepRiskModel::new(20, 5)?;
    
    // Generate synthetic data with appropriate size based on BLAS availability
    #[cfg(not(feature = "no-blas"))]
    let (n_samples, n_assets) = (1000, 100);
    
    #[cfg(feature = "no-blas")]
    let (n_samples, n_assets) = (200, 20);
    
    let features = Array2::<f32>::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
    let returns = Array2::<f32>::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
    let market_data = MarketData::new(returns, features);
    
    // Train model
    model.train(&market_data).await?;
    
    // Generate risk factors
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    println!("Generated {} risk factors", risk_factors.factors().shape()[1]);
    
    // Estimate covariance
    let covariance = model.estimate_covariance(&market_data).await?;
    println!("Estimated {}x{} covariance matrix", covariance.shape()[0], covariance.shape()[1]);
    
    // Validate symmetry
    for i in 0..std::cmp::min(5, covariance.shape()[0]) {
        for j in 0..std::cmp::min(5, covariance.shape()[1]) {
            assert_eq!(covariance[[i, j]], covariance[[j, i]]);
        }
    }
    println!("Covariance matrix is symmetric");

    Ok(())
} 