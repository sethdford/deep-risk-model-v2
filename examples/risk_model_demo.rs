use deep_risk_model::prelude::{
    MarketData,
    RiskModel,
    DeepRiskModel,
};
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::Rng;
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
    let mut model = DeepRiskModel::new(
        20,  // n_assets - reduced from 50 to avoid complexity
        3,   // n_factors - reduced from 5 to ensure we can find factors 
        10,  // max_seq_len - reduced from 20
        40,  // d_model - reduced from 100
        2,   // n_heads - reduced from 4
        64,  // d_ff - reduced from 256
        1    // n_layers - reduced from 2
    )?;
    
    #[cfg(feature = "no-blas")]
    let mut model = DeepRiskModel::new(
        10, // n_assets
        3,  // n_factors
        10, // max_seq_len
        20, // d_model
        2,  // n_heads
        32, // d_ff
        1   // n_layers
    )?;
    
    // Generate synthetic data with appropriate size based on BLAS availability
    #[cfg(not(feature = "no-blas"))]
    let (n_samples, n_assets) = (100, 20); // Reduced from 200 samples and 50 assets
    
    #[cfg(feature = "no-blas")]
    let (n_samples, n_assets) = (100, 10);
    
    println!("Generating synthetic data with {} samples and {} assets...", n_samples, n_assets);
    
    // Generate correlated returns data
    let mut rng = rand::thread_rng();
    let common_factor = Array2::<f32>::random((n_samples, 1), Normal::new(0.0, 0.05).unwrap());
    
    // Create returns with common factor to ensure factors can be extracted
    let mut returns = Array2::<f32>::zeros((n_samples, n_assets));
    for i in 0..n_assets {
        let asset_specific = Array2::<f32>::random((n_samples, 1), Normal::new(0.0, 0.02).unwrap());
        let factor_loading = 0.3 + 0.7 * rng.gen::<f32>(); // Random loading between 0.3 and 1.0
        
        for j in 0..n_samples {
            returns[[j, i]] = factor_loading * common_factor[[j, 0]] + asset_specific[[j, 0]];
        }
    }
    
    // Generate features data with twice the number of columns as n_assets
    let features = Array2::<f32>::random((n_samples, n_assets * 2), Normal::new(0.0, 1.0).unwrap());
    
    let market_data = MarketData::new(returns, features);
    
    println!("Training model...");
    // Train model
    model.train(&market_data).await?;
    
    // Set extremely lenient thresholds
    println!("Setting very lenient factor selection thresholds...");
    model.set_factor_selection_thresholds(0.01, 100.0, 0.5)?;
    
    // In no-blas mode, skip operations that require matrix inversion
    #[cfg(not(feature = "no-blas"))]
    {
        println!("Generating risk factors...");
        
        // Generate risk factors with more lenient thresholds
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        println!("Generated {} risk factors from {} returns", risk_factors.factors().shape()[1], risk_factors.factors().shape()[0]);
        
        // Estimate covariance
        println!("Estimating covariance matrix...");
        let covariance = model.estimate_covariance(&market_data).await?;
        println!("Estimated {}x{} covariance matrix", covariance.shape()[0], covariance.shape()[1]);
        
        // Validate symmetry
        let mut symmetric = true;
        for i in 0..covariance.shape()[0] {
            for j in 0..covariance.shape()[1] {
                if (covariance[[i, j]] - covariance[[j, i]]).abs() > 1e-5 {
                    symmetric = false;
                    println!("Asymmetry found at [{},{}]: {} vs {}", i, j, covariance[[i, j]], covariance[[j, i]]);
                }
            }
        }
        println!("Covariance matrix is symmetric: {}", symmetric);
    }
    
    #[cfg(feature = "no-blas")]
    {
        println!("Skipping risk factor generation and covariance estimation in no-blas mode");
        println!("These operations require matrix inversion, which is limited in no-blas mode");
    }

    println!("Demo completed successfully!");
    Ok(())
} 