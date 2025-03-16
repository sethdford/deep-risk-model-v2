use deep_risk_model::prelude::{TransformerRiskModel, RiskModel, MarketData};
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;
use deep_risk_model::transformer::TransformerConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Deep Risk Model - Quantization Example");
    println!("======================================\n");
    
    // Check if we're running with BLAS support
    #[cfg(feature = "no-blas")]
    {
        println!("Running without BLAS support. Using fallback implementations.");
        println!("Note: Performance will be significantly slower without BLAS.");
        println!("This example is primarily for demonstration purposes.");
        
        // Use smaller dimensions for no-blas mode
        let d_model = 16;
        let n_heads = 2;
        let d_ff = 32;
        let n_layers = 1;
        let n_samples = 30;
        let n_assets = d_model;
        
        println!("Creating transformer risk model with reduced dimensions for no-blas mode:");
        println!("  d_model = {}", d_model);
        println!("  n_heads = {}", n_heads);
        println!("  d_ff = {}", d_ff);
        println!("  n_layers = {}", n_layers);
        
        // Create a custom config with smaller max_seq_len
        let mut config = TransformerConfig::new(n_assets, d_model, n_heads, d_ff, n_layers);
        config.max_seq_len = 10; // Use a smaller max_seq_len for no-blas mode
        
        let model = TransformerRiskModel::with_config(config)?;
        
        println!("\nGenerating synthetic market data (reduced size):");
        println!("  n_samples = {}", n_samples);
        println!("  n_assets = {}", n_assets);
        
        let features = Array::random((n_samples, n_assets), Uniform::new(-1.0, 1.0));
        let returns = Array::random((n_samples, n_assets), Uniform::new(-0.05, 0.05));
        let market_data = MarketData::new(returns, features);
        
        // Measure full precision model performance
        println!("\nRunning full precision model...");
        
        let start = Instant::now();
        let full_precision_factors = model.generate_risk_factors(&market_data).await?;
        let full_precision_duration = start.elapsed();
        
        println!("  Inference time: {:?}", full_precision_duration);
        
        // Run the model again to get baseline performance
        println!("\nRunning model again for baseline performance...");
        let start = Instant::now();
        let baseline_factors = model.generate_risk_factors(&market_data).await?;
        let baseline_duration = start.elapsed();
        
        println!("  Inference time: {:?}", baseline_duration);
        
        // Calculate error between runs (should be minimal)
        let baseline_factors_data = baseline_factors.factors();
        let fp_factors = full_precision_factors.factors();
        
        let mut baseline_mse = 0.0;
        for i in 0..fp_factors.shape()[0] {
            for j in 0..fp_factors.shape()[1] {
                let diff = fp_factors[[i, j]] - baseline_factors_data[[i, j]];
                baseline_mse += diff * diff;
            }
        }
        baseline_mse /= (fp_factors.shape()[0] * fp_factors.shape()[1]) as f32;
        println!("  Baseline MSE between runs: {:.6}", baseline_mse);
        
        // Print summary
        println!("\nSummary:");
        println!("  Baseline accuracy (MSE): {:.6}", baseline_mse);
        println!("  Running without BLAS support (pure Rust implementation)");
        println!("\nTo run with BLAS support for better performance:");
        println!("  cargo run --example quantization_example --features openblas");
        
        return Ok(());
    }
    
    #[cfg(not(feature = "no-blas"))]
    {
        // Create a transformer risk model
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        println!("Creating transformer risk model with:");
        println!("  d_model = {}", d_model);
        println!("  n_heads = {}", n_heads);
        println!("  d_ff = {}", d_ff);
        println!("  n_layers = {}", n_layers);
        
        // Create a custom config with appropriate max_seq_len
        let mut config = TransformerConfig::new(d_model, d_model, n_heads, d_ff, n_layers);
        config.max_seq_len = 50; // Use a smaller max_seq_len than the default 100
        
        let model = TransformerRiskModel::with_config(config)?;
        
        // Generate synthetic market data
        let n_samples = 100;
        let n_assets = d_model;
        
        println!("\nGenerating synthetic market data:");
        println!("  n_samples = {}", n_samples);
        println!("  n_assets = {}", n_assets);
        
        let features = Array::random((n_samples, n_assets), Uniform::new(-1.0, 1.0));
        let returns = Array::random((n_samples, n_assets), Uniform::new(-0.05, 0.05));
        let market_data = MarketData::new(returns, features);
        
        // Measure full precision model performance
        println!("\nRunning full precision model...");
        
        let start = Instant::now();
        let full_precision_factors = model.generate_risk_factors(&market_data).await?;
        let full_precision_duration = start.elapsed();
        
        println!("  Inference time: {:?}", full_precision_duration);
        
        // Run the model again to get baseline performance
        println!("\nRunning model again for baseline performance...");
        let start = Instant::now();
        let baseline_factors = model.generate_risk_factors(&market_data).await?;
        let baseline_duration = start.elapsed();
        
        println!("  Inference time: {:?}", baseline_duration);
        
        // Calculate error between runs (should be minimal)
        let baseline_factors_data = baseline_factors.factors();
        let fp_factors = full_precision_factors.factors();
        
        let mut baseline_mse = 0.0;
        for i in 0..fp_factors.shape()[0] {
            for j in 0..fp_factors.shape()[1] {
                let diff = fp_factors[[i, j]] - baseline_factors_data[[i, j]];
                baseline_mse += diff * diff;
            }
        }
        baseline_mse /= (fp_factors.shape()[0] * fp_factors.shape()[1]) as f32;
        println!("  Baseline MSE between runs: {:.6}", baseline_mse);
        
        // Calculate speedup
        let speedup = full_precision_duration.as_secs_f32() / baseline_duration.as_secs_f32();
        println!("  Baseline speedup: {:.2}x", speedup);
        
        // Compare covariance matrices
        let fp_cov = full_precision_factors.covariance();
        let q_cov = baseline_factors.covariance();
        
        let mut cov_mse = 0.0;
        for i in 0..fp_cov.shape()[0] {
            for j in 0..fp_cov.shape()[1] {
                let diff = fp_cov[[i, j]] - q_cov[[i, j]];
                cov_mse += diff * diff;
            }
        }
        
        cov_mse /= (fp_cov.shape()[0] * fp_cov.shape()[1]) as f32;
        println!("  Baseline covariance MSE: {:.6}", cov_mse);
        
        // Print summary
        println!("\nSummary:");
        println!("  Baseline accuracy (MSE): {:.6}", baseline_mse);
        
        // Print BLAS configuration
        #[cfg(feature = "openblas")]
        println!("  Running with OpenBLAS support");
        
        #[cfg(feature = "netlib")]
        println!("  Running with Netlib support");
        
        #[cfg(feature = "intel-mkl")]
        println!("  Running with Intel MKL support");
        
        #[cfg(feature = "accelerate")]
        println!("  Running with Accelerate support");
    }
    
    Ok(())
} 