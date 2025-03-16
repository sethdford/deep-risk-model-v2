use deep_risk_model::{
    prelude::{
        TransformerRiskModel,
        TransformerConfig,
        MarketData, RiskModel
    },
    gpu::{is_cuda_available, get_gpu_info},
};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use std::time::Instant;

/// This example demonstrates how to use GPU-accelerated models in the deep_risk_model crate.
/// It shows:
/// 1. How to create and configure GPU models
/// 2. How to check for GPU availability
/// 3. How to compare performance between CPU and GPU implementations
/// 4. How to use different configuration options
/// 5. How to handle cases when BLAS is not available
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Acceleration Example ===\n");
    
    // Check GPU availability
    println!("Checking GPU availability...");
    let gpu_available = is_cuda_available();
    println!("GPU available: {}", gpu_available);
    println!("GPU info: {}", get_gpu_info());
    
    // Generate synthetic market data
    println!("\nGenerating synthetic market data...");
    let n_samples = 200;
    let n_assets = 64;
    let d_model = 64;
    let n_heads = 8;
    let d_ff = 256;
    let n_layers = 3;
    
    // Check if we're running with BLAS support
    #[cfg(feature = "no-blas")]
    {
        println!("Running without BLAS support. Using fallback implementations.");
        println!("Note: Performance will be significantly slower without BLAS.");
        println!("This example is primarily for demonstration purposes.");
        
        // Create a smaller dataset for no-blas mode to avoid excessive computation
        let small_n_samples = 50;
        let small_n_assets = 16;
        let small_d_model = 8;
        let small_n_heads = 2;
        let small_d_ff = 32;
        let small_n_layers = 1;
        
        let small_features = Array::random((small_n_samples, small_n_assets), Normal::new(0.0, 1.0).unwrap());
        let small_returns = Array::random((small_n_samples, small_n_assets), Normal::new(0.0, 0.1).unwrap());
        let small_market_data = MarketData::new(small_returns, small_features);
        
        // Create a basic transformer model with CPU computation
        let transformer_config = TransformerConfig::new(small_n_assets, small_d_model, small_n_heads, small_d_ff, small_n_layers);
        let mut cpu_model = TransformerRiskModel::with_config(transformer_config.clone())?;
        
        // Train the CPU model
        println!("Training CPU model...");
        let cpu_start = Instant::now();
        cpu_model.train(&small_market_data).await?;
        let cpu_duration = cpu_start.elapsed();
        println!("CPU training completed in {:?}", cpu_duration);
        
        // Skip GPU examples in no-blas mode
        println!("\nSkipping GPU examples in no-blas mode.");
        println!("To run GPU examples, build with BLAS support:");
        println!("cargo run --example gpu_example --features openblas");
        
        return Ok(());
    }
    
    #[cfg(not(feature = "no-blas"))]
    {
        let features = Array::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
        let market_data = MarketData::new(returns.clone(), features.clone());
        
        println!("Created market data with {} samples and {} assets", n_samples, n_assets);
        
        // Create a basic transformer model with CPU computation
        let transformer_config = TransformerConfig::new(n_assets, d_model, n_heads, d_ff, n_layers);
        let mut cpu_model = TransformerRiskModel::with_config(transformer_config.clone())?;
        
        // Train the CPU model
        println!("Training CPU model...");
        let cpu_start = Instant::now();
        cpu_model.train(&market_data).await?;
        let cpu_duration = cpu_start.elapsed();
        println!("CPU training completed in {:?}", cpu_duration);
        
        // Skip GPU-specific examples in this version
        println!("\nSkipping GPU-specific examples in this version.");
        println!("This example is focused on demonstrating CPU fallback functionality.");
        println!("For full GPU support, please use the GPU-enabled version of the library.");
        
        // Run the model again to get baseline performance
        println!("\nRunning model again for baseline performance...");
        let start = Instant::now();
        let factors = cpu_model.generate_risk_factors(&market_data).await?;
        let duration = start.elapsed();
        println!("  Inference time: {:?}", duration);
        println!("  Risk factors shape: {:?}", factors.factors().shape());
        
        // Print summary
        println!("\nSummary:");
        println!("  CPU model training time: {:?}", cpu_duration);
        println!("  CPU model inference time: {:?}", duration);
    }
    
    Ok(())
} 