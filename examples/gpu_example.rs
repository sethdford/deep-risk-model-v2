use deep_risk_model::{
    prelude::{
        GPUDeepRiskModel,
        DeepRiskModel,
        ComputeDevice, GPUConfig,
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
    let n_factors = 5;
    
    let features = Array::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
    let returns = Array::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
    let market_data = MarketData::new(returns, features);
    
    println!("Created market data with {} samples and {} assets", n_samples, n_assets);
    
    // Example 1: Basic GPU model with default configuration
    println!("\n=== Example 1: Basic GPU Model ===");
    
    // Create GPU model with default configuration
    let gpu_config = if gpu_available {
        Some(GPUConfig {
            device: ComputeDevice::GPU,
            ..GPUConfig::default()
        })
    } else {
        println!("GPU not available, using CPU fallback");
        Some(GPUConfig::default())
    };
    
    let mut gpu_model = GPUDeepRiskModel::new(n_assets, n_factors, gpu_config.clone())?;
    
    // Create CPU model for comparison
    let mut cpu_model = DeepRiskModel::new(n_assets, n_factors)?;
    
    // Compare training performance
    println!("\nTraining models...");
    
    let cpu_start = Instant::now();
    cpu_model.train(&market_data).await?;
    let cpu_train_time = cpu_start.elapsed();
    println!("CPU training time: {:?}", cpu_train_time);
    
    let gpu_start = Instant::now();
    gpu_model.train(&market_data).await?;
    let gpu_train_time = gpu_start.elapsed();
    println!("GPU training time: {:?}", gpu_train_time);
    
    // Compare risk factor generation performance
    println!("\nGenerating risk factors...");
    
    let cpu_start = Instant::now();
    let cpu_factors = cpu_model.generate_risk_factors(&market_data).await?;
    let cpu_factor_time = cpu_start.elapsed();
    println!("CPU factor generation time: {:?}", cpu_factor_time);
    println!("CPU factors shape: {:?}", cpu_factors.factors().shape());
    
    let gpu_start = Instant::now();
    let gpu_factors = gpu_model.generate_risk_factors(&market_data).await?;
    let gpu_factor_time = gpu_start.elapsed();
    println!("GPU factor generation time: {:?}", gpu_factor_time);
    println!("GPU factors shape: {:?}", gpu_factors.factors().shape());
    
    // Compare covariance estimation performance
    println!("\nEstimating covariance matrices...");
    
    let cpu_start = Instant::now();
    let cpu_cov = cpu_model.estimate_covariance(&market_data).await?;
    let cpu_cov_time = cpu_start.elapsed();
    println!("CPU covariance estimation time: {:?}", cpu_cov_time);
    
    let gpu_start = Instant::now();
    let gpu_cov = gpu_model.estimate_covariance(&market_data).await?;
    let gpu_cov_time = gpu_start.elapsed();
    println!("GPU covariance estimation time: {:?}", gpu_cov_time);
    
    // Calculate speedup
    println!("\nPerformance comparison:");
    let factor_speedup = cpu_factor_time.as_secs_f64() / gpu_factor_time.as_secs_f64();
    let cov_speedup = cpu_cov_time.as_secs_f64() / gpu_cov_time.as_secs_f64();
    println!("Factor generation speedup: {:.2}x", factor_speedup);
    println!("Covariance estimation speedup: {:.2}x", cov_speedup);
    
    // Example 2: Custom GPU configuration
    println!("\n=== Example 2: Custom GPU Configuration ===");
    
    // Create custom transformer configuration
    let transformer_config = TransformerConfig {
        d_model: 128,
        n_heads: 8,
        d_ff: 512,
        n_layers: 3,
        dropout: 0.1,
        max_seq_len: 10,
        num_static_features: 5,
        num_temporal_features: 10,
        hidden_size: 64,
    };
    
    // Create custom GPU configuration
    let custom_gpu_config = GPUConfig {
        device: ComputeDevice::GPU,
        use_mixed_precision: true,
        batch_size: 128,
        use_tensor_cores: true,
    };
    
    // Create GPU model with custom configuration
    println!("Creating GPU model with custom configuration...");
    let custom_gpu_model = GPUDeepRiskModel::with_transformer_config(
        n_assets,
        n_factors,
        transformer_config,
        Some(custom_gpu_config.clone()),
    )?;
    
    println!("Custom GPU model created successfully");
    println!("Using GPU: {}", custom_gpu_model.is_using_gpu());
    println!("GPU config: {:?}", custom_gpu_model.gpu_config());
    
    // Example 3: Switching between CPU and GPU
    println!("\n=== Example 3: Switching Between CPU and GPU ===");
    
    // Create model with CPU configuration
    let mut switchable_model = GPUDeepRiskModel::new(
        n_assets,
        n_factors,
        Some(GPUConfig {
            device: ComputeDevice::CPU,
            ..GPUConfig::default()
        }),
    )?;
    
    println!("Model initially using CPU: {}", !switchable_model.is_using_gpu());
    
    // Run on CPU
    let cpu_start = Instant::now();
    let _cpu_result = switchable_model.generate_risk_factors(&market_data).await?;
    let cpu_time = cpu_start.elapsed();
    println!("Time with CPU: {:?}", cpu_time);
    
    // Switch to GPU if available
    if gpu_available {
        println!("\nSwitching to GPU...");
        switchable_model.set_gpu_config(GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: true,
            batch_size: 64,
            use_tensor_cores: true,
        });
        
        println!("Now using GPU: {}", switchable_model.is_using_gpu());
        
        // Run on GPU
        let gpu_start = Instant::now();
        let _gpu_result = switchable_model.generate_risk_factors(&market_data).await?;
        let gpu_time = gpu_start.elapsed();
        println!("Time with GPU: {:?}", gpu_time);
        
        // Calculate speedup
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("Speedup after switching to GPU: {:.2}x", speedup);
    } else {
        println!("GPU not available for switching");
    }
    
    // Example 4: Large-scale performance test
    println!("\n=== Example 4: Large-scale Performance Test ===");
    
    // Create larger dataset
    println!("Creating larger dataset...");
    let large_n_samples = 500;
    let large_n_assets = 128;
    
    let large_features = Array::random((large_n_samples, large_n_assets), Normal::new(0.0, 1.0).unwrap());
    let large_returns = Array::random((large_n_samples, large_n_assets), Normal::new(0.0, 0.1).unwrap());
    let large_market_data = MarketData::new(large_returns, large_features);
    
    println!("Created market data with {} samples and {} assets", large_n_samples, large_n_assets);
    
    // Create models
    let mut large_cpu_model = DeepRiskModel::new(large_n_assets, n_factors)?;
    
    let large_gpu_model = GPUDeepRiskModel::new(
        large_n_assets,
        n_factors,
        Some(GPUConfig {
            device: if gpu_available { ComputeDevice::GPU } else { ComputeDevice::CPU },
            use_mixed_precision: true,
            batch_size: 128,
            use_tensor_cores: true,
        }),
    )?;
    
    // Compare performance
    println!("\nGenerating risk factors for large dataset...");
    
    let cpu_start = Instant::now();
    let _large_cpu_factors = large_cpu_model.generate_risk_factors(&large_market_data).await?;
    let cpu_time = cpu_start.elapsed();
    println!("CPU time for large dataset: {:?}", cpu_time);
    
    let gpu_start = Instant::now();
    let _large_gpu_factors = large_gpu_model.generate_risk_factors(&large_market_data).await?;
    let gpu_time = gpu_start.elapsed();
    println!("GPU time for large dataset: {:?}", gpu_time);
    
    if gpu_available {
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("Speedup for large dataset: {:.2}x", speedup);
    }
    
    println!("\n=== GPU Example Completed Successfully ===");
    
    Ok(())
} 