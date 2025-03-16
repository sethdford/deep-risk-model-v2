use deep_risk_model::{
    prelude::{
        GPUTransformerRiskModel,
        TransformerRiskModel,
        ComputeDevice, GPUConfig,
        TransformerConfig,
        MarketData, RiskModel
    },
    gpu::{is_cuda_available, get_gpu_info},
    error::ModelError,
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
    
    // Try to create GPU model, handle potential BLAS unavailability
    let gpu_result = GPUTransformerRiskModel::new(d_model, n_heads, d_ff, n_layers, gpu_config.clone());
    
    let mut gpu_model = match gpu_result {
        Ok(model) => model,
        Err(e) => {
            println!("Failed to create GPU model: {}", e);
            println!("Using CPU-only implementation instead");
            // Fall back to CPU model
            TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?
        }
    };
    
    // Create CPU model for comparison
    let mut cpu_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
    
    // Compare training performance
    println!("\nTraining models...");
    
    let cpu_start = Instant::now();
    cpu_model.train(&market_data).await?;
    let cpu_train_time = cpu_start.elapsed();
    println!("CPU training time: {:?}", cpu_train_time);
    
    let gpu_start = Instant::now();
    // Handle potential errors during GPU training
    match gpu_model.train(&market_data).await {
        Ok(_) => {
            let gpu_train_time = gpu_start.elapsed();
            println!("GPU training time: {:?}", gpu_train_time);
        },
        Err(e) => {
            println!("GPU training failed: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
    }
    
    // Compare risk factor generation performance
    println!("\nGenerating risk factors...");
    
    let cpu_start = Instant::now();
    let cpu_factors = cpu_model.generate_risk_factors(&market_data).await?;
    let cpu_factor_time = cpu_start.elapsed();
    println!("CPU factor generation time: {:?}", cpu_factor_time);
    println!("CPU factors shape: {:?}", cpu_factors.factors().shape());
    
    let gpu_start = Instant::now();
    // Handle potential errors during GPU risk factor generation
    match gpu_model.generate_risk_factors(&market_data).await {
        Ok(gpu_factors) => {
            let gpu_factor_time = gpu_start.elapsed();
            println!("GPU factor generation time: {:?}", gpu_factor_time);
            println!("GPU factors shape: {:?}", gpu_factors.factors().shape());
            
            // Calculate speedup
            println!("\nPerformance comparison:");
            let factor_speedup = cpu_factor_time.as_secs_f64() / gpu_factor_time.as_secs_f64();
            println!("Factor generation speedup: {:.2}x", factor_speedup);
        },
        Err(e) => {
            println!("GPU factor generation failed: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
    }
    
    // Compare covariance estimation performance
    println!("\nEstimating covariance matrices...");
    
    let cpu_start = Instant::now();
    let cpu_cov = cpu_model.estimate_covariance(&market_data).await?;
    let cpu_cov_time = cpu_start.elapsed();
    println!("CPU covariance estimation time: {:?}", cpu_cov_time);
    
    let gpu_start = Instant::now();
    // Handle potential errors during GPU covariance estimation
    match gpu_model.estimate_covariance(&market_data).await {
        Ok(gpu_cov) => {
            let gpu_cov_time = gpu_start.elapsed();
            println!("GPU covariance estimation time: {:?}", gpu_cov_time);
            
            // Calculate speedup
            let cov_speedup = cpu_cov_time.as_secs_f64() / gpu_cov_time.as_secs_f64();
            println!("Covariance estimation speedup: {:.2}x", cov_speedup);
        },
        Err(e) => {
            println!("GPU covariance estimation failed: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
    }
    
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
    match GPUTransformerRiskModel::new(
        transformer_config.d_model,
        transformer_config.n_heads,
        transformer_config.d_ff,
        transformer_config.n_layers,
        Some(custom_gpu_config.clone()),
    ) {
        Ok(custom_gpu_model) => {
            println!("Custom GPU model created successfully");
            println!("Using GPU: {}", custom_gpu_model.is_using_gpu());
            println!("GPU config: {:?}", custom_gpu_model.gpu_config());
        },
        Err(e) => {
            println!("Failed to create custom GPU model: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
    }
    
    // Example 3: Switching between CPU and GPU
    println!("\n=== Example 3: Switching Between CPU and GPU ===");
    
    // Try to create switchable model, handle potential errors
    match GPUTransformerRiskModel::new(
        d_model,
        n_heads,
        d_ff,
        n_layers,
        Some(GPUConfig {
            device: ComputeDevice::CPU,
            ..GPUConfig::default()
        }),
    ) {
        Ok(mut switchable_model) => {
            println!("Model initially using CPU: {}", !switchable_model.is_using_gpu());
            
            // Run on CPU
            let cpu_start = Instant::now();
            match switchable_model.generate_risk_factors(&market_data).await {
                Ok(_cpu_result) => {
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
                        match switchable_model.generate_risk_factors(&market_data).await {
                            Ok(_gpu_result) => {
                                let gpu_time = gpu_start.elapsed();
                                println!("Time with GPU: {:?}", gpu_time);
                                
                                // Calculate speedup
                                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
                                println!("Speedup after switching to GPU: {:.2}x", speedup);
                            },
                            Err(e) => {
                                println!("GPU operation failed after switching: {}", e);
                                println!("This may be due to missing BLAS libraries or GPU drivers");
                            }
                        }
                    } else {
                        println!("GPU not available for switching");
                    }
                },
                Err(e) => {
                    println!("CPU operation failed: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Failed to create switchable model: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
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
    let mut large_cpu_model = TransformerRiskModel::new(large_n_assets, n_heads, d_ff, n_layers)?;
    
    // Try to create large GPU model, handle potential errors
    match GPUTransformerRiskModel::new(
        large_n_assets,
        n_heads,
        d_ff,
        n_layers,
        Some(GPUConfig {
            device: if gpu_available { ComputeDevice::GPU } else { ComputeDevice::CPU },
            use_mixed_precision: true,
            batch_size: 128,
            use_tensor_cores: true,
        }),
    ) {
        Ok(large_gpu_model) => {
            // Compare performance
            println!("\nGenerating risk factors for large dataset...");
            
            let cpu_start = Instant::now();
            match large_cpu_model.generate_risk_factors(&large_market_data).await {
                Ok(_large_cpu_factors) => {
                    let cpu_time = cpu_start.elapsed();
                    println!("CPU time for large dataset: {:?}", cpu_time);
                    
                    let gpu_start = Instant::now();
                    match large_gpu_model.generate_risk_factors(&large_market_data).await {
                        Ok(_large_gpu_factors) => {
                            let gpu_time = gpu_start.elapsed();
                            println!("GPU time for large dataset: {:?}", gpu_time);
                            
                            if gpu_available {
                                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
                                println!("Speedup for large dataset: {:.2}x", speedup);
                            }
                        },
                        Err(e) => {
                            println!("GPU operation failed for large dataset: {}", e);
                            println!("This may be due to missing BLAS libraries or GPU drivers");
                        }
                    }
                },
                Err(e) => {
                    println!("CPU operation failed for large dataset: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Failed to create large GPU model: {}", e);
            println!("This may be due to missing BLAS libraries or GPU drivers");
        }
    }
    
    println!("\n=== GPU Example Completed Successfully ===");
    
    Ok(())
} 