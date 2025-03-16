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
    fallback,
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
    
    // Check if we're running with BLAS support
    #[cfg(feature = "no-blas")]
    {
        println!("Running without BLAS support. Using fallback implementations.");
        println!("Note: Performance will be significantly slower without BLAS.");
        println!("This example is primarily for demonstration purposes.");
        
        // Create a smaller dataset for no-blas mode to avoid excessive computation
        let small_n_samples = 50;
        let small_n_assets = 16;
        let small_features = Array::random((small_n_samples, small_n_assets), Normal::new(0.0, 1.0).unwrap());
        let small_returns = Array::random((small_n_samples, small_n_assets), Normal::new(0.0, 0.1).unwrap());
        let small_market_data = MarketData::new(small_returns, small_features);
        
        // Create a basic transformer model with CPU computation
        let transformer_config = TransformerConfig::new(small_n_assets, d_model, n_heads, d_ff, n_layers);
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
        // Create a basic transformer model with CPU computation
        let transformer_config = TransformerConfig::new(n_assets, d_model, n_heads, d_ff, n_layers);
        let mut cpu_model = TransformerRiskModel::with_config(transformer_config.clone())?;
        
        // Train the CPU model
        println!("Training CPU model...");
        let cpu_start = Instant::now();
        cpu_model.train(&market_data).await?;
        let cpu_duration = cpu_start.elapsed();
        println!("CPU training completed in {:?}", cpu_duration);
        
        // Create a GPU configuration
        let gpu_config = GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: false,
            batch_size: 64,
            use_tensor_cores: true,
        };
        
        // Create a GPU model with the same transformer configuration
        let mut gpu_model = GPUTransformerRiskModel::with_config(transformer_config, gpu_config)?;
        
        // Train the GPU model
        println!("Training GPU model...");
        let gpu_start = Instant::now();
        gpu_model.train(&market_data).await?;
        let gpu_duration = gpu_start.elapsed();
        println!("GPU training completed in {:?}", gpu_duration);
        
        // Compare performance
        println!("\nPerformance comparison:");
        println!("CPU training time: {:?}", cpu_duration);
        println!("GPU training time: {:?}", gpu_duration);
        if gpu_duration < cpu_duration {
            let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
            println!("GPU speedup: {:.2}x", speedup);
        } else {
            println!("No GPU speedup observed. This could be due to:");
            println!("- Small model size (GPU overhead exceeds benefits)");
            println!("- GPU not properly utilized");
            println!("- System configuration issues");
        }
        
        // Example 2: Mixed precision
        println!("\n=== Example 2: Mixed Precision ===");
        
        // Create a GPU configuration with mixed precision
        let mixed_precision_config = GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: true,
            batch_size: 64,
            use_tensor_cores: true,
        };
        
        // Create a GPU model with mixed precision
        let mut mixed_precision_model = GPUTransformerRiskModel::with_config(
            transformer_config.clone(),
            mixed_precision_config
        )?;
        
        // Train the mixed precision model
        println!("Training GPU model with mixed precision...");
        let mixed_start = Instant::now();
        mixed_precision_model.train(&market_data).await?;
        let mixed_duration = mixed_start.elapsed();
        println!("Mixed precision training completed in {:?}", mixed_duration);
        
        // Compare with full precision
        println!("\nMixed precision vs. full precision:");
        println!("Full precision GPU time: {:?}", gpu_duration);
        println!("Mixed precision GPU time: {:?}", mixed_duration);
        if mixed_duration < gpu_duration {
            let speedup = gpu_duration.as_secs_f64() / mixed_duration.as_secs_f64();
            println!("Mixed precision speedup: {:.2}x", speedup);
        } else {
            println!("No mixed precision speedup observed.");
        }
        
        // Example 3: Batch size optimization
        println!("\n=== Example 3: Batch Size Optimization ===");
        
        // Try different batch sizes
        let batch_sizes = [16, 32, 64, 128, 256];
        let mut best_batch_size = 0;
        let mut best_time = std::time::Duration::from_secs(u64::MAX);
        
        for &batch_size in &batch_sizes {
            // Create a GPU configuration with the current batch size
            let batch_config = GPUConfig {
                device: ComputeDevice::GPU,
                use_mixed_precision: false,
                batch_size,
                use_tensor_cores: true,
            };
            
            // Create a GPU model with the current batch size
            let mut batch_model = GPUTransformerRiskModel::with_config(
                transformer_config.clone(),
                batch_config
            )?;
            
            // Train the model
            println!("Training with batch size {}...", batch_size);
            let batch_start = Instant::now();
            batch_model.train(&market_data).await?;
            let batch_duration = batch_start.elapsed();
            println!("Batch size {} completed in {:?}", batch_size, batch_duration);
            
            // Update best batch size if this one is faster
            if batch_duration < best_time {
                best_time = batch_duration;
                best_batch_size = batch_size;
            }
        }
        
        println!("\nBatch size optimization results:");
        println!("Best batch size: {}", best_batch_size);
        println!("Best time: {:?}", best_time);
        
        // Example 4: CPU fallback
        println!("\n=== Example 4: CPU Fallback ===");
        
        // Create a GPU configuration that falls back to CPU
        let fallback_config = GPUConfig {
            device: ComputeDevice::CPU, // Explicitly use CPU
            use_mixed_precision: false,
            batch_size: 64,
            use_tensor_cores: false,
        };
        
        // Create a model with CPU fallback
        let mut fallback_model = GPUTransformerRiskModel::with_config(
            transformer_config.clone(),
            fallback_config
        )?;
        
        // Train the model
        println!("Training with CPU fallback...");
        let fallback_start = Instant::now();
        fallback_model.train(&market_data).await?;
        let fallback_duration = fallback_start.elapsed();
        println!("CPU fallback training completed in {:?}", fallback_duration);
        
        // Compare with GPU
        println!("\nCPU fallback vs. GPU:");
        println!("GPU time: {:?}", gpu_duration);
        println!("CPU fallback time: {:?}", fallback_duration);
        
        // Example 5: Memory optimization
        println!("\n=== Example 5: Memory Optimization ===");
        
        // Create a larger dataset to demonstrate memory optimization
        let large_n_samples = 1000;
        let large_n_assets = 128;
        println!("Generating large dataset with {} samples and {} assets...", large_n_samples, large_n_assets);
        let large_features = Array::random((large_n_samples, large_n_assets), Normal::new(0.0, 1.0).unwrap());
        let large_returns = Array::random((large_n_samples, large_n_assets), Normal::new(0.0, 0.1).unwrap());
        let large_market_data = MarketData::new(large_returns, large_features);
        
        // Create a GPU configuration with memory optimization
        let memory_config = GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: true, // Use mixed precision to reduce memory usage
            batch_size: 32, // Smaller batch size to reduce memory usage
            use_tensor_cores: true,
        };
        
        // Create a GPU model with memory optimization
        let large_transformer_config = TransformerConfig::new(large_n_assets, d_model, n_heads, d_ff, n_layers);
        let mut memory_model = GPUTransformerRiskModel::with_config(
            large_transformer_config,
            memory_config
        )?;
        
        // Train the model
        println!("Training large model with memory optimization...");
        let memory_start = Instant::now();
        memory_model.train(&large_market_data).await?;
        let memory_duration = memory_start.elapsed();
        println!("Large model training completed in {:?}", memory_duration);
        
        println!("\nGPU acceleration example completed successfully!");
    }
    
    Ok(())
} 