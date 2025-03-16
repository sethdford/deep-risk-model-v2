# GPU Acceleration Guide

This guide explains how to use the GPU acceleration features in the Deep Risk Model library to significantly improve performance for large datasets and complex models.

## Table of Contents

- [GPU Acceleration Guide](#gpu-acceleration-guide)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Enabling GPU Support](#enabling-gpu-support)
  - [Creating GPU-Accelerated Models](#creating-gpu-accelerated-models)
    - [Basic Usage](#basic-usage)
    - [With Custom Configuration](#with-custom-configuration)
  - [GPU Configuration Options](#gpu-configuration-options)
    - [ComputeDevice](#computedevice)
    - [Mixed Precision](#mixed-precision)
    - [Batch Size](#batch-size)
    - [Tensor Cores](#tensor-cores)
  - [Checking GPU Availability](#checking-gpu-availability)
  - [Switching Between CPU and GPU](#switching-between-cpu-and-gpu)
  - [Performance Considerations](#performance-considerations)
  - [Troubleshooting](#troubleshooting)
    - [GPU Not Detected](#gpu-not-detected)
    - [Performance Issues](#performance-issues)
    - [Memory Errors](#memory-errors)
  - [Examples](#examples)
    - [Basic GPU Example](#basic-gpu-example)
    - [Performance Comparison Example](#performance-comparison-example)
  - [Further Reading](#further-reading)

## Prerequisites

To use GPU acceleration, you need:

- CUDA Toolkit 11.0 or later installed on your system
- A CUDA-compatible GPU (NVIDIA)
- The `gpu` feature enabled in your Cargo.toml

## Enabling GPU Support

To enable GPU support, add the `gpu` feature to your dependencies in `Cargo.toml`:

```toml
[dependencies]
deep_risk_model = { version = "0.1.0", features = ["gpu"] }
```

Or when building your project:

```bash
cargo build --release --features gpu
```

## Creating GPU-Accelerated Models

The library provides GPU-accelerated versions of the core models:

- `GPUDeepRiskModel` - GPU-accelerated version of `DeepRiskModel`
- `GPUTransformerRiskModel` - GPU-accelerated version of `TransformerRiskModel`

### Basic Usage

```rust
use deep_risk_model::{
    gpu_model::GPUDeepRiskModel,
    gpu::{ComputeDevice, GPUConfig},
    types::{MarketData, RiskModel},
};

// Create GPU configuration
let gpu_config = GPUConfig {
    device: ComputeDevice::GPU,
    use_mixed_precision: true,
    batch_size: 64,
    use_tensor_cores: true,
};

// Create GPU model
let mut model = GPUDeepRiskModel::new(
    64,  // n_assets
    5,   // n_factors
    Some(gpu_config),
)?;

// Use the model as you would use the CPU version
model.train(&market_data).await?;
let risk_factors = model.generate_risk_factors(&market_data).await?;
let covariance = model.estimate_covariance(&market_data).await?;
```

### With Custom Configuration

You can create a GPU model with custom transformer configuration:

```rust
use deep_risk_model::{
    gpu_model::GPUDeepRiskModel,
    gpu::{ComputeDevice, GPUConfig},
    transformer::TransformerConfig,
    types::{MarketData, RiskModel},
};

// Create transformer configuration
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

// Create GPU configuration
let gpu_config = GPUConfig {
    device: ComputeDevice::GPU,
    use_mixed_precision: true,
    batch_size: 64,
    use_tensor_cores: true,
};

// Create GPU model with custom configuration
let mut model = GPUDeepRiskModel::with_transformer_config(
    64,  // n_assets
    5,   // n_factors
    transformer_config,
    Some(gpu_config),
)?;
```

## GPU Configuration Options

The `GPUConfig` struct provides several options to customize GPU acceleration:

```rust
pub struct GPUConfig {
    /// Compute device to use (CPU or GPU)
    pub device: ComputeDevice,
    
    /// Whether to use mixed precision (FP16/FP32)
    pub use_mixed_precision: bool,
    
    /// Batch size for GPU operations
    pub batch_size: usize,
    
    /// Whether to use tensor cores (if available)
    pub use_tensor_cores: bool,
}
```

### ComputeDevice

The `ComputeDevice` enum specifies which device to use for computation:

```rust
pub enum ComputeDevice {
    /// CPU computation (default)
    CPU,
    /// GPU computation (if available)
    GPU,
}
```

### Mixed Precision

Setting `use_mixed_precision` to `true` enables mixed precision computation, which can significantly improve performance on GPUs with tensor cores (NVIDIA Volta, Turing, Ampere, or newer architectures).

### Batch Size

The `batch_size` parameter controls how many samples are processed in parallel on the GPU. Larger batch sizes generally improve GPU utilization but require more memory.

### Tensor Cores

Setting `use_tensor_cores` to `true` enables the use of tensor cores on supported GPUs, which can provide significant speedups for matrix operations.

## Checking GPU Availability

You can check if CUDA is available on the system:

```rust
use deep_risk_model::gpu::{is_cuda_available, get_gpu_info};

// Check if CUDA is available
let gpu_available = is_cuda_available();
println!("GPU available: {}", gpu_available);

// Get information about available GPUs
println!("GPU info: {}", get_gpu_info());
```

## Switching Between CPU and GPU

You can dynamically switch between CPU and GPU computation:

```rust
use deep_risk_model::{
    gpu_model::GPUDeepRiskModel,
    gpu::{ComputeDevice, GPUConfig},
};

// Create model with CPU configuration
let mut model = GPUDeepRiskModel::new(
    64,  // n_assets
    5,   // n_factors
    Some(GPUConfig {
        device: ComputeDevice::CPU,
        ..GPUConfig::default()
    }),
)?;

// Run on CPU
let cpu_result = model.generate_risk_factors(&market_data).await?;

// Switch to GPU
model.set_gpu_config(GPUConfig {
    device: ComputeDevice::GPU,
    use_mixed_precision: true,
    batch_size: 64,
    use_tensor_cores: true,
});

// Run on GPU
let gpu_result = model.generate_risk_factors(&market_data).await?;
```

## Performance Considerations

For optimal GPU performance:

1. **Batch Size**: Experiment with different batch sizes to find the optimal value for your GPU.
2. **Mixed Precision**: Enable mixed precision for modern GPUs to get significant speedups.
3. **Data Transfer**: Minimize data transfers between CPU and GPU.
4. **GPU Memory**: Monitor GPU memory usage, especially for large datasets.
5. **Tensor Cores**: Enable tensor cores on supported GPUs for maximum performance.

## Troubleshooting

### GPU Not Detected

If the GPU is not detected:

1. Ensure CUDA Toolkit is properly installed
2. Check that the GPU driver is up to date
3. Verify that the `gpu` feature is enabled in Cargo.toml
4. Run `nvidia-smi` to check if the GPU is recognized by the system

### Performance Issues

If GPU performance is not as expected:

1. Check GPU utilization using `nvidia-smi`
2. Experiment with different batch sizes
3. Enable mixed precision if using a modern GPU
4. Ensure the model is large enough to benefit from GPU acceleration

### Memory Errors

If you encounter GPU memory errors:

1. Reduce batch size
2. Use mixed precision to reduce memory usage
3. Simplify the model (fewer layers, smaller dimensions)
4. Process data in smaller chunks

## Examples

### Basic GPU Example

```rust
use deep_risk_model::{
    gpu_model::GPUDeepRiskModel,
    gpu::{ComputeDevice, GPUConfig, is_cuda_available},
    types::{MarketData, RiskModel},
};
use ndarray::Array2;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check GPU availability
    let gpu_available = is_cuda_available();
    println!("GPU available: {}", gpu_available);
    
    // Create synthetic market data
    let n_samples = 200;
    let n_assets = 64;
    let features = Array2::zeros((n_samples, n_assets));
    let returns = Array2::zeros((n_samples, n_assets));
    let market_data = MarketData::new(returns, features);
    
    // Create GPU configuration
    let gpu_config = if gpu_available {
        GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: true,
            batch_size: 64,
            use_tensor_cores: true,
        }
    } else {
        GPUConfig::default() // Falls back to CPU
    };
    
    // Create GPU model
    let mut model = GPUDeepRiskModel::new(n_assets, 5, Some(gpu_config))?;
    
    // Train model
    model.train(&market_data).await?;
    
    // Measure performance
    let start = Instant::now();
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    let elapsed = start.elapsed();
    
    println!("Generated {} risk factors in {:?}", 
             risk_factors.factors().shape()[1], elapsed);
    
    Ok(())
}
```

### Performance Comparison Example

```rust
use deep_risk_model::{
    model::DeepRiskModel,
    gpu_model::GPUDeepRiskModel,
    gpu::{ComputeDevice, GPUConfig},
    types::{MarketData, RiskModel},
};
use ndarray::Array2;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create synthetic market data
    let n_samples = 200;
    let n_assets = 64;
    let features = Array2::zeros((n_samples, n_assets));
    let returns = Array2::zeros((n_samples, n_assets));
    let market_data = MarketData::new(returns, features);
    
    // Create CPU model
    let mut cpu_model = DeepRiskModel::new(n_assets, 5)?;
    
    // Create GPU model
    let mut gpu_model = GPUDeepRiskModel::new(
        n_assets, 
        5, 
        Some(GPUConfig {
            device: ComputeDevice::GPU,
            use_mixed_precision: true,
            batch_size: 64,
            use_tensor_cores: true,
        }),
    )?;
    
    // Train models
    cpu_model.train(&market_data).await?;
    gpu_model.train(&market_data).await?;
    
    // Measure CPU performance
    let cpu_start = Instant::now();
    let _cpu_factors = cpu_model.generate_risk_factors(&market_data).await?;
    let cpu_time = cpu_start.elapsed();
    println!("CPU time: {:?}", cpu_time);
    
    // Measure GPU performance
    let gpu_start = Instant::now();
    let _gpu_factors = gpu_model.generate_risk_factors(&market_data).await?;
    let gpu_time = gpu_start.elapsed();
    println!("GPU time: {:?}", gpu_time);
    
    // Calculate speedup
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("GPU speedup: {:.2}x", speedup);
    
    Ok(())
}
```

For more examples, see the `examples/gpu_example.rs` file in the repository.

## Further Reading

- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/) 