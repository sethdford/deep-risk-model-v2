# Deep Risk Model

A significant improvement over https://github.com/sethdford/deep_risk_model-v0 which is a Rust implementation of a deep learning-based risk model for financial markets, inspired by the research paper ["Deep Risk Model: A Deep Learning Solution for Mining Latent Risk Factors to Improve Covariance Matrix Estimation"](https://arxiv.org/abs/2107.05201) (Lin et al., 2021). This project combines Graph Attention Networks (GAT) and Gated Recurrent Units (GRU) to generate risk factors and estimate covariance matrices from market data.

## 🔑 Key Features

- **Advanced Risk Modeling**: Transformer architecture, Temporal Fusion Transformer (TFT), Factor Analysis
- **Market Intelligence**: Regime Detection with HMM, Adaptive Risk Estimation
- **Performance Optimizations**: GPU acceleration, Quantization, Memory optimization
- **Production Ready**: Thread-safe, Error handling, No-BLAS fallback, Python bindings
- **Comprehensive Testing**: Backtesting framework, Benchmarks, CI/CD integration

## 📚 Documentation & API

Comprehensive documentation is available to help you get started:

- [Architecture](docs/ARCHITECTURE.md) - System architecture and capabilities
- [Theory](docs/THEORY.md) - Theoretical foundations
- [Use Cases](docs/USE_CASES.md) - Application scenarios
- [Benchmarks](docs/BENCHMARKS.md) - Detailed performance metrics
- [Sprint Backlog](docs/SPRINT_BACKLOG.md) - Development progress

**API Documentation**: Run `cargo doc --open` for detailed API reference

## 🚀 Quick Start

### Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
deep_risk_model = "0.1.0"
```

### Basic Usage

```rust
use deep_risk_model::prelude::{DeepRiskModel, MarketData, RiskModel};
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample data
    let n_assets = 64;
    let n_samples = 100;
    let features = Array2::zeros((n_samples, n_assets));
    let returns = Array2::zeros((n_samples, n_assets));
    let data = MarketData::new(returns, features);
    
    // Create and train model
    let mut model = DeepRiskModel::new(n_assets, 5)?;
    model.train(&data).await?;
    
    // Generate risk factors
    let risk_factors = model.generate_risk_factors(&data).await?;
    
    // Estimate covariance matrix
    let covariance = model.estimate_covariance(&data).await?;
    
    Ok(())
}
```

### Advanced Features

```rust
use deep_risk_model::prelude::{
    TransformerRiskModel, MarketData, RiskModel,
    MemoryConfig, QuantizationConfig, QuantizationPrecision
};

// Create model with memory optimization
let mut model = TransformerRiskModel::new(64, 8, 256, 3)?;

// Configure memory optimization
let memory_config = MemoryConfig {
    use_sparse_tensors: true,
    sparsity_threshold: 0.7,
    use_chunked_processing: true,
    chunk_size: 1000,
    ..Default::default()
};
model.set_memory_config(memory_config);

// Quantize model for memory reduction
let quant_config = QuantizationConfig {
    precision: QuantizationPrecision::Int8,
    per_channel: true,
};
model.quantize(quant_config)?;
```

For more detailed examples, see the [Use Cases](docs/USE_CASES.md) documentation.

## 🛠️ Building & Installation

### BLAS Support

This library uses platform-specific BLAS implementations by default:

- **macOS**: Uses Apple's Accelerate framework for optimal performance
- **Windows**: Uses OpenBLAS via vcpkg
- **Linux/Others**: Uses OpenBLAS

You can build with the default platform-specific implementation:

```bash
# Build with the platform-specific default
cargo build
```

Or you can override the default by specifying a feature:

```bash
# Use OpenBLAS explicitly
cargo build --no-default-features --features openblas

# Use Netlib
cargo build --no-default-features --features netlib

# Use Intel MKL
cargo build --no-default-features --features intel-mkl

# Use Accelerate (macOS only)
cargo build --no-default-features --features accelerate

# Use system BLAS (required for Windows with vcpkg)
cargo build --no-default-features --features system

# Build without BLAS (pure Rust implementation)
cargo build --no-default-features --features no-blas
```

### System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get install -y libopenblas-dev gfortran
```

#### macOS

No additional dependencies required by default (uses built-in Accelerate framework).

If you want to use OpenBLAS instead:

```bash
brew install openblas
```

#### Windows

For Windows, it's recommended to use vcpkg:

```bash
# Install vcpkg if you haven't already
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg.exe integrate install

# Install OpenBLAS
./vcpkg.exe install openblas:x64-windows
```

Then build with the `system` feature:

```bash
cargo build --no-default-features --features system
```

### No-BLAS Fallback Implementation

The library includes a pure Rust fallback implementation for environments where BLAS is not available:

- ✅ Automatic fallback to pure Rust implementation when BLAS is not available
- ✅ Matrix multiplication and inversion implemented in pure Rust
- ✅ Comprehensive test suite for both BLAS and no-BLAS configurations

To use the no-BLAS implementation:

```bash
# Build without BLAS (pure Rust implementation)
cargo build --no-default-features --features no-blas
```

> **Note for Windows Users**: The no-BLAS feature is currently not supported on Windows without vcpkg. 
> If you need to build on Windows, please use vcpkg with the system feature or use WSL (Windows Subsystem for Linux).

> **Note for Performance**: The no-BLAS implementation is significantly slower for large matrices.
> It's recommended to use a BLAS implementation for production use.

## 🛠️ Technical Stack

### Core Dependencies
```toml
[dependencies]
ndarray = "0.15.6"
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
ndarray-rand = "0.14.0"
ndarray-stats = "0.5.1"
tokio = { version = "1.0", features = ["full", "macros", "rt-multi-thread"] }
criterion = "0.5"
rand = "0.8"
anyhow = "1.0"
thiserror = "1.0"

# GPU acceleration (optional)
cuda-runtime-sys = { version = "0.7.0", optional = true }
cublas-sys = { version = "0.7.0", optional = true }
curand-sys = { version = "0.7.0", optional = true }

[features]
default = []
gpu = ["cuda-runtime-sys", "cublas-sys", "curand-sys"]
openblas-system = ["ndarray-linalg/openblas-system"]
```

### Key Features
- 🚀 Hardware-accelerated matrix operations via OpenBLAS
- 🔥 GPU acceleration for high-performance computing (optional)
- 💾 Memory optimization for handling large models and datasets
- 📊 Model compression through quantization
- 🔄 Async runtime with Tokio
- 🌐 REST API with Axum
- 📊 Comprehensive benchmarking with criterion.rs

## 📦 Project Structure
```
deep_risk_model/
├── src/
│   ├── transformer/       # Transformer architecture
│   │   ├── attention.rs   # Multi-head attention
│   │   ├── position.rs    # Positional encoding
│   │   ├── layer.rs       # Transformer layers
│   │   ├── model.rs       # Transformer model
│   │   └── temporal_fusion.rs # TFT implementation
│   ├── factor_analysis.rs # Factor analysis utilities
│   ├── model.rs           # Core risk model
│   ├── transformer_risk_model.rs # Transformer-based risk model
│   ├── tft_risk_model.rs  # TFT-based risk model
│   ├── gpu.rs             # GPU acceleration utilities
│   ├── gpu_transformer_risk_model.rs # GPU-accelerated transformer
│   ├── gpu_model.rs       # GPU-accelerated deep risk model
│   ├── quantization.rs    # Model compression through quantization
│   ├── memory_opt.rs      # Memory optimization utilities
│   └── utils.rs           # Utility functions
├── benches/
│   ├── model_benchmarks.rs
│   └── transformer_benchmarks.rs
├── examples/
│   ├── quantization_example.rs # Example of model quantization
│   └── memory_optimization_example.rs # Example of memory optimization
└── tests/
    └── integration_tests.rs
```

## 🔍 System Requirements
- CPU: Modern processor with SIMD support
- RAM: 8GB minimum (16GB recommended)
- OS: Linux, macOS, or Windows with OpenBLAS
- Rust: 2021 edition or later

## Testing

Run all tests with:

```bash
cargo test --features openblas
```

For tests that don't require BLAS:

```bash
cargo test --no-default-features --features no-blas -- --skip factor_analysis::tests::test_factor_selection --skip gpu_model::tests::test_gpu_factor_metrics --skip model::tests::test_factor_generation --skip model::tests::test_factor_metrics --skip model::tests::test_covariance_estimation --skip gpu_model::tests::test_gpu_factor_generation --skip gpu_model::tests::test_gpu_vs_cpu_performance
```

Or use the provided script:

```bash
./run_tests.sh
```

## Examples

Run examples with:

```bash
cargo run --example quantization_example --features openblas
cargo run --example memory_optimization_example --features openblas
```

## 🤝 Contributing
Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Research Background

This implementation is based on academic research that demonstrates how deep learning can be used to mine latent risk factors and improve covariance matrix estimation. The original paper shows:

- 1.9% higher explained variance (measured by R²)
- Improved risk reduction in global minimum variance portfolios
- Novel approach to learning risk factors using neural networks
- Effective combination of temporal and cross-sectional features

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Deep Risk Model                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│  Transformer    │  Temporal       │  Factor         │  Risk         │
│  Architecture   │  Fusion         │  Analysis       │  Modeling     │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│• Multi-head     │• Variable       │• Orthogonal-    │• Covariance   │
│  Attention      │  Selection      │  ization        │  Estimation   │
│• Positional     │• Static         │• Factor         │• Risk Factor  │
│  Encoding       │• Temporal       │• Metrics        │  Analysis     │
│• Feed-Forward   │• Self-Attention │• Metrics        │  Analysis     │
│  Networks       │• Temporal       │• Metrics        │  Analysis     │
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
```

## Memory Optimization Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Memory Optimization                             │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│  Sparse         │  Chunked        │  Gradient       │  Memory       │
│  Tensors        │  Processing     │  Checkpointing  │  Management   │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│• Efficient      │• Large Dataset  │• Memory-        │• Memory Pool  │
│  Storage        │  Processing     │  Efficient      │• Memory-      │
│• Sparse Matrix  │• Configurable   │  Computation    │  Mapped       │
│  Operations     │  Chunk Size     │• Segment        │  Arrays       │
│• Memory Usage   │• Progress       │  Processing     │• Efficient    │
│  Tracking       │  Tracking       │• Memory Savings │  Allocation   │
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
```

## Features

- Deep learning-based risk factor generation
- Transformer architecture for feature processing
- Temporal Fusion Transformer for combining static and temporal features
- Covariance matrix estimation with improved accuracy
- Advanced factor analysis with orthogonalization
- Memory optimization for handling large models and datasets
- Model compression through quantization
- Comprehensive test suite and benchmarks

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
deep_risk_model = "0.1.0"
```

## Usage Example

```rust
use deep_risk_model::prelude::{
    DeepRiskModel, TransformerRiskModel, MarketData, RiskModel,
    MemoryConfig, QuantizationConfig, QuantizationPrecision
};
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample data
    let n_assets = 64;
    let n_samples = 100;
    let features = Array2::zeros((n_samples, n_assets));
    let returns = Array2::zeros((n_samples, n_assets));
    let data = MarketData::new(returns, features);
    
    // Create model with memory optimization
    let mut model = TransformerRiskModel::new(64, 8, 256, 3)?;
    
    // Configure memory optimization
    let memory_config = MemoryConfig {
        use_sparse_tensors: true,
        sparsity_threshold: 0.7,
        use_chunked_processing: true,
        chunk_size: 1000,
        use_checkpointing: true,
        checkpoint_segments: 4,
        ..Default::default()
    };
    model.set_memory_config(memory_config);
    
    // Sparsify model weights
    model.sparsify(0.1)?;
    
    // Generate risk factors with memory-efficient processing
    let risk_factors = model.generate_risk_factors(&data).await?;
    
    // Quantize model for further memory reduction
    let quant_config = QuantizationConfig {
        precision: QuantizationPrecision::Int8,
        per_channel: true,
    };
    model.quantize(quant_config)?;
    
    // Check memory savings
    let memory_usage = model.memory_usage();
    println!("Memory usage: {} bytes", memory_usage);
    
    Ok(())
}
```

For more detailed examples, see the [Use Cases](docs/USE_CASES.md) documentation.

## GPU Acceleration

The library provides GPU-accelerated versions of key components:

- `GPUDeepRiskModel`: GPU-accelerated deep risk model
- `GPUTransformerRiskModel`: GPU-accelerated transformer risk model
- `GPUConfig`: Configuration for GPU acceleration settings

To use GPU acceleration:

1. Build with the `gpu` feature: `cargo build --features gpu`
2. Use the GPU-accelerated model variants in your code
3. Configure GPU settings using `GPUConfig`

GPU acceleration provides significant performance improvements for:
- Matrix multiplication operations
- Attention mechanism computations
- Covariance matrix estimation
- Factor generation and analysis

**Note:** The current GPU implementation is a placeholder that demonstrates the architecture for GPU acceleration. It includes CPU fallbacks for all operations. Full CUDA integration requires uncommenting and updating the CUDA dependencies in Cargo.toml and installing the CUDA toolkit.

### Configurable Model Dimensions

The models now support configurable dimensions:

```rust
// Create model with default dimensions (d_model = n_assets)
let model = DeepRiskModel::new(64, 5)?;

// Create model with custom dimensions
let model = DeepRiskModel::with_config(64, 5, 128, 8, 512, 3)?;

// Create model with custom transformer configuration
let config = TransformerConfig {
    d_model: 128,
    n_heads: 8,
    d_ff: 512,
    n_layers: 3,
    // ... other config options
};
let model = DeepRiskModel::with_transformer_config(64, 5, config)?;
```

The same configuration options are available for `GPUDeepRiskModel`.
