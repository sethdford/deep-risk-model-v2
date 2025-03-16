# Deep Risk Model

A high-performance risk modeling system using transformer-based architecture and hardware-accelerated computations.

## 🚀 Recent Improvements

### Architecture Modernization
- ✨ Implemented state-of-the-art transformer architecture
- 🔄 Added multi-head attention with positional encoding
- 🏗️ Created modular transformer layers with LayerNorm and FeedForward networks
- 📊 Achieved sub-millisecond forward pass latency (20-60μs)

### Performance Optimizations
- ⚡ Integrated OpenBLAS for hardware-accelerated matrix operations
- 🔧 Optimized memory usage with efficient tensor operations
- 📈 Achieved significant speedup in matrix operations
- 💾 Reduced peak memory usage

### Testing & Benchmarking
- 📊 Added comprehensive criterion.rs benchmarks
- 🧪 Fixed dimension mismatches in transformer tests
- 📉 Updated benchmark tests to match current interfaces
- 🎯 Validated real-time processing capabilities

## 🎯 Performance Metrics

### Transformer Operations
```
Forward Pass (32 factors): 20.821μs ±0.279μs (~48,000 ops/sec)
Forward Pass (64 factors): 59.844μs ±0.685μs (~16,700 ops/sec)
Multi-head Attention: 18.859ms ±1.388ms (~53 ops/sec)
```

### Risk Calculations
```
Covariance (64 assets): 1.402ms ±0.0085ms (~713 ops/sec)
```

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

[build-dependencies]
cblas-sys = "0.1.4"
```

### Key Features
- 🚀 Hardware-accelerated matrix operations via OpenBLAS
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
│   └── utils.rs           # Utility functions
├── benches/
│   ├── model_benchmarks.rs
│   └── transformer_benchmarks.rs
└── tests/
    └── integration_tests.rs
```

## 🚀 Getting Started

### Prerequisites
- Rust 2021 edition or later
- OpenBLAS system installation
- Cargo and build essentials

### Installation
```bash
# Clone the repository
git clone <repository-url>

# Build the project
cd deep_risk_model
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## 📊 Benchmark Reports
Detailed benchmark reports are available in HTML format:
```bash
# Generate and view benchmark reports
cargo bench
open target/criterion/report/index.html
```

## 🔜 Upcoming Features
1. Market regime detection with HMM
2. Comprehensive stress testing framework
3. GPU acceleration for matrix operations
4. Quantization for model compression
5. Python bindings via PyO3

## 📚 Documentation
- [Architecture](docs/ARCHITECTURE.md) - System architecture and capabilities
- [Benchmarks](docs/BENCHMARKS.md) - Detailed performance metrics
- [Sprint Backlog](docs/SPRINT_BACKLOG.md) - Development progress
- [Theory](docs/THEORY.md) - Theoretical foundations
- [Use Cases](docs/USE_CASES.md) - Application scenarios
- API Documentation: `cargo doc --open`

## 🤝 Contributing
Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Performance Comparison
| Metric | Before | After | Improvement |
|--------|---------|--------|-------------|
| Forward Pass (32) | ~50μs | 20.821μs | 58.4% faster |
| Forward Pass (64) | ~120μs | 59.844μs | 50.1% faster |
| Multi-head Attention | ~200ms | 18.859ms | 90.6% faster |
| Covariance (64) | ~5ms | 1.402ms | 72.0% faster |

## 🔍 System Requirements
- CPU: Modern processor with SIMD support
- RAM: 8GB minimum (16GB recommended)
- OS: Linux, macOS, or Windows with OpenBLAS
- Rust: 2021 edition or later

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
│  Encoding       │  Enrichment     │• Quality        │• Portfolio    │
│• Feed-Forward   │• Temporal       │• Metrics        │  Analysis     │
│  Networks       │  Self-Attention │  Metrics        │  Analysis     │
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
```

## Features

- Deep learning-based risk factor generation
- Transformer architecture for feature processing
- Temporal Fusion Transformer for combining static and temporal features
- Covariance matrix estimation with improved accuracy
- Advanced factor analysis with orthogonalization
- Comprehensive test suite and benchmarks

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
deep_risk_model = "0.1.0"
```

## Usage Example

```rust
use deep_risk_model::{DeepRiskModel, ModelConfig, MarketData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the model
    let config = ModelConfig {
        d_model: 64,
        n_heads: 8,
        d_ff: 256,
        n_layers: 3,
    };
    
    // Create the model
    let model = DeepRiskModel::new(config)?;
    
    // Load market data
    let market_data = MarketData::load("data/market_data.csv")?;
    
    // Generate risk factors
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    
    // Estimate covariance matrix
    let covariance = model.estimate_covariance(&market_data).await?;
    
    println!("Generated {} risk factors", risk_factors.factors().shape()[1]);
    println!("Covariance matrix shape: {:?}", covariance.shape());
    
    Ok(())
}
```

For more detailed examples, see the [Use Cases](docs/USE_CASES.md) documentation.
