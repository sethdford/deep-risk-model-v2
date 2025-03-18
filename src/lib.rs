#![recursion_limit = "256"]

//! Deep Risk Model - A Rust library for financial risk modeling using deep learning
//! 
//! This library provides tools for analyzing and modeling financial risk using advanced deep learning
//! techniques. It includes implementations of:
//! 
//! - Deep Risk Model with transformer architecture
//! - Temporal Fusion Transformer (TFT) for time series analysis
//! - Factor Analysis for risk decomposition
//! - Graph Attention Networks (GAT) for asset relationships
//! - Gated Recurrent Units (GRU) for temporal dependencies
//! - Market Regime Detection using Hidden Markov Models
//! - Regime-Aware Risk Models for adaptive risk estimation
//! - Backtesting framework for model evaluation
//! - GPU acceleration for high-performance risk modeling
//! - Quantization for model compression and inference acceleration
//! 
//! # Main Components
//! 
//! - [`DeepRiskModel`]: The primary model combining deep learning with risk factor analysis
//! - [`TransformerRiskModel`]: Risk modeling using transformer architecture
//! - [`TFTRiskModel`]: Risk modeling using temporal fusion transformer
//! - [`FactorAnalyzer`]: Advanced factor analysis and quality metrics
//! - [`RiskModel`]: Trait defining the interface for all risk models
//! - [`MarketRegimeDetector`]: Hidden Markov Model for market regime detection
//! - [`RegimeAwareRiskModel`]: Risk model that adapts to different market regimes
//! - [`Backtest`]: Framework for backtesting risk models
//! - [`GPUDeepRiskModel`]: GPU-accelerated version of the deep risk model
//! - [`Quantizable`]: Trait for models that support quantization
//! 
//! # Example
//! 
//! ```rust,no_run
//! use deep_risk_model::prelude::{DeepRiskModel, MarketData, RiskModel};
//! use ndarray::Array2;
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a model for 64 assets with 5 risk factors
//! let mut model = DeepRiskModel::new(
//!     64, // n_assets
//!     5,  // n_factors
//!     50, // max_seq_len
//!     128, // d_model
//!     4,  // n_heads
//!     256, // d_ff
//!     3   // n_layers
//! )?;
//! 
//! // Generate synthetic market data
//! let features = Array2::<f32>::zeros((100, 64));
//! let returns = Array2::<f32>::zeros((100, 64));
//! let market_data = MarketData::new(returns, features);
//! 
//! // Train the model
//! model.train(&market_data).await?;
//! 
//! // Generate risk factors
//! let risk_factors = model.generate_risk_factors(&market_data).await?;
//! 
//! // Estimate covariance matrix
//! let covariance = model.estimate_covariance(&market_data).await?;
//! # Ok(())
//! # }

// Main module groups
pub mod core;
pub mod models;
pub mod nn;
pub mod optimization;
pub mod gpu;
pub mod api;

// Direct access to modules for backward compatibility
// These will be deprecated in future versions
pub mod error {
    pub use crate::core::error::*;
}

pub mod types {
    pub use crate::core::types::*;
}

pub mod utils {
    pub use crate::core::utils::*;
}

pub mod linalg {
    pub use crate::core::linalg::*;
}

pub mod model {
    pub use crate::models::model::*;
}

pub mod factor_analysis {
    pub use crate::models::factor_analysis::*;
}

pub mod transformer_risk_model {
    pub use crate::models::transformer_risk_model::*;
}

pub mod tft_risk_model {
    pub use crate::models::tft_risk_model::*;
}

pub mod regime_risk_model {
    pub use crate::models::regime_risk_model::*;
}

pub mod regime {
    pub use crate::models::regime::*;
}

pub mod gru {
    pub use crate::nn::gru::*;
}

pub mod gat {
    pub use crate::nn::gat::*;
}

pub mod transformer {
    pub use crate::nn::transformer::*;
}

pub mod backtest {
    pub use crate::optimization::backtest::*;
}

pub mod stress_testing {
    pub use crate::optimization::stress_testing::*;
}

pub mod fallback {
    pub use crate::optimization::fallback::*;
}

pub mod quantization {
    pub use crate::optimization::quantization::*;
}

pub mod memory_opt {
    pub use crate::optimization::memory_opt::*;
}

// Re-export ndarray_linalg as our linalg module for backward compatibility
pub mod ndarray_linalg {
    pub use crate::core::linalg::*;
}

// Public exports
pub mod prelude {
    // Core types
    pub use crate::core::{ModelError, MarketData, RiskFactors, RiskModel, ModelConfig, MCPConfig};
    
    // Main model implementations
    pub use crate::models::{
        DeepRiskModel, 
        TransformerRiskModel, 
        TFTRiskModel,
        FactorAnalyzer, 
        FactorQualityMetrics,
        MarketRegimeDetector, 
        RegimeType, 
        RegimeConfig,
        RegimeAwareRiskModel, 
        RegimeParameters
    };
    
    // Configuration
    pub use crate::nn::TransformerConfig;
    
    // Optimization
    pub use crate::optimization::{
        Quantizable, 
        QuantizationConfig, 
        QuantizationPrecision, 
        Quantizer,
        MemoryConfig, 
        SparseTensor, 
        ChunkedProcessor, 
        GradientCheckpointer, 
        MemoryMappedArray, 
        MemoryPool,
        Backtest, 
        BacktestResults, 
        ScenarioGenerator, 
        HistoricalScenarioGenerator, 
        StressScenarioGenerator,
        EnhancedStressScenarioGenerator, 
        StressTestExecutor, 
        StressTestResults, 
        StressScenario, 
        HistoricalPeriod, 
        ScenarioCombinationSettings, 
        StressTestSettings, 
        ReportDetail, 
        ScenarioComparison
    };
    
    // API
    pub use crate::api::{AppState, run_server};
    
    // GPU acceleration types
    #[cfg(feature = "gpu")]
    pub use crate::gpu::{ComputeDevice, GPUConfig, GPUTransformerRiskModel, GPUDeepRiskModel};
}

// Re-export utilities
pub mod array_utils {
    pub use crate::core::utils::*;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use ndarray::Array;

    #[test]
    fn test_transformer_risk_model() -> Result<(), ModelError> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let n_assets = 64;
            let d_model = 64;
            let n_heads = 8;
            let d_ff = 256;
            let n_layers = 3;
            
            // Create a custom config with smaller max_seq_len
            let mut config = TransformerConfig::new(n_assets, d_model, n_heads, d_ff, n_layers);
            config.max_seq_len = 3; // Use a smaller max_seq_len for testing
            
            let model = TransformerRiskModel::with_config(config)?;
            
            let n_samples = 6;
            let features = Array::zeros((n_samples, n_assets));
            let returns = Array::zeros((n_samples, n_assets));
            
            let data = MarketData::new(returns, features);
            let risk_factors = model.generate_risk_factors(&data).await?;
            
            // The output shape is [n_samples - max_seq_len + 1, d_model]
            // With n_samples = 6 and max_seq_len = 3, we expect [4, 64]
            assert_eq!(risk_factors.factors().shape(), &[4, d_model]);
            assert_eq!(risk_factors.covariance().shape(), &[d_model, d_model]);
            
            Ok(())
        })
    }
    
    #[test]
    fn test_tft_risk_model() -> Result<(), ModelError> {
        // Skip test when BLAS is not enabled
        #[cfg(not(feature = "blas-enabled"))]
        {
            println!("Skipping test_tft_risk_model when BLAS is not enabled");
            return Ok(());
        }
        
        // Create a TFTRiskModel
        let model = TFTRiskModel::new(
            16, // n_assets
            2   // n_factors
        )?;
        
        // Just test that the model can be created without error
        assert!(true);
        
        Ok(())
    }
}