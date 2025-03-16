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
//! let mut model = DeepRiskModel::new(64, 5)?;
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

pub mod error;
pub mod factor_analysis;
pub mod gru;
pub mod model;
pub mod regime;
pub mod regime_risk_model;
pub mod transformer;
pub mod transformer_risk_model;
pub mod tft_risk_model;
pub mod types;
pub mod utils;
pub mod backtest;
pub mod stress_testing;
pub mod fallback;

// GPU acceleration modules
pub mod gpu;
pub mod gpu_transformer_risk_model;
pub mod gpu_model;

// Model compression and optimization
pub mod quantization;
pub mod memory_opt;

// Public exports
pub mod prelude {
    pub use crate::error::ModelError;
    pub use crate::model::DeepRiskModel;
    pub use crate::transformer::TransformerConfig;
    pub use crate::transformer_risk_model::TransformerRiskModel;
    pub use crate::tft_risk_model::TFTRiskModel;
    pub use crate::types::{MarketData, RiskFactors, RiskModel, ModelConfig, MCPConfig};
    pub use crate::factor_analysis::{FactorAnalyzer, FactorQualityMetrics};
    pub use crate::regime::{MarketRegimeDetector, RegimeType, RegimeConfig};
    pub use crate::regime_risk_model::{RegimeAwareRiskModel, RegimeParameters};
    pub use crate::backtest::{Backtest, BacktestResults, ScenarioGenerator, HistoricalScenarioGenerator, StressScenarioGenerator};
    pub use crate::stress_testing::{EnhancedStressScenarioGenerator, StressTestExecutor, StressTestResults, 
                                StressScenario, HistoricalPeriod, ScenarioCombinationSettings, 
                                StressTestSettings, ReportDetail, ScenarioComparison};
    
    // GPU acceleration types
    pub use crate::gpu::{ComputeDevice, GPUConfig};
    pub use crate::gpu_transformer_risk_model::GPUTransformerRiskModel;
    pub use crate::gpu_model::GPUDeepRiskModel;
    
    // Quantization types
    pub use crate::quantization::{Quantizable, QuantizationConfig, QuantizationPrecision, Quantizer};
    
    // Memory optimization types
    pub use crate::memory_opt::{MemoryConfig, SparseTensor, ChunkedProcessor, GradientCheckpointer, MemoryMappedArray, MemoryPool};
}

// Re-export utilities
pub mod array_utils {
    pub use crate::utils::*;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::transformer::TransformerConfig;
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let n_assets = 100;
            let n_factors = 5;
            
            let model = TFTRiskModel::new(n_assets, n_factors)?;
            
            let n_samples = 100;
            let features = Array::zeros((n_samples, n_assets));
            let returns = Array::zeros((n_samples, n_assets));
            
            let data = MarketData::new(returns, features);
            let risk_factors = model.generate_risk_factors(&data).await?;
            
            assert_eq!(risk_factors.factors().shape()[0], n_samples);
            assert_eq!(risk_factors.covariance().shape()[0], risk_factors.covariance().shape()[1]);
            
            Ok(())
        })
    }
}

// Conditional imports based on features
#[cfg(all(feature = "blas-enabled", not(feature = "no-blas")))]
pub use ndarray_linalg;

// When no-blas is enabled or blas-enabled is not set, provide fallback implementations
#[cfg(any(feature = "no-blas", not(feature = "blas-enabled")))]
pub mod ndarray_linalg {
    // Empty module to satisfy imports when BLAS is not available
    pub mod error {
        #[derive(Debug)]
        pub struct LinalgError;
        
        impl std::fmt::Display for LinalgError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Linear algebra operation failed (fallback implementation)")
            }
        }
        
        impl std::error::Error for LinalgError {}
    }
    
    pub trait Solve<T> {
        type Output;
        fn solve(&self, _b: &T) -> Result<Self::Output, error::LinalgError>;
    }
    
    pub trait Eigh {
        type Output;
        fn eigh(&self) -> Result<Self::Output, error::LinalgError>;
    }
    
    pub trait Svd {
        type Output;
        fn svd(&self) -> Result<Self::Output, error::LinalgError>;
    }
    
    // Implement the traits for Array2 to provide fallback implementations
    impl<S, D> Solve<ndarray::ArrayBase<S, D>> for ndarray::Array2<f32>
    where
        S: ndarray::Data<Elem = f32>,
        D: ndarray::Dimension,
    {
        type Output = ndarray::Array<f32, D>;
        
        fn solve(&self, _b: &ndarray::ArrayBase<S, D>) -> Result<Self::Output, error::LinalgError> {
            // Use fallback implementation
            Err(error::LinalgError)
        }
    }
    
    impl Eigh for ndarray::Array2<f32> {
        type Output = (ndarray::Array<f32, ndarray::Ix1>, ndarray::Array<f32, ndarray::Ix2>);
        
        fn eigh(&self) -> Result<Self::Output, error::LinalgError> {
            // Use fallback implementation
            Err(error::LinalgError)
        }
    }
    
    impl Svd for ndarray::Array2<f32> {
        type Output = (ndarray::Array<f32, ndarray::Ix2>, ndarray::Array<f32, ndarray::Ix1>, ndarray::Array<f32, ndarray::Ix2>);
        
        fn svd(&self) -> Result<Self::Output, error::LinalgError> {
            // Use fallback implementation
            Err(error::LinalgError)
        }
    }
}