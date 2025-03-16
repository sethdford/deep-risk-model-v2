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
//! 
//! # Main Components
//! 
//! - [`DeepRiskModel`]: The primary model combining deep learning with risk factor analysis
//! - [`TransformerRiskModel`]: Risk modeling using transformer architecture
//! - [`TFTRiskModel`]: Risk modeling using temporal fusion transformer
//! - [`FactorAnalyzer`]: Advanced factor analysis and quality metrics
//! - [`RiskModel`]: Trait defining the interface for all risk models
//! 
//! # Example
//! 
//! ```rust
//! use deep_risk_model::{DeepRiskModel, MarketData, RiskModel};
//! use ndarray::Array2;
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize model
//!     let mut model = DeepRiskModel::new(10, 5)?;
//!     
//!     // Prepare market data
//!     let features = Array2::zeros((100, 10));
//!     let returns = Array2::zeros((100, 10));
//!     let data = MarketData::new(returns, features);
//!     
//!     // Train model and generate risk factors
//!     model.train(&data).await?;
//!     let risk_factors = model.generate_risk_factors(&data).await?;
//!     
//!     // Estimate covariance matrix
//!     let covariance = model.estimate_covariance(&data).await?;
//!     
//!     Ok(())
//! }

#![recursion_limit = "32768"]

pub mod error;
pub mod gat;
pub mod gru;
pub mod model;
pub mod transformer;
pub mod transformer_risk_model;
pub mod tft_risk_model;
pub mod types;
pub mod utils;
pub mod factor_analysis;

// Public exports
pub use error::ModelError;
pub use model::DeepRiskModel;
pub use transformer::TransformerConfig;
pub use transformer_risk_model::TransformerRiskModel;
pub use tft_risk_model::TFTRiskModel;
pub use types::{MarketData, RiskFactors, RiskModel, ModelConfig, MCPConfig};
pub use factor_analysis::{FactorAnalyzer, FactorQualityMetrics};

// Re-export utilities
pub mod array_utils {
    pub use crate::utils::*;
}

#[cfg(test)]
mod tests {
    use crate::transformer_risk_model::TransformerRiskModel;
    use crate::tft_risk_model::TFTRiskModel;
    use crate::types::{MarketData, RiskModel};
    use crate::error::ModelError;
    use ndarray::Array;

    #[tokio::test]
    async fn test_transformer_risk_model() -> Result<(), ModelError> {
        let n_assets = 64;
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        
        let n_samples = 6;
        let features = Array::zeros((n_samples, n_assets));
        let returns = Array::zeros((n_samples, n_assets));
        
        let data = MarketData::new(returns, features);
        let risk_factors = model.generate_risk_factors(&data).await?;
        
        // The output shape is [n_samples - max_seq_len + 1, d_model]
        // With n_samples = 6 and max_seq_len = 5, we expect [2, 64]
        assert_eq!(risk_factors.factors().shape(), &[2, d_model]);
        assert_eq!(risk_factors.covariance().shape(), &[d_model, d_model]);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_tft_risk_model() -> Result<(), ModelError> {
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
    }
}