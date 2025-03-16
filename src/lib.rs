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

// Conditional imports based on features
#[cfg(not(feature = "no-blas"))]
pub use ndarray_linalg;

// When no-blas is enabled, provide fallback implementations
#[cfg(feature = "no-blas")]
pub mod ndarray_linalg {
    // Empty module to satisfy imports when BLAS is not available
    pub mod error {
        #[derive(Debug)]
        pub struct LinalgError;
    }
}

// Fallback implementations for when BLAS is not available
pub mod fallback {
    //! Fallback implementations for when BLAS is not available
    use ndarray::{Array2, ArrayBase, Data, Ix2};
    use crate::error::ModelError;

    /// A simple matrix multiplication fallback when BLAS is not available
    pub fn matmul<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<Array2<f32>, ModelError>
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
    {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        if a_cols != b_rows {
            return Err(ModelError::InvalidDimension(format!(
                "Matrix dimensions don't match for multiplication: ({}, {}) and ({}, {})",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let mut result = Array2::zeros((a_rows, b_cols));

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// A simple matrix inversion fallback when BLAS is not available
    /// This is a very basic implementation and should only be used when BLAS is not available
    pub fn inv(a: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let (rows, cols) = a.dim();
        if rows != cols {
            return Err(ModelError::InvalidDimension(format!(
                "Matrix must be square for inversion, got: ({}, {})",
                rows, cols
            )));
        }

        // For simplicity, we'll only implement 2x2 and 3x3 matrix inversion
        // For larger matrices, we'll return an error suggesting to use BLAS
        match rows {
            2 => {
                let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
                if det.abs() < 1e-10 {
                    return Err(ModelError::NumericalError("Matrix is singular".to_string()));
                }
                let inv_det = 1.0 / det;
                let mut result = Array2::zeros((2, 2));
                result[[0, 0]] = a[[1, 1]] * inv_det;
                result[[0, 1]] = -a[[0, 1]] * inv_det;
                result[[1, 0]] = -a[[1, 0]] * inv_det;
                result[[1, 1]] = a[[0, 0]] * inv_det;
                Ok(result)
            }
            3 => {
                // 3x3 matrix inversion using cofactors
                let mut result = Array2::zeros((3, 3));
                let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                        - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                        + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
                
                if det.abs() < 1e-10 {
                    return Err(ModelError::NumericalError("Matrix is singular".to_string()));
                }
                
                let inv_det = 1.0 / det;
                
                // Calculate cofactors
                result[[0, 0]] = (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]]) * inv_det;
                result[[0, 1]] = (a[[0, 2]] * a[[2, 1]] - a[[0, 1]] * a[[2, 2]]) * inv_det;
                result[[0, 2]] = (a[[0, 1]] * a[[1, 2]] - a[[0, 2]] * a[[1, 1]]) * inv_det;
                result[[1, 0]] = (a[[1, 2]] * a[[2, 0]] - a[[1, 0]] * a[[2, 2]]) * inv_det;
                result[[1, 1]] = (a[[0, 0]] * a[[2, 2]] - a[[0, 2]] * a[[2, 0]]) * inv_det;
                result[[1, 2]] = (a[[0, 2]] * a[[1, 0]] - a[[0, 0]] * a[[1, 2]]) * inv_det;
                result[[2, 0]] = (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]) * inv_det;
                result[[2, 1]] = (a[[0, 1]] * a[[2, 0]] - a[[0, 0]] * a[[2, 1]]) * inv_det;
                result[[2, 2]] = (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]) * inv_det;
                
                Ok(result)
            }
            _ => Err(ModelError::UnsupportedOperation(
                "Matrix inversion for matrices larger than 3x3 requires BLAS. Please enable the BLAS feature.".to_string()
            )),
        }
    }

    /// A simple eigenvalue decomposition fallback when BLAS is not available
    /// This is a very basic implementation and should only be used when BLAS is not available
    pub fn eigh(a: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>), ModelError> {
        let (rows, cols) = a.dim();
        if rows != cols {
            return Err(ModelError::InvalidDimension(format!(
                "Matrix must be square for eigendecomposition, got: ({}, {})",
                rows, cols
            )));
        }

        // For simplicity, we'll only implement 2x2 matrix eigendecomposition
        // For larger matrices, we'll return an error suggesting to use BLAS
        match rows {
            2 => {
                let a11 = a[[0, 0]];
                let a12 = a[[0, 1]];
                let a21 = a[[1, 0]];
                let a22 = a[[1, 1]];
                
                // Check if the matrix is symmetric (within numerical precision)
                if (a12 - a21).abs() > 1e-10 {
                    return Err(ModelError::InvalidInput("Matrix must be symmetric for eigendecomposition".to_string()));
                }
                
                // Calculate eigenvalues
                let trace = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                
                let discriminant = trace * trace - 4.0 * det;
                if discriminant < 0.0 {
                    return Err(ModelError::NumericalError("Complex eigenvalues not supported in fallback mode".to_string()));
                }
                
                let sqrt_discriminant = discriminant.sqrt();
                let lambda1 = (trace + sqrt_discriminant) / 2.0;
                let lambda2 = (trace - sqrt_discriminant) / 2.0;
                
                // Calculate eigenvectors
                let mut eigenvectors = Array2::zeros((2, 2));
                
                // First eigenvector
                if a12.abs() > 1e-10 {
                    let v1 = [a12, lambda1 - a11];
                    let norm = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
                    eigenvectors[[0, 0]] = v1[0] / norm;
                    eigenvectors[[1, 0]] = v1[1] / norm;
                } else if (a11 - lambda1).abs() > 1e-10 {
                    eigenvectors[[0, 0]] = 0.0;
                    eigenvectors[[1, 0]] = 1.0;
                } else {
                    eigenvectors[[0, 0]] = 1.0;
                    eigenvectors[[1, 0]] = 0.0;
                }
                
                // Second eigenvector
                if a12.abs() > 1e-10 {
                    let v2 = [a12, lambda2 - a11];
                    let norm = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
                    eigenvectors[[0, 1]] = v2[0] / norm;
                    eigenvectors[[1, 1]] = v2[1] / norm;
                } else if (a11 - lambda2).abs() > 1e-10 {
                    eigenvectors[[0, 1]] = 0.0;
                    eigenvectors[[1, 1]] = 1.0;
                } else {
                    eigenvectors[[0, 1]] = 1.0;
                    eigenvectors[[1, 1]] = 0.0;
                }
                
                let eigenvalues = Array2::from_diag(&ndarray::Array1::from(vec![lambda1, lambda2]));
                
                Ok((eigenvalues, eigenvectors))
            }
            _ => Err(ModelError::UnsupportedOperation(
                "Eigendecomposition for matrices larger than 2x2 requires BLAS. Please enable the BLAS feature.".to_string()
            )),
        }
    }
}