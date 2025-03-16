use ndarray::Array2;
use async_trait::async_trait;
use crate::error::ModelError;
use crate::types::{MarketData, RiskFactors, RiskModel};
use crate::gpu::{ComputeDevice, GPUConfig, compute_covariance};
use crate::factor_analysis::{FactorAnalyzer, FactorQualityMetrics};
use crate::gpu_transformer_risk_model::GPUTransformerRiskModel;
use crate::transformer::TransformerConfig;
use crate::model::DeepRiskModel;

/// GPU-accelerated deep learning model for risk factor generation and covariance estimation.
/// 
/// This model combines GPU-accelerated transformer-based deep learning with factor analysis to:
/// 1. Generate risk factors from market data with GPU acceleration
/// 2. Analyze factor quality and significance
/// 3. Estimate covariance matrices for risk assessment using GPU
/// 
/// The model leverages CUDA for high-performance matrix operations, providing
/// significant speedups for large datasets compared to CPU-only implementations.
/// 
/// # Example
/// 
/// ```rust,no_run
/// use deep_risk_model::prelude::{GPUDeepRiskModel, RiskModel, MarketData};
/// use ndarray::Array2;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create model with 10 assets and 5 risk factors
///     let mut model = GPUDeepRiskModel::new(10, 5, None)?;
///     
///     // Generate sample data
///     let returns = Array2::zeros((100, 10));
///     let features = Array2::zeros((100, 10));
///     let data = MarketData::new(returns, features);
///     
///     // Train model
///     model.train(&data).await?;
///     
///     // Generate risk factors and their covariance
///     let risk_factors = model.generate_risk_factors(&data).await?;
///     let factors = risk_factors.factors();
///     let factor_cov = risk_factors.covariance();
///     
///     // Estimate asset covariance
///     let asset_cov = model.estimate_covariance(&data).await?;
///     
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct GPUDeepRiskModel {
    n_assets: usize,
    n_factors: usize,
    transformer: GPUTransformerRiskModel,
    factor_analyzer: FactorAnalyzer,
    gpu_config: GPUConfig,
}

impl GPUDeepRiskModel {
    /// Create a new GPU-accelerated deep risk model with default transformer configuration
    /// 
    /// # Arguments
    /// 
    /// * `n_assets` - Number of assets
    /// * `n_factors` - Number of risk factors
    /// * `gpu_config` - GPU configuration (optional, uses default if not provided)
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, ModelError>` - New GPU deep risk model or error if initialization fails
    pub fn new(
        n_assets: usize,
        n_factors: usize,
        gpu_config: Option<GPUConfig>,
    ) -> Result<Self, ModelError> {
        // Default configuration values
        let d_model = n_assets; // Use n_assets as d_model by default
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        Self::with_config(n_assets, n_factors, d_model, n_heads, d_ff, n_layers, gpu_config)
    }
    
    /// Create a new GPU-accelerated deep risk model with custom transformer dimensions
    /// 
    /// # Arguments
    /// 
    /// * `n_assets` - Number of assets
    /// * `n_factors` - Number of risk factors
    /// * `d_model` - Dimension of the model's hidden state
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Dimension of feed-forward network
    /// * `n_layers` - Number of transformer layers
    /// * `gpu_config` - GPU configuration (optional, uses default if not provided)
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, ModelError>` - New GPU deep risk model or error if initialization fails
    pub fn with_config(
        n_assets: usize,
        n_factors: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        gpu_config: Option<GPUConfig>,
    ) -> Result<Self, ModelError> {
        if d_model < n_assets {
            return Err(ModelError::InvalidInput(
                format!("d_model ({}) must be at least as large as n_assets ({})", d_model, n_assets)
            ));
        }
        
        let transformer = GPUTransformerRiskModel::new(d_model, n_heads, d_ff, n_layers, gpu_config.clone())?;
        
        // Initialize factor analyzer with default parameters
        let factor_analyzer = FactorAnalyzer::new(
            0.1,  // min_explained_variance
            5.0,  // max_vif
            1.96, // significance_level (95% confidence)
        );
        
        Ok(Self {
            n_assets,
            n_factors,
            transformer,
            factor_analyzer,
            gpu_config: gpu_config.unwrap_or_default(),
        })
    }
    
    /// Create a new GPU-accelerated deep risk model with a custom transformer configuration
    /// 
    /// # Arguments
    /// 
    /// * `n_assets` - Number of assets
    /// * `n_factors` - Number of risk factors
    /// * `config` - Transformer configuration
    /// * `gpu_config` - GPU configuration (optional, uses default if not provided)
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, ModelError>` - New GPU deep risk model or error if initialization fails
    pub fn with_transformer_config(
        n_assets: usize,
        n_factors: usize,
        config: TransformerConfig,
        gpu_config: Option<GPUConfig>,
    ) -> Result<Self, ModelError> {
        if config.d_model < n_assets {
            return Err(ModelError::InvalidInput(
                format!("config.d_model ({}) must be at least as large as n_assets ({})", 
                        config.d_model, n_assets)
            ));
        }
        
        let transformer = GPUTransformerRiskModel::new(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers,
            gpu_config.clone(),
        )?;
        
        // Initialize factor analyzer with default parameters
        let factor_analyzer = FactorAnalyzer::new(
            0.1,  // min_explained_variance
            5.0,  // max_vif
            1.96, // significance_level (95% confidence)
        );
        
        Ok(Self {
            n_assets,
            n_factors,
            transformer,
            factor_analyzer,
            gpu_config: gpu_config.unwrap_or_default(),
        })
    }

    /// Set the GPU configuration for this model
    pub fn set_gpu_config(&mut self, gpu_config: GPUConfig) {
        self.gpu_config = gpu_config.clone();
        self.transformer.set_gpu_config(gpu_config);
    }
    
    /// Get the current GPU configuration
    pub fn gpu_config(&self) -> &GPUConfig {
        &self.gpu_config
    }
    
    /// Check if the model is using GPU acceleration
    pub fn is_using_gpu(&self) -> bool {
        self.gpu_config.device == ComputeDevice::GPU
    }

    /// Get factor quality metrics with GPU acceleration
    pub async fn get_factor_metrics(&self, data: &MarketData) -> Result<Vec<FactorQualityMetrics>, ModelError> {
        let risk_factors = self.transformer.generate_risk_factors(data).await?;
        self.factor_analyzer.calculate_metrics(
            risk_factors.factors(),
            data.returns(),
        )
    }
}

#[async_trait]
impl RiskModel for GPUDeepRiskModel {
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
        // Validate input dimensions
        if data.returns().shape()[1] != self.n_assets {
            return Err(ModelError::DimensionMismatch(
                "Returns must have n_assets columns".into()
            ));
        }
        
        Ok(())
    }
    
    async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
        if data.features().shape()[1] != self.n_assets {
            return Err(ModelError::DimensionMismatch(
                "Market data must have n_assets columns".into()
            ));
        }
        
        // Generate initial factors using GPU-accelerated transformer
        let transformer_factors = self.transformer.generate_risk_factors(data).await?;
        let mut factors = transformer_factors.factors().to_owned();
        
        // Orthogonalize factors
        self.factor_analyzer.orthogonalize_factors(&mut factors)?;
        
        // Calculate factor metrics
        let metrics = self.factor_analyzer.calculate_metrics(&factors, data.returns())?;
        
        // Select optimal factors
        let selected_factors = self.factor_analyzer.select_optimal_factors(&factors, &metrics)?;
        
        // Compute reduced covariance matrix using GPU acceleration
        let covariance = compute_covariance(&selected_factors.view(), &self.gpu_config)?;
        
        Ok(RiskFactors::new(selected_factors, covariance))
    }
    
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        if data.features().shape()[1] != self.n_assets {
            return Err(ModelError::DimensionMismatch(
                "Market data must have n_assets columns".into()
            ));
        }
        
        // Generate risk factors first
        let risk_factors = self.generate_risk_factors(data).await?;
        let factors = risk_factors.factors();
        let n_samples = factors.shape()[0];
        let n_factors = factors.shape()[1];
        
        // Compute factor loadings using regression
        let mut factor_loadings = Array2::zeros((self.n_assets, n_factors));
        
        // Estimate factor loadings using regression
        for i in 0..self.n_assets {
            let returns_i = data.returns().slice(ndarray::s![.., i]);
            for j in 0..n_factors {
                let factor_j = factors.slice(ndarray::s![.., j]);
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;
                for k in 0..n_samples {
                    sum_xy += factor_j[k] * returns_i[k];
                    sum_xx += factor_j[k] * factor_j[k];
                }
                factor_loadings[[i, j]] = if sum_xx > 1e-10 { sum_xy / sum_xx } else { 0.0 };
            }
        }
        
        // Compute covariance using factor model and GPU acceleration
        // In a full implementation, we would use matrix_multiply for this operation
        let mut covariance = Array2::zeros((self.n_assets, self.n_assets));
        for i in 0..self.n_assets {
            for j in 0..self.n_assets {
                let mut cov = 0.0;
                for k in 0..n_factors {
                    cov += factor_loadings[[i, k]] * factor_loadings[[j, k]];
                }
                covariance[[i, j]] = cov;
            }
        }
        
        Ok(covariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DeepRiskModel;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;
    use std::time::Instant;

    #[tokio::test]
    async fn test_gpu_factor_generation() -> Result<(), ModelError> {
        // Skip this test when no-blas feature is enabled
        #[cfg(feature = "no-blas")]
        {
            println!("Skipping test_gpu_factor_generation in no-blas mode");
            return Ok(());
        }
        
        #[cfg(not(feature = "no-blas"))]
        {
            let model = GPUDeepRiskModel::new(64, 5, None)?;
            let features = Array::random((100, 64), StandardNormal);
            let returns = Array::random((100, 64), StandardNormal);
            let data = MarketData::new(returns, features);
            
            let factors = model.generate_risk_factors(&data).await?;
            assert!(factors.factors().shape()[1] <= 5); // Should have at most n_factors columns
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_factor_metrics() -> Result<(), ModelError> {
        // Skip this test when no-blas feature is enabled
        #[cfg(feature = "no-blas")]
        {
            println!("Skipping test_gpu_factor_metrics in no-blas mode");
            return Ok(());
        }
        
        #[cfg(not(feature = "no-blas"))]
        {
            let model = GPUDeepRiskModel::new(64, 5, None)?;
            let features = Array::random((100, 64), StandardNormal);
            let returns = Array::random((100, 64), StandardNormal);
            let data = MarketData::new(returns, features);
            
            let metrics = model.get_factor_metrics(&data).await?;
            assert!(!metrics.is_empty());
            
            for metric in metrics {
                assert!(metric.information_coefficient.abs() <= 1.0);
                assert!(metric.vif >= 1.0);
                assert!(metric.explained_variance >= 0.0 && metric.explained_variance <= 1.0);
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_vs_cpu_performance() -> Result<(), ModelError> {
        // Skip this test when no-blas feature is enabled
        #[cfg(feature = "no-blas")]
        {
            println!("Skipping test_gpu_vs_cpu_performance in no-blas mode");
            return Ok(());
        }
        
        #[cfg(not(feature = "no-blas"))]
        {
            let n_assets = 100;
            let n_factors = 10;
            let n_samples = 1000;
            
            // Create models
            let gpu_model = GPUDeepRiskModel::new(n_assets, n_factors, None)?;
            let cpu_model = DeepRiskModel::new(n_assets, n_factors)?;
            
            // Generate test data
            let features = Array::random((n_samples, n_assets), StandardNormal);
            let returns = Array::random((n_samples, n_assets), StandardNormal);
            let data = MarketData::new(returns, features);
            
            // Measure GPU performance
            let gpu_start = Instant::now();
            let _gpu_factors = gpu_model.generate_risk_factors(&data).await?;
            let gpu_duration = gpu_start.elapsed();
            
            // Measure CPU performance
            let cpu_start = Instant::now();
            let _cpu_factors = cpu_model.generate_risk_factors(&data).await?;
            let cpu_duration = cpu_start.elapsed();
            
            println!("GPU time: {:?}, CPU time: {:?}", gpu_duration, cpu_duration);
            
            // We don't assert that GPU is faster because it depends on hardware
            // Just make sure both complete successfully
        }
        
        Ok(())
    }
} 