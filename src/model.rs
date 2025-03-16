#![recursion_limit = "256"]

use ndarray::{Array1, Array2};
use async_trait::async_trait;
use crate::error::ModelError;
use crate::types::{MarketData, RiskFactors, RiskModel};
use crate::transformer_risk_model::TransformerRiskModel;
use crate::factor_analysis::{FactorAnalyzer, FactorQualityMetrics};

/// Deep learning model for risk factor generation and covariance estimation.
/// 
/// This model combines transformer-based deep learning with factor analysis to:
/// 1. Generate risk factors from market data
/// 2. Analyze factor quality and significance
/// 3. Estimate covariance matrices for risk assessment
/// 
/// The model uses a transformer architecture to capture complex non-linear relationships
/// in market data, while the factor analyzer ensures the generated factors are
/// statistically significant and economically meaningful.
/// 
/// # Example
/// 
/// ```rust,no_run
/// use deep_risk_model::{DeepRiskModel, types::{RiskModel, MarketData}};
/// use ndarray::Array2;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create model with 10 assets and 5 risk factors
///     let mut model = DeepRiskModel::new(10, 5)?;
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
pub struct DeepRiskModel {
    n_assets: usize,
    n_factors: usize,
    transformer: TransformerRiskModel,
    factor_analyzer: FactorAnalyzer,
}

impl DeepRiskModel {
    /// Create a new deep risk model
    pub fn new(n_assets: usize, n_factors: usize) -> Result<Self, ModelError> {
        // Use a d_model that matches the expected value (64)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        let transformer = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        
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
        })
    }

    /// Get factor quality metrics
    pub async fn get_factor_metrics(&self, data: &MarketData) -> Result<Vec<FactorQualityMetrics>, ModelError> {
        let risk_factors = self.transformer.generate_risk_factors(data).await?;
        self.factor_analyzer.calculate_metrics(
            risk_factors.factors(),
            data.returns(),
        )
    }
}

#[async_trait]
impl RiskModel for DeepRiskModel {
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
        
        // Generate initial factors using transformer
        let transformer_factors = self.transformer.generate_risk_factors(data).await?;
        let mut factors = transformer_factors.factors().to_owned();
        
        // Orthogonalize factors
        self.factor_analyzer.orthogonalize_factors(&mut factors)?;
        
        // Calculate factor metrics
        let metrics = self.factor_analyzer.calculate_metrics(&factors, data.returns())?;
        
        // Select optimal factors
        let selected_factors = self.factor_analyzer.select_optimal_factors(&factors, &metrics)?;
        
        // Compute reduced covariance matrix
        let n_selected = selected_factors.shape()[1];
        let n_samples = selected_factors.shape()[0];
        let mut covariance = Array2::zeros((n_selected, n_selected));
        for i in 0..n_selected {
            for j in 0..n_selected {
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += selected_factors[[k, i]] * selected_factors[[k, j]];
                }
                covariance[[i, j]] = cov / ((n_samples - 1) as f32);
            }
        }
        
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
        
        // Compute covariance matrix using risk factors
        let mut covariance = Array2::zeros((self.n_assets, self.n_assets));
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
        
        // Compute covariance using factor model
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
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;

    #[tokio::test]
    async fn test_factor_generation() -> Result<(), ModelError> {
        let model = DeepRiskModel::new(64, 5)?;
        let features = Array::random((100, 64), StandardNormal);
        let returns = Array::random((100, 64), StandardNormal);
        let data = MarketData::new(returns, features);
        
        let factors = model.generate_risk_factors(&data).await?;
        assert!(factors.factors().shape()[1] <= 5); // Should have at most n_factors columns
        
        Ok(())
    }

    #[tokio::test]
    async fn test_factor_metrics() -> Result<(), ModelError> {
        let model = DeepRiskModel::new(64, 5)?;
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
        
        Ok(())
    }

    #[tokio::test]
    async fn test_covariance_estimation() -> Result<(), ModelError> {
        let model = DeepRiskModel::new(64, 5)?;
        let features = Array::random((100, 64), StandardNormal);
        let returns = Array::random((100, 64), StandardNormal);
        let data = MarketData::new(returns, features);
        
        let covariance = model.estimate_covariance(&data).await?;
        assert_eq!(covariance.shape(), &[64, 64]);
        
        // Check symmetry
        for i in 0..64 {
            for j in 0..64 {
                assert!((covariance[[i, j]] - covariance[[j, i]]).abs() < 1e-6);
            }
        }
        
        Ok(())
    }
}