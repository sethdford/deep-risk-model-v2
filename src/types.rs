use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::error::ModelError;
use async_trait::async_trait;

/// Container for risk factors and their covariance matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactors {
    factors: Array2<f32>,
    covariance: Array2<f32>,
}

impl RiskFactors {
    /// Creates a new RiskFactors instance
    pub fn new(factors: Array2<f32>, covariance: Array2<f32>) -> Self {
        Self { factors, covariance }
    }

    /// Returns the risk factors matrix
    pub fn factors(&self) -> &Array2<f32> {
        &self.factors
    }

    /// Returns the covariance matrix
    pub fn covariance(&self) -> &Array2<f32> {
        &self.covariance
    }
}

/// Core trait for risk modeling components.
/// 
/// This trait defines the interface that all risk models must implement. It provides
/// methods for training, generating risk factors, and estimating covariance matrices.
/// 
/// Risk models are responsible for:
/// 1. Learning risk factors from market data
/// 2. Generating risk factors for new data
/// 3. Estimating covariance matrices for risk assessment
/// 4. Providing quality metrics for risk factors
/// 
/// # Example
/// 
/// ```rust,no_run
/// use deep_risk_model::{RiskModel, types::{MarketData, RiskFactors}, error::ModelError};
/// use ndarray::Array2;
/// 
/// struct SimpleRiskModel {
///     factors: Array2<f32>,
/// }
/// 
/// #[async_trait::async_trait]
/// impl RiskModel for SimpleRiskModel {
///     async fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
///         // Training logic here
///         Ok(())
///     }
///     
///     async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
///         // Generate risk factors
///         let factors = self.factors.clone();
///         let covariance = Array2::eye(factors.shape()[1]);
///         Ok(RiskFactors::new(factors, covariance))
///     }
///     
///     async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
///         // Estimate covariance matrix
///         Ok(Array2::eye(self.factors.shape()[1]))
///     }
/// }
/// ```
#[async_trait]
pub trait RiskModel: Send + Sync {
    /// Trains the risk model using historical market data.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Market data containing historical asset returns and features
    /// 
    /// # Returns
    /// 
    /// * `Result<(), ModelError>` - Success or error during training
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError>;

    /// Generates risk factors from market data.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Market data containing current asset returns and features
    /// 
    /// # Returns
    /// 
    /// * `Result<RiskFactors, ModelError>` - Generated risk factors and their covariance matrix
    async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError>;

    /// Estimates covariance matrix from market data.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Market data containing current asset returns and features
    /// 
    /// # Returns
    /// 
    /// * `Result<Array2<f32>, ModelError>` - Covariance matrix with shape (n_factors, n_factors)
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError>;
}

/// Container for market data used in risk modeling
#[derive(Debug, Clone)]
pub struct MarketData {
    returns: Array2<f32>,
    features: Array2<f32>,
}

impl MarketData {
    /// Creates a new MarketData instance
    pub fn new(returns: Array2<f32>, features: Array2<f32>) -> Self {
        Self { returns, features }
    }

    /// Returns the asset returns matrix
    pub fn returns(&self) -> &Array2<f32> {
        &self.returns
    }

    /// Returns the feature matrix
    pub fn features(&self) -> &Array2<f32> {
        &self.features
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    api_key: String,
    base_url: String,
}

impl MCPConfig {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { api_key, base_url }
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    hidden_size: usize,
    n_heads: usize,
    n_layers: usize,
    dropout: f32,
}

impl ModelConfig {
    pub fn new(hidden_size: usize, n_heads: usize, n_layers: usize, dropout: f32) -> Self {
        Self {
            hidden_size,
            n_heads,
            n_layers,
            dropout,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn dropout(&self) -> f32 {
        self.dropout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_market_data() {
        let features = Array2::zeros((10, 5));
        let returns = Array2::zeros((10, 3));
        let data = MarketData::new(returns, features);
        
        assert_eq!(data.features().shape(), &[10, 5]);
        assert_eq!(data.returns().shape(), &[10, 3]);
    }

    #[test]
    fn test_risk_factors() {
        let factors = Array2::zeros((10, 5));
        let covariance = Array2::zeros((5, 5));
        let risk_factors = RiskFactors::new(factors, covariance);
        
        assert_eq!(risk_factors.factors().shape(), &[10, 5]);
        assert_eq!(risk_factors.covariance().shape(), &[5, 5]);
    }
} 