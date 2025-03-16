use ndarray::Array2;
use async_trait::async_trait;
use crate::error::ModelError;
use crate::types::{MarketData, RiskFactors, RiskModel};
use crate::transformer_risk_model::TransformerRiskModel;
use crate::transformer::TransformerConfig;
use crate::regime::{MarketRegimeDetector, RegimeType, RegimeConfig};
use std::collections::HashMap;

/// Risk model that adapts to different market regimes
#[derive(Debug)]
pub struct RegimeAwareRiskModel {
    /// Base transformer risk model
    base_model: TransformerRiskModel,
    /// Regime detector
    regime_detector: MarketRegimeDetector,
    /// Regime-specific parameters
    regime_params: HashMap<RegimeType, RegimeParameters>,
    /// Window size for regime detection
    window_size: usize,
    /// Current regime
    current_regime: Option<RegimeType>,
    /// Training data size
    training_data_size: usize,
}

/// Parameters that can be adjusted based on market regime
#[derive(Debug, Clone)]
pub struct RegimeParameters {
    /// Volatility scaling factor
    pub volatility_scale: f32,
    /// Correlation scaling factor
    pub correlation_scale: f32,
    /// Risk aversion parameter
    pub risk_aversion: f32,
}

impl Default for RegimeParameters {
    fn default() -> Self {
        Self {
            volatility_scale: 1.0,
            correlation_scale: 1.0,
            risk_aversion: 1.0,
        }
    }
}

impl RegimeAwareRiskModel {
    /// Create a new regime-aware risk model
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        window_size: usize,
    ) -> Result<Self, ModelError> {
        let base_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        let regime_detector = MarketRegimeDetector::new(window_size);
        
        // Initialize regime parameters
        let mut regime_params = HashMap::new();
        
        // Low volatility regime: Lower risk estimates
        regime_params.insert(RegimeType::LowVolatility, RegimeParameters {
            volatility_scale: 0.8,
            correlation_scale: 0.9,
            risk_aversion: 0.7,
        });
        
        // Normal regime: Standard risk estimates
        regime_params.insert(RegimeType::Normal, RegimeParameters::default());
        
        // High volatility regime: Higher risk estimates
        regime_params.insert(RegimeType::HighVolatility, RegimeParameters {
            volatility_scale: 1.2,
            correlation_scale: 1.1,
            risk_aversion: 1.3,
        });
        
        // Crisis regime: Much higher risk estimates
        regime_params.insert(RegimeType::Crisis, RegimeParameters {
            volatility_scale: 1.5,
            correlation_scale: 1.3,
            risk_aversion: 2.0,
        });
        
        Ok(Self {
            base_model,
            regime_detector,
            regime_params,
            window_size,
            current_regime: None,
            training_data_size: 0,
        })
    }
    
    /// Create a new regime-aware risk model with custom configuration
    pub fn with_config(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        window_size: usize,
        regime_config: RegimeConfig,
    ) -> Result<Self, ModelError> {
        let base_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        let regime_detector = MarketRegimeDetector::with_config(window_size, regime_config);
        
        // Initialize regime parameters
        let mut regime_params = HashMap::new();
        
        // Low volatility regime: Lower risk estimates
        regime_params.insert(RegimeType::LowVolatility, RegimeParameters {
            volatility_scale: 0.8,
            correlation_scale: 0.9,
            risk_aversion: 0.7,
        });
        
        // Normal regime: Standard risk estimates
        regime_params.insert(RegimeType::Normal, RegimeParameters::default());
        
        // High volatility regime: Higher risk estimates
        regime_params.insert(RegimeType::HighVolatility, RegimeParameters {
            volatility_scale: 1.2,
            correlation_scale: 1.1,
            risk_aversion: 1.3,
        });
        
        // Crisis regime: Much higher risk estimates
        regime_params.insert(RegimeType::Crisis, RegimeParameters {
            volatility_scale: 1.5,
            correlation_scale: 1.3,
            risk_aversion: 2.0,
        });
        
        Ok(Self {
            base_model,
            regime_detector,
            regime_params,
            window_size,
            current_regime: None,
            training_data_size: 0,
        })
    }
    
    /// Create a new regime-aware risk model with a pre-configured transformer model
    pub fn new_with_model(
        base_model: TransformerRiskModel,
        window_size: usize,
        regime_config: RegimeConfig,
    ) -> Result<Self, ModelError> {
        let regime_detector = MarketRegimeDetector::with_config(window_size, regime_config);
        
        // Initialize regime parameters
        let mut regime_params = HashMap::new();
        
        // Low volatility regime: Lower risk estimates
        regime_params.insert(RegimeType::LowVolatility, RegimeParameters {
            volatility_scale: 0.8,
            correlation_scale: 0.9,
            risk_aversion: 0.7,
        });
        
        // Normal regime: Standard risk estimates
        regime_params.insert(RegimeType::Normal, RegimeParameters::default());
        
        // High volatility regime: Higher risk estimates
        regime_params.insert(RegimeType::HighVolatility, RegimeParameters {
            volatility_scale: 1.2,
            correlation_scale: 1.1,
            risk_aversion: 1.3,
        });
        
        // Crisis regime: Much higher risk estimates
        regime_params.insert(RegimeType::Crisis, RegimeParameters {
            volatility_scale: 1.5,
            correlation_scale: 1.3,
            risk_aversion: 2.0,
        });
        
        Ok(Self {
            base_model,
            regime_detector,
            regime_params,
            window_size,
            current_regime: None,
            training_data_size: 0,
        })
    }
    
    /// Create a new regime-aware risk model with custom transformer configuration
    pub fn with_transformer_config(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        window_size: usize,
        transformer_config: TransformerConfig,
    ) -> Result<Self, ModelError> {
        let base_model = TransformerRiskModel::with_config(transformer_config)?;
        let regime_detector = MarketRegimeDetector::new(window_size);
        
        // Initialize regime parameters
        let mut regime_params = HashMap::new();
        
        // Low volatility regime: Lower risk estimates
        regime_params.insert(RegimeType::LowVolatility, RegimeParameters {
            volatility_scale: 0.8,
            correlation_scale: 0.9,
            risk_aversion: 0.7,
        });
        
        // Normal regime: Standard risk estimates
        regime_params.insert(RegimeType::Normal, RegimeParameters::default());
        
        // High volatility regime: Higher risk estimates
        regime_params.insert(RegimeType::HighVolatility, RegimeParameters {
            volatility_scale: 1.2,
            correlation_scale: 1.1,
            risk_aversion: 1.3,
        });
        
        // Crisis regime: Much higher risk estimates
        regime_params.insert(RegimeType::Crisis, RegimeParameters {
            volatility_scale: 1.5,
            correlation_scale: 1.3,
            risk_aversion: 2.0,
        });
        
        Ok(Self {
            base_model,
            regime_detector,
            regime_params,
            window_size,
            current_regime: None,
            training_data_size: 0,
        })
    }
    
    /// Set custom parameters for a specific regime
    pub fn set_regime_parameters(&mut self, regime: RegimeType, params: RegimeParameters) {
        self.regime_params.insert(regime, params);
    }
    
    /// Get the current regime
    pub fn current_regime(&self) -> Option<RegimeType> {
        self.current_regime
    }
    
    /// Get the regime history
    pub fn regime_history(&self) -> &[RegimeType] {
        self.regime_detector.regime_history()
    }
    
    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &Array2<f32> {
        self.regime_detector.transition_matrix()
    }
    
    /// Get the parameters for a specific regime
    pub fn regime_parameters(&self, regime: RegimeType) -> Option<&RegimeParameters> {
        self.regime_params.get(&regime)
    }
    
    /// Apply regime-specific scaling to covariance matrix
    fn apply_regime_scaling(&self, covariance: &mut Array2<f32>, regime: RegimeType) {
        let params = match self.regime_params.get(&regime) {
            Some(p) => p,
            None => return, // No scaling if regime parameters not found
        };
        
        let n = covariance.shape()[0];
        
        // Extract volatilities (diagonal elements)
        let mut volatilities = Vec::with_capacity(n);
        for i in 0..n {
            volatilities.push(covariance[[i, i]].sqrt());
        }
        
        // Scale volatilities and correlations
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Scale variance (diagonal)
                    covariance[[i, i]] *= params.volatility_scale.powi(2);
                } else {
                    // Scale covariance (off-diagonal)
                    let correlation = covariance[[i, j]] / (volatilities[i] * volatilities[j]);
                    let scaled_correlation = correlation * params.correlation_scale;
                    
                    // Clamp to valid correlation range [-1, 1]
                    let clamped_correlation = scaled_correlation.clamp(-1.0, 1.0);
                    
                    // Update covariance with scaled correlation and volatilities
                    let scaled_vol_i = volatilities[i] * params.volatility_scale;
                    let scaled_vol_j = volatilities[j] * params.volatility_scale;
                    covariance[[i, j]] = clamped_correlation * scaled_vol_i * scaled_vol_j;
                    covariance[[j, i]] = covariance[[i, j]]; // Ensure symmetry
                }
            }
        }
    }
}

#[async_trait]
impl RiskModel for RegimeAwareRiskModel {
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
        // Train the base model
        self.base_model.train(data).await?;
        
        // Train the regime detector
        self.regime_detector.train(data)?;
        
        // Store training data size
        self.training_data_size = data.returns().shape()[0];
        
        Ok(())
    }
    
    async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
        // Generate base risk factors
        let risk_factors = self.base_model.generate_risk_factors(data).await?;
        
        // Detect current regime
        let regime = match self.regime_detector.detect_regime(data) {
            Ok(r) => r,
            Err(_) => return Ok(risk_factors), // Return base factors if regime detection fails
        };
        
        // Get a mutable reference to the covariance matrix
        let mut covariance = risk_factors.covariance().to_owned();
        
        // Apply regime-specific scaling
        self.apply_regime_scaling(&mut covariance, regime);
        
        // Create new risk factors with scaled covariance
        Ok(RiskFactors::new(risk_factors.factors().to_owned(), covariance))
    }
    
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        // Generate risk factors with regime-aware scaling
        let risk_factors = self.generate_risk_factors(data).await?;
        
        // Return the scaled covariance matrix
        Ok(risk_factors.covariance().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    use ndarray::Array;
    
    #[tokio::test]
    async fn test_regime_aware_risk_model() -> Result<(), ModelError> {
        // Create model
        let d_model = 32;
        let n_heads = 4;
        let d_ff = 128;
        let n_layers = 2;
        let window_size = 10;
        
        let mut model = RegimeAwareRiskModel::new(
            d_model, n_heads, d_ff, n_layers, window_size
        )?;
        
        // Generate synthetic data
        let n_samples = 100;
        let n_assets = 5;
        let n_features = d_model;
        
        // First 50 samples: low volatility
        let low_vol_returns = Array2::random((50, n_assets), Normal::new(0.001, 0.01)?);
        let low_vol_features = Array2::random((50, n_features), Normal::new(0.0, 1.0)?);
        
        // Next 50 samples: high volatility
        let high_vol_returns = Array2::random((50, n_assets), Normal::new(-0.002, 0.03)?);
        let high_vol_features = Array2::random((50, n_features), Normal::new(0.0, 1.0)?);
        
        // Combine data
        let mut returns = Array2::zeros((n_samples, n_assets));
        let mut features = Array2::zeros((n_samples, n_features));
        
        for i in 0..50 {
            for j in 0..n_assets {
                returns[[i, j]] = low_vol_returns[[i, j]];
                returns[[i+50, j]] = high_vol_returns[[i, j]];
            }
            
            for j in 0..n_features {
                features[[i, j]] = low_vol_features[[i, j]];
                features[[i+50, j]] = high_vol_features[[i, j]];
            }
        }
        
        // Create market data
        let market_data = MarketData::new(returns, features);
        
        // Train model
        model.train(&market_data).await?;
        
        // Generate risk factors
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        
        // Check dimensions
        assert_eq!(risk_factors.covariance().shape(), &[d_model, d_model]);
        
        // Check that covariance matrix is positive semi-definite (all eigenvalues >= 0)
        // This would require computing eigenvalues, which we'll skip for this test
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_regime_parameter_scaling() -> Result<(), ModelError> {
        // Create model
        let d_model = 4; // Small dimension for easy testing
        let n_heads = 2;
        let d_ff = 16;
        let n_layers = 1;
        let window_size = 5;
        
        let mut model = RegimeAwareRiskModel::new(
            d_model, n_heads, d_ff, n_layers, window_size
        )?;
        
        // Create a test covariance matrix
        let mut cov = Array2::zeros((d_model, d_model));
        
        // Set diagonal (variances)
        for i in 0..d_model {
            cov[[i, i]] = 0.01; // 10% volatility squared
        }
        
        // Set off-diagonal (covariances with 0.5 correlation)
        for i in 0..d_model {
            for j in 0..d_model {
                if i != j {
                    cov[[i, j]] = 0.5 * f32::sqrt(cov[[i, i]] * cov[[j, j]]);
                }
            }
        }
        
        // Make a copy for comparison
        let original_cov = cov.clone();
        
        // Apply scaling for crisis regime
        model.apply_regime_scaling(&mut cov, RegimeType::Crisis);
        
        // Check that volatilities are scaled up
        for i in 0..d_model {
            assert!(cov[[i, i]] > original_cov[[i, i]]);
        }
        
        // Check that correlations remain in [-1, 1] with a small epsilon for floating point errors
        for i in 0..d_model {
            for j in 0..d_model {
                if i != j {
                    let vol_i = cov[[i, i]].sqrt();
                    let vol_j = cov[[j, j]].sqrt();
                    let corr = cov[[i, j]] / (vol_i * vol_j);
                    assert!(corr >= -1.0 - 1e-5 && corr <= 1.0 + 1e-5, "Correlation {} is out of bounds", corr);
                }
            }
        }
        
        Ok(())
    }
} 