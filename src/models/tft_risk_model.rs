use ndarray::{Array2, Array3, Axis, s};
use async_trait::async_trait;
use crate::error::ModelError;
use crate::transformer::temporal_fusion::{TemporalFusionTransformer, TFTConfig};
use crate::types::{MarketData, RiskFactors, RiskModel};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

/// Risk model using Temporal Fusion Transformer
#[derive(Debug)]
pub struct TFTRiskModel {
    transformer: TemporalFusionTransformer,
    config: TFTConfig,
}

impl TFTRiskModel {
    /// Create a new TFT risk model
    pub fn new(n_assets: usize, n_factors: usize) -> Result<Self, ModelError> {
        let config = TFTConfig {
            d_model: n_factors,
            n_heads: 8,
            d_ff: n_factors * 4,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 1024,
            num_static_features: n_assets,
            num_temporal_features: n_assets,
            hidden_size: n_factors,
        };
        
        let transformer = TemporalFusionTransformer::new(config.clone())?;
        
        Ok(Self {
            transformer,
            config,
        })
    }
    
    /// Process market data through sliding windows
    async fn process_windows(&self, features: &Array2<f32>) -> Result<Array3<f32>, ModelError> {
        let n_samples = features.shape()[0];
        let n_features = features.shape()[1];
        
        if n_samples < self.config.max_seq_len {
            return Err(ModelError::DimensionMismatch(
                "Not enough samples for window size".into()
            ));
        }
        
        let n_windows = n_samples - self.config.max_seq_len + 1;
        let mut windows = Vec::with_capacity(n_windows);
        
        // Create normalized windows
        for i in 0..n_windows {
            let window = features.slice(s![i..i + self.config.max_seq_len, ..]).to_owned();
            
            // Normalize each window
            let mut normalized = Array2::zeros((self.config.max_seq_len, n_features));
            for j in 0..self.config.max_seq_len {
                let row = window.slice(s![j, ..]);
                let mean = row.mean().unwrap_or(0.0);
                let std = row.std(0.0);
                
                for k in 0..n_features {
                    normalized[[j, k]] = if std > 1e-10 {
                        (window[[j, k]] - mean) / std
                    } else {
                        0.0
                    };
                }
            }
            
            windows.push(normalized);
        }
        
        // Stack windows into batch tensor
        let mut batch_tensor = Array3::zeros((n_windows, self.config.max_seq_len, n_features));
        for (i, window) in windows.iter().enumerate() {
            batch_tensor.slice_mut(s![i, .., ..]).assign(window);
        }
        
        Ok(batch_tensor)
    }
    
    /// Compute volatility features
    fn compute_volatility(&self, returns: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let n_samples = returns.shape()[0];
        let n_assets = returns.shape()[1];
        let window = 21; // 1-month volatility
        
        let mut volatility = Array2::zeros((n_samples, n_assets));
        
        for i in window..n_samples {
            let window_returns = returns.slice(s![i-window..i, ..]);
            let vol = window_returns.std_axis(ndarray::Axis(0), 0.0) * (252.0f32).sqrt(); // Annualized
            volatility.slice_mut(s![i, ..]).assign(&vol);
        }
        
        Ok(volatility)
    }

    fn process_batch(&self, batch_tensor: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let shape = batch_tensor.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_features = shape[2];

        if num_features != self.config.num_static_features * 3 {
            return Err(ModelError::InvalidDimension(
                format!("Expected {} features, got {}", self.config.num_static_features * 3, num_features)
            ));
        }

        // Extract static features (asset-specific features) from all timesteps
        // We take the mean across time to get a single static feature vector per batch
        let static_features = batch_tensor
            .slice(s![.., .., ..self.config.num_static_features])
            .mean_axis(Axis(1))
            .ok_or_else(|| ModelError::InvalidDimension("Failed to compute mean of static features".into()))?;

        // Extract temporal features (returns and volatility) for all timesteps
        // These are already in the correct shape (batch_size, seq_len, num_temporal_features)
        let temporal_features = batch_tensor
            .slice(s![.., .., self.config.num_static_features..])
            .to_owned();

        // Reshape temporal features to 2D for the new API
        let (_, seq_len, n_temporal) = temporal_features.dim();
        let temporal_flat = temporal_features.into_shape((batch_size, seq_len * n_temporal))?;
        
        // Combine static and temporal features
        let mut combined = Array2::zeros((batch_size, self.config.num_static_features + seq_len * n_temporal));
        combined.slice_mut(s![.., ..self.config.num_static_features]).assign(&static_features);
        combined.slice_mut(s![.., self.config.num_static_features..]).assign(&temporal_flat);
        
        // Process through transformer
        let processed = self.transformer.forward(&combined)?;
        
        // Reshape back to 3D for compatibility with existing code
        let output_shape = (batch_size, seq_len, self.config.d_model / seq_len);
        processed.into_shape(output_shape)
            .map_err(|e| ModelError::Shape(e))
    }

    /// Train the model on historical data
    pub fn train(&mut self, returns: &Array2<f32>, factors: &Array3<f32>) -> Result<(), ModelError> {
        let (n_samples, n_assets) = returns.dim();
        let (_, seq_len, n_factors) = factors.dim();
        
        // Reshape factors to match the expected input format for the transformer
        // Combine static and temporal features into a single 2D array
        let mut static_features = Array2::zeros((n_samples, self.config.num_static_features));
        
        // For simplicity, we'll use asset returns as static features
        // In a real application, you might use other asset characteristics
        for i in 0..n_samples {
            for j in 0..n_assets.min(self.config.num_static_features) {
                static_features[[i, j]] = returns[[i, j]];
            }
        }
        
        // Reshape temporal features (factors) to 2D
        let temporal_features_flat = factors.clone().into_shape((n_samples, seq_len * n_factors)).unwrap();
        
        // Combine static and temporal features
        let mut combined_features = Array2::zeros((n_samples, self.config.num_static_features + seq_len * n_factors));
        for i in 0..n_samples {
            // Copy static features
            for j in 0..self.config.num_static_features {
                combined_features[[i, j]] = static_features[[i, j]];
            }
            
            // Copy temporal features
            for j in 0..(seq_len * n_factors) {
                combined_features[[i, self.config.num_static_features + j]] = temporal_features_flat[[i, j]];
            }
        }
        
        // Process through transformer
        let _processed = self.transformer.forward(&combined_features)?;
        
        // In a real implementation, you would update model parameters based on some loss function
        // For this example, we'll just return success
        Ok(())
    }
}

#[async_trait]
impl RiskModel for TFTRiskModel {
    async fn train(&mut self, data: &MarketData) -> Result<(), ModelError> {
        // Extract features and returns from MarketData
        let returns = data.returns();
        let features = data.features();
        
        // Reshape features to match the expected input format for the transformer
        let (n_samples, n_features) = features.dim();
        
        // Combine static and temporal features into a single 2D array
        let mut combined_features = Array2::zeros((n_samples, self.config.num_static_features + self.config.num_temporal_features));
        
        // Copy features to combined array
        for i in 0..n_samples {
            for j in 0..n_features.min(self.config.num_static_features + self.config.num_temporal_features) {
                combined_features[[i, j]] = features[[i, j]];
            }
        }
        
        // Process through transformer
        let _processed = self.transformer.forward(&combined_features)?;
        
        // In a real implementation, you would update model parameters based on some loss function
        // For this example, we'll just return success
        Ok(())
    }
    
    async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
        // Extract features from MarketData
        let features = data.features();
        
        // Prepare input features
        let (n_samples, n_features) = features.dim();
        
        // Combine features into a single 2D array
        let mut combined_features = Array2::zeros((n_samples, self.config.num_static_features + self.config.num_temporal_features));
        
        // Copy features to combined array
        for i in 0..n_samples {
            for j in 0..n_features.min(self.config.num_static_features + self.config.num_temporal_features) {
                combined_features[[i, j]] = features[[i, j]];
            }
        }
        
        // Process through transformer to get risk factors
        let processed = self.transformer.forward(&combined_features)?;
        
        // Compute covariance matrix
        let n_samples = processed.shape()[0];
        let mut covariance = Array2::zeros((self.config.d_model, self.config.d_model));
        
        // Center the data
        let mean = processed.mean_axis(ndarray::Axis(0)).unwrap();
        let centered = &processed - &mean;
        
        // Compute covariance
        for i in 0..self.config.d_model {
            for j in 0..self.config.d_model {
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += centered[[k, i]] * centered[[k, j]];
                }
                covariance[[i, j]] = cov / ((n_samples - 1) as f32);
            }
        }
        
        Ok(RiskFactors::new(processed, covariance))
    }
    
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        // Call the async trait method directly
        let risk_factors = self.generate_risk_factors(data).await?;
        Ok(risk_factors.covariance().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    
    #[tokio::test]
    async fn test_tft_risk_model() -> Result<(), ModelError> {
        let n_assets = 10;
        let n_factors = 5;
        
        let model = TFTRiskModel::new(n_assets, n_factors)?;
        
        let n_samples = 20;
        let features = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0)?);
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0)?);
        
        let data = MarketData::new(returns, features);
        let risk_factors = model.generate_risk_factors(&data).await?;
        
        assert_eq!(risk_factors.factors().shape(), &[n_samples, model.config.d_model]);
        assert_eq!(risk_factors.covariance().shape(), &[model.config.d_model, model.config.d_model]);
        
        Ok(())
    }
} 