use ndarray::{Array2, Array3, s};
use async_trait::async_trait;
use crate::error::ModelError;
use crate::transformer::{
    TransformerComponent,
    TransformerConfig,
    TransformerLayer,
    position::PositionalEncoder,
};
use crate::types::{MarketData, RiskFactors, RiskModel};
use crate::transformer::temporal_fusion::{TemporalFusionTransformer, TFTConfig};

/// Risk model based on transformer architecture
#[derive(Debug)]
pub struct TransformerRiskModel {
    layers: Vec<TransformerLayer>,
    pos_encoder: PositionalEncoder,
    config: TransformerConfig,
}

impl TransformerRiskModel {
    /// Create a new transformer-based risk model
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, n_layers: usize) -> Result<Self, ModelError> {
        let config = TransformerConfig {
            d_model,
            n_heads,
            d_ff,
            dropout: 0.1,
            n_layers,
            max_seq_len: 5,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        };
        
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerLayer::new(d_model, d_ff, n_heads)?);
        }
        
        let pos_encoder = PositionalEncoder::new(d_model, config.max_seq_len)?;
        
        Ok(Self {
            layers,
            pos_encoder,
            config,
        })
    }
}

impl TransformerComponent<f32> for TransformerRiskModel {
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error + Send + Sync>> {
        // Add positional encoding
        let mut output = self.pos_encoder.forward(x)?;
        
        // Pass through transformer layers
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        
        Ok(output)
    }
}

#[async_trait]
impl RiskModel for TransformerRiskModel {
    async fn train(&mut self, _data: &MarketData) -> Result<(), ModelError> {
        // The transformer model is self-supervised and doesn't require explicit training
        Ok(())
    }

    async fn generate_risk_factors(&self, data: &MarketData) -> Result<RiskFactors, ModelError> {
        let features = data.features();
        let returns = data.returns();
        
        // Validate input dimensions
        if features.shape()[0] != returns.shape()[0] {
            return Err(ModelError::InvalidDimension(
                "Features and returns must have same number of samples".into(),
            ));
        }
        
        let n_samples = features.shape()[0];
        
        if n_samples < self.config.max_seq_len {
            return Err(ModelError::InvalidDimension(
                format!("Number of samples ({}) must be >= max_seq_len ({})", n_samples, self.config.max_seq_len)
            ));
        }
        
        // Process through transformer
        let processed = self.forward(&features)?;
        
        // Extract features by taking mean of sequence dimension
        let mut factors = Array2::zeros((n_samples - self.config.max_seq_len + 1, self.config.d_model));
        for i in 0..n_samples - self.config.max_seq_len + 1 {
            for j in 0..self.config.d_model {
                let mut sum = 0.0;
                for k in 0..self.config.max_seq_len {
                    sum += processed[[i + k, j]];
                }
                factors[[i, j]] = sum / (self.config.max_seq_len as f32);
            }
        }
        
        // Normalize factors
        let f_mean = factors.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| ModelError::ComputationError("Failed to compute mean".into()))?;
        let f_std = factors.std_axis(ndarray::Axis(0), 0.0);
        
        for i in 0..n_samples - self.config.max_seq_len + 1 {
            for j in 0..self.config.d_model {
                factors[[i, j]] = if f_std[j] > 1e-10 {
                    (factors[[i, j]] - f_mean[j]) / f_std[j]
                } else {
                    0.0
                };
            }
        }
        
        // Compute covariance matrix
        let mut covariance = Array2::zeros((self.config.d_model, self.config.d_model));
        for i in 0..self.config.d_model {
            for j in 0..self.config.d_model {
                let mut cov = 0.0;
                for k in 0..n_samples - self.config.max_seq_len + 1 {
                    cov += factors[[k, i]] * factors[[k, j]];
                }
                covariance[[i, j]] = cov / ((n_samples - self.config.max_seq_len) as f32);
            }
        }
        
        Ok(RiskFactors::new(factors, covariance))
    }

    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        let risk_factors = self.generate_risk_factors(data).await?;
        Ok(risk_factors.covariance().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    
    #[test]
    fn test_transformer_risk_model() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let n_layers = 2;
        
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        
        let batch_size = 2;
        let seq_len = 10;
        let input = Array2::random((batch_size * seq_len, d_model), Normal::new(0.0, 1.0)?);
        
        let output = model.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size * seq_len, d_model]);
        
        Ok(())
    }
} 