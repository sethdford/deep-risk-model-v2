use ndarray::{s, Array1, Array2, ArrayD, Axis};
use crate::error::ModelError;
use crate::transformer::TransformerComponent;
use crate::transformer::attention::MultiHeadAttention;
use std::error::Error;
use rand_distr::NormalError;

impl From<NormalError> for ModelError {
    fn from(err: NormalError) -> Self {
        ModelError::InitializationError(err.to_string())
    }
}

/// Transformer layer that combines multi-head attention and feed-forward networks
#[derive(Debug)]
pub struct TransformerLayer {
    d_model: usize,
    d_ff: usize,
    attention: MultiHeadAttention,
    w1: Array2<f32>,
    w2: Array2<f32>,
    norm1_scale: Array1<f32>,
    norm1_bias: Array1<f32>,
    norm2_scale: Array1<f32>,
    norm2_bias: Array1<f32>,
}

impl TransformerLayer {
    /// Create a new transformer layer
    pub fn new(d_model: usize, d_ff: usize, n_heads: usize) -> Result<Self, ModelError> {
        let attention = MultiHeadAttention::new(d_model, n_heads)?;
        
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        
        // Initialize feed-forward weights
        let w1_normal = Normal::new(0.0, (2.0 / (d_model as f32)).sqrt())?;
        let w2_normal = Normal::new(0.0, (2.0 / (d_ff as f32)).sqrt())?;
        
        let w1 = Array2::random_using((d_model, d_ff), w1_normal, &mut rng);
        let w2 = Array2::random_using((d_ff, d_model), w2_normal, &mut rng);
        
        let norm1_scale = Array1::ones(d_model);
        let norm1_bias = Array1::zeros(d_model);
        let norm2_scale = Array1::ones(d_model);
        let norm2_bias = Array1::zeros(d_model);

        Ok(Self {
            d_model,
            d_ff,
            attention,
            w1,
            w2,
            norm1_scale,
            norm1_bias,
            norm2_scale,
            norm2_bias,
        })
    }
    
    /// Get the first feed-forward weight matrix
    pub fn w1(&self) -> &Array2<f32> {
        &self.w1
    }
    
    /// Get the second feed-forward weight matrix
    pub fn w2(&self) -> &Array2<f32> {
        &self.w2
    }
    
    /// Get the first layer normalization scale
    pub fn norm1_scale(&self) -> &Array1<f32> {
        &self.norm1_scale
    }
    
    /// Get the first layer normalization bias
    pub fn norm1_bias(&self) -> &Array1<f32> {
        &self.norm1_bias
    }
    
    /// Get the second layer normalization scale
    pub fn norm2_scale(&self) -> &Array1<f32> {
        &self.norm2_scale
    }
    
    /// Get the second layer normalization bias
    pub fn norm2_bias(&self) -> &Array1<f32> {
        &self.norm2_bias
    }
    
    /// Get the attention module
    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }
    
    /// Apply feed-forward network
    fn feed_forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let h = x.dot(&self.w1).mapv(|x| if x > 0.0 { x } else { 0.0 });
        let output = h.dot(&self.w2);
        Ok(output)
    }
    
    /// Apply layer normalization to the input
    fn layer_norm(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let shape = x.shape();
        let batch_size = shape[0];
        let d_model = shape[1];
        
        if d_model != self.d_model {
            return Err(ModelError::InvalidDimension(
                format!("Expected d_model {}, got {}", self.d_model, d_model)
            ));
        }
        
        // Compute mean and variance along the feature dimension
        let mut mean = Array2::zeros((batch_size, 1));
        let mut var = Array2::zeros((batch_size, 1));
        
        for i in 0..batch_size {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            
            for j in 0..d_model {
                let val = x[[i, j]];
                sum += val;
                sum_sq += val * val;
            }
            
            let mean_val = sum / (d_model as f32);
            let var_val = sum_sq / (d_model as f32) - mean_val * mean_val;
            
            mean[[i, 0]] = mean_val;
            var[[i, 0]] = var_val;
        }
        
        // Normalize
        let mut normalized = Array2::zeros((batch_size, d_model));
        
        for i in 0..batch_size {
            for j in 0..d_model {
                normalized[[i, j]] = (x[[i, j]] - mean[[i, 0]]) / (var[[i, 0]] + 1e-5).sqrt();
                normalized[[i, j]] = normalized[[i, j]] * self.norm1_scale[j] + self.norm1_bias[j];
            }
        }
        
        Ok(normalized)
    }
}

impl TransformerComponent<f32> for TransformerLayer {
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        // Layer normalization and attention
        let norm1 = self.layer_norm(x)?;
        let attended = self.attention.forward(&norm1)?;
        let res1 = x + &attended;
        
        // Layer normalization and feed-forward
        let norm2 = self.layer_norm(&res1)?;
        let ff = self.feed_forward(&norm2)?;
        let output = &res1 + &ff;
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_transformer_layer() -> Result<(), Box<dyn Error + Send + Sync>> {
        let d_model = 64;
        let d_ff = 256;
        let n_heads = 8;
        let batch_size = 16;
        
        let layer = TransformerLayer::new(d_model, d_ff, n_heads)?;
        let input = Array2::zeros((batch_size, d_model));
        
        let output = layer.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, d_model]);
        
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<(), ModelError> {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;

        let layer = TransformerLayer::new(64, 256, 8)?;
        let batch_size = 10;
        
        // Use random data instead of ones to get meaningful variance
        let input = Array2::random((batch_size, 64), Normal::new(0.0, 1.0).unwrap());
        let normalized = layer.layer_norm(&input)?;
        
        // Check shape
        assert_eq!(normalized.shape(), input.shape());
        
        // Check mean and variance
        let mut mean_sum = 0.0;
        let mut var_sum = 0.0;
        let mut count = 0;
        
        for b in 0..batch_size {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for d in 0..64 {
                let val = normalized[[b, d]];
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / 64.0;
            let var = sum_sq / 64.0 - mean * mean;
            mean_sum += mean;
            var_sum += var;
            count += 1;
        }
        
        // Check that mean is close to 0 and variance is close to 1
        let avg_mean = mean_sum / (count as f32);
        let avg_var = var_sum / (count as f32);
        
        assert!(avg_mean.abs() < 0.1, "Mean should be close to 0, got {}", avg_mean);
        assert!((avg_var - 1.0).abs() < 0.5, "Variance should be close to 1, got {}", avg_var);
        
        Ok(())
    }
} 