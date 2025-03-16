use ndarray::Array2;
use crate::error::ModelError;
use crate::transformer::{TransformerComponent, TransformerConfig, TransformerLayer};
use std::error::Error;

/// Main transformer model
#[derive(Debug)]
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    config: TransformerConfig,
}

impl Transformer {
    /// Create a new transformer model
    pub fn new(config: TransformerConfig) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(config.n_layers);
        
        for _ in 0..config.n_layers {
            layers.push(TransformerLayer::new(config.d_model, config.d_ff, config.n_heads)?);
        }
        
        Ok(Self { layers, config })
    }
}

impl TransformerComponent<f32> for Transformer {
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        let mut output = x.clone();
        
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    
    #[test]
    fn test_transformer() -> Result<(), Box<dyn Error + Send + Sync>> {
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let n_layers = 2;
        
        let config = TransformerConfig {
            d_model,
            n_heads,
            d_ff,
            dropout: 0.1,
            max_seq_len: 100,
            n_layers,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        };
        
        let transformer = Transformer::new(config)?;
        
        let batch_size = 2;
        let seq_len = 10;
        
        let x = Array2::random((batch_size * seq_len, d_model), Normal::new(0.0, 1.0)?);
        let output = transformer.forward(&x)?;
        
        assert_eq!(output.shape(), &[batch_size * seq_len, d_model]);
        Ok(())
    }
} 