use ndarray::{s, Array1, Array2, ArrayD, Axis};
use crate::error::ModelError;
use crate::transformer::TransformerComponent;
use std::error::Error;
use std::ops::AddAssign;

/// Positional encoder for transformer models
#[derive(Debug)]
pub struct PositionalEncoder {
    d_model: usize,
    max_seq_len: usize,
    encoding: Array2<f32>,
}

impl PositionalEncoder {
    /// Create a new positional encoder
    pub fn new(d_model: usize, max_seq_len: usize) -> Result<Self, ModelError> {
        let mut encoding = Array2::zeros((max_seq_len, d_model));
        
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let div_term = (2.0 * (i / 2) as f32 / d_model as f32).exp();
                if i % 2 == 0 {
                    encoding[[pos, i]] = (pos as f32 / div_term).sin();
                } else {
                    encoding[[pos, i]] = (pos as f32 / div_term).cos();
                }
            }
        }
        
        Ok(Self {
            d_model,
            max_seq_len,
            encoding,
        })
    }
}

impl TransformerComponent<f32> for PositionalEncoder {
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        let shape = x.shape();
        let batch_size = shape[0];
        let d_model = shape[1];
        
        if d_model != self.d_model {
            return Err(Box::new(ModelError::InvalidDimension(
                format!("Expected d_model {}, got {}", self.d_model, d_model)
            )));
        }
        
        // Create output with same shape as input
        let mut output = x.to_owned();
        
        // Add positional encoding to each item in the batch
        // The positional encoding is a 2D array of shape [max_seq_len, d_model]
        // We need to add it to each item in the batch
        for i in 0..batch_size {
            // Get the slice for this batch item
            let mut slice = output.slice_mut(s![i, ..]);
            
            // Add the positional encoding for position 0
            // This assumes each row in the batch is a separate sequence at position 0
            for j in 0..d_model {
                slice[j] += self.encoding[[0, j]];
            }
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_positional_encoder() -> Result<(), Box<dyn Error + Send + Sync>> {
        let d_model = 64;
        let max_seq_len = 100;
        let batch_size = 16;
        let seq_len = 50;
        
        let encoder = PositionalEncoder::new(d_model, max_seq_len)?;
        let input = Array2::zeros((batch_size * seq_len, d_model));
        
        let output = encoder.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size * seq_len, d_model]);
        
        Ok(())
    }
} 