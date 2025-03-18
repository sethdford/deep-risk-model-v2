use crate::transformer::TransformerConfig;
use crate::error::ModelError;
use ndarray::{Array1, Array2, ArrayD};
use std::error::Error;
use crate::transformer::{TransformerComponent};

/// Multi-head attention implementation
#[derive(Debug)]
pub struct MultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    d_k: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Result<Self, ModelError> {
        let d_k = d_model / n_heads;
        
        let mut attention = Self {
            d_model,
            n_heads,
            d_k,
            w_q: Array2::zeros((d_model, d_model)),
            w_k: Array2::zeros((d_model, d_model)),
            w_v: Array2::zeros((d_model, d_model)),
            w_o: Array2::zeros((d_model, d_model)),
        };
        
        attention.init_weights()?;
        Ok(attention)
    }

    /// Get the model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Get the number of attention heads
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    
    /// Get the dimension of each attention head
    pub fn d_k(&self) -> usize {
        self.d_k
    }
    
    /// Get the query weight matrix
    pub fn w_q(&self) -> &Array2<f32> {
        &self.w_q
    }
    
    /// Get the key weight matrix
    pub fn w_k(&self) -> &Array2<f32> {
        &self.w_k
    }
    
    /// Get the value weight matrix
    pub fn w_v(&self) -> &Array2<f32> {
        &self.w_v
    }
    
    /// Get the output weight matrix
    pub fn w_o(&self) -> &Array2<f32> {
        &self.w_o
    }

    fn init_weights(&mut self) -> Result<(), ModelError> {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, (1.0 / (self.d_k as f32)).sqrt())
            .map_err(|e| ModelError::InitializationError(e.to_string()))?;

        self.w_q = Array2::random_using((self.d_model, self.d_model), normal, &mut rng);
        self.w_k = Array2::random_using((self.d_model, self.d_model), normal, &mut rng);
        self.w_v = Array2::random_using((self.d_model, self.d_model), normal, &mut rng);
        self.w_o = Array2::random_using((self.d_model, self.d_model), normal, &mut rng);

        Ok(())
    }
}

impl TransformerComponent<f32> for MultiHeadAttention {
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let batch_size = x.shape()[0];
        
        // Linear projections
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);
        
        // Scaled dot-product attention
        let qk = q.dot(&k.t()) / (self.d_k as f32).sqrt();
        let attention = qk.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let context = attention.dot(&v);
        
        // Output projection
        let output = context.dot(&self.w_o);
        
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
    fn test_attention() -> Result<(), Box<dyn Error + Send + Sync>> {
        let d_model = 64;
        let n_heads = 8;
        let batch_size = 16;
        
        let attention = MultiHeadAttention::new(d_model, n_heads)?;
        let input = Array2::zeros((batch_size, d_model));
        
        let output = attention.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, d_model]);
        
        Ok(())
    }
} 