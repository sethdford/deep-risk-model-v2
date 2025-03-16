use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};
use std::sync::Arc;
use anyhow::Result;
use rand::thread_rng;
use ndarray::linalg::general_mat_mul;
use std::ops::AddAssign;

use crate::error::ModelError;

/// A Gated Recurrent Unit (GRU) module for processing sequential data.
/// 
/// The GRU is a type of recurrent neural network that uses gating mechanisms to
/// control information flow through the network. It is particularly effective
/// for capturing temporal dependencies in sequential data.
/// 
/// # Fields
/// 
/// * `input_size` - Size of the input features
/// * `hidden_size` - Size of the hidden state
/// * `w_ih` - Input-to-hidden weights for all gates
/// * `w_hh` - Hidden-to-hidden weights for all gates
/// * `b_ih` - Input-to-hidden biases
/// * `b_hh` - Hidden-to-hidden biases
/// 
/// # Example
/// 
/// ```rust
/// use deep_risk_model::gru::GRUModule;
/// use ndarray::Array2;
/// 
/// let input_size = 10;
/// let hidden_size = 20;
/// let gru = GRUModule::new(input_size, hidden_size)?;
/// 
/// // Process a sequence
/// let batch_size = 32;
/// let seq_len = 50;
/// let input = Array3::zeros((batch_size, seq_len, input_size));
/// let output = gru.forward(&input)?;
/// ```
#[derive(Debug, Clone)]
pub struct GRUModule {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Array2<f32>,
    pub w_hh: Array2<f32>,
    pub b_ih: Array2<f32>,
    pub b_hh: Array2<f32>,
}

// Implement Send and Sync for GRUModule
// This is safe because GRUModule only contains primitive types (usize)
// and ndarray::Array2<f32> which are already Send and Sync
unsafe impl Send for GRUModule {}
unsafe impl Sync for GRUModule {}

impl GRUModule {
    /// Create a new GRU module
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, crate::error::ModelError> {
        // Validate input parameters
        if input_size == 0 {
            return Err(crate::error::ModelError::InvalidInput(
                "Input size must be greater than 0".to_string()
            ));
        }
        
        if hidden_size == 0 {
            return Err(crate::error::ModelError::InvalidInput(
                "Hidden size must be greater than 0".to_string()
            ));
        }
        
        let w_ih = Array2::zeros((3 * hidden_size, input_size));
        let w_hh = Array2::zeros((3 * hidden_size, hidden_size));
        let b_ih = Array2::zeros((3 * hidden_size, 1));
        let b_hh = Array2::zeros((3 * hidden_size, 1));
        
        let mut gru = Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        };
        
        gru.init_weights()?;
        Ok(gru)
    }
    
    /// Initialize weights with appropriate values
    pub fn init_weights(&mut self) -> Result<(), crate::error::ModelError> {
        // Initialize weights with small random values
        // For simplicity, we'll use zeros here, but in practice you'd want to use
        // proper initialization like Xavier/Glorot
        self.w_ih.fill(0.1);
        self.w_hh.fill(0.1);
        self.b_ih.fill(0.0);
        self.b_hh.fill(0.0);
        Ok(())
    }
    
    /// Forward pass through the GRU
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, crate::error::ModelError> {
        let batch_size = x.shape()[0];
        let mut h = Array2::zeros((batch_size, self.hidden_size));
        
        // Compute gates
        let gates = self.compute_gates(x, &h)?;
        
        // Update gate
        let z = gates.slice(s![.., 0..self.hidden_size]).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        
        // Reset gate
        let r = gates.slice(s![.., self.hidden_size..2*self.hidden_size]).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        
        // New gate
        let n = gates.slice(s![.., 2*self.hidden_size..]).mapv(|x| x.tanh());
        
        // Update hidden state
        for i in 0..batch_size {
            for j in 0..self.hidden_size {
                h[[i, j]] = z[[i, j]] * h[[i, j]] + (1.0 - z[[i, j]]) * n[[i, j]];
            }
        }
        
        Ok(h)
    }
    
    /// Compute gates for the GRU
    fn compute_gates(&self, x: &Array2<f32>, h: &Array2<f32>) -> Result<Array2<f32>, crate::error::ModelError> {
        let batch_size = x.shape()[0];
        let mut gates = Array2::zeros((batch_size, 3 * self.hidden_size));
        
        // Add bias terms first
        for i in 0..batch_size {
            for j in 0..3*self.hidden_size {
                gates[[i, j]] = self.b_ih[[j, 0]] + self.b_hh[[j, 0]];
            }
        }
        
        // Use matrix multiplication for input and hidden contributions
        // This is much more efficient than nested loops
        for i in 0..batch_size {
            let x_i = x.slice(s![i, ..]);
            let h_i = h.slice(s![i, ..]);
            
            for j in 0..3*self.hidden_size {
                // Input contribution
                for k in 0..self.input_size {
                    gates[[i, j]] += x_i[k] * self.w_ih[[j, k]];
                }
                
                // Hidden state contribution
                for k in 0..self.hidden_size {
                    gates[[i, j]] += h_i[k] * self.w_hh[[j, k]];
                }
            }
        }
        
        Ok(gates)
    }
}

// Helper functions
fn split_gates(gates: &Array2<f32>, hidden_size: usize) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), ModelError> {
    let shape = gates.shape();
    let batch_size = shape[0];
    
    Ok((
        gates.slice(s![.., 0..hidden_size]).to_owned(),
        gates.slice(s![.., hidden_size..2*hidden_size]).to_owned(),
        gates.slice(s![.., 2*hidden_size..3*hidden_size]).to_owned(),
    ))
}

fn sigmoid(x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
    Ok(x.mapv(|v| 1.0 / (1.0 + (-v).exp())))
}

fn tanh(x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
    Ok(x.mapv(|v| v.tanh()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_gru_creation() {
        let gru = GRUModule::new(10, 20);
        assert!(gru.is_ok());

        let gru = GRUModule::new(0, 20);
        assert!(gru.is_err());
    }

    #[test]
    fn test_gru_forward() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let input_size = 10;
        let hidden_size = 20;
        let batch_size = 5;
        
        let gru = GRUModule::new(input_size, hidden_size)?;
        let input = Array2::zeros((batch_size, input_size));
        
        let output = gru.forward(&input)?;
        assert_eq!(output.shape(), &[batch_size, hidden_size]);
        
        Ok(())
    }
}