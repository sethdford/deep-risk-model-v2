use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use ndarray::linalg::general_mat_mul;
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform}};
use std::ops::AddAssign;
use crate::error::ModelError;

/// Graph Attention Network (GAT) module for learning asset relationships.
/// 
/// The GAT module implements a graph neural network that uses attention mechanisms
/// to learn and model relationships between assets in a financial portfolio. It is
/// particularly effective at capturing market structure and cross-asset dependencies.
/// 
/// Key features:
/// - Multi-head attention for robust relationship learning
/// - Learnable attention weights for each asset pair
/// - Nonlinear transformations of node features
/// - Residual connections for stable training
/// 
/// # Example
/// 
/// ```rust
/// use deep_risk_model::gat::GATModule;
/// use ndarray::Array2;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let gat = GATModule::new(64, 32, 4, 0.1)?;  // in_features=64, out_features=32, n_heads=4
/// let features = Array2::zeros((100, 64));     // 100 assets with 64 features each
/// let adj_matrix = Array2::ones((100, 100));   // Fully connected graph
/// let output = gat.forward(&features, &adj_matrix)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GATModule {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Dropout rate for regularization
    dropout: f32,
    /// Linear transformation for input features: W
    weight: Array2<f32>,
    /// Attention mechanism parameters: a
    attention: Array2<f32>,
}

impl GATModule {
    /// Creates a new GAT module with specified dimensions.
    /// 
    /// # Arguments
    /// 
    /// * `in_features` - Number of input features per node
    /// * `out_features` - Number of output features per node
    /// * `n_heads` - Number of attention heads
    /// * `dropout` - Dropout rate for regularization
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, ModelError>` - New GAT module or error if dimensions are invalid
    pub fn new(
        in_features: usize,
        out_features: usize,
        n_heads: usize,
        dropout: f32,
    ) -> Result<Self, ModelError> {
        if dropout < 0.0 || dropout > 1.0 {
            return Err(ModelError::InvalidDimension(format!(
                "Dropout rate must be between 0 and 1, got {}", dropout
            )));
        }

        // Initialize weights with Xavier/Glorot initialization
        let normal = Normal::new(0.0, (2.0 / (in_features + out_features) as f32).sqrt()).unwrap();
        let weight = Array2::random((out_features, in_features), normal);
        
        // Initialize attention parameters
        let attention = Array2::random((n_heads, 2 * out_features), normal);

        Ok(Self {
            in_features,
            out_features,
            n_heads,
            dropout,
            weight,
            attention,
        })
    }

    /// Performs forward pass through the GAT network.
    /// 
    /// # Arguments
    /// 
    /// * `x` - Node features of shape (n_nodes, in_features)
    /// * `adj` - Adjacency matrix of shape (n_nodes, n_nodes)
    /// 
    /// # Returns
    /// 
    /// * `Result<Array2<f32>, ModelError>` - Updated node features of shape (n_nodes, out_features)
    pub fn forward(&self, x: &Array2<f32>, adj: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let (n_nodes, in_dim) = x.dim();
        if in_dim != self.in_features {
            return Err(ModelError::InvalidDimension(format!(
                "Expected input dimension {}, got {}",
                self.in_features, in_dim
            )));
        }

        // Linear transformation
        let h = x.dot(&self.weight.t());
        
        // Multi-head attention
        let mut outputs = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let attention_head = self.attention.slice(s![head, ..]);
            let alpha = self.compute_attention_weights(&h, &attention_head, adj)?;
            let head_output = self.apply_attention(&h, &alpha)?;
            outputs.push(head_output);
        }
        
        // Combine attention heads
        let mut output = Array2::zeros((n_nodes, self.out_features));
        for head_output in outputs {
            output += &head_output;
        }
        output /= self.n_heads as f32;
        
        // Apply dropout during training
        if self.dropout > 0.0 {
            output = self.apply_dropout(output);
        }
        
        Ok(output)
    }

    /// Computes attention weights between nodes.
    fn compute_attention_weights(
        &self,
        h: &Array2<f32>,
        attention_params: &ArrayView1<f32>,
        adj: &Array2<f32>,
    ) -> Result<Array2<f32>, ModelError> {
        let n_nodes = h.shape()[0];
        let mut alpha = Array2::zeros((n_nodes, n_nodes));
        
        // Compute attention coefficients
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if adj[[i, j]] > 0.0 {
                    let a1 = h.slice(s![i, ..]).dot(&attention_params.slice(s![..self.out_features]));
                    let a2 = h.slice(s![j, ..]).dot(&attention_params.slice(s![self.out_features..]));
                    alpha[[i, j]] = self.leaky_relu(a1 + a2);
                }
            }
        }
        
        // Normalize attention weights
        for i in 0..n_nodes {
            let row_sum: f32 = alpha.slice(s![i, ..]).sum();
            if row_sum > 0.0 {
                alpha.slice_mut(s![i, ..]).mapv_inplace(|x| x / row_sum);
            }
        }
        
        Ok(alpha)
    }

    /// Applies attention weights to node features.
    fn apply_attention(&self, h: &Array2<f32>, alpha: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let n_nodes = h.shape()[0];
        let mut output = Array2::zeros((n_nodes, self.out_features));
        
        for i in 0..n_nodes {
            let mut weighted_sum = Array1::zeros(self.out_features);
            for j in 0..n_nodes {
                let weight = alpha[[i, j]];
                weighted_sum += &(h.slice(s![j, ..]).to_owned() * weight);
            }
            output.slice_mut(s![i, ..]).assign(&weighted_sum);
        }
        
        Ok(output)
    }

    /// Applies dropout to the output features.
    fn apply_dropout(&self, mut x: Array2<f32>) -> Array2<f32> {
        let mask = Array2::random(x.dim(), Uniform::new(0.0, 1.0));
        x.zip_mut_with(&mask, |v, &m| {
            if m < self.dropout {
                *v = 0.0;
            } else {
                *v /= 1.0 - self.dropout;
            }
        });
        x
    }

    /// Applies leaky ReLU activation function.
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gat_creation() -> Result<(), ModelError> {
        let gat = GATModule::new(64, 32, 4, 0.1)?;
        assert_eq!(gat.in_features, 64);
        assert_eq!(gat.out_features, 32);
        assert_eq!(gat.n_heads, 4);
        assert!((gat.dropout - 0.1).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_gat_forward() -> Result<(), ModelError> {
        let gat = GATModule::new(64, 32, 4, 0.1)?;
        let features = Array2::zeros((100, 64));
        let adj_matrix = Array2::ones((100, 100));
        let output = gat.forward(&features, &adj_matrix)?;
        assert_eq!(output.shape(), &[100, 32]);
        Ok(())
    }

    #[test]
    fn test_invalid_dropout() {
        assert!(GATModule::new(64, 32, 4, 1.5).is_err());
        assert!(GATModule::new(64, 32, 4, -0.1).is_err());
    }
}