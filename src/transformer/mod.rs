use ndarray::ArrayD;
use ndarray::Array2;
use crate::error::ModelError;
use std::error::Error;

/// Configuration for transformer-based risk models.
/// 
/// This struct defines the architecture and training parameters for transformer models
/// used in risk factor generation and analysis.
/// 
/// # Example
/// 
/// ```rust
/// use deep_risk_model::transformer::TransformerConfig;
/// 
/// let config = TransformerConfig {
///     n_heads: 4,
///     d_model: 64,
///     d_ff: 256,
///     dropout: 0.1,
///     n_layers: 3,
///     max_seq_len: 100,
///     num_static_features: 5,
///     num_temporal_features: 10,
///     hidden_size: 32,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Number of attention heads in multi-head attention
    pub n_heads: usize,
    /// Dimension of the model's hidden state
    pub d_model: usize,
    /// Dimension of the feed-forward network
    pub d_ff: usize,
    /// Dropout rate for regularization
    pub dropout: f32,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of static features
    pub num_static_features: usize,
    /// Number of temporal features
    pub num_temporal_features: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            n_heads: 8,
            d_model: 512,
            d_ff: 2048,
            dropout: 0.1,
            n_layers: 6,
            max_seq_len: 1024,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        }
    }
}

/// Core trait for transformer components.
/// 
/// This trait defines the interface that all transformer components must implement.
/// It provides a method for forward propagation through the transformer network.
pub trait TransformerComponent<T> {
    /// Performs forward propagation through the transformer network.
    /// 
    /// # Arguments
    /// 
    /// * `x` - Input tensor of shape (batch_size, sequence_length, d_model)
    /// 
    /// # Returns
    /// 
    /// * `Result<Array2<T>, ModelError>` - Output tensor after transformer processing
    fn forward(&self, x: &Array2<T>) -> Result<Array2<T>, ModelError>;
}

/// Utility functions for transformer components
pub mod utils;

/// Multi-head attention implementation
pub mod attention;

/// Position encoding implementation
pub mod position;

/// Transformer layer implementation
pub mod layer;

/// Complete transformer model
pub mod model;

/// Temporal fusion transformer implementation
pub mod temporal_fusion;

pub use attention::MultiHeadAttention;
pub use layer::TransformerLayer;
pub use model::Transformer;
pub use temporal_fusion::{TemporalFusionTransformer, TFTConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.d_ff, 2048);
        assert_eq!(config.dropout, 0.1);
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.num_static_features, 5);
        assert_eq!(config.num_temporal_features, 10);
        assert_eq!(config.hidden_size, 32);
    }
    
    #[test]
    fn test_xavier_init() {
        let shape = vec![64, 512];
        let weights = utils::xavier_init(&shape).unwrap();
        assert_eq!(weights.shape(), &shape);
        
        let mean = weights.mean().unwrap();
        let std = weights.std(0.0);
        
        // Check if weights follow expected distribution
        assert!(mean.abs() < 0.1);
        assert!(std > 0.0 && std < 1.0);
    }
} 