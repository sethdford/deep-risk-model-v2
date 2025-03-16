use ndarray::{s, Array1, Array2, Array3, ArrayD, Axis, Ix2};
use crate::error::ModelError;
use crate::gru::GRUModule;
use crate::transformer::{MultiHeadAttention, TransformerComponent, TransformerConfig};
use crate::utils::xavier_init;
use anyhow::Result as AnyResult;
use ndarray_rand::{RandomExt, rand_distr::Normal};
use super::TransformerConfig as SuperTransformerConfig;
use std::ops::AddAssign;
use ndarray::ShapeError;
use crate::transformer::layer::TransformerLayer;
use rand::thread_rng;

/// Configuration for the Temporal Fusion Transformer
#[derive(Debug, Clone)]
pub struct TFTConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f32,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub num_static_features: usize,
    pub num_temporal_features: usize,
    pub hidden_size: usize,
}

impl Default for TFTConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            n_heads: 8,
            d_ff: 256,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 64,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        }
    }
}

/// Configuration for gradient checkpointing
#[derive(Debug, Clone, Copy)]
pub struct CheckpointConfig {
    /// Whether to enable gradient checkpointing
    pub enabled: bool,
    /// Number of segments to divide the sequence into
    pub num_segments: usize,
    /// Whether to checkpoint the variable selection networks
    pub checkpoint_vsn: bool,
    /// Whether to checkpoint the attention layers
    pub checkpoint_attention: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_segments: 4,
            checkpoint_vsn: true,
            checkpoint_attention: true,
        }
    }
}

/// Temporal Fusion Transformer for processing temporal data with static features
#[derive(Debug)]
pub struct TemporalFusionTransformer {
    config: TFTConfig,
    gru: GRUModule,
    static_encoder: Array2<f32>,
    temporal_encoder: Array2<f32>,
    selection_weights: Array2<f32>,
}

impl TemporalFusionTransformer {
    /// Create a new Temporal Fusion Transformer
    pub fn new(config: TFTConfig) -> Result<Self, ModelError> {
        let input_size = config.num_static_features + config.num_temporal_features;
        let hidden_size = config.hidden_size;
        
        let gru = GRUModule::new(input_size, hidden_size)?;
        
        let static_encoder = Array2::zeros((config.num_static_features, config.d_model));
        let temporal_encoder = Array2::zeros((config.num_temporal_features, config.d_model));
        
        // Selection weights should be (hidden_size, 2) for static and temporal features
        let selection_weights = Array2::zeros((hidden_size, 2));
        
        Ok(Self {
            config,
            gru,
            static_encoder,
            temporal_encoder,
            selection_weights,
        })
    }
    
    /// Process input through the temporal fusion transformer
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let (batch_size, total_features) = x.dim();
        
        // Validate input dimensions
        let expected_features = self.config.num_static_features + self.config.num_temporal_features;
        if total_features != expected_features {
            return Err(ModelError::InvalidDimension(
                format!("Expected {} features, got {}", expected_features, total_features)
            ));
        }
        
        // Split features into static and temporal
        let static_features = x.slice(s![.., 0..self.config.num_static_features]).to_owned();
        let temporal_features = x.slice(s![.., self.config.num_static_features..]).to_owned();
        
        // Process through GRU
        let hidden = self.gru.forward(x)?;
        
        // Apply feature selection
        let mut output = Array2::zeros((batch_size, self.config.d_model));
        for i in 0..batch_size {
            let h = hidden.slice(s![i..i+1, ..]);
            let mut scores = h.dot(&self.selection_weights);
            
            // Apply softmax
            let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            scores.mapv_inplace(|x| (x - max_score).exp());
            let sum = scores.sum();
            scores.mapv_inplace(|x| x / sum);
            
            // Ensure projection matrices have correct dimensions
            // Static encoder should be (num_static_features, d_model)
            // Temporal encoder should be (num_temporal_features, d_model)
            let static_proj = static_features.slice(s![i..i+1, ..]).dot(&self.static_encoder);
            let temporal_proj = temporal_features.slice(s![i..i+1, ..]).dot(&self.temporal_encoder);
            
            output.slice_mut(s![i..i+1, ..]).assign(&(scores[[0, 0]] * static_proj + scores[[0, 1]] * temporal_proj));
        }
        
        Ok(output)
    }
}

/// Variable Selection Network for feature selection
#[derive(Debug)]
pub struct VariableSelectionNetwork {
    /// GRU for processing input features
    gru: GRUModule,
    /// Weights for feature selection
    selection_weights: Array2<f32>,
    /// Bias for feature selection
    selection_bias: Array1<f32>,
}

impl VariableSelectionNetwork {
    /// Create a new Variable Selection Network
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        let gru = GRUModule::new(input_size, hidden_size)?;
        let selection_weights = Array2::zeros((hidden_size, input_size));
        let selection_bias = Array1::zeros(input_size);
        
        let mut vsn = Self {
            gru,
            selection_weights,
            selection_bias,
        };
        vsn.init_weights()?;
        Ok(vsn)
    }
    
    /// Initialize weights
    pub fn init_weights(&mut self) -> Result<(), ModelError> {
        self.selection_weights = xavier_init(self.gru.hidden_size, self.gru.input_size)?;
        Ok(())
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        let (batch_size, num_features) = x.dim();
        
        // Process features through GRU
        let processed = self.gru.forward(x)?;
        
        // Compute selection weights
        let mut selected = Array2::zeros((batch_size, num_features));
        
        for i in 0..batch_size {
            // Get GRU output for this timestep
            let h = processed.slice(s![i..i+1, ..]);
            
            // Compute selection scores
            let mut scores = h.dot(&self.selection_weights);
            scores += &self.selection_bias;
            
            // Apply softmax
            let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            // Convert to 1D array before applying mapv
            let scores_1d = scores.into_shape(num_features).unwrap();
            let exp_scores: Array1<f32> = scores_1d.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();
            let weights = exp_scores / sum_exp;
            
            // Apply selection weights
            for k in 0..num_features {
                selected[[i, k]] = weights[k] * x[[i, k]];
            }
        }
        
        Ok(selected)
    }
}

/// Gating Layer for controlling information flow
#[derive(Debug)]
pub struct GatingLayer {
    /// Input dimension
    input_size: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Gate weights
    gate_weights: Array2<f32>,
    /// Gate bias
    gate_bias: Array1<f32>,
}

impl GatingLayer {
    /// Create a new Gating Layer
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        let gate_weights = Array2::zeros((hidden_size, input_size));
        let gate_bias = Array1::zeros(input_size);
        
        let mut gate = Self {
            input_size,
            hidden_size,
            gate_weights,
            gate_bias,
        };
        gate.init_weights()?;
        Ok(gate)
    }
    
    /// Initialize weights
    pub fn init_weights(&mut self) -> Result<(), ModelError> {
        self.gate_weights = xavier_init(self.hidden_size, self.input_size)?;
        Ok(())
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Array3<f32>, context: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let (batch_size, seq_len, _) = x.dim();
        let mut gated = Array3::zeros(x.dim());
        
        for i in 0..batch_size {
            for j in 0..seq_len {
                // Compute gate values
                let c = context.slice(s![i, j, ..]);
                let mut gate = c.dot(&self.gate_weights);
                gate += &self.gate_bias;
                
                // Apply sigmoid activation
                gate.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                
                // Apply gate
                for k in 0..self.input_size {
                    gated[[i, j, k]] = gate[k] * x[[i, j, k]];
                }
            }
        }
        
        Ok(gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    
    #[test]
    fn test_temporal_fusion_transformer() -> Result<(), ModelError> {
        let config = TFTConfig {
            d_model: 64,
            n_heads: 8,
            d_ff: 256,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 1024,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        };
        
        let transformer = TemporalFusionTransformer::new(config)?;
        
        let batch_size = 2;
        let total_features = transformer.config.num_static_features + transformer.config.num_temporal_features;
        
        let x = Array2::random((batch_size, total_features), Normal::new(0.0, 1.0)?);
        let output = transformer.forward(&x)?;
        
        assert_eq!(output.shape(), &[batch_size, transformer.config.d_model]);
        Ok(())
    }
    
    #[test]
    fn test_variable_selection_network() -> Result<(), ModelError> {
        let input_size = 10;
        let hidden_size = 32;
        let batch_size = 5;
        
        let vsn = VariableSelectionNetwork::new(input_size, hidden_size)?;
        
        let x = Array2::random((batch_size, input_size), Normal::new(0.0, 1.0)?);
        let output = vsn.forward(&x)?;
        
        assert_eq!(output.shape(), &[batch_size, input_size]);
        Ok(())
    }

    #[test]
    fn test_tft_static_enrichment() -> Result<(), ModelError> {
        let config = TFTConfig {
            d_model: 64,
            n_heads: 8,
            d_ff: 256,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 64,
            num_static_features: 3,
            num_temporal_features: 5,
            hidden_size: 16,
        };
        
        let tft = TemporalFusionTransformer::new(config.clone())?;
        
        // Small batch for detailed testing
        let batch_size = 1;
        let total_features = tft.config.num_static_features + tft.config.num_temporal_features;
        
        let x = Array2::ones((batch_size, total_features));
        
        let output = tft.forward(&x)?;
        
        // Check that output values are finite
        for &value in output.iter() {
            assert!(value.is_finite(), "Output contains non-finite values");
        }
        Ok(())
    }

    #[test]
    fn test_tft_error_handling() {
        let config = TFTConfig {
            d_model: 64,
            n_heads: 8,
            d_ff: 256,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 64,
            num_static_features: 5,
            num_temporal_features: 6,
            hidden_size: 32,
        };
        
        let tft = TemporalFusionTransformer::new(config.clone()).unwrap();
        
        // Test with wrong features dimension
        let wrong_features = Array2::ones((2, config.num_static_features + config.num_temporal_features + 1));
        let result = tft.forward(&wrong_features);
        assert!(result.is_err(), "Should error on wrong features dimension");
        
        // Test with correct features dimension
        let correct_features = Array2::ones((2, config.num_static_features + config.num_temporal_features));
        let result = tft.forward(&correct_features);
        assert!(result.is_ok(), "Should not error on correct features dimension");
    }

    #[test]
    fn test_tft_with_variable_selection() -> Result<(), ModelError> {
        let config = TFTConfig {
            d_model: 64,
            n_heads: 8,
            d_ff: 256,
            dropout: 0.1,
            n_layers: 3,
            max_seq_len: 64,
            num_static_features: 5,
            num_temporal_features: 10,
            hidden_size: 32,
        };
        
        let tft = TemporalFusionTransformer::new(config.clone())?;
        
        // Create sample input
        let batch_size = 2;
        let total_features = tft.config.num_static_features + tft.config.num_temporal_features;
        
        let x = Array2::random((batch_size, total_features), Normal::new(0.0, 1.0)?);
        
        // Run forward pass
        let output = tft.forward(&x)?;
        
        // Check output shape
        assert_eq!(output.shape(), &[batch_size, tft.config.d_model]);
        
        // Check that output values are finite
        for &value in output.iter() {
            assert!(value.is_finite(), "Output contains non-finite values");
        }
        
        Ok(())
    }
} 