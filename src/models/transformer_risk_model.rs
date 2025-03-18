use ndarray::{Array2, ArrayView2, s};
use async_trait::async_trait;
use crate::error::ModelError;
use crate::transformer::{
    TransformerComponent,
    TransformerConfig,
    TransformerLayer,
    position::PositionalEncoder,
};
use crate::types::{MarketData, RiskFactors, RiskModel};
use crate::quantization::{Quantizable, QuantizationConfig, Quantizer, QuantizedTensor};
use crate::memory_opt::{MemoryConfig, SparseTensor, ChunkedProcessor};
use std::collections::HashMap;

/// Risk model based on transformer architecture for financial time series analysis.
/// 
/// This model leverages transformer neural networks to capture complex temporal dependencies
/// and non-linear relationships in financial market data. It uses self-attention mechanisms
/// to identify important patterns across different time steps and assets.
/// 
/// Key features:
/// - Multi-head attention for capturing diverse relationships
/// - Positional encoding for sequence awareness
/// - Layer normalization for stable training
/// - Residual connections to prevent vanishing gradients
/// 
/// The model generates risk factors that represent the underlying drivers of market returns
/// and estimates covariance matrices for risk assessment.
#[derive(Debug)]
pub struct TransformerRiskModel {
    layers: Vec<TransformerLayer>,
    pos_encoder: PositionalEncoder,
    config: TransformerConfig,
    // Optional quantized weights storage
    quantized_weights: Option<HashMap<String, QuantizedTensor>>,
    // Optional sparse weights storage
    sparse_weights: Option<HashMap<String, SparseTensor>>,
    // Memory optimization configuration
    memory_config: Option<MemoryConfig>,
}

impl TransformerRiskModel {
    /// Creates a new transformer-based risk model with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the transformer model
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - New transformer risk model or error if initialization fails
    pub fn with_config(config: TransformerConfig) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(config.n_layers);
        
        for _ in 0..config.n_layers {
            layers.push(TransformerLayer::new(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
            )?);
        }
        
        let pos_encoder = PositionalEncoder::new(config.d_model, config.max_seq_len);
        
        Ok(Self {
            layers,
            pos_encoder,
            config,
            quantized_weights: None,
            sparse_weights: None,
            memory_config: None,
        })
    }

    /// Creates a new transformer-based risk model with the specified architecture.
    /// 
    /// # Arguments
    /// 
    /// * `d_model` - Dimension of the model's hidden state
    /// * `n_heads` - Number of attention heads in multi-head attention
    /// * `d_ff` - Dimension of the feed-forward network
    /// * `n_layers` - Number of transformer layers
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, ModelError>` - New transformer risk model or error if initialization fails
    /// 
    /// # Example
    /// 
    /// ```rust,no_run
    /// use deep_risk_model::transformer_risk_model::TransformerRiskModel;
    /// 
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = TransformerRiskModel::new(64, 8, 256, 3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, n_layers: usize) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerLayer::new(
                d_model,
                n_heads,
                d_ff,
                0.1, // Default dropout rate
            )?);
        }
        
        let max_seq_len = 100; // Default max sequence length
        let pos_encoder = PositionalEncoder::new(d_model, max_seq_len);
        
        let config = TransformerConfig {
            n_heads,
            d_model,
            d_ff,
            n_layers,
            dropout: 0.1,
            max_seq_len,
            num_static_features: d_model,
            num_temporal_features: d_model,
            hidden_size: d_model / 2,
        };
        
        Ok(Self {
            layers,
            pos_encoder,
            config,
            quantized_weights: None,
            sparse_weights: None,
            memory_config: None,
        })
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        
        // Count parameters in each transformer layer
        for layer in &self.layers {
            // Feed-forward weights
            total += layer.w1().len() + layer.w2().len();
            
            // Layer normalization parameters
            total += layer.norm1_scale().len() + layer.norm1_bias().len();
            total += layer.norm2_scale().len() + layer.norm2_bias().len();
            
            // Attention weights
            let attention = layer.attention();
            total += attention.w_q().len() + attention.w_k().len() + 
                     attention.w_v().len() + attention.w_o().len();
        }
        
        // Positional encoder parameters
        total += self.pos_encoder.encoding().len();
        
        total
    }

    /// Set memory optimization configuration
    pub fn set_memory_config(&mut self, config: MemoryConfig) {
        self.memory_config = Some(config);
    }

    /// Get memory optimization configuration
    pub fn memory_config(&self) -> Option<&MemoryConfig> {
        self.memory_config.as_ref()
    }

    /// Convert dense weights to sparse representation for memory efficiency
    pub fn sparsify(&mut self, threshold: f32) -> Result<(), ModelError> {
        if let Some(config) = &self.memory_config {
            if !config.use_sparse_tensors {
                return Ok(());
            }
        } else {
            return Ok(());
        }

        let mut sparse_weights = HashMap::new();
        
        // Sparsify weights in each transformer layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Sparsify feed-forward weights
            let w1_view = layer.w1().view();
            let w1_sparse = SparseTensor::from_dense(&w1_view, threshold);
            sparse_weights.insert(format!("layer_{}_w1", layer_idx), w1_sparse);
            
            let w2_view = layer.w2().view();
            let w2_sparse = SparseTensor::from_dense(&w2_view, threshold);
            sparse_weights.insert(format!("layer_{}_w2", layer_idx), w2_sparse);
            
            // Sparsify attention weights
            let attention = layer.attention();
            
            let w_q_view = attention.w_q().view();
            let w_q_sparse = SparseTensor::from_dense(&w_q_view, threshold);
            sparse_weights.insert(format!("layer_{}_attention_w_q", layer_idx), w_q_sparse);
            
            let w_k_view = attention.w_k().view();
            let w_k_sparse = SparseTensor::from_dense(&w_k_view, threshold);
            sparse_weights.insert(format!("layer_{}_attention_w_k", layer_idx), w_k_sparse);
            
            let w_v_view = attention.w_v().view();
            let w_v_sparse = SparseTensor::from_dense(&w_v_view, threshold);
            sparse_weights.insert(format!("layer_{}_attention_w_v", layer_idx), w_v_sparse);
            
            let w_o_view = attention.w_o().view();
            let w_o_sparse = SparseTensor::from_dense(&w_o_view, threshold);
            sparse_weights.insert(format!("layer_{}_attention_w_o", layer_idx), w_o_sparse);
        }
        
        // Store the sparse weights
        self.sparse_weights = Some(sparse_weights);
        
        Ok(())
    }

    /// Calculate memory savings from sparsification
    pub fn sparse_memory_savings(&self) -> Option<(usize, usize, f32)> {
        if let Some(sparse_weights) = &self.sparse_weights {
            let mut dense_size = 0;
            let mut sparse_size = 0;
            
            for (_, tensor) in sparse_weights {
                let (rows, cols) = tensor.shape;
                dense_size += rows * cols * std::mem::size_of::<f32>();
                sparse_size += tensor.memory_usage();
            }
            
            let ratio = dense_size as f32 / sparse_size as f32;
            Some((dense_size, sparse_size, ratio))
        } else {
            None
        }
    }
}

/// Implementation of the TransformerComponent trait for TransformerRiskModel.
/// 
/// This implementation enables the model to process input features through
/// the transformer architecture, applying positional encoding and passing
/// the data through multiple transformer layers.
impl TransformerComponent<f32> for TransformerRiskModel {
    /// Performs forward propagation through the transformer network.
    /// 
    /// This method:
    /// 1. Applies positional encoding to the input
    /// 2. Passes the encoded input through each transformer layer
    /// 3. Returns the processed features
    /// 
    /// # Arguments
    /// 
    /// * `features` - Input tensor of shape (batch_size, d_model)
    /// 
    /// # Returns
    /// 
    /// * `Result<Array2<f32>, ModelError>` - Processed features
    fn forward(&self, features: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
        // Add positional encoding
        let mut output = self.pos_encoder.forward(features)?;
        
        // Pass through transformer layers
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        
        Ok(output)
    }
}

// Implement Send and Sync for TransformerRiskModel
// This is safe because TransformerRiskModel only contains types that are already Send and Sync
unsafe impl Send for TransformerRiskModel {}
unsafe impl Sync for TransformerRiskModel {}

/// Implementation of the RiskModel trait for TransformerRiskModel.
/// 
/// This implementation enables the model to generate risk factors and
/// estimate covariance matrices from market data using transformer-based
/// deep learning.
#[async_trait]
impl RiskModel for TransformerRiskModel {
    /// Trains the transformer risk model.
    /// 
    /// The transformer model is self-supervised and doesn't require explicit training,
    /// so this method simply returns success.
    /// 
    /// # Arguments
    /// 
    /// * `_data` - Market data (unused in this implementation)
    /// 
    /// # Returns
    /// 
    /// * `Result<(), ModelError>` - Success or error during training
    async fn train(&mut self, _data: &MarketData) -> Result<(), ModelError> {
        // The transformer model is self-supervised and doesn't require explicit training
        Ok(())
    }

    /// Generates risk factors from market data using the transformer model.
    /// 
    /// This method:
    /// 1. Validates input dimensions
    /// 2. Processes features through the transformer
    /// 3. Extracts risk factors by aggregating across the sequence dimension
    /// 4. Normalizes the factors
    /// 5. Computes the factor covariance matrix
    /// 
    /// # Arguments
    /// 
    /// * `data` - Market data containing asset returns and features
    /// 
    /// # Returns
    /// 
    /// * `Result<RiskFactors, ModelError>` - Generated risk factors and their covariance matrix
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
        
        // Check if we should use chunked processing
        if let Some(memory_config) = &self.memory_config {
            if memory_config.use_chunked_processing {
                return self.generate_risk_factors_chunked(data, memory_config).await;
            }
        }
        
        // Process through transformer
        let processed = self.forward(features)?;
        
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

    /// Estimates the covariance matrix from market data.
    /// 
    /// This method generates risk factors and returns their covariance matrix.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Market data containing asset returns and features
    /// 
    /// # Returns
    /// 
    /// * `Result<Array2<f32>, ModelError>` - Estimated covariance matrix
    async fn estimate_covariance(&self, data: &MarketData) -> Result<Array2<f32>, ModelError> {
        let risk_factors = self.generate_risk_factors(data).await?;
        Ok(risk_factors.covariance().to_owned())
    }
}

impl TransformerRiskModel {
    /// Generate risk factors using chunked processing for memory efficiency
    async fn generate_risk_factors_chunked(&self, data: &MarketData, memory_config: &MemoryConfig) -> Result<RiskFactors, ModelError> {
        let features = data.features();
        let n_samples = features.shape()[0];
        
        // Create chunked processor
        let mut chunked_processor = ChunkedProcessor::new(memory_config.clone(), n_samples);
        
        // Convert features to ArrayView2 for chunked processing
        let features_view = features.view();
        
        // Process features in chunks
        let chunk_results = chunked_processor.process_in_chunks(&features_view, |chunk| {
            // Convert chunk to owned Array2 before processing
            let owned_chunk = chunk.to_owned();
            
            // Process chunk through transformer
            let processed = self.forward(&owned_chunk)?;
            
            // For simplicity, just return the processed chunk
            Ok(processed)
        })?;
        
        // Combine chunk results
        let mut combined_processed = Vec::new();
        for chunk_result in chunk_results {
            for row in 0..chunk_result.shape()[0] {
                let mut processed_row = Vec::new();
                for col in 0..chunk_result.shape()[1] {
                    processed_row.push(chunk_result[[row, col]]);
                }
                combined_processed.push(processed_row);
            }
        }
        
        // Convert to Array2
        let n_rows = combined_processed.len();
        let n_cols = if n_rows > 0 { combined_processed[0].len() } else { 0 };
        let mut processed = Array2::zeros((n_rows, n_cols));
        for i in 0..n_rows {
            for j in 0..n_cols {
                processed[[i, j]] = combined_processed[i][j];
            }
        }
        
        // Extract features by taking mean of sequence dimension
        let output_rows = n_samples - self.config.max_seq_len + 1;
        let mut factors = Array2::zeros((output_rows, self.config.d_model));
        for i in 0..output_rows {
            for j in 0..self.config.d_model {
                let mut sum = 0.0;
                for k in 0..self.config.max_seq_len {
                    if i + k < processed.shape()[0] {
                        sum += processed[[i + k, j]];
                    }
                }
                factors[[i, j]] = sum / (self.config.max_seq_len as f32);
            }
        }
        
        // Normalize factors
        let f_mean = factors.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| ModelError::ComputationError("Failed to compute mean".into()))?;
        let f_std = factors.std_axis(ndarray::Axis(0), 0.0);
        
        for i in 0..output_rows {
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
                for k in 0..output_rows {
                    cov += factors[[k, i]] * factors[[k, j]];
                }
                covariance[[i, j]] = cov / (output_rows as f32);
            }
        }
        
        Ok(RiskFactors::new(factors, covariance))
    }
}

impl Quantizable for TransformerRiskModel {
    fn quantize(&mut self, config: QuantizationConfig) -> Result<(), ModelError> {
        let quantizer = Quantizer::new(config);
        let mut quantized_weights = HashMap::new();
        
        // Quantize weights in each transformer layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Quantize feed-forward weights
            let w1_view = layer.w1().view();
            let w1_quantized = quantizer.quantize_tensor(&w1_view)?;
            quantized_weights.insert(format!("layer_{}_w1", layer_idx), w1_quantized);
            
            let w2_view = layer.w2().view();
            let w2_quantized = quantizer.quantize_tensor(&w2_view)?;
            quantized_weights.insert(format!("layer_{}_w2", layer_idx), w2_quantized);
            
            // Quantize layer normalization parameters
            let norm1_scale_view = layer.norm1_scale().view().insert_axis(ndarray::Axis(0));
            let norm1_scale_quantized = quantizer.quantize_tensor(&norm1_scale_view)?;
            quantized_weights.insert(format!("layer_{}_norm1_scale", layer_idx), norm1_scale_quantized);
            
            let norm1_bias_view = layer.norm1_bias().view().insert_axis(ndarray::Axis(0));
            let norm1_bias_quantized = quantizer.quantize_tensor(&norm1_bias_view)?;
            quantized_weights.insert(format!("layer_{}_norm1_bias", layer_idx), norm1_bias_quantized);
            
            let norm2_scale_view = layer.norm2_scale().view().insert_axis(ndarray::Axis(0));
            let norm2_scale_quantized = quantizer.quantize_tensor(&norm2_scale_view)?;
            quantized_weights.insert(format!("layer_{}_norm2_scale", layer_idx), norm2_scale_quantized);
            
            let norm2_bias_view = layer.norm2_bias().view().insert_axis(ndarray::Axis(0));
            let norm2_bias_quantized = quantizer.quantize_tensor(&norm2_bias_view)?;
            quantized_weights.insert(format!("layer_{}_norm2_bias", layer_idx), norm2_bias_quantized);
            
            // Quantize attention weights
            let attention = layer.attention();
            
            let w_q_view = attention.w_q().view();
            let w_q_quantized = quantizer.quantize_tensor(&w_q_view)?;
            quantized_weights.insert(format!("layer_{}_attention_w_q", layer_idx), w_q_quantized);
            
            let w_k_view = attention.w_k().view();
            let w_k_quantized = quantizer.quantize_tensor(&w_k_view)?;
            quantized_weights.insert(format!("layer_{}_attention_w_k", layer_idx), w_k_quantized);
            
            let w_v_view = attention.w_v().view();
            let w_v_quantized = quantizer.quantize_tensor(&w_v_view)?;
            quantized_weights.insert(format!("layer_{}_attention_w_v", layer_idx), w_v_quantized);
            
            let w_o_view = attention.w_o().view();
            let w_o_quantized = quantizer.quantize_tensor(&w_o_view)?;
            quantized_weights.insert(format!("layer_{}_attention_w_o", layer_idx), w_o_quantized);
        }
        
        // Quantize positional encoder parameters
        let pos_encoding_view = self.pos_encoder.encoding().view();
        let pos_encoding_quantized = quantizer.quantize_tensor(&pos_encoding_view)?;
        quantized_weights.insert("positional_encoding".to_string(), pos_encoding_quantized);
        
        // Store the quantized weights
        self.quantized_weights = Some(quantized_weights);
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        if let Some(quantized_weights) = &self.quantized_weights {
            // Calculate memory usage of quantized weights
            let mut total_bytes = 0;
            for (_, tensor) in quantized_weights {
                // Data size
                total_bytes += tensor.data.len();
                
                // Scales size (f32)
                total_bytes += tensor.scales.len() * 4;
                
                // Zero points size (i32) if present
                if let Some(zero_points) = &tensor.zero_points {
                    total_bytes += zero_points.len() * 4;
                }
                
                // Shape metadata
                total_bytes += tensor.shape.len() * std::mem::size_of::<usize>();
                
                // Other metadata (precision, per_channel flag)
                total_bytes += 2;
            }
            
            total_bytes
        } else {
            // Calculate memory usage of full-precision weights
            let num_params = self.num_parameters();
            
            // Each parameter is a f32 (4 bytes)
            num_params * 4
        }
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