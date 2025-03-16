use ndarray::{Array, Array2, Array4};
use rand_distr::{Normal, Distribution};
use crate::error::ModelError;

/// Initialize weights using Xavier/Glorot initialization
pub fn xavier_init(shape: &[usize]) -> Result<Array2<f32>, ModelError> {
    let n_in = shape[0] as f32;
    let n_out = shape[1] as f32;
    let limit = (6.0 / (n_in + n_out)).sqrt();
    
    let normal = Normal::new(0.0, limit).map_err(|e| ModelError::Other(e.to_string()))?;
    let mut rng = rand::thread_rng();
    
    let mut data = Vec::with_capacity(shape.iter().product());
    for _ in 0..shape.iter().product() {
        data.push(normal.sample(&mut rng));
    }
    
    Ok(Array::from_vec(data).into_shape((shape[0], shape[1]))?)
}

/// Compute scaled dot-product attention
pub fn compute_attention(
    query: &Array4<f32>,
    key: &Array4<f32>,
    value: &Array4<f32>,
    d_k: f32,
) -> Result<Array4<f32>, ModelError> {
    let batch_size = query.shape()[0];
    let n_heads = query.shape()[1];
    let seq_len = query.shape()[2];
    let d_v = value.shape()[3];

    // Compute attention scores: (batch_size, n_heads, seq_len, seq_len)
    let mut scores = Array4::zeros((batch_size, n_heads, seq_len, seq_len));
    for b in 0..batch_size {
        for h in 0..n_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0;
                    for k in 0..d_k as usize {
                        score += query[[b, h, i, k]] * key[[b, h, j, k]];
                    }
                    scores[[b, h, i, j]] = score / d_k.sqrt();
                }
            }
        }
    }

    // Apply softmax
    let mut attention_weights = Array4::zeros((batch_size, n_heads, seq_len, seq_len));
    for b in 0..batch_size {
        for h in 0..n_heads {
            for i in 0..seq_len {
                let mut max_val = f32::NEG_INFINITY;
                let mut sum_exp = 0.0;
                
                // Find max value for numerical stability
                for j in 0..seq_len {
                    max_val = max_val.max(scores[[b, h, i, j]]);
                }
                
                // Compute softmax
                for j in 0..seq_len {
                    let exp_val = (scores[[b, h, i, j]] - max_val).exp();
                    attention_weights[[b, h, i, j]] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize
                for j in 0..seq_len {
                    attention_weights[[b, h, i, j]] /= sum_exp;
                }
            }
        }
    }

    // Apply attention weights to values
    let mut output = Array4::zeros((batch_size, n_heads, seq_len, d_v));
    for b in 0..batch_size {
        for h in 0..n_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    for k in 0..d_v {
                        output[[b, h, i, k]] += attention_weights[[b, h, i, j]] * value[[b, h, j, k]];
                    }
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_xavier_init() -> Result<(), ModelError> {
        let shape = [100, 100];
        let weights = xavier_init(&shape)?;
        
        assert_eq!(weights.shape(), &shape);
        
        // Calculate mean and std manually
        let sum: f32 = weights.iter().sum();
        let mean = sum / (weights.len() as f32);
        
        let var_sum: f32 = weights.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        let std = (var_sum / (weights.len() as f32)).sqrt();
        
        assert!(mean.abs() < 0.1);
        assert!(std > 0.0 && std < 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_attention_computation() -> Result<(), ModelError> {
        let batch_size = 2;
        let n_heads = 4;
        let seq_len = 10;
        let d_k = 16;
        
        let query = Array4::zeros((batch_size, n_heads, seq_len, d_k));
        let key = Array4::zeros((batch_size, n_heads, seq_len, d_k));
        let value = Array4::zeros((batch_size, n_heads, seq_len, d_k));
        
        let attention = compute_attention(&query, &key, &value, d_k as f32)?;
        
        assert_eq!(attention.shape(), &[batch_size, n_heads, seq_len, d_k]);
        Ok(())
    }
} 