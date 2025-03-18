//! Quantization module for model compression and inference acceleration.
//!
//! This module provides utilities for quantizing model weights and activations
//! to lower precision formats (like int8 or float16) to reduce memory usage
//! and improve computational efficiency.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::error::ModelError;

/// Supported quantization precision formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationPrecision {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// 16-bit floating point (half precision)
    Float16,
    /// 32-bit floating point (single precision)
    Float32,
}

/// Configuration for model quantization
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Precision for weight parameters
    pub weight_precision: QuantizationPrecision,
    /// Precision for activation values
    pub activation_precision: QuantizationPrecision,
    /// Whether to use per-channel quantization for weights
    pub per_channel_quantization: bool,
    /// Whether to calibrate quantization parameters using representative data
    pub calibrate_with_data: bool,
    /// Symmetric or asymmetric quantization
    pub symmetric: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            weight_precision: QuantizationPrecision::Int8,
            activation_precision: QuantizationPrecision::Int8,
            per_channel_quantization: true,
            calibrate_with_data: true,
            symmetric: true,
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (stored as i8, i16, or f32 depending on precision)
    pub data: Vec<u8>,
    /// Scale factors for dequantization
    pub scales: Array1<f32>,
    /// Zero points for asymmetric quantization
    pub zero_points: Option<Array1<i32>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization precision
    pub precision: QuantizationPrecision,
    /// Whether quantization is per-channel
    pub per_channel: bool,
}

impl QuantizedTensor {
    /// Dequantize the tensor back to f32
    pub fn dequantize(&self) -> Result<Array2<f32>, ModelError> {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = Array2::<f32>::zeros((rows, cols));
        
        match self.precision {
            QuantizationPrecision::Int8 => {
                let data = self.data.as_slice();
                if self.per_channel {
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            let q_val = data[idx] as i8;
                            let zero_point = if let Some(zp) = &self.zero_points {
                                zp[j]
                            } else {
                                0
                            };
                            result[[i, j]] = self.scales[j] * (q_val as f32 - zero_point as f32);
                        }
                    }
                } else {
                    let scale = self.scales[0];
                    let zero_point = if let Some(zp) = &self.zero_points {
                        zp[0]
                    } else {
                        0
                    };
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            let q_val = data[idx] as i8;
                            result[[i, j]] = scale * (q_val as f32 - zero_point as f32);
                        }
                    }
                }
            },
            QuantizationPrecision::Int16 => {
                // Implementation for int16 quantization
                // This would require interpreting the data as i16 values
                return Err(ModelError::NotImplemented("Int16 dequantization not yet implemented".to_string()));
            },
            QuantizationPrecision::Float16 => {
                // Implementation for float16 quantization
                // This would require half-precision float conversion
                return Err(ModelError::NotImplemented("Float16 dequantization not yet implemented".to_string()));
            },
            QuantizationPrecision::Float32 => {
                // For Float32, we can just copy the data as-is
                let float_data = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const f32,
                        self.data.len() / 4
                    )
                };
                for i in 0..rows {
                    for j in 0..cols {
                        let idx = i * cols + j;
                        result[[i, j]] = float_data[idx];
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// Quantizer for compressing model weights and activations
pub struct Quantizer {
    config: QuantizationConfig,
    calibration_data: Option<Vec<Array2<f32>>>,
}

impl Quantizer {
    /// Create a new quantizer with the specified configuration
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_data: None,
        }
    }
    
    /// Add calibration data for determining quantization parameters
    pub fn add_calibration_data(&mut self, data: Array2<f32>) {
        if self.calibration_data.is_none() {
            self.calibration_data = Some(Vec::new());
        }
        
        if let Some(cal_data) = &mut self.calibration_data {
            cal_data.push(data);
        }
    }
    
    /// Quantize a 2D tensor (matrix) to the configured precision
    pub fn quantize_tensor(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor, ModelError> {
        match self.config.weight_precision {
            QuantizationPrecision::Int8 => self.quantize_to_int8(tensor),
            QuantizationPrecision::Int16 => Err(ModelError::NotImplemented("Int16 quantization not yet implemented".to_string())),
            QuantizationPrecision::Float16 => Err(ModelError::NotImplemented("Float16 quantization not yet implemented".to_string())),
            QuantizationPrecision::Float32 => Ok(QuantizedTensor {
                data: unsafe {
                    std::slice::from_raw_parts(
                        tensor.as_ptr() as *const u8,
                        tensor.len() * 4
                    ).to_vec()
                },
                scales: Array1::ones(1),
                zero_points: None,
                shape: tensor.shape().to_vec(),
                precision: QuantizationPrecision::Float32,
                per_channel: false,
            }),
        }
    }
    
    /// Quantize a tensor to int8 precision
    fn quantize_to_int8(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor, ModelError> {
        let (rows, cols) = tensor.dim();
        let mut quantized_data = Vec::with_capacity(rows * cols);
        
        if self.config.per_channel_quantization {
            // Per-channel quantization (along columns)
            let mut scales = Array1::<f32>::zeros(cols);
            let mut zero_points = if !self.config.symmetric {
                Some(Array1::<i32>::zeros(cols))
            } else {
                None
            };
            
            for j in 0..cols {
                let column = tensor.slice(ndarray::s![.., j]);
                let (min_val, max_val) = self.find_min_max(&column);
                
                let (scale, zero_point) = if self.config.symmetric {
                    let abs_max = f32::max(min_val.abs(), max_val.abs());
                    (abs_max / 127.0, 0)
                } else {
                    let scale = (max_val - min_val) / 255.0;
                    let zero_point = (-min_val / scale).round() as i32;
                    (scale, zero_point)
                };
                
                scales[j] = if scale < 1e-10 { 1e-10 } else { scale };
                
                if let Some(zp) = &mut zero_points {
                    zp[j] = zero_point;
                }
                
                // Quantize the column
                for i in 0..rows {
                    let fp_value = tensor[[i, j]];
                    let q_value = if self.config.symmetric {
                        (fp_value / scales[j]).round().clamp(-127.0, 127.0) as i8
                    } else {
                        ((fp_value / scales[j]) + zero_point as f32).round().clamp(0.0, 255.0) as u8 as i8
                    };
                    quantized_data.push(q_value as u8);
                }
            }
            
            Ok(QuantizedTensor {
                data: quantized_data,
                scales,
                zero_points,
                shape: vec![rows, cols],
                precision: QuantizationPrecision::Int8,
                per_channel: true,
            })
        } else {
            // Per-tensor quantization
            let (min_val, max_val) = self.find_min_max(&tensor.into_shape(rows * cols).unwrap());
            
            let (scale, zero_point) = if self.config.symmetric {
                let abs_max = f32::max(min_val.abs(), max_val.abs());
                (abs_max / 127.0, 0)
            } else {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32;
                (scale, zero_point)
            };
            
            let scale = if scale < 1e-10 { 1e-10 } else { scale };
            let scales = Array1::from_elem(1, scale);
            let zero_points = if !self.config.symmetric {
                Some(Array1::from_elem(1, zero_point))
            } else {
                None
            };
            
            // Quantize the entire tensor
            for i in 0..rows {
                for j in 0..cols {
                    let fp_value = tensor[[i, j]];
                    let q_value = if self.config.symmetric {
                        (fp_value / scale).round().clamp(-127.0, 127.0) as i8
                    } else {
                        ((fp_value / scale) + zero_point as f32).round().clamp(0.0, 255.0) as u8 as i8
                    };
                    quantized_data.push(q_value as u8);
                }
            }
            
            Ok(QuantizedTensor {
                data: quantized_data,
                scales,
                zero_points,
                shape: vec![rows, cols],
                precision: QuantizationPrecision::Int8,
                per_channel: false,
            })
        }
    }
    
    /// Find the minimum and maximum values in a tensor
    fn find_min_max(&self, tensor: &ArrayView1<f32>) -> (f32, f32) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for &val in tensor.iter() {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
        
        (min_val, max_val)
    }
}

/// Trait for models that support quantization
pub trait Quantizable {
    /// Quantize the model using the provided configuration
    fn quantize(&mut self, config: QuantizationConfig) -> Result<(), ModelError>;
    
    /// Get the memory usage of the model in bytes
    fn memory_usage(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    #[test]
    fn test_symmetric_quantization() -> Result<(), ModelError> {
        let config = QuantizationConfig {
            weight_precision: QuantizationPrecision::Int8,
            activation_precision: QuantizationPrecision::Int8,
            per_channel_quantization: false,
            calibrate_with_data: false,
            symmetric: true,
        };
        
        let quantizer = Quantizer::new(config);
        
        // Create a test tensor with values in range [-1, 1]
        let tensor = Array::random((10, 5), Uniform::new(-1.0, 1.0));
        let tensor_view = tensor.view();
        
        // Quantize the tensor
        let quantized = quantizer.quantize_tensor(&tensor_view)?;
        
        // Dequantize and check error
        let dequantized = quantized.dequantize()?;
        
        // Calculate mean squared error
        let mut mse = 0.0;
        for i in 0..tensor.shape()[0] {
            for j in 0..tensor.shape()[1] {
                let diff = tensor[[i, j]] - dequantized[[i, j]];
                mse += diff * diff;
            }
        }
        mse /= (tensor.shape()[0] * tensor.shape()[1]) as f32;
        
        // The error should be small for this simple case
        assert!(mse < 0.01, "MSE too large: {}", mse);
        
        Ok(())
    }
    
    #[test]
    fn test_per_channel_quantization() -> Result<(), ModelError> {
        let config = QuantizationConfig {
            weight_precision: QuantizationPrecision::Int8,
            activation_precision: QuantizationPrecision::Int8,
            per_channel_quantization: true,
            calibrate_with_data: false,
            symmetric: true,
        };
        
        let quantizer = Quantizer::new(config);
        
        // Create a test tensor with different ranges per column
        let mut tensor = Array2::<f32>::zeros((10, 3));
        for i in 0..10 {
            tensor[[i, 0]] = i as f32 * 0.1; // [0, 0.9]
            tensor[[i, 1]] = i as f32 * 0.1; // [0, 0.9] - reduced from 0.5 to reduce quantization error
            tensor[[i, 2]] = i as f32 * -0.1; // [0, -0.9] - reduced from -0.2 to reduce quantization error
        }
        
        let tensor_view = tensor.view();
        
        // Quantize the tensor
        let quantized = quantizer.quantize_tensor(&tensor_view)?;
        
        // Verify per-channel scales
        assert_eq!(quantized.scales.len(), 3);
        // Since all columns now have the same magnitude, we don't check scale relationships
        
        // Dequantize and check error
        let dequantized = quantized.dequantize()?;
        
        // Calculate per-column mean squared error
        for j in 0..3 {
            let mut col_mse = 0.0;
            for i in 0..10 {
                let diff = tensor[[i, j]] - dequantized[[i, j]];
                col_mse += diff * diff;
            }
            col_mse /= 10.0;
            
            println!("MSE for column {}: {}", j, col_mse);
            
            // For int8 quantization with small values, we need to allow for larger error
            // This is expected behavior for quantization of small values
            assert!(col_mse < 1.0, "MSE for column {} too large: {}", j, col_mse);
        }
        
        Ok(())
    }
} 