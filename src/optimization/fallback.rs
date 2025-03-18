//! Fallback implementations for matrix operations when BLAS is not available.
//! 
//! This module provides pure Rust implementations of matrix operations that would
//! normally require BLAS. These implementations are slower but allow the library
//! to function without BLAS dependencies.
//!
//! # Performance Considerations
//!
//! These implementations are significantly slower than BLAS for large matrices.
//! They are intended for use in environments where BLAS is not available or
//! for small matrices where the overhead of BLAS might outweigh its benefits.
//!
//! For production use with large matrices, it's recommended to use a BLAS
//! implementation such as OpenBLAS, Intel MKL, or Apple's Accelerate framework.
//!
//! # Usage
//!
//! These functions are automatically used when the `no-blas` feature is enabled
//! and the BLAS features are disabled:
//!
//! ```bash
//! cargo build --no-default-features --features no-blas
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::error::ModelError;

/// Computes the inverse of a small matrix (up to 3x3) without BLAS.
/// 
/// # Arguments
/// 
/// * `a` - Square matrix to invert
/// 
/// # Returns
/// 
/// * `Result<Array2<f32>, ModelError>` - Inverted matrix or error
///
/// # Performance
///
/// This implementation is limited to matrices of size 3x3 or smaller.
/// For larger matrices, a BLAS implementation is required.
pub fn inv(a: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(ModelError::InvalidDimension(
            "Matrix must be square for inversion".into()
        ));
    }
    
    let n = shape[0];
    
    // When BLAS is enabled, use ndarray_linalg for matrix inversion
    #[cfg(feature = "blas-enabled")]
    {
        use ndarray_linalg::Inverse;
        
        // For very small matrices, the pure Rust implementation might be faster
        if n <= 3 {
            // Use the pure Rust implementation for small matrices
            return fallback_inv(a);
        }
        
        // Use BLAS-accelerated matrix inversion for larger matrices
        match a.inv() {
            Ok(inv_a) => return Ok(inv_a),
            Err(_) => return Err(ModelError::NumericalError(
                "Matrix inversion failed".into()
            )),
        }
    }
    
    // When BLAS is not enabled, use the pure Rust implementation
    #[cfg(not(feature = "blas-enabled"))]
    {
        // For matrices larger than 3x3, we need BLAS
        if n > 3 {
            return Err(ModelError::UnsupportedOperation(
                "Matrix inversion for matrices larger than 3x3 requires BLAS. Please enable the BLAS feature.".into()
            ));
        }
        
        return fallback_inv(a);
    }
}

/// Pure Rust implementation of matrix inversion for small matrices (up to 3x3)
#[inline]
fn fallback_inv(a: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
    let shape = a.shape();
    let n = shape[0];
    
    match n {
        1 => {
            // 1x1 matrix
            let val = a[[0, 0]];
            if val.abs() < 1e-10 {
                return Err(ModelError::NumericalError(
                    "Matrix is singular".into()
                ));
            }
            let mut result = Array2::zeros((1, 1));
            result[[0, 0]] = 1.0 / val;
            Ok(result)
        },
        2 => {
            // 2x2 matrix
            let a11 = a[[0, 0]];
            let a12 = a[[0, 1]];
            let a21 = a[[1, 0]];
            let a22 = a[[1, 1]];
            
            let det = a11 * a22 - a12 * a21;
            if det.abs() < 1e-10 {
                return Err(ModelError::NumericalError(
                    "Matrix is singular".into()
                ));
            }
            
            let mut result = Array2::zeros((2, 2));
            result[[0, 0]] = a22 / det;
            result[[0, 1]] = -a12 / det;
            result[[1, 0]] = -a21 / det;
            result[[1, 1]] = a11 / det;
            
            Ok(result)
        },
        3 => {
            // 3x3 matrix
            let a11 = a[[0, 0]];
            let a12 = a[[0, 1]];
            let a13 = a[[0, 2]];
            let a21 = a[[1, 0]];
            let a22 = a[[1, 1]];
            let a23 = a[[1, 2]];
            let a31 = a[[2, 0]];
            let a32 = a[[2, 1]];
            let a33 = a[[2, 2]];
            
            // Calculate determinant
            let det = a11 * (a22 * a33 - a23 * a32)
                    - a12 * (a21 * a33 - a23 * a31)
                    + a13 * (a21 * a32 - a22 * a31);
            
            if det.abs() < 1e-10 {
                return Err(ModelError::NumericalError(
                    "Matrix is singular".into()
                ));
            }
            
            let mut result = Array2::zeros((3, 3));
            
            // Calculate cofactors and adjugate
            result[[0, 0]] = (a22 * a33 - a23 * a32) / det;
            result[[0, 1]] = (a13 * a32 - a12 * a33) / det;
            result[[0, 2]] = (a12 * a23 - a13 * a22) / det;
            result[[1, 0]] = (a23 * a31 - a21 * a33) / det;
            result[[1, 1]] = (a11 * a33 - a13 * a31) / det;
            result[[1, 2]] = (a13 * a21 - a11 * a23) / det;
            result[[2, 0]] = (a21 * a32 - a22 * a31) / det;
            result[[2, 1]] = (a12 * a31 - a11 * a32) / det;
            result[[2, 2]] = (a11 * a22 - a12 * a21) / det;
            
            Ok(result)
        },
        _ => {
            Err(ModelError::UnsupportedOperation(
                "Matrix inversion for matrices larger than 3x3 requires BLAS. Please enable the BLAS feature.".into()
            ))
        }
    }
}

/// Matrix multiplication without BLAS.
/// 
/// # Arguments
/// 
/// * `a` - First matrix
/// * `b` - Second matrix
/// 
/// # Returns
/// 
/// * `Result<Array2<f32>, ModelError>` - Result of a * b
///
/// # Performance
///
/// This implementation has O(n³) time complexity, which is significantly
/// slower than optimized BLAS implementations for large matrices.
pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, ModelError> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape[1] != b_shape[0] {
        return Err(ModelError::InvalidDimension(
            format!("Matrix dimensions don't match for multiplication: ({}, {}) and ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1])
        ));
    }
    
    let mut result = Array2::zeros((a_shape[0], b_shape[1]));
    
    // Basic matrix multiplication algorithm
    for i in 0..a_shape[0] {
        for j in 0..b_shape[1] {
            let mut sum = 0.0;
            for k in 0..a_shape[1] {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }
    
    Ok(result)
}

/// Dot product of two vectors without BLAS.
/// 
/// # Arguments
/// 
/// * `a` - First vector
/// * `b` - Second vector
/// 
/// # Returns
/// 
/// * `f32` - Dot product
///
/// # Performance
///
/// This implementation has O(n) time complexity and is reasonably efficient,
/// but still slower than BLAS for very large vectors.
pub fn dot(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    if a.len() != b.len() {
        panic!("Vector dimensions don't match for dot product: {} and {}", a.len(), b.len());
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Matrix-vector multiplication without BLAS.
/// 
/// # Arguments
/// 
/// * `a` - Matrix
/// * `x` - Vector
/// 
/// # Returns
/// 
/// * `Result<Array1<f32>, ModelError>` - Result of a * x
///
/// # Performance
///
/// This implementation has O(n²) time complexity, which is slower than
/// optimized BLAS implementations for large matrices.
pub fn matvec(a: &Array2<f32>, x: &Array1<f32>) -> Result<Array1<f32>, ModelError> {
    let a_shape = a.shape();
    
    if a_shape[1] != x.len() {
        return Err(ModelError::InvalidDimension(
            format!("Dimensions don't match for matrix-vector multiplication: ({}, {}) and ({})",
                a_shape[0], a_shape[1], x.len())
        ));
    }
    
    let mut result = Array1::zeros(a_shape[0]);
    
    for i in 0..a_shape[0] {
        let mut sum = 0.0;
        for j in 0..a_shape[1] {
            sum += a[[i, j]] * x[j];
        }
        result[i] = sum;
    }
    
    Ok(result)
}

/// Transpose a matrix without BLAS.
/// 
/// # Arguments
/// 
/// * `a` - Matrix to transpose
/// 
/// # Returns
/// 
/// * `Array2<f32>` - Transposed matrix
pub fn transpose(a: &Array2<f32>) -> Array2<f32> {
    let shape = a.shape();
    let mut result = Array2::zeros((shape[1], shape[0]));
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            result[[j, i]] = a[[i, j]];
        }
    }
    
    result
}

/// Compute the trace of a matrix (sum of diagonal elements).
/// 
/// # Arguments
/// 
/// * `a` - Square matrix
/// 
/// # Returns
/// 
/// * `Result<f32, ModelError>` - Trace of the matrix
pub fn trace(a: &Array2<f32>) -> Result<f32, ModelError> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(ModelError::InvalidDimension(
            "Matrix must be square to compute trace".into()
        ));
    }
    
    let mut sum = 0.0;
    for i in 0..shape[0] {
        sum += a[[i, i]];
    }
    
    Ok(sum)
}

/// Compute the Frobenius norm of a matrix (square root of sum of squares of all elements).
/// 
/// # Arguments
/// 
/// * `a` - Matrix
/// 
/// # Returns
/// 
/// * `f32` - Frobenius norm
pub fn frobenius_norm(a: &Array2<f32>) -> f32 {
    let mut sum_sq = 0.0;
    for &val in a.iter() {
        sum_sq += val * val;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_inv_1x1() {
        let a = array![[2.0]];
        let inv_a = inv(&a).unwrap();
        assert!((inv_a[[0, 0]] - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_inv_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let inv_a = inv(&a).unwrap();
        let expected = array![[-2.0, 1.0], [1.5, -0.5]];
        
        for i in 0..2 {
            for j in 0..2 {
                assert!((inv_a[[i, j]] - expected[[i, j]]).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_inv_3x3() {
        let a = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]];
        let inv_a = inv(&a).unwrap();
        let expected = array![[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]];
        
        for i in 0..3 {
            for j in 0..3 {
                assert!((inv_a[[i, j]] - expected[[i, j]]).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b).unwrap();
        let expected = array![[19.0, 22.0], [43.0, 50.0]];
        
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - expected[[i, j]]).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_dot() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let result = dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_matvec() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![7.0, 8.0, 9.0];
        let result = matvec(&a, &x).unwrap();
        let expected = array![50.0, 122.0];
        
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_transpose() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = transpose(&a);
        let expected = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        
        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_trace() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let result = trace(&a).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_frobenius_norm() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let result = frobenius_norm(&a);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt();
        assert!((result - expected).abs() < 1e-6);
    }
} 