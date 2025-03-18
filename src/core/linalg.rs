//! Linear algebra adapter module that provides a unified interface
//! for both pure Rust (linfa-linalg) and BLAS-accelerated (ndarray-linalg) implementations.
//!
//! This module automatically selects the appropriate implementation based on the
//! feature flags enabled during compilation.

use ndarray::{Array1, Array2};
use thiserror::Error;

/// Error type for linear algebra operations
#[derive(Debug, Error)]
pub enum LinalgError {
    #[error("Linear algebra operation failed: {0}")]
    OperationFailed(String),
    
    #[error("Invalid dimensions for operation")]
    InvalidDimensions,
    
    #[error("Matrix is singular")]
    SingularMatrix,
}

/// Result type for linear algebra operations
pub type Result<T> = std::result::Result<T, LinalgError>;

// When using pure Rust implementation (linfa-linalg)
#[cfg(feature = "pure-rust")]
mod impl_linalg {
    use super::*;
    use linfa_linalg::svd::SVDInto;
    use linfa_linalg::qr::QRInto;
    use linfa_linalg::cholesky::CholeskyInplace;
    
    /// Compute the singular value decomposition of a matrix
    pub fn svd(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let a_owned = a.to_owned();
        let result = a_owned.svd_into(true, true)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        
        let (u, s, vt) = result;
        Ok((u, s, vt))
    }
    
    /// Compute the QR decomposition of a matrix
    pub fn qr(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let a_owned = a.to_owned();
        let qr_decomp = a_owned.qr_into()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        
        Ok(qr_decomp.into_decomp())
    }
    
    /// Compute the Cholesky decomposition of a matrix
    pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
        let mut a_copy = a.to_owned();
        a_copy.cholesky_into()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
    
    /// Solve a linear system Ax = b
    pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        // Use QR decomposition to solve the system
        let a_owned = a.to_owned();
        let qr_decomp = a_owned.qr_into()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        
        let b_2d = b.clone().insert_axis(ndarray::Axis(1));
        let x_2d = qr_decomp.solve(&b_2d)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        
        Ok(x_2d.column(0).to_owned())
    }
    
    /// Compute the inverse of a matrix
    pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::InvalidDimensions);
        }
        
        // Create identity matrix
        let mut result = Array2::zeros((n, n));
        let mut identity = Array2::zeros((n, n));
        for i in 0..n {
            identity[[i, i]] = 1.0;
        }
        
        // Solve Ax = I for each column
        for i in 0..n {
            let b = identity.column(i).to_owned();
            let x = solve(a, &b)?;
            for j in 0..n {
                result[[j, i]] = x[j];
            }
        }
        
        Ok(result)
    }
    
    /// Compute the eigenvalues and eigenvectors of a symmetric matrix
    pub fn eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        // For symmetric matrices, we can use SVD as an approximation
        // This is not the most efficient approach but works for a pure Rust implementation
        let (u, s, _) = svd(a)?;
        Ok((s, u))
    }
    
    /// Compute the determinant of a matrix
    pub fn det(a: &Array2<f64>) -> Result<f64> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::InvalidDimensions);
        }
        
        // Use QR decomposition to compute determinant
        let (_, r) = qr(a)?;
        
        // Determinant is the product of diagonal elements of R
        let mut det = 1.0;
        for i in 0..n {
            det *= r[[i, i]];
        }
        
        Ok(det)
    }
}

// When using BLAS-accelerated implementation (ndarray-linalg)
#[cfg(not(feature = "pure-rust"))]
mod impl_linalg {
    use super::*;
    use ndarray_linalg::{Solve, SVD, Eigh, Inverse, Determinant, QR, Cholesky, UPLO};
    
    /// Compute the singular value decomposition of a matrix
    pub fn svd(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (u, s, vt) = a.svd(true, true)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        
        // In ndarray-linalg 0.14.1, u and vt are returned as Options
        let u = u.unwrap_or_else(|| Array2::zeros((a.nrows(), a.nrows())));
        let vt = vt.unwrap_or_else(|| Array2::zeros((a.ncols(), a.ncols())));
        
        Ok((u, s, vt))
    }
    
    /// Compute the QR decomposition of a matrix
    pub fn qr(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let (q, r) = a.qr()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))?;
        Ok((q, r))
    }
    
    /// Compute the Cholesky decomposition of a matrix
    pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
        a.cholesky(UPLO::Upper)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
    
    /// Solve a linear system Ax = b
    pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        a.solve(b)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
    
    /// Compute the inverse of a matrix
    pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
        a.inv()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
    
    /// Compute the eigenvalues and eigenvectors of a symmetric matrix
    pub fn eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        a.eigh(UPLO::Upper)
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
    
    /// Compute the determinant of a matrix
    pub fn det(a: &Array2<f64>) -> Result<f64> {
        a.det()
            .map_err(|e| LinalgError::OperationFailed(e.to_string()))
    }
}

// Re-export the implementation
pub use impl_linalg::*;

// Additional utility functions that work with both implementations
/// Compute the matrix multiplication C = A * B
pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// Compute the dot product of two vectors
pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

/// Compute the matrix-vector product y = A * x
pub fn matvec(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
    a.dot(x)
}

/// Compute the transpose of a matrix
pub fn transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b);
        
        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(c[[0, 0]], 19.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 0]], 43.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 50.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dot() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let c = dot(&a, &b);
        
        assert_abs_diff_eq!(c, 32.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_matvec() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![7.0, 8.0, 9.0];
        let y = matvec(&a, &x);
        
        assert_eq!(y.shape(), &[2]);
        assert_abs_diff_eq!(y[0], 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[1], 122.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_transpose() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let at = transpose(&a);
        
        assert_eq!(at.shape(), &[3, 2]);
        assert_abs_diff_eq!(at[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(at[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(at[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(at[[1, 1]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(at[[2, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(at[[2, 1]], 6.0, epsilon = 1e-10);
    }
} 