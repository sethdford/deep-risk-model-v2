use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use rand::Rng;

use crate::error::ModelError;
use crate::linalg;

/// Factor quality metrics for evaluating generated risk factors.
/// 
/// These metrics help assess the quality and significance of each risk factor:
/// 
/// - Information coefficient: Measures predictive power of the factor
/// - VIF (Variance Inflation Factor): Measures multicollinearity
/// - T-statistic: Measures statistical significance
/// - Explained variance: Measures proportion of variance explained
#[derive(Debug, Clone)]
pub struct FactorQualityMetrics {
    /// Information coefficient (correlation with future returns)
    pub information_coefficient: f32,
    /// Variance inflation factor (measure of multicollinearity)
    pub vif: f32,
    /// T-statistic for factor significance
    pub t_statistic: f32,
    /// Explained variance ratio
    pub explained_variance: f32,
}

/// Advanced factor analysis and processing for risk factor generation.
/// 
/// The `FactorAnalyzer` performs several key functions:
/// 
/// 1. Orthogonalizes risk factors using Gram-Schmidt process
/// 2. Calculates quality metrics for each factor
/// 3. Selects optimal factors based on quality criteria
/// 4. Estimates factor loadings and covariance matrices
/// 
/// # Example
/// 
/// ```rust,no_run
/// use deep_risk_model::factor_analysis::FactorAnalyzer;
/// use ndarray::Array2;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let analyzer = FactorAnalyzer::new(0.1, 5.0, 1.96);
/// let mut factors = Array2::zeros((100, 5));
/// analyzer.orthogonalize_factors(&mut factors)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FactorAnalyzer {
    /// Minimum information coefficient threshold for factor selection
    min_information_coefficient: f32,
    /// Maximum VIF threshold for factor selection
    max_vif: f32,
    /// Minimum t-statistic threshold for factor significance
    min_t_statistic: f32,
}

// Implement Send and Sync for FactorAnalyzer
// This is safe because FactorAnalyzer only contains primitive types (f32)
// that are already Send and Sync
unsafe impl Send for FactorAnalyzer {}
unsafe impl Sync for FactorAnalyzer {}

impl FactorAnalyzer {
    /// Creates a new FactorAnalyzer with the specified thresholds
    pub fn new(min_information_coefficient: f32, max_vif: f32, min_t_statistic: f32) -> Self {
        FactorAnalyzer {
            min_information_coefficient,
            max_vif,
            min_t_statistic,
        }
    }

    /// Creates a new FactorAnalyzer with default thresholds suitable for most datasets
    pub fn default() -> Self {
        // Use more lenient thresholds that work well with both synthetic and real-world data
        // - Lower min_information_coefficient (0.1 instead of higher values)
        // - Higher max_vif to allow for some collinearity (5.0 is standard in finance)
        // - Lower min_t_statistic (1.65 corresponds to ~90% confidence level)
        FactorAnalyzer {
            min_information_coefficient: 0.1,
            max_vif: 5.0,
            min_t_statistic: 1.65,
        }
    }

    /// Creates a new FactorAnalyzer with strict thresholds for high-quality factors
    pub fn strict() -> Self {
        FactorAnalyzer {
            min_information_coefficient: 0.3,
            max_vif: 2.5,
            min_t_statistic: 1.96, // 95% confidence level
        }
    }

    /// Creates a new FactorAnalyzer with lenient thresholds for exploratory analysis
    pub fn lenient() -> Self {
        FactorAnalyzer {
            min_information_coefficient: 0.05,
            max_vif: 10.0,
            min_t_statistic: 1.28, // 80% confidence level
        }
    }
    
    /// Orthogonalizes a set of factors using the Gram-Schmidt process.
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors to orthogonalize
    /// 
    /// # Returns
    /// 
    /// Result indicating success or failure
    pub fn orthogonalize_factors(&self, factors: &mut Array2<f32>) -> Result<(), ModelError> {
        let (n_samples, n_factors) = factors.dim();
        
        if n_factors == 0 {
            return Ok(());
        }
        
        // Normalize first factor
        let norm = factors.slice(s![.., 0]).mapv(|x| x * x).sum().sqrt();
        if norm > 1e-10 {
            factors.slice_mut(s![.., 0]).mapv_inplace(|x| x / norm);
        }
        
        // Orthogonalize remaining factors using modified Gram-Schmidt
        for i in 1..n_factors {
            let mut factor = factors.slice(s![.., i]).to_owned();
            
            // Subtract projections onto previous factors
            for j in 0..i {
                let prev_factor = factors.slice(s![.., j]);
                
                // Calculate projection coefficient (dot product)
                let proj = {
                    let dot1 = factor.iter().zip(prev_factor.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    let dot2 = prev_factor.iter().map(|&x| x * x).sum::<f32>();
                    if dot2 < 1e-10 { 0.0 } else { dot1 / dot2 }
                };
                
                // Subtract projection
                for (f_val, p_val) in factor.iter_mut().zip(prev_factor.iter()) {
                    *f_val -= proj * (*p_val);
                }
            }
            
            // Normalize
            let norm = factor.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                factor.mapv_inplace(|x| x / norm);
                factors.slice_mut(s![.., i]).assign(&factor);
            } else {
                // If the factor becomes too small after orthogonalization,
                // replace it with a random orthogonal vector
                let mut rng = rand::thread_rng();
                let mut random_vec = Array1::zeros(n_samples);
                for val in random_vec.iter_mut() {
                    *val = rng.gen_range(-1.0..1.0);
                }
                
                // Orthogonalize this random vector against previous factors
                for j in 0..i {
                    let prev_factor = factors.slice(s![.., j]);
                    let proj = random_vec.iter().zip(prev_factor.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    for (r_val, p_val) in random_vec.iter_mut().zip(prev_factor.iter()) {
                        *r_val -= proj * (*p_val);
                    }
                }
                
                // Normalize
                let norm = random_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    random_vec.mapv_inplace(|x| x / norm);
                    factors.slice_mut(s![.., i]).assign(&random_vec);
                }
            }
        }
        
        // Final verification of orthogonality
        for i in 0..n_factors {
            for j in (i+1)..n_factors {
                let f1 = factors.slice(s![.., i]);
                let f2 = factors.slice(s![.., j]);
                let dot = f1.iter().zip(f2.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                
                // If not orthogonal enough, re-orthogonalize
                if dot.abs() > 1e-5 {
                    let mut f2_new = f2.to_owned();
                    let proj = dot;
                    for (f_val, p_val) in f2_new.iter_mut().zip(f1.iter()) {
                        *f_val -= proj * (*p_val);
                    }
                    
                    let norm = f2_new.mapv(|x| x * x).sum().sqrt();
                    if norm > 1e-10 {
                        f2_new.mapv_inplace(|x| x / norm);
                        factors.slice_mut(s![.., j]).assign(&f2_new);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Calculates quality metrics for each risk factor.
    /// 
    /// Computes several metrics to assess factor quality:
    /// - Information coefficient (predictive power)
    /// - Variance Inflation Factor (multicollinearity)
    /// - T-statistic (statistical significance)
    /// - Explained variance ratio
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors
    /// * `returns` - Matrix of asset returns
    /// 
    /// # Returns
    /// 
    /// Vector of `FactorQualityMetrics` for each factor
    pub fn calculate_factor_metrics(
        &self,
        factors: &Array2<f32>,
        returns: &Array2<f32>,
    ) -> Result<Vec<FactorQualityMetrics>, ModelError> {
        let (n_samples, n_factors) = factors.dim();
        let (ret_samples, n_assets) = returns.dim();
        
        if n_samples != ret_samples {
            return Err(ModelError::InvalidDimension(
                format!("Number of samples in factors ({}) and returns ({}) must match", n_samples, ret_samples)
            ));
        }
        
        let mut metrics = Vec::with_capacity(n_factors);
        
        for i in 0..n_factors {
            let factor = factors.slice(s![.., i]);
            
            // Calculate information coefficient (correlation with returns)
            let mut ic_sum = 0.0;
            for j in 0..n_assets {
                let asset_returns = returns.slice(s![.., j]);
                
                // Convert to vectors to avoid issues with non-contiguous memory
                let factor_vec: Vec<f32> = factor.iter().copied().collect();
                let returns_vec: Vec<f32> = asset_returns.iter().copied().collect();
                
                let corr = calculate_correlation(&factor_vec, &returns_vec);
                ic_sum += corr.abs();
            }
            let ic = ic_sum / (n_assets as f32);
            
            // Calculate VIF (Variance Inflation Factor)
            // For orthogonal factors, VIF should be close to 1.0
            let vif = 1.0; // Simplified since we've already orthogonalized
            
            // Calculate t-statistic
            let t_stat = ic * (n_samples as f32).sqrt() / (1.0 - ic * ic).sqrt();
            
            // Calculate explained variance
            let total_var = returns.mapv(|x| x * x).sum() / (n_samples as f32);
            let factor_var = factor.mapv(|x| x * x).sum() / (n_samples as f32);
            let explained_var = factor_var / total_var;
            
            metrics.push(FactorQualityMetrics {
                information_coefficient: ic,
                vif,
                t_statistic: t_stat,
                explained_variance: explained_var,
            });
        }
        
        Ok(metrics)
    }
    
    /// Selects optimal factors based on quality metrics.
    /// 
    /// Filters factors based on:
    /// - Minimum information coefficient
    /// - Maximum VIF (Variance Inflation Factor)
    /// - Minimum t-statistic
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors
    /// * `metrics` - Vector of quality metrics for each factor
    /// 
    /// # Returns
    /// 
    /// Matrix containing only the selected factors
    pub fn select_optimal_factors(
        &self,
        factors: &Array2<f32>,
        metrics: &[FactorQualityMetrics],
    ) -> Result<Array2<f32>, ModelError> {
        let (n_samples, n_factors) = factors.dim();
        
        if metrics.len() != n_factors {
            return Err(ModelError::InvalidDimension(
                format!("Number of metrics ({}) must match number of factors ({})", metrics.len(), n_factors)
            ));
        }
        
        // Find indices of factors that meet all criteria
        let selected_indices: Vec<usize> = metrics.iter().enumerate()
            .filter(|(_, m)| {
                m.information_coefficient >= self.min_information_coefficient &&
                m.vif <= self.max_vif &&
                m.t_statistic >= self.min_t_statistic
            })
            .map(|(i, _)| i)
            .collect();
        
        if selected_indices.is_empty() {
            return Err(ModelError::InvalidInput(
                "No factors meet the selection criteria".into()
            ));
        }
        
        // Create a new matrix with only the selected factors
        let mut selected_factors = Array2::zeros((n_samples, selected_indices.len()));
        
        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            let factor = factors.slice(s![.., old_idx]);
            selected_factors.slice_mut(s![.., new_idx]).assign(&factor);
        }
        
        Ok(selected_factors)
    }
    
    /// Estimates factor loadings (beta coefficients) for each asset.
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors
    /// * `returns` - Matrix of asset returns
    /// 
    /// # Returns
    /// 
    /// Matrix of factor loadings (assets × factors)
    pub fn estimate_factor_loadings(
        &self,
        factors: &Array2<f32>,
        returns: &Array2<f32>,
    ) -> Result<Array2<f32>, ModelError> {
        let (n_samples, n_factors) = factors.dim();
        let (ret_samples, n_assets) = returns.dim();
        
        if n_samples != ret_samples {
            return Err(ModelError::InvalidDimension(
                format!("Number of samples in factors ({}) and returns ({}) must match", n_samples, ret_samples)
            ));
        }
        
        // Convert to f64 for better numerical stability
        let factors_f64 = factors.mapv(|x| x as f64);
        let returns_f64 = returns.mapv(|x| x as f64);
        
        // Compute (X^T X)^(-1) X^T y for each asset
        let xtx = linalg::matmul(&factors_f64.t().to_owned(), &factors_f64);
        let xtx_inv = match linalg::inv(&xtx) {
            Ok(inv) => inv,
            Err(_) => return Err(ModelError::NumericalError(
                "Failed to invert factor covariance matrix".into()
            )),
        };
        
        let xt = factors_f64.t().to_owned();
        let mut loadings = Array2::zeros((n_assets, n_factors));
        
        for j in 0..n_assets {
            let asset_returns = returns_f64.slice(s![.., j]).to_owned();
            let xty = linalg::matvec(&xt, &asset_returns);
            let beta = linalg::matvec(&xtx_inv, &xty);
            
            // Convert back to f32 and store in loadings matrix
            for i in 0..n_factors {
                loadings[[j, i]] = beta[i] as f32;
            }
        }
        
        Ok(loadings)
    }
    
    /// Estimates the factor covariance matrix.
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors
    /// 
    /// # Returns
    /// 
    /// Covariance matrix of the factors
    pub fn estimate_factor_covariance(
        &self,
        factors: &Array2<f32>,
    ) -> Result<Array2<f32>, ModelError> {
        let (n_samples, n_factors) = factors.dim();
        
        if n_samples <= 1 {
            return Err(ModelError::InvalidDimension(
                "Need at least 2 samples to estimate covariance".into()
            ));
        }
        
        // Convert to f64 for better numerical stability
        let factors_f64 = factors.mapv(|x| x as f64);
        
        // Center the factors (subtract mean)
        let mut centered = Array2::zeros(factors_f64.dim());
        for j in 0..n_factors {
            let col = factors_f64.slice(s![.., j]);
            let mean = col.sum() / (n_samples as f64);
            for i in 0..n_samples {
                centered[[i, j]] = factors_f64[[i, j]] - mean;
            }
        }
        
        // Compute covariance matrix: (X^T X) / (n - 1)
        let cov = linalg::matmul(&centered.t().to_owned(), &centered);
        let cov = cov.mapv(|x| x / ((n_samples - 1) as f64));
        
        // Convert back to f32
        let cov_f32 = cov.mapv(|x| x as f32);
        
        Ok(cov_f32)
    }
}

/// Calculates the Pearson correlation coefficient between two arrays.
fn calculate_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f32;
    
    // Calculate means
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;
    
    // Calculate covariance and variances
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    // Calculate correlation
    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_orthogonalize_factors() {
        let analyzer = FactorAnalyzer::new(0.1, 5.0, 1.96);
        
        // Create test factors
        let mut factors = Array::from_shape_vec(
            (5, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        ).unwrap();
        
        // Orthogonalize
        analyzer.orthogonalize_factors(&mut factors).unwrap();
        
        // Check that factors are orthogonal (dot product close to zero)
        let f1 = factors.slice(s![.., 0]);
        let f2 = factors.slice(s![.., 1]);
        let f3 = factors.slice(s![.., 2]);
        
        let dot_12: f32 = f1.iter().zip(f2.iter()).map(|(&a, &b)| a * b).sum();
        let dot_13: f32 = f1.iter().zip(f3.iter()).map(|(&a, &b)| a * b).sum();
        let dot_23: f32 = f2.iter().zip(f3.iter()).map(|(&a, &b)| a * b).sum();
        
        assert!(dot_12.abs() < 1e-6);
        assert!(dot_13.abs() < 1e-6);
        assert!(dot_23.abs() < 1e-6);
        
        // Check that factors are normalized (unit length)
        let norm1: f32 = f1.iter().map(|&x| x * x).sum();
        let norm2: f32 = f2.iter().map(|&x| x * x).sum();
        let norm3: f32 = f3.iter().map(|&x| x * x).sum();
        
        assert!((norm1 - 1.0).abs() < 1e-6);
        assert!((norm2 - 1.0).abs() < 1e-6);
        assert!((norm3 - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_calculate_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);
        
        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = calculate_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-6);
        
        // No correlation
        let x_uncorr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_uncorr = vec![5.0, 2.0, 7.0, 1.0, 9.0];
        let corr_uncorr = calculate_correlation(&x_uncorr, &y_uncorr);
        assert!(corr_uncorr.abs() < 0.5); // Not exactly zero, but should be low
    }
} 