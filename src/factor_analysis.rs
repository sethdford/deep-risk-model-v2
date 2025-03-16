use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s, Axis};
#[cfg(feature = "blas-enabled")]
use ndarray_linalg::Solve;
use crate::error::ModelError;

#[cfg(feature = "no-blas")]
use crate::fallback;

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
    /// Minimum explained variance ratio to keep a factor
    pub(crate) min_explained_variance: f32,
    /// Maximum acceptable VIF (variance inflation factor)
    pub(crate) max_vif: f32,
    /// Significance level for t-tests
    pub(crate) significance_level: f32,
}

// Implement Send and Sync for FactorAnalyzer
// This is safe because FactorAnalyzer only contains primitive types (f32)
// that are already Send and Sync
unsafe impl Send for FactorAnalyzer {}
unsafe impl Sync for FactorAnalyzer {}

impl FactorAnalyzer {
    /// Creates a new FactorAnalyzer with specified quality thresholds.
    /// 
    /// # Arguments
    /// 
    /// * `min_explained_variance` - Minimum variance ratio (0 to 1) a factor must explain
    /// * `max_vif` - Maximum allowed Variance Inflation Factor (typically 5-10)
    /// * `significance_level` - T-statistic threshold for significance (typically 1.96 for 95% confidence)
    pub fn new(min_explained_variance: f32, max_vif: f32, significance_level: f32) -> Self {
        Self {
            min_explained_variance,
            max_vif,
            significance_level,
        }
    }

    /// Orthogonalizes risk factors using the Gram-Schmidt process.
    /// 
    /// This ensures that risk factors are uncorrelated with each other, which is
    /// important for stable risk decomposition and factor analysis.
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Matrix of risk factors to orthogonalize (modified in-place)
    pub fn orthogonalize_factors(&self, factors: &mut Array2<f32>) -> Result<(), ModelError> {
        let (n_samples, n_factors) = factors.dim();
        
        // Normalize first factor
        let norm = factors.slice(s![.., 0]).mapv(|x| x * x).sum().sqrt();
        if norm > 1e-10 {
            factors.slice_mut(s![.., 0]).mapv_inplace(|x| x / norm);
        }
        
        // Orthogonalize remaining factors
        for i in 1..n_factors {
            let mut factor = factors.slice(s![.., i]).to_owned();
            
            // Subtract projections onto previous factors
            for j in 0..i {
                let prev_factor = factors.slice(s![.., j]);
                
                // Calculate projection coefficient (dot product)
                #[cfg(not(feature = "no-blas"))]
                let proj = factor.dot(&prev_factor) / prev_factor.dot(&prev_factor);
                
                #[cfg(feature = "no-blas")]
                let proj = {
                    let dot1 = factor.iter().zip(prev_factor.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    let dot2 = prev_factor.iter().map(|&x| x * x).sum::<f32>();
                    if dot2 < 1e-10 { 0.0 } else { dot1 / dot2 }
                };
                
                // Subtract projection
                factor = &factor - &(&prev_factor * proj);
            }
            
            // Normalize
            let norm = factor.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                factor.mapv_inplace(|x| x / norm);
                factors.slice_mut(s![.., i]).assign(&factor);
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
    pub fn calculate_metrics(
        &self,
        factors: &Array2<f32>,
        returns: &Array2<f32>,
    ) -> Result<Vec<FactorQualityMetrics>, ModelError> {
        let (n_samples, n_factors) = factors.dim();
        let mut metrics = Vec::with_capacity(n_factors);
        
        for i in 0..n_factors {
            let factor = factors.slice(s![.., i]);
            
            // Calculate information coefficient (correlation with returns)
            let mut ic = 0.0;
            for j in 0..returns.shape()[1] {
                let asset_returns = returns.slice(s![.., j]);
                ic += self.correlation(&factor, &asset_returns)?;
            }
            ic /= returns.shape()[1] as f32;
            
            // Calculate variance inflation factor
            let vif = self.calculate_vif(factors, i)?;
            
            // Calculate t-statistic
            let t_stat = self.calculate_t_statistic(&factor, returns)?;
            
            // Calculate explained variance
            let exp_var = self.calculate_explained_variance(&factor, returns)?;
            
            metrics.push(FactorQualityMetrics {
                information_coefficient: ic,
                vif,
                t_statistic: t_stat,
                explained_variance: exp_var,
            });
        }
        
        Ok(metrics)
    }

    /// Selects optimal risk factors based on quality metrics.
    /// 
    /// Factors are selected if they meet all criteria:
    /// - Explained variance >= min_explained_variance
    /// - VIF <= max_vif
    /// - |t-statistic| > significance_level
    /// 
    /// # Arguments
    /// 
    /// * `factors` - Original factor matrix
    /// * `metrics` - Quality metrics for each factor
    pub fn select_optimal_factors(
        &self,
        factors: &Array2<f32>,
        metrics: &[FactorQualityMetrics],
    ) -> Result<Array2<f32>, ModelError> {
        let mut selected_indices = Vec::new();
        
        // Select factors that meet all criteria
        for (i, metric) in metrics.iter().enumerate() {
            if metric.explained_variance >= self.min_explained_variance
                && metric.vif <= self.max_vif
                && metric.t_statistic.abs() > self.significance_level
            {
                selected_indices.push(i);
            }
        }
        
        // Create new array with selected factors
        let n_samples = factors.shape()[0];
        let n_selected = selected_indices.len();
        let mut selected_factors = Array2::zeros((n_samples, n_selected));
        
        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            selected_factors.slice_mut(s![.., new_idx])
                .assign(&factors.slice(s![.., old_idx]));
        }
        
        Ok(selected_factors)
    }

    /// Calculate correlation between two arrays
    fn correlation(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> Result<f32, ModelError> {
        let n_x = x.len();
        let n_y = y.len();
        let n: usize = n_x.min(n_y);
        
        if n == 0 {
            return Ok(0.0);
        }
        
        // Explicitly calculate mean to avoid Option issues
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for i in 0..n {
            sum_x += x[i];
            sum_y += y[i];
        }
        let mean_x = sum_x / (n as f32);
        let mean_y = sum_y / (n as f32);
        
        // Explicitly calculate standard deviation
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        for i in 0..n {
            sum_sq_x += (x[i] - mean_x).powi(2);
            sum_sq_y += (y[i] - mean_y).powi(2);
        }
        let std_x = (sum_sq_x / (n as f32)).sqrt();
        let std_y = (sum_sq_y / (n as f32)).sqrt();
        
        if std_x < 1e-10 || std_y < 1e-10 {
            return Ok(0.0);
        }
        
        // Calculate covariance
        let mut cov = 0.0;
        for i in 0..n {
            cov += (x[i] - mean_x) * (y[i] - mean_y);
        }
        
        // Convert n to f32 before subtraction to avoid type errors
        let n_f32 = n as f32;
        cov /= (n_f32 - 1.0);
        
        Ok(cov / (std_x * std_y))
    }

    /// Calculate variance inflation factor
    fn calculate_vif(
        &self,
        factors: &Array2<f32>,
        factor_idx: usize,
    ) -> Result<f32, ModelError> {
        let n_factors = factors.shape()[1];
        if n_factors < 2 {
            return Ok(1.0);
        }
        
        // Create array of other factors
        let mut other_factors = Vec::with_capacity(n_factors - 1);
        for i in 0..n_factors {
            if i != factor_idx {
                other_factors.push(factors.slice(s![.., i]));
            }
        }
        
        // Calculate R-squared of factor_idx with all other factors
        let r_squared = self.calculate_r_squared(
            &factors.slice(s![.., factor_idx]),
            &other_factors,
        )?;
        
        // VIF = 1 / (1 - R²)
        if r_squared >= 1.0 {
            Ok(f32::MAX)
        } else {
            Ok(1.0 / (1.0 - r_squared))
        }
    }

    /// Calculate t-statistic for factor significance
    fn calculate_t_statistic(
        &self,
        factor: &ArrayView1<f32>,
        returns: &Array2<f32>,
    ) -> Result<f32, ModelError> {
        let n_samples = factor.len();
        let n_assets = returns.shape()[1];
        
        // Calculate mean t-statistic across all assets
        let mut t_stat_sum = 0.0;
        for j in 0..n_assets {
            let asset_returns = returns.slice(s![.., j]);
            let corr = self.correlation(factor, &asset_returns)?;
            let t = corr * ((n_samples - 2) as f32).sqrt() / (1.0 - corr * corr).sqrt();
            t_stat_sum += t;
        }
        
        Ok(t_stat_sum / n_assets as f32)
    }

    /// Calculate explained variance ratio for a factor
    fn calculate_explained_variance(
        &self,
        factor: &ArrayView1<f32>,
        returns: &Array2<f32>,
    ) -> Result<f32, ModelError> {
        let n_assets = returns.shape()[1];
        let mut exp_var_sum = 0.0;
        
        for j in 0..n_assets {
            let asset_returns = returns.slice(s![.., j]);
            let corr = self.correlation(factor, &asset_returns)?;
            exp_var_sum += corr * corr;  // R² is explained variance
        }
        
        Ok(exp_var_sum / n_assets as f32)
    }

    /// Calculate R-squared (coefficient of determination)
    fn calculate_r_squared(
        &self,
        target: &ArrayView1<f32>,
        predictors: &[ArrayView1<f32>],
    ) -> Result<f32, ModelError> {
        if predictors.is_empty() {
            return Ok(0.0);
        }
        
        let n_samples = target.len();
        let n_predictors = predictors.len();
        
        // Create design matrix X (with intercept)
        let mut x = Array2::ones((n_samples, n_predictors + 1));
        for (j, predictor) in predictors.iter().enumerate() {
            x.slice_mut(s![.., j + 1]).assign(predictor);
        }
        
        // Calculate coefficients using OLS: β = (X'X)^(-1)X'y
        #[cfg(not(feature = "no-blas"))]
        let coefficients = {
            let xtx = x.t().dot(&x);
            let xty = x.t().dot(target);
            xtx.solve(&xty)?
        };
        
        #[cfg(feature = "no-blas")]
        let coefficients = {
            let xtx = x.t().dot(&x);
            let xty = x.t().dot(target);
            
            // Use our fallback matrix inversion
            let xtx_inv = fallback::inv(&xtx)?;
            
            // Manually compute (X'X)^(-1)X'y
            let result = fallback::matmul(&xtx_inv, &xty.into_shape((n_predictors + 1, 1))?)?;
            result.column(0).to_owned()
        };
        
        // Calculate predicted values
        let y_pred = x.dot(&coefficients);
        
        // Calculate R-squared = 1 - SSR/SST
        let y_mean = target.mean().unwrap_or(0.0);
        
        let mut ss_total = 0.0;
        let mut ss_residual = 0.0;
        
        for i in 0..n_samples {
            ss_total += (target[i] - y_mean).powi(2);
            ss_residual += (target[i] - y_pred[i]).powi(2);
        }
        
        if ss_total < 1e-10 {
            Ok(0.0)
        } else {
            Ok((ss_total - ss_residual) / ss_total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;

    #[test]
    fn test_orthogonalization() -> Result<(), ModelError> {
        let analyzer = FactorAnalyzer::new(0.1, 5.0, 1.96);
        let mut factors = Array::random((100, 5), StandardNormal);
        
        analyzer.orthogonalize_factors(&mut factors)?;
        
        // Check orthogonality
        for i in 0..5 {
            for j in 0..i {
                let factor_i = factors.slice(s![.., i]).to_owned().into_shape(100).unwrap();
                let factor_j = factors.slice(s![.., j]).to_owned().into_shape(100).unwrap();
                let dot_product = factor_i.dot(&factor_j);
                assert!(dot_product.abs() < 1e-6);
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_factor_metrics() -> Result<(), ModelError> {
        let analyzer = FactorAnalyzer::new(0.1, 5.0, 1.96);
        let factors = Array::random((100, 3), StandardNormal);
        let returns = Array::random((100, 5), StandardNormal);
        
        let metrics = analyzer.calculate_metrics(&factors, &returns)?;
        
        assert_eq!(metrics.len(), 3);
        for metric in metrics {
            assert!(metric.information_coefficient.abs() <= 1.0);
            assert!(metric.vif >= 1.0);
            assert!(metric.explained_variance >= 0.0 && metric.explained_variance <= 1.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_factor_selection() -> Result<(), ModelError> {
        // Skip this test when no-blas feature is enabled
        #[cfg(feature = "no-blas")]
        {
            println!("Skipping test_factor_selection in no-blas mode");
            return Ok(());
        }
        
        #[cfg(not(feature = "no-blas"))]
        {
            let analyzer = FactorAnalyzer::new(0.1, 5.0, 1.96);
            let factors = Array::random((100, 5), StandardNormal);
            let returns = Array::random((100, 3), StandardNormal);
            
            let metrics = analyzer.calculate_metrics(&factors, &returns)?;
            let selected = analyzer.select_optimal_factors(&factors, &metrics)?;
            
            assert!(selected.shape()[1] <= factors.shape()[1]);
            assert_eq!(selected.shape()[0], factors.shape()[0]);
        }
        
        Ok(())
    }
} 