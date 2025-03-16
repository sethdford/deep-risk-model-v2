use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use deep_risk_model::{
    error::ModelError,
    model::DeepRiskModel,
    types::{
        MarketData,
        ModelConfig,
        RiskModel,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(num_stocks: usize, seq_len: usize) -> MarketData {
        let features = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 0.1).unwrap());
        MarketData::new(returns, features)
    }

    #[tokio::test]
    async fn test_basic_model_operations() -> Result<(), ModelError> {
        // Generate synthetic market data
        let n_samples = 100;
        let n_assets = 10;
        let features = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);

        // Initialize model
        let mut model = DeepRiskModel::new(10, 5)?;

        // Train model
        model.train(&data).await?;

        // Generate risk factors
        let risk_factors = model.generate_risk_factors(&data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = n_samples - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(risk_factors.factors().shape()[1] <= 16); // d_model is rounded up to next multiple of 8 (10 -> 16)

        // Estimate covariance
        let covariance = model.estimate_covariance(&data).await?;
        assert_eq!(covariance.shape(), &[10, 10]);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_different_sizes() -> Result<(), ModelError> {
        // Generate synthetic market data with different dimensions
        let n_samples = 150;
        let n_assets = 20;
        let features = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);

        // Initialize model
        let mut model = DeepRiskModel::new(20, 10)?;

        // Train model
        model.train(&data).await?;

        // Generate risk factors
        let risk_factors = model.generate_risk_factors(&data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = n_samples - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(risk_factors.factors().shape()[1] <= 24); // d_model is rounded up to next multiple of 8 (20 -> 24)

        // Estimate covariance
        let covariance = model.estimate_covariance(&data).await?;
        assert_eq!(covariance.shape(), &[20, 20]);

        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<(), ModelError> {
        // Initialize model
        let mut model = DeepRiskModel::new(10, 5)?;

        // Test with invalid data dimensions
        let features = Array2::random((100, 20), Normal::new(0.0, 1.0).unwrap()); // Wrong number of assets
        let returns = Array2::random((100, 10), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);

        // Should return error due to dimension mismatch
        assert!(model.generate_risk_factors(&data).await.is_err());

        Ok(())
    }
} 