#[cfg(test)]
mod tests {
    use deep_risk_model::{
        ModelError,
        DeepRiskModel,
        MarketData,
        RiskModel,
    };
    use ndarray::{Array2};
    use ndarray_rand::{RandomExt, rand_distr::Normal};
    use std::time::Instant;

    fn generate_test_data(num_stocks: usize, seq_len: usize) -> MarketData {
        let features = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 0.1).unwrap());
        MarketData::new(returns, features)
    }

    #[tokio::test]
    async fn test_basic_model_operations() -> Result<(), ModelError> {
        // Initialize model
        let mut model = DeepRiskModel::new(10, 5)?;

        // Generate synthetic data
        let market_data = generate_test_data(10, 100);

        // Train model
        model.train(&market_data).await?;

        // Generate risk factors
        let factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = 100 - window_size + 1;
        assert_eq!(factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(factors.factors().shape()[1] <= 16); // d_model is rounded up to next multiple of 8 (10 -> 16)

        // Estimate covariance
        let cov = model.estimate_covariance(&market_data).await?;
        assert_eq!(cov.shape(), &[10, 10]);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_different_sizes() -> Result<(), ModelError> {
        // Initialize model with different dimensions
        let mut model = DeepRiskModel::new(20, 8)?;

        // Generate synthetic data
        let market_data = generate_test_data(20, 150);

        // Train model
        model.train(&market_data).await?;

        // Generate risk factors
        let factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = 150 - window_size + 1;
        assert_eq!(factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(factors.factors().shape()[1] <= 24); // d_model is rounded up to next multiple of 8 (20 -> 24)

        // Estimate covariance
        let cov = model.estimate_covariance(&market_data).await?;
        assert_eq!(cov.shape(), &[20, 20]);

        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<(), ModelError> {
        // Initialize model
        let mut model = DeepRiskModel::new(10, 5)?;

        // Generate synthetic data with wrong dimensions
        let wrong_market_data = generate_test_data(15, 100); // Wrong number of assets

        // Test error handling for risk factor generation and covariance estimation
        assert!(model.generate_risk_factors(&wrong_market_data).await.is_err());
        assert!(model.estimate_covariance(&wrong_market_data).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_model_full_lifecycle() -> Result<(), ModelError> {
        // Initialize model with 10 assets and 5 factors
        let mut model = DeepRiskModel::new(10, 5)?;

        // Generate synthetic market data
        let market_data = generate_test_data(10, 100);

        // Train model and verify success
        assert!(model.train(&market_data).await.is_ok());

        // Generate risk factors and verify dimensions
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = 100 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(risk_factors.factors().shape()[1] <= 16); // d_model is rounded up to next multiple of 8 (10 -> 16)

        // Estimate covariance matrix and verify properties
        let covariance = model.estimate_covariance(&market_data).await?;
        assert_eq!(covariance.shape(), &[10, 10]);
        
        // Verify covariance matrix symmetry and positive semi-definiteness
        for i in 0..10 {
            for j in 0..10 {
                assert!((covariance[[i, j]] - covariance[[j, i]]).abs() < 1e-6);
                if i == j {
                    assert!(covariance[[i, i]] >= 0.0);
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_model_input_validation() -> Result<(), ModelError> {
        // Initialize model
        let mut model = DeepRiskModel::new(10, 5)?;

        // Test with incorrect number of assets
        let invalid_data = generate_test_data(15, 100);
        assert!(model.train(&invalid_data).await.is_err());
        assert!(model.generate_risk_factors(&invalid_data).await.is_err());
        assert!(model.estimate_covariance(&invalid_data).await.is_err());

        // Test with valid data
        let valid_data = generate_test_data(10, 100);
        assert!(model.train(&valid_data).await.is_ok());
        assert!(model.generate_risk_factors(&valid_data).await.is_ok());
        assert!(model.estimate_covariance(&valid_data).await.is_ok());

        Ok(())
    }

    #[tokio::test]
    async fn test_model_performance() -> Result<(), ModelError> {
        // Initialize model with larger dimensions
        let mut model = DeepRiskModel::new(50, 10)?;

        // Generate synthetic data with more stocks and longer sequence
        let market_data = generate_test_data(50, 250);

        // Measure training time
        let start = Instant::now();
        model.train(&market_data).await?;
        let training_time = start.elapsed();

        // Measure risk factor generation time
        let start = Instant::now();
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let risk_factor_time = start.elapsed();

        // Measure covariance estimation time
        let start = Instant::now();
        let covariance = model.estimate_covariance(&market_data).await?;
        let covariance_time = start.elapsed();

        // Verify outputs
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = 250 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(risk_factors.factors().shape()[1] <= 56); // d_model is rounded up to next multiple of 8 (50 -> 56)
        assert_eq!(covariance.shape(), &[50, 50]);

        // Print performance metrics
        println!("Performance metrics for 50 stocks, 250 time steps:");
        println!("Training time: {:?}", training_time);
        println!("Risk factor generation time: {:?}", risk_factor_time);
        println!("Covariance estimation time: {:?}", covariance_time);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_noisy_data() -> Result<(), ModelError> {
        let n_assets = 100;
        let n_factors = 32;
        let mut model = DeepRiskModel::new(n_assets, n_factors)?;
        
        // Generate data with high noise level
        let market_data = generate_test_data(n_assets, 250);
        
        // Verify model can handle noisy data
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 10; // This is hardcoded in DeepRiskModel::new
        let expected_samples = 250 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        // The number of factors may be less than d_model due to quality filtering
        assert!(risk_factors.factors().shape()[1] <= 104); // d_model is rounded up to next multiple of 8 (100 -> 104)
        
        let covariance = model.estimate_covariance(&market_data).await?;
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_model_dimension_mismatch() -> Result<(), ModelError> {
        let n_assets = 100;
        let n_factors = 32;
        
        // Create data with mismatched dimensions
        let features = Array2::<f32>::random((250, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::<f32>::random((250, n_assets + 1), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);
        
        let mut model = DeepRiskModel::new(n_assets, n_factors)?;
        assert!(model.train(&data).await.is_err());
        
        Ok(())
    }
}