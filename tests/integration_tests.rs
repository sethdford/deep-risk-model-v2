#[cfg(test)]
mod tests {
    use deep_risk_model::prelude::{
        ModelError,
        MarketData,
        RiskModel,
        TransformerRiskModel,
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
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate synthetic data with n_assets = d_model
        let n_assets = d_model;
        let market_data = generate_test_data(n_assets, 100);

        // Generate risk factors
        let factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = 100 - window_size + 1;
        assert_eq!(factors.factors().shape()[0], expected_samples);
        assert_eq!(factors.factors().shape()[1], d_model);

        // Estimate covariance
        let cov = model.estimate_covariance(&market_data).await?;
        assert_eq!(cov.shape(), &[n_assets, n_assets]);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_different_sizes() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate synthetic data with n_assets = d_model
        let n_assets = d_model;
        let market_data = generate_test_data(n_assets, 150);

        // Generate risk factors
        let factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = 150 - window_size + 1;
        assert_eq!(factors.factors().shape()[0], expected_samples);
        assert_eq!(factors.factors().shape()[1], d_model);

        // Estimate covariance
        let cov = model.estimate_covariance(&market_data).await?;
        assert_eq!(cov.shape(), &[n_assets, n_assets]);

        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate synthetic data with wrong dimensions
        let wrong_market_data = generate_test_data(d_model + 5, 100); // Wrong number of assets

        // Test error handling for risk factor generation and covariance estimation
        assert!(model.generate_risk_factors(&wrong_market_data).await.is_err());
        assert!(model.estimate_covariance(&wrong_market_data).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_model_full_lifecycle() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate synthetic market data with n_assets = d_model
        let n_assets = d_model;
        let market_data = generate_test_data(n_assets, 100);

        // Generate risk factors and verify dimensions
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = 100 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        assert_eq!(risk_factors.factors().shape()[1], d_model);

        // Estimate covariance matrix and verify properties
        let covariance = model.estimate_covariance(&market_data).await?;
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);
        
        // Verify covariance matrix symmetry and positive semi-definiteness
        for i in 0..n_assets {
            for j in 0..n_assets {
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
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Test with incorrect number of assets
        let invalid_data = generate_test_data(d_model + 5, 100);
        assert!(model.generate_risk_factors(&invalid_data).await.is_err());
        assert!(model.estimate_covariance(&invalid_data).await.is_err());

        // Test with valid data
        let valid_data = generate_test_data(d_model, 100);
        assert!(model.generate_risk_factors(&valid_data).await.is_ok());
        assert!(model.estimate_covariance(&valid_data).await.is_ok());

        Ok(())
    }

    #[tokio::test]
    async fn test_model_performance() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate synthetic data with n_assets = d_model
        let n_assets = d_model;
        let market_data = generate_test_data(n_assets, 250);

        // Measure risk factor generation time
        let start = Instant::now();
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let risk_factor_time = start.elapsed();

        // Measure covariance estimation time
        let start = Instant::now();
        let covariance = model.estimate_covariance(&market_data).await?;
        let covariance_time = start.elapsed();

        // Verify outputs
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = 250 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        assert_eq!(risk_factors.factors().shape()[1], d_model);
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);

        // Print performance metrics
        println!("Performance metrics for {} stocks, 250 time steps:", n_assets);
        println!("Risk factor generation time: {:?}", risk_factor_time);
        println!("Covariance estimation time: {:?}", covariance_time);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_noisy_data() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        
        // Generate data with high noise level
        let n_assets = d_model;
        let market_data = generate_test_data(n_assets, 250);
        
        // Verify model can handle noisy data
        let risk_factors = model.generate_risk_factors(&market_data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = 250 - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        assert_eq!(risk_factors.factors().shape()[1], d_model);
        
        let covariance = model.estimate_covariance(&market_data).await?;
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_model_dimension_mismatch() -> Result<(), ModelError> {
        // Initialize model with d_model = 64 (required by TransformerRiskModel)
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        // Create data with mismatched dimensions between features and returns
        // This data is not used in the test, but is kept for documentation purposes
        let _features = Array2::<f32>::random((250, d_model), Normal::new(0.0, 1.0).unwrap());
        let _returns = Array2::<f32>::random((250, d_model + 1), Normal::new(0.0, 0.1).unwrap());
        
        // Create data with mismatched dimensions between d_model and features
        let wrong_features = Array2::<f32>::random((250, d_model + 5), Normal::new(0.0, 1.0).unwrap());
        let matching_returns = Array2::<f32>::random((250, d_model + 5), Normal::new(0.0, 0.1).unwrap());
        let wrong_data = MarketData::new(matching_returns, wrong_features);
        
        let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
        
        // This should fail because the feature dimension doesn't match d_model
        assert!(model.generate_risk_factors(&wrong_data).await.is_err());
        
        Ok(())
    }
}