#[cfg(test)]
mod tests {
    use deep_risk_model::prelude::{
        MarketData,
        RiskModel,
        ModelError,
        TransformerRiskModel,
    };
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    fn generate_test_data(num_stocks: usize, seq_len: usize) -> MarketData {
        let features = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 0.1).unwrap());
        MarketData::new(returns, features)
    }

    #[tokio::test]
    async fn test_basic_model_operations() -> Result<(), ModelError> {
        // Generate synthetic market data
        let n_samples = 100;
        let n_assets = 64; // Set to d_model to match the transformer's expected dimension
        let features = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);

        // Create a TransformerRiskModel directly with the correct parameters
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let transformer_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate risk factors using the transformer model
        let risk_factors = transformer_model.generate_risk_factors(&data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = n_samples - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        assert_eq!(risk_factors.factors().shape()[1], d_model);

        // Estimate covariance
        let covariance = transformer_model.estimate_covariance(&data).await?;
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);

        Ok(())
    }

    #[tokio::test]
    async fn test_model_with_different_sizes() -> Result<(), ModelError> {
        // Generate synthetic market data with different dimensions
        let n_samples = 150;
        let n_assets = 64; // Set to d_model to match the transformer's expected dimension
        
        // Create a TransformerRiskModel directly with the correct parameters
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        
        // Create features and returns with the correct dimensions
        let features = Array2::random((n_samples, n_assets), Normal::new(0.0, 1.0).unwrap());
        let returns = Array2::random((n_samples, n_assets), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);
        
        let transformer_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Generate risk factors
        let risk_factors = transformer_model.generate_risk_factors(&data).await?;
        let window_size = 5; // Default max_seq_len in TransformerConfig
        let expected_samples = n_samples - window_size + 1;
        assert_eq!(risk_factors.factors().shape()[0], expected_samples);
        assert_eq!(risk_factors.factors().shape()[1], d_model);

        // Estimate covariance
        let covariance = transformer_model.estimate_covariance(&data).await?;
        assert_eq!(covariance.shape(), &[n_assets, n_assets]);

        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<(), ModelError> {
        // Create a TransformerRiskModel directly with the correct parameters
        let d_model = 64;
        let n_heads = 8;
        let d_ff = 256;
        let n_layers = 3;
        let transformer_model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;

        // Test with invalid data dimensions
        let n_assets = 10; // Different from the features dimension below
        let features = Array2::random((100, 20), Normal::new(0.0, 1.0).unwrap()); // Wrong number of assets
        let returns = Array2::random((100, n_assets), Normal::new(0.0, 0.1).unwrap());
        let data = MarketData::new(returns, features);

        // Should return error due to dimension mismatch
        assert!(transformer_model.generate_risk_factors(&data).await.is_err());

        Ok(())
    }
} 