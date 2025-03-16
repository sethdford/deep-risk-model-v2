use deep_risk_model::{
    ModelError,
    MarketData,
    ModelConfig,
};
use ndarray::{Array2};
use ndarray_rand::{RandomExt, rand_distr::Normal};

/// Creates a default model configuration for testing
pub fn create_test_config() -> ModelConfig {
    ModelConfig::new(64, 8, 3, 0.1)
}

/// Generates synthetic market data with specified parameters
pub fn create_synthetic_data(
    num_stocks: usize,
    seq_len: usize,
    noise_level: f32,
) -> Result<MarketData, ModelError> {
    let features = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, noise_level).unwrap());
    let returns = Array2::<f32>::random((seq_len, num_stocks), Normal::new(0.0, 0.1).unwrap());
    Ok(MarketData::new(returns, features))
}