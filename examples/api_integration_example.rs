use deep_risk_model::{error::ModelError, model::DeepRiskModel};
use ndarray::{Array, Array3, ArrayD, IxDyn, Array2};
use serde::{Deserialize, Serialize};
use tokio;
use std::time::Instant;
use deep_risk_model::{
    types::{ModelConfig, MarketData, RiskModel},
};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Serialize, Deserialize)]
struct APIConfig {
    input_size: i64,
    hidden_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_layers: i64,
    output_size: i64,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            input_size: 32,
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            num_layers: 2,
            output_size: 16,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    features: Vec<Vec<f32>>,
    returns: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    factors: Vec<Vec<f32>>,
    covariance: Vec<Vec<f32>>,
}

fn array_to_vec(array: &ArrayD<f32>) -> Vec<Vec<f32>> {
    let shape = array.shape();
    let mut result = Vec::new();
    
    match shape.len() {
        2 => {
            // Handle 2D array directly
            for i in 0..shape[0] {
                let mut row = Vec::new();
                for j in 0..shape[1] {
                    row.push(array[[i, j]]);
                }
                result.push(row);
            }
        },
        3 => {
            // For 3D array, take the last 2D slice
            let last_slice_idx = shape[0] - 1;
            for i in 0..shape[1] {
                let mut row = Vec::new();
                for j in 0..shape[2] {
                    row.push(array[[last_slice_idx, i, j]]);
                }
                result.push(row);
            }
        },
        _ => panic!("Unsupported array dimension"),
    }
    
    result
}

fn array2_to_vec(array: &Array2<f32>) -> Vec<Vec<f32>> {
    let (rows, cols) = array.dim();
    let mut result = Vec::new();
    for i in 0..rows {
        let mut row = Vec::new();
        for j in 0..cols {
            row.push(array[[i, j]]);
        }
        result.push(row);
    }
    result
}

async fn test_with_real_data(n_assets: usize, n_factors: usize, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = DeepRiskModel::new(n_assets, n_factors)?;
    model.train(&market_data).await?;
    
    let factors = model.generate_risk_factors(&market_data).await?;
    let covariance = model.estimate_covariance(&market_data).await?;
    
    println!("Generated factors shape: {:?}", factors.factors().shape());
    println!("Estimated covariance shape: {:?}", covariance.shape());
    
    Ok(())
}

async fn test_performance(n_assets: usize, n_factors: usize, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = DeepRiskModel::new(n_assets, n_factors)?;
    
    // Train the model
    let start = std::time::Instant::now();
    model.train(&market_data).await?;
    let training_time = start.elapsed();
    println!("Training time: {:?}", training_time);
    
    // Generate factors
    let start = std::time::Instant::now();
    let factors = model.generate_risk_factors(&market_data).await?;
    let factor_time = start.elapsed();
    println!("Factor generation time: {:?}", factor_time);
    println!("Generated factors shape: {:?}", factors.factors().shape());
    
    // Estimate covariance
    let start = std::time::Instant::now();
    let covariance = model.estimate_covariance(&market_data).await?;
    let covariance_time = start.elapsed();
    println!("Covariance estimation time: {:?}", covariance_time);
    println!("Estimated covariance shape: {:?}", covariance.shape());
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create synthetic market data
    let n_samples = 100;
    let n_assets = 10;
    let returns = Array2::from_shape_vec((n_samples, n_assets), vec![0.0; n_samples * n_assets])?;
    let features = Array2::from_shape_vec((n_samples, n_assets), vec![0.0; n_samples * n_assets])?;
    let market_data = MarketData::new(returns, features);

    // Initialize model
    let mut model = DeepRiskModel::new(n_assets, 5)?;

    // Basic model operations
    println!("Training model...");
    let start = Instant::now();
    model.train(&market_data).await?;
    println!("Training completed in {:?}", start.elapsed());

    println!("\nGenerating risk factors...");
    let start = Instant::now();
    let factors = model.generate_risk_factors(&market_data).await?;
    println!("Risk factors generated in {:?}", start.elapsed());
    println!("Risk factors shape: {:?}", factors.factors().shape());

    println!("\nEstimating covariance...");
    let start = Instant::now();
    let covariance = model.estimate_covariance(&market_data).await?;
    println!("Covariance estimated in {:?}", start.elapsed());
    println!("Covariance shape: {:?}", covariance.shape());

    // Error handling example
    println!("\nTesting error handling with incorrect data...");
    let incorrect_returns = Array2::from_shape_vec((50, 5), vec![0.0; 250])?;
    let incorrect_features = Array2::from_shape_vec((50, 5), vec![0.0; 250])?;
    let incorrect_market_data = MarketData::new(incorrect_returns, incorrect_features);

    match model.train(&incorrect_market_data).await {
        Ok(_) => println!("Training succeeded (unexpected)"),
        Err(e) => println!("Training failed as expected: {}", e),
    }

    match model.generate_risk_factors(&incorrect_market_data).await {
        Ok(_) => println!("Risk factor generation succeeded (unexpected)"),
        Err(e) => println!("Risk factor generation failed as expected: {}", e),
    }

    match model.estimate_covariance(&incorrect_market_data).await {
        Ok(_) => println!("Covariance estimation succeeded (unexpected)"),
        Err(e) => println!("Covariance estimation failed as expected: {}", e),
    }

    Ok(())
}