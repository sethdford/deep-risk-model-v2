use deep_risk_model::prelude::*;
use serde_json::{json, Value};
use std::io::{self, Read};
use std::error::Error;
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Read input from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    
    // Parse the JSON input
    let data: Value = serde_json::from_str(&input)?;
    
    // Extract features and returns
    let features = data["features"].as_array()
        .ok_or_else(|| "Missing features array")?;
    let returns = data["returns"].as_array()
        .ok_or_else(|| "Missing returns array")?;
    
    // Convert to ndarray
    let n_samples = features.len();
    let n_assets = if n_samples > 0 {
        features[0].as_array()
            .ok_or_else(|| "Invalid features format")?
            .len() / 2 // Divide by 2 because features has both static and temporal features
    } else {
        0
    };
    
    println!("Input: {} samples, {} assets", n_samples, n_assets);
    
    let mut features_array = Array2::zeros((n_samples, n_assets * 2));
    let mut returns_array = Array2::zeros((n_samples, n_assets));
    
    for i in 0..n_samples {
        let feature_row = features[i].as_array()
            .ok_or_else(|| "Invalid features format")?;
        
        // Ensure feature row has the right length
        if feature_row.len() != n_assets * 2 {
            return Err(format!(
                "Feature row {} has incorrect length: expected {}, got {}", 
                i, n_assets * 2, feature_row.len()
            ).into());
        }
        
        for j in 0..n_assets * 2 {
            features_array[[i, j]] = feature_row[j].as_f64()
                .ok_or_else(|| "Invalid feature value")? as f32;
        }
        
        let return_row = returns[i].as_array()
            .ok_or_else(|| "Invalid returns format")?;
        
        // Ensure return row has the right length
        if return_row.len() != n_assets {
            return Err(format!(
                "Return row {} has incorrect length: expected {}, got {}", 
                i, n_assets, return_row.len()
            ).into());
        }
        
        for j in 0..n_assets {
            returns_array[[i, j]] = return_row[j].as_f64()
                .ok_or_else(|| "Invalid return value")? as f32;
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns_array, features_array);
    
    // Initialize the model
    let model = DeepRiskModel::new(
        n_assets,
        2,  // n_factors
        20, // max_seq_len
        n_assets * 2, // d_model - match the number of columns in our input
        2,  // n_heads
        64, // d_ff
        2   // n_layers
    )?;
    
    // Generate risk factors
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    
    // Output the results
    println!("Risk factors shape: {:?}", risk_factors.factors().shape());
    println!("Covariance matrix shape: {:?}", risk_factors.covariance().shape());
    
    // Convert to JSON response
    let response = json!({
        "factors": risk_factors.factors().to_owned(),
        "covariance": risk_factors.covariance().to_owned(),
    });
    
    println!("Response: {}", response);
    
    Ok(())
} 