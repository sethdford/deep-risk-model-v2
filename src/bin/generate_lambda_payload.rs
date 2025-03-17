use deep_risk_model::prelude::*;
use ndarray::Array2;
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create test data
    let n_samples = 24;
    let n_assets = 16;
    
    // Create sample data
    let mut features_array = Array2::<f32>::zeros((n_samples, n_assets * 2));
    let mut returns_array = Array2::<f32>::zeros((n_samples, n_assets));
    
    // Fill with sample data
    for i in 0..n_samples {
        for j in 0..n_assets {
            returns_array[[i, j]] = (i as f32 * 0.1) + (j as f32 * 0.1);
            features_array[[i, j]] = (i as f32 * 0.1) + (j as f32 * 0.1);
            features_array[[i, j + n_assets]] = (i as f32 * 0.1) + (j as f32 * 0.1) + 1.0;
        }
    }
    
    // Convert to JSON format expected by Lambda
    let mut features_json = Vec::new();
    let mut returns_json = Vec::new();
    
    for i in 0..n_samples {
        let mut feature_row = Vec::new();
        for j in 0..n_assets * 2 {
            feature_row.push(features_array[[i, j]]);
        }
        features_json.push(feature_row);
        
        let mut return_row = Vec::new();
        for j in 0..n_assets {
            return_row.push(returns_array[[i, j]]);
        }
        returns_json.push(return_row);
    }
    
    // Create JSON payload
    let payload = json!({
        "features": features_json,
        "returns": returns_json
    });
    
    // Write to file
    let mut file = File::create("lambda_test_payload.json")?;
    file.write_all(payload.to_string().as_bytes())?;
    
    println!("Created test payload in lambda_test_payload.json");
    println!("To test the Lambda function, run:");
    println!("cargo run --bin lambda < lambda_test_payload.json");
    
    Ok(())
} 