use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use ndarray::{Array2, s};
use deep_risk_model::prelude::*;
use serde_json::{json, Value};
use tracing::{info, error};
use tracing_subscriber;

// Initialize logging with CloudWatch format
fn init_logging() {
    tracing_subscriber::fmt::init();
}

// Old logging function - to be removed
// fn log(message: &str) {
//     let timestamp = chrono::Utc::now().to_rfc3339();
//     writeln!(io::stderr(), "[{}] {}", timestamp, message).unwrap();
// }

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize logging
    init_logging();
    info!("Lambda function initialized");
    
    run(service_fn(function_handler)).await
}

async fn function_handler(event: LambdaEvent<Value>) -> Result<Value, Error> {
    // Log the incoming event
    info!(event = ?event.payload, "Received event");
    
    // Extract the matrix from the event
    let matrix = match event.payload.as_array() {
        Some(arr) => arr,
        None => {
            error!("Event payload is not an array");
            return Err("Event payload must be a 2D array".into());
        }
    };
    
    // Validate matrix dimensions
    if matrix.is_empty() {
        error!("Empty matrix");
        return Err("Matrix cannot be empty".into());
    }
    
    let n_rows = matrix.len();
    let first_row = match matrix[0].as_array() {
        Some(row) => row,
        None => {
            error!("Matrix row is not an array");
            return Err("Matrix must be a 2D array".into());
        }
    };
    let n_cols = first_row.len();
    
    // Ensure all rows have the same number of columns
    for (i, row) in matrix.iter().enumerate() {
        let row = match row.as_array() {
            Some(r) => r,
            None => {
                error!(row_index = i, "Matrix row is not an array");
                return Err(format!("Row {} is not an array", i).into());
            }
        };
        
        if row.len() != n_cols {
            error!(row_index = i, expected = n_cols, actual = row.len(), "Inconsistent row length");
            return Err(format!("Row {} has inconsistent length", i).into());
        }
    }
    
    info!(rows = n_rows, cols = n_cols, "Matrix dimensions");
    
    // Convert to ndarray
    let mut data = Array2::<f32>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        let row = matrix[i].as_array().unwrap();
        for j in 0..n_cols {
            data[[i, j]] = match row[j].as_f64() {
                Some(val) => val as f32,
                None => {
                    error!(row = i, col = j, "Matrix element is not a number");
                    return Err(format!("Element at [{}, {}] is not a number", i, j).into());
                }
            };
        }
    }
    
    // For this demo, we'll treat the data as both features and returns
    // In a real application, you would separate these appropriately
    // We'll use half the columns as returns and all columns as features
    let n_assets = n_cols / 2;
    if n_assets == 0 {
        error!("Not enough columns for assets");
        return Err("Matrix must have at least 2 columns".into());
    }
    
    let returns = data.slice(s![.., ..n_assets]).to_owned();
    let features = data.clone(); // Use all columns as features
    
    // Create market data
    let market_data = MarketData::new(returns, features);
    
    // Create a model with appropriate dimensions
    // Use 10% of assets as factors, with a minimum of 2
    let n_factors = std::cmp::max(2, (n_assets as f32 * 0.1) as usize);
    info!(n_assets = n_assets, n_factors = n_factors, "Creating model");
    
    let mut model = match DeepRiskModel::new(
        n_assets,
        n_factors,
        50,  // max_seq_len
        n_assets * 2, // d_model - twice the number of assets
        4,   // n_heads
        256, // d_ff
        3    // n_layers
    ) {
        Ok(m) => m,
        Err(e) => {
            error!(error = %e, "Failed to create model");
            return Err(format!("Failed to create model: {}", e).into());
        }
    };
    
    // Set lower thresholds for factor selection in demo mode
    if let Err(e) = model.set_factor_selection_thresholds(0.01, 10.0, 0.01) {
        error!(error = %e, "Failed to set factor selection thresholds");
        return Err(format!("Failed to set factor selection thresholds: {}", e).into());
    }
    
    // Generate risk factors
    let risk_factors = match model.generate_risk_factors(&market_data).await {
        Ok(rf) => rf,
        Err(e) => {
            error!(error = %e, "Failed to generate risk factors");
            return Err(format!("Failed to generate risk factors: {}", e).into());
        }
    };
    
    // Convert factors and covariance to Vec<Vec<f32>> for JSON serialization
    let factors: Vec<Vec<f32>> = risk_factors.factors()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
        
    let covariance: Vec<Vec<f32>> = risk_factors.covariance()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    
    // Prepare response
    let response = json!({
        "factors": factors,
        "covariance": covariance,
        "message": "Successfully generated risk factors"
    });
    
    info!("Function completed successfully");
    Ok(response)
} 