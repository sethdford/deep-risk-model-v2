use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use ndarray::{Array2, s};
use deep_risk_model::prelude::*;
use serde_json::{json, Value};
use tracing::{info, error};
use tracing_subscriber;
use std::sync::Arc;
use tokio::sync::Mutex;

// Initialize logging with CloudWatch format
fn init_logging() {
    tracing_subscriber::fmt::init();
}

// Old logging function - to be removed
// fn log(message: &str) {
//     let timestamp = chrono::Utc::now().to_rfc3339();
//     writeln!(io::stderr(), "[{}] {}", timestamp, message).unwrap();
// }

type SharedModel = Arc<Mutex<DeepRiskModel>>;

async fn function_handler(event: LambdaEvent<Value>, model: SharedModel) -> Result<Value, Error> {
    // Parse request body
    let data = event.payload;
    
    // Extract features and returns
    let features = data["features"].as_array()
        .ok_or_else(|| Error::from("Missing features array"))?;
    let returns = data["returns"].as_array()
        .ok_or_else(|| Error::from("Missing returns array"))?;
    
    // Convert to ndarray
    let n_samples = features.len();
    let n_assets = if n_samples > 0 {
        features[0].as_array()
            .ok_or_else(|| Error::from("Invalid features format"))?
            .len() / 2 // Divide by 2 because features has both static and temporal features
    } else {
        0
    };
    
    let mut features_array = Array2::zeros((n_samples, n_assets * 2));
    let mut returns_array = Array2::zeros((n_samples, n_assets));
    
    for i in 0..n_samples {
        let feature_row = features[i].as_array()
            .ok_or_else(|| Error::from("Invalid features format"))?;
        
        // Ensure feature row has the right length
        if feature_row.len() != n_assets * 2 {
            return Err(Error::from(format!(
                "Feature row {} has incorrect length: expected {}, got {}", 
                i, n_assets * 2, feature_row.len()
            )));
        }
        
        for j in 0..n_assets * 2 {
            features_array[[i, j]] = feature_row[j].as_f64()
                .ok_or_else(|| Error::from("Invalid feature value"))? as f32;
        }
        
        let return_row = returns[i].as_array()
            .ok_or_else(|| Error::from("Invalid returns format"))?;
        
        // Ensure return row has the right length
        if return_row.len() != n_assets {
            return Err(Error::from(format!(
                "Return row {} has incorrect length: expected {}, got {}", 
                i, n_assets, return_row.len()
            )));
        }
        
        for j in 0..n_assets {
            returns_array[[i, j]] = return_row[j].as_f64()
                .ok_or_else(|| Error::from("Invalid return value"))? as f32;
        }
    }
    
    // Create market data
    let market_data = MarketData::new(returns_array, features_array);
    
    // Generate risk factors
    let model = model.lock().await;
    let risk_factors = model.generate_risk_factors(&market_data).await?;
    
    // Convert to JSON response
    let response = json!({
        "factors": risk_factors.factors().to_owned(),
        "covariance": risk_factors.covariance().to_owned(),
    });
    
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize logging
    init_logging();
    info!("Lambda function initialized");
    
    // Initialize model with appropriate parameters
    let n_assets = 16;
    let n_factors = 2;
    
    let model = DeepRiskModel::new(
        n_assets,
        n_factors,
        20,   // max_seq_len
        32,   // d_model - match the number of columns in our input (16 * 2)
        2,    // n_heads
        64,   // d_ff
        2,    // n_layers
    )?;
    
    let shared_model = Arc::new(Mutex::new(model));
    
    // Start lambda service
    run(service_fn(|event| function_handler(event, shared_model.clone()))).await
} 