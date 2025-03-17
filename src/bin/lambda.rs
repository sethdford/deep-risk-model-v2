use deep_risk_model::prelude::{
    MarketData,
    RiskModel,
    DeepRiskModel,
};
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use ndarray::Array2;

type SharedModel = Arc<Mutex<DeepRiskModel>>;

#[cfg(not(feature = "no-blas"))]
async fn function_handler(event: Request, model: SharedModel) -> Result<Response<Body>, Error> {
    // Parse request body
    let body = event.body();
    let data: serde_json::Value = serde_json::from_slice(body)?;
    
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
            .len()
    } else {
        0
    };
    
    let mut features_array = Array2::zeros((n_samples, n_assets * 2)); // Double the features for static and temporal
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
    let response = Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(json!({
            "factors": risk_factors.factors().to_owned(),
            "covariance": risk_factors.covariance().to_owned(),
        }).to_string().into())?;
    
    Ok(response)
}

#[cfg(feature = "no-blas")]
async fn function_handler(event: Request, _model: SharedModel) -> Result<Response<Body>, Error> {
    // In no-blas mode, return a message indicating that this functionality requires BLAS
    let response = Response::builder()
        .status(501)
        .header("content-type", "application/json")
        .body(json!({
            "error": "This Lambda function requires BLAS support for matrix operations",
            "message": "The no-blas feature is enabled, which disables advanced matrix operations required for risk factor generation",
            "solution": "Please rebuild the Lambda function with BLAS support enabled"
        }).to_string().into())?;
    
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize model with appropriate parameters
    let n_assets = 100;
    let n_factors = 10;
    
    #[cfg(not(feature = "no-blas"))]
    let mut model = DeepRiskModel::new(
        n_assets,
        n_factors,
        20,   // max_seq_len
        200,  // d_model
        4,    // n_heads
        128,  // d_ff
        2,    // n_layers
    )?;
    
    #[cfg(not(feature = "no-blas"))]
    // Set lower thresholds for factor selection in demo mode
    model.set_factor_selection_thresholds(0.01, 10.0, 0.01)?;
    
    #[cfg(feature = "no-blas")]
    let model = {
        // In no-blas mode, use a smaller configuration to avoid matrix inversion issues
        let small_n_assets = 10;
        let small_n_factors = 2;
        DeepRiskModel::new(
            small_n_assets,
            small_n_factors,
            10,   // max_seq_len
            40,   // d_model
            2,    // n_heads
            64,   // d_ff
            1,    // n_layers
        )?
    };
    
    let shared_model = Arc::new(Mutex::new(model));
    
    // Start lambda service
    run(service_fn(|event| function_handler(event, shared_model.clone()))).await
} 