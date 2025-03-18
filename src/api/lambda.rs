use deep_risk_model::prelude::{
    MarketData,
    RiskModel,
    TransformerRiskModel,
};
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use ndarray::Array2;

type SharedModel = Arc<Mutex<TransformerRiskModel>>;

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
    let n_assets = features[0].as_array()
        .ok_or_else(|| Error::from("Invalid features format"))?
        .len();
    
    let mut features_array = Array2::zeros((n_samples, n_assets));
    let mut returns_array = Array2::zeros((n_samples, n_assets));
    
    for i in 0..n_samples {
        let row = features[i].as_array()
            .ok_or_else(|| Error::from("Invalid features format"))?;
        for j in 0..n_assets {
            features_array[[i, j]] = row[j].as_f64()
                .ok_or_else(|| Error::from("Invalid feature value"))? as f32;
        }
        
        let row = returns[i].as_array()
            .ok_or_else(|| Error::from("Invalid returns format"))?;
        for j in 0..n_assets {
            returns_array[[i, j]] = row[j].as_f64()
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
    // Initialize model with appropriate parameters for TransformerRiskModel
    let d_model = 64;
    let n_heads = 8;
    let d_ff = 256;
    let n_layers = 3;
    
    #[cfg(not(feature = "no-blas"))]
    let model = TransformerRiskModel::new(d_model, n_heads, d_ff, n_layers)?;
    
    #[cfg(feature = "no-blas")]
    let model = {
        // In no-blas mode, use a smaller configuration to avoid matrix inversion issues
        let small_d_model = 6;
        let small_n_heads = 1;
        let small_d_ff = 8;
        let small_n_layers = 1;
        TransformerRiskModel::new(small_d_model, small_n_heads, small_d_ff, small_n_layers)?
    };
    
    let shared_model = Arc::new(Mutex::new(model));
    
    // Start lambda service
    run(service_fn(|event| function_handler(event, shared_model.clone()))).await
}